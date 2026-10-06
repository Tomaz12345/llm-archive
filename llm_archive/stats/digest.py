"""What happened in a window of time: the archive as a weekly report.

Extractive and offline. Nothing here summarises anything -- every line is a field the
archive already holds, arranged so that "what did I do last week, and where" can be
read in a minute: sessions per project, what each one opened with, which files it
wrote, what it cost, and which ones you never came back to.

**The window is about activity, not birth.** A session is in the digest if any of it
happened inside the window -- `ended_at >= since` and `started_at < until` -- so a
conversation resumed on Monday that began the Friday before is on this week's list,
where the work was. `metrics.LIVE` is applied on top: a session wholly replayed by the
one that resumed it would otherwise appear twice and be paid for twice.

**Subagents fold into their parent.** A Claude Code session that spawned three
explorers is one piece of work, so its children's tokens are added to its cost and it
is listed once, with the count. Listing each transcript would make a busy afternoon
read as a dozen sessions whose "first prompt" is an agent's instruction to another.

**Cost is computed, never read.** `session.cost_usd` is set on 11 of ~600 rows; the
number that means something is `pricing.cost()` over the session's own counters. It is
the list-price value of the tokens, in the sense stats/pricing.py spells out -- what
this volume would cost at API rates, which for a Claude Code subscription is not what
left the account. Copilot and `:free` models are priced but kept apart as `unbilled`.

**Files come from the derived tables**, so they are only as fresh as the last
`llma index`; `facts_available` says whether that has run at all, and the renderers
say so rather than showing an empty column. Files inside the project come first and
the rest are counted: a Claude Code session writes its scratch patches and memory
notes under the temp directory and `~/.claude`, and those would otherwise crowd out
the six lines that name what the session actually changed.

The builder returns plain data. `render_text` is for a terminal, `render_markdown`
for a file or the web page, and `--json` is the dict itself.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

from ..core import unfinished
from . import metrics, pricing

PROMPT_CHARS = 140
FILES_PER_SESSION = 6
HOT_FILES = 8
NO_PROJECT = "(no project)"

# The window, as a CTE every query joins: activity overlaps [since, until), and the
# session is not a replayed copy of a later one.
_WINDOW = f"""
WITH win AS (
  SELECT s.id FROM session s
   WHERE COALESCE(s.ended_at, s.started_at) >= ? AND s.started_at < ?
     AND {metrics.LIVE})
"""

_SESSIONS = _WINDOW + """
SELECT s.id, s.title, s.title_source, s.started_at, s.ended_at, s.host,
       s.model_primary, s.msg_count, s.turn_count, s.parent_session_id,
       s.tok_in, s.tok_out, s.tok_cache_read, s.tok_cache_write,
       src.kind AS source, COALESCE(w.label, '') AS workspace
  FROM win JOIN session s ON s.id = win.id
  JOIN source src ON src.id = s.source_id
  LEFT JOIN workspace w ON w.id = s.workspace_id
 ORDER BY s.started_at
"""

# The first thing the person typed on the live path. The window function rather than
# a correlated LIMIT 1 per session: one pass, and no id list to chunk.
_FIRST_PROMPTS = _WINDOW + """
SELECT session_id, text FROM (
  SELECT m.session_id, p.text,
         ROW_NUMBER() OVER (PARTITION BY m.session_id ORDER BY m.seq, p.seq) AS rn
    FROM win JOIN message m ON m.session_id = win.id
    JOIN part p ON p.message_id = m.id
   WHERE m.on_active_path = 1 AND m.is_turn = 1 AND m.role = 'user'
     AND p.kind = 'text' AND p.text IS NOT NULL AND TRIM(p.text) != '')
 WHERE rn = 1
"""

# Per session, the files it touched, most-written first. Grouped on the
# workspace-relative path for the reason stats/workspaces.py gives: the same file from
# two roots is one row.
_SESSION_FILES = _WINDOW + """
SELECT tf.session_id, COALESCE(tf.rel, tf.norm) AS key, MIN(tf.path) AS path,
       tf.rel IS NOT NULL AS inside,
       COUNT(*) AS calls,
       SUM(CASE WHEN tf.action IN ('write','edit','delete') THEN 1 ELSE 0 END) AS writes
  FROM win JOIN touched_file tf ON tf.session_id = win.id
 WHERE tf.ok IS NOT 0
 GROUP BY tf.session_id, key
 ORDER BY tf.session_id, inside DESC, writes DESC, calls DESC
"""


def _iso(ms: int | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def _facts_available(con: sqlite3.Connection) -> bool:
    try:
        return con.execute("SELECT 1 FROM touched_file LIMIT 1").fetchone() is not None
    except sqlite3.OperationalError:
        return False              # pre-v11 archive


def _cost(row) -> tuple[float, bool]:
    return pricing.cost(row["model_primary"], row["tok_in"] or 0, row["tok_out"] or 0,
                        row["tok_cache_read"] or 0, row["tok_cache_write"] or 0)


def build(con: sqlite3.Connection, since: int, until: int | None = None, *,
          now_ms: int | None = None, label: str | None = None) -> dict:
    """The digest for [since, until) as plain data. `label` is how the window was
    asked for (`last-week`), kept so the heading can say it back."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    until = until if until is not None else now
    params = (since, until)
    facts = _facts_available(con)

    rows = con.execute(_SESSIONS, params).fetchall()
    prompts = {r["session_id"]: r["text"]
               for r in con.execute(_FIRST_PROMPTS, params)}
    files: dict[int, list[dict]] = {}
    if facts:
        for r in con.execute(_SESSION_FILES, params):
            files.setdefault(r["session_id"], []).append(
                {"key": r["key"], "path": r["path"], "calls": r["calls"],
                 "writes": r["writes"], "inside": bool(r["inside"])})

    # Everything unfinished, then narrowed to the window: `find` already knows about
    # resumed pairs, subagents, dismissals and the min-age grace period.
    ids = {r["id"] for r in rows}
    open_items = [it for it in unfinished.find(con, now_ms=now)
                  if it.session_id in ids]
    open_by_id = {it.session_id: it for it in open_items}

    sessions: dict[int, dict] = {}
    for r in rows:
        usd, billed = _cost(r)
        item = open_by_id.get(r["id"])
        inside = [f for f in files.get(r["id"], []) if f["inside"]]
        outside = len(files.get(r["id"], [])) - len(inside)
        sessions[r["id"]] = {
            "session_id": r["id"],
            "title": r["title"],
            "source": r["source"],
            "workspace": r["workspace"],
            "host": r["host"],
            "model": r["model_primary"],
            "started_at": _iso(r["started_at"]),
            "ended_at": _iso(r["ended_at"]),
            # began on an earlier DAY than the window: the digest shows where the
            # session was, not when it was born, so the renderers say "since <day>"
            # on these. Compared by day, because a window that opens at 14:07 on
            # Tuesday should not call a session started that morning carried over.
            "carried_over": _day(r["started_at"]) < _day(since),
            "_started": r["started_at"],
            "turns": r["turn_count"] or 0,
            "messages": r["msg_count"] or 0,
            "first_prompt": unfinished.one_line(prompts.get(r["id"]), PROMPT_CHARS),
            "usd": usd,
            "billed": billed,
            "priced": pricing.lookup(r["model_primary"])[0] is not None,
            "subagents": 0,
            "files": [{k: v for k, v in f.items() if k != "inside"}
                      for f in inside[:FILES_PER_SESSION]],
            "more_files": max(0, len(inside) - FILES_PER_SESSION),
            "outside_files": outside,
            "unfinished": item.reason if item else None,
            "_parent": r["parent_session_id"],
        }

    # Fold subagents into the parent that spawned them. A child whose parent is not
    # in the window stays on its own: its work happened here, and dropping it would
    # lose the tokens.
    for sid in list(sessions):
        parent = sessions[sid]["_parent"]
        if parent is None or parent not in sessions or parent == sid:
            continue
        child = sessions.pop(sid)
        target = sessions[parent]
        target["subagents"] += 1
        target["usd"] += child["usd"]
        target["turns"] += child["turns"]
        target["messages"] += child["messages"]

    by_ws: dict[str, list[dict]] = {}
    for s in sessions.values():
        by_ws.setdefault(s["workspace"] or NO_PROJECT, []).append(s)

    workspaces = []
    for name, items in by_ws.items():
        items.sort(key=lambda s: s["_started"])
        hot: dict[str, dict] = {}
        for s in items:
            for f in files.get(s["session_id"], []):
                if not f["inside"]:
                    continue
                slot = hot.setdefault(f["key"], {"key": f["key"], "path": f["path"],
                                                 "writes": 0, "calls": 0, "sessions": 0})
                slot["writes"] += f["writes"]
                slot["calls"] += f["calls"]
                slot["sessions"] += 1
        workspaces.append({
            "workspace": name,
            "sessions": len(items),
            "turns": sum(s["turns"] for s in items),
            "usd": sum(s["usd"] for s in items if s["billed"]),
            "unbilled_usd": sum(s["usd"] for s in items if not s["billed"]),
            "unfinished": sum(1 for s in items if s["unfinished"]),
            "hot_files": sorted(hot.values(),
                                key=lambda f: (-f["writes"], -f["calls"]))[:HOT_FILES],
            "sessions_list": items,
        })
    # busiest first; the unattributed pile goes after a project it ties with
    workspaces.sort(key=lambda w: (-w["sessions"], -w["turns"],
                                   w["workspace"] == NO_PROJECT, w["workspace"]))

    by_source: dict[str, int] = {}
    for s in sessions.values():
        by_source[s["source"]] = by_source.get(s["source"], 0) + 1

    for s in sessions.values():
        s.pop("_started", None)
        s.pop("_parent", None)

    listed = list(sessions.values())
    return {
        "since": _iso(since),
        "until": _iso(until),
        "label": label,
        "generated_at": _iso(now),
        "facts_available": facts,
        "totals": {
            "sessions": len(listed),
            "subagents": sum(s["subagents"] for s in listed),
            "workspaces": len(workspaces),
            "turns": sum(s["turns"] for s in listed),
            "usd": sum(s["usd"] for s in listed if s["billed"]),
            "unbilled_usd": sum(s["usd"] for s in listed if not s["billed"]),
            "unpriced_sessions": sum(1 for s in listed if not s["priced"]),
            "unfinished": len(open_items),
            "by_source": [{"source": k, "sessions": n}
                          for k, n in sorted(by_source.items(), key=lambda kv: -kv[1])],
        },
        "workspaces": workspaces,
        "unfinished": [{
            "session_id": it.session_id, "reason": it.reason,
            "title": it.title, "workspace": it.workspace or None,
            "source": it.source, "last_at": _iso(it.last_at), "excerpt": it.excerpt,
        } for it in open_items],
    }


# ------------------------------------------------------------------ rendering

def _money(usd: float, billed: bool, priced: bool = True) -> str:
    """`$1.23` at list price, `($1.23)` on a subscription or free tier, `—` unpriced."""
    if not priced:
        return "—"
    if usd < 0.005:
        return "$0" if billed else "($0)"
    return f"${usd:,.2f}" if billed else f"(${usd:,.2f})"


def _cost_line(usd: float, unbilled: float) -> str:
    """The list-price value, the way stats/pricing.py means it: what the tokens would
    cost at API rates, not what was paid."""
    bits = []
    if usd >= 0.005:
        bits.append(f"${usd:,.2f} at list price")
    if unbilled >= 0.005:
        bits.append(f"(${unbilled:,.2f} covered by a subscription)")
    return " ".join(bits) if bits else "$0"


def _when(s: dict) -> str:
    """The day the session was last active -- and where it came from, if that was
    before the window."""
    last = _day_of(s["ended_at"] or s["started_at"])
    return f"{last} (since {_day_of(s['started_at'])})" if s["carried_over"] else last


def _files_line(s: dict, quote: str = "`") -> str:
    names = ", ".join(f"{quote}{f['key']}{quote}" for f in s["files"])
    extra = []
    if s["more_files"]:
        extra.append(f"+{s['more_files']} more")
    if s["outside_files"]:
        extra.append(f"{s['outside_files']} outside the project")
    if names and extra:
        return f"{names} ({', '.join(extra)})"
    return names or ", ".join(extra)


def _heading(d: dict) -> str:
    span = f"{d['since'][:10]} → {d['until'][:10]}"
    return f"{span}  ({d['label']})" if d.get("label") else span


def render_text(d: dict) -> str:
    """For a terminal: dense, aligned, one session per three lines."""
    t = d["totals"]
    out = [f"DIGEST  {_heading(d)}"]
    if not t["sessions"]:
        out.append("nothing happened in this window")
        return "\n".join(out) + "\n"

    parts = [f"{t['sessions']} session{'s' if t['sessions'] != 1 else ''}"]
    if t["subagents"]:
        parts[-1] += f" (+{t['subagents']} subagent)"
    parts += [f"{t['workspaces']} project{'s' if t['workspaces'] != 1 else ''}",
              f"{t['turns']:,} turns", _cost_line(t["usd"], t["unbilled_usd"])]
    if t["unfinished"]:
        parts.append(f"{t['unfinished']} unfinished")
    out.append("  " + " · ".join(parts))
    out.append("  " + " · ".join(f"{s['sessions']} {s['source']}"
                                 for s in t["by_source"]))
    if not d["facts_available"]:
        out.append("  (no file history yet -- run `llma index` to see files)")

    for w in d["workspaces"]:
        out.append("")
        head = (f"{w['workspace']}  --  {w['sessions']} session"
                f"{'s' if w['sessions'] != 1 else ''} · {w['turns']:,} turns"
                f" · {_cost_line(w['usd'], w['unbilled_usd'])}")
        out.append(head)
        if w["hot_files"]:
            hot = " · ".join(f"{f['key']} ({f['writes']}w)" if f["writes"]
                             else f"{f['key']}" for f in w["hot_files"][:5])
            out.append(f"  hot: {hot}")
        for s in w["sessions_list"]:
            out.append("")
            flag = f"   {s['unfinished'].upper()}" if s["unfinished"] else ""
            sub = f" +{s['subagents']} sub" if s["subagents"] else ""
            out.append(f"  #{s['session_id']:<5} {_when(s)}  "
                       f"{s['source']:<12} {s['turns']:>4} turns{sub}   "
                       f"{_money(s['usd'], s['billed'], s['priced'])}{flag}")
            if s["title"]:
                out.append(f"         {s['title']}")
            if s["first_prompt"] and s["first_prompt"] != s["title"]:
                out.append(f"         › {s['first_prompt']}")
            if s["files"] or s["outside_files"]:
                out.append(f"         files: {_files_line(s, quote='')}")

    if d["unfinished"]:
        out.append("")
        out.append(f"UNFINISHED ({len(d['unfinished'])})  -- llma inbox")
        for it in d["unfinished"]:
            where = f" · {it['workspace']}" if it["workspace"] else ""
            out.append(f"  #{it['session_id']:<5} {it['reason']:<10} "
                       f"{it['title'] or '(untitled)'}{where}")
            if it["excerpt"]:
                out.append(f"         {it['excerpt'][:120]}")
    return "\n".join(out) + "\n"


def _day_of(iso_str: str | None) -> str:
    return iso_str[5:10] if iso_str else "??-??"


def render_markdown(d: dict, session_url=None) -> str:
    """For a file, or the web page. `session_url(id)` makes ids into links; without
    it a session is cited as `#412`, which `llma show 412` understands."""
    t = d["totals"]

    def ref(sid: int) -> str:
        return f"[#{sid}]({session_url(sid)})" if session_url else f"#{sid}"

    out = [f"# Digest — {_heading(d)}", ""]
    if not t["sessions"]:
        out.append("Nothing happened in this window.")
        return "\n".join(out) + "\n"

    summary = (f"**{t['sessions']} session{'s' if t['sessions'] != 1 else ''}**"
               + (f" (+{t['subagents']} subagent)" if t["subagents"] else "")
               + f" across {t['workspaces']} project{'s' if t['workspaces'] != 1 else ''}"
               f" · {t['turns']:,} turns · {_cost_line(t['usd'], t['unbilled_usd'])}")
    if t["unfinished"]:
        summary += f" · **{t['unfinished']} unfinished**"
    out.append(summary)
    out.append("")
    out.append(" · ".join(f"{s['sessions']} {s['source']}" for s in t["by_source"]))
    if not d["facts_available"]:
        out.append("")
        out.append("_No file history yet — run `llma index` to see files._")

    for w in d["workspaces"]:
        out += ["", f"## {w['workspace']}", ""]
        out.append(f"{w['sessions']} session{'s' if w['sessions'] != 1 else ''} · "
                   f"{w['turns']:,} turns · {_cost_line(w['usd'], w['unbilled_usd'])}")
        if w["hot_files"]:
            out.append("")
            out.append("Hot files: " + ", ".join(
                f"`{f['key']}`" + (f" ({f['writes']} write{'s' if f['writes'] != 1 else ''})"
                                   if f["writes"] else "")
                for f in w["hot_files"]))
        for s in w["sessions_list"]:
            out.append("")
            flag = f" · **{s['unfinished']}**" if s["unfinished"] else ""
            sub = f" +{s['subagents']} subagent" if s["subagents"] else ""
            out.append(f"- {ref(s['session_id'])} **{s['title'] or '(untitled)'}** — "
                       f"{_when(s)} · {s['source']} · {s['turns']} turns"
                       f"{sub} · {_money(s['usd'], s['billed'], s['priced'])}{flag}")
            if s["first_prompt"] and s["first_prompt"] != s["title"]:
                out.append(f"  › {s['first_prompt']}")
            if s["files"] or s["outside_files"]:
                out.append(f"  files: {_files_line(s)}")

    if d["unfinished"]:
        out += ["", f"## Unfinished ({len(d['unfinished'])})", ""]
        for it in d["unfinished"]:
            where = f" · {it['workspace']}" if it["workspace"] else ""
            line = f"- {ref(it['session_id'])} `{it['reason']}` {it['title'] or '(untitled)'}{where}"
            if it["excerpt"]:
                line += f" — _{it['excerpt'][:120]}_"
            out.append(line)
        out.append("")
        out.append("`llma inbox` lists these; `llma inbox dismiss <id>` drops one.")
    return "\n".join(out) + "\n"
