"""Aggregation queries for the dashboard.

Timestamps are epoch milliseconds UTC everywhere (normalised at the adapter
boundary — Claude Code writes ISO strings, opencode epoch ms, web exports epoch
seconds), so every date bucket goes through `datetime(x/1000,'unixepoch')`.

Abandoned branches are excluded from every count by default. They are real history
but they are not what happened, and counting them would inflate volume by ~4.6%.

Every message-shaped figure comes in two flavours, `messages` and `turns`. Read the
note on TURN below before comparing any of them across sources.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from ..core import freshness, redact
from ..search import selection
from . import model_types, pricing

ACTIVE = "m.on_active_path = 1"

# Messages and turns are different questions and both are reported.
#
# `messages` counts records as the source wrote them, and the sources disagree about
# what a record is: Claude Code emits one per content block (a tool call and its result
# are two messages, the result filed under role='user'), Codex splits call from output,
# while Gemini, T3 Chat, VS Code chat and the web exports fold tool_use/tool_result into
# the assistant message. So `messages` measures volume but cannot be compared BETWEEN
# sources -- two thirds of this archive's messages carry no text at all.
#
# `turns` counts only messages carrying said-or-shown content (models.TURN_KINDS,
# materialised as message.is_turn at ingest). That question means the same thing in
# every source, so it is what cross-source tables rank by.
TURN = "m.on_active_path = 1 AND m.is_turn = 1"


def overview(con: sqlite3.Connection) -> dict:
    row = con.execute(f"""
        SELECT (SELECT COUNT(*) FROM session) sessions,
               (SELECT COUNT(*) FROM message m WHERE {ACTIVE}) messages,
               (SELECT COUNT(*) FROM message m WHERE {TURN}) turns,
               (SELECT COUNT(*) FROM message m WHERE m.on_active_path = 0) abandoned,
               (SELECT COUNT(*) FROM chunk) chunks,
               (SELECT COUNT(*) FROM source) sources,
               (SELECT COUNT(DISTINCT host) FROM session WHERE host IS NOT NULL) hosts,
               (SELECT COALESCE(SUM(tok_in),0) FROM session) tok_in,
               (SELECT COALESCE(SUM(tok_out),0) FROM session) tok_out,
               (SELECT COALESCE(SUM(tok_cache_read),0) FROM session) cache_read,
               (SELECT COALESCE(SUM(tok_cache_write),0) FROM session) cache_write,
               (SELECT MIN(started_at) FROM session WHERE started_at > 0) first_at,
               (SELECT MAX(started_at) FROM session) last_at
    """).fetchone()
    out = dict(row)
    out["tool_steps"] = out["messages"] - out["turns"]
    return out


def by_source(con) -> list[dict]:
    rows = con.execute("""
        SELECT src.kind, src.label, src.surface, COUNT(*) sessions,
               COALESCE(SUM(s.msg_count),0) messages,
               COALESCE(SUM(s.turn_count),0) turns,
               COALESCE(SUM(s.tok_in),0) tok_in,
               COALESCE(SUM(s.tok_out),0) tok_out,
               COALESCE(SUM(s.tok_cache_read),0) cache_read,
               COALESCE(SUM(s.tok_cache_write),0) cache_write
        FROM session s JOIN source src ON src.id = s.source_id
        GROUP BY src.id ORDER BY sessions DESC""").fetchall()
    return [dict(r) for r in rows]


def participants(con) -> list[dict]:
    """Who answered, inside sources where several assistants share one store.

    VS Code's chat panel is shared ground: Copilot Chat, the Remote-SSH participant and
    any other chat extension write to the same files, so `by_source` alone cannot say
    how much of it was Copilot. Sources with a single assistant never set this.
    """
    rows = con.execute("""
        SELECT json_extract(s.meta,'$.participant') key,
               COALESCE(json_extract(s.meta,'$.participant_label'),
                        json_extract(s.meta,'$.participant')) label,
               src.label AS source, COUNT(*) sessions,
               COALESCE(SUM(s.msg_count),0) messages,
               COALESCE(SUM(s.turn_count),0) turns
        FROM session s JOIN source src ON src.id = s.source_id
        WHERE json_extract(s.meta,'$.participant') IS NOT NULL
        -- group by the expression: an output alias would bind to a table column first
        GROUP BY json_extract(s.meta,'$.participant'), src.id
        ORDER BY sessions DESC""").fetchall()
    return [dict(r) for r in rows]


def volume_by_month(con) -> dict:
    """Sessions per month, stacked by source."""
    rows = con.execute("""
        SELECT strftime('%Y-%m', datetime(s.started_at/1000,'unixepoch')) month,
               src.kind, COUNT(*) n
        FROM session s JOIN source src ON src.id = s.source_id
        WHERE s.started_at > 0
        GROUP BY month, src.kind ORDER BY month""").fetchall()
    months, series = [], defaultdict(dict)
    for r in rows:
        if r["month"] not in months:
            months.append(r["month"])
        series[r["kind"]][r["month"]] = r["n"]
    return {
        "months": months,
        "series": {k: [v.get(m, 0) for m in months] for k, v in series.items()},
    }


def messages_by_month(con) -> dict:
    """Messages per month, stacked by source."""
    rows = con.execute(f"""
        SELECT strftime('%Y-%m', datetime(m.created_at/1000,'unixepoch')) month,
               src.kind, COUNT(*) n
        FROM message m JOIN session s ON s.id = m.session_id
                       JOIN source src ON src.id = s.source_id
        WHERE {ACTIVE} AND m.created_at > 0
        GROUP BY month, src.kind ORDER BY month""").fetchall()
    months, series = [], defaultdict(dict)
    for r in rows:
        if r["month"] not in months:
            months.append(r["month"])
        series[r["kind"]][r["month"]] = r["n"]
    return {
        "months": months,
        "series": {k: [v.get(m, 0) for m in months] for k, v in series.items()},
    }


def cumulative_messages(con) -> dict:
    rows = con.execute(f"""
        SELECT strftime('%Y-%m', datetime(m.created_at/1000,'unixepoch')) month,
               COUNT(*) n
        FROM message m WHERE {ACTIVE} AND m.created_at > 0
        GROUP BY month ORDER BY month""").fetchall()
    counts = {r["month"]: r["n"] for r in rows}
    months, running, total = [], [], 0
    # Quiet months are carried forward rather than dropped: on a time axis an
    # omitted month silently compresses the gap and misstates when growth happened.
    for month in _month_span(rows[0]["month"], rows[-1]["month"]) if rows else []:
        total += counts.get(month, 0)
        months.append(month)
        running.append(total)
    return {"months": months, "values": running}


def _month_span(first: str, last: str) -> list[str]:
    """Every 'YYYY-MM' from first to last inclusive, gaps included."""
    y, m = (int(x) for x in first.split("-"))
    end_y, end_m = (int(x) for x in last.split("-"))
    out = []
    while (y, m) <= (end_y, end_m):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def activity_heatmap(con) -> dict:
    """Messages by weekday x hour, local-naive UTC. The shape of your working week."""
    rows = con.execute(f"""
        SELECT CAST(strftime('%w', datetime(m.created_at/1000,'unixepoch')) AS INT) dow,
               CAST(strftime('%H', datetime(m.created_at/1000,'unixepoch')) AS INT) hour,
               COUNT(*) n
        FROM message m WHERE {ACTIVE} AND m.created_at > 0
        GROUP BY dow, hour""").fetchall()
    grid = [[0] * 24 for _ in range(7)]
    for r in rows:
        grid[r["dow"]][r["hour"]] = r["n"]
    # SQLite %w is 0=Sunday; present Monday-first
    order = [1, 2, 3, 4, 5, 6, 0]
    return {"labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "grid": [grid[i] for i in order],
            "peak": max((max(r) for r in grid), default=0)}


# Four sources record no model on any record — Gemini and Mistral because the export
# has no such field at all (§2.4, §2.6), claude.ai and part of VS Code chat because
# theirs is empty. Requiring one dropped 186 of 471 sessions out of this table in
# silence, Mistral's 1,266 output tokens with them. They are labelled by whoever
# answered instead: the participant the adapter recorded, or failing that the source.
#
# The label is NOT a model, and every row built this way carries `model_recorded:
# False` so the reader is told rather than left to assume. Pricing never sees it —
# an app name would sail past `normalise()` into some unrelated price key.
_ANSWERED_BY = """COALESCE(json_extract(s.meta, '$.participant_label'),
                           json_extract(s.meta, '$.participant'),
                           src.label)"""


def models(con) -> list[dict]:
    """Per-model volume and estimated cost, with billing honesty preserved."""
    rows = con.execute(f"""
        SELECT COALESCE(s.model_primary, m.model, {_ANSWERED_BY}) model,
               COALESCE(s.model_primary, m.model) IS NOT NULL model_recorded,
               COUNT(DISTINCT s.id) sessions, COUNT(*) messages,
               COALESCE(SUM(m.is_turn),0) turns,
               COALESCE(SUM(s.tok_in),0) tok_in
        FROM message m JOIN session s ON s.id = m.session_id
                       JOIN source src ON src.id = s.source_id
        GROUP BY 1, 2 ORDER BY messages DESC""").fetchall()

    # token totals live on session, so attribute them per session to avoid
    # multiplying a session's tokens by its message count
    per_model_tokens = con.execute(f"""
        SELECT COALESCE(s.model_primary, {_ANSWERED_BY}) model,
               s.model_primary IS NOT NULL model_recorded,
               COALESCE(SUM(s.tok_in),0) tin, COALESCE(SUM(s.tok_out),0) tout,
               COALESCE(SUM(s.tok_cache_read),0) cr,
               COALESCE(SUM(s.tok_cache_write),0) cw
        FROM session s JOIN source src ON src.id = s.source_id
        GROUP BY 1, 2""").fetchall()
    tokens = {(r["model"], r["model_recorded"]): r for r in per_model_tokens}

    out = []
    for r in rows:
        model = r["model"]
        recorded = bool(r["model_recorded"])
        tok = tokens.get((model, r["model_recorded"]))
        tin = tok["tin"] if tok else 0
        tout = tok["tout"] if tok else 0
        cr = tok["cr"] if tok else 0
        cw = tok["cw"] if tok else 0
        if recorded:
            usd, billed = pricing.cost(model, tin, tout, cr, cw)
            price, reason = pricing.lookup(model)
        else:
            # `lookup(None)` already words this exactly right, and pricing an app
            # name would be worse than pricing nothing.
            usd, billed = 0.0, False
            price, reason = pricing.lookup(None)
        out.append({
            "model": model, "model_recorded": recorded,
            "type": model_types.classify(model) if recorded else None,
            "sessions": r["sessions"], "messages": r["messages"],
            "turns": r["turns"],
            "tok_in": tin, "tok_out": tout, "cache_read": cr, "cache_write": cw,
            "usd": usd, "billed": billed, "reason": reason,
            "verified": price.verified if price else False,
        })
    return out


def cost_by_workspace(con) -> list[dict]:
    rows = con.execute("""
        SELECT COALESCE(w.label,'(none)') workspace, s.model_primary model,
               COALESCE(SUM(s.tok_in),0) tin, COALESCE(SUM(s.tok_out),0) tout,
               COALESCE(SUM(s.tok_cache_read),0) cr,
               COALESCE(SUM(s.tok_cache_write),0) cw,
               COUNT(*) sessions
        FROM session s LEFT JOIN workspace w ON w.id = s.workspace_id
        GROUP BY workspace, model""").fetchall()
    totals: dict[str, dict] = {}
    for r in rows:
        usd, billed = pricing.cost(r["model"], r["tin"], r["tout"], r["cr"], r["cw"])
        entry = totals.setdefault(r["workspace"],
                                  {"workspace": r["workspace"], "usd": 0.0,
                                   "sessions": 0, "unbilled": 0.0})
        entry["sessions"] += r["sessions"]
        if billed:
            entry["usd"] += usd
        else:
            entry["unbilled"] += usd
    return sorted(totals.values(), key=lambda e: -e["usd"])[:12]


def tools(con) -> list[dict]:
    rows = con.execute("""
        SELECT p.tool_name name, COUNT(*) calls,
               SUM(CASE WHEN p.tool_ok = 0 THEN 1 ELSE 0 END) failures,
               COUNT(p.duration_ms) timed_calls,
               COALESCE(SUM(p.duration_ms),0) total_ms,
               AVG(p.duration_ms) avg_ms
        FROM part p JOIN message m ON m.id = p.message_id
        WHERE p.tool_name IS NOT NULL AND p.kind = 'tool_use'
        GROUP BY p.tool_name ORDER BY calls DESC LIMIT 14""").fetchall()
    return [dict(r) | {
        "fail_pct": (100.0 * r["failures"] / r["calls"]) if r["calls"] else 0,
        "avg_ms": r["avg_ms"] or 0,
    } for r in rows]


def tools_by_model(con, tool_limit: int = 10, model_limit: int = 6) -> dict:
    """Which models drove each tool's calls, for the legend on the tools chart.

    Same model resolution as `models()`: session, then message, then whoever
    answered when no model was ever recorded. A tool used by more than
    `model_limit` distinct models folds the smallest into "Other" rather than
    blowing past the fixed series-color slots.
    """
    top_tools = [r["name"] for r in con.execute("""
        SELECT p.tool_name name, COUNT(*) calls
        FROM part p JOIN message m ON m.id = p.message_id
        WHERE p.tool_name IS NOT NULL AND p.kind = 'tool_use'
        GROUP BY p.tool_name ORDER BY calls DESC LIMIT ?""", (tool_limit,)).fetchall()]

    rows = con.execute(f"""
        SELECT p.tool_name tool, COALESCE(s.model_primary, m.model, {_ANSWERED_BY}) model,
               COUNT(*) calls
        FROM part p JOIN message m ON m.id = p.message_id
                    JOIN session s ON s.id = m.session_id
                    JOIN source src ON src.id = s.source_id
        WHERE p.tool_name IS NOT NULL AND p.kind = 'tool_use'
        GROUP BY tool, model""").fetchall()

    model_totals: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["tool"] in top_tools:
            model_totals[r["model"]] += r["calls"]
    top_models = [m for m, _ in
                  sorted(model_totals.items(), key=lambda kv: -kv[1])[:model_limit]]

    tool_index = {t: i for i, t in enumerate(top_tools)}
    series: dict[str, list[int]] = {m: [0] * len(top_tools) for m in top_models}
    series["Other"] = [0] * len(top_tools)
    for r in rows:
        i = tool_index.get(r["tool"])
        if i is None:
            continue
        key = r["model"] if r["model"] in top_models else "Other"
        series[key][i] += r["calls"]
    if not any(series["Other"]):
        del series["Other"]

    return {"tools": top_tools, "series": series}


def tool_duration(con) -> dict:
    """Wall-clock time spent waiting on tools, where the source timestamps both ends.

    Only Claude Code, Codex and opencode record a call and its result far enough apart
    to time it; everything else leaves `duration_ms` NULL rather than guessing, so this
    is a partial total, not "every tool call ever made".
    """
    row = con.execute("""
        SELECT COUNT(*) timed_calls, COALESCE(SUM(duration_ms),0) total_ms,
               AVG(duration_ms) avg_ms, MAX(duration_ms) max_ms
        FROM part WHERE kind = 'tool_use' AND duration_ms IS NOT NULL""").fetchone()
    total_calls = con.execute(
        "SELECT COUNT(*) n FROM part WHERE kind = 'tool_use'").fetchone()["n"]
    return {
        "timed_calls": row["timed_calls"] or 0,
        "total_calls": total_calls,
        "total_ms": row["total_ms"] or 0,
        "avg_ms": row["avg_ms"] or 0,
        "max_ms": row["max_ms"] or 0,
        "coverage_pct": (100.0 * row["timed_calls"] / total_calls) if total_calls else 0.0,
    }


def tool_outcomes(con) -> dict:
    row = con.execute("""
        SELECT COUNT(*) total,
               SUM(CASE WHEN tool_ok = 0 THEN 1 ELSE 0 END) failed
        FROM part WHERE kind = 'tool_result' AND tool_ok IS NOT NULL""").fetchone()
    total = row["total"] or 0
    failed = row["failed"] or 0
    return {"total": total, "failed": failed,
            "pct": (100.0 * failed / total) if total else 0.0}


SURFACE_LABELS = {"cli": "Terminal", "web": "Web chat", "editor_panel": "Editor panel"}


def surface_split(con) -> dict:
    """Terminal versus editor panel versus web chat over time — how work has shifted."""
    rows = con.execute("""
        SELECT strftime('%Y-%m', datetime(s.started_at/1000,'unixepoch')) month,
               src.surface, COUNT(*) n
        FROM session s JOIN source src ON src.id = s.source_id
        WHERE s.started_at > 0 GROUP BY month, src.surface ORDER BY month""").fetchall()
    months, series = [], defaultdict(dict)
    for r in rows:
        if r["month"] not in months:
            months.append(r["month"])
        series[r["surface"]][r["month"]] = r["n"]
    return {"months": months,
            "labels": {k: SURFACE_LABELS.get(k, k) for k in series},
            "series": {k: [v.get(m, 0) for m in months] for k, v in series.items()}}


def workspaces(con, limit: int = 12) -> list[dict]:
    # Grouped by (workspace, host) rather than workspace alone: the same project
    # name can exist on two machines as distinct workspace rows, and sessions with
    # no workspace ("(none)") span both real hosts and host-less web sessions —
    # without the host in the label those all render as indistinguishable bars.
    rows = con.execute("""
        SELECT COALESCE(w.label,'(none)') label, s.host host, COUNT(*) sessions,
               COALESCE(SUM(s.msg_count),0) messages,
               COALESCE(SUM(s.turn_count),0) turns
        FROM session s LEFT JOIN workspace w ON w.id = s.workspace_id
        GROUP BY s.workspace_id, s.host ORDER BY messages DESC LIMIT ?""", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        host = (d.pop("host") or "").strip()
        machine = host[:10] if host else "web"
        d["label"] = f'{d["label"]} - {machine}'
        out.append(d)
    return out


def hosts(con) -> list[dict]:
    rows = con.execute("""
        SELECT COALESCE(host,'(web)') host, COUNT(*) sessions,
               COALESCE(SUM(msg_count),0) messages,
               COALESCE(SUM(turn_count),0) turns
        FROM session GROUP BY host ORDER BY sessions DESC""").fetchall()
    return [dict(r) for r in rows]


def content_mix(con) -> list[dict]:
    rows = con.execute("""
        SELECT kind, COUNT(*) parts, COALESCE(SUM(bytes),0) bytes,
               COALESCE(SUM(CASE WHEN embed_eligible=1 THEN LENGTH(text) ELSE 0 END),0) embedded
        FROM part GROUP BY kind ORDER BY bytes DESC""").fetchall()
    return [dict(r) for r in rows]


def session_shape(con) -> dict:
    """Turn-count distribution, in buckets. Most sessions are short.

    Bucketed on `turn_count`, not `msg_count`: on raw messages every agentic session
    landed in the top bucket regardless of how much conversation it held, because a
    single request that runs twenty tools is twenty-one Claude Code messages and one
    T3 Chat message. Sessions that are pure tool traffic get their own '0' bucket
    rather than being folded into '1-2'.
    """
    rows = con.execute("""
        SELECT CASE
                 WHEN turn_count = 0  THEN '0'
                 WHEN turn_count <= 2  THEN '1-2'
                 WHEN turn_count <= 5  THEN '3-5'
                 WHEN turn_count <= 15 THEN '6-15'
                 WHEN turn_count <= 50 THEN '16-50'
                 WHEN turn_count <= 200 THEN '51-200'
                 ELSE '200+' END bucket,
               COUNT(*) n
        FROM session GROUP BY bucket""").fetchall()
    order = ["0", "1-2", "3-5", "6-15", "16-50", "51-200", "200+"]
    found = {r["bucket"]: r["n"] for r in rows}
    return {"labels": order, "values": [found.get(b, 0) for b in order]}


def hygiene(con) -> dict:
    runs = con.execute("""
        SELECT source_kind, finished_at, sessions_new, sessions_updated,
               sessions_skipped, stats
        FROM ingest_run ORDER BY id DESC LIMIT 8""").fetchall()
    unpriced = pricing.coverage(
        [r["model_primary"] for r in con.execute(
            "SELECT model_primary FROM session")])
    return {"runs": [dict(r) for r in runs], "pricing_coverage": unpriced}


def _has_index(con) -> bool:
    """Is there a usable index at all, whatever we know about when it was made?

    `SELECT COUNT(*) FROM part_fts` is not the question it looks like: part_fts is
    external-content, so that counts rows in `part`, and reports a full index for an
    archive that has never been indexed at all. `part_fts_docsize` holds one row per
    *indexed* document, which is the thing actually being asked about.
    """
    if con.execute("SELECT 1 FROM chunk LIMIT 1").fetchone():
        return True
    try:
        return bool(con.execute("SELECT 1 FROM part_fts_docsize LIMIT 1").fetchone())
    except sqlite3.OperationalError:
        return False


def index_health(con) -> dict:
    """Whether search can be trusted right now, and when it was last rebuilt.

    Ingesting deletes and rewrites a session's messages and parts, so part ids move.
    Both indexes are keyed on those ids: FTS5 is external-content and keeps matching the
    old rowids against new rows, and every `chunk` row for a re-ingested session points
    at a part that no longer exists. Neither failure raises anything — keyword search
    returns the wrong passages and vector search quietly stops seeing those sessions —
    so the only honest answer is to compare when the index was built against when the
    archive last changed.

    `unindexed_parts` is the exact count of embeddable text vector search cannot reach;
    it mirrors the query `index._embeddable_parts` uses to decide what to embed.
    """
    # Measured the same way in every state: coverage is a fact about the tables, not
    # something only a recorded build gets to have an opinion about. The count comes
    # from `search.selection` so it can never drift from what the indexer embeds.
    unindexed = selection.unindexed_count(con)
    empty = {"state": "unrecorded", "last": None, "sessions_since": 0,
             "unindexed_parts": unindexed, "reasons": [],
             "notes": ([f"{unindexed:,} embeddable part(s) are not in the vector index"]
                       if unindexed else [])}

    recorded = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='index_run'"
    ).fetchone()
    row = con.execute("""
        SELECT started_at, finished_at, fts_rows, chunks, vectors, model_tag,
               with_vectors, seconds, warnings
        FROM index_run ORDER BY id DESC LIMIT 1""").fetchone() if recorded else None
    if row is None:
        # No record is not the same as no index: an archive indexed before builds were
        # logged has working search and an unknown build date. Saying "never built" there
        # would be a flat lie about data that is sitting right in the tables.
        return {**empty, "state": "unrecorded" if _has_index(con) else "never"}

    last = dict(row)
    since = con.execute(
        "SELECT COUNT(*) n FROM session WHERE ingested_at > ?",
        (last["finished_at"],)).fetchone()["n"]

    # `reasons` are why the index no longer matches the archive; `notes` are true of a
    # perfectly current index. A deliberate --no-vectors build is the second kind: its
    # keyword index is exactly right, and calling that "out of date" would train you to
    # ignore the warning that matters.
    reasons, notes = [], []
    if since:
        reasons.append(f"{since} session(s) ingested since the last build")
    if unindexed and last["with_vectors"]:
        reasons.append(f"{unindexed:,} embeddable part(s) not in the vector index")
    if not last["with_vectors"]:
        notes.append("built with --no-vectors — keyword search only, "
                     "semantic search needs a full rebuild")
    if last["warnings"]:
        # say what the warning was; "reported warnings" sends you to the terminal
        # history for something already sitting in the row
        try:
            notes.extend(str(w) for w in json.loads(last["warnings"]))
        except (ValueError, TypeError):
            notes.append(str(last["warnings"]))

    return {"state": "stale" if reasons else "fresh", "last": last,
            "sessions_since": since, "unindexed_parts": unindexed,
            "reasons": reasons, "notes": notes}


def everything(con) -> dict:
    ov = overview(con)
    model_rows = models(con)
    billed_total = sum(m["usd"] for m in model_rows if m["billed"])
    notional_total = sum(m["usd"] for m in model_rows)
    return {
        "overview": ov,
        "sources": by_source(con),
        "participants": participants(con),
        "volume": volume_by_month(con),
        "messages_volume": messages_by_month(con),
        "cumulative": cumulative_messages(con),
        "heatmap": activity_heatmap(con),
        "models": model_rows,
        "models_unrecorded": {
            "rows": sum(1 for m in model_rows if not m["model_recorded"]),
            "sessions": sum(m["sessions"] for m in model_rows
                            if not m["model_recorded"]),
            "tok_out": sum(m["tok_out"] for m in model_rows
                           if not m["model_recorded"]),
        },
        "cost_total_billed": billed_total,
        "cost_total_notional": notional_total,
        "cost_by_workspace": cost_by_workspace(con),
        "tools": tools(con),
        "tools_by_model": tools_by_model(con),
        "tool_outcomes": tool_outcomes(con),
        "tool_duration": tool_duration(con),
        "surface_split": surface_split(con),
        "workspaces": workspaces(con),
        "hosts": hosts(con),
        "content": content_mix(con),
        "shape": session_shape(con),
        "hygiene": hygiene(con),
        "index": index_health(con),
        # Both are questions about the archive's *upkeep* rather than its contents,
        # and both are invisible from the charts: a source whose export stopped
        # arriving simply flattens out in the volume bars, which reads as a quiet
        # month rather than as a missing feed.
        "freshness": freshness.report(con),
        "redaction": redact.summary(con),
    }
