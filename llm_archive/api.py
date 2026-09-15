"""The archive as data: the JSON view behind `--json` and the MCP server.

`llma search` and `llma show` print for a person — aligned columns, a 150-character
snippet, a transcript cut where it stops being readable. Nothing but a person can
consume that, which is why the archive has been legible to its owner and opaque to every
tool they run. This module is the other half: the same three reads (a query, one
session, its neighbours) shaped as dicts, so `--json` and `mcp/server.py` hand back the
same payload and the field names are decided in exactly one place.

Two rules shape every payload.

**Bounded.** A single `tool_result` part in this archive runs to 32 KB and a long agent
session to megabytes. Whatever is reading this has a context window, so every text field
has a ceiling and the caller sets it.

**Honest about the bound.** `chars` is the true length and `truncated` says whether
`text` is all of it. Without that pair a reader cannot tell a short answer from a long
one that was cut, which is the difference between "this session did not solve it" and
"the answer is further down".
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .core import lineage, reopen
from .search.hybrid import Filters, Hit, related as run_related, search as run_search

# Ceilings, all overridable per call. A snippet is a hook to decide whether to open the
# session, not the answer; a part is generous enough to hold a whole reply; the session
# budget is what one transcript may spend before the rest is declared omitted.
SNIPPET_CHARS = 400
PART_CHARS = 4000

# The session budget counts the rendered document, not the body text inside it. Charging
# text alone was wrong in the way that matters: an agent session is hundreds of *short*
# parts, and at ~400 characters of envelope each, the structure outweighs the prose. A
# 12,000-character request against one such session returned a 63 KB document — over the
# inline tool-result limit of the client that asked for it, so the whole thing spilled to
# a file and the model had to ask again, smaller. Measured at 350-450 across real
# sessions: per-part keys, the message wrapper amortised over its parts, and the
# escaping of newlines and quotes inside `text`.
PART_OVERHEAD = 400
SESSION_CHARS = 24_000

TOOL_KINDS = ("tool_use", "tool_result")
TEXT_KINDS = ("text", "thinking")

_META = """
    SELECT s.id, s.title, s.started_at, s.ended_at, s.host, s.model_primary,
           s.msg_count, s.turn_count, s.tok_in, s.tok_out, s.tok_cache_read,
           s.tok_cache_write, s.cost_usd, s.raw_path, s.meta, s.native_id,
           s.parent_session_id,
           src.kind AS source, COALESCE(w.label,'') AS workspace,
           w.key AS workspace_key
    FROM session s
    JOIN source src ON src.id = s.source_id
    LEFT JOIN workspace w ON w.id = s.workspace_id
    WHERE s.id = ?
"""


def iso(ms: int | None) -> str | None:
    """Epoch ms to UTC ISO-8601. Everything stored is UTC ms; nothing else reads dates."""
    if not ms:
        return None
    return (datetime.fromtimestamp(ms / 1000, timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"))


def parse_day(value: str | None) -> int | None:
    """`YYYY-MM-DD` to the epoch ms of its UTC midnight, for --since / --until."""
    if not value:
        return None
    try:
        return int(datetime.strptime(value, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}") from None


def _text(value: str | None, limit: int) -> dict:
    body = (value or "").strip()
    return {"text": body[:limit], "chars": len(body), "truncated": len(body) > limit}


def session_row(con: sqlite3.Connection, session_id: int) -> sqlite3.Row | None:
    return con.execute(_META, (session_id,)).fetchone()


def session_brief(row: sqlite3.Row) -> dict:
    """The header every payload carries: enough to cite a session without opening it."""
    try:
        meta = json.loads(row["meta"] or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return {
        "session_id": row["id"],
        "title": row["title"],
        "source": row["source"],
        "workspace": row["workspace"] or None,
        "host": row["host"],
        # which assistant answered, where one store holds several (vscode_chat)
        "participant": meta.get("participant"),
        "started_at": iso(row["started_at"]),
        "ended_at": iso(row["ended_at"]),
        "model": row["model_primary"],
        "messages": row["msg_count"],
        "turns": row["turn_count"],
        "tokens": {"in": row["tok_in"], "out": row["tok_out"],
                   "cache_read": row["tok_cache_read"],
                   "cache_write": row["tok_cache_write"]},
        "cost_usd": row["cost_usd"],
        "raw_path": row["raw_path"],
        # Where the conversation still lives, so a reader can cite the real thread
        # rather than an archive id — or be told plainly that it cannot be reached.
        "open": reopen.resolve(row).as_dict(),
    }


def _hit(hit: Hit, snippet_chars: int,
         targets: dict[int, reopen.Target] | None = None) -> dict:
    payload = {
        "session_id": hit.session_id,
        "title": hit.title,
        "source": hit.source,
        "workspace": hit.workspace,
        "host": hit.host,
        "started_at": iso(hit.started_at),
        "score": hit.score,
        "matched_by": hit.matched_by,
        # each retriever's own position for this session before fusion, or null where
        # that retriever did not return it — the thing to read when a ranking surprises
        "ranks": {"keyword": hit.keyword_rank, "semantic": hit.semantic_rank},
        "snippets": [{"part_id": s.part_id, "kind": s.kind, "role": s.role,
                      **_text(s.text, snippet_chars)}
                     for s in hit.snippets],
    }
    # A `Hit` knows nothing about native ids, so the targets are resolved in one query
    # by the caller and handed in here rather than looked up per result.
    target = (targets or {}).get(hit.session_id)
    if target is not None:
        payload["open"] = target.as_dict()
    return payload


def _hits(con: sqlite3.Connection, hits: list[Hit], snippet_chars: int) -> list[dict]:
    targets = reopen.targets_for(con, [h.session_id for h in hits])
    return [_hit(h, snippet_chars, targets) for h in hits]


def search_payload(con: sqlite3.Connection, vectors_dir: Path, query: str, *,
                   limit: int = 10, filters: Filters | None = None,
                   mode: str = "hybrid",
                   snippet_chars: int = SNIPPET_CHARS) -> dict:
    hits = run_search(con, vectors_dir, query, limit=limit, filters=filters, mode=mode)
    return {"query": query, "mode": mode, "count": len(hits),
            "results": _hits(con, hits, snippet_chars)}


def related_payload(con: sqlite3.Connection, vectors_dir: Path, session_id: int, *,
                    limit: int = 10, filters: Filters | None = None,
                    mode: str = "hybrid",
                    snippet_chars: int = SNIPPET_CHARS) -> dict | None:
    """None when the session does not exist — the caller decides what that means."""
    row = session_row(con, session_id)
    if row is None:
        return None
    hits = run_related(con, vectors_dir, session_id, limit=limit,
                       filters=filters, mode=mode)
    return {"session": session_brief(row), "mode": mode, "count": len(hits),
            "results": _hits(con, hits, snippet_chars)}


def session_payload(con: sqlite3.Connection, session_id: int, *,
                    tools: bool = False, abandoned: bool = False,
                    part_chars: int = PART_CHARS,
                    budget: int = SESSION_CHARS) -> dict | None:
    """One session as messages and parts, in about `budget` characters of JSON.

    `budget` sizes the *document*, not the prose in it — each part is charged its text
    plus `PART_OVERHEAD` for the structure around it. A caller deciding how much of its
    context to spend is asking about the thing it will receive, and for a session made
    of many short tool steps the two differ by an order of magnitude.

    Whole parts, so it is a threshold rather than a hard ceiling: a part is kept if the
    budget was not already spent, which lets one part carry the total over. Splitting a
    reply in half to hit an exact number would cost more than the overshoot does.

    The cut is reported rather than hidden: `omitted_parts` counts what did not fit, so
    a caller that needs the rest knows to ask again with a larger budget or without tool
    traffic, which is usually most of the volume.
    """
    row = session_row(con, session_id)
    if row is None:
        return None

    kinds = TEXT_KINDS + TOOL_KINDS if tools else TEXT_KINDS
    rows = con.execute(f"""
        SELECT m.id AS mid, m.role, m.seq, m.created_at, m.model, m.on_active_path,
               p.id AS pid, p.kind, p.text, p.tool_name, p.tool_ok, p.duration_ms
        FROM message m JOIN part p ON p.message_id = m.id
        WHERE m.session_id = ? AND p.kind IN ({','.join('?' * len(kinds))})
          {'' if abandoned else 'AND m.on_active_path = 1'}
        ORDER BY m.seq, p.seq""", (session_id, *kinds)).fetchall()

    messages: dict[int, dict] = {}
    order: list[int] = []
    spent = omitted = kept = 0

    for r in rows:
        body = (r["text"] or "").strip()
        if not body:            # a signature-only thinking block, an empty tool result
            continue
        if spent >= budget:
            omitted += 1
            continue
        part = {"part_id": r["pid"], "kind": r["kind"], **_text(body, part_chars)}
        if r["tool_name"]:
            part["tool_name"] = r["tool_name"]
        if r["tool_ok"] is not None:
            part["tool_ok"] = bool(r["tool_ok"])
        if r["duration_ms"] is not None:
            part["duration_ms"] = r["duration_ms"]
        spent += len(part["text"]) + PART_OVERHEAD
        kept += 1

        if r["mid"] not in messages:
            messages[r["mid"]] = {"seq": r["seq"], "role": r["role"],
                                  "created_at": iso(r["created_at"]),
                                  "model": r["model"],
                                  "on_active_path": bool(r["on_active_path"]),
                                  "parts": []}
            order.append(r["mid"])
        messages[r["mid"]]["parts"].append(part)

    return {
        "session": session_brief(row),
        # Half a conversation reads as a whole one otherwise: a resumed session replays
        # its predecessor, so a caller citing this transcript needs to know the rest of it
        # is filed under another id.
        "lineage": lineage.chain(con, session_id),
        "included_kinds": list(kinds),
        "messages": [messages[mid] for mid in order],
        "parts": kept,
        "omitted_parts": omitted,
        "truncated": omitted > 0,
    }


# --- derived tool facts ------------------------------------------------------
#
# Both reads join back up to `message`/`session`/`source`/`workspace` so `Filters.sql()`
# applies unchanged -- its clauses are written against those aliases. That is worth the
# joins: `--source`, `--workspace`, `--since` and `--abandoned` all work here for free,
# and the last one matters more than it looks. An edit made on a branch that was later
# rewound is still in `part`, and excluding it is exactly what "which sessions actually
# changed this file" means.

_FACT_FROM = """
  FROM touched_file tf
  JOIN part p    ON p.id = tf.part_id
  JOIN message m ON m.id = p.message_id
  JOIN session s ON s.id = m.session_id
  JOIN source src ON src.id = s.source_id
  LEFT JOIN workspace w ON w.id = s.workspace_id
"""

_CMD_FROM = """
  FROM command c
  JOIN part p    ON p.id = c.part_id
  JOIN message m ON m.id = p.message_id
  JOIN session s ON s.id = m.session_id
  JOIN source src ON src.id = s.source_id
  LEFT JOIN workspace w ON w.id = s.workspace_id
"""

WRITE_ACTIONS = ("write", "edit", "delete")


def _where(filters, extra: list[str], params: list) -> tuple[str, list]:
    clause, fparams = (filters.sql() if filters is not None else ("", ()))
    parts = [x for x in ([clause] if clause else []) + extra if x]
    return (" WHERE " + " AND ".join(parts) if parts else ""), params + list(fparams)


def file_match(query: str, exact: bool = False) -> tuple[str, list, str]:
    """How a path query is matched, and the name of the strategy that fired.

    Three strategies, most specific first, because the three ways people name a file
    are genuinely different questions. `db.py` means "anywhere"; `core/db.py` means
    "this path, under any root, on any machine"; an absolute path means that file.
    The strategy is reported back so a surprising result set explains itself.
    """
    from .core import toolinput

    raw = (query or "").strip()
    if not raw:
        return "", [], "none"

    normal = toolinput.normalise_path(raw)
    norm = normal.norm if normal else raw.replace(chr(92), "/").casefold()

    if exact:
        return "tf.norm = ?", [norm], "exact"
    if "/" not in raw and chr(92) not in raw:
        return "tf.base = ?", [raw.casefold()], "name"
    # A path fragment matches any root: the same file reached from a Windows checkout
    # and over SSH is the same file.
    return "(tf.rel = ? OR tf.norm = ? OR tf.norm LIKE ?)", [norm, norm, f"%/{norm}"], "path"


def who_touched_payload(con: sqlite3.Connection, query: str, *, limit: int = 20,
                        actions: tuple[str, ...] = (), writes_only: bool = False,
                        exact: bool = False, filters=None) -> dict:
    """Which sessions read, wrote or edited a file."""
    match, params, strategy = file_match(query, exact)
    if not match:
        return {"query": query, "matched": "none", "count": 0,
                "calls": 0, "results": []}

    extra = [match]
    if writes_only:
        extra.append(f"tf.action IN ({','.join('?' * len(WRITE_ACTIONS))})")
        params = params + list(WRITE_ACTIONS)
    elif actions:
        extra.append(f"tf.action IN ({','.join('?' * len(actions))})")
        params = params + list(actions)

    where, params = _where(filters, extra, params)
    rows = con.execute(f"""
        SELECT tf.session_id, COUNT(*) AS calls,
               SUM(CASE WHEN tf.action IN ('write','edit','delete') THEN 1 ELSE 0 END)
                   AS writes,
               MIN(tf.at) AS first_at, MAX(tf.at) AS last_at,
               GROUP_CONCAT(DISTINCT tf.action) AS actions,
               MIN(tf.path) AS path, MIN(tf.norm) AS norm
        {_FACT_FROM} {where}
        GROUP BY tf.session_id
        ORDER BY last_at DESC
        LIMIT ?""", (*params, limit)).fetchall()

    results = []
    for row in rows:
        brief_row = session_row(con, row["session_id"])
        results.append({
            **(session_brief(brief_row) if brief_row is not None
               else {"session_id": row["session_id"]}),
            "path": row["path"],
            "norm": row["norm"],
            "calls": row["calls"],
            "writes": row["writes"],
            "actions": sorted((row["actions"] or "").split(",")),
            "first_at": iso(row["first_at"]),
            "last_at": iso(row["last_at"]),
        })
    return {"query": query, "matched": strategy, "count": len(results),
            "calls": sum(r["calls"] for r in results), "results": results}


def commands_payload(con: sqlite3.Connection, substring: str | None = None, *,
                     limit: int = 20, program: str | None = None,
                     filters=None, text_chars: int = PART_CHARS) -> dict:
    """Shell commands an agent ran, or -- with no substring -- what gets run most.

    The two modes answer different questions. "What did I run" wants the occurrences;
    "what do I run" wants the ranking, and is the discoverable entry point.
    """
    extra, params = [], []
    if substring:
        extra.append("c.text LIKE ?")
        params.append(f"%{substring}%")
    if program:
        extra.append("c.argv0 = ?")
        params.append(program.casefold())
    where, params = _where(filters, extra, params)

    if not substring and not program:
        rows = con.execute(f"""
            SELECT c.argv0, COUNT(*) AS runs,
                   COUNT(DISTINCT c.session_id) AS sessions, MAX(c.at) AS last_at
            {_CMD_FROM} {where}
            GROUP BY c.argv0 ORDER BY runs DESC LIMIT ?""",
            (*params, limit)).fetchall()
        programs = []
        for row in rows:
            subs = con.execute(f"""
                SELECT c.subcommand, COUNT(*) AS runs
                {_CMD_FROM} {where} {'AND' if where else 'WHERE'}
                      c.argv0 = ? AND c.subcommand IS NOT NULL
                GROUP BY c.subcommand ORDER BY runs DESC LIMIT 3""",
                (*params, row["argv0"])).fetchall()
            programs.append({
                "program": row["argv0"], "runs": row["runs"],
                "sessions": row["sessions"], "last_at": iso(row["last_at"]),
                "top_subcommands": [{"subcommand": x["subcommand"], "runs": x["runs"]}
                                    for x in subs],
            })
        return {"query": None, "mode": "programs", "count": len(programs),
                "programs": programs}

    rows = con.execute(f"""
        SELECT c.id, c.session_id, c.at, c.argv0, c.subcommand, c.text, c.shell,
               c.cwd, c.ok, c.duration_ms,
               s.title, src.kind AS source, COALESCE(w.label,'') AS workspace
        {_CMD_FROM} {where}
        ORDER BY c.at DESC LIMIT ?""", (*params, limit)).fetchall()
    return {
        "query": substring, "mode": "runs", "count": len(rows),
        "results": [{
            "session_id": row["session_id"],
            "title": row["title"],
            "source": row["source"],
            "workspace": row["workspace"] or None,
            "at": iso(row["at"]),
            "program": row["argv0"],
            "subcommand": row["subcommand"],
            "shell": row["shell"],
            "cwd": row["cwd"],
            "ok": None if row["ok"] is None else bool(row["ok"]),
            "duration_ms": row["duration_ms"],
            **_text(row["text"], text_chars),
        } for row in rows],
    }


def session_files(con: sqlite3.Connection, session_id: int,
                  limit: int = 40) -> list[dict]:
    """What one session opened, wrote and ran. Grouped by `rel` so the same file
    reached from two roots is one row."""
    rows = con.execute("""
        SELECT COALESCE(tf.rel, tf.norm) AS key, MIN(tf.path) AS path,
               COUNT(*) AS calls,
               SUM(CASE WHEN tf.action IN ('write','edit','delete') THEN 1 ELSE 0 END)
                   AS writes,
               GROUP_CONCAT(DISTINCT tf.action) AS actions,
               MAX(tf.at) AS last_at
        FROM touched_file tf WHERE tf.session_id = ?
        GROUP BY key ORDER BY writes DESC, calls DESC LIMIT ?""",
        (session_id, limit)).fetchall()
    return [{"key": r["key"], "path": r["path"], "calls": r["calls"],
             "writes": r["writes"], "actions": sorted((r["actions"] or "").split(",")),
             "last_at": iso(r["last_at"])} for r in rows]


def session_commands(con: sqlite3.Connection, session_id: int,
                     limit: int = 10) -> list[dict]:
    """The programs one session ran. A 6,000-command session is not a list."""
    rows = con.execute("""
        SELECT argv0, COUNT(*) AS runs, MAX(at) AS last_at
        FROM command WHERE session_id = ?
        GROUP BY argv0 ORDER BY runs DESC LIMIT ?""", (session_id, limit)).fetchall()
    return [{"program": r["argv0"], "runs": r["runs"], "last_at": iso(r["last_at"])}
            for r in rows]


# --- git blame bridge --------------------------------------------------------
#
# line -> commit -> the sessions in the commit's window. `core/gitblame.py` supplies the
# commits and each one's window; this is the join. Two kinds of evidence, kept apart in
# the payload because they mean different things: a session that *edited the file*
# inside the window is the one that wrote the code, and a session that *ran the git
# commit* is the one that decided it was done. Usually the same session; not always.

# How far a transcript's clock may sit from git's. Same machine almost always, but a
# session over SSH is timestamped by the local client while the commit is stamped by the
# remote, and the two are not synchronised to the second.
BLAME_SLACK_MS = 5 * 60 * 1000

# `git add -A && git commit -m ...` is one `command` row whose subcommand is `add`, so
# the commit is found in the text rather than by column: `git` at the head of a shell
# segment, allowed its own options (`-c k=v`, `-C dir`), then `commit`. Anchoring on
# the segment is what keeps a docstring that *mentions* `git commit` inside a heredoc
# from counting as one.
_GIT_COMMIT = re.compile(
    r"(?:^|[;&|(]|\n)\s*(?:\w+=\S*\s+)*(?:sudo\s+)?git\s+"
    r"(?:(?:-[a-zA-Z]\S*|--\S+)\s+(?:[^-\s]\S*\s+)?)*commit\b")
_COMMIT_MSG_FLAG = re.compile(r"(?:^|\s)(?:-m|--message)(?:\s|=)")


def _blame_file_match(bl) -> tuple[str, list]:
    """Rows for this file: this checkout's absolute path, the repo-relative path from
    any root or machine, and any earlier name a blamed commit knew it by."""
    from .core import toolinput

    names = {bl.rel.casefold()}
    names.update(c.filename.casefold() for c in bl.commits.values() if c.filename)
    clauses, params = [], []
    absolute = toolinput.normalise_path(str(bl.repo / bl.rel))
    if absolute is not None:
        clauses.append("tf.norm = ?")
        params.append(absolute.norm)
    for name in sorted(names):
        clauses += ["tf.rel = ?", "tf.norm = ?", "tf.norm LIKE ?"]
        params += [name, name, f"%/{name}"]
    return "(" + " OR ".join(clauses) + ")", params


def _inside(cwd: str | None, repo_norm: str | None) -> bool:
    from .core import toolinput

    if not cwd or not repo_norm:
        return False
    here = toolinput.normalise_path(cwd)
    return here is not None and (here.norm == repo_norm
                                 or here.norm.startswith(repo_norm + "/"))


def _is_the_commit(commit, text: str, cwd: str | None, repo_norm: str | None) -> bool:
    """Whether one `git commit` command line made `commit`.

    Its subject in the text settles it either way: present means yes, and a different
    `-m` message means no even if the times line up. Only a command that carries no
    message at all — `-F`, `--amend --no-edit`, an editor — falls back to "run from
    inside this checkout at the right time".
    """
    if commit.summary and commit.summary in text:
        return True
    if _COMMIT_MSG_FLAG.search(text):
        return False
    return _inside(cwd, repo_norm)


def _blame_hit(path: str | None = None) -> dict:
    return {"evidence": [], "edits": 0, "first_edit_at": None, "last_edit_at": None,
            "committed_at": None, "path": path}


def blame_payload(con: sqlite3.Connection, path: str, *, start: int | None = None,
                  end: int | None = None, cwd: str | None = None, limit: int = 20,
                  filters=None) -> dict:
    """Which sessions produced each commit behind a range of lines.

    Raises `gitblame.GitError` when git cannot answer — no repository, an untracked
    file, git not installed — since the archive cannot say anything about lines it
    cannot attribute to a commit first.
    """
    from .core import gitblame, toolinput

    target = Path(path)
    if not target.is_absolute() and cwd:
        target = Path(cwd) / target
    bl = gitblame.blame(target, start, end)
    repo_np = toolinput.normalise_path(str(bl.repo))
    repo_norm = repo_np.norm if repo_np else None

    # Windows in ms, per commit: (previous commit touching the file, this commit].
    # Uncommitted lines are the working tree's: everything since the newest commit.
    windows: dict[str, tuple[int, int | None]] = {}
    for sha, commit in bl.commits.items():
        prev = bl.previous.get(sha)
        low = prev.opened_at * 1000 if prev else 0
        high = None if commit.uncommitted else commit.closed_at * 1000 + BLAME_SLACK_MS
        windows[sha] = (low, high)

    match, params = _blame_file_match(bl)
    extra = [match, f"tf.action IN ({','.join('?' * len(WRITE_ACTIONS))})",
             "COALESCE(tf.ok, 1) = 1"]         # an Edit that errored changed nothing
    where, params = _where(filters, extra, params + list(WRITE_ACTIONS))
    edits = con.execute(f"""
        SELECT tf.session_id, tf.at, tf.action, tf.path
        {_FACT_FROM} {where} ORDER BY tf.at""", params).fetchall()

    committed = [c for c in bl.commits.values() if not c.uncommitted]
    commits_run: list = []
    if committed:
        span = [min(c.opened_at for c in committed) * 1000 - BLAME_SLACK_MS,
                max(c.closed_at for c in committed) * 1000 + BLAME_SLACK_MS]
        where, cparams = _where(filters, ["instr(c.text, 'commit') > 0",
                                          "c.at >= ?", "c.at <= ?"], span)
        commits_run = [r for r in con.execute(f"""
            SELECT c.session_id, c.at, c.text, c.cwd {_CMD_FROM} {where}""", cparams)
                       if _GIT_COMMIT.search(r["text"] or "")]

    briefs: dict[int, dict] = {}

    def brief(session_id: int) -> dict:
        if session_id not in briefs:
            row = session_row(con, session_id)
            briefs[session_id] = (session_brief(row) if row is not None
                                  else {"session_id": session_id})
        return briefs[session_id]

    order = bl.in_order()
    out = []
    for sha in order[:limit]:
        commit = bl.commits[sha]
        low, high = windows[sha]
        prev = bl.previous.get(sha)
        found: dict[int, dict] = {}

        for row in edits:
            if row["at"] <= low or (high is not None and row["at"] > high):
                continue
            hit = found.setdefault(row["session_id"], _blame_hit(row["path"]))
            if "edit" not in hit["evidence"]:
                hit["evidence"].append("edit")
            hit["edits"] += 1
            hit["first_edit_at"] = hit["first_edit_at"] or iso(row["at"])
            hit["last_edit_at"] = iso(row["at"])

        if not commit.uncommitted:
            lo = commit.opened_at * 1000 - BLAME_SLACK_MS
            for row in commits_run:
                if not (lo <= row["at"] <= high):
                    continue
                if not _is_the_commit(commit, row["text"] or "", row["cwd"], repo_norm):
                    continue
                hit = found.setdefault(row["session_id"], _blame_hit())
                if "commit" not in hit["evidence"]:
                    hit["evidence"].insert(0, "commit")
                hit["committed_at"] = hit["committed_at"] or iso(row["at"])

        sessions = [{**brief(sid), **hit} for sid, hit in found.items()]
        sessions.sort(key=lambda s: ("commit" not in s["evidence"], -s["edits"],
                                     s["last_edit_at"] or ""))
        ranges = bl.ranges(sha)
        out.append({
            "sha": None if commit.uncommitted else sha,
            "short": None if commit.uncommitted else sha[:7],
            "uncommitted": commit.uncommitted,
            "author": None if commit.uncommitted else commit.author,
            "summary": "not committed yet" if commit.uncommitted else commit.summary,
            "authored_at": None if commit.uncommitted else iso(commit.author_time * 1000),
            "committed_at": (None if commit.uncommitted
                             else iso(commit.committer_time * 1000)),
            "filename": commit.filename,
            "lines": [list(r) for r in ranges],
            "line_count": sum(b - a + 1 for a, b in ranges),
            "previous": ({"sha": prev.sha, "short": prev.sha[:7],
                          "committed_at": iso(prev.closed_at * 1000)}
                         if prev else None),
            "window": {"from": iso(low) if low else None, "to": iso(high)},
            "session_count": len(sessions),
            "sessions": sessions,
        })

    return {
        "path": path,
        "repo": str(bl.repo),
        "file": bl.rel,
        "range": {"start": bl.start, "end": bl.end},
        "lines": len(bl.lines),
        "count": len(out),
        "total": len(order),
        "commits": out,
    }
