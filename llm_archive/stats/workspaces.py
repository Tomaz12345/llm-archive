"""One project, across every agent and every machine that worked on it.

A workspace row is per-source and per-root, so one project is routinely several rows:
`document2markdown` is a Windows checkout and an SSH box, and `Language_learning` is
Claude Code and Codex. The label is what a person means by "the project", so every read
here takes a SET of workspace ids and the page shows the merge rather than hiding it.

Hot files group on `COALESCE(rel, norm)` — the workspace-relative path — which is what
makes `code/train.py` edited over SSH and the same file on Windows one row instead of
two. That merge is the whole reason `touched_file.rel` exists.
"""

from __future__ import annotations

import json
import sqlite3


def resolve(con: sqlite3.Connection, label: str,
            key: str | None = None) -> list[sqlite3.Row]:
    """Every workspace row a label names, newest-busiest first.

    `key` narrows to a single root, which is what the per-root links on the page do.
    """
    sql = """
        SELECT w.id, w.key, w.label, w.git_remote, src.kind AS source,
               COUNT(s.id) AS sessions, MIN(s.started_at) AS first_at,
               MAX(s.started_at) AS last_at
        FROM workspace w
        JOIN source src ON src.id = w.source_id
        LEFT JOIN session s ON s.workspace_id = w.id
        WHERE LOWER(COALESCE(w.label,'')) = LOWER(?)
    """
    params: list = [label]
    if key:
        sql += " AND w.key = ?"
        params.append(key)
    sql += " GROUP BY w.id ORDER BY sessions DESC, w.key"
    return con.execute(sql, params).fetchall()


def _ids(rows) -> tuple[str, list]:
    ids = [r["id"] for r in rows]
    return ",".join("?" * len(ids)), ids


def overview(con: sqlite3.Connection, rows) -> dict:
    marks, ids = _ids(rows)
    if not ids:
        return {}
    row = con.execute(f"""
        SELECT COUNT(*) AS sessions, SUM(msg_count) AS messages,
               SUM(turn_count) AS turns, SUM(tok_in) AS tok_in,
               SUM(tok_out) AS tok_out, SUM(cost_usd) AS cost_usd,
               MIN(started_at) AS first_at, MAX(started_at) AS last_at
        FROM session WHERE workspace_id IN ({marks})""", ids).fetchone()
    return dict(row) if row else {}


def hot_files(con: sqlite3.Connection, rows, limit: int = 30) -> list[dict]:
    """The files this project actually works on, most-written first."""
    marks, ids = _ids(rows)
    if not ids:
        return []
    out = con.execute(f"""
        SELECT COALESCE(tf.rel, tf.norm) AS key, MIN(tf.path) AS path,
               COUNT(*) AS calls,
               SUM(CASE WHEN tf.action IN ('write','edit','delete') THEN 1 ELSE 0 END)
                   AS writes,
               COUNT(DISTINCT tf.session_id) AS sessions,
               COUNT(DISTINCT COALESCE(tf.host_key,'')) AS hosts,
               MAX(tf.at) AS last_at
        FROM touched_file tf
        WHERE tf.workspace_id IN ({marks})
        GROUP BY key ORDER BY writes DESC, calls DESC LIMIT ?""",
        (*ids, limit)).fetchall()
    return [dict(r) for r in out]


def top_programs(con: sqlite3.Connection, rows, limit: int = 15) -> list[dict]:
    marks, ids = _ids(rows)
    if not ids:
        return []
    out = con.execute(f"""
        SELECT c.argv0 AS program, COUNT(*) AS runs,
               COUNT(DISTINCT c.session_id) AS sessions
        FROM command c WHERE c.workspace_id IN ({marks})
        GROUP BY c.argv0 ORDER BY runs DESC LIMIT ?""", (*ids, limit)).fetchall()
    return [dict(r) for r in out]


def sessions(con: sqlite3.Connection, rows, limit: int = 50, offset: int = 0,
             file: str | None = None) -> list[sqlite3.Row]:
    """The project's sessions, optionally only those that touched one file."""
    marks, ids = _ids(rows)
    if not ids:
        return []
    where = f"s.workspace_id IN ({marks})"
    params: list = list(ids)
    if file:
        where += (" AND s.id IN (SELECT tf.session_id FROM touched_file tf "
                  "WHERE COALESCE(tf.rel, tf.norm) = ?)")
        params.append(file)
    return con.execute(f"""
        SELECT s.id, s.title, s.started_at, s.msg_count, s.turn_count, s.host,
               src.kind AS source
        FROM session s JOIN source src ON src.id = s.source_id
        WHERE {where}
        ORDER BY s.started_at DESC LIMIT ? OFFSET ?""",
        (*params, limit, offset)).fetchall()


def session_count(con: sqlite3.Connection, rows, file: str | None = None) -> int:
    marks, ids = _ids(rows)
    if not ids:
        return 0
    where = f"workspace_id IN ({marks})"
    params: list = list(ids)
    if file:
        where += (" AND id IN (SELECT tf.session_id FROM touched_file tf "
                  "WHERE COALESCE(tf.rel, tf.norm) = ?)")
        params.append(file)
    return con.execute(f"SELECT COUNT(*) FROM session WHERE {where}",
                       params).fetchone()[0]


def branches(con: sqlite3.Connection, rows, limit: int = 8) -> list[dict]:
    """Git branches the sessions ran on.

    `session.meta.git_branches` has been stored by the Claude Code adapter since the
    beginning and has never been shown anywhere. It is the closest thing the archive
    has to a link between a conversation and a commit.
    """
    marks, ids = _ids(rows)
    if not ids:
        return []
    counts: dict[str, int] = {}
    for row in con.execute(
            f"SELECT meta FROM session WHERE workspace_id IN ({marks})", ids):
        try:
            meta = json.loads(row["meta"] or "{}")
        except (ValueError, TypeError):
            continue
        found = meta.get("git_branches") if isinstance(meta, dict) else None
        if isinstance(found, dict):
            for name, n in found.items():
                if isinstance(n, int):
                    counts[name] = counts.get(name, 0) + n
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return [{"branch": name, "turns": n} for name, n in ranked]
