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
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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
           s.tok_cache_write, s.cost_usd, s.raw_path, s.meta,
           src.kind AS source, COALESCE(w.label,'') AS workspace
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
    }


def _hit(hit: Hit, snippet_chars: int) -> dict:
    return {
        "session_id": hit.session_id,
        "title": hit.title,
        "source": hit.source,
        "workspace": hit.workspace,
        "host": hit.host,
        "started_at": iso(hit.started_at),
        "score": hit.score,
        "matched_by": hit.matched_by,
        "snippets": [{"part_id": s.part_id, "kind": s.kind, "role": s.role,
                      **_text(s.text, snippet_chars)}
                     for s in hit.snippets],
    }


def search_payload(con: sqlite3.Connection, vectors_dir: Path, query: str, *,
                   limit: int = 10, filters: Filters | None = None,
                   mode: str = "hybrid",
                   snippet_chars: int = SNIPPET_CHARS) -> dict:
    hits = run_search(con, vectors_dir, query, limit=limit, filters=filters, mode=mode)
    return {"query": query, "mode": mode, "count": len(hits),
            "results": [_hit(h, snippet_chars) for h in hits]}


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
            "results": [_hit(h, snippet_chars) for h in hits]}


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
        "included_kinds": list(kinds),
        "messages": [messages[mid] for mid in order],
        "parts": kept,
        "omitted_parts": omitted,
        "truncated": omitted > 0,
    }
