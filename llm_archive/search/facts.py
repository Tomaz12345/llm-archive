"""Derive `touched_file` and `command` from the stored tool payloads.

Wholesale, exactly like `fts.rebuild`, and for the same two reasons: there are no
triggers anywhere in this repo, and a change to the extractor has to take effect
without re-ingesting 572 sessions.

The staleness contract, which belongs in one sentence: **these tables are as stale as
the last `llma index`, and never staler than the last ingest.** The second half is free
— `part_id` is `ON DELETE CASCADE` and ingest deletes a re-parsed message's parts, so
rows for a call that no longer exists go with it. That is a better position than FTS,
whose rowids go stale silently.

`core/toolinput.py` decides what a call did; this module only reads rows, normalises
paths against the workspace, and writes. One malformed payload must never take an index
run down, so every per-row failure is counted rather than raised.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..core import blobs as blobstore
from ..core import redact, toolinput

# Everything a fact needs, in one pass, so nothing is looked up per row.
PARTS_SQL = """
SELECT p.id AS part_id, p.tool_name, p.tool_ok, p.duration_ms,
       p.tool_input, p.tool_input_blob_id,
       tb.sha256 AS payload_sha, tb.path AS payload_path,
       m.session_id, m.created_at,
       s.workspace_id, s.meta AS session_meta,
       src.kind AS source_kind, w.key AS workspace_key
  FROM part p
  JOIN message m ON m.id = p.message_id
  JOIN session s ON s.id = m.session_id
  JOIN source src ON src.id = s.source_id
  LEFT JOIN workspace w ON w.id = s.workspace_id
  LEFT JOIN blob tb ON tb.id = p.tool_input_blob_id
 WHERE p.kind = 'tool_use'
 ORDER BY p.id
"""

BATCH = 500


@dataclass
class FactsResult:
    files: int = 0
    commands: int = 0
    payloads: int = 0          # tool_use parts that had a payload to read
    missing: int = 0           # ...and those that did not: needs `ingest --force`
    unreadable: int = 0        # payload in a blob that is not there any more
    failed: int = 0            # payload that could not be parsed at all
    unknown_tools: dict[str, int] = field(default_factory=dict)

    def drift(self, tool: str) -> None:
        self.unknown_tools[tool] = self.unknown_tools.get(tool, 0) + 1


def _session_cwd(meta_json: str | None) -> str | None:
    """The cwd a session mostly ran from, for resolving a relative path.

    `meta.cwds` is a {raw path: turns} counter the adapters already store; the most-used
    one is the same choice `reopen` makes when deciding where to resume a session.
    """
    if not meta_json:
        return None
    try:
        meta = json.loads(meta_json)
    except ValueError:
        return None
    cwds = meta.get("cwds") if isinstance(meta, dict) else None
    if isinstance(cwds, dict) and cwds:
        return max(cwds.items(), key=lambda kv: kv[1])[0]
    return None


def _payload(row, blob_dir: Path | None, result: FactsResult):
    """The call's arguments, from the column or from the blob it overflowed into.

    A payload over INLINE_LIMIT is stored truncated in the column, and a truncated head
    does not parse as JSON. Falling back to it would silently derive facts from half a
    payload, so an unreachable blob is counted and skipped instead.
    """
    if row["tool_input_blob_id"] is not None:
        if blob_dir is None or not row["payload_sha"]:
            result.unreadable += 1
            return None
        path = blobstore.blob_path(blob_dir, row["payload_sha"], row["payload_path"])
        if path is None:
            result.unreadable += 1
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            result.unreadable += 1
            return None
    return row["tool_input"]


def rebuild(con: sqlite3.Connection, blob_dir: Path | None = None) -> FactsResult:
    """Re-derive both tables from every stored tool payload."""
    result = FactsResult()
    con.execute("DELETE FROM touched_file")
    con.execute("DELETE FROM command")

    # Blob-offloaded payloads are outside `redact.apply`'s reach (the standing caveat
    # about data/blobs/), and a command line is the most leak-prone text in the archive.
    # Redacting here closes both paths at once.
    redactor = None
    try:
        if redact.is_enabled(con):
            redactor = redact.active_rules(redact.enabled_wide(con))
    except sqlite3.Error:
        redactor = None

    files: list[tuple] = []
    commands: list[tuple] = []
    cwd_cache: tuple[int, str | None] = (-1, None)

    for row in con.execute(PARTS_SQL):
        raw = _payload(row, blob_dir, result)
        if raw is None and row["tool_input_blob_id"] is None:
            result.missing += 1
            continue
        if raw is None:
            continue
        result.payloads += 1

        tool = row["tool_name"] or ""
        source = row["source_kind"]
        try:
            facts = toolinput.extract(source, tool, raw)
        except Exception:  # noqa: BLE001 - one bad payload is not an index failure
            result.failed += 1
            continue
        if not toolinput.is_known(source, tool):
            result.drift(f"{source}:{tool}")
        if not facts:
            continue

        if cwd_cache[0] != row["session_id"]:
            cwd_cache = (row["session_id"], _session_cwd(row["session_meta"]))
        cwd = cwd_cache[1]
        ws_key = row["workspace_key"]

        seen: set[tuple[str, str]] = set()
        for fact in facts.files:
            try:
                norm = toolinput.normalise_path(fact.path, cwd=cwd, workspace_key=ws_key)
            except Exception:  # noqa: BLE001
                norm = None
            if norm is None or (norm.norm, fact.action) in seen:
                continue
            seen.add((norm.norm, fact.action))
            files.append((row["part_id"], row["session_id"], row["workspace_id"],
                          row["created_at"], fact.action, tool, row["tool_ok"],
                          norm.path, norm.norm, norm.base, norm.rel, norm.host_key))

        for cmd in facts.commands:
            text = cmd.text
            if redactor is not None:
                text = redact.redact_text(text, redactor)[0]
            argv0, sub = toolinput.split_argv0(text, cmd.shell)
            if not argv0:
                continue
            ok = row["tool_ok"] if cmd.ok is None else int(cmd.ok)
            commands.append((row["part_id"], row["session_id"], row["workspace_id"],
                             row["created_at"], tool, cmd.shell, argv0, sub, text,
                             cmd.cwd or cwd, ok, row["duration_ms"]))

        if len(files) >= BATCH:
            _flush_files(con, files)
            result.files += len(files)
            files = []
        if len(commands) >= BATCH:
            _flush_commands(con, commands)
            result.commands += len(commands)
            commands = []

    _flush_files(con, files)
    result.files += len(files)
    _flush_commands(con, commands)
    result.commands += len(commands)
    con.commit()
    return result


def _flush_files(con: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        con.executemany(
            "INSERT INTO touched_file(part_id,session_id,workspace_id,at,action,"
            "tool_name,ok,path,norm,base,rel,host_key) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def _flush_commands(con: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        con.executemany(
            "INSERT INTO command(part_id,session_id,workspace_id,at,tool_name,shell,"
            "argv0,subcommand,text,cwd,ok,duration_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
