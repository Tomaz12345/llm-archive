"""Keyword search over every part, including tool output.

FTS5 indexes *all* text, not just the embeddable slice: you still want to find which
session ran a particular migration, even though 62% of the archive by volume is tool
output that would only dilute vector search.

The index is external-content (`content='part'`), so the text is stored once.

`remove_diacritics 2` handles Slovene. FTS5 ships no Slovene stemmer, so a trailing `*`
is added to each term to recover some morphology — `predlog` finds `predlogi`. That is
crude but it is the difference between finding an inflected word and not.
"""

from __future__ import annotations

import re
import sqlite3

# FTS5 treats these as syntax. A user typing "what's the C++ flag?" must not produce a
# malformed MATCH expression.
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

STOP = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "on", "with", "is",
        "are", "was", "were", "be", "been", "that", "this", "it", "its", "from", "at",
        "as", "how", "why", "what", "when", "i", "my", "me", "do", "did", "does"}


def rebuild(con: sqlite3.Connection) -> int:
    """Repopulate the whole index from `part`. Cheap enough to always do in full."""
    con.execute("INSERT INTO part_fts(part_fts) VALUES('rebuild')")
    con.commit()
    return con.execute("SELECT COUNT(*) n FROM part WHERE text IS NOT NULL").fetchone()["n"]


def build_match(query: str, prefix: bool = True) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression, or None if nothing usable."""
    terms = [t for t in TOKEN_RE.findall(query)]
    kept = [t for t in terms if len(t) > 2 and t.lower() not in STOP] or terms
    if not kept:
        return None
    parts = []
    for term in kept[:24]:
        safe = term.replace('"', "")
        if not safe:
            continue
        # quote to neutralise syntax, then optionally prefix-match for morphology
        parts.append(f'"{safe}"*' if prefix and len(safe) > 3 else f'"{safe}"')
    return " OR ".join(parts) if parts else None


def search(con: sqlite3.Connection, query: str, limit: int = 60,
           where: str = "", params: tuple = ()) -> list[tuple[int, float]]:
    """Return (part_id, bm25_score) best-first. bm25() is negative-is-better in SQLite."""
    match = build_match(query)
    if not match:
        return []
    # source and workspace are joined even when unfiltered: callers build WHERE
    # fragments that may reference src.kind or w.label.
    sql = f"""
        SELECT p.id AS part_id, bm25(part_fts) AS score
        FROM part_fts
        JOIN part p ON p.id = part_fts.rowid
        JOIN message m ON m.id = p.message_id
        JOIN session s ON s.id = m.session_id
        JOIN source src ON src.id = s.source_id
        LEFT JOIN workspace w ON w.id = s.workspace_id
        WHERE part_fts MATCH ? {where}
        ORDER BY score
        LIMIT ?
    """
    try:
        rows = con.execute(sql, (match, *params, limit)).fetchall()
    except sqlite3.OperationalError:
        # a term combination FTS5 still refuses: retry without prefix matching
        match = build_match(query, prefix=False)
        if not match:
            return []
        rows = con.execute(sql, (match, *params, limit)).fetchall()
    return [(r["part_id"], -float(r["score"])) for r in rows]
