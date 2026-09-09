"""Which sessions are continuations of other sessions, and what that costs the statistics.

A resumed conversation does not always keep writing into the file it started in. Claude Code's
`--resume` forks: it opens a new `<uuid>.jsonl` and replays the old one's records into it, so one
conversation ends up as two session rows whose opening messages are byte-identical. Nothing
noticed, and the shared prefix was counted twice in every figure on /stats.

Phase 0 measured the opposite and said so in `adapters/claude_code.py` — that finding was true
when taken (9 projects, 93 files, 23,735 records, zero forks) and is not any more (14 / 154 /
36,586, one fork). `tools/probe_resume.py` is the file-level probe that asks the question of a
`~/.claude/projects` tree; this module asks it of the archive, which covers every source at once
and is the only one of the two that can write the answer down.

Detection is over `message.native_id`, not text: a replay reuses the source's own message ids, so
a shared LEADING run of them is the signal. Not a shared set — two sessions that merely quote the
same ids somewhere in the middle are not a continuation, and `probe_resume` reports that case
separately for the same reason.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

# 1-2 shared leading ids are coincidence, not lineage. Two Gemini sessions collide on exactly
# one because that adapter synthesises ids as f"{stem}:user" rather than reading them from the
# export; four such pairs exist in this archive today and none is a continuation.
# `tools/probe_resume.py` uses the same floor for the same reason.
MIN_PREFIX = 3


def _leading_ids(con: sqlite3.Connection) -> tuple[dict[int, list[str]], dict[int, int]]:
    """Every session's active-path native_ids in order, plus the source each belongs to.

    Restricted to the active path because that is what the statistics count: an abandoned
    branch is already excluded everywhere by `metrics.ACTIVE`, so letting it into the
    comparison would find prefixes that no figure was ever double-counting. Blank ids are
    dropped rather than compared -- Codex leaves them None on plain text messages and T3 Chat
    falls back to "", and treating those as equal would match every such message to every
    other.
    """
    seqs: dict[int, list[str]] = defaultdict(list)
    source_of: dict[int, int] = {}
    for row in con.execute("""
        SELECT s.id AS sid, s.source_id, m.native_id
        FROM session s JOIN message m ON m.session_id = s.id
        WHERE m.on_active_path = 1 AND m.native_id IS NOT NULL AND m.native_id <> ''
        ORDER BY s.id, m.seq"""):
        seqs[row["sid"]].append(row["native_id"])
        source_of[row["sid"]] = row["source_id"]
    return dict(seqs), source_of


def _candidate_pairs(seqs: dict[int, list[str]],
                     source_of: dict[int, int]) -> set[tuple[int, int]]:
    """Session pairs worth a full prefix comparison: those sharing any id within one source.

    Comparing all pairs is 150k comparisons of long lists at this size and grows as the square.
    Inverting id -> owners first costs one pass and leaves five candidates out of 550 sessions,
    because sharing a message id at all is the rare event.
    """
    owners: dict[tuple[int, str], set[int]] = defaultdict(set)
    for sid, ids in seqs.items():
        src = source_of[sid]
        for native_id in ids:
            owners[(src, native_id)].add(sid)

    pairs: set[tuple[int, int]] = set()
    for sids in owners.values():
        if len(sids) < 2:
            continue
        ordered = sorted(sids)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pairs.add((a, b))
    return pairs


def _shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _would_cycle(links: dict[int, int], child: int, parent: int) -> bool:
    """True if child -> parent closes a loop. Cheap, and a loop would hang every reader."""
    seen = {child}
    node = parent
    while node is not None:
        if node in seen:
            return True
        seen.add(node)
        node = links.get(node)
    return False


def detect(con: sqlite3.Connection) -> dict:
    """Find and record every continuation. Idempotent: clears its own marks first.

    Rebuilding from scratch rather than amending is what makes this safe to run after every
    ingest. A child that grew since the last run gets a longer overlap, a session re-ingested
    from a fresh export gets re-derived, and a pair that stops matching stops being linked.
    """
    con.execute("UPDATE session SET continues_session_id = NULL, continues_overlap = NULL "
                "WHERE continues_session_id IS NOT NULL")
    con.execute("UPDATE message SET superseded = 0 WHERE superseded <> 0")

    seqs, source_of = _leading_ids(con)
    links: dict[int, int] = {}
    found: list[dict] = []

    # Longest overlap first, so when a session could be attached to several the strongest
    # claim wins and the weaker ones are the ones a cycle check has to refuse.
    scored = []
    for a, b in _candidate_pairs(seqs, source_of):
        if source_of[a] != source_of[b]:
            continue
        overlap = _shared_prefix(seqs[a], seqs[b])
        if overlap >= MIN_PREFIX:
            scored.append((overlap, a, b))
    scored.sort(key=lambda t: -t[0])

    for overlap, a, b in scored:
        # The shorter transcript is the one that was resumed; the longer carries it plus what
        # happened after. Equal lengths mean neither continues the other -- the same
        # conversation ingested twice under two ids, which is a duplicate, not lineage.
        if len(seqs[a]) == len(seqs[b]):
            continue
        parent, child = (a, b) if len(seqs[a]) < len(seqs[b]) else (b, a)
        if child in links or _would_cycle(links, child, parent):
            continue
        links[child] = parent
        found.append({"child": child, "parent": parent, "overlap": overlap,
                      "parent_messages": len(seqs[parent]),
                      "whole_parent": overlap == len(seqs[parent])})

    for link in found:
        con.execute("UPDATE session SET continues_session_id = ?, continues_overlap = ? "
                    "WHERE id = ?", (link["parent"], link["overlap"], link["child"]))
        # Mark the PARENT's copies, not the child's: the child is where the conversation
        # actually continued, so it is the live row. Marking messages rather than the whole
        # session is what makes a mid-session fork come out right -- only the replayed prefix
        # stops counting and the parent's own divergent tail still does.
        con.execute("""
            UPDATE message SET superseded = 1 WHERE id IN (
                SELECT id FROM message
                 WHERE session_id = ? AND on_active_path = 1
                   AND native_id IS NOT NULL AND native_id <> ''
                 ORDER BY seq LIMIT ?)""", (link["parent"], link["overlap"]))
    con.commit()

    return {"pairs": found, "continuations": len(found),
            "superseded_messages": sum(link["overlap"] for link in found)}


def chain(con: sqlite3.Connection, session_id: int) -> dict:
    """What this session continues and what continues it, for the session page."""
    row = con.execute("""
        SELECT s.continues_session_id AS parent, s.continues_overlap AS overlap,
               p.title AS parent_title
        FROM session s LEFT JOIN session p ON p.id = s.continues_session_id
        WHERE s.id = ?""", (session_id,)).fetchone()
    child = con.execute("""
        SELECT id, title, continues_overlap AS overlap FROM session
        WHERE continues_session_id = ?""", (session_id,)).fetchone()
    return {
        "continues": ({"id": row["parent"], "title": row["parent_title"],
                       "overlap": row["overlap"]}
                      if row is not None and row["parent"] is not None else None),
        "continued_by": ({"id": child["id"], "title": child["title"],
                          "overlap": child["overlap"]} if child is not None else None),
    }
