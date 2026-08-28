"""Hybrid retrieval: BM25 + dense vectors, fused by rank.

Phase 0 measured why both are needed. With queries that reuse the conversation's own
words, BM25 wins (MRR 0.863 vs 0.830). With paraphrased queries — the reason this
project exists — **BM25 collapses**: MRR falls 66% and it ranks the right session first
for one query in nine (R@1 0.114). Dense degrades gracefully instead.

You cannot tell in advance which kind a query is, so both run and the ranks are fused.

Fusion is **weighted** RRF, not plain RRF. Two independent measurements showed unweighted
fusion actively hurting: it cost mpnet 0.077 MRR against dense-alone, and cost MiniLM
0.057 R@10. Mixing in a retriever that is near-random for a given query evicts correct
results from the deep tail. Weighting the dense side higher keeps the ranking gain
without paying that price; `tools/tune_fusion.py` fits the weight on the eval set.

Scoring is session-level because that is what a person is looking for — "the chat where
I worked out the offside thing" — with the best-matching parts returned as snippets.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import fts
from .chunker import strip_header
from .embed import DEFAULT_MODEL, MODEL_TAG, VectorStore, get_embedder

RRF_K = 60

# (keyword, semantic). Fitted by tools/tune_fusion.py against the live archive on 35
# paraphrase queries — the case that matters, where you search in words you did not
# originally use:
#
#   keyword only     R@1 0.200   R@10 0.543   MRR 0.277
#   semantic only    R@1 0.457   R@10 0.771   MRR 0.560
#   1.0 / 1.0        R@1 0.343   R@10 0.771   MRR 0.469   <- plain RRF, worse than either
#   1.0 / 3.0        R@1 0.400   R@10 0.800   MRR 0.514   <- best fusion
#
# Semantic alone still beats fusion on MRR. BM25 is kept anyway, for a reason this
# benchmark cannot see: **62% of the archive is tool output, which is never embedded**
# (PLAN.md §1.1). Vector search physically cannot find the session that ran a given
# command or hit a given stack trace; only FTS can. The eval measures conversational
# recall, so it understates keyword's job rather than proving it useless.
DEFAULT_WEIGHTS = (1.0, 3.0)

CANDIDATES = 80                  # per retriever, before fusion


@dataclass
class Snippet:
    part_id: int
    kind: str
    text: str
    role: str


@dataclass
class Hit:
    session_id: int
    score: float
    title: str | None
    source: str
    workspace: str | None
    host: str | None
    started_at: int | None
    matched_by: str                      # keyword | semantic | both
    snippets: list[Snippet] = field(default_factory=list)


@dataclass
class Filters:
    sources: tuple[str, ...] = ()
    # which assistant answered inside a shared panel (vscode_chat only, for now):
    # 'copilot', 'remote-ssh', … See adapters/vscode_chat.py PARTICIPANTS.
    participant: str | None = None
    workspace: str | None = None
    host: str | None = None
    since: int | None = None
    until: int | None = None
    include_abandoned: bool = False

    def sql(self) -> tuple[str, tuple]:
        clauses, params = [], []
        if self.sources:
            clauses.append(f"src.kind IN ({','.join('?' * len(self.sources))})")
            params.extend(self.sources)
        if self.participant:
            clauses.append("json_extract(s.meta,'$.participant') = ?")
            params.append(self.participant)
        if self.workspace:
            clauses.append("LOWER(COALESCE(w.label,'')) LIKE ?")
            params.append(f"%{self.workspace.lower()}%")
        if self.host:
            clauses.append("COALESCE(s.host,'') = ?")
            params.append(self.host)
        if self.since:
            clauses.append("s.started_at >= ?")
            params.append(self.since)
        if self.until:
            clauses.append("s.started_at <= ?")
            params.append(self.until)
        if not self.include_abandoned:
            clauses.append("m.on_active_path = 1")
        return (" AND ".join(clauses), tuple(params))


def _rank_map(pairs: list[tuple[int, float]]) -> dict[int, int]:
    return {key: rank for rank, (key, _) in enumerate(pairs, start=1)}


def search(con: sqlite3.Connection, vectors_dir: Path, query: str, *,
           limit: int = 20, filters: Filters | None = None,
           mode: str = "hybrid", weights: tuple[float, float] = DEFAULT_WEIGHTS,
           model_name: str = DEFAULT_MODEL, model_tag: str = MODEL_TAG,
           snippets_per_hit: int = 3) -> list[Hit]:
    filters = filters or Filters()
    where, params = filters.sql()
    where_sql = f" AND {where}" if where else ""

    # ---- keyword ----
    keyword_parts: list[tuple[int, float]] = []
    if mode in ("hybrid", "keyword"):
        keyword_parts = fts.search(con, query, limit=CANDIDATES,
                                   where=where_sql, params=params)

    # ---- semantic ----
    semantic_parts: list[tuple[int, float]] = []
    if mode in ("hybrid", "semantic"):
        semantic_parts = _semantic(con, vectors_dir, query, filters,
                                   model_name, model_tag)

    if not keyword_parts and not semantic_parts:
        return []

    kw_sessions = _to_sessions(con, keyword_parts)
    sem_sessions = _to_sessions(con, semantic_parts)

    kw_rank = _rank_map(kw_sessions)
    sem_rank = _rank_map(sem_sessions)

    w_kw, w_sem = weights
    fused: dict[int, float] = {}
    for sid, rank in kw_rank.items():
        fused[sid] = fused.get(sid, 0.0) + w_kw / (RRF_K + rank)
    for sid, rank in sem_rank.items():
        fused[sid] = fused.get(sid, 0.0) + w_sem / (RRF_K + rank)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
    if not ordered:
        return []

    best_parts: dict[int, list[int]] = {}
    for pid, sid in _part_sessions(con, [p for p, _ in keyword_parts + semantic_parts]):
        best_parts.setdefault(sid, []).append(pid)

    hits: list[Hit] = []
    for sid, score in ordered:
        meta = con.execute("""
            SELECT s.id, s.title, s.started_at, s.host,
                   src.kind AS source, COALESCE(w.label,'') AS workspace
            FROM session s
            JOIN source src ON src.id = s.source_id
            LEFT JOIN workspace w ON w.id = s.workspace_id
            WHERE s.id = ?""", (sid,)).fetchone()
        if meta is None:
            continue
        in_kw, in_sem = sid in kw_rank, sid in sem_rank
        hits.append(Hit(
            session_id=sid, score=round(score, 6), title=meta["title"],
            source=meta["source"], workspace=meta["workspace"] or None,
            host=meta["host"], started_at=meta["started_at"],
            matched_by="both" if in_kw and in_sem else
                       ("keyword" if in_kw else "semantic"),
            snippets=_snippets(con, best_parts.get(sid, [])[:snippets_per_hit]),
        ))
    return hits


def _semantic(con, vectors_dir: Path, query: str, filters: Filters,
              model_name: str, model_tag: str) -> list[tuple[int, float]]:
    store = VectorStore(vectors_dir, model_tag)
    if store.load() is None:
        return []
    qvec = get_embedder(model_name, model_tag).encode_queries([query])[0]
    rows = store.search(qvec, top=CANDIDATES * 3)
    if not rows:
        return []

    by_row = {r: s for r, s in rows}
    placeholders = ",".join("?" * len(by_row))
    where, params = filters.sql()
    where_sql = f" AND {where}" if where else ""
    sql = f"""
        SELECT c.vec_row, c.part_id
        FROM chunk c
        JOIN part p ON p.id = c.part_id
        JOIN message m ON m.id = p.message_id
        JOIN session s ON s.id = m.session_id
        JOIN source src ON src.id = s.source_id
        LEFT JOIN workspace w ON w.id = s.workspace_id
        WHERE c.model_tag = ? AND c.vec_row IN ({placeholders}){where_sql}
    """
    found = con.execute(sql, (model_tag, *by_row.keys(), *params)).fetchall()
    scored: dict[int, float] = {}
    for row in found:
        score = by_row.get(row["vec_row"], 0.0)
        pid = row["part_id"]
        if score > scored.get(pid, -1.0):
            scored[pid] = score
    return sorted(scored.items(), key=lambda kv: -kv[1])[:CANDIDATES]


def _to_sessions(con, parts: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Collapse ranked parts to ranked sessions, keeping each session's best score."""
    if not parts:
        return []
    scores = dict(parts)          # built once; rebuilding it per row was O(n^2)
    best: dict[int, float] = {}
    for pid, sid in _part_sessions(con, [p for p, _ in parts]):
        score = scores.get(pid, 0.0)
        if score > best.get(sid, float("-inf")):
            best[sid] = score
    return sorted(best.items(), key=lambda kv: -kv[1])


def _part_sessions(con, part_ids: list[int]) -> list[tuple[int, int]]:
    if not part_ids:
        return []
    out = []
    for start in range(0, len(part_ids), 400):
        window = part_ids[start:start + 400]
        placeholders = ",".join("?" * len(window))
        out.extend((r["pid"], r["sid"]) for r in con.execute(
            f"""SELECT p.id AS pid, m.session_id AS sid
                FROM part p JOIN message m ON m.id = p.message_id
                WHERE p.id IN ({placeholders})""", window))
    return out


def _snippets(con, part_ids: list[int]) -> list[Snippet]:
    if not part_ids:
        return []
    placeholders = ",".join("?" * len(part_ids))
    rows = con.execute(f"""
        SELECT p.id, p.kind, p.text, m.role
        FROM part p JOIN message m ON m.id = p.message_id
        WHERE p.id IN ({placeholders})""", part_ids).fetchall()
    return [Snippet(part_id=r["id"], kind=r["kind"], role=r["role"],
                    text=strip_header((r["text"] or "").strip()))
            for r in rows]
