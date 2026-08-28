"""Fit the RRF weights against the real archive, end to end.

Phase 0 established that unweighted RRF can make things *worse*: fusing a retriever that
is near-random for a given query evicts correct results from the deep tail. It cost
mpnet 0.077 MRR and MiniLM 0.057 R@10. So the weights have to be measured, not guessed.

This differs from tools/bench_embed.py in an important way: that one scored an offline
corpus, this one drives `llm_archive.search.hybrid.search()` against the live database —
the same code path the CLI and UI use. What it reports is what you will actually get.

Queries come from data/fixtures/paraphrase_queries.json, keyed by session title, and are
matched to archive sessions by title. Ground truth is the session the title names.

    python tools/tune_fusion.py
    python tools/tune_fusion.py --titles     # also score the easy (title) query set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_archive.core import db, ingest            # noqa: E402
from llm_archive.search.hybrid import Filters, search  # noqa: E402

PARAPHRASE = ROOT / "data" / "fixtures" / "paraphrase_queries.json"

# (keyword, semantic)
GRID = [
    (1.0, 0.0),    # keyword only
    (0.0, 1.0),    # semantic only
    (1.0, 1.0),    # plain RRF — the Phase 0 baseline that sometimes hurt
    (1.0, 1.3),
    (1.0, 1.6),    # current default
    (1.0, 2.0),
    (1.0, 3.0),
    (0.5, 1.0),
]
KS = (1, 5, 10)


def load_queries(con) -> list[tuple[str, int, str]]:
    """Return (query, target_session_id, kind) triples."""
    if not PARAPHRASE.exists():
        sys.exit("missing data/fixtures/paraphrase_queries.json")
    mapping = json.loads(PARAPHRASE.read_text(encoding="utf-8"))["queries"]

    by_title: dict[str, int] = {}
    for row in con.execute("SELECT id, title FROM session WHERE title IS NOT NULL"):
        by_title.setdefault(row["title"], row["id"])

    out = []
    missing = []
    for title, phrase in mapping.items():
        sid = by_title.get(title)
        if sid is None:
            missing.append(title)
            continue
        out.append((phrase, sid, "paraphrase"))
    if missing:
        print(f"note: {len(missing)} titles not in the archive "
              f"(e.g. {missing[:2]})")
    return out


def title_queries(con, limit: int = 80) -> list[tuple[str, int, str]]:
    """CONTAMINATED — kept only to document why it must not be trusted.

    `chunker.context_header()` prepends `[source · workspace · date · TITLE]` to every
    embedded chunk, so a chunk literally contains its session's title. Querying by title
    then matches its own header. FTS indexes `part.text` and never sees the header, so
    the advantage goes to the semantic side alone.

    That is why this set reports semantic MRR 0.981 against keyword 0.820, reversing the
    Phase 0 offline result where BM25 won on titles. The header is worth keeping — it is
    real context for real queries, standard contextual-retrieval practice — but titles
    can never again serve as queries against it. Fit weights on the paraphrase set.
    """
    rows = con.execute("""SELECT id, title FROM session
                          WHERE title IS NOT NULL AND LENGTH(title) > 8
                          ORDER BY msg_count DESC LIMIT ?""", (limit,)).fetchall()
    return [(r["title"], r["id"], "title") for r in rows]


def evaluate(con, vectors_dir, queries, weights, mode="hybrid") -> dict:
    hits_at = {k: 0 for k in KS}
    rr = 0.0
    for query, target, _ in queries:
        results = search(con, vectors_dir, query, limit=max(KS),
                         filters=Filters(), mode=mode, weights=weights)
        ids = [h.session_id for h in results]
        pos = ids.index(target) + 1 if target in ids else None
        for k in KS:
            if pos and pos <= k:
                hits_at[k] += 1
        if pos:
            rr += 1.0 / pos
    n = max(len(queries), 1)
    return {**{f"R@{k}": round(hits_at[k] / n, 3) for k in KS},
            "MRR": round(rr / n, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", action="store_true",
                    help="score the title query set — CONTAMINATED, see note below")
    args = ap.parse_args()

    db_path, _ = ingest.default_paths(None)
    con = db.connect(db_path)
    vectors_dir = db_path.parent / "vectors"

    n_chunks = con.execute("SELECT COUNT(*) n FROM chunk").fetchone()["n"]
    if not n_chunks:
        sys.exit("no vectors yet — run: llma index")

    queries = load_queries(con)
    print(f"archive: {n_chunks} chunks | paraphrase queries: {len(queries)}\n")

    sets = [("paraphrase", queries)]
    if args.titles:
        sets.append(("titles", title_queries(con)))

    for name, qs in sets:
        print(f"=== {name} (n={len(qs)}) ===")
        print(f"{'weights (kw, sem)':<22} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
        best = None
        for weights in GRID:
            metrics = evaluate(con, vectors_dir, qs, weights)
            label = ("keyword only" if weights == (1.0, 0.0) else
                     "semantic only" if weights == (0.0, 1.0) else
                     f"{weights[0]:.1f} / {weights[1]:.1f}")
            star = ""
            if weights not in ((1.0, 0.0), (0.0, 1.0)):
                if best is None or metrics["MRR"] > best[1]["MRR"]:
                    best = (weights, metrics)
                    star = ""
            print(f"{label:<22} {metrics['R@1']:>6} {metrics['R@5']:>6} "
                  f"{metrics['R@10']:>6} {metrics['MRR']:>6}{star}")
        if best:
            print(f"\n  best fused weights: {best[0]}  MRR={best[1]['MRR']}  "
                  f"R@10={best[1]['R@10']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
