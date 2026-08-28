"""Phase 0 — decide the embedding model by measurement, not by leaderboard.

Two query sets, because the first one turned out to be unable to answer the question:

  titles      Session titles, harvested automatically (69 queries).
              Measured mean lexical overlap with their target session: 0.84.
              Titles are GENERATED FROM the conversation, so they reuse its words.
              This set measures "find a session you can already name" — real, but easy,
              and structurally biased toward keyword search.

  paraphrase  Hand-authored queries describing the same sessions in deliberately
              different vocabulary (data/fixtures/paraphrase_queries.json).
              This is the case hybrid search exists for: searching months later in
              words you did not originally use.

Both retrievers face identical queries, so the comparison stays fair. Reported overlap
makes the disjointness verifiable instead of asserted.

Vectors are cached per model, keyed by a hash of the corpus, so re-analysis is free.

    python tools/bench_embed.py                    # all candidates, both query sets
    python tools/bench_embed.py --quick            # smallest model only
    python tools/bench_embed.py --models e5-small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVALSET = ROOT / "data" / "fixtures" / "evalset.json"
PARAPHRASE = ROOT / "data" / "fixtures" / "paraphrase_queries.json"
VECDIR = ROOT / "data" / "vectors"
RESULTS = ROOT / "docs" / "formats" / "bench_embed.md"

KS = (1, 5, 10)

CANDIDATES = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "pm-MiniLM-L12", False),
    ("intfloat/multilingual-e5-small", "e5-small", True),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "pm-mpnet-base", False),
    ("intfloat/multilingual-e5-large", "e5-large", True),
]

STOP = set(
    "the a an of to in for and or on with is are was were be been this that it its from "
    "at by as into how why what when make create fix add use using run my me i".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9žčšćđ]+", text.lower())
            if len(w) > 2 and w not in STOP}


# --------------------------------------------------------------------------

def load_corpus():
    if not EVALSET.exists():
        sys.exit("evalset missing — run: python tools/build_evalset.py")
    data = json.loads(EVALSET.read_text(encoding="utf-8"))
    return data["docs"], data["queries"]


def build_query_sets(docs, title_queries):
    """Return {name: [query dicts]} for every query set we can construct."""
    sets = {"titles": title_queries}

    if PARAPHRASE.exists():
        mapping = json.loads(PARAPHRASE.read_text(encoding="utf-8"))["queries"]
        by_title = {q["query"]: q for q in title_queries}
        para = []
        for title, phrase in mapping.items():
            src = by_title.get(title)
            if src is None:
                continue
            para.append({**src, "query": phrase, "source_title": title})
        if para:
            sets["paraphrase"] = para
        missing = [t for t in mapping if t not in by_title]
        if missing:
            print(f"  note: {len(missing)} paraphrase titles not in evalset "
                  f"(sessions may be too thin): {missing[:3]}")

    # measure overlap so the "disjoint vocabulary" claim is verifiable
    session_words: dict[str, set[str]] = {}
    for d in docs:
        session_words.setdefault(d["session_id"], set()).update(content_words(d["text"]))
    for name, qs in sets.items():
        vals = []
        for q in qs:
            qw = content_words(q["query"])
            if not qw:
                continue
            vals.append(len(qw & session_words.get(q["target_session"], set())) / len(qw))
        print(f"  {name:11s} n={len(qs):<4} mean query/target lexical overlap = "
              f"{sum(vals)/len(vals):.3f}")
    return sets


def score(ranked_sessions, queries) -> dict:
    out = {f"recall@{k}": 0.0 for k in KS}
    rr = 0.0
    sl_hits, sl_n = 0, 0
    for ranked, q in zip(ranked_sessions, queries):
        target = q["target_session"]
        pos = ranked.index(target) + 1 if target in ranked else None
        for k in KS:
            if pos and pos <= k:
                out[f"recall@{k}"] += 1
        if pos:
            rr += 1.0 / pos
        if q.get("slovene_session"):
            sl_n += 1
            if pos and pos <= 10:
                sl_hits += 1
    n = max(len(queries), 1)
    for k in KS:
        out[f"recall@{k}"] = round(out[f"recall@{k}"] / n, 3)
    out["MRR"] = round(rr / n, 3)
    out["sl_recall@10"] = round(sl_hits / sl_n, 3) if sl_n else None
    return out


def rank_sessions(sim, doc_sessions, top_docs: int = 60):
    ranked = []
    for row in np.argsort(-sim, axis=1)[:, :top_docs]:
        seen, out = set(), []
        for d in row:
            s = doc_sessions[d]
            if s not in seen:
                seen.add(s)
                out.append(s)
        ranked.append(out)
    return ranked


def run_bm25(docs, queries):
    con = sqlite3.connect(":memory:")
    try:
        con.execute('CREATE VIRTUAL TABLE d USING fts5(text, '
                    'tokenize="unicode61 remove_diacritics 2")')
    except sqlite3.OperationalError as exc:
        print(f"  FTS5 unavailable ({exc})")
        return None
    con.executemany("INSERT INTO d(rowid, text) VALUES (?,?)",
                    [(i, d["text"]) for i, d in enumerate(docs)])
    con.commit()
    doc_sessions = [d["session_id"] for d in docs]
    ranked = []
    for q in queries:
        terms = [t for t in re.findall(r"\w+", q["query"]) if len(t) > 2]
        seen, out = set(), []
        if terms:
            expr = " OR ".join(f'"{t}"' for t in terms)
            try:
                for (rowid,) in con.execute(
                    "SELECT rowid FROM d WHERE d MATCH ? ORDER BY bm25(d) LIMIT 60", (expr,)
                ):
                    s = doc_sessions[rowid]
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
            except sqlite3.OperationalError:
                pass
        ranked.append(out)
    con.close()
    return ranked


def embed_docs(name, label, prefixes, docs, corpus_hash):
    """Encode the corpus once per model, cached on disk."""
    VECDIR.mkdir(parents=True, exist_ok=True)
    slug = label.replace("/", "_")
    cache = VECDIR / f"{slug}.{corpus_hash}.npy"
    if cache.exists():
        print(f"  cached vectors: {cache.name}")
        return np.load(cache), 0.0

    from fastembed import TextEmbedding
    try:
        model = TextEmbedding(model_name=name)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! cannot load ({type(exc).__name__}: {str(exc)[:110]})")
        return None, 0.0

    texts = [("passage: " + d["text"]) if prefixes else d["text"] for d in docs]
    t0 = time.perf_counter()
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    elapsed = time.perf_counter() - t0
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    np.save(cache, vecs)
    (VECDIR / f"{slug}.{corpus_hash}.json").write_text(
        json.dumps({"model": name, "encode_seconds": elapsed, "docs": len(docs)}),
        encoding="utf-8")
    return vecs, elapsed


def embed_queries(name, prefixes, queries):
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=name)
    texts = [("query: " + q["query"]) if prefixes else q["query"] for q in queries]
    qv = np.array(list(model.embed(texts)), dtype=np.float32)
    return qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)


def rrf(rank_lists, k: int = 60):
    fused = []
    for i in range(len(rank_lists[0])):
        scores: dict[str, float] = {}
        for rl in rank_lists:
            for rank, sess in enumerate(rl[i], 1):
                scores[sess] = scores.get(sess, 0.0) + 1.0 / (k + rank)
        fused.append([s for s, _ in sorted(scores.items(), key=lambda x: -x[1])])
    return fused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", nargs="*", help="labels to run")
    args = ap.parse_args()

    docs, title_queries = load_corpus()
    corpus_hash = hashlib.sha1(
        "".join(d["text"] for d in docs).encode("utf-8")).hexdigest()[:10]

    sl_sessions = {d["session_id"] for d in docs if re.search(r"[žčšŽČŠ]", d["text"])}
    for q in title_queries:
        q["slovene_session"] = q["target_session"] in sl_sessions

    print(f"corpus: {len(docs)} docs  hash={corpus_hash}")
    query_sets = build_query_sets(docs, title_queries)
    print()

    candidates = CANDIDATES
    if args.quick:
        candidates = CANDIDATES[:1]
    elif args.models:
        candidates = [c for c in CANDIDATES if c[1] in args.models]

    doc_sessions = [d["session_id"] for d in docs]
    results: dict[str, list] = {name: [] for name in query_sets}

    # --- baseline ---
    for qname, qs in query_sets.items():
        ranked = run_bm25(docs, qs)
        if ranked:
            results[qname].append(("BM25 / FTS5", "—", score(ranked, qs), None))
            query_sets[qname] = qs
            globals().setdefault("_bm25", {})[qname] = ranked

    # --- models ---
    for name, label, prefixes in candidates:
        print(f"{label}")
        dv, elapsed = embed_docs(name, label, prefixes, docs, corpus_hash)
        if dv is None:
            continue
        if elapsed:
            print(f"  encoded {len(docs)} docs in {elapsed:.1f}s "
                  f"({len(docs)/elapsed:.1f} docs/s)")
        for qname, qs in query_sets.items():
            qv = embed_queries(name, prefixes, qs)
            ranked = rank_sessions(qv @ dv.T, doc_sessions)
            m = score(ranked, qs)
            results[qname].append((label, dv.shape[1], m, elapsed or None))
            print(f"  [{qname:10s}] R@10={m['recall@10']}  MRR={m['MRR']}")

            fused = rrf([globals()["_bm25"][qname], ranked])
            mf = score(fused, qs)
            results[qname].append((f"HYBRID BM25+{label}", "—", mf, None))
            print(f"  [{qname:10s}] HYBRID R@10={mf['recall@10']}  MRR={mf['MRR']}")
        print()

    # --- report ---
    lines = [
        "# Phase 0 — embedding benchmark",
        "",
        f"Corpus: **{len(docs)} docs** (hash `{corpus_hash}`). "
        "Measured on this machine, i7-8565U, CPU only.",
        "",
        "Scoring is session-level: a document from the right session inside the top k "
        "counts as a hit, because that is what the UI returns.",
        "",
    ]
    for qname, rows in results.items():
        if not rows:
            continue
        n = len(query_sets[qname])
        lines += [
            f"## Query set: `{qname}` ({n} queries)",
            "",
            "| retriever | dim | R@1 | R@5 | R@10 | MRR | SL R@10 | encode |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, dim, m, elapsed in rows:
            enc = f"{elapsed:.0f}s" if elapsed else "—"
            lines.append(
                f"| {label} | {dim} | {m['recall@1']} | {m['recall@5']} | "
                f"{m['recall@10']} | {m['MRR']} | {m['sl_recall@10']} | {enc} |")
        lines.append("")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
