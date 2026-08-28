# Phase 0 — embedding benchmark

Corpus: **1632 docs** (hash `b89b42ce70`). Measured on this machine, i7-8565U, CPU only.

Scoring is session-level: a document from the right session inside the top k counts as a hit, because that is what the UI returns.

## Query set: `titles` (64 queries)

| retriever | dim | R@1 | R@5 | R@10 | MRR | SL R@10 | encode |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 / FTS5 | — | 0.812 | 0.922 | 0.953 | 0.863 | 1.0 | — |
| pm-MiniLM-L12 | 384 | 0.766 | 0.906 | 0.922 | 0.83 | 0.923 | — |
| HYBRID BM25+pm-MiniLM-L12 | — | 0.797 | 0.938 | 0.953 | 0.86 | 0.923 | — |
| pm-mpnet-base | 768 | 0.797 | 0.906 | 0.938 | 0.85 | 0.923 | — |
| HYBRID BM25+pm-mpnet-base | — | 0.781 | 0.953 | 0.953 | 0.857 | 0.923 | — |

## Query set: `paraphrase` (35 queries)

| retriever | dim | R@1 | R@5 | R@10 | MRR | SL R@10 | encode |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 / FTS5 | — | 0.114 | 0.514 | 0.714 | 0.295 | 0.7 | — |
| pm-MiniLM-L12 | 384 | 0.286 | 0.686 | 0.886 | 0.446 | 0.8 | — |
| HYBRID BM25+pm-MiniLM-L12 | — | 0.343 | 0.714 | 0.829 | 0.492 | 0.8 | — |
| pm-mpnet-base | 768 | 0.4 | 0.743 | 0.829 | 0.544 | 0.8 | — |
| HYBRID BM25+pm-mpnet-base | — | 0.286 | 0.743 | 0.8 | 0.467 | 0.8 | — |
