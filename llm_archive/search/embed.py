"""Local embedding model and the vector store.

Model: `paraphrase-multilingual-MiniLM-L12-v2` (118M, 384-dim), chosen in Phase 0 on
measured numbers rather than a leaderboard — see docs/phase0-findings.md §6.2d. It beat
the larger mpnet on R@10 (0.886 vs 0.829) at 6x the throughput, and the MRR gap sat
inside the noise band at n=35. Multilingual is required: session titles are English
while ~14% of the corpus is Slovene, so the real task is cross-lingual.

Everything runs locally. Google's Gemini free tier would embed at no charge, but its
terms permit training on submissions and human review of them — the wrong trade for an
index of your entire chat history (§1.3).

Store: a numpy memmap, not a vector database. At the realistic ceiling (~50K chunks x
384 dims = 77 MB float32) a brute-force matmul takes ~10 ms. FAISS/Chroma/sqlite-vec buy
nothing at this scale and add a dependency that breaks on Python upgrades.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_TAG = "pm-MiniLM-L12"
DIM = 384

# e5-family models need "query: " / "passage: " prefixes; this one does not.
NEEDS_PREFIX = {"intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large"}


_EMBEDDERS: dict[tuple[str, str], "Embedder"] = {}


def get_embedder(model_name: str = DEFAULT_MODEL, tag: str = MODEL_TAG) -> "Embedder":
    """Process-wide cache.

    Loading the ONNX model costs seconds. Constructing a fresh Embedder per query — as
    the first cut of hybrid search did — pays that on every single search, which is
    invisible in a one-off CLI call and crippling anywhere that runs many (the tuner, a
    web UI serving requests).
    """
    key = (model_name, tag)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = Embedder(model_name, tag)
    return _EMBEDDERS[key]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, tag: str = MODEL_TAG):
        self.model_name = model_name
        self.tag = tag
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if self.model_name in NEEDS_PREFIX:
            return [f"{kind}: {t}" for t in texts]
        return texts

    def encode_passages(self, texts: list[str], batch: int = 512):
        """Yield (offset, vectors) so callers can stream progress on a slow CPU.

        Batch size is a real throughput knob, not just a progress granularity. Each call
        re-enters fastembed's pipeline and gives up its internal batching, so small
        windows cost measurable speed: the first full index ran at 2.8 chunks/s with
        batch=64, against 4.1 chunks/s when Phase 0's benchmark passed the whole list at
        once. 512 keeps progress readable (~20 updates over 10k chunks) without paying
        most of that penalty.
        """
        model = self._load()
        prepared = self._prefix(texts, "passage")
        for start in range(0, len(prepared), batch):
            window = prepared[start:start + batch]
            vecs = np.asarray(list(model.embed(window)), dtype=np.float32)
            yield start, _normalise(vecs)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = np.asarray(list(model.embed(self._prefix(texts, "query"))),
                          dtype=np.float32)
        return _normalise(vecs)


def _normalise(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


class VectorStore:
    """One .npy per model tag; row index is `chunk.vec_row`."""

    def __init__(self, root: Path, tag: str = MODEL_TAG):
        self.root = root
        self.tag = tag
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.root / f"archive.{self.tag}.npy"

    def save(self, vectors: np.ndarray) -> None:
        np.save(self.path, vectors.astype(np.float32))

    def load(self) -> np.ndarray | None:
        if not self.path.exists():
            return None
        return np.load(self.path, mmap_mode="r")

    def search(self, query_vec: np.ndarray, top: int = 60) -> list[tuple[int, float]]:
        """Brute-force cosine over the whole store. Returns (vec_row, score)."""
        matrix = self.load()
        if matrix is None or len(matrix) == 0:
            return []
        scores = np.asarray(matrix) @ query_vec.reshape(-1)
        top = min(top, len(scores))
        rows = np.argpartition(-scores, top - 1)[:top]
        rows = rows[np.argsort(-scores[rows])]
        return [(int(r), float(scores[r])) for r in rows]
