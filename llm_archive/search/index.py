"""Build the search indexes: FTS5 over everything, vectors over the embeddable slice."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import db
from . import facts, fts, selection
from .chunker import chunk_part, context_header
from .embed import DEFAULT_MODEL, MODEL_TAG, Embedder, VectorStore


@dataclass
class IndexResult:
    fts_rows: int = 0
    chunks: int = 0
    vectors: int = 0
    seconds: float = 0.0
    chars: int = 0
    skipped_vectors: bool = False
    model_tag: str = MODEL_TAG
    topics: int = 0
    files: int = 0
    commands: int = 0
    warnings: list[str] = field(default_factory=list)


def _embeddable_parts(con: sqlite3.Connection):
    """Every part worth embedding, with the context its chunk header needs.

    The FROM/WHERE live in `selection` so the health check counts exactly this set.
    """
    return con.execute(f"""
        SELECT p.id AS part_id, p.text, p.kind,
               m.id AS message_id, m.created_at,
               s.id AS session_id, s.title,
               src.kind AS source_kind,
               COALESCE(w.label, '') AS workspace
        {selection.EMBEDDABLE_FROM}
        {selection.EMBEDDABLE_WHERE}
        ORDER BY p.id
    """).fetchall()


def build(con: sqlite3.Connection, vectors_dir: Path,
          model_name: str = DEFAULT_MODEL, model_tag: str = MODEL_TAG,
          with_vectors: bool = True, progress=None,
          blob_dir: Path | None = None) -> IndexResult:
    """Build the indexes and record that it happened.

    The record is the point: an index carries no timestamp of its own, so without one
    nothing downstream can tell a fresh index from one invalidated by a later ingest.
    """
    started = int(time.time() * 1000)
    result = _build(con, vectors_dir, model_name, model_tag, with_vectors, progress,
                    blob_dir)
    db.record_index_run(con, started, result)
    return result


def _build(con: sqlite3.Connection, vectors_dir: Path,
           model_name: str = DEFAULT_MODEL, model_tag: str = MODEL_TAG,
           with_vectors: bool = True, progress=None,
           blob_dir: Path | None = None) -> IndexResult:
    result = IndexResult(model_tag=model_tag)
    t0 = time.perf_counter()

    result.fts_rows = fts.rebuild(con)

    # Derived here, beside FTS, and deliberately ABOVE the --no-vectors return below:
    # both are wholesale rebuilds from `part`, neither needs a vector, and a
    # keyword-only run must not leave the archive with a fresh keyword index and a
    # month-old answer to "who touched this file". Wrapped for the same reason the
    # topic build is: search has to survive a derivation failure.
    try:
        derived = facts.rebuild(con, blob_dir)
        result.files, result.commands = derived.files, derived.commands
        if derived.missing:
            result.warnings.append(
                f"{derived.missing} tool call(s) have no stored payload — run "
                f"`llma ingest --force` to derive their files and commands")
        if derived.unreadable:
            result.warnings.append(
                f"{derived.unreadable} tool payload(s) are in a blob that is missing")
        if derived.unknown_tools:
            top = sorted(derived.unknown_tools.items(), key=lambda kv: -kv[1])[:3]
            result.warnings.append(
                "unrecognised tools produced no files or commands: "
                + ", ".join(f"{name} x{n}" for name, n in top))
    except Exception as exc:  # noqa: BLE001 - search must survive a derivation failure
        result.warnings.append(f"fact derivation failed ({type(exc).__name__}: "
                               f"{str(exc)[:120]}); search is unaffected")

    rows = _embeddable_parts(con)
    chunks = []
    for row in rows:
        header = context_header(row["source_kind"], row["workspace"] or None,
                                row["created_at"], row["title"])
        chunks.extend(chunk_part(
            row["text"], header, row["part_id"], row["message_id"],
            row["session_id"], start_seq=0))

    result.chunks = len(chunks)
    result.chars = sum(len(c.text) for c in chunks)

    # Only a build that is about to re-insert them. This was unconditional, which
    # made every keyword-only run wipe the vector index a full build had just spent
    # half an hour of CPU on — so a nightly `--no-vectors` schedule left semantic
    # search dead six days in seven. Keeping them is sound: `chunk.part_id` is
    # ON DELETE CASCADE and ingest deletes a re-ingested session's parts, so chunks
    # whose text moved are already gone, and every surviving chunk's `vec_row` still
    # addresses the same row of the untouched vector file. What remains is coverage
    # missing for new sessions, which is exactly what `selection.unindexed_count`
    # reports. (`redact --apply` rewrites part.text in place without moving ids —
    # that is why it tells you to run a full `llma index` afterwards.)
    if with_vectors:
        con.execute("DELETE FROM chunk WHERE model_tag = ?", (model_tag,))
        con.commit()

    if not with_vectors or not chunks:
        result.skipped_vectors = True
        result.seconds = time.perf_counter() - t0
        return result

    try:
        embedder = Embedder(model_name, model_tag)
        store = VectorStore(vectors_dir, model_tag)
        matrix = np.zeros((len(chunks), 0), dtype=np.float32)
        buffers: list[np.ndarray] = []
        done = 0
        for offset, vecs in embedder.encode_passages([c.text for c in chunks]):
            buffers.append(vecs)
            done = offset + len(vecs)
            if progress:
                progress(done, len(chunks))
        matrix = np.vstack(buffers) if buffers else matrix
        store.save(matrix)
        result.vectors = len(matrix)
    except Exception as exc:  # noqa: BLE001 — FTS must survive a model failure
        result.warnings.append(f"vector build failed ({type(exc).__name__}: "
                               f"{str(exc)[:120]}); keyword search still works")
        result.skipped_vectors = True
        result.seconds = time.perf_counter() - t0
        return result

    con.executemany(
        "INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,model_tag) "
        "VALUES (?,?,?,?,?,?,?)",
        [(c.part_id, c.message_id, c.session_id, c.seq, c.text, i, model_tag)
         for i, c in enumerate(chunks)])
    con.commit()

    # Topic groups are pooled from the vectors that were just written, so this is the one
    # moment they are guaranteed to describe the current archive. Only ever reached on a
    # build that produced vectors: a --no-vectors run returned above, which is what keeps a
    # keyword-only rebuild from dissolving a perfectly good set of groups -- the same
    # reasoning as the chunk DELETE it also skips.
    try:
        from . import topics as topic_build
        result.topics = topic_build.build(con, vectors_dir,
                                          model_tag=model_tag)["topics"]
    except Exception as exc:  # noqa: BLE001 - search must survive a grouping failure
        result.warnings.append(f"topic build failed ({type(exc).__name__}: "
                               f"{str(exc)[:120]}); search is unaffected")

    result.seconds = time.perf_counter() - t0
    return result
