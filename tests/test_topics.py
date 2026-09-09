"""Derived topic groups: the clustering, the labels, and what happens with no vectors."""

from __future__ import annotations

import numpy as np
import pytest

from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.search import topics
from llm_archive.search.embed import MODEL_TAG, VectorStore


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "archive.db")


def _blobs(rng, centre, n, spread=0.05, dim=32):
    v = centre + rng.normal(0, spread, size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _corpus(seed=0, dim=32):
    """Three tight groups (8, 6, 5) plus two vectors that belong to nothing."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dim))
    parts = [_blobs(rng, centres[0], 8), _blobs(rng, centres[1], 6),
             _blobs(rng, centres[2], 5)]
    outliers = rng.normal(size=(2, dim))
    outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
    vecs = np.vstack(parts + [outliers]).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------------------------------------------------------------- clustering

def test_finds_the_groups_and_leaves_the_outliers_out():
    groups = topics.cluster(_corpus(), threshold=0.55, min_size=3)

    assert [len(g) for g in groups] == [8, 6, 5]
    assert sorted(groups[0]) == list(range(8))
    # the blobs fill rows 0-18; the two outliers are the last two, and are assigned to
    # nothing rather than forced into a group
    assigned = {i for g in groups for i in g}
    assert 19 not in assigned
    assert 20 not in assigned


def test_is_deterministic():
    vecs = _corpus()
    first = [sorted(g) for g in topics.cluster(vecs)]
    second = [sorted(g) for g in topics.cluster(vecs)]
    assert first == second


def test_average_linkage_matches_the_definition():
    """The sum-vector shortcut must equal the mean pairwise cosine it stands in for.

    This is the whole reason there is no distance matrix, so it is worth pinning: for
    unit vectors, mean(a.b over all pairs) == (SUM_A . SUM_B) / (|A| |B|).
    """
    rng = np.random.default_rng(7)
    a = rng.normal(size=(5, 16))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(3, 16))
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    brute = np.mean([x @ y for x in a for y in b])
    shortcut = (a.sum(axis=0) @ b.sum(axis=0)) / (len(a) * len(b))

    assert brute == pytest.approx(shortcut, abs=1e-6)


def test_threshold_controls_granularity():
    vecs = _corpus()
    assert topics.cluster(vecs, threshold=0.999, min_size=3) == []      # nothing merges
    coarse = topics.cluster(vecs, threshold=-1.0, min_size=3)           # everything does
    assert len(coarse) == 1 and len(coarse[0]) == 21


def test_min_size_dissolves_small_groups():
    assert [len(g) for g in topics.cluster(_corpus(), min_size=7)] == [8]


def test_degenerate_inputs():
    assert topics.cluster(np.zeros((0, 4), dtype=np.float32)) == []
    assert topics.cluster(np.array([[1.0, 0, 0, 0]], dtype=np.float32), min_size=1) == [[0]]


# ---------------------------------------------------------------- labels

def _session(con, src, native, title, texts):
    msgs = []
    for i, text in enumerate(texts):
        m = Message(native_id=f"{native}-{i}", role="user",
                    created_at=1771200000000 + i, seq=i)
        m.parts.append(Part(kind="text", seq=0, text=text, embed_eligible=True))
        msgs.append(m)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id=native, title=title,
        started_at=1771200000000, raw_path=f"/raw/{native}", raw_hash=native,
        messages=msgs))
    return con.execute("SELECT id FROM session WHERE native_id=?", (native,)).fetchone()[0]


def test_labels_come_from_what_distinguishes_a_group(con):
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    ids = [
        _session(con, src, "a1", "Packet tracer topology", ["router vlan topology"]),
        _session(con, src, "a2", "Packet tracer vlan", ["vlan trunk router"]),
        _session(con, src, "b1", "Blender shading", ["shader material blender"]),
        _session(con, src, "b2", "Blender geometry", ["blender geometry material"]),
    ]
    labelled = topics.label_groups(con, ids, [[0, 1], [2, 3]])

    assert "router" in labelled[0]["label"] or "vlan" in labelled[0]["label"]
    assert "blender" in labelled[1]["label"] or "material" in labelled[1]["label"]
    # a term spent on one group is not reused as another's headline
    assert labelled[0]["label"] != labelled[1]["label"]
    assert labelled[0]["slug"] and " " not in labelled[0]["slug"]


def test_stopwords_never_become_a_label(con):
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    ids = [_session(con, src, "s1", "Kako lahko naredim", ["kako lahko to naredim prosim"]),
           _session(con, src, "s2", "Kako lahko popravim", ["kako lahko tudi popravim"])]

    label = topics.label_groups(con, ids, [[0, 1]])[0]["label"]

    for stop in ("kako", "lahko", "the", "and"):
        assert stop not in label.split(" · ")


def test_diacritics_fold_together(con):
    assert topics._fold("Črke") == topics._fold("crke")


# ---------------------------------------------------------------- build

def _index(con, vectors_dir, session_ids, vecs):
    """Write vectors and the chunk rows that address them."""
    VectorStore(vectors_dir, MODEL_TAG).save(vecs)
    for row, sid in enumerate(session_ids):
        part = con.execute("SELECT p.id, p.message_id FROM part p JOIN message m "
                           "ON m.id=p.message_id WHERE m.session_id=? LIMIT 1",
                           (sid,)).fetchone()
        con.execute("INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,"
                    "model_tag) VALUES (?,?,?,?,?,?,?)",
                    (part[0], part[1], sid, 0, "chunk text", row, MODEL_TAG))
    con.commit()


def test_build_persists_groups_and_leaves_outliers_unassigned(con, tmp_path):
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    ids = [_session(con, src, f"s{i}", f"session {i}", [f"vlan router text {i}"])
           for i in range(8)]
    rng = np.random.default_rng(3)
    centre = rng.normal(size=16)
    tight = _blobs(rng, centre, 6, dim=16)
    loose = rng.normal(size=(2, 16))
    loose /= np.linalg.norm(loose, axis=1, keepdims=True)
    vecs = np.vstack([tight, loose]).astype(np.float32)

    vectors_dir = tmp_path / "vectors"
    _index(con, vectors_dir, ids, vecs)

    res = topics.build(con, vectors_dir, threshold=0.55, min_size=3)

    assert res["skipped"] is False
    assert res["topics"] == 1
    assert res["assigned"] == 6
    assert res["unclustered"] == 2
    assert con.execute("SELECT COUNT(*) FROM topic").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM session_topic").fetchone()[0] == 6
    assert topics.topic_facet(con)[0]["n"] == 6


def test_build_is_wholesale_and_re_runnable(con, tmp_path):
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    ids = [_session(con, src, f"s{i}", f"session {i}", [f"vlan router {i}"])
           for i in range(6)]
    rng = np.random.default_rng(4)
    vecs = _blobs(rng, rng.normal(size=16), 6, dim=16).astype(np.float32)
    vectors_dir = tmp_path / "vectors"
    _index(con, vectors_dir, ids, vecs)

    first = topics.build(con, vectors_dir)
    second = topics.build(con, vectors_dir)

    assert first["topics"] == second["topics"]
    # replaced, never appended: a rebuild must not leave the old groups behind
    assert con.execute("SELECT COUNT(*) FROM topic").fetchone()[0] == first["topics"]
    assert con.execute("SELECT COUNT(*) FROM session_topic").fetchone()[0] == first["assigned"]


def test_build_without_vectors_keeps_what_is_there(con, tmp_path):
    """A --no-vectors rebuild must not dissolve good groups -- the same trap `index.py`
    avoids by not deleting chunks it is not about to re-insert."""
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    ids = [_session(con, src, f"s{i}", f"session {i}", [f"vlan router {i}"])
           for i in range(6)]
    rng = np.random.default_rng(5)
    vectors_dir = tmp_path / "vectors"
    _index(con, vectors_dir, ids, _blobs(rng, rng.normal(size=16), 6, dim=16).astype(np.float32))
    topics.build(con, vectors_dir)
    before = con.execute("SELECT COUNT(*) FROM topic").fetchone()[0]
    assert before == 1

    con.execute("DELETE FROM chunk")            # what --no-vectors leaves behind
    con.commit()
    res = topics.build(con, vectors_dir)

    assert res["skipped"] is True
    assert con.execute("SELECT COUNT(*) FROM topic").fetchone()[0] == before


def test_build_with_no_vector_file_at_all(con, tmp_path):
    res = topics.build(con, tmp_path / "nothing")
    assert res["skipped"] is True
    assert res["topics"] == 0
