"""Continuation detection, and the arithmetic it is there to fix.

The premise is measured, not assumed: `tools/probe_resume.py` found zero forks in phase 0
and one on a corpus half again as large, so these tests pin the behaviour that flip
depends on rather than the conclusion of either run.
"""

from __future__ import annotations

import pytest

from llm_archive.core import db, lineage
from llm_archive.core.models import Message, Part, Session
from llm_archive.stats import metrics


def _session(con, src, native, ids, *, title="s", started=1771200000000):
    """A session whose messages carry exactly `ids` as their native ids, in order."""
    msgs = []
    for i, native_id in enumerate(ids):
        m = Message(native_id=native_id, role="user" if i % 2 == 0 else "assistant",
                    created_at=started + i * 1000, seq=i)
        m.parts.append(Part(kind="text", seq=0, text=f"text for {native_id}",
                            embed_eligible=True))
        msgs.append(m)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id=native, title=title,
        workspace_key="proj", workspace_label="proj", host="box",
        started_at=started, raw_path=f"/raw/{native}.jsonl", raw_hash=native,
        messages=msgs, tok_out=100 * len(ids), tok_cache_read=1000 * len(ids)))
    return con.execute("SELECT id FROM session WHERE native_id = ?",
                       (native,)).fetchone()["id"]


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "archive.db")


@pytest.fixture
def src(con):
    return db.source_id(con, "claude_code", "Claude Code", "cli")


def test_detects_a_full_replay(con, src):
    """The shape actually observed: the child replays the whole parent, then continues."""
    parent = _session(con, src, "a", ["u1", "u2", "u3", "u4"])
    child = _session(con, src, "b", ["u1", "u2", "u3", "u4", "u5", "u6"])

    res = lineage.detect(con)

    assert res["continuations"] == 1
    pair = res["pairs"][0]
    assert (pair["parent"], pair["child"]) == (parent, child)
    assert pair["overlap"] == 4
    assert pair["whole_parent"] is True
    assert con.execute("SELECT continues_session_id FROM session WHERE id = ?",
                       (child,)).fetchone()[0] == parent
    # the PARENT's copies are the stale ones, so they are what stops counting
    assert con.execute("SELECT COUNT(*) FROM message WHERE session_id = ? AND superseded = 1",
                       (parent,)).fetchone()[0] == 4
    assert con.execute("SELECT COUNT(*) FROM message WHERE session_id = ? AND superseded = 1",
                       (child,)).fetchone()[0] == 0


def test_a_mid_session_fork_keeps_the_parents_own_tail(con, src):
    """Only the shared prefix is superseded; what the parent did alone still counts."""
    parent = _session(con, src, "a", ["u1", "u2", "u3", "OWN1", "OWN2"])
    _session(con, src, "b", ["u1", "u2", "u3", "NEW1", "NEW2", "NEW3"])

    res = lineage.detect(con)

    assert res["pairs"][0]["whole_parent"] is False
    assert res["pairs"][0]["overlap"] == 3
    kept = con.execute("SELECT native_id FROM message WHERE session_id = ? "
                       "AND superseded = 0 ORDER BY seq", (parent,)).fetchall()
    assert [r[0] for r in kept] == ["OWN1", "OWN2"]


def test_one_shared_id_is_not_lineage(con, src):
    """The Gemini case: synthesised ids collide, and MIN_PREFIX is what refuses them."""
    _session(con, src, "a", ["shared"])
    _session(con, src, "b", ["shared", "x2"])

    assert lineage.detect(con)["continuations"] == 0
    assert con.execute(
        "SELECT COUNT(*) FROM session WHERE continues_session_id IS NOT NULL").fetchone()[0] == 0


def test_missing_ids_are_never_matched(con, src):
    """Codex leaves plain text messages with no id at all (adapters/codex.py), and
    `UNIQUE(session_id, native_id)` permits many NULLs per session because SQLite NULLs
    do not collide. Treating them as equal would link every such session to every other.
    """
    _session(con, src, "a", [None, None, None])
    _session(con, src, "b", [None, None, None, None])

    assert lineage.detect(con)["continuations"] == 0


def test_a_blank_id_does_not_seed_a_prefix(con, src):
    """T3 Chat falls back to "" when a message carries neither id. One per session is
    all the UNIQUE constraint allows, but it must not count toward a shared prefix."""
    _session(con, src, "a", ["", "u2", "u3"])
    _session(con, src, "b", ["", "u2", "u3", "u4"])

    # "" is dropped before comparison, so the real shared prefix is u2,u3 -- two, under
    # MIN_PREFIX, and therefore not lineage.
    assert lineage.detect(con)["continuations"] == 0


def test_sessions_from_different_sources_never_link(con, src):
    other = db.source_id(con, "codex", "Codex", "cli")
    _session(con, src, "a", ["u1", "u2", "u3"])
    msgs = [Message(native_id=n, role="user", created_at=1771200000000, seq=i)
            for i, n in enumerate(["u1", "u2", "u3", "u4"])]
    for m in msgs:
        m.parts.append(Part(kind="text", seq=0, text="t", embed_eligible=True))
    db.upsert_session(con, other, Session(
        source_kind="codex", native_id="b", title="s", started_at=1771200000000,
        raw_path="/raw/b.jsonl", raw_hash="b", messages=msgs))

    assert lineage.detect(con)["continuations"] == 0


def test_equal_length_twins_are_not_lineage(con, src):
    """Neither continues the other -- that is a duplicate, and linking it would pick a
    parent by coin flip and delete half the archive's numbers."""
    _session(con, src, "a", ["u1", "u2", "u3"])
    _session(con, src, "b", ["u1", "u2", "u3"])

    assert lineage.detect(con)["continuations"] == 0


def test_chains_link_each_step(con, src):
    a = _session(con, src, "a", ["u1", "u2", "u3"])
    b = _session(con, src, "b", ["u1", "u2", "u3", "u4", "u5"])
    c = _session(con, src, "c", ["u1", "u2", "u3", "u4", "u5", "u6", "u7"])

    lineage.detect(con)

    links = dict(con.execute(
        "SELECT id, continues_session_id FROM session "
        "WHERE continues_session_id IS NOT NULL").fetchall())
    # b continues a, c continues b -- and nothing points at itself or loops
    assert links == {b: a, c: b}


def test_detect_is_idempotent(con, src):
    _session(con, src, "a", ["u1", "u2", "u3"])
    _session(con, src, "b", ["u1", "u2", "u3", "u4"])

    first = lineage.detect(con)
    second = lineage.detect(con)

    assert first == second
    assert con.execute("SELECT COUNT(*) FROM message WHERE superseded = 1").fetchone()[0] == 3


def test_a_link_that_stops_matching_is_dropped(con, src):
    """Re-derived from scratch, so a stale pointer cannot survive a re-ingest."""
    _session(con, src, "a", ["u1", "u2", "u3"])
    _session(con, src, "b", ["u1", "u2", "u3", "u4"])
    lineage.detect(con)
    assert con.execute("SELECT COUNT(*) FROM session "
                       "WHERE continues_session_id IS NOT NULL").fetchone()[0] == 1

    # the child is re-ingested from an export that no longer shares the prefix
    _session(con, src, "b", ["z1", "z2", "z3", "z4"])
    lineage.detect(con)

    assert con.execute("SELECT COUNT(*) FROM session "
                       "WHERE continues_session_id IS NOT NULL").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM message WHERE superseded = 1").fetchone()[0] == 0


def test_chain_reports_both_directions(con, src):
    parent = _session(con, src, "a", ["u1", "u2", "u3"], title="first half")
    child = _session(con, src, "b", ["u1", "u2", "u3", "u4"], title="second half")
    lineage.detect(con)

    assert lineage.chain(con, parent)["continued_by"]["id"] == child
    assert lineage.chain(con, parent)["continues"] is None
    assert lineage.chain(con, child)["continues"]["id"] == parent
    assert lineage.chain(con, child)["continues"]["overlap"] == 3


# ------------------------------------------------------------------ the arithmetic

def test_replayed_messages_leave_the_counts(con, src):
    _session(con, src, "a", ["u1", "u2", "u3", "u4"])
    _session(con, src, "b", ["u1", "u2", "u3", "u4", "u5", "u6"])

    before = metrics.overview(con)
    lineage.detect(con)
    after = metrics.overview(con)

    assert before["messages"] == 10          # 4 + 6, the double count
    assert after["messages"] == 6            # the conversation as it actually ran
    assert after["conversations"] == before["conversations"] - 1
    assert after["sessions"] == before["sessions"]     # the rows are still there


def test_a_wholly_replayed_parents_tokens_leave_the_totals(con, src):
    _session(con, src, "a", ["u1", "u2", "u3", "u4"])          # tok_out 400
    _session(con, src, "b", ["u1", "u2", "u3", "u4", "u5"])    # tok_out 500

    before = metrics.overview(con)
    lineage.detect(con)
    after = metrics.overview(con)

    assert before["tok_out"] == 900
    assert after["tok_out"] == 500
    assert sum(r["tok_out"] for r in metrics.by_source(con)) == 500


def test_a_mid_session_forks_parent_keeps_its_tokens(con, src):
    """It is not wholly contained in the child, so dropping it would lose real volume."""
    _session(con, src, "a", ["u1", "u2", "u3", "OWN"])         # tok_out 400
    _session(con, src, "b", ["u1", "u2", "u3", "NEW", "N2"])   # tok_out 500

    lineage.detect(con)

    assert metrics.overview(con)["tok_out"] == 900
    # ...but the replayed prefix still stops being counted twice
    assert metrics.overview(con)["messages"] == 6              # 9 - 3


def test_lineage_metric_reports_what_it_removed(con, src):
    _session(con, src, "a", ["u1", "u2", "u3", "u4"])
    _session(con, src, "b", ["u1", "u2", "u3", "u4", "u5"])
    lineage.detect(con)

    rep = metrics.lineage(con)

    assert rep["continuations"] == 1
    assert rep["merged"] == 1
    assert rep["superseded_messages"] == 4
    assert rep["excluded_tok_out"] == 400
    assert metrics.everything(con)["lineage"]["continuations"] == 1


def test_an_archive_with_no_continuations_is_unchanged(con, src):
    _session(con, src, "a", ["u1", "u2", "u3"])
    _session(con, src, "b", ["z1", "z2", "z3"])

    before = metrics.overview(con)
    assert lineage.detect(con)["continuations"] == 0

    assert metrics.overview(con) == before
    assert metrics.lineage(con)["continuations"] == 0
