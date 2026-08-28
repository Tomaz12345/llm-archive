"""Tests for the DB-level DAG structural validator (llm_archive/core/validate.py).

The validator inspects whatever ends up in `message`, independent of how it got there,
so fixtures build sessions directly through `db.upsert_session` rather than through any
adapter -- these are the shapes the checks are meant to accept or reject, not a replay
of adapter parsing.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llm_archive.cli import app as cli_app
from llm_archive.core import db, validate
from llm_archive.core.models import Message, Session

T0 = 1771200000000


@pytest.fixture
def archive(tmp_path):
    con = db.connect(tmp_path / "data" / "archive.db")
    return con, tmp_path


def _msg(native_id, parent=None, *, active=True, sidechain=False, seq=0):
    return Message(native_id=native_id, role="user", created_at=T0 + seq,
                   seq=seq, parent_native_id=parent, on_active_path=active,
                   is_sidechain=sidechain)


def _session(con, kind, native_id, messages, *, label=None):
    src = db.source_id(con, kind, label or kind, "cli")
    sid, _ = db.upsert_session(con, src, Session(
        source_kind=kind, native_id=native_id, title=native_id,
        started_at=T0, raw_path=f"/raw/{native_id}", raw_hash=native_id,
        messages=messages))
    con.commit()
    return sid


def _codes(findings, severity=None):
    return {f.code for f in findings if severity is None or f.severity == severity}


# --------------------------------------------------------------------- basic shapes

def test_clean_linear_session_has_no_findings(archive):
    con, _ = archive
    sid = _session(con, "codex", "s1", [
        _msg("a"), _msg("b", "a"), _msg("c", "b")])
    assert validate.check_session(con, sid) == []


def test_flat_source_with_no_parent_links_is_not_flagged(archive):
    con, _ = archive
    sid = _session(con, "codex", "s1", [
        _msg("a"), _msg("b"), _msg("c")])
    assert validate.check_session(con, sid) == []


# -------------------------------------------------------------------------- cycles

@pytest.mark.parametrize("kind", ["codex", "claude_code"])
def test_cycle_is_a_hard_error_regardless_of_source(archive, kind):
    con, _ = archive
    sid = _session(con, kind, "s1", [
        _msg("a", "c"), _msg("b", "a"), _msg("c", "b")])
    findings = validate.check_session(con, sid)
    cycles = [f for f in findings if f.code == "cycle"]
    assert len(cycles) == 1
    assert cycles[0].severity == "error"


# ------------------------------------------------------------------- active path

def test_branching_active_path_is_an_error(archive):
    con, _ = archive
    sid = _session(con, "claude_code", "s1", [
        _msg("root"),
        _msg("a", "root"),
        _msg("b", "root"),  # two active children of the same parent
    ])
    findings = validate.check_session(con, sid)
    assert "active_path_branches" in _codes(findings, "error")


def test_zero_active_path_rows_is_an_error(archive):
    con, _ = archive
    sid = _session(con, "codex", "s1", [
        _msg("a", active=False), _msg("b", "a", active=False)])
    findings = validate.check_session(con, sid)
    assert "no_active_path" in _codes(findings, "error")


def test_active_path_collapse_is_flagged(archive):
    con, _ = archive
    msgs = [_msg("root")]
    for i in range(1, 40):
        msgs.append(_msg(f"n{i}", f"n{i-1}" if i > 1 else "root", active=False, seq=i))
    sid = _session(con, "claude_code", "s1", msgs)
    findings = validate.check_session(con, sid)
    collapse = [f for f in findings if f.code == "active_path_collapse"]
    assert len(collapse) == 1
    assert collapse[0].severity == "warning"


# --------------------------------------------------------------- known parent gaps

def test_claude_code_bookkeeping_gap_does_not_false_positive(archive):
    con, _ = archive
    # "b"'s real parent is a bookkeeping record (file-history-snapshot etc.) that was
    # never persisted, splitting one logical chain into two apparent segments.
    sid = _session(con, "claude_code", "s1", [
        _msg("root"),
        _msg("a", "root"),
        _msg("b", "unpersisted-bookkeeping-record"),
        _msg("c", "b"),
    ])
    findings = validate.check_session(con, sid)
    assert _codes(findings, "error") == set()
    assert "multi_root_active_path" in _codes(findings, "info")


def test_claude_web_dropped_empty_node_does_not_false_positive(archive):
    con, _ = archive
    # session-128-style: "b"'s parent is an empty assistant turn that produced zero
    # content parts and was dropped before persistence, not a second conversation.
    sid = _session(con, "claude_web", "s1", [
        _msg("root"),
        _msg("a", "root"),
        _msg("b", "dropped-empty-assistant-turn"),
        _msg("c", "b"),
    ])
    findings = validate.check_session(con, sid)
    assert _codes(findings, "error") == set()
    assert "multi_root_active_path" in _codes(findings, "info")


def test_unresolved_parent_in_a_clean_source_is_an_error(archive):
    con, _ = archive
    # openrouter has no documented parent-resolution gap, so a mid-chain break here
    # (unlike the claude_code/claude_web cases above) is a real regression signal.
    sid = _session(con, "openrouter", "s1", [
        _msg("root"),
        _msg("a", "root"),
        _msg("b", "nowhere"),
        _msg("c", "b"),
    ])
    findings = validate.check_session(con, sid)
    assert "multi_root_active_path" in _codes(findings, "error")


def test_gemini_per_cell_pairing_does_not_false_positive(archive):
    con, _ = archive
    # gemini.py:_messages() pairs each reply with its own prompt inside one activity
    # cell; cells are never chained, so N independent user->model pairs is the normal
    # shape of a gemini session, not a corrupted DAG.
    sid = _session(con, "gemini", "s1", [
        _msg("c1:user"), _msg("c1:model", "c1:user", seq=1),
        _msg("c2:user", seq=2), _msg("c2:model", "c2:user", seq=3),
        _msg("c3:user", seq=4), _msg("c3:model", "c3:user", seq=5),
    ])
    findings = validate.check_session(con, sid)
    assert _codes(findings, "error") == set()
    assert "multi_root_active_path" in _codes(findings, "info")


# ------------------------------------------------------------------------ multi-root

def test_multi_root_session_is_informational_not_error(archive):
    con, _ = archive
    sentinel = "00000000-0000-4000-8000-000000000000"
    sid = _session(con, "claude_web", "s1", [
        _msg("old-user", sentinel, active=False, seq=0),
        _msg("old-asst", "old-user", active=False, seq=1),
        _msg("new-user", sentinel, seq=2),
        _msg("new-asst", "new-user", seq=3),
    ])
    findings = validate.check_session(con, sid)
    assert _codes(findings, "error") == set()
    assert "multi_root_session" in _codes(findings, "info")


# -------------------------------------------------------------------------- sidechain

def test_sidechain_rows_are_excluded_from_shape_checks(archive):
    con, _ = archive
    sid = _session(con, "claude_code", "s1", [
        _msg("root"),
        _msg("a", "root"),
        _msg("sub", "root", sidechain=True),  # shares a parent with an active message
    ])
    findings = validate.check_session(con, sid)
    assert "active_path_branches" not in _codes(findings)


# -------------------------------------------------------------------------------- CLI

def test_cli_validate_exits_nonzero_on_error(archive):
    con, tmp_path = archive
    _session(con, "codex", "s1", [
        _msg("a", "c"), _msg("b", "a"), _msg("c", "b")])
    con.close()

    result = CliRunner().invoke(cli_app, [
        "validate", "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 1, result.output
    assert "cycle" in result.output


def test_cli_validate_session_argument_scopes_to_one_session(archive):
    con, tmp_path = archive
    good = _session(con, "codex", "s1", [_msg("a")])
    _session(con, "codex", "s2", [
        _msg("a", "c"), _msg("b", "a"), _msg("c", "b")])
    con.close()

    result = CliRunner().invoke(cli_app, [
        "validate", str(good), "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 0, result.output


def test_cli_validate_unknown_session_fails(archive):
    con, tmp_path = archive
    con.close()
    result = CliRunner().invoke(cli_app, [
        "validate", "999999", "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 1
