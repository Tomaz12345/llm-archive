"""Regression tests for the Claude Code adapter.

The DAG test exists because of a real bug: the first implementation built the parent
graph over conversational records only. Parent links hop through bookkeeping records,
so that shattered one 764-node session into 33 roots and collapsed its active path to a
single message. Nothing crashed; it just silently discarded half the archive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_archive.adapters.claude_code import (
    ClaudeCodeAdapter, _clean, derive_workspace,
)
from llm_archive.core.models import ParseStats
from collections import Counter


def write_session(tmp_path: Path, name: str, records: list[dict]) -> Path:
    project = tmp_path / "c--Users-x-Projekti-demo"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def msg(uid, parent, role, text, ts, **extra):
    return {
        "type": role, "uuid": uid, "parentUuid": parent,
        "timestamp": ts, "cwd": "C:\\Users\\x\\Projekti\\demo",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
        **extra,
    }


# --------------------------------------------------------------------------

def test_parent_links_hop_through_bookkeeping_records(tmp_path):
    """The regression. A snapshot record sits between two user turns."""
    records = [
        msg("a", None, "user", "first question", "2026-01-01T10:00:00Z"),
        msg("b", "a", "assistant", "first answer", "2026-01-01T10:00:01Z"),
        # not a conversational type, but it carries the chain forward
        {"type": "file-history-snapshot", "uuid": "snap", "parentUuid": "b",
         "timestamp": "2026-01-01T10:00:02Z"},
        msg("c", "snap", "user", "second question", "2026-01-01T10:00:03Z"),
        msg("d", "c", "assistant", "second answer", "2026-01-01T10:00:04Z"),
    ]
    path = write_session(tmp_path, "s1", records)
    adapter = ClaudeCodeAdapter(root=tmp_path)
    stats = ParseStats()
    sessions = list(adapter.parse(path.parent, stats))

    assert len(sessions) == 1
    active = [m for m in sessions[0].messages if m.on_active_path]
    # all four conversational records survive; none is orphaned by the snapshot
    assert len(active) == 4
    assert stats.orphaned_messages == 0


def test_abandoned_branch_is_kept_but_flagged(tmp_path):
    """A rewind leaves a dead branch. Keep it, exclude it from the active path."""
    records = [
        msg("a", None, "user", "question", "2026-01-01T10:00:00Z"),
        msg("dead", "a", "assistant", "abandoned answer", "2026-01-01T10:00:01Z"),
        msg("live", "a", "assistant", "kept answer", "2026-01-01T10:00:09Z"),
    ]
    path = write_session(tmp_path, "s2", records)
    stats = ParseStats()
    session = next(ClaudeCodeAdapter(root=tmp_path).parse(path.parent, stats))

    by_text = {m.parts[0].text: m for m in session.messages}
    assert by_text["kept answer"].on_active_path is True
    assert by_text["abandoned answer"].on_active_path is False
    assert by_text["question"].on_active_path is True
    assert stats.orphaned_messages == 1
    # the abandoned turn is still stored, not dropped
    assert len(session.messages) == 3


def test_empty_thinking_blocks_are_dropped(tmp_path):
    """Claude Code stores thinking as '' plus a signature — never a usable part."""
    rec = msg("a", None, "assistant", "visible", "2026-01-01T10:00:00Z")
    rec["message"]["content"] = [
        {"type": "thinking", "thinking": "", "signature": "x" * 900},
        {"type": "text", "text": "visible"},
    ]
    path = write_session(tmp_path, "s3", [rec])
    session = next(ClaudeCodeAdapter(root=tmp_path).parse(path.parent, ParseStats()))
    kinds = [p.kind for m in session.messages for p in m.parts]
    assert kinds == ["text"]


def test_tool_result_is_indexed_but_not_embedded(tmp_path):
    """§1.1: tool output is 62x the conversation. Searchable, never embedded."""
    rec = msg("a", None, "user", "x", "2026-01-01T10:00:00Z")
    rec["message"]["content"] = [
        {"type": "tool_result", "tool_use_id": "t1", "content": "huge file dump"},
    ]
    path = write_session(tmp_path, "s4", [rec])
    session = next(ClaudeCodeAdapter(root=tmp_path).parse(path.parent, ParseStats()))
    part = session.messages[0].parts[0]
    assert part.kind == "tool_result"
    assert part.embed_eligible is False
    assert part.text == "huge file dump"


def test_tool_result_takes_the_name_of_the_call_it_answers(tmp_path):
    """Results are filed under `user` with no name of their own; pairing is by id.

    Two calls answered out of order, so a positional match would swap the names.
    The same pairing times the call, and that must keep working.
    """
    call = msg("a", None, "assistant", "x", "2026-01-01T10:00:00Z")
    call["message"]["content"] = [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
        {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "f"}},
    ]
    result = msg("b", "a", "user", "x", "2026-01-01T10:00:02Z")
    result["message"]["content"] = [
        {"type": "tool_result", "tool_use_id": "t2", "content": "file body"},
        {"type": "tool_result", "tool_use_id": "t1", "content": "a.py  b.py"},
        {"type": "tool_result", "tool_use_id": "gone", "content": "orphan"},
    ]
    path = write_session(tmp_path, "s6", [call, result])
    session = next(ClaudeCodeAdapter(root=tmp_path).parse(path.parent, ParseStats()))

    uses = {p.tool_name: p for p in session.messages[0].parts}
    results = {p.text: p for p in session.messages[1].parts}
    assert results["file body"].tool_name == "Read"
    assert results["a.py  b.py"].tool_name == "Bash"
    assert results["orphan"].tool_name is None       # unknown call: no guess, no crash
    assert uses["Bash"].duration_ms == uses["Read"].duration_ms == 2000


def test_persisted_output_resolves_relative_to_session_dir(tmp_path):
    """The marker's absolute path breaks if ~/.claude moves; resolve by structure."""
    from llm_archive.core.blobs import BlobStore

    project = tmp_path / "c--Users-x-Projekti-demo"
    sidecar = project / "s5" / "tool-results"
    sidecar.mkdir(parents=True)
    (sidecar / "abc123.txt").write_text("THE REAL FULL OUTPUT", encoding="utf-8")

    rec = msg("a", None, "user", "x", "2026-01-01T10:00:00Z")
    rec["message"]["content"] = [{
        "type": "tool_result", "tool_use_id": "t1",
        "persistedOutputSize": 999,
        # deliberately a stale path from another machine
        "content": "<persisted-output>\nOutput too large (0.9KB). Full output saved to: "
                   "D:\\elsewhere\\gone\\abc123.txt\n\nPreview (first 2KB):\nprev",
    }]
    write_session(tmp_path, "s5", [rec])

    blobs = BlobStore(tmp_path / "blobs")
    stats = ParseStats()
    session = next(ClaudeCodeAdapter(root=tmp_path, blobs=blobs).parse(project, stats))
    part = session.messages[0].parts[0]

    assert part.blob_sha is not None, "sidecar file was not resolved"
    assert Path(part.blob_path).read_text(encoding="utf-8") == "THE REAL FULL OUTPUT"
    assert part.bytes == 999           # true size, from persistedOutputSize
    assert stats.blobs == 1


def test_subagent_transcripts_are_found_and_linked(tmp_path):
    """They live at <project>/<sessionId>/subagents/ — a non-recursive glob misses them."""
    project = tmp_path / "c--Users-x-Projekti-demo"
    write_session(tmp_path, "parent", [
        msg("a", None, "user", "delegate this", "2026-01-01T10:00:00Z")])
    sub = project / "parent" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-xyz.jsonl").write_text(json.dumps(
        msg("s", None, "assistant", "subagent work", "2026-01-01T10:00:05Z",
            isSidechain=True)), encoding="utf-8")

    sessions = list(ClaudeCodeAdapter(root=tmp_path).parse(project, ParseStats()))
    agent = [s for s in sessions if s.native_id == "agent-xyz"]
    assert agent, "subagent transcript was not discovered"
    assert agent[0].parent_native_id == "parent"
    assert agent[0].messages[0].is_sidechain is True


# --------------------------------------------------------------------------

def test_workspace_collapses_case_variants_and_subdirs():
    """One project dir held 6 cwds, two differing only by drive-letter case."""
    cwds = Counter({
        r"c:\Users\t\Projekti\Invoice_integration": 1402,
        r"C:\Users\t\Projekti\Invoice_integration\admin_dashboard": 1129,
        r"C:\Users\t\Projekti\Invoice_integration": 915,
        r"C:\Users\t\Projekti\Invoice_integration\output\render_batch": 14,
    })
    key, label = derive_workspace(cwds)
    assert key == "c:/users/t/projekti/invoice_integration"
    assert label == "Invoice_integration"


def test_workspace_empty_is_none():
    assert derive_workspace(Counter()) == (None, None)


@pytest.mark.parametrize("raw,expected", [
    ("<ide_opened_file>C:\\a\\b.py</ide_opened_file>real question", "real question"),
    ("<system-reminder>noise</system-reminder> kept", "kept"),
    ("<local-command-caveat>c</local-command-caveat>text", "text"),
    ("plain text", "plain text"),
])
def test_clean_strips_harness_noise(raw, expected):
    assert _clean(raw) == expected


def test_metadata_only_session_is_not_stored(tmp_path):
    """A file that records a mode and a slash command is not a conversation.

    Stored anyway it lands with msg_count 0 and started_at 0: counted under `by_source`
    but invisible in every dated chart, so the two disagree by one for no reason.
    """
    path = write_session(tmp_path, "empty", [
        {"type": "last-prompt", "leafUuid": "e1", "sessionId": "empty"},
        {"type": "mode", "mode": "normal", "sessionId": "empty"},
        {"type": "permission-mode", "permissionMode": "default", "sessionId": "empty"},
        {"type": "system", "subtype": "local_command", "uuid": "s1",
         "parentUuid": None, "isSidechain": False,
         "timestamp": "2026-01-01T10:00:00Z", "content": ""},
    ])
    adapter = ClaudeCodeAdapter(root=tmp_path)
    stats = ParseStats()
    assert list(adapter.parse(path.parent, stats)) == []
    assert stats.sessions == 0


def test_a_session_with_one_real_turn_is_still_stored(tmp_path):
    """The empty-session guard must not swallow short but real conversations."""
    path = write_session(tmp_path, "short", [
        {"type": "mode", "mode": "normal", "sessionId": "short"},
        msg("a", None, "user", "one question", "2026-01-01T10:00:00Z"),
    ])
    adapter = ClaudeCodeAdapter(root=tmp_path)
    sessions = list(adapter.parse(path.parent, ParseStats()))
    assert len(sessions) == 1
    assert sessions[0].started_at > 0
