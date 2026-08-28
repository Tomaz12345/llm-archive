"""Regression tests for the Codex, opencode and VS Code adapters.

Each test here corresponds to a bug that actually happened while building them, or to a
format subtlety that would silently drop content if mishandled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_archive.adapters.codex import CodexAdapter
from llm_archive.adapters.opencode import OpenCodeAdapter
from llm_archive.adapters.vscode_chat import VSCodeChatAdapter
from llm_archive.core.models import ParseStats


# ---------------------------------------------------------------- codex ----

def write_rollout(tmp_path: Path, records: list[dict]) -> Path:
    d = tmp_path / "sessions" / "2026" / "01" / "01"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "rollout-2026-01-01T00-00-00-abc.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def env(ptype, payload, outer="event_msg", ts="2026-01-01T10:00:00Z"):
    return {"timestamp": ts, "type": outer, "payload": {"type": ptype, **payload}}


def test_codex_prefers_event_msg_over_injected_context(tmp_path):
    """response_item role=user also carries AGENTS.md and environment_context."""
    write_rollout(tmp_path, [
        {"timestamp": "2026-01-01T10:00:00Z", "type": "session_meta",
         "payload": {"id": "sess-1", "cwd": "C:\\work\\proj",
                     "cli_version": "1.0", "originator": "cli",
                     "base_instructions": {"text": "HUGE SYSTEM PROMPT" * 500}}},
        env("user_message", {"message": "what the human typed"}),
        {"timestamp": "2026-01-01T10:00:01Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text",
                                  "text": "# AGENTS.md instructions ..."}]}},
        env("agent_message", {"message": "the real reply"}),
    ])
    adapter = CodexAdapter(root=tmp_path, host="box")
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))

    texts = [p.text for m in session.messages for p in m.parts]
    assert "what the human typed" in texts
    assert "the real reply" in texts
    assert not any("AGENTS.md" in (t or "") for t in texts)
    # the system prompt must never be stored as conversation
    assert not any("HUGE SYSTEM PROMPT" in (t or "") for t in texts)
    assert session.host == "box"


def test_codex_falls_back_when_no_event_msg(tmp_path):
    """Older/other builds may emit only response_item records."""
    write_rollout(tmp_path, [
        {"timestamp": "2026-01-01T10:00:00Z", "type": "session_meta",
         "payload": {"id": "sess-2", "cwd": "C:\\work\\proj"}},
        {"timestamp": "2026-01-01T10:00:01Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "only here"}]}},
    ])
    adapter = CodexAdapter(root=tmp_path)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    assert [p.text for m in session.messages for p in m.parts] == ["only here"]
    assert "codex:used-response_item-fallback" in stats.unknown_types


def test_codex_tool_output_not_embedded(tmp_path):
    write_rollout(tmp_path, [
        {"timestamp": "2026-01-01T10:00:00Z", "type": "session_meta",
         "payload": {"id": "s3", "cwd": "C:\\w"}},
        {"timestamp": "2026-01-01T10:00:01Z", "type": "response_item",
         "payload": {"type": "function_call", "name": "shell_command",
                     "call_id": "c1",
                     "arguments": json.dumps({"command": "ls -la"})}},
        {"timestamp": "2026-01-01T10:00:02Z", "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": "c1",
                     "output": "a huge directory listing"}},
    ])
    adapter = CodexAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    parts = {p.kind: p for m in session.messages for p in m.parts}
    assert parts["tool_use"].embed_eligible is True
    assert "ls -la" in parts["tool_use"].text
    assert parts["tool_result"].embed_eligible is False
    assert parts["tool_result"].tool_name == "shell_command"


# ------------------------------------------------------------- opencode ----

def test_opencode_parts_are_keyed_by_message_not_session(tmp_path):
    """The regression: part/ is keyed by messageID, message/ by sessionID."""
    storage = tmp_path / "storage"
    (storage / "session" / "global").mkdir(parents=True)
    (storage / "message" / "ses_1").mkdir(parents=True)
    (storage / "part" / "msg_1").mkdir(parents=True)

    (storage / "session" / "global" / "ses_1.json").write_text(json.dumps({
        "id": "ses_1", "slug": "demo", "directory": "C:\\work\\demo",
        "title": "A demo session", "time": {"created": 1700000000000,
                                            "updated": 1700000100000}}),
        encoding="utf-8")
    (storage / "message" / "ses_1" / "msg_1.json").write_text(json.dumps({
        "id": "msg_1", "sessionID": "ses_1", "role": "user",
        "time": {"created": 1700000000000},
        "model": {"providerID": "opencode", "modelID": "kimi-k2.5"}}),
        encoding="utf-8")
    (storage / "part" / "msg_1" / "prt_1.json").write_text(json.dumps({
        "id": "prt_1", "sessionID": "ses_1", "messageID": "msg_1",
        "type": "text", "text": "hello from opencode"}), encoding="utf-8")

    adapter = OpenCodeAdapter(root=storage)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.title == "A demo session"
    assert [p.text for m in session.messages for p in m.parts] == ["hello from opencode"]


def test_opencode_reasoning_is_kept_and_cost_accumulated(tmp_path):
    storage = tmp_path / "storage"
    (storage / "session" / "global").mkdir(parents=True)
    (storage / "message" / "ses_1").mkdir(parents=True)
    (storage / "part" / "msg_1").mkdir(parents=True)
    (storage / "session" / "global" / "ses_1.json").write_text(json.dumps({
        "id": "ses_1", "directory": "C:\\w", "title": "t",
        "time": {"created": 1, "updated": 2}}), encoding="utf-8")
    (storage / "message" / "ses_1" / "msg_1.json").write_text(json.dumps({
        "id": "msg_1", "sessionID": "ses_1", "role": "assistant",
        "time": {"created": 1}}), encoding="utf-8")
    for i, part in enumerate([
        {"type": "reasoning", "text": "thinking out loud"},
        {"type": "step-finish", "cost": 0.0125,
         "tokens": {"input": 100, "output": 50}},
        {"type": "step-start"},
    ]):
        (storage / "part" / "msg_1" / f"prt_{i}.json").write_text(json.dumps(
            {"id": f"prt_{i}", "messageID": "msg_1", **part}), encoding="utf-8")

    adapter = OpenCodeAdapter(root=storage)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    kinds = [p.kind for m in session.messages for p in m.parts]
    assert kinds == ["thinking"]           # step-* produce no parts
    assert session.cost_usd == 0.0125      # but do contribute cost
    assert session.tok_in == 100 and session.tok_out == 50


# -------------------------------------------------------------- vscode ----

def write_vscode(tmp_path: Path, session: dict, folder_uri: str | None = None) -> Path:
    ws = tmp_path / "workspaceStorage" / "hash1"
    (ws / "chatSessions").mkdir(parents=True, exist_ok=True)
    if folder_uri:
        (ws / "workspace.json").write_text(json.dumps({"folder": folder_uri}),
                                           encoding="utf-8")
    path = ws / "chatSessions" / "s1.json"
    path.write_text(json.dumps(session), encoding="utf-8")
    return path


def test_vscode_assistant_prose_has_no_kind(tmp_path):
    """Plain markdown replies carry their text in `value` with kind absent."""
    write_vscode(tmp_path, {
        "sessionId": "vs1", "customTitle": "A chat",
        "creationDate": 1700000000000, "lastMessageDate": 1700000100000,
        "requests": [{
            "requestId": "r1", "timestamp": 1700000000000, "modelId": "copilot/auto",
            "message": {"text": "the question"},
            "response": [
                {"value": "the answer", "supportHtml": False},
                {"kind": "thinking", "value": "internal reasoning"},
                {"kind": "undoStop"},
            ],
        }],
    }, folder_uri="file:///c%3A/work/demo")

    adapter = VSCodeChatAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    parts = [(p.kind, p.text) for m in session.messages for p in m.parts]
    assert ("text", "the question") in parts
    assert ("text", "the answer") in parts
    assert ("thinking", "internal reasoning") in parts
    assert not any(k == "undoStop" for k, _ in parts)


def test_vscode_remote_session_is_attributed_to_the_remote_host(tmp_path):
    """VS Code keeps remote chats locally; they belong to the remote machine."""
    write_vscode(tmp_path, {
        "sessionId": "vs2", "customTitle": "remote work",
        "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    }, folder_uri="vscode-remote://ssh-remote%2Bdevbox/home/alex/doc2html")

    adapter = VSCodeChatAdapter(root=tmp_path, host="desktop")
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.host == "devbox", "remote session wrongly attributed to the local box"
    assert session.workspace_label == "doc2html"


def test_vscode_local_session_keeps_local_host(tmp_path):
    write_vscode(tmp_path, {
        "sessionId": "vs3", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    }, folder_uri="file:///c%3A/work/demo")
    adapter = VSCodeChatAdapter(root=tmp_path, host="desktop")
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.host == "desktop"


def test_vscode_unknown_response_kind_counted_not_fatal(tmp_path):
    write_vscode(tmp_path, {
        "sessionId": "vs4", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"kind": "someNewBlock"}, {"value": "ok"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    assert stats.unknown_types == {"vscode_chat:response:someNewBlock": 1}
    assert any(p.text == "ok" for m in session.messages for p in m.parts)


# ------------------------------------------------- vscode: who answered ----

def test_vscode_surface_is_an_editor_panel():
    """Not a CLI: it is a panel inside the editor, and the surface split says so."""
    assert VSCodeChatAdapter.surface == "editor_panel"


def test_vscode_participant_comes_from_the_agent_extension(tmp_path):
    """Copilot is named per request, not in the session header."""
    write_vscode(tmp_path, {
        "sessionId": "vs5", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "agent": {"extensionId": {"value": "GitHub.copilot-chat"},
                                "id": "github.copilot.editsAgent"},
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.meta["participant"] == "copilot"
    assert session.meta["participant_label"] == "GitHub Copilot"


def test_vscode_participant_falls_back_to_the_responder_name(tmp_path):
    """Most sessions carry no selectedModel; some carry no agent either."""
    write_vscode(tmp_path, {
        "sessionId": "vs6", "creationDate": 1, "lastMessageDate": 2,
        "responderUsername": "GitHub Copilot",
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.meta["participant"] == "copilot"


def test_vscode_unknown_extension_keeps_its_own_id(tmp_path):
    """A chat extension nobody has mapped yet is still a distinct participant."""
    write_vscode(tmp_path, {
        "sessionId": "vs7", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "agent": {"extensionId": {"value": "Acme.some-chat"},
                                "extensionDisplayName": "Some Chat"},
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert session.meta["participant"] == "acme.some-chat"
    assert session.meta["participant_label"] == "Some Chat"


def test_vscode_session_without_an_assistant_sets_no_participant(tmp_path):
    write_vscode(tmp_path, {
        "sessionId": "vs8", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"value": "hello"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert "participant" not in session.meta


def test_vscode_flagged_markdown_is_kept_as_answer_text(tmp_path):
    """`markdownVuln` is prose the panel would not make click-to-run. Still the answer."""
    write_vscode(tmp_path, {
        "sessionId": "vs9", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "how do I check the health endpoint"},
                      "response": [
                          {"kind": "markdownVuln",
                           "content": {"value": "Invoke-WebRequest 'http://127.0.0.1:8000/health'"}},
                      ]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    parts = [(p.kind, p.text) for m in session.messages for p in m.parts]
    assert ("text", "Invoke-WebRequest 'http://127.0.0.1:8000/health'") in parts
    assert stats.unknown_types == {}


@pytest.mark.parametrize("kind", ["progressTaskSerialized", "elicitation",
                                  "notebookEditGroup"])
def test_vscode_known_chrome_is_dropped_without_being_flagged(tmp_path, kind):
    """Chrome must not show up in `doctor` as drift needing attention."""
    write_vscode(tmp_path, {
        "sessionId": f"vs-{kind}", "creationDate": 1, "lastMessageDate": 2,
        "requests": [{"requestId": "r1", "timestamp": 1,
                      "message": {"text": "hi"},
                      "response": [{"kind": kind, "content": {"value": "noise"}},
                                   {"value": "hello"}]}],
    })
    adapter = VSCodeChatAdapter(root=tmp_path)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    assert stats.unknown_types == {}
    texts = [p.text for m in session.messages for p in m.parts]
    assert "hello" in texts
    assert "noise" not in texts
