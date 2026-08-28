"""Regression tests for the claude.ai export adapter and the shared tree resolver."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_archive.adapters._tree import resolve_active_path
from llm_archive.adapters.claude_web import ClaudeWebAdapter
from llm_archive.core.models import ParseStats


def write_export(tmp_path: Path, conversations: list[dict], as_zip=True) -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(conversations, ensure_ascii=False)
    if not as_zip:
        path = drops / "conversations.json"
        path.write_text(payload, encoding="utf-8")
        return drops
    path = drops / "conversations-000.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", payload)
    return drops


def conv(uuid, name, messages, created="2026-01-01T10:00:00Z"):
    return {"uuid": uuid, "name": name, "summary": "", "created_at": created,
            "updated_at": created, "account": {"uuid": "acct"},
            "chat_messages": messages}


def cmsg(uuid, parent, sender, blocks, ts="2026-01-01T10:00:00Z"):
    return {"uuid": uuid, "parent_message_uuid": parent, "sender": sender,
            "created_at": ts, "updated_at": ts, "text": "",
            "content": blocks, "attachments": [], "files": []}


def text_block(t):
    return {"type": "text", "text": t}


# --------------------------------------------------------------------------

def test_reads_conversations_from_zip(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "First chat", [
        cmsg("m1", None, "human", [text_block("hello")]),
        cmsg("m2", "m1", "assistant", [text_block("hi there")]),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    targets = adapter.discover()
    assert len(targets) == 1

    sessions = list(adapter.parse(targets[0], ParseStats()))
    assert len(sessions) == 1
    assert sessions[0].title == "First chat"
    assert [m.role for m in sessions[0].messages] == ["user", "assistant"]


def test_thinking_is_kept_here_unlike_claude_code(tmp_path):
    """The web export carries real reasoning text; Claude Code stores it empty."""
    drops = write_export(tmp_path, [conv("c1", "T", [
        cmsg("m1", None, "assistant", [
            {"type": "thinking", "thinking": "let me reason about this",
             "signature": "sig"},
            text_block("the answer"),
        ]),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    parts = {p.kind: p for p in session.messages[0].parts}
    assert "thinking" in parts
    assert parts["thinking"].text == "let me reason about this"
    assert parts["thinking"].embed_eligible is True


def test_empty_thinking_still_dropped(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "T", [
        cmsg("m1", None, "assistant", [
            {"type": "thinking", "thinking": "   ", "signature": "sig"},
            text_block("answer"),
        ]),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    assert [p.kind for p in session.messages[0].parts] == ["text"]


def test_branch_is_flagged_not_dropped(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "B", [
        cmsg("m1", None, "human", [text_block("q")], "2026-01-01T10:00:00Z"),
        cmsg("dead", "m1", "assistant", [text_block("abandoned")],
             "2026-01-01T10:00:01Z"),
        cmsg("live", "m1", "assistant", [text_block("kept")],
             "2026-01-01T10:00:09Z"),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    by_text = {m.parts[0].text: m for m in session.messages}
    assert by_text["kept"].on_active_path is True
    assert by_text["abandoned"].on_active_path is False
    assert stats.orphaned_messages == 1
    assert len(session.messages) == 3


def test_tool_result_not_embedded(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "T", [
        cmsg("m1", None, "assistant", [
            {"type": "tool_use", "name": "web_search",
             "input": {"query": "how to do X"}},
            {"type": "tool_result", "name": "web_search",
             "content": [{"type": "text", "text": "a big pile of results"}]},
        ]),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    session = next(adapter.parse(adapter.discover()[0], ParseStats()))
    parts = {p.kind: p for p in session.messages[0].parts}
    assert parts["tool_use"].embed_eligible is True
    assert "how to do X" in parts["tool_use"].text
    assert parts["tool_result"].embed_eligible is False


def test_offloaded_parts_are_counted_in_the_run_report(tmp_path):
    """`blobs 0` while writing blobs made the ingest summary quietly wrong."""
    from llm_archive.core.blobs import BlobStore
    from llm_archive.core.models import INLINE_LIMIT

    huge = "x" * (INLINE_LIMIT + 1000)
    drops = write_export(tmp_path, [conv("c1", "T", [
        cmsg("m1", None, "assistant", [text_block(huge)]),
    ])])
    stats = ParseStats()
    adapter = ClaudeWebAdapter(drops=drops, blobs=BlobStore(tmp_path / "blobs"))
    session = next(adapter.parse(adapter.discover()[0], stats))

    assert session.messages[0].parts[0].blob_sha is not None
    assert stats.blobs == 1
    assert stats.blob_bytes == len(huge)


def test_per_conversation_hashing(tmp_path):
    """A re-export must not report every untouched conversation as changed."""
    a = conv("c1", "one", [cmsg("m1", None, "human", [text_block("hello")])])
    b = conv("c2", "two", [cmsg("m2", None, "human", [text_block("world")])])
    drops = write_export(tmp_path, [a, b])
    adapter = ClaudeWebAdapter(drops=drops)
    first = {s.native_id: s.raw_hash for s in
             adapter.parse(adapter.discover()[0], ParseStats())}

    # a fresh export where only c2 gained a message
    b2 = conv("c2", "two", [cmsg("m2", None, "human", [text_block("world")]),
                            cmsg("m3", "m2", "assistant", [text_block("reply")])])
    drops2 = write_export(tmp_path / "second", [a, b2])
    adapter2 = ClaudeWebAdapter(drops=drops2)
    second = {s.native_id: s.raw_hash for s in
              adapter2.parse(adapter2.discover()[0], ParseStats())}

    assert first["c1"] == second["c1"], "untouched conversation changed hash"
    assert first["c2"] != second["c2"], "changed conversation kept its hash"


def test_unknown_block_types_counted_not_fatal(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "T", [
        cmsg("m1", None, "assistant", [
            {"type": "some_future_block", "payload": 1},
            text_block("still here"),
        ]),
    ])])
    adapter = ClaudeWebAdapter(drops=drops)
    stats = ParseStats()
    session = next(adapter.parse(adapter.discover()[0], stats))
    assert [p.kind for p in session.messages[0].parts] == ["text"]
    assert stats.unknown_types == {"claude_web:content:some_future_block": 1}


# --------------------------------------------------------------------------

def test_tree_resolver_picks_newest_leaf():
    parents = {"a": None, "b": "a", "c": "a"}
    times = {"a": "1", "b": "2", "c": "9"}
    active = resolve_active_path(parents, parents.get, times.get)
    assert active == {"a", "c"}


def test_tree_resolver_survives_a_cycle():
    parents = {"a": "b", "b": "a"}
    active = resolve_active_path(parents, parents.get, lambda x: x)
    assert active == {"a", "b"}, "a cycle must not lose every node"


def test_tree_resolver_handles_missing_parent():
    """A parent outside the node set makes the child a root, not a crash."""
    parents = {"a": "gone", "b": "a"}
    active = resolve_active_path(parents, parents.get, lambda x: x)
    assert active == {"a", "b"}
