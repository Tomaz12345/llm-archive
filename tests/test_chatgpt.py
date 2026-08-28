"""Regression tests for the ChatGPT export adapter.

The things this format gets wrong in ways that fail silently: a `conversations.json`
that two other adapters also answer to, nine content shapes where reading `parts`
unconditionally loses two of them, a `current_node` the sniff window cannot see, and a
role field that says `assistant` on a tool call.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_archive.adapters.chatgpt import ChatGPTAdapter
from llm_archive.adapters.claude_web import ClaudeWebAdapter
from llm_archive.adapters.deepseek import DeepSeekAdapter
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import (
    KIND_ATTACHMENT, KIND_IMAGE, KIND_TEXT, KIND_THINKING, KIND_TOOL_RESULT,
    KIND_TOOL_USE, ParseStats,
)

T0 = 1787000000.0


def node(nid, parent, children, message=None):
    return {"id": nid, "parent": parent, "children": children, "message": message}


def msg(nid, role, content, ts=T0, model="gpt-5", recipient="all", **extra):
    meta = {"model_slug": model}
    meta.update(extra.pop("metadata", {}))
    return {"id": nid, "author": {"role": role, "name": None, "metadata": {}},
            "create_time": ts, "status": "finished_successfully", "end_turn": True,
            "weight": extra.pop("weight", 1.0), "recipient": recipient,
            "content": content, "metadata": meta, **extra}


def text(*parts):
    return {"content_type": "text", "parts": list(parts)}


def simple_conv(cid="c1", title="A chat", extra_nodes=None, current="n2"):
    """root -> user -> assistant, the minimum a real export ever contains."""
    nodes = {
        "root": node("root", None, ["n1"], None),
        "n1": node("n1", "root", ["n2"], msg("m1", "user", text("hello"))),
        "n2": node("n2", "n1", [], msg("m2", "assistant", text("hi there"))),
    }
    nodes.update(extra_nodes or {})
    return {"id": cid, "conversation_id": cid, "title": title,
            "create_time": T0, "update_time": T0 + 60,
            "current_node": current, "default_model_slug": "gpt-5",
            "mapping": nodes}


def write_export(tmp_path: Path, conversations, as_zip=True,
                 name="chatgpt-export.zip", assets=None) -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(conversations, ensure_ascii=False)
    if not as_zip:
        (drops / "conversations.json").write_text(payload, encoding="utf-8")
        return drops
    with zipfile.ZipFile(drops / name, "w") as zf:
        zf.writestr("conversations.json", payload)
        zf.writestr("user.json", json.dumps({"id": "u1", "email": "a@b.c"}))
        for member, data in (assets or {}).items():
            zf.writestr(member, data)
    return drops


def parse(drops: Path, blobs=None):
    stats = ParseStats()
    adapter = ChatGPTAdapter(drops=drops, blobs=blobs)
    sessions = [s for target in adapter.discover()
                for s in adapter.parse(target, stats)]
    return sessions, stats


# -- discovery -------------------------------------------------------------


def test_discovers_its_own_export(tmp_path):
    drops = write_export(tmp_path, [simple_conv()])
    sessions, _ = parse(drops)
    assert [s.native_id for s in sessions] == ["c1"]


def test_bare_conversations_json_is_read(tmp_path):
    drops = write_export(tmp_path, [simple_conv()], as_zip=False)
    sessions, _ = parse(drops)
    assert len(sessions) == 1


def test_the_three_conversations_json_formats_do_not_claim_each_other(tmp_path):
    """claude.ai, DeepSeek and ChatGPT all ship a file by that name."""
    drops = write_export(tmp_path, [simple_conv()])
    drop = next(drops.iterdir())

    assert ChatGPTAdapter.claims(drop)
    assert not ClaudeWebAdapter.claims(drop)
    assert not DeepSeekAdapter.claims(drop)

    # And the reverse: a DeepSeek-shaped export is not ChatGPT's.
    deepseek = tmp_path / "ds"
    deepseek.mkdir()
    with zipfile.ZipFile(deepseek / "deepseek_data-2026-08-27.zip", "w") as zf:
        zf.writestr("conversations.json", json.dumps([{
            "id": "d1", "title": "t", "inserted_at": "2026-08-26T23:15:46+08:00",
            "mapping": {"root": {"id": "root", "parent": None, "children": [],
                                 "message": None}}}]))
    assert not ChatGPTAdapter.claims(next(deepseek.iterdir()))


def test_identified_without_current_node_in_the_sniff_window(tmp_path):
    """`current_node` is written after the mapping; one long chat pushes it past 64 KB.

    This is why `author` is the positive marker. A 200 KB first conversation is enough
    to reproduce the miss that using `current_node` would have caused.
    """
    filler = "x" * 400
    nodes = {"root": node("root", None, ["n0"], None)}
    prev = "root"
    for i in range(500):
        nid = f"n{i}"
        nodes[nid] = node(nid, prev, [f"n{i + 1}"],
                          msg(f"m{i}", "user", text(filler), ts=T0 + i))
        prev = nid
    conv = simple_conv(extra_nodes=nodes, current=prev)
    drops = write_export(tmp_path, [conv], as_zip=False)
    assert ChatGPTAdapter.claims(drops / "conversations.json")


# -- the tree --------------------------------------------------------------


def test_current_node_decides_the_active_path(tmp_path):
    """Two answers to one prompt; `current_node` says which one survived.

    Deliberately points at the OLDER of the two, which is what separates this from the
    resolver's "newest leaf wins" guess — if the pointer were ignored, the newer
    regenerated answer would win and this would fail.
    """
    extra = {
        "n2": node("n2", "n1", [], msg("m2", "assistant", text("first answer"),
                                       ts=T0 + 10)),
        "n3": node("n3", "n1", [], msg("m3", "assistant", text("regenerated"),
                                       ts=T0 + 99)),
    }
    extra["n1"] = node("n1", "root", ["n2", "n3"], msg("m1", "user", text("hello")))
    conv = simple_conv(extra_nodes=extra, current="n2")
    sessions, stats = parse(write_export(tmp_path, [conv]))

    on_path = {m.native_id for m in sessions[0].messages if m.on_active_path}
    assert "m2" in on_path and "m3" not in on_path
    assert stats.orphaned_messages == 1
    assert sessions[0].meta["branched"] is True


def test_a_dangling_current_node_falls_back_to_the_resolver(tmp_path):
    conv = simple_conv(current="no-such-node")
    sessions, _ = parse(write_export(tmp_path, [conv]))
    # Newest leaf wins, so the assistant reply is still on the path.
    assert {m.native_id for m in sessions[0].messages if m.on_active_path} == {"m1", "m2"}


def test_hidden_system_messages_stay_in_the_graph(tmp_path):
    """Dropped as content, kept as structure — they join the first turn to the root."""
    extra = {
        "sys": node("sys", "root", ["n1"], msg(
            "ms", "system", text(""),
            metadata={"is_visually_hidden_from_conversation": True})),
        "root": node("root", None, ["sys"], None),
        "n1": node("n1", "sys", ["n2"], msg("m1", "user", text("hello"))),
    }
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    ids = [m.native_id for m in sessions[0].messages]
    assert ids == ["m1", "m2"]                       # the system turn is not content
    assert all(m.on_active_path for m in sessions[0].messages)   # the chain held


# -- content shapes --------------------------------------------------------


def test_reasoning_is_kept_and_embedded(tmp_path):
    """`thoughts` nests its text; reading `parts` would return nothing at all."""
    extra = {"n2": node("n2", "n1", [], msg("m2", "assistant", {
        "content_type": "thoughts",
        "thoughts": [{"summary": "Planning", "content": "First I will check the file."}],
    }))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    part = sessions[0].messages[-1].parts[0]
    assert part.kind == KIND_THINKING
    assert "First I will check the file." in part.text
    assert part.embed_eligible


def test_reasoning_recap_is_reasoning_too(tmp_path):
    extra = {"n2": node("n2", "n1", [], msg("m2", "assistant", {
        "content_type": "reasoning_recap", "content": "Thought for 8 seconds"}))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))
    assert sessions[0].messages[-1].parts[0].kind == KIND_THINKING


def test_recipient_makes_an_assistant_message_a_tool_call(tmp_path):
    """Role is still `assistant` when it calls python; only `recipient` says otherwise."""
    extra = {"n2": node("n2", "n1", [], msg(
        "m2", "assistant", {"content_type": "code", "language": "python",
                            "text": "print(1)"}, recipient="python"))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    message = sessions[0].messages[-1]
    part = message.parts[0]
    assert part.kind == KIND_TOOL_USE and part.tool_name == "python"
    assert message.role == "assistant"
    assert not message.is_turn          # a tool step, not a conversational turn
    assert sessions[0].turn_count == 1  # only the user's prompt


def test_execution_output_is_a_tool_result_and_is_not_embedded(tmp_path):
    extra = {"n2": node("n2", "n1", [], msg(
        "m2", "tool", {"content_type": "execution_output", "text": "1\n"},
        recipient="all"))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    part = sessions[0].messages[-1].parts[0]
    assert part.kind == KIND_TOOL_RESULT
    assert not part.embed_eligible          # §1.1
    assert part.tool_ok is True


def test_system_error_is_a_failed_tool_result(tmp_path):
    extra = {"n2": node("n2", "n1", [], msg(
        "m2", "tool", {"content_type": "system_error", "name": "boom",
                       "text": "it broke"}))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))
    assert sessions[0].messages[-1].parts[0].tool_ok is False


def test_unknown_content_types_are_counted_not_fatal(tmp_path):
    extra = {"n2": node("n2", "n1", [], msg(
        "m2", "assistant", {"content_type": "some_future_thing", "blob": 1}))}
    sessions, stats = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    assert stats.unknown_types["chatgpt:content:some_future_thing"] == 1
    assert len(sessions) == 1               # the conversation still lands


def test_weight_zero_turns_are_kept_off_the_path(tmp_path):
    extra = {"n2": node("n2", "n1", [], msg("m2", "assistant", text("forget this"),
                                            weight=0))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))
    forgotten = [m for m in sessions[0].messages if m.native_id == "m2"]
    assert forgotten and not forgotten[0].on_active_path


# -- images ----------------------------------------------------------------


def test_uploaded_images_come_out_of_the_zip(tmp_path):
    payload = b"\x89PNG\r\n\x1a\n" + b"pixels" * 40
    extra = {"n1": node("n1", "root", ["n2"], msg("m1", "user", {
        "content_type": "multimodal_text",
        "parts": [{"content_type": "image_asset_pointer",
                   "asset_pointer": "file-service://file-ABC123",
                   "size_bytes": len(payload), "width": 8, "height": 8},
                  "what is this?"]}))}
    drops = write_export(tmp_path, [simple_conv(extra_nodes=extra)],
                         assets={"file-ABC123-photo.png": payload})
    blobs = BlobStore(tmp_path / "blobs")
    sessions, stats = parse(drops, blobs=blobs)

    kinds = [p.kind for p in sessions[0].messages[0].parts]
    assert KIND_IMAGE in kinds and KIND_TEXT in kinds
    image = next(p for p in sessions[0].messages[0].parts if p.kind == KIND_IMAGE)
    assert image.blob_sha and Path(image.blob_path).read_bytes() == payload
    assert stats.blobs == 1


def test_a_missing_image_member_keeps_the_reference(tmp_path):
    """DALL·E output ages out of the export; the pointer is all there is."""
    extra = {"n1": node("n1", "root", ["n2"], msg("m1", "user", {
        "content_type": "multimodal_text",
        "parts": [{"content_type": "image_asset_pointer",
                   "asset_pointer": "file-service://file-GONE",
                   "size_bytes": 1234, "width": 4, "height": 4}]}))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]),
                        blobs=BlobStore(tmp_path / "blobs"))

    image = sessions[0].messages[0].parts[0]
    assert image.kind == KIND_IMAGE and image.blob_sha is None
    assert "not in export" in image.text
    assert image.bytes == 1234          # the size the export claims, not zero


def test_attachments_named_in_metadata_are_kept(tmp_path):
    extra = {"n1": node("n1", "root", ["n2"], msg(
        "m1", "user", text("see attached"),
        metadata={"attachments": [{"name": "report.pdf", "size": 900}]}))}
    sessions, _ = parse(write_export(tmp_path, [simple_conv(extra_nodes=extra)]))

    kinds = [p.kind for p in sessions[0].messages[0].parts]
    assert KIND_ATTACHMENT in kinds


# -- session level ---------------------------------------------------------


def test_session_fields(tmp_path):
    sessions, _ = parse(write_export(tmp_path, [simple_conv(title="Naslov")]))
    session = sessions[0]

    assert session.source_kind == "chatgpt"
    assert session.title == "Naslov" and session.title_source == "provider"
    assert session.started_at == int(T0 * 1000)
    assert session.ended_at == int((T0 + 60) * 1000)
    assert session.model_primary == "gpt-5"
    assert session.meta["participant"] == "gpt-5"
    assert session.exported_at is not None       # dates the drop, for the stale guard
    assert session.tok_in is None and session.cost_usd is None   # a gap, not a zero


def test_rehash_is_per_conversation_not_per_drop(tmp_path):
    """A monthly re-export must not report every chat as changed."""
    a = parse(write_export(tmp_path / "a", [simple_conv("c1"), simple_conv("c2", "B")]))[0]
    grown = simple_conv("c2", "B")
    grown["mapping"]["n3"] = node("n3", "n2", [], msg("m3", "user", text("more")))
    grown["mapping"]["n2"]["children"] = ["n3"]
    grown["current_node"] = "n3"
    b = parse(write_export(tmp_path / "b", [simple_conv("c1"), grown]))[0]

    by_id_a = {s.native_id: s for s in a}
    by_id_b = {s.native_id: s for s in b}
    assert by_id_a["c1"].raw_hash == by_id_b["c1"].raw_hash      # untouched
    assert by_id_a["c2"].raw_hash != by_id_b["c2"].raw_hash      # grew
