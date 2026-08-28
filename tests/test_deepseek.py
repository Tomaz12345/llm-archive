"""Regression tests for the DeepSeek export adapter.

The three things this format gets wrong in ways that fail silently — a `conversations.json`
that another adapter also answers to, turn timestamps that run backwards, and a tree with
no `current_node` — each get a test here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_archive.adapters.claude_web import ClaudeWebAdapter
from llm_archive.adapters.deepseek import DeepSeekAdapter
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import (
    KIND_TEXT, KIND_THINKING, KIND_TOOL_RESULT, KIND_TOOL_USE, ParseStats,
)

BEIJING = "2026-08-26T23:16:23.422000+08:00"


def write_export(tmp_path: Path, conversations: list[dict], as_zip=True,
                 name="deepseek_data-2026-08-27.zip") -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(conversations, ensure_ascii=False)
    if not as_zip:
        (drops / "conversations.json").write_text(payload, encoding="utf-8")
        return drops
    with zipfile.ZipFile(drops / name, "w") as zf:
        zf.writestr("conversations.json", payload)
        zf.writestr("user.json", json.dumps(
            {"user_id": "u1", "email": "someone@example.com", "oauth_profiles": []}))
    return drops


def conv(cid, title, mapping, created="2026-08-26T23:15:46.069000+08:00"):
    return {"id": cid, "title": title, "inserted_at": created,
            "updated_at": created, "mapping": mapping}


def node(nid, parent, children, fragments=None, model="deepseek-chat", ts=BEIJING):
    message = None if fragments is None else {
        "model": model, "inserted_at": ts, "fragments": fragments}
    return {"id": nid, "parent": parent, "children": children, "message": message}


def ask(text):
    return {"type": "REQUEST", "content": text}


def reply(text):
    return {"type": "RESPONSE", "content": text}


def simple_mapping(question="hello", answer="hi there"):
    """root -> 1 (question) -> 2 (answer), the shape the real export produces."""
    return {
        "root": node("root", None, ["1"]),
        "1": node("1", "root", ["2"], [ask(question)]),
        "2": node("2", "1", [], [reply(answer)]),
    }


def parse_one(drops: Path, stats: ParseStats | None = None, **kw):
    adapter = DeepSeekAdapter(drops=drops, **kw)
    targets = adapter.discover()
    assert len(targets) == 1
    return next(adapter.parse(targets[0], stats or ParseStats()))


# -- discovery -------------------------------------------------------------

def test_reads_conversations_from_zip(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "V Language Overview", simple_mapping())])
    session = parse_one(drops)
    assert session.source_kind == "deepseek"
    assert session.native_id == "c1"
    assert session.title == "V Language Overview"
    assert session.title_source == "provider"
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_reads_bare_conversations_json(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "Bare", simple_mapping())], as_zip=False)
    assert parse_one(drops).native_id == "c1"


def test_user_json_is_never_read(tmp_path):
    """The only email address in any of the nine sources stays out of the archive."""
    drops = write_export(tmp_path, [conv("c1", "V", simple_mapping())])
    session = parse_one(drops)
    blob = json.dumps([session.meta] + [p.text for m in session.messages
                                        for p in m.parts])
    assert "example.com" not in blob


def test_claude_export_is_not_claimed_as_deepseek(tmp_path):
    """Both exports ship a conversations.json; only the payload tells them apart."""
    drops = tmp_path / "drops"
    drops.mkdir()
    claude = [{"uuid": "x", "name": "n", "created_at": "2026-01-01T00:00:00Z",
               "updated_at": "2026-01-01T00:00:00Z", "account": {"uuid": "a"},
               "chat_messages": [{"uuid": "m1", "parent_message_uuid": None,
                                  "sender": "human", "created_at": "2026-01-01T00:00:00Z",
                                  "text": "hi", "content": []}]}]
    with zipfile.ZipFile(drops / "conversations-000.zip", "w") as zf:
        zf.writestr("conversations.json", json.dumps(claude))
    assert DeepSeekAdapter(drops=drops).discover() == []
    assert len(ClaudeWebAdapter(drops=drops).discover()) == 1


def test_deepseek_export_is_not_claimed_as_claude(tmp_path):
    """The other half of the same question — this is the bug the sniffing fixed."""
    drops = write_export(tmp_path, [conv("c1", "V", simple_mapping())])
    assert ClaudeWebAdapter(drops=drops).discover() == []
    assert len(DeepSeekAdapter(drops=drops).discover()) == 1


def test_unrelated_drops_are_ignored(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    (drops / "threads-export.json").write_text('{"threads": [], "messages": []}',
                                               encoding="utf-8")
    (drops / "notes.txt").write_text("hello", encoding="utf-8")
    (drops / "broken.zip").write_bytes(b"not a zip at all")
    assert DeepSeekAdapter(drops=drops).discover() == []


def test_missing_drops_folder_is_not_an_error(tmp_path):
    assert DeepSeekAdapter(drops=tmp_path / "nope").discover() == []


# -- ordering --------------------------------------------------------------

def test_order_comes_from_the_tree_not_the_clock(tmp_path):
    """The verified export stamps the answer 4 ms BEFORE the prompt it answers."""
    mapping = {
        "root": node("root", None, ["1"]),
        "1": node("1", "root", ["2"], [ask("question")],
                  ts="2026-08-26T23:16:23.422000+08:00"),
        "2": node("2", "1", [], [reply("answer")],
                  ts="2026-08-26T23:16:23.418000+08:00"),
    }
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]))
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert [m.seq for m in session.messages] == [0, 1]
    # The answer really is older; ordering by created_at would invert the thread.
    assert session.messages[1].created_at < session.messages[0].created_at


def test_document_order_is_not_relied_on(tmp_path):
    """A mapping serialised leaf-first still reads root-first."""
    mapping = dict(reversed(list(simple_mapping().items())))
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]))
    assert [p.text for m in session.messages for p in m.parts] == ["hello", "hi there"]


def test_timestamps_convert_from_beijing_to_utc(tmp_path):
    session = parse_one(write_export(tmp_path, [conv("c1", "V", simple_mapping())]))
    # 23:16:23.422 +08:00 == 15:16:23.422 UTC
    assert session.messages[0].created_at == 1787757383422
    assert session.started_at == 1787757346069


# -- the tree --------------------------------------------------------------

def test_regenerated_branch_is_kept_but_off_the_active_path(tmp_path):
    """No current_node in this format, so the newest leaf decides (§8.1)."""
    mapping = {
        "root": node("root", None, ["1"]),
        "1": node("1", "root", ["2", "3"], [ask("question")],
                  ts="2026-08-26T23:16:00+08:00"),
        "2": node("2", "1", [], [reply("first attempt")],
                  ts="2026-08-26T23:16:10+08:00"),
        "3": node("3", "1", [], [reply("regenerated")],
                  ts="2026-08-26T23:17:00+08:00"),
    }
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]), stats)
    active = {m.native_id: m for m in session.messages if m.on_active_path}
    assert set(active) == {"1", "3"}
    assert [m.native_id for m in session.messages if not m.on_active_path] == ["2"]
    assert stats.orphaned_messages == 1
    assert session.msg_count == 2
    assert session.meta["branched"] is True


def test_root_node_stays_in_the_graph(tmp_path):
    """Dropping the message-less root would shatter the tree into false roots."""
    mapping = simple_mapping()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]))
    assert all(m.on_active_path for m in session.messages)
    assert session.messages[0].parent_native_id == "root"
    assert session.meta["branched"] is False


def test_broken_parent_link_still_yields_its_message(tmp_path):
    mapping = {
        "1": node("1", "gone", [], [ask("orphan")]),
        "2": node("2", None, [], [reply("root-level")]),
    }
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]))
    assert {m.native_id for m in session.messages} == {"1", "2"}


# -- fragments -------------------------------------------------------------

def test_search_becomes_a_tool_call_plus_numbered_hits(tmp_path):
    frag = {"type": "SEARCH", "results": [
        {"url": "https://vlang.io", "title": "The V Programming Language"},
        {"url": "https://en.wikipedia.org/wiki/V", "title": "V - Wikipedia"}]}
    mapping = {
        "root": node("root", None, ["1"]),
        "1": node("1", "root", ["2"], [ask("about V")]),
        "2": node("2", "1", [], [frag, reply("V is [citation:1] a language.")]),
    }
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_TOOL_RESULT, KIND_TEXT]
    assert [p.seq for p in parts] == [0, 1, 2]
    assert parts[0].tool_name == "search"
    # Numbered from one so a [citation:n] marker lands on the right hit.
    assert parts[1].text == ("[1] The V Programming Language · https://vlang.io\n"
                             "[2] V - Wikipedia · https://en.wikipedia.org/wiki/V")
    assert parts[1].embed_eligible is False     # §1.1: tool results are never embedded
    assert parts[2].embed_eligible is True
    assert session.meta["searches"] == 1


def test_search_without_results_is_counted_not_dropped(tmp_path):
    mapping = {"1": node("1", None, ["2"], [ask("q")]),
               "2": node("2", "1", [], [{"type": "SEARCH", "results": []},
                                        reply("answer")])}
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]), stats)
    assert stats.unknown_types == {"deepseek:search-without-results": 1}
    assert [p.kind for p in session.messages[1].parts] == [KIND_TOOL_USE, KIND_TEXT]


def test_thinking_is_kept_and_embedded(tmp_path):
    mapping = {"1": node("1", None, ["2"], [ask("q")], model="deepseek-reasoner"),
               "2": node("2", "1", [], [{"type": "THINKING", "content": "let me think"},
                                        reply("done")], model="deepseek-reasoner")}
    session = parse_one(write_export(tmp_path, [conv("c1", "R", mapping)]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_THINKING, KIND_TEXT]
    assert parts[0].text == "let me think"
    assert parts[0].embed_eligible is True


def test_unknown_fragment_type_is_counted_never_fatal(tmp_path):
    mapping = {"1": node("1", None, ["2"], [ask("q")]),
               "2": node("2", "1", [], [{"type": "IMAGE_GENERATION", "url": "x"},
                                        reply("here it is")])}
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]), stats)
    assert stats.unknown_types == {"deepseek:fragment:IMAGE_GENERATION": 1}
    assert [p.kind for p in session.messages[1].parts] == [KIND_TEXT]


def test_a_node_mixing_request_and_response_is_split(tmp_path):
    mapping = {"1": node("1", None, [], [ask("question"), reply("answer")])}
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]), stats)
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert [m.native_id for m in session.messages] == ["1", "1:assistant"]
    assert [m.model for m in session.messages] == [None, "deepseek-chat"]
    assert stats.unknown_types == {"deepseek:mixed-fragments": 1}


def test_empty_fragment_list_is_counted(tmp_path):
    mapping = {"1": node("1", None, ["2"], [ask("q")]),
               "2": node("2", "1", [], [])}
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "V", mapping)]), stats)
    assert [m.role for m in session.messages] == ["user"]
    assert stats.unknown_types == {"deepseek:message-without-fragments": 1}


# -- session fields --------------------------------------------------------

def test_model_is_recorded_on_answers_only(tmp_path):
    """`model` on a prompt is the model that was selected, not the one that spoke."""
    session = parse_one(write_export(tmp_path, [conv("c1", "V", simple_mapping())]))
    assert session.messages[0].model is None
    assert session.messages[1].model == "deepseek-chat"
    assert session.model_primary == "deepseek-chat"
    assert session.meta["participant"] == "deepseek-chat"
    assert session.meta["models"] == ["deepseek-chat"]


def test_no_usage_columns_are_invented(tmp_path):
    """This export carries no token counts and no cost. A gap, not a zero."""
    session = parse_one(write_export(tmp_path, [conv("c1", "V", simple_mapping())]))
    assert session.tok_in is None and session.tok_out is None
    assert session.cost_usd is None
    assert all(m.tok_in is None and m.tok_out is None for m in session.messages)


def test_hash_is_per_conversation_not_per_export(tmp_path):
    """A fresh monthly export must not report every untouched chat as changed."""
    first = [conv("c1", "V", simple_mapping()), conv("c2", "Other", simple_mapping())]
    before = {s.native_id: s.raw_hash for s in DeepSeekAdapter(
        drops=write_export(tmp_path / "a", first)).parse(
            (tmp_path / "a" / "drops" / "deepseek_data-2026-08-27.zip"), ParseStats())}

    grown = [conv("c1", "V", simple_mapping()),
             conv("c2", "Other", simple_mapping(answer="a longer answer"))]
    after = {s.native_id: s.raw_hash for s in DeepSeekAdapter(
        drops=write_export(tmp_path / "b", grown)).parse(
            (tmp_path / "b" / "drops" / "deepseek_data-2026-08-27.zip"), ParseStats())}

    assert before["c1"] == after["c1"]
    assert before["c2"] != after["c2"]


def test_conversation_without_id_is_counted(tmp_path):
    drops = write_export(tmp_path, [{"title": "no id", "mapping": simple_mapping()}])
    stats = ParseStats()
    adapter = DeepSeekAdapter(drops=drops)
    assert list(adapter.parse(adapter.discover()[0], stats)) == []
    assert stats.unknown_types == {"deepseek:conversation-without-id": 1}


def test_conversation_without_mapping_is_counted(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "V", simple_mapping()),
                                    {"id": "c2", "title": "empty", "mapping": {},
                                     "inserted_at": BEIJING, "updated_at": BEIJING}])
    stats = ParseStats()
    adapter = DeepSeekAdapter(drops=drops)
    sessions = list(adapter.parse(adapter.discover()[0], stats))
    assert [s.native_id for s in sessions] == ["c1"]
    assert stats.unknown_types == {"deepseek:conversation-without-mapping": 1}


def test_unreadable_drop_is_an_error_not_a_crash(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    with zipfile.ZipFile(drops / "deepseek_data.zip", "w") as zf:
        zf.writestr("conversations.json", '[{"mapping": {, "inserted_at": "x"}]')
    adapter = DeepSeekAdapter(drops=drops)
    stats = ParseStats()
    assert list(adapter.parse(adapter.discover()[0], stats)) == []
    assert stats.errors == {"unreadable:deepseek_data.zip": 1}


def test_oversized_answer_goes_to_the_blob_store(tmp_path):
    long_answer = "x" * (40 * 1024)
    drops = write_export(tmp_path, [conv("c1", "V", simple_mapping(answer=long_answer))])
    stats = ParseStats()
    session = parse_one(drops, stats, blobs=BlobStore(tmp_path / "blobs"))
    part = session.messages[1].parts[0]
    assert part.blob_sha and part.bytes == len(long_answer)
    assert part.text.endswith("<truncated, full text in blob>")
    assert stats.blobs == 1
