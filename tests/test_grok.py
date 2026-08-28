"""Regression tests for the Grok export adapter.

The things this format gets wrong in ways that fail silently — a ZIP named after a bare
uuid with the chats buried four directories down, two clock formats that both stamp the
whole turn at once, a `model` field holding the UI picker rather than the model, and
search hits aggregated three times over — each get a test here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_archive.adapters.claude_web import ClaudeWebAdapter
from llm_archive.adapters.deepseek import DeepSeekAdapter
from llm_archive.adapters.grok import MEMBER, GrokAdapter
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import (
    KIND_TEXT, KIND_THINKING, KIND_TOOL_RESULT, KIND_TOOL_USE, ParseStats,
)

# Where the export actually puts the file: a retention prefix, then the account uuid.
INSIDE = f"ttl/30d/export_data/c2f0fd62-8d42-489a-863d-c03521fc8558/{MEMBER}"
ZIP_NAME = "5f4c8d58-c8d3-4b68-9256-13f01162dcb6.zip"

# The root every first turn points at, which the export never includes.
ROOT = "bcfa4ad2-9e62-5353-b941-7fc191f105ad"

# prod-mc-auth-mgmt-api.json, trimmed to the fields that must never reach the archive.
IDENTITY = {
    "user": {"userId": "u1", "email": "someone@example.com",
             "googleEmail": "someone@example.com", "birthDate": "1990-01-01"},
    "sessions": [{"sessionId": "s1", "cfMetadata": {"ipAddress": "203.0.113.7",
                                                    "city": "Ljubljana",
                                                    "latitude": 46.05, "longitude": 14.5}}],
    "api_keys": [{"apiKeyHash": "deadbeef", "redactedApiKey": "xai-...abcd"}],
}


# -- builders --------------------------------------------------------------

def mongo(ms: int) -> dict:
    """The Mongo extended JSON every response stamps its clock in."""
    return {"$date": {"$numberLong": str(ms)}}


def document(conversations: list[dict], **extra) -> dict:
    payload = {"conversations": conversations, "projects": [], "tasks": [],
               "media_posts": []}
    payload.update(extra)
    return payload


def write_export(tmp_path: Path, conversations: list[dict], as_zip=True,
                 name=ZIP_NAME, **extra) -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document(conversations, **extra), ensure_ascii=False)
    if not as_zip:
        (drops / MEMBER).write_text(payload, encoding="utf-8")
        return drops
    with zipfile.ZipFile(drops / name, "w") as zf:
        zf.writestr(INSIDE, payload)
        zf.writestr(INSIDE.replace(MEMBER, "prod-mc-auth-mgmt-api.json"),
                    json.dumps(IDENTITY))
        zf.writestr(INSIDE.replace(MEMBER, "prod-mc-billing.json"),
                    json.dumps({"balance_map": {"t1": 0}}))
    return drops


def conv(cid, title, responses, leaf=None, created="2026-08-26T20:23:32.934050Z",
         modified="2026-08-26T20:24:10.259142Z", **fields):
    conversation = {"id": cid, "user_id": "c2f0fd62-8d42-489a-863d-c03521fc8558",
                    "x_user_id": "x-9", "title": title, "create_time": created,
                    "modify_time": modified, "starred": False, "temporary": False,
                    "asset_ids": [], "leaf_response_id": leaf}
    conversation.update(fields)
    return {"conversation": conversation, "responses": responses}


def ask(rid, text, parent=ROOT, ms=1787775850033):
    return {"response": {"_id": rid, "sender": "human", "message": text,
                         "parent_response_id": parent, "create_time": mongo(ms),
                         "model": "", "metadata": {}}}


def reply(rid, text, parent, ms=1787775850043, picker="build",
          resolved="grok-chat-app-builder-free", steps=None, **fields):
    response = {"_id": rid, "sender": "assistant", "message": text,
                "parent_response_id": parent, "create_time": mongo(ms),
                "model": picker, "web_search_results": [],
                "metadata": {"request_metadata": {"model": picker,
                                                  "resolved_model": resolved,
                                                  "source": "Web"}}}
    if steps is not None:
        response["steps"] = steps
    response.update(fields)
    return {"response": response}


def header(text):
    return {"tag_order": ["header"], "tagged_text": {"header": text},
            "web_search_results": [], "tool_usage_results": []}


def call(card_id, tool, args, result=None, hits=None):
    """One step holding a single tool call, shaped the way the export ships it."""
    step = {
        "tag_order": ["tool_usage_card", "raw_function_result"],
        "tagged_text": {"tool_usage_card": _markup(card_id, tool, args),
                        "raw_function_result": ""},
        "tool_usage_cards": [{"tool_usage_card_id": card_id, "intent": None,
                              "tool": {tool: {"args": args}}}],
        "tool_usage_results": [] if result is None else [
            {"tool_usage_card_id": card_id, "result": result}],
        # The export repeats every hit here as well; reading it would double-count.
        "web_search_results": hits or [],
        "x_posts_ids": [],
    }
    return step


def _markup(card_id, tool, args):
    return ("<xai:tool_usage_card>\n"
            f"  <xai:tool_usage_card_id>{card_id}</xai:tool_usage_card_id>\n"
            f"  <xai:tool_name>{tool}</xai:tool_name>\n"
            f"  <xai:tool_args><![CDATA[{json.dumps({'args': args})}]]></xai:tool_args>\n"
            "</xai:tool_usage_card>")


def hit(url, title, preview):
    return {"url": url, "title": title, "preview": preview}


def exchange(question="Hey, can you give me short tutorial about Odin?",
             answer="Odin is a compiled, statically typed systems language.",
             steps=None):
    """root -> r1 (question) -> r2 (answer), the shape the real export produces."""
    return [ask("r1", question), reply("r2", answer, "r1", steps=steps)]


def parse_one(drops: Path, stats: ParseStats | None = None, **kw):
    adapter = GrokAdapter(drops=drops, **kw)
    targets = adapter.discover()
    assert len(targets) == 1
    return next(adapter.parse(targets[0], stats or ParseStats()))


# -- discovery -------------------------------------------------------------

def test_reads_chats_from_a_uuid_named_zip(tmp_path):
    """Nothing outside the archive names this export; the member is the only marker."""
    drops = write_export(tmp_path, [conv("c1", "Odin Programming Language Tutorial",
                                         exchange(), leaf="r2")])
    session = parse_one(drops)
    assert session.source_kind == "grok"
    assert session.native_id == "c1"
    assert session.title == "Odin Programming Language Tutorial"
    assert session.title_source == "provider"
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_reads_bare_backend_json(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2")],
                         as_zip=False)
    assert parse_one(drops).native_id == "c1"


def test_identity_and_billing_members_are_never_read(tmp_path):
    """The email, the IP address and the api key hashes stay out of the archive."""
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2")])
    session = parse_one(drops)
    blob = json.dumps([session.meta, session.title]
                      + [m.meta for m in session.messages]
                      + [p.text for m in session.messages for p in m.parts])
    for secret in ("example.com", "203.0.113.7", "deadbeef", "1990-01-01",
                   "Ljubljana", "balance_map"):
        assert secret not in blob


def test_account_ids_are_not_carried_into_the_session(tmp_path):
    """`user_id`/`x_user_id` identify the account, and nothing here is keyed on them."""
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2")])
    session = parse_one(drops)
    assert "c2f0fd62-8d42-489a-863d-c03521fc8558" not in json.dumps(session.meta)
    assert "x-9" not in json.dumps(session.meta)


def test_other_exports_are_not_claimed_as_grok(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    claude = [{"uuid": "x", "name": "n", "created_at": "2026-01-01T00:00:00Z",
               "updated_at": "2026-01-01T00:00:00Z", "account": {"uuid": "a"},
               "chat_messages": []}]
    with zipfile.ZipFile(drops / "conversations-000.zip", "w") as zf:
        zf.writestr("conversations.json", json.dumps(claude))
    (drops / "threads-export.json").write_text('{"threads": [], "messages": []}',
                                               encoding="utf-8")
    (drops / "notes.txt").write_text("hello", encoding="utf-8")
    (drops / "broken.zip").write_bytes(b"not a zip at all")
    assert GrokAdapter(drops=drops).discover() == []


def test_grok_export_is_not_claimed_by_the_conversations_json_adapters(tmp_path):
    """Both of those sniff a member Grok does not ship; this is the collision test."""
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2")])
    assert ClaudeWebAdapter(drops=drops).discover() == []
    assert DeepSeekAdapter(drops=drops).discover() == []
    assert len(GrokAdapter(drops=drops).discover()) == 1


def test_missing_drops_folder_is_not_an_error(tmp_path):
    assert GrokAdapter(drops=tmp_path / "nope").discover() == []


def test_unreadable_drop_is_an_error_not_a_crash(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    with zipfile.ZipFile(drops / ZIP_NAME, "w") as zf:
        zf.writestr(INSIDE, '{"conversations": [ {, ]}')
    adapter = GrokAdapter(drops=drops)
    stats = ParseStats()
    assert list(adapter.parse(adapter.discover()[0], stats)) == []
    assert stats.errors == {f"unreadable:{ZIP_NAME}": 1}


# -- clocks ----------------------------------------------------------------

def test_both_clock_formats_are_read(tmp_path):
    """ISO on the conversation, Mongo extended JSON on every response."""
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", exchange(),
                                                     leaf="r2")]))
    assert session.started_at == 1787775812934      # 2026-08-26T20:23:32.934050Z
    assert session.ended_at == 1787775850259
    assert [m.created_at for m in session.messages] == [1787775850033, 1787775850043]


def test_order_comes_from_the_tree_not_the_clock(tmp_path):
    """Both turns are stamped at the end of the turn, 10 ms apart — see the docstring."""
    responses = [ask("r1", "question", ms=1787775850053),
                 reply("r2", "answer", "r1", ms=1787775850043)]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]))
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert [m.seq for m in session.messages] == [0, 1]
    # The answer really is older; ordering by created_at would invert the thread.
    assert session.messages[1].created_at < session.messages[0].created_at


def test_document_order_is_not_relied_on(tmp_path):
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", list(reversed(exchange(question="q", answer="a"))), leaf="r2")]))
    assert [p.text for m in session.messages for p in m.parts] == ["q", "a"]


def test_thinking_span_is_kept_because_create_time_cannot_show_it(tmp_path):
    responses = [ask("r1", "q"),
                 reply("r2", "a", "r1", thinking_start_time=mongo(1787775813216),
                       thinking_end_time=mongo(1787775831251))]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]))
    assert session.messages[1].meta["thinking_ms"] == 18035


# -- the tree --------------------------------------------------------------

def test_dangling_root_parent_is_a_root_not_an_orphan(tmp_path):
    """The first turn points at a node the export never includes."""
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", exchange(),
                                                     leaf="r2")]))
    assert all(m.on_active_path for m in session.messages)
    assert session.messages[0].parent_native_id == ROOT
    assert session.meta["branched"] is False
    assert session.msg_count == 2


def test_named_leaf_decides_the_active_path(tmp_path):
    """Unlike DeepSeek, this export says which branch survived — even the older one."""
    responses = [ask("r1", "question"),
                 reply("r2", "first attempt", "r1", ms=1787775850100),
                 reply("r3", "regenerated", "r1", ms=1787775850900)]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]), stats)
    active = {m.native_id for m in session.messages if m.on_active_path}
    assert active == {"r1", "r2"}
    assert [m.native_id for m in session.messages if not m.on_active_path] == ["r3"]
    assert stats.orphaned_messages == 1
    assert session.meta["branched"] is True
    assert [m.seq for m in session.messages if m.on_active_path] == [0, 1]


def test_missing_leaf_falls_back_to_the_newest_leaf(tmp_path):
    """A leaf that was never exported must not orphan the whole conversation."""
    responses = [ask("r1", "question"),
                 reply("r2", "first attempt", "r1", ms=1787775850100),
                 reply("r3", "regenerated", "r1", ms=1787775850900)]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="never-exported")]))
    assert {m.native_id for m in session.messages if m.on_active_path} == {"r1", "r3"}


def test_conversation_without_leaf_field_still_resolves(tmp_path):
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", exchange())]))
    assert all(m.on_active_path for m in session.messages)


# -- steps: reasoning, tools, results --------------------------------------

def test_step_headers_are_the_only_reasoning_and_are_embedded(tmp_path):
    steps = [header("Thinking about your request"),
             header("Providing a short tutorial on the Odin programming language")]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_THINKING, KIND_THINKING, KIND_TEXT]
    assert parts[0].text == "Thinking about your request"
    assert parts[0].embed_eligible is True


def test_tool_call_and_its_result_are_paired_by_card_id(tmp_path):
    steps = [call("card-1", "WebSearch", {"query": "Odin tutorial"},
                  result={"WebSearchResults": [
                      hit("https://odin-lang.org/", "Odin Programming Language",
                          "A general-purpose programming language."),
                      hit("https://learnxinyminutes.com/odin/", "Learn Odin in Y Minutes",
                          "Odin was created by Bill Hall.")]})]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_TOOL_RESULT, KIND_TEXT]
    assert [p.seq for p in parts] == [0, 1, 2]
    assert parts[0].tool_name == "WebSearch"
    assert json.loads(parts[0].text) == {"query": "Odin tutorial"}
    # Numbered from one so a citation marker lands on the right hit.
    assert parts[1].text == (
        "[1] Odin Programming Language · https://odin-lang.org/\n"
        "A general-purpose programming language.\n\n"
        "[2] Learn Odin in Y Minutes · https://learnxinyminutes.com/odin/\n"
        "Odin was created by Bill Hall.")
    assert parts[1].tool_name == "WebSearch"
    assert parts[1].embed_eligible is False     # §1.1: tool results are never embedded
    assert parts[2].embed_eligible is True
    assert session.meta["tools"] == {"WebSearch": 1}
    assert session.meta["searches"] == 1


def test_a_call_with_no_result_is_still_recorded(tmp_path):
    """ReadFile and InitTerminalSession record no result at all in the real export."""
    steps = [call("card-1", "ReadFile", {"file_path": "/workspace/AGENTS.md"}),
             call("card-2", "InitTerminalSession", {"preview_url": "https://x"})]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_TOOL_USE, KIND_TEXT]
    assert [p.tool_name for p in parts[:2]] == ["ReadFile", "InitTerminalSession"]
    assert stats.unknown_types == {}


def test_tool_ok_is_never_invented(tmp_path):
    """Nothing in this format flags success or failure, so the column stays unknown."""
    steps = [call("card-1", "WebSearch", {"query": "x"},
                  result={"WebSearchResults": [hit("https://a", "A", "p")]})]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    assert all(p.tool_ok is None for m in session.messages for p in m.parts)


def test_flat_tool_args_are_kept_without_an_args_wrapper(tmp_path):
    steps = [{"tag_order": ["tool_usage_card"], "tagged_text": {"tool_usage_card": "x"},
              "tool_usage_cards": [{"tool_usage_card_id": "c",
                                    "tool": {"InitTerminalSession":
                                             {"preview_url": "https://sandbox"}}}],
              "tool_usage_results": []}]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    call_part = session.messages[1].parts[0]
    assert json.loads(call_part.text) == {"preview_url": "https://sandbox"}


def test_step_search_aggregate_is_not_double_counted(tmp_path):
    """Each hit ships in the tool result AND in the step's own aggregate list."""
    hits = [hit("https://odin-lang.org/", "Odin", "A language.")]
    steps = [call("card-1", "WebSearch", {"query": "Odin"},
                  result={"WebSearchResults": hits}, hits=hits)]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    results = [p for p in session.messages[1].parts if p.kind == KIND_TOOL_RESULT]
    assert len(results) == 1


def test_response_aggregate_is_not_double_counted(tmp_path):
    """The response repeats every step's hits once more at the top level."""
    hits = [hit("https://odin-lang.org/", "Odin", "A language.")]
    steps = [call("card-1", "WebSearch", {"query": "Odin"},
                  result={"WebSearchResults": hits}, hits=hits)]
    responses = [ask("r1", "q"),
                 reply("r2", "a", "r1", steps=steps, web_search_results=hits)]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]))
    results = [p for p in session.messages[1].parts if p.kind == KIND_TOOL_RESULT]
    assert len(results) == 1


def test_response_aggregate_is_the_fallback_when_there_are_no_steps(tmp_path):
    """With no steps it is the only record of what the answer searched."""
    hits = [hit("https://odin-lang.org/", "Odin", "A language.")]
    responses = [ask("r1", "q"), reply("r2", "a", "r1", web_search_results=hits)]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_RESULT, KIND_TEXT]
    assert parts[0].text.startswith("[1] Odin · https://odin-lang.org/")
    assert parts[0].embed_eligible is False


def test_raw_function_result_is_kept_when_it_carries_anything(tmp_path):
    steps = [{"tag_order": ["raw_function_result"],
              "tagged_text": {"raw_function_result": "exit 0\nbuilt in 2.1s"},
              "tool_usage_results": []}]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_RESULT, KIND_TEXT]
    assert parts[0].text == "exit 0\nbuilt in 2.1s"


def test_empty_raw_function_result_is_not_a_part(tmp_path):
    """Empty on every step of the verified export."""
    steps = [call("card-1", "ReadFile", {"file_path": "/workspace/AGENTS.md"})]
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]))
    assert [p.kind for p in session.messages[1].parts] == [KIND_TOOL_USE, KIND_TEXT]


def test_markup_is_parsed_when_the_card_list_is_missing(tmp_path):
    steps = [{"tag_order": ["tool_usage_card"],
              "tagged_text": {"tool_usage_card": _markup(
                  "card-1", "BrowsePage", {"url": "https://odin-lang.org/"})},
              "tool_usage_results": []}]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    call_part = session.messages[1].parts[0]
    assert call_part.kind == KIND_TOOL_USE and call_part.tool_name == "BrowsePage"
    assert stats.unknown_types == {"grok:card-markup-only": 1}


def test_unknown_step_tag_is_counted_never_fatal(tmp_path):
    steps = [{"tag_order": ["image_generation_card"],
              "tagged_text": {"image_generation_card": "<xai:image/>"},
              "tool_usage_results": []}]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    assert stats.unknown_types == {"grok:tag:image_generation_card": 1}
    assert [p.kind for p in session.messages[1].parts] == [KIND_TEXT]


def test_unknown_result_shape_is_counted_and_kept_as_json(tmp_path):
    steps = [call("card-1", "CodeExecution", {"code": "print(1)"},
                  result={"ExecutionOutput": {"stdout": "1\n"}})]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    parts = session.messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_TOOL_RESULT, KIND_TEXT]
    assert json.loads(parts[1].text) == {"ExecutionOutput": {"stdout": "1\n"}}
    assert stats.unknown_types == {"grok:result:ExecutionOutput": 1}


def test_result_without_a_matching_card_is_kept(tmp_path):
    steps = [{"tag_order": [], "tagged_text": {}, "tool_usage_cards": [],
              "tool_usage_results": [{"tool_usage_card_id": "orphan",
                                      "result": {"WebSearchResults": [
                                          hit("https://a", "A", "preview")]}}]}]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    assert [p.kind for p in session.messages[1].parts] == [KIND_TOOL_RESULT, KIND_TEXT]
    assert stats.unknown_types == {"grok:result-without-card": 1}


# -- roles and models ------------------------------------------------------

def test_resolved_model_wins_and_the_picker_is_kept_separately(tmp_path):
    """`model` holds the UI mode ('build'), not the model that answered."""
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", exchange(),
                                                     leaf="r2")]))
    assert session.messages[0].model is None        # a prompt names no speaker
    assert session.messages[1].model == "grok-chat-app-builder-free"
    assert session.messages[1].meta["picker"] == "build"
    assert session.messages[1].meta["source"] == "Web"
    assert session.model_primary == "grok-chat-app-builder-free"
    assert session.meta["models"] == ["grok-chat-app-builder-free"]
    assert session.meta["pickers"] == ["build"]
    assert session.meta["participant"] == "grok-chat-app-builder-free"


def test_picker_is_the_model_when_nothing_resolved_it(tmp_path):
    responses = [ask("r1", "q"), reply("r2", "a", "r1", picker="grok-4", resolved="")]
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]))
    assert session.messages[1].model == "grok-4"
    assert "picker" not in session.messages[1].meta


def test_unknown_sender_is_counted_never_dropped(tmp_path):
    responses = [ask("r1", "q"),
                 {"response": {"_id": "r2", "sender": "agent", "message": "a",
                               "parent_response_id": "r1",
                               "create_time": mongo(1787775850043), "metadata": {}}}]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r2")]), stats)
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert stats.unknown_types == {"grok:sender:agent": 1}


def test_response_without_content_is_counted(tmp_path):
    responses = [ask("r1", "q"), reply("r2", "", "r1")]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", responses,
                                                     leaf="r1")]), stats)
    assert [m.role for m in session.messages] == ["user"]
    assert stats.unknown_types == {"grok:response-without-content": 1}


# -- session fields --------------------------------------------------------

def test_no_usage_columns_are_invented(tmp_path):
    """This export carries no token counts and no cost. A gap, not a zero."""
    session = parse_one(write_export(tmp_path, [conv("c1", "Odin", exchange(),
                                                     leaf="r2")]))
    assert session.tok_in is None and session.tok_out is None
    assert session.cost_usd is None
    assert all(m.tok_in is None and m.tok_out is None for m in session.messages)


def test_starred_and_temporary_are_carried(tmp_path):
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(), leaf="r2", starred=True, temporary=True)]))
    assert session.meta["starred"] is True
    assert session.meta["temporary"] is True


def test_hash_is_per_conversation_not_per_export(tmp_path):
    """A fresh export must not report every untouched chat as changed."""
    def hashes(where, answer):
        drops = write_export(tmp_path / where, [
            conv("c1", "Odin", exchange(), leaf="r2"),
            conv("c2", "Other", [ask("s1", "q"), reply("s2", answer, "s1")], leaf="s2")])
        adapter = GrokAdapter(drops=drops)
        return {s.native_id: s.raw_hash
                for s in adapter.parse(adapter.discover()[0], ParseStats())}

    before, after = hashes("a", "short"), hashes("b", "a much longer answer")
    assert before["c1"] == after["c1"]
    assert before["c2"] != after["c2"]


def test_conversation_without_id_is_counted(tmp_path):
    drops = write_export(tmp_path, [{"conversation": {"title": "no id"},
                                     "responses": exchange()}])
    stats = ParseStats()
    adapter = GrokAdapter(drops=drops)
    assert list(adapter.parse(adapter.discover()[0], stats)) == []
    assert stats.unknown_types == {"grok:conversation-without-id": 1}


def test_conversation_without_responses_is_counted(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2"),
                                    conv("c2", "Empty", [])])
    stats = ParseStats()
    adapter = GrokAdapter(drops=drops)
    sessions = list(adapter.parse(adapter.discover()[0], stats))
    assert [s.native_id for s in sessions] == ["c1"]
    assert stats.unknown_types == {"grok:conversation-without-responses": 1}


def test_side_collections_are_counted_not_silently_dropped(tmp_path):
    """Empty in the verified export; the next one that fills them should say so."""
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2")],
                         projects=[{"id": "p1"}, {"id": "p2"}], tasks=[{"id": "t1"}])
    stats = ParseStats()
    adapter = GrokAdapter(drops=drops)
    assert len(list(adapter.parse(adapter.discover()[0], stats))) == 1
    assert stats.unknown_types == {"grok:projects": 2, "grok:tasks": 1}


def test_conversation_assets_are_flagged(tmp_path):
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(), leaf="r2",
                                         asset_ids=["a1", "a2"])])
    stats = ParseStats()
    session = parse_one(drops, stats)
    assert session.meta["asset_ids"] == 2
    assert stats.unknown_types == {"grok:conversation-assets": 1}


def test_oversized_answer_goes_to_the_blob_store(tmp_path):
    long_answer = "x" * (40 * 1024)
    drops = write_export(tmp_path, [conv("c1", "Odin", exchange(answer=long_answer),
                                         leaf="r2")])
    stats = ParseStats()
    session = parse_one(drops, stats, blobs=BlobStore(tmp_path / "blobs"))
    part = session.messages[1].parts[0]
    assert part.blob_sha and part.bytes == len(long_answer)
    assert part.text.endswith("<truncated, full text in blob>")
    assert stats.blobs == 1


def test_multi_key_result_envelope_is_flagged(tmp_path):
    """Every result in the verified export is a single-key envelope."""
    steps = [call("card-1", "CodeExecution", {"code": "print(1)"},
                  result={"stdout": "1\n", "exit_code": 0})]
    stats = ParseStats()
    session = parse_one(write_export(tmp_path, [conv(
        "c1", "Odin", exchange(steps=steps), leaf="r2")]), stats)
    parts = session.messages[1].parts
    assert json.loads(parts[1].text) == {"exit_code": 0, "stdout": "1\n"}
    assert stats.unknown_types == {"grok:result-shape": 1}
