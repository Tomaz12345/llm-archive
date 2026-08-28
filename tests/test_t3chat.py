"""Regression tests for the T3 Chat bulk export adapter.

Fixtures are synthesised from the shapes verified in the 14 MB account export
(170 threads / 1430 messages): the provider usage blocks differ per vendor, the two
token counters disagree on reasoning models, and the two server tools return payloads of
completely different kinds. Those are the places this adapter makes a decision, so those
are what is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_archive.adapters.t3chat import T3ChatAdapter
from llm_archive.core.models import (
    KIND_ATTACHMENT, KIND_IMAGE, KIND_TEXT, KIND_THINKING, KIND_TOOL_RESULT,
    KIND_TOOL_USE, ParseStats,
)

T0 = 1787672878879
NAME = "threads-export-2026-08-26T15_25_39.018Z.json"


def export(tmp_path: Path, threads: list[dict], messages: list[dict],
           name: str = NAME) -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    doc = {"threads": threads, "messages": messages, "version": "11.0.1"}
    (drops / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return drops


def thread(tid="t1", title="Reformulacija raziskovalnih vprašanj", model="claude-4.5-opus",
           **kw) -> dict:
    row = {"_id": "jd7" + tid, "_creationTime": T0 + 0.05, "threadId": tid, "id": tid,
           "title": title, "model": model, "modelParams": {"reasoningEffort": "medium"},
           "createdAt": T0, "updatedAt": T0, "lastMessageAt": T0 + 9000,
           "pinned": False, "visibility": "archived", "generationStatus": "completed",
           "userSetTitle": False, "user_edited_title": False, "status": "completed",
           "profileId": "default", "userId": "user_01K"}
    row.update(kw)
    return row


def umsg(mid="m1", tid="t1", content="How do I speed this up?", at=T0, **kw) -> dict:
    row = {"_id": "j97" + mid, "id": "j97" + mid, "_creationTime": at + 0.4,
           "messageId": mid, "threadId": tid, "role": "user", "content": content,
           "created_at": at, "updated_at": at, "status": "done", "attachmentIds": [],
           "model": "claude-4.5-opus", "modelParams": {"reasoningEffort": "medium",
                                                       "includeSearch": False},
           "providerMetadata": {}, "userId": "user_01K"}
    row.update(kw)
    return row


def amsg(mid="m2", tid="t1", parts=None, at=T0 + 1, model="claude-4.5-opus",
         **kw) -> dict:
    parts = [{"type": "text", "text": "Use a bigger batch."}] if parts is None else parts
    row = {"_id": "j97" + mid, "id": "j97" + mid, "_creationTime": at + 0.4,
           "messageId": mid, "threadId": tid, "role": "assistant",
           "content": "".join(p.get("text") or "" for p in parts
                              if p.get("type") == "text"),
           "parts": parts, "created_at": at, "updated_at": at, "status": "done",
           "attachmentIds": [], "model": model,
           "modelParams": {"reasoningEffort": "medium", "includeSearch": False},
           "providerMetadata": {}, "userId": "user_01K"}
    row.update(kw)
    return row


def anthropic(inp=3, out=553, cache_read=0, cache_write=2973) -> dict:
    return {"anthropic": {"cacheCreationInputTokens": cache_write,
                          "usage": {"input_tokens": inp, "output_tokens": out,
                                    "cache_read_input_tokens": cache_read,
                                    "cache_creation_input_tokens": cache_write}}}


def parse_all(drops: Path, stats: ParseStats | None = None):
    adapter = T3ChatAdapter(drops=drops)
    targets = adapter.discover()
    assert len(targets) == 1
    return list(adapter.parse(targets[0], stats or ParseStats()))


def parse_one(drops: Path, stats: ParseStats | None = None):
    sessions = parse_all(drops, stats)
    assert len(sessions) == 1
    return sessions[0]


# -- discovery -------------------------------------------------------------

def test_discovers_by_content_regardless_of_filename(tmp_path):
    drops = export(tmp_path, [thread()], [umsg(), amsg()], name="renamed.json")
    assert [p.name for p in T3ChatAdapter(drops=drops).discover()] == ["renamed.json"]


def test_discovers_by_filename_when_the_marker_is_past_the_sniff_window(tmp_path):
    """`threads` is written first today; the export's own name is the backstop."""
    drops = tmp_path / "drops"
    drops.mkdir(parents=True)
    doc = {"messages": [umsg(), amsg()], "threads": [thread()], "version": "11.0.1"}
    (drops / NAME).write_text(json.dumps(doc), encoding="utf-8")

    import llm_archive.adapters.t3chat as mod
    sniff = mod.SNIFF_BYTES
    mod.SNIFF_BYTES = 8
    try:
        assert [p.name for p in T3ChatAdapter(drops=drops).discover()] == [NAME]
    finally:
        mod.SNIFF_BYTES = sniff


def test_ignores_foreign_json_in_the_same_drop_folder(tmp_path):
    drops = export(tmp_path, [thread()], [umsg(), amsg()])
    (drops / "conversations.json").write_text(
        json.dumps([{"uuid": "c1", "chat_messages": []}]), encoding="utf-8")
    (drops / "OpenRouter Chat.json").write_text(
        json.dumps({"version": "orpg.3.0", "messages": {}}), encoding="utf-8")
    (drops / "broken.json").write_text("{not json", encoding="utf-8")

    assert [p.name for p in T3ChatAdapter(drops=drops).discover()] == [NAME]


def test_a_json_that_is_not_an_export_is_an_error_not_a_crash(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir(parents=True)
    (drops / NAME).write_text(json.dumps({"threads": "nope"}), encoding="utf-8")
    stats = ParseStats()
    assert parse_all(drops, stats) == []
    assert stats.errors == {f"not-an-export:{NAME}": 1}


# -- one file, many chats --------------------------------------------------

def test_one_export_yields_one_session_per_thread(tmp_path):
    drops = export(
        tmp_path,
        [thread("t1", title="First"), thread("t2", title="Second")],
        [umsg("m1", "t1"), amsg("m2", "t1"),
         umsg("m3", "t2"), amsg("m4", "t2")])
    sessions = {s.native_id: s for s in parse_all(drops)}
    assert set(sessions) == {"t1", "t2"}
    assert sessions["t1"].title == "First"
    assert [m.native_id for m in sessions["t2"].messages] == ["m3", "m4"]


def test_hash_is_per_thread_so_one_new_chat_does_not_rewrite_the_rest(tmp_path):
    first = parse_all(export(tmp_path / "a", [thread("t1")], [umsg(), amsg()]))
    grown = parse_all(export(
        tmp_path / "b", [thread("t1"), thread("t2", title="new")],
        [umsg(), amsg(), umsg("m3", "t2"), amsg("m4", "t2")]))
    by_id = {s.native_id: s for s in grown}
    assert by_id["t1"].raw_hash == first[0].raw_hash
    assert by_id["t2"].raw_hash != first[0].raw_hash


def test_empty_thread_yields_no_session(tmp_path):
    assert parse_all(export(tmp_path, [thread("t1")], [])) == []


def test_messages_whose_thread_row_is_missing_are_kept_and_flagged(tmp_path):
    stats = ParseStats()
    session = parse_one(export(tmp_path, [], [umsg(), amsg()]), stats)
    assert session.native_id == "t1"
    assert session.title is None
    assert session.meta["thread_record"] is False
    assert stats.unknown_types == {"t3chat:thread-record-missing": 1}


def test_messages_are_ordered_by_created_at_not_declaration_order(tmp_path):
    drops = export(tmp_path, [thread()],
                   [amsg("m2", at=T0 + 5), umsg("m1", at=T0),
                    amsg("m4", at=T0 + 9), umsg("m3", content="and?", at=T0 + 7)])
    session = parse_one(drops)
    assert [m.native_id for m in session.messages] == ["m1", "m2", "m3", "m4"]
    assert [m.seq for m in session.messages] == [0, 1, 2, 3]
    # No parent pointers exist in this format; none are invented.
    assert all(m.parent_native_id is None for m in session.messages)
    assert all(m.on_active_path for m in session.messages)


# -- parts -----------------------------------------------------------------

def test_parts_win_over_the_duplicated_content_field(tmp_path):
    parts = [{"type": "reasoning", "reasoning": "they want throughput"},
             {"type": "text", "text": "Use a bigger batch."}]
    session = parse_one(export(tmp_path, [thread()],
                               [umsg(), amsg(parts=parts, content="Use a bigger batch.")]))
    answer = session.messages[1]
    assert [p.kind for p in answer.parts] == [KIND_THINKING, KIND_TEXT]
    assert [p.seq for p in answer.parts] == [0, 1]
    assert answer.parts[0].embed_eligible is True     # real reasoning text, unlike CC


def test_user_turns_fall_back_to_content_because_they_carry_no_parts(tmp_path):
    session = parse_one(export(tmp_path, [thread()], [umsg(content="hi"), amsg()]))
    prompt = session.messages[0]
    assert [p.kind for p in prompt.parts] == [KIND_TEXT]
    assert prompt.parts[0].text == "hi"
    assert prompt.model is None            # the selected model, not who answered


def test_empty_reasoning_blocks_are_dropped(tmp_path):
    parts = [{"type": "reasoning", "reasoning": "", "providerMetadata": {"anthropic": {}}},
             {"type": "text", "text": "Use a bigger batch."}]
    session = parse_one(export(tmp_path, [thread()], [umsg(), amsg(parts=parts)]))
    assert [p.kind for p in session.messages[1].parts] == [KIND_TEXT]


def test_attachments_are_counted_but_have_no_recoverable_content(tmp_path):
    drops = export(tmp_path, [thread()],
                   [umsg(content="I have this:", attachmentIds=["a1", "a2"]), amsg()])
    prompt = parse_one(drops).messages[0]
    assert [p.kind for p in prompt.parts] == [KIND_TEXT, KIND_ATTACHMENT,
                                              KIND_ATTACHMENT]
    assert all(p.text is None for p in prompt.parts[1:])


def test_an_image_only_prompt_survives_on_its_attachment_alone(tmp_path):
    drops = export(tmp_path, [thread()],
                   [umsg(content="", attachmentIds=["a1"]), amsg()])
    assert [p.kind for p in parse_one(drops).messages[0].parts] == [KIND_ATTACHMENT]


def test_a_failed_answer_with_no_text_is_counted_not_stored(tmp_path):
    stats = ParseStats()
    error = amsg(parts=[], content="", status="error",
                 serverError={"message": "Your chat is too long.",
                              "type": "input_too_long"})
    session = parse_one(export(tmp_path, [thread()], [umsg(), error]), stats)
    assert [m.role for m in session.messages] == ["user"]
    assert stats.unknown_types == {"t3chat:empty-message:error": 1}


def test_unknown_part_type_is_counted_never_fatal(tmp_path):
    stats = ParseStats()
    parts = [{"type": "video_generation"}, {"type": "text", "text": "done"}]
    session = parse_one(export(tmp_path, [thread()], [umsg(), amsg(parts=parts)]), stats)
    assert [p.kind for p in session.messages[1].parts] == [KIND_TEXT]
    assert stats.unknown_types == {"t3chat:part:video_generation": 1}


# -- server tools ----------------------------------------------------------

def test_web_search_keeps_the_scraped_page_but_never_embeds_it(tmp_path):
    call = {"type": "tool_call", "toolName": "webSearch", "toolCallId": "tc1",
            "status": "completed",
            "args": {"queries": ["docker compose up --build"], "category": "news"},
            "result": [{"title": "Using the build cache",
                        "url": "https://docs.docker.com/x/",
                        "summary": "How the cache works",
                        "content": "Consider the following Dockerfile"}]}
    parts = parse_one(export(tmp_path, [thread()],
                             [umsg(), amsg(parts=[call,
                                                  {"type": "text", "text": "In short:"}])
                              ])).messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_TOOL_RESULT, KIND_TEXT]
    assert [p.seq for p in parts] == [0, 1, 2]
    assert parts[0].text == "docker compose up --build"
    assert parts[0].tool_name == "webSearch"
    assert "Consider the following Dockerfile" in parts[1].text
    assert "https://docs.docker.com/x/" in parts[1].text
    assert parts[1].embed_eligible is False           # §1.1
    assert parts[1].tool_ok is True


def test_generated_images_are_recorded_as_the_reference_the_export_gives(tmp_path):
    call = {"type": "tool_call", "toolName": "image_generation", "toolCallId": "tc2",
            "status": "completed", "args": {},
            "result": [{"completed": True, "fileName": "anime--t3chat--1.jpg",
                        "url": "https://upoevdcxa3.ufs.sh/f/IN4"}]}
    parts = parse_one(export(tmp_path, [thread()],
                             [umsg(), amsg(parts=[call])])).messages[1].parts
    assert [p.kind for p in parts] == [KIND_TOOL_USE, KIND_IMAGE]
    assert parts[1].text == "anime--t3chat--1.jpg https://upoevdcxa3.ufs.sh/f/IN4"
    assert parts[1].embed_eligible is False


# -- usage, cost and attribution -------------------------------------------

def test_anthropic_usage_splits_cache_reads_from_cache_writes(tmp_path):
    answer = amsg(tokens=553, providerMetadata=anthropic(inp=3, out=553,
                                                         cache_read=61, cache_write=2973))
    session = parse_one(export(tmp_path, [thread()], [umsg(), answer]))
    assert (session.tok_in, session.tok_out) == (3, 553)
    assert (session.tok_cache_read, session.tok_cache_write) == (61, 2973)
    assert session.messages[1].meta["provider"] == "anthropic"


def test_t3s_own_counter_wins_because_the_provider_drops_reasoning_tokens(tmp_path):
    """gemini-3-flash-thinking: 1240 streamed, 471 in `candidatesTokenCount`."""
    answer = amsg(model="gemini-3-flash-thinking", tokens=1240,
                  providerMetadata={"google": {"usageMetadata": {
                      "promptTokenCount": 1968, "candidatesTokenCount": 471,
                      "totalTokenCount": 2439}}})
    session = parse_one(export(tmp_path, [thread()], [umsg(), answer]))
    assert session.tok_out == 1240
    assert session.tok_in == 1968
    assert session.tok_cache_read is None      # Google's block does not split cache


def test_openai_answers_have_no_prompt_count_anywhere(tmp_path):
    answer = amsg(model="gpt-5.2-instant", tokens=101,
                  providerMetadata={"openai": {"responseId": "resp_03e",
                                               "serviceTier": "default"}})
    session = parse_one(export(tmp_path, [thread()], [umsg(), answer]))
    assert session.tok_out == 101
    assert session.tok_in is None              # not zero — the export has no number


def test_only_openrouter_routed_answers_carry_a_real_charge(tmp_path):
    billed = amsg("m2", model="kimi-k2-0905", tokens=148, byok=True,
                  providerMetadata={"openrouter": {"provider": "Groq", "usage": {
                      "promptTokens": 595, "completionTokens": 148,
                      "cost": 0.001039}}})
    free = amsg("m4", at=T0 + 3, tokens=553, providerMetadata=anthropic())
    session = parse_one(export(tmp_path, [thread()],
                               [umsg(), billed, umsg("m3", at=T0 + 2), free]))
    assert session.cost_usd == 0.001039
    assert session.messages[1].meta["upstream_provider"] == "Groq"
    assert session.messages[1].meta["byok"] is True
    assert "cost_usd" not in session.messages[3].meta


def test_a_subscription_thread_reports_no_cost_at_all(tmp_path):
    session = parse_one(export(tmp_path, [thread()],
                               [umsg(), amsg(tokens=553,
                                             providerMetadata=anthropic())]))
    assert session.cost_usd is None            # covered by the T3 subscription


def test_primary_model_is_who_answered_not_the_selected_one(tmp_path):
    """Switching model mid-thread leaves `thread.model` pointing at the last pick."""
    drops = export(
        tmp_path,
        [thread(model="gemini-3-flash")],
        [umsg("m1", at=T0), amsg("m2", at=T0 + 1, model="claude-4.5-opus"),
         umsg("m3", at=T0 + 2), amsg("m4", at=T0 + 3, model="claude-4.5-opus"),
         umsg("m5", at=T0 + 4), amsg("m6", at=T0 + 5, model="gemini-3-flash")])
    session = parse_one(drops)
    assert session.model_primary == "claude-4.5-opus"
    assert session.meta["models"] == ["claude-4.5-opus", "gemini-3-flash"]
    assert session.meta["selected_model"] == "gemini-3-flash"
    assert "participant" not in session.meta       # no single answerer to file it under


def test_single_model_thread_opts_into_the_by_assistant_breakdown(tmp_path):
    session = parse_one(export(tmp_path, [thread()], [umsg(), amsg()]))
    assert session.meta["participant"] == "claude-4.5-opus"
    assert session.meta["participant_label"] == "claude-4.5-opus"


# -- session fields --------------------------------------------------------

def test_thread_flags_and_timestamps_survive(tmp_path):
    row = thread(pinned=True, visibility="visible", generationStatus="failed",
                 userSetTitle=True, forkedFromSharedThread="bufswn9opb")
    session = parse_one(export(tmp_path, [row], [umsg(), amsg()]))
    assert session.started_at == T0
    assert session.ended_at == T0 + 9000
    assert session.title_source == "provider"
    assert session.meta["pinned"] is True
    assert session.meta["visibility"] == "visible"
    assert session.meta["generation_status"] == "failed"
    assert session.meta["user_set_title"] is True
    # A fork of someone else's shared chat: provenance, not a session we can link to.
    assert session.meta["forked_from_shared"] == "bufswn9opb"
    assert session.parent_native_id is None
    assert session.meta["export_version"] == "11.0.1"


def test_message_meta_keeps_the_generation_settings(tmp_path):
    answer = amsg(tokens=553, timeToFirstToken=0.434, tokensPerSecond=142.7,
                  modelParams={"reasoningEffort": "high", "includeSearch": True},
                  providerMetadata=anthropic())
    meta = parse_one(export(tmp_path, [thread()], [umsg(), answer])).messages[1].meta
    assert meta["reasoning_effort"] == "high"
    assert meta["include_search"] is True
    assert meta["time_to_first_token"] == 0.434
    assert meta["tokens_per_second"] == 142.7


def test_scraped_pages_overflow_to_blobs_and_are_counted(tmp_path):
    from llm_archive.core.blobs import BlobStore
    from llm_archive.core.models import INLINE_LIMIT

    page = "y" * (INLINE_LIMIT + 1000)
    call = {"type": "tool_call", "toolName": "webSearch", "toolCallId": "tc1",
            "status": "completed", "args": {"queries": ["big page"]},
            "result": [{"title": "T", "url": "https://x/", "content": page}]}
    drops = export(tmp_path, [thread()], [umsg(), amsg(parts=[call])])
    stats = ParseStats()
    adapter = T3ChatAdapter(drops=drops, blobs=BlobStore(tmp_path / "blobs"))
    session = next(adapter.parse(adapter.discover()[0], stats))

    result = session.messages[1].parts[1]
    assert result.kind == KIND_TOOL_RESULT
    assert result.blob_sha is not None
    assert result.text.endswith("<truncated, full text in blob>")
    assert stats.blobs == 1
