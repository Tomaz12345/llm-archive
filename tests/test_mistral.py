"""Regression tests for the Mistral (Le Chat) export adapter.

The three things this format gets wrong in ways that fail silently — reasoning that is
typed `text` like the answer, an answer that ships twice (`content` *and* the last
chunk), and a ZIP whose name identifies nothing — each get a test here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_archive.adapters.mistral import MistralAdapter
from llm_archive.adapters.openrouter import OpenRouterAdapter
from llm_archive.adapters.t3chat import T3ChatAdapter
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import (
    KIND_ATTACHMENT, KIND_TEXT, KIND_THINKING, KIND_TOOL_USE, ParseStats,
)

CHAT = "bda873f6-e352-4b55-9d77-398dd4ba6446"
ASKED = "2026-08-26T20:45:30.594Z"
ANSWERED = "2026-08-26T20:45:31.006Z"
COMPLETED_MS = 1787777155420        # 24.4s after the answer's own createdAt


def write_export(tmp_path: Path, chats: dict[str, list], as_zip=True,
                 name="chat-export-1787777208552.zip") -> Path:
    """`chats` maps chat id -> message array, one member per chat."""
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    if not as_zip:
        for cid, records in chats.items():
            (drops / f"chat-{cid}.json").write_text(
                json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return drops
    with zipfile.ZipFile(drops / name, "w") as zf:
        for cid, records in chats.items():
            zf.writestr(f"chat-{cid}.json", json.dumps(records, ensure_ascii=False))
    return drops


def ask(text="Hey, give me a tutorial about Go programming language.", *,
        mid="0a4cefe3", created=ASKED, version=0, chat=CHAT):
    return {"id": mid, "version": version, "chatId": chat, "content": text,
            "contentChunks": None, "role": "user", "createdAt": created,
            "reaction": "neutral", "reactionDetail": None, "reactionComment": None,
            "preference": None, "preferenceOver": None,
            "context": {"openCanvaId": None, "initiatedFromUrlQuery": False},
            "canvas": [], "files": []}


def reply(text="### Golang Tutorial", reasoning="The user wants a tutorial about Go.", *,
          mid="90aab74b", created=ANSWERED, version=0, chat=CHAT, tokens=1266,
          chunks=..., tools=()):
    if chunks is ...:
        chunks = []
        if reasoning:
            chunks.append({"text": reasoning, "type": "text",
                           "_context": {"type": "reasoning", "startTime": 1787777148026,
                                        "endTime": 1787777149176,
                                        "contextId": "70dac4d2"}})
        chunks.append({"text": text, "type": "text"})
    return {"id": mid, "version": version, "chatId": chat, "content": text,
            "contentChunks": chunks, "role": "assistant", "createdAt": created,
            "reaction": "neutral", "reactionDetail": None, "reactionComment": None,
            "preference": None, "preferenceOver": None,
            "context": {
                "openCanvaId": None,
                "completionTiming": {
                    "startedAtMs": 1787777131016, "completedAtMs": COMPLETED_MS,
                    "firstTokenAtMs": 1787777148016, "timeToFirstTokenMs": 17000,
                    "completionDurationMs": 24404,
                    "generationStats": {"outputTokens": tokens, "completionCount": 1,
                                        "generationDurationMs": 7165,
                                        "outputTokensPerSecond": 176.69225401256105}},
                "assistantAnswerSignals": {"toolNames": list(tools),
                                           "integrationNames": []}},
            "canvas": [], "files": []}


def parse_one(drops: Path, stats: ParseStats | None = None, **kw):
    adapter = MistralAdapter(drops=drops, **kw)
    targets = adapter.discover()
    assert len(targets) == 1
    return next(adapter.parse(targets[0], stats or ParseStats()))


# -- discovery -------------------------------------------------------------

def test_reads_chat_from_zip(tmp_path):
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]})
    session = parse_one(drops)
    assert session.source_kind == "mistral"
    assert session.native_id == CHAT
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_reads_bare_chat_json(tmp_path):
    """The ZIP is trivial to unpack by hand; an unzipped drop still gets ingested."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]}, as_zip=False)
    assert parse_one(drops).native_id == CHAT


def test_export_name_is_not_the_marker(tmp_path):
    """`chat-export-<epoch ms>.zip` names the moment, not the source — sniff the bytes."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]}, name="whatever.zip")
    assert parse_one(drops).native_id == CHAT


def test_ignores_foreign_drops(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir(parents=True)
    (drops / "conversations.json").write_text(
        json.dumps([{"uuid": "c1", "chat_messages": []}]), encoding="utf-8")
    (drops / "threads-export-2026.json").write_text(
        json.dumps({"threads": [], "messages": []}), encoding="utf-8")
    (drops / "chat-notes.txt").write_text("not json", encoding="utf-8")
    assert MistralAdapter(drops=drops).discover() == []


def test_does_not_feed_the_other_drop_adapters(tmp_path):
    """A bare Le Chat member must not be claimed by the two adapters that scan .json."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]}, as_zip=False)
    assert OpenRouterAdapter(drops=drops).discover() == []
    assert T3ChatAdapter(drops=drops).discover() == []


def test_multiple_chats_in_one_zip(tmp_path):
    """One member per chat; each becomes its own session, hashed separately."""
    other = "11111111-2222-3333-4444-555555555555"
    drops = write_export(tmp_path, {
        CHAT: [ask(), reply()],
        other: [ask("Second chat", mid="aaa", chat=other),
                reply("Second answer", mid="bbb", chat=other)],
    })
    adapter = MistralAdapter(drops=drops)
    sessions = list(adapter.parse(adapter.discover()[0], ParseStats()))
    assert sorted(s.native_id for s in sessions) == sorted([CHAT, other])
    assert len({s.raw_hash for s in sessions}) == 2


# -- parts -----------------------------------------------------------------

def test_reasoning_chunk_is_thinking_not_text(tmp_path):
    """Point 3: both chunks are typed `text`; only `_context.type` separates them."""
    session = parse_one(tmp_path_drops := write_export(
        tmp_path, {CHAT: [ask(), reply(text="Answer body", reasoning="Planning notes")]}))
    assert tmp_path_drops.exists()
    answer = session.messages[1]
    assert [(p.kind, p.text) for p in answer.parts] == [
        (KIND_THINKING, "Planning notes"), (KIND_TEXT, "Answer body")]
    # Real prose, unlike Claude Code's signature-only blocks, so it is embedded.
    assert all(p.embed_eligible for p in answer.parts)


def test_answer_is_not_stored_twice(tmp_path):
    """Point 2: `content` duplicates the final chunk byte for byte."""
    body = "### Golang Tutorial\n\nGo is a language."
    drops = write_export(tmp_path, {CHAT: [ask(), reply(text=body)]})
    answer = parse_one(drops).messages[1]
    assert [p.text for p in answer.parts if p.kind == KIND_TEXT] == [body]


def test_content_is_the_fallback_when_there_are_no_chunks(tmp_path):
    """Every user turn has `contentChunks: null`, and so does an older assistant turn."""
    drops = write_export(tmp_path, {CHAT: [ask("Just a question"),
                                           reply(text="Just an answer", chunks=None)]})
    session = parse_one(drops)
    assert [p.text for m in session.messages for p in m.parts] == [
        "Just a question", "Just an answer"]


def test_unknown_chunk_type_is_counted_not_guessed(tmp_path):
    drops = write_export(tmp_path, {CHAT: [ask(), reply(
        chunks=[{"text": "{}", "type": "tool_call"}, {"text": "Answer", "type": "text"}])]})
    stats = ParseStats()
    session = parse_one(drops, stats)
    assert stats.unknown_types["mistral:chunk:tool_call"] == 1
    assert [p.text for p in session.messages[1].parts] == ["Answer"]


def test_tool_signals_become_bare_markers(tmp_path):
    """The export names the tools a turn used and never their arguments or results."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply(tools=["web_search"])]})
    stats = ParseStats()
    session = parse_one(drops, stats)
    call = [p for p in session.messages[1].parts if p.kind == KIND_TOOL_USE]
    assert [p.tool_name for p in call] == ["web_search"]
    assert not call[0].embed_eligible and not call[0].text
    assert stats.unknown_types["mistral:tool-signals-without-payload"] == 1


def test_attachment_names_kept_bytes_flagged_missing(tmp_path):
    prompt = ask()
    prompt["files"] = [{"id": "f1", "name": "spec.pdf"}]
    drops = write_export(tmp_path, {CHAT: [prompt, reply()]})
    stats = ParseStats()
    session = parse_one(drops, stats)
    attached = [p for p in session.messages[0].parts if p.kind == KIND_ATTACHMENT]
    assert [p.text for p in attached] == ["spec.pdf"]
    assert stats.unknown_types["mistral:attachment-bytes-not-exported"] == 1


def test_canvas_is_counted_not_dropped_silently(tmp_path):
    answer = reply()
    answer["canvas"] = [{"id": "canva-1"}]
    drops = write_export(tmp_path, {CHAT: [ask(), answer]})
    stats = ParseStats()
    parse_one(drops, stats)
    assert stats.unknown_types["mistral:canvas-not-parsed"] == 1


def test_oversized_answer_goes_to_the_blob_store(tmp_path):
    huge = "x" * 40_000
    drops = write_export(tmp_path, {CHAT: [ask(), reply(text=huge, reasoning="")]})
    stats = ParseStats()
    session = parse_one(drops, stats, blobs=BlobStore(tmp_path / "blobs"))
    part = session.messages[1].parts[0]
    assert part.blob_sha and part.bytes == 40_000
    assert part.text.endswith("<truncated, full text in blob>")
    assert stats.blobs == 1


# -- session fields --------------------------------------------------------

def test_title_comes_from_the_first_prompt(tmp_path):
    """Point 1: there is no chat object in the file, so there is no exported title."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]})
    session = parse_one(drops)
    assert session.title == "Hey, give me a tutorial about Go programming language."
    assert session.title_source == "first_prompt"


def test_long_prompt_is_clipped_on_a_word_boundary(tmp_path):
    prompt = "word " * 40
    drops = write_export(tmp_path, {CHAT: [ask(prompt), reply()]})
    title = parse_one(drops).title
    assert title.endswith("…") and len(title) <= 81 and "  " not in title


def test_ended_at_uses_completion_not_the_start_stamp(tmp_path):
    """Point 4: an assistant turn is stamped when generation began, not when it ended."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply()]})
    session = parse_one(drops)
    assert session.started_at == 1787777130594          # the prompt, in UTC ms
    assert session.ended_at == COMPLETED_MS
    assert session.ended_at > session.messages[1].created_at

def test_output_tokens_only_no_model_no_cost(tmp_path):
    """Points 4 and 5: the three columns this export genuinely cannot fill."""
    drops = write_export(tmp_path, {CHAT: [ask(), reply(tokens=1266)]})
    session = parse_one(drops)
    assert session.tok_out == 1266
    assert session.tok_in is None and session.cost_usd is None
    assert session.model_primary is None
    assert all(m.model is None for m in session.messages)
    # No model means no answerer to file the session under, so the app stands in.
    assert session.meta["participant"] == "mistral"
    assert session.meta["model_recorded"] is False


def test_reaction_recorded_only_when_the_user_voted(tmp_path):
    liked = reply()
    liked["reaction"] = "good"
    liked["reactionComment"] = "clear"
    drops = write_export(tmp_path, {CHAT: [ask(), liked]})
    session = parse_one(drops)
    assert session.messages[0].meta.get("reaction") is None      # 'neutral' is the default
    assert session.messages[1].meta["reaction"] == "good"
    assert session.messages[1].meta["reaction_comment"] == "clear"
    assert session.meta["reactions"] == ["good"]


def test_rehash_is_per_chat_not_per_drop(tmp_path):
    """Re-exporting lands under a new filename; the same chat must not fork a session."""
    first = parse_one(write_export(tmp_path / "a", {CHAT: [ask(), reply()]},
                                   name="chat-export-1787777208552.zip"))
    second = parse_one(write_export(tmp_path / "b", {CHAT: [ask(), reply()]},
                                    name="chat-export-1787999999999.zip"))
    assert first.native_id == second.native_id
    assert first.raw_hash == second.raw_hash


# -- versions --------------------------------------------------------------

def test_higher_version_supersedes_the_same_message_id(tmp_path):
    """UNVERIFIED shape: `version` is 0 throughout the sample. Counted when it fires."""
    drops = write_export(tmp_path, {CHAT: [
        ask(), reply(text="First take", version=0),
        reply(text="Second take", version=1, created="2026-08-26T20:46:00.000Z")]})
    stats = ParseStats()
    session = parse_one(drops, stats)
    active = [m for m in session.messages if m.on_active_path]
    assert [p.text for m in active for p in m.parts if p.kind == KIND_TEXT] == [
        "Hey, give me a tutorial about Go programming language.", "Second take"]
    orphan = next(m for m in session.messages if not m.on_active_path)
    assert orphan.native_id == "90aab74b:v0"          # unique on (session, native_id)
    assert stats.orphaned_messages == 1
    assert stats.unknown_types["mistral:duplicate-message-id"] == 1
    # A superseded answer's tokens are not part of the chat's total.
    assert session.tok_out == 1266
