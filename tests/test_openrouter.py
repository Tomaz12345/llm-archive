"""Regression tests for the OpenRouter per-chat export adapter.

The three sample exports in `data/drops/` are all single-model, single-turn, so the
branch cases below are synthesised: multi-model fan-out, retries and edited prompts are
the situations where this adapter deliberately diverges from the shared tree resolver,
and nothing on disk exercises them.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_archive.adapters.openrouter import OpenRouterAdapter
from llm_archive.core.models import KIND_TEXT, KIND_THINKING, ParseStats

TS = "2026-08-25T10:00:00.000Z"


def export(tmp_path: Path, messages: list[dict], items: list[dict],
           characters: list[dict] | None = None, title: str = "Hey, can you tell me",
           name: str = "OpenRouter Chat Tue Aug 25 2026.json") -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "orpg.3.0",
        "title": title,
        "characters": {c["id"]: c for c in (characters or [])},
        "messages": {m["id"]: m for m in messages},
        "items": {i["id"]: i for i in items},
        "artifacts": {}, "artifactFiles": {},
        "artifactVersions": {}, "artifactFileContents": {},
    }
    (drops / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return drops


def char(cid, model, removed=False):
    return {"id": cid, "model": model, "isRemoved": removed, "isDisabled": False}


def umsg(mid, item_id, ts=TS, parent=None):
    return {"id": mid, "characterId": "USER", "type": "user", "context": "main-chat",
            "parentMessageId": parent, "createdAt": ts, "updatedAt": ts,
            "isEdited": False, "isRetrying": False,
            "items": [{"id": item_id, "type": "message"}]}


def amsg(mid, cid, parent, refs, ts=TS, model=None, tokens=None, cost="0"):
    return {"id": mid, "characterId": cid, "type": "assistant", "context": "main-chat",
            "parentMessageId": parent, "createdAt": ts, "updatedAt": ts,
            "isEdited": False, "isRetrying": False,
            "metadata": {"variantSlug": model, "tokensCount": tokens, "cost": cost,
                         "duration": 1000, "generationId": "gen-x"},
            "items": refs}


def ref(item_id, seq):
    return {"id": item_id, "outputIndex": seq, "type": "message", "sequenceIndex": seq}


def item(iid, mid, dtype, text, role=None):
    ctype = {"message": "output_text", "reasoning": "reasoning_text"}[dtype]
    if role == "user":
        ctype = "input_text"
    data = {"type": dtype, "content": [{"type": ctype, "text": text}]}
    if role:
        data["role"] = role
    return {"id": iid, "messageId": mid, "data": data}


def simple(tmp_path, **kw):
    """One prompt, one answer with reasoning + text."""
    return export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0), ref("i3", 1)],
              model="minimax/minimax-m3:free", tokens=776)],
        [item("i1", "m1", "message", "Hey, can you tell me about V?", role="user"),
         item("i2", "m2", "reasoning", "thinking about V"),
         item("i3", "m2", "message", "V is a compiled language.")],
        [char("c1", "minimax/minimax-m3:free")],
        **kw)


def parse_one(drops: Path, stats: ParseStats | None = None):
    adapter = OpenRouterAdapter(drops=drops)
    targets = adapter.discover()
    assert len(targets) == 1
    sessions = list(adapter.parse(targets[0], stats or ParseStats()))
    assert len(sessions) == 1
    return sessions[0]


# -- discovery -------------------------------------------------------------

def test_discovers_by_schema_version_not_filename(tmp_path):
    drops = simple(tmp_path, name="OpenRouter Chat Tue Aug 25 2026(2).json")
    assert [p.name for p in OpenRouterAdapter(drops=drops).discover()] == [
        "OpenRouter Chat Tue Aug 25 2026(2).json"]


def test_ignores_foreign_json_in_the_same_drop_folder(tmp_path):
    drops = simple(tmp_path)
    (drops / "conversations.json").write_text(
        json.dumps([{"uuid": "c1", "chat_messages": []}]), encoding="utf-8")
    (drops / "settings.json").write_text(json.dumps({"version": 3}), encoding="utf-8")
    (drops / "broken.json").write_text("{not json", encoding="utf-8")

    found = [p.name for p in OpenRouterAdapter(drops=drops).discover()]
    assert found == ["OpenRouter Chat Tue Aug 25 2026.json"]


# -- basic parse -----------------------------------------------------------

def test_joins_the_three_id_keyed_maps(tmp_path):
    session = parse_one(simple(tmp_path))
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert [p.kind for p in session.messages[1].parts] == [KIND_THINKING, KIND_TEXT]
    assert session.messages[1].parts[1].text == "V is a compiled language."


def test_reasoning_is_kept_and_embedded(tmp_path):
    thinking = parse_one(simple(tmp_path)).messages[1].parts[0]
    assert thinking.kind == KIND_THINKING
    assert thinking.text == "thinking about V"
    assert thinking.embed_eligible is True


def test_parts_follow_sequence_index_not_declaration_order(tmp_path):
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i3", 1), ref("i2", 0)], model="x")],
        [item("i1", "m1", "message", "hi", role="user"),
         item("i2", "m2", "reasoning", "first"),
         item("i3", "m2", "message", "second")],
        [char("c1", "x")])
    assert [p.kind for p in parse_one(drops).messages[1].parts] == [
        KIND_THINKING, KIND_TEXT]


def test_tokens_count_is_output_only_and_cost_parses(tmp_path):
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], model="x", tokens=776,
              cost="0.00042")],
        [item("i1", "m1", "message", "hi", role="user"),
         item("i2", "m2", "message", "hello")],
        [char("c1", "x")])
    session = parse_one(drops)
    assert session.tok_out == 776
    assert session.tok_in is None            # the export records no prompt tokens
    assert session.cost_usd == 0.00042
    assert session.messages[0].tok_out is None   # never attributed to the user turn


def test_free_tier_zero_cost_stays_null_not_zero(tmp_path):
    assert parse_one(simple(tmp_path)).cost_usd is None


# -- attribution -----------------------------------------------------------

def test_removed_characters_are_not_counted_as_models(tmp_path):
    """The sample 3-model chat registers two models that never answered."""
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c2", "m1", [ref("i2", 0)],
              model="dots-studio/dots-3-note-preview:free")],
        [item("i1", "m1", "message", "weather?", role="user"),
         item("i2", "m2", "message", "no live data")],
        [char("c1", "minimax/minimax-m3:free", removed=True),
         char("c2", "dots-studio/dots-3-note-preview:free"),
         char("c3", "google/gemma-4-31b-it:free", removed=True)])
    session = parse_one(drops)
    assert session.meta["models"] == ["dots-studio/dots-3-note-preview:free"]
    assert session.model_primary == "dots-studio/dots-3-note-preview:free"
    assert session.meta["multi_model"] is False


def test_provider_and_router_strategy_land_in_message_meta(tmp_path):
    msg = amsg("m2", "c1", "m1", [ref("i2", 0)], model="x", tokens=5)
    msg["metadata"]["routerMetadata"] = {
        "strategy": "direct",
        "endpoints": {"available": [{"provider": "GMICloud", "selected": True},
                                    {"provider": "Other", "selected": False}]}}
    drops = export(tmp_path, [umsg("m1", "i1"), msg],
                   [item("i1", "m1", "message", "hi", role="user"),
                    item("i2", "m2", "message", "yo")],
                   [char("c1", "x")])
    meta = parse_one(drops).messages[1].meta
    assert meta["provider"] == "GMICloud"
    assert meta["router_strategy"] == "direct"


# -- the branch rules ------------------------------------------------------

def test_parallel_model_answers_all_stay_on_the_active_path(tmp_path):
    """A multi-model chat fans out; the shared newest-leaf resolver would keep one."""
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], ts="2026-08-25T10:00:01.000Z",
              model="model-a", tokens=10),
         amsg("m3", "c2", "m1", [ref("i3", 0)], ts="2026-08-25T10:00:02.000Z",
              model="model-b", tokens=20),
         amsg("m4", "c3", "m1", [ref("i4", 0)], ts="2026-08-25T10:00:03.000Z",
              model="model-c", tokens=30)],
        [item("i1", "m1", "message", "one prompt", role="user"),
         item("i2", "m2", "message", "answer A"),
         item("i3", "m3", "message", "answer B"),
         item("i4", "m4", "message", "answer C")],
        [char("c1", "model-a"), char("c2", "model-b"), char("c3", "model-c")])

    stats = ParseStats()
    session = parse_one(drops, stats)
    assert all(m.on_active_path for m in session.messages)
    assert stats.orphaned_messages == 0
    assert session.meta["models"] == ["model-a", "model-b", "model-c"]
    assert session.meta["multi_model"] is True
    assert session.tok_out == 60          # every parallel answer counts


def test_multi_model_session_opts_out_of_the_participant_breakdown(tmp_path):
    """One session holds one `participant`; three answerers cannot pick one."""
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], model="model-a"),
         amsg("m3", "c2", "m1", [ref("i3", 0)], model="model-b")],
        [item("i1", "m1", "message", "q", role="user"),
         item("i2", "m2", "message", "A"),
         item("i3", "m3", "message", "B")],
        [char("c1", "model-a"), char("c2", "model-b")])
    assert "participant" not in parse_one(drops).meta


def test_single_model_session_fills_the_participant_breakdown(tmp_path):
    meta = parse_one(simple(tmp_path)).meta
    assert meta["participant"] == "minimax/minimax-m3:free"
    assert meta["participant_label"] == "minimax/minimax-m3:free"


def test_retrying_the_same_model_supersedes_the_earlier_answer(tmp_path):
    """Same character, same parent — a retry, not a second opinion."""
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], ts="2026-08-25T10:00:01.000Z",
              model="model-a", tokens=10),
         amsg("m3", "c1", "m1", [ref("i3", 0)], ts="2026-08-25T10:00:09.000Z",
              model="model-a", tokens=20)],
        [item("i1", "m1", "message", "q", role="user"),
         item("i2", "m2", "message", "first attempt"),
         item("i3", "m3", "message", "retried answer")],
        [char("c1", "model-a")])

    stats = ParseStats()
    session = parse_one(drops, stats)
    live = [m for m in session.messages if m.on_active_path]
    assert [m.native_id for m in live] == ["m1", "m3"]
    assert stats.orphaned_messages == 1
    assert session.tok_out == 20          # the abandoned attempt is not billed twice


def test_editing_a_prompt_orphans_the_whole_branch_below_it(tmp_path):
    drops = export(
        tmp_path,
        [umsg("m1", "i1", ts="2026-08-25T10:00:00.000Z"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], ts="2026-08-25T10:00:01.000Z",
              model="model-a", tokens=10),
         umsg("m3", "i3", ts="2026-08-25T10:00:05.000Z"),
         amsg("m4", "c1", "m3", [ref("i4", 0)], ts="2026-08-25T10:00:06.000Z",
              model="model-a", tokens=20)],
        [item("i1", "m1", "message", "original prompt", role="user"),
         item("i2", "m2", "message", "answer to original"),
         item("i3", "m3", "message", "edited prompt", role="user"),
         item("i4", "m4", "message", "answer to edit")],
        [char("c1", "model-a")])

    stats = ParseStats()
    session = parse_one(drops, stats)
    live = [m.native_id for m in session.messages if m.on_active_path]
    assert live == ["m3", "m4"]
    assert stats.orphaned_messages == 2    # the answer under the dead prompt goes too
    assert session.msg_count == 2


def test_a_missing_parent_link_is_treated_as_a_root(tmp_path):
    drops = export(
        tmp_path,
        [umsg("m1", "i1", parent="gone-in-export"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], model="model-a")],
        [item("i1", "m1", "message", "q", role="user"),
         item("i2", "m2", "message", "a")],
        [char("c1", "model-a")])
    session = parse_one(drops)
    assert all(m.on_active_path for m in session.messages)


# -- identity and idempotence ---------------------------------------------

def test_native_id_is_the_root_message_not_the_filename(tmp_path):
    """`(1)`/`(2)` suffixes are browser dedup noise, not chat identity."""
    first = parse_one(simple(tmp_path))
    second = parse_one(simple(tmp_path / "again",
                              name="OpenRouter Chat Tue Aug 25 2026(1).json"))
    assert first.native_id == "m1"
    assert first.native_id == second.native_id
    assert first.raw_hash == second.raw_hash   # re-export updates, never forks


def test_editing_the_first_prompt_does_not_fork_the_session(tmp_path):
    """Identity anchors on the earliest message, superseded or not."""
    before = parse_one(simple(tmp_path))
    drops = export(
        tmp_path / "edited",
        [umsg("m1", "i1", ts="2026-08-25T10:00:00.000Z"),
         amsg("m2", "c1", "m1", [ref("i2", 0)], ts="2026-08-25T10:00:01.000Z",
              model="minimax/minimax-m3:free"),
         umsg("m3", "i3", ts="2026-08-25T10:04:00.000Z"),
         amsg("m4", "c1", "m3", [ref("i4", 0)], ts="2026-08-25T10:04:01.000Z",
              model="minimax/minimax-m3:free")],
        [item("i1", "m1", "message", "Hey, can you tell me about V?", role="user"),
         item("i2", "m2", "message", "V is a compiled language."),
         item("i3", "m3", "message", "Hey, tell me about Zig instead", role="user"),
         item("i4", "m4", "message", "Zig is a systems language.")],
        [char("c1", "minimax/minimax-m3:free")],
        title="Hey, tell me about Zig")          # the stub follows the edited prompt

    after = parse_one(drops)
    assert after.native_id == before.native_id == "m1"
    assert after.title == "Hey, tell me about Zig instead"   # title follows the edit
    assert after.raw_hash != before.raw_hash                 # same session, new content


def test_growing_a_chat_changes_the_hash(tmp_path):
    before = parse_one(simple(tmp_path))
    drops = export(
        tmp_path / "later",
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0), ref("i3", 1)],
              model="minimax/minimax-m3:free", tokens=776),
         umsg("m4", "i4", ts="2026-08-25T10:05:00.000Z", parent="m2")],
        [item("i1", "m1", "message", "Hey, can you tell me about V?", role="user"),
         item("i2", "m2", "reasoning", "thinking about V"),
         item("i3", "m2", "message", "V is a compiled language."),
         item("i4", "m4", "message", "and how fast is it?", role="user")],
        [char("c1", "minimax/minimax-m3:free")])
    after = parse_one(drops)
    assert after.native_id == before.native_id
    assert after.raw_hash != before.raw_hash


# -- titles ----------------------------------------------------------------

def test_truncated_stub_is_replaced_by_the_full_prompt(tmp_path):
    session = parse_one(simple(tmp_path, title="Hey, can you tell me"))
    assert session.title == "Hey, can you tell me about V?"
    assert session.title_source == "first_prompt"


def test_a_hand_renamed_chat_keeps_its_name(tmp_path):
    session = parse_one(simple(tmp_path, title="V language research"))
    assert session.title == "V language research"
    assert session.title_source == "provider"


def test_a_long_prompt_title_is_cut_on_a_word_boundary(tmp_path):
    prompt = ("Hey, could you please explain in considerable detail how the V "
              "programming language handles memory management")
    drops = export(
        tmp_path,
        [umsg("m1", "i1"), amsg("m2", "c1", "m1", [ref("i2", 0)], model="x")],
        [item("i1", "m1", "message", prompt, role="user"),
         item("i2", "m2", "message", "sure")],
        [char("c1", "x")], title=prompt[:40])
    title = parse_one(drops).title
    assert title.endswith("…")
    assert len(title) <= 81
    assert not title[:-1].endswith(" ")
    assert prompt.startswith(title[:-1])


# -- robustness ------------------------------------------------------------

def test_unknown_item_types_are_counted_never_fatal(tmp_path):
    drops = export(
        tmp_path,
        [umsg("m1", "i1"),
         amsg("m2", "c1", "m1", [ref("i2", 0), ref("i3", 1)], model="x")],
        [item("i1", "m1", "message", "hi", role="user"),
         {"id": "i2", "messageId": "m2",
          "data": {"type": "image_generation_call", "content": []}},
         item("i3", "m2", "message", "here you go")],
        [char("c1", "x")])
    stats = ParseStats()
    session = parse_one(drops, stats)
    assert [p.kind for p in session.messages[1].parts] == [KIND_TEXT]
    assert stats.unknown_types == {"openrouter:item:image_generation_call": 1}


def test_artifacts_are_flagged_rather_than_silently_dropped(tmp_path):
    drops = simple(tmp_path)
    path = next(drops.glob("*.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["artifacts"] = {"art-1": {"id": "art-1"}}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    stats = ParseStats()
    session = parse_one(drops, stats)
    assert session.meta["artifacts"] == 1
    assert stats.unknown_types == {"openrouter:artifacts-not-parsed": 1}


def test_an_export_with_no_messages_is_counted_not_raised(tmp_path):
    drops = export(tmp_path, [], [])
    adapter = OpenRouterAdapter(drops=drops)
    stats = ParseStats()
    assert list(adapter.parse(adapter.discover()[0], stats)) == []
    assert stats.unknown_types == {"openrouter:export-without-messages": 1}
