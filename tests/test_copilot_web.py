"""Regression tests for the GitHub Copilot (github.com) share-capture adapter.

The things this format gets wrong in ways that fail silently each get a test here:
the three duplicated fields (`content`, `skillExecutions`, `references`) that would
double-count if read alongside the ones actually parsed, the `"root"` string sentinel
that would shatter the tree if treated as falsy, and a HAR whose timing fields change on
every capture and would defeat idempotence if hashed.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_archive.adapters.copilot_web import CopilotWebAdapter
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import (
    KIND_TEXT, KIND_TOOL_RESULT, KIND_TOOL_USE, ParseStats,
)

THREAD = "2bb64435-8210-4588-854e-0b75188d3819"
SHARED = "805e1228-4b24-8046-a043-b646004e604d"
ENDPOINT = (f"https://api.individual.githubcopilot.com"
            f"/github/chat/shared/{SHARED}/messages")

ASKED = "2026-08-26T20:57:17.234245630Z"          # RFC3339Nano: nine digits
ANSWERED = "2026-08-26T20:57:17.238656762Z"


def thread(**over) -> dict:
    base = {
        "id": THREAD, "name": "Maze solver agent setup",
        "manuallyNamed": False, "repoID": 0, "repoOwnerID": 0,
        "createdAt": "2026-08-26T20:55:46.800881435Z",
        "updatedAt": "2026-08-26T20:57:36.194295828Z",
        "sharedID": SHARED, "sharedAt": "2026-08-26T20:57:36.194295828Z",
        "associatedRepoIDs": [572177136], "autoPickedModel": "gpt-5-mini",
    }
    return {**base, **over}


def ask(text="Give me comments about my maze_agent project.", *,
        mid="f148f73d", parent="root", created=ASKED) -> dict:
    return {"id": mid, "parentMessageID": parent, "intent": "conversation",
            "role": "user", "content": text, "createdAt": created,
            "threadID": THREAD, "references": [], "skillExecutions": [],
            "interrupted": False}


def getfile(path="README.md", repo="example-user/maze_agent") -> dict:
    """A completed skill execution. Note `content` is empty — see the adapter docstring."""
    return {
        "slug": "getfile", "status": "completed", "callId": "c1",
        "arguments": json.dumps({"path": path, "ref": None, "repo": repo}),
        "references": [{
            "type": "file", "ref": "", "repoID": 572177136,
            "repoName": repo.split("/")[-1], "repoOwner": "",
            "url": f"https://github.com/{repo}/blob/a0b778d/{path}",
            "path": path, "commitOID": "a0b778d", "languageName": "Markdown",
            "content": "", "addLineNums": True, "sha": "576cc68",
        }],
    }


def answer(text="## What this is\nA student-facing template.", *,
           mid="7bbe41ac", parent="f148f73d", created=ANSWERED,
           skills=None, model="gpt-5-mini", usage=(141844, 5414)) -> dict:
    skills = [getfile()] if skills is None else skills
    parts = [{"type": "toolCall", "skillExecution": s} for s in skills]
    parts.append({"type": "text", "content": text})
    flat = [r for s in skills for r in s["references"]]
    return {
        "id": mid, "parentMessageID": parent, "intent": "conversation",
        "role": "assistant", "model": model, "generatedWithAuto": True,
        # `content` duplicates the text parts and `references` duplicates the skills'
        # own lists, exactly as the real capture does.
        "content": text, "contentParts": parts,
        "skillExecutions": skills, "references": list(reversed(flat)),
        "createdAt": created, "threadID": THREAD, "interrupted": False,
        "usage": {"inputTokens": usage[0], "outputTokens": usage[1]},
    }


def write_response(tmp_path: Path, messages, th=None, name="copilot-share.json") -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    (drops / name).write_text(
        json.dumps({"thread": th or thread(), "messages": messages}),
        encoding="utf-8")
    return drops


def write_har(tmp_path: Path, messages, th=None, *, started="2026-08-27T01:31:03.000Z",
              name="github.com_Archive.har") -> Path:
    """A HAR with the transcript buried among unrelated traffic, as a real one is."""
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    har = {"log": {"version": "1.2", "startedDateTime": started, "entries": [
        {"request": {"url": "https://github.com/copilot/share/" + SHARED,
                     "headers": [{"name": "Cookie", "value": "user_session=SECRET"}]},
         "response": {"content": {"mimeType": "text/html", "text": "<html></html>"}}},
        {"request": {"url": "https://github.githubassets.com/chunk-67800.js"},
         "response": {"content": {"mimeType": "text/javascript", "text": "void 0;"}}},
        {"request": {"url": "https://avatars.githubusercontent.com/u/1?v=4"},
         "response": {"content": {"mimeType": "image/png", "encoding": "base64",
                                  "text": "iVBORw0KGgo="}}},
        {"request": {"url": ENDPOINT,
                     "headers": [{"name": "Authorization", "value": "Bearer SECRET"}]},
         "response": {"content": {"mimeType": "text/plain", "text": json.dumps(
             {"thread": th or thread(), "messages": messages})}}},
    ]}}
    (drops / name).write_text(json.dumps(har), encoding="utf-8")
    return drops


def parse_one(drops: Path, blobs=None):
    adapter = CopilotWebAdapter(drops=drops, blobs=blobs)
    stats = ParseStats()
    found = [s for p in adapter.discover() for s in adapter.parse(p, stats)]
    return found, stats


# -- the format's own traps -------------------------------------------------

def test_reads_transcript_out_of_a_har(tmp_path):
    """The HAR holds four responses; only one is the transcript."""
    sessions, stats = parse_one(write_har(tmp_path, [ask(), answer()]))
    assert len(sessions) == 1
    assert not stats.errors and not stats.unknown_types
    assert sessions[0].meta["endpoint"] == ENDPOINT


def test_reads_a_bare_response_body(tmp_path):
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    assert len(sessions) == 1
    assert sessions[0].native_id == THREAD


def test_duplicated_content_is_not_counted_twice(tmp_path):
    """`content` repeats the text parts; reading both would store the answer twice."""
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    reply = sessions[0].messages[1]
    texts = [p for p in reply.parts if p.kind == KIND_TEXT]
    assert len(texts) == 1
    assert texts[0].text == "## What this is\nA student-facing template."


def test_duplicated_skills_and_references_are_not_counted_twice(tmp_path):
    """One tool call in, one tool_use and one tool_result out — never two of each."""
    sessions, _ = parse_one(write_response(
        tmp_path, [ask(), answer(skills=[getfile(), getfile("myTeam.py")])]))
    reply = sessions[0].messages[1]
    assert len([p for p in reply.parts if p.kind == KIND_TOOL_USE]) == 2
    assert len([p for p in reply.parts if p.kind == KIND_TOOL_RESULT]) == 2


def test_tool_results_are_stored_but_never_embedded(tmp_path):
    """The §1.1 rule: tool payloads are searchable, not embedded."""
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    reply = sessions[0].messages[1]
    for part in reply.parts:
        assert part.embed_eligible is (part.kind in (KIND_TEXT, KIND_TOOL_USE))


def test_root_sentinel_keeps_the_first_turn_on_the_active_path(tmp_path):
    """`parentMessageID` is the string 'root', not null — treating it as a real id
    would leave the opening prompt parentless and drop it off the active path."""
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    messages = sessions[0].messages
    assert [m.on_active_path for m in messages] == [True, True]
    assert messages[0].parent_native_id is None


def test_abandoned_branch_is_kept_but_marked(tmp_path):
    """Retrying a prompt fans the tree out; the newest leaf wins and the loser stays."""
    retry = answer(mid="99999999", created="2026-08-26T20:58:00.000000000Z",
                   text="A second, better answer.")
    sessions, stats = parse_one(write_response(
        tmp_path, [ask(), answer(), retry]))
    by_id = {m.native_id: m for m in sessions[0].messages}
    assert by_id["99999999"].on_active_path
    assert not by_id["7bbe41ac"].on_active_path
    assert stats.orphaned_messages == 1


def test_nanosecond_timestamps_parse(tmp_path):
    """Go's RFC3339Nano has nine fractional digits; fromisoformat predates tolerating
    them on the versions this project supports."""
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    assert sessions[0].messages[0].created_at == 1787777837234
    assert sessions[0].started_at == 1787777746800


def test_tokens_come_from_assistant_turns_and_cost_stays_null(tmp_path):
    """Copilot bills premium requests, not tokens — cost is absent, not zero."""
    session, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    session = session[0]
    assert (session.tok_in, session.tok_out) == (141844, 5414)
    assert session.cost_usd is None


def test_repo_becomes_the_workspace(tmp_path):
    """The full name lives only in tool payloads; the key is casefolded like the rest."""
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    session = sessions[0]
    assert session.workspace_key == "github.com/example-user/maze_agent"
    assert session.workspace_label == "maze_agent"
    assert session.meta["repo"] == "example-user/maze_agent"


def test_most_cited_repo_wins(tmp_path):
    """A passing mention of someone else's repo must not relabel the conversation."""
    skills = [getfile(repo="example-user/maze_agent"),
              getfile(repo="example-user/maze_agent"),
              getfile(repo="someone/other")]
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer(skills=skills)]))
    assert sessions[0].workspace_label == "maze_agent"


def test_identity_is_the_thread_not_the_share(tmp_path):
    """Re-sharing mints a new sharedID; keying on it would file the chat twice."""
    first, _ = parse_one(write_response(tmp_path, [ask(), answer()]))
    reshared = thread(sharedID="ffffffff-0000-0000-0000-000000000000")
    second, _ = parse_one(write_response(tmp_path, [ask(), answer()], th=reshared,
                                         name="copilot-share-2.json"))
    assert first[0].native_id == second[0].native_id


def test_hash_ignores_the_har_wrapper(tmp_path):
    """Two captures of one conversation differ in timing but must hash the same, or
    every re-drop reports the session as updated."""
    messages = [ask(), answer()]
    a, _ = parse_one(write_har(tmp_path / "a", messages, started="2026-08-27T01:00:00Z"))
    b, _ = parse_one(write_har(tmp_path / "b", messages, started="2026-08-27T09:45:12Z"))
    assert a[0].raw_hash == b[0].raw_hash


def test_har_credentials_never_reach_the_session(tmp_path):
    """Request headers carry a live cookie and bearer token. Nothing reads them."""
    sessions, _ = parse_one(write_har(tmp_path, [ask(), answer()]))
    assert "SECRET" not in json.dumps(
        {"meta": sessions[0].meta,
         "parts": [p.text for m in sessions[0].messages for p in m.parts]},
        default=str)


# -- discovery --------------------------------------------------------------

def test_other_sources_json_is_not_discovered(tmp_path):
    """The drops folder already holds T3 and OpenRouter exports; opening a 14 MB
    unrelated dump on every ingest is exactly what the sniff exists to prevent."""
    drops = write_response(tmp_path, [ask(), answer()])
    (drops / "threads-export-2026.json").write_text(
        json.dumps({"version": "11.0.1", "threads": [], "messages": []}),
        encoding="utf-8")
    (drops / "OpenRouter Chat.json").write_text(
        json.dumps({"version": "orpg.1", "messages": {}}), encoding="utf-8")
    found = CopilotWebAdapter(drops=drops).discover()
    assert [p.name for p in found] == ["copilot-share.json"]


def test_har_without_a_transcript_is_counted_not_fatal(tmp_path):
    """Capturing the page but missing the fetch is the likely user error (risk R5)."""
    drops = tmp_path / "drops"
    drops.mkdir(parents=True)
    (drops / "other.har").write_text(json.dumps({"log": {"entries": [
        {"request": {"url": "https://example.com/"},
         "response": {"content": {"mimeType": "text/html", "text": "<html></html>"}}}]}}),
        encoding="utf-8")
    sessions, stats = parse_one(drops)
    assert sessions == []
    assert not stats.errors
    assert stats.unknown_types == {"copilot_web:har-without-transcript": 1}


def test_unknown_content_part_is_counted_not_fatal(tmp_path):
    """A part type GitHub adds later must show up in `llma doctor`, not crash ingest."""
    reply = answer()
    reply["contentParts"].insert(0, {"type": "somethingNew", "payload": {}})
    sessions, stats = parse_one(write_response(tmp_path, [ask(), reply]))
    assert len(sessions) == 1
    assert stats.unknown_types == {"copilot_web:contentPart:somethingNew": 1}


def test_large_tool_result_goes_to_a_blob(tmp_path):
    big = getfile()
    big["references"][0]["content"] = "x" * 40_000
    blobs = BlobStore(tmp_path / "blobs")
    sessions, _ = parse_one(write_response(tmp_path, [ask(), answer(skills=[big])]),
                            blobs=blobs)
    result = next(p for p in sessions[0].messages[1].parts
                  if p.kind == KIND_TOOL_RESULT)
    assert result.blob_sha and result.bytes > 40_000
    assert result.text.endswith("<truncated, full text in blob>")
