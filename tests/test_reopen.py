"""Tests for `core.reopen` — where a session can be reopened, and where it cannot.

Two things are worth guarding here. The URL templates were derived by matching every
session's `native_id` against this machine's browser history, so a typo in one of them
produces a link that looks right and 404s; the table below pins the exact shape.

And the refusals matter as much as the links. A session recorded on another machine, a
subagent transcript, an OpenRouter export with no conversation id in it — each has to say
why rather than offer a guess, because a link that silently goes nowhere is worse than
being told the conversation cannot be reached from here.
"""

from __future__ import annotations

import json

import pytest

from llm_archive.core import reopen

UUID = "5be9ff86-8b9e-40c6-afd1-2fa3cf0e2e34"


def row(kind, native_id=UUID, meta=None, host=None, parent=None, workspace_key=None):
    """A session row as the queries in `web/app.py` and `api.py` shape one."""
    return {"source_kind": kind, "native_id": native_id,
            "meta": json.dumps(meta or {}), "host": host,
            "parent_session_id": parent, "workspace_key": workspace_key}


# ------------------------------------------------------------------ web links

@pytest.mark.parametrize("kind,expected", [
    ("t3chat", f"https://t3.chat/chat/{UUID}"),
    ("claude_web", f"https://claude.ai/chat/{UUID}"),
    ("chatgpt", f"https://chatgpt.com/c/{UUID}"),
    ("deepseek", f"https://chat.deepseek.com/a/chat/s/{UUID}"),
    ("mistral", f"https://chat.mistral.ai/chat/{UUID}"),
    ("copilot_web", f"https://github.com/copilot/c/{UUID}"),
    ("grok", f"https://grok.com/chat/{UUID}"),
    ("gemini", f"https://gemini.google.com/app/{UUID}"),
])
def test_web_sources_rebuild_their_chat_url(kind, expected):
    target = reopen.resolve(row(kind))
    assert target.mode == "url"
    assert target.url == expected
    assert target.blocked is None


def test_gemini_prefers_the_url_its_adapter_recorded():
    """Gemini is the one source that stored the real URL at ingest; an account prefix
    in it (`/u/0/app/…`) would be lost by rebuilding from the template."""
    stored = "https://gemini.google.com/app/6356e0503d6be5da"
    target = reopen.resolve(row("gemini", meta={"conversation_url": stored}))
    assert target.url == stored


def test_a_gemini_canvas_doc_has_no_conversation_to_open():
    target = reopen.resolve(row("gemini", meta={"unlinked": True}))
    assert target.blocked and "canvas" in target.blocked


def test_openrouter_admits_it_cannot_be_linked():
    """`native_id` there is a root *message* id — chat URLs are `?room=orc-…`, and the
    export carries nothing that maps between them."""
    target = reopen.resolve(row("openrouter", native_id="msg-1787652204-7U7e1MmNVDmK"))
    assert target.blocked and "conversation id" in target.blocked
    assert target.url is None


def test_an_id_that_is_not_url_shaped_is_refused_rather_than_interpolated():
    target = reopen.resolve(row("t3chat", native_id="../../evil?x=1"))
    assert target.blocked
    assert target.url is None


# --------------------------------------------------------- archived / temporary

def test_an_archived_t3chat_thread_still_links_but_says_so():
    """Archiving hides a thread from the T3 sidebar; it does not delete it. 132 of the
    170 t3chat sessions in the real archive are archived, so refusing the link there
    would remove it from most of the source."""
    target = reopen.resolve(row("t3chat", meta={"visibility": "archived"}))
    assert target.url and target.blocked is None
    assert target.warn == reopen.ARCHIVED_WARN


def test_an_archived_chatgpt_conversation_is_flagged_the_same_way():
    assert reopen.resolve(row("chatgpt", meta={"archived": True})).warn == \
        reopen.ARCHIVED_WARN


def test_a_visible_thread_carries_no_warning():
    assert reopen.resolve(row("t3chat", meta={"visibility": "visible"})).warn is None


def test_a_temporary_grok_chat_was_never_saved_server_side():
    target = reopen.resolve(row("grok", meta={"temporary": True}))
    assert target.blocked and "temporary" in target.blocked


# ----------------------------------------------------------------- VS Code

def test_a_local_workspace_becomes_a_vscode_file_uri():
    target = reopen.resolve(row("vscode_chat", workspace_key="c:/users/t/projekti/gapy"))
    assert target.url == "vscode://file/c:/users/t/projekti/gapy"
    # the editor cannot be pointed at one chat, and the label must not imply it can
    assert "no per-chat link" in target.warn


def test_a_remote_workspace_keeps_its_authority():
    target = reopen.resolve(row("vscode_chat",
                                workspace_key="ssh-remote+jon/home/tomaz/wall_e"))
    assert target.url == "vscode://vscode-remote/ssh-remote+jon/home/tomaz/wall_e"


def test_folder_uri_wins_over_the_casefolded_workspace_key():
    """`derive_workspace` casefolds, which is right for grouping and wrong for a
    case-sensitive remote path — so the adapter now records the folder verbatim."""
    target = reopen.resolve(row(
        "vscode_chat", meta={"folder_uri": "ssh-remote+jon/home/tomaz/Wall_E"},
        workspace_key="ssh-remote+jon/home/tomaz/wall_e"))
    assert target.url.endswith("/Wall_E")


def test_a_panel_with_no_folder_has_nothing_to_open():
    target = reopen.resolve(row("vscode_chat"))
    assert target.blocked and "folder" in target.blocked


# ------------------------------------------------------------- CLI agents

@pytest.fixture
def here(monkeypatch, tmp_path):
    """Pretend every tool is installed and this is the machine that recorded it."""
    monkeypatch.setattr(reopen, "_local_host", lambda: "thisbox")
    monkeypatch.setattr(reopen.shutil, "which", lambda name: f"/bin/{name}")
    return tmp_path


@pytest.mark.parametrize("kind,cwd_key,argv", [
    ("claude_code", "cwds", ("claude", "--resume", UUID)),
    ("codex", "cwds", ("codex", "resume", UUID)),
    ("opencode", "directory", ("opencode", "-s", UUID)),
])
def test_cli_agents_resume_by_id_in_their_own_directory(here, kind, cwd_key, argv):
    meta = ({"cwds": {str(here): 3}} if cwd_key == "cwds"
            else {"directory": str(here)})
    target = reopen.resolve(row(kind, meta=meta, host="thisbox"))
    assert target.mode == "launch"
    assert target.blocked is None
    assert target.argv == argv
    assert target.cwd == str(here)


def test_the_busiest_cwd_wins(here):
    """A session drifts between a project root and its subdirectories; the directory
    that recorded the most turns is the one it was mostly run from."""
    sub = here / "sub"
    sub.mkdir()
    target = reopen.resolve(row("claude_code", host="thisbox",
                                meta={"cwds": {str(here): 2, str(sub): 9}}))
    assert target.cwd == str(sub)


def test_a_session_from_another_machine_is_refused(here):
    target = reopen.resolve(row("claude_code", host="laptop",
                                meta={"cwds": {str(here): 1}}))
    assert target.blocked and "laptop" in target.blocked
    # Blocked here is not blocked everywhere: the command is exactly what you would run
    # once you are on that machine, so it stays available to copy.
    assert target.display == f"claude --resume {UUID}"
    assert target.argv is None


def test_a_remote_session_is_not_judged_by_this_disk(here):
    """The recorded cwd is a path on the other machine. Testing whether it exists here
    would report the wrong reason for the right refusal."""
    target = reopen.resolve(row("claude_code", host="laptop",
                                meta={"cwds": {"/home/t/somewhere-that-is-not-here": 1}}))
    assert "laptop" in target.blocked
    assert "no longer exists" not in target.blocked


def test_a_subagent_transcript_points_at_its_parent(here):
    target = reopen.resolve(row("claude_code", host="thisbox", parent=41,
                                meta={"cwds": {str(here): 1}}))
    assert target.blocked and "spawned" in target.blocked


def test_a_directory_that_is_gone_is_named_in_the_reason(here):
    missing = str(here / "deleted")
    target = reopen.resolve(row("claude_code", host="thisbox",
                                meta={"cwds": {missing: 1}}))
    assert target.blocked and missing in target.blocked


def test_a_session_with_no_recorded_cwd_is_refused(here):
    assert reopen.resolve(row("claude_code", host="thisbox")).blocked


def test_a_tool_that_is_not_installed_blocks_but_still_offers_the_command(
        here, monkeypatch):
    """Opening a terminal only to print "not recognized" is worse than saying so up
    front — but the command is still filled in, so copy-and-paste works on a box that
    does have it."""
    monkeypatch.setattr(reopen.shutil, "which", lambda name: None)
    target = reopen.resolve(row("codex", host="thisbox",
                                meta={"cwds": {str(here): 1}}))
    assert target.blocked and "PATH" in target.blocked
    assert target.display == f"codex resume {UUID}"
    assert str(here) in target.copy_text


def test_launch_refuses_a_blocked_target():
    with pytest.raises(reopen.LaunchError):
        reopen.launch(reopen.Target(mode="launch", label="x", blocked="nope"))


def test_launch_refuses_a_url_target():
    with pytest.raises(reopen.LaunchError):
        reopen.launch(reopen.Target(mode="url", label="x", url="https://example.com"))


# ------------------------------------------------------------------- payload

def test_an_unknown_source_says_so_rather_than_raising():
    assert reopen.resolve(row("some_new_thing")).blocked


def test_as_dict_omits_the_keys_that_do_not_apply():
    payload = reopen.resolve(row("t3chat")).as_dict()
    assert payload["mode"] == "url" and payload["url"].startswith("https://t3.chat/")
    assert "blocked" not in payload and "warn" not in payload and "cwd" not in payload
