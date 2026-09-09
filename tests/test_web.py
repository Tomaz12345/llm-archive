"""Tests for the local web UI.

The escaping tests matter most: search results and transcripts are rendered with
`|safe` so query terms can be wrapped in <mark>. Archive text is arbitrary — it contains
HTML, script tags and shell fragments from real web pages that tools fetched. If
`_highlight` ever stops escaping before marking, every session becomes stored XSS
against the person reading their own archive.
"""

from __future__ import annotations

import sqlite3

import json
import pytest
from starlette.testclient import TestClient

from llm_archive.core import db, reopen
from llm_archive.core.models import Message, Part, Session
from llm_archive.search import facts, fts
from llm_archive.web.app import _highlight, create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    con = db.connect(data / "archive.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    panel = db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel")
    t3 = db.source_id(con, "t3chat", "T3 Chat", "web")
    orouter = db.source_id(con, "openrouter", "OpenRouter", "web")

    def add(native, title, texts, workspace="proj", host="box",
            source=None, kind="claude_code", meta=None):
        msgs = []
        for i, (role, text, kind) in enumerate(texts):
            m = Message(native_id=f"{native}-{i}", role=role,
                        created_at=1771200000000 + i * 1000, seq=i)
            m.parts.append(Part(kind=kind, seq=0, text=text,
                                embed_eligible=kind in ("text", "thinking")))
            msgs.append(m)
        db.upsert_session(con, source if source is not None else src, Session(
            source_kind=kind, native_id=native, title=title,
            workspace_key=workspace, workspace_label=workspace, host=host,
            started_at=1771200000000, raw_path=f"/raw/{native}.jsonl",
            raw_hash=native, messages=msgs, meta=meta or {}))

    add("s1", "Offside detection work", [
        ("user", "how do I detect an offside line", "text"),
        ("assistant", "use the second-rearmost defender", "text"),
        ("assistant", "ran the pipeline", "tool_result")])
    add("s2", "Dangerous content", [
        ("user", "<script>alert('xss')</script> & <img src=x onerror=alert(1)>", "text"),
        ("assistant", "here is <b>markup</b> from a fetched page", "tool_result")])
    add("s3", "Copilot refactor", [
        ("user", "rename the argparse flag", "text"),
        ("assistant", "renamed it", "text")],
        source=panel, kind="vscode_chat", meta={"participant": "copilot",
                                                "participant_label": "GitHub Copilot"})
    # #4-#6 exist for the reopen links: a web chat that has a URL, one whose export
    # carries no conversation id, and a CLI session whose recorded cwd is real.
    add("db0ee0a2-d373-452b-9617-f6bd406e10e0", "Archived thread", [
        ("user", "a question I asked T3", "text"),
        ("assistant", "an answer", "text")],
        workspace=None, host=None, source=t3, kind="t3chat",
        meta={"visibility": "archived"})
    add("msg-1787652204-7U7e1MmNVDmK", "Exported from OpenRouter", [
        ("user", "compare these models", "text"),
        ("assistant", "here is the comparison", "text")],
        workspace=None, host=None, source=orouter, kind="openrouter")
    add("resumable", "Local agent run", [
        ("user", "fix the failing test", "text"),
        ("assistant", "fixed", "text")],
        meta={"cwds": {str(tmp_path): 4}})
    # Two workspace rows under ONE label, on two machines -- the merge the workspace
    # page has to show rather than hide. A path containing markup is here on purpose:
    # a path is arbitrary archive text on its way into HTML.
    def add_files(native, title, calls, ws_key, ws_label, host, source=None,
                  kind="claude_code"):
        msgs = []
        for i, (tool, path) in enumerate(calls):
            m = Message(native_id=f"{native}-{i}", role="assistant", seq=i,
                        created_at=1771200000000 + i * 1000)
            part = Part(kind="tool_use", seq=0, text=f"file_path: {path}",
                        tool_name=tool)
            part.tool_input = json.dumps({"file_path": path})
            m.parts.append(part)
            msgs.append(m)
        db.upsert_session(con, source if source is not None else src, Session(
            source_kind=kind, native_id=native, title=title,
            workspace_key=ws_key, workspace_label=ws_label, host=host,
            started_at=1771200000000, raw_path=f"/raw/{native}.jsonl",
            raw_hash=native, messages=msgs))

    add_files("w1", "Trained on the box", [
        ("Edit", "/home/t/vision/code/train.py"),
        ("Read", "/home/t/vision/code/train.py"),
    ], "ssh-remote+jon/home/t/vision", "vision", "jon")
    add_files("w2", "Trained on Windows", [
        ("Edit", r"C:/proj/vision/code/train.py"),
        ("Write", r"C:/proj/vision/<script>alert(1)</script>.py"),
    ], "c:/proj/vision", "vision", "box")

    cmd_msg = Message(native_id="cmd-0", role="assistant", seq=0,
                      created_at=1771200000000)
    cmd_part = Part(kind="tool_use", seq=0, text="command: pytest -q",
                    tool_name="Bash")
    cmd_part.tool_input = json.dumps({"command": "pytest -q tests/"})
    cmd_msg.parts.append(cmd_part)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="w3", title="Ran the tests",
        workspace_key="c:/proj/vision", workspace_label="vision", host="box",
        started_at=1771200000000, raw_path="/raw/w3.jsonl", raw_hash="w3",
        messages=[cmd_msg]))

    con.commit()
    fts.rebuild(con)
    facts.rebuild(con)
    con.close()

    return TestClient(create_app(data))


# ------------------------------------------------------------------- routes

def test_import_page_renders(client):
    r = client.get("/import")
    assert r.status_code == 200
    assert "Drop export files here" in r.text
    # It doubles as the "what should I go and download?" page, so the freshness table
    # and its export-button instructions have to be on it.
    assert "Still to fetch" in r.text
    assert "Data controls" in r.text


def test_uploading_an_export_identifies_and_ingests_it(client, tmp_path):
    import io
    import json
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps([{
            "uuid": "web-1", "name": "Uploaded chat",
            "created_at": "2026-08-01T10:00:00Z",
            "updated_at": "2026-08-01T11:00:00Z",
            "account": {"uuid": "a"},
            "chat_messages": [
                {"uuid": "m1", "parent_message_uuid": None, "sender": "human",
                 "created_at": "2026-08-01T10:00:00Z", "text": "hello",
                 "content": [{"type": "text", "text": "hello"}],
                 "attachments": [], "files": []}]}]))

    r = client.post("/import", files={
        "upload": ("conversations-000.zip", buf.getvalue(), "application/zip")})

    assert r.status_code == 200
    assert "claude_web" in r.text
    assert "Uploaded chat" in client.get("/browse?source=claude_web").text


def test_uploading_a_non_export_is_reported_not_ingested(client):
    r = client.post("/import", files={
        "upload": ("holiday.mp4", b"\x00\x00\x00 ftypisom", "video/mp4")})

    assert r.status_code == 200
    assert "skipped" in r.text
    assert "not an export" in r.text


@pytest.mark.parametrize("path", ["/", "/browse", "/?q=offside", "/?q=offside&mode=keyword"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "Session" in r.text


def test_search_finds_session(client):
    r = client.get("/?q=offside&mode=keyword")
    assert "Offside detection work" in r.text


def test_query_terms_are_highlighted(client):
    r = client.get("/?q=offside&mode=keyword")
    assert "<mark>" in r.text


def test_unknown_session_is_404(client):
    assert client.get("/session/999999").status_code == 404


def test_session_reader_shows_transcript(client):
    r = client.get("/session/1")
    assert r.status_code == 200
    assert "second-rearmost defender" in r.text
    assert "/raw/s1.jsonl" in r.text          # pointer back to the source file


def test_tools_can_be_hidden(client):
    with_tools = client.get("/session/1?tools=true").text
    without = client.get("/session/1?tools=false").text
    assert "ran the pipeline" in with_tools
    assert "ran the pipeline" not in without


# ------------------------------------------------------------------ escaping

def test_highlight_escapes_before_marking():
    out = _highlight("<script>alert('x')</script>", ["script"])
    assert "<script>" not in out
    assert "&lt;" in out
    assert "<mark>" in out          # still highlighted, just not executable


def test_highlight_without_terms_still_escapes():
    assert "<img" not in _highlight("<img src=x onerror=alert(1)>", [])


def test_script_tags_from_archive_never_reach_the_page(client):
    """Real tool output contains fetched HTML. It must render as text."""
    for path in ["/session/2", "/?q=script&mode=keyword"]:
        text = client.get(path).text
        # what matters is that no *tag* survives — the escaped characters appearing as
        # visible text (&lt;img src=x onerror=alert(1)&gt;) is the correct outcome
        assert "<script>alert" not in text
        assert "<img src=x" not in text
        assert "&lt;img src=x onerror=alert(1)&gt;" in text


def test_regex_metacharacters_in_query_do_not_crash(client):
    for query in ["c++", "a(b", "*", "[x]", "\\", "a|b"]:
        assert client.get("/", params={"q": query, "mode": "keyword"}).status_code == 200


# ------------------------------------------------------------------- filters

def test_workspace_filter_narrows_results(client):
    assert client.get("/browse?workspace=proj").status_code == 200
    empty = client.get("/browse?workspace=nope")
    assert "Offside detection work" not in empty.text


def test_source_filter_excludes_other_sources(client):
    r = client.get("/?q=offside&mode=keyword&source=codex")
    assert "Offside detection work" not in r.text


# ------------------------------------------------------------------- tagging

def test_tag_roundtrip(client):
    client.post("/session/1/tag", data={"name": "important"}, follow_redirects=False)
    assert "important" in client.get("/session/1").text

    client.post("/session/1/untag", data={"name": "important"},
                follow_redirects=False)
    assert "important" not in client.get("/session/1").text


def test_blank_tag_is_ignored(client):
    r = client.post("/session/1/tag", data={"name": "   "}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/session/1").status_code == 200


def test_search_survives_a_broken_vector_store(client):
    """No vectors built yet: hybrid must fall back to keyword, not 500."""
    r = client.get("/?q=offside&mode=hybrid")
    assert r.status_code == 200
    assert "Offside detection work" in r.text


def test_assistant_facet_offers_copilot(client):
    """The source chip says "VS Code chat"; the facet says who answered in it."""
    r = client.get("/browse")
    assert "GitHub Copilot" in r.text
    assert 'name="participant"' in r.text


def test_browse_filters_by_participant(client):
    kept = client.get("/browse?participant=copilot")
    assert "Copilot refactor" in kept.text
    assert "Offside detection work" not in kept.text

    empty = client.get("/browse?participant=nobody")
    assert "Copilot refactor" not in empty.text


def test_search_filters_by_participant(client):
    r = client.get("/?q=rename&mode=keyword&participant=copilot")
    assert "Copilot refactor" in r.text
    r = client.get("/?q=offside&mode=keyword&participant=copilot")
    assert "Offside detection work" not in r.text


def test_stats_labels_model_less_rows_without_pricing_them(client):
    """A row labelled by its participant must never read as a priced model."""
    r = client.get("/stats")
    assert r.status_code == 200
    assert "no model recorded" in r.text
    # The disclaimer is the whole point: the label is not a model.
    assert "<b>not models</b>" in r.text
    assert "an app name is not a rate" in r.text


def test_stats_renders_the_surface_chart(client):
    """The surface split was computed for weeks with nothing rendering it."""
    r = client.get("/stats")
    assert r.status_code == 200
    assert "Where you work" in r.text
    # both surfaces present in the fixture must appear in the legend
    assert "Terminal" in r.text
    assert "Editor panel" in r.text


# ------------------------------------------------------------- index state

def _log_index(tmp_path, *, offset_ms, with_vectors=1):
    """Write an index_run row dated relative to now, then let the page read it."""
    import time

    con = sqlite3.connect(tmp_path / "data" / "archive.db")
    when = int(time.time() * 1000) + offset_ms
    con.execute("INSERT INTO index_run(started_at,finished_at,fts_rows,chunks,vectors,"
                "model_tag,with_vectors,seconds) VALUES (?,?,?,?,?,?,?,?)",
                (when - 1000, when, 12, 4, 4, "pm-MiniLM-L12", with_vectors, 1.0))
    # a fresh index means every embeddable part has a chunk; the fixture builds none
    con.execute("""INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,
                                     model_tag)
                   SELECT p.id, m.id, m.session_id, 0, p.text, p.id, 'pm-MiniLM-L12'
                   FROM part p JOIN message m ON m.id = p.message_id
                   WHERE p.embed_eligible = 1 AND p.text IS NOT NULL
                     AND LENGTH(p.text) >= 40 AND m.on_active_path = 1""")
    con.commit()
    con.close()


def test_an_index_with_no_recorded_build_raises_no_alarm(client):
    """The fixture has a real keyword index and no record of building it — the state
    every existing archive lands in on upgrade. It must not be cried wolf over."""
    r = client.get("/?q=offside&mode=keyword")
    assert "Search index is out of date" not in r.text
    assert "No search index yet" not in r.text
    assert "Offside detection work" in r.text, "results should still render"

    stats = client.get("/stats").text
    assert "Unknown" in stats, "/stats should still explain the unknown build date"


def test_search_page_warns_when_the_index_is_stale(client, tmp_path):
    """The exact failure mode: results render, and they are wrong."""
    _log_index(tmp_path, offset_ms=-60_000)   # built before the sessions were ingested
    r = client.get("/?q=offside&mode=keyword")
    assert "Search index is out of date" in r.text
    assert "session(s) ingested since the last build" in r.text


def test_stats_shows_the_index_panel(client, tmp_path):
    _log_index(tmp_path, offset_ms=-60_000)
    r = client.get("/stats")
    assert "Search index" in r.text
    assert "Out of date" in r.text


def test_a_fresh_index_is_not_announced_as_stale(client, tmp_path):
    """A rebuilt index must clear the warning, not leave it permanently on."""
    _log_index(tmp_path, offset_ms=60_000)
    assert "Search index is out of date" not in client.get("/?q=offside&mode=keyword").text
    assert "Up to date" in client.get("/stats").text


def test_keyword_only_build_reads_as_current_with_a_caveat(client, tmp_path):
    _log_index(tmp_path, offset_ms=60_000, with_vectors=0)
    stats = client.get("/stats").text
    assert "Up to date" in stats
    assert "--no-vectors" in stats
    assert "Search index is out of date" not in client.get("/?q=offside").text


# ------------------------------------------------------ reopening a session

def test_a_web_chat_links_back_to_the_provider(client):
    r = client.get("/session/4")
    assert "https://t3.chat/chat/db0ee0a2-d373-452b-9617-f6bd406e10e0" in r.text
    assert "open in T3 Chat" in r.text


def test_an_archived_thread_is_badged_but_still_linked(client):
    """Archiving hides a thread from the sidebar at the provider; it does not delete
    it. Dropping the link on those would drop it on most of the T3 Chat archive."""
    r = client.get("/session/4").text
    assert "archived there" in r
    assert "https://t3.chat/chat/" in r


def test_a_session_that_cannot_be_reopened_says_why(client):
    r = client.get("/session/5").text
    assert "can’t reopen" in r
    assert "no conversation id" in r


def test_a_local_agent_session_offers_a_terminal_not_a_link(client, monkeypatch):
    monkeypatch.setattr(reopen, "_local_host", lambda: "box")
    monkeypatch.setattr(reopen.shutil, "which", lambda name: f"/bin/{name}")
    r = client.get("/session/6").text
    assert "resume in Claude Code" in r
    assert "/session/6/open" in r
    assert "claude --resume resumable" in r


def test_a_blocked_session_still_offers_the_command_when_there_is_one(
        client, monkeypatch):
    """#6 is recorded on "box"; from anywhere else the button cannot run it, but the
    command is still the right one to paste over there."""
    monkeypatch.setattr(reopen, "_local_host", lambda: "somewhere-else")
    r = client.get("/session/6").text
    assert "can’t reopen" in r
    assert "copy command" in r


def test_a_blocked_session_with_no_command_offers_nothing_to_copy(client):
    """OpenRouter has no URL and no command — there is nothing to hand over."""
    r = client.get("/session/5").text
    assert "can’t reopen" in r
    assert "copy command" not in r


def test_browse_links_out_for_web_sessions_only(client):
    """A terminal launch from a result row is one mis-click from a window nobody
    asked for, so only URL targets get the jump-out arrow."""
    assert "row-open" in client.get("/browse?source=t3chat").text
    assert "row-open" not in client.get("/browse?source=claude_code").text


def test_search_results_link_out(client):
    assert "hit-open" in client.get("/?q=T3&mode=keyword").text


# ------------------------------------------------- POST /session/{id}/open

def _launched(monkeypatch):
    """Capture the launch instead of spawning a terminal in the test suite."""
    calls = []
    monkeypatch.setattr(reopen, "_local_host", lambda: "box")
    monkeypatch.setattr(reopen.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(reopen, "launch", calls.append)
    return calls


def test_open_launches_a_local_session(client, monkeypatch, tmp_path):
    calls = _launched(monkeypatch)
    r = client.post("/session/6/open")
    assert r.status_code == 200
    assert calls and calls[0].argv == ("claude", "--resume", "resumable")
    assert calls[0].cwd == str(tmp_path)


def test_open_refuses_a_cross_site_request(client, monkeypatch):
    """The only endpoint here that starts a process, and there is no CSRF token
    anywhere in this app — so the header the browser sets on its own carries it."""
    calls = _launched(monkeypatch)
    r = client.post("/session/6/open", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert not calls


def test_open_allows_the_uis_own_fetch(client, monkeypatch):
    calls = _launched(monkeypatch)
    assert client.post("/session/6/open",
                       headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert calls


def test_open_refuses_a_session_from_another_machine(client, monkeypatch):
    calls = _launched(monkeypatch)
    monkeypatch.setattr(reopen, "_local_host", lambda: "somewhere-else")
    r = client.post("/session/6/open")
    assert r.status_code == 409
    assert "somewhere-else" not in r.json()["error"]   # names the recording host, not this one
    assert "box" in r.json()["error"]
    assert not calls


def test_open_refuses_a_session_that_opens_with_a_link(client, monkeypatch):
    calls = _launched(monkeypatch)
    r = client.post("/session/4/open")
    assert r.status_code == 400
    assert r.json()["url"].startswith("https://t3.chat/")
    assert not calls


def test_open_on_an_unknown_session_is_404(client, monkeypatch):
    _launched(monkeypatch)
    assert client.post("/session/999999/open").status_code == 404


# ------------------------------------------------------- session grouping (v10)

def test_related_panel_survives_a_missing_vector_store(client):
    """The fixture builds no vectors. The panel must degrade, never 500 the transcript."""
    r = client.get("/session/1")
    assert r.status_code == 200
    assert "Offside detection work" in r.text


def test_topic_facet_is_hidden_until_groups_exist(client):
    """An archive that has never been indexed with vectors has no groups, and an empty
    dropdown labelled Topic is worse than no dropdown."""
    for path in ("/", "/browse"):
        assert 'name="topic"' not in client.get(path).text


def test_topic_facet_appears_and_filters(client, tmp_path):
    con = sqlite3.connect(tmp_path / "data" / "archive.db")
    con.execute("INSERT INTO topic(id,slug,label,size,built_at,model_tag) "
                "VALUES (1,'offside-vlan','offside · vlan',1,1,'t')")
    con.execute("INSERT INTO session_topic(session_id,topic_id) VALUES (1,1)")
    con.commit()
    con.close()

    listing = client.get("/browse")
    assert 'name="topic"' in listing.text
    assert "offside · vlan" in listing.text

    kept = client.get("/browse?topic=offside-vlan")
    assert "Offside detection work" in kept.text
    assert "Copilot refactor" not in kept.text


def test_lineage_banner_names_both_halves(client, tmp_path):
    con = sqlite3.connect(tmp_path / "data" / "archive.db")
    con.execute("UPDATE session SET continues_session_id=1, continues_overlap=2 "
                "WHERE id=2")
    con.commit()
    con.close()

    child = client.get("/session/2")
    assert "Continues" in child.text and 'href="/session/1"' in child.text
    parent = client.get("/session/1")
    assert "Continued in" in parent.text and 'href="/session/2"' in parent.text


def test_no_lineage_banner_when_a_session_stands_alone(client):
    assert "Continued in" not in client.get("/session/3").text


def test_stats_reports_what_lineage_removed(client, tmp_path):
    con = sqlite3.connect(tmp_path / "data" / "archive.db")
    con.execute("UPDATE session SET continues_session_id=1, continues_overlap=2 "
                "WHERE id=2")
    con.execute("UPDATE message SET superseded=1 WHERE session_id=1")
    con.commit()
    con.close()

    r = client.get("/stats")
    assert r.status_code == 200
    assert "Continued sessions" in r.text
    assert "continuation" in r.text


# --- derived tool facts on the web -------------------------------------------

def test_the_session_page_lists_the_files_it_touched(client):
    """The panel is the shortest path from a transcript to what it actually did."""
    sid = _session_id(client, "Trained on Windows")
    body = client.get(f"/session/{sid}").text
    assert "Files touched" in body
    assert "code/train.py" in body


def test_the_session_page_links_each_file_to_the_workspace(client):
    """One click from "this session edited train.py" to everything that touched it."""
    sid = _session_id(client, "Trained on Windows")
    body = client.get(f"/session/{sid}").text
    assert "/workspace/vision?file=" in body


def test_the_session_page_survives_an_unindexed_archive(client, tmp_path):
    """`_files_panel` returns [] rather than 500ing a page whose job is the transcript."""
    import sqlite3
    con = sqlite3.connect(tmp_path / "data" / "archive.db")
    con.execute("DROP TABLE touched_file")
    con.execute("DROP TABLE command")
    con.commit()
    con.close()
    sid = _session_id(client, "Trained on Windows")
    assert client.get(f"/session/{sid}").status_code == 200


def test_the_workspace_page_merges_two_roots_under_one_label(client):
    """`vision` is an SSH box and a Windows checkout. Both are named, and their
    `code/train.py` is ONE hot-file row -- that merge is why `rel` exists."""
    body = client.get("/workspace/vision").text
    assert "ssh-remote+jon/home/t/vision" in body
    assert "c:/proj/vision" in body
    # Once in the hot-file table -- the chart names it too, hence the anchor.
    assert body.count(">code/train.py</a>") == 1
    assert "2 machines" in body


def test_the_workspace_page_can_narrow_to_one_root(client):
    body = client.get("/workspace/vision?key=c:/proj/vision").text
    assert "Trained on Windows" in body
    assert "Trained on the box" not in body


def test_the_workspace_page_filters_sessions_by_file(client):
    body = client.get("/workspace/vision?file=code/train.py").text
    assert "Trained on Windows" in body and "Trained on the box" in body
    assert "Ran the tests" not in body


def test_the_workspace_page_shows_what_gets_run(client):
    assert "pytest" in client.get("/workspace/vision").text


def test_the_workspace_page_404s_for_an_unknown_label(client):
    assert client.get("/workspace/nosuchproject").status_code == 404


def test_a_path_containing_markup_is_escaped_on_the_workspace_page(client):
    """A path is arbitrary text from someone else's disk on its way into HTML."""
    body = client.get("/workspace/vision").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def _session_id(client, title: str) -> int:
    import re
    body = client.get("/browse").text
    match = re.search(r'href="/session/(\d+)">' + re.escape(title), body)
    assert match, f"{title!r} not on /browse"
    return int(match.group(1))
