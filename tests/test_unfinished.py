"""The inbox: which sessions ended waiting on you, and which only look like they did.

Most of these are named after the false positive they prevent. The two rules that were
measured before they were written -- `asked-you` is agent-only, `cut-off` needs the
session to file tool results as messages of their own -- each get the case that would
have flooded the list without them.
"""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from llm_archive import api
from llm_archive.cli import app as cli_app
from llm_archive.core import db, unfinished
from llm_archive.core.models import Message, Part, Session

T0 = 1771200000000          # 2026-02-16, comfortably older than any min_age


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "archive.db")


@pytest.fixture
def sources(con):
    return {
        "claude_code": db.source_id(con, "claude_code", "Claude Code", "cli"),
        "vscode_chat": db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel"),
        "claude_web": db.source_id(con, "claude_web", "Claude.ai", "web"),
    }


def add(con, sources, native, steps, *, kind="claude_code", title=None, started=T0,
        workspace="proj", parent=None, tok_out=None):
    """A session from (role, kind, text) steps, one part per message unless a step's
    text is a list -- then one message with a part per piece, the streamed shape."""
    msgs = []
    for i, (role, part_kind, text) in enumerate(steps):
        m = Message(native_id=f"{native}-{i}", role=role, seq=i,
                    created_at=started + i * 1000)
        pieces = text if isinstance(text, list) else [text]
        for j, piece in enumerate(pieces):
            m.parts.append(Part(kind=part_kind, seq=j, text=piece,
                                tool_name="Bash" if part_kind.startswith("tool") else None))
        msgs.append(m)
    db.upsert_session(con, sources[kind], Session(
        source_kind=kind, native_id=native, title=title or native,
        workspace_key=workspace, workspace_label=workspace, host="box",
        started_at=started, raw_path=f"/raw/{native}", raw_hash=native,
        parent_native_id=parent, messages=msgs, tok_out=tok_out))
    con.commit()
    return con.execute("SELECT id FROM session WHERE native_id = ?",
                       (native,)).fetchone()["id"]


def reasons(con, **kw):
    return {it.session_id: it for it in unfinished.find(con, **kw)}


# ------------------------------------------------------------------ the rules

def test_a_session_that_ends_on_your_turn_is_unanswered(con, sources):
    sid = add(con, sources, "u", [("user", "text", "how do I deploy this?"),
                                  ("assistant", "text", "like so"),
                                  ("user", "text", "and to a second region?")],
              kind="claude_web")
    got = reasons(con)
    assert got[sid].reason == "unanswered"
    assert got[sid].excerpt == "and to a second region?"


def test_a_session_that_ends_on_a_plain_reply_is_finished(con, sources):
    add(con, sources, "f", [("user", "text", "fix it"),
                            ("assistant", "text", "Fixed, tests pass.")])
    assert reasons(con) == {}


def test_a_trailing_tool_call_is_cut_off_when_results_come_as_messages(con, sources):
    """Claude Code: the result would have been its own message, and it never came."""
    sid = add(con, sources, "c", [
        ("user", "text", "run the tests"),
        ("assistant", "tool_use", "command: pytest -q"),
        ("user", "tool_result", "3 passed"),
        ("assistant", "text", "green, now the migration"),
        ("assistant", "tool_use", "command: alembic upgrade head")])
    got = reasons(con)
    assert got[sid].reason == "cut-off"
    assert got[sid].excerpt == "command: alembic upgrade head"


def test_a_trailing_tool_call_is_not_cut_off_where_results_are_never_stored(con, sources):
    """VS Code chat writes no tool_result at all, so a closing tool call is just how
    every one of its sessions ends. Flagging it would list all of them."""
    add(con, sources, "v", [
        ("user", "text", "add argparse"),
        ("assistant", "tool_use", 'Using "Apply Patch"'),
        ("assistant", "text", "done"),
        ("assistant", "tool_use", 'Using "Manage and track todo items"')],
        kind="vscode_chat")
    assert reasons(con) == {}


def test_an_interrupted_reply_is_a_stop_not_a_question(con, sources):
    sid = add(con, sources, "i", [
        ("user", "text", "refactor the module"),
        ("assistant", "text", "starting"),
        ("user", "text", "[Request interrupted by user for tool use]")])
    assert reasons(con)[sid].reason == "cut-off"


def test_an_agents_closing_question_is_asked_you(con, sources):
    sid = add(con, sources, "q", [
        ("user", "text", "look at the failing build"),
        ("assistant", "text", "Two blockers, both in CI.\n\nWant me to start on them?")])
    got = reasons(con)
    assert got[sid].reason == "asked-you"
    assert got[sid].excerpt == "Want me to start on them?"


def test_a_web_chats_closing_question_is_a_sign_off(con, sources):
    """50 of the 75 question-ending sessions in the real archive are web chats, and
    they are 'Would you like me to explain any part in more detail?' -- filler."""
    add(con, sources, "w", [
        ("user", "text", "explain transformers"),
        ("assistant", "text", "...\n\nWould you like me to explain any part in more detail?")],
        kind="claude_web")
    assert reasons(con) == {}


def test_anything_else_is_not_a_question_waiting_on_you(con, sources):
    add(con, sources, "a", [("user", "text", "rename the flag"),
                            ("assistant", "text", "Renamed and tested.\n\nAnything else?")])
    assert reasons(con) == {}


def test_a_question_asked_in_streamed_fragments_is_read_whole(con, sources):
    """VS Code stores a reply as shards -- ', or', 'to set', a lone '?' -- so the last
    part alone says nothing about how the reply ends."""
    sid = add(con, sources, "s", [
        ("user", "text", "make the batch size configurable"),
        ("assistant", "text", ["Done. Want me to ", "run the suite", " now", "?"])],
        kind="vscode_chat")
    got = reasons(con)
    assert got[sid].reason == "asked-you"
    assert got[sid].excerpt == "Want me to run the suite now?"     # the last sentence


def test_a_question_mark_inside_closing_markup_still_counts(con, sources):
    assert unfinished.asks_a_question("Shall I add the index? **")
    assert unfinished.asks_a_question("(Should I proceed?)")
    assert not unfinished.asks_a_question("Added the index.")
    assert not unfinished.asks_a_question("")


# --------------------------------------------------------------- exclusions

def test_a_session_you_resumed_is_no_longer_waiting(con, sources):
    """The parent of a --resume pair ended mid-thought by definition; the child is
    where the conversation went on."""
    parent = add(con, sources, "p", [("user", "text", "start"),
                                     ("assistant", "text", "Want me to go on?")])
    child = add(con, sources, "k", [("user", "text", "start"),
                                    ("assistant", "text", "Want me to go on?"),
                                    ("user", "text", "yes"),
                                    ("assistant", "text", "Done.")])
    con.execute("UPDATE session SET continues_session_id = ? WHERE id = ?",
                (parent, child))
    con.commit()
    assert reasons(con) == {}


def test_a_subagent_transcript_is_not_yours_to_answer(con, sources):
    add(con, sources, "main", [("user", "text", "go"),
                               ("assistant", "text", "Done.")])
    add(con, sources, "sub", [("user", "text", "explore the repo"),
                              ("assistant", "text", "Should I go deeper?")],
        parent="main")
    assert reasons(con) == {}


def test_a_sidechain_message_does_not_stand_in_for_the_last_turn(con, sources):
    sid = add(con, sources, "sc", [("user", "text", "go"),
                                   ("assistant", "text", "Which of the two?")])
    # a subagent's closing line, filed after the main conversation's last message
    con.execute("""INSERT INTO message(session_id, native_id, seq, role, created_at,
                                       is_sidechain, is_turn)
                   VALUES (?, 'side', 9, 'assistant', ?, 1, 1)""", (sid, T0 + 9000))
    mid = con.execute("SELECT id FROM message WHERE native_id = 'side'").fetchone()["id"]
    con.execute("INSERT INTO part(message_id, seq, kind, text) VALUES (?, 0, 'text', 'Done.')",
                (mid,))
    con.commit()
    assert reasons(con)[sid].reason == "asked-you"


def test_a_session_still_open_somewhere_is_not_an_inbox_item_yet(con, sources):
    just_now = int(time.time() * 1000) - 10 * 60 * 1000
    sid = add(con, sources, "live", [("user", "text", "one more thing")],
              started=just_now)
    assert reasons(con) == {}
    assert sid in reasons(con, min_age_ms=0)


def test_dismissing_hides_and_restoring_brings_back(con, sources):
    sid = add(con, sources, "d", [("user", "text", "no answer here")])
    assert unfinished.dismiss(con, [sid, 999]) == 1
    assert reasons(con) == {}
    assert reasons(con, include_dismissed=True)[sid].reason == "unanswered"
    # dismissing twice is not an error and not a second row
    assert unfinished.dismiss(con, [sid]) == 0
    assert unfinished.restore(con, [sid]) == 1
    assert sid in reasons(con)
    # the tag row goes when nothing carries it, as the web UI's untag does
    assert con.execute("SELECT COUNT(*) FROM tag").fetchone()[0] == 0


def test_filters_narrow_and_counts_cover_the_whole_set(con, sources):
    a = add(con, sources, "fa", [("user", "text", "x")], workspace="alpha")
    b = add(con, sources, "fb", [("user", "text", "y")], workspace="beta",
            kind="claude_web", started=T0 + 10 * 86_400_000)
    c = add(con, sources, "fc", [("user", "text", "z"),
                                 ("assistant", "text", "Want me to?")], workspace="alpha")
    assert set(reasons(con, workspace="alph")) == {a, c}
    assert set(reasons(con, sources=("claude_web",))) == {b}
    assert set(reasons(con, since=T0 + 86_400_000)) == {b}
    assert set(reasons(con, reasons=("asked-you",))) == {c}
    assert unfinished.counts(unfinished.find(con)) == {
        "unanswered": 2, "cut-off": 0, "asked-you": 1}
    with pytest.raises(ValueError):
        unfinished.find(con, reasons=("abandoned",))


def test_the_list_is_most_recently_stopped_first(con, sources):
    old = add(con, sources, "old", [("user", "text", "x")], started=T0)
    new = add(con, sources, "new", [("user", "text", "y")], started=T0 + 5000)
    assert [it.session_id for it in unfinished.find(con)] == [new, old]


# ----------------------------------------------------------------- surfaces

def test_the_payload_carries_the_brief_and_the_reason(con, sources):
    sid = add(con, sources, "pl", [("user", "text", "waiting")])
    payload = api.inbox_payload(con)
    assert payload["count"] == 1 and payload["shown"] == 1
    hit = payload["results"][0]
    assert hit["session_id"] == sid
    assert hit["reason"] == "unanswered"
    assert hit["excerpt"] == "waiting"
    assert hit["last_at"] == "2026-02-16T00:00:00Z"
    assert "open" in hit          # the reopen target, same as every other payload
    # a limit trims the list, not the count
    add(con, sources, "pl2", [("user", "text", "also waiting")])
    payload = api.inbox_payload(con, limit=1)
    assert (payload["count"], payload["shown"]) == (2, 1)
    assert payload["by_reason"]["unanswered"] == 2


def test_the_cli_lists_dismisses_and_restores(tmp_path, con, sources):
    sid = add(con, sources, "cli", [("user", "text", "please answer")], title="Lonely")
    con.close()
    data = str(tmp_path)

    r = CliRunner().invoke(cli_app, ["inbox", "--data-dir", data])
    assert r.exit_code == 0, r.output
    assert "1 unfinished session(s)" in r.output
    assert "Lonely" in r.output and "please answer" in r.output
    assert f"llma inbox dismiss {sid}" in r.output

    r = CliRunner().invoke(cli_app, ["inbox", "dismiss", str(sid), "--data-dir", data])
    assert r.exit_code == 0 and "dismissed 1" in r.output
    r = CliRunner().invoke(cli_app, ["inbox", "--data-dir", data])
    assert "nothing waiting on you" in r.output
    r = CliRunner().invoke(cli_app, ["inbox", "--all", "--json", "--data-dir", data])
    assert '"reason": "unanswered"' in r.output

    r = CliRunner().invoke(cli_app, ["inbox", "restore", str(sid), "--data-dir", data])
    assert "restored 1" in r.output
    r = CliRunner().invoke(cli_app, ["inbox", "--reason", "cut-off", "--data-dir", data])
    assert "nothing waiting on you" in r.output
    r = CliRunner().invoke(cli_app, ["inbox", "--reason", "abandoned", "--data-dir", data])
    assert r.exit_code != 0


def test_the_web_page_lists_and_a_done_click_hides(tmp_path, con, sources):
    from llm_archive.web.app import create_app

    sid = add(con, sources, "web", [("user", "text", "<b>unanswered</b> prompt")],
              title="Waiting <script>")
    con.close()
    client = TestClient(create_app(tmp_path, fetch_images=False))

    r = client.get("/inbox")
    assert r.status_code == 200
    assert "1 session waiting on you" in r.text
    assert "Waiting &lt;script&gt;" in r.text          # archive text is never HTML
    assert "&lt;b&gt;unanswered&lt;/b&gt; prompt" in r.text
    assert f'formaction="/inbox/{sid}/dismiss"' in r.text

    r = client.post(f"/inbox/{sid}/dismiss", follow_redirects=False,
                    headers={"referer": "http://testserver/inbox?reason=unanswered"})
    assert r.status_code == 303
    assert r.headers["location"] == "/inbox?reason=unanswered"
    assert "0 sessions waiting on you" in client.get("/inbox").text
    page = client.get("/inbox?dismissed=true").text
    assert "1 session waiting on you" in page and f'formaction="/inbox/{sid}/restore"' in page

    client.post(f"/inbox/{sid}/restore")
    assert "1 session waiting on you" in client.get("/inbox").text


# ------------------------------------------------------------- date parsing

def test_parse_when_takes_days_words_and_distances():
    now = 1_800_000_000_000
    day = 86_400_000
    assert api.parse_when("2026-09-01") == api.parse_day("2026-09-01")
    assert api.parse_when("7d", now_ms=now) == now - 7 * day
    assert api.parse_when("2w", now_ms=now) == now - 14 * day
    assert api.parse_when("6h", now_ms=now) == now - 6 * 3_600_000
    assert api.parse_when("last-week", now_ms=now) == now - 7 * day
    assert api.parse_when("last-month", now_ms=now) == now - 30 * day
    assert api.parse_when("today", now_ms=now) == now - now % day
    assert api.parse_when("yesterday", now_ms=now) == now - now % day - day
    assert api.parse_when(None) is None and api.parse_when("") is None
    with pytest.raises(ValueError):
        api.parse_when("lastweek")


def test_parse_duration():
    assert api.parse_duration("30m") == 30 * 60_000
    assert api.parse_duration("1h") == 3_600_000
    assert api.parse_duration("2d") == 2 * 86_400_000
    assert api.parse_duration("0") == 0
    assert api.parse_duration(None) is None
    with pytest.raises(ValueError):
        api.parse_duration("soon")
