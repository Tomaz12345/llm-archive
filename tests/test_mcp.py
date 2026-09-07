"""Tests for the JSON view (api.py), the stdio MCP server, and the CLI's --json.

The protocol tests carry the most weight. There is no SDK validating frames on the way
out, so the shape of what this server writes is only as correct as what is asserted
here: one line per frame, pure ASCII, a reply for every request and silence for every
notification. A frame that splits over two lines, or a notification that gets answered,
desynchronises a client in a way that looks like the archive is broken rather than the
transport.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_archive import api
from llm_archive.cli import app as cli_app
from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.mcp import server as mcp
from llm_archive.search import fts

LONG_REPLY = ("Both workers take the same two row locks in opposite order. " * 120)
LONG_TOOL_OUTPUT = ("pid 4412 waiting on transactionid 91823; blocked by pid 4409. " * 90)


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """An archive on disk at the location `--data-dir` and the MCP server expect."""
    con = db.connect(tmp_path / "archive.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    web = db.source_id(con, "chatgpt", "ChatGPT", "web")

    def add(native, title, parts, workspace="payments-api", source=None,
            kind="claude_code"):
        msgs = []
        for i, (role, kind_, text) in enumerate(parts):
            m = Message(native_id=f"{native}-{i}", role=role, seq=i,
                        created_at=1771200000000 + i * 1000)
            m.parts.append(Part(kind=kind_, seq=0, text=text,
                                embed_eligible=kind_ == "text",
                                tool_name="Bash" if kind_ == "tool_use" else None))
            msgs.append(m)
        db.upsert_session(con, source if source is not None else src, Session(
            source_kind=kind, native_id=native, title=title,
            workspace_key=workspace, workspace_label=workspace, host="dell",
            started_at=1771200000000, raw_path=f"/raw/{native}", raw_hash=native,
            messages=msgs))

    add("s1", "Postgres advisory locks in the batch job", [
        ("user", "text", "we keep deadlocking when two workers run the migration"),
        ("assistant", "text", LONG_REPLY),
        ("assistant", "tool_use", "psql -c 'select * from pg_locks'"),
        ("user", "tool_result", LONG_TOOL_OUTPUT),
    ])
    add("s2", "Migration deadlock again", [
        ("user", "text", "the batch job deadlocked on the migration lock a second time"),
        ("assistant", "text", "take the advisory lock before the row locks, one order"),
    ])
    add("s3", "Offside detection work", [
        ("user", "text", "how do I detect an offside line from the tracking data"),
        ("assistant", "text", "compute the second-rearmost defender position"),
    ], workspace="football")
    add("s4", "Slovenske opombe", [
        ("user", "text", "prosim pripravi predloge za intervju o čebelarstvu"),
        ("assistant", "text", "pripravil sem nekaj predlogov za tvoj intervju"),
    ], workspace="notes", source=web, kind="chatgpt")

    con.commit()
    fts.rebuild(con)
    con.close()
    return tmp_path


@pytest.fixture
def con(data_dir):
    return db.connect(data_dir / "archive.db")


@pytest.fixture
def archive(data_dir):
    return mcp.Archive(data_dir)


def call(archive, name, **arguments):
    """Drive one tools/call through the dispatcher and return (payload, is_error)."""
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
    result = frame["result"]
    body = result["content"][0]["text"]
    if result.get("isError"):
        return body, True
    return json.loads(body), False


# ------------------------------------------------------------------- api ----

def test_text_declares_the_length_it_is_hiding():
    """A caller must be able to tell a short answer from a long one that was cut."""
    short = api._text("brief", 100)
    assert short == {"text": "brief", "chars": 5, "truncated": False}

    cut = api._text("x" * 500, 100)
    assert cut["text"] == "x" * 100
    assert cut["chars"] == 500 and cut["truncated"] is True


def test_parse_day_is_utc_midnight():
    assert api.iso(api.parse_day("2026-04-12")) == "2026-04-12T00:00:00Z"
    assert api.parse_day(None) is None


def test_parse_day_rejects_a_non_date():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        api.parse_day("last tuesday")


def test_search_payload_carries_hits_and_bounded_snippets(con, data_dir):
    payload = api.search_payload(con, data_dir / "vectors", "deadlock migration",
                                 mode="keyword", snippet_chars=40)
    assert payload["query"] == "deadlock migration"
    assert payload["count"] == len(payload["results"]) >= 1
    hit = payload["results"][0]
    assert hit["started_at"] == "2026-02-16T00:00:00Z"
    assert hit["matched_by"] == "keyword"
    assert all(len(s["text"]) <= 40 for s in hit["snippets"])
    assert any(s["truncated"] for s in hit["snippets"])


def test_session_payload_leaves_tool_traffic_out_by_default(con):
    plain = api.session_payload(con, 1)
    assert {p["kind"] for m in plain["messages"] for p in m["parts"]} == {"text"}

    with_tools = api.session_payload(con, 1, tools=True)
    kinds = {p["kind"] for m in with_tools["messages"] for p in m["parts"]}
    assert kinds == {"text", "tool_use", "tool_result"}
    tool_use = next(p for m in with_tools["messages"] for p in m["parts"]
                    if p["kind"] == "tool_use")
    assert tool_use["tool_name"] == "Bash"


def test_session_payload_reports_what_the_budget_cut(con):
    """Silent truncation would read as 'the session says nothing more'."""
    whole = api.session_payload(con, 1, tools=True)
    assert whole["truncated"] is False and whole["omitted_parts"] == 0

    clipped = api.session_payload(con, 1, tools=True, part_chars=200, budget=100)
    assert clipped["truncated"] is True
    assert clipped["omitted_parts"] == 3
    assert clipped["parts"] == whole["parts"] - 3


def test_the_budget_overshoots_by_at_most_one_part(con):
    """Whole parts only — a budget of 1 still returns the first one rather than none."""
    payload = api.session_payload(con, 1, tools=True, part_chars=200, budget=1)
    assert payload["parts"] == 1
    assert sum(len(p["text"]) for m in payload["messages"] for p in m["parts"]) <= 200


def test_the_budget_charges_for_structure_not_just_prose(con):
    """A session of many short parts is mostly envelope; text-only accounting hid that.

    Four one-character parts hold four characters of prose and roughly 1600 of JSON. A
    budget of 1000 has to stop partway through, not wave all four past.
    """
    con.execute("UPDATE part SET text = 'x' WHERE message_id IN "
                "(SELECT id FROM message WHERE session_id = 1)")
    con.commit()
    payload = api.session_payload(con, 1, tools=True, budget=1000)
    assert payload["truncated"] is True
    assert payload["parts"] == 1000 // api.PART_OVERHEAD + 1 == 3


def test_the_default_budget_fits_a_client_inline_result(con):
    """The default has to come back inline; spilling to a file costs a whole round trip."""
    rendered = json.dumps(api.session_payload(con, 1, tools=True),
                          ensure_ascii=False, indent=2)
    assert len(rendered) <= api.SESSION_CHARS * 1.3


def test_session_payload_marks_a_part_it_shortened(con):
    payload = api.session_payload(con, 1, part_chars=100)
    reply = next(p for m in payload["messages"] for p in m["parts"]
                 if p["chars"] > 100)
    assert reply["truncated"] is True and len(reply["text"]) == 100


def test_session_payload_is_none_for_an_unknown_id(con):
    assert api.session_payload(con, 999) is None


def test_related_payload_names_the_session_it_was_asked_about(con, data_dir):
    payload = api.related_payload(con, data_dir / "vectors", 1)
    assert payload["session"]["session_id"] == 1
    assert payload["session"]["source"] == "claude_code"
    assert [r["session_id"] for r in payload["results"]] == [2]


# -------------------------------------------------------------- protocol ----

def test_initialize_echoes_a_version_it_supports(archive):
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {"protocolVersion": "2024-11-05"}})
    result = frame["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["serverInfo"]["name"] == "llm-archive"
    assert result["instructions"]


def test_initialize_falls_back_for_a_version_it_has_never_heard_of(archive):
    """A client newer than this file gets the newest revision we know, not a refusal."""
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                   "params": {"protocolVersion": "2099-01-01"}})
    assert frame["result"]["protocolVersion"] == mcp.LATEST_PROTOCOL


def test_tools_list_is_the_three_read_only_reads(archive):
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = frame["result"]["tools"]
    assert [t["name"] for t in tools] == ["search", "show", "related"]
    assert all(t["annotations"]["readOnlyHint"] for t in tools)


@pytest.mark.parametrize("tool", mcp.TOOLS, ids=lambda t: t["name"])
def test_every_tool_schema_is_well_formed(tool):
    schema = tool["inputSchema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(schema["properties"])
    assert tool["description"] and tool["title"]
    for prop in schema["properties"].values():
        assert "type" in prop


def test_notifications_are_not_answered(archive):
    assert mcp.dispatch(archive, {"jsonrpc": "2.0",
                                  "method": "notifications/initialized"}) is None
    assert mcp.dispatch(archive, {"jsonrpc": "2.0", "method": "notifications/cancelled",
                                  "params": {"requestId": 1}}) is None
    # an unknown *notification* is ignored; only a request earns an error frame
    assert mcp.dispatch(archive, {"jsonrpc": "2.0", "method": "who/knows"}) is None


def test_a_response_frame_is_ignored(archive):
    """This server sends no requests, so anything without a method is not ours."""
    assert mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 4, "result": {}}) is None


def test_unknown_method_with_an_id_is_an_error_frame(archive):
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 3,
                                   "method": "resources/list"})
    assert frame["error"]["code"] == mcp.METHOD_NOT_FOUND


def test_ping_is_answered(archive):
    assert mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 5,
                                  "method": "ping"})["result"] == {}


def test_an_unknown_tool_is_a_protocol_error(archive):
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                   "params": {"name": "delete_everything"}})
    assert frame["error"]["code"] == mcp.INVALID_PARAMS


# ----------------------------------------------------------------- tools ----

def test_search_tool_returns_ranked_sessions(archive):
    payload, failed = call(archive, "search", query="deadlock migration",
                           mode="keyword", limit=3)
    assert not failed
    assert payload["results"][0]["title"] in {
        "Postgres advisory locks in the batch job", "Migration deadlock again"}


def test_search_tool_applies_filters(archive):
    payload, _ = call(archive, "search", query="offside deadlock", mode="keyword",
                      workspace="football")
    assert [r["title"] for r in payload["results"]] == ["Offside detection work"]

    payload, _ = call(archive, "search", query="deadlock", mode="keyword",
                      source=["chatgpt"])
    assert payload["results"] == []


def test_search_tool_accepts_a_bare_source_string(archive):
    """The schema says array, but a model that sends one name should still be served."""
    payload, failed = call(archive, "search", query="predloge intervju", mode="keyword",
                           source="chatgpt")
    assert not failed
    assert [r["title"] for r in payload["results"]] == ["Slovenske opombe"]


def test_search_tool_limit_is_clamped_not_obeyed(archive):
    payload, failed = call(archive, "search", query="deadlock", mode="keyword",
                           limit=10_000)
    assert not failed and len(payload["results"]) <= 50


def test_show_tool_can_be_asked_for_tool_traffic(archive):
    plain, _ = call(archive, "show", session_id=1)
    assert {p["kind"] for m in plain["messages"] for p in m["parts"]} == {"text"}

    full, _ = call(archive, "show", session_id=1, include_tools=True)
    assert "tool_result" in {p["kind"] for m in full["messages"] for p in m["parts"]}


def test_show_tool_honours_max_chars(archive):
    payload, _ = call(archive, "show", session_id=1, include_tools=True,
                      max_chars=1000)
    assert payload["truncated"] is True and payload["omitted_parts"] > 0


def test_related_tool_never_returns_the_session_asked_about(archive):
    payload, failed = call(archive, "related", session_id=1)
    assert not failed
    assert payload["results"] and all(r["session_id"] != 1 for r in payload["results"])


def test_a_missing_session_is_a_readable_result_not_an_error_frame(archive):
    """The model has to be able to read the failure to recover by searching again."""
    body, failed = call(archive, "show", session_id=999)
    assert failed and "999" in body

    body, failed = call(archive, "related", session_id=999)
    assert failed and "999" in body


def test_a_bad_argument_comes_back_as_a_readable_tool_error(archive):
    body, failed = call(archive, "search", query="x", since="last tuesday")
    assert failed and "YYYY-MM-DD" in body

    body, failed = call(archive, "search", query="   ")
    assert failed and "query is required" in body

    body, failed = call(archive, "search", query="x", mode="magic")
    assert failed and "hybrid" in body

    body, failed = call(archive, "show", session_id="not a number")
    assert failed and "integer" in body


# ------------------------------------------------------------ stdio loop ----

def drive(data_dir, *frames) -> list[dict]:
    out = io.StringIO()
    mcp.serve(data_dir=data_dir,
              stdin=io.StringIO("".join(json.dumps(f) + "\n" for f in frames)),
              stdout=out)
    lines = [ln for ln in out.getvalue().split("\n") if ln]
    assert all(ln.isascii() for ln in lines), "a frame reached the wire as non-ASCII"
    return [json.loads(ln) for ln in lines]


def test_the_loop_answers_requests_and_stays_silent_on_notifications(data_dir):
    replies = drive(
        data_dir,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "search",
                    "arguments": {"query": "deadlock", "mode": "keyword"}}},
    )
    assert [r["id"] for r in replies] == [1, 2, 3]


def test_slovene_survives_the_ascii_wire(data_dir):
    """Escaped in the frame, intact once the client parses it — half the archive is."""
    replies = drive(data_dir,
                    {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "search",
                                "arguments": {"query": "predloge za intervju",
                                              "mode": "keyword"}}})
    payload = json.loads(replies[0]["result"]["content"][0]["text"])
    assert payload["results"][0]["title"] == "Slovenske opombe"
    assert "čebelarstvu" in payload["results"][0]["snippets"][0]["text"]


def test_a_malformed_line_does_not_end_the_session(data_dir):
    out = io.StringIO()
    mcp.serve(data_dir=data_dir, stdout=out, stdin=io.StringIO(
        "\n"
        "{not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}) + "\n"))
    frames = [json.loads(ln) for ln in out.getvalue().split("\n") if ln]
    assert frames[0]["error"]["code"] == mcp.PARSE_ERROR
    assert frames[1]["id"] == 9


def test_a_json_scalar_is_rejected_as_a_request(data_dir):
    frames = drive(data_dir, 42)
    assert frames[0]["error"]["code"] == mcp.INVALID_REQUEST


def test_a_batch_is_answered_as_a_batch(data_dir):
    """Allowed before protocol 2025-06-18; answering one cannot break a client."""
    out = io.StringIO()
    mcp.serve(data_dir=data_dir, stdout=out, stdin=io.StringIO(json.dumps([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]) + "\n"))
    batch = json.loads(out.getvalue())
    assert [f["id"] for f in batch] == [1, 2]


def test_the_database_is_not_opened_until_a_tool_is_called(tmp_path):
    """A client spawns this at session start; an unused server should cost nothing."""
    empty = tmp_path / "untouched"
    out = io.StringIO()
    mcp.serve(data_dir=empty, stdout=out, stdin=io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {}}) + "\n"))
    assert not empty.exists(), "initialize created an archive"


# ------------------------------------------------------------------- cli ----

def run_cli(data_dir, *args):
    return CliRunner().invoke(cli_app, [*args, "--data-dir", str(data_dir)])


def test_search_json_is_machine_readable(data_dir):
    result = run_cli(data_dir, "search", "deadlock migration", "--mode", "keyword",
                     "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["count"] >= 1
    assert payload["results"][0]["session_id"]


def test_search_json_with_no_matches_is_still_json(data_dir):
    """`no matches` on stdout would break every consumer of --json."""
    result = run_cli(data_dir, "search", "zzzznothinghere", "--mode", "keyword",
                     "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"query": "zzzznothinghere", "mode": "keyword",
                                         "count": 0, "results": []}


def test_show_json_is_machine_readable(data_dir):
    result = run_cli(data_dir, "show", "1", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["session"]["title"] == "Postgres advisory locks in the batch job"
    assert payload["messages"][0]["role"] == "user"


def test_show_json_reports_a_missing_session_as_json(data_dir):
    result = run_cli(data_dir, "show", "999", "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "no session #999"}


def test_related_json_is_machine_readable(data_dir):
    result = run_cli(data_dir, "related", "1", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["session"]["session_id"] == 1
    assert all(r["session_id"] != 1 for r in payload["results"])


def test_related_prints_a_list_for_a_person(data_dir):
    result = run_cli(data_dir, "related", "1")
    assert result.exit_code == 0
    assert "like #1" in result.stdout
    assert "Migration deadlock again" in result.stdout


def test_related_on_a_missing_session_exits_nonzero(data_dir):
    result = run_cli(data_dir, "related", "999")
    assert result.exit_code == 1
    assert "no session #999" in result.stdout
