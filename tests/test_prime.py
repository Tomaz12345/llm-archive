"""The primer: a session cut down to what a fresh session needs from it.

The rules are extractive, so each test checks that a span the next session needs is
still there verbatim, and that a span it does not need -- a tool call that worked, a
thinking block, a lead-in line, a leaked key -- is not.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from llm_archive import api
from llm_archive.cli import app as cli_app
from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.export import prime
from llm_archive.mcp import server as mcp
from llm_archive.search import facts

T0 = 1771200000000
WS = "c:/proj/app"


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "archive.db")


@pytest.fixture
def src(con):
    return db.source_id(con, "claude_code", "Claude Code", "cli")


def add(con, src, native, steps, *, title=None, parent=None, started=T0):
    """Steps are (role, kind, text[, extra]); extra is tool_ok for results, a tool_input
    dict for calls, or a tool_name. One message per step, the Claude Code shape."""
    msgs = []
    for i, step in enumerate(steps):
        role, kind, text = step[:3]
        extra = step[3] if len(step) > 3 else None
        m = Message(native_id=f"{native}-{i}", role=role, seq=i,
                    created_at=started + i * 1000)
        part = Part(kind=kind, seq=0, text=text)
        if kind in ("tool_use", "tool_result"):
            part.tool_name = "Bash"
        if kind == "tool_use" and isinstance(extra, dict):
            part.tool_name = extra.pop("tool", "Bash")
            part.tool_input = json.dumps(extra)
        if kind == "tool_result":
            part.tool_ok = extra if isinstance(extra, bool) else True
        m.parts.append(part)
        msgs.append(m)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id=native, title=title or native,
        workspace_key=WS, workspace_label="app", host="box", started_at=started,
        ended_at=started + len(steps) * 1000, raw_path=f"/raw/{native}",
        raw_hash=native, parent_native_id=parent, model_primary="claude-opus-5",
        messages=msgs))
    con.commit()
    return con.execute("SELECT id FROM session WHERE native_id = ?",
                       (native,)).fetchone()["id"]


# ------------------------------------------------------------------ text rules

def test_a_trimmed_reply_keeps_the_opening_the_argument_and_short_code():
    text = "\n\n".join([
        "Two things changed the plan.",
        "Step one: read the file.",
        "I chose SQLite instead of a JSON file because the joins are free.",
        "```\nx = 1\n```",
        "```\n" + "\n".join(f"line {i}" for i in range(30)) + "\n```",
        "## Where it stops",
        "Then I ran the tests.",
    ])
    kept, cut = prime.trim_reply(text, limit=400)
    assert kept.startswith("Two things changed the plan.")
    assert "because the joins are free" in kept
    assert "x = 1" in kept                     # a short fence, after kept prose
    assert "line 29" not in kept               # a long fence goes
    assert "## Where it stops" in kept         # headings give shape
    assert "read the file" not in kept and "ran the tests" not in kept
    assert cut == len(text) - len(kept) > 0


def test_a_short_reply_is_not_trimmed_at_all():
    assert prime.trim_reply("Fixed, tests pass.") == ("Fixed, tests pass.", 0)


def test_a_lead_in_is_a_short_line_ending_in_a_colon():
    assert prime.is_lead_in("Now the tests. Writing via heredoc:")
    assert not prime.is_lead_in("Now the tests.")
    assert not prime.is_lead_in("Two paragraphs:\n\nof detail:")
    assert not prime.is_lead_in("x" * 300 + ":")


@pytest.mark.parametrize("command, expected", [
    ('cd x && git add -A && git commit -q -m "Fix the thing" && git log -1', ["Fix the thing"]),
    ("git commit -m 'single quoted'", ["single quoted"]),
    ("git -c core.x=1 commit -m 'with options'", ["with options"]),
    ("git commit -F - <<'EOF'\nAdd llma blame\n\nlong body\nEOF", ["Add llma blame"]),
    ('git commit -m "$(cat <<\'EOF\'\nSubject line\n\nBody\nEOF\n)"', ["Subject line"]),
    # a script that mentions a commit is not a commit
    ("python - <<'PYEOF'\nprint('git commit -m x')\nPYEOF", []),
    ("python -c \"for t in ['git add && git commit -m x', 'git commit -F - <<EOF']:\n"
     "    print(t)\"", []),
    ("git log --grep commit", []),
])
def test_commit_messages_are_read_off_the_commit_stage(command, expected):
    assert prime.commits_in(command) == expected


def test_a_test_run_is_recognised_by_program_not_by_mention():
    assert prime._is_test_run("pytest", None, "pytest -q")
    assert prime._is_test_run("python", "pytest", "python -m pytest tests/")
    assert prime._is_test_run("npm", "test", "npm test")
    assert not prime._is_test_run("cat", None, "cat > test_x.py <<'EOF'\nimport pytest\nEOF")
    assert not prime._is_test_run("uv", "run", "uv run main.py")
    assert prime._is_test_run("uv", "run", "uv run pytest -x")


# ---------------------------------------------------------------- the build

@pytest.fixture
def worked(con, src):
    """A session that did a thing: a goal, tool traffic, one failure, a decision."""
    sid = add(con, src, "w", [
        ("user", "text", "Make the batch size configurable. It is hardcoded to 32."),
        ("assistant", "thinking", "let me think about where it lives"),
        ("assistant", "text", "Reading the config module:"),
        ("assistant", "tool_use", "file_path: C:/proj/app/src/config.py",
         {"tool": "Read", "file_path": "C:/proj/app/src/config.py"}),
        ("user", "tool_result", "BATCH = 32", True),
        ("assistant", "tool_use", "command: pytest -q",
         {"command": "cd C:/proj/app && pytest -q"}),
        ("user", "tool_result", "Exit code 1\nlots\nof\noutput\nE   AssertionError: 32 != 64", False),
        ("assistant", "text", "The default lives in config.py. I'll read it from the "
                              "environment instead of the constant, because the trainer "
                              "already reads its other knobs that way.\n\n"
                              + "Long unrelated paragraph. " * 40),
        ("assistant", "tool_use", "file_path: C:/proj/app/src/config.py",
         {"tool": "Edit", "file_path": "C:/proj/app/src/config.py",
          "old_string": "32", "new_string": "int(os.environ.get('BATCH', 32))"}),
        ("user", "tool_result", "ok", True),
        ("assistant", "tool_use", "command: git commit",
         {"command": 'cd C:/proj/app && git add -A && git commit -m "Make batch size an env var"'}),
        ("user", "tool_result", "[main abc123] Make batch size an env var", True),
        ("user", "text", "And a CLI flag too?"),
        ("assistant", "text", "Done: `--batch` on the trainer, falling back to the env var. "
                              "Tests pass. The key sk-ant-api03-" + "x" * 40 + " in the "
                              "fixture should be rotated."),
    ], title="Batch size")
    facts.rebuild(con)
    return sid


def test_the_primer_keeps_the_goal_the_decision_the_failure_and_the_end(con, worked):
    p = prime.build(con, worked)
    md = p.markdown
    assert md.startswith("# Primer: Batch size")
    assert "## Goal" in md and "> Make the batch size configurable." in md
    # the person's later turn, whole
    assert "**you:** And a CLI flag too?" in md
    # the decision paragraph survives; under a budget, the padding does not
    assert "because the trainer already reads its other knobs that way" in md
    tight = prime.build(con, worked, chars=2000).markdown
    assert "because the trainer already reads its other knobs that way" in tight
    assert "Long unrelated paragraph." not in tight and "chars trimmed" in tight
    # the failed call, with its intent and the tail of the error
    assert "✗ **Bash** `command: pytest -q`" in md
    assert "AssertionError: 32 != 64" in md
    assert "lots\n" not in md
    # successful tool traffic and thinking are gone; lead-ins too
    assert "BATCH = 32" not in md and "let me think" not in md
    assert "Reading the config module:" not in md
    # the outcome, from the derived tables
    assert "**Files changed:** `src/config.py`" in md
    assert "- Make batch size an env var" in md
    assert "**Last test run:** `cd C:/proj/app && pytest -q`" in md
    assert "### Last reply" in md and "Done: `--batch` on the trainer" in md
    assert p.stats["tool_calls"] == 4 and p.stats["failures"] == 1
    assert p.stats["thinking_dropped"] == 1 and p.stats["lead_ins_dropped"] == 1


def test_secrets_are_redacted_on_the_way_out(con, worked):
    md = prime.build(con, worked).markdown
    assert "sk-ant-api03-" not in md
    assert "[redacted:anthropic_key:" in md
    assert "sk-ant-api03-" in prime.build(con, worked, redact=False).markdown


def test_the_budget_caps_the_document_and_reports_the_cut(con, worked):
    p = prime.build(con, worked, chars=1200)
    assert len(p.markdown) <= 1200 + 400          # whole sections, not a hard cut
    assert p.stats["turns_omitted"] + p.stats["assistant_trimmed"] > 0
    assert "## Goal" in p.markdown and "### Last reply" in p.markdown
    assert p.stats["chars"] < prime.build(con, worked).stats["chars"]


def test_an_unfinished_session_says_so_and_why(con, src):
    sid = add(con, src, "u", [("user", "text", "start"),
                              ("assistant", "text", "Two options. Which do you want?")])
    md = prime.build(con, sid).markdown
    assert "**Unfinished** (asked-you)" in md
    assert "Which do you want?" in md


def test_lineage_points_at_the_other_half(con, src):
    parent = add(con, src, "p", [("user", "text", "start"), ("assistant", "text", "ok")],
                 title="First half")
    child = add(con, src, "c", [("user", "text", "start"), ("assistant", "text", "ok"),
                                ("user", "text", "go on"), ("assistant", "text", "done")],
                title="Second half")
    con.execute("UPDATE session SET continues_session_id = ?, continues_overlap = 2 "
                "WHERE id = ?", (parent, child))
    con.commit()
    assert f"Continues #{parent} “First half”" in prime.build(con, child).markdown
    assert f"Continued by #{child} “Second half”" in prime.build(con, parent).markdown


def test_subagent_messages_are_left_out_and_counted(con, src):
    sid = add(con, src, "s", [("user", "text", "explore"), ("assistant", "text", "found it")])
    con.execute("""INSERT INTO message(session_id, native_id, seq, role, created_at,
                                       is_sidechain) VALUES (?, 'side', 5, 'user', ?, 1)""",
                (sid, T0 + 5000))
    mid = con.execute("SELECT id FROM message WHERE native_id = 'side'").fetchone()["id"]
    con.execute("INSERT INTO part(message_id, seq, kind, text) VALUES (?, 0, 'text', "
                "'subagent instruction')", (mid,))
    con.commit()
    md = prime.build(con, sid).markdown
    assert "subagent instruction" not in md
    assert "1 subagent messages left out" in md


def test_a_web_chat_without_tools_has_no_files_line(con):
    web = db.source_id(con, "claude_web", "Claude.ai", "web")
    sid = add(con, web, "chat", [("user", "text", "explain X"),
                                 ("assistant", "text", "X is Y.")])
    md = prime.build(con, sid).markdown
    assert "Files changed" not in md
    assert "### Last reply\n\nX is Y." in md


def test_missing_sessions_are_none_everywhere(con):
    assert prime.build(con, 999) is None
    assert api.prime_payload(con, 999) is None


# ------------------------------------------------------------------ surfaces

def test_the_cli_prints_writes_and_refuses_a_missing_id(tmp_path, con, worked):
    con.close()
    data = str(tmp_path)
    r = CliRunner().invoke(cli_app, ["prime", str(worked), "--data-dir", data])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("# Primer: Batch size")

    out = tmp_path / "primer.md"
    r = CliRunner().invoke(cli_app, ["prime", str(worked), "-o", str(out), "--data-dir", data])
    assert r.exit_code == 0 and out.read_text(encoding="utf-8").startswith("# Primer")

    r = CliRunner().invoke(cli_app, ["prime", str(worked), "--json", "--data-dir", data])
    payload = json.loads(r.output)
    assert payload["stats"]["failures"] == 1 and payload["markdown"].startswith("# Primer")

    r = CliRunner().invoke(cli_app, ["prime", "999", "--data-dir", data])
    assert r.exit_code == 1 and "no session #999" in r.output


def test_the_mcp_tool_returns_the_primer(tmp_path, con, worked):
    con.close()
    archive = mcp.Archive(tmp_path)
    listed = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "prime" in {t["name"] for t in listed["result"]["tools"]}
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                   "params": {"name": "prime",
                                              "arguments": {"session_id": worked,
                                                            "max_chars": 3000}}})
    payload = json.loads(frame["result"]["content"][0]["text"])
    assert payload["markdown"].startswith("# Primer: Batch size")
    assert payload["session"]["session_id"] == worked
    frame = mcp.dispatch(archive, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "prime",
                                              "arguments": {"session_id": 999}}})
    assert frame["result"]["isError"]


def test_the_web_serves_it_as_a_download(tmp_path, con, worked):
    from llm_archive.web.app import create_app

    con.close()
    client = TestClient(create_app(tmp_path, fetch_images=False))
    r = client.get(f"/session/{worked}/prime.md")
    assert r.status_code == 200
    assert r.text.startswith("# Primer: Batch size")
    assert "primer.md" in r.headers["content-disposition"]
    assert client.get("/session/999/prime.md").status_code == 404
    # the session page links to it
    assert f"/session/{worked}/prime.md" in client.get(f"/session/{worked}").text
