"""The git blame bridge: a line -> its commit -> the sessions in that commit's window.

The tests that matter are the ones about the *window*. Which sessions a commit is
attributed to is entirely a question of where its window opens and closes, and the
tempting answer — the parent commit — is wrong in a way that quietly loses edits. So
the repo built here has exactly the history that would expose that, with every
timestamp pinned, and the archive's sessions are placed inside and outside each edge.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_archive import api
from llm_archive.cli import app as cli_app
from llm_archive.core import db, gitblame
from llm_archive.core.models import Message, Part, Session
from llm_archive.mcp import server as mcp
from llm_archive.search import facts

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")

# Epoch seconds, every event on its own minute so the story below reads in order.
T0 = 1_800_000_000
MINUTE = 60


def at(minutes: int) -> int:
    return T0 + minutes * MINUTE


def ms(minutes: int) -> int:
    return at(minutes) * 1000


# ----------------------------------------------------------------- fixtures

def git(repo: Path, *args: str, when: int | None = None) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo.parent)}
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"{when} +0000"
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), env=env, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str, when: int) -> str:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message, when=when)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory) -> Path:
    """A history built to catch the parent-commit mistake.

        t+10  A  foo.py created (lines 1-3)         <- previous commit touching foo.py
        t+40  B  bar.py only                        <- the parent of C, NOT foo.py's edge
        t+70  C  foo.py line 2 rewritten, line 4 added

    An edit to foo.py at t+30 sits after A and *before B*. It is in C's diff — nothing
    committed it in between — so C's window must open at A, not at its parent B.

    Built once: five git processes cost two seconds on Windows, and every test gets
    its own copy of the result instead, which costs nothing.
    """
    root = tmp_path_factory.mktemp("base") / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    commit(root, {"foo.py": "one\ntwo\nthree\n"}, "Create foo", at(10))
    commit(root, {"bar.py": "unrelated\n"}, "Add bar", at(40))
    commit(root, {"foo.py": "one\nTWO\nthree\nfour\n"}, "Rewrite line two", at(70))
    return root


@pytest.fixture
def repo(tmp_path, base_repo) -> Path:
    """A private copy, so a test may rename, edit or commit without touching another."""
    root = tmp_path / "repo"
    shutil.copytree(base_repo, root)
    return root


@pytest.fixture
def archive(tmp_path, repo) -> Path:
    """Sessions placed around the edges of C's window (A at t+10, C at t+70)."""
    con = db.connect(tmp_path / "archive.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    foo = str(repo / "foo.py")
    ws_key = repo.as_posix().casefold()

    def add(native, title, events):
        msgs = []
        for i, (tool, payload, minute, ok) in enumerate(events):
            m = Message(native_id=f"{native}-{i}", role="assistant", seq=i,
                        created_at=ms(minute))
            part = Part(kind="tool_use", seq=0, text=tool, tool_name=tool, tool_ok=ok)
            part.tool_input = json.dumps(payload)
            m.parts.append(part)
            msgs.append(m)
        db.upsert_session(con, src, Session(
            source_kind="claude_code", native_id=native, title=title,
            workspace_key=ws_key, workspace_label="repo", host="dell",
            started_at=ms(0), raw_path=f"/raw/{native}", raw_hash=native,
            messages=msgs, meta={"cwds": {str(repo): 1}}))

    add("before-a", "Edited before foo existed in git", [
        ("Edit", {"file_path": foo}, 5, True),
    ])
    add("in-window", "Edited foo between A and C", [
        ("Edit", {"file_path": foo}, 30, True),          # before parent B: still C's
        ("Edit", {"file_path": foo}, 60, True),
    ])
    add("failed-edit", "An Edit that errored", [
        ("Edit", {"file_path": foo}, 55, False),
    ])
    add("committer", "Ran the commit", [
        ("Bash", {"command": 'git add -A && git commit -m "Rewrite line two"'}, 71, True),
    ])
    add("other-commit", "Committed something else at the same time", [
        ("Bash", {"command": 'git commit -m "Bump version"'}, 70, True),
    ])
    add("after-c", "Edited foo after the last commit", [
        ("Edit", {"file_path": foo}, 90, True),
    ])
    con.commit()
    facts.rebuild(con)
    con.close()
    return tmp_path


@pytest.fixture
def con(archive):
    return db.connect(archive / "archive.db")


def by_summary(payload: dict, summary: str) -> dict:
    return next(c for c in payload["commits"] if c["summary"] == summary)


def titles(commit: dict) -> list[str]:
    return [s["title"] for s in commit["sessions"]]


# ------------------------------------------------------------- gitblame.py

def test_porcelain_is_parsed_into_lines_and_commits():
    """Attributes are printed once per commit; the second group from the same commit
    carries only a header, and must still resolve to the same Commit."""
    sha_a, sha_b = "a" * 40, "b" * 40
    text = "\n".join([
        f"{sha_a} 1 1 2", "author Ann", "author-time 100", "committer-time 200",
        "summary first", "filename foo.py", "\tone",
        f"{sha_a} 2 2", "\ttwo",
        f"{sha_b} 1 3 1", "author Bob", "author-time 300", "committer-time 300",
        "summary second", "previous " + sha_a + " foo.py", "filename foo.py", "\tthree",
        f"{sha_a} 3 4 1", "\tfour",
    ])
    bl = gitblame._parse_porcelain(text, Path("/r"), "foo.py", 1, 4)
    assert bl.lines == {1: sha_a, 2: sha_a, 3: sha_b, 4: sha_a}
    assert bl.commits[sha_a].summary == "first"
    assert bl.commits[sha_a].opened_at == 100 and bl.commits[sha_a].closed_at == 200
    assert bl.ranges(sha_a) == [(1, 2), (4, 4)]
    assert bl.in_order() == [sha_a, sha_b]


def test_parse_range_accepts_gits_syntax_and_a_bare_line():
    assert gitblame.parse_range("10,20") == (10, 20)
    assert gitblame.parse_range("10-20") == (10, 20)
    assert gitblame.parse_range("7") == (7, 7)
    assert gitblame.parse_range(None) == (None, None)
    with pytest.raises(ValueError):
        gitblame.parse_range("ten")


def test_a_line_suffix_is_split_off_but_a_drive_letter_is_not():
    assert gitblame.split_line_suffix("api.py:137") == ("api.py", 137, 137)
    assert gitblame.split_line_suffix("api.py:120-140") == ("api.py", 120, 140)
    assert gitblame.split_line_suffix(r"C:\x\api.py") == (r"C:\x\api.py", None, None)
    assert gitblame.split_line_suffix("C:/x/api.py:3") == ("C:/x/api.py", 3, 3)


def test_blame_resolves_the_previous_commit_that_touched_the_file(repo):
    bl = gitblame.blame(repo / "foo.py")
    c = next(x for x in bl.commits.values() if x.summary == "Rewrite line two")
    assert bl.previous[c.sha].summary == "Create foo"       # not "Add bar"
    a = next(x for x in bl.commits.values() if x.summary == "Create foo")
    assert bl.previous[a.sha] is None
    assert bl.ranges(c.sha) == [(2, 2), (4, 4)]
    assert bl.ranges(a.sha) == [(1, 1), (3, 3)]


def test_blame_outside_a_repository_is_a_git_error(tmp_path):
    loose = tmp_path / "loose.txt"
    loose.write_text("x\n")
    with pytest.raises(gitblame.GitError, match="not inside a git repository"):
        gitblame.blame(loose)


def test_blame_of_an_untracked_file_carries_gits_own_message(repo):
    (repo / "new.py").write_text("x\n")
    with pytest.raises(gitblame.GitError, match="no such path"):
        gitblame.blame(repo / "new.py")


def test_an_empty_range_is_refused_before_git_is_asked(repo):
    with pytest.raises(gitblame.GitError, match="empty"):
        gitblame.blame(repo / "foo.py", 4, 2)


# ------------------------------------------------------- api.blame_payload

def test_the_window_opens_at_the_previous_commit_of_the_file_not_the_parent(con, repo):
    """The edit at t+30 came before C's parent (B, t+40) and is in C's diff all the
    same. Opening the window at the parent would lose it."""
    payload = api.blame_payload(con, str(repo / "foo.py"), start=2, end=2)
    c = by_summary(payload, "Rewrite line two")
    assert c["previous"]["short"] and c["window"]["from"] == api.iso(ms(10))
    hit = next(s for s in c["sessions"] if s["title"] == "Edited foo between A and C")
    assert hit["edits"] == 2 and hit["evidence"] == ["edit"]
    assert hit["first_edit_at"] == api.iso(ms(30))


def test_an_edit_before_the_window_belongs_to_the_earlier_commit(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"))
    a = by_summary(payload, "Create foo")
    c = by_summary(payload, "Rewrite line two")
    assert "Edited before foo existed in git" in titles(a)
    assert "Edited before foo existed in git" not in titles(c)
    assert a["window"]["from"] is None          # the commit that created the file


def test_the_session_that_ran_the_commit_is_named_and_listed_first(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"), start=2, end=2)
    c = by_summary(payload, "Rewrite line two")
    assert titles(c)[0] == "Ran the commit"
    assert c["sessions"][0]["evidence"] == ["commit"]
    assert c["sessions"][0]["committed_at"] == api.iso(ms(71))


def test_a_commit_with_a_different_message_at_the_same_time_is_not_it(con, repo):
    """Same repo, same minute, `-m "Bump version"`: the subject settles it."""
    payload = api.blame_payload(con, str(repo / "foo.py"), start=2, end=2)
    c = by_summary(payload, "Rewrite line two")
    assert "Committed something else at the same time" not in titles(c)


def test_an_edit_that_errored_did_not_touch_the_file(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"))
    assert all("An Edit that errored" not in titles(c) for c in payload["commits"])


def test_uncommitted_lines_are_attributed_to_edits_since_the_newest_commit(con, repo):
    (repo / "foo.py").write_text("one\nTWO\nthree\nfour\nfive\n", encoding="utf-8")
    payload = api.blame_payload(con, str(repo / "foo.py"), start=5, end=5)
    (wt,) = payload["commits"]
    assert wt["uncommitted"] and wt["sha"] is None
    assert wt["previous"]["committed_at"] == api.iso(ms(70))
    assert wt["window"] == {"from": api.iso(ms(70)), "to": None}
    assert titles(wt) == ["Edited foo after the last commit"]


def test_an_edit_after_the_last_commit_is_not_that_commits(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"))
    c = by_summary(payload, "Rewrite line two")
    assert "Edited foo after the last commit" not in titles(c)


def test_a_range_reports_only_the_commits_inside_it(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"), start=1, end=1)
    assert [c["summary"] for c in payload["commits"]] == ["Create foo"]
    assert payload["range"] == {"start": 1, "end": 1} and payload["lines"] == 1


def test_commits_come_in_line_order_and_limit_cuts_the_tail(con, repo):
    payload = api.blame_payload(con, str(repo / "foo.py"), limit=1)
    assert payload["total"] == 2 and payload["count"] == 1
    assert payload["commits"][0]["summary"] == "Create foo"       # owns line 1


def test_a_relative_path_is_resolved_against_cwd(con, repo):
    payload = api.blame_payload(con, "foo.py", cwd=str(repo), start=2, end=2)
    assert payload["file"] == "foo.py"
    assert "Ran the commit" in titles(by_summary(payload, "Rewrite line two"))


def test_the_workspace_filter_applies(con, repo):
    from llm_archive.search.hybrid import Filters

    payload = api.blame_payload(con, str(repo / "foo.py"),
                                filters=Filters(workspace="nothing-called-this"))
    assert all(c["sessions"] == [] for c in payload["commits"])


def test_a_renamed_file_still_finds_the_edits_made_under_its_old_name(con, repo):
    """The commit knew the file as foo.py; the sessions that edited it recorded foo.py;
    the reader asks about the new name. `filename` from blame is the bridge."""
    git(repo, "mv", "foo.py", "renamed.py")
    git(repo, "commit", "-q", "-m", "Rename foo", when=at(100))
    payload = api.blame_payload(con, str(repo / "renamed.py"), start=2, end=2)
    c = by_summary(payload, "Rewrite line two")
    assert c["filename"] == "foo.py"
    assert c["window"]["from"] == api.iso(ms(10))
    assert "Edited foo between A and C" in titles(c)


def test_git_commit_is_recognised_at_the_head_of_a_segment_only():
    """A docstring that mentions `git commit` inside a heredoc is not a commit."""
    yes = ['git commit -m x', 'cd /p && git commit -m "y"', 'FOO=1 git commit',
           'git -c core.x=1 commit -F msg', 'x=$(git commit -m y)',
           "python - <<'EOF'\nprint(1)\nEOF\ngit commit -q -F- <<'EOF'\nmsg\nEOF"]
    no = ['git log --grep commit', 'echo git commit',
          'python - <<\'EOF\'\n"""run `git commit` here"""\nEOF']
    assert all(api._GIT_COMMIT.search(t) for t in yes)
    assert not any(api._GIT_COMMIT.search(t) for t in no)


# --------------------------------------------------------------- mcp / cli

def test_blame_is_listed_as_a_read_only_tool(archive):
    frame = mcp.dispatch(mcp.Archive(archive),
                         {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tool = next(t for t in frame["result"]["tools"] if t["name"] == "blame")
    assert tool["annotations"]["readOnlyHint"]
    assert tool["inputSchema"]["required"] == ["path"]


def call(archive, **arguments):
    frame = mcp.dispatch(mcp.Archive(archive), {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "blame", "arguments": arguments}})
    result = frame["result"]
    body = result["content"][0]["text"]
    return (body, True) if result.get("isError") else (json.loads(body), False)


def test_blame_over_mcp_takes_a_line_suffix_and_a_cwd(archive, repo):
    payload, err = call(archive, path="foo.py:2", cwd=str(repo))
    assert not err
    assert payload["range"] == {"start": 2, "end": 2}
    assert titles(payload["commits"][0])[0] == "Ran the commit"


def test_blame_over_mcp_reports_a_non_repository_as_a_tool_error(archive, tmp_path):
    loose = tmp_path / "loose.txt"
    loose.write_text("x\n")
    body, err = call(archive, path=str(loose))
    assert err and "not inside a git repository" in body


def test_blame_over_mcp_rejects_line_end_alone(archive, repo):
    body, err = call(archive, path=str(repo / "foo.py"), line_end=3)
    assert err and "line_start" in body


def run_cli(archive, *args):
    return CliRunner().invoke(cli_app, [*args, "--data-dir", str(archive)])


def test_blame_cli_prints_commits_and_their_sessions(archive, repo):
    result = run_cli(archive, "blame", f"{repo / 'foo.py'}:2")
    assert result.exit_code == 0, result.output
    assert "foo.py:2   1 commit, 1 line" in result.output
    assert "Rewrite line two" in result.output
    assert "ran the commit" in result.output
    assert "2 edits" in result.output
    assert "read one with: llma show" in result.output


def test_blame_cli_range_flag_and_json(archive, repo):
    result = run_cli(archive, "blame", str(repo / "foo.py"), "-L", "1,4", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["range"] == {"start": 1, "end": 4} and payload["total"] == 2


def test_blame_cli_says_when_no_session_is_in_the_window(archive, repo):
    (repo / "foo.py").write_text("one\nTWO\nthree\nfour\nfive\nsix\n", encoding="utf-8")
    result = run_cli(archive, "blame", str(repo / "foo.py"), "-L", "6",
                     "--workspace", "nothing-called-this")
    assert result.exit_code == 0, result.output
    assert "not committed yet" in result.output
    assert "no session in the archive edited this file since" in result.output


def test_blame_cli_fails_plainly_outside_a_repository(archive, tmp_path):
    loose = tmp_path / "loose.txt"
    loose.write_text("x\n")
    result = run_cli(archive, "blame", str(loose))
    assert result.exit_code == 1
    assert "not inside a git repository" in result.output
