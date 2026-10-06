"""The digest: a window of sessions, arranged so a week can be read in a minute.

The window is about activity, subagents fold into their parent, the cost is the
list-price value and files inside the project come first -- each of those is a case
here, because each was a wrong first draft against the real archive.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from llm_archive.cli import app as cli_app
from llm_archive.core import db, schedule
from llm_archive.core.models import Message, Part, Session
from llm_archive.search import facts
from llm_archive.stats import digest

DAY = 86_400_000
T0 = 1771200000000              # 2026-02-16 00:00 UTC
SINCE, UNTIL = T0 + 7 * DAY, T0 + 14 * DAY
WS = "c:/proj/app"


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "archive.db")


@pytest.fixture
def src(con):
    return db.source_id(con, "claude_code", "Claude Code", "cli")


def add(con, src, native, *, started, ended=None, steps=(), files=(), model="claude-opus-5",
        tok_out=1_000_000, title=None, parent=None, workspace=WS, label="app"):
    """A session with text steps and, after them, one Edit per file path."""
    msgs = []
    for i, (role, text, active) in enumerate(steps):
        m = Message(native_id=f"{native}-{i}", role=role, seq=i,
                    created_at=started + i * 1000, on_active_path=active)
        m.parts.append(Part(kind="text", seq=0, text=text))
        msgs.append(m)
    for j, path in enumerate(files):
        m = Message(native_id=f"{native}-f{j}", role="assistant", seq=100 + j,
                    created_at=started + (100 + j) * 1000)
        part = Part(kind="tool_use", seq=0, text=f"file_path: {path}", tool_name="Edit",
                    tool_ok=True)
        part.tool_input = json.dumps({"file_path": path, "old_string": "a",
                                      "new_string": "b"})
        m.parts.append(part)
        msgs.append(m)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id=native, title=title or native,
        workspace_key=workspace, workspace_label=label, host="box",
        started_at=started, ended_at=ended or (started + 200_000),
        raw_path=f"/raw/{native}", raw_hash=native, parent_native_id=parent,
        model_primary=model, tok_out=tok_out, messages=msgs))
    con.commit()
    return con.execute("SELECT id FROM session WHERE native_id = ?",
                       (native,)).fetchone()["id"]


def build(con, **kw):
    return digest.build(con, SINCE, UNTIL, now_ms=UNTIL, **kw)


def ids(d):
    return {s["session_id"] for w in d["workspaces"] for s in w["sessions_list"]}


# ------------------------------------------------------------------ the window

def test_the_window_is_about_activity_not_birth(con, src):
    before = add(con, src, "before", started=SINCE - 3 * DAY, ended=SINCE - 2 * DAY)
    carried = add(con, src, "carried", started=SINCE - 3 * DAY, ended=SINCE + DAY)
    inside = add(con, src, "inside", started=SINCE + DAY)
    after = add(con, src, "after", started=UNTIL + DAY)
    d = build(con)
    assert ids(d) == {carried, inside}
    rows = {s["session_id"]: s for w in d["workspaces"] for s in w["sessions_list"]}
    assert rows[carried]["carried_over"] is True
    assert rows[inside]["carried_over"] is False
    assert d["totals"]["sessions"] == 2
    assert before not in ids(d) and after not in ids(d)


def test_a_session_replayed_by_its_continuation_is_not_counted_twice(con, src):
    parent = add(con, src, "p", started=SINCE + DAY,
                 steps=[("user", "start", True), ("assistant", "ok", True)])
    child = add(con, src, "c", started=SINCE + 2 * DAY,
                steps=[("user", "start", True), ("assistant", "ok", True),
                       ("user", "more", True), ("assistant", "done", True)])
    con.execute("UPDATE session SET continues_session_id = ?, continues_overlap = 2 "
                "WHERE id = ?", (parent, child))
    con.commit()
    assert ids(build(con)) == {child}


def test_subagents_fold_into_the_session_that_spawned_them(con, src):
    parent = add(con, src, "main", started=SINCE + DAY, tok_out=1_000_000,
                 steps=[("user", "explore", True), ("assistant", "did", True)])
    add(con, src, "sub1", started=SINCE + DAY, tok_out=1_000_000, parent="main",
        steps=[("user", "look at x", True), ("assistant", "x", True)])
    add(con, src, "sub2", started=SINCE + DAY, tok_out=1_000_000, parent="main",
        steps=[("user", "look at y", True), ("assistant", "y", True)])
    d = build(con)
    assert ids(d) == {parent}
    row = d["workspaces"][0]["sessions_list"][0]
    assert row["subagents"] == 2
    # three sessions' worth of opus output at $25/M
    assert row["usd"] == pytest.approx(75.0)
    assert row["turns"] == 6
    assert d["totals"]["subagents"] == 2


# ----------------------------------------------------------------- per session

def test_the_first_prompt_is_the_first_live_user_turn(con, src):
    sid = add(con, src, "fp", started=SINCE + DAY, steps=[
        ("user", "the  abandoned   version", False),
        ("user", "make the batch size\nconfigurable please", True),
        ("assistant", "sure", True),
        ("user", "and a flag", True)])
    row = build(con)["workspaces"][0]["sessions_list"][0]
    assert row["session_id"] == sid
    assert row["first_prompt"] == "make the batch size configurable please"


def test_files_inside_the_project_come_first_and_the_rest_are_counted(con, src):
    inside = [f"C:/proj/app/src/f{i}.py" for i in range(8)]
    outside = ["C:/Users/t/AppData/Local/Temp/claude/x/scratchpad/patch.py",
               "C:/Users/t/.claude/projects/app/memory/note.md"]
    add(con, src, "files", started=SINCE + DAY, files=outside + inside)
    facts.rebuild(con)
    d = build(con)
    assert d["facts_available"] is True
    row = d["workspaces"][0]["sessions_list"][0]
    assert len(row["files"]) == digest.FILES_PER_SESSION
    assert all(f["key"].startswith("src/") for f in row["files"])
    assert row["more_files"] == 2
    assert row["outside_files"] == 2
    hot = d["workspaces"][0]["hot_files"]
    assert hot and all(f["key"].startswith("src/") for f in hot)


def test_without_an_index_the_digest_says_so_instead_of_showing_nothing(con, src):
    add(con, src, "noidx", started=SINCE + DAY, files=["C:/proj/app/a.py"])
    d = build(con)
    assert d["facts_available"] is False
    assert "llma index" in digest.render_text(d)
    assert "llma index" in digest.render_markdown(d)


def test_cost_is_the_list_price_value_with_subscriptions_kept_apart(con, src):
    a = add(con, src, "opus", started=SINCE + DAY, model="claude-opus-5",
            tok_out=1_000_000)
    b = add(con, src, "copilot", started=SINCE + DAY, model="copilot/gpt-5",
            tok_out=1_000_000)
    c = add(con, src, "mystery", started=SINCE + DAY, model="some-model-2099",
            tok_out=1_000_000)
    d = build(con)
    rows = {s["session_id"]: s for w in d["workspaces"] for s in w["sessions_list"]}
    assert rows[a]["usd"] == pytest.approx(25.0) and rows[a]["billed"] is True
    assert rows[b]["billed"] is False and rows[b]["priced"] is True
    assert rows[c]["priced"] is False
    assert d["totals"]["usd"] == pytest.approx(25.0)
    assert d["totals"]["unpriced_sessions"] == 1
    text = digest.render_text(d)
    assert "$25.00 at list price" in text
    assert "—" in text                       # the unpriced one


def test_unfinished_sessions_in_the_window_are_listed(con, src):
    waiting = add(con, src, "wait", started=SINCE + DAY,
                  steps=[("user", "still here?", True)])
    add(con, src, "old-wait", started=SINCE - 5 * DAY, ended=SINCE - 5 * DAY + 1000,
        steps=[("user", "forgotten", True)])
    d = build(con)
    assert [u["session_id"] for u in d["unfinished"]] == [waiting]
    assert d["unfinished"][0]["reason"] == "unanswered"
    rows = {s["session_id"]: s for w in d["workspaces"] for s in w["sessions_list"]}
    assert rows[waiting]["unfinished"] == "unanswered"
    assert "UNFINISHED (1)" in digest.render_text(d)


def test_workspaces_are_grouped_by_label_busiest_first(con, src):
    add(con, src, "a1", started=SINCE + DAY, workspace="c:/a", label="alpha")
    add(con, src, "b1", started=SINCE + DAY, workspace="c:/b", label="beta")
    add(con, src, "b2", started=SINCE + DAY, workspace="c:/b2", label="beta")
    add(con, src, "n1", started=SINCE + DAY, workspace=None, label=None)
    d = build(con)
    assert [w["workspace"] for w in d["workspaces"]] == ["beta", "alpha", digest.NO_PROJECT]
    assert d["workspaces"][0]["sessions"] == 2


def test_an_empty_window_renders_as_such(con, src):
    d = build(con)
    assert d["totals"]["sessions"] == 0
    assert "nothing happened" in digest.render_text(d)
    assert "Nothing happened" in digest.render_markdown(d)


def test_markdown_links_sessions_when_given_a_url_maker(con, src):
    sid = add(con, src, "md", started=SINCE + DAY, title="Fix the build")
    d = build(con, label="last-week")
    plain = digest.render_markdown(d)
    linked = digest.render_markdown(d, session_url=lambda s: f"/session/{s}")
    assert f"#{sid} **Fix the build**" in plain
    assert f"[#{sid}](/session/{sid}) **Fix the build**" in linked
    assert "(last-week)" in linked


# ------------------------------------------------------------------- surfaces

def test_the_cli_prints_files_and_dates_the_output(tmp_path, con, src):
    add(con, src, "cli", started=SINCE + DAY, title="Digest me")
    con.close()
    data = str(tmp_path)
    r = CliRunner().invoke(cli_app, ["digest", "--since", "2026-02-23",
                                     "--until", "2026-03-02", "--data-dir", data])
    assert r.exit_code == 0, r.output
    assert "Digest me" in r.output and "1 session" in r.output

    r = CliRunner().invoke(cli_app, ["digest", "--since", "2026-02-23",
                                     "--until", "2026-03-02", "--json", "--data-dir", data])
    assert json.loads(r.output)["totals"]["sessions"] == 1

    out = tmp_path / "digests"
    r = CliRunner().invoke(cli_app, ["digest", "--since", "2026-02-23",
                                     "--until", "2026-03-02", "--format", "md",
                                     "--out", str(out), "--data-dir", data])
    assert r.exit_code == 0, r.output
    written = list(out.glob("digest-*.md"))
    assert [p.name for p in written] == ["digest-2026-03-02.md"]
    assert "# Digest" in written[0].read_text(encoding="utf-8")

    r = CliRunner().invoke(cli_app, ["digest", "--since", "whenever", "--data-dir", data])
    assert r.exit_code != 0


def test_the_web_page_renders_with_session_links(tmp_path, con, src):
    from llm_archive.web.app import create_app

    sid = add(con, src, "web", started=SINCE + DAY, title="On the <page>")
    con.close()
    client = TestClient(create_app(tmp_path, fetch_images=False))
    r = client.get("/digest?since=2026-02-23&until=2026-03-02")
    assert r.status_code == 200
    assert f'href="/session/{sid}"' in r.text
    assert "On the &lt;page&gt;" in r.text             # archive text is never HTML
    assert client.get("/digest?since=nope").status_code == 200


# ----------------------------------------------------------------- scheduling

def test_the_digest_task_is_weekly_short_and_writes_to_the_data_folder():
    plan = schedule.make_digest_plan()
    xml = schedule.build_xml(plan)
    assert plan.task_name == schedule.DIGEST_TASK_NAME
    assert "<ScheduleByWeek>" in xml and "<Monday/>" in xml
    assert "<ExecutionTimeLimit>PT10M</ExecutionTimeLimit>" in xml
    assert "T08:00:00" in xml
    assert "digest --since last-week --format md --out" in plan.arguments
    assert plan.out_dir.name == "digests"
    # the sync task is untouched by any of this
    assert "<ScheduleByDay>" in schedule.build_xml(schedule.make_plan())


def test_the_digest_task_takes_a_weekday_and_refuses_a_bad_one():
    assert schedule.make_digest_plan(on="friday").weekly_on == "Friday"
    with pytest.raises(ValueError):
        schedule.make_digest_plan(on="someday")
