"""Taking data in, and the promise that a re-ingest can only add.

Two halves. `llma add`'s identification and filing, and — the part that matters — what
happens when the same account is exported twice. Nothing before this drove `ingest.run`
end to end, so `skipped`, `stale`, `appended` and `retained` had no coverage at all,
and the delete-and-rewrite behaviour that made a shrinking export lose history was
never exercised.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest

from llm_archive.core import db, ingest, intake
from llm_archive.core.models import Message, Part, Session

T0 = 1787000000000


# -- helpers ---------------------------------------------------------------


@pytest.fixture
def archive(tmp_path):
    con = db.connect(tmp_path / "archive.db")
    yield con, tmp_path
    con.close()


def claude_export(path: Path, conversations) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json",
                    json.dumps(conversations, ensure_ascii=False))
    return path


def conv(cid, name, messages):
    return {"uuid": cid, "name": name, "summary": "",
            "created_at": "2026-08-01T10:00:00Z",
            "updated_at": "2026-08-01T11:00:00Z",
            "account": {"uuid": "acct"},
            "chat_messages": messages}


def turn(uid, parent, sender, body):
    return {"uuid": uid, "parent_message_uuid": parent, "sender": sender,
            "created_at": "2026-08-01T10:00:00Z", "text": body,
            "content": [{"type": "text", "text": body}],
            "attachments": [], "files": []}


def chain(n, prefix="u"):
    """A straight user/assistant chain of `n` messages."""
    out, parent = [], None
    for i in range(n):
        uid = f"{prefix}{i}"
        out.append(turn(uid, parent, "human" if i % 2 == 0 else "assistant",
                        f"message {i}"))
        parent = uid
    return out


def run_claude(con, drops: Path, blobs=None, **kw):
    from llm_archive.adapters.claude_web import ClaudeWebAdapter
    adapter = ClaudeWebAdapter(drops=drops, blobs=blobs)
    return ingest.run(adapter, con, blobs, archive_drops=False, **kw)


def stored_messages(con, native_id="c1"):
    return list(con.execute(
        "SELECT m.native_id, m.absent_since FROM message m "
        "  JOIN session s ON s.id = m.session_id "
        " WHERE s.native_id = ? ORDER BY m.seq", (native_id,)))


def age(path: Path, seconds: int) -> Path:
    """Backdate a drop's mtime — the clock that says when an export was taken."""
    stamp = time.time() - seconds
    import os
    os.utime(path, (stamp, stamp))
    return path


# -- identification --------------------------------------------------------


def test_identifies_each_export_format(tmp_path):
    """One fixture per drop adapter, each claimed by exactly one of them."""
    from llm_archive.adapters import _drops

    drops = tmp_path / "drops"
    drops.mkdir()

    claude_export(drops / "conversations-000.zip", [conv("c1", "A", chain(2))])
    with zipfile.ZipFile(drops / "deepseek_data-2026-08-27.zip", "w") as zf:
        zf.writestr("conversations.json", json.dumps([{
            "id": "d1", "title": "t", "inserted_at": "2026-08-26T23:15:46+08:00",
            "mapping": {"root": {"id": "root", "parent": None, "children": [],
                                 "message": None}}}]))
    with zipfile.ZipFile(drops / "5f4c8d58.zip", "w") as zf:
        zf.writestr("ttl/30d/export_data/u1/prod-grok-backend.json",
                    json.dumps({"conversations": []}))
    (drops / "OpenRouter Chat.json").write_text(
        json.dumps({"version": "orpg-1", "messages": []}), encoding="utf-8")
    (drops / "threads-export-2026.json").write_text(
        json.dumps({"version": "11.0.1", "threads": [], "messages": []}),
        encoding="utf-8")
    (drops / "share.json").write_text(
        json.dumps({"sharedID": "abc", "turns": []}), encoding="utf-8")

    found = {p.name: intake.identify(p) for p in _drops.candidates(drops)}
    assert found == {
        "conversations-000.zip": "claude_web",
        "deepseek_data-2026-08-27.zip": "deepseek",
        "5f4c8d58.zip": "grok",
        "OpenRouter Chat.json": "openrouter",
        "threads-export-2026.json": "t3chat",
        "share.json": "copilot_web",
    }


def test_a_non_export_is_identified_as_nothing(tmp_path):
    """~/Downloads is mostly videos and PDFs; none of them may be claimed."""
    for name, body in (("holiday.mp4", b"\x00\x00\x00 ftyp"),
                       ("paper.pdf", b"%PDF-1.7\n"),
                       ("notes.txt", b"hello"),
                       ("random.json", b'{"unrelated": true}')):
        (tmp_path / name).write_bytes(body)
    assert all(intake.identify(p) is None for p in tmp_path.iterdir())


def test_add_copies_into_drops_and_leaves_the_original(archive):
    con, tmp_path = archive
    source = claude_export(tmp_path / "dl" / "conversations-000.zip",
                           [conv("c1", "A", chain(2))])
    drops = tmp_path / "drops"

    taken = intake.take_all([tmp_path / "dl"], con, drops)

    assert [t.action for t in taken] == ["added"]
    assert taken[0].kind == "claude_web"
    assert source.exists(), "the user's download must be left where it was"
    assert (drops / "conversations-000.zip").exists()


def test_the_same_export_under_a_new_name_is_recognised(archive):
    """Identity is the sha256: Mistral renames its export on every download."""
    con, tmp_path = archive
    payload = [conv("c1", "A", chain(2))]
    first = claude_export(tmp_path / "dl" / "conversations-000.zip", payload)
    drops = tmp_path / "drops"
    intake.take_all([first], con, drops)

    again = claude_export(tmp_path / "dl2" / "conversations-000 (1).zip", payload)
    taken = intake.take_all([again], con, drops)

    assert taken[0].action == "held"
    assert len(list(drops.iterdir())) == 1


def test_a_name_collision_does_not_overwrite(archive):
    """Every claude.ai export is called conversations-000.zip; last month's is not this."""
    con, tmp_path = archive
    drops = tmp_path / "drops"
    intake.take_all([claude_export(tmp_path / "a" / "conversations-000.zip",
                                   [conv("c1", "A", chain(2))])], con, drops)
    intake.take_all([claude_export(tmp_path / "b" / "conversations-000.zip",
                                   [conv("c1", "A", chain(4))])], con, drops)

    assert sorted(p.name for p in drops.iterdir()) == [
        "conversations-000-2.zip", "conversations-000.zip"]


def test_a_file_already_in_drops_is_recorded_not_copied(archive):
    """Pointing `add` at the drops folder itself must not file everything twice."""
    con, tmp_path = archive
    drops = tmp_path / "drops"
    claude_export(drops / "conversations-000.zip", [conv("c1", "A", chain(2))])

    taken = intake.take_all([drops], con, drops)

    assert taken[0].action == "added" and taken[0].detail == "already in drops"
    assert [p.name for p in drops.iterdir()] == ["conversations-000.zip"]


def test_a_non_export_is_recorded_so_it_is_not_re_sniffed(archive):
    con, tmp_path = archive
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.7\n")
    taken = intake.take_all([tmp_path / "paper.pdf"], con, tmp_path / "drops")

    assert taken[0].action == "skipped"
    row = con.execute("SELECT kind FROM drop_file").fetchone()
    assert row["kind"] is None

    # Recognised on sight the second time, and still reported as what it is rather
    # than as something the archive is holding.
    again = intake.take_all([tmp_path / "paper.pdf"], con, tmp_path / "drops")
    assert again[0].action == "skipped" and again[0].detail == "not an export"
    assert con.execute("SELECT COUNT(*) c FROM drop_file").fetchone()["c"] == 1


def test_a_duplicate_non_export_under_another_name_is_still_not_an_export(archive):
    """`thing.pdf` and `thing (1).pdf` are byte-identical; neither is held."""
    con, tmp_path = archive
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.7\n")
    (tmp_path / "a (1).pdf").write_bytes(b"%PDF-1.7\n")

    taken = intake.take_all([tmp_path / "a.pdf", tmp_path / "a (1).pdf"], con,
                            tmp_path / "drops")
    assert [t.action for t in taken] == ["skipped", "skipped"]


def test_scanning_a_folder_ignores_what_is_not_an_export(archive):
    con, tmp_path = archive
    downloads = tmp_path / "dl"
    downloads.mkdir()
    claude_export(downloads / "conversations-000.zip", [conv("c1", "A", chain(2))])
    (downloads / "holiday.mp4").write_bytes(b"\x00\x00\x00 ftyp")
    (downloads / "paper.pdf").write_bytes(b"%PDF-1.7\n")

    taken = intake.take_all([downloads], con, tmp_path / "drops")
    assert sorted((t.source.name, t.action) for t in taken) == [
        ("conversations-000.zip", "added"),
        ("holiday.mp4", "skipped"),
        ("paper.pdf", "skipped"),
    ]


# -- the append promise ----------------------------------------------------


def test_a_grown_conversation_appends_rather_than_duplicating(archive):
    """The case this whole change exists for: export in August, export again later."""
    con, tmp_path = archive
    drops = tmp_path / "drops"

    claude_export(drops / "export-a.zip", [conv("c1", "A", chain(10))])
    first = run_claude(con, drops)
    assert (first.new, first.updated, first.appended) == (1, 0, 0)
    assert len(stored_messages(con)) == 10

    # The same chat, 15 messages longer, in a newer drop.
    (drops / "export-a.zip").unlink()
    age(claude_export(drops / "export-b.zip", [conv("c1", "A", chain(25))]), 0)
    second = run_claude(con, drops)

    assert (second.new, second.updated) == (0, 1)
    assert second.appended == 15
    assert len(stored_messages(con)) == 25, "no duplicates, no truncation"
    assert con.execute("SELECT COUNT(*) c FROM session").fetchone()["c"] == 1


def test_an_older_export_cannot_overwrite_a_newer_one(archive):
    """Re-dropping last month's export must not undo this month's.

    Before `exported_at`, discovery ordered drops by filename — meaningless for Grok's
    bare-uuid ZIP — so whichever sorted last won, and a stale snapshot silently
    truncated a conversation that had grown.
    """
    con, tmp_path = archive
    drops = tmp_path / "drops"

    age(claude_export(drops / "new.zip", [conv("c1", "A", chain(25))]), 0)
    run_claude(con, drops)
    (drops / "new.zip").unlink()

    age(claude_export(drops / "old.zip", [conv("c1", "A", chain(10))]), 60 * 86400)
    result = run_claude(con, drops)

    assert result.stale == 1
    assert (result.new, result.updated) == (0, 0)
    assert len(stored_messages(con)) == 25, "the older export was refused"


def test_both_exports_present_the_newer_one_still_wins(archive):
    """Order within one run: discovery hands drops over oldest-first."""
    con, tmp_path = archive
    drops = tmp_path / "drops"

    # Named so that filename order is the WRONG order — 'a' sorts first but is newer.
    age(claude_export(drops / "a.zip", [conv("c1", "A", chain(25))]), 0)
    age(claude_export(drops / "z.zip", [conv("c1", "A", chain(10))]), 60 * 86400)

    run_claude(con, drops)
    assert len(stored_messages(con)) == 25


def test_a_shrinking_export_never_removes_history(archive):
    """A chat pruned or deleted server-side must not take the archive's copy with it."""
    con, tmp_path = archive
    drops = tmp_path / "drops"

    age(claude_export(drops / "full.zip", [conv("c1", "A", chain(25))]), 10)
    run_claude(con, drops)
    (drops / "full.zip").unlink()

    # Same export, three turns gone from the middle, and genuinely newer.
    short = chain(25)
    del short[10:13]
    short[10]["parent_message_uuid"] = short[9]["uuid"]
    age(claude_export(drops / "pruned.zip", [conv("c1", "A", short)]), 0)
    result = run_claude(con, drops)

    rows = stored_messages(con)
    assert len(rows) == 25, "nothing was deleted"
    assert result.retained == 3
    absent = [r["native_id"] for r in rows if r["absent_since"] is not None]
    assert sorted(absent) == ["u10", "u11", "u12"]


def test_absent_since_dates_the_loss_not_the_latest_ingest(archive):
    con, tmp_path = archive
    drops = tmp_path / "drops"

    age(claude_export(drops / "full.zip", [conv("c1", "A", chain(6))]), 20)
    run_claude(con, drops)
    (drops / "full.zip").unlink()

    age(claude_export(drops / "p1.zip", [conv("c1", "A", chain(4))]), 10)
    run_claude(con, drops)
    first_stamp = {r["native_id"]: r["absent_since"] for r in stored_messages(con)}
    (drops / "p1.zip").unlink()

    # A third, still-shorter export. The already-missing pair keeps its original stamp.
    age(claude_export(drops / "p2.zip", [conv("c1", "A", chain(3))]), 0)
    run_claude(con, drops, force=True)
    second = {r["native_id"]: r["absent_since"] for r in stored_messages(con)}

    assert second["u4"] == first_stamp["u4"] and second["u5"] == first_stamp["u5"]
    assert second["u3"] is not None and second["u3"] >= first_stamp["u4"]


def test_a_message_that_comes_back_is_no_longer_absent(archive):
    con, tmp_path = archive
    drops = tmp_path / "drops"

    age(claude_export(drops / "a.zip", [conv("c1", "A", chain(6))]), 20)
    run_claude(con, drops)
    (drops / "a.zip").unlink()
    age(claude_export(drops / "b.zip", [conv("c1", "A", chain(4))]), 10)
    run_claude(con, drops)
    assert sum(1 for r in stored_messages(con) if r["absent_since"]) == 2
    (drops / "b.zip").unlink()

    age(claude_export(drops / "c.zip", [conv("c1", "A", chain(6))]), 0)
    run_claude(con, drops)
    assert all(r["absent_since"] is None for r in stored_messages(con))


def test_counts_describe_the_stored_rows_not_the_snapshot(archive):
    """After a merge the tree can hold more than the snapshot supplied."""
    con, tmp_path = archive
    drops = tmp_path / "drops"

    age(claude_export(drops / "full.zip", [conv("c1", "A", chain(10))]), 10)
    run_claude(con, drops)
    (drops / "full.zip").unlink()
    age(claude_export(drops / "short.zip", [conv("c1", "A", chain(6))]), 0)
    run_claude(con, drops)

    row = con.execute("SELECT msg_count, turn_count FROM session").fetchone()
    stored = con.execute(
        "SELECT COUNT(*) c FROM message WHERE on_active_path=1").fetchone()["c"]
    assert row["msg_count"] == stored == 10


def test_an_unchanged_export_is_skipped_not_rewritten(archive):
    con, tmp_path = archive
    drops = tmp_path / "drops"
    claude_export(drops / "a.zip", [conv("c1", "A", chain(4))])

    run_claude(con, drops)
    again = run_claude(con, drops)

    assert (again.skipped, again.new, again.updated) == (1, 0, 0)


def test_force_reparses_and_still_does_not_duplicate(archive):
    con, tmp_path = archive
    drops = tmp_path / "drops"
    claude_export(drops / "a.zip", [conv("c1", "A", chain(4))])

    run_claude(con, drops)
    forced = run_claude(con, drops, force=True)

    assert (forced.skipped, forced.updated) == (0, 1)
    assert len(stored_messages(con)) == 4


def test_a_session_whose_messages_have_no_ids_falls_back_to_rewrite(archive):
    """Codex synthesises native_id=None; the merge key cannot address those rows."""
    con, _ = archive
    src = db.source_id(con, "codex", "Codex", "cli")

    def build(n):
        return Session(
            source_kind="codex", native_id="s1", started_at=T0,
            raw_path="/x", raw_hash=f"h{n}",
            messages=[Message(native_id=None, role="user", created_at=T0, seq=i,
                              parts=[Part(kind="text", seq=0, text=f"m{i}")])
                      for i in range(n)])

    db.upsert_session(con, src, build(4))
    merged = db.upsert_session(con, src, build(2))

    assert merged.wholesale is True
    assert merged.retained == 0
    assert con.execute("SELECT COUNT(*) c FROM message").fetchone()["c"] == 2


# -- retiring a spent drop -------------------------------------------------


def test_an_ingested_drop_is_archived_and_raw_path_follows_it(archive):
    """`freshness` stats raw_path to date an export; a stale path silently changes
    which clock the report is answering with."""
    con, tmp_path = archive
    drops = tmp_path / "drops"
    drop = claude_export(drops / "conversations-000.zip", [conv("c1", "A", chain(4))])
    intake.take_all([drop], con, drops)     # gives it a ledger row

    from llm_archive.adapters.claude_web import ClaudeWebAdapter
    result = ingest.run(ClaudeWebAdapter(drops=drops), con)

    assert result.drops_archived == 1
    assert not drop.exists()
    moved = next((drops / intake.ARCHIVE).rglob("conversations-000.zip"))
    raw_path = con.execute("SELECT raw_path FROM session").fetchone()["raw_path"]
    assert Path(raw_path) == moved and moved.exists()

    row = con.execute("SELECT ingested_at, sessions, path FROM drop_file").fetchone()
    assert row["ingested_at"] and row["sessions"] == 1
    assert Path(row["path"]) == moved


def test_discovery_does_not_descend_into_the_archive(archive):
    """Otherwise every export ever taken would be re-sniffed on every run, forever."""
    con, tmp_path = archive
    drops = tmp_path / "drops"
    claude_export(drops / "conversations-000.zip", [conv("c1", "A", chain(4))])

    from llm_archive.adapters.claude_web import ClaudeWebAdapter
    ingest.run(ClaudeWebAdapter(drops=drops), con)
    second = ingest.run(ClaudeWebAdapter(drops=drops), con)

    assert second.files == 0 and second.skipped == 0


def test_an_archived_drop_can_still_be_re_read(archive):
    """The raw bytes stay: `raw_path` + `raw_hash` is a promise you can re-parse."""
    con, tmp_path = archive
    drops = tmp_path / "drops"
    claude_export(drops / "conversations-000.zip", [conv("c1", "A", chain(4))])

    from llm_archive.adapters.claude_web import ClaudeWebAdapter
    ingest.run(ClaudeWebAdapter(drops=drops), con)

    raw_path = Path(con.execute("SELECT raw_path FROM session").fetchone()["raw_path"])
    with zipfile.ZipFile(raw_path) as zf:
        again = json.loads(zf.read("conversations.json"))
    assert len(again[0]["chat_messages"]) == 4


# -- the ledger ------------------------------------------------------------


def test_ledger_lists_what_is_held(archive):
    con, tmp_path = archive
    drops = tmp_path / "drops"
    intake.take_all([claude_export(tmp_path / "dl" / "conversations-000.zip",
                                   [conv("c1", "A", chain(2))])], con, drops)
    (tmp_path / "dl" / "paper.pdf").write_bytes(b"%PDF-1.7\n")
    intake.take_all([tmp_path / "dl" / "paper.pdf"], con, drops)

    # "What am I holding?" means the exports. The rejects are recorded so a repeat
    # scan does not reopen them, and counted separately rather than listed — this
    # machine has 139 of them against 10 real exports.
    assert {r["name"]: r["kind"] for r in intake.ledger(con)} == {
        "conversations-000.zip": "claude_web"}
    assert intake.passed_over(con) == 1
    assert len(intake.ledger(con, exports_only=False)) == 2


# -- schema ----------------------------------------------------------------


def test_v8_archive_migrates_to_v9(tmp_path):
    """The three columns arrive without losing sessions, and re-running is a no-op."""
    path = tmp_path / "old.db"
    con = db.connect(path)
    src = db.source_id(con, "claude_web", "Claude.ai", "web")
    db.upsert_session(con, src, Session(
        source_kind="claude_web", native_id="c1", started_at=T0,
        raw_path="/x", raw_hash="h",
        messages=[Message(native_id="m1", role="user", created_at=T0,
                          parts=[Part(kind="text", seq=0, text="hi")])]))
    con.commit()
    # Pretend it was written before v9 existed.
    con.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
    con.commit()
    con.close()

    con = db.connect(path)
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'"
                       ).fetchone()["value"] == str(db.SCHEMA_VERSION)
    assert con.execute("SELECT COUNT(*) c FROM session").fetchone()["c"] == 1
    assert con.execute("SELECT exported_at FROM session").fetchone()["exported_at"] is None
    assert con.execute("SELECT absent_since FROM message").fetchone()[0] is None
    con.close()

    db.connect(path).close()        # re-running the ladder must not raise
