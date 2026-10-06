"""Snapshots and garbage collection.

The archive's promise is that history is never lost, so most of these tests are about
what gc must NOT touch: a session whose raw file vanished, a drop that was deleted, a
blob one part still points at, the hand-made backups. The snapshot tests are about the
one property a file copy lacks -- consistency under WAL -- and the migration hook that
replaces the five copies made by hand.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from typer.testing import CliRunner

from llm_archive.cli import app as cli_app
from llm_archive.core import db, gc, snapshot
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import Message, Part, Session

TAG = "pm-MiniLM-L12"


@pytest.fixture
def data(tmp_path):
    (tmp_path / "vectors").mkdir()
    return tmp_path


@pytest.fixture
def con(data):
    return db.connect(data / "archive.db")


@pytest.fixture
def src(con):
    return db.source_id(con, "claude_code", "Claude Code", "cli")


def add(con, src, native, big_text: str | None = None, blobs: BlobStore | None = None,
        raw_path: str | None = None):
    """One session with a text part and, optionally, a tool_result held in a blob."""
    m = Message(native_id=f"{native}-0", role="user", seq=0, created_at=1771200000000)
    m.parts.append(Part(kind="text", seq=0, text="hello"))
    if big_text is not None:
        sha, size, path = blobs.put_text(big_text)
        m.parts.append(Part(kind="tool_result", seq=1, text=big_text[:40], bytes=size,
                            blob_sha=sha, blob_path=path, tool_name="Bash"))
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id=native, title=native,
        workspace_key="c:/p", workspace_label="p", host="box",
        started_at=1771200000000, raw_path=raw_path or f"/raw/{native}",
        raw_hash=native, messages=[m]))
    con.commit()
    return con.execute("SELECT id FROM session WHERE native_id = ?",
                       (native,)).fetchone()["id"]


def report(con, data, **kw):
    return gc.plan(con, data / "archive.db", data / "blobs", data / "vectors", TAG, **kw)


def run(con, data, **kw):
    return gc.apply(con, data / "archive.db", data / "blobs", data / "vectors", TAG, **kw)


# ------------------------------------------------------------------ snapshot

def test_a_snapshot_holds_what_is_still_in_the_wal(data, con, src):
    """The reason for the backup API: a file copy of archive.db alone misses this."""
    add(con, src, "s1")
    # a write that sits in the -wal until a checkpoint
    con.execute("UPDATE session SET title = 'renamed' WHERE native_id = 's1'")
    con.commit()
    assert (data / "archive.db-wal").stat().st_size > 0

    snap = snapshot.take(data / "archive.db", "test")
    assert snap.path.parent == data / "snapshots"
    assert snap.label == "test" and snap.bytes > 0
    copy = sqlite3.connect(snap.path)
    assert copy.execute("SELECT title FROM session").fetchone()[0] == "renamed"
    copy.close()


def test_snapshots_list_newest_first_and_never_overwrite(data, con, src):
    add(con, src, "s1")
    p = data / "archive.db"
    a = snapshot.take(p, "a", now=1_800_000_000)
    b = snapshot.take(p, "b", now=1_800_000_060)
    same_minute = snapshot.take(p, "b", now=1_800_000_070)
    names = [s.name for s in snapshot.list_snapshots(p)]
    assert names[0].endswith("-b-2.db") and names[1] == b.name and names[2] == a.name
    assert same_minute.path != b.path
    (data / "snapshots" / "notes.txt").write_text("not a snapshot")
    assert len(snapshot.list_snapshots(p)) == 3


def test_prune_keeps_the_newest(data, con, src):
    add(con, src, "s1")
    p = data / "archive.db"
    for i in range(4):
        snapshot.take(p, f"n{i}", now=1_800_000_000 + i * 60)
    gone = snapshot.prune(p, keep=2)
    assert [s.label for s in gone] == ["n1", "n0"]
    assert [s.label for s in snapshot.list_snapshots(p)] == ["n3", "n2"]
    with pytest.raises(snapshot.SnapshotError):
        snapshot.prune(p, keep=-1)


def test_restore_puts_the_old_content_back_and_keeps_an_undo(data, con, src):
    add(con, src, "s1")
    p = data / "archive.db"
    before = snapshot.take(p, "before")
    add(con, src, "s2")
    con.close()

    restored, safety = snapshot.restore(p, before.name.removesuffix(".db"))
    assert restored.path == before.path and safety.label == "pre-restore"
    con = db.connect(p)
    assert con.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1
    undo = sqlite3.connect(safety.path)
    assert undo.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 2
    undo.close()
    with pytest.raises(snapshot.SnapshotError):
        snapshot.restore(p, "archive-19990101-0000-nope")


def test_a_bad_label_is_refused_before_anything_is_written(data, con):
    with pytest.raises(snapshot.SnapshotError):
        snapshot.take(data / "archive.db", "no spaces here")
    assert not (data / "snapshots").exists()


def test_a_migration_takes_its_own_snapshot_first(data):
    """What the five archive.db.bak-* files were, done by the code."""
    p = data / "archive.db"
    con = db.connect(p)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('schema_version','9')")
    con.commit()
    con.close()
    db.connect(p).close()
    snaps = snapshot.list_snapshots(p)
    assert [s.label for s in snaps] == ["schema-v9"]
    old = sqlite3.connect(snaps[0].path)
    assert old.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "9"
    old.close()
    # a fresh archive, and one already current, take none
    db.connect(p).close()
    assert len(snapshot.list_snapshots(p)) == 1


# ------------------------------------------------------------------------ gc

def test_a_clean_archive_reclaims_nothing_and_reports_so(data, con, src):
    blobs = BlobStore(data / "blobs")
    add(con, src, "s1", "x" * 100_000, blobs)
    r = report(con, data)
    assert r.reclaimable_bytes == 0
    assert all(i.count == 0 for i in r.items if i.reclaim)
    assert "nothing changed" in gc.render(r)


def test_a_blob_nothing_points_at_goes_row_and_file(data, con, src):
    blobs = BlobStore(data / "blobs")
    sid = add(con, src, "s1", "x" * 100_000, blobs)
    keep = add(con, src, "s2", "y" * 100_000, blobs)
    # the part that referenced s1's blob is rewritten without it (a re-parse)
    con.execute("DELETE FROM part WHERE message_id IN "
                "(SELECT id FROM message WHERE session_id = ?) AND kind = 'tool_result'",
                (sid,))
    con.commit()
    r = report(con, data)
    assert r.item("orphan_blobs").count == 1 and r.item("orphan_blobs").bytes == 100_000
    assert list((data / "blobs").rglob("*")) and r.reclaimable_bytes == 100_000

    r = run(con, data)
    assert r.applied and r.reclaimed_bytes == 100_000
    assert con.execute("SELECT COUNT(*) FROM blob").fetchone()[0] == 1
    files = [f for f in (data / "blobs").rglob("*") if f.is_file()]
    assert len(files) == 1
    # the survivor is the one s2 still points at
    assert con.execute("""SELECT COUNT(*) FROM part p JOIN blob b ON b.id = p.blob_id
                          JOIN message m ON m.id = p.message_id
                          WHERE m.session_id = ?""", (keep,)).fetchone()[0] == 1
    assert report(con, data).reclaimable_bytes == 0


def test_a_blob_two_parts_share_is_kept_while_either_lives(data, con, src):
    """Content-addressed: the same bytes in two sessions are one blob."""
    blobs = BlobStore(data / "blobs")
    a = add(con, src, "s1", "z" * 50_000, blobs)
    add(con, src, "s2", "z" * 50_000, blobs)
    assert con.execute("SELECT COUNT(*) FROM blob").fetchone()[0] == 1
    con.execute("DELETE FROM part WHERE message_id IN "
                "(SELECT id FROM message WHERE session_id = ?) AND kind = 'tool_result'", (a,))
    con.commit()
    assert report(con, data).item("orphan_blobs").count == 0


def test_a_tool_input_blob_counts_as_referenced(data, con, src):
    blobs = BlobStore(data / "blobs")
    sid = add(con, src, "s1")
    sha, size, path = blobs.put_text("{" + "k" * 50_000 + "}")
    bid = db.blob_id(con, sha, size, path)
    mid = con.execute("SELECT id FROM message WHERE session_id = ?", (sid,)).fetchone()[0]
    con.execute("INSERT INTO part(message_id, seq, kind, text, tool_input_blob_id) "
                "VALUES (?, 1, 'tool_use', 'command: x', ?)", (mid, bid))
    con.commit()
    assert report(con, data).item("orphan_blobs").count == 0


def test_a_file_with_no_row_is_a_stray(data, con, src):
    blobs = BlobStore(data / "blobs")
    add(con, src, "s1", "x" * 100_000, blobs)
    sha, _, _ = blobs.put_text("left behind by a restore")     # file, no row
    (data / "blobs" / "zz").mkdir()
    (data / "blobs" / "zz" / "not-a-sha.txt").write_text("ignored")
    r = report(con, data)
    assert r.item("stray_files").count == 1
    run(con, data)
    assert not (data / "blobs" / sha[:2] / sha).exists()
    assert (data / "blobs" / "zz" / "not-a-sha.txt").exists()      # not ours
    assert report(con, data).item("stray_files").count == 0


def test_chunks_and_vectors_of_another_model_are_stale(data, con, src):
    sid = add(con, src, "s1")
    pid, mid = con.execute("SELECT p.id, m.id FROM part p JOIN message m ON m.id = p.message_id "
                           "WHERE m.session_id = ?", (sid,)).fetchone()
    for tag, row in ((TAG, 0), ("old-model", 0), ("old-model", 1)):
        con.execute("INSERT INTO chunk(part_id, message_id, session_id, seq, text, vec_row, "
                    "model_tag) VALUES (?,?,?,?,?,?,?)", (pid, mid, sid, row, "t", row, tag))
    con.commit()
    np.save(data / "vectors" / f"archive.{TAG}.npy", np.zeros((1, 4), np.float32))
    np.save(data / "vectors" / "archive.old-model.npy", np.zeros((2, 4), np.float32))

    r = report(con, data)
    assert r.item("stale_vectors").count == 3           # two rows, one file
    assert "old-model" in r.item("stale_vectors").details
    run(con, data)
    assert [r[0] for r in con.execute("SELECT DISTINCT model_tag FROM chunk")] == [TAG]
    assert not (data / "vectors" / "archive.old-model.npy").exists()
    assert (data / "vectors" / f"archive.{TAG}.npy").exists()


def test_missing_raw_files_and_drops_are_reported_never_deleted(data, con, src):
    add(con, src, "s1", raw_path=str(data / "nowhere.jsonl"))
    con.execute("INSERT INTO drop_file(sha256, name, kind, bytes, added_at, path) "
                "VALUES ('a'*64, 'x.zip', 'chatgpt', 1, 0, ?)", (str(data / "gone.zip"),))
    con.commit()
    r = run(con, data)
    assert r.item("missing_raw").count == 1 and not r.item("missing_raw").reclaim
    assert r.item("missing_drops").count == 1
    assert con.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM drop_file").fetchone()[0] == 1
    text = gc.render(r)
    assert "kept" in text


def test_legacy_backups_are_listed_with_sizes_and_left_alone(data, con, src):
    add(con, src, "s1")
    (data / "archive.db.bak-prefacts").write_bytes(b"0" * 1000)
    (data / "archive.db.bak-preturns").write_bytes(b"0" * 2000)
    r = run(con, data)
    item = r.item("legacy_backups")
    assert item.count == 2 and item.bytes == 3000 and not item.reclaim
    assert (data / "archive.db.bak-prefacts").exists()
    assert "archive.db.bak-prefacts" in gc.render(r)
    assert [p.name for p, _ in snapshot.legacy_backups(data / "archive.db")] == [
        "archive.db.bak-prefacts", "archive.db.bak-preturns"]


def test_vacuum_runs_when_asked_and_reports_the_shrink(data, con, src):
    blobs = BlobStore(data / "blobs")
    for i in range(30):
        add(con, src, f"s{i}")
    # inline content large enough to free real pages when it goes
    con.execute("UPDATE part SET text = ?", ("q" * 20_000,))
    con.commit()
    con.execute("DELETE FROM part")
    con.commit()
    assert report(con, data).item("free_pages").count > 0
    r = run(con, data, vacuum=True)
    assert r.vacuumed
    assert con.execute("PRAGMA freelist_count").fetchone()[0] == 0
    # small archives are not rewritten unasked
    r2 = run(con, data)
    assert not r2.vacuumed


def test_gc_stands_down_while_a_sync_holds_the_lock(data, con, src):
    add(con, src, "s1")
    from llm_archive.core import sync
    with sync.Lock(data / sync.LOCK_NAME):
        with pytest.raises(gc.Busy):
            run(con, data)


# ------------------------------------------------------------------- cli

def test_the_cli_reports_then_reclaims(data, con, src):
    blobs = BlobStore(data / "blobs")
    sid = add(con, src, "s1", "x" * 100_000, blobs)
    con.execute("DELETE FROM part WHERE message_id IN "
                "(SELECT id FROM message WHERE session_id = ?) AND kind = 'tool_result'", (sid,))
    con.commit()
    con.close()
    d = str(data)
    r = CliRunner().invoke(cli_app, ["gc", "--data-dir", d])
    assert r.exit_code == 0, r.output
    assert "dry run" in r.output and "blob rows nothing points at" in r.output
    assert "reclaimable 100.0 KB" in r.output

    r = CliRunner().invoke(cli_app, ["gc", "--json", "--data-dir", d])
    assert json.loads(r.output)["reclaimable_bytes"] == 100_000

    r = CliRunner().invoke(cli_app, ["gc", "--apply", "--data-dir", d])
    assert r.exit_code == 0 and "reclaimed 100.0 KB" in r.output
    r = CliRunner().invoke(cli_app, ["gc", "--data-dir", d])
    assert "reclaimable 0 B" in r.output


def test_the_snapshot_cli_takes_lists_prunes_and_restores(data, con, src):
    add(con, src, "s1")
    con.close()
    d = str(data)
    r = CliRunner().invoke(cli_app, ["snapshot", "--label", "one", "--data-dir", d])
    assert r.exit_code == 0 and "-one.db" in r.output
    r = CliRunner().invoke(cli_app, ["snapshot", "list", "--data-dir", d])
    assert "-one.db" in r.output
    name = [ln.split()[0] for ln in r.output.splitlines() if "-one.db" in ln][0]

    con = db.connect(data / "archive.db")
    add(con, src, "s2")
    con.close()
    r = CliRunner().invoke(cli_app, ["snapshot", "restore", name, "--data-dir", d])
    assert r.exit_code == 0 and "restore that to undo" in r.output
    con = db.connect(data / "archive.db")
    assert con.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1
    con.close()

    r = CliRunner().invoke(cli_app, ["snapshot", "prune", "--keep", "1", "--data-dir", d])
    assert "1 removed, 1 kept" in r.output
    r = CliRunner().invoke(cli_app, ["snapshot", "--label", "bad label", "--data-dir", d])
    assert r.exit_code == 1
