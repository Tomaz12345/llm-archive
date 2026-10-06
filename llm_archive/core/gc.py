"""Reclaim what nothing references any more, and say what is merely missing.

The archive's contract is that history is never lost: an export that shrank does not
take messages out (`message.absent_since`), a raw file that vanished does not take its
session out. So a garbage collector here has a short list of things it may touch, all
of them derived or unreferenced, and a longer list it may only report.

**Reclaimed, with `--apply`:**

* Blob rows no part points at, and their files. `upsert_session` rewrites a
  re-parsed message's parts and never deletes a blob, so every `--force` re-ingest
  after a parser change can strand the previous overflow content. Content-addressed
  storage keeps this rare -- the same bytes re-hash to the same blob -- which is why
  the real archive had none when this was written, and why the count is worth
  printing rather than assuming.
* Files in `blobs/xx/` with no row at all. The reverse case: restoring an older
  snapshot leaves files the restored database never heard of.
* `chunk` rows and `vectors/archive.<tag>.npy` for any model tag but the current one.
  `index.build` deletes only its own tag's chunks, so switching embedding models
  leaves the old ones behind as dead weight that search never reads.
* Free pages, with `VACUUM` -- but only when there is enough of them to be worth a
  full rewrite of the file, or when asked. A `wal_checkpoint(TRUNCATE)` runs first
  either way, which is what shrinks a `-wal` that has grown past the database.

**Reported, never touched:** sessions whose `raw_path` no longer exists (the archive
is now the only copy -- that is the point), `drop_file` rows whose file is gone (the
row is the memory that stops the same export being re-sniffed), and the hand-made
`archive.db.bak-*` copies beside the archive, with their sizes -- 370 MB here -- since
`llma snapshot` is what those were reaching for and deleting them is a person's call.

**Order of operations.** Rows go first, in one transaction; files go after the
commit. A crash between the two leaves a stray file the next run finds, which is
harmless. The other order leaves a row pointing at nothing, which the viewer renders
as a broken blob. The whole run holds the sync lock, so it cannot race the nightly
ingest that would be writing new blobs while this deletes old ones. `VACUUM` needs
roughly the database's size free on disk and is skipped, with a note, when it is not.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import snapshot
from .blobs import SHA256_RE, blob_path
from .sync import LOCK_NAME, Busy, Lock  # noqa: F401 - Busy is this module's too

# VACUUM rewrites the whole file; below this much reclaimable space it is not worth
# the minutes and the doubled disk, unless asked with --vacuum.
VACUUM_WORTH_BYTES = 16 << 20
VACUUM_WORTH_FRACTION = 0.10


@dataclass
class Item:
    key: str
    label: str
    count: int = 0
    bytes: int = 0
    reclaim: bool = True          # False: reported only
    note: str = ""
    # what would go, so a surprising count can be explained
    details: list[str] = field(default_factory=list)


@dataclass
class Report:
    items: list[Item]
    applied: bool = False
    vacuumed: bool = False
    reclaimed_bytes: int = 0
    notes: list[str] = field(default_factory=list)
    db_bytes: int = 0
    wal_bytes: int = 0

    def item(self, key: str) -> Item:
        return next(i for i in self.items if i.key == key)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(i.bytes for i in self.items if i.reclaim)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _orphan_blobs(con: sqlite3.Connection, blob_dir: Path) -> tuple[Item, list[tuple[int, Path | None]]]:
    # NOT IN over two materialised sets, not a correlated NOT EXISTS per blob: there is
    # no index on either column, and the correlated form scanned `part` twice for each
    # of 190 blobs -- five seconds, against a tenth of one this way.
    rows = con.execute("""
        SELECT b.id, b.sha256, b.bytes, b.path FROM blob b
         WHERE b.id NOT IN (SELECT blob_id FROM part WHERE blob_id IS NOT NULL)
           AND b.id NOT IN (SELECT tool_input_blob_id FROM part
                             WHERE tool_input_blob_id IS NOT NULL)
         ORDER BY b.id""").fetchall()
    victims = [(r["id"], blob_path(blob_dir, r["sha256"], r["path"])) for r in rows]
    item = Item("orphan_blobs", "blob rows nothing points at", len(rows),
                sum(r["bytes"] or 0 for r in rows),
                details=[r["sha256"][:12] for r in rows[:20]])
    return item, victims


def _stray_files(con: sqlite3.Connection, blob_dir: Path) -> tuple[Item, list[Path]]:
    known = {r[0] for r in con.execute("SELECT sha256 FROM blob")}
    strays: list[Path] = []
    if blob_dir.is_dir():
        for shard in blob_dir.iterdir():
            if not shard.is_dir():
                continue
            for f in shard.iterdir():
                if f.is_file() and SHA256_RE.match(f.name) and f.name not in known:
                    strays.append(f)
    item = Item("stray_files", "blob files with no row", len(strays),
                sum(_size(f) for f in strays),
                details=[f.name[:12] for f in strays[:20]])
    return item, strays


def _stale_vectors(con: sqlite3.Connection, vectors_dir: Path, model_tag: str
                   ) -> tuple[Item, list[str], list[Path]]:
    tags = [r[0] for r in con.execute(
        "SELECT model_tag FROM chunk WHERE model_tag != ? GROUP BY model_tag", (model_tag,))]
    rows = con.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM chunk "
                       "WHERE model_tag != ?", (model_tag,)).fetchone()
    files = []
    if vectors_dir.is_dir():
        keep = f"archive.{model_tag}.npy"
        files = sorted(f for f in vectors_dir.glob("archive.*.npy") if f.name != keep)
    item = Item("stale_vectors", "chunks and vectors of a previous model",
                rows[0] + len(files), (rows[1] or 0) + sum(_size(f) for f in files),
                details=[*tags, *(f.name for f in files)])
    return item, tags, files


def _free_pages(con: sqlite3.Connection) -> Item:
    page = con.execute("PRAGMA page_size").fetchone()[0]
    free = con.execute("PRAGMA freelist_count").fetchone()[0]
    return Item("free_pages", "free pages inside the file (VACUUM)", free, free * page)


def _missing_raw(con: sqlite3.Connection) -> Item:
    gone = [r[0] for r in con.execute("SELECT raw_path FROM session")
            if r[0] and not os.path.exists(r[0])]
    return Item("missing_raw", "sessions whose raw file is gone", len(gone), 0,
                reclaim=False, note="the archive is now the only copy; kept",
                details=gone[:20])


def _missing_drops(con: sqlite3.Connection) -> Item:
    try:
        gone = [r[0] for r in con.execute("SELECT path FROM drop_file")
                if r[0] and not os.path.exists(r[0])]
    except sqlite3.OperationalError:
        gone = []
    return Item("missing_drops", "drop files gone from disk", len(gone), 0,
                reclaim=False, note="the row is what stops a re-sniff; kept",
                details=gone[:20])


def _legacy(db_path: Path) -> Item:
    found = snapshot.legacy_backups(db_path)
    return Item("legacy_backups", "hand-made archive.db.bak-* copies", len(found),
                sum(b for _, b in found), reclaim=False,
                note="use `llma snapshot`; delete these by hand when ready",
                details=[p.name for p, _ in found])


def plan(con: sqlite3.Connection, db_path: Path, blob_dir: Path, vectors_dir: Path,
         model_tag: str) -> Report:
    """What `apply` would do. Read-only."""
    orphans, _ = _orphan_blobs(con, blob_dir)
    strays, _ = _stray_files(con, blob_dir)
    stale, _, _ = _stale_vectors(con, vectors_dir, model_tag)
    report = Report(items=[orphans, strays, stale, _free_pages(con),
                           _missing_raw(con), _missing_drops(con), _legacy(db_path)],
                    db_bytes=_size(db_path), wal_bytes=_size(_wal(db_path)))
    return report


def _wal(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + "-wal")


def vacuum_worth_it(report: Report) -> bool:
    free = report.item("free_pages").bytes
    return free >= VACUUM_WORTH_BYTES or (report.db_bytes and
                                          free >= report.db_bytes * VACUUM_WORTH_FRACTION)


def apply(con: sqlite3.Connection, db_path: Path, blob_dir: Path, vectors_dir: Path,
          model_tag: str, *, vacuum: bool | None = None) -> Report:
    """Do it. `vacuum=None` means "when worth it"; True forces, False never.

    Holds the sync lock for the whole run. Rows first, in one transaction; files
    after the commit; checkpoint; then VACUUM, which cannot run inside a transaction
    and is the one step that needs the disk space check.
    """
    with Lock(db_path.parent / LOCK_NAME):
        report = plan(con, db_path, blob_dir, vectors_dir, model_tag)
        report.applied = True

        _, victims = _orphan_blobs(con, blob_dir)
        _, stray_files = _stray_files(con, blob_dir)
        _, tags, npy_files = _stale_vectors(con, vectors_dir, model_tag)

        # ---- rows, one transaction
        if victims:
            con.executemany("DELETE FROM blob WHERE id = ?", [(bid,) for bid, _ in victims])
        if tags:
            con.execute("DELETE FROM chunk WHERE model_tag != ?", (model_tag,))
        con.commit()

        # ---- files, after the commit
        freed = 0
        for _, path in victims:
            freed += _unlink(path)
        for path in stray_files:
            freed += _unlink(path)
        for path in npy_files:
            freed += _unlink(path)
        # a shard folder emptied by the above is noise in a listing
        if blob_dir.is_dir():
            for shard in blob_dir.iterdir():
                if shard.is_dir() and not any(shard.iterdir()):
                    try:
                        shard.rmdir()
                    except OSError:
                        pass
        # what actually left the disk: an orphan row whose file was already gone
        # reclaims a row, not bytes
        report.reclaimed_bytes = freed

        # ---- the file itself
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError as exc:
            report.notes.append(f"wal checkpoint skipped: {exc}")

        want = vacuum if vacuum is not None else vacuum_worth_it(report)
        if want:
            need = report.db_bytes + report.wal_bytes
            free_disk = shutil.disk_usage(db_path.parent).free
            if free_disk < need * 1.1:
                report.notes.append(
                    f"VACUUM skipped: needs ~{need / 1e6:.0f} MB free beside the "
                    f"archive, {free_disk / 1e6:.0f} MB available")
            else:
                before = _size(db_path)
                con.execute("VACUUM")
                report.vacuumed = True
                report.reclaimed_bytes += max(0, before - _size(db_path))
        elif vacuum is None and report.item("free_pages").bytes:
            report.notes.append("VACUUM not run: too little to reclaim for a full "
                                "rewrite; force it with --vacuum")
        report.db_bytes = _size(db_path)
        report.wal_bytes = _size(_wal(db_path))
    return report


def _unlink(path: Path | None) -> int:
    if path is None:
        return 0
    size = _size(path)
    try:
        path.unlink()
    except OSError:
        return 0
    return size


# ------------------------------------------------------------------ rendering

def fmt_bytes(n: int) -> str:
    """Decimal units, as `llma stats` and the snapshot listing print them."""
    if n < 1000:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1000
        if n < 1000 or unit == "GB":
            return f"{n:,.1f} {unit}"
    return f"{n:,.1f} GB"


def render(report: Report) -> str:
    head = ("GC  applied" if report.applied
            else "GC  dry run -- nothing changed; add --apply to reclaim")
    out = [head, f"  archive {fmt_bytes(report.db_bytes)}"
                 + (f" + wal {fmt_bytes(report.wal_bytes)}" if report.wal_bytes else "")]
    out.append("")
    for it in report.items:
        kind = "reclaim" if it.reclaim else "report "
        size = fmt_bytes(it.bytes) if it.bytes else ""
        line = f"  {kind}  {it.label:<42} {it.count:>6}  {size:>10}"
        if it.note and it.count:
            line += f"   {it.note}"
        out.append(line)
        if it.count and it.details and it.key in ("stale_vectors", "legacy_backups"):
            out.append(f"           {', '.join(it.details)}")
    out.append("")
    if report.applied:
        out.append(f"  reclaimed {fmt_bytes(report.reclaimed_bytes)}"
                   + (" (vacuumed)" if report.vacuumed else ""))
    else:
        out.append(f"  reclaimable {fmt_bytes(report.reclaimable_bytes)}")
    for note in report.notes:
        out.append(f"  note: {note}")
    return "\n".join(out) + "\n"


def as_dict(report: Report) -> dict:
    return {
        "applied": report.applied, "vacuumed": report.vacuumed,
        "db_bytes": report.db_bytes, "wal_bytes": report.wal_bytes,
        "reclaimable_bytes": report.reclaimable_bytes,
        "reclaimed_bytes": report.reclaimed_bytes,
        "items": [{"key": i.key, "label": i.label, "count": i.count, "bytes": i.bytes,
                   "reclaim": i.reclaim, "note": i.note, "details": i.details}
                  for i in report.items],
        "notes": report.notes,
    }
