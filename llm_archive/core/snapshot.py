"""Point-in-time copies of the archive, taken the one way that is safe under WAL.

`data/` held five hand-made copies when this was written -- `archive.db.bak-prefacts`,
`bak-preimages`, `bak-preintake`, `bak-pretoolname`, `bak-preturns` -- one before each
schema migration, 370 MB between them, made with a file copy. They were the right
instinct and the wrong mechanism: the archive runs in WAL mode, so a copy of
`archive.db` alone can miss every write still sitting in `archive.db-wal`, and a copy
taken while an ingest is mid-transaction is a database that opens and is wrong. The
sqlite3 online backup API is the correct tool -- it reads pages under the database's
own locking, folds in the WAL, and produces a file that is a consistent snapshot even
while the source is being written. That is what `take()` uses, and what `restore()`
uses in the other direction, for the same reason: writing pages into the live file
through the backup API replaces its content without leaving a stale `-wal` beside it.

Snapshots live in `<data>/snapshots/` and are named for when they were taken, with an
optional label: `archive-20260917-0303-schema-v11.db`. The name is the metadata; there
is no ledger to fall out of date.

**The one that takes itself.** `db.connect()` calls `take()` before running any
migration, labelled with the schema version the archive had. That is precisely what
the five `.bak-*` files were for, done by hand each time and forgotten once; a
migration that goes wrong halfway is the one moment a copy from a minute earlier is
worth everything. `prune --keep N` is how they stop accumulating.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SNAPSHOT_DIR = "snapshots"
_NAME = re.compile(r"^archive-(\d{8}-\d{4})(?:-([A-Za-z0-9][A-Za-z0-9_.-]*))?\.db$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# The hand-made copies this module replaces. Reported by `llma gc`, never deleted by
# it: they are not the archive's to reclaim, and 370 MB is a decision for a person.
LEGACY_GLOB = "archive.db.bak-*"


class SnapshotError(RuntimeError):
    """A bad label, a name that is not a snapshot, or a restore that cannot proceed."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    taken_at: datetime
    label: str | None
    bytes: int

    @property
    def name(self) -> str:
        return self.path.name


def snapshots_dir(db_path: Path) -> Path:
    return db_path.parent / SNAPSHOT_DIR


def _parse(path: Path) -> Snapshot | None:
    m = _NAME.match(path.name)
    if not m:
        return None
    try:
        taken = datetime.strptime(m.group(1), "%Y%m%d-%H%M")
        size = path.stat().st_size
    except (ValueError, OSError):
        return None
    return Snapshot(path=path, taken_at=taken, label=m.group(2), bytes=size)


def list_snapshots(db_path: Path) -> list[Snapshot]:
    """Newest first. Files in the folder that do not fit the name are not snapshots."""
    folder = snapshots_dir(db_path)
    if not folder.is_dir():
        return []
    found = [s for p in folder.iterdir() if p.is_file() and (s := _parse(p)) is not None]
    # two in one minute share a stamp; the file's own clock says which came second
    return sorted(found, key=lambda s: (s.taken_at, _mtime(s.path)), reverse=True)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _backup(src: sqlite3.Connection, dest_path: Path) -> None:
    """Copy `src` into a fresh file at `dest_path` through the backup API."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(dest_path)
    try:
        src.backup(dest)
    finally:
        dest.close()


def take(db_path: Path, label: str | None = None, *, con: sqlite3.Connection | None = None,
         now: float | None = None) -> Snapshot:
    """Snapshot the archive at `db_path`. Pass `con` to reuse an open connection --
    `db.connect()` does, from inside the migration path, before the ladder runs.

    Two snapshots within one minute would collide on the name; the second gets a
    `-2` suffix rather than overwriting the first, because a snapshot that replaces
    another is not a snapshot.
    """
    if label is not None and not _LABEL.match(label):
        raise SnapshotError(f"label must be letters, digits, '.', '_' or '-'; got {label!r}")
    if not db_path.exists():
        raise SnapshotError(f"no archive at {db_path}")
    stamp = datetime.fromtimestamp(now if now is not None else time.time())
    base = f"archive-{stamp.strftime('%Y%m%d-%H%M')}" + (f"-{label}" if label else "")
    folder = snapshots_dir(db_path)
    dest = folder / f"{base}.db"
    n = 2
    while dest.exists():
        dest = folder / f"{base}-{n}.db"
        n += 1

    own = con is None
    src = con if con is not None else sqlite3.connect(db_path)
    try:
        _backup(src, dest)
    finally:
        if own:
            src.close()
    return _parse(dest)


def find(db_path: Path, name: str) -> Snapshot:
    """A snapshot by file name, or by the name without `.db`."""
    want = name if name.endswith(".db") else f"{name}.db"
    for snap in list_snapshots(db_path):
        if snap.name == want:
            return snap
    raise SnapshotError(f"no snapshot named {name!r} in {snapshots_dir(db_path)}")


def prune(db_path: Path, keep: int) -> list[Snapshot]:
    """Delete all but the newest `keep`. Returns what went."""
    if keep < 0:
        raise SnapshotError("--keep wants a count of snapshots to leave, 0 or more")
    gone = []
    for snap in list_snapshots(db_path)[keep:]:
        try:
            snap.path.unlink()
        except OSError:
            continue
        gone.append(snap)
    return gone


def restore(db_path: Path, name: str) -> tuple[Snapshot, Snapshot]:
    """Replace the archive with a snapshot. Returns (restored, safety copy).

    The current archive is snapshotted first, labelled `pre-restore`, so a restore
    is itself undoable. The write goes through the backup API into the live file,
    which is what keeps a `-wal` from a previous session from being replayed over
    the restored pages on the next open.
    """
    snap = find(db_path, name)
    safety = take(db_path, "pre-restore")
    src = sqlite3.connect(snap.path)
    dest = sqlite3.connect(db_path)
    try:
        src.backup(dest)
    finally:
        dest.close()
        src.close()
    return snap, safety


def legacy_backups(db_path: Path) -> list[tuple[Path, int]]:
    """The hand-made `archive.db.bak-*` copies beside the archive, with sizes."""
    out = []
    for p in sorted(db_path.parent.glob(LEGACY_GLOB)):
        try:
            out.append((p, p.stat().st_size))
        except OSError:
            continue
    return out
