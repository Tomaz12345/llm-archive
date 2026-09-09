"""One unattended pass: ingest every source, then rebuild the search index.

This is what the scheduled task runs, and the two steps are one command on purpose.
`llma index` is not optional housekeeping after an ingest — ingesting a session deletes
and rewrites its parts, so part ids move, and both indexes are keyed on those ids
(see `metrics.index_health`). An automated ingest that did not re-index would leave
keyword search matching the wrong rows every night, silently.

**The lock.** A full embed pass is minutes long. The scheduler's `IgnoreNew` policy
stops the task overlapping *itself*, but nothing stops a nightly run landing on top of
a manual `llma index` — two writers renumbering `chunk` against the same vector file.
So a lock file beside the archive makes the second one stand down instead of racing.

Staleness of that lock is decided two ways, and it needs both. **Liveness** releases an
orphaned lock immediately — a run killed mid-embed never reaches `__exit__`, and this
happened on the very first real sync, leaving a lock that would otherwise have blocked
the archive for two hours. **Age** covers what liveness cannot: a holder that is alive
but hung, and a recycled pid that makes a dead holder look alive.

The liveness check is `OpenProcess(SYNCHRONIZE)` via ctypes, **never**
`os.kill(pid, 0)`: on Windows that call does not test liveness the way it does on
POSIX — it routes to `TerminateProcess`, so the "check" would kill the very run it was
asking about. The age limit matches the task's own `ExecutionTimeLimit`, past which the
scheduler has already given up on the previous run anyway.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

LOCK_NAME = ".sync.lock"
STALE_AFTER_S = 2 * 3600          # == schedule.TIME_LIMIT
LOG_MAX_BYTES = 1 << 20           # keep the tail, not a year of nightly runs


class Busy(RuntimeError):
    """Another sync (or a manual index) holds the lock."""


def _holder_alive(pid: int) -> bool:
    """Is the process that wrote the lock still running?

    Never `os.kill(pid, 0)` on Windows: that call does not test liveness there the way
    it does on POSIX — it routes to `TerminateProcess`, so the "check" would kill the
    very run it was asking about. `OpenProcess(SYNCHRONIZE)` is the read-only
    equivalent; a failure to open means the pid is gone.

    Pid reuse can make a dead holder look alive, which is why this is an *addition* to
    the age check rather than a replacement for it: liveness releases a lock early,
    age releases it eventually, and neither on its own covers both failures.
    """
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True       # exists, owned by someone else
        return True

    import ctypes
    SYNCHRONIZE = 0x00100000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


class Lock:
    """Exclusive, best-effort, and never blocking — a skipped nightly run is fine."""

    def __init__(self, path: Path):
        self.path = path
        self._held = False

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self._mtime()
            holder = self._pid()
            # A killed run never reaches __exit__, so its lock is orphaned. Waiting the
            # full STALE_AFTER_S for that is two hours of an archive that could have
            # been updating — and a run killed by the scheduler's own time limit, or by
            # a reboot mid-embed, is not rare enough to treat as impossible.
            if holder and not _holder_alive(holder):
                self.path.unlink(missing_ok=True)
            elif age < STALE_AFTER_S:
                raise Busy(f"another sync (pid {holder or '?'}) started "
                           f"{age/60:.0f} min ago ({self.path}); "
                           f"skipping this run") from None
            else:
                # Still alive, but past the scheduler's own time limit: hung, and
                # nothing is served by never running again.
                self.path.unlink(missing_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {int(time.time())}\n".encode())
        os.close(fd)
        self._held = True
        return self

    def __exit__(self, *exc) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def _mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def _pid(self) -> int | None:
        try:
            return int(self.path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None


@dataclass
class SyncResult:
    ingested: list = field(default_factory=list)     # (label, IngestResult)
    index: object | None = None
    seconds: float = 0.0
    skipped_vectors: bool = False
    # Bulk sources that need re-exporting. Carried on the result, not just logged, so
    # the caller can surface it — `llma sync` prints it, and a notifier could too.
    stale_exports: list = field(default_factory=list)

    @property
    def new(self) -> int:
        return sum(r.new for _, r in self.ingested)

    @property
    def updated(self) -> int:
        return sum(r.updated for _, r in self.ingested)

    @property
    def errors(self) -> dict:
        out: dict[str, int] = {}
        for _, res in self.ingested:
            for key, count in res.errors.items():
                out[f"{res.kind}:{key}"] = out.get(f"{res.kind}:{key}", 0) + count
        return out


def _heartbeat(log: Callable[[str], None], every_pct: int = 25
               ) -> Callable[[int, int], None]:
    """Progress callback that writes at most a handful of lines to the log."""
    state = {"last": -every_pct}

    def progress(done: int, total: int) -> None:
        if not total:
            return
        pct = int(done * 100 / total)
        if pct >= state["last"] + every_pct:
            state["last"] = pct - pct % every_pct
            log(f"    embedding {done}/{total} ({pct}%)")

    return progress


def _report_freshness(con, log: Callable[[str], None]) -> list[dict]:
    """Name the exports that have gone stale, and say where to get them.

    Written to the same log the run itself goes to, because that is the file you open
    when you wonder whether the archive is still working. A source needing action is
    listed with the instruction; everything current is one summary line, so a healthy
    run stays short enough to keep reading.
    """
    from . import freshness

    try:
        rows = freshness.report(con)
    except Exception as exc:                            # noqa: BLE001
        log(f"  freshness check failed — {type(exc).__name__}: {exc}")
        return []

    summary = freshness.summary(rows)
    needs = summary["needs_action"]
    soon = summary["due_soon"]

    for row in needs:
        age = (f"{row['export_age_days']:.0f}d ago" if row["export_age_days"]
               else "never")
        log(f"  EXPORT DUE  {row['label']}: {row['reason']} (last {age})")
        log(f"              {row['how']}")
        if row["retention"]:
            log(f"              {row['retention']}")
    for row in soon:
        log(f"  due soon    {row['label']}: {row['reason']}")

    # A deadline that runs whatever you do is worth saying on a source that is
    # otherwise perfectly current — it is the one warning that expires by itself.
    for row in rows:
        if row["retention"] and row not in needs and row["sessions"]:
            log(f"  retention   {row['label']}: {row['retention']}")

    if not needs and not soon:
        log(f"  exports current ({summary['by_state'].get('current', 0)} bulk "
            f"source(s) up to date)")
    return needs


def run(data_dir: Path | None = None, with_vectors: bool = True,
        log: Callable[[str], None] = print) -> SyncResult:
    """Ingest everything discoverable on this machine, then rebuild the indexes."""
    # `selection` is deliberately dependency-free, so importing it does not drag in
    # numpy or the ONNX model the way `index` does.
    from ..search import index as search_index
    from ..search import selection
    from . import db, idb, ingest
    from .blobs import BlobStore

    t0 = time.perf_counter()
    db_path, blob_dir = ingest.default_paths(data_dir)
    result = SyncResult(skipped_vectors=not with_vectors)

    with Lock(db_path.parent / LOCK_NAME):
        con = db.connect(db_path)
        blobs = BlobStore(blob_dir)

        for adapter in ingest.build_adapters(
                blobs, drops=ingest.drops_dir(data_dir),
                browser=idb.is_enabled(con)):
            # An adapter that throws must not take the other eleven with it: this runs
            # unattended, so the failure mode to avoid is one drifted format (§8.5)
            # stopping the archive updating at all, with only a log file to say why.
            try:
                res = ingest.run(adapter, con, blobs)
            except Exception as exc:                        # noqa: BLE001
                log(f"  {adapter.kind}: FAILED — {type(exc).__name__}: {exc}")
                continue
            result.ingested.append((adapter.label, res))
            if res.new or res.updated:
                log(f"  {adapter.kind}: +{res.new} new, {res.updated} updated, "
                    f"{res.skipped} unchanged")

        # "Nothing was ingested" is not the same as "the index is fine". A build killed
        # part-way — by the scheduler's time limit, a reboot, or a Ctrl-C — commits its
        # `DELETE FROM chunk` and dies before the re-insert, leaving an archive with an
        # empty vector index and no ingest to trigger a repair. That state then
        # survives every subsequent nightly run. Asking `selection` how much embeddable
        # text is currently unreachable catches it, and is the same measure `/stats`
        # reports staleness with, so the two can never disagree.
        changed = result.new + result.updated
        unindexed = selection.unindexed_count(con) if with_vectors else 0
        if not changed and unindexed:
            log(f"  nothing ingested, but {unindexed:,} embeddable part(s) are not "
                f"in the vector index — repairing")

        if changed or unindexed:
            # `index.build` is a full rebuild — it drops every chunk for the model tag
            # and re-embeds the corpus, however little changed. On this machine that is
            # ~20 minutes of CPU, during which a job with no console writes nothing at
            # all, and a log that goes silent for twenty minutes is indistinguishable
            # from a hang. Hence the heartbeat.
            log(f"  indexing ({changed} session(s) changed; full re-embed)...")
            result.index = search_index.build(con, db_path.parent / "vectors",
                                              blob_dir=db_path.parent / "blobs",
                                              with_vectors=with_vectors,
                                              progress=_heartbeat(log))
            log(f"  index: {result.index.fts_rows} keyword rows, "
                f"{result.index.chunks} chunks, {result.index.vectors} vectors")
        else:
            # Rebuilding an index nothing invalidated is minutes of CPU for an
            # identical result — on a laptop, every night.
            log("  nothing changed; index left alone")

        # The four local stores keep themselves current; the bulk web exports cannot,
        # and nothing was telling you so. `freshness` already knows which are overdue
        # and where each provider's export button is — but only if you run it, and the
        # whole point of a scheduled sync is that you have stopped running things.
        # Retention is reported even for a source that is not overdue: Gemini's My
        # Activity deletes on a rolling window and Grok's bundle sits behind a 30-day
        # TTL, so "you have three weeks left" is worth saying while it is still true.
        result.stale_exports = _report_freshness(con, log)

        con.close()

    result.seconds = time.perf_counter() - t0
    return result


def logger(path: Path | None) -> Callable[[str], None]:
    """A `log` callable that timestamps to `path` (and stdout when there is none).

    Truncates from the front past `LOG_MAX_BYTES`. A scheduled job's log is only read
    after something went wrong, so what matters is that the recent runs are there and
    that it never grows without bound.
    """
    if path is None:
        return print

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > LOG_MAX_BYTES:
            tail = path.read_bytes()[-LOG_MAX_BYTES // 2:]
            path.write_bytes(b"[...truncated...]\n" + tail)
    except OSError:
        pass

    def write(line: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with path.open("a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"{stamp}Z {line}\n")
        except OSError:
            pass

    return write
