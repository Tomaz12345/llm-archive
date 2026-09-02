"""Pull image parts that are only a URL into the blob store.

One source records generated images as a CDN reference and nothing else: T3 Chat's
`image_generation` result is a filename plus an uploadthing link, so the archive holds
118 bytes where a 700 KB JPEG should be. Rendering that link in the page was the other
option and it is the wrong one — `web/app.py` is explicit that a page which fetches from
the network is both a leak (the CDN learns when you read your own archive) and a thing
that breaks offline.

So the network reach happens here instead, never while rendering a page. `ingest` and
`sync` still do not call it — an unattended nightly job that has just pulled in new
sessions must not also start talking to third-party CDNs about them. `serve` does, via
`backfill_on_start`: these uploadthing links expire, and an archive that waits for you to
remember a command backfills them once they are already dead.

Only `blob_id` and `bytes` are written. `part.text` keeps the original reference, which
matters more than it looks: `part_fts` is an external-content index over `part.text`, so
rewriting it here would desynchronise search until the next full re-index.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import db
from .blobs import IMAGE_MIMES, BlobStore, sniff_mime

URL_RE = re.compile(r"https?://[^\s<>\"']+")

# uploadthing answers a default urllib User-Agent with 403 and a browser one with the
# file. Nothing here depends on being mistaken for a browser beyond getting served.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

MAX_BYTES = 32 * 1024 * 1024      # a generated image is ~1 MB; 32 is a runaway guard
TIMEOUT = 30

# Consecutive failures that mean "there is no network", not "one link has died". Only
# the unattended `serve` backfill uses it; a typed command works the whole list.
START_STOP_AFTER = 3


@dataclass
class FetchResult:
    candidates: int = 0
    fetched: int = 0
    bytes_fetched: int = 0
    skipped: int = 0                          # no URL in the reference
    failed: list[tuple[int, str]] = field(default_factory=list)
    aborted: bool = False                     # gave up early, candidates left untried

    def as_dict(self) -> dict:
        return {"candidates": self.candidates, "fetched": self.fetched,
                "bytes_fetched": self.bytes_fetched, "skipped": self.skipped,
                "failed": len(self.failed), "aborted": self.aborted}


def _download(url: str, timeout: int = TIMEOUT) -> bytes:
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"larger than {MAX_BYTES // (1024 * 1024)} MB")
    return data


def candidates(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Image parts with no bytes and a URL to get them from."""
    return con.execute("""
        SELECT p.id, p.text, m.session_id
        FROM part p JOIN message m ON m.id = p.message_id
        WHERE p.kind = 'image' AND p.blob_id IS NULL AND p.text LIKE '%http%'
        ORDER BY p.id""").fetchall()


def url_in(text: str | None) -> str | None:
    match = URL_RE.search(text or "")
    return match.group(0) if match else None


def run(con: sqlite3.Connection, blobs: BlobStore, *, dry_run: bool = False,
        limit: int | None = None, stop_after_failures: int | None = None,
        fetch: Callable[[str], bytes] = _download) -> FetchResult:
    """Download every un-stored image reference. Safe to re-run: stored parts are skipped.

    `stop_after_failures` gives up once that many attempts in a row have failed, leaving
    the rest for the next run. The `serve` backfill passes it because it starts at logon,
    when the network often is not up yet: without it a laptop spends TIMEOUT seconds per
    candidate discovering the same thing twenty-two times over. A typed
    `llma fetch-images` leaves it None and works the whole list — you are watching, and
    one dead CDN link is not a reason to skip the twenty live ones behind it.

    `fetch` is injectable so the tests never touch the network.
    """
    result = FetchResult()
    rows = candidates(con)
    result.candidates = len(rows)
    if limit is not None:
        rows = rows[:limit]

    consecutive = 0
    for row in rows:
        url = url_in(row["text"])
        if not url:
            result.skipped += 1
            continue
        if dry_run:
            continue

        why: str | None = None
        try:
            data = fetch(url)
        except Exception as exc:  # noqa: BLE001 - one dead link must not end the run
            why = f"{type(exc).__name__}: {exc}"
        else:
            # A CDN that has forgotten the file answers with an HTML error page, and
            # storing that as the image would be worse than storing nothing.
            if sniff_mime(data[:16]) not in IMAGE_MIMES:
                why = "response is not an image"

        if why is not None:
            result.failed.append((row["id"], why))
            consecutive += 1
            if stop_after_failures and consecutive >= stop_after_failures:
                result.aborted = True
                break
            continue

        sha, size, dest = blobs.put_bytes(data)
        bid = db.blob_id(con, sha, size, dest)
        con.execute("UPDATE part SET blob_id = ?, bytes = ? WHERE id = ?",
                    (bid, size, row["id"]))
        result.fetched += 1
        result.bytes_fetched += size
        consecutive = 0

    if not dry_run:
        con.commit()
    return result


def _echo(line: str) -> None:
    """Flushed, because these lines are written from a background thread and the
    interesting case is reading them after the server was killed rather than shut down.
    Redirected to a file (`tools/serve_windowless.py` does exactly that), a bare `print`
    is block-buffered and a `Stop-Process` takes the report with it."""
    print(line, flush=True)


def pending(db_path: Path) -> int:
    """How many parts a backfill would try. Plain `sqlite3.connect`, not `db.connect`:
    this asks a question of an existing archive and has no business running the DDL
    script and the migration ladder to do it."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return len(candidates(con))
    finally:
        con.close()


def backfill_on_start(db_path: Path, blob_dir: Path, *,
                      echo: Callable[[str], None] = _echo,
                      fetch: Callable[[str], bytes] = _download) -> threading.Thread | None:
    """Backfill URL-only images in the background, for `serve` to call as it starts.

    Returns the running thread, or None when there is nothing pending — the usual case,
    and it costs one indexed query and no network whatsoever. That short-circuit is the
    whole reason this is tolerable on every start: an archive with no T3 Chat images in
    it never opens a socket.

    A thread rather than the lifespan body: this reaches a third-party CDN, over links
    that may be dead, on a machine whose network may not be up yet, and none of that is
    a reason for the port to stay unbound while it finds out. Daemon, so a Ctrl-C during
    a slow fetch still exits at once — a half-finished backfill loses nothing, because
    the next start picks up exactly the parts that still have no bytes.
    """
    try:
        n = pending(db_path)
    except sqlite3.Error as exc:
        echo(f"  image backfill skipped: {type(exc).__name__}: {exc}")
        return None
    if not n:
        return None

    def work() -> None:
        # Its own connection: sqlite3 objects belong to the thread that made them, and
        # this one writes while the request handlers are reading. WAL (set in db.py's
        # schema) is what makes that a concurrent reader/writer and not a locked file.
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            result = run(con, BlobStore(blob_dir), fetch=fetch,
                         stop_after_failures=START_STOP_AFTER)
        except Exception as exc:  # noqa: BLE001 - a backfill must never take serve down
            echo(f"  image backfill failed: {type(exc).__name__}: {exc}")
            return
        finally:
            con.close()

        if result.fetched:
            echo(f"  image backfill: {result.fetched} of {n} fetched "
                 f"({result.bytes_fetched / 1024 / 1024:.1f} MB) — reload to see them")
        if result.aborted:
            echo(f"  image backfill: stopped after {START_STOP_AFTER} failures in a row; "
                 f"run `llma fetch-images` once the network is back")
        elif result.failed:
            echo(f"  image backfill: {len(result.failed)} could not be fetched")

    echo(f"  fetching {n} image(s) held as a URL only, in the background…")
    thread = threading.Thread(target=work, name="fetch-images", daemon=True)
    thread.start()
    return thread


def default_store(data_dir: Path | None = None) -> tuple[Path, BlobStore]:
    from .ingest import default_paths

    db_path, blob_dir = default_paths(data_dir)
    return db_path, BlobStore(blob_dir)
