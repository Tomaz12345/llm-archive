"""Pull image parts that are only a URL into the blob store.

One source records generated images as a CDN reference and nothing else: T3 Chat's
`image_generation` result is a filename plus an uploadthing link, so the archive holds
118 bytes where a 700 KB JPEG should be. Rendering that link in the page was the other
option and it is the wrong one — `web/app.py` is explicit that a page which fetches from
the network is both a leak (the CDN learns when you read your own archive) and a thing
that breaks offline.

So the network reach happens here instead: once, deliberately, on a command you type.
Nothing in `ingest` or `sync` calls this — an unattended nightly job must not start
talking to third-party CDNs on its own.

Only `blob_id` and `bytes` are written. `part.text` keeps the original reference, which
matters more than it looks: `part_fts` is an external-content index over `part.text`, so
rewriting it here would desynchronise search until the next full re-index.
"""

from __future__ import annotations

import re
import sqlite3
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


@dataclass
class FetchResult:
    candidates: int = 0
    fetched: int = 0
    bytes_fetched: int = 0
    skipped: int = 0                          # no URL in the reference
    failed: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"candidates": self.candidates, "fetched": self.fetched,
                "bytes_fetched": self.bytes_fetched, "skipped": self.skipped,
                "failed": len(self.failed)}


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
        limit: int | None = None,
        fetch: Callable[[str], bytes] = _download) -> FetchResult:
    """Download every un-stored image reference. Safe to re-run: stored parts are skipped.

    `fetch` is injectable so the tests never touch the network.
    """
    result = FetchResult()
    rows = candidates(con)
    result.candidates = len(rows)
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        url = url_in(row["text"])
        if not url:
            result.skipped += 1
            continue
        if dry_run:
            continue
        try:
            data = fetch(url)
        except Exception as exc:  # noqa: BLE001 - one dead link must not end the run
            result.failed.append((row["id"], f"{type(exc).__name__}: {exc}"))
            continue
        # A CDN that has forgotten the file answers with an HTML error page, and storing
        # that as the image would be worse than storing nothing.
        if sniff_mime(data[:16]) not in IMAGE_MIMES:
            result.failed.append((row["id"], "response is not an image"))
            continue

        sha, size, dest = blobs.put_bytes(data)
        bid = db.blob_id(con, sha, size, dest)
        con.execute("UPDATE part SET blob_id = ?, bytes = ? WHERE id = ?",
                    (bid, size, row["id"]))
        result.fetched += 1
        result.bytes_fetched += size

    if not dry_run:
        con.commit()
    return result


def default_store(data_dir: Path | None = None) -> tuple[Path, BlobStore]:
    from .ingest import default_paths

    db_path, blob_dir = default_paths(data_dir)
    return db_path, BlobStore(blob_dir)
