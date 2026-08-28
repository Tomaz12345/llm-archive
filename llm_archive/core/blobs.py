"""Content-addressed store for oversized part content.

Two things land here:

  * parts whose inline text exceeds INLINE_LIMIT;
  * files Claude Code already externalised itself, via the `<persisted-output>` marker
    it writes into oversized tool_result blocks (docs/phase0-findings.md §3d).

Addressing by sha256 means the same 350 KB job-board dump captured in three sessions
is stored once.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Magic-byte prefixes, longest-first where one is a prefix of another. Sniffing beats
# trusting a recorded type: nothing in the schema stores a mime, filenames are absent
# for half the sources, and the store deliberately writes blobs without an extension.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"%PDF-", "application/pdf"),
)

IMAGE_MIMES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp",
    "image/avif", "image/heic", "image/tiff",
})


def sniff_mime(head: bytes) -> str:
    """Content type from the first bytes of a blob, `application/octet-stream` if unsure.

    RIFF and ISO-BMFF containers carry their real type at a fixed offset rather than at
    byte 0, which is why WEBP/AVIF/HEIC cannot be matched by a plain prefix table.
    """
    for prefix, mime in _MAGIC:
        if head.startswith(prefix):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1"):
            return "image/heic"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "application/octet-stream"


def sniff_file(path: Path) -> str:
    """`sniff_mime` on a file's header. Unreadable files are octet-stream, not an error."""
    try:
        with open(path, "rb") as fh:
            return sniff_mime(fh.read(16))
    except OSError:
        return "application/octet-stream"


def blob_path(root: Path, sha: str, recorded: str | None = None) -> Path | None:
    """Locate a blob by hash.

    `blob.path` holds an absolute path baked in at ingest time, so it is wrong the
    moment `data/` moves to another disk or another machine. The content-addressed
    layout is derivable, so derive first and fall back to what was recorded.
    """
    if not SHA256_RE.match(sha or ""):
        return None
    derived = root / sha[:2] / sha
    if derived.is_file():
        return derived
    if recorded:
        candidate = Path(recorded)
        if candidate.is_file():
            return candidate
    return None


class BlobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.written = 0
        self.bytes_written = 0
        self.deduped = 0

    def _dest(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def put_bytes(self, data: bytes) -> tuple[str, int, str]:
        """Store raw bytes. Returns (sha256, size, absolute path)."""
        sha = hashlib.sha256(data).hexdigest()
        dest = self._dest(sha)
        if dest.exists():
            self.deduped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            self.written += 1
            self.bytes_written += len(data)
        return sha, len(data), str(dest)

    def put_text(self, text: str) -> tuple[str, int, str]:
        return self.put_bytes(text.encode("utf-8", errors="replace"))

    def put_file(self, src: Path) -> tuple[str, int, str] | None:
        """Copy an existing file in. Returns None if it cannot be read."""
        try:
            return self.put_bytes(src.read_bytes())
        except OSError:
            return None
