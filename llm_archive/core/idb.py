"""Reading a browser's own IndexedDB store, for sources that have no export.

§2.1's unfinished business. OpenRouter is local-first: there is no bulk export, so
capturing 50 chats means 50 manual clicks on 50 "Export Chat" menus, and the documented
mitigation is to read the store the site already keeps on your disk.

`tools/probe_openrouter_idb.py` was written to decide whether that was worth doing, and
it looked in the wrong place — only the Chromium family. **The data here is in Firefox**,
which is a completely different and much more tractable format:

    <profile>/storage/default/https+++openrouter.ai/idb/<mangled>.sqlite

That is a plain SQLite database. Chromium's equivalent is a LevelDB SSTable holding V8
structured-clone blobs, which needs two parsers that are not in the standard library;
Firefox needs `sqlite3` (which is) plus the two decoders below, both small and both
fully determined by the bytes.

**Two layers to get through.**

1. `object_data.data` is raw Snappy — a length varint then literal/back-reference tags.
   No framing, no checksums. ~40 lines, and it self-checks: the varint states the
   uncompressed length up front, so a wrong decode is caught rather than guessed at.

2. The result is a Firefox structured clone: 8-byte little-endian words of
   `(uint32 data, uint32 tag)`, strings written inline and padded to the next 8-byte
   boundary. Objects are `OBJECT_OBJECT`, then alternating key/value, then
   `END_OF_KEYS`. Arrays are the same with integer keys.

**Why the unknown-tag handling is the whole safety story.** This is an undocumented
on-disk format that Firefox may revise (§8.5). Every tag whose *width* this module does
not know is fatal to the record, never skipped: a tag skipped by the wrong number of
bytes desynchronises the stream and everything after it decodes as plausible garbage —
strings that are really floats, objects that are really arrays. A record that raises is
counted and dropped; a record that silently shifts would poison the archive. So
`decode()` raises `CloneError` and the caller reports it.

The store's own key prefix is the version check the probe's docstring said this format
lacked: every record is keyed `/v3:<type>:<id>` inside a store named
`openrouter:playground:v3`. A schema change bumps that, and the caller can refuse to
guess rather than mis-parse.

**This is your own data on your own disk** — but it lives in a browser profile, beside
session cookies and saved passwords. So, as in the probe: only paths whose directory
name is the site's own origin are ever opened, the store is copied before it is read
(a running browser holds a lock, and a half-written `-wal` is how you corrupt a
profile), and nothing else in the profile is touched.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

# ------------------------------------------------------------------ setting

SETTING = "browser_store"


def is_enabled(con: sqlite3.Connection) -> bool:
    """Persisted, the same way redaction is, and for the same reason.

    The nightly sync runs unattended, so a flag passed once on a command line would
    read one browser store and then never again. Off by default: this opens a browser
    profile, and that is a decision to make on purpose rather than inherit.
    """
    row = con.execute("SELECT value FROM meta WHERE key=?", (SETTING,)).fetchone()
    return bool(row) and str(row[0]) == "on"


def set_enabled(con: sqlite3.Connection, on: bool) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                (SETTING, "on" if on else "off"))
    con.commit()


# ---------------------------------------------------------------- discovery

FIREFOX_ROOTS = [
    r"%APPDATA%\Mozilla\Firefox\Profiles",
    r"%LOCALAPPDATA%\Mozilla\Firefox\Profiles",
]


def origin_dir(origin: str) -> str:
    """Firefox's on-disk name for an origin: `https://openrouter.ai` -> `https+++openrouter.ai`."""
    return origin.replace("://", "+++").replace("/", "+")


def stores(origin: str, roots: list[Path] | None = None) -> list[Path]:
    """Every IndexedDB SQLite file this machine holds for `origin`.

    Never a recursive walk of a browser profile: the path from the profile root to the
    store is fixed and fully known, so it is spelled out. A probe that goes looking
    through a profile for something interesting is a credential scanner whatever its
    author meant by it.
    """
    wanted = origin_dir(origin)
    found: list[Path] = []
    for root in (roots or [Path(os.path.expandvars(r)) for r in FIREFOX_ROOTS]):
        if not root.is_dir():
            continue
        for profile in sorted(root.iterdir()):
            idb = profile / "storage" / "default" / wanted / "idb"
            if not idb.is_dir():
                continue
            found.extend(sorted(p for p in idb.glob("*.sqlite")))
    return found


@contextmanager
def opened(path: Path):
    """A read-only connection to a *copy* of the store.

    Firefox holds a lock on the live file, and reading one mid-write is how a profile
    gets corrupted. The `-wal` and `-shm` siblings come along so the copy is consistent
    with whatever the browser had committed.
    """
    with tempfile.TemporaryDirectory(prefix="llma-idb-") as tmp:
        local = Path(tmp) / path.name
        shutil.copy2(path, local)
        for suffix in ("-wal", "-shm"):
            side = path.with_name(path.name + suffix)
            if side.exists():
                try:
                    shutil.copy2(side, local.with_name(local.name + suffix))
                except OSError:
                    pass
        con = sqlite3.connect(f"file:{local}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()


def store_names(con: sqlite3.Connection) -> dict[int, str]:
    try:
        return {r["id"]: r["name"] for r in con.execute(
            "SELECT id, name FROM object_store")}
    except sqlite3.DatabaseError:
        return {}


def decode_key(raw: bytes) -> str:
    """Firefox's IndexedDB key encoding for a string: every ASCII byte shifted up by one.

    `dibsbdufs` is `character`. The type prefix byte and any trailing structure are not
    decoded — callers match on the readable middle, not on the exact bytes.
    """
    return "".join(chr(b - 1) if 32 < b < 128 else "�" for b in raw)


def records(con: sqlite3.Connection, store_id: int | None = None):
    """(key, value) for every record, values already decompressed and decoded.

    A record that will not decode is yielded as `(key, CloneError)` rather than
    dropped silently — see the module docstring on why a bad record must be visible.
    """
    sql = "SELECT key, data FROM object_data"
    args: tuple = ()
    if store_id is not None:
        sql += " WHERE object_store_id=?"
        args = (store_id,)
    for row in con.execute(sql, args):
        key = decode_key(row["key"])
        try:
            yield key, decode(snappy_decompress(row["data"]))
        except (CloneError, IndexError, struct.error, UnicodeDecodeError) as exc:
            yield key, CloneError(str(exc))


# ---------------------------------------------------------------- snappy


class SnappyError(ValueError):
    pass


def snappy_decompress(blob: bytes) -> bytes:
    """Raw Snappy (no stream framing), which is what Firefox writes.

    Self-checking: the leading varint states the uncompressed length, so a decode that
    lands anywhere else is raised rather than returned.
    """
    size, shift, i = 0, 0, 0
    try:
        while True:
            byte = blob[i]
            i += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
            if shift > 35:
                raise SnappyError("length varint never terminated")
    except IndexError:
        raise SnappyError("truncated before the length varint ended") from None

    out = bytearray()
    n = len(blob)
    while i < n:
        tag = blob[i]
        i += 1
        kind = tag & 0x03
        if kind == 0:                                   # literal
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(blob[i:i + extra], "little")
                i += extra
            length += 1
            chunk = blob[i:i + length]
            if len(chunk) != length:
                raise SnappyError("literal runs past the end of the blob")
            out += chunk
            i += length
            continue

        if kind == 1:                                   # 1-byte offset copy
            length = ((tag >> 2) & 0x07) + 4
            offset = ((tag >> 5) << 8) | blob[i]
            i += 1
        elif kind == 2:                                 # 2-byte offset copy
            length = (tag >> 2) + 1
            offset = int.from_bytes(blob[i:i + 2], "little")
            i += 2
        else:                                           # 4-byte offset copy
            length = (tag >> 2) + 1
            offset = int.from_bytes(blob[i:i + 4], "little")
            i += 4

        if offset == 0 or offset > len(out):
            raise SnappyError(f"back-reference of {offset} past {len(out)} bytes")
        # Overlapping copies are legal and common — they are how snappy encodes a run,
        # so this cannot be a slice copy.
        for _ in range(length):
            out.append(out[-offset])

    if len(out) != size:
        raise SnappyError(f"decoded {len(out)} bytes, header claimed {size}")
    return bytes(out)


# ------------------------------------------------------- structured clone


class CloneError(ValueError):
    """A structured clone this module will not guess at. Never skipped, always raised."""


# Firefox's SCTAG values. Only the ones a JSON-shaped record can contain are handled;
# everything else is deliberately absent so it raises rather than being skipped by a
# guessed width. See the module docstring.
TAG_HEADER = 0xFFF10000
TAG_NULL = 0xFFFF0000
TAG_UNDEFINED = 0xFFFF0001
TAG_BOOLEAN = 0xFFFF0002
TAG_INT32 = 0xFFFF0003
TAG_STRING = 0xFFFF0004
TAG_DATE = 0xFFFF0005
TAG_ARRAY = 0xFFFF0007
TAG_OBJECT = 0xFFFF0008
TAG_BOOLEAN_OBJECT = 0xFFFF000A
TAG_STRING_OBJECT = 0xFFFF000B
TAG_NUMBER_OBJECT = 0xFFFF000C
TAG_BACK_REFERENCE = 0xFFFF000D
TAG_END_OF_KEYS = 0xFFFF0013

# A double is not tagged: any word whose high half is below this is the raw IEEE754
# bits of a number. That is why doubles are read by looking at the tag range rather
# than by matching a constant.
TAG_FLOAT_MAX = 0xFFF00000

# Latin-1 rather than UTF-16 is a flag in the string's length word.
STRING_LATIN1 = 0x80000000
STRING_LENGTH = 0x7FFFFFFF

MAX_DEPTH = 64          # a cycle in a corrupt record must not become a stack overflow


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.at = 0
        # Every container in the order it was written. A `TAG_BACK_REFERENCE` is an
        # index into this, which is how Firefox stores an object that appears twice
        # in one record — OpenRouter's `character` records share a model descriptor
        # between fields, and without this every one of them fails to decode.
        # Containers are registered BEFORE they are filled, so a self-reference
        # resolves to the same (still-growing) object rather than recursing.
        self.seen: list = []

    def word(self) -> tuple[int, int]:
        if self.at + 8 > len(self.buf):
            raise CloneError("ran off the end of the record")
        data, tag = struct.unpack_from("<II", self.buf, self.at)
        self.at += 8
        return data, tag

    def boxed(self, value):
        """Record a JS object that decodes to a plain value, and return the value."""
        self.seen.append(value)
        return value

    def double(self) -> float:
        if self.at + 8 > len(self.buf):
            raise CloneError("ran off the end reading a double")
        value = struct.unpack_from("<d", self.buf, self.at)[0]
        self.at += 8
        return value

    def string(self, data: int) -> str:
        length = data & STRING_LENGTH
        width = length if data & STRING_LATIN1 else length * 2
        raw = self.buf[self.at:self.at + width]
        if len(raw) != width:
            raise CloneError("string runs past the end of the record")
        # Every value is 8-byte aligned; the padding after a string is not content.
        self.at += (width + 7) // 8 * 8
        return raw.decode("latin-1") if data & STRING_LATIN1 \
            else raw.decode("utf-16-le")


def decode(buf: bytes):
    """One structured clone -> plain Python. Raises CloneError on anything unfamiliar."""
    reader = _Reader(buf)
    data, tag = reader.word()
    if tag != TAG_HEADER:
        raise CloneError(f"not a structured clone (leading tag {tag:#010x})")
    return _value(reader, 0)


def _value(reader: _Reader, depth: int):
    if depth > MAX_DEPTH:
        raise CloneError("nested past the depth limit")
    data, tag = reader.word()
    return _decoded(reader, data, tag, depth)


def _decoded(reader: _Reader, data: int, tag: int, depth: int):
    if tag < TAG_FLOAT_MAX:
        # Not a tag at all: the word is the raw bit pattern of a double.
        reader.at -= 8
        return reader.double()

    if tag == TAG_NULL or tag == TAG_UNDEFINED:
        return None
    if tag == TAG_BOOLEAN:
        return bool(data)
    if tag == TAG_INT32:
        return struct.unpack("<i", struct.pack("<I", data))[0]
    if tag == TAG_STRING:
        return reader.string(data)

    # The boxed forms are JS *objects*, so each one takes a slot in the back-reference
    # numbering even though it decodes to a plain value here. Miss one and every later
    # back-reference in the record is off by that many — which is exactly how three of
    # OpenRouter's `character` records failed with "index 49, only 49 seen".
    if tag == TAG_BOOLEAN_OBJECT:
        return reader.boxed(bool(data))
    if tag == TAG_STRING_OBJECT:
        return reader.boxed(reader.string(data))
    if tag == TAG_NUMBER_OBJECT:
        return reader.boxed(reader.double())
    if tag == TAG_DATE:
        # Epoch milliseconds as a double, which is exactly what the archive stores.
        return reader.boxed(int(reader.double()))
    if tag == TAG_OBJECT:
        return _object(reader, depth)
    if tag == TAG_ARRAY:
        return _array(reader, depth)
    if tag == TAG_BACK_REFERENCE:
        if data >= len(reader.seen):
            raise CloneError(
                f"back-reference to object {data}, only {len(reader.seen)} seen")
        return reader.seen[data]

    raise CloneError(f"unknown tag {tag:#010x} — refusing to guess its width")


def _object(reader: _Reader, depth: int) -> dict:
    out: dict = {}
    reader.seen.append(out)
    while True:
        data, tag = reader.word()
        if tag == TAG_END_OF_KEYS:
            return out
        if tag == TAG_STRING:
            key = reader.string(data)
        elif tag == TAG_INT32:
            key = str(struct.unpack("<i", struct.pack("<I", data))[0])
        else:
            raise CloneError(f"object key has tag {tag:#010x}")
        out[key] = _value(reader, depth + 1)


def _array(reader: _Reader, depth: int) -> list:
    """Arrays are objects with integer keys, and may be sparse.

    The length in the tag word is trusted only as a size hint — the keys decide what
    goes where, so a truncated array reads short instead of shifting its contents.
    """
    holes: dict[int, object] = {}
    # Registered as a list so a back-reference into it is the same object; the entries
    # are filled in below and the identity is preserved by `[:] =` rather than rebinding.
    out: list = []
    reader.seen.append(out)
    while True:
        data, tag = reader.word()
        if tag == TAG_END_OF_KEYS:
            break
        if tag == TAG_INT32:
            index = struct.unpack("<i", struct.pack("<I", data))[0]
        elif tag == TAG_STRING:
            key = reader.string(data)
            if not key.isdigit():
                raise CloneError(f"array key {key!r} is not an index")
            index = int(key)
        else:
            raise CloneError(f"array key has tag {tag:#010x}")
        holes[index] = _value(reader, depth + 1)

    if holes:
        out[:] = [holes.get(i) for i in range(max(holes) + 1)]
    return out
