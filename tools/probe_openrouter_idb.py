"""Is OpenRouter's chat history actually in your browser profile, and is it readable?

**ANSWERED, and this probe looked in the wrong place.** It searched only the Chromium
family and reported "not available here" — on this machine the store is in **Firefox**,
which is a different and far more tractable format: a plain SQLite file holding
Snappy-compressed structured clones, rather than Chromium's LevelDB SSTable of V8
blobs. Firefox is now searched too (below), and the reader that came out of it lives in
`llm_archive/core/idb.py` behind `llma browser --enable`.

The measurement that settled it, on this account: **3 rooms, 6 messages, 26 records,
all decodable** — which is exactly what had already been exported by hand, so the
reader adds nothing retroactively and everything from here on. Re-run this whenever you
want that count again.

§2.1's unfinished business. OpenRouter has no bulk export — 50 chats means 50 manual
clicks — and the documented mitigation is to read your own browser storage, which is
your disk and not their site. Before writing that reader, one question has to be
answered with a measurement rather than a shrug: **is the text there in a form worth
parsing, and how much of it?**

This probe answers exactly that and nothing more. It is deliberately read-only,
deliberately dependency-free, and deliberately *not* a parser:

* it finds `https_openrouter.ai_0.indexeddb.leveldb` under the Chromium-family
  profiles on this machine;
* it copies the store to a temp directory before reading a byte of it, because a
  running browser holds a lock on the live one and a half-written `.log` is exactly
  how you get a corrupt profile;
* it scans the copy for printable runs and reports what shape they are in — how many
  records look like chat messages, how much text, newest mtime.

**Why this is a probe and not the adapter.** A Chromium IndexedDB value is not JSON.
It is a V8 structured-clone blob inside a LevelDB SSTable, so a real reader needs both
an SSTable reader and a structured-clone decoder — neither of which is in the standard
library, and both of which are a bad thing to write against an undocumented on-disk
format that Chrome revises without notice (§8.5, with none of the mitigations: there is
no `version` field to key off here). Whether that is worth building depends entirely on
what this prints. If it reports three chats you have already exported by hand, the
answer is no.

**This is your own data on your own disk.** It is also a browser profile, which is
where session cookies and saved passwords live, so:

* the probe only ever opens paths whose directory name contains `openrouter`;
* it prints counts, sizes and dates — never record contents;
* nothing it reads goes into the archive. Ingestion would be a separate, later step.

Run it yourself, with the browser closed:

    python tools/probe_openrouter_idb.py
    python tools/probe_openrouter_idb.py --profiles-root "D:/some/User Data"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STORE_DIR = "https_openrouter.ai_0.indexeddb.leveldb"

# Only these are searched, and only for a directory whose name already contains
# "openrouter". A probe that walks a whole browser profile looking for something
# interesting is a credential scanner, whatever its author meant by it.
CHROMIUM_ROOTS = [
    r"%LOCALAPPDATA%\Google\Chrome\User Data",
    r"%LOCALAPPDATA%\Google\Chrome Beta\User Data",
    r"%LOCALAPPDATA%\Microsoft\Edge\User Data",
    r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data",
    r"%LOCALAPPDATA%\Vivaldi\User Data",
    r"%APPDATA%\Opera Software\Opera Stable",
]

PRINTABLE = re.compile(rb"[\x20-\x7e\xc2-\xf4][\x20-\x7e\x80-\xbf]{15,}")

# Markers from the per-chat export schema (§2.1, `orpg.3.0`). If the browser store is
# the same object graph, these are what it is made of — and if none of them appear,
# the store is not worth an SSTable reader.
MARKERS = [b"parentMessageId", b"variantSlug", b"generationId", b"characters",
           b"orpg", b"tokensCount", b"routerMetadata", b"createdAt"]


def candidate_stores(extra_root: Path | None = None) -> list[Path]:
    roots = [Path(os.path.expandvars(r)) for r in CHROMIUM_ROOTS]
    if extra_root is not None:
        roots.insert(0, extra_root)

    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        # Profiles are `Default`, `Profile 1`, ... — one level down, then a fixed
        # subpath. Never a recursive walk of the profile.
        for profile in sorted(root.iterdir()):
            if not profile.is_dir():
                continue
            store = profile / "IndexedDB" / STORE_DIR
            if store.is_dir():
                found.append(store)
    return found


def copy_store(store: Path) -> Path:
    """Snapshot the store. Chrome holds a LOCK on the live one while it runs."""
    dest = Path(tempfile.mkdtemp(prefix="or-idb-")) / "store"
    shutil.copytree(store, dest, ignore=shutil.ignore_patterns("LOCK"))
    return dest


def describe(store: Path) -> dict:
    files = [p for p in sorted(store.rglob("*")) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    newest = max((p.stat().st_mtime for p in files), default=0)

    marker_hits = {m.decode(): 0 for m in MARKERS}
    strings, text_bytes = 0, 0
    for path in files:
        if path.suffix.lower() not in (".ldb", ".log", ".sst") and path.name != "CURRENT":
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for marker in MARKERS:
            marker_hits[marker.decode()] += blob.count(marker)
        for match in PRINTABLE.finditer(blob):
            strings += 1
            text_bytes += len(match.group(0))

    return {
        "files": len(files),
        "bytes": total,
        "newest": datetime.fromtimestamp(newest, timezone.utc).isoformat(" ", "seconds")
                  if newest else "-",
        "strings": strings,
        "text_bytes": text_bytes,
        "markers": {k: v for k, v in marker_hits.items() if v},
    }


def describe_firefox() -> list[str]:
    """What the Firefox store holds, counted — never its contents.

    Uses the shipped reader rather than a second copy of the format, so this probe and
    the ingest path can never disagree about what is in there.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from llm_archive.core import idb
    except ImportError:
        return []

    lines: list[str] = []
    for path in idb.stores("https://openrouter.ai"):
        try:
            with idb.opened(path) as con:
                for store_id, name in idb.store_names(con).items():
                    rows = list(idb.records(con, store_id))
                    if not rows:
                        continue
                    kinds: dict[str, int] = {}
                    bad = 0
                    for key, value in rows:
                        if isinstance(value, Exception):
                            bad += 1
                        parts = key.split(":")
                        kind = parts[1] if len(parts) > 2 else "?"
                        kinds[kind] = kinds.get(kind, 0) + 1
                    lines.append(f"  {path.parents[4].name}/{path.name}")
                    lines.append(f"    store   {name}")
                    lines.append(f"    records {len(rows)}"
                                 + (f", {bad} undecodable" if bad else
                                    ", all decodable"))
                    lines.append("    by type " + ", ".join(
                        f"{k} {n}" for k, n in sorted(kinds.items())))
                    lines.append(f"    -> {kinds.get('room', 0)} conversation(s)")
        except OSError as exc:
            lines.append(f"  {path}: could not read ({exc})")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profiles-root", type=Path, default=None,
                    help="an extra Chromium 'User Data' directory to look in")
    args = ap.parse_args()

    firefox = describe_firefox()
    if firefox:
        print("Firefox store(s) found — this is the supported route.\n")
        for line in firefox:
            print(line)
        print("\n  llma browser            what is on this machine")
        print("  llma browser --enable   read it on every ingest")
        print("\nChats read this way merge with ones you exported by hand: both key a")
        print("session on the same root message id, so nothing is stored twice.")

    stores = candidate_stores(args.profiles_root)
    if not stores:
        if firefox:
            print("\nNo Chromium store — not needed, the Firefox one is readable.")
            return 0
        print("No OpenRouter IndexedDB store found in any browser profile.")
        print("\nThat is a real answer, not a failure. It means one of:")
        print("  - you use a browser this cannot read (only Firefox is supported;")
        print("    Chromium's LevelDB/V8 format is not), or")
        print("  - a profile outside the standard roots (pass --profiles-root), or")
        print("  - the site keeps nothing locally for this account.")
        print("\nEither way §2.1's IndexedDB route is not available here, and per-chat")
        print("export stays the only way in.")
        return 1

    print(f"{len(stores)} store(s) found.\n")
    verdict_ready = False
    for store in stores:
        print(f"  {store}")
        try:
            copy = copy_store(store)
        except OSError as exc:
            print(f"    could not snapshot it: {exc}")
            print("    close the browser and re-run — Chrome holds the store open.")
            continue
        try:
            info = describe(copy)
        finally:
            shutil.rmtree(copy.parent, ignore_errors=True)

        print(f"    {info['files']} files, {info['bytes']/1e6:.1f} MB, "
              f"newest write {info['newest']}")
        print(f"    {info['strings']:,} printable runs, "
              f"{info['text_bytes']/1e6:.2f} MB of text")
        if info["markers"]:
            verdict_ready = True
            print("    schema markers:", ", ".join(
                f"{k}x{v}" for k, v in sorted(info["markers"].items(),
                                              key=lambda kv: -kv[1])))
        else:
            print("    no orpg.3.0 markers — this store is not the chat history")

    print()
    if verdict_ready:
        print("VERDICT: the chat graph is in there. Building the reader means an "
              "SSTable\n         reader plus a V8 structured-clone decoder — neither "
              "in the stdlib,\n         both against an undocumented format. Weigh "
              "that against the number\n         of chats above before starting.")
    else:
        print("VERDICT: nothing recognisable as OpenRouter chat data. Do not build "
              "the reader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
