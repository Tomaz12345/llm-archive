"""Taking a file in: identify it, file it, remember it, move it aside when it is spent.

Before this, adding a web export meant knowing which of eight formats you had, knowing
that `data/drops/` was where it went, and remembering to run `llma ingest` afterwards.
`data/drops/` and `~/Downloads` on this machine were byte-identical for all ten exports,
which is what that workflow looks like when it is working — and there is no version of
it that survives being forgotten for a month.

Nothing here parses anything. It asks each adapter the one question the adapter is
already able to answer about a single file — `claims(path)`, lifted out of its
`discover()` so the rule has exactly one home — and then does the filing.

Three rules that are not obvious:

* **Copy, never move.** The file in ~/Downloads is the user's, and an import tool that
  empties the folder it read from is a tool nobody trusts twice. The archive takes its
  own copy; the original stays where it was put.
* **Identity is the sha256, not the name.** The same Mistral chat re-exported is
  `chat-export-<a new epoch ms>.zip`; the same claude.ai archive downloaded twice is
  `conversations-000.zip` and `conversations-000 (1).zip`. Only the content is stable,
  so "have I already got this?" is a hash lookup.
* **A file that is not an export is recorded as one that is not.** Otherwise every scan
  of ~/Downloads re-sniffs the same 52 MB PDF forever.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

READ_CHUNK = 1 << 20

# Where a consumed drop goes, under the drops folder itself: the raw bytes stay beside
# the archive that was built from them, which is what makes `raw_path` + `raw_hash` a
# real "you can re-parse this from scratch" promise rather than a stored filename.
ARCHIVE = "_archive"

# Where an upload lands while it is being identified. A ZIP member cannot be sniffed
# from a stream, so the bytes have to be on disk first — but a file sitting here is not
# yet a drop, and nothing may treat it as one. Dot-prefixed so `_drops.candidates`
# skips it during discovery.
STAGING = ".uploads"

# What `identify` will not even open. Everything a browser has ever put in a Downloads
# folder passes through here, and sniffing a 50 MB video for JSON markers is a waste of
# a read on every scan.
SKIP_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".m4a",
    ".pdf", ".docx", ".xlsx", ".pptx", ".exe", ".msi", ".dmg", ".iso",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico",
    ".ttf", ".otf", ".woff", ".woff2", ".css", ".md", ".txt", ".log", ".csv",
}

# Suffixes an export actually arrives with. A directory is also allowed through, since
# the Gemini adapter accepts an unpacked Takeout tree.
EXPORT_SUFFIXES = {".zip", ".json", ".har", ".html", ".htm"}


@dataclass
class Taken:
    """One file `add` looked at, and what became of it."""
    source: Path
    kind: str | None = None
    action: str = ""          # added | held | skipped | failed
    detail: str = ""
    dest: Path | None = None
    sha: str | None = None

    @property
    def ok(self) -> bool:
        return self.action == "added"


def sha256_file(path: Path) -> str:
    """Streamed, because the T3 export is 14 MB and this runs over whole folders."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    """Content identity for an unpacked export tree: names and hashes of its files."""
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).replace("\\", "/").encode())
        digest.update(sha256_file(child).encode())
    return digest.hexdigest()


def content_id(path: Path) -> str:
    return sha256_dir(path) if path.is_dir() else sha256_file(path)


def exported_at(path: Path) -> int | None:
    """When the export was taken. One definition, shared with the adapters."""
    from ..adapters._drops import taken_at
    return taken_at(path)


# -- identification --------------------------------------------------------


def drop_adapters() -> list:
    """The adapters that read `data/drops/`, in the order intake asks them.

    Ordered most-specific first. `claude_web` and `deepseek` both answer about a
    `conversations.json` and exclude each other explicitly, but `copilot_web` claims any
    `.har` on its extension alone, so it goes last — a `.har` is the one drop no other
    adapter wants, and asking it first would be asking the least discriminating question
    first.
    """
    from ..adapters.claude_web import ClaudeWebAdapter
    from ..adapters.chatgpt import ChatGPTAdapter
    from ..adapters.deepseek import DeepSeekAdapter
    from ..adapters.gemini import GeminiAdapter
    from ..adapters.grok import GrokAdapter
    from ..adapters.mistral import MistralAdapter
    from ..adapters.openrouter import OpenRouterAdapter
    from ..adapters.t3chat import T3ChatAdapter
    from ..adapters.copilot_web import CopilotWebAdapter

    return [ClaudeWebAdapter, ChatGPTAdapter, DeepSeekAdapter, GrokAdapter,
            MistralAdapter, OpenRouterAdapter, T3ChatAdapter, GeminiAdapter,
            CopilotWebAdapter]


def identify(path: Path) -> str | None:
    """Which source produced this file, or None if it is not an export at all.

    Asks the adapters' own `claims()`, so intake and discovery can never disagree about
    what a file is. A file two adapters both claim goes to the first that answers —
    which is why the order above is deliberate rather than alphabetical.
    """
    if not path.is_dir():
        suffix = path.suffix.lower()
        if suffix in SKIP_SUFFIXES or suffix not in EXPORT_SUFFIXES:
            return None
    for cls in drop_adapters():
        try:
            if cls.claims(path):
                return cls.kind
        except (OSError, ValueError):
            continue        # unreadable is a "not mine", not a failure worth raising
    return None


def expand(paths) -> list[Path]:
    """Files to consider, from a mix of files, folders and globs.

    A folder is scanned one level deep, not recursively: `llma add ~/Downloads` should
    look at what is in Downloads, not walk into every project checkout underneath it.
    A folder that is itself an export tree (an unpacked Takeout) is taken whole.
    """
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            if identify(path) is not None:
                out.append(path)
                continue
            out.extend(sorted(p for p in path.iterdir() if p.is_file()))
        elif path.exists():
            out.append(path)
        else:
            # An unexpanded glob: the shell leaves it verbatim when nothing matched.
            out.extend(sorted(Path(path.parent).glob(path.name)))
    seen, unique = set(), []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


# -- the drop ledger -------------------------------------------------------


def known(con: sqlite3.Connection, sha: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM drop_file WHERE sha256=?", (sha,)).fetchone()


def record(con: sqlite3.Connection, sha: str, name: str, kind: str | None,
           size: int, taken_at: int | None, path: Path,
           origin: str | None = None) -> None:
    con.execute(
        "INSERT OR IGNORE INTO drop_file(sha256,name,kind,bytes,exported_at,added_at,"
        "path,origin) VALUES (?,?,?,?,?,?,?,?)",
        (sha, name, kind, size, taken_at, int(time.time() * 1000), str(path), origin))


def ledger(con: sqlite3.Connection, exports_only: bool = True) -> list[sqlite3.Row]:
    """Drops the archive has taken in, newest export first.

    Non-exports are recorded too — that is what stops a scan of ~/Downloads re-sniffing
    the same 50 MB PDF every time — but there are 139 of them here against 10 real
    exports, so they are not what "what am I holding?" means. `exports_only=False`
    returns them as well.
    """
    where = "" if not exports_only else " WHERE kind IS NOT NULL"
    return list(con.execute(
        f"SELECT * FROM drop_file{where}"
        " ORDER BY COALESCE(exported_at, added_at) DESC, id DESC"))


def passed_over(con: sqlite3.Connection) -> int:
    """How many files were looked at and were not exports."""
    return con.execute(
        "SELECT COUNT(*) c FROM drop_file WHERE kind IS NULL").fetchone()["c"]


# -- taking files in -------------------------------------------------------


def existing_copy(drops: Path, sha: str, size: int,
                  ignore: Path | None = None) -> Path | None:
    """A byte-identical file already in the drops folder, ledger or no ledger.

    The ledger lives in the database and the drops live on disk, and those two can come
    apart: rebuild the archive from scratch — which is a supported thing to do, since
    every session can be re-parsed from its drop — and the folder still holds ten
    exports that the empty ledger has never heard of. Without this check, the next
    `llma add ~/Downloads` copies all ten in again beside themselves, and each one gets
    parsed twice more. (Harmless to the data, because the upsert is idempotent, but it
    turns a 1-second ingest into a 3-second one and leaves the folder a mess.)

    Size is checked first so this costs one `stat` per drop and a hash only for the
    handful that could actually match.

    `ignore` is the file being taken in. An upload is staged inside `drops/.uploads/`
    before it can be identified — a ZIP member cannot be sniffed from a stream — so
    without this the scan finds the staged copy, concludes the export is already held,
    and files nothing. Dot-prefixed directories are skipped for the same reason.
    """
    if not drops.exists():
        return None
    skip = ignore.resolve() if ignore is not None else None
    for path in drops.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if any(part.startswith(".") for part in path.relative_to(drops).parts[:-1]):
            continue
        try:
            if path.stat().st_size != size:
                continue
            if skip is not None and path.resolve() == skip:
                continue
        except OSError:
            continue
        if sha256_file(path) == sha:
            return path
    return None


def _free_name(drops: Path, name: str) -> Path:
    """A destination that does not tread on a drop already sitting there.

    Same name, different bytes is routine — `conversations-000.zip` is what every
    claude.ai export is called, and last month's is not this month's.
    """
    dest = drops / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    for n in range(2, 1000):
        candidate = drops / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
    return drops / f"{stem}-{int(time.time())}{suffix}"


def take(path: Path, con: sqlite3.Connection, drops: Path) -> Taken:
    """Identify one file and, if it is an export, copy it into the drops folder."""
    result = Taken(source=path)
    try:
        if not path.exists():
            result.action, result.detail = "failed", "no such file"
            return result
        size = (sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                if path.is_dir() else path.stat().st_size)
        kind = identify(path)
        sha = content_id(path)
    except OSError as exc:
        result.action, result.detail = "failed", f"unreadable: {exc.strerror or exc}"
        return result

    result.kind, result.sha = kind, sha

    prior = known(con, sha)
    if prior is not None:
        result.action = "held"
        result.dest = Path(prior["path"])
        if prior["kind"] is None:
            # Two byte-identical non-exports under different names — a Downloads folder
            # is full of `thing.pdf` and `thing (1).pdf`. Saying "already in drops"
            # would claim the archive took something it deliberately did not.
            result.action = "skipped"
            result.detail = "not an export"
        elif prior["ingested_at"]:
            result.detail = "already ingested"
        else:
            result.detail = "already in drops"
        return result

    if kind is None:
        # Recorded so a repeat scan of the same folder does not sniff it again.
        record(con, sha, path.name, None, size, exported_at(path), path,
               origin=str(path))
        result.action, result.detail = "skipped", "not an export"
        return result

    drops.mkdir(parents=True, exist_ok=True)

    # Already where it needs to be — someone copied it in by hand, or pointed `add` at
    # the drops folder itself. Record it and leave it alone; copying would file the
    # same export twice under two names.
    # Already where it needs to be — copied in by hand, pointed at the drops folder
    # itself, or held over from before the database was rebuilt. Record it and leave it
    # alone; copying would file the same export twice under two names.
    here = None
    try:
        resolved, root = path.resolve(), drops.resolve()
        # Inside the drops tree already — except the upload staging area, which is a
        # holding pen on the way in rather than a place a drop lives.
        inside = root in resolved.parents and STAGING not in resolved.relative_to(
            root).parts
        here = resolved if inside else existing_copy(drops, sha, size, ignore=path)
    except (OSError, ValueError):
        here = None
    if here is not None:
        record(con, sha, here.name, kind, size, exported_at(here), here,
               origin=str(path))
        result.action, result.dest = "added", here
        result.detail = "already in drops"
        return result

    dest = _free_name(drops, path.name)
    try:
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)     # copy2 keeps the mtime, which dates the export
    except OSError as exc:
        result.action, result.detail = "failed", f"copy failed: {exc.strerror or exc}"
        return result

    record(con, sha, path.name, kind, size, exported_at(dest), dest, origin=str(path))
    result.action, result.dest = "added", dest
    return result


def take_all(paths, con: sqlite3.Connection, drops: Path) -> list[Taken]:
    results = [take(path, con, drops) for path in expand(paths)]
    con.commit()
    return results


# -- retiring a spent drop -------------------------------------------------


def archive_consumed(con: sqlite3.Connection, consumed: dict[Path, int],
                     drops: Path) -> int:
    """Move drops whose sessions are now stored into `_archive/<YYYY-MM>/`.

    Two reasons this is not just tidiness. Discovery re-sniffs every file in the folder
    on every run, so a year of exports is a year of re-reads. And two exports of one
    account sitting side by side is exactly the situation where the older one used to
    win — moving the spent one out is the structural version of the `exported_at`
    guard, not a duplicate of it.

    `session.raw_path` is repointed in the same breath. `freshness` stats that path to
    date an export and falls back to the ingest clock when the file has gone, so
    leaving it behind would silently swap one promise for another
    (freshness.py's `_newest_export_ms`). Anything that fails to move is left alone and
    simply not counted: a locked file is a reason to try again next run, not to lose
    track of it.
    """
    moved = 0
    for path, count in consumed.items():
        if not count or not path.exists() or ARCHIVE in path.parts:
            continue
        bucket = drops / ARCHIVE / datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc).strftime("%Y-%m")
        try:
            bucket.mkdir(parents=True, exist_ok=True)
            dest = _free_name(bucket, path.name)
            shutil.move(str(path), str(dest))
        except OSError:
            continue
        con.execute("UPDATE session SET raw_path=? WHERE raw_path=?",
                    (str(dest), str(path)))
        con.execute(
            "UPDATE drop_file SET path=?, ingested_at=?, sessions=? WHERE path=?",
            (str(dest), int(time.time() * 1000), count, str(path)))
        # A drop ingested before it was ever added through `llma add` has no ledger row
        # yet; backfilling here is what stops the ledger from starting out incomplete
        # on an archive that predates this module.
        if not con.execute("SELECT 1 FROM drop_file WHERE path=?",
                           (str(dest),)).fetchone():
            try:
                record(con, content_id(dest), path.name, None,
                       dest.stat().st_size, exported_at(dest), dest)
                con.execute(
                    "UPDATE drop_file SET ingested_at=?, sessions=? WHERE path=?",
                    (int(time.time() * 1000), count, str(dest)))
            except OSError:
                pass
        moved += 1
    return moved
