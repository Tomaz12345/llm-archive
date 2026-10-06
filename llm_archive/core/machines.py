"""Machines this archive cannot reach: a bundle packed there, merged in here.

`llma ingest --root` covers a store copied across once, by hand. A machine with no SSH
route back -- a laptop, a work box, the one Claude Code was tried out on -- needs that
to be repeatable, and three things about it are not obvious:

* **The bundle names its machine.** Nothing in any of these formats records a hostname,
  so the packer writes the machine's own `gethostname()` into `llma-machine.json`. A
  name typed at import time is a name typed differently next month, and "laptop" and
  "Laptop" then count as two machines in every statistic. A machine filed under another
  name once (`--host`) remembers the hostname it was given with, so the next bundle from
  that box lands in the same place without being told.
* **Merge, never replace.** Claude Code deletes a transcript 30 days after its last
  activity unless `cleanupPeriodDays` says otherwise. A bundle packed after that has
  forgotten the session; the copy here must not forget it too. So a bundle's files are
  merged into `data/machines/<host>/`: new files are added, newer ones replace older,
  and a file the bundle no longer carries is kept. A file only ever moves forward in
  time, so an old bundle found and added late cannot roll a transcript back.
* **Only what an adapter reads.** `RULES` lists, per source, the paths its adapter
  opens. Everything else is ignored on both sides: the packer never reads
  `.credentials.json` or `auth.json`, and the unpacker would not write them if a
  hand-made zip carried them anyway. The list is also the contract with
  `scripts/pack-machine.ps1`, which runs where there is no Python and re-states it.

The merged tree is then just another root for the portable adapters, read under its
machine's name by every `llma sync`. It lives under `data/`, so `raw_path` keeps
pointing at a file the archive owns rather than at a USB stick that has gone home.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

KIND = "machine"                    # drop_file.kind of a bundle in the ledger
MANIFEST = "llma-machine.json"      # at the root of a bundle
FORMAT = "llma-machine-bundle"
VERSION = 1
DIRNAME = "machines"                # data/machines/, beside data/drops/
MACHINE_FILE = "machine.json"       # data/machines/<slug>/machine.json

# Claude Code's own default for `cleanupPeriodDays`, and how far ahead of it the next
# bundle is due. A bundle at least every (cleanup - margin) days means no transcript can
# age out on the machine between two of them.
CLAUDE_CLEANUP_DAYS = 30
MARGIN_DAYS = 7
MAX_INTERVAL_DAYS = 30

# A zip entry's timestamp is a DOS one: two-second resolution. Two copies of one file
# compared across a zip round trip differ by up to that much without either being newer.
MTIME_SLACK = 2.0

DAY_MS = 86_400_000

# Per source: the paths its adapter opens, relative to the store's root. Nothing else
# goes into a bundle or comes out of one. Keep `scripts/pack-machine.ps1` in step --
# tests/test_machines.py runs the script and compares.
RULES: dict[str, tuple[re.Pattern, ...]] = {
    # Transcripts, subagent transcripts, and the sidecar files a `<persisted-output>`
    # marker points at. Not `memory/`, which no adapter reads.
    "claude_code": (
        re.compile(r"projects/[^/]+/[^/]+\.jsonl"),
        re.compile(r"projects/[^/]+/[^/]+/(?:subagents|tool-results)/[^/]+"),
    ),
    "codex": (
        re.compile(r"sessions/(?:[^/]+/)*[^/]+\.jsonl"),
        re.compile(r"session_index\.jsonl"),          # the only place titles live
    ),
    "opencode": (
        re.compile(r"storage/(?:session|message|part)/(?:[^/]+/)*[^/]+\.json"),
    ),
    "vscode_chat": (
        re.compile(r"workspaceStorage/[^/]+/workspace\.json"),
        re.compile(r"workspaceStorage/[^/]+/chatSessions/[^/]+\.jsonl?"),
        re.compile(r"globalStorage/emptyWindowChatSessions/[^/]+\.jsonl?"),
    ),
}

# Where under each store's root the packer starts walking. Walking all of VS Code's
# User folder to keep forty files would read every extension's private storage.
WALK = {
    "claude_code": ("projects",),
    "codex": ("sessions", "session_index.jsonl"),
    "opencode": ("storage/session", "storage/message", "storage/part"),
    "vscode_chat": ("workspaceStorage", "globalStorage/emptyWindowChatSessions"),
}

_VSCODE_CHAT = re.compile(r"workspaceStorage/([^/]+)/chatSessions/")
_VSCODE_META = re.compile(r"workspaceStorage/([^/]+)/workspace\.json")
# A bare Claude Code store, zipped by hand: `projects/<dir>/<file>.jsonl` under some
# prefix (`.claude/`, or nothing when the folder's contents were zipped).
_BARE = re.compile(r"(.*/)?projects/[^/]+/[^/]+\.jsonl")


class BundleError(ValueError):
    """A bundle that cannot be filed, with the reason in words."""


def allowed(source: str, rel: str) -> bool:
    """Is `rel` (posix, relative to the store root) a file this source's adapter reads?"""
    rules = RULES.get(source)
    if not rules or not _safe(rel):
        return False
    return any(rule.fullmatch(rel) for rule in rules)


def _safe(rel: str) -> bool:
    """No absolute paths, drive letters, or `..` -- a zip names its own destinations."""
    if not rel or rel.startswith("/") or "\\" in rel or ":" in rel or "\x00" in rel:
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def script_path() -> Path:
    """The packer for a machine without llma: Windows PowerShell 5.1 and nothing else."""
    return Path(__file__).resolve().parent.parent / "scripts" / "pack-machine.ps1"


# -- packing ---------------------------------------------------------------


def local_stores() -> dict[str, Path]:
    """This machine's portable stores, at the places the adapters read them from.

    Computed per call rather than borrowed from the adapters' class attributes, which
    are fixed at import time and so would ignore a test's -- or a user's -- HOME.
    """
    from ..adapters.vscode_chat import _default_roots

    home = Path.home()
    stores = {
        "claude_code": home / ".claude",
        "codex": home / ".codex",
        "opencode": home / ".local" / "share" / "opencode",
    }
    vscode = next(iter(_default_roots()), None)
    if vscode is not None:
        stores["vscode_chat"] = vscode
    return stores


def collect(stores: dict[str, Path]) -> list[tuple[str, str, Path]]:
    """(source, rel, file) for every file a bundle of these stores carries."""
    picked: list[tuple[str, str, Path]] = []
    for source, root in stores.items():
        if source not in RULES or root is None or not root.is_dir():
            continue
        for start in WALK[source]:
            begin = root / start
            if begin.is_file():
                files = [begin]
            elif begin.is_dir():
                files = sorted(p for p in begin.rglob("*") if p.is_file())
            else:
                continue
            for path in files:
                rel = path.relative_to(root).as_posix()
                if allowed(source, rel):
                    picked.append((source, rel, path))

    # A workspace.json is only worth carrying for a workspace that has chats: it is
    # what names the folder they belong to, and there are hundreds of the rest.
    chats = {m.group(1) for s, rel, _ in picked
             if s == "vscode_chat" and (m := _VSCODE_CHAT.match(rel))}
    return [(s, rel, p) for s, rel, p in picked
            if not (s == "vscode_chat" and (m := _VSCODE_META.fullmatch(rel))
                    and m.group(1) not in chats)]


def cleanup_days(claude_root: Path) -> int | None:
    """`cleanupPeriodDays` from Claude Code's settings, or None when it is the default.

    Only that one key is read. settings.json is otherwise none of the archive's business.
    """
    try:
        settings = json.loads((claude_root / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = settings.get("cleanupPeriodDays") if isinstance(settings, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class Packed:
    path: Path
    host: str
    files: dict[str, int] = field(default_factory=dict)
    bytes: int = 0
    cleanup_period_days: int | None = None


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip(".-_").lower() or "machine"


def _zip_time(mtime: float) -> tuple:
    # DOS timestamps start in 1980; a file claiming to be older is clamped, not refused.
    return time.localtime(max(mtime, 315532800 + 86400))[:6]


def pack(out: Path, host: str | None = None, stores: dict[str, Path] | None = None,
         packed_by: str = "llma pack") -> Packed:
    """Write a bundle of `stores` (default: this machine's) to `out`, a folder or a .zip.

    Each file is read whole before it is written. A transcript Claude Code is appending
    to as this runs then goes in as one consistent snapshot rather than a zip entry
    whose length disagrees with its header.
    """
    from .ingest import local_host

    host = host or local_host()
    stores = local_stores() if stores is None else stores
    files = collect(stores)
    if not files:
        raise BundleError("nothing to pack: no transcripts under "
                          + ", ".join(str(p) for p in stores.values()))

    if out.suffix.lower() == ".zip":
        dest = out
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        dest = out / f"llma-machine-{_slug(host)}-{stamp}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    result = Packed(path=dest, host=host)
    sources: dict[str, dict] = {}
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for source, rel, path in files:
            try:
                data = path.read_bytes()
                mtime = path.stat().st_mtime
            except OSError:
                continue            # vanished or locked mid-walk; the next bundle has it
            info = zipfile.ZipInfo(f"{source}/{rel}", _zip_time(mtime))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
            row = sources.setdefault(source, {"files": 0, "bytes": 0})
            row["files"] += 1
            row["bytes"] += len(data)
        if "claude_code" in sources:
            result.cleanup_period_days = cleanup_days(stores["claude_code"])
            sources["claude_code"]["cleanup_period_days"] = result.cleanup_period_days
        zf.writestr(MANIFEST, json.dumps({
            "format": FORMAT, "version": VERSION, "host": host,
            "packed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "packed_by": packed_by, "os": platform.system(), "sources": sources,
        }, indent=2))
    os.replace(part, dest)

    result.files = {s: row["files"] for s, row in sources.items()}
    result.bytes = sum(row["bytes"] for row in sources.values())
    return result


# -- recognising one -------------------------------------------------------


@dataclass
class Layout:
    """How the names inside a bundle map onto (source, rel).

    A real bundle has a manifest and `<source>/<rel>` names. A bare Claude Code store --
    the `.claude` folder zipped by hand -- has neither, only `projects/` under some
    `strip` prefix (or, given the projects folder itself, `add` supplies the prefix).
    """
    manifest: dict | None
    strip: str = ""
    add: str = ""

    @property
    def bundle(self) -> bool:
        return self.manifest is not None

    def place(self, name: str) -> tuple[str, str] | None:
        if self.bundle:
            source, _, rel = name.partition("/")
            return (source, rel) if source in RULES and rel else None
        if not name.startswith(self.strip):
            return None
        return "claude_code", self.add + name[len(self.strip):]


def _norm(name: str) -> str:
    # Windows PowerShell 5.1's zip writers have used backslashes in entry names.
    return name.replace("\\", "/")


def inspect(path: Path) -> Layout | None:
    """Is this a machine bundle, or a Claude Code store, and how is it laid out?

    Cheap on anything else: `llma add ~/Downloads` asks this of every folder and zip it
    passes, so a folder is judged by two `stat`s and a zip by its name list alone.
    """
    try:
        if path.is_dir():
            if (path / MANIFEST).is_file():
                return Layout(manifest=_read_manifest_file(path / MANIFEST))
            if path.name.casefold() == "projects" and any(path.glob("*/*.jsonl")):
                return Layout(manifest=None, add="projects/")
            if (path / "projects").is_dir() and any((path / "projects").glob("*/*.jsonl")):
                return Layout(manifest=None)
            return None
        if path.suffix.lower() != ".zip":
            return None
        with zipfile.ZipFile(path) as zf:
            names = [_norm(n) for n in zf.namelist()]
            if MANIFEST in names:
                return Layout(manifest=_parse_manifest(zf.read(
                    zf.namelist()[names.index(MANIFEST)])))
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None
    for name in names:
        if match := _BARE.fullmatch(name):
            return Layout(manifest=None, strip=match.group(1) or "")
    return None


def _parse_manifest(raw: bytes) -> dict:
    """The manifest as a dict; an unreadable one is still a bundle, just a broken one."""
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_manifest_file(path: Path) -> dict:
    return _parse_manifest(path.read_bytes())


def _check(manifest: dict) -> None:
    if manifest.get("format") != FORMAT:
        raise BundleError(f"{MANIFEST} is not a machine bundle manifest")
    version = manifest.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise BundleError(f"bundle format v{version} is newer than this llma reads "
                          f"(v{VERSION}); update llma")
    if not isinstance(manifest.get("host"), str) or not manifest["host"].strip():
        raise BundleError(f"{MANIFEST} names no machine")


def tree_id(path: Path) -> tuple[str, int]:
    """Identity and size of a store given as a folder: (sha256, bytes).

    A folder is not a file with one hash, and hashing every byte of a `.claude` tree --
    shell snapshots, file history, debug logs -- to recognise it again is minutes spent
    on files a bundle never reads. Names, sizes and mtimes of the files it does read
    are enough to tell "the same folder again" from "it has changed since".
    """
    layout = inspect(path)
    digest, size = hashlib.sha256(), 0
    for entry in _entries(path, layout) if layout else ():
        if entry.wanted:
            digest.update(f"{entry.source}/{entry.rel}\0{entry.size}\0"
                          f"{int(entry.mtime)}\n".encode())
            size += entry.size
    return digest.hexdigest(), size


# -- merging one in --------------------------------------------------------


@dataclass
class Entry:
    source: str
    rel: str
    mtime: float
    size: int
    read: Callable[[], bytes]
    wanted: bool


def _entries(path: Path, layout: Layout) -> Iterator[Entry]:
    """Every file in the bundle, with whether RULES let it through."""
    def make(name: str, mtime: float, size: int, read) -> Entry | None:
        if name == MANIFEST:
            return None
        placed = layout.place(name)
        if placed is None:
            return Entry("", name, mtime, size, read, wanted=False)
        source, rel = placed
        return Entry(source, rel, mtime, size, read, wanted=allowed(source, rel))

    if path.is_dir():
        # A bare store is walked from its projects folder only: the rest of a `.claude`
        # tree is shell snapshots and file history that RULES would refuse one by one.
        base = path if layout.bundle or layout.add else path / "projects"
        for file in sorted(p for p in base.rglob("*") if p.is_file()):
            st = file.stat()
            entry = make(file.relative_to(path).as_posix(), st.st_mtime, st.st_size,
                         file.read_bytes)
            if entry is not None:
                yield entry
        return

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.endswith(("/", "\\")):
                continue
            entry = make(_norm(info.filename),
                         time.mktime(info.date_time + (0, 0, -1)), info.file_size,
                         lambda info=info: zf.read(info))
            if entry is not None:
                yield entry


@dataclass
class Machine:
    """One machine filed under `data/machines/`, as its machine.json describes it."""
    host: str
    dir: Path
    hostnames: list[str] = field(default_factory=list)
    packed_at: int | None = None        # newest bundle's own packing time, epoch ms
    added_at: int | None = None
    bundles: int = 0
    cleanup_period_days: int | None = None
    sources: list[str] = field(default_factory=list)


def listing(machines: Path) -> list[Machine]:
    """Every machine filed under `machines`, in folder order."""
    if not machines.is_dir():
        return []
    out = []
    for folder in sorted(p for p in machines.iterdir() if p.is_dir()):
        try:
            data = json.loads((folder / MACHINE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        host = data.get("host") if isinstance(data, dict) else None
        if not isinstance(host, str) or not host:
            continue
        out.append(Machine(
            host=host, dir=folder,
            hostnames=[h for h in data.get("hostnames") or [] if isinstance(h, str)],
            packed_at=data.get("packed_at"), added_at=data.get("added_at"),
            bundles=data.get("bundles") or 0,
            cleanup_period_days=data.get("cleanup_period_days"),
            sources=[s for s in RULES if (folder / s).is_dir()]))
    return out


def _resolve(machines: Path, host: str | None, packed_host: str | None) -> Machine:
    """The machine a bundle belongs to: an existing one if any name matches, else new.

    Matching is case-insensitive throughout, since the point is that a casing slip does
    not make a second machine. `--host` decides when given; otherwise the bundle's own
    hostname is looked up among the names each machine has been seen with first, so a
    box filed as "laptop" once stays "laptop" without being told again.
    """
    known = listing(machines)
    if not host and packed_host:
        for machine in known:
            if packed_host.casefold() in {h.casefold() for h in machine.hostnames}:
                return machine
    name = (host or packed_host or "").strip()
    if not name:
        raise BundleError("a Claude Code store with no machine name: add --host NAME")
    for machine in known:
        if machine.host.casefold() == name.casefold() or machine.dir.name == _slug(name):
            return machine
    return Machine(host=name, dir=machines / _slug(name))


def _iso_ms(value) -> int | None:
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                   .timestamp() * 1000)
    except ValueError:
        return None


@dataclass
class Unpacked:
    host: str
    dir: Path
    packed_at: int | None = None
    added: dict[str, int] = field(default_factory=dict)
    updated: dict[str, int] = field(default_factory=dict)
    unchanged: int = 0
    kept: int = 0           # ours was newer than the bundle's copy
    ignored: int = 0        # outside RULES: settings, credentials, memory, ...
    failed: int = 0

    def describe(self) -> str:
        new, upd = sum(self.added.values()), sum(self.updated.values())
        bits = [f"{new} new", f"{upd} updated"]
        if self.unchanged:
            bits.append(f"{self.unchanged} unchanged")
        if self.kept:
            bits.append(f"{self.kept} older than ours, kept ours")
        if self.ignored:
            bits.append(f"{self.ignored} not read, skipped")
        if self.failed:
            bits.append(f"{self.failed} failed")
        return f"{self.host}: " + ", ".join(bits)


def _merge(entry: Entry, dest: Path) -> str:
    """Write one file forward in time only. Returns what happened to it."""
    try:
        st = dest.stat()
    except FileNotFoundError:
        st = None
    if st is not None:
        if entry.mtime < st.st_mtime - MTIME_SLACK:
            return "kept"
        if entry.mtime <= st.st_mtime + MTIME_SLACK and entry.size == st.st_size:
            return "unchanged"
    data = entry.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and renamed over, so a sync reading the tree at the same moment
    # sees the old transcript or the new one, never half of one. The suffix keeps the
    # half-written file out of every adapter's glob.
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.utime(tmp, (entry.mtime, entry.mtime))
    os.replace(tmp, dest)
    return "added" if st is None else "updated"


def _live_store(path: Path) -> bool:
    """Is `path` this machine's own store, which `llma ingest` already reads live?"""
    try:
        resolved = path.resolve()
        roots = [root.resolve() for root in local_stores().values()]
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def unpack(path: Path, machines: Path, host: str | None = None) -> Unpacked:
    """Merge a bundle (or a bare Claude Code store) into its machine's tree."""
    from .ingest import local_host

    layout = inspect(path)
    if layout is None:
        raise BundleError("not a machine bundle or a Claude Code store")
    if layout.bundle:
        _check(layout.manifest)
    elif path.is_dir() and _live_store(path):
        raise BundleError("this is this machine's own store, which `llma ingest` "
                          "already reads")
    packed_host = layout.manifest.get("host") if layout.bundle else None

    # Filed under another name or not, a bundle packed here is this machine's own
    # sessions. Taking it in would hold each one twice, from two paths under two hosts,
    # and every sync would flip it from one to the other.
    here = local_host().casefold()
    machine = _resolve(machines, host, packed_host)
    for name in (packed_host, machine.host):
        if name and name.casefold() == here:
            raise BundleError(f"packed on {name}, which is this machine: its stores "
                              f"are read live by `llma ingest`, not from a bundle")

    packed_at = (_iso_ms(layout.manifest.get("packed_at")) if layout.bundle else None) \
        or int(path.stat().st_mtime * 1000)
    result = Unpacked(host=machine.host, dir=machine.dir, packed_at=packed_at)
    root = machine.dir.resolve() if machine.dir.exists() else None
    wanted = 0

    for entry in _entries(path, layout):
        if not entry.wanted:
            result.ignored += 1
            continue
        wanted += 1
        dest = machine.dir / entry.source / entry.rel
        try:
            if root is None:
                machine.dir.mkdir(parents=True, exist_ok=True)
                root = machine.dir.resolve()
            # RULES already refuse `..`; this is the belt to that pair of braces.
            if root not in dest.resolve().parents:
                result.ignored += 1
                continue
            what = _merge(entry, dest)
        except (OSError, zipfile.BadZipFile, RuntimeError):
            result.failed += 1
            continue
        if what == "added":
            result.added[entry.source] = result.added.get(entry.source, 0) + 1
        elif what == "updated":
            result.updated[entry.source] = result.updated.get(entry.source, 0) + 1
        elif what == "kept":
            result.kept += 1
        else:
            result.unchanged += 1

    if not wanted:
        raise BundleError("no transcripts in it: nothing a source adapter reads")
    if root is None:
        raise BundleError(f"could not write under {machine.dir}")

    newest = packed_at >= (machine.packed_at or 0)
    hostnames = {h.casefold(): h for h in machine.hostnames}
    if packed_host:
        hostnames.setdefault(packed_host.casefold(), packed_host)
    cleanup = machine.cleanup_period_days
    if newest and layout.bundle:
        cleanup = ((layout.manifest.get("sources") or {}).get("claude_code") or {}) \
            .get("cleanup_period_days")
    (machine.dir / MACHINE_FILE).write_text(json.dumps({
        "host": machine.host,
        "hostnames": sorted(hostnames.values()),
        "packed_at": max(packed_at, machine.packed_at or 0),
        "added_at": int(time.time() * 1000),
        "bundles": machine.bundles + 1,
        "cleanup_period_days": cleanup,
    }, indent=2), encoding="utf-8")
    return result


# -- due -------------------------------------------------------------------


def interval_days(machine: Machine) -> int:
    """How often a bundle is wanted from this machine, in days.

    Set by Claude Code's retention there, when it has Claude Code: the last bundle's
    own `cleanupPeriodDays`, less a week's margin. Never longer than a month even when
    nothing would be lost, because then it is the archive that has gone stale.
    """
    if "claude_code" not in machine.sources:
        return MAX_INTERVAL_DAYS
    keep = machine.cleanup_period_days or CLAUDE_CLEANUP_DAYS
    return max(1, min(MAX_INTERVAL_DAYS, keep - MARGIN_DAYS))


def due(machines: Path, now_ms: int | None = None) -> list[dict]:
    """Machines whose last bundle is older than their interval, oldest first."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    rows = []
    for machine in listing(machines):
        if machine.packed_at is None:
            continue
        age = (now - machine.packed_at) / DAY_MS
        if age < interval_days(machine):
            continue
        reason = f"last bundle packed {age:.0f}d ago"
        if "claude_code" in machine.sources:
            keep = machine.cleanup_period_days or CLAUDE_CLEANUP_DAYS
            reason += (f"; Claude Code there deletes a transcript {keep}d after "
                       f"its last activity")
        rows.append({"host": machine.host, "age_days": age, "reason": reason,
                     "how": "run pack-machine.ps1 there (`llma pack --script <dir>` "
                            "writes it), then `llma add <zip>` here"})
    return sorted(rows, key=lambda r: -r["age_days"])
