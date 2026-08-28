"""Reading the drops folder when the filename does not say which source a file is.

Two separate problems, one answer — look at the bytes.

claude.ai and DeepSeek both export a `conversations.json`, bare or inside a ZIP, and
neither filename says which one it is. Grok has the opposite problem: its export ZIP is
named after a bare uuid (`5f4c8d58-….zip`), so nothing outside the archive identifies it
at all, and the member that holds the chats is buried at
`ttl/30d/export_data/<user id>/prod-grok-backend.json`.

Both are settled by reading a window off the front of the interesting member rather than
parsing it — the same rule `tools/probe_exports.py` already follows — so sniffing a
14 MB export costs a read of the first page and nothing else.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

MEMBER = "conversations.json"
SNIFF_BYTES = 1 << 16

# Where consumed drops are moved once their sessions are in the database. Discovery
# must not descend into it, or every archived export would be re-parsed forever.
ARCHIVE_DIR = "_archive"


def candidates(drops: Path) -> list[Path]:
    """Everything in the drops folder that is worth sniffing.

    Skips the `_archive/` subtree and dotfiles. Directories survive because the Gemini
    adapter accepts an unpacked Takeout tree as well as a ZIP.
    """
    if not drops.exists():
        return []
    return [p for p in sorted(drops.iterdir())
            if p.name != ARCHIVE_DIR and not p.name.startswith(".")]


def taken_at(path: Path) -> int | None:
    """Epoch ms of the drop's mtime — when this export was taken.

    Set on every session parsed out of a drop, and the only thing that lets ingest
    refuse to apply an older snapshot over a newer one. The same clock `freshness`
    already uses to date an export, so the two cannot disagree about which is newer.

    The live local stores (Claude Code, Codex, opencode, VS Code) deliberately leave
    this NULL. Their files are not snapshots of a remote account — they are the account
    — so "this file is older than what I stored" is never a reason to refuse a read,
    and a tree copied off another machine with `--root` would trip exactly that guard.
    """
    try:
        return int(path.stat().st_mtime * 1000)
    except OSError:
        return None


def by_recency(paths: Iterable[Path]) -> list[Path]:
    """Oldest export first, so the newest one is parsed last and therefore wins.

    Discovery used to return drops in filename order, which is meaningless for two of
    these sources: Grok names its ZIP after a bare uuid and OpenRouter names its JSON
    after the chat title. With two exports of one account present, whichever sorted
    last overwrote the other regardless of age -- a stale snapshot could truncate a
    conversation that had since grown. mtime is the same clock `freshness` already
    trusts to date an export, so the two agree about which drop is newer.
    """
    def when(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
    return sorted(paths, key=lambda p: (when(p), p.name))


def member_head(path: Path, member: str, limit: int = SNIFF_BYTES) -> str | None:
    """First `limit` decoded bytes of the member ending in `member` inside `path`.

    Returns None when the drop holds no such member — including when it is not a ZIP at
    all, which is a "not mine" answer for every adapter, not an error worth reporting.
    A bare file whose own name ends in `member` is read directly, since several of these
    exports arrive unzipped.
    """
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                name = next((n for n in zf.namelist() if n.endswith(member)), None)
                if name is None:
                    return None
                with zf.open(name) as fh:
                    raw = fh.read(limit)
        elif path.name.endswith(member):
            with path.open("rb") as fh:
                raw = fh.read(limit)
        else:
            return None
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return None
    return raw.decode("utf-8", errors="replace")


def conversations_head(path: Path, limit: int = SNIFF_BYTES) -> str | None:
    """The claude.ai / DeepSeek case: whichever of them wrote this `conversations.json`."""
    return member_head(path, MEMBER, limit)
