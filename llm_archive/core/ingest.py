"""Adapter runner.

Owns the things no adapter should know about: the database, hashing, run bookkeeping,
and the idempotence guarantee. An adapter yields Sessions; this decides whether each one
is new, changed, or unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import db, intake, redact
from .blobs import BlobStore
from .models import ParseStats


@dataclass
class IngestResult:
    kind: str
    files: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    # A session the archive already holds from a NEWER snapshot than the one just
    # parsed. Distinct from `skipped`, which means "unchanged since last time": this
    # one changed, and the change was refused because it would have gone backwards.
    stale: int = 0
    # Messages an update added to a session that was already stored, and messages the
    # archive kept because the new snapshot no longer mentioned them. Together they are
    # how you see that a re-drop grew a conversation rather than replacing it.
    appended: int = 0
    retained: int = 0
    messages: int = 0
    parts: int = 0
    orphaned: int = 0
    blobs: int = 0
    blob_bytes: int = 0
    drops_archived: int = 0
    unknown_types: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    redacted: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "files": self.files, "new": self.new, "updated": self.updated,
            "skipped": self.skipped, "stale": self.stale, "appended": self.appended,
            "retained": self.retained,
            "messages": self.messages, "parts": self.parts,
            "orphaned": self.orphaned, "blobs": self.blobs,
            "blob_bytes": self.blob_bytes, "drops_archived": self.drops_archived,
            "unknown_types": self.unknown_types,
            "errors": self.errors, "redacted": self.redacted,
            "seconds": round(self.seconds, 2),
        }


def run(adapter, con, blobs: BlobStore | None = None,
        force: bool = False, archive_drops: bool = True) -> IngestResult:
    """Ingest one adapter.

    `force` re-parses sessions whose source bytes are unchanged. Needed whenever the
    adapter itself changes: the raw_hash short-circuit is keyed on the input, not on the
    parser, so a parsing fix would otherwise be skipped for every existing session. It
    also lifts the staleness guard below, since "this export is older" is exactly the
    situation you are overriding when you re-run a fixed parser over an old drop.

    `archive_drops` moves consumed web exports aside once their sessions are stored.
    Off for the live local stores, which are not drops at all, and off in tests that
    want the input left where they put it.
    """
    started = int(time.time() * 1000)
    t0 = time.perf_counter()
    src = db.source_id(con, adapter.kind, adapter.label, adapter.surface)

    stats = ParseStats()
    result = IngestResult(kind=adapter.kind)
    consumed: dict[Path, int] = {}      # drop -> sessions it put in the database

    for target in adapter.discover():
        for session in adapter.parse(target, stats):
            # Counted before the short-circuits: a drop whose every conversation is
            # already stored has still been fully consumed and is still ready to be
            # archived. Only a drop that yielded nothing at all stays put.
            if isinstance(target, Path):
                consumed[target] = consumed.get(target, 0) + 1

            prior = db.session_state(con, src, session.native_id)
            if prior is not None and prior.raw_hash == session.raw_hash and not force:
                result.skipped += 1
                continue
            # An older snapshot must not overwrite a newer one. Discovery already hands
            # drops over oldest-first, but that only orders one run: the drop archived
            # last month is gone from the folder, and a re-download of an OLD export
            # would otherwise walk in and rewrite a session's title, end time and token
            # counts with superseded values.
            if (not force and prior is not None
                    and prior.exported_at is not None
                    and session.exported_at is not None
                    and session.exported_at < prior.exported_at):
                result.stale += 1
                continue
            merged = db.upsert_session(con, src, session)
            if merged.was_new:
                result.new += 1
            else:
                result.updated += 1
            result.appended += merged.appended
            result.retained += merged.retained
        con.commit()

    # Redaction is a stored setting, not a command you re-run: ingest re-reads the
    # original files, so a secret redacted last week is back in `part.text` the moment
    # its session is re-parsed. Applying it here — before anything indexes or embeds
    # the text — is the only placement where that does not open a window.
    if redact.is_enabled(con):
        red = redact.apply(con, wide=redact.enabled_wide(con), only_new=True)
        result.redacted = red.secrets

    # Only now, with the sessions committed: a drop is moved aside once the archive
    # genuinely holds what was in it, never before.
    if archive_drops and getattr(adapter, "surface", None) == "web":
        drops_dir = getattr(adapter, "drops", None)
        if drops_dir is not None:
            result.drops_archived = intake.archive_consumed(
                con, consumed, Path(drops_dir))
            con.commit()

    result.files = stats.files
    result.messages = stats.messages
    result.parts = stats.parts
    result.orphaned = stats.orphaned_messages
    result.blobs = stats.blobs
    result.blob_bytes = stats.blob_bytes
    result.unknown_types = dict(sorted(stats.unknown_types.items(),
                                       key=lambda kv: -kv[1]))
    result.errors = dict(stats.errors)
    result.seconds = time.perf_counter() - t0

    db.record_run(con, adapter.kind, started, result.as_dict())
    return result


def local_host() -> str:
    import socket
    return socket.gethostname()


# Adapters that can be pointed at a tree copied from another machine.
PORTABLE = {"claude_code", "codex", "opencode", "vscode_chat"}


def build_adapters(blobs: BlobStore, only: str | None = None,
                   root: Path | None = None, host: str | None = None,
                   drops: Path | None = None, browser: bool = False) -> list:
    """Instantiate adapters.

    `root` + `host` import a copy taken from a different machine. Since nothing in any
    of these formats records a hostname, `host` is the only way sessions from three
    laptops stay distinguishable.

    `drops` is where the web adapters look for exports. It was never passed before, so
    every drop adapter fell back to its own hard-coded constant pointing at the package's
    own `data/drops` — which meant `--data-dir` moved the database and the blob store
    somewhere else and then read exports from the original tree anyway.
    """
    from ..adapters.chatgpt import ChatGPTAdapter
    from ..adapters.claude_code import ClaudeCodeAdapter
    from ..adapters.claude_web import ClaudeWebAdapter
    from ..adapters.codex import CodexAdapter
    from ..adapters.copilot_web import CopilotWebAdapter
    from ..adapters.deepseek import DeepSeekAdapter
    from ..adapters.gemini import GeminiAdapter
    from ..adapters.grok import GrokAdapter
    from ..adapters.mistral import MistralAdapter
    from ..adapters.opencode import OpenCodeAdapter
    from ..adapters.openrouter import OpenRouterAdapter
    from ..adapters.t3chat import T3ChatAdapter
    from ..adapters.vscode_chat import VSCodeChatAdapter

    host = host or local_host()

    if root is not None:
        if not only:
            raise ValueError("--root requires --source (which tree is this?)")
        cls = {"claude_code": ClaudeCodeAdapter, "codex": CodexAdapter,
               "opencode": OpenCodeAdapter, "vscode_chat": VSCodeChatAdapter}.get(only)
        if cls is None:
            raise ValueError(f"--root is not supported for source {only!r}; "
                             f"portable sources are {sorted(PORTABLE)}")
        return [cls(root=root, blobs=blobs, host=host)]

    all_adapters = [
        ClaudeCodeAdapter(blobs=blobs, host=host),
        CodexAdapter(blobs=blobs, host=host),
        OpenCodeAdapter(blobs=blobs, host=host),
        VSCodeChatAdapter(blobs=blobs, host=host),
        ChatGPTAdapter(blobs=blobs, drops=drops),
        ClaudeWebAdapter(blobs=blobs, drops=drops),   # web chats have no host
        CopilotWebAdapter(blobs=blobs, drops=drops),
        DeepSeekAdapter(blobs=blobs, drops=drops),
        GeminiAdapter(blobs=blobs, drops=drops),
        GrokAdapter(blobs=blobs, drops=drops),
        MistralAdapter(blobs=blobs, drops=drops),
        # `browser` reads the site's own IndexedDB store, which is the only route that
        # is not one manual export click per conversation (§2.1). Opt-in.
        OpenRouterAdapter(blobs=blobs, drops=drops, browser=browser),
        T3ChatAdapter(blobs=blobs, drops=drops),
    ]
    if only:
        return [a for a in all_adapters if a.kind == only]
    return all_adapters


def default_paths(root: Path | None = None) -> tuple[Path, Path]:
    base = root or Path(__file__).resolve().parent.parent.parent / "data"
    return base / "archive.db", base / "blobs"


def drops_dir(root: Path | None = None) -> Path:
    """Where exports land. Derived from the same base as the database and the blobs.

    The drop adapters each carry their own fallback constant for the same folder; this
    is what `--data-dir` uses so that all three move together.
    """
    base = root or Path(__file__).resolve().parent.parent.parent / "data"
    return base / "drops"
