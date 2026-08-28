"""Canonical records every adapter emits.

Adapters know their own native format and nothing else — no database, no search, no
dedup logic. They read files and yield these dataclasses. Everything downstream works
against this vocabulary regardless of which of the ten sources produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

# Parts larger than this are written to the blob store; a head excerpt stays inline.
INLINE_LIMIT = 32 * 1024

# Part kinds.
#
# `thinking` is source-dependent, not universally droppable: Claude Code stores those
# blocks EMPTY (signature only), while the claude.ai web export carries real reasoning
# text in 39 of 66 blocks. So the Claude Code adapter drops them and the web adapter
# keeps them — the decision belongs to the adapter, not to this vocabulary.
KIND_TEXT = "text"
KIND_THINKING = "thinking"
KIND_TOOL_USE = "tool_use"
KIND_TOOL_RESULT = "tool_result"
KIND_IMAGE = "image"
KIND_ATTACHMENT = "attachment"

# What separates a conversational TURN from a tool step.
#
# Sources disagree about what one "message" is. Claude Code writes one record per API
# content block, so a tool call and its result are two messages and the result arrives
# under role='user'; Codex splits the call and its output into an `assistant` and a
# `tool` message; Gemini, T3 Chat, VS Code chat and the web exports put tool_use and
# tool_result INSIDE the assistant message, so a tool call costs no messages at all.
# Counting raw messages therefore inflates agentic sources several-fold against chat
# ones -- 67% of this archive's messages carry no text at all.
#
# A message is a turn when it carries something a person or a model actually said or
# showed. A message made only of tool_use/tool_result is a tool step: real work, but
# not a turn. That makes the count comparable across all ten sources.
TURN_KINDS = frozenset({KIND_TEXT, KIND_THINKING, KIND_IMAGE, KIND_ATTACHMENT})


@dataclass(slots=True)
class Part:
    kind: str
    seq: int
    text: str | None = None
    tool_name: str | None = None
    tool_ok: bool | None = None
    bytes: int = 0                 # TRUE original size, even when `text` is truncated
    blob_sha: str | None = None    # sha256 of overflow content, from BlobStore
    blob_path: str | None = None   # where BlobStore put it
    embed_eligible: bool = False   # the §1.1 rule, materialised
    duration_ms: int | None = None # tool_use only: call -> matching result, when timed

    def __post_init__(self) -> None:
        if not self.bytes and self.text:
            self.bytes = len(self.text.encode("utf-8", errors="replace"))


@dataclass(slots=True)
class Message:
    native_id: str | None
    role: str                      # user | assistant | system | tool
    created_at: int                # epoch ms, UTC
    seq: int = 0
    parent_native_id: str | None = None
    on_active_path: bool = True    # False once a rewind/edit orphans it (risk R1)
    is_sidechain: bool = False
    model: str | None = None
    tok_in: int | None = None
    tok_out: int | None = None
    parts: list[Part] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_turn(self) -> bool:
        """See TURN_KINDS. A part-less message is never a turn."""
        return any(p.kind in TURN_KINDS for p in self.parts)


@dataclass(slots=True)
class Session:
    source_kind: str
    native_id: str
    started_at: int                # epoch ms, UTC
    raw_path: str
    raw_hash: str
    # When the snapshot this came from was taken — the drop's mtime for a web export,
    # the file's mtime for a live local store. Ingest refuses to apply an older
    # snapshot over a newer one, which is what stops a stale re-drop of an export
    # rewriting a session that has grown since. None means "no clock available".
    exported_at: int | None = None
    title: str | None = None
    title_source: str | None = None      # provider | first_prompt | generated
    parent_native_id: str | None = None  # subagent -> the session that spawned it
    host: str | None = None              # which machine this came from; see below
    workspace_key: str | None = None     # CASEFOLDED project root, never a raw cwd
    workspace_label: str | None = None
    model_primary: str | None = None
    ended_at: int | None = None
    tok_in: int | None = None
    tok_out: int | None = None
    tok_cache_read: int | None = None
    tok_cache_write: int | None = None
    cost_usd: float | None = None
    messages: list[Message] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def msg_count(self) -> int:
        """Everything on the active path -- the volume of the session."""
        return sum(1 for m in self.messages if m.on_active_path)

    @property
    def turn_count(self) -> int:
        """Only the conversational turns -- the shape of the session, comparably.

        Always <= msg_count; the difference is tool steps.
        """
        return sum(1 for m in self.messages if m.on_active_path and m.is_turn)


@dataclass(slots=True)
class ParseStats:
    """What an adapter saw. Unknown shapes are counted, never fatal (risk R5)."""
    files: int = 0
    sessions: int = 0
    messages: int = 0
    parts: int = 0
    orphaned_messages: int = 0     # records left off the active path
    unknown_types: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    blobs: int = 0
    blob_bytes: int = 0

    def unknown(self, key: str) -> None:
        self.unknown_types[key] = self.unknown_types.get(key, 0) + 1

    def error(self, key: str) -> None:
        self.errors[key] = self.errors.get(key, 0) + 1


class Adapter(Protocol):
    """Every source implements exactly this."""

    kind: str
    label: str
    surface: str   # 'cli' | 'web' | 'editor_panel'

    def discover(self) -> list:
        """Return the files/roots this adapter can parse on this machine."""

    def parse(self, target, stats: ParseStats) -> Iterator[Session]:
        """Yield canonical Sessions from one discovered target."""
