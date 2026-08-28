"""claude.ai adapter — reads the official data export.

Export flow, which is easy to get wrong: the emailed "Download data" button returns a
1.2 KB *manifest* listing four single-use URLs. The conversations live behind one of
them, in `conversations-000.zip::conversations.json`. The manifest itself contains no
chat content at all.

Shape:

    [ { uuid, name, summary, created_at, updated_at, account: {uuid},
        chat_messages: [
          { uuid, parent_message_uuid, sender: 'human'|'assistant',
            created_at, text, content: [...], attachments: [], files: [] } ] } ]

`parent_message_uuid` makes this a tree, same as Claude Code — 3 of 53 conversations
branch and 2 have multiple roots, so the shared resolver is required here too.

Unlike Claude Code, **thinking blocks carry real text** (39 of 66, ~32 KB) and are kept.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_ATTACHMENT,
    KIND_TEXT,
    KIND_THINKING,
    KIND_TOOL_RESULT,
    KIND_TOOL_USE,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import MEMBER, by_recency, candidates, conversations_head, taken_at

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"


def _ts(value) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


class ClaudeWebAdapter:
    kind = "claude_web"
    label = "Claude.ai"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this one file look like a claude.ai export?

        The filename is not the marker: DeepSeek's export ships a `conversations.json`
        too, and matching on the name alone claimed it for this adapter — which then
        read it as 53 conversations without `chat_messages` and yielded nothing, in
        silence. `chat_messages` is what makes the file Claude's.

        Lifted out of `discover` so `core.intake` can ask the same question about a
        single arbitrary path — a file someone dropped on the import page, or one
        sitting in ~/Downloads — without a second copy of the rule to keep in step.
        """
        return '"chat_messages"' in (conversations_head(path) or "")

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    def _load(self, path: Path) -> tuple[list, str]:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(MEMBER))
                data = json.loads(zf.read(name).decode("utf-8", errors="replace"))
        else:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        return (data if isinstance(data, list) else []), digest

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        from ._tree import resolve_active_path

        stats.files += 1
        try:
            conversations, digest = self._load(path)
        except (json.JSONDecodeError, OSError, StopIteration, zipfile.BadZipFile):
            stats.error(f"unreadable:{path.name}")
            return

        for conv in conversations:
            if not isinstance(conv, dict) or not conv.get("uuid"):
                stats.unknown(f"{self.kind}:conversation-without-uuid")
                continue

            raw_msgs = [m for m in (conv.get("chat_messages") or [])
                        if isinstance(m, dict) and m.get("uuid")]
            if not raw_msgs:
                continue

            by_id = {m["uuid"]: m for m in raw_msgs}
            active = resolve_active_path(
                by_id.keys(),
                lambda u: by_id[u].get("parent_message_uuid"),
                lambda u: str(by_id[u].get("created_at") or ""),
            )

            messages: list[Message] = []
            seq = 0
            for i, raw in enumerate(raw_msgs):
                on_path = raw["uuid"] in active
                msg = self._build_message(raw, seq if on_path else i, on_path, stats)
                if msg is None:
                    continue
                messages.append(msg)
                stats.messages += 1
                stats.parts += len(msg.parts)
                if on_path:
                    seq += 1
                else:
                    stats.orphaned_messages += 1

            if not messages:
                continue

            times = [m.created_at for m in messages if m.created_at]
            title = conv.get("name") or None
            stats.sessions += 1
            yield Session(
                source_kind=self.kind,
                native_id=conv["uuid"],
                title=title,
                title_source="provider" if title else None,
                started_at=_ts(conv.get("created_at")) or (min(times) if times else 0),
                ended_at=_ts(conv.get("updated_at")) or (max(times) if times else None),
                raw_path=str(path),
                exported_at=taken_at(path),
                # Hash this conversation, not the whole archive: a fresh monthly export
                # otherwise reports every conversation as changed and rewrites all of
                # them. `digest` still salts it so a re-download is traceable.
                raw_hash=hashlib.sha256(
                    json.dumps(conv, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                messages=messages,
                meta={
                    "summary": conv.get("summary") or None,
                    "account": (conv.get("account") or {}).get("uuid"),
                    "branched": len(messages) - len(active) > 0,
                    "export_digest": digest[:16],   # which drop this came from
                },
            )

    def _build_message(self, raw: dict, seq: int, on_path: bool,
                       stats: ParseStats) -> Message | None:
        sender = raw.get("sender")
        role = {"human": "user", "assistant": "assistant"}.get(sender, str(sender))

        msg = Message(
            native_id=raw["uuid"],
            parent_native_id=raw.get("parent_message_uuid"),
            seq=seq,
            on_active_path=on_path,
            role=role,
            created_at=_ts(raw.get("created_at")) or 0,
        )

        blocks = raw.get("content")
        if not isinstance(blocks, list) or not blocks:
            # Older exports carry only the flat `text` field.
            text = (raw.get("text") or "").strip()
            if text:
                msg.parts.append(self._offload(
                    Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True), stats))
            return msg if msg.parts else None

        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            part = self._build_part(blk, len(msg.parts), stats)
            if part is not None:
                msg.parts.append(part)

        for att in (raw.get("attachments") or []) + (raw.get("files") or []):
            if isinstance(att, dict):
                name = att.get("file_name") or att.get("name") or "attachment"
                extracted = att.get("extracted_content") or ""
                msg.parts.append(self._offload(Part(
                    kind=KIND_ATTACHMENT, seq=len(msg.parts),
                    text=f"{name}\n{extracted}".strip() if extracted else name,
                    bytes=att.get("file_size") or len(extracted),
                    embed_eligible=bool(extracted)), stats))

        return msg if msg.parts else None

    def _build_part(self, blk: dict, seq: int, stats: ParseStats) -> Part | None:
        btype = str(blk.get("type"))

        if btype == "text":
            text = (blk.get("text") or "").strip()
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_TEXT, seq=seq, text=text, embed_eligible=True), stats)

        if btype == "thinking":
            # Real content here, unlike Claude Code. Worth keeping and worth embedding.
            text = (blk.get("thinking") or "").strip()
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_THINKING, seq=seq, text=text, embed_eligible=True), stats)

        if btype == "tool_use":
            name = str(blk.get("name") or "tool")
            payload = blk.get("input")
            return Part(
                kind=KIND_TOOL_USE, seq=seq, tool_name=name,
                text=self._summarise(payload),
                bytes=len(json.dumps(payload or {}, ensure_ascii=False)),
                embed_eligible=True)

        if btype == "tool_result":
            content = blk.get("content")
            if isinstance(content, list):
                text = " ".join(c.get("text", "") for c in content
                                if isinstance(c, dict))
            elif isinstance(content, str):
                text = content
            else:
                text = json.dumps(content, ensure_ascii=False) if content else ""
            return self._offload(Part(
                kind=KIND_TOOL_RESULT, seq=seq, text=text,
                tool_name=blk.get("name"),
                tool_ok=not blk.get("is_error"),
                embed_eligible=False), stats)   # §1.1

        if btype in ("image", "document"):
            return Part(kind=KIND_ATTACHMENT, seq=seq,
                        bytes=len(json.dumps(blk.get("source") or {})))

        stats.unknown(f"{self.kind}:content:{btype}")
        return None

    def _offload(self, part: Part, stats: ParseStats | None = None) -> Part:
        """Oversized parts go to the blob store; a head excerpt stays inline.

        `stats` is what makes the run report say `blobs 3 (1.2 MB)` rather than
        `blobs 0` while quietly writing three of them.
        """
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                if stats is not None:
                    stats.blobs += 1
                    stats.blob_bytes += size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part

    @staticmethod
    def _summarise(value) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("query", "command", "prompt", "path", "url", "description", "code"):
            if isinstance(value.get(key), str) and value[key].strip():
                return f"{key}: {value[key][:300]}"
        keys = ", ".join(sorted(value)[:6])
        return f"({keys})" if keys else ""
