"""ChatGPT adapter — reads the official account data export.

Export flow: Settings → Data controls → **Export data**, which mails a link to a ZIP:

    <export>.zip
      ├── conversations.json    every chat in the account
      ├── chat.html             the same thing rendered; not read
      ├── user.json             account identity; not read, as with DeepSeek (§8.6)
      ├── message_feedback.json thumbs up/down; not read
      └── file-<id>-<name>.png  the images you uploaded and the ones DALL·E made

Shape — the node map DeepSeek's export imitates, with the pointer DeepSeek is missing:

    [ { id, conversation_id, title, create_time, update_time, current_node,
        default_model_slug,
        mapping: { '<node id>': { id, parent, children: [...],
                                  message: { id, author: {role, name, metadata},
                                             create_time, status, end_turn, weight,
                                             recipient,
                                             content: {content_type, ...},
                                             metadata: {model_slug, ...} } | null } } } ]

Seven things that decide how this is read:

1. **`current_node` says which leaf survived.** DeepSeek's identical tree lacks it and
   has to guess with "newest leaf wins" (§8.1). Here the file states the answer, so the
   active path is walked up from it and the shared resolver is only the fallback for an
   export whose pointer is missing or dangles. That matters because ChatGPT's regenerate
   button is used far more than any other source's, and a guessed leaf silently files
   the abandoned answer as the real one.

2. **`content_type` is nine different shapes, not one.** `text` carries `parts` (a list
   of strings); `code` and `execution_output` carry `text`; `thoughts` carries a list of
   `{summary, content}`; `multimodal_text` mixes image pointers with strings. Reading
   `parts` unconditionally — the obvious implementation — drops every reasoning block
   and every tool call in the archive. Anything not listed in `_PART_BUILDERS` is
   counted in `unknown_types` rather than guessed at.

3. **Hidden system messages are structural, not content.** A conversation's root is
   normally an empty `system` message, and custom-instruction turns carry
   `metadata.is_visually_hidden_from_conversation`. They are dropped as *messages* but
   kept in the *graph*: they are what joins the first real turn to the root, and
   filtering before building the tree shatters the chain — the same trap §8.1 describes
   for Claude Code.

4. **`recipient` is what makes a message a tool call.** The role is still `assistant`
   when it calls `python` or `browser`; only `recipient != 'all'` says so. Without it,
   generated code reads as something the model said out loud, and `is_turn` counts a
   tool step as a conversational turn.

5. **`weight: 0` marks a turn the model was told to forget.** It stays, off the active
   path, for the same reason abandoned branches stay: it is real history.

6. **Images ship as real bytes**, as in Gemini and unlike every web export. The pointer
   is `file-service://file-<id>` and the member is `file-<id>-<name>.<ext>`, so they are
   joined on that id. When the member is missing — DALL·E images age out of the export —
   the part records the reference and the size the export claims, rather than nothing.

7. **No usage anywhere.** No token counts, no cost, on any record. Those columns are a
   real gap for this source, not a zero; `model_slug` is the only usage-adjacent field
   and it names the model, not what it spent.

`create_time` is a float epoch **second** (`1699999999.123`), unlike every other source
here, and is null on system nodes and on messages still streaming when the export ran.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_ATTACHMENT,
    KIND_IMAGE,
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

# `file-service://file-ABC123` / `sediment://file_ABC123`, and the member that holds it.
ASSET_POINTER = re.compile(r"(?:file-service://|sediment://)?(file[-_][A-Za-z0-9]+)")

# recipient on a message addressed to the person rather than to a tool.
TO_USER = "all"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _ts(value) -> int | None:
    """Epoch ms from ChatGPT's float epoch seconds. Null on system and streaming rows."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value * 1000)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _joined(parts) -> str:
    """The string parts of a `parts` list, which may also hold image pointer dicts."""
    if not isinstance(parts, list):
        return ""
    return "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


class ChatGPTAdapter:
    kind = "chatgpt"
    label = "ChatGPT"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Is this drop's conversations.json ChatGPT's rather than one of the other two?

        Three sources ship a file by that name. claude.ai is ruled out by
        `chat_messages`, DeepSeek by `inserted_at` — which it stamps on every node and
        ChatGPT never writes.

        `author` is the positive marker rather than `current_node`, which would be the
        obvious choice and does not work: the sniff window is 64 KB off the front of the
        file, `current_node` is written *after* the whole of the first conversation's
        mapping, and one long chat pushes it far past that. `author` appears inside the
        first message node, a few hundred bytes in.
        """
        head = conversations_head(path)
        return bool(head) and '"mapping"' in head and '"author"' in head \
            and '"chat_messages"' not in head and '"inserted_at"' not in head

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    def _load(self, path: Path) -> tuple[list, str, dict[str, str]]:
        """The conversations, the drop's digest, and the id -> member map for assets."""
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assets: dict[str, str] = {}
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(MEMBER))
                data = json.loads(zf.read(name).decode("utf-8", errors="replace"))
                for member in zf.namelist():
                    match = ASSET_POINTER.match(member.rsplit("/", 1)[-1])
                    if match:
                        assets.setdefault(match.group(1), member)
        else:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        return (data if isinstance(data, list) else []), digest, assets

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            conversations, digest, assets = self._load(path)
        except (json.JSONDecodeError, OSError, StopIteration, zipfile.BadZipFile):
            stats.error(f"unreadable:{path.name}")
            return

        for conv in conversations:
            if not isinstance(conv, dict):
                stats.unknown(f"{self.kind}:conversation-not-an-object")
                continue
            if not (conv.get("id") or conv.get("conversation_id")):
                stats.unknown(f"{self.kind}:conversation-without-id")
                continue
            session = self._session(path, digest, assets, conv, stats)
            if session is not None:
                stats.sessions += 1
                yield session

    def _session(self, path: Path, digest: str, assets: dict[str, str],
                 conv: dict, stats: ParseStats) -> Session | None:
        mapping = conv.get("mapping")
        nodes = {str(k): v for k, v in mapping.items()
                 if isinstance(v, dict)} if isinstance(mapping, dict) else {}
        if not nodes:
            stats.unknown(f"{self.kind}:conversation-without-mapping")
            return None

        active = self._active(nodes, conv.get("current_node"))

        messages: list[Message] = []
        seq = 0
        for i, nid in enumerate(self._walk(nodes)):
            msg = self._build_message(nodes[nid], nid, path, assets, stats)
            if msg is None:
                continue
            # `and msg.on_active_path` rather than a plain assignment: a weight-0 turn
            # has already taken itself off the path (point 5), and it sits on the
            # surviving chain, so overwriting here would put it straight back on.
            msg.on_active_path = nid in active and msg.on_active_path
            msg.seq = seq if msg.on_active_path else i
            messages.append(msg)
            stats.messages += 1
            stats.parts += len(msg.parts)
            if msg.on_active_path:
                seq += 1
            else:
                stats.orphaned_messages += 1

        if not messages:
            return None

        answers = [m for m in messages if m.role == "assistant"]
        models = sorted({m.model for m in answers if m.model})
        times = [m.created_at for m in messages if m.created_at]
        title = _text(conv.get("title")) or None

        meta = {
            "models": models,
            "branched": any(not m.on_active_path for m in messages),
            "tools": sorted({p.tool_name for m in messages for p in m.parts
                             if p.kind == KIND_TOOL_USE and p.tool_name}),
            "export_file": path.name,
            "export_digest": digest[:16],       # which drop this came from
        }
        if conv.get("is_archived"):
            meta["archived"] = True
        if len(models) == 1:
            # Drives the BY ASSISTANT breakdown, which holds one value per session.
            meta["participant"] = models[0]
            meta["participant_label"] = models[0]

        return Session(
            source_kind=self.kind,
            native_id=str(conv.get("conversation_id") or conv["id"]),
            title=title,
            title_source="provider" if title else None,
            started_at=_ts(conv.get("create_time")) or (min(times) if times else 0),
            ended_at=_ts(conv.get("update_time")) or (max(times) if times else None),
            raw_path=str(path),
            exported_at=taken_at(path),
            # Hash this conversation, not the export: the file is account-wide, so
            # hashing the whole document reports every chat as changed whenever one of
            # them grows. `digest` still salts it so a re-download stays traceable.
            raw_hash=hashlib.sha256(
                json.dumps(conv, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            model_primary=self._primary(answers, conv),
            messages=messages,      # no token counts and no cost in this export
            meta=meta,
        )

    # -- tree --------------------------------------------------------------

    @staticmethod
    def _parent_of(nodes: dict, nid: str) -> str | None:
        parent = nodes[nid].get("parent")
        return str(parent) if parent is not None else None

    def _active(self, nodes: dict, current: object) -> set:
        """The surviving root-to-leaf path.

        Walked up from `current_node`, which is the export stating the answer outright.
        Falls back to the shared resolver's "newest leaf wins" when the pointer is
        absent or names a node that is not in the mapping — both of which happen in
        exports taken while a response was still streaming.
        """
        cursor = str(current) if current is not None else None
        if cursor in nodes:
            active: set = set()
            while cursor is not None and cursor in nodes and cursor not in active:
                active.add(cursor)
                cursor = self._parent_of(nodes, cursor)
            return active

        from ._tree import resolve_active_path
        return resolve_active_path(
            nodes.keys(),
            lambda n: self._parent_of(nodes, n),
            lambda n: (_ts((nodes[n].get("message") or {}).get("create_time")) or 0,
                       self._depth(nodes, n)),
        )

    @classmethod
    def _depth(cls, nodes: dict, nid: str) -> int:
        """Distance to the root, used only to break a timestamp tie between leaves."""
        depth, cursor, seen = 0, nid, {nid}
        while True:
            parent = cls._parent_of(nodes, cursor)
            if parent is None or parent not in nodes or parent in seen:
                return depth
            depth, cursor = depth + 1, parent
            seen.add(parent)

    @classmethod
    def _walk(cls, nodes: dict) -> list[str]:
        """Node ids in reading order: depth-first from each root, children as listed.

        The parent → child links are the only ordering this format guarantees.
        `create_time` is null on system nodes and on anything that was still streaming,
        so sorting by it would file those first.
        """
        roots = [n for n in nodes if cls._parent_of(nodes, n) not in nodes]
        order: list[str] = []
        seen: set[str] = set()
        stack = list(reversed(roots))
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            order.append(nid)
            children = [str(c) for c in (nodes[nid].get("children") or [])]
            stack.extend(reversed([c for c in children if c in nodes and c not in seen]))
        # A node orphaned by a cycle or a broken link is still content; keep it last
        # rather than lose it.
        order.extend(n for n in nodes if n not in seen)
        return order

    # -- messages ----------------------------------------------------------

    def _build_message(self, node: dict, nid: str, path: Path,
                       assets: dict[str, str], stats: ParseStats) -> Message | None:
        raw = node.get("message")
        if not isinstance(raw, dict):
            return None                     # the empty root node

        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        author = raw.get("author") if isinstance(raw.get("author"), dict) else {}
        role = str(author.get("role") or "unknown")

        # Point 3: dropped as a message, already counted in the graph above.
        if meta.get("is_visually_hidden_from_conversation"):
            return None
        if role == "system" and not _joined((raw.get("content") or {}).get("parts")):
            return None

        recipient = str(raw.get("recipient") or TO_USER)
        message = Message(
            native_id=str(raw.get("id") or nid),
            parent_native_id=(str(node["parent"])
                              if node.get("parent") is not None else None),
            role=role,
            model=meta.get("model_slug") or meta.get("default_model_slug"),
            created_at=_ts(raw.get("create_time")) or 0,
        )
        if recipient != TO_USER:
            message.meta["recipient"] = recipient
        if author.get("name"):
            message.meta["author_name"] = author["name"]
        # Point 5: kept, but off the path — a turn the model was told to forget.
        if raw.get("weight") == 0:
            message.on_active_path = False
            message.meta["weight"] = 0

        content = raw.get("content")
        if isinstance(content, dict):
            self._build_parts(content, message, recipient, path, assets, stats)

        for att in (meta.get("attachments") or []):
            if not isinstance(att, dict):
                continue
            message.parts.append(Part(
                kind=KIND_ATTACHMENT, seq=len(message.parts),
                text=str(att.get("name") or "attachment"),
                bytes=att.get("size") or 0))

        return message if message.parts else None

    def _build_parts(self, content: dict, msg: Message, recipient: str, path: Path,
                     assets: dict[str, str], stats: ParseStats) -> None:
        """Point 2: nine content shapes, each with its own place to look for the text."""
        ctype = str(content.get("content_type"))

        if ctype == "text":
            text = _joined(content.get("parts"))
            if not text:
                return
            # Point 4: role stays `assistant` on a tool call; `recipient` is the tell.
            if recipient != TO_USER:
                msg.parts.append(Part(
                    kind=KIND_TOOL_USE, seq=len(msg.parts), tool_name=recipient,
                    text=text[:2000], bytes=len(text.encode("utf-8")),
                    embed_eligible=True))
            else:
                msg.parts.append(self._offload(Part(
                    kind=KIND_TEXT, seq=len(msg.parts), text=text,
                    embed_eligible=True), stats))
            return

        if ctype == "multimodal_text":
            for item in content.get("parts") or []:
                if isinstance(item, str) and item.strip():
                    msg.parts.append(self._offload(Part(
                        kind=KIND_TEXT, seq=len(msg.parts), text=item.strip(),
                        embed_eligible=True), stats))
                elif isinstance(item, dict):
                    part = self._image(item, len(msg.parts), path, assets, stats)
                    if part is not None:
                        msg.parts.append(part)
            return

        if ctype == "code":
            text = _text(content.get("text"))
            if text:
                msg.parts.append(self._offload(Part(
                    kind=KIND_TOOL_USE, seq=len(msg.parts),
                    tool_name=recipient if recipient != TO_USER else "code",
                    text=text, embed_eligible=True), stats))
            return

        if ctype in ("execution_output", "tether_browsing_display", "system_error"):
            text = _text(content.get("text")) or _text(content.get("result"))
            if text:
                msg.parts.append(self._offload(Part(
                    kind=KIND_TOOL_RESULT, seq=len(msg.parts),
                    tool_name=recipient if recipient != TO_USER else None,
                    tool_ok=ctype != "system_error",
                    text=text, embed_eligible=False), stats))   # §1.1
            return

        if ctype == "thoughts":
            # Real reasoning prose, so kept and embedded — as in claude.ai and unlike
            # Claude Code's signature-only blocks.
            blocks = content.get("thoughts")
            text = "\n\n".join(
                "\n".join(filter(None, (_text(t.get("summary")), _text(t.get("content")))))
                for t in blocks if isinstance(t, dict)) if isinstance(blocks, list) else ""
            if text.strip():
                msg.parts.append(self._offload(Part(
                    kind=KIND_THINKING, seq=len(msg.parts), text=text.strip(),
                    embed_eligible=True), stats))
            return

        if ctype == "reasoning_recap":
            text = _text(content.get("content"))
            if text:
                msg.parts.append(self._offload(Part(
                    kind=KIND_THINKING, seq=len(msg.parts), text=text,
                    embed_eligible=True), stats))
            return

        if ctype == "tether_quote":
            text = "\n".join(filter(None, (
                _text(content.get("title")), _text(content.get("url")),
                _text(content.get("text")))))
            if text:
                msg.parts.append(self._offload(Part(
                    kind=KIND_TOOL_RESULT, seq=len(msg.parts), tool_name="browser",
                    tool_ok=True, text=text, embed_eligible=False), stats))
            return

        if ctype in ("user_editable_context", "model_editable_context"):
            text = "\n".join(filter(None, (
                _text(content.get("user_profile")),
                _text(content.get("user_instructions")),
                _text(content.get("model_set_context")))))
            if text:
                msg.parts.append(self._offload(Part(
                    kind=KIND_TEXT, seq=len(msg.parts), text=text,
                    embed_eligible=True), stats))
            return

        stats.unknown(f"{self.kind}:content:{ctype}")

    def _image(self, item: dict, seq: int, path: Path, assets: dict[str, str],
               stats: ParseStats) -> Part | None:
        """Point 6: join the asset pointer to the member that holds the bytes."""
        if str(item.get("content_type")) != "image_asset_pointer":
            stats.unknown(f"{self.kind}:multimodal:{item.get('content_type')}")
            return None

        pointer = str(item.get("asset_pointer") or "")
        match = ASSET_POINTER.search(pointer)
        member = assets.get(match.group(1)) if match else None
        size = item.get("size_bytes") or 0

        part = Part(kind=KIND_IMAGE, seq=seq, bytes=size,
                    text=f"image {item.get('width')}x{item.get('height')}".strip())

        if member and self.blobs is not None and path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    data = zf.read(member)
            except (OSError, KeyError, zipfile.BadZipFile):
                data = None
            if data:
                stored = self.blobs.put_bytes(data)
                if stored:
                    sha, stored_size, dest = stored
                    part.blob_sha, part.blob_path = sha, dest
                    part.bytes = stored_size
                    stats.blobs += 1
                    stats.blob_bytes += stored_size
        elif not member:
            # DALL·E output ages out of the export; the reference is all there is.
            part.text = f"{part.text} (not in export)".strip()
        return part

    def _offload(self, part: Part, stats: ParseStats | None = None) -> Part:
        """Oversized parts go to the blob store; a head excerpt stays inline."""
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
    def _primary(answers: list[Message], conv: dict) -> str | None:
        """The model that did most of the talking, or the chat's declared default."""
        counts: dict[str, int] = {}
        for msg in answers:
            if msg.model:
                counts[msg.model] = counts.get(msg.model, 0) + 1
        if counts:
            return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return _text(conv.get("default_model_slug")) or None
