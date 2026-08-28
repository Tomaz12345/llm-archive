"""Mistral (Le Chat) adapter — reads the per-chat "Export chat" ZIP.

Export flow: chat UI → ⋯ menu → **Export chat** → `chat-export-<epoch ms>.zip`, holding
one member per chat named `chat-<chatId>.json`. Per-chat, like OpenRouter (§2.1) and
unlike the account-wide bulk exports, so drops accumulate one ZIP per conversation and
the same chat gets re-exported whenever it grows.

Shape — the flattest format in the archive. The member is not an object with a chat
around it, it is a **bare array of message records**:

    [ { id, version, chatId, role, createdAt,
        content: '<the whole turn as one string>',
        contentChunks: [ {text, type, _context: {type, startTime, endTime}} ] | null,
        context: {completionTiming: {...}, assistantAnswerSignals: {...}},
        reaction, reactionDetail, preference, preferenceOver, canvas: [], files: [] } ]

No tree, no parent pointers, no `mapping`, no `current_node` — the ordering problem §8.1
is about does not exist here. What it does instead:

1. **No chat object, so no title and no chat timestamps.** The array is all there is;
   `chatId` is repeated on every message and is the only conversation identity in the
   file. Title comes from the first prompt (`title_source: 'first_prompt'`), and the
   session's clock is derived from the messages.

2. **`content` and `contentChunks` say the same thing twice.** On the verified assistant
   turn the final chunk is byte-identical to `content` (3,617 chars each). Storing both
   would index every answer twice in FTS — the same trap Grok's triplicated search hits
   set (§2.5). Chunks win when present because they are the only place reasoning
   survives; `content` is the fallback for turns that carry no chunks at all, which is
   every user message (`contentChunks: null`).

3. **Reasoning is a chunk that looks exactly like text.** `type` is `'text'` on both the
   reasoning chunk and the answer chunk; what separates them is `_context.type ==
   'reasoning'`. Read `type` alone and 840 characters of the model's planning get filed
   as part of its answer. It is real prose, so it is kept and embedded, as in claude.ai
   and OpenRouter and unlike Claude Code's signature-only blocks.

4. **Output tokens only, and a clock that starts before the answer exists.**
   `context.completionTiming.generationStats.outputTokens` is the completion count;
   nothing in the export records prompt tokens, so `tok_in` stays NULL rather than being
   guessed. `createdAt` on an assistant turn is stamped when generation *started* —
   24.4 s before it finished in the verified chat — so `ended_at` comes from
   `completedAtMs` where the export offers it.

5. **No model, anywhere.** Not on the message, not on the chat, not in `context`. Le Chat
   records that an assistant answered and never which model did, so `model_primary` and
   every `model` column stay empty and the source itself is named as the participant —
   the same accommodation Gemini needs (§2.4). No cost figure either.

`version` is 0 on every verified record. It is treated as an edit counter — same id,
higher number wins, losers stay as orphans — which is **unverified**: no edited turn
exists in the sample. It fires a counted `unknown` when it ever triggers, which is how
the next export will tell us whether the reading was right. `canvas` and `files` are
empty arrays throughout and are counted the same way rather than guessed at.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_ATTACHMENT,
    KIND_TEXT,
    KIND_THINKING,
    KIND_TOOL_USE,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import by_recency, candidates, taken_at

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"

# Members that hold a chat. The ZIP is named for the moment of export, not for the chat,
# so the conversation id is only ever in here.
MEMBER_PREFIX = "chat-"
MEMBER_SUFFIX = ".json"

# A per-chat export is tens of KB; the whole verified drop is 10 KB. Anything past this
# is a bulk export from some other source, and sniffing it would mean parsing it.
MAX_EXPORT_BYTES = 8 * 1024 * 1024
SNIFF_BYTES = 1 << 16

# `_context.type` on a chunk. 'reasoning' is the model thinking, and it is the only
# thing separating that from the answer — see point 3 above.
CTX_REASONING = "reasoning"

CHUNK_TEXT = "text"

# Default reaction on an un-rated turn; recorded only when the user actually voted.
REACTION_NEUTRAL = "neutral"


def _ts(value) -> int | None:
    """Epoch ms, UTC. The export stamps `Z`; a naive stamp is read as UTC."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_export(head: str) -> bool:
    """Le Chat's array of message records, judged from a window off the front.

    `chatId` beside `contentChunks` is the pair no other source in the archive writes:
    T3 Chat has `threadId`, DeepSeek and ChatGPT nest a `mapping`, OpenRouter declares
    an `orpg` version. The leading `[` keeps this from claiming an object-shaped export
    that happens to mention either word.
    """
    return (head.lstrip().startswith("[")
            and '"chatId"' in head and '"contentChunks"' in head)


class MistralAdapter:
    kind = "mistral"
    label = "Mistral"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this drop hold at least one Le Chat member? Payload, not name.

        `chat-export-<epoch ms>.zip` says when the export was taken and nothing about
        who produced it, so — as with Grok's bare-uuid ZIP — the name cannot be the
        marker. Each candidate member is read one page deep and never parsed, so a drop
        from some other source costs a seek rather than a parse.
        """
        try:
            if path.is_dir() or path.stat().st_size > MAX_EXPORT_BYTES:
                return False
        except OSError:
            return False
        return any(_is_export(head) for head in MistralAdapter._heads(path))

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    @staticmethod
    def _members(path: Path) -> list[str]:
        try:
            with zipfile.ZipFile(path) as zf:
                return [n for n in zf.namelist()
                        if n.rsplit("/", 1)[-1].startswith(MEMBER_PREFIX)
                        and n.endswith(MEMBER_SUFFIX)]
        except (OSError, zipfile.BadZipFile, RuntimeError):
            return []

    @classmethod
    def _heads(cls, path: Path, limit: int = SNIFF_BYTES) -> Iterator[str]:
        """First `limit` decoded bytes of each candidate member inside `path`.

        A bare `chat-<id>.json` is read directly: the ZIP is trivial to unpack by hand,
        and one that arrives unzipped should not read as "no Mistral data".
        """
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in cls._members(path):
                        with zf.open(name) as fh:
                            yield fh.read(limit).decode("utf-8", errors="replace")
            except (OSError, zipfile.BadZipFile, RuntimeError):
                return
        elif (path.suffix.lower() == MEMBER_SUFFIX
              and path.name.startswith(MEMBER_PREFIX)):
            try:
                with path.open("rb") as fh:
                    yield fh.read(limit).decode("utf-8", errors="replace")
            except OSError:
                return

    def _chats(self, path: Path) -> Iterator[tuple[str, list]]:
        """(member name, message array) for every chat in the drop."""
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in self._members(path):
                    data = json.loads(zf.read(name).decode("utf-8", errors="replace"))
                    if isinstance(data, list):
                        yield name.rsplit("/", 1)[-1], data
        else:
            data = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
            if isinstance(data, list):
                yield path.name, data

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            chats = list(self._chats(path))
        except (json.JSONDecodeError, OSError, zipfile.BadZipFile, RuntimeError):
            stats.error(f"unreadable:{path.name}")
            return

        for member, records in chats:
            session = self._session(path, member, records, stats)
            if session is not None:
                stats.sessions += 1
                yield session

    def _session(self, path: Path, member: str, records: list,
                 stats: ParseStats) -> Session | None:
        raw = [r for r in records if isinstance(r, dict)]
        if not raw:
            stats.unknown(f"{self.kind}:chat-without-messages")
            return None

        # Point 1: the conversation id lives on the messages, not above them. The member
        # name carries it too, so a chat whose records somehow lack it still lands under
        # the id the export filed it as.
        chat_id = next((_text(r.get("chatId")) for r in raw if _text(r.get("chatId"))), "")
        native_id = chat_id or self._id_from_member(member)
        if not chat_id:
            stats.unknown(f"{self.kind}:chat-without-id")

        superseded = self._superseded(raw, stats)
        # Document order is the true order here — the array is written as the thread
        # reads — but `createdAt` agrees with it and a re-serialised export need not
        # preserve position, so the stamp leads and position breaks its ties.
        ordered = sorted(enumerate(raw),
                         key=lambda kv: (_ts(kv[1].get("createdAt")) or 0, kv[0]))

        messages: list[Message] = []
        seq = 0
        for i, (_, record) in enumerate(ordered):
            on_path = id(record) not in superseded
            msg = self._build_message(record, seq if on_path else i, on_path, stats)
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
            return None

        answers = [m for m in messages if m.on_active_path and m.role == "assistant"]
        starts = [m.created_at for m in messages if m.created_at]
        # Point 4: an assistant turn is stamped when generation began, so the last
        # `createdAt` understates the end of the chat by however long the answer took.
        ends = starts + [m.meta["completed_at"] for m in messages
                         if m.meta.get("completed_at")]
        tok_out = sum(m.tok_out or 0 for m in answers)

        meta = {
            # Point 5: no model on any record, so the model columns stay empty and the
            # app is named as the participant instead — as for Gemini.
            "model_recorded": False,
            "participant": self.kind,
            "participant_label": self.label,
            "reactions": sorted({m.meta["reaction"] for m in messages
                                 if m.meta.get("reaction")}),
            "export_file": path.name,
            "export_member": member,
        }

        return Session(
            source_kind=self.kind,
            native_id=native_id,
            title=self._title(messages),
            title_source="first_prompt",
            started_at=min(starts) if starts else 0,
            ended_at=max(ends) if ends else None,
            raw_path=str(path),
            exported_at=taken_at(path),
            # Hash this chat's records, not the drop: the same chat re-exported lands
            # under a different filename (the ZIP is named for the export instant), and
            # a multi-chat drop would otherwise report every chat as changed whenever
            # any one of them grew.
            raw_hash=hashlib.sha256(
                json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            # No model_primary, no tok_in and no cost — none of the three is exported.
            tok_out=tok_out or None,
            messages=messages,
            meta=meta,
        )

    @staticmethod
    def _id_from_member(member: str) -> str:
        stem = member.rsplit("/", 1)[-1]
        if stem.startswith(MEMBER_PREFIX) and stem.endswith(MEMBER_SUFFIX):
            return stem[len(MEMBER_PREFIX):-len(MEMBER_SUFFIX)] or stem
        return stem

    def _superseded(self, raw: list[dict], stats: ParseStats) -> set[int]:
        """Records replaced by a higher `version` of the same message id.

        UNVERIFIED — `version` is 0 on every record in the sample, so this has never
        fired. It is the only field shaped like an edit counter, and an edited turn
        arriving as a second record under the same id is the failure that would
        otherwise read back as the same question asked twice. Counted when it triggers,
        so the next export says whether the reading was right.
        """
        best: dict[str, dict] = {}
        for record in raw:
            mid = _text(record.get("id"))
            if not mid:
                continue
            held = best.get(mid)
            if held is None:
                best[mid] = record
                continue
            stats.unknown(f"{self.kind}:duplicate-message-id")
            if self._version(record) >= self._version(held):
                best[mid] = record
        kept = {id(r) for r in best.values()}
        return {id(r) for r in raw if _text(r.get("id")) and id(r) not in kept}

    @staticmethod
    def _version(record: dict) -> int:
        value = record.get("version")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    # -- messages ----------------------------------------------------------

    def _build_message(self, raw: dict, seq: int, on_path: bool,
                       stats: ParseStats) -> Message | None:
        role = str(raw.get("role") or "").lower()
        if role not in ("user", "assistant", "system"):
            stats.unknown(f"{self.kind}:role:{role or 'missing'}")
            role = role or "assistant"

        mid = _text(raw.get("id")) or None
        msg = Message(
            # A superseded record needs an id of its own: message rows are unique on
            # (session, native_id), so two versions filed under one id would collide.
            native_id=mid if (on_path or not mid) else f"{mid}:v{self._version(raw)}",
            role=role,
            created_at=_ts(raw.get("createdAt")) or 0,
            seq=seq,
            on_active_path=on_path,
            model=None,                 # point 5: nothing records which model answered
        )

        for part in self._build_parts(raw, stats):
            part.seq = len(msg.parts)
            msg.parts.append(part)

        self._apply_context(msg, raw.get("context"), role, stats)

        reaction = _text(raw.get("reaction"))
        if reaction and reaction != REACTION_NEUTRAL:
            msg.meta["reaction"] = reaction
            if _text(raw.get("reactionComment")):
                msg.meta["reaction_comment"] = _text(raw["reactionComment"])
        # The A/B picker: which sibling answer the user preferred. Null throughout the
        # sample, kept because it is the only record that a comparison happened at all.
        if raw.get("preference") is not None:
            msg.meta["preference"] = raw["preference"]
        if raw.get("preferenceOver") is not None:
            msg.meta["preference_over"] = raw["preferenceOver"]
        if not on_path:
            msg.meta["superseded_by_version"] = True

        return msg if msg.parts else None

    def _apply_context(self, msg: Message, context, role: str,
                       stats: ParseStats) -> None:
        timing = (context or {}).get("completionTiming") \
            if isinstance(context, dict) else None
        if not isinstance(timing, dict):
            if role == "assistant":
                # Every sample answer carries one; its absence is worth knowing about,
                # since `ended_at` and `tok_out` both come out of it.
                stats.unknown(f"{self.kind}:answer-without-timing")
            return

        completed = _ts(timing.get("completedAtMs"))
        if completed:
            msg.meta["completed_at"] = completed
        for key, name in (("completionDurationMs", "duration_ms"),
                          ("timeToFirstTokenMs", "ttft_ms")):
            if isinstance(timing.get(key), (int, float)):
                msg.meta[name] = timing[key]

        gen = timing.get("generationStats")
        if isinstance(gen, dict):
            # Point 4: output only. `tok_in` is a real gap in this export, not a zero.
            if isinstance(gen.get("outputTokens"), int):
                msg.tok_out = gen["outputTokens"]
            if isinstance(gen.get("outputTokensPerSecond"), (int, float)):
                msg.meta["tokens_per_second"] = round(gen["outputTokensPerSecond"], 2)
            count = gen.get("completionCount")
            if isinstance(count, int) and count > 1:
                msg.meta["completions"] = count

    # -- parts -------------------------------------------------------------

    def _build_parts(self, raw: dict, stats: ParseStats) -> list[Part]:
        """Point 2: chunks when there are any, `content` only as the fallback.

        Never both — the last chunk of the verified answer is byte-identical to
        `content`, so emitting the two would put every answer into FTS twice.
        """
        parts = self._chunk_parts(raw.get("contentChunks"), stats)
        if not parts:
            text = _text(raw.get("content"))
            if text:
                parts.append(self._offload(
                    Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True), stats))

        parts.extend(self._signal_parts(raw.get("context"), stats))
        parts.extend(self._file_parts(raw.get("files"), stats))

        # Canvas documents are their own entity in Le Chat and the array is empty in
        # every sample chat, so their shape is unverified. Counted loudly rather than
        # dropped in silence — a non-zero here is the signal to come back with a fixture.
        canvas = raw.get("canvas")
        if isinstance(canvas, list) and canvas:
            stats.unknown(f"{self.kind}:canvas-not-parsed")
        return parts

    def _chunk_parts(self, chunks, stats: ParseStats) -> list[Part]:
        if not isinstance(chunks, list):
            return []
        parts: list[Part] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            ctype = str(chunk.get("type") or "")
            if ctype != CHUNK_TEXT:
                # Tool calls and citations will likely arrive as chunk types of their
                # own once a chat uses them; none does yet.
                stats.unknown(f"{self.kind}:chunk:{ctype or 'missing'}")
                continue
            text = _text(chunk.get("text"))
            if not text:
                continue
            ctx = chunk.get("_context")
            # Point 3: `type` is 'text' on reasoning too. `_context.type` is the tell.
            reasoning = isinstance(ctx, dict) and str(ctx.get("type") or "") == CTX_REASONING
            parts.append(self._offload(Part(
                kind=KIND_THINKING if reasoning else KIND_TEXT, seq=0, text=text,
                embed_eligible=True), stats))       # real prose either way, so embedded
        return parts

    def _signal_parts(self, context, stats: ParseStats) -> list[Part]:
        """`assistantAnswerSignals` — the names of the tools a turn used, and no more.

        Empty in every sample answer. The export records that a tool ran and never its
        arguments or its result, so a call becomes a bare marker: enough for the tool
        breakdown to count it, with nothing to put in the index (§1.1).
        """
        if not isinstance(context, dict):
            return []
        signals = context.get("assistantAnswerSignals")
        if not isinstance(signals, dict):
            return []
        names = [_text(n) for key in ("toolNames", "integrationNames")
                 for n in (signals.get(key) or []) if _text(n)]
        if names:
            stats.unknown(f"{self.kind}:tool-signals-without-payload")
        return [Part(kind=KIND_TOOL_USE, seq=0, tool_name=name, tool_ok=True,
                     embed_eligible=False) for name in names]

    def _file_parts(self, files, stats: ParseStats) -> list[Part]:
        """Attachment names. The ZIP holds chat JSON and nothing else, so the bytes a
        message refers to are not in the export — unlike Gemini's, which ships them."""
        if not isinstance(files, list) or not files:
            return []
        stats.unknown(f"{self.kind}:attachment-bytes-not-exported")
        parts = []
        for entry in files:
            name = (_text(entry.get("name")) or _text(entry.get("fileName"))
                    or _text(entry.get("id"))) if isinstance(entry, dict) else _text(entry)
            if name:
                parts.append(Part(kind=KIND_ATTACHMENT, seq=0, text=name,
                                  embed_eligible=False))
        return parts

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

    # -- session fields ----------------------------------------------------

    @staticmethod
    def _title(messages: list[Message]) -> str | None:
        """Point 1: the export carries no title, so the first prompt is the title."""
        prompt = next((p.text for m in messages
                       if m.role == "user" and m.on_active_path
                       for p in m.parts if p.kind == KIND_TEXT and p.text), None)
        if not prompt:
            return None
        clean = " ".join(prompt.split())
        if len(clean) > 80:
            cut = clean[:80].rsplit(" ", 1)[0]
            clean = (cut or clean[:80]) + "…"
        return clean
