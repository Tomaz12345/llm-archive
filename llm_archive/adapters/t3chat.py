"""T3 Chat adapter — reads the account-wide "export message history" JSON.

Export flow: Settings → History & Sync → Export, which downloads one file for the whole
account, `threads-export-<ISO instant>.json`. Unlike OpenRouter (§2.1) this *is* a bulk
export, so the per-chat friction described there does not apply here.

Shape — two flat arrays joined on `threadId`, plus a schema version:

    { version: '11.0.1',
      threads:  [{ threadId, title, model, modelParams, createdAt, lastMessageAt,
                   pinned, visibility, generationStatus, userSetTitle, ... }],
      messages: [{ messageId, threadId, role, content, parts: [...], model, status,
                   tokens, providerMetadata, attachmentIds, serverError, ... }] }

Every record is doubled: Convex's `_id`/`_creationTime` sit beside the app's own
`messageId`/`created_at`, and each camelCase field is repeated in snake_case. The
app-level uuids are used as ids here — `_id` is a Convex document id that a
re-sync could reissue, while `threadId`/`messageId` are what the app itself
addresses a chat and a turn by.

Six things this format does that the others do not:

1. **There is no tree.** No record carries a parent pointer, and no superseded siblings
   survive: in the verified export all 170 threads alternate user/assistant strictly by
   `created_at`, with no duplicate timestamp inside a thread. T3 rewrites a message in
   place on retry or edit, so the export holds only the surviving turn — everything read
   here is on the active path, and this adapter never needs the shared tree resolver. If
   a future export does fan out, ordering degrades to a stable sort rather than
   inventing a structure that is not in the file.

2. **Two token counters that disagree on purpose.** `tokens` is T3's own stream counter
   (686 of 715 answers); `providerMetadata` carries the provider's usage block in
   whatever shape that provider uses (Anthropic, Google, OpenRouter — OpenAI's block has
   `responseId` and no usage at all, so 235 answers have no prompt count anywhere). The
   two agree except on reasoning models, where the provider's completion count *excludes*
   reasoning tokens that `tokens` includes: one `gemini-3-flash-thinking` turn reports
   1240 against 471. `tok_out` therefore prefers `tokens` and falls back to the provider
   count; `tok_in` can only come from the provider block. Anthropic's block is the only
   one that splits cache creation from cache reads, so cache columns are Claude-only
   here — a real gap, not a zero.

3. **`content` is a duplicate, not a summary.** On all 715 answers it equals the
   concatenation of the `text` parts exactly. `parts` wins because it also carries
   reasoning and tool calls; `content` is the fallback for records with no `parts` array,
   which is every user turn but one.

4. **Two server tools, both with real payloads.** `webSearch` returns whole scraped pages
   (4.5 MB across 38 calls, one of them 264 KB) — kept as tool results, indexed by FTS,
   never embedded (§1.1). `image_generation` returns a filename and a CDN URL; the bytes
   live behind that URL and are not in the export, so the part records the reference.

5. **Attachments are ids and nothing else.** 79 user turns reference 90 files by id, and
   the export contains no attachment table, no filenames and no bytes. The count is
   preserved as empty attachment parts; the content is simply not recoverable from this
   file, and a future export that adds one should be read here.

6. **Cost is real but partial.** Only OpenRouter-routed models report a charge (45 of 715
   answers, $3.07 total). Everything else is covered by the T3 subscription and reports
   no per-token price, so `cost_usd` is set only where the provider actually billed one
   and is never extrapolated to the rest of the thread.

Reasoning text is real (284 blocks, 17 of them empty), so it is kept and embedded, as
with claude.ai and unlike Claude Code.
"""

from __future__ import annotations

import hashlib
import json
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
    attach_tool_input,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import by_recency, candidates, taken_at

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"

# Discovery reads a window off the front of each candidate rather than parsing it: the
# verified export is 14 MB and writes `threads` as its first key. A file that hides the
# marker past this window is still caught by the filename the export ships with.
SNIFF_BYTES = 1 << 20
EXPORT_PREFIX = "threads-export"


def _ts(value) -> int | None:
    """Epoch ms, already UTC. Floats appear on `_creationTime`; ints elsewhere."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


class T3ChatAdapter:
    kind = "t3chat"
    label = "T3 Chat"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this drop look like a T3 export? Content first, name second."""
        if path.suffix.lower() != ".json" or path.name == "conversations.json":
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(SNIFF_BYTES).decode("utf-8", errors="replace")
        except OSError:
            return False
        marked = head.lstrip().startswith("{") and '"threads"' in head
        return marked or path.name.lower().startswith(EXPORT_PREFIX)

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            doc = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            stats.error(f"unreadable:{path.name}")
            return
        if not isinstance(doc, dict) or not isinstance(doc.get("threads"), list):
            stats.error(f"not-an-export:{path.name}")
            return

        version = doc.get("version")
        threads = {t["threadId"]: t for t in doc["threads"]
                   if isinstance(t, dict) and t.get("threadId")}

        by_thread: dict[str, list[dict]] = {}
        for raw in doc.get("messages") or []:
            if not isinstance(raw, dict) or not raw.get("threadId"):
                stats.unknown(f"{self.kind}:message-without-thread")
                continue
            by_thread.setdefault(raw["threadId"], []).append(raw)

        # A thread row that carries no messages is an empty chat, not a loss. The
        # reverse — messages whose thread row is missing — would lose real content, so
        # those are rebuilt from the messages alone and flagged.
        for tid in by_thread:
            if tid not in threads:
                stats.unknown(f"{self.kind}:thread-record-missing")
                threads[tid] = {"threadId": tid}

        for tid, thread in threads.items():
            session = self._session(path, version, tid, thread,
                                    by_thread.get(tid, []), stats)
            if session is not None:
                stats.sessions += 1
                yield session

    def _session(self, path: Path, version, tid: str, thread: dict,
                 raw_msgs: list[dict], stats: ParseStats) -> Session | None:
        messages: list[Message] = []
        seq = 0
        for raw in sorted(raw_msgs, key=self._sort_key):
            msg = self._build_message(raw, seq, stats)
            if msg is None:
                continue
            messages.append(msg)
            stats.messages += 1
            stats.parts += len(msg.parts)
            seq += 1

        if not messages:
            return None

        answers = [m for m in messages if m.role == "assistant"]
        models = sorted({m.model for m in answers if m.model})
        times = [m.created_at for m in messages if m.created_at]
        cost = sum(m.meta.get("cost_usd") or 0.0 for m in answers)

        meta = {
            "models": models,
            "selected_model": thread.get("model") or None,   # last model chosen, not
            "pinned": bool(thread.get("pinned")),            # necessarily the answerer
            "visibility": thread.get("visibility") or None,
            "generation_status": thread.get("generationStatus") or None,
            "user_set_title": bool(thread.get("userSetTitle")),
            "export_file": path.name,
            "export_version": version,
        }
        if thread.get("forkedFromSharedThread"):
            # A fork of someone else's shared chat. The parent lives in that person's
            # account, so it is a note about provenance, not a `parent_native_id` we
            # could ever resolve to a session in this archive.
            meta["forked_from_shared"] = thread["forkedFromSharedThread"]
        if "threadId" in thread and len(thread) == 1:
            meta["thread_record"] = False
        if len(models) == 1:
            meta["participant"] = models[0]
            meta["participant_label"] = models[0]

        title = _text(thread.get("title")) or None
        return Session(
            source_kind=self.kind,
            native_id=tid,
            title=title,
            # T3 titles a chat with a model-generated summary of the first prompt and
            # `userSetTitle` marks the ones renamed by hand. Both are the provider's
            # stored title; unlike OpenRouter's export there is no truncated stub to
            # second-guess.
            title_source="provider" if title else None,
            started_at=_ts(thread.get("createdAt")) or (min(times) if times else 0),
            ended_at=_ts(thread.get("lastMessageAt")) or (max(times) if times else None),
            raw_path=str(path),
            exported_at=taken_at(path),
            # Hash this thread, not the file: the export is account-wide, so hashing the
            # whole document would report all 170 chats as changed every time one of
            # them grows.
            raw_hash=hashlib.sha256(
                json.dumps({"thread": thread, "messages": raw_msgs},
                           sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            model_primary=self._primary(answers),
            tok_in=sum(m.tok_in or 0 for m in answers) or None,
            tok_out=sum(m.tok_out or 0 for m in answers) or None,
            tok_cache_read=sum(m.meta.get("cache_read") or 0 for m in answers) or None,
            tok_cache_write=sum(m.meta.get("cache_write") or 0 for m in answers) or None,
            cost_usd=round(cost, 8) if cost else None,
            messages=messages,
            meta=meta,
        )

    @staticmethod
    def _sort_key(raw: dict) -> tuple:
        """`created_at` orders every thread in the sample; the rest only break ties."""
        return (_ts(raw.get("created_at")) or 0,
                raw.get("_creationTime") or 0,
                str(raw.get("messageId") or ""))

    # -- messages ----------------------------------------------------------

    def _build_message(self, raw: dict, seq: int, stats: ParseStats) -> Message | None:
        role = str(raw.get("role") or "").lower()
        if role not in ("user", "assistant", "system", "tool"):
            stats.unknown(f"{self.kind}:role:{role or 'missing'}")
            role = role or "assistant"

        msg = Message(
            native_id=str(raw.get("messageId") or raw.get("_id") or ""),
            seq=seq,
            role=role,
            created_at=_ts(raw.get("created_at")) or 0,
            # A user record carries the model that was *selected* when the prompt was
            # sent, which says nothing about who answered it.
            model=raw.get("model") if role != "user" else None,
        )

        parts = raw.get("parts")
        if isinstance(parts, list) and parts:
            for blk in parts:
                if isinstance(blk, dict):
                    msg.parts.extend(self._build_parts(blk, len(msg.parts), stats))
        else:
            text = _text(raw.get("content"))
            if text:
                msg.parts.append(self._offload(
                    Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True), stats))

        # Ids into an attachment table the export does not ship. Keep the count so the
        # reader shows that a file was there; there is nothing else to keep.
        for _ in (raw.get("attachmentIds") or []):
            msg.parts.append(Part(kind=KIND_ATTACHMENT, seq=len(msg.parts),
                                  text=None, embed_eligible=False))

        self._usage(raw, msg)
        params = raw.get("modelParams") or {}
        for key, name in (("reasoningEffort", "reasoning_effort"),
                          ("includeSearch", "include_search")):
            if params.get(key) is not None:
                msg.meta[name] = params[key]
        for key, name in (("tokensPerSecond", "tokens_per_second"),
                          ("timeToFirstToken", "time_to_first_token")):
            if raw.get(key) is not None:
                msg.meta[name] = raw[key]
        if raw.get("byok"):
            msg.meta["byok"] = True          # answered on the user's own API key
        if raw.get("status") and raw["status"] != "done":
            msg.meta["status"] = raw["status"]
        if isinstance(raw.get("serverError"), dict):
            msg.meta["server_error"] = raw["serverError"].get("type") or "error"

        # An answer that failed or was stopped before its first token carries no parts
        # and no content. Counted rather than stored as an empty row (§ risk R5).
        if not msg.parts:
            stats.unknown(f"{self.kind}:empty-message:{raw.get('status') or 'unknown'}")
            return None
        return msg

    def _usage(self, raw: dict, msg: Message) -> None:
        """Fold whichever provider's usage block this answer carries into the message."""
        provider, block = next(
            ((k, v) for k, v in (raw.get("providerMetadata") or {}).items()
             if isinstance(v, dict)), (None, None))
        if provider:
            msg.meta["provider"] = provider

        tok_in = cache_read = cache_write = tok_out = None
        cost = None
        if provider == "anthropic":
            usage = block.get("usage") or {}
            tok_in = usage.get("input_tokens")
            tok_out = usage.get("output_tokens")
            cache_read = usage.get("cache_read_input_tokens")
            cache_write = usage.get("cache_creation_input_tokens")
        elif provider == "google":
            usage = block.get("usageMetadata") or {}
            tok_in = usage.get("promptTokenCount")
            tok_out = usage.get("candidatesTokenCount")
        elif provider == "openrouter":
            usage = block.get("usage") or {}
            tok_in = usage.get("promptTokens")
            tok_out = usage.get("completionTokens")
            cost = usage.get("cost")
            if block.get("provider"):
                msg.meta["upstream_provider"] = block["provider"]

        msg.tok_in = tok_in
        # T3's own counter first: it is the one that includes reasoning tokens.
        msg.tok_out = raw.get("tokens") if isinstance(raw.get("tokens"), int) else tok_out
        if cache_read:
            msg.meta["cache_read"] = cache_read
        if cache_write:
            msg.meta["cache_write"] = cache_write
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            cost = 0.0
        if cost:
            msg.meta["cost_usd"] = cost

    # -- parts -------------------------------------------------------------

    def _build_parts(self, blk: dict, seq: int, stats: ParseStats) -> list[Part]:
        """One source block can yield several parts, so `seq` is assigned on the way out."""
        parts = self._parts_for(blk, stats)
        for i, part in enumerate(parts):
            part.seq = seq + i
        return parts

    def _parts_for(self, blk: dict, stats: ParseStats) -> list[Part]:
        ptype = str(blk.get("type"))

        if ptype == "text":
            text = _text(blk.get("text"))
            return [self._offload(Part(kind=KIND_TEXT, seq=0, text=text,
                                       embed_eligible=True), stats)] if text else []

        if ptype == "reasoning":
            # Real reasoning text, unlike Claude Code's signature-only blocks. 17 of the
            # 284 blocks in the sample are empty and carry only a provider signature.
            text = _text(blk.get("reasoning"))
            return [self._offload(Part(kind=KIND_THINKING, seq=0, text=text,
                                       embed_eligible=True), stats)] if text else []

        if ptype == "tool_call":
            return self._tool_parts(blk, stats)

        stats.unknown(f"{self.kind}:part:{ptype}")
        return []

    def _tool_parts(self, blk: dict, stats: ParseStats) -> list[Part]:
        name = str(blk.get("toolName") or "tool")
        args = blk.get("args") or {}
        result = blk.get("result")
        ok = blk.get("status") in (None, "completed")

        if name == "webSearch":
            queries = args.get("queries")
            call = "; ".join(q for q in queries if isinstance(q, str)) \
                if isinstance(queries, list) else json.dumps(args, ensure_ascii=False)
        else:
            call = json.dumps(args, ensure_ascii=False) if args else ""

        parts = [attach_tool_input(
            Part(kind=KIND_TOOL_USE, seq=0, tool_name=name, text=call or None,
                 bytes=len(json.dumps(args, ensure_ascii=False)),
                 embed_eligible=bool(call)), args, self.blobs)]

        if name == "image_generation":
            # The bytes live behind a CDN URL that the export does not include, so the
            # part is the reference: filename plus where it was served from.
            for entry in (result if isinstance(result, list) else []):
                if not isinstance(entry, dict):
                    continue
                ref = " ".join(str(entry.get(k)) for k in ("fileName", "url")
                               if entry.get(k))
                parts.append(Part(kind=KIND_IMAGE, seq=0,
                                  tool_name=name, tool_ok=bool(entry.get("completed")),
                                  text=ref or None, embed_eligible=False))
            return parts

        if result:
            text = self._flatten(result)
            if text:
                parts.append(self._offload(Part(
                    kind=KIND_TOOL_RESULT, seq=0, tool_name=name, tool_ok=ok,
                    text=text, embed_eligible=False), stats))   # §1.1: never embedded
        elif result is None:
            stats.unknown(f"{self.kind}:tool-without-result:{name}")
        return parts

    @staticmethod
    def _flatten(result) -> str:
        """Search hits as readable text: title, url, then the scraped page."""
        if isinstance(result, str):
            return result.strip()
        if not isinstance(result, list):
            return json.dumps(result, ensure_ascii=False) if result else ""
        blocks = []
        for entry in result:
            if not isinstance(entry, dict):
                blocks.append(str(entry))
                continue
            head = " · ".join(str(entry[k]) for k in ("title", "url") if entry.get(k))
            body = entry.get("content") or entry.get("summary") or ""
            blocks.append("\n".join(p for p in (head, str(body).strip()) if p))
        return "\n\n".join(b for b in blocks if b).strip()

    def _offload(self, part: Part, stats: ParseStats | None = None) -> Part:
        """Oversized parts go to the blob store; a head excerpt stays inline.

        Scraped search pages make overflow the rule here rather than the exception:
        31 of the 38 tool results in the sample export are over the limit.
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

    # -- session fields ----------------------------------------------------

    @staticmethod
    def _primary(answers: list[Message]) -> str | None:
        """The model that did most of the talking; earliest answer breaks a tie.

        A T3 thread is single-model until you switch models mid-chat, which the export
        records per message — so this is a real question here, not a formality.
        """
        counts: dict[str, int] = {}
        first: dict[str, int] = {}
        for i, m in enumerate(answers):
            if m.model:
                counts[m.model] = counts.get(m.model, 0) + 1
                first.setdefault(m.model, i)
        if not counts:
            return None
        return max(counts, key=lambda k: (counts[k], -first[k]))
