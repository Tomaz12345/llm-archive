"""OpenRouter adapter — reads the per-chat "Export Chat" JSON (schema `orpg.3.0`).

Export flow: chat UI → gear cog → Export Chat → one `.json` per conversation. There is
no bulk export (§2.1), so drops accumulate one file per chat, and the same chat gets
re-exported whenever it grows. Both facts drive the design below.

Shape — three id-keyed maps that must be joined, not three arrays:

    { version: 'orpg.3.0', title, characters: {charId: {model, isRemoved, ...}},
      messages: {msgId: {characterId, type, parentMessageId, createdAt, metadata,
                         items: [{id, sequenceIndex}]}},
      items:    {itemId: {messageId, data: {type, content: [{type, text}]}}} }

Four things this format does that the others do not:

1. **No conversation id.** The file carries no chat identity at all, and the filename is
   a date plus a browser dedup suffix — `…2026.json`, `…2026(1).json` — so it cannot
   supply one either. `native_id` is therefore the id of the root message, which is
   stable for the life of the chat. Re-exporting a chat you have already ingested
   updates that session instead of forking a second copy.

2. **`characters` is a registry, not a cast.** It keeps models the user added and later
   removed: in the sample three-model chat, two of the three carry `isRemoved: true` and
   never produced a message. Attribution comes from what actually generated text —
   `metadata.variantSlug` — with the character map only as a fallback.

3. **Siblings are usually legitimate, not abandoned.** In a multi-model chat every model
   answers the same prompt, so one parent has several assistant children and all of them
   are real. That is the opposite of the Claude Code / claude.ai case, so the shared
   `_tree.resolve_active_path` (newest leaf wins, everything else orphaned) would throw
   away every model's answer but one. `_active_subtree` below keeps one child per *slot*
   — per character for assistants, one for the user — which drops retries and edited
   prompts while keeping parallel answers.

4. **`tokensCount` is output only.** `tokensCount / duration` reproduces the recorded
   `tokensPerSecond` exactly on all sample generations, so it is the completion count;
   the export records no prompt tokens. `tok_in` stays NULL rather than being guessed.

Reasoning items carry real text (3 KB in the sample nemotron turn), so they are kept and
embedded, as with claude.ai and unlike Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
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

# A per-chat export is tens of KB. Anything past this is not one of these files, and
# sniffing it would mean parsing a bulk export we have no business reading here.
MAX_EXPORT_BYTES = 8 * 1024 * 1024


def _ts(value) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _sort_key(msg: dict) -> tuple:
    """Newest wins. `createdAt` ties on same-millisecond fan-out, so id breaks it."""
    return (str(msg.get("createdAt") or ""), str(msg.get("id") or ""))


def _active_subtree(messages: dict) -> set[str]:
    """Ids that survive edits and retries, keeping parallel model answers.

    One child per (parent, slot): an assistant's slot is its characterId, so two models
    replying to one prompt occupy different slots and both live, while a retry of the
    same model lands in the same slot and only the newest survives. User messages share
    a single slot, so editing a prompt supersedes the original — and, because the walk
    only descends into survivors, the whole abandoned branch under it drops out too.
    """
    if not messages:
        return set()

    children: dict[str | None, list[str]] = {}
    for mid, msg in messages.items():
        parent = msg.get("parentMessageId")
        if parent not in messages:
            parent = None          # missing parent link ⇒ treat as a root
        children.setdefault(parent, []).append(mid)

    active: set[str] = set()
    frontier: list[str | None] = [None]
    while frontier:
        parent = frontier.pop()
        slots: dict[str, str] = {}
        for mid in children.get(parent, []):
            msg = messages[mid]
            slot = ("user" if msg.get("type") == "user"
                    else "assistant:" + str(msg.get("characterId")))
            best = slots.get(slot)
            if best is None or _sort_key(msg) > _sort_key(messages[best]):
                slots[slot] = mid
        for mid in slots.values():
            if mid not in active:          # a cycle would otherwise spin here
                active.add(mid)
                frontier.append(mid)
    return active


ORIGIN = "https://openrouter.ai"

# The store's own name, and the prefix on every key inside it. This is the version
# marker §8.5 says a format like this needs: a schema change bumps `v3`, and the reader
# then finds no rooms and says so instead of mis-parsing a shape it has never seen.
STORE_NAME = "openrouter:playground:v3"
STORE_VERSION = "v3"


@dataclass(frozen=True)
class BrowserStore:
    """One IndexedDB object store on this machine holding OpenRouter's chats."""
    path: Path
    store_id: int
    profile: str

    @property
    def name(self) -> str:
        return f"{self.profile}/{self.path.name}"


def browser_stores() -> list[BrowserStore]:
    """Every local browser store for openrouter.ai, or [] if there are none.

    Read-only and failure-tolerant by design: a locked profile, a browser that is not
    installed, or a schema this reader does not know are all "nothing here", never an
    error that stops the rest of an unattended sync.
    """
    from ..core import idb

    found: list[BrowserStore] = []
    for path in idb.stores(ORIGIN):
        try:
            with idb.opened(path) as con:
                for store_id, name in idb.store_names(con).items():
                    if name == STORE_NAME:
                        # <profile>/storage/default/<origin>/idb/<file>.sqlite — the
                        # profile is five levels up, and naming it is what tells two
                        # browser profiles apart in the run report.
                        profile = path.parents[4].name if len(path.parents) > 4 else "?"
                        found.append(BrowserStore(path, store_id, profile))
        except (OSError, sqlite3.DatabaseError):
            continue
    return found


class OpenRouterAdapter:
    kind = "openrouter"
    label = "OpenRouter"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None,
                 browser: bool = False):
        self.drops = drops or DROPS
        self.blobs = blobs
        # Off unless asked for: this reads a browser profile, which is where session
        # cookies and saved passwords also live. `llma browser --enable` stores the
        # setting so an unattended sync keeps doing it.
        self.browser = browser

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this JSON declare an `orpg` schema version?

        Sniffed by content, not by filename: the export is named for the day it was
        taken, so the name says nothing about which chat — or which tool — produced it.
        """
        if path.suffix.lower() != ".json" or path.name == "conversations.json":
            return False
        try:
            if path.stat().st_size > MAX_EXPORT_BYTES:
                return False
            data = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return False
        return isinstance(data, dict) and str(data.get("version", "")).startswith("orpg")

    def discover(self) -> list:
        """Drop files, and — when switched on — the browser's own store.

        The store goes LAST so that a chat present both ways is written by the browser
        copy, which is the live one. An export you clicked is a snapshot of the moment
        you clicked it; the store is what the chat is now.
        """
        targets: list = by_recency(p for p in candidates(self.drops) if self.claims(p))
        if self.browser:
            targets.extend(browser_stores())
        return targets

    # -- parsing -----------------------------------------------------------

    def parse(self, target, stats: ParseStats) -> Iterator[Session]:
        """One drop file, or the browser's own store.

        The store yields documents in exactly the shape an exported file has, so
        everything below this line is shared. That is deliberate: the two paths must
        agree about what a session's `native_id` is, or the same chat would land twice
        — once from the export you clicked and once from the browser it came out of.
        """
        if isinstance(target, BrowserStore):
            yield from self._parse_store(target, stats)
            return

        stats.files += 1
        try:
            raw = target.read_bytes()
            doc = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            stats.error(f"unreadable:{target.name}")
            return
        if not isinstance(doc, dict):
            stats.error(f"not-an-export:{target.name}")
            return
        yield from self._sessions(doc, raw, target, taken_at(target), stats)

    def _parse_store(self, store: BrowserStore, stats: ParseStats) -> Iterator[Session]:
        """Rebuild each room in the browser store into an export-shaped document.

        The store keeps five record types keyed `/v3:<type>:<id>` — `master:rooms` lists
        the rooms, each room has a `manifest` naming its messages, items and characters,
        and those live as their own records. Reassembling them here means the exported
        JSON stays the only shape the parser below knows.
        """
        from ..core import idb

        stats.files += 1
        try:
            with idb.opened(store.path) as con:
                records: dict[str, dict] = {}
                failed = 0
                for key, value in idb.records(con, store.store_id):
                    if isinstance(value, Exception):
                        failed += 1
                        continue
                    if isinstance(value, dict):
                        records[key] = value.get("value", value)
        except (OSError, sqlite3.DatabaseError) as exc:
            stats.error(f"browser-store-unreadable:{type(exc).__name__}")
            return

        if failed:
            # Never silent: a record this cannot decode is a chat that will be missing
            # from the archive, and the count is what says so out loud in `llma doctor`.
            stats.unknown(f"{self.kind}:browser-record-undecodable")
            for _ in range(failed - 1):
                stats.unknown(f"{self.kind}:browser-record-undecodable")

        def of(kind: str) -> dict:
            prefix = f"{STORE_VERSION}:{kind}:"
            return {k.split(":", 2)[2]: v for k, v in records.items()
                    if k.startswith(prefix) or k.startswith("/" + prefix)}

        rooms, manifests = of("room"), of("manifest")
        messages, items, characters = of("message"), of("item"), of("character")

        if not rooms:
            stats.unknown(f"{self.kind}:browser-store-without-rooms")
            return

        # When this capture was taken — the store file's own mtime, which is the true
        # analogue of a drop's mtime: both say "this is what the source held at this
        # moment". The room's `updatedAt` is a different clock entirely (when the chat
        # last changed), and comparing the two would refuse a fresh browser read simply
        # because the chat in it had been quiet for a while. `freshness` keeps those two
        # promises apart for the same reason.
        captured = taken_at(store.path)

        for room_id, room in rooms.items():
            manifest = manifests.get(room_id) or {}
            ids = manifest.get("messageIds") or []
            doc = {
                "version": f"orpg.browser.{STORE_VERSION}",
                "title": room.get("title"),
                "messages": {i: messages[i] for i in ids if i in messages},
                # Messages are filtered by the manifest because that is what assigns a
                # chat to a room. Items and characters are NOT: they are globally keyed
                # and a message only ever looks up its own, so filtering them buys
                # nothing and can lose content — one room's manifest here omits an item
                # its own message references, which showed up as a `dangling-item`.
                "items": items,
                "characters": characters,
                "artifacts": {i: True for i in (manifest.get("artifactIds") or [])},
            }
            if not doc["messages"]:
                continue        # an empty room is a chat you opened and never used
            yield from self._sessions(
                doc, json.dumps(doc, sort_keys=True).encode("utf-8"),
                store.path, captured, stats, origin="browser",
                extra={"room_id": room_id,
                       "room_updated_at": room.get("updatedAt")})

    def _sessions(self, doc: dict, raw: bytes, path: Path,
                  exported_at: int | None, stats: ParseStats,
                  origin: str = "export",
                  extra: dict | None = None) -> Iterator[Session]:
        raw_msgs = {k: v for k, v in (doc.get("messages") or {}).items()
                    if isinstance(v, dict)}
        items = doc.get("items") or {}
        characters = doc.get("characters") or {}
        if not raw_msgs:
            stats.unknown(f"{self.kind}:export-without-messages")
            return

        for mid, msg in raw_msgs.items():      # `_sort_key` needs the id on the record
            msg.setdefault("id", mid)
        active = _active_subtree(raw_msgs)

        ordered = sorted(raw_msgs.items(), key=lambda kv: _sort_key(kv[1]))
        messages: list[Message] = []
        seq = 0
        for i, (mid, raw_msg) in enumerate(ordered):
            on_path = mid in active
            msg = self._build_message(mid, raw_msg, items, characters,
                                      seq if on_path else i, on_path, stats)
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
            return

        # Artifacts (OpenRouter's code canvases) are empty in every sample export, so
        # their file shape is unverified. Count them loudly rather than dropping them
        # silently — a non-zero unknown here is the signal to come back with a fixture.
        artifacts = len(doc.get("artifacts") or {})
        if artifacts:
            stats.unknown(f"{self.kind}:artifacts-not-parsed")

        # Identity comes from the earliest message in the file, including one that has
        # since been superseded: editing the *first* prompt would otherwise move the
        # root and re-ingest the same chat as a second session. The abandoned original
        # stays in the export, so it is the more stable anchor.
        root = min(raw_msgs.values(), key=_sort_key, default=None)
        native_id = (str((root or {}).get("id") or "")
                     or hashlib.sha256(raw).hexdigest()[:32])

        answers = [m for m in messages if m.on_active_path and m.role == "assistant"]
        models = sorted({m.model for m in answers if m.model})
        tok_out = sum(m.tok_out or 0 for m in answers)
        cost = sum(m.meta.get("cost_usd") or 0.0 for m in answers)
        times = [m.created_at for m in messages if m.created_at]

        title, title_source = self._title(doc.get("title"), messages)

        meta = {
            "models": models,
            "multi_model": len(models) > 1,
            "schema_version": doc.get("version"),
            "artifacts": artifacts,
            "export_file": path.name,
            # Which route this came in by. The same chat can arrive both ways and they
            # merge on `native_id`; this says which one wrote what is stored now.
            "origin": origin,
            **(extra or {}),
        }
        # `participant` drives the BY ASSISTANT breakdown, which holds one value per
        # session. A multi-model chat has no single answerer, so it opts out and is
        # represented by `models` instead of being filed under an arbitrary one.
        if len(models) == 1:
            meta["participant"] = models[0]
            meta["participant_label"] = models[0]

        stats.sessions += 1
        yield Session(
            source_kind=self.kind,
            native_id=native_id,
            title=title,
            title_source=title_source,
            started_at=min(times) if times else 0,
            ended_at=max(times) if times else None,
            raw_path=str(path),
            exported_at=exported_at,
            # Hash the parsed document, not the file bytes: the same chat re-exported
            # is identical in content but lands under a different name, and key order
            # is not guaranteed across export versions.
            raw_hash=hashlib.sha256(
                json.dumps(doc, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            model_primary=self._primary(answers),
            tok_out=tok_out or None,
            cost_usd=round(cost, 8) if cost else None,
            messages=messages,
            meta=meta,
        )

    # -- messages ----------------------------------------------------------

    def _build_message(self, mid: str, raw: dict, items: dict, characters: dict,
                       seq: int, on_path: bool, stats: ParseStats) -> Message | None:
        role = str(raw.get("type") or "").lower()
        if role not in ("user", "assistant", "system"):
            stats.unknown(f"{self.kind}:message-type:{role or 'missing'}")
            role = role or "assistant"

        char = characters.get(raw.get("characterId")) or {}
        metadata = raw.get("metadata") or {}
        model = metadata.get("variantSlug") or char.get("model")

        msg = Message(
            native_id=mid,
            parent_native_id=raw.get("parentMessageId"),
            seq=seq,
            on_active_path=on_path,
            role=role,
            created_at=_ts(raw.get("createdAt")) or 0,
            model=model if role != "user" else None,
            tok_out=metadata.get("tokensCount") if role != "user" else None,
        )

        # `items` on the message are pointers; the payload lives in the top-level map.
        # `sequenceIndex` is the render order and is absent on user turns, where the
        # declared order is already correct.
        refs = [r for r in (raw.get("items") or [])
                if isinstance(r, dict) and r.get("id")]
        refs.sort(key=lambda r: r["sequenceIndex"]
                  if isinstance(r.get("sequenceIndex"), int) else 0)
        for ref in refs:
            entry = items.get(ref["id"])
            if not isinstance(entry, dict):
                stats.unknown(f"{self.kind}:dangling-item")
                continue
            part = self._build_part(entry.get("data") or {}, len(msg.parts), stats)
            if part is not None:
                msg.parts.append(part)

        if raw.get("context") and raw["context"] != "main-chat":
            msg.meta["context"] = raw["context"]
        if raw.get("isEdited"):
            msg.meta["edited"] = True
        for key in ("generationId", "duration", "tokensPerSecond"):
            if metadata.get(key) is not None:
                msg.meta[key] = metadata[key]
        router = metadata.get("routerMetadata") or {}
        if router.get("strategy"):
            msg.meta["router_strategy"] = router["strategy"]
        provider = next((e.get("provider") for e in
                         ((router.get("endpoints") or {}).get("available") or [])
                         if isinstance(e, dict) and e.get("selected")), None)
        if provider:
            msg.meta["provider"] = provider
        try:
            cost = float(metadata.get("cost"))
        except (TypeError, ValueError):
            cost = 0.0
        if cost:
            msg.meta["cost_usd"] = cost

        return msg if msg.parts else None

    def _build_part(self, data: dict, seq: int, stats: ParseStats) -> Part | None:
        dtype = str(data.get("type"))
        text = "".join(
            blk.get("text") or "" for blk in (data.get("content") or [])
            if isinstance(blk, dict)
        ).strip()

        if dtype == "message":
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_TEXT, seq=seq, text=text, embed_eligible=True), stats)

        if dtype == "reasoning":
            # Real reasoning text, unlike Claude Code's signature-only blocks.
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_THINKING, seq=seq, text=text, embed_eligible=True), stats)

        if dtype in ("function_call", "web_search_call", "web_fetch_call"):
            # Server tools are enabled in every sample export but never fired, so the
            # argument shape is unconfirmed; keep the call's intent, per §1.1.
            args = data.get("arguments")
            return Part(
                kind=KIND_TOOL_USE, seq=seq,
                tool_name=str(data.get("name") or dtype),
                text=args if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False) if args else text,
                embed_eligible=True)

        stats.unknown(f"{self.kind}:item:{dtype}")
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

    # -- session fields ----------------------------------------------------

    @staticmethod
    def _primary(answers: list[Message]) -> str | None:
        """The model that did most of the talking; earliest answer breaks a tie."""
        counts: dict[str, int] = {}
        first: dict[str, int] = {}
        for i, m in enumerate(answers):
            if m.model:
                counts[m.model] = counts.get(m.model, 0) + 1
                first.setdefault(m.model, i)
        if not counts:
            return None
        return max(counts, key=lambda k: (counts[k], -first[k]))

    @staticmethod
    def _title(exported: object,
               messages: list[Message]) -> tuple[str | None, str | None]:
        """Prefer the user's own words to OpenRouter's 40-character stub.

        The export titles a chat with a hard prefix of the first prompt, cut mid-word:
        "Hey, can you tell me some things about p". When the stub is exactly that — a
        prefix — the prompt itself is the better title. A stub that is *not* a prefix
        means the chat was renamed by hand, and that name wins.
        """
        prompt = next((p.text for m in messages
                       if m.role == "user" and m.on_active_path
                       for p in m.parts if p.kind == KIND_TEXT and p.text), None)
        stub = exported.strip() if isinstance(exported, str) else ""

        if prompt and (not stub or prompt.startswith(stub)):
            clean = " ".join(prompt.split())
            if len(clean) > 80:
                cut = clean[:80].rsplit(" ", 1)[0]
                clean = (cut or clean[:80]) + "…"
            return clean, "first_prompt"
        return (stub or None), ("provider" if stub else None)
