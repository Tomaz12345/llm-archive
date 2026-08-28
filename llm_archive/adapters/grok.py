"""Grok adapter — reads the official x.ai account data export.

Export flow: Settings → Data Controls → **Export your data**, which mails a link to a ZIP
named after a bare uuid — `5f4c8d58-c8d3-4b68-9256-13f01162dcb6.zip`. Nothing outside
the archive says what it is, and the chats sit four directories down:

    <uuid>.zip
      └── ttl/30d/export_data/<user id>/
            ├── prod-grok-backend.json      every conversation in the account
            ├── prod-mc-auth-mgmt-api.json  identity: email, sessions, api key hashes
            ├── prod-mc-billing.json        credit balance
            └── prod-mc-asset-server/       profile picture and other uploaded assets

Only `prod-grok-backend.json` is read. `prod-mc-auth-mgmt-api.json` is the most sensitive
file in any of the ten sources — it carries the account email, the linked Google email, a
birth date, and per-session IP address, city, latitude/longitude and user agent — so it
is skipped for the same reason DeepSeek's `user.json` is: it is account identity, and
nothing in the archive is keyed on it. `user_id`/`x_user_id` are dropped off each
conversation for the same reason. The `ttl/30d` prefix is x.ai's own retention marker on
the download, not a schema version, so discovery matches on the member name.

Shape of `prod-grok-backend.json`:

    { conversations: [ { conversation: { id, title, create_time, modify_time,
                                         leaf_response_id, starred, temporary, ... },
                         responses: [ { response: { _id, sender, message, model,
                                                    parent_response_id, path, children,
                                                    create_time, metadata, steps,
                                                    web_search_results, ... } } ] } ],
      projects: [], tasks: [], media_posts: [] }

Six things this format does that the others do not:

1. **The surviving leaf is named, and the root is not in the file.** `leaf_response_id`
   is the pointer DeepSeek's export is missing (§8.1), so the active path is walked
   directly from it rather than inferred from the newest leaf. The first turn's
   `parent_response_id` points at a node that does not appear in `responses` — a dangling
   root, not a broken link — so a response whose parent is absent is treated as a root.
   The shared resolver stays as the fallback for an export whose `leaf_response_id` is
   missing or names a response that was never included.

2. **Two clock formats, and neither one orders the turns.** The conversation stamps ISO
   (`2026-08-26T20:23:32.934050Z`); every response stamps Mongo extended JSON
   (`{"$date": {"$numberLong": "1787775850033"}}`, epoch ms). Both turns of the verified
   exchange land 10 ms apart — 1787775850033 and 1787775850043 — because `create_time`
   records when the record was persisted at the end of the turn, not when it was sent.
   The answer's real start is 37 s earlier, in `thinking_start_time`. Ordering therefore
   comes from the tree, exactly as in DeepSeek, and `seq` is assigned by walking
   parent → child.

3. **`model` is the picker, not the model.** The response carries `model: 'build'` — the
   name of the UI mode that was selected. What actually answered is
   `metadata.request_metadata.resolved_model` (`grok-chat-app-builder-free`), which is
   what `model_primary` and the BY ASSISTANT breakdown use; the picker is kept in
   `meta.pickers` because it is the only record of which mode produced the turn.

4. **Reasoning survives only as step headers.** There is no reasoning-text field. What is
   left of the trace is one short `header` per step ("Providing a short tutorial on the
   Odin programming language"), kept and embedded the way claude.ai's reasoning is. The
   model's narration of its own plan is not separated out at all — it is concatenated
   onto the front of `message` with no delimiter, so `message` is stored whole rather
   than split on a boundary the export does not mark.

5. **Tool calls arrive twice, parsed and as XML.** Each step's `tagged_text` holds
   `<xai:tool_usage_card>` markup, and `tool_usage_cards` holds the same calls already
   parsed into `{tool_usage_card_id, tool: {<ToolName>: {args}}}`. The parsed list is
   used; the markup is read only as a fallback for a step that omits it. Results join
   back by `tool_usage_card_id` — and often do not: of the five calls in the verified
   export, `ReadFile` and `InitTerminalSession` have no result recorded at all. Nothing
   in the format flags success or failure, so `tool_ok` stays None rather than claiming
   every Grok tool call worked.

6. **Search results are aggregated three times over.** A `BrowsePage` or `WebSearch`
   result appears in `tool_usage_results`, again in that step's `web_search_results`, and
   again in the response's top-level `web_search_results` — byte for byte, verified. Only
   the tool results are read; the two aggregates would triple-count every hit. The
   top-level list is the fallback for a response that has no steps. Page text comes back
   capped at 1024 characters per hit, so unlike T3's webSearch this source cannot produce
   a multi-hundred-KB tool result.

No usage anywhere: no token counts and no cost on any record — the same real gap DeepSeek
has, not a zero. `projects`, `tasks` and `media_posts` are all empty in the verified
export and are counted rather than guessed at, which is how the next export will say what
they hold.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_TEXT,
    KIND_THINKING,
    KIND_TOOL_RESULT,
    KIND_TOOL_USE,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import by_recency, candidates, member_head, taken_at

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"

# The one member of the export that holds chats. The rest is identity and billing.
MEMBER = "prod-grok-backend.json"

# Top-level arrays that are empty in the verified export. Counted, never parsed.
SIDE_COLLECTIONS = ("projects", "tasks", "media_posts")

# `tagged_text` keys seen in the verified export.
TAG_HEADER = "header"
TAG_TOOL_CARD = "tool_usage_card"
TAG_RAW_RESULT = "raw_function_result"

# Fallback for a step that ships the markup but not the parsed `tool_usage_cards`.
_CARD = re.compile(
    r"<xai:tool_usage_card_id>(?P<id>.*?)</xai:tool_usage_card_id>\s*"
    r"<xai:tool_name>(?P<name>.*?)</xai:tool_name>\s*"
    r"(?:<xai:tool_args><!\[CDATA\[(?P<args>.*?)\]\]></xai:tool_args>)?",
    re.DOTALL)

ROLES = {"human": "user", "assistant": "assistant", "system": "system"}


def _ts(value) -> int | None:
    """Epoch ms, UTC, from either clock format this export uses.

    The conversation stamps ISO-8601; every response stamps Mongo extended JSON. A naive
    ISO stamp is read as UTC.
    """
    if isinstance(value, dict):
        inner = value.get("$date")
        if isinstance(inner, dict):
            inner = inner.get("$numberLong")
        if isinstance(inner, bool):
            return None
        if isinstance(inner, int):
            return inner
        if isinstance(inner, str):
            return int(inner) if inner.lstrip("-").isdigit() else _ts(inner)
        return None
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


class GrokAdapter:
    kind = "grok"
    label = "Grok"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this drop hold a `prod-grok-backend.json`, wherever it sits inside?

        The ZIP is named after a uuid, so the member name is the only marker. The head
        read confirms the payload is the conversation document rather than a file that
        merely borrowed the name.
        """
        return GrokAdapter._is_export(member_head(path, MEMBER))

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    @staticmethod
    def _is_export(head: str | None) -> bool:
        return bool(head) and '"conversations"' in head

    def _load(self, path: Path) -> tuple[dict, str]:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(MEMBER))
                data = json.loads(zf.read(name).decode("utf-8", errors="replace"))
        else:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        return (data if isinstance(data, dict) else {}), digest

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            document, digest = self._load(path)
        except (json.JSONDecodeError, OSError, StopIteration, zipfile.BadZipFile):
            stats.error(f"unreadable:{path.name}")
            return

        for name in SIDE_COLLECTIONS:
            held = document.get(name)
            if isinstance(held, list) and held:
                # Empty in the verified export. Say so loudly rather than dropping them
                # silently, so the first export that fills one is visible.
                key = f"{self.kind}:{name}"
                stats.unknown_types[key] = stats.unknown_types.get(key, 0) + len(held)

        for entry in document.get("conversations") or []:
            if not isinstance(entry, dict):
                continue
            conv = entry.get("conversation")
            if not isinstance(conv, dict) or not conv.get("id"):
                stats.unknown(f"{self.kind}:conversation-without-id")
                continue
            session = self._session(path, digest, entry, conv, stats)
            if session is not None:
                stats.sessions += 1
                yield session

    def _session(self, path: Path, digest: str, entry: dict, conv: dict,
                 stats: ParseStats) -> Session | None:
        responses: dict[str, dict] = {}
        for wrapper in entry.get("responses") or []:
            raw = wrapper.get("response") if isinstance(wrapper, dict) else None
            if isinstance(raw, dict) and raw.get("_id"):
                responses[str(raw["_id"])] = raw
        if not responses:
            stats.unknown(f"{self.kind}:conversation-without-responses")
            return None

        active = self._active_path(conv, responses)

        messages: list[Message] = []
        for i, rid in enumerate(self._walk(responses)):
            msg = self._build_message(responses[rid], rid, stats)
            if msg is None:
                continue
            msg.on_active_path = rid in active
            msg.seq = i
            messages.append(msg)
            stats.messages += 1
            stats.parts += len(msg.parts)
            if not msg.on_active_path:
                stats.orphaned_messages += 1

        if not messages:
            return None
        # The active path is renumbered into a dense 0..n-1 run, which is what the reader
        # orders by; an orphaned turn keeps its walk position.
        for seq, msg in enumerate(m for m in messages if m.on_active_path):
            msg.seq = seq

        answers = [m for m in messages if m.role == "assistant"]
        models = sorted({m.model for m in answers if m.model})
        times = [m.created_at for m in messages if m.created_at]
        title = _text(conv.get("title")) or None

        tools: dict[str, int] = {}
        for msg in messages:
            for part in msg.parts:
                if part.kind == KIND_TOOL_USE and part.tool_name:
                    tools[part.tool_name] = tools.get(part.tool_name, 0) + 1

        meta = {
            "models": models,
            # The UI mode that was selected ('build', 'expert', …). Not a model, and the
            # only record of which mode produced the turn.
            "pickers": sorted({p for m in answers if (p := m.meta.get("picker"))}),
            "branched": any(not m.on_active_path for m in messages),
            "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
            "searches": sum(n for name, n in tools.items()
                            if "search" in name.lower() or "browse" in name.lower()),
            "starred": bool(conv.get("starred")),
            "temporary": bool(conv.get("temporary")),
            "export_file": path.name,
            "export_digest": digest[:16],       # which drop this came from
        }
        if conv.get("asset_ids"):
            # Never seen populated; the bytes would live in the export's asset folder,
            # which nothing here reads yet.
            meta["asset_ids"] = len(conv["asset_ids"])
            stats.unknown(f"{self.kind}:conversation-assets")
        if len(models) == 1:
            # Drives the BY ASSISTANT breakdown, which holds one value per session.
            meta["participant"] = models[0]
            meta["participant_label"] = models[0]

        return Session(
            source_kind=self.kind,
            native_id=str(conv["id"]),
            title=title,
            title_source="provider" if title else None,
            started_at=_ts(conv.get("create_time")) or (min(times) if times else 0),
            ended_at=_ts(conv.get("modify_time")) or (max(times) if times else None),
            raw_path=str(path),
            exported_at=taken_at(path),
            # Hash this conversation, not the export: the file is account-wide, so
            # hashing the whole document reports every chat as changed whenever one of
            # them grows. `digest` still salts it so a re-download stays traceable.
            raw_hash=hashlib.sha256(
                json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            model_primary=self._primary(answers),
            messages=messages,      # no token counts and no cost in this export
            meta=meta,
        )

    # -- tree --------------------------------------------------------------

    @staticmethod
    def _parent_of(responses: dict, rid: str) -> str | None:
        parent = responses[rid].get("parent_response_id")
        return str(parent) if parent is not None else None

    def _active_path(self, conv: dict, responses: dict) -> set[str]:
        """Walk up from `leaf_response_id`; fall back to the shared newest-leaf rule.

        Unlike DeepSeek's, this export names the surviving leaf outright — but a
        conversation whose leaf was excluded from the export (a deleted turn, a partial
        download) would otherwise resolve to an empty path and orphan every message.
        """
        named = conv.get("leaf_response_id")
        leaf = str(named) if named is not None else None
        if leaf in responses:
            active: set[str] = set()
            cursor: str | None = leaf
            while cursor is not None and cursor in responses and cursor not in active:
                active.add(cursor)
                cursor = self._parent_of(responses, cursor)
            return active

        from ._tree import resolve_active_path
        return resolve_active_path(
            responses.keys(),
            lambda n: self._parent_of(responses, n),
            lambda n: (_ts(responses[n].get("create_time")) or 0,
                       self._depth(responses, n)),
        )

    @classmethod
    def _depth(cls, responses: dict, rid: str) -> int:
        """Distance to the root, used only to break a timestamp tie between leaves."""
        depth, cursor, seen = 0, rid, {rid}
        while True:
            parent = cls._parent_of(responses, cursor)
            if parent is None or parent not in responses or parent in seen:
                return depth
            depth, cursor = depth + 1, parent
            seen.add(parent)

    @classmethod
    def _walk(cls, responses: dict) -> list[str]:
        """Response ids in reading order: depth-first from each root, oldest child first.

        Document order happens to be topological in the verified export, but the turn
        timestamps are not reliably ordered (point 2 of the module docstring), so the
        parent → child links are the only ordering this format actually guarantees. A
        response whose parent is absent is a root: the first turn of every conversation
        points at a root node the export does not include.
        """
        children: dict[str, list[str]] = {}
        roots: list[str] = []
        for rid in responses:
            parent = cls._parent_of(responses, rid)
            if parent in responses:
                children.setdefault(parent, []).append(rid)
            else:
                roots.append(rid)

        def born(rid: str) -> int:
            return _ts(responses[rid].get("create_time")) or 0

        order: list[str] = []
        seen: set[str] = set()
        stack = sorted(roots, key=born, reverse=True)
        while stack:
            rid = stack.pop()
            if rid in seen:
                continue
            seen.add(rid)
            order.append(rid)
            # Siblings are regenerations of one turn; oldest first keeps the abandoned
            # branch ahead of the one that replaced it.
            stack.extend(sorted((c for c in children.get(rid, []) if c not in seen),
                                key=born, reverse=True))
        # A response orphaned by a cycle is still content; keep it last rather than
        # lose it.
        order.extend(r for r in responses if r not in seen)
        return order

    # -- messages ----------------------------------------------------------

    def _build_message(self, raw: dict, rid: str, stats: ParseStats) -> Message | None:
        sender = str(raw.get("sender") or "").lower()
        role = ROLES.get(sender)
        if role is None:
            # `sender` is the only role signal in this format, so an unrecognised one is
            # worth surfacing; the turn is still kept, filed as the model's.
            stats.unknown(f"{self.kind}:sender:{sender or '<missing>'}")
            role = "assistant"

        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        request = metadata.get("request_metadata")
        request = request if isinstance(request, dict) else {}
        picker = _text(raw.get("model"))
        resolved = _text(request.get("resolved_model")) or picker or None

        parent = raw.get("parent_response_id")
        msg = Message(
            native_id=rid,
            parent_native_id=str(parent) if parent is not None else None,
            role=role,
            created_at=_ts(raw.get("create_time")) or 0,
            # A prompt records the mode that was *selected*, which says nothing about
            # who answered it — the same rule as T3 Chat and DeepSeek.
            model=resolved if role == "assistant" else None,
        )
        if role == "assistant":
            if picker and picker != resolved:
                msg.meta["picker"] = picker
            if _text(request.get("source")):
                msg.meta["source"] = _text(request["source"])
            started, ended = (_ts(raw.get("thinking_start_time")),
                              _ts(raw.get("thinking_end_time")))
            if started and ended and ended >= started:
                # The only honest "how long did this take" in the export: create_time is
                # stamped when the record was persisted, identically for both turns.
                msg.meta["thinking_ms"] = ended - started
            msg.parts.extend(self._answer_parts(raw, stats))
        else:
            text = _text(raw.get("message"))
            if text:
                msg.parts.append(self._offload(
                    Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True), stats))

        if not msg.parts:
            stats.unknown(f"{self.kind}:response-without-content")
            return None
        return msg

    # -- parts -------------------------------------------------------------

    def _answer_parts(self, raw: dict, stats: ParseStats) -> list[Part]:
        """Steps first, then the answer — the order the turn actually happened in."""
        parts: list[Part] = []
        steps = [s for s in (raw.get("steps") or []) if isinstance(s, dict)]
        for step in steps:
            parts.extend(self._step_parts(step, stats))

        if not steps:
            # With no steps, the response's own aggregate is the only record of what it
            # searched. With steps present it is a byte-for-byte duplicate of the tool
            # results above, and reading it would triple-count every hit.
            parts.extend(self._hit_parts(raw.get("web_search_results"), None, stats))

        text = _text(raw.get("message"))
        if text:
            # Stored whole: the model's narration of its own plan is concatenated onto
            # the front with no delimiter, and there is no boundary to split on.
            parts.append(self._offload(
                Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True), stats))

        for seq, part in enumerate(parts):
            part.seq = seq
        return parts

    def _step_parts(self, step: dict, stats: ParseStats) -> list[Part]:
        tagged = step.get("tagged_text")
        tagged = tagged if isinstance(tagged, dict) else {}
        order = [t for t in (step.get("tag_order") or []) if isinstance(t, str)]
        # A tag carrying text that `tag_order` forgot to list is still content.
        order += [t for t in tagged if t not in order]

        cards = self._cards(step, tagged.get(TAG_TOOL_CARD), stats)
        results = {str(r.get("tool_usage_card_id")): r.get("result")
                   for r in (step.get("tool_usage_results") or [])
                   if isinstance(r, dict)}

        parts: list[Part] = []
        for tag in order:
            if tag == TAG_HEADER:
                header = _text(tagged.get(TAG_HEADER))
                if header:
                    # All that is left of the reasoning trace; kept and embedded the way
                    # claude.ai's reasoning is.
                    parts.append(Part(kind=KIND_THINKING, seq=0, text=header,
                                      embed_eligible=True))
            elif tag == TAG_TOOL_CARD:
                for card_id, name, args in cards:
                    parts.append(Part(
                        kind=KIND_TOOL_USE, seq=0, tool_name=name,
                        # Nothing in this format flags success or failure.
                        tool_ok=None, text=args or None, embed_eligible=False))
                    if card_id in results:
                        parts.extend(self._result_parts(results.pop(card_id), name,
                                                        stats))
            elif tag == TAG_RAW_RESULT:
                body = _text(tagged.get(TAG_RAW_RESULT))
                if body:
                    parts.append(self._offload(Part(
                        kind=KIND_TOOL_RESULT, seq=0, tool_ok=None, text=body,
                        embed_eligible=False), stats))       # §1.1: never embedded
            else:
                stats.unknown(f"{self.kind}:tag:{tag}")

        # A result whose call was never listed is still what the model saw.
        for result in list(results.values()):
            stats.unknown(f"{self.kind}:result-without-card")
            parts.extend(self._result_parts(result, None, stats))
        return parts

    def _cards(self, step: dict, markup, stats: ParseStats) -> list[tuple[str, str, str]]:
        """`(card id, tool name, args json)` — parsed list first, markup as fallback."""
        listed = [c for c in (step.get("tool_usage_cards") or []) if isinstance(c, dict)]
        if listed:
            out = []
            for card in listed:
                tool = card.get("tool") if isinstance(card.get("tool"), dict) else {}
                name = next(iter(tool), None)
                if name is None:
                    stats.unknown(f"{self.kind}:card-without-tool")
                    continue
                payload = tool[name]
                # Most tools nest their arguments one level down; a few carry them flat.
                if isinstance(payload, dict) and "args" in payload:
                    payload = payload["args"]
                out.append((str(card.get("tool_usage_card_id")), str(name),
                            json.dumps(payload, ensure_ascii=False, sort_keys=True)))
            return out

        if not _text(markup):
            return []
        stats.unknown(f"{self.kind}:card-markup-only")
        return [(m.group("id").strip(), m.group("name").strip(),
                 (m.group("args") or "").strip())
                for m in _CARD.finditer(markup)]

    def _result_parts(self, result, tool: str | None,
                      stats: ParseStats) -> list[Part]:
        """One tool result. Search-shaped payloads are rendered; the rest kept as JSON."""
        if result is None:
            return []
        if isinstance(result, dict):
            if len(result) == 1:
                key, payload = next(iter(result.items()))
                if key == "WebSearchResults":
                    return self._hit_parts(payload, tool, stats)
                stats.unknown(f"{self.kind}:result:{key}")
            else:
                # Every result in the verified export is a single-key envelope. Anything
                # else is kept verbatim as JSON and flagged rather than reshaped.
                stats.unknown(f"{self.kind}:result-shape")
        body = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, sort_keys=True)
        if not _text(body):
            return []
        return [self._offload(Part(kind=KIND_TOOL_RESULT, seq=0, tool_name=tool,
                                   tool_ok=None, text=body, embed_eligible=False),
                              stats)]

    def _hit_parts(self, hits, tool: str | None, stats: ParseStats) -> list[Part]:
        """Hits, numbered so a citation marker lands on the one it names."""
        rows = [h for h in (hits or []) if isinstance(h, dict)]
        if not rows:
            return []
        blocks = []
        for i, hit in enumerate(rows, 1):
            head = " · ".join(_text(hit.get(k)) for k in ("title", "url")
                              if _text(hit.get(k)))
            # `preview` is the page text, capped at 1024 chars by the export.
            body = _text(hit.get("preview")) or _text(hit.get("description"))
            blocks.append(f"[{i}] {head}\n{body}".rstrip())
        return [self._offload(Part(
            kind=KIND_TOOL_RESULT, seq=0, tool_name=tool, tool_ok=None,
            text="\n\n".join(blocks), embed_eligible=False), stats)]  # §1.1: not embedded

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
