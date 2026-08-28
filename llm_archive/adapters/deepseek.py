"""DeepSeek adapter — reads the official account data export.

Export flow: Profile → Settings → Data → **Export data**, which mails a link to
`deepseek_data-<YYYY-MM-DD>.zip`. Two members, and only one of them holds chats:

    deepseek_data-*.zip
      ├── conversations.json   every chat in the account
      └── user.json            user_id, email, OAuth profile — no chat content

`user.json` is deliberately not read. It is account identity, it is one of only two
places in the ten sources where an email address appears (Grok's
`prod-mc-auth-mgmt-api.json` is the other), and nothing in the archive is keyed on it.

Shape of `conversations.json` — an array of conversations, each a ChatGPT-style node map:

    [ { id, title, inserted_at, updated_at,
        mapping: { '<node id>': { id, parent, children: [...],
                                  message: { model, inserted_at,
                                             fragments: [ ... ] } | null } } } ]

Section 2 rated this the highest-risk export of the five, and it earned that. Five
things it does that no other source in this archive does:

1. **A tree with no `current_node`.** The mapping is the same DAG shape as ChatGPT's,
   minus the pointer that says which leaf survived. The shared resolver (§8.1) supplies
   that — newest leaf wins — so an edited or regenerated turn does not read back as the
   same question answered twice. A synthetic `root` node carries `message: null` and
   must stay in the graph: it is what joins otherwise separate first turns.

2. **No role field anywhere.** Who spoke is implied by fragment type: `REQUEST` is the
   human, `RESPONSE`/`SEARCH` are the model. Both nodes of the verified pair carry
   `model: 'deepseek-chat'`, so `model` says which model was *selected*, not who spoke —
   it is recorded on answers only, as in T3 Chat. A node whose fragments mix a request
   with a response is split into two messages rather than filed under one role.

3. **Timestamps that run backwards inside a turn.** In the verified export the answer is
   stamped 23:16:23.418 and the question that produced it 23:16:23.422 — the answer is
   4 ms *older* than the prompt. Ordering therefore comes from the tree, never from
   `inserted_at`, and `seq` is assigned by walking parent → child. The reader orders by
   `seq`, so this is the difference between a readable thread and an inverted one.

4. **Search results with no content, and citations that point at them.** A `SEARCH`
   fragment carries `{url, title}` per hit and no page text at all — unlike T3's
   webSearch, which returns whole scraped pages. Answer text then cites them positionally
   as `[citation:6]`, one-based into that list. The hits are kept as a tool result so the
   citation markers still resolve to something; there is no body to keep.

5. **No usage anywhere.** No token counts, no cost, on any record. Those columns are a
   real gap for this source, not a zero — DeepSeek's export simply does not carry them.

`THINKING` is handled but **unverified**: this account has no `deepseek-reasoner` chat,
so no reasoning fragment appears in the sample. It is mapped the way claude.ai's is
(kept, embedded), and any fragment type not listed here is counted in `unknown_types`
rather than guessed at — which is how the next export will tell us what to add.
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

# Fragment types, as they appear in the export.
FRAG_REQUEST = "REQUEST"
FRAG_RESPONSE = "RESPONSE"
FRAG_SEARCH = "SEARCH"
FRAG_THINKING = "THINKING"

SEARCH_TOOL = "search"


def _ts(value) -> int | None:
    """Epoch ms, UTC. The export stamps `+08:00`; a naive stamp is read as UTC."""
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


class DeepSeekAdapter:
    kind = "deepseek"
    label = "DeepSeek"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Is this drop's conversations.json DeepSeek-shaped rather than Claude-shaped?

        Both exports ship that filename, so the marker is the payload: DeepSeek nests a
        `mapping` and stamps `inserted_at`, claude.ai carries `chat_messages`. ChatGPT
        also nests a `mapping`, and is told apart by `inserted_at` — which only DeepSeek
        writes — rather than by adding a third exclusion here.
        """
        return DeepSeekAdapter._is_export(conversations_head(path))

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    @staticmethod
    def _is_export(head: str | None) -> bool:
        return bool(head) and '"mapping"' in head and '"inserted_at"' in head \
            and '"chat_messages"' not in head

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
        stats.files += 1
        try:
            conversations, digest = self._load(path)
        except (json.JSONDecodeError, OSError, StopIteration, zipfile.BadZipFile):
            stats.error(f"unreadable:{path.name}")
            return

        for conv in conversations:
            if not isinstance(conv, dict) or not conv.get("id"):
                stats.unknown(f"{self.kind}:conversation-without-id")
                continue
            session = self._session(path, digest, conv, stats)
            if session is not None:
                stats.sessions += 1
                yield session

    def _session(self, path: Path, digest: str, conv: dict,
                 stats: ParseStats) -> Session | None:
        from ._tree import resolve_active_path

        mapping = conv.get("mapping")
        nodes = {str(k): v for k, v in mapping.items()
                 if isinstance(v, dict)} if isinstance(mapping, dict) else {}
        if not nodes:
            stats.unknown(f"{self.kind}:conversation-without-mapping")
            return None

        # Every node goes into the graph, the message-less `root` included: dropping it
        # would split the first turn off from the rest of the tree (§8.1).
        active = resolve_active_path(
            nodes.keys(),
            lambda n: self._parent_of(nodes, n),
            lambda n: (_ts((nodes[n].get("message") or {}).get("inserted_at")) or 0,
                       self._depth(nodes, n)),
        )

        messages: list[Message] = []
        seq = 0
        for i, nid in enumerate(self._walk(nodes)):
            on_path = nid in active
            for msg in self._build_messages(nodes[nid], nid, stats):
                msg.on_active_path = on_path
                msg.seq = seq if on_path else i
                messages.append(msg)
                stats.messages += 1
                stats.parts += len(msg.parts)
                if on_path:
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
            "searches": sum(1 for m in messages for p in m.parts
                            if p.kind == KIND_TOOL_USE),
            "export_file": path.name,
            "export_digest": digest[:16],       # which drop this came from
        }
        if len(models) == 1:
            # Drives the BY ASSISTANT breakdown, which holds one value per session.
            meta["participant"] = models[0]
            meta["participant_label"] = models[0]

        return Session(
            source_kind=self.kind,
            native_id=str(conv["id"]),
            title=title,
            title_source="provider" if title else None,
            started_at=_ts(conv.get("inserted_at")) or (min(times) if times else 0),
            ended_at=_ts(conv.get("updated_at")) or (max(times) if times else None),
            raw_path=str(path),
            exported_at=taken_at(path),
            # Hash this conversation, not the export: the file is account-wide, so
            # hashing the whole document reports every chat as changed whenever one of
            # them grows. `digest` still salts it so a re-download stays traceable.
            raw_hash=hashlib.sha256(
                json.dumps(conv, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            model_primary=self._primary(answers),
            messages=messages,      # no token counts and no cost in this export
            meta=meta,
        )

    # -- tree --------------------------------------------------------------

    @staticmethod
    def _parent_of(nodes: dict, nid: str) -> str | None:
        parent = nodes[nid].get("parent")
        return str(parent) if parent is not None else None

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

        Document order would do for the verified export, but only because it happens to
        be topological. The turn timestamps are not reliably ordered (point 3 of the
        module docstring), so the parent → child links are the only ordering this
        format actually guarantees.
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

    def _build_messages(self, node: dict, nid: str,
                        stats: ParseStats) -> list[Message]:
        """One node, normally one message — two when it mixes a request with a reply."""
        raw = node.get("message")
        if not isinstance(raw, dict):
            return []                       # the synthetic `root`, or an empty node
        fragments = [f for f in (raw.get("fragments") or []) if isinstance(f, dict)]
        if not fragments:
            stats.unknown(f"{self.kind}:message-without-fragments")
            return []

        asked = [f for f in fragments if str(f.get("type")).upper() == FRAG_REQUEST]
        answered = [f for f in fragments if str(f.get("type")).upper() != FRAG_REQUEST]
        if asked and answered:
            # Never seen in the verified export; filing both under one role would
            # attribute the prompt to the model, so they are split instead.
            stats.unknown(f"{self.kind}:mixed-fragments")

        created = _ts(raw.get("inserted_at")) or 0
        parent = None if node.get("parent") is None else str(node["parent"])
        model = _text(raw.get("model")) or None

        built: list[Message] = []
        for role, group in (("user", asked), ("assistant", answered)):
            if not group:
                continue
            msg = Message(
                # A split node needs two ids; the plain node id stays with the first so
                # ids do not churn for the shape the export actually produces.
                native_id=nid if not built else f"{nid}:{role}",
                parent_native_id=parent,
                role=role,
                created_at=created,
                # As in T3 Chat: a prompt records the model that was *selected*, which
                # says nothing about who answered it.
                model=model if role == "assistant" else None,
            )
            for frag in group:
                msg.parts.extend(self._build_parts(frag, len(msg.parts), stats))
            if msg.parts:
                built.append(msg)
        return built

    # -- parts -------------------------------------------------------------

    def _build_parts(self, frag: dict, seq: int, stats: ParseStats) -> list[Part]:
        """One fragment can yield several parts, so `seq` is assigned on the way out."""
        parts = self._parts_for(frag, stats)
        for i, part in enumerate(parts):
            part.seq = seq + i
        return parts

    def _parts_for(self, frag: dict, stats: ParseStats) -> list[Part]:
        ftype = str(frag.get("type")).upper()

        if ftype in (FRAG_REQUEST, FRAG_RESPONSE):
            text = _text(frag.get("content"))
            return [self._offload(Part(kind=KIND_TEXT, seq=0, text=text,
                                       embed_eligible=True), stats)] if text else []

        if ftype == FRAG_THINKING:
            # Unverified — no deepseek-reasoner chat in the sample. Treated like
            # claude.ai's reasoning: real prose, kept and embedded.
            text = _text(frag.get("content")) or _text(frag.get("thinking"))
            return [self._offload(Part(kind=KIND_THINKING, seq=0, text=text,
                                       embed_eligible=True), stats)] if text else []

        if ftype == FRAG_SEARCH:
            return self._search_parts(frag, stats)

        stats.unknown(f"{self.kind}:fragment:{ftype}")
        return []

    def _search_parts(self, frag: dict, stats: ParseStats) -> list[Part]:
        """The hits a `[citation:n]` marker points at — titles and urls, no page text."""
        results = [r for r in (frag.get("results") or []) if isinstance(r, dict)]
        # The call itself carries no query: the export records what came back, not what
        # was asked. Kept as a marker so the tool breakdown counts the search at all.
        parts = [Part(kind=KIND_TOOL_USE, seq=0, tool_name=SEARCH_TOOL,
                      tool_ok=True, embed_eligible=False)]
        if not results:
            stats.unknown(f"{self.kind}:search-without-results")
            return parts

        # Numbered to match the citation markers, which index this list from one.
        text = "\n".join(
            f"[{i}] " + " · ".join(_text(r.get(k)) for k in ("title", "url")
                                   if _text(r.get(k)))
            for i, r in enumerate(results, 1))
        parts.append(self._offload(Part(
            kind=KIND_TOOL_RESULT, seq=0, tool_name=SEARCH_TOOL, tool_ok=True,
            text=text, embed_eligible=False), stats))     # §1.1: never embedded
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
