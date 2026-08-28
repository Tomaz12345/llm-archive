"""GitHub Copilot (github.com) adapter — reads a captured share-link response.

The tenth source, and the only one with **no export of any kind**. What GitHub offers is
a share link, `github.com/copilot/share/<uuid>`, and three things about it shaped this
adapter:

1. **The link is not public.** Fetched without a browser session it 302s to `/login`, and
   a `gh` OAuth token does not open it either — the route authenticates on session
   cookies. Nothing here downloads it, and per §0 nothing here stores GitHub cookies to
   try.

2. **Ctrl+S captures nothing, silently.** The route server-renders a mount point and no
   more: `react-app.embeddedData.payload` is `{}` and the document holds ~450 characters
   of visible chrome. The messages are fetched afterwards by the client. A saved page is
   73 KB with no conversation in it and no sign that anything went wrong.

3. So the transcript is reachable only as **captured traffic** — the response to

       GET https://api.individual.githubcopilot.com/github/chat/shared/<shared id>/messages

   which is what `docs/export-requests.md` §9 walks you through saving, as either a whole
   HAR or that one response body. Both are read here.

Shape, verified against one captured conversation (2 messages, 130 KB of JSON):

    { thread:   { id, name, manuallyNamed, createdAt, updatedAt,
                  sharedID, sharedAt, associatedRepoIDs, autoPickedModel, ... },
      messages: [ { id, parentMessageID, role, content, createdAt, threadID,
                    model, usage: { inputTokens, outputTokens },
                    contentParts: [ { type: 'text', content }
                                  | { type: 'toolCall', skillExecution } ],
                    skillExecutions, references, interrupted, ... } ] }

Six things this format does that are worth knowing:

1. **It is a tree.** `parentMessageID` chains the turns, with the literal string `root`
   as the sentinel for "no parent" — not null, not absent. Editing or retrying a prompt
   is what makes it fan out, so the shared tree resolver runs here exactly as it does for
   Claude Code and claude.ai. The verified capture is a single chain; the resolver
   degrades to keeping that chain rather than assuming one.

2. **Two duplicate fields, both skipped.** `content` on an assistant turn equals the
   concatenation of its `text` contentParts exactly (4,920 chars, verified), and
   `skillExecutions` is element-for-element identical to the `toolCall` entries inside
   `contentParts`. `contentParts` wins because it is the only field that preserves the
   *order* tool calls and prose were emitted in. `content` remains the fallback for a
   turn that has no `contentParts` at all, which is every user turn.

3. **`references` is a third duplicate, reordered.** The message-level `references` array
   holds the same 59 objects as the skill executions' own reference lists — same set,
   grouped differently. Reading both would double-count 37 KB, so only the per-skill
   lists are read, where each payload still sits with the tool call that produced it.

4. **Tool results name their files but do not ship them.** Every `file` reference carries
   `path`, `url`, `commitOID` and `sha` with `content` set to `""` — 0 of 8 populated.
   The share endpoint strips the bytes. Same class of gap as Mistral's attachments: the
   reference is recorded, the content is simply not in the capture.

5. **Token counts are real, and only on assistant turns.** `usage.inputTokens` /
   `outputTokens`, with no cache split and no cost anywhere — Copilot bills in
   "premium requests", not tokens, so `cost_usd` is left null rather than invented.
   The one verified answer reports 141,844 in against 5,414 out, which is what a
   repo-reading agent costs.

6. **The repo is the workspace.** These conversations are scoped to a repository the way
   a CLI session is scoped to a project root, so the repo full name becomes the workspace
   key — that is what makes a Copilot chat about `maze_agent` land beside the Claude
   Code sessions for the same project under one facet.

Timestamps are Go's RFC3339Nano — nine fractional digits. `fromisoformat` accepts those
on 3.13 but not on every version this project supports (`requires-python >=3.11`), so
they are truncated to microseconds before parsing rather than trusted to the runtime.

`participant` is set to `copilot` so the BY ASSISTANT rollup in `llma stats` joins these
sessions with the Copilot Chat conversations that `vscode_chat.py` reads off local disk.
They stay separate *sources* — the formats share nothing — while still answering
"how much have I asked Copilot" in one line.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_TEXT,
    KIND_TOOL_RESULT,
    KIND_TOOL_USE,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import by_recency, candidates, taken_at
from ._tree import resolve_active_path

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"

# The endpoint that answers with the transcript. Matched loosely on purpose: the host is
# `api.individual.githubcopilot.com` for a personal plan and differs for business ones,
# so the path is the stable half.
SHARED_PATH = re.compile(r"/github/chat/shared/([0-9a-f-]{8,})/messages")

# `sharedID` is distinctive enough to identify a bare response body without parsing it,
# and it sits inside `thread`, the first key, so a small window always covers it.
SNIFF_BYTES = 1 << 16

# The owner/name pair out of a blob or tree URL — the only place a `file` reference
# records the owner, since its own `repoOwner` comes through empty.
REPO_URL = re.compile(r"github\.com/([^/\s]+/[^/\s]+)")

# `parentMessageID` on the first turn. A string, not null — filtering on falsiness would
# make the root its own parent and shatter the tree.
ROOT = "root"

# Nine-digit fractional seconds, which `fromisoformat` rejects before 3.13.
NANOS = re.compile(r"(\.\d{6})\d+")


def _ts(value) -> int | None:
    """Epoch ms, UTC. RFC3339Nano in, microsecond precision out."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(
            NANOS.sub(r"\1", value.strip()).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_transcript(doc) -> bool:
    return (isinstance(doc, dict) and isinstance(doc.get("thread"), dict)
            and isinstance(doc.get("messages"), list))


class CopilotWebAdapter:
    kind = "copilot_web"
    label = "GitHub Copilot"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Could this drop hold a shared Copilot transcript?

        A `.har` is taken on its extension — no other source here produces one, and
        confirming it would mean reading megabytes of unrelated traffic. A `.json` has to
        show `sharedID` in its first window, which keeps the T3 and OpenRouter exports
        already sitting in this folder from being opened.
        """
        suffix = path.suffix.lower()
        if suffix == ".har":
            return True
        if suffix != ".json":
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(SNIFF_BYTES).decode("utf-8", errors="replace")
        except OSError:
            return False
        return '"sharedID"' in head or bool(SHARED_PATH.search(head))

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    # -- parsing -----------------------------------------------------------

    def _transcripts(self, path: Path, stats: ParseStats) -> list[tuple[dict, str | None]]:
        """Every (transcript, source url) this drop holds.

        A HAR is a whole browsing session, so the response bodies are searched for the
        shared-messages endpoint rather than assumed to be the only thing captured.
        Request headers are never touched: a HAR recorded against a logged-in session
        carries a live session cookie and bearer token in them, and nothing here needs
        either. See docs/export-requests.md §9 — that file is worth deleting once
        ingested.
        """
        try:
            doc = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            stats.error(f"unreadable:{path.name}")
            return []

        if _is_transcript(doc):
            return [(doc, None)]

        if not (isinstance(doc, dict) and isinstance(doc.get("log"), dict)):
            stats.error(f"not-a-transcript:{path.name}")
            return []

        out = []
        for entry in doc["log"].get("entries") or []:
            if not isinstance(entry, dict):
                continue
            url = (entry.get("request") or {}).get("url") or ""
            if not SHARED_PATH.search(url):
                continue
            content = (entry.get("response") or {}).get("content") or {}
            body = content.get("text")
            # The browser sends a CORS preflight to this same URL, and it answers with an
            # empty body. Skipped as uninteresting rather than reported as a bad
            # response, which is what it would otherwise look like from here.
            if not isinstance(body, str) or not body.strip():
                continue
            if content.get("encoding") == "base64":
                continue
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                stats.error(f"bad-response-body:{path.name}")
                continue
            if _is_transcript(payload):
                out.append((payload, url))
        if not out:
            # A HAR that simply captured something else is not an error worth reporting;
            # a HAR of the share page that missed the fetch is, and they look identical
            # from here. Counted, never fatal (risk R5).
            stats.unknown(f"{self.kind}:har-without-transcript")
        return out

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        for doc, url in self._transcripts(path, stats):
            session = self._build(doc, path, url, stats)
            if session is not None:
                yield session

    def _build(self, doc: dict, path: Path, url: str | None,
               stats: ParseStats) -> Session | None:
        thread = doc["thread"]
        records = [m for m in doc["messages"] if isinstance(m, dict) and m.get("id")]
        if not records:
            return None

        # Hash the transcript, never the container. A HAR re-recorded from the same
        # conversation differs in every timing field, so hashing the file would report
        # the session as updated on every capture and defeat the skip in ingest.run.
        raw_hash = hashlib.sha256(
            json.dumps(doc, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        parents = {m["id"]: (None if m.get("parentMessageID") == ROOT
                             else m.get("parentMessageID"))
                   for m in records}
        stamps = {m["id"]: _ts(m.get("createdAt")) or 0 for m in records}
        active = resolve_active_path(
            ids=list(parents),
            parent_of=parents.get,
            sort_key=lambda mid: (stamps[mid], mid),
        )

        records.sort(key=lambda m: (stamps[m["id"]], m["id"]))

        messages: list[Message] = []
        for seq, record in enumerate(records):
            message = self._message(record, seq, stats)
            message.on_active_path = record["id"] in active
            if not message.on_active_path:
                stats.orphaned_messages += 1
            messages.append(message)

        stats.messages += len(messages)
        stats.parts += sum(len(m.parts) for m in messages)
        stats.sessions += 1

        tok_in = sum(m.tok_in or 0 for m in messages if m.on_active_path)
        tok_out = sum(m.tok_out or 0 for m in messages if m.on_active_path)
        ws_key, ws_label, repo = self._workspace(records)
        title = _text(thread.get("name")) or None
        models = Counter(m.model for m in messages if m.model)

        return Session(
            source_kind=self.kind,
            # The thread id, not the share id: re-sharing a conversation mints a new
            # `sharedID`, and keying on that would file the same chat twice.
            native_id=str(thread.get("id") or thread.get("sharedID") or path.stem),
            host=None,                          # a web chat ran on GitHub's machines
            title=title,
            # GitHub ships the name either way; `manuallyNamed` only says whether a
            # human or the model wrote it, and that distinction lives in meta.
            title_source="provider" if title else None,
            workspace_key=ws_key,
            workspace_label=ws_label,
            model_primary=(_text(thread.get("autoPickedModel")) or
                           (models.most_common(1)[0][0] if models else None)),
            started_at=_ts(thread.get("createdAt")) or 0,
            ended_at=_ts(thread.get("updatedAt")),
            tok_in=tok_in or None,
            tok_out=tok_out or None,
            # No cost: Copilot bills premium requests, not tokens. Nothing to prorate.
            raw_path=str(path),
            exported_at=taken_at(path),
            raw_hash=raw_hash,
            messages=messages,
            meta={k: v for k, v in {
                "participant": "copilot",
                "participant_label": "GitHub Copilot",
                "shared_id": thread.get("sharedID"),
                "shared_at": _ts(thread.get("sharedAt")),
                "auto_picked_model": thread.get("autoPickedModel"),
                "manually_named": thread.get("manuallyNamed") or None,
                "repo": repo,
                "endpoint": url,
                "messages": len(records),
            }.items() if v is not None},
        )

    # -- one message -------------------------------------------------------

    def _message(self, record: dict, seq: int, stats: ParseStats) -> Message:
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        parent = record.get("parentMessageID")
        message = Message(
            native_id=str(record["id"]),
            role=_text(record.get("role")) or "assistant",
            created_at=_ts(record.get("createdAt")) or 0,
            seq=seq,
            parent_native_id=None if parent == ROOT else parent,
            model=_text(record.get("model")) or None,
            tok_in=usage.get("inputTokens"),
            tok_out=usage.get("outputTokens"),
            meta={k: v for k, v in {
                "intent": record.get("intent"),
                "interrupted": record.get("interrupted") or None,
                "generated_with_auto": record.get("generatedWithAuto") or None,
            }.items() if v is not None},
        )

        parts = record.get("contentParts")
        if isinstance(parts, list) and parts:
            for block in parts:
                if isinstance(block, dict):
                    self._add_part(message, block, stats)
        else:
            # Every user turn, and any answer the API returned unsegmented.
            body = _text(record.get("content"))
            if body:
                message.parts.append(self._offload(Part(
                    kind=KIND_TEXT, seq=0, text=body, embed_eligible=True)))
        return message

    def _add_part(self, message: Message, block: dict, stats: ParseStats) -> None:
        kind = block.get("type")
        seq = len(message.parts)

        if kind == "text":
            body = _text(block.get("content"))
            if body:
                message.parts.append(self._offload(Part(
                    kind=KIND_TEXT, seq=seq, text=body, embed_eligible=True)))
            return

        if kind == "toolCall":
            skill = block.get("skillExecution")
            if not isinstance(skill, dict):
                stats.unknown(f"{self.kind}:toolCall-without-skillExecution")
                return
            status = skill.get("status")
            message.parts.append(Part(
                kind=KIND_TOOL_USE, seq=seq,
                tool_name=_text(skill.get("slug")) or "skill",
                text=_text(skill.get("arguments"))[:300],
                tool_ok=(status == "completed") if status else None,
                # `arguments` is the whole call; the excerpt above is only what is shown.
                bytes=len(_text(skill.get("arguments")).encode("utf-8")),
                embed_eligible=True))

            # The references ARE the result — the files read and the API responses
            # returned. Stored and keyword-searchable, never embedded: this is the
            # §1.1 rule, and it is 40 KB against 4.9 KB of actual prose here.
            refs = skill.get("references")
            if isinstance(refs, list) and refs:
                message.parts.append(self._offload(Part(
                    kind=KIND_TOOL_RESULT, seq=len(message.parts),
                    tool_name=_text(skill.get("slug")) or "skill",
                    text=json.dumps(refs, ensure_ascii=False, default=str),
                    embed_eligible=False)))
            return

        stats.unknown(f"{self.kind}:contentPart:{kind}")

    # -- repo as workspace -------------------------------------------------

    @staticmethod
    def _workspace(records: list[dict]) -> tuple[str | None, str | None, str | None]:
        """The repository this conversation was about, as (key, label, full name).

        `thread.repoID` is 0 on the verified capture and `associatedRepoIDs` holds a bare
        numeric id, so neither names the repo. The full name only ever appears down in the
        tool payloads, and no single field carries it reliably: `api-response` references
        have `repo`, while `file` references leave `repoOwner` **empty** and name the
        owner only inside their `url`. Reading just one of those would leave a
        conversation that happened to use only `getfile` with no workspace at all, so all
        four carriers are tried. The most-cited repo wins, which keeps a passing mention
        of someone else's repository from relabelling the chat.
        """
        seen: Counter = Counter()

        def note(full: str) -> None:
            full = full.strip().removesuffix(".git")
            if full.count("/") == 1 and all(part for part in full.split("/")):
                seen[full] += 1

        for record in records:
            for skill in record.get("skillExecutions") or []:
                if not isinstance(skill, dict):
                    continue
                # The call's own arguments name the repo on every repo-scoped skill.
                try:
                    args = json.loads(skill.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if isinstance(args, dict):
                    note(_text(args.get("repo")))
                for ref in skill.get("references") or []:
                    if not isinstance(ref, dict):
                        continue
                    note(_text(ref.get("repo")))
                    owner, name = _text(ref.get("repoOwner")), _text(ref.get("repoName"))
                    if owner and name:
                        note(f"{owner}/{name}")
                    match = REPO_URL.search(_text(ref.get("url")))
                    if match:
                        note(match.group(1))
        if not seen:
            return None, None, None
        repo = seen.most_common(1)[0][0]
        # Casefolded like every other workspace key, so the same repo cannot arrive
        # twice under different capitalisation.
        return f"github.com/{repo}".casefold(), repo.split("/")[-1], repo

    def _offload(self, part: Part) -> Part:
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part
