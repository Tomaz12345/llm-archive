"""Claude Code adapter.

The hard part is risk R1: the JSONL is a DAG, not a list. 47 of 66 sessions on this
machine branch for real, one with 139 leaves in a single file. Reading top-to-bottom
yields the same question answered several different ways as if it were one conversation.

Resolution: build the parent->child map, take the newest leaf, walk up to the root.
Everything else is kept but flagged `on_active_path = False` — abandoned branches are
real history, they are just not what happened.

Also handled, all found in Phase 0:
  * `<persisted-output>` markers pointing at sidecar tool-results files (§3d)
  * `<ide_opened_file>` / `<ide_selection>` noise carrying absolute paths (R8)
  * `thinking` blocks stored empty — dropped, never stored as empty rows (§1.2)
  * one project directory holding up to 6 cwds, some differing only by case (§3d)

**Resume sometimes forks, and the fork is linked by `core.lineage`.** §8.2 kept
`parent_session_id` against the possibility that `--resume` opens a new `<uuid>.jsonl`
replaying the old one's history, which makes one conversation read as two sessions and
counts its shared prefix twice.

That possibility was measured and dismissed, and the dismissal has since expired. The
phase 0 run of `tools/probe_resume.py` saw 9 projects, 93 files and 23,735 records and
found zero shared prefixes; re-run against 14 projects, 154 files and 36,586 records it
finds one, a clean whole-parent replay. Both readings were correct when taken -- Claude
Code's behaviour changed between them, so this is a fact with a date on it rather than a
property of the format.

What that means here: a resumed session may arrive either as a file whose bytes grew
(handled by the merge in `db.upsert_session`) or as a new file replaying an old one
(detected after ingest by `core.lineage`, which sets `session.continues_session_id` and
marks the replayed copies `message.superseded` so they are counted once). The pointer is
deliberately not `parent_session_id`, which still means "subagent transcript of" and
nothing else.

The probe is read-only and takes a few seconds. Re-run it if this looks wrong again --
it is expected to, eventually.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_ATTACHMENT,
    KIND_IMAGE,
    KIND_TEXT,
    KIND_TOOL_RESULT,
    KIND_TOOL_USE,
    attach_tool_input,
    Message,
    ParseStats,
    Part,
    Session,
)

CONTENT_TYPES = {"user", "assistant", "system"}

# Only what Claude Code actually accepts as an attachment; anything else keeps no suffix
# rather than inventing one from the mime subtype.
MEDIA_SUFFIXES = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "application/pdf": ".pdf",
}


def _suffix_for(media_type: str | None) -> str:
    return MEDIA_SUFFIXES.get((media_type or "").lower(), "")

IDE_TAGS = re.compile(
    r"<(ide_opened_file|ide_selection|ide_diagnostics|system-reminder|"
    r"command-name|command-message|command-args|local-command-stdout|"
    r"local-command-caveat|task-notification|task-id|tool-use-id|output-file|"
    r"status|summary|event)>.*?</\1>",
    re.S)

PERSISTED = re.compile(
    r"<persisted-output>\s*Output too large \(([^)]+)\)\.\s*"
    r"Full output saved to:\s*(.+?)\n", re.S)


def _ts(value) -> int | None:
    """ISO-8601 (Claude Code always uses UTC 'Z') -> epoch ms."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").rstrip("/").casefold()


def derive_workspace(cwds: Counter) -> tuple[str | None, str | None]:
    """Collapse a project directory's many cwds into one workspace key.

    Invoice_integration has 6 distinct cwds, two differing only by drive-letter case.
    Keying on the raw cwd would split one project into six and double-count the rest,
    so pick the shortest normalised path that prefixes the majority of records.
    """
    if not cwds:
        return None, None
    norm = Counter()
    labels: dict[str, str] = {}
    for raw, n in cwds.items():
        key = _norm_path(raw)
        norm[key] += n
        labels.setdefault(key, raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])

    total = sum(norm.values())
    best = None
    for cand in sorted(norm, key=len):
        covered = sum(n for k, n in norm.items() if k == cand or k.startswith(cand + "/"))
        if covered * 2 >= total:
            best = cand
            break
    if best is None:
        best = norm.most_common(1)[0][0]
    return best, labels.get(best)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", IDE_TAGS.sub(" ", text)).strip()


class ClaudeCodeAdapter:
    kind = "claude_code"
    label = "Claude Code"
    surface = "cli"

    def __init__(self, root: Path | None = None, blobs: BlobStore | None = None,
                 host: str | None = None):
        # `root` may point at a copy taken from another machine; nothing in the JSONL
        # records which host produced it, so `host` has to be supplied.
        if root is not None and root.name != "projects" and (root / "projects").is_dir():
            root = root / "projects"
        self.root = root or (Path.home() / ".claude" / "projects")
        self.blobs = blobs
        self.host = host

    # -- discovery ---------------------------------------------------------

    def discover(self) -> list[Path]:
        """Project directories, not files: workspace resolution needs the whole dir."""
        if not self.root.exists():
            return []
        return sorted(d for d in self.root.iterdir()
                      if d.is_dir() and any(d.glob("*.jsonl")))

    # -- parsing -----------------------------------------------------------

    def parse(self, project_dir: Path, stats: ParseStats) -> Iterator[Session]:
        files = sorted(project_dir.glob("*.jsonl"))

        # one pass over the directory to settle the workspace before any session
        dir_cwds: Counter = Counter()
        for path in files:
            for rec in self._records(path, stats):
                if rec.get("cwd"):
                    dir_cwds[rec["cwd"]] += 1
        ws_key, ws_label = derive_workspace(dir_cwds)

        for path in files:
            stats.files += 1
            session = self._parse_file(path, ws_key, ws_label, stats)
            if session is not None:
                stats.sessions += 1
                yield session

        # Subagent transcripts live one level down, at
        # <project>/<sessionId>/subagents/agent-*.jsonl — a non-recursive glob misses
        # them entirely. Parents are yielded first so the link can resolve.
        for path in sorted(project_dir.glob("*/subagents/*.jsonl")):
            stats.files += 1
            session = self._parse_file(path, ws_key, ws_label, stats)
            if session is not None:
                session.parent_native_id = path.parent.parent.name
                session.meta["subagent_of"] = session.parent_native_id
                stats.sessions += 1
                yield session

    def _records(self, path: Path, stats: ParseStats) -> Iterator[dict]:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stats.error(f"unreadable:{path.name}")
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.error("json_decode")
                continue
            if isinstance(rec, dict):
                yield rec

    def _parse_file(self, path: Path, ws_key, ws_label, stats: ParseStats) -> Session | None:
        records = list(self._records(path, stats))
        if not records:
            return None

        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        native_id = path.stem

        title = None
        cwds: Counter = Counter()
        versions: Counter = Counter()
        branches: Counter = Counter()
        entrypoints: Counter = Counter()
        models: Counter = Counter()
        tok_in = tok_out = cache_read = cache_write = 0

        # The DAG spans EVERY record type, not just conversational ones: a user record's
        # parentUuid routinely points at a file-history-snapshot. Building the graph over
        # content records alone shatters the chains — one session went from 1 root to 33,
        # and the active path collapsed to a single node.
        graph: dict[str, dict] = {}      # uuid -> record, ALL types
        nodes: dict[str, dict] = {}      # uuid -> record, content records only
        order: list[str] = []
        sidechain: set[str] = set()

        for rec in records:
            rtype = str(rec.get("type"))
            if isinstance(rec.get("uuid"), str):
                graph[rec["uuid"]] = rec
            if rec.get("cwd"):
                cwds[rec["cwd"]] += 1
            if rec.get("version"):
                versions[rec["version"]] += 1
            if rec.get("gitBranch"):
                branches[rec["gitBranch"]] += 1
            if rec.get("entrypoint"):
                entrypoints[rec["entrypoint"]] += 1

            if rtype == "ai-title" and rec.get("aiTitle"):
                title = rec["aiTitle"]
                continue

            if rtype not in CONTENT_TYPES:
                # R5: never fatal. Count it so `doctor` can report drift.
                stats.unknown(f"{self.kind}:{rtype}")
                continue

            uid = rec.get("uuid")
            if not isinstance(uid, str):
                stats.unknown(f"{self.kind}:{rtype}:no-uuid")
                continue
            nodes[uid] = rec
            order.append(uid)
            if rec.get("isSidechain"):
                sidechain.add(uid)

            usage = ((rec.get("message") or {}).get("usage") or {}) \
                if isinstance(rec.get("message"), dict) else {}
            tok_in += usage.get("input_tokens") or 0
            tok_out += usage.get("output_tokens") or 0
            cache_read += usage.get("cache_read_input_tokens") or 0
            cache_write += usage.get("cache_creation_input_tokens") or 0
            model = (rec.get("message") or {}).get("model") \
                if isinstance(rec.get("message"), dict) else None
            if model:
                models[model] += 1

        if not nodes:
            return None

        active = self._resolve_active_path(graph, sidechain, stats)

        messages: list[Message] = []
        seq_active = 0
        tool_calls: dict[str, tuple[int, Part]] = {}   # tool_use id -> (start_ms, part)
        for i, uid in enumerate(order):
            rec = nodes[uid]
            on_path = uid in active or uid in sidechain
            if on_path and uid not in sidechain:
                seq = seq_active
                seq_active += 1
            else:
                seq = i
            msg = self._build_message(rec, uid, seq, on_path, uid in sidechain, path, stats,
                                      tool_calls)
            if msg is not None:
                messages.append(msg)
                stats.messages += 1
                stats.parts += len(msg.parts)
                if not on_path:
                    stats.orphaned_messages += 1   # count kept messages only

        # Content records that all render to nothing — a file holding only `last-prompt`,
        # `mode` and a `system`/local_command record is a session that never happened.
        # `if not nodes` above does not catch it: those records ARE content types, they
        # just build no message. Stored, such a session has msg_count 0 and started_at 0,
        # so it counts in `by_source` but silently drops out of every dated chart. The
        # other three adapters already return early on an empty message list.
        if not messages:
            return None

        times = [m.created_at for m in messages if m.created_at]
        ws_session_key, ws_session_label = (ws_key, ws_label)
        if ws_session_key is None:
            ws_session_key, ws_session_label = derive_workspace(cwds)

        return Session(
            source_kind=self.kind,
            native_id=native_id,
            host=self.host,
            title=title,
            title_source="provider" if title else None,
            workspace_key=ws_session_key,
            workspace_label=ws_session_label,
            model_primary=models.most_common(1)[0][0] if models else None,
            started_at=min(times) if times else 0,
            ended_at=max(times) if times else None,
            tok_in=tok_in or None,
            tok_out=tok_out or None,
            tok_cache_read=cache_read or None,
            tok_cache_write=cache_write or None,
            raw_path=str(path),
            raw_hash=raw_hash,
            messages=messages,
            meta={
                "cwds": dict(cwds),
                "versions": dict(versions),
                "git_branches": dict(branches),
                "entrypoints": dict(entrypoints),
                "dag": {
                    "graph_nodes": len(graph),        # every uuid-bearing record
                    "content_nodes": len(nodes),      # user/assistant/system only
                    "active_in_graph": len(active),
                    "content_active": len(set(nodes) & active),
                    "sidechain": len(sidechain),
                },
            },
        )

    @staticmethod
    def _resolve_active_path(graph: dict[str, dict], sidechain: set[str],
                             stats: ParseStats) -> set[str]:
        """Walk up from the newest leaf. Everything not on that path is abandoned.

        `graph` must contain EVERY uuid-bearing record, not only conversational ones —
        parent links hop through bookkeeping records, so filtering first fragments the
        tree into dozens of false roots.

        Sidechains are subagent conversations hanging off the main thread, so they are
        excluded from the walk and kept unconditionally — they are not dead branches.
        """
        main = {u: r for u, r in graph.items() if u not in sidechain}
        if not main:
            return set()

        children: dict[str | None, list[str]] = defaultdict(list)
        for uid, rec in main.items():
            parent = rec.get("parentUuid")
            children[parent if parent in main else None].append(uid)

        leaves = [u for u in main if not children.get(u)]
        if not leaves:
            return set(main)

        def when(uid: str) -> str:
            return str(main[uid].get("timestamp") or "")

        newest = max(leaves, key=when)

        active: set[str] = set()
        cursor: str | None = newest
        guard = 0
        while cursor and cursor in main and guard <= len(main):
            active.add(cursor)
            cursor = main[cursor].get("parentUuid")
            guard += 1
        if guard > len(main):
            stats.error("dag_cycle")
        return active

    def _build_message(self, rec: dict, uid: str, seq: int, on_path: bool,
                       is_side: bool, path: Path, stats: ParseStats,
                       tool_calls: dict[str, tuple[int, Part]]) -> Message | None:
        msg = rec.get("message")
        role = rec.get("type")
        if isinstance(msg, dict) and msg.get("role"):
            role = msg["role"]

        usage = (msg or {}).get("usage") or {} if isinstance(msg, dict) else {}
        out = Message(
            native_id=uid,
            parent_native_id=rec.get("parentUuid"),
            seq=seq,
            on_active_path=on_path,
            is_sidechain=is_side,
            role=str(role),
            model=(msg or {}).get("model") if isinstance(msg, dict) else None,
            created_at=_ts(rec.get("timestamp")) or 0,
            tok_in=usage.get("input_tokens"),
            tok_out=usage.get("output_tokens"),
        )
        if rec.get("isMeta"):
            out.meta["is_meta"] = True

        content = (msg or {}).get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return out if out.parts else None

        for blk in content:
            if not isinstance(blk, dict):
                continue
            part = self._build_part(blk, len(out.parts), path, stats,
                                    out.created_at, tool_calls)
            if part is not None:
                out.parts.append(part)
        return out if out.parts else None

    def _build_part(self, blk: dict, seq: int, path: Path, stats: ParseStats,
                    when: int, tool_calls: dict[str, tuple[int, Part]]) -> Part | None:
        btype = str(blk.get("type"))

        if btype == "thinking":
            # Stored empty on disk (signature only) — nothing to keep. §1.2
            return None

        if btype == "text":
            text = _clean(blk.get("text") or "")
            if not text:
                return None
            return self._maybe_offload(
                Part(kind=KIND_TEXT, seq=seq, text=text, embed_eligible=True))

        if btype == "tool_use":
            name = str(blk.get("name") or "tool")
            summary = self._summarise_tool_input(blk.get("input"))
            part = Part(kind=KIND_TOOL_USE, seq=seq, text=summary, tool_name=name,
                        bytes=len(json.dumps(blk.get("input") or {})),
                        embed_eligible=True)  # intent is worth embedding; the payload is not
            attach_tool_input(part, blk.get("input"), self.blobs)
            tool_id = blk.get("id")
            if isinstance(tool_id, str):
                tool_calls[tool_id] = (when, part)
            return part

        if btype == "tool_result":
            return self._build_tool_result(blk, seq, path, stats, when, tool_calls)

        if btype in ("image", "document"):
            return self._build_media(blk, btype, seq, stats)

        stats.unknown(f"{self.kind}:content:{btype}")
        return None

    def _build_media(self, blk: dict, btype: str, seq: int, stats: ParseStats) -> Part:
        """A pasted screenshot or dropped document.

        The bytes are right there in the record as base64, so keeping only a size — as
        this did until now — threw away the one thing that makes the part worth showing.
        A `url` source has no bytes to recover; it degrades to the old reference-only part.
        """
        source = blk.get("source") or {}
        kind = KIND_IMAGE if btype == "image" else KIND_ATTACHMENT
        media_type = str(source.get("media_type") or "") or None
        part = Part(kind=kind, seq=seq, text=None,
                    bytes=len(json.dumps(source)), embed_eligible=False)

        if source.get("type") == "base64" and source.get("data") and self.blobs:
            try:
                data = base64.b64decode(str(source["data"]), validate=False)
            except (ValueError, TypeError):
                stats.unknown(f"{self.kind}:{btype}-undecodable")
                return part
            sha, size, dest = self.blobs.put_bytes(data)
            part.blob_sha, part.blob_path, part.bytes = sha, dest, size
            # Claude Code names nothing it pastes, and the viewer wants a caption. The
            # media_type is the only naming information the record carries.
            part.text = f"pasted-{btype}{_suffix_for(media_type)}"
            stats.blobs += 1
            stats.blob_bytes += size
        elif source.get("type") == "url" and source.get("url"):
            part.text = str(source["url"])
        return part

    def _build_tool_result(self, blk: dict, seq: int, path: Path, stats: ParseStats,
                           when: int, tool_calls: dict[str, tuple[int, Part]]) -> Part:
        content = blk.get("content")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text")
        elif isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False) if content is not None else ""

        part = Part(kind=KIND_TOOL_RESULT, seq=seq, text=text,
                    tool_ok=not blk.get("is_error"),
                    bytes=blk.get("persistedOutputSize") or len(text.encode("utf-8", "replace")),
                    embed_eligible=False)   # §1.1: indexed for FTS, never embedded

        # A result block names no tool -- only the call it answers, by id. Carry the
        # call's name across: every other agent source stamps its results, and without
        # it here a `--tool Bash` search could reach the command but never its output.
        tool_use_id = blk.get("tool_use_id")
        call = tool_calls.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
        if call is not None:
            started, use_part = call
            part.tool_name = use_part.tool_name
            if when and started and when >= started:
                use_part.duration_ms = when - started

        match = PERSISTED.search(text)
        if match and self.blobs is not None:
            # Resolve against the session directory, NOT the absolute path baked into
            # the marker — that path breaks the moment ~/.claude moves. §3d
            filename = Path(match.group(2).strip().strip('"')).name
            sidecar = path.parent / path.stem / "tool-results" / filename
            if sidecar.exists():
                stored = self.blobs.put_file(sidecar)
                if stored:
                    sha, size, dest = stored
                    part.blob_sha, part.blob_path = sha, dest
                    part.bytes = blk.get("persistedOutputSize") or size
                    stats.blobs += 1
                    stats.blob_bytes += size
            else:
                stats.error("persisted_output_missing")
        return part

    def _maybe_offload(self, part: Part) -> Part:
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path = sha, dest
                part.bytes = size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part

    @staticmethod
    def _summarise_tool_input(value) -> str:
        """One line of intent, never the payload."""
        if not isinstance(value, dict):
            return ""
        for key in ("command", "file_path", "path", "pattern", "query", "url",
                    "description", "prompt", "notebook_path"):
            if isinstance(value.get(key), str) and value[key].strip():
                # A command line is the one payload worth carrying whole into `text`:
                # `part_fts` is external-content over THIS column and nothing else, so
                # a 300-character cap left 47% of Bash calls unfindable by search.
                cap = 2000 if key == "command" else 300
                return f"{key}: {value[key][:cap]}"
        keys = ", ".join(sorted(value)[:6])
        return f"({keys})" if keys else ""
