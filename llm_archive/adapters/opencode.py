"""opencode adapter.

Storage is normalised into one JSON file per entity, joined by id:

    storage/session/<ses_*>.json    id, slug, projectID, directory, title, time.*
    storage/message/<ses>/<msg>.json  role, agent, model.{providerID,modelID}
    storage/part/<ses>/<prt>.json     type: text | reasoning | tool | step-start | step-finish

Two things differ from the other CLI sources:

  * `reasoning` parts carry REAL text here (Claude Code stores its thinking empty), so
    they are kept and embedded.
  * `step-finish` parts carry `cost` and `tokens` — the only source so far that reports
    cost directly rather than requiring a price table.

Timestamps are epoch **milliseconds**, not ISO strings.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
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
from .claude_code import derive_workspace


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


class OpenCodeAdapter:
    kind = "opencode"
    label = "opencode"
    surface = "cli"

    DEFAULT_ROOTS = [
        Path.home() / ".local" / "share" / "opencode" / "storage",
    ]

    def __init__(self, root: Path | None = None, blobs: BlobStore | None = None,
                 host: str | None = None):
        if root is not None:
            self.root = root if root.name == "storage" else root / "storage"
        else:
            self.root = next((r for r in self.DEFAULT_ROOTS if r.exists()),
                             self.DEFAULT_ROOTS[0])
        self.blobs = blobs
        self.host = host

    def discover(self) -> list[Path]:
        sessions = self.root / "session"
        if not sessions.exists():
            return []
        return sorted(sessions.rglob("*.json"))

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        sess = _load(path)
        if not isinstance(sess, dict) or not sess.get("id"):
            stats.error("session_unreadable")
            return
        sid = sess["id"]

        raw_msgs = []
        for mp in sorted((self.root / "message" / sid).glob("*.json")) \
                if (self.root / "message" / sid).exists() else []:
            rec = _load(mp)
            if isinstance(rec, dict) and rec.get("id"):
                raw_msgs.append(rec)
        raw_msgs.sort(key=lambda m: (m.get("time") or {}).get("created") or 0)

        # part/ is keyed by messageID, NOT sessionID — unlike message/, which is keyed
        # by sessionID. Looking in part/<sessionID>/ silently finds nothing, which
        # empties every message and drops the whole session.
        parts_by_msg: dict[str, list[dict]] = {}
        for raw in raw_msgs:
            part_dir = self.root / "part" / raw["id"]
            if not part_dir.exists():
                continue
            for pp in sorted(part_dir.glob("*.json")):
                rec = _load(pp)
                if isinstance(rec, dict):
                    parts_by_msg.setdefault(raw["id"], []).append(rec)

        messages: list[Message] = []
        models: Counter = Counter()
        cost = 0.0
        tok_in = tok_out = 0

        for seq, raw in enumerate(raw_msgs):
            model = raw.get("model") or {}
            model_id = model.get("modelID")
            if model_id:
                models[model_id] += 1

            msg = Message(
                native_id=raw["id"],
                role=str(raw.get("role") or "unknown"),
                created_at=(raw.get("time") or {}).get("created") or 0,
                seq=seq,
                model=model_id,
                meta={k: v for k, v in {
                    "agent": raw.get("agent"),
                    "provider": model.get("providerID"),
                }.items() if v},
            )

            for prt in parts_by_msg.get(raw["id"], []):
                ptype = str(prt.get("type"))

                if ptype in ("step-start",):
                    continue
                if ptype == "step-finish":
                    cost += prt.get("cost") or 0
                    toks = prt.get("tokens") or {}
                    tok_in += toks.get("input") or 0
                    tok_out += toks.get("output") or 0
                    continue

                if ptype == "text":
                    text = (prt.get("text") or "").strip()
                    if text:
                        msg.parts.append(self._offload(Part(
                            kind=KIND_TEXT, seq=len(msg.parts), text=text,
                            embed_eligible=True)))
                elif ptype == "reasoning":
                    text = (prt.get("text") or "").strip()
                    if text:      # real content here, unlike Claude Code
                        msg.parts.append(self._offload(Part(
                            kind=KIND_THINKING, seq=len(msg.parts), text=text,
                            embed_eligible=True)))
                elif ptype == "tool":
                    self._add_tool(msg, prt)
                else:
                    stats.unknown(f"{self.kind}:part:{ptype}")

            if msg.parts:
                messages.append(msg)

        if not messages:
            return

        stats.messages += len(messages)
        stats.parts += sum(len(m.parts) for m in messages)
        stats.sessions += 1

        directory = sess.get("directory")
        ws_key, ws_label = derive_workspace(Counter({directory: 1})) \
            if directory else (None, None)
        times = [m.created_at for m in messages if m.created_at]

        yield Session(
            source_kind=self.kind,
            native_id=sid,
            host=self.host,
            title=sess.get("title"),
            title_source="provider" if sess.get("title") else None,
            workspace_key=ws_key,
            workspace_label=ws_label,
            model_primary=models.most_common(1)[0][0] if models else None,
            started_at=(sess.get("time") or {}).get("created")
            or (min(times) if times else 0),
            ended_at=(sess.get("time") or {}).get("updated")
            or (max(times) if times else None),
            tok_in=tok_in or None,
            tok_out=tok_out or None,
            cost_usd=round(cost, 6) or None,
            raw_path=str(path),
            raw_hash=hashlib.sha256(
                json.dumps([sess, raw_msgs, parts_by_msg], sort_keys=True,
                           default=str).encode("utf-8")).hexdigest(),
            messages=messages,
            meta={k: v for k, v in {
                "slug": sess.get("slug"),
                "projectID": sess.get("projectID"),
                "directory": directory,
                "version": sess.get("version"),
            }.items() if v},
        )

    def _add_tool(self, msg: Message, prt: dict) -> None:
        name = prt.get("tool") or "tool"
        state = prt.get("state") or {}
        timing = state.get("time") or {}
        start, end = timing.get("start"), timing.get("end")
        duration = (end - start if isinstance(start, (int, float))
                    and isinstance(end, (int, float)) and end >= start else None)
        msg.parts.append(attach_tool_input(Part(
            kind=KIND_TOOL_USE, seq=len(msg.parts), tool_name=str(name),
            text=self._summarise(state.get("input")),
            bytes=len(json.dumps(state.get("input") or {}, default=str)),
            embed_eligible=True, duration_ms=duration),
            state.get("input"), self.blobs))
        output = state.get("output")
        if output:
            text = output if isinstance(output, str) else json.dumps(
                output, ensure_ascii=False, default=str)
            msg.parts.append(self._offload(Part(
                kind=KIND_TOOL_RESULT, seq=len(msg.parts), text=text,
                tool_name=str(name),
                tool_ok=state.get("status") != "error",
                embed_eligible=False)))     # §1.1

    def _offload(self, part: Part) -> Part:
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part

    @staticmethod
    def _summarise(value) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("command", "filePath", "path", "pattern", "query", "url"):
            if isinstance(value.get(key), str) and value[key].strip():
                return f"{key}: {value[key][:300]}"
        return f"({', '.join(sorted(value)[:6])})" if value else ""
