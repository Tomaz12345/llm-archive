"""Codex adapter.

Rollout files are an event log, not a tree — `{timestamp, type, payload}` per line, in
order. No `parentUuid`, so no R1 resolution needed; a `thread_rolled_back` event marks
the rare rewind and is recorded in meta.

The subtlety is that the same turn appears twice in different forms:

    event_msg/user_message      what the human actually typed          <- use this
    response_item/message role=user
                                the full API payload, which also carries
                                injected AGENTS.md, <environment_context>
                                and IDE tab lists                      <- not conversation

Counts confirm it: one session has 4 `user_message` events but 6 role=user response
items. So conversation comes from `event_msg`, tool activity from `response_item`, and
`response_item/message` is skipped — with a fallback if a Codex version emits no
`event_msg` at all.

Two things are deliberately dropped:
  * `session_meta.base_instructions` — the entire system prompt, repeated in every file
  * `response_item/reasoning` — `encrypted_content`, unreadable (same as Claude Code)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
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
from .claude_code import derive_workspace

TOOL_CALLS = {"function_call", "custom_tool_call", "web_search_call", "local_shell_call"}
TOOL_OUTPUTS = {"function_call_output", "custom_tool_call_output",
                "local_shell_call_output"}


def _ts(value) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


class CodexAdapter:
    kind = "codex"
    label = "Codex"
    surface = "cli"

    def __init__(self, root: Path | None = None, blobs: BlobStore | None = None,
                 host: str | None = None):
        self.root = root or (Path.home() / ".codex")
        self.blobs = blobs
        self.host = host

    def discover(self) -> list[Path]:
        sessions = self.root / "sessions"
        if not sessions.exists():
            return []
        return sorted(sessions.rglob("*.jsonl"))

    def _thread_names(self) -> dict[str, str]:
        """Titles live in a separate index file, not in the rollouts."""
        index = self.root / "session_index.jsonl"
        names: dict[str, str] = {}
        if not index.exists():
            return names
        for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec.get("thread_name"):
                names[rec["id"]] = rec["thread_name"]
        return names

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            raw = path.read_bytes()
        except OSError:
            stats.error(f"unreadable:{path.name}")
            return
        digest = hashlib.sha256(raw).hexdigest()

        titles = getattr(self, "_titles_cache", None)
        if titles is None:
            titles = self._titles_cache = self._thread_names()

        native_id = None
        cwds: Counter = Counter()
        meta: dict = {}
        tok_in = tok_out = cached = reasoning = 0
        rollbacks = 0
        started = None

        messages: list[Message] = []
        pending_calls: dict[str, tuple[str, int, Part]] = {}   # call_id -> (name, when, part)
        fallback_msgs: list[Message] = []
        seq = 0

        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.error("json_decode")
                continue

            outer = str(rec.get("type"))
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = str(payload.get("type") or "")
            when = _ts(rec.get("timestamp")) or 0

            if outer == "session_meta":
                native_id = payload.get("id") or native_id
                started = _ts(payload.get("timestamp")) or when
                if payload.get("cwd"):
                    cwds[payload["cwd"]] += 1
                meta.update({
                    "originator": payload.get("originator"),
                    "cli_version": payload.get("cli_version"),
                    "source": payload.get("source"),
                    "model_provider": payload.get("model_provider"),
                    "git": payload.get("git"),
                })
                # base_instructions deliberately not stored: the whole system prompt,
                # identical across every rollout on the machine.
                continue

            if outer == "turn_context":
                if payload.get("cwd"):
                    cwds[payload["cwd"]] += 1
                meta.setdefault("timezone", payload.get("timezone"))
                meta.setdefault("model", payload.get("model"))
                continue

            if ptype == "token_count":
                info = payload.get("info") or {}
                total = info.get("total_token_usage") or {}
                tok_in = max(tok_in, total.get("input_tokens") or 0)
                tok_out = max(tok_out, total.get("output_tokens") or 0)
                cached = max(cached, total.get("cached_input_tokens") or 0)
                reasoning = max(reasoning, total.get("reasoning_output_tokens") or 0)
                continue

            if ptype == "thread_rolled_back":
                rollbacks += payload.get("num_turns") or 1
                continue

            # ---- conversation ----
            if ptype in ("user_message", "agent_message"):
                text = (payload.get("message") or payload.get("text") or "").strip()
                if not text:
                    continue
                role = "user" if ptype == "user_message" else "assistant"
                messages.append(self._text_message(role, text, when, seq))
                seq += 1
                continue

            # ---- tool activity ----
            if outer == "response_item" and ptype in TOOL_CALLS:
                name = payload.get("name") or ptype.replace("_call", "")
                call_id = payload.get("call_id")
                summary = self._call_summary(payload)
                msg = Message(native_id=call_id, role="assistant", created_at=when,
                              seq=seq)
                call_part = Part(
                    kind=KIND_TOOL_USE, seq=0, text=summary, tool_name=str(name),
                    bytes=len(json.dumps(payload, ensure_ascii=False)),
                    embed_eligible=True)
                msg.parts.append(call_part)
                if call_id:
                    pending_calls[call_id] = (str(name), when, call_part)
                messages.append(msg)
                seq += 1
                continue

            if outer == "response_item" and ptype in TOOL_OUTPUTS:
                call_id = payload.get("call_id")
                out = payload.get("output")
                text = out if isinstance(out, str) else json.dumps(
                    out, ensure_ascii=False) if out is not None else ""
                call = pending_calls.get(call_id or "")
                if call and when and when >= call[1]:
                    call[2].duration_ms = when - call[1]
                msg = Message(native_id=f"{call_id}:out" if call_id else None,
                              role="tool", created_at=when, seq=seq)
                msg.parts.append(self._offload(Part(
                    kind=KIND_TOOL_RESULT, seq=0, text=text,
                    tool_name=call[0] if call else None,
                    embed_eligible=False)))     # §1.1
                messages.append(msg)
                seq += 1
                continue

            if outer == "response_item" and ptype == "message":
                # Injected context or a duplicate of agent_message. Kept only as a
                # fallback for Codex builds that emit no event_msg records.
                if payload.get("role") in ("user", "assistant"):
                    text = " ".join(
                        c.get("text", "") for c in (payload.get("content") or [])
                        if isinstance(c, dict)).strip()
                    if text:
                        fallback_msgs.append(self._text_message(
                            str(payload["role"]), text, when, len(fallback_msgs)))
                continue

            if ptype == "reasoning" or outer == "response_item" and ptype == "reasoning":
                continue    # encrypted_content, unreadable

            if outer == "event_msg":
                stats.unknown(f"{self.kind}:event:{ptype}")
            else:
                stats.unknown(f"{self.kind}:{outer}:{ptype}")

        if native_id is None:
            native_id = path.stem

        has_conversation = any(m.role in ("user", "assistant") and
                               m.parts and m.parts[0].kind == KIND_TEXT
                               for m in messages)
        if not has_conversation and fallback_msgs:
            stats.unknown(f"{self.kind}:used-response_item-fallback")
            messages = fallback_msgs + [m for m in messages
                                        if m.parts and m.parts[0].kind != KIND_TEXT]
            for i, m in enumerate(messages):
                m.seq = i

        if not messages:
            return

        stats.messages += len(messages)
        stats.parts += sum(len(m.parts) for m in messages)
        stats.sessions += 1

        ws_key, ws_label = derive_workspace(cwds)
        times = [m.created_at for m in messages if m.created_at]
        meta["cwds"] = dict(cwds)
        if rollbacks:
            meta["rolled_back_turns"] = rollbacks
        if reasoning:
            meta["reasoning_output_tokens"] = reasoning

        yield Session(
            source_kind=self.kind,
            native_id=native_id,
            host=self.host,
            title=titles.get(native_id),
            title_source="provider" if titles.get(native_id) else None,
            workspace_key=ws_key,
            workspace_label=ws_label,
            model_primary=meta.get("model"),
            started_at=started or (min(times) if times else 0),
            ended_at=max(times) if times else None,
            tok_in=tok_in or None,
            tok_out=tok_out or None,
            tok_cache_read=cached or None,
            raw_path=str(path),
            raw_hash=digest,
            messages=messages,
            meta={k: v for k, v in meta.items() if v is not None},
        )

    def _text_message(self, role: str, text: str, when: int, seq: int) -> Message:
        msg = Message(native_id=None, role=role, created_at=when, seq=seq)
        msg.parts.append(self._offload(
            Part(kind=KIND_TEXT, seq=0, text=text, embed_eligible=True)))
        return msg

    def _offload(self, part: Part) -> Part:
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part

    @staticmethod
    def _call_summary(payload: dict) -> str:
        """One line of intent. Arguments arrive as a JSON *string*."""
        args = payload.get("arguments") or payload.get("input")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                return args[:300]
            args = parsed
        if isinstance(args, dict):
            for key in ("command", "path", "file_path", "query", "pattern"):
                value = args.get(key)
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                if isinstance(value, str) and value.strip():
                    return f"{key}: {value[:300]}"
            return f"({', '.join(sorted(args)[:6])})"
        action = payload.get("action")
        if isinstance(action, dict) and action.get("query"):
            return f"query: {str(action['query'])[:300]}"
        return ""
