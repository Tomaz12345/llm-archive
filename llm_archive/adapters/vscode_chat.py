"""VS Code chat adapter — Copilot Chat and other chat extensions.

Found during Phase 0 while answering "is VS Code storage worth probing?". It was:
110 sessions, 59.7 MB, needing no export at all — the second-largest source on the
machine and the simplest to parse, because it is **flat**:

    workspaceStorage/<hash>/chatSessions/<uuid>.json
      { sessionId, customTitle, creationDate, lastMessageDate, mode, selectedModel,
        requests: [ { requestId, timestamp, modelId,
                      message: { text, parts },
                      response: [ ... ] } ] }

Response blocks are discriminated by `kind`, except plain assistant prose which has
**no kind at all** and carries its markdown in `value` — easy to miss, and it is the
actual answer text.

`thinking` blocks here hold real reasoning text, as in the claude.ai export.

One panel, several assistants: Copilot Chat, the Remote-SSH chat participant and any
other extension that registers one all write into the same store. Which one answered is
in `requests[].agent.extensionId`, so it is recorded per session as `meta.participant`
rather than split into separate sources — the *format* is what an adapter owns, and this
is one format. See PARTICIPANTS below.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

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
from .claude_code import derive_workspace


def _default_roots() -> list[Path]:
    appdata = os.environ.get("APPDATA")
    home = Path.home()
    candidates = [
        Path(appdata) / "Code" / "User" if appdata else None,
        Path(appdata) / "Code - Insiders" / "User" if appdata else None,
        home / ".config" / "Code" / "User",                       # linux
        home / "Library" / "Application Support" / "Code" / "User",  # macos
    ]
    return [c for c in candidates if c and c.exists()]


# Extension ids seen in this store, mapped to a stable key and a display name.
# An unknown extension is not an error: it keeps its own id as the key, so a newly
# installed chat extension shows up as its own participant without a code change.
PARTICIPANTS = {
    "github.copilot-chat":          ("copilot", "GitHub Copilot"),
    "ms-vscode-remote.remote-ssh":  ("remote-ssh", "Remote - SSH"),
    "andrepimenta.claude-code-chat": ("claude-code-chat", "Claude Code Chat"),
}

# Older sessions predate `agent`, but still name the responder.
RESPONDERS = {"github copilot": ("copilot", "GitHub Copilot")}

# `selectedModel.metadata.vendor`, the weakest signal — present on a dozen sessions.
VENDORS = {"copilot": "GitHub Copilot"}


class VSCodeChatAdapter:
    kind = "vscode_chat"
    label = "VS Code chat"
    surface = "editor_panel"

    def __init__(self, root: Path | None = None, blobs: BlobStore | None = None,
                 host: str | None = None):
        self.roots = [root] if root is not None else _default_roots()
        self.blobs = blobs
        self.host = host

    def discover(self) -> list[Path]:
        found: list[Path] = []
        for root in self.roots:
            base = root / "workspaceStorage"
            if base.exists():
                found.extend(sorted(base.glob("*/chatSessions/*.json")))
            empty = root / "globalStorage" / "emptyWindowChatSessions"
            if empty.exists():
                found.extend(sorted(empty.glob("*.json")))
        return found

    @staticmethod
    def _workspace_for(path: Path) -> tuple[str | None, str | None, str | None,
                                            str | None]:
        """Resolve (workspace_key, label, remote_host, folder) from workspace.json.

        VS Code stores chats for **remote** sessions on the local disk, tagged with a
        `vscode-remote://ssh-remote+<authority>/…` folder URI. So a laptop's worth of
        work done over SSH is already here — it just needs attributing to the machine it
        actually ran on rather than to this desktop.

        The fourth element is that folder with its case intact. `derive_workspace`
        casefolds, which is what makes six spellings of one project collapse into one
        workspace row — and is exactly wrong for handing the path back to VS Code over
        SSH, where `/home/tomaz/Wall_E` and `/home/tomaz/wall_e` are different
        directories. See `core/reopen.py`.
        """
        meta = path.parent.parent / "workspace.json"
        if not meta.exists():
            return None, None, None, None
        try:
            data = json.loads(meta.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return None, None, None, None
        uri = data.get("folder") or data.get("workspace")
        if not isinstance(uri, str):
            return None, None, None, None

        folder = unquote(uri)
        remote_host = None
        for prefix in ("file:///", "vscode-remote://"):
            if folder.startswith(prefix):
                folder = folder[len(prefix):]
        if folder.startswith("ssh-remote+"):
            authority, _, rest = folder[len("ssh-remote+"):].partition("/")
            # authority can be host, host:port, or an opaque proxy label
            remote_host = authority.split("%2B")[-1] or None
            folder = f"ssh-remote+{authority}/{rest}"
        elif folder.startswith(("wsl+", "dev-container+", "attached-container+")):
            scheme, _, rest = folder.partition("+")
            authority, _, tail = rest.partition("/")
            remote_host = f"{scheme}:{authority}"
            folder = f"{scheme}+{authority}/{tail}"

        if len(folder) > 1 and folder[1] == "%3A":
            folder = folder.replace("%3A", ":", 1)
        key, label = derive_workspace(Counter({folder: 1}))
        return key, label, remote_host, folder

    @staticmethod
    def _participant(data: dict, requests: list) -> tuple[str | None, str | None]:
        """Which assistant answered in this panel — (key, display name).

        `requests[].agent.extensionId` is the reliable signal: it survives on sessions
        that never recorded a `selectedModel`, which is most of them. `responderUsername`
        covers sessions written before the agent field existed, and the model vendor is
        the last resort.
        """
        seen: Counter = Counter()
        display: dict[str, str] = {}
        for req in requests:
            if not isinstance(req, dict):
                continue
            agent = req.get("agent")
            if not isinstance(agent, dict):
                continue
            ext = agent.get("extensionId")
            value = ext.get("value") if isinstance(ext, dict) else ext
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            seen[value.lower()] += 1
            name = agent.get("extensionDisplayName")
            if isinstance(name, str) and name:
                display.setdefault(value.lower(), name)
        if seen:
            ext_id = seen.most_common(1)[0][0]
            key, label = PARTICIPANTS.get(ext_id, (None, None))
            return key or ext_id, label or display.get(ext_id) or ext_id

        responder = data.get("responderUsername")
        if isinstance(responder, str) and responder.strip():
            name = responder.strip()
            key, label = RESPONDERS.get(name.lower(), (None, None))
            return key or name.lower().replace(" ", "-"), label or name

        vendor = ((data.get("selectedModel") or {}).get("metadata") or {}).get("vendor")
        if isinstance(vendor, str) and vendor.strip():
            key = vendor.strip().lower()
            return key, VENDORS.get(key, vendor.strip())
        return None, None

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        stats.files += 1
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            stats.error(f"unreadable:{path.name}")
            return
        if not isinstance(data, dict):
            return

        requests = data.get("requests")
        if not isinstance(requests, list) or not requests:
            return

        model = data.get("selectedModel") or {}
        model_meta = model.get("metadata") or {}
        model_name = model_meta.get("family") or model.get("identifier")

        messages: list[Message] = []
        seq = 0
        for req in requests:
            if not isinstance(req, dict):
                continue
            when = req.get("timestamp") or 0

            user_text = ""
            message = req.get("message")
            if isinstance(message, dict):
                user_text = (message.get("text") or "").strip()
            elif isinstance(message, str):
                user_text = message.strip()
            if user_text:
                msg = Message(native_id=req.get("requestId"), role="user",
                              created_at=when, seq=seq)
                msg.parts.append(self._offload(Part(
                    kind=KIND_TEXT, seq=0, text=user_text, embed_eligible=True)))
                messages.append(msg)
                seq += 1

            reply = Message(native_id=req.get("responseId") or
                            (f"{req.get('requestId')}:r" if req.get("requestId") else None),
                            role="assistant", created_at=when, seq=seq,
                            model=req.get("modelId") or model_name)
            for blk in req.get("response") or []:
                if not isinstance(blk, dict):
                    continue
                part = self._build_part(blk, len(reply.parts), stats)
                if part is not None:
                    reply.parts.append(part)
            if reply.parts:
                messages.append(reply)
                seq += 1

        if not messages:
            return

        stats.messages += len(messages)
        stats.parts += sum(len(m.parts) for m in messages)
        stats.sessions += 1

        participant, participant_label = self._participant(data, requests)
        ws_key, ws_label, remote_host, folder_uri = self._workspace_for(path)
        title = data.get("customTitle") or None
        times = [m.created_at for m in messages if m.created_at]

        yield Session(
            source_kind=self.kind,
            native_id=str(data.get("sessionId") or path.stem),
            # a remote session ran on the remote box, not on this one
            host=remote_host or self.host,
            title=title,
            title_source="provider" if title else None,
            workspace_key=ws_key,
            workspace_label=ws_label,
            model_primary=model_name,
            started_at=data.get("creationDate") or (min(times) if times else 0),
            ended_at=data.get("lastMessageDate") or (max(times) if times else None),
            raw_path=str(path),
            raw_hash=hashlib.sha256(raw).hexdigest(),
            messages=messages,
            meta={k: v for k, v in {
                "participant": participant,
                "participant_label": participant_label,
                "mode": (data.get("mode") or {}).get("kind"),
                "model_identifier": model.get("identifier"),
                "vendor": model_meta.get("vendor"),
                "extension": ((model_meta.get("extension") or {}).get("value")
                              if isinstance(model_meta.get("extension"), dict) else None),
                "requests": len(requests),
                # the workspace path with its case intact, for reopening in VS Code
                "folder_uri": folder_uri,
            }.items() if v is not None},
        )

    def _build_part(self, blk: dict, seq: int, stats: ParseStats) -> Part | None:
        kind = blk.get("kind")

        # Plain assistant prose has NO kind and hides its markdown in `value`.
        if kind is None:
            value = blk.get("value")
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_TEXT, seq=seq, text=text, embed_eligible=True))

        if kind == "thinking":
            value = blk.get("value")
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_THINKING, seq=seq, text=text, embed_eligible=True))

        # Assistant prose the panel flagged as risky to render (a command it does not
        # want click-to-run). It is still the answer, and it hides one level deeper
        # than plain prose — in `content.value` rather than `value`.
        if kind == "markdownVuln":
            content = blk.get("content")
            value = content.get("value") if isinstance(content, dict) else content
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                return None
            return self._offload(
                Part(kind=KIND_TEXT, seq=seq, text=text, embed_eligible=True))

        if kind in ("toolInvocationSerialized", "prepareToolInvocation"):
            name = blk.get("toolId") or blk.get("toolName") or "tool"
            label = blk.get("invocationMessage") or blk.get("pastTenseMessage")
            if isinstance(label, dict):
                label = label.get("value")
            return Part(
                kind=KIND_TOOL_USE, seq=seq, tool_name=str(name),
                text=str(label)[:300] if label else "",
                tool_ok=blk.get("isComplete") if "isComplete" in blk else None,
                bytes=len(json.dumps(blk, default=str)),
                embed_eligible=True)

        # Structural/editor chrome carries no conversational value. `elicitation` is a
        # modal asking the user to decide something ("Continue waiting?"), the same
        # class of thing as `confirmation`; the *Serialized suffixes are the persisted
        # forms of blocks whose live variants are already here.
        if kind in ("undoStop", "mcpServersStarting", "codeblockUri", "inlineReference",
                    "textEditGroup", "progressMessage", "progressTask", "codeCitation",
                    "command", "confirmation", "warning", "notebookEdit", "extensions",
                    "pullRequest", "toolInvocation", "treeData",
                    "progressTaskSerialized", "elicitation", "notebookEditGroup"):
            return None

        stats.unknown(f"{self.kind}:response:{kind}")
        return None

    def _offload(self, part: Part) -> Part:
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part
