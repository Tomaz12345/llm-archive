"""Where a session actually lives — and how to get back to it.

The archive can show you a transcript but not return you to the conversation. Every row
already holds what that would take: `native_id` is the provider's own id, and for the
seven linkable web sources it is verbatim the last path segment of the chat's URL. For
the CLI agents it is the id `claude --resume` expects, with the real working directory
sitting in `meta`. Nothing turned any of that into a link, so reopening a session meant
hunting for it by hand in a sidebar, or retyping a `cd` and a `--resume`.

This module is the one place that decides. `resolve()` maps a session row to a `Target`;
the web UI, `llma open`, and the JSON/MCP payloads all render the same answer.

**The URL templates were verified, not guessed.** Each was checked by formatting every
session's `native_id` and testing for an exact hit in this machine's browser history:
t3chat matched 170/170, claude_web 50/53, and deepseek, mistral and copilot_web 1/1.
`grok` is the one template with no local evidence behind it and is marked as such below.

**Two sources cannot be linked at all**, and saying so is the point:

- `openrouter` — the Export Chat JSON carries no conversation id (see the adapter's
  docstring), so `native_id` is a root *message* id. Chat URLs are `?room=orc-…`; the two
  id spaces have no derivable relation, and the history confirms no overlap.
- a `vscode_chat` panel with no folder attached — there is nothing to open.

**A link that would 404 is worse than no link**, so a `Target` can be `blocked` with the
reason, or clickable but carrying a `warn`. The distinction matters most for T3 Chat,
where 132 of 170 sessions here are `visibility: archived`: archiving hides a thread from
the sidebar but does *not* delete it, so those stay clickable and merely say so. Grok's
`temporary` flag is the opposite — that conversation was never stored server-side.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import quote


class LaunchError(RuntimeError):
    """Raised when a `launch` target could not be started."""


# `native_id` reaches a URL and an argv, so it is checked against what the sources
# actually produce: uuids, Convex ids, `ses_…`, `msg-…`, Gemini's hex. Anything with a
# slash, a space or a quote in it is not one of those and does not get interpolated.
NATIVE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")

# Display names, so the button can say "open in T3 Chat" rather than "open in t3chat".
# `source.label` in the database says the same thing, but `resolve` is given one row and
# should not need a second query to name what it found.
LABELS = {
    "t3chat": "T3 Chat", "claude_web": "Claude.ai", "chatgpt": "ChatGPT",
    "gemini": "Gemini", "deepseek": "DeepSeek", "mistral": "Mistral",
    "grok": "Grok", "copilot_web": "GitHub Copilot", "openrouter": "OpenRouter",
    "vscode_chat": "VS Code", "claude_code": "Claude Code", "codex": "Codex",
    "opencode": "opencode",
}

WEB = {
    "t3chat": "https://t3.chat/chat/{id}",
    "claude_web": "https://claude.ai/chat/{id}",
    "chatgpt": "https://chatgpt.com/c/{id}",
    "gemini": "https://gemini.google.com/app/{id}",
    "deepseek": "https://chat.deepseek.com/a/chat/s/{id}",
    "mistral": "https://chat.mistral.ai/chat/{id}",
    "copilot_web": "https://github.com/copilot/c/{id}",
    # Unverified: there is no grok.com history on this machine to check against, and the
    # account reached Grok through x.ai's sign-in. Shape follows every other provider's.
    "grok": "https://grok.com/chat/{id}",
}

# Each CLI agent's resume invocation, and which `meta` key holds the directory to run it
# in. All three were read off `--help` on this machine rather than recalled.
CLI = {
    "claude_code": (("claude", "--resume", "{id}"), "cwds"),
    "codex": (("codex", "resume", "{id}"), "cwds"),
    "opencode": (("opencode", "-s", "{id}"), "directory"),
}

REMOTE_PREFIXES = ("ssh-remote+", "wsl+", "dev-container+", "attached-container+")

ARCHIVED_WARN = ("archived at the provider — the URL still opens the thread, but it no "
                 "longer appears in the sidebar there")


@dataclass(frozen=True)
class Target:
    """Where one session can be reopened, or why it cannot be."""

    mode: str                               # "url" | "launch"
    label: str                              # "open in T3 Chat"
    url: str | None = None                  # mode="url": an href, the OS opens it
    argv: tuple[str, ...] | None = None     # mode="launch": exact argv, never a shell string
    cwd: str | None = None
    display: str | None = None              # the command as text, for copy + tooltip
    warn: str | None = None                 # still clickable, but say this
    blocked: str | None = None              # not clickable; this is why

    @property
    def ok(self) -> bool:
        return self.blocked is None

    @property
    def copy_text(self) -> str | None:
        """What the "copy command" button puts on the clipboard.

        Plain `cd "…"` rather than cmd's `cd /d`: it is the spelling both cmd and
        PowerShell accept, and the person pasting this chose their own shell.
        """
        if not self.display:
            return None
        return f'cd "{self.cwd}"\n{self.display}' if self.cwd else self.display

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape carried by `--json` and the MCP payloads."""
        out: dict[str, Any] = {"mode": self.mode, "label": self.label}
        for key in ("url", "cwd", "warn", "blocked"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.display:
            out["command"] = self.display
        return out


def _blocked(kind: str, why: str, mode: str = "url") -> Target:
    return Target(mode=mode, label=f"open in {LABELS.get(kind, kind)}", blocked=why)


def _meta(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _vscode_uri(folder: str) -> str:
    """A `vscode://` URI for a workspace folder, local or remote.

    VS Code registers the scheme, so this needs no cooperation from the archive's server
    — the browser hands it straight to the editor. `safe` keeps `+` unescaped because the
    remote authority is spelled `ssh-remote+jon` and VS Code matches it literally.
    """
    path = quote(folder, safe="/:+@._-~")
    if folder.startswith(REMOTE_PREFIXES):
        return f"vscode://vscode-remote/{path}"
    return f"vscode://file/{path}"


def _cwd_from(meta: dict, key: str) -> str | None:
    """The directory to resume in — from `meta`, never from `workspace.key`.

    `derive_workspace` casefolds and slash-normalises its key so that six spellings of
    one project collapse into a single workspace row. That is right for grouping and
    wrong for `cd`: `meta` is where the raw, case-preserving path survived.
    """
    if key == "directory":
        value = meta.get("directory")
        return value if isinstance(value, str) and value else None
    cwds = meta.get("cwds")
    if not isinstance(cwds, dict) or not cwds:
        return None
    # A session drifts between a project root and its subdirectories; the one that
    # recorded the most turns is the one it was mostly run from.
    return max(cwds, key=lambda k: cwds.get(k) or 0)


def _local_host() -> str:
    from .ingest import local_host
    return local_host()


def resolve(row: Mapping[str, Any] | sqlite3.Row) -> Target:
    """Map one session row to where it can be reopened.

    Needs `source_kind` (or `source`), `native_id`, `meta`, `host`, `parent_session_id`
    and `workspace_key`; missing optional keys are treated as absent rather than fatal,
    so partial rows from different queries can all be passed in.
    """
    data = dict(row)
    kind = data.get("source_kind") or data.get("source") or ""
    name = LABELS.get(kind, kind or "the original app")
    native_id = data.get("native_id") or ""
    meta = _meta(data.get("meta"))
    label = f"open in {name}"

    if kind in WEB or kind == "openrouter":
        return _resolve_web(kind, name, native_id, meta, label)
    if kind == "vscode_chat":
        return _resolve_vscode(kind, native_id, meta, data, label)
    if kind in CLI:
        return _resolve_cli(kind, name, native_id, meta, data)
    return Target(mode="url", label=label,
                  blocked=f"no way to reopen a {kind or 'session'} from here")


def _resolve_web(kind: str, name: str, native_id: str, meta: dict,
                 label: str) -> Target:
    if kind == "openrouter":
        return _blocked(kind, "OpenRouter's per-chat export carries no conversation id, "
                              "so there is no URL to rebuild — only the root message id")
    # A temporary Grok chat was never written to the server; there is no thread to open.
    if kind == "grok" and meta.get("temporary"):
        return _blocked(kind, "a temporary chat — Grok never saved it server-side")
    # Gemini is the one source whose adapter already recorded the real URL; `unlinked`
    # marks the canvas documents it found with no conversation behind them.
    if kind == "gemini":
        if meta.get("unlinked"):
            return _blocked(kind, "this Gemini export is a canvas document with no "
                                  "conversation behind it")
        stored = meta.get("conversation_url")
        if isinstance(stored, str) and stored.startswith("https://"):
            return Target(mode="url", label=label, url=stored)

    if not NATIVE_ID.fullmatch(native_id):
        return _blocked(kind, "the stored id is not the shape this provider's URLs use")

    warn = None
    if (kind == "t3chat" and meta.get("visibility") == "archived") or \
       (kind == "chatgpt" and meta.get("archived")):
        warn = ARCHIVED_WARN
    return Target(mode="url", label=label,
                  url=WEB[kind].format(id=native_id), warn=warn)


def _resolve_vscode(kind: str, native_id: str, meta: dict, data: dict,
                    label: str) -> Target:
    """VS Code opens the *workspace*, which is as close as the editor allows.

    There is no `vscode://` URI for one chat session — the id in `chatSessions/<id>.json`
    is private to the workbench. Opening the folder puts the chat one click away in the
    Chat view's history, and the warning says so rather than implying more.
    """
    # `folder_uri` is the raw, case-preserving path recorded at ingest; `workspace_key`
    # is the casefolded fallback for sessions ingested before that field existed. The
    # difference only bites on a case-sensitive remote path.
    folder = meta.get("folder_uri") or data.get("workspace_key")
    if not isinstance(folder, str) or not folder:
        return _blocked(kind, "this chat was not attached to a folder, so there is no "
                              "workspace to open")
    return Target(
        mode="url", label=label, url=_vscode_uri(folder),
        warn="VS Code has no per-chat link — this opens the workspace; the conversation "
             "is in the Chat view's history")


def _resolve_cli(kind: str, name: str, native_id: str, meta: dict,
                 data: dict) -> Target:
    """A terminal, at the right directory, already inside the resumed session.

    The refusals below are ordered by how much is still worth saying. An unusable id or
    a missing directory leaves nothing to offer. But "you recorded this on the laptop"
    and "the tool is not on PATH here" are both cases where the command itself is still
    exactly right — it just has to run somewhere else — so those keep `cwd` and
    `display` filled in and the UI keeps offering copy-the-command beside the reason.
    """
    template, cwd_key = CLI[kind]
    label = f"resume in {name}"

    def refuse(why: str, **kw) -> Target:
        return Target(mode="launch", label=label, blocked=why, **kw)

    if not NATIVE_ID.fullmatch(native_id):
        return refuse("the stored id is not a resumable session id")
    if data.get("parent_session_id") is not None:
        return refuse("a subagent transcript — resume the session that spawned it")

    cwd = _cwd_from(meta, cwd_key)
    if not cwd:
        return refuse("no working directory was recorded for this session")

    argv = tuple(part.format(id=native_id) for part in template)
    display = " ".join(argv)

    # A session recorded elsewhere has paths that do not exist on this disk, so the
    # directory check below would be meaningless — and misleading — for one of those.
    # `--root`/`--host` ingests are exactly why the archive holds three machines.
    host = data.get("host")
    if host and host != _local_host():
        return refuse(f"recorded on {host}, not on this machine",
                      cwd=cwd, display=display)
    if not os.path.isdir(cwd):
        return refuse(f"{cwd} no longer exists", cwd=cwd, display=display)
    # Opening a terminal only for it to print "not recognized" is worse than saying so
    # up front; the command stays available to copy for wherever the tool does live.
    if shutil.which(argv[0]) is None:
        return refuse(f"{argv[0]} is not on PATH on this machine",
                      cwd=cwd, display=display)
    return Target(mode="launch", label=label, argv=argv, cwd=cwd, display=display)


_META_SQL = """
    SELECT s.id, s.native_id, s.meta, s.host, s.parent_session_id,
           src.kind AS source_kind, w.key AS workspace_key
    FROM session s JOIN source src ON src.id = s.source_id
    LEFT JOIN workspace w ON w.id = s.workspace_id
    WHERE s.id IN ({})
"""


def targets_for(con: sqlite3.Connection,
                ids: Iterable[int]) -> dict[int, Target]:
    """Resolve many sessions in one query, for list pages and search payloads."""
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}
    rows = con.execute(_META_SQL.format(",".join("?" * len(ids))),
                       tuple(ids)).fetchall()
    return {row["id"]: resolve(row) for row in rows}


def target_for(con: sqlite3.Connection, session_id: int) -> Target | None:
    """The one-session case. None when there is no such session."""
    return targets_for(con, [session_id]).get(session_id)


def launch(target: Target) -> None:
    """Open a terminal in `target.cwd`, already running `target.argv`.

    Windows Terminal when it is present, `start` into a console window otherwise. The
    command is always an argv list and never a shell string, and `cmd /k` keeps the
    window open so a failure is something you can read rather than a flash of black.
    """
    if target.mode != "launch" or target.blocked or not target.argv or not target.cwd:
        raise LaunchError(target.blocked or "this session cannot be launched")
    if os.name != "nt":
        raise LaunchError("launching a terminal is implemented for Windows only; "
                          f'run: cd "{target.cwd}" && {target.display}')

    terminal = shutil.which("wt.exe")
    if terminal:
        cmd = [terminal, "-d", target.cwd, "cmd", "/k", *target.argv]
    else:
        cmd = ["cmd.exe", "/c", "start", "", "/D", target.cwd,
               "cmd", "/k", *target.argv]
    try:
        subprocess.Popen(
            cmd, cwd=target.cwd, close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP)
    except OSError as exc:
        raise LaunchError(f"could not start a terminal: {exc}") from exc


def describe(target: Target) -> str:
    """One line for a terminal — what `llma open --print` prints."""
    if target.blocked:
        return f"cannot reopen: {target.blocked}"
    if target.mode == "url":
        line = f"{target.label}: {target.url}"
    else:
        line = f"{target.label}: {target.display}   (in {target.cwd})"
    return f"{line}\n  note: {target.warn}" if target.warn else line


__all__ = ["LaunchError", "Target", "describe", "launch", "resolve",
           "target_for", "targets_for"]
