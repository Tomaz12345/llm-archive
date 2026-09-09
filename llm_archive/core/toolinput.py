"""What a tool call actually did, read out of its arguments.

`part.text` has always held a 300-character line of intent — `command: pytest -q …`,
`file_path: C:\\…` — which is the right thing to show a human and the right thing to
put in FTS, and the wrong thing to compute over. This module turns the payload kept
alongside it (`part.tool_input`) into facts: which files a call touched, and which
shell commands it ran.

Pure by design: no database, no filesystem, no I/O. `normalise_path` must never stat
anything, because most of the paths in this archive are on a machine that is not this
one — an SSH box, a WSL guest, a GitHub repo. It is vocabulary, in the same sense
`models.py` is, which is why it lives in `core/` and not in `search/`.

The two hard-won rules, both of which have a test named after the failure they prevent:

1. Rules are keyed on `(source_kind, tool_name)`, never on the shape of the payload.
   claude.ai's `artifacts` tool takes an `input.command` of `create`/`update`/`rewrite`;
   a rule of "has a `command` key, therefore it is a shell command" invents 64 shell
   commands that were never run.

2. Nothing is inferred from shell text. See `_shell_paths`.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Callable
from urllib.parse import unquote

# What a call did to a file. Deliberately no `run` — running is what CommandFact is
# for, and keeping the two disjoint in verb space is what stops `who-touched` answering
# with files that were never opened.
ACTIONS = ("read", "write", "edit", "delete", "search", "list", "other")

WRITE_ACTIONS = frozenset({"write", "edit", "delete"})


@dataclass(frozen=True, slots=True)
class FileFact:
    path: str
    action: str


@dataclass(frozen=True, slots=True)
class CommandFact:
    text: str
    shell: str = "unknown"          # posix | powershell | unknown
    cwd: str | None = None
    ok: bool | None = None          # some sources carry an exit code the part does not


@dataclass(frozen=True, slots=True)
class ToolFacts:
    files: tuple[FileFact, ...] = ()
    commands: tuple[CommandFact, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.files or self.commands)


@dataclass(frozen=True, slots=True)
class NormalPath:
    path: str                       # the literal string the tool wrote
    norm: str                       # the match key, host-qualified; see normalise_path
    base: str                       # final segment of norm
    rel: str | None                 # norm with the workspace key stripped
    host_key: str | None            # ssh-remote+jon | wsl+ubuntu | github.com/o/r | None


EMPTY = ToolFacts()


# --------------------------------------------------------------------------- paths

# \\wsl$\Ubuntu\home\x and \\wsl.localhost\Ubuntu\home\x are the same guest.
_WSL = re.compile(r"^//(?:wsl\$|wsl\.localhost)/([^/]+)(/.*)?$", re.IGNORECASE)
_UNC = re.compile(r"^//([^/]+)(/.*)?$")
_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://([^/]*)(/.*)?$")
_DRIVE = re.compile(r"^[a-zA-Z]:(/|$)")


def _lexical(path: str) -> str:
    """Collapse `.`, `..` and repeated slashes without touching the filesystem.

    posixpath.normpath is the right tool and the wrong default: it turns "" into "."
    and would happily walk `..` above the root.
    """
    if not path:
        return ""
    out = posixpath.normpath(path)
    return "" if out == "." else out


def normalise_path(raw, *, cwd: str | None = None,
                   workspace_key: str | None = None) -> NormalPath | None:
    """Turn whatever a tool recorded into a comparable key, or None if it is not a path.

    `norm` is host-qualified: a remote path comes back as
    `ssh-remote+jon/home/tomaz/proj/main.py`, which is exactly the shape
    `workspace.key` already uses for remote roots (see the vscode_chat adapter). That
    is what makes `rel` a plain prefix strip, and what keeps `/home/tomaz/x` on two
    different boxes from colliding.

    Accepts VS Code's `{"path", "scheme", "authority"}` dict directly — that is how the
    only paths VS Code records arrive, and re-serialising them to a URI first would
    just mean parsing them back out again.
    """
    host_key: str | None = None

    if isinstance(raw, dict):
        # VS Code URI object. `path` is already decoded; `external` is not.
        text = raw.get("path") or raw.get("fsPath") or raw.get("external") or ""
        if not isinstance(text, str) or not text.strip():
            return None
        scheme, authority = raw.get("scheme"), raw.get("authority")
        if isinstance(authority, str) and authority.strip():
            host_key = unquote(authority).strip()
        elif isinstance(scheme, str) and scheme not in ("file", "untitled"):
            host_key = scheme
        original = text
        text = text.replace("\\", "/")
    elif isinstance(raw, str):
        original = raw
        text = raw.strip().replace("\\", "/")
    else:
        return None

    if not text:
        return None

    match = _SCHEME.match(text)
    if match:
        scheme, authority, rest = match.group(1).lower(), match.group(2), match.group(3) or "/"
        if scheme == "file":
            # file://host/path — an empty or localhost authority means "here".
            if authority and unquote(authority).lower() not in ("", "localhost"):
                host_key = unquote(authority)
        elif scheme in ("http", "https"):
            return None          # a URL is not a file this machine can be said to touch
        else:
            # vscode-remote://ssh-remote%2Bjon/home/… and friends.
            host_key = unquote(authority) if authority else host_key
        text = unquote(rest)

    if text.startswith("//"):
        wsl = _WSL.match(text)
        if wsl:
            host_key = f"wsl+{wsl.group(1).casefold()}"
            text = wsl.group(2) or "/"
        else:
            unc = _UNC.match(text)
            if unc:
                host_key = host_key or unc.group(1)
                text = unc.group(2) or "/"

    text = _lexical(text)
    if not text:
        return None

    absolute = text.startswith("/") or bool(_DRIVE.match(text))
    if not absolute and cwd:
        joined = normalise_path(cwd)
        if joined is not None:
            # Join on the raw form so the cwd's own host/drive survives.
            text = _lexical(f"{joined.norm}/{text}")
            host_key = host_key or joined.host_key
            absolute = True

    norm = text.rstrip("/").casefold()
    if not norm:
        return None

    ws = workspace_key.rstrip("/").casefold() if workspace_key else None

    # A remote workspace key is `<host>/<abs path>` (`ssh-remote+jon/home/t/proj`).
    # Most tools record a bare `/home/t/proj/...` with nothing in the string saying
    # which machine that is -- but the session's workspace does. Without this, every
    # path from a remote session is unqualified and never matches its own project.
    if ws and not host_key and not ws.startswith("/") and not _DRIVE.match(ws):
        ws_host, _, ws_path = ws.partition("/")
        if ws_path and norm.startswith("/" + ws_path + "/"):
            host_key = ws_host

    if host_key:
        norm = f"{host_key.casefold()}{'' if norm.startswith('/') else '/'}{norm}"

    base = norm.rsplit("/", 1)[-1]

    rel = None
    if ws and norm.startswith(ws + "/"):
        rel = norm[len(ws) + 1:]

    return NormalPath(path=original, norm=norm, base=base, rel=rel, host_key=host_key)


# ------------------------------------------------------------------------ commands

# Sequencing separates one command from the next; a pipe does not -- in `ls | grep x`
# the program IS `ls`. So the two are split separately.
_SEQ = re.compile(r"&&|\|\||[;\n]")
_PIPE = re.compile(r"\|")
# Navigation that PRECEDES the real command rather than being it. Nearly every Bash
# call in this archive opens `cd <project> && ...`, which made `cd` 66% of every
# command derived -- a ranking that says nothing about what actually runs. A bare
# `cd /x` with nothing after it is still a cd.
_NAVIGATION = frozenset({"cd", "pushd", "popd", "export", "set",
                         "source", ".", "cls", "clear"})
# Shell grammar, not programs. `for f in *.py; do python "$f"; done` is a python run,
# and walking past the keyword stages is what finds it -- which is also why `do` and
# `then` are here rather than only the openers.
_KEYWORDS = frozenset({"for", "while", "until", "if", "elif", "done", "fi",
                       "case", "esac", "select", "function", "in"})
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Leading tokens that stand in front of the real program WITHIN a stage. `do` and
# `then` are here rather than in _KEYWORDS because `do python x` is one stage whose
# command is python -- skipping the whole stage would land on `done`.
_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "command", "nice",
                       "doas", "xargs", "do", "then", "else"})
_SUFFIXES = (".exe", ".cmd", ".bat", ".ps1", ".sh")

# Programs whose first argument is a verb rather than an operand. Only these get a
# `subcommand`: `git` alone is a third of everything and says nothing, but `sed`'s
# first argument is a script (`1,60p`), `grep`'s is a pattern and `cat`'s is a
# redirection -- recording those as subcommands fills the ranking with noise.
_MULTIPLEXERS = frozenset({
    "git", "npm", "npx", "yarn", "pnpm", "uv", "pip", "pip3", "poetry", "conda",
    "docker", "podman", "kubectl", "helm", "cargo", "go", "dotnet", "gh", "apt",
    "apt-get", "brew", "systemctl", "terraform", "aws", "az", "gcloud", "alembic",
    "django-admin", "rails", "composer", "gem", "bundle", "nix", "pyenv", "nvm",
    # `python -m pytest` is the same shape as `git commit`.
    "python", "python3", "py",
})

POSIX, POWERSHELL, UNKNOWN = "posix", "powershell", "unknown"


def _argv0_of(head: str, shell: str) -> tuple[str, str | None]:
    """One stage of a command line, resolved to its program and subcommand.

    Returns ("", None) for a stage that is only environment assignments -- the caller
    treats that as a prefix and moves on to the next stage.
    """
    # POSIX shlex reads a backslash as an escape, which turns a Windows path into
    # "C:Pythonpython.exe". Half this archive's commands are Windows paths, so the
    # first token decides the mode rather than the shell alone.
    first = head.split()[0].lstrip(chr(34) + chr(39)) if head.split() else ""
    windows_ish = len(first) > 2 and first[0].isalpha() and first[1] == ":"
    try:
        tokens = shlex.split(head, posix=(shell != POWERSHELL and not windows_ish))
    except ValueError:
        # Unbalanced quotes are common in real command lines and must not be fatal.
        tokens = head.split()
    if not tokens:
        return "", None

    while tokens and (_ENV_ASSIGN.match(tokens[0])
                      or tokens[0].casefold() in _WRAPPERS):
        tokens = tokens[1:]
    if not tokens:
        return "", None

    argv0 = tokens[0].strip("'" + chr(34))
    argv0 = argv0.replace(chr(92), "/").rsplit("/", 1)[-1].casefold()
    for suffix in _SUFFIXES:
        if argv0.endswith(suffix) and len(argv0) > len(suffix):
            argv0 = argv0[: -len(suffix)]
            break
    if not argv0:
        return "", None

    sub = None
    for token in tokens[1:] if argv0 in _MULTIPLEXERS else ():
        candidate = token.strip("'" + chr(34))
        # A flag, a path, a variable standing in for one, or a redirection: `python -
        # <<PY` is a heredoc, not a `python <<py` subcommand.
        if not candidate or candidate.startswith(("-", "/", "$", "%", "~", "<", ">")):
            continue
        if "/" in candidate or chr(92) in candidate:
            break
        sub = candidate.casefold()
        break
    return argv0, sub


def split_argv0(command: str, shell: str = POSIX) -> tuple[str, str | None]:
    """The program a command line runs, and its subcommand when it has one.

    `git` alone is uninformative -- it is a third of everything -- so `git commit` and
    `git status` are kept apart. This is what makes `llma commands` with no argument a
    useful answer rather than a list of five verbs.

    Stages are walked rather than just taking the first, because the first is so often
    scaffolding: `cd <project> && pytest` is a pytest run, and `f="..."; python "$f"` is
    a python run. Taking the head made `cd` two thirds of this archive and dropped 476
    commands whose opening stage was nothing but a variable assignment.
    """
    if not command or not command.strip():
        return "", None

    stages = [x.strip() for x in _SEQ.split(command.strip()) if x.strip()]
    fallback: tuple[str, str | None] = ("", None)
    for i, stage in enumerate(stages):
        head = _PIPE.split(stage, 1)[0].strip()
        if not head:
            continue
        argv0, sub = _argv0_of(head, shell)
        if not argv0:
            continue                    # assignments only; the command is further on
        fallback = (argv0, sub)
        # Shell syntax that no amount of parsing turns into a program name: a
        # substitution, a subshell, a brace group, or a program held in a variable.
        if argv0[0] in "$(`{[<>&!*":
            continue
        if argv0 in _NAVIGATION or argv0 in _KEYWORDS:
            if i < len(stages) - 1:
                continue                # a prefix; the real command is further on
        return argv0, sub
    return fallback


def _shell_paths(command: str) -> tuple[FileFact, ...]:
    """Files a shell command touched. Deliberately always empty in v1.

    A shell line is not parseable without a shell. `git commit -m "fix src/foo.py"`
    names a path that was never touched; `for f in *.py` names none; `rg 'def load'
    src/` is a search, not a touch. The entire value of `who-touched` is that its
    answer is trustworthy — one false "session #412 edited src/db.py" that was really
    a grep of a commit message costs more than ten misses, because it turns the table
    into something you have to double-check, which is the same as not having it.

    The unambiguous subset does exist — a redirection target is always a write, `rm`'s
    arguments are always deletes — and this is where it goes when the table has earned
    the trust. Shipping it first means the first thing anyone sees at the top of every
    hot-files list is `logs/out.txt`.
    """
    return ()


# --------------------------------------------------------------------------- rules

Rule = tuple[tuple[str, str], ...] | Callable[[dict], ToolFacts]

_PATH_KEYS = ("file_path", "filePath", "notebook_path", "notebookPath", "path")

# An MCP server's arguments are somebody else's vocabulary; guessing at them is how
# `(code)` ends up in a hot-files list.
_SKIP_PREFIXES = ("mcp__", "mcp_")


def _files(payload: dict, *pairs: tuple[str, str]) -> ToolFacts:
    out = []
    for key, action in pairs:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            out.append(FileFact(value.strip(), action))
    return ToolFacts(files=tuple(out))


def _command(payload: dict, key: str = "command", *, shell: str = POSIX,
             cwd_key: str = "workdir") -> ToolFacts:
    value = payload.get(key)
    if isinstance(value, list):           # codex records argv, not a line
        value = " ".join(str(v) for v in value)
    if not isinstance(value, str) or not value.strip():
        return EMPTY
    cwd = payload.get(cwd_key)
    return ToolFacts(commands=(CommandFact(
        text=value.strip(), shell=shell,
        cwd=cwd if isinstance(cwd, str) and cwd.strip() else None),))


def _file_list(payload: dict, key: str, action: str) -> ToolFacts:
    values = payload.get(key)
    if not isinstance(values, list):
        return EMPTY
    return ToolFacts(files=tuple(
        FileFact(v.strip(), action) for v in values
        if isinstance(v, str) and v.strip()))


def _powershell(payload: dict) -> ToolFacts:
    return _command(payload, shell=POWERSHELL)


# --- codex apply_patch -------------------------------------------------------

_PATCH_LINE = re.compile(
    r"^\*\*\*\s+(Add|Update|Delete)\s+File:\s*(.+?)\s*$", re.MULTILINE)
_PATCH_ACTION = {"add": "write", "update": "edit", "delete": "delete"}


def _apply_patch(payload) -> ToolFacts:
    """`*** Begin Patch / *** Update File: <path>` — N files in one call.

    The patch arrives as a raw string, not JSON, which is why the payload for this
    tool is stored verbatim rather than re-serialised.
    """
    text = payload if isinstance(payload, str) else None
    if text is None and isinstance(payload, dict):
        for key in ("input", "patch", "arguments", "command"):
            value = payload.get(key)
            if isinstance(value, str) and "*** " in value:
                text = value
                break
    if not text:
        return EMPTY
    return ToolFacts(files=tuple(
        FileFact(path.strip(), _PATCH_ACTION[verb.casefold()])
        for verb, path in _PATCH_LINE.findall(text)))


# --- vscode_chat -------------------------------------------------------------

def _uris(payload: dict, action: str, key: str = "uris") -> ToolFacts:
    """VS Code names its files in a `uris` map, and its edits in `editedUris`.

    Only the completed record is allowed to produce facts: VS Code writes a
    `prepareToolInvocation` block AND a `toolInvocationSerialized` block for the same
    call, so counting both would double every file touch in the archive.
    """
    if payload.get("phase") == "prepare":
        return EMPTY
    values = payload.get(key)
    if isinstance(values, dict):
        values = list(values.values())
    if not isinstance(values, list):
        return EMPTY
    return ToolFacts(files=tuple(
        FileFact(v, action) for v in values if isinstance(v, (str, dict)) and v))


def _vscode_edit(payload: dict) -> ToolFacts:
    return _uris(payload, "edit", "editedUris")


def _run_in_terminal(payload: dict) -> ToolFacts:
    """The command is in `toolSpecificData`, never in the label.

    `isComplete` is true for a command that exited 127, so the exit code is the only
    honest source for `ok`.
    """
    if payload.get("phase") == "prepare":
        return EMPTY
    terminal = payload.get("terminal")
    if not isinstance(terminal, dict):
        return EMPTY
    line = terminal.get("commandLine")
    text = line.get("original") if isinstance(line, dict) else line
    if not isinstance(text, str) or not text.strip():
        return EMPTY

    language = str(terminal.get("language") or "").casefold()
    shell = POWERSHELL if language in ("pwsh", "powershell", "ps1") else (
        POSIX if language in ("sh", "bash", "zsh", "shellscript") else UNKNOWN)

    ok = None
    state = terminal.get("terminalCommandState")
    if isinstance(state, dict) and isinstance(state.get("exitCode"), int):
        ok = state["exitCode"] == 0

    return ToolFacts(commands=(CommandFact(text=text.strip(), shell=shell, ok=ok),))


# --- copilot_web -------------------------------------------------------------

def _github_file(payload: dict) -> ToolFacts:
    """A file read out of a GitHub repo, which is a host in its own right."""
    repo, path = payload.get("repo"), payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return EMPTY
    if isinstance(repo, str) and repo.strip():
        return ToolFacts(files=(FileFact(f"//github.com/{repo.strip()}/{path.strip('/')}",
                                         "read"),))
    return ToolFacts(files=(FileFact(path.strip(), "read"),))


_RULES: dict[tuple[str, str], Rule] = {
    # claude_code
    ("claude_code", "Read"):         (("file_path", "read"),),
    ("claude_code", "Write"):        (("file_path", "write"),),
    ("claude_code", "Edit"):         (("file_path", "edit"),),
    ("claude_code", "MultiEdit"):    (("file_path", "edit"),),
    ("claude_code", "NotebookEdit"): (("notebook_path", "edit"),),
    ("claude_code", "Artifact"):     (("file_path", "write"),),
    ("claude_code", "Grep"):         (("path", "search"),),
    ("claude_code", "Glob"):         (("path", "search"),),
    ("claude_code", "SendUserFile"): lambda p: _file_list(p, "files", "read"),
    ("claude_code", "Bash"):         _command,
    ("claude_code", "PowerShell"):   _powershell,

    # codex
    ("codex", "shell_command"):    _command,
    ("codex", "local_shell_call"): _command,
    ("codex", "apply_patch"):      _apply_patch,

    # opencode
    ("opencode", "read"):  (("filePath", "read"),),
    ("opencode", "write"): (("filePath", "write"),),
    ("opencode", "edit"):  (("filePath", "edit"),),
    ("opencode", "bash"):  _command,
    ("opencode", "glob"):  (("path", "search"),),
    ("opencode", "grep"):  (("path", "search"),),

    # vscode_chat — see _uris on why `phase` matters
    ("vscode_chat", "copilot_readFile"):           lambda p: _uris(p, "read"),
    ("vscode_chat", "copilot_getErrors"):          lambda p: _uris(p, "read"),
    ("vscode_chat", "copilot_listDirectory"):      lambda p: _uris(p, "list"),
    ("vscode_chat", "copilot_createFile"):         _vscode_edit,
    ("vscode_chat", "copilot_createDirectory"):    lambda p: _uris(p, "write"),
    ("vscode_chat", "copilot_applyPatch"):         _vscode_edit,
    ("vscode_chat", "copilot_replaceString"):      _vscode_edit,
    ("vscode_chat", "copilot_multiReplaceString"): _vscode_edit,
    ("vscode_chat", "run_in_terminal"):            _run_in_terminal,

    # copilot_web
    ("copilot_web", "getfile"): _github_file,
}

# `(source, tool)` pairs that carry a path- or command-shaped key which is NOT one.
# claude.ai's artifacts tool is the reason the rule table is keyed on the tool at all.
_MUTED: frozenset[tuple[str, str]] = frozenset({
    ("claude_web", "artifacts"),
    ("vscode_chat", "copilot_findTextInFiles"),
    ("vscode_chat", "copilot_findFiles"),
    ("vscode_chat", "copilot_searchCodebase"),
    ("vscode_chat", "manage_todo_list"),
    ("claude_code", "Grep"),          # pattern-only calls fall through to the fallback
    ("claude_code", "Glob"),          # a glob is a pattern, never a file
    ("opencode", "websearch"),
    ("opencode", "webfetch"),
    ("opencode", "todowrite"),
})

# Tools that touch no file and run no command. Listed so `unknown_tools` stays a real
# drift signal instead of 702 calls of steady-state noise. `Monitor` is here on purpose:
# its `until` is prose that often quotes a path, which is the shell-text trap wearing a
# different hat.
_NO_FACTS = frozenset({
    "WebFetch", "WebSearch", "ToolSearch", "AskUserQuestion", "ExitPlanMode",
    "EnterPlanMode", "TaskOutput", "TaskStop", "Agent", "Skill", "Monitor",
    "TodoWrite", "manage_todo_list", "ListAgents", "SendMessage", "ReportFindings",
    "ScheduleWakeup", "CronCreate", "CronList", "CronDelete", "PushNotification",
    "RemoteTrigger", "DesignSync", "web_search", "webSearch", "web_fetch",
    # VS Code panel tools that answer a question rather than touch a file.
    "runTests", "copilot_listCodeUsages", "copilot_searchWorkspaceSymbols",
    "copilot_think", "copilot_getChangedFiles", "copilot_runVscodeCmd",
    "copilot_vscodeAPI", "copilot_fetchWebPage", "copilot_githubRepo",
})


def _as_payload(payload):
    """Adapters store what the source gave them; that is a dict, or a JSON string, or
    neither. Codex hands over `arguments` as a JSON string and `apply_patch` as a raw
    patch, so a failed parse is data, not an error."""
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except ValueError:
                return payload
        return payload
    return None


def extract(source_kind: str, tool_name: str | None, payload) -> ToolFacts:
    """Facts from one tool call. Never raises; an unreadable payload yields nothing."""
    if not tool_name:
        return EMPTY
    if tool_name.startswith(_SKIP_PREFIXES):
        return EMPTY

    data = _as_payload(payload)
    if data is None:
        return EMPTY

    if tool_name in _NO_FACTS:
        return EMPTY

    key = (source_kind, tool_name)
    rule = _RULES.get(key)

    if rule is not None:
        try:
            if callable(rule):
                return rule(data)
            if isinstance(data, dict):
                facts = _files(data, *rule)
                if facts:
                    return facts
        except (AttributeError, TypeError, ValueError):
            return EMPTY

    if key in _MUTED or not isinstance(data, dict):
        return EMPTY

    # Fallback: an unlisted tool that names something path-shaped. Recorded as `other`
    # so a surprising row can be explained, and counted by the caller as drift.
    for name in _PATH_KEYS:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        # A rule-matched key is a path because the rule says so. Here nothing says so,
        # so require it to look like one: a bare word is far more often a mode, an id
        # or a language than a file.
        if "/" in value or chr(92) in value or "." in value.rsplit("/", 1)[-1]:
            return ToolFacts(files=(FileFact(value.strip(), "other"),))
    return EMPTY


def is_known(source_kind: str, tool_name: str | None) -> bool:
    """Whether the rule table has an opinion about this tool — the drift signal."""
    if not tool_name or tool_name.startswith(_SKIP_PREFIXES):
        return True
    return (tool_name in _NO_FACTS
            or (source_kind, tool_name) in _RULES
            or (source_kind, tool_name) in _MUTED)
