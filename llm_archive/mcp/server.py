"""A stdio MCP server over the archive: `search`, `show`, `related`. Read-only.

The archive answers "have I solved this before?" — but until now only to whoever was
willing to open a browser tab and read. The agent already sitting in the terminal, about
to re-derive last March's fix from scratch, had no way in. This is that way in: three
tools over `api.py`, so an assistant can consult the archive mid-session.

**Read-only, and offline by construction.** Every tool is a SELECT. Nothing here ingests,
redacts, exports or deletes, and the server opens no socket — stdin and stdout are the
whole transport. The privacy claim in the README survives unchanged: an agent can read
the archive because it is already running on this machine, not because anything left it.

**No SDK.** The official `mcp` package would pull anyio, starlette, sse-starlette,
uvicorn and httpx into a project whose CLI installs on `typer` and `numpy` alone — a web
stack, to read newline-delimited JSON-RPC off a pipe. The three methods a tools-only
server must answer are `initialize`, `tools/list` and `tools/call`; they are below, and
they are not the part of this file that will need maintaining.

**stdout is the wire.** A frame is one line of JSON, and anything else written to
stdout — fastembed's first-run model download, a numpy warning, a debug print left in —
lands between two frames and desynchronises the client. So `serve()` points the *name*
`sys.stdout` at stderr and keeps the real handle privately: noise becomes visible in the
client's log instead of corrupting the protocol.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .. import api
from ..core import db, ingest
from ..search.hybrid import Filters

SERVER_NAME = "llm-archive"

# Protocol revisions this server's surface is correct for. Tools-only servers changed
# very little across them, so an unrecognised version — a client newer than this file —
# is answered with the newest we know rather than refused.
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL = SUPPORTED_PROTOCOLS[0]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

INSTRUCTIONS = (
    "A local, offline index of this machine's past LLM conversations — every session "
    "with any assistant, across chat sites and CLI coding agents. Search it before "
    "solving a problem from scratch: the decision, the fix or the command that worked "
    "is often already in here. Every tool is read-only."
)

_FILTERS = {
    "source": {
        "type": "array", "items": {"type": "string"},
        "description": "Restrict to these sources, e.g. claude_code, codex, opencode, "
                       "vscode_chat, chatgpt, claude_web, gemini, grok, deepseek, "
                       "mistral, t3chat, openrouter, copilot_web.",
    },
    "workspace": {
        "type": "string",
        "description": "Substring of the project/workspace label, e.g. payments-api. "
                       "Only agent sources carry one.",
    },
    "participant": {
        "type": "string",
        "description": "Which assistant answered inside a shared panel (vscode_chat): "
                       "copilot, remote-ssh, …",
    },
    "host": {
        "type": "string",
        "description": "Exact machine tag, for an archive holding several machines.",
    },
    "since": {"type": "string", "description": "Only sessions on or after YYYY-MM-DD."},
    "until": {"type": "string", "description": "Only sessions on or before YYYY-MM-DD."},
    "include_abandoned": {
        "type": "boolean", "default": False,
        "description": "Also search branches abandoned by an edit or rewind.",
    },
}

_READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

TOOLS = [
    {
        "name": "search",
        "title": "Search past sessions",
        "description":
            "Search the local archive of past LLM conversations — every session with "
            "any assistant, from chat sites and CLI coding agents alike. Use it before "
            "solving something from scratch, to find whether this error, decision or "
            "design has already come up: \"have I hit this deadlock before\", \"what "
            "did I decide about the retry policy\". Retrieval is hybrid (BM25 over "
            "everything including tool output, plus local embeddings), so a paraphrase "
            "finds a session that used entirely different words, and a query in one "
            "language finds a conversation held in another. Returns ranked sessions "
            "with the snippets that matched; pass a session_id to `show` to read one. "
            "Each result also carries `open`: where that conversation still "
            "lives — the provider URL, or the reason it cannot be reached — so "
            "you can cite the real thread rather than an archive id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are looking for. Describe the problem in "
                                   "your own words; it need not match the original "
                                   "wording.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "mode": {
                    "type": "string", "enum": ["hybrid", "keyword", "semantic"],
                    "default": "hybrid",
                    "description": "hybrid is right almost always. keyword for an exact "
                                   "string — a stack trace, a flag, a file path — which "
                                   "is also the only way to reach tool output, as that "
                                   "is indexed but never embedded.",
                },
                **_FILTERS,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "show",
        "title": "Read one session",
        "description":
            "Return one archived session as an ordered transcript of messages. The "
            "session_id comes from `search` or `related`. Tool calls and their results "
            "are left out unless you ask for them — they are most of an agent session's "
            "volume — so set include_tools when what matters is which command was run "
            "or what it returned. A long transcript is cut to about max_chars of "
            "JSON and says so: check `truncated` and `omitted_parts`, and ask again "
            "with a larger max_chars if the part you needed was past the cut.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "integer",
                               "description": "From a `search` or `related` result."},
                "include_tools": {
                    "type": "boolean", "default": False,
                    "description": "Include tool_use and tool_result parts.",
                },
                "include_abandoned": {
                    "type": "boolean", "default": False,
                    "description": "Include branches abandoned by an edit or rewind.",
                },
                "max_chars": {
                    "type": "integer", "minimum": 1000, "maximum": 400000,
                    "default": api.SESSION_CHARS,
                    "description": "Roughly how many characters of JSON to "
                                   "return, structure included — i.e. how much of "
                                   "your context this call may spend.",
                },
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "related",
        "title": "Sessions like this one",
        "description":
            "Find the sessions most similar to one you already have, with no query — "
            "the session's own content becomes the query. Use it when `show` turned up "
            "something close but not quite right, or to collect every time a recurring "
            "problem was worked on, which is usually under different words on different "
            "days. Returns the same ranked shape as `search`, never including the "
            "session asked about.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "integer",
                               "description": "The session to find neighbours for."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                **_FILTERS,
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
]


class Archive:
    """The database, opened on the first tool call rather than at startup.

    A client spawns this server when its own session begins, whether or not the archive
    is ever consulted. Connecting lazily keeps that free — and keeps `llma mcp` pointed
    at the wrong `--data-dir` from creating an empty archive there as a side effect.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir
        self._con = None
        self._vectors: Path | None = None

    def open(self):
        if self._con is None:
            db_path, _ = ingest.default_paths(self.data_dir)
            self._con = db.connect(db_path)
            self._vectors = (self.data_dir or db_path.parent) / "vectors"
        return self._con, self._vectors


# ------------------------------------------------------------------ tools ----

def _bounded(value, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    return max(low, min(high, int(value)))


def _mode(value) -> str:
    mode = value or "hybrid"
    if mode not in ("hybrid", "keyword", "semantic"):
        raise ValueError(f"mode must be hybrid, keyword or semantic; got {mode!r}")
    return mode


def _session_id(args: dict) -> int:
    if args.get("session_id") is None:
        raise ValueError("session_id is required")
    try:
        return int(args["session_id"])
    except (TypeError, ValueError):
        raise ValueError(f"session_id must be an integer, "
                         f"got {args['session_id']!r}") from None


def _filters(args: dict) -> Filters:
    source = args.get("source") or ()
    if isinstance(source, str):       # schema says array; a model may still send one name
        source = (source,)
    return Filters(sources=tuple(source),
                   participant=args.get("participant"),
                   workspace=args.get("workspace"),
                   host=args.get("host"),
                   since=api.parse_day(args.get("since")),
                   until=api.parse_day(args.get("until")),
                   include_abandoned=bool(args.get("include_abandoned")))


def _tool_search(archive: Archive, args: dict) -> dict:
    con, vectors = archive.open()
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    return api.search_payload(con, vectors, query,
                              limit=_bounded(args.get("limit"), 10, 1, 50),
                              filters=_filters(args), mode=_mode(args.get("mode")))


def _tool_show(archive: Archive, args: dict) -> dict:
    con, _ = archive.open()
    session_id = _session_id(args)
    payload = api.session_payload(
        con, session_id,
        tools=bool(args.get("include_tools")),
        abandoned=bool(args.get("include_abandoned")),
        budget=_bounded(args.get("max_chars"), api.SESSION_CHARS, 1000, 400_000))
    if payload is None:
        raise LookupError(f"no session #{session_id} in the archive")
    return payload


def _tool_related(archive: Archive, args: dict) -> dict:
    con, vectors = archive.open()
    session_id = _session_id(args)
    payload = api.related_payload(con, vectors, session_id,
                                  limit=_bounded(args.get("limit"), 10, 1, 50),
                                  filters=_filters(args))
    if payload is None:
        raise LookupError(f"no session #{session_id} in the archive")
    return payload


HANDLERS = {"search": _tool_search, "show": _tool_show, "related": _tool_related}


# --------------------------------------------------------------- protocol ----

def _log(message: str) -> None:
    """Diagnostics, to stderr, never through `print`.

    `print(..., file=sys.stderr)` falls back to *stdout* when stderr is None — which
    would put a log line in the middle of the protocol, the one thing this module is
    built to prevent.
    """
    if sys.stderr is not None:
        sys.stderr.write(f"{SERVER_NAME}: {message}\n")
        sys.stderr.flush()


def _result(rid, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": payload}


def _error(rid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _version() -> str:
    try:
        return version("llm-archive")
    except PackageNotFoundError:
        return "0+source"          # running from a clone that was never pip-installed


def _initialize(params: dict) -> dict:
    asked = params.get("protocolVersion")
    return {
        "protocolVersion": asked if asked in SUPPORTED_PROTOCOLS else LATEST_PROTOCOL,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "title": "LLM archive",
                       "version": _version()},
        "instructions": INSTRUCTIONS,
    }


def _call(archive: Archive, rid, params: dict) -> dict:
    name = params.get("name")
    handler = HANDLERS.get(name)
    if handler is None:
        return _error(rid, INVALID_PARAMS, f"unknown tool: {name!r}")
    try:
        payload = handler(archive, params.get("arguments") or {})
    except Exception as exc:
        # A tool that fails comes back as a *result* flagged isError, not an error
        # frame: an id that is not in the archive should send the model back to
        # `search`, and it can only do that if it gets to read what went wrong. Error
        # frames are reserved for the protocol itself.
        _log(f"tool {name} failed: {exc!r}")
        return _result(rid, {"isError": True,
                             "content": [{"type": "text",
                                          "text": f"{type(exc).__name__}: {exc}"}]})
    return _result(rid, {"content": [
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]})


def dispatch(archive: Archive, message: dict) -> dict | None:
    """One request in, one response out — or None for anything not owed an answer."""
    method = message.get("method")
    if method is None:
        return None                      # a response frame; this server sends no requests
    rid = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _result(rid, _initialize(params))
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": TOOLS})
    if method == "tools/call":
        return _call(archive, rid, params)
    if rid is None:
        return None                      # notifications/initialized, …/cancelled, …
    return _error(rid, METHOD_NOT_FOUND, f"unknown method: {method}")


def _write(stream, frame) -> None:
    # ensure_ascii keeps a frame pure ASCII on the wire. Half this archive is Slovene
    # and its tool output is arbitrary bytes; escaping at the envelope means neither can
    # be mangled by whatever encoding stdout was handed, and no frame can contain a raw
    # newline and split itself into two.
    stream.write(json.dumps(frame, ensure_ascii=True) + "\n")
    stream.flush()


def serve(data_dir: Path | None = None, stdin=None, stdout=None) -> None:
    """Answer JSON-RPC on stdin until the client closes it."""
    source = stdin or sys.stdin
    wire = stdout or sys.stdout
    for stream in (wire, source):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    if stdout is None and sys.stderr is not None:
        # See the module docstring: from here the real stdout is `wire` alone, and
        # everything that thinks it is printing goes to the client's log instead.
        sys.stdout = sys.stderr

    archive = Archive(data_dir)
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write(wire, _error(None, PARSE_ERROR, "invalid JSON"))
            continue
        if isinstance(message, list):
            # Batches, which the protocol allowed before 2025-06-18 and no longer does.
            # Answering one costs four lines and cannot break a client that never sends.
            replies = [r for r in (dispatch(archive, m) for m in message
                                   if isinstance(m, dict)) if r is not None]
            if replies:
                _write(wire, replies)
        elif isinstance(message, dict):
            reply = dispatch(archive, message)
            if reply is not None:
                _write(wire, reply)
        else:
            _write(wire, _error(None, INVALID_REQUEST, "request must be an object"))
