"""Local web UI for the archive.

Deliberately dependency-light: server-rendered Jinja, plain HTML forms, and `<details>`
for disclosure. No bundler, no vendored JavaScript framework, no CDN — the archive is
private (risk R6) and a page that fetches from the network is both a leak and a thing
that breaks when offline.

Binds to 127.0.0.1 only. Never expose this: the database concentrates API keys, .env
contents and private code from ten sources into one searchable place.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core import db, idb, ingest, lineage, reopen
from ..core.blobs import IMAGE_MIMES, BlobStore, blob_path, sniff_file
from ..export.render import (
    caption_of as _caption,
)
from ..export.render import (
    dag_summary as _dag_summary,
)
from ..export.render import (
    fmt_ms as _fmt_ms,
)
from ..export.render import (
    fmt_when as _fmt,
)
from ..export.render import (
    looks_like_markdown as _looks_like_markdown,
)
from ..export.render import (
    ref_url_of as _ref_url,
)
from ..export.render import (
    render_markdown as _render_markdown_raw,
)
from ..export.render import (
    resolve_full_text,
)
from ..export.render import (
    tags_for as _tags_for,
)
from ..search.chunker import strip_header
from ..search.fts import TOKEN_RE
from ..search.hybrid import DEFAULT_WEIGHTS, Filters
from ..search.hybrid import search as run_search

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))

MESSAGE_PAGE = 300      # a 1,000-turn session must not render as one wall

# Comfortably past the largest export here (a 14 MB T3 bulk file) and past a plausible
# ChatGPT archive with images, while still refusing to spool a mistaken 4 GB video into
# the drops folder.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _export_name(session_id: int, message_id: int | None = None) -> str:
    """Download filename stem. Message exports carry both ids: the message id alone
    would say nothing about which conversation the turn came out of, and these land in
    a downloads folder next to each other."""
    if message_id is None:
        return f"session-{session_id}"
    return f"session-{session_id}-message-{message_id}"


def _asset_version() -> str:
    """Stylesheet mtime, appended to its URL.

    StaticFiles answers with `Last-Modified` and no `Cache-Control`, so a browser is
    free to apply heuristic freshness and never revalidate. That is how a CSS change
    ships invisibly: the template updates, the rules for it do not, and the page renders
    with markup its stylesheet has never heard of.
    """
    try:
        return str(int((HERE / "static" / "app.css").stat().st_mtime))
    except OSError:
        return "0"


def create_app(data_dir: Path | None = None, *,
               fetch_images: bool = True) -> FastAPI:
    db_path, blob_dir = ingest.default_paths(data_dir)
    vectors_dir = db_path.parent / "vectors"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Started first, because it is the one thing here that waits on somebody else's
        # server: it should be running while the embedder loads, not after it.
        if fetch_images:
            from ..core.fetch_images import backfill_on_start
            backfill_on_start(db_path, blob_dir)
        # Load the embedding model before the first visitor waits on it. Cold ONNX
        # load is ~5 s in-process and was measured at 17 s through the server, which
        # reads as "this is broken" rather than "this is starting".
        try:
            from ..search.embed import get_embedder
            get_embedder().encode_queries(["warm up"])
        except Exception:  # noqa: BLE001 - keyword search must work regardless
            pass
        yield

    app = FastAPI(title="LLM Session Archive", docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    templates.env.globals["asset_v"] = _asset_version()

    # Schema and migrations run ONCE here. db.connect() executes the whole DDL script
    # and the migration ladder every call, which is fine for a CLI invocation and
    # wasteful per HTTP request — it was costing most of a second on every search.
    db.connect(db_path).close()

    def connect() -> sqlite3.Connection:
        con = sqlite3.connect(db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ---------------------------------------------------------------- helpers

    def model_type_facet(con) -> list[dict]:
        from ..stats.model_types import model_type_index

        index = model_type_index(con)
        return sorted(
            ({"type": k, "n": sum(n for _, n in v)} for k, v in index.items()),
            key=lambda e: -e["n"])

    def topic_facet(con) -> list[dict]:
        from ..search.topics import topic_facet as build_facet
        try:
            return build_facet(con)
        except sqlite3.OperationalError:
            return []            # pre-v10 archive; the migration adds the tables

    def facets(con) -> dict:
        return {
            "sources": con.execute("""
                SELECT src.kind, src.label, COUNT(*) n
                FROM session s JOIN source src ON src.id = s.source_id
                GROUP BY src.id ORDER BY n DESC""").fetchall(),
            # Several assistants share the VS Code chat panel; the source alone
            # cannot tell Copilot from the Remote-SSH participant.
            "participants": con.execute("""
                SELECT json_extract(s.meta,'$.participant') key,
                       COALESCE(json_extract(s.meta,'$.participant_label'),
                                json_extract(s.meta,'$.participant')) label,
                       COUNT(*) n
                FROM session s
                WHERE json_extract(s.meta,'$.participant') IS NOT NULL
                GROUP BY json_extract(s.meta,'$.participant')
                ORDER BY n DESC""").fetchall(),
            "workspaces": con.execute("""
                SELECT w.label, COUNT(*) n
                FROM session s JOIN workspace w ON w.id = s.workspace_id
                WHERE w.label IS NOT NULL AND w.label != ''
                GROUP BY w.label ORDER BY n DESC LIMIT 40""").fetchall(),
            "hosts": con.execute("""
                SELECT COALESCE(host,'') h, COUNT(*) n FROM session
                WHERE host IS NOT NULL AND host != '' GROUP BY host
                ORDER BY n DESC""").fetchall(),
            "tags": con.execute("""
                SELECT t.id, t.name, COUNT(st.session_id) n FROM tag t
                LEFT JOIN session_tag st ON st.tag_id = t.id
                GROUP BY t.id ORDER BY t.name""").fetchall(),
            "model_types": model_type_facet(con),
            # Derived, not typed: see search/topics.py. Empty until an index with
            # vectors has run, and the sidebar hides the group entirely when so.
            "topics": topic_facet(con),
            # Counted per session, not per call: a tool used 6,000 times in one
            # session is one place to look, and this is a filter on places to look.
            "tools": con.execute("""
                SELECT p.tool_name AS name, COUNT(DISTINCT m.session_id) n
                FROM part p JOIN message m ON m.id = p.message_id
                WHERE p.tool_name IS NOT NULL AND p.kind = 'tool_use'
                GROUP BY p.tool_name ORDER BY n DESC LIMIT 40""").fetchall(),
        }

    def totals(con) -> dict:
        row = con.execute("""
            SELECT (SELECT COUNT(*) FROM session) sessions,
                   (SELECT COUNT(*) FROM message WHERE on_active_path=1) messages,
                   (SELECT COUNT(*) FROM chunk) chunks""").fetchone()
        return dict(row)

    def _related_panel(con, session_id: int, limit: int = 6) -> list[dict]:
        """Sessions that resemble this one, for the reader's sidebar.

        `hybrid.related` has existed since the MCP server shipped and was reachable only
        from a terminal; this is the same call, rendered. Nothing is precomputed — the
        session becomes a query on the spot.

        Never raises. An archive indexed with --no-vectors still gets the keyword half,
        one with no index at all gets an empty panel, and neither is worth a 500 on a page
        whose actual job is showing the transcript.
        """
        from ..search.hybrid import related as run_related
        try:
            hits = run_related(con, vectors_dir, session_id, limit=limit,
                               snippets_per_hit=1)
        except Exception:  # noqa: BLE001 - a broken vector store must not hide the session
            return []
        # Condensed like a search result: `_snippets` strips the chunk header but leaves
        # the part's own newlines, which render as a ragged block under a one-line title.
        # Not highlighted -- there is no query here, only a session.
        return [{"hit": h, "when": _fmt(h.started_at),
                 "snippet": _condense(h.snippets[0].text) if h.snippets else "",
                 "role": h.snippets[0].role if h.snippets else ""}
                for h in hits]

    def _files_panel(con, session_id: int, limit: int = 40) -> list[dict]:
        """What this session opened, wrote and ran, from the derived tables.

        Same contract as `_related_panel`: never raises. A pre-v11 archive has no
        `touched_file` at all, and an archive ingested but never indexed has an empty
        one -- neither is worth a 500 on a page whose job is showing the transcript.
        """
        from .. import api
        try:
            return api.session_files(con, session_id, limit)
        except Exception:  # noqa: BLE001 - the transcript matters more than the panel
            return []

    def _commands_panel(con, session_id: int, limit: int = 10) -> list[dict]:
        """The programs this session ran. A 6,000-command session is not a list."""
        from .. import api
        try:
            return api.session_commands(con, session_id, limit)
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ views

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, q: str = "", mode: str = "hybrid",
             # Query() is required for a repeated param: without it FastAPI treats a
             # list annotation as a request body, so ?source=x&source=y arrives empty
             # and the filter silently does nothing.
             source: list[str] = Query(default=[]), participant: str = "",
             workspace: str = "", host: str = "", since: str = "", until: str = "",
             tag: str = "", model_type: str = "", topic: str = "",
             role: str = "", kind: str = "", tool: str = "",
             abandoned: bool = False, limit: int = 25):
        from ..stats import metrics
        from ..stats.model_types import model_type_index

        con = connect()
        selected_sources = tuple(s for s in (source or []) if s)

        hits = []
        error = None
        elapsed = 0.0
        if q.strip():
            import time
            t0 = time.perf_counter()
            try:
                hits = run_search(
                    con, vectors_dir, q, limit=limit,
                    filters=Filters(sources=selected_sources,
                                    participant=participant or None,
                                    workspace=workspace or None,
                                    host=host or None,
                                    topic=topic or None,
                                    role=role or None, kind=kind or None,
                                    tool=tool or None,
                                    since=_as_ms(since), until=_as_ms(until),
                                    include_abandoned=abandoned),
                    mode=mode, weights=DEFAULT_WEIGHTS)
            except Exception as exc:  # noqa: BLE001 - surface, never 500
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0

            if tag:
                keep = {r["session_id"] for r in con.execute(
                    "SELECT session_id FROM session_tag st JOIN tag t "
                    "ON t.id=st.tag_id WHERE t.name = ?", (tag,))}
                hits = [h for h in hits if h.session_id in keep]

            if model_type:
                models = [m for m, _ in model_type_index(con).get(model_type, [])]
                keep_mt = {r["id"] for r in con.execute(
                    f"""SELECT id FROM session
                        WHERE model_primary IN ({",".join("?" * len(models))})""",
                    models)} if models else set()
                hits = [h for h in hits if h.session_id in keep_mt]

        terms = _terms(q)
        # A `Hit` carries no native_id or meta, so the reopen targets come from one
        # extra query over the hit ids rather than from widening the search stack.
        targets = reopen.targets_for(con, [h.session_id for h in hits])
        rendered = [{
            "hit": h,
            "when": _fmt(h.started_at),
            "tags": _tags_for(con, h.session_id),
            "target": targets.get(h.session_id),
            "snippets": [{
                "role": s.role, "kind": s.kind,
                "html": _highlight(_condense(s.text), terms),
            } for s in h.snippets[:2]],
        } for h in hits]

        return templates.TemplateResponse(request, "search.html", {
            "q": q, "mode": mode, "hits": rendered,
            "facets": facets(con), "totals": totals(con),
            "selected_sources": selected_sources, "participant": participant,
            "workspace": workspace,
            "host": host, "since": since, "until": until, "tag": tag,
            "model_type": model_type, "topic": topic,
            "role": role, "kind": kind, "tool": tool,
            "abandoned": abandoned, "elapsed": elapsed, "error": error,
            # a stale index misleads here more than anywhere else: these are the results
            "index": metrics.index_health(con), "fmt_when": _fmt,
        })

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    def session_view(request: Request, session_id: int, offset: int = 0,
                     tools: bool = True, abandoned: bool = False, q: str = ""):
        from ..stats import model_types

        con = connect()
        meta = con.execute("""
            SELECT s.*, src.kind AS source_kind, src.label AS source_label,
                   COALESCE(w.label,'') AS workspace, w.key AS workspace_key,
                   json_extract(s.meta,'$.participant_label') AS participant
            FROM session s JOIN source src ON src.id = s.source_id
            LEFT JOIN workspace w ON w.id = s.workspace_id
            WHERE s.id = ?""", (session_id,)).fetchone()
        if meta is None:
            return HTMLResponse("<h1>404</h1><p>No such session.</p>", status_code=404)

        where_abandoned = "" if abandoned else "AND m.on_active_path = 1"
        total = con.execute(
            f"SELECT COUNT(*) n FROM message m WHERE m.session_id = ? "
            f"{where_abandoned}", (session_id,)).fetchone()["n"]

        rows = con.execute(f"""
            SELECT m.id, m.role, m.seq, m.on_active_path, m.is_sidechain,
                   m.created_at, m.model
            FROM message m WHERE m.session_id = ? {where_abandoned}
            ORDER BY m.seq, m.id LIMIT ? OFFSET ?""",
            (session_id, MESSAGE_PAGE, offset)).fetchall()

        terms = _terms(q)
        messages = []
        for row in rows:
            parts = con.execute("""
                SELECT p.id, p.kind, p.text, p.tool_name, p.tool_ok, p.bytes, p.blob_id,
                       p.duration_ms, b.sha256
                FROM part p LEFT JOIN blob b ON b.id = p.blob_id
                WHERE p.message_id = ? ORDER BY p.seq""",
                (row["id"],)).fetchall()
            # "hide tool calls" means the machinery, not the pictures: an image is
            # content the way text is, and hiding it hid half of what the turn said.
            shown = [p for p in parts
                     if tools or p["kind"] in ("text", "thinking", "image")]
            if not shown:
                continue

            def _full_text(p) -> str:
                return resolve_full_text(p["kind"], p["text"], p["blob_id"],
                                          p["sha256"], blob_dir)

            def _render_part(p) -> dict:
                raw = strip_header(_full_text(p))
                # Prose (a chat turn) is always markdown. Tool output (a file read, a
                # command result) is raw bytes by default — rendering an unfenced code
                # dump as markdown would collapse the indentation that makes it readable —
                # except when it's plainly itself a markdown document, e.g. `cat`ing a
                # plan or README back into context.
                narrative = p["kind"] in ("text", "thinking")
                md = narrative or _looks_like_markdown(raw)
                # A char count alone misjudges markdown: a 2 KB doc with a table and a
                # few headings is short in bytes but tall on screen — every `|a|b|` row
                # and `##` line costs a table cell's or a heading's worth of vertical
                # space, not one line of prose. Line count tracks the render height
                # that actually matters here.
                big = len(raw) > RENDER_CAP_CHARS or raw.count("\n") > BIG_LINE_THRESHOLD
                source = raw[:RENDER_CAP_CHARS]
                html = _render_markdown(source, terms) if md else f"<pre>{_highlight(source, terms)}</pre>"
                if len(raw) > RENDER_CAP_CHARS:
                    html += (f'<p class="clip-note">— cut off at {RENDER_CAP_CHARS // 1000} KB '
                             f'of {len(raw) // 1000} KB; use "save" to get the rest —</p>')
                return {
                    "kind": p["kind"], "tool_name": p["tool_name"],
                    "tool_ok": p["tool_ok"], "bytes": p["bytes"],
                    "duration": _fmt_ms(p["duration_ms"]),
                    "has_blob": p["blob_id"] is not None,
                    "sha": p["sha256"],
                    "caption": _caption(p["text"]),
                    "ref_url": _ref_url(p["text"]),
                    "md": md,
                    "big": big,
                    "html": html,
                    "raw": raw,
                }

            messages.append({
                "row": row, "when": _fmt(row["created_at"], time=True),
                "parts": [_render_part(p) for p in shown],
            })

        # There's no server-side publish step: the Artifact tool that turns a file into
        # a shared link only exists inside a Claude agent session, not this process. So
        # "export link" copies the ask, not a URL — `json.dumps` because a session title
        # is untrusted archive text and this is going straight into a JS string literal.
        share_prompt = (f'Export session #{session_id} '
                        f'("{meta["title"] or "(untitled)"}") from my llm-archive and '
                        'publish it as a shared link.')

        # Where this conversation still lives — a provider URL, a vscode:// hand-off, or
        # a terminal this server can spawn. See `core/reopen.py` for why some are refused.
        target = reopen.resolve(meta)
        # `json.dumps` for the same reason `share_prompt` uses it: a cwd is archive text
        # heading straight into a JS string literal.
        copy_cmd_json = json.dumps(target.copy_text or "")

        return templates.TemplateResponse(request, "session.html", {
            "totals": totals(con),
            "meta": meta, "messages": messages, "target": target,
            "copy_cmd_json": copy_cmd_json,
            # only set when the provider itself says the thread is archived, so the
            # badge means "hidden in their sidebar", not "old"
            "archived": (target.warn if target.warn == reopen.ARCHIVED_WARN else None),
            "model_type": model_types.classify(meta["model_primary"]),
            "when": _fmt(meta["started_at"], time=True),
            "tags": _tags_for(con, session_id),
            "all_tags": con.execute("SELECT name FROM tag ORDER BY name").fetchall(),
            "offset": offset, "page": MESSAGE_PAGE, "total": total,
            "tools": tools, "abandoned": abandoned, "q": q,
            "dag": _dag_summary(con, session_id),
            "share_prompt_json": json.dumps(share_prompt),
            # Two sessions can be one conversation: --resume forks a new transcript and
            # replays the old one into it. Without this the reader has no way to know the
            # other half exists.
            "lineage": lineage.chain(con, session_id),
            "related": _related_panel(con, session_id),
            # What the session actually did, as opposed to what it said. Derived at
            # index time; empty rather than absent when the archive has not been
            # indexed since the tool payloads landed.
            "files": _files_panel(con, session_id),
            "commands": _commands_panel(con, session_id),
        })

    def _export_html(session_id: int, *, download: bool, tools: bool, abandoned: bool,
                     message_id: int | None = None):
        """A session (or one message of it) as a self-contained HTML file — no `/blob/`
        src's, no server needed to open it. See `export/render.py`/`export/html.py`."""
        from ..export.html import to_html
        from ..export.render import build_session

        con = connect()
        session = build_session(con, blob_dir, session_id, tools=tools,
                                abandoned=abandoned, message_id=message_id)
        if session is None:
            return HTMLResponse("<h1>404</h1><p>No such session.</p>", status_code=404)

        headers = {}
        if download:
            headers["Content-Disposition"] =                 f'attachment; filename="{_export_name(session_id, message_id)}.html"'
        return HTMLResponse(to_html(session, blob_dir, standalone=True), headers=headers)

    def _export_md(session_id: int, *, tools: bool, abandoned: bool,
                   message_id: int | None = None):
        """Plain Markdown when there are no images to bundle (the common case); a
        `.md.zip` of the text plus an `assets/` folder when there are — one download
        link either way, the content decides the container."""
        import io
        import zipfile

        from ..export.markdown import to_markdown
        from ..export.render import build_session

        con = connect()
        session = build_session(con, blob_dir, session_id, tools=tools,
                                abandoned=abandoned, message_id=message_id)
        if session is None:
            return HTMLResponse("<h1>404</h1><p>No such session.</p>", status_code=404)

        name = _export_name(session_id, message_id)
        export = to_markdown(session, blob_dir)
        if not export.assets:
            return PlainTextResponse(
                export.text, media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{name}.md"'})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{name}.md", export.text)
            for asset_name, data in export.assets.items():
                zf.writestr(f"assets/{asset_name}", data)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}.md.zip"'})

    @app.get("/session/{session_id}/export.html")
    def session_export_html(session_id: int, download: bool = False,
                            tools: bool = True, abandoned: bool = True):
        return _export_html(session_id, download=download, tools=tools,
                            abandoned=abandoned)

    @app.get("/session/{session_id}/message/{message_id}/export.html")
    def message_export_html(session_id: int, message_id: int, download: bool = True,
                            tools: bool = True):
        """One message on its own, still carrying the session header that says where it
        came from. `download` defaults true here: unlike the whole-session export there
        is no reason to preview a single turn in the browser — it is already on screen
        in the live view, which is where this link is clicked from."""
        return _export_html(session_id, download=download, tools=tools,
                            abandoned=True, message_id=message_id)

    @app.get("/session/{session_id}/message/{message_id}/export.md")
    def message_export_md(session_id: int, message_id: int, tools: bool = True):
        """One message as Markdown — see `message_export_html`."""
        return _export_md(session_id, tools=tools, abandoned=True,
                          message_id=message_id)

    @app.get("/session/{session_id}/export.md")
    def session_export_md(session_id: int, tools: bool = True, abandoned: bool = True):
        return _export_md(session_id, tools=tools, abandoned=abandoned)

    @app.get("/export")
    def export_many(workspace: str = "", source: str = "", participant: str = "",
                    host: str = "", tag: str = "", model_type: str = ""):
        """Every session matching `/browse`'s current filters, as one zip — the same
        filter vocabulary and the same batch writer (`export/batch.py`) the CLI uses."""
        import io

        from ..export.batch import export_batch, select_session_ids

        con = connect()
        ids = select_session_ids(
            con, workspace=workspace or None, source=source or None,
            participant=participant or None, host=host or None, tag=tag or None,
            model_type=model_type or None)
        if not ids:
            return HTMLResponse("<h1>404</h1><p>No sessions matched.</p>", status_code=404)

        buf = io.BytesIO()
        export_batch(con, blob_dir, ids, buf, fmt={"html", "md"}, zip_output=True)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="export.zip"'})

    @app.get("/blob/{sha}")
    def blob(sha: str, download: bool = False):
        """Serve one stored blob, keyed by content hash.

        Keyed on the sha rather than the part id because ingest renumbers part ids —
        an `<img src>` pinned to a part id would rot on the next `--force` re-ingest.
        The hash is validated as 64 hex characters before it is ever joined to a path,
        so no request can walk out of the blob directory.
        """
        con = connect()
        row = con.execute("SELECT path FROM blob WHERE sha256 = ?", (sha,)).fetchone()
        if row is None:
            return HTMLResponse("<h1>404</h1><p>No such blob.</p>", status_code=404)

        path = blob_path(blob_dir, sha, row["path"])
        if path is None:
            return HTMLResponse(
                "<h1>410</h1><p>Recorded, but the file is no longer on disk.</p>",
                status_code=410)

        mime = sniff_file(path)
        # Anything not recognised as an image is offered as a download rather than
        # rendered: the store is full of raw tool output, and a text dump served as
        # a guessed type is how a local viewer turns archive content into markup.
        inline = mime in IMAGE_MIMES and not download
        return FileResponse(
            path, media_type=mime if inline else "application/octet-stream",
            headers={"Content-Disposition":
                     ("inline" if inline else f'attachment; filename="{sha[:16]}.bin"'),
                     "Cache-Control": "private, max-age=86400"})

    @app.post("/session/{session_id}/open")
    def open_session(request: Request, session_id: int):
        """Spawn a terminal already inside a CLI session — `claude --resume <id>` and
        friends, in the directory that session actually ran in.

        This is the only endpoint in the app that starts a process, so it is deliberately
        narrow. The target is re-resolved from the database on every call: nothing about
        the command, its arguments or its directory is ever taken from the request, so
        the worst a caller can do is ask for a session id.

        There is no CSRF token anywhere in this app — every other POST is a plain form —
        so `Sec-Fetch-Site` carries the weight instead. Browsers set it on their own and
        a page cannot forge it, and `fetch` from this UI sends `same-origin`. A missing
        header means a client that is not a browser (curl, a test), which is fine; a
        cross-site one means some other page tried, which is not.
        """
        site = request.headers.get("sec-fetch-site")
        if site is not None and site not in ("same-origin", "none"):
            return JSONResponse({"error": "cross-site requests cannot open sessions"},
                                status_code=403)

        target = reopen.target_for(connect(), session_id)
        if target is None:
            return JSONResponse({"error": f"no session #{session_id}"}, status_code=404)
        if target.mode != "launch":
            return JSONResponse(
                {"error": "this session opens with a link, not a terminal",
                 "url": target.url}, status_code=400)
        if target.blocked:
            return JSONResponse({"error": target.blocked}, status_code=409)
        try:
            reopen.launch(target)
        except reopen.LaunchError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"opened": target.display, "cwd": target.cwd})

    @app.post("/session/{session_id}/tag")
    def add_tag(session_id: int, name: str = Form(...)):
        name = name.strip()
        if name:
            con = connect()
            con.execute("INSERT OR IGNORE INTO tag(name) VALUES (?)", (name,))
            tag_id = con.execute("SELECT id FROM tag WHERE name = ?",
                                 (name,)).fetchone()["id"]
            con.execute("INSERT OR IGNORE INTO session_tag(session_id, tag_id) "
                        "VALUES (?,?)", (session_id, tag_id))
            con.commit()
        return RedirectResponse(f"/session/{session_id}", status_code=303)

    @app.post("/session/{session_id}/untag")
    def remove_tag(session_id: int, name: str = Form(...)):
        con = connect()
        con.execute("""DELETE FROM session_tag WHERE session_id = ? AND tag_id =
                       (SELECT id FROM tag WHERE name = ?)""", (session_id, name))
        con.execute("""DELETE FROM tag WHERE id NOT IN
                       (SELECT tag_id FROM session_tag)""")
        con.commit()
        return RedirectResponse(f"/session/{session_id}", status_code=303)

    @app.get("/stats", response_class=HTMLResponse)
    def stats_view(request: Request):
        from ..stats import charts, metrics

        con = connect()
        data = metrics.everything(con)

        vol = data["volume"]
        source_labels = {s["kind"]: s["label"] for s in data["sources"]}
        ordered = [k for k in source_labels if k in vol["series"]]

        legend = [charts.Legend(source_labels.get(k, k),
                                f"{sum(vol['series'][k]):,}", i)
                  for i, k in enumerate(ordered)]

        msg_vol = data["messages_volume"]
        msg_ordered = [k for k in source_labels if k in msg_vol["series"]]
        msg_legend = [charts.Legend(source_labels.get(k, k),
                                    f"{sum(msg_vol['series'][k]):,}", i)
                      for i, k in enumerate(msg_ordered)]

        # Terminal / editor panel / web, in that order — the question is where the work
        # happens, so keep the surfaces in a fixed order rather than by volume.
        surf = data["surface_split"]
        surf_ordered = [k for k in ("cli", "editor_panel", "web") if k in surf["series"]]
        surf_ordered += [k for k in surf["series"] if k not in surf_ordered]
        surf_legend = [charts.Legend(surf["labels"].get(k, k),
                                     f"{sum(surf['series'][k]):,}", i)
                       for i, k in enumerate(surf_ordered)]

        tbm = data["tools_by_model"]
        tool_legend = [charts.Legend(name, f"{sum(counts):,}", i)
                       for i, (name, counts) in enumerate(tbm["series"].items())]

        return templates.TemplateResponse(request, "stats.html", {
            "totals": totals(con),
            "d": data,
            "chart_volume": charts.stacked_bars(
                vol["months"], {k: vol["series"][k] for k in ordered}),
            "legend_volume": charts.legend_html(legend),
            "chart_messages_volume": charts.stacked_bars(
                msg_vol["months"], {k: msg_vol["series"][k] for k in msg_ordered}),
            "legend_messages_volume": charts.legend_html(msg_legend),
            "chart_surface": charts.stacked_bars(
                surf["months"], {k: surf["series"][k] for k in surf_ordered}),
            "legend_surface": charts.legend_html(surf_legend),
            "chart_cumulative": charts.line_chart(
                data["cumulative"]["months"], data["cumulative"]["values"]),
            "chart_heatmap": charts.heatmap(
                data["heatmap"]["grid"], data["heatmap"]["labels"],
                data["heatmap"]["peak"]),
            "chart_tools": charts.stacked_hbars(tbm["tools"], tbm["series"]),
            "legend_tools": charts.legend_html(tool_legend),
            "chart_workspaces": charts.hbars(
                [(w["label"], w["turns"], f"{w['turns']:,}")
                 for w in data["workspaces"]], accent_index=2),
            "chart_shape": charts.hbars(
                [(b, v, f"{v}") for b, v in zip(data["shape"]["labels"],
                                                data["shape"]["values"])],
                accent_index=1),
            "fmt_when": _fmt,
            "fmt_ms": _fmt_ms,
        })

    @app.get("/workspace/{label}", response_class=HTMLResponse)
    def workspace_view(request: Request, label: str, key: str = "", file: str = "",
                       limit: int = 50, offset: int = 0):
        """One project, across every agent and machine that worked on it.

        A label routinely names several `workspace` rows -- per source, and per root --
        so the page resolves to a SET of ids and names the roots it merged rather than
        quietly picking one.
        """
        from ..stats import charts, workspaces as ws

        con = connect()
        try:
            rows = ws.resolve(con, label, key or None)
            if not rows:
                return HTMLResponse(f"no workspace {label!r}", status_code=404)

            files = ws.hot_files(con, rows)
            programs = ws.top_programs(con, rows)
            total = ws.session_count(con, rows, file or None)
            listed = ws.sessions(con, rows, limit=limit, offset=offset,
                                 file=file or None)
            return templates.TemplateResponse(request, "workspace.html", {
                "totals": totals(con),
                "label": rows[0]["label"] or label,
                "roots": rows,
                "key": key,
                "file": file,
                "overview": ws.overview(con, rows),
                "files": files,
                "programs": programs,
                "branches": ws.branches(con, rows),
                "sessions": listed,
                "total": total,
                "offset": offset,
                "page": limit,
                "chart_files": charts.hbars(
                    [(f["key"], f["calls"], f"{f['calls']:,}") for f in files[:15]],
                    accent_index=2),
                "chart_programs": charts.hbars(
                    [(p["program"], p["runs"], f"{p['runs']:,}") for p in programs],
                    accent_index=1),
                "fmt": _fmt,
            })
        finally:
            con.close()

    @app.get("/browse", response_class=HTMLResponse)
    def browse(request: Request, workspace: str = "", source: str = "",
               participant: str = "", host: str = "", tag: str = "",
               model_type: str = "", topic: str = "",
               limit: int = 100, offset: int = 0):
        from ..stats import model_types

        con = connect()
        clauses, params = [], []
        if workspace:
            clauses.append("COALESCE(w.label,'') = ?")
            params.append(workspace)
        if source:
            clauses.append("src.kind = ?")
            params.append(source)
        if participant:
            clauses.append("json_extract(s.meta,'$.participant') = ?")
            params.append(participant)
        if host:
            clauses.append("COALESCE(s.host,'') = ?")
            params.append(host)
        if tag:
            clauses.append("""s.id IN (SELECT st.session_id FROM session_tag st
                                       JOIN tag t ON t.id=st.tag_id WHERE t.name=?)""")
            params.append(tag)
        if topic:
            clauses.append("""s.id IN (SELECT st.session_id FROM session_topic st
                                       JOIN topic t ON t.id=st.topic_id
                                      WHERE t.slug=?)""")
            params.append(topic)
        if model_type:
            # no `model_type` column to join on — resolve the matching raw model
            # strings in Python (model_types.classify) and filter on those instead.
            matching = [m for m, _ in model_types.model_type_index(con).get(model_type, [])]
            if matching:
                clauses.append(f"s.model_primary IN ({','.join('?' * len(matching))})")
                params.extend(matching)
            else:
                clauses.append("0")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = con.execute(f"""
            SELECT s.id, s.title, s.started_at, s.msg_count, s.host, s.model_primary,
                   s.continues_session_id,
                   src.kind AS source_kind, COALESCE(w.label,'') AS workspace,
                   json_extract(s.meta,'$.participant_label') AS participant
            FROM session s JOIN source src ON src.id = s.source_id
            LEFT JOIN workspace w ON w.id = s.workspace_id
            {where}
            ORDER BY s.started_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset)).fetchall()
        count = con.execute(f"""
            SELECT COUNT(*) n FROM session s
            JOIN source src ON src.id = s.source_id
            LEFT JOIN workspace w ON w.id = s.workspace_id {where}""",
            tuple(params)).fetchone()["n"]

        targets = reopen.targets_for(con, [r["id"] for r in rows])
        return templates.TemplateResponse(request, "browse.html", {
            "rows": [{"r": r, "when": _fmt(r["started_at"]),
                      "target": targets.get(r["id"]),
                      "model_type": model_types.classify(r["model_primary"])}
                     for r in rows],
            "facets": facets(con), "totals": totals(con), "count": count,
            "workspace": workspace, "source": source, "participant": participant,
            "host": host, "tag": tag, "model_type": model_type, "topic": topic,
            "offset": offset, "limit": limit,
        })

    # -------------------------------------------------------------- importing

    def import_page(request: Request, con, taken=None, results=None,
                    error: str | None = None):
        """The one page that answers both halves of "how do I add data?".

        What the archive already holds (the drop ledger), and what it is still waiting
        for (the freshness table, which knows where each provider's export button is).
        Splitting those across two pages means the answer to "what should I go and
        download?" lives somewhere you only visit when you already know.
        """
        from ..core import freshness, intake

        rows = freshness.report(con)
        return templates.TemplateResponse(request, "import.html", {
            "drops": [{"r": r,
                       "when": _fmt(r["exported_at"] or r["added_at"]),
                       "archived": intake.ARCHIVE in Path(r["path"]).parts}
                      for r in intake.ledger(con)],
            "freshness": rows,
            "summary": freshness.summary(rows),
            "passed_over": intake.passed_over(con),
            "taken": taken or [],
            "results": results or [],
            "error": error,
            "totals": totals(con),
        })

    @app.get("/import", response_class=HTMLResponse)
    def import_view(request: Request):
        return import_page(request, connect())

    @app.post("/import", response_class=HTMLResponse)
    async def import_upload(request: Request, upload: list[UploadFile] = File(...)):
        """Take an uploaded export, identify it, ingest it.

        The file is streamed to a scratch path first and identified from there, because
        `intake.identify` reads ZIP members and needs a real file rather than a stream.
        Nothing is copied into drops/ until it has been recognised as something.
        """
        from ..core import intake

        con = connect()
        blobs = BlobStore(blob_dir)
        drops = ingest.drops_dir(data_dir)
        staging = drops / intake.STAGING
        staging.mkdir(parents=True, exist_ok=True)

        taken, kinds = [], set()
        try:
            for item in upload:
                if not item.filename:
                    continue
                temp = staging / Path(item.filename).name
                size = 0
                try:
                    with temp.open("wb") as fh:
                        while chunk := await item.read(1 << 20):
                            size += len(chunk)
                            if size > MAX_UPLOAD_BYTES:
                                raise ValueError("too large")
                            fh.write(chunk)
                except ValueError:
                    temp.unlink(missing_ok=True)
                    taken.append(intake.Taken(
                        source=Path(item.filename), action="failed",
                        detail=f"over the {MAX_UPLOAD_BYTES // (1 << 20)} MB limit"))
                    continue

                result = intake.take(temp, con, drops)
                result.source = Path(item.filename)   # report the name they uploaded
                taken.append(result)
                if result.ok and result.kind:
                    kinds.add(result.kind)
                temp.unlink(missing_ok=True)
            con.commit()
        finally:
            for leftover in staging.glob("*"):
                leftover.unlink(missing_ok=True)

        results = []
        for kind in sorted(kinds):
            for adapter in ingest.build_adapters(blobs, kind, drops=drops,
                                                 browser=idb.is_enabled(con)):
                res = ingest.run(adapter, con, blobs)
                results.append({"label": adapter.label, "r": res})

        return import_page(request, con, taken=taken, results=results)

    return app


# ------------------------------------------------------------------ utilities

def _as_ms(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.strptime(value, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def _terms(query: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(query or "") if len(t) > 2]


def _condense(text: str) -> str:
    return " ".join((text or "").split())


def _highlight(text: str, terms: list[str]) -> str:
    """Escape first, then mark query terms. Never trust archive text as HTML."""
    import html
    safe = html.escape(text or "")
    if not terms:
        return safe
    pattern = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.sub(f"({pattern})", r"<mark>\1</mark>", safe, flags=re.IGNORECASE)


def _highlight_html(rendered: str, terms: list[str]) -> str:
    """Mark query terms in already-rendered HTML, skipping tag markup itself."""
    if not terms:
        return rendered
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)),
        re.IGNORECASE)
    pieces = re.split(r"(<[^>]+>)", rendered)
    for i in range(0, len(pieces), 2):  # odd indices are the tags themselves
        pieces[i] = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", pieces[i])
    return "".join(pieces)


def _render_markdown(text: str, terms: list[str]) -> str:
    """A message's text part, as markdown: headings, tables, bold, lists render
    instead of showing up as literal `##`/`**`/`|` — most sessions carry plans,
    specs and notes written as markdown, not prose. The render itself is shared with
    the export path (`export.render.render_markdown`); only the query-term highlight
    on top of it is web-view-only."""
    return _highlight_html(_render_markdown_raw(text), terms)


BIG_LINE_THRESHOLD = 20          # more source lines than this: clip and scroll
RENDER_CAP_CHARS = 200_000       # a safety net against a pathological single blob
