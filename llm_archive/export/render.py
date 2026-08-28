"""One session, from the DB's raw rows to a source-agnostic intermediate representation.

This is the single place that decides what a session *is* once resolved: full text read
back from the blob store where the DB only kept a preview, secrets redacted, markdown vs.
plain-text told apart. Both `export/html.py` and `export/markdown.py` consume the same
`RenderedSession` this module builds — one render step, two dumb serializers — which is
what makes a Claude Code session and a ChatGPT session come out looking the same: they
were already normalized into the same `core.models.Session`/`Message`/`Part` vocabulary
long before this module ever sees them.

Most of this was lifted out of `web/app.py`'s `session_view`, which still uses these same
functions (re-imported under their old private names) so the live web view and a static
export share one code path rather than two that can quietly drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..core.blobs import IMAGE_MIMES, blob_path, sniff_file
from ..core.redact import active_rules, redact_text
from ..search.chunker import strip_header
from ..stats import model_types

# A pathological single tool_result (a multi-megabyte log dump), not normal content — a
# saved record should be complete, so this is a guard rail, not the live view's tighter
# RENDER_CAP_CHARS. Above it, text is spilled to a sibling asset file instead of being
# either inlined whole (bloats the HTML/Markdown past usefulness) or truncated (silently
# incomplete, which a saved record must never be).
EXPORT_INLINE_TEXT_LIMIT = 5_000_000

# Blobs are stored content-addressed with no extension (`core/blobs.py`); both serializers
# need a real filename when a part becomes a sibling asset file.
IMAGE_EXT: dict[str, str] = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/bmp": ".bmp", "image/webp": ".webp", "image/avif": ".avif",
    "image/heic": ".heic", "image/tiff": ".tiff",
}


@dataclass(slots=True)
class RenderedPart:
    kind: str
    tool_name: str | None
    tool_ok: bool | None
    bytes: int
    duration: str
    sha: str | None            # blob sha256; image bytes resolved lazily by serializers
    caption: str
    ref_url: str                # an unfetched image's source URL, "" otherwise
    is_markdown: bool
    text: str                   # fully resolved (blob-read where needed) + redacted
    spilled_to: str | None = None   # sibling asset filename if EXPORT_INLINE_TEXT_LIMIT hit


@dataclass(slots=True)
class RenderedMessage:
    role: str
    when: str
    on_active_path: bool
    is_sidechain: bool
    model: str | None
    parts: list[RenderedPart] = field(default_factory=list)


@dataclass(slots=True)
class RenderedSession:
    id: int
    title: str
    source_kind: str
    source_label: str
    workspace: str
    host: str
    participant: str | None
    model_primary: str | None
    model_type: str | None
    started_at: str
    tags: list[str]
    dag: dict
    tok_out: int | None
    tok_cache_read: int | None
    raw_path: str
    messages: list[RenderedMessage]
    redacted: bool
    generated_at: str
    # True when this holds a single message pulled out of a longer session. The
    # serializers say so on the page: a one-turn file that presents itself as the whole
    # session is a quote taken out of context, which is exactly what an archive is for
    # avoiding. `dag` still describes the session it came from, so "1 of N" is honest.
    excerpt: bool = False


def build_session(con, blob_dir: Path, session_id: int, *,
                   tools: bool = True, abandoned: bool = True,
                   redact: bool = True, wide: bool = False,
                   message_id: int | None = None) -> RenderedSession | None:
    """Everything needed to render one session, resolved and ready to serialize.

    `tools`/`abandoned` mirror `web/app.py`'s `session_view` query params, but the
    defaults are flipped: the live view hides abandoned branches and can show tool calls
    because a "show" toggle is one click away; a static export has no toggle, so it
    defaults to complete.

    `message_id` narrows the result to that one message, keeping the session's header,
    tags and provenance around it — an excerpt with its source attached, not a bare
    fragment. `abandoned` is ignored in that mode: you asked for that message by id, so
    which branch it sits on is not a reason to hand back nothing.
    """
    meta = con.execute("""
        SELECT s.*, src.kind AS source_kind, src.label AS source_label,
               COALESCE(w.label,'') AS workspace,
               json_extract(s.meta,'$.participant_label') AS participant
        FROM session s JOIN source src ON src.id = s.source_id
        LEFT JOIN workspace w ON w.id = s.workspace_id
        WHERE s.id = ?""", (session_id,)).fetchone()
    if meta is None:
        return None

    if message_id is not None:
        where, params = "AND m.id = ?", (session_id, message_id)
    else:
        where = "" if abandoned else "AND m.on_active_path = 1"
        params = (session_id,)
    rows = con.execute(f"""
        SELECT m.id, m.role, m.seq, m.on_active_path, m.is_sidechain,
               m.created_at, m.model
        FROM message m WHERE m.session_id = ? {where}
        ORDER BY m.seq, m.id""", params).fetchall()
    if message_id is not None and not rows:
        return None      # no such message, or not this session's — a 404 either way

    rules = active_rules(wide) if redact else None

    messages: list[RenderedMessage] = []
    for row in rows:
        parts = con.execute("""
            SELECT p.id, p.kind, p.text, p.tool_name, p.tool_ok, p.bytes, p.blob_id,
                   p.duration_ms, b.sha256
            FROM part p LEFT JOIN blob b ON b.id = p.blob_id
            WHERE p.message_id = ? ORDER BY p.seq""", (row["id"],)).fetchall()
        shown = [p for p in parts
                 if tools or p["kind"] in ("text", "thinking", "image")]
        if not shown:
            continue
        messages.append(RenderedMessage(
            role=row["role"], when=fmt_when(row["created_at"], time=True),
            on_active_path=bool(row["on_active_path"]),
            is_sidechain=bool(row["is_sidechain"]), model=row["model"],
            parts=[_render_part(p, blob_dir, rules) for p in shown]))

    return RenderedSession(
        id=session_id, title=meta["title"] or "(untitled)",
        source_kind=meta["source_kind"], source_label=meta["source_label"],
        workspace=meta["workspace"] or "", host=meta["host"] or "",
        participant=meta["participant"], model_primary=meta["model_primary"],
        model_type=model_types.classify(meta["model_primary"]),
        started_at=fmt_when(meta["started_at"], time=True),
        tags=tags_for(con, session_id), dag=dag_summary(con, session_id),
        tok_out=meta["tok_out"], tok_cache_read=meta["tok_cache_read"],
        raw_path=meta["raw_path"], messages=messages, redacted=redact,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        excerpt=message_id is not None)


def _render_part(p, blob_dir: Path, rules) -> RenderedPart:
    raw = strip_header(resolve_full_text(p["kind"], p["text"], p["blob_id"],
                                          p["sha256"], blob_dir))
    if rules is not None:
        raw, _ = redact_text(raw, rules)

    spilled_to = None
    if len(raw.encode("utf-8", errors="replace")) > EXPORT_INLINE_TEXT_LIMIT:
        import hashlib
        key = p["sha256"] or hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()
        spilled_to = f"{key}.txt"

    is_md = p["kind"] in ("text", "thinking") or looks_like_markdown(raw)
    return RenderedPart(
        kind=p["kind"], tool_name=p["tool_name"], tool_ok=p["tool_ok"],
        bytes=p["bytes"], duration=fmt_ms(p["duration_ms"]), sha=p["sha256"],
        caption=caption_of(p["text"]), ref_url=ref_url_of(p["text"]),
        is_markdown=is_md, text=raw, spilled_to=spilled_to)


# --------------------------------------------------------------- blob resolution

def resolve_full_text(kind: str, text: str | None, blob_id: int | None,
                       sha256: str | None, blob_dir: Path) -> str:
    """The DB's `part.text` is only ever a preview once a big tool result has been
    offloaded to the blob store (`core.models.INLINE_LIMIT`). Read the real thing back
    from disk instead so a saved export isn't quietly missing the bulk of what a turn
    actually produced.
    """
    if kind == "tool_result" and blob_id is not None and sha256:
        path = blob_path(blob_dir, sha256)
        if path is not None and sniff_file(path) not in IMAGE_MIMES:
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return text or ""


# ----------------------------------------------------------------- markdown sniff/render

_MD_TABLE_SEP_RE = re.compile(r"(?m)^ {0,3}\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MD_HEADING_RE = re.compile(r"(?m)^#{2,6} +\S")
_MD_BOLD_RE = re.compile(r"\*\*[^\s*][^*]*\*\*")


def looks_like_markdown(text: str) -> bool:
    """A cheap sniff for "this tool output is itself a markdown document" (a file read
    of a plan/README) as opposed to a code dump or JSON. A table separator row is
    markdown-specific enough alone; headings need a second signal (bold text) since a
    lone `##` divider shows up in code too."""
    if _MD_TABLE_SEP_RE.search(text):
        return True
    return len(_MD_HEADING_RE.findall(text)) >= 2 and _MD_BOLD_RE.search(text) is not None


def markdown_renderer():
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False, "linkify": False})
    md.enable(["table"])
    # See web/app.py's identical setting: a plain sentence followed by a "---" divider
    # (a turn separator, a frontmatter fence) must not become a setext heading.
    md.disable(["lheading"])
    return md


_MD = None


def render_markdown(text: str) -> str:
    global _MD
    if _MD is None:
        _MD = markdown_renderer()
    return _MD.render(text or "")


# ----------------------------------------------------------------------- small helpers

def fmt_when(ms: int | None, time: bool = False) -> str:
    if not ms:
        return "—"
    stamp = datetime.fromtimestamp(ms / 1000, timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M" if time else "%Y-%m-%d")


def fmt_ms(ms: int | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def caption_of(text: str | None) -> str:
    """The filename an image part carries, if it carries one — see web/app.py's
    identical helper for why this is "first whitespace token, unless it's a URL"."""
    head = (text or "").strip().split()
    if not head or head[0].startswith(("http://", "https://")):
        return ""
    return head[0][:120]


def ref_url_of(text: str | None) -> str:
    """The URL an un-fetched image part points at. Never triggers a fetch itself —
    `llma fetch-images` is the only place that touches the network."""
    from ..core.fetch_images import url_in

    return url_in(text) or ""


def tags_for(con, session_id: int) -> list[str]:
    return [r["name"] for r in con.execute(
        "SELECT t.name FROM tag t JOIN session_tag st ON st.tag_id = t.id "
        "WHERE st.session_id = ? ORDER BY t.name", (session_id,))]


def dag_summary(con, session_id: int) -> dict:
    row = con.execute("""
        SELECT COUNT(*) total, SUM(on_active_path) active, SUM(is_sidechain) side
        FROM message WHERE session_id = ?""", (session_id,)).fetchone()
    return {"total": row["total"] or 0,
            "abandoned": (row["total"] or 0) - (row["active"] or 0),
            "sidechain": row["side"] or 0}
