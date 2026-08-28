"""Render a `RenderedSession` (see `export/render.py`) as a self-contained HTML page.

No external requests: CSS is inlined from the web UI's own stylesheet (already
system-fonts-only, per its header comment), images are inlined as base64 data URIs
(falling back to a sibling `assets/` file, or a "not included" note, above
`IMAGE_INLINE_MAX_BYTES`). Opens by double-click, works offline, and needs nothing the
archive's own web app doesn't already serve locally.

`standalone=False` returns just the `<style>` block and the wrapping `<div class="export">`
fragment, byte-identical to what's embedded in the standalone document — deliberately, so
handing this file to something that publishes HTML fragments (a Claude Code session using
its Artifact tool, say) is a substring slice, not a rewrite.
"""

from __future__ import annotations

import base64
from html import escape as esc
from pathlib import Path

from ..core.blobs import blob_path, sniff_file
from .render import IMAGE_EXT, RenderedPart, RenderedSession, render_markdown

IMAGE_INLINE_MAX_BYTES = 10 * 1024 * 1024

_CSS_PATH = Path(__file__).resolve().parents[1] / "web" / "static" / "app.css"


def _load_css() -> str:
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def to_html(session: RenderedSession, blob_dir: Path, *,
            assets_dir: Path | None = None, standalone: bool = True) -> str:
    style = f"<style>\n{_load_css()}\n</style>"
    body = _fragment(session, blob_dir, assets_dir)
    if not standalone:
        return _scrub(style + body)
    return _scrub(
        "<!doctype html>\n"
        f'<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(session.title)} — {esc(session.source_label)}</title>\n"
        f"{style}\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _scrub(text: str) -> str:
    """A raw U+FFFD can end up in tool output that read a binary file as text
    (`errors="replace"` upstream, at ingest or in `resolve_full_text`) — valid UTF-8
    once encoded, but some HTML consumers reject the literal replacement character and
    want it spelled out as an entity instead."""
    return text.replace("�", "&#xFFFD;")


def _fragment(session: RenderedSession, blob_dir: Path, assets_dir: Path | None) -> str:
    return (
        '<div class="export">\n<div class="reader">\n'
        f"{_head(session)}\n"
        f'<div class="thread">\n{_thread(session, blob_dir, assets_dir)}\n</div>\n'
        f"{_footer(session)}\n"
        "</div>\n</div>\n"
    )


def _head(s: RenderedSession) -> str:
    chips = [f'<span class="chip src-{esc(s.source_kind)}">{esc(s.source_label)}</span>']
    if s.participant:
        chips.append(f'<span class="chip">{esc(s.participant)}</span>')
    chips.append(f"<span>{esc(s.started_at)}</span>")
    if s.workspace:
        chips.append(f"<span>· {esc(s.workspace)}</span>")
    if s.host:
        chips.append(f"<span>· {esc(s.host)}</span>")
    if s.model_primary:
        chips.append(f"<span>· {esc(s.model_primary)}</span>")
        if s.model_type:
            chips.append(f'<span class="tag type-{esc(s.model_type)}">{esc(s.model_type)}</span>')

    if s.excerpt:
        stats = ['<span class="badge">excerpt</span>',
                 f"<span>{len(s.messages)} of {s.dag['total']} messages</span>"]
    else:
        stats = [f"<span>{s.dag['total']} messages</span>"]
    if not s.excerpt and s.dag["abandoned"]:
        stats.append(f'<span title="Turns left behind by a rewind or prompt edit">'
                      f'{s.dag["abandoned"]} abandoned</span>')
    if not s.excerpt and s.dag["sidechain"]:
        stats.append(f"<span>{s.dag['sidechain']} subagent</span>")
    if not s.excerpt and s.tok_out:
        stats.append(f"<span>{s.tok_out:,} out</span>")
    if not s.excerpt and s.tok_cache_read:
        stats.append(f"<span>{s.tok_cache_read:,} cache-read</span>")

    tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in s.tags)

    return (
        '<div class="rhead">\n'
        f"<h1>{esc(s.title)}</h1>\n"
        f'<div class="hit-meta">{"".join(chips)}</div>\n'
        f'<div class="rstats">{"".join(stats)}</div>\n'
        '<div class="rtools">'
        f'<span class="path" title="the file this was parsed from">{esc(s.raw_path)}</span>'
        "</div>\n"
        + (f'<div class="tagbar">{tags}</div>\n' if tags else "")
        + "</div>"
    )


def _footer(s: RenderedSession) -> str:
    note = " · secrets redacted" if s.redacted else ""
    return f'<p class="foot">exported {esc(s.generated_at)}{note}</p>'


def _thread(session: RenderedSession, blob_dir: Path, assets_dir: Path | None) -> str:
    out = []
    for m in session.messages:
        classes = ["msg", f"role-{esc(m.role)}"]
        if not m.on_active_path:
            classes.append("dead")
        if m.is_sidechain:
            classes.append("side")
        badges = ""
        if m.is_sidechain:
            badges += '<span class="badge">subagent</span>'
        if not m.on_active_path:
            badges += '<span class="badge warn">abandoned</span>'
        parts_html = "".join(_part(p, blob_dir, assets_dir) for p in m.parts)
        out.append(
            f'<div class="{" ".join(classes)}">\n'
            f'<div class="mhead"><b>{esc(m.role)}</b>{badges}'
            f'<span class="when">{esc(m.when)}</span></div>\n'
            f"{parts_html}\n</div>"
        )
    return "\n".join(out)


def _part(p: RenderedPart, blob_dir: Path, assets_dir: Path | None) -> str:
    if p.kind in ("text", "thinking"):
        label = '<span class="plabel">thinking</span>' if p.kind == "thinking" else ""
        return (f'<div class="part {p.kind}">{label}'
                f'<div class="body md">{_body_html(p, assets_dir)}</div></div>')

    if p.kind == "image":
        return _image_part(p, blob_dir, assets_dir)

    # tool_use / tool_result / attachment / anything else
    ok_badge = '<span class="badge warn">error</span>' if p.tool_ok == 0 else ""
    return (
        '<details class="part tool" open>\n<summary>'
        f'<span class="tname">{esc(p.tool_name or p.kind)}</span>'
        f'<span class="tkind">{esc(p.kind.replace("_", " "))}</span>{ok_badge}'
        + (f'<span class="tduration">{esc(p.duration)}</span>' if p.duration else "")
        + f'<span class="tsize">{p.bytes / 1024:.1f} KB</span>'
        + "</summary>\n"
        f'<div class="body{" md" if p.is_markdown else ""}">{_body_html(p, assets_dir)}</div>\n'
        "</details>"
    )


def _body_html(p: RenderedPart, assets_dir: Path | None) -> str:
    if p.spilled_to:
        return _spill_note(p, assets_dir)
    if p.is_markdown:
        return render_markdown(p.text)
    return f"<pre>{esc(p.text)}</pre>"


def _spill_note(p: RenderedPart, assets_dir: Path | None) -> str:
    size_mb = len(p.text.encode("utf-8", "replace")) / (1024 * 1024)
    if assets_dir is not None:
        (assets_dir / p.spilled_to).write_text(p.text, encoding="utf-8")
        return (f'<p class="clip-note">— {size_mb:.1f} MB, too large to inline; '
                f'full output at <a href="assets/{esc(p.spilled_to)}">assets/{esc(p.spilled_to)}</a> —</p>')
    return f'<p class="clip-note">— {size_mb:.1f} MB, not included in a single-file export —</p>'


def _image_part(p: RenderedPart, blob_dir: Path, assets_dir: Path | None) -> str:
    caption = (f'<span class="fname">{esc(p.caption)}</span>' if p.caption else "")
    if p.sha:
        path = blob_path(blob_dir, p.sha)
        if path is None:
            return (f'<div class="part image missing"><span class="plabel">image not included'
                     f'</span>{caption}<span class="hint">recorded, but the file is no longer '
                     "on disk</span></div>")
        src = _image_src(path, p.sha, assets_dir)
        if src is not None:
            size = f'<span class="tsize">{p.bytes / 1024:.1f} KB</span>'
            return (f'<figure class="part image"><img src="{src}" loading="lazy" '
                     f'alt="{esc(p.caption or "image")}">'
                     f"<figcaption>{caption}{size}</figcaption></figure>")
        return (f'<div class="part image missing"><span class="plabel">image not included'
                f'</span>{caption}<span class="hint">over {IMAGE_INLINE_MAX_BYTES // (1024*1024)} MB '
                "— re-run export with an assets folder to include it</span></div>")
    hint = ('<span class="hint">referenced by URL — run <code>llma fetch-images</code> '
            "to pull it into the archive</span>" if p.ref_url else
            '<span class="hint">the source export shipped no bytes for this one</span>')
    return f'<div class="part image missing"><span class="plabel">image not stored</span>{caption}{hint}</div>'


def _image_src(path: Path, sha: str, assets_dir: Path | None) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    mime = sniff_file(path)
    if len(data) <= IMAGE_INLINE_MAX_BYTES:
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    if assets_dir is not None:
        ext = IMAGE_EXT.get(mime, "")
        dest = assets_dir / f"{sha}{ext}"
        if not dest.exists():
            dest.write_bytes(data)
        return f"assets/{sha}{ext}"
    return None
