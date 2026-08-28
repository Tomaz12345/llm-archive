"""Render a `RenderedSession` (see `export/render.py`) as plain Markdown.

Unlike the HTML serializer, images are never inlined as data URIs here — base64 in a
`.md` file defeats the format's whole point (readable, diffable, pasteable) — they are
always collected into `MarkdownExport.assets` for the caller to write to a sibling
`assets/` folder, referenced by relative path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.blobs import blob_path, sniff_file
from .render import IMAGE_EXT, RenderedMessage, RenderedPart, RenderedSession


@dataclass(slots=True)
class MarkdownExport:
    text: str
    assets: dict[str, bytes] = field(default_factory=dict)


def to_markdown(session: RenderedSession, blob_dir: Path) -> MarkdownExport:
    assets: dict[str, bytes] = {}
    lines = [f"# {session.title}", "", _meta_line(session), f"`{session.raw_path}`"]
    if session.tags:
        lines.append("Tags: " + ", ".join(session.tags))
    lines += [_stats_line(session), "", "---", ""]

    for m in session.messages:
        lines.append(_message_header(m))
        lines.append("")
        for p in m.parts:
            lines.append(_part_md(p, blob_dir, assets))
            lines.append("")

    footer = f"_exported {session.generated_at}"
    footer += " · secrets redacted_" if session.redacted else "_"
    lines.append(footer)

    return MarkdownExport(text="\n".join(lines).rstrip() + "\n", assets=assets)


def write(export: MarkdownExport, out_path: Path, assets_dir: Path | None = None) -> None:
    """Write the markdown text and, if it collected any, the asset files beside it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(export.text, encoding="utf-8")
    if export.assets:
        target = assets_dir or (out_path.parent / f"{out_path.stem}.assets")
        target.mkdir(parents=True, exist_ok=True)
        for name, data in export.assets.items():
            (target / name).write_bytes(data)


def _meta_line(s: RenderedSession) -> str:
    bits = [s.source_label]
    if s.participant:
        bits.append(s.participant)
    bits.append(s.started_at)
    if s.workspace:
        bits.append(s.workspace)
    if s.host:
        bits.append(s.host)
    if s.model_primary:
        bits.append(f"{s.model_primary} ({s.model_type})" if s.model_type else s.model_primary)
    return " · ".join(bits)


def _stats_line(s: RenderedSession) -> str:
    if s.excerpt:
        return f"Excerpt: {len(s.messages)} of {s.dag['total']} messages"
    bits = [f"{s.dag['total']} messages"]
    if s.dag["abandoned"]:
        bits.append(f"{s.dag['abandoned']} abandoned")
    if s.dag["sidechain"]:
        bits.append(f"{s.dag['sidechain']} subagent")
    if s.tok_out:
        bits.append(f"{s.tok_out:,} out")
    if s.tok_cache_read:
        bits.append(f"{s.tok_cache_read:,} cache-read")
    return " · ".join(bits)


def _message_header(m: RenderedMessage) -> str:
    badges = []
    if m.is_sidechain:
        badges.append("subagent")
    if not m.on_active_path:
        badges.append("abandoned")
    tag = f" _{', '.join(badges)}_" if badges else ""
    return f"### {m.role}{tag} — {m.when}"


def _part_md(p: RenderedPart, blob_dir: Path, assets: dict[str, bytes]) -> str:
    if p.kind == "image":
        return _image_md(p, blob_dir, assets)

    header = ""
    if p.kind not in ("text", "thinking"):
        label = p.tool_name or p.kind
        bits = [f"**{label}**", p.kind.replace("_", " "), f"{p.bytes / 1024:.1f} KB"]
        if p.tool_ok is False:
            bits.append("error")
        if p.duration:
            bits.append(p.duration)
        header = " · ".join(bits) + "\n\n"
    elif p.kind == "thinking":
        header = "_thinking:_\n\n"

    if p.spilled_to:
        size_mb = len(p.text.encode("utf-8", "replace")) / (1024 * 1024)
        assets[p.spilled_to] = p.text.encode("utf-8", "replace")
        body = (f"_{size_mb:.1f} MB, too large to inline — full output at "
                f"[assets/{p.spilled_to}](assets/{p.spilled_to})_")
    elif p.is_markdown:
        body = p.text
    else:
        fence = "````" if "```" in p.text else "```"
        body = f"{fence}\n{p.text}\n{fence}"

    return header + body


def _image_md(p: RenderedPart, blob_dir: Path, assets: dict[str, bytes]) -> str:
    if p.sha:
        path = blob_path(blob_dir, p.sha)
        if path is not None:
            try:
                data = path.read_bytes()
            except OSError:
                data = None
            if data is not None:
                mime = sniff_file(path)
                fname = f"{p.sha}{IMAGE_EXT.get(mime, '')}"
                assets[fname] = data
                return f"![{p.caption or 'image'}](assets/{fname})"
        return "_image recorded, but the file is no longer on disk_"
    if p.ref_url:
        return f"_image not stored — referenced by {p.ref_url}; run `llma fetch-images`_"
    return "_image not stored — the source export shipped no bytes for this one_"
