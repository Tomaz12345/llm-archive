"""Split embeddable parts into retrieval chunks.

Chunk on message boundaries first, splitting only what is too long. A short exchange is
one chunk; a 4,000-word design discussion becomes several with a little overlap so an
idea straddling a boundary is still findable.

Each chunk gets a small context header — `[claude_code · telemetry_analysis · 2026-02-14]`
— so the embedding carries date and project signal, not just prose. The header is
stripped before anything is shown to a human.

Only `embed_eligible` parts arrive here. Tool *results* never do: they are 62x the
conversation by volume and are copies of files already on disk (PLAN.md §1.1). They stay
searchable through FTS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

# ~380 tokens at roughly 4 chars/token, with a little headroom for the header.
MAX_CHARS = 1500
OVERLAP_CHARS = 220          # ~15%
MIN_CHARS = 40               # below this a chunk carries no retrievable signal

HEADER_RE = re.compile(r"^\[[^\]\n]{0,120}\]\n")


@dataclass(slots=True)
class Chunk:
    part_id: int
    message_id: int
    session_id: int
    seq: int
    text: str          # includes the context header


def context_header(source: str, workspace: str | None, when_ms: int | None,
                   title: str | None = None) -> str:
    bits = [source]
    if workspace:
        bits.append(workspace)
    if when_ms:
        bits.append(datetime.fromtimestamp(when_ms / 1000, timezone.utc)
                    .strftime("%Y-%m-%d"))
    if title:
        bits.append(title[:60])
    return "[" + " · ".join(bits) + "]"


def strip_header(text: str) -> str:
    """Remove the context header before display."""
    return HEADER_RE.sub("", text, count=1)


def _split(text: str) -> list[str]:
    """Paragraph-aware split, falling back to hard slicing for solid blocks."""
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return [text] if len(text) >= MIN_CHARS else []

    pieces: list[str] = []
    buf = ""
    for para in re.split(r"\n{2,}", text):
        if len(para) > MAX_CHARS:
            if buf:
                pieces.append(buf)
                buf = ""
            # a single huge paragraph (log dump, minified blob): slice with overlap
            start = 0
            while start < len(para):
                pieces.append(para[start:start + MAX_CHARS])
                start += MAX_CHARS - OVERLAP_CHARS
            continue
        if len(buf) + len(para) + 2 > MAX_CHARS and buf:
            pieces.append(buf)
            # carry a tail of the previous chunk so ideas spanning the seam survive
            tail = buf[-OVERLAP_CHARS:]
            buf = f"{tail}\n\n{para}" if tail.strip() else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        pieces.append(buf)
    return [p.strip() for p in pieces if len(p.strip()) >= MIN_CHARS]


def chunk_part(text: str, header: str, part_id: int, message_id: int,
               session_id: int, start_seq: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    for i, piece in enumerate(_split(text)):
        out.append(Chunk(
            part_id=part_id, message_id=message_id, session_id=session_id,
            seq=start_seq + i, text=f"{header}\n{piece}"))
    return out
