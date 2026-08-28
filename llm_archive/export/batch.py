"""Filter-based session selection and multi-session export.

`select_session_ids` mirrors `/browse`'s hand-rolled filter clauses (`web/app.py`)
exactly, so a CLI `llma export --workspace foo` and the web `/browse?workspace=foo` filter
bar mean the same thing. `export_batch` writes one subfolder per session (`session.html`
and/or `session.md` + `assets/`) plus a top-level `manifest.json`, either straight to a
directory or zipped — `out` can be a real path (CLI) or an in-memory `io.BytesIO` (the web
route streams a zip without touching disk), since `zipfile.ZipFile` accepts either.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import html as html_mod
from . import markdown as markdown_mod
from . import render

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(title: str | None) -> str:
    s = _SLUG_RE.sub("-", (title or "untitled").lower()).strip("-")
    return (s or "untitled")[:60]


def select_session_ids(con, *, ids: list[int] | None = None,
                        workspace: str | None = None, source: str | None = None,
                        participant: str | None = None, host: str | None = None,
                        tag: str | None = None, model_type: str | None = None) -> list[int]:
    if ids:
        return list(ids)

    clauses: list[str] = []
    params: list = []
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
    if model_type:
        from ..stats.model_types import model_type_index
        matching = [m for m, _ in model_type_index(con).get(model_type, [])]
        if matching:
            clauses.append(f"s.model_primary IN ({','.join('?' * len(matching))})")
            params.extend(matching)
        else:
            clauses.append("0")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = con.execute(f"""
        SELECT s.id FROM session s
        JOIN source src ON src.id = s.source_id
        LEFT JOIN workspace w ON w.id = s.workspace_id
        {where}
        ORDER BY s.started_at DESC""", tuple(params)).fetchall()
    return [r["id"] for r in rows]


@dataclass(slots=True)
class BatchResult:
    generated_at: str
    count: int
    sessions: list[dict]


def export_batch(con, blob_dir: Path, session_ids: list[int], out, *, fmt: set[str],
                  redact: bool = True, wide: bool = False, tools: bool = True,
                  abandoned: bool = True, zip_output: bool = False) -> BatchResult:
    if zip_output:
        with tempfile.TemporaryDirectory() as tmp:
            result = _export_all(con, blob_dir, session_ids, Path(tmp), fmt=fmt,
                                  redact=redact, wide=wide, tools=tools, abandoned=abandoned)
            _zip_dir(Path(tmp), out)
        return result
    return _export_all(con, blob_dir, session_ids, Path(out), fmt=fmt, redact=redact,
                        wide=wide, tools=tools, abandoned=abandoned)


def _export_all(con, blob_dir: Path, session_ids: list[int], root: Path, *,
                 fmt: set[str], redact: bool, wide: bool, tools: bool,
                 abandoned: bool) -> BatchResult:
    root.mkdir(parents=True, exist_ok=True)
    sessions_meta = []
    for sid in session_ids:
        session = render.build_session(con, blob_dir, sid, tools=tools,
                                        abandoned=abandoned, redact=redact, wide=wide)
        if session is None:
            continue
        folder_name = f"{sid:06d}-{slug(session.title)}"
        folder = root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        if "html" in fmt:
            assets_dir = folder / "assets"
            assets_dir.mkdir(exist_ok=True)
            (folder / "session.html").write_text(
                html_mod.to_html(session, blob_dir, assets_dir=assets_dir, standalone=True),
                encoding="utf-8")
            if not any(assets_dir.iterdir()):
                assets_dir.rmdir()

        if "md" in fmt:
            markdown_mod.write(markdown_mod.to_markdown(session, blob_dir),
                                folder / "session.md", folder / "assets")

        sessions_meta.append({
            "id": sid, "title": session.title, "source": session.source_kind,
            "started_at": session.started_at, "folder": folder_name,
        })

    result = BatchResult(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        count=len(sessions_meta), sessions=sessions_meta)
    (root / "manifest.json").write_text(
        json.dumps({"generated_at": result.generated_at, "count": result.count,
                    "sessions": result.sessions}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return result


def _zip_dir(src_root: Path, out) -> None:
    if isinstance(out, (str, Path)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_root).as_posix())
