"""Tests for the session export feature (`llm_archive/export/`).

Redaction gets the most scrutiny here: export is the one place content is meant to leave
the machine, so `build_session`'s default (`redact=True`, independent of the archive's own
`redact_enabled` setting) is the load-bearing safety property of the whole feature.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from llm_archive.cli import app as cli_app
from llm_archive.core import db
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import Message, Part, Session
from llm_archive.export import batch, render
from llm_archive.export import html as export_html
from llm_archive.export import markdown as export_markdown
from llm_archive.web.app import create_app

TINY_PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 100   # real magic bytes, fake payload


@pytest.fixture
def archive(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    con = db.connect(data / "archive.db")
    store = BlobStore(data / "blobs")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    return con, store, src


def _add(con, source_id, native, title, *, workspace="proj", host="box",
         kind="claude_code", model_primary=None, meta=None, tags=None):
    m = Message(native_id=f"{native}-0", role="user", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="text", seq=0, text=f"hello from {native}",
                         embed_eligible=True))
    sid, _ = db.upsert_session(con, source_id, Session(
        source_kind=kind, native_id=native, title=title,
        workspace_key=workspace, workspace_label=workspace, host=host,
        started_at=1771200000000, raw_path=f"/raw/{native}.jsonl", raw_hash=native,
        model_primary=model_primary, messages=[m], meta=meta or {}))
    if tags:
        for t in tags:
            con.execute("INSERT OR IGNORE INTO tag(name) VALUES (?)", (t,))
            tid = con.execute("SELECT id FROM tag WHERE name=?", (t,)).fetchone()["id"]
            con.execute("INSERT OR IGNORE INTO session_tag(session_id,tag_id) VALUES (?,?)",
                        (sid, tid))
    return sid


# ------------------------------------------------------------------ build_session

def test_build_session_resolves_blob_backed_tool_result(archive):
    con, store, src = archive
    full = "line\n" * 5000
    sha, size, path = store.put_text(full)
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="tool_result", seq=0, text="preview only",
                         blob_sha=sha, blob_path=path, bytes=size))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Blob test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()

    session = render.build_session(con, store.root, sid)
    assert session.messages[0].parts[0].text == full


def test_redaction_fires_by_default_and_is_skippable(archive):
    con, store, src = archive
    secret = "sk-ant-api03-FAKESECRETVALUE1234567890ABCDEFGHIJK"
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="text", seq=0, text=f"here is my key {secret}",
                         embed_eligible=True))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Secret test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()

    redacted = render.build_session(con, store.root, sid)
    assert secret not in redacted.messages[0].parts[0].text
    assert "[redacted:" in redacted.messages[0].parts[0].text

    raw = render.build_session(con, store.root, sid, redact=False)
    assert secret in raw.messages[0].parts[0].text


def test_abandoned_and_tools_toggle(archive):
    con, store, src = archive
    m1 = Message(native_id="m1", role="user", created_at=1771200000000, seq=0)
    m1.parts.append(Part(kind="text", seq=0, text="hello", embed_eligible=True))
    m2 = Message(native_id="m2", role="assistant", created_at=1771200001000, seq=1,
                on_active_path=False)
    m2.parts.append(Part(kind="text", seq=0, text="dead end", embed_eligible=True))
    m3 = Message(native_id="m3", role="assistant", created_at=1771200002000, seq=2)
    m3.parts.append(Part(kind="tool_use", seq=0, text="ran a tool", tool_name="Bash"))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Toggle test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m1, m2, m3]))
    con.commit()

    full = render.build_session(con, store.root, sid)
    assert len(full.messages) == 3

    no_abandoned = render.build_session(con, store.root, sid, abandoned=False)
    assert len(no_abandoned.messages) == 2

    no_tools = render.build_session(con, store.root, sid, tools=False)
    kinds = {p.kind for m in no_tools.messages for p in m.parts}
    assert "tool_use" not in kinds


def test_unknown_session_returns_none(archive):
    con, store, _ = archive
    assert render.build_session(con, store.root, 999999) is None


# ------------------------------------------------------------------------ to_html

def test_html_fragment_matches_standalone_embed(archive):
    con, store, src = archive
    sid = _add(con, src, "s1", "Frag test")
    con.commit()
    session = render.build_session(con, store.root, sid)

    standalone = export_html.to_html(session, store.root, standalone=True)
    fragment = export_html.to_html(session, store.root, standalone=False)

    assert "<!doctype" in standalone.lower()
    assert "<!doctype" not in fragment.lower()
    assert fragment.count('<div class="export">') == 1

    style, _, rest = fragment.partition('<div class="export">')
    body = '<div class="export">' + rest
    assert style in standalone
    assert body in standalone


def test_html_export_has_no_external_urls(archive):
    con, store, src = archive
    sid = _add(con, src, "s1", "No network test")
    con.commit()
    session = render.build_session(con, store.root, sid)
    out = export_html.to_html(session, store.root)
    for m in re.finditer(r'(?:src|href)="([^"]*)"', out):
        assert not m.group(1).startswith(("http://", "https://")), m.group(0)


def test_image_inlines_as_data_uri(archive):
    con, store, src = archive
    sha, size, path = store.put_bytes(TINY_PNG)
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="image", seq=0, text="pic.png",
                         blob_sha=sha, blob_path=path, bytes=size))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Img test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()

    session = render.build_session(con, store.root, sid)
    out = export_html.to_html(session, store.root)
    assert "data:image/png;base64," in out


def test_oversized_image_degrades_without_assets_dir_and_copies_with_one(
        archive, monkeypatch, tmp_path):
    con, store, src = archive
    sha, size, path = store.put_bytes(TINY_PNG)
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="image", seq=0, text="pic.png",
                         blob_sha=sha, blob_path=path, bytes=size))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Big img test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()

    monkeypatch.setattr(export_html, "IMAGE_INLINE_MAX_BYTES", 4)
    session = render.build_session(con, store.root, sid)

    single_file = export_html.to_html(session, store.root)
    assert "data:image" not in single_file
    assert "not included" in single_file

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    with_assets = export_html.to_html(session, store.root, assets_dir=assets_dir)
    assert f"assets/{sha}.png" in with_assets
    assert (assets_dir / f"{sha}.png").exists()


# -------------------------------------------------------------------- to_markdown

def test_markdown_export_never_inlines_images(archive):
    con, store, src = archive
    sha, size, path = store.put_bytes(TINY_PNG)
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="image", seq=0, text="pic.png",
                         blob_sha=sha, blob_path=path, bytes=size))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="Img md test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()

    session = render.build_session(con, store.root, sid)
    export = export_markdown.to_markdown(session, store.root)
    assert "data:image" not in export.text
    assert f"assets/{sha}.png" in export.text
    assert f"{sha}.png" in export.assets


# --------------------------------------------------------------- select_session_ids

def test_select_session_ids_filters(archive):
    con, _, src = archive
    panel = db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel")
    s1 = _add(con, src, "s1", "Offside work", workspace="proj-a", host="box1")
    s2 = _add(con, panel, "s2", "Copilot refactor", workspace="proj-b", host="box2",
              kind="vscode_chat", meta={"participant": "copilot"})
    con.commit()

    assert batch.select_session_ids(con, workspace="proj-a") == [s1]
    assert batch.select_session_ids(con, source="vscode_chat") == [s2]
    assert batch.select_session_ids(con, participant="copilot") == [s2]
    assert batch.select_session_ids(con, host="box2") == [s2]
    assert set(batch.select_session_ids(con)) == {s1, s2}
    assert batch.select_session_ids(con, ids=[s1]) == [s1]


def test_select_session_ids_tag_filter(archive):
    con, _, src = archive
    s1 = _add(con, src, "s1", "Tagged", tags=["important"])
    _add(con, src, "s2", "Untagged")
    con.commit()
    assert batch.select_session_ids(con, tag="important") == [s1]


def test_select_session_ids_model_type_filter(archive):
    con, _, src = archive
    s1 = _add(con, src, "s1", "Legacy model", model_primary="claude-3.5-sonnet")
    _add(con, src, "s2", "General model", model_primary="claude-4.5-sonnet")
    con.commit()
    assert batch.select_session_ids(con, model_type="legacy") == [s1]


# ------------------------------------------------------------------------------ CLI

def test_cli_single_session_export(archive, tmp_path):
    con, _, src = archive
    sid = _add(con, src, "s1", "CLI export test")
    con.commit()
    con.close()

    result = CliRunner().invoke(cli_app, [
        "export", str(sid), "--data-dir", str(tmp_path / "data"),
        "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output

    folder = tmp_path / "out" / f"{sid:06d}-cli-export-test"
    assert (folder / "session.html").exists()
    assert (folder / "session.md").exists()


def test_cli_unknown_session_fails(archive, tmp_path):
    con, _, _ = archive
    con.close()
    result = CliRunner().invoke(cli_app, [
        "export", "999999", "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 1


def test_cli_batch_zip_export_writes_manifest(archive, tmp_path):
    con, _, src = archive
    _add(con, src, "s1", "Batch one", workspace="proj-x")
    _add(con, src, "s2", "Batch two", workspace="proj-x")
    con.commit()
    con.close()

    zip_path = tmp_path / "out.zip"
    result = CliRunner().invoke(cli_app, [
        "export", "--data-dir", str(tmp_path / "data"), "--workspace", "proj-x",
        "--zip", "--out", str(zip_path)])
    assert result.exit_code == 0, result.output

    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["count"] == 2


def test_cli_no_redact_keeps_the_secret(archive, tmp_path):
    con, _, src = archive
    secret = "sk-ant-api03-FAKESECRETVALUE1234567890ABCDEFGHIJK"
    m = Message(native_id="m1", role="assistant", created_at=1771200000000, seq=0)
    m.parts.append(Part(kind="text", seq=0, text=secret, embed_eligible=True))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", title="No redact test",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s1.jsonl", raw_hash="h1", messages=[m]))
    con.commit()
    con.close()

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(cli_app, [
        "export", str(sid), "--format", "md", "--no-redact",
        "--data-dir", str(tmp_path / "data"), "--out", str(out_dir)])
    assert result.exit_code == 0, result.output

    md = (out_dir / f"{sid:06d}-no-redact-test" / "session.md").read_text(encoding="utf-8")
    assert secret in md


# ------------------------------------------------------------------------ web routes

@pytest.fixture
def web_client(archive, tmp_path):
    con, _, src = archive
    _add(con, src, "s1", "Web export test")
    con.commit()
    con.close()
    return TestClient(create_app(tmp_path / "data"))


def test_session_export_html_route(web_client):
    r = web_client.get("/session/1/export.html")
    assert r.status_code == 200
    assert '<div class="export">' in r.text


def test_session_export_md_route(web_client):
    r = web_client.get("/session/1/export.md")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")


def test_message_export_routes_return_one_message(archive, tmp_path):
    con, store, src = archive
    m0 = Message(native_id="m0", role="user", created_at=1771200000000, seq=0)
    m0.parts.append(Part(kind="text", seq=0, text="first question"))
    m1 = Message(native_id="m1", role="assistant", created_at=1771200001000, seq=1)
    m1.parts.append(Part(kind="text", seq=0, text="second answer"))
    sid, _ = db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s-multi", title="Two turns",
        workspace_key="proj", workspace_label="proj", started_at=1771200000000,
        raw_path="/raw/s-multi.jsonl", raw_hash="hm", messages=[m0, m1]))
    con.commit()
    mid = con.execute("SELECT id FROM message WHERE session_id=? ORDER BY seq",
                      (sid,)).fetchall()[1]["id"]
    con.close()
    client = TestClient(create_app(tmp_path / "data"))

    r = client.get(f"/session/{sid}/message/{mid}/export.md")
    assert r.status_code == 200
    assert "second answer" in r.text and "first question" not in r.text
    # the session it came from travels with the excerpt, and says it is one
    assert "Two turns" in r.text
    assert "Excerpt: 1 of 2 messages" in r.text
    assert f'filename="session-{sid}-message-{mid}.md"' in r.headers["content-disposition"]

    r = client.get(f"/session/{sid}/message/{mid}/export.html")
    assert r.status_code == 200
    assert "second answer" in r.text and "first question" not in r.text
    assert "attachment" in r.headers["content-disposition"]


def test_message_export_rejects_id_from_another_session(archive, tmp_path):
    con, _, src = archive
    a = _add(con, src, "s-a", "Session A")
    b = _add(con, src, "s-b", "Session B")
    con.commit()
    mid_b = con.execute("SELECT id FROM message WHERE session_id=?",
                        (b,)).fetchone()["id"]
    con.close()
    client = TestClient(create_app(tmp_path / "data"))

    assert client.get(f"/session/{a}/message/{mid_b}/export.md").status_code == 404
    assert client.get(f"/session/{a}/message/999999/export.html").status_code == 404


def test_message_export_reaches_an_abandoned_turn(archive, tmp_path):
    """The session view hides abandoned branches by default, but a link asked for that
    message by id — the branch it sits on must not turn the download into a 404."""
    con, _, src = archive
    sid = _add(con, src, "s-dead", "Rewound")
    con.execute("UPDATE message SET on_active_path = 0 WHERE session_id = ?", (sid,))
    con.commit()
    mid = con.execute("SELECT id FROM message WHERE session_id=?", (sid,)).fetchone()["id"]
    con.close()
    client = TestClient(create_app(tmp_path / "data"))

    r = client.get(f"/session/{sid}/message/{mid}/export.md")
    assert r.status_code == 200
    assert "hello from s-dead" in r.text


def test_session_view_offers_a_per_message_download(web_client):
    r = web_client.get("/session/1")
    assert r.status_code == 200
    assert "/message/1/export.md" in r.text
    assert "/message/1/export.html" in r.text


def test_session_export_unknown_is_404(web_client):
    assert web_client.get("/session/999999/export.html").status_code == 404
    assert web_client.get("/session/999999/export.md").status_code == 404


def test_batch_export_route_returns_zip(web_client):
    r = web_client.get("/export?workspace=proj")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "manifest.json" in zf.namelist()


def test_batch_export_route_no_match_is_404(web_client):
    assert web_client.get("/export?workspace=nope").status_code == 404
