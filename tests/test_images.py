"""Tests for image capture and image rendering.

Three sources put images in the archive and no two of them do it the same way: Gemini
ships real bytes, Claude Code ships base64 inside the record, T3 Chat ships a CDN URL and
nothing else. The viewer has to end up showing a picture in all three cases, and has to
say so plainly in the one case where the bytes genuinely do not exist.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from llm_archive.adapters.claude_code import ClaudeCodeAdapter
from llm_archive.core import db, fetch_images
from llm_archive.core.blobs import BlobStore, blob_path, sniff_mime
from llm_archive.core.models import KIND_IMAGE, Message, ParseStats, Part, Session
from llm_archive.web.app import _caption, create_app


def png_bytes(width: int = 2, height: int = 2) -> bytes:
    """A real, decodable PNG — magic bytes alone would not prove the route works."""
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload)))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 32
PNG = png_bytes()


# ------------------------------------------------------------------ sniffing

@pytest.mark.parametrize("data,expected", [
    (PNG, "image/png"),
    (JPEG, "image/jpeg"),
    (b"GIF89a" + b"\x00" * 10, "image/gif"),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    (b"\x00\x00\x00 ftypavif", "image/avif"),
    (b"%PDF-1.7\n", "application/pdf"),
    (b"just some tool output", "application/octet-stream"),
    (b"", "application/octet-stream"),
])
def test_sniff_mime(data, expected):
    assert sniff_mime(data[:16]) == expected


def test_blob_path_prefers_the_derived_location(tmp_path):
    """`blob.path` is an absolute path baked in at ingest; the layout is derivable."""
    store = BlobStore(tmp_path / "blobs")
    sha, _, recorded = store.put_bytes(PNG)

    found = blob_path(tmp_path / "blobs", sha, "D:/gone/somewhere/else")
    assert found == Path(recorded) and found.read_bytes() == PNG


def test_blob_path_rejects_anything_that_is_not_a_hash(tmp_path):
    for sha in ("../../etc/passwd", "", "ZZ" * 32, "abc"):
        assert blob_path(tmp_path, sha) is None


# --------------------------------------------------- claude code base64 images

def image_record(uid: str, media_type: str, data: bytes) -> dict:
    return {
        "type": "user", "uuid": uid, "parentUuid": None,
        "timestamp": "2026-01-01T10:00:00Z", "cwd": "C:\\x\\demo",
        "message": {"role": "user", "content": [
            {"type": "text", "text": "what is wrong with this screenshot"},
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.b64encode(data).decode()}},
        ]},
    }


def parse_records(tmp_path: Path, records: list[dict], blobs: BlobStore | None):
    project = tmp_path / "c--Users-x-Projekti-demo"
    project.mkdir(parents=True, exist_ok=True)
    (project / "s1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")
    stats = ParseStats()
    adapter = ClaudeCodeAdapter(root=tmp_path, blobs=blobs)
    return list(adapter.parse(project, stats)), stats


def test_pasted_screenshot_bytes_reach_the_blob_store(tmp_path):
    """The regression: the adapter used to keep the size and throw away the image."""
    blobs = BlobStore(tmp_path / "blobs")
    sessions, stats = parse_records(
        tmp_path, [image_record("a", "image/png", PNG)], blobs)

    part = sessions[0].messages[0].parts[1]
    assert part.kind == KIND_IMAGE
    assert part.blob_sha and Path(part.blob_path).read_bytes() == PNG
    assert part.bytes == len(PNG)          # true size, not the size of the base64
    assert part.text == "pasted-image.png"
    assert stats.blobs == 1 and stats.blob_bytes == len(PNG)


def test_pasted_document_keeps_its_media_type(tmp_path):
    blobs = BlobStore(tmp_path / "blobs")
    pdf = b"%PDF-1.7\n" + b"x" * 100
    record = image_record("a", "application/pdf", pdf)
    record["message"]["content"][1]["type"] = "document"
    sessions, _ = parse_records(tmp_path, [record], blobs)

    part = sessions[0].messages[0].parts[1]
    assert part.kind == "attachment" and part.text == "pasted-document.pdf"
    assert Path(part.blob_path).read_bytes() == pdf


def test_undecodable_image_does_not_break_the_parse(tmp_path):
    blobs = BlobStore(tmp_path / "blobs")
    record = image_record("a", "image/png", PNG)
    record["message"]["content"][1]["source"]["data"] = None
    sessions, _ = parse_records(tmp_path, [record], blobs)

    part = sessions[0].messages[0].parts[1]
    assert part.kind == KIND_IMAGE and part.blob_sha is None


def test_no_blob_store_still_yields_a_part(tmp_path):
    sessions, _ = parse_records(
        tmp_path, [image_record("a", "image/png", PNG)], None)
    assert sessions[0].messages[0].parts[1].kind == KIND_IMAGE


# --------------------------------------------------------------- url backfill

@pytest.fixture
def archive(tmp_path):
    """An archive holding one stored image, one URL-only image, and one dead end."""
    data = tmp_path / "data"
    data.mkdir()
    con = db.connect(data / "archive.db")
    blobs = BlobStore(data / "blobs")
    sha, size, path = blobs.put_bytes(PNG)
    src = db.source_id(con, "t3chat", "T3 Chat", "web")

    stored = Part(kind=KIND_IMAGE, seq=0, text="slika.png", bytes=size)
    stored.blob_sha, stored.blob_path = sha, path
    messages = [
        Message(native_id="m0", role="user", created_at=1771200000000, seq=0,
                parts=[Part(kind="text", seq=0, text="generate a CV template"),
                       stored]),
        Message(native_id="m1", role="assistant", created_at=1771200001000, seq=1,
                parts=[Part(kind=KIND_IMAGE, seq=0, bytes=118,
                            text="generated.jpg https://cdn.example/f/abc123")]),
        Message(native_id="m2", role="assistant", created_at=1771200002000, seq=2,
                parts=[Part(kind=KIND_IMAGE, seq=0, bytes=0, text=None)]),
    ]
    db.upsert_session(con, src, Session(
        source_kind="t3chat", native_id="t1", title="Visual CV Template Design",
        workspace_key=None, workspace_label=None, started_at=1771200000000,
        raw_path="/raw/threads.json", raw_hash="t1", messages=messages))
    con.commit()
    return data, con, blobs, sha


def test_fetch_pulls_a_url_only_image_into_the_store(archive):
    _, con, blobs, _ = archive
    result = fetch_images.run(con, blobs, fetch=lambda url: JPEG)

    assert result.candidates == 1 and result.fetched == 1
    row = con.execute("""SELECT p.bytes, b.sha256 FROM part p JOIN blob b
                         ON b.id = p.blob_id WHERE p.text LIKE 'generated%'""").fetchone()
    assert row["bytes"] == len(JPEG)
    assert blob_path(blobs.root, row["sha256"]).read_bytes() == JPEG


def test_fetch_keeps_the_reference_text_intact(archive):
    """`part_fts` is an external-content index over `part.text`; rewriting it desyncs."""
    _, con, blobs, _ = archive
    fetch_images.run(con, blobs, fetch=lambda url: JPEG)

    assert con.execute("SELECT text FROM part WHERE bytes = ?",
                       (len(JPEG),)).fetchone()["text"].endswith("abc123")


def test_fetch_is_idempotent(archive):
    _, con, blobs, _ = archive
    fetch_images.run(con, blobs, fetch=lambda url: JPEG)
    again = fetch_images.run(con, blobs, fetch=lambda url: pytest.fail("refetched"))
    assert again.candidates == 0 and again.fetched == 0


def test_an_error_page_is_never_stored_as_an_image(archive):
    """A CDN that forgot the file answers 200 with HTML. That is not a picture."""
    _, con, blobs, _ = archive
    result = fetch_images.run(
        con, blobs, fetch=lambda url: b"<html><body>Not found</body></html>")

    assert result.fetched == 0 and len(result.failed) == 1
    assert "not an image" in result.failed[0][1]


def test_one_dead_link_does_not_end_the_run(archive):
    _, con, blobs, _ = archive

    def boom(url):
        raise TimeoutError("timed out")

    result = fetch_images.run(con, blobs, fetch=boom)
    assert result.fetched == 0 and len(result.failed) == 1


def test_dry_run_writes_nothing(archive):
    _, con, blobs, _ = archive
    result = fetch_images.run(con, blobs, dry_run=True,
                              fetch=lambda url: pytest.fail("fetched"))
    assert result.candidates == 1 and result.fetched == 0


@pytest.mark.parametrize("text,expected", [
    ("slika-52785bee74b1e9bd.png", "slika-52785bee74b1e9bd.png"),
    ("generated.jpg https://cdn.example/f/abc", "generated.jpg"),
    ("https://cdn.example/f/abc", ""),
    (None, ""),
    ("", ""),
])
def test_caption_handles_every_adapter_shape(text, expected):
    assert _caption(text) == expected


# ------------------------------------------------------------------- the page

@pytest.fixture
def client(archive):
    data, con, _, _ = archive
    con.close()
    return TestClient(create_app(data))


def test_stored_image_renders_inline(client, archive):
    _, _, _, sha = archive
    body = client.get("/session/1").text
    assert f'<img src="/blob/{sha}"' in body
    assert f'href="/blob/{sha}"' in body        # click-through to full size


def test_blob_route_serves_the_real_bytes(client, archive):
    _, _, _, sha = archive
    r = client.get(f"/blob/{sha}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")
    assert r.content == PNG


def test_blob_route_refuses_a_path_instead_of_a_hash(client):
    assert client.get("/blob/..%2F..%2Farchive.db").status_code == 404
    assert client.get("/blob/" + "z" * 64).status_code == 404


def test_non_image_blob_is_offered_as_a_download_never_rendered(archive):
    """The store is mostly raw tool output. Serving that inline is how it becomes markup."""
    data, con, blobs, _ = archive
    sha, size, path = blobs.put_bytes(b"<script>alert(1)</script>")
    db.blob_id(con, sha, size, path)
    con.commit()
    con.close()

    r = TestClient(create_app(data)).get(f"/blob/{sha}")
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert "attachment" in r.headers["content-disposition"]


def test_url_only_image_says_so_and_does_not_hotlink(client):
    body = client.get("/session/1").text
    assert "image not stored" in body
    assert "llma fetch-images" in body
    # the whole point of the backfill: the page must never fetch from the CDN
    assert "https://cdn.example" not in body.split("image not stored")[1][:400]


def test_stylesheet_url_is_cache_busted(client):
    """A CSS change shipping invisibly is how the images first rendered unstyled.

    StaticFiles sets no `Cache-Control`, so a browser may hold the old stylesheet
    indefinitely while happily taking the new markup.
    """
    body = client.get("/session/1").text
    match = re.search(r'href="/static/app\.css\?v=(\d+)"', body)
    assert match and int(match.group(1)) > 0


def test_images_survive_hiding_tool_calls(client, archive):
    """An image is content, not machinery — 'hide tool calls' used to swallow it."""
    _, _, _, sha = archive
    assert f'<img src="/blob/{sha}"' in client.get("/session/1?tools=false").text


def test_fetched_image_then_shows_up_in_the_page(archive):
    data, con, blobs, _ = archive
    fetch_images.run(con, blobs, fetch=lambda url: JPEG)
    con.close()

    body = TestClient(create_app(data)).get("/session/1").text
    assert body.count("<img src=\"/blob/") == 2
    assert "image not stored" in body      # the one with no URL at all still says so


# ------------------------------------------------- the backfill `serve` runs at start

def test_start_backfill_fetches_and_the_page_then_shows_the_image(archive):
    """The whole point: start the server, the picture is there without typing anything."""
    data, con, _, sha = archive
    con.close()

    thread = fetch_images.backfill_on_start(
        data / "archive.db", data / "blobs", echo=lambda line: None,
        fetch=lambda url: JPEG)
    assert thread is not None
    thread.join(timeout=10)
    assert not thread.is_alive()

    body = TestClient(create_app(data)).get("/session/1").text
    assert body.count('<img src="/blob/') == 2


def test_start_backfill_opens_no_socket_when_nothing_is_pending(archive):
    """The short circuit that makes this tolerable on every single start."""
    data, con, blobs, _ = archive
    fetch_images.run(con, blobs, fetch=lambda url: JPEG)
    con.close()

    thread = fetch_images.backfill_on_start(
        data / "archive.db", data / "blobs", echo=lambda line: None,
        fetch=lambda url: pytest.fail("reached the network with nothing to fetch"))
    assert thread is None


def test_start_backfill_survives_a_failure_instead_of_killing_serve(archive):
    data, con, _, _ = archive
    con.close()
    lines: list[str] = []

    def boom(url):
        raise OSError("no route to host")

    thread = fetch_images.backfill_on_start(
        data / "archive.db", data / "blobs", echo=lines.append, fetch=boom)
    thread.join(timeout=10)

    assert not thread.is_alive()
    # and the page still renders, with the honest placeholder it had before
    assert "image not stored" in TestClient(create_app(data)).get("/session/1").text


def test_start_backfill_runs_as_a_daemon(archive):
    """A slow CDN must not be the reason Ctrl-C does not exit."""
    data, con, _, _ = archive
    con.close()
    thread = fetch_images.backfill_on_start(
        data / "archive.db", data / "blobs", echo=lambda line: None,
        fetch=lambda url: JPEG)
    assert thread.daemon
    thread.join(timeout=10)


def test_lifespan_starts_the_backfill(archive, monkeypatch):
    """Wired to the app's start, not just callable on its own."""
    data, con, _, _ = archive
    con.close()
    calls: list[tuple] = []
    monkeypatch.setattr(fetch_images, "backfill_on_start",
                        lambda *a, **kw: calls.append(a) or None)

    with TestClient(create_app(data)):
        pass
    assert calls and calls[0][0] == data / "archive.db"


def test_no_fetch_images_keeps_the_server_offline(archive, monkeypatch):
    data, con, _, _ = archive
    con.close()
    monkeypatch.setattr(fetch_images, "backfill_on_start",
                        lambda *a, **kw: pytest.fail("backfill ran despite the opt-out"))

    with TestClient(create_app(data, fetch_images=False)):
        pass


# ------------------------------------------------------------ the give-up guard

def test_run_gives_up_after_a_run_of_failures(tmp_path):
    """A laptop whose network is not up at logon must not spend TIMEOUT per candidate."""
    data = tmp_path / "data"
    data.mkdir()
    con = db.connect(data / "archive.db")
    blobs = BlobStore(data / "blobs")
    src = db.source_id(con, "t3chat", "T3 Chat", "web")
    messages = [
        Message(native_id=f"m{i}", role="assistant", created_at=1771200000000 + i, seq=i,
                parts=[Part(kind=KIND_IMAGE, seq=0, bytes=118,
                            text=f"gen{i}.jpg https://cdn.example/f/{i}")])
        for i in range(10)
    ]
    db.upsert_session(con, src, Session(
        source_kind="t3chat", native_id="t2", title="many", workspace_key=None,
        workspace_label=None, started_at=1771200000000, raw_path="/raw/x",
        raw_hash="t2", messages=messages))
    con.commit()

    attempts = []

    def offline(url):
        attempts.append(url)
        raise OSError("network is unreachable")

    result = fetch_images.run(con, blobs, fetch=offline, stop_after_failures=3)

    assert result.aborted and result.candidates == 10
    assert len(attempts) == 3           # not all ten
    assert len(result.failed) == 3


def test_a_success_resets_the_failure_run(tmp_path):
    """Two dead links either side of a live one is not 'the network is down'."""
    data = tmp_path / "data"
    data.mkdir()
    con = db.connect(data / "archive.db")
    blobs = BlobStore(data / "blobs")
    src = db.source_id(con, "t3chat", "T3 Chat", "web")
    messages = [
        Message(native_id=f"m{i}", role="assistant", created_at=1771200000000 + i, seq=i,
                parts=[Part(kind=KIND_IMAGE, seq=0, bytes=118,
                            text=f"gen{i}.jpg https://cdn.example/f/{i}")])
        for i in range(5)
    ]
    db.upsert_session(con, src, Session(
        source_kind="t3chat", native_id="t3", title="mixed", workspace_key=None,
        workspace_label=None, started_at=1771200000000, raw_path="/raw/x",
        raw_hash="t3", messages=messages))
    con.commit()

    def flaky(url):
        if url.endswith(("/1", "/3")):
            return JPEG
        raise OSError("dead")

    result = fetch_images.run(con, blobs, fetch=flaky, stop_after_failures=3)

    assert not result.aborted
    assert result.fetched == 2 and len(result.failed) == 3


def test_a_typed_command_never_gives_up_early(archive):
    """`stop_after_failures` defaults off: the CLI works the whole list."""
    _, con, blobs, _ = archive
    result = fetch_images.run(con, blobs, fetch=lambda url: (_ for _ in ()).throw(
        OSError("dead")))
    assert not result.aborted
