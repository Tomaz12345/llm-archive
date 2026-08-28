"""Reading a browser's own IndexedDB store.

Two decoders, and the reason they get their own tests is that both fail *silently* when
they are wrong. A snappy back-reference off by one byte, or a structured-clone tag
skipped by a guessed width, does not raise — it desynchronises the stream and everything
after it decodes into plausible nonsense. So the tests here are mostly about what the
decoders REFUSE to do.

Fixtures are built by hand rather than copied out of a browser profile: the on-disk
format is what is being tested, and a fixture nobody can read is one nobody can fix.
"""

from __future__ import annotations

import sqlite3
import struct

import pytest

from llm_archive.core import idb

# -- building structured clones -------------------------------------------


def word(data: int, tag: int) -> bytes:
    return struct.pack("<II", data & 0xFFFFFFFF, tag)


def sc_string(text: str) -> bytes:
    raw = text.encode("latin-1")
    pad = (-len(raw)) % 8
    return word(len(raw) | idb.STRING_LATIN1, idb.TAG_STRING) + raw + b"\0" * pad


def sc_object(pairs) -> bytes:
    out = word(0, idb.TAG_OBJECT)
    for key, value in pairs:
        out += sc_string(key) + value
    return out + word(0, idb.TAG_END_OF_KEYS)


def sc_array(values) -> bytes:
    out = word(len(values), idb.TAG_ARRAY)
    for i, value in enumerate(values):
        out += word(i, idb.TAG_INT32) + value
    return out + word(0, idb.TAG_END_OF_KEYS)


def sc_int(n: int) -> bytes:
    return word(n, idb.TAG_INT32)


def sc_null() -> bytes:
    return word(0, idb.TAG_NULL)


def sc_bool(v: bool) -> bytes:
    return word(int(v), idb.TAG_BOOLEAN)


def sc_date(ms: int) -> bytes:
    return word(0, idb.TAG_DATE) + struct.pack("<d", float(ms))


def sc_double(x: float) -> bytes:
    return struct.pack("<d", x)


def clone(body: bytes) -> bytes:
    return word(3, idb.TAG_HEADER) + body


# -- snappy ----------------------------------------------------------------


def snappy_literal(payload: bytes) -> bytes:
    """The simplest legal snappy encoding: a length varint then one literal run."""
    size, out = len(payload), bytearray()
    n = size
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            break
    if len(payload) - 1 < 60:
        out.append((len(payload) - 1) << 2)
    else:
        extra = (len(payload) - 1).to_bytes(4, "little")
        out.append((63) << 2)
        out += extra
    return bytes(out) + payload


def test_snappy_round_trips_a_literal():
    payload = b"hello, this is a chat message" * 4
    assert idb.snappy_decompress(snappy_literal(payload)) == payload


def test_snappy_expands_a_back_reference():
    # Uncompressed length 7; literal "ab"; then a 1-byte-offset copy of 5 from offset 2.
    # The copy overlaps what it is writing — that is how snappy encodes a repeat, and
    # the reason the decoder cannot do it as a slice.
    body = bytes([0x07, 0x04]) + b"ab" + bytes([(1 << 2) | 1, 2])
    assert idb.snappy_decompress(body) == b"abababa"


def test_snappy_rejects_a_length_that_does_not_match():
    """The header states the uncompressed size, so a bad decode is caught, not returned."""
    good = snappy_literal(b"12345678")
    lied = bytes([99]) + good[1:]           # claim 99 bytes, deliver 8
    with pytest.raises(idb.SnappyError):
        idb.snappy_decompress(lied)


def test_snappy_rejects_a_back_reference_past_the_start():
    with pytest.raises(idb.SnappyError):
        idb.snappy_decompress(bytes([0x04, (1 << 2) | 1, 200]))


# -- structured clone ------------------------------------------------------


def test_decodes_a_record_shaped_like_openrouters():
    body = sc_object([
        ("id", sc_string("msg-1787652204-7U7e1MmNVDmK9dA40X2h")),
        ("parentMessageId", sc_null()),
        ("createdAt", sc_date(1787652204001)),
        ("isEdited", sc_bool(False)),
        ("type", sc_string("user")),
        ("items", sc_array([sc_object([("id", sc_string("item-1"))])])),
    ])
    assert idb.decode(clone(body)) == {
        "id": "msg-1787652204-7U7e1MmNVDmK9dA40X2h",
        "parentMessageId": None,
        "createdAt": 1787652204001,
        "isEdited": False,
        "type": "user",
        "items": [{"id": "item-1"}],
    }


def test_utf16_strings_decode():
    raw = "žolč".encode("utf-16-le")
    pad = (-len(raw)) % 8
    body = sc_object([("t", word(len("žolč"), idb.TAG_STRING) + raw + b"\0" * pad)])
    assert idb.decode(clone(body)) == {"t": "žolč"}


def test_doubles_are_untagged():
    """Any word whose high half is below TAG_FLOAT_MAX is raw IEEE754, not a tag."""
    body = sc_object([("ratio", sc_double(0.25))])
    assert idb.decode(clone(body)) == {"ratio": 0.25}


def test_nested_objects_and_arrays():
    body = sc_object([("data", sc_object([
        ("role", sc_string("user")),
        ("content", sc_array([
            sc_object([("type", sc_string("input_text")),
                       ("text", sc_string("hello"))])])),
    ]))])
    assert idb.decode(clone(body))["data"]["content"][0]["text"] == "hello"


def test_a_sparse_array_keeps_its_indices():
    """Keys decide placement, so a gap reads as a hole rather than shifting entries."""
    body = sc_object([("xs", word(3, idb.TAG_ARRAY)
                       + word(0, idb.TAG_INT32) + sc_string("a")
                       + word(2, idb.TAG_INT32) + sc_string("c")
                       + word(0, idb.TAG_END_OF_KEYS))])
    assert idb.decode(clone(body)) == {"xs": ["a", None, "c"]}


def test_back_references_resolve_to_the_same_object():
    """OpenRouter's `character` records share one model descriptor between fields."""
    shared = sc_object([("model", sc_string("minimax/minimax-m3:free"))])
    body = sc_object([
        ("a", shared),
        # object 0 is the outer record, object 1 is `shared`
        ("b", word(1, idb.TAG_BACK_REFERENCE)),
    ])
    out = idb.decode(clone(body))
    assert out["a"] == out["b"] == {"model": "minimax/minimax-m3:free"}
    assert out["a"] is out["b"]


def test_dates_take_a_slot_in_the_back_reference_numbering():
    """A boxed value is a JS object. Miss it and every later reference is off by one.

    This is the real bug that made three `character` records fail with
    "back-reference to object 49, only 49 seen".
    """
    shared = sc_object([("k", sc_string("v"))])
    body = sc_object([
        ("when", sc_date(1787652204001)),   # object 1
        ("a", shared),                      # object 2
        ("b", word(2, idb.TAG_BACK_REFERENCE)),
    ])
    out = idb.decode(clone(body))
    assert out["b"] == {"k": "v"}


def test_an_unknown_tag_is_refused_not_skipped():
    """The whole safety story: a guessed width desynchronises everything after it."""
    body = sc_object([("x", word(0, 0xFFFF00FE))])
    with pytest.raises(idb.CloneError, match="unknown tag"):
        idb.decode(clone(body))


def test_a_truncated_record_is_refused():
    body = sc_object([("id", sc_string("abc"))])
    with pytest.raises(idb.CloneError):
        idb.decode(clone(body)[:-6])


def test_something_that_is_not_a_clone_is_refused():
    with pytest.raises(idb.CloneError, match="not a structured clone"):
        idb.decode(word(0, idb.TAG_OBJECT) + word(0, idb.TAG_END_OF_KEYS))


def test_a_dangling_back_reference_is_refused():
    body = sc_object([("b", word(9, idb.TAG_BACK_REFERENCE))])
    with pytest.raises(idb.CloneError, match="back-reference"):
        idb.decode(clone(body))


def test_deep_nesting_stops_rather_than_blowing_the_stack():
    body = sc_string("leaf")
    for _ in range(idb.MAX_DEPTH + 5):
        body = sc_object([("n", body)])
    with pytest.raises(idb.CloneError, match="depth"):
        idb.decode(clone(body))


# -- the store -------------------------------------------------------------


def make_store(path, records: dict[str, bytes], store_name="openrouter:playground:v3"):
    """A minimal Firefox-shaped IndexedDB file."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE object_store (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE object_data (object_store_id INTEGER, key BLOB, data BLOB);
    """)
    con.execute("INSERT INTO object_store VALUES (1, ?)", (store_name,))
    for key, blob in records.items():
        # Firefox writes string keys with every ASCII byte shifted up by one.
        encoded = bytes(b + 1 for b in key.encode("ascii"))
        con.execute("INSERT INTO object_data VALUES (1, ?, ?)", (encoded, blob))
    con.commit()
    con.close()
    return path


def test_reads_records_out_of_a_store(tmp_path):
    blob = snappy_literal(clone(sc_object([("id", sc_string("msg-1"))])))
    db = make_store(tmp_path / "x.sqlite", {"/v3:message:msg-1": blob})

    with idb.opened(db) as con:
        assert idb.store_names(con) == {1: "openrouter:playground:v3"}
        rows = dict(idb.records(con, 1))
    assert rows == {"/v3:message:msg-1": {"id": "msg-1"}}


def test_a_bad_record_is_reported_not_dropped(tmp_path):
    """A chat that will not decode must be visible, not quietly missing."""
    good = snappy_literal(clone(sc_object([("id", sc_string("ok"))])))
    bad = snappy_literal(clone(sc_object([("x", word(0, 0xFFFF00FE))])))
    db = make_store(tmp_path / "x.sqlite",
                    {"/v3:message:a": good, "/v3:message:b": bad})

    with idb.opened(db) as con:
        rows = dict(idb.records(con, 1))
    assert rows["/v3:message:a"] == {"id": "ok"}
    assert isinstance(rows["/v3:message:b"], idb.CloneError)


def test_the_store_is_copied_before_being_read(tmp_path):
    """A running browser holds a lock, and reading a live profile can corrupt it."""
    db = make_store(tmp_path / "x.sqlite",
                    {"/v3:message:a": snappy_literal(clone(sc_object([])))})
    before = db.stat().st_mtime_ns
    with idb.opened(db) as con:
        con.execute("SELECT 1").fetchone()
    assert db.stat().st_mtime_ns == before


def test_key_decoding():
    assert idb.decode_key(bytes(b + 1 for b in b"/v3:character:char-1")) \
        == "/v3:character:char-1"


def test_origin_dir():
    assert idb.origin_dir("https://openrouter.ai") == "https+++openrouter.ai"


def test_no_profiles_is_not_an_error(tmp_path):
    assert idb.stores("https://openrouter.ai", roots=[tmp_path / "nope"]) == []


def test_finds_a_store_under_a_profile(tmp_path):
    idbdir = (tmp_path / "profiles" / "abc.default" / "storage" / "default"
              / "https+++openrouter.ai" / "idb")
    idbdir.mkdir(parents=True)
    make_store(idbdir / "x.sqlite", {})
    found = idb.stores("https://openrouter.ai", roots=[tmp_path / "profiles"])
    assert [p.name for p in found] == ["x.sqlite"]


def test_the_setting_is_off_until_switched_on(tmp_path):
    from llm_archive.core import db

    con = db.connect(tmp_path / "a.db")
    assert idb.is_enabled(con) is False
    idb.set_enabled(con, True)
    assert idb.is_enabled(con) is True
    idb.set_enabled(con, False)
    assert idb.is_enabled(con) is False


# -- the OpenRouter browser route -----------------------------------------


def openrouter_store(tmp_path, rooms):
    """A store shaped like OpenRouter's: master -> room -> manifest -> message/item."""
    records: dict[str, bytes] = {}
    room_ids = []
    for room_id, title, turns in rooms:
        room_ids.append(room_id)
        message_ids, item_ids = [], []
        for mid, role, text in turns:
            iid = f"item-{mid}"
            message_ids.append(mid)
            item_ids.append(iid)
            records[f"/v3:message:{mid}"] = snappy_literal(clone(sc_object([
                ("id", sc_string(mid)),
                ("type", sc_string(role)),
                ("characterId", sc_string("USER" if role == "user" else "char-1")),
                ("parentMessageId", sc_null()),
                ("createdAt", sc_date(int(mid.split("-")[1]) * 1000)),
                ("items", sc_array([sc_object([("id", sc_string(iid))])])),
            ])))
            records[f"/v3:item:{iid}"] = snappy_literal(clone(sc_object([
                ("id", sc_string(iid)),
                ("messageId", sc_string(mid)),
                ("data", sc_object([
                    ("type", sc_string("message")),
                    ("role", sc_string(role)),
                    ("content", sc_array([sc_object([
                        ("type", sc_string("input_text")),
                        ("text", sc_string(text))])])),
                ])),
            ])))
        records[f"/v3:room:{room_id}"] = snappy_literal(clone(sc_object([
            ("id", sc_string(room_id)), ("title", sc_string(title)),
            ("createdAt", sc_date(1787652203000)),
            ("updatedAt", sc_date(1787652483000)),
        ])))
        records[f"/v3:manifest:{room_id}"] = snappy_literal(clone(sc_object([
            ("messageIds", sc_array([sc_string(m) for m in message_ids])),
            ("itemIds", sc_array([sc_string(i) for i in item_ids])),
            ("characterIds", sc_array([])),
            ("artifactIds", sc_array([])),
        ])))
    records["/v3:master:rooms"] = snappy_literal(clone(sc_object([
        ("roomIds", sc_array([sc_string(r) for r in room_ids]))])))

    idbdir = (tmp_path / "profiles" / "p.default" / "storage" / "default"
              / "https+++openrouter.ai" / "idb")
    idbdir.mkdir(parents=True)
    return make_store(idbdir / "or.sqlite", records)


def parse_store(store_path, monkeypatch, tmp_path):
    from llm_archive.adapters import openrouter
    from llm_archive.core.models import ParseStats

    monkeypatch.setattr(
        openrouter, "browser_stores",
        lambda: [openrouter.BrowserStore(store_path, 1, "p.default")])
    adapter = openrouter.OpenRouterAdapter(drops=tmp_path / "none", browser=True)
    stats = ParseStats()
    out = [s for t in adapter.discover() for s in adapter.parse(t, stats)]
    return out, stats


def test_a_browser_room_becomes_a_session(tmp_path, monkeypatch):
    store = openrouter_store(tmp_path, [(
        "orc-1787652203-A", "About V",
        [("msg-1787652204-a", "user", "tell me about V"),
         ("msg-1787652300-b", "assistant", "V is a compiled language")])])

    sessions, stats = parse_store(store, monkeypatch, tmp_path)

    assert len(sessions) == 1
    session = sessions[0]
    # Keyed on the ROOT MESSAGE, exactly as the exported-file route keys it — which is
    # what makes the two routes merge instead of storing the chat twice.
    assert session.native_id == "msg-1787652204-a"
    assert session.meta["origin"] == "browser"
    assert session.exported_at is not None
    assert [p.text for m in session.messages for p in m.parts] == [
        "tell me about V", "V is a compiled language"]


def test_the_browser_route_appends_to_a_chat_you_exported(tmp_path, monkeypatch):
    """The point of the whole feature: the chat grew after you exported it."""
    from llm_archive.core import db, ingest

    con = db.connect(tmp_path / "a.db")
    src = db.source_id(con, "openrouter", "OpenRouter", "web")

    exported = openrouter_store(tmp_path / "old", [(
        "orc-1787652203-A", "About V",
        [("msg-1787652204-a", "user", "tell me about V"),
         ("msg-1787652300-b", "assistant", "V is compiled")])])
    first, _ = parse_store(exported, monkeypatch, tmp_path / "old")
    for session in first:
        db.upsert_session(con, src, session)

    # Same chat, two turns longer, captured later.
    grown = openrouter_store(tmp_path / "new", [(
        "orc-1787652203-A", "About V",
        [("msg-1787652204-a", "user", "tell me about V"),
         ("msg-1787652300-b", "assistant", "V is compiled"),
         ("msg-1787652400-c", "user", "and its GC?"),
         ("msg-1787652500-d", "assistant", "optional, autofree")])])
    import os
    later = os.path.getmtime(grown) + 3600
    os.utime(grown, (later, later))

    second, _ = parse_store(grown, monkeypatch, tmp_path / "new")
    merged = db.upsert_session(con, src, second[0])

    assert merged.was_new is False
    assert merged.appended == 2
    assert merged.retained == 0
    rows = con.execute("SELECT COUNT(*) c FROM message").fetchone()["c"]
    assert rows == 4, "grown chat appended, not duplicated"
    assert con.execute("SELECT COUNT(*) c FROM session").fetchone()["c"] == 1
    del ingest


def test_an_undecodable_record_is_counted_loudly(tmp_path, monkeypatch):
    store = openrouter_store(tmp_path, [(
        "orc-1", "Chat", [("msg-1787652204-a", "user", "hi")])])
    con = sqlite3.connect(store)
    con.execute("INSERT INTO object_data VALUES (1, ?, ?)",
                (bytes(b + 1 for b in b"/v3:message:msg-bad"),
                 snappy_literal(clone(sc_object([("x", word(0, 0xFFFF00FE))])))))
    con.commit()
    con.close()

    _, stats = parse_store(store, monkeypatch, tmp_path)
    assert stats.unknown_types.get("openrouter:browser-record-undecodable") == 1


def test_a_store_with_no_rooms_says_so(tmp_path, monkeypatch):
    """A schema bump renames the keys; the reader must notice rather than mis-parse."""
    store = openrouter_store(tmp_path, [])
    sessions, stats = parse_store(store, monkeypatch, tmp_path)
    assert sessions == []
    assert stats.unknown_types.get("openrouter:browser-store-without-rooms") == 1
