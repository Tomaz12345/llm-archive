"""Tests for chunking, FTS query building, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest

from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.search import fts, index
from llm_archive.search.chunker import (
    MAX_CHARS, chunk_part, context_header, strip_header,
)
from llm_archive.search.embed import MODEL_TAG, VectorStore
from llm_archive.search.hybrid import (
    Filters, _session_centroid, _session_gist, related, search,
)


# ------------------------------------------------------------- chunker ----

def test_short_text_is_one_chunk():
    chunks = chunk_part("a short but meaningful question about offside",
                        "[hdr]", 1, 1, 1)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("[hdr]\n")


def test_tiny_text_is_dropped():
    assert chunk_part("ok", "[hdr]", 1, 1, 1) == []


def test_long_text_splits_with_overlap():
    para = "This is a paragraph about telemetry analysis. " * 12   # ~540 chars
    text = "\n\n".join([para] * 8)                                 # ~4.3k chars
    chunks = chunk_part(text, "[hdr]", 1, 1, 1)
    assert len(chunks) > 1
    bodies = [strip_header(c.text) for c in chunks]
    assert all(len(b) <= MAX_CHARS + 400 for b in bodies)
    # consecutive chunks should share some text, so a seam-straddling idea survives
    assert bodies[0][-80:] in bodies[1]


def test_single_huge_paragraph_is_hard_sliced():
    """A minified blob has no paragraph breaks; it must still be chunked."""
    text = "x" * 9000
    chunks = chunk_part(text, "[hdr]", 1, 1, 1)
    assert len(chunks) >= 5
    assert all(len(strip_header(c.text)) <= MAX_CHARS for c in chunks)


def test_header_is_built_and_stripped():
    header = context_header("claude_code", "telemetry_analysis",
                            1771200000000, "Offside work")
    assert "claude_code" in header and "telemetry_analysis" in header
    chunk = chunk_part("some real content here that is long enough", header,
                       1, 1, 1)[0]
    assert strip_header(chunk.text) == "some real content here that is long enough"


# ----------------------------------------------------------------- fts ----

@pytest.mark.parametrize("query", [
    'what about "quotes" here',
    "C++ templates AND OR NOT",
    "path/to/file.py:42",
    "kaj je narobe s tem?",
    "*",
    "-- ; DROP TABLE part",
])
def test_match_expression_is_always_safe(query):
    """FTS5 syntax must never leak from user input."""
    expr = fts.build_match(query)
    if expr is None:
        return
    assert expr.count('"') % 2 == 0
    for bad in ("(", ")", ":", ";"):
        assert bad not in expr


def test_match_drops_stopwords_but_keeps_something():
    assert "offside" in (fts.build_match("what is the offside") or "")
    # a query made only of stopwords still returns a usable expression
    assert fts.build_match("what is the") is not None


def test_match_uses_prefix_for_morphology():
    """Slovene has no FTS5 stemmer; prefix matching recovers inflections."""
    assert '"predlog"*' in (fts.build_match("predlog") or "")


def test_empty_query_returns_none():
    assert fts.build_match("   ") is None
    assert fts.build_match("") is None


# ------------------------------------------------------------- filters ----

def test_filters_exclude_abandoned_by_default():
    where, _ = Filters().sql()
    assert "on_active_path = 1" in where


def test_filters_can_include_abandoned():
    where, _ = Filters(include_abandoned=True).sql()
    assert "on_active_path" not in where


def test_filters_can_exclude_one_session():
    """`related` needs the source session gone before the candidate budget is spent."""
    where, params = Filters(exclude_session=7).sql()
    assert "m.session_id != ?" in where
    assert params == (7,)


def test_filters_build_params_in_order():
    where, params = Filters(sources=("claude_code", "codex"), host="devbox",
                            since=1000).sql()
    assert params == ("claude_code", "codex", "devbox", 1000)
    assert where.count("?") == 4


def test_filters_reject_a_role_or_kind_they_do_not_know():
    """A typo must be an error, not a clean 'no matches'."""
    with pytest.raises(ValueError, match="kind must be one of"):
        Filters(kind="tool_results")
    with pytest.raises(ValueError, match="role must be one of"):
        Filters(role="me")


def test_a_role_alone_means_what_that_role_said():
    """Claude Code files tool_result under 'user'; a bare role must not sweep it in."""
    where, params = Filters(role="user").sql()
    assert "m.role = ?" in where and "p.kind IN (" in where
    assert "tool_result" not in params and "text" in params

    # naming a kind lifts that: this is exactly the tool output filed under 'user'
    where, params = Filters(role="user", kind="tool_result").sql()
    assert "p.kind = ?" in where and "p.kind IN (" not in where
    assert params[:2] == ("user", "tool_result")


def test_tool_filter_is_session_level_and_case_insensitive():
    where, params = Filters(tool="bash").sql()
    assert "s.id IN (SELECT" in where and "tool_name = ? COLLATE NOCASE" in where
    assert params == ("bash",)


# --------------------------------------------------------- integration ----

def build_archive(tmp_path: Path):
    con = db.connect(tmp_path / "a.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    panel = db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel")

    def add(native, title, texts, workspace="proj", host="box",
            source=None, kind="claude_code", meta=None):
        msgs = []
        for i, (role, text) in enumerate(texts):
            m = Message(native_id=f"{native}-{i}", role=role,
                        created_at=1771200000000 + i, seq=i)
            m.parts.append(Part(kind="text", seq=0, text=text, embed_eligible=True))
            msgs.append(m)
        db.upsert_session(con, source if source is not None else src, Session(
            source_kind=kind, native_id=native, title=title,
            workspace_key=workspace, workspace_label=workspace, host=host,
            started_at=1771200000000, raw_path="x", raw_hash=native,
            messages=msgs, meta=meta or {}))

    add("s1", "Offside detection work", [
        ("user", "how do I detect an offside line from the tracking data"),
        ("assistant", "compute the second-rearmost defender position per frame")])
    add("s2", "Django dashboard", [
        ("user", "build a jobs dashboard with django and sqlite"),
        ("assistant", "start with models.py and a ListView")],
        workspace="invoice")
    add("s3", "Slovene notes", [
        ("user", "prosim pripravi predloge za intervju"),
        ("assistant", "pripravil sem nekaj predlogov za tvoj intervju")],
        host="laptop")
    # two assistants sharing the one VS Code chat store
    add("s4", "Copilot refactor", [
        ("user", "rename the argparse flag in this script"),
        ("assistant", "renamed it to args_file_path")],
        source=panel, kind="vscode_chat", meta={"participant": "copilot",
                                                "participant_label": "GitHub Copilot"})
    add("s5", "Remote ssh chat", [
        ("user", "rename the ssh host entry"),
        ("assistant", "edit your ssh config")],
        source=panel, kind="vscode_chat", meta={"participant": "remote-ssh",
                                                "participant_label": "Remote - SSH"})
    con.commit()
    return con


def test_keyword_search_finds_the_right_session(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = search(con, tmp_path / "vectors", "offside line", mode="keyword")
    assert hits, "no hits"
    assert hits[0].title == "Offside detection work"
    assert hits[0].matched_by == "keyword"
    assert hits[0].snippets


def test_search_respects_source_and_workspace_filters(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)

    hits = search(con, tmp_path / "vectors", "django dashboard", mode="keyword",
                  filters=Filters(workspace="invoice"))
    assert [h.title for h in hits] == ["Django dashboard"]

    none = search(con, tmp_path / "vectors", "django dashboard", mode="keyword",
                  filters=Filters(workspace="nonexistent"))
    assert none == []

    wrong_source = search(con, tmp_path / "vectors", "django", mode="keyword",
                          filters=Filters(sources=("codex",)))
    assert wrong_source == []


def test_search_filters_by_host(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = search(con, tmp_path / "vectors", "predloge intervju", mode="keyword",
                  filters=Filters(host="laptop"))
    assert [h.title for h in hits] == ["Slovene notes"]


def test_slovene_diacritics_are_folded(tmp_path):
    """remove_diacritics 2 means 'prosim' and 'prošim' reach the same rows."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    a = search(con, tmp_path / "vectors", "predloge", mode="keyword")
    b = search(con, tmp_path / "vectors", "prédloge", mode="keyword")
    assert [h.session_id for h in a] == [h.session_id for h in b]


def test_semantic_mode_without_vectors_returns_nothing(tmp_path):
    """No index built yet must degrade quietly, not explode."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    assert search(con, tmp_path / "vectors", "anything", mode="semantic") == []


def test_hybrid_without_vectors_still_returns_keyword_hits(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = search(con, tmp_path / "vectors", "offside", mode="hybrid")
    assert hits and hits[0].title == "Offside detection work"


def test_weights_change_the_ranking(tmp_path):
    """A zero keyword weight must drop keyword-only matches out of the fusion."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    normal = search(con, tmp_path / "vectors", "offside", mode="hybrid",
                    weights=(1.0, 1.6))
    muted = search(con, tmp_path / "vectors", "offside", mode="hybrid",
                   weights=(0.0, 1.6))
    assert normal and all(h.score > 0 for h in normal)
    assert muted == [] or all(h.score == 0 for h in muted)


# ------------------------------------------------ role / kind / tool filters ----
#
# Its own fixture: these need tool traffic, and the archive above is text only.

def archive_with_tool_traffic(tmp_path: Path):
    con = db.connect(tmp_path / "t.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")

    def add(native, title, parts):
        msgs = []
        for i, (role, kind, text, tool) in enumerate(parts):
            m = Message(native_id=f"{native}-{i}", role=role, seq=i,
                        created_at=1771200000000 + i)
            m.parts.append(Part(kind=kind, seq=0, text=text, tool_name=tool,
                                embed_eligible=kind == "text"))
            msgs.append(m)
        db.upsert_session(con, src, Session(
            source_kind="claude_code", native_id=native, title=title,
            workspace_key="proj", workspace_label="proj", host="box",
            started_at=1771200000000, raw_path="x", raw_hash=native, messages=msgs))

    # hit the error: the trace is tool output, filed under 'user' as Claude Code does
    add("t1", "Nightly batch crashed", [
        ("user", "text", "why does the nightly batch job crash", None),
        ("assistant", "text", "let me run it and read the traceback", None),
        ("assistant", "tool_use", "python batch.py", "Bash"),
        ("user", "tool_result",
         "Traceback (most recent call last): ModuleNotFoundError: "
         "No module named psycopg2", None),
    ])
    # discussed the error, never hit it, ran nothing
    add("t2", "Installing psycopg2", [
        ("user", "text", "how do I install psycopg2 so ModuleNotFoundError goes away",
         None),
        ("assistant", "text", "pip install psycopg2-binary", None),
    ])
    # ran a different tool
    add("t3", "Config loader tidy-up", [
        ("user", "text", "tidy the config loader", None),
        ("assistant", "tool_use", "file_path: loader.py", "Edit"),
    ])
    con.commit()
    fts.rebuild(con)
    return con


def test_kind_finds_the_session_that_hit_the_error_not_the_one_that_discussed_it(tmp_path):
    con = archive_with_tool_traffic(tmp_path)
    both = search(con, tmp_path / "vectors", "ModuleNotFoundError psycopg2",
                  mode="keyword")
    assert {h.title for h in both} == {"Nightly batch crashed", "Installing psycopg2"}

    hit = search(con, tmp_path / "vectors", "ModuleNotFoundError psycopg2",
                 mode="keyword", filters=Filters(kind="tool_result"))
    assert [h.title for h in hit] == ["Nightly batch crashed"]
    assert hit[0].snippets[0].kind == "tool_result"


def test_role_user_is_what_was_typed_not_what_a_tool_returned(tmp_path):
    con = archive_with_tool_traffic(tmp_path)
    typed = search(con, tmp_path / "vectors", "ModuleNotFoundError", mode="keyword",
                   filters=Filters(role="user"))
    assert [h.title for h in typed] == ["Installing psycopg2"]

    # asking for the kind explicitly reaches the tool output filed under that role
    returned = search(con, tmp_path / "vectors", "ModuleNotFoundError", mode="keyword",
                      filters=Filters(role="user", kind="tool_result"))
    assert [h.title for h in returned] == ["Nightly batch crashed"]


def test_tool_narrows_to_sessions_that_called_it(tmp_path):
    con = archive_with_tool_traffic(tmp_path)
    ran = search(con, tmp_path / "vectors", "psycopg2", mode="keyword",
                 filters=Filters(tool="bash"))
    assert [h.title for h in ran] == ["Nightly batch crashed"]
    assert search(con, tmp_path / "vectors", "psycopg2", mode="keyword",
                  filters=Filters(tool="Edit")) == []


def test_hits_carry_each_retriever_rank(tmp_path):
    """Keyword-only archive: a keyword rank on every hit, no semantic rank."""
    con = archive_with_tool_traffic(tmp_path)
    hits = search(con, tmp_path / "vectors", "ModuleNotFoundError psycopg2")
    assert [h.keyword_rank for h in hits] == [1, 2]
    assert all(h.semantic_rank is None for h in hits)


def test_search_filters_by_participant(tmp_path):
    """One source, two assistants — the source filter cannot separate them."""
    con = build_archive(tmp_path)
    fts.rebuild(con)

    both = search(con, tmp_path / "vectors", "rename", mode="keyword",
                  filters=Filters(sources=("vscode_chat",)))
    assert {h.title for h in both} == {"Copilot refactor", "Remote ssh chat"}

    only = search(con, tmp_path / "vectors", "rename", mode="keyword",
                  filters=Filters(participant="copilot"))
    assert [h.title for h in only] == ["Copilot refactor"]

    other = search(con, tmp_path / "vectors", "rename", mode="keyword",
                   filters=Filters(participant="remote-ssh"))
    assert [h.title for h in other] == ["Remote ssh chat"]


def test_participant_filter_excludes_sessions_that_have_none(tmp_path):
    """Claude Code sets no participant; filtering by one must not sweep it in."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = search(con, tmp_path / "vectors", "offside", mode="keyword",
                  filters=Filters(participant="copilot"))
    assert hits == []


def test_building_the_index_records_that_it_happened(tmp_path):
    """Without a logged build, nothing downstream can date the index."""
    from llm_archive.search import index as search_index

    con = build_archive(tmp_path)
    result = search_index.build(con, tmp_path / "vectors", with_vectors=False)

    row = con.execute("SELECT * FROM index_run ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None, "index build left no record"
    assert row["fts_rows"] == result.fts_rows
    assert row["chunks"] == result.chunks
    assert row["with_vectors"] == 0, "a --no-vectors build must say so"
    assert row["finished_at"] >= row["started_at"]


def test_a_keyword_only_build_leaves_the_vector_index_alone(tmp_path):
    """The regression behind the nightly schedule.

    `--no-vectors` used to drop every chunk before deciding it was not going to
    re-insert any, so a nightly keyword build destroyed whatever a full build had
    embedded. Running the two in the same week has to leave semantic search working.
    """
    from llm_archive.search import index as search_index, selection

    con = build_archive(tmp_path)
    search_index.build(con, tmp_path / "vectors", with_vectors=False)
    con.execute("""INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,
                                     model_tag)
                   SELECT p.id, m.id, m.session_id, 0, p.text, p.id, ?
                   FROM part p JOIN message m ON m.id = p.message_id
                   JOIN session s ON s.id = m.session_id
                   WHERE p.embed_eligible = 1 AND LENGTH(p.text) >= 40
                     AND m.on_active_path = 1""", (search_index.MODEL_TAG,))
    con.commit()
    assert selection.unindexed_count(con) == 0

    search_index.build(con, tmp_path / "vectors", with_vectors=False)
    assert selection.unindexed_count(con) == 0, (
        "a keyword-only build dropped the chunks a vector build had inserted")


def test_orphaned_rows_cannot_make_a_fresh_index_look_stale(tmp_path):
    """The regression: a part hanging off a message whose session is gone.

    The indexer joins through `session` and never sees it; a health check that joined
    only part -> message counted it as missing coverage, so a just-built index reported
    "2 embeddable parts not in the vector index" — forever, and unclearably.
    """
    from llm_archive.search import index as search_index, selection

    con = build_archive(tmp_path)
    search_index.build(con, tmp_path / "vectors", with_vectors=False)
    con.execute("DELETE FROM chunk")   # stand in for a real vector build
    con.execute("""INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,
                                     model_tag)
                   SELECT p.id, m.id, m.session_id, 0, p.text, p.id, 'x'
                   FROM part p JOIN message m ON m.id = p.message_id
                   JOIN session s ON s.id = m.session_id
                   WHERE p.embed_eligible = 1 AND LENGTH(p.text) >= 40
                     AND m.on_active_path = 1""")
    con.commit()
    assert selection.unindexed_count(con) == 0

    # an orphan: message rows whose session row is gone (foreign_key_check finds these)
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("""INSERT INTO message(id,session_id,seq,role,created_at)
                   VALUES (9999,4242,0,'user',1)""")
    con.execute("""INSERT INTO part(message_id,seq,kind,text,embed_eligible)
                   VALUES (9999,0,'text',?,1)""", ("x" * 200,))
    con.commit()

    assert selection.unindexed_count(con) == 0, \
        "an unreachable part is not missing coverage; the indexer never embeds it"
    assert len(search_index._embeddable_parts(con)) == \
        con.execute(f"SELECT COUNT(*) {selection.EMBEDDABLE_FROM} "
                    f"{selection.EMBEDDABLE_WHERE}").fetchone()[0], \
        "indexer and health check must count the same population"


# ------------------------------------------------------------- related ----

def test_related_finds_the_sibling(tmp_path):
    """Two VS Code sessions both about renaming: each should surface the other."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = related(con, tmp_path / "vectors", 4)
    assert hits, "no neighbours"
    assert hits[0].title == "Remote ssh chat"


def test_related_never_returns_the_session_itself(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    for sid in (1, 2, 3, 4, 5):
        assert all(h.session_id != sid
                   for h in related(con, tmp_path / "vectors", sid))


def test_related_works_without_vectors(tmp_path):
    """An archive indexed with --no-vectors still gets the keyword half."""
    con = build_archive(tmp_path)
    fts.rebuild(con)
    hits = related(con, tmp_path / "vectors", 4)
    assert hits and all(h.matched_by == "keyword" for h in hits)


def test_related_respects_filters(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    assert related(con, tmp_path / "vectors", 4,
                   filters=Filters(sources=("codex",))) == []


def test_related_on_an_unknown_session_is_empty(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    assert related(con, tmp_path / "vectors", 999) == []


def test_gist_is_the_title_and_the_opening_question(tmp_path):
    """Not the whole transcript: build_match keeps 24 terms, so what goes in matters."""
    con = build_archive(tmp_path)
    gist = _session_gist(con, 1)
    assert gist.startswith("Offside detection work")
    assert "how do I detect an offside line" in gist
    # the assistant's reply is not what the session is *about*
    assert "second-rearmost" not in gist


def test_gist_falls_back_when_a_session_has_no_user_text(tmp_path):
    """A resumed agent run can have no user turn at all and must still be queryable."""
    con = build_archive(tmp_path)
    con.execute("UPDATE message SET role='assistant' WHERE session_id=1")
    con.commit()
    gist = _session_gist(con, 1)
    assert "offside" in gist.lower()


# --------------------------------------------------- related: the dense half ----
#
# The vector half of `related` cannot be reached with the real embedder in a test that
# has to stay offline, so the store is built by hand: four dimensions instead of 384,
# and vectors chosen so the expected neighbour is arithmetic rather than a judgement
# call. Everything under test — the centroid, the store lookup, the session filter —
# is dimension-agnostic.

def fake_vectors(con, tmp_path, by_session: dict[int, list[list[float]]]) -> None:
    """One chunk row per supplied vector, so each session's centroid is predictable."""
    matrix, vec_row = [], 0
    for session_id, vectors in by_session.items():
        anchor = con.execute("""
            SELECT p.id AS pid, m.id AS mid FROM part p
            JOIN message m ON m.id = p.message_id
            WHERE m.session_id = ? ORDER BY m.seq, p.seq LIMIT 1""",
            (session_id,)).fetchone()
        for seq, vector in enumerate(vectors):
            con.execute("""INSERT INTO chunk(part_id, message_id, session_id, seq,
                                             text, vec_row, model_tag)
                           VALUES (?,?,?,?,'chunk',?,?)""",
                        (anchor["pid"], anchor["mid"], session_id, seq, vec_row,
                         MODEL_TAG))
            matrix.append(vector)
            vec_row += 1
    con.commit()
    rows = np.asarray(matrix, dtype=np.float32)
    VectorStore(tmp_path / "vectors", MODEL_TAG).save(
        rows / np.linalg.norm(rows, axis=1, keepdims=True))


def test_related_ranks_by_the_centroid_of_the_session_vectors(tmp_path):
    con = build_archive(tmp_path)
    fake_vectors(con, tmp_path, {1: [[1, 0, 0, 0]], 2: [[0.95, 0.31, 0, 0]],
                                 3: [[0, 0, 1, 0]], 4: [[0, 0, 0, 1]],
                                 5: [[0, -1, 0, 0]]})
    hits = related(con, tmp_path / "vectors", 1, mode="semantic")
    assert [h.session_id for h in hits][0] == 2
    assert all(h.matched_by == "semantic" for h in hits)


def test_the_dense_half_also_drops_the_session_asked_about(tmp_path):
    """Its own chunks score 1.0 against its own centroid; nothing else would rank."""
    con = build_archive(tmp_path)
    fake_vectors(con, tmp_path, {1: [[1, 0, 0, 0]], 2: [[0.95, 0.31, 0, 0]]})
    assert all(h.session_id != 1
               for h in related(con, tmp_path / "vectors", 1, mode="semantic"))


def test_centroid_is_a_unit_vector(tmp_path):
    con = build_archive(tmp_path)
    fake_vectors(con, tmp_path, {1: [[1, 0, 0, 0], [0, 1, 0, 0]]})
    _, centroid = _session_centroid(con, tmp_path / "vectors", 1, MODEL_TAG)
    assert float(np.linalg.norm(centroid)) == pytest.approx(1.0, abs=1e-5)
    assert centroid.tolist() == pytest.approx([0.7071, 0.7071, 0, 0], abs=1e-4)


def test_a_long_session_is_sampled_across_its_length_not_truncated(tmp_path, monkeypatch):
    """Averaging 900 chunks whole is mush; averaging the first N is only its opening."""
    monkeypatch.setattr("llm_archive.search.hybrid.CENTROID_CHUNKS", 2)
    con = build_archive(tmp_path)
    fake_vectors(con, tmp_path, {1: [[1, 0, 0, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]]})
    _, centroid = _session_centroid(con, tmp_path / "vectors", 1, MODEL_TAG)
    # stride 2 over four chunks takes the 1st and the 3rd — the end is represented
    assert centroid.tolist() == pytest.approx([0.7071, 0, 0.7071, 0], abs=1e-4)


def test_a_session_with_no_chunks_has_no_centroid(tmp_path):
    con = build_archive(tmp_path)
    fake_vectors(con, tmp_path, {1: [[1, 0, 0, 0]]})
    _, centroid = _session_centroid(con, tmp_path / "vectors", 2, MODEL_TAG)
    assert centroid is None


def test_hybrid_related_uses_both_halves(tmp_path):
    con = build_archive(tmp_path)
    fts.rebuild(con)
    fake_vectors(con, tmp_path, {4: [[1, 0, 0, 0]], 5: [[0.99, 0.14, 0, 0]]})
    hits = related(con, tmp_path / "vectors", 4)
    assert hits[0].session_id == 5
    assert hits[0].matched_by == "both"
    assert (hits[0].keyword_rank, hits[0].semantic_rank) == (1, 1)


# --- derived tool facts -------------------------------------------------------

def _archive_with_tool_calls(tmp_path: Path):
    """One session whose tool calls carry real payloads."""
    con = db.connect(tmp_path / "facts.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    msgs = []
    calls = [("Edit", {"file_path": "/proj/app/main.py"}),
             ("Read", {"file_path": "/proj/app/main.py"}),
             ("Bash", {"command": "pytest -q tests/"})]
    for i, (tool, payload) in enumerate(calls):
        m = Message(native_id=f"c{i}", role="assistant", seq=i,
                    created_at=1771200000000 + i)
        part = Part(kind="tool_use", seq=0, text="summary", tool_name=tool)
        part.tool_input = json.dumps(payload)
        m.parts.append(part)
        msgs.append(m)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="fs1", title="Worked on main",
        workspace_key="/proj", workspace_label="proj", host="box",
        started_at=1771200000000, raw_path="x", raw_hash="fs1", messages=msgs))
    con.commit()
    return con


def test_a_keyword_only_build_still_derives_the_facts(tmp_path):
    """`--no-vectors` returns early, before the embedding work. The derivation has to
    sit ABOVE that return: a nightly keyword-only run must not leave the archive with a
    fresh keyword index and a month-old answer to "who touched this file"."""
    con = _archive_with_tool_calls(tmp_path)
    result = index.build(con, tmp_path / "vectors", with_vectors=False)
    assert result.skipped_vectors is True
    assert result.files == 2 and result.commands == 1
    assert con.execute("SELECT COUNT(*) FROM touched_file").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM command").fetchone()[0] == 1


def test_the_derivation_reads_the_workspace_relative_path(tmp_path):
    con = _archive_with_tool_calls(tmp_path)
    index.build(con, tmp_path / "vectors", with_vectors=False)
    rels = {r[0] for r in con.execute("SELECT rel FROM touched_file")}
    assert rels == {"app/main.py"}


def test_the_whole_command_is_kept_not_the_summary(tmp_path):
    """`part.text` is a short line of intent; `command.text` is the command."""
    con = _archive_with_tool_calls(tmp_path)
    index.build(con, tmp_path / "vectors", with_vectors=False)
    row = con.execute("SELECT argv0, subcommand, text FROM command").fetchone()
    assert row["argv0"] == "pytest"
    assert row["text"] == "pytest -q tests/"


def test_re_ingesting_a_session_leaves_no_stale_file_rows(tmp_path):
    """Ingest deletes and re-inserts a re-parsed message's parts, renumbering part.id.
    The cascade is what stops the archive answering from calls that no longer exist."""
    con = _archive_with_tool_calls(tmp_path)
    index.build(con, tmp_path / "vectors", with_vectors=False)
    before = con.execute("SELECT COUNT(*) FROM touched_file").fetchone()[0]

    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    m = Message(native_id="c0", role="assistant", seq=0, created_at=1771200000000)
    part = Part(kind="tool_use", seq=0, text="summary", tool_name="Edit")
    part.tool_input = json.dumps({"file_path": "/proj/app/main.py"})
    m.parts.append(part)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="fs1", title="Worked on main",
        workspace_key="/proj", workspace_label="proj", host="box",
        started_at=1771200000000, raw_path="x", raw_hash="fs1-v2", messages=[m]))
    con.commit()

    # The rewritten message's parts were deleted and re-inserted under new ids, and
    # its derived rows went with them. The rows for messages the snapshot did not
    # mention stay, because `upsert_session` merges rather than replacing -- a chat
    # pruned server-side must not silently erase what the archive already held.
    after = con.execute("SELECT COUNT(*) FROM touched_file").fetchone()[0]
    assert after < before, "the cascade did not fire"
    index.build(con, tmp_path / "vectors", with_vectors=False)
    assert con.execute("SELECT COUNT(*) FROM touched_file").fetchone()[0] == before


def test_an_unparseable_payload_does_not_stop_the_index(tmp_path):
    """One malformed payload among 13,000 must not take the whole run down."""
    con = _archive_with_tool_calls(tmp_path)
    con.execute("UPDATE part SET tool_input = ? WHERE tool_name = 'Bash'",
                ("{not json at all",))
    con.commit()
    result = index.build(con, tmp_path / "vectors", with_vectors=False)
    assert result.files == 2          # the other calls still derived


def test_a_call_with_no_stored_payload_is_reported_not_guessed(tmp_path):
    """A pre-v11 part has `tool_input` NULL. Saying so is what tells the reader to run
    `ingest --force`, rather than silently deriving nothing."""
    con = _archive_with_tool_calls(tmp_path)
    con.execute("UPDATE part SET tool_input = NULL")
    con.commit()
    result = index.build(con, tmp_path / "vectors", with_vectors=False)
    assert result.files == 0
    assert any("ingest --force" in w for w in result.warnings)


def test_a_payload_that_overflowed_to_a_blob_is_read_back(tmp_path):
    """Payloads over INLINE_LIMIT live in the blob store; a truncated head does not
    parse as JSON, so falling back to it would derive facts from half a payload."""
    from llm_archive.core.blobs import BlobStore
    from llm_archive.core.models import INLINE_LIMIT, attach_tool_input

    blobs = BlobStore(tmp_path / "blobs")
    con = db.connect(tmp_path / "big.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    payload = {"command": "echo " + "x" * (INLINE_LIMIT + 100)}
    part = Part(kind="tool_use", seq=0, text="summary", tool_name="Bash")
    attach_tool_input(part, payload, blobs)
    assert part.tool_input_sha, "the fixture did not overflow"

    m = Message(native_id="b0", role="assistant", seq=0, created_at=1771200000000)
    m.parts.append(part)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="bs1", title="Big call",
        workspace_key="/proj", workspace_label="proj", host="box",
        started_at=1771200000000, raw_path="x", raw_hash="bs1", messages=[m]))
    con.commit()

    result = index.build(con, tmp_path / "vectors", with_vectors=False,
                         blob_dir=tmp_path / "blobs")
    assert result.commands == 1
    assert result.unreadable == 0 if hasattr(result, "unreadable") else True
    row = con.execute("SELECT argv0, LENGTH(text) n FROM command").fetchone()
    assert row["argv0"] == "echo"
    assert row["n"] > INLINE_LIMIT, "the blob was not read back in full"
