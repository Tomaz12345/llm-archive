"""Tests for chunking, FTS query building, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.search import fts
from llm_archive.search.chunker import (
    MAX_CHARS, chunk_part, context_header, strip_header,
)
from llm_archive.search.hybrid import Filters, search


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


def test_filters_build_params_in_order():
    where, params = Filters(sources=("claude_code", "codex"), host="devbox",
                            since=1000).sql()
    assert params == ("claude_code", "codex", "devbox", 1000)
    assert where.count("?") == 4


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
