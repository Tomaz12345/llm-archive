"""Tests for pricing and metrics.

The cache-multiplier tests matter most. This archive holds 1.4 billion cache-read
tokens; pricing them at the input rate instead of 0.1x would overstate the total by
roughly six thousand dollars, and nothing about the resulting number would look
obviously wrong.
"""

from __future__ import annotations


import pytest

from llm_archive.core import db
from llm_archive.core.models import Message, Part, Session
from llm_archive.stats import charts, metrics, pricing


# ------------------------------------------------------------------ pricing

def test_claude_opus_5_list_rates():
    price, reason = pricing.lookup("claude-opus-5")
    assert (price.input, price.output) == (5.0, 25.0)
    assert reason == "list price"
    assert price.billed is True


def test_cache_read_is_a_tenth_of_input():
    price, _ = pricing.lookup("claude-opus-5")
    assert price.read_rate() == pytest.approx(0.5)


def test_cache_write_is_1_25x_input():
    price, _ = pricing.lookup("claude-opus-5")
    assert price.write_rate() == pytest.approx(6.25)


def test_cost_uses_cache_rates_not_input_rate():
    """The whole point: a billion cache reads are not a billion input tokens."""
    cheap, _ = pricing.cost("claude-opus-5", cache_read=1_000_000_000)
    expensive, _ = pricing.cost("claude-opus-5", tok_in=1_000_000_000)
    assert cheap == pytest.approx(500.0)
    assert expensive == pytest.approx(5000.0)
    assert expensive == pytest.approx(cheap * 10)


def test_cost_combines_all_four_token_kinds():
    usd, billed = pricing.cost("claude-opus-5", tok_in=1_000_000,
                               tok_out=1_000_000, cache_read=1_000_000,
                               cache_write=1_000_000)
    assert usd == pytest.approx(5.0 + 25.0 + 0.5 + 6.25)
    assert billed is True


def test_copilot_models_are_not_metered():
    usd, billed = pricing.cost("copilot/gpt-5-mini", tok_in=10_000_000)
    assert billed is False, "Copilot usage is a subscription, not per-token"
    price, reason = pricing.lookup("copilot/gpt-5-mini")
    assert reason == "subscription"


def test_free_endpoints_cost_nothing():
    usd, billed = pricing.cost("kimi-k2.5-free", tok_in=50_000_000)
    assert usd == 0.0
    assert billed is False


def test_unknown_model_contributes_zero_and_is_reported():
    usd, billed = pricing.cost("some-model-from-2029", tok_in=1_000_000)
    assert usd == 0.0 and billed is False
    cov = pricing.coverage(["some-model-from-2029"])
    assert cov["unpriced"] == 1
    assert "some-model-from-2029" in cov["missing_models"]


@pytest.mark.parametrize("recorded,expected", [
    ("claude-haiku-4.5", "claude-haiku-4-5"),      # VS Code writes dots
    ("claude-opus-5", "claude-opus-5"),
    ("copilot/claude-haiku-4.5", "claude-haiku-4-5"),
    ("CLAUDE-OPUS-5", "claude-opus-5"),
    ("<synthetic>", None),
    (None, None),
])
def test_model_name_normalisation(recorded, expected):
    assert pricing.normalise(recorded) == expected


def test_no_model_means_no_cost_not_a_crash():
    assert pricing.cost(None, tok_in=999) == (0.0, False)


# ------------------------------------------------------------------ metrics

@pytest.fixture
def con(tmp_path):
    connection = db.connect(tmp_path / "a.db")
    src = db.source_id(connection, "claude_code", "Claude Code", "cli")

    def add(native, title, model, when, msgs, **tokens):
        messages = []
        for i, (role, text, kind) in enumerate(msgs):
            m = Message(native_id=f"{native}-{i}", role=role,
                        created_at=when + i * 1000, seq=i, model=model)
            m.parts.append(Part(kind=kind, seq=0, text=text,
                                tool_name="Bash" if kind == "tool_use" else None,
                                tool_ok=False if kind == "tool_result" else None,
                                embed_eligible=kind == "text"))
            messages.append(m)
        db.upsert_session(connection, src, Session(
            source_kind="claude_code", native_id=native, title=title,
            model_primary=model, workspace_key="proj", workspace_label="proj",
            host="box", started_at=when, raw_path="x", raw_hash=native,
            messages=messages, **tokens))

    # 2026-02-16T12:00Z is a Monday
    add("s1", "One", "claude-opus-5", 1771243200000,
        [("user", "hello there friend", "text"),
         ("assistant", "ls -la", "tool_use"),
         ("assistant", "output", "tool_result")],
        tok_in=1_000_000, tok_out=1_000_000,
        tok_cache_read=1_000_000, tok_cache_write=1_000_000)
    add("s2", "Two", "copilot/gpt-5-mini", 1771243200000,
        [("user", "another question here", "text")],
        tok_in=5_000_000)
    connection.commit()
    return connection


def test_overview_counts(con):
    ov = metrics.overview(con)
    assert ov["sessions"] == 2
    assert ov["cache_read"] == 1_000_000


def test_subscription_usage_excluded_from_billed_total(con):
    data = metrics.everything(con)
    # s1 only: 5 + 25 + 0.5 + 6.25
    assert data["cost_total_billed"] == pytest.approx(36.75)
    copilot = [m for m in data["models"] if m["model"].startswith("copilot/")]
    assert copilot and copilot[0]["billed"] is False


def test_model_less_session_is_labelled_by_its_participant(con):
    """Gemini and Mistral export no model at all; requiring one hid them entirely."""
    src = db.source_id(con, "mistral", "Mistral", "web")
    db.upsert_session(con, src, Session(
        source_kind="mistral", native_id="c1", title="Go tutorial",
        started_at=1771243200000, raw_path="x", raw_hash="c1", tok_out=1266,
        messages=[Message(native_id="m1", role="assistant", created_at=1771243200000,
                          parts=[Part(kind="text", seq=0, text="hi",
                                      embed_eligible=True)])],
        meta={"participant": "mistral", "participant_label": "Mistral",
              "model_recorded": False}))
    con.commit()

    row = next(m for m in metrics.models(con) if m["model"] == "Mistral")
    assert row["model_recorded"] is False
    assert row["reason"] == "no model recorded"
    assert row["tok_out"] == 1266          # the number the old filter threw away


def test_a_participant_label_is_never_priced(con):
    """An app name must not reach the price table — `normalise` would prefix-match it."""
    src = db.source_id(con, "mistral", "Mistral", "web")
    db.upsert_session(con, src, Session(
        source_kind="mistral", native_id="c1", title="t",
        started_at=1771243200000, raw_path="x", raw_hash="c1",
        tok_in=10_000_000, tok_out=10_000_000,
        messages=[Message(native_id="m1", role="assistant", created_at=1771243200000,
                          parts=[Part(kind="text", seq=0, text="hi")])],
        meta={"participant_label": "Mistral"}))
    con.commit()

    data = metrics.everything(con)
    row = next(m for m in data["models"] if m["model"] == "Mistral")
    assert row["usd"] == 0.0 and row["billed"] is False
    # 20M tokens of an unpriced participant must not move the headline figure.
    assert data["cost_total_billed"] == pytest.approx(36.75)
    assert data["models_unrecorded"]["sessions"] == 1
    assert data["models_unrecorded"]["tok_out"] == 10_000_000


def test_session_without_participant_falls_back_to_its_source(con):
    """claude.ai records neither a model nor a participant — the source names it."""
    src = db.source_id(con, "claude_web", "Claude.ai", "web")
    db.upsert_session(con, src, Session(
        source_kind="claude_web", native_id="c1", title="t",
        started_at=1771243200000, raw_path="x", raw_hash="c1",
        messages=[Message(native_id="m1", role="assistant", created_at=1771243200000,
                          parts=[Part(kind="text", seq=0, text="hi")])]))
    con.commit()

    row = next(m for m in metrics.models(con) if m["model"] == "Claude.ai")
    assert row["model_recorded"] is False and row["sessions"] == 1


def test_recorded_models_are_unaffected(con):
    rows = {m["model"]: m for m in metrics.models(con)}
    assert rows["claude-opus-5"]["model_recorded"] is True
    assert rows["claude-opus-5"]["usd"] == pytest.approx(36.75)
    assert rows["copilot/gpt-5-mini"]["model_recorded"] is True


def test_tool_metrics(con):
    tools = metrics.tools(con)
    assert tools[0]["name"] == "Bash"
    assert tools[0]["calls"] == 1
    outcomes = metrics.tool_outcomes(con)
    assert outcomes["total"] == 1 and outcomes["failed"] == 1


def test_heatmap_is_monday_first(con):
    heat = metrics.activity_heatmap(con)
    assert heat["labels"][0] == "Mon"
    assert heat["peak"] > 0
    assert sum(heat["grid"][0]) > 0, "Monday messages landed on the wrong row"


def test_volume_buckets_by_month(con):
    vol = metrics.volume_by_month(con)
    assert vol["months"] == ["2026-02"]
    assert vol["series"]["claude_code"] == [2]


def test_everything_runs_on_an_empty_archive(tmp_path):
    empty = db.connect(tmp_path / "empty.db")
    data = metrics.everything(empty)
    assert data["overview"]["sessions"] == 0
    assert data["cost_total_billed"] == 0


# ------------------------------------------------------------------- charts

def test_charts_escape_hostile_labels():
    svg = charts.hbars([("<script>alert(1)</script>", 5, "5")])
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_charts_handle_no_data():
    assert "No dated sessions" in charts.stacked_bars([], {})
    assert "Nothing to plot" in charts.line_chart([], [])
    assert "Nothing recorded" in charts.hbars([])
    assert "No timestamped" in charts.heatmap([[0] * 24] * 7, ["Mon"], 0)


def test_series_slots_never_cycle():
    """A 9th series folds into the last slot rather than inventing a hue."""
    assert charts.series_var(0) == "var(--series-1)"
    assert charts.series_var(7) == "var(--series-8)"
    assert charts.series_var(20) == "var(--series-8)"


def test_every_mark_carries_a_title_for_hover():
    svg = charts.stacked_bars(["2026-01"], {"claude_code": [3]})
    assert "<title>" in svg


def test_participants_breakdown_separates_assistants_in_one_source(con):
    """Two assistants share the VS Code store; by_source cannot tell them apart."""
    panel = db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel")
    for native, who, label in (("v1", "copilot", "GitHub Copilot"),
                               ("v2", "copilot", "GitHub Copilot"),
                               ("v3", "remote-ssh", "Remote - SSH")):
        m = Message(native_id=f"{native}-0", role="user", created_at=1771243200000,
                    seq=0)
        m.parts.append(Part(kind="text", seq=0, text="a question", embed_eligible=True))
        db.upsert_session(con, panel, Session(
            source_kind="vscode_chat", native_id=native, title=native,
            started_at=1771243200000, raw_path="x", raw_hash=native, messages=[m],
            meta={"participant": who, "participant_label": label}))
    con.commit()

    rows = {r["label"]: r["sessions"] for r in metrics.participants(con)}
    assert rows == {"GitHub Copilot": 2, "Remote - SSH": 1}, \
        "assistants collapsed into one row — check the GROUP BY binds the expression"


def test_participants_is_empty_when_no_source_shares_a_panel(con):
    assert metrics.participants(con) == []


def test_surface_split_labels_the_editor_panel(con):
    db.source_id(con, "vscode_chat", "VS Code chat", "editor_panel")
    m = Message(native_id="v1-0", role="user", created_at=1771243200000, seq=0)
    m.parts.append(Part(kind="text", seq=0, text="a question", embed_eligible=True))
    db.upsert_session(con, db.source_id(con, "vscode_chat", "VS Code chat",
                                        "editor_panel"),
                      Session(source_kind="vscode_chat", native_id="v1", title="v1",
                              started_at=1771243200000, raw_path="x", raw_hash="v1",
                              messages=[m]))
    con.commit()

    split = metrics.surface_split(con)
    assert split["labels"]["editor_panel"] == "Editor panel"
    assert split["labels"]["cli"] == "Terminal"
    assert sum(split["series"]["editor_panel"]) == 1


# ------------------------------------------------------------- index health

def _record_index(connection, *, finished_at, chunks=0, vectors=0, with_vectors=1,
                  warnings=None):
    connection.execute(
        "INSERT INTO index_run(started_at,finished_at,fts_rows,chunks,vectors,"
        "model_tag,with_vectors,seconds,warnings) VALUES (?,?,?,?,?,?,?,?,?)",
        (finished_at - 1000, finished_at, 42, chunks, vectors, "pm-MiniLM-L12",
         with_vectors, 1.5, warnings))
    connection.commit()


def test_index_health_is_unrecorded_when_an_index_predates_the_log(con):
    """An archive indexed before builds were logged is unknown, not broken."""
    from llm_archive.search import fts

    fts.rebuild(con)                      # a real keyword index...
    con.execute("DROP TABLE index_run")   # ...with no record of when it was built
    health = metrics.index_health(con)
    assert health["state"] == "unrecorded"
    assert health["reasons"] == []


def test_index_health_reports_coverage_even_with_no_recorded_build(con):
    """Vector coverage is a fact about the tables; it does not need a build record."""
    from llm_archive.search import fts

    # the fixture's turns are all under the 40-char embedding floor
    con.execute("""INSERT INTO part(message_id,seq,kind,text,embed_eligible)
                   SELECT id, 9, 'text', ?, 1 FROM message LIMIT 1""", ("x" * 100,))
    fts.rebuild(con)
    con.execute("DROP TABLE index_run")
    health = metrics.index_health(con)
    assert health["unindexed_parts"] == 1, "nothing is chunked, so nothing is covered"
    assert "not in the vector index" in " ".join(health["notes"])


def test_index_health_reports_never_built(con):
    assert metrics.index_health(con)["state"] == "never"


def test_index_health_is_fresh_when_nothing_changed_since_the_build(con):
    latest = con.execute("SELECT MAX(ingested_at) FROM session").fetchone()[0]
    _record_index(con, finished_at=latest + 60_000)
    health = metrics.index_health(con)
    assert health["state"] == "fresh"
    assert health["sessions_since"] == 0
    assert health["last"]["fts_rows"] == 42


def test_index_health_goes_stale_when_a_session_is_ingested_after_it(con):
    """The exact failure that made search silently wrong after `ingest --force`."""
    earliest = con.execute("SELECT MIN(ingested_at) FROM session").fetchone()[0]
    _record_index(con, finished_at=earliest - 60_000)
    health = metrics.index_health(con)
    assert health["state"] == "stale"
    assert health["sessions_since"] == 2
    assert "ingested since" in " ".join(health["reasons"])


def test_index_health_counts_embeddable_parts_missing_from_the_vector_index(con):
    latest = con.execute("SELECT MAX(ingested_at) FROM session").fetchone()[0]
    _record_index(con, finished_at=latest + 60_000)
    # long enough to be embeddable, on the active path, but no chunk row
    con.execute("""INSERT INTO part(message_id,seq,kind,text,embed_eligible)
                   SELECT id, 9, 'text', ?, 1 FROM message LIMIT 1""",
                ("x" * 100,))
    con.commit()
    health = metrics.index_health(con)
    assert health["state"] == "stale"
    assert health["unindexed_parts"] >= 1
    assert "vector index" in " ".join(health["reasons"])


def test_a_keyword_only_build_is_a_note_not_staleness(con):
    """--no-vectors is a deliberate choice. Its keyword index is exactly current, and
    crying "out of date" at it would train you to ignore the warning that matters."""
    latest = con.execute("SELECT MAX(ingested_at) FROM session").fetchone()[0]
    _record_index(con, finished_at=latest + 60_000, with_vectors=0)
    health = metrics.index_health(con)
    assert health["state"] == "fresh"
    assert health["reasons"] == []
    assert "--no-vectors" in " ".join(health["notes"])


# ------------------------------------------------- turns vs tool steps (v8)

def test_a_tool_call_and_its_result_are_messages_but_not_turns(con):
    """The fixture's s1 is one prompt plus a Bash call and its output."""
    ov = metrics.overview(con)
    assert ov["messages"] == 4
    assert ov["turns"] == 2, "only the two text messages said anything"
    assert ov["tool_steps"] == 2
    assert ov["messages"] == ov["turns"] + ov["tool_steps"]


def test_turn_count_is_stored_per_session_at_ingest(con):
    rows = dict(con.execute("SELECT native_id, turn_count FROM session").fetchall())
    assert rows["s1"] == 1, "3 messages, 1 turn"
    assert rows["s2"] == 1
    totals = con.execute("SELECT SUM(msg_count) m, SUM(turn_count) t FROM session"
                         ).fetchone()
    assert (totals["m"], totals["t"]) == (4, 2)


def test_a_message_mixing_text_and_a_tool_call_is_one_turn(tmp_path):
    """Sources that fold tool_use into the assistant reply must not lose the turn."""
    connection = db.connect(tmp_path / "b.db")
    src = db.source_id(connection, "t3chat", "T3 Chat", "web")
    msg = Message(native_id="m1", role="assistant", created_at=1771243200000)
    msg.parts.append(Part(kind="text", seq=0, text="here you go"))
    msg.parts.append(Part(kind="tool_use", seq=1, text="search", tool_name="web"))
    db.upsert_session(connection, src, Session(
        source_kind="t3chat", native_id="w1", started_at=1771243200000,
        raw_path="x", raw_hash="h", messages=[msg]))
    connection.commit()

    assert msg.is_turn is True
    assert metrics.overview(connection)["turns"] == 1


def test_an_image_only_message_is_a_turn(tmp_path):
    """A pasted screenshot said something; only tool plumbing does not."""
    shown = Message(native_id="m1", role="user", created_at=1)
    shown.parts.append(Part(kind="image", seq=0, text=None, bytes=99))
    silent = Message(native_id="m2", role="user", created_at=2)
    silent.parts.append(Part(kind="tool_result", seq=0, text="exit 0"))
    assert (shown.is_turn, silent.is_turn) == (True, False)

    sess = Session(source_kind="claude_code", native_id="s", started_at=1,
                   raw_path="x", raw_hash="h", messages=[shown, silent])
    assert (sess.msg_count, sess.turn_count) == (2, 1)


def test_abandoned_messages_are_left_out_of_turn_count():
    dead = Message(native_id="m1", role="user", created_at=1, on_active_path=False)
    dead.parts.append(Part(kind="text", seq=0, text="a rewound prompt"))
    live = Message(native_id="m2", role="user", created_at=2)
    live.parts.append(Part(kind="text", seq=0, text="the prompt that stuck"))
    sess = Session(source_kind="claude_code", native_id="s", started_at=1,
                   raw_path="x", raw_hash="h", messages=[dead, live])
    assert (sess.msg_count, sess.turn_count) == (1, 1)


def test_session_shape_buckets_on_turns_not_messages(con):
    """s1 is 3 messages but 1 turn; on messages it landed a bucket too high."""
    shape = metrics.session_shape(con)
    counts = dict(zip(shape["labels"], shape["values"]))
    assert counts["1-2"] == 2, "both sessions are one-turn sessions"
    assert counts["3-5"] == 0
    assert shape["labels"][0] == "0", "pure tool traffic needs its own bucket"


def test_by_source_reports_turns_alongside_messages(con):
    row = metrics.by_source(con)[0]
    assert (row["messages"], row["turns"]) == (4, 2)
