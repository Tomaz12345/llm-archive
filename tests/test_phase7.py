"""Phase 7: export freshness, scheduled sync, and the opt-in redaction pass.

The redaction tests carry the most weight. This pass rewrites `part.text` in place and
there is no undo inside the database — the plaintext survives only in the source files
— so a rule that is too greedy destroys archive content permanently, and one that is
too narrow gives a false all-clear about a leaked key. Both failures are silent, which
is why the false-positive cases here are as explicit as the true ones.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from llm_archive.core import db, freshness, redact, schedule, sync
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import Message, Part, Session

DAY = freshness.DAY_MS


# ------------------------------------------------------------------ helpers

def _archive(tmp_path: Path) -> sqlite3.Connection:
    return db.connect(tmp_path / "archive.db")


def _session(native_id: str, started: int, raw_path: str, text: str = "hello",
             kind: str = "text") -> Session:
    return Session(
        source_kind="x", native_id=native_id, started_at=started,
        raw_path=raw_path, raw_hash=f"h-{native_id}",
        messages=[Message(native_id=f"m-{native_id}", role="user", created_at=started,
                          parts=[Part(kind=kind, seq=0, text=text,
                                      embed_eligible=True)])])


def _add(con, kind: str, label: str, surface: str, native_id: str,
         started: int, raw_path: str, **kw) -> None:
    src = db.source_id(con, kind, label, surface)
    db.upsert_session(con, src, _session(native_id, started, raw_path, **kw))
    con.commit()


# ------------------------------------------------------------- freshness

def test_every_adapter_has_an_export_route(tmp_path):
    """A source with no entry vanishes from the one report meant to notice it."""
    from llm_archive.core import ingest

    kinds = {a.kind for a in ingest.build_adapters(BlobStore(tmp_path / "b"))}
    assert kinds <= set(freshness.ROUTES), \
        f"no export route for: {sorted(kinds - set(freshness.ROUTES))}"


def test_bulk_source_with_no_rows_is_reported_missing(tmp_path):
    """The case a query over `session` cannot produce, and the whole point of ROUTES.

    ChatGPT has no adapter yet, so it has no sessions, so it appears in no join — and
    "you have never exported ChatGPT" is exactly the thing this report exists to say.
    """
    con = _archive(tmp_path)
    report = freshness.report(con)
    rows = {r["kind"]: r for r in report}
    assert rows["chatgpt"]["state"] == "missing"
    assert rows["chatgpt"]["sessions"] == 0
    # and `missing` sorts ahead of everything merely stale
    assert report[0]["state"] == "missing"
    assert all(r["state"] == "missing"
               for r in report if r["mode"] == freshness.BULK)


def test_overdue_is_measured_from_the_drop_file_not_the_newest_chat(tmp_path):
    """The measurement §7 turns on: mtime of the export, not the age of its contents."""
    con = _archive(tmp_path)
    now = int(time.time() * 1000)
    drop = tmp_path / "conversations.json"
    drop.write_text("{}", encoding="utf-8")

    # An export taken 90 days ago whose newest chat is from yesterday.
    old = (time.time() - 90 * 86400)
    import os
    os.utime(drop, (old, old))
    _add(con, "t3chat", "T3 Chat", "web", "s1", now - DAY, str(drop))

    row = {r["kind"]: r for r in freshness.report(con, interval_days=30)}["t3chat"]
    assert row["state"] == "overdue"
    assert row["export_age_days"] == pytest.approx(90, abs=1)
    assert row["exported_clock"] == "file"


def test_fresh_export_of_an_idle_account_is_not_a_warning(tmp_path):
    """A service you stopped using must not read as an export you neglected."""
    con = _archive(tmp_path)
    now = int(time.time() * 1000)
    drop = tmp_path / "conversations.json"
    drop.write_text("{}", encoding="utf-8")            # written just now

    _add(con, "deepseek", "DeepSeek", "web", "s1", now - 200 * DAY, str(drop))

    row = {r["kind"]: r for r in freshness.report(con, interval_days=30)}["deepseek"]
    assert row["state"] == "current"
    assert "idle" in row and "account is idle" in row["idle"]


def test_missing_drop_falls_back_to_ingest_time(tmp_path):
    con = _archive(tmp_path)
    now = int(time.time() * 1000)
    _add(con, "grok", "Grok", "web", "s1", now, str(tmp_path / "gone.zip"))

    row = {r["kind"]: r for r in freshness.report(con)}["grok"]
    assert row["exported_clock"] == "ingest"
    assert row["state"] == "current"


def test_per_chat_and_live_sources_are_never_overdue(tmp_path):
    """OpenRouter has no action that makes it current; nagging about it is noise."""
    con = _archive(tmp_path)
    ancient = int(time.time() * 1000) - 900 * DAY
    _add(con, "openrouter", "OpenRouter", "web", "s1", ancient,
         str(tmp_path / "nope.json"))
    _add(con, "claude_code", "Claude Code", "cli", "s2", ancient,
         str(tmp_path / "nope.jsonl"))

    rows = {r["kind"]: r for r in freshness.report(con)}
    assert rows["openrouter"]["state"] == "manual"
    assert rows["claude_code"]["state"] == "live"
    assert not any(r["mode"] != freshness.BULK
                   for r in freshness.summary(freshness.report(con))["needs_action"])


def test_retention_is_reported_separately_from_notes():
    """Gemini's rolling delete is true of a perfectly current export, so it cannot
    live in the field that only prints when a source needs attention."""
    assert freshness.ROUTES["gemini"].retention
    assert freshness.ROUTES["gemini"].note is None


# -------------------------------------------------------------- scheduling

def test_task_xml_carries_the_three_settings_that_make_it_run_on_a_laptop():
    plan = schedule.make_plan(at="03:30")
    xml = schedule.build_xml(plan)
    # asleep at 03:30 -> run on next wake, rather than skip
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml
    # the default is true, which on a laptop means "never runs"
    assert "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>" in xml
    # two embed passes writing chunks at once is the one self-inflicted corruption
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "T03:30:00" in xml


def test_task_xml_has_no_encoding_declaration():
    """`Register-ScheduledTask -Xml` takes a decoded string and rejects a declaration
    claiming UTF-16 on it."""
    assert not schedule.build_xml(schedule.make_plan()).lstrip().startswith("<?xml")


def test_paths_with_spaces_are_quoted_in_the_action():
    plan = schedule.make_plan(data_dir=Path("C:/a b/data"))
    assert '"' in plan.arguments and "a b" in plan.arguments
    assert "--data-dir" in plan.arguments


@pytest.mark.parametrize("bad", ["9pm", "25:00", "3:70", "", "03-30"])
def test_bad_start_time_is_rejected_before_anything_is_registered(bad):
    with pytest.raises(ValueError):
        schedule.build_xml(schedule.make_plan(at=bad))


def test_no_vectors_reaches_the_scheduled_command():
    plan = schedule.make_plan(with_vectors=False)
    assert "--no-vectors" in plan.arguments


# -------------------------------------------------------------------- sync

def test_lock_is_exclusive(tmp_path):
    path = tmp_path / ".sync.lock"
    with sync.Lock(path):
        with pytest.raises(sync.Busy):
            with sync.Lock(path):
                pass


def test_lock_is_released_on_the_way_out(tmp_path):
    path = tmp_path / ".sync.lock"
    with sync.Lock(path):
        pass
    with sync.Lock(path):
        assert path.exists()


def test_a_lock_whose_holder_is_gone_is_taken_immediately(tmp_path):
    """Found by the first real sync: the run was killed mid-embed, never reached
    __exit__, and left a lock that would have blocked the archive for two hours."""
    import os
    path = tmp_path / ".sync.lock"
    path.write_text(f"{2**22 + 7} {int(time.time())}")   # a pid that is not running

    with sync.Lock(path):
        assert path.read_text().startswith(str(os.getpid()))


def test_a_live_holder_still_wins(tmp_path):
    """The liveness check must not become a way to steal a lock that is in use."""
    import os
    path = tmp_path / ".sync.lock"
    path.write_text(f"{os.getpid()} {int(time.time())}")
    with pytest.raises(sync.Busy):
        with sync.Lock(path):
            pass


def test_this_process_is_reported_alive():
    import os
    assert sync._holder_alive(os.getpid()) is True
    assert sync._holder_alive(0) is False


def test_a_lock_older_than_the_task_time_limit_is_stolen(tmp_path):
    """Past the scheduler's own ExecutionTimeLimit the previous holder is gone or hung;
    refusing forever would mean the archive silently stops updating."""
    import os
    path = tmp_path / ".sync.lock"
    path.write_text("9999999 0")
    old = time.time() - sync.STALE_AFTER_S - 60
    os.utime(path, (old, old))

    with sync.Lock(path):
        assert path.read_text().startswith(str(os.getpid()))


def test_sync_reindexes_when_the_index_is_broken_but_nothing_changed(tmp_path,
                                                                     monkeypatch):
    """The state the first real sync left behind: a build killed part-way commits its
    `DELETE FROM chunk` and dies before the re-insert. Nothing is then ingested on the
    next run, so a "changed?" test alone would leave the archive unsearchable forever.
    """
    from llm_archive.core import ingest as core_ingest
    from llm_archive.search import index as search_index

    con = _archive(tmp_path)
    _add(con, "claude_code", "Claude Code", "cli", "s1", 1_700_000_000_000,
         str(tmp_path / "a.jsonl"), text="a" * 200)      # embeddable: >= 40 chars
    con.close()

    monkeypatch.setattr(core_ingest, "build_adapters", lambda *a, **k: [])
    built: list[bool] = []

    class _Res:
        fts_rows = chunks = vectors = 0
        model_tag = "t"
        skipped_vectors = False
        warnings: list[str] = []
        seconds = 0.0

    monkeypatch.setattr(search_index, "build",
                        lambda *a, **k: (built.append(True), _Res())[1])

    res = sync.run(tmp_path, log=lambda _: None)
    assert built, "an empty vector index must be repaired even with nothing ingested"
    assert res.new == 0


def test_sync_leaves_a_current_index_alone(tmp_path, monkeypatch):
    """The other half: re-embedding what nothing invalidated is minutes of CPU for an
    identical result, on a laptop, every night."""
    from llm_archive.core import ingest as core_ingest
    from llm_archive.search import index as search_index

    _archive(tmp_path).close()          # no sessions, so nothing is unindexed
    monkeypatch.setattr(core_ingest, "build_adapters", lambda *a, **k: [])
    monkeypatch.setattr(search_index, "build",
                        lambda *a, **k: pytest.fail("must not rebuild"))

    sync.run(tmp_path, log=lambda _: None)


def test_logger_truncates_from_the_front(tmp_path):
    path = tmp_path / "sync.log"
    path.write_bytes(b"x" * (sync.LOG_MAX_BYTES + 1000))
    write = sync.logger(path)
    write("after truncation")
    body = path.read_text(encoding="utf-8", errors="replace")
    assert body.startswith("[...truncated...]")
    assert "after truncation" in body
    assert len(body) < sync.LOG_MAX_BYTES


def test_heartbeat_writes_a_few_lines_not_thousands():
    """A scheduled embed pass is ~20 minutes with no console. Silence for that long is
    indistinguishable from a hang, and a line per chunk is a log nobody reads."""
    lines: list[str] = []
    progress = sync._heartbeat(lines.append, every_pct=25)
    for done in range(0, 1001):
        progress(done, 1000)
    assert len(lines) == 5
    assert "100%" in lines[-1]


def test_heartbeat_survives_an_empty_corpus():
    sync._heartbeat(lambda _: None)(0, 0)      # must not divide by zero


def test_logger_survives_an_unwritable_path(tmp_path):
    """A scheduled run must not die because its log file is locked."""
    write = sync.logger(tmp_path / "no" / "such" / "dir" / "sync.log")
    write("still fine")          # must not raise


# ---------------------------------------------------------------- redaction

REAL = [
    ("anthropic_key", "sk-ant-api03-" + "A" * 80),
    ("openai_key", "sk-" + "a1B2c3D4" * 6),
    ("openrouter_key", "sk-or-v1-" + "0123456789abcdef" * 4),
    ("github_token", "ghp_" + "b" * 36),
    ("github_pat", "github_pat_" + "c" * 60),
    ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
    ("google_api_key", "AIza" + "D" * 35),
    # Filler rather than a realistic token: the literal tripped GitHub push
    # protection. The rule is xox[abprse]-[A-Za-z0-9-]{10,}, so this exercises it
    # identically.
    ("slack_token", "xoxb-" + "s" * 24),
    ("stripe_key", "sk_live_" + "e" * 24),
    ("huggingface_token", "hf_" + "f" * 34),
    ("groq_key", "gsk_" + "g" * 48),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            + "h" * 40),
]


@pytest.mark.parametrize("rule_name,secret", REAL)
def test_each_rule_matches_its_own_token(rule_name, secret):
    text = f"here is the value: {secret} and nothing else"
    out, applied = redact.redact_text(text, redact.active_rules())
    assert applied, f"{rule_name} did not match"
    assert applied[0][0] == rule_name
    assert secret not in out


def test_a_prefix_in_the_middle_of_a_word_is_not_a_key():
    """The false positive the first real dry run produced: `sk-` out of `...Task-`."""
    text = ("--backgroundupdate MozillaBackgroundTask-308046B0AF4A39CB-"
            "backgroundupdate=fj5549no")
    _, applied = redact.redact_text(text, redact.active_rules())
    assert applied == []


def test_a_pair_of_key_markers_with_no_body_is_prose_not_a_key():
    text = ("ssh-keygen -y -f id_rsa prints the public half; the file starts with "
            "-----BEGIN PRIVATE KEY----- and ends -----END PRIVATE KEY-----")
    _, applied = redact.redact_text(text, redact.active_rules())
    assert applied == []


def test_a_real_key_block_with_a_body_is_matched():
    text = ("-----BEGIN RSA PRIVATE KEY-----\n" + "MIIEow" * 40
            + "\n-----END RSA PRIVATE KEY-----")
    out, applied = redact.redact_text(text, redact.active_rules())
    assert [name for name, _ in applied] == ["private_key"]
    assert "MIIEow" not in out


def test_wide_rules_are_off_by_default():
    text = 'DATABASE_PASSWORD="hunter2-hunter2-hunter2"'
    assert redact.redact_text(text, redact.active_rules())[1] == []
    assert redact.redact_text(text, redact.active_rules(wide=True))[1]


def test_url_credentials_keeps_the_host_and_takes_only_the_password():
    text = "postgres://appuser:s3cr3t-pa55word@db.internal:5432/prod"
    out, applied = redact.redact_text(text, redact.active_rules(wide=True))
    assert applied[0][0] == "url_credentials"
    assert "db.internal:5432/prod" in out and "appuser" in out
    assert "s3cr3t-pa55word" not in out


def test_the_placeholder_names_the_rule_and_fingerprints_the_secret():
    secret = "sk-ant-api03-" + "Z" * 80
    out, _ = redact.redact_text(f"key={secret}", redact.active_rules())
    assert out == f"key=[redacted:anthropic_key:{redact.fingerprint(secret)}]"


def test_the_same_secret_fingerprints_the_same_and_a_different_one_does_not():
    a = "ghp_" + "a" * 36
    b = "ghp_" + "b" * 36
    assert redact.fingerprint(a) == redact.fingerprint(a)
    assert redact.fingerprint(a) != redact.fingerprint(b)


def test_redaction_is_idempotent():
    """Re-running must be a no-op — the placeholder's own hex tail is close enough to
    the shapes some rules look for that a second pass could eat its own output."""
    text = "token " + "ghp_" + "a" * 36
    once, _ = redact.redact_text(text, redact.active_rules(wide=True))
    twice, applied = redact.redact_text(once, redact.active_rules(wide=True))
    assert twice == once and applied == []


def test_a_specific_rule_wins_over_a_looser_one():
    """`sk-ant-...` is an Anthropic key, not an OpenAI key that starts oddly."""
    secret = "sk-ant-api03-" + "Q" * 80
    _, applied = redact.redact_text(secret, redact.active_rules())
    assert [name for name, _ in applied] == ["anthropic_key"]


def test_apply_rewrites_the_row_and_records_what_it_did(tmp_path):
    con = _archive(tmp_path)
    secret = "ghp_" + "k" * 36
    _add(con, "claude_code", "Claude Code", "cli", "s1", 1_700_000_000_000,
         str(tmp_path / "a.jsonl"), text=f"the token is {secret}")

    res = redact.apply(con)
    assert (res.parts_changed, res.secrets) == (1, 1)

    stored = con.execute("SELECT text, redacted FROM part").fetchone()
    assert secret not in stored["text"]
    assert stored["redacted"] == 1

    row = con.execute("SELECT rule, fingerprint, hits FROM redaction").fetchone()
    assert row["rule"] == "github_token"
    assert row["fingerprint"] == redact.fingerprint(secret)


def test_dry_run_changes_nothing(tmp_path):
    con = _archive(tmp_path)
    secret = "ghp_" + "m" * 36
    _add(con, "claude_code", "Claude Code", "cli", "s1", 1_700_000_000_000,
         str(tmp_path / "a.jsonl"), text=f"token {secret}")

    res = redact.apply(con, dry_run=True)
    assert res.secrets == 1
    assert secret in con.execute("SELECT text FROM part").fetchone()["text"]
    assert con.execute("SELECT COUNT(*) FROM redaction").fetchone()[0] == 0


def test_one_key_repeated_in_a_dump_is_one_leak_not_forty(tmp_path):
    con = _archive(tmp_path)
    secret = "ghp_" + "n" * 36
    _add(con, "claude_code", "Claude Code", "cli", "s1", 1_700_000_000_000,
         str(tmp_path / "a.jsonl"), text="\n".join(f"VAR{i}={secret}" for i in range(40)))

    redact.apply(con)
    rows = con.execute("SELECT fingerprint, hits FROM redaction").fetchall()
    assert len(rows) == 1
    assert rows[0]["hits"] == 40


def test_the_setting_survives_and_drives_ingest(tmp_path):
    """The point of storing it: ingest re-reads the plaintext source, so a one-off
    redaction is undone by the next run."""
    con = _archive(tmp_path)
    assert redact.is_enabled(con) is False
    redact.set_enabled(con, True, wide=True)
    assert redact.is_enabled(con) and redact.enabled_wide(con)
    redact.set_enabled(con, False)
    assert redact.is_enabled(con) is False


def test_reingesting_a_session_re_redacts_it(tmp_path):
    """A re-upserted session's parts come back at redacted = 0, which is what makes
    `only_new` pick them up again rather than skip them as already clean."""
    con = _archive(tmp_path)
    secret = "ghp_" + "p" * 36
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    sess = _session("s1", 1_700_000_000_000, str(tmp_path / "a.jsonl"),
                    text=f"token {secret}")
    db.upsert_session(con, src, sess)
    con.commit()

    assert redact.apply(con, only_new=True).secrets == 1

    db.upsert_session(con, src, sess)       # ingest re-reads the untouched source file
    con.commit()
    assert secret in con.execute("SELECT text FROM part").fetchone()["text"]

    assert redact.apply(con, only_new=True).secrets == 1
    assert secret not in con.execute("SELECT text FROM part").fetchone()["text"]


def test_summary_counts_distinct_secrets_not_rows(tmp_path):
    con = _archive(tmp_path)
    same = "ghp_" + "q" * 36
    other = "ghp_" + "r" * 36
    _add(con, "claude_code", "Claude Code", "cli", "s1", 1_700_000_000_000,
         str(tmp_path / "a.jsonl"), text=f"{same} and {other}")
    _add(con, "codex", "Codex", "cli", "s2", 1_700_000_000_000,
         str(tmp_path / "b.jsonl"), text=f"{same} again")

    redact.apply(con)
    info = redact.summary(con)
    assert info["rows"] == 3 and info["distinct"] == 2 and info["sessions"] == 2


def test_blobs_are_scanned_but_never_rewritten(tmp_path):
    blob_dir = tmp_path / "blobs" / "ab"
    blob_dir.mkdir(parents=True)
    target = blob_dir / "abcdef"
    secret = "ghp_" + "s" * 36
    target.write_text(f"AWS_TOKEN={secret}\n", encoding="utf-8")
    before = target.read_bytes()

    findings = redact.scan_blobs(tmp_path / "blobs")
    assert findings and findings[0]["by_rule"] == {"github_token": 1}
    assert target.read_bytes() == before, "the blob store must not be edited in place"


def test_an_empty_archive_reports_no_redactions(tmp_path):
    assert redact.summary(_archive(tmp_path))["rows"] == 0


# ------------------------------------------------------------------ schema

def test_v5_archive_migrates_to_v6(tmp_path):
    """The ordering bug this caught: `CREATE INDEX ... ON part(redacted)` sitting in
    the fresh-database DDL runs *before* the ALTER that adds the column, so every
    command on an existing archive died with "no such column: redacted"."""
    path = tmp_path / "old.db"
    con = db.connect(path)
    con.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
    con.execute("DROP INDEX IF EXISTS idx_part_redacted")
    con.execute("DROP TABLE IF EXISTS redaction")
    con.commit()
    con.close()

    con = db.connect(path)            # must not raise
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'"
                       ).fetchone()["value"] == str(db.SCHEMA_VERSION)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(part)")}
    assert "redacted" in cols
    assert con.execute("SELECT COUNT(*) FROM redaction").fetchone()[0] == 0


def test_a_secret_in_a_tool_payload_is_redacted_too(tmp_path):
    """A Bash payload is the single most leak-prone content in the archive -- it is
    where `export AWS_SECRET=...` actually lives -- and `command.text` is printed by
    the CLI, the web UI and the MCP server. Scanning only `part.text` would leave the
    secret in the column the derived tables are computed from."""
    from llm_archive.core import db, redact
    from llm_archive.core.models import Message, Part, Session

    con = db.connect(tmp_path / "a.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    secret = "ghp_" + "a" * 36
    part = Part(kind="tool_use", seq=0, text="command: deploy.sh", tool_name="Bash")
    part.tool_input = '{"command": "GITHUB_TOKEN=' + secret + ' ./deploy.sh"}'
    msg = Message(native_id="m1", role="assistant", created_at=1, seq=0)
    msg.parts.append(part)
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", started_at=1,
        raw_path="x", raw_hash="h", messages=[msg]))

    stored = con.execute("SELECT tool_input FROM part").fetchone()[0]
    assert secret in stored, "fixture is wrong; the payload never reached the column"

    result = redact.apply(con)
    assert result.secrets >= 1

    after = con.execute("SELECT text, tool_input, redacted FROM part").fetchone()
    assert secret not in after["tool_input"]
    assert "[redacted:" in after["tool_input"]
    assert after["redacted"] >= 1
    assert after["text"] == "command: deploy.sh"   # the summary is untouched


def test_redaction_leaves_a_part_with_no_payload_alone(tmp_path):
    """Most parts are plain text with tool_input NULL; the wider scan must not write
    an empty string over them."""
    from llm_archive.core import db, redact
    from llm_archive.core.models import Message, Part, Session

    con = db.connect(tmp_path / "b.db")
    src = db.source_id(con, "claude_code", "Claude Code", "cli")
    secret = "ghp_" + "b" * 36
    msg = Message(native_id="m1", role="user", created_at=1, seq=0)
    msg.parts.append(Part(kind="text", seq=0, text=f"my key is {secret}"))
    db.upsert_session(con, src, Session(
        source_kind="claude_code", native_id="s1", started_at=1,
        raw_path="x", raw_hash="h", messages=[msg]))

    redact.apply(con)
    row = con.execute("SELECT text, tool_input FROM part").fetchone()
    assert secret not in row["text"]
    assert row["tool_input"] is None
