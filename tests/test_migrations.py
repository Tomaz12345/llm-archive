"""Schema migrations run against databases that already hold data.

The archive is not rebuilt from scratch — re-ingesting 60k messages costs minutes and a
re-index costs more — so every migration has to survive on a live file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llm_archive.core import db


def make_v3(path: Path) -> None:
    """A database as it looked before 'editor_panel' existed."""
    con = sqlite3.connect(path)
    schema = db.SCHEMA.replace(
        "CHECK (surface IN ('cli','web','editor_panel'))",
        "CHECK (surface IN ('cli','web'))")
    # index_run arrived in v5; a v3 database has no such table
    start = schema.index("CREATE TABLE IF NOT EXISTS index_run")
    schema = schema[:start] + schema[schema.index(";", start) + 1:]
    con.executescript(schema)
    con.execute("INSERT INTO meta(key,value) VALUES ('schema_version','3')")
    con.executemany(
        "INSERT INTO source(id,kind,label,surface) VALUES (?,?,?,?)",
        [(1, "claude_code", "Claude Code", "cli"),
         (2, "vscode_chat", "VS Code chat", "cli"),
         (3, "claude_web", "Claude.ai", "web")])
    con.execute("""INSERT INTO session(id,source_id,native_id,started_at,
                                       raw_path,raw_hash,ingested_at)
                   VALUES (10,2,'vs1',1,'x','h',1)""")
    con.commit()
    con.close()


def test_v4_relabels_vscode_chat_as_an_editor_panel(tmp_path):
    path = tmp_path / "a.db"
    make_v3(path)
    con = db.connect(path)

    surfaces = dict(con.execute("SELECT kind, surface FROM source"))
    assert surfaces["vscode_chat"] == "editor_panel"
    assert surfaces["claude_code"] == "cli", "other sources must not move"
    assert surfaces["claude_web"] == "web"


def test_v4_rebuild_keeps_sessions_and_their_foreign_keys(tmp_path):
    """`source` is rebuilt to widen a CHECK; `session` points at it."""
    path = tmp_path / "a.db"
    make_v3(path)
    con = db.connect(path)

    row = con.execute("""SELECT src.kind FROM session s
                         JOIN source src ON src.id = s.source_id
                         WHERE s.id = 10""").fetchone()
    assert row["kind"] == "vscode_chat", "session lost its source in the rebuild"
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1, "keys left off"


def test_v4_accepts_the_new_surface_and_still_rejects_nonsense(tmp_path):
    path = tmp_path / "a.db"
    make_v3(path)
    con = db.connect(path)

    db.source_id(con, "other_panel", "Other panel", "editor_panel")
    assert con.execute("SELECT surface FROM source WHERE kind='other_panel'"
                       ).fetchone()[0] == "editor_panel"
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO source(kind,label,surface) "
                    "VALUES ('x','X','telepathy')")


def test_migrating_twice_is_a_no_op(tmp_path):
    path = tmp_path / "a.db"
    make_v3(path)
    db.connect(path).close()
    con = db.connect(path)
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'"
                       ).fetchone()[0] == str(db.SCHEMA_VERSION)
    assert dict(con.execute("SELECT kind, surface FROM source"))["vscode_chat"] \
        == "editor_panel"


def test_v5_adds_the_index_run_table(tmp_path):
    """Without it nothing can date the search index, so nothing can call it stale."""
    path = tmp_path / "a.db"
    make_v3(path)
    assert sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='index_run'").fetchone()[0] == 0

    con = db.connect(path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(index_run)")}
    assert {"started_at", "finished_at", "fts_rows", "chunks", "vectors",
            "model_tag", "with_vectors", "seconds", "warnings"} <= cols
    assert con.execute("SELECT COUNT(*) FROM index_run").fetchone()[0] == 0


def test_an_archive_indexed_before_v5_is_unrecorded_not_never_built(tmp_path):
    """The archive that triggered this feature already had a full index and no record
    of it. Reporting "never built — search will return nothing" would be a flat lie."""
    from llm_archive.stats import metrics

    path = tmp_path / "a.db"
    make_v3(path)
    con = db.connect(path)
    con.execute("""INSERT INTO message(id,session_id,seq,role,created_at)
                   VALUES (1,10,0,'user',1)""")
    con.execute("""INSERT INTO part(id,message_id,seq,kind,text,embed_eligible)
                   VALUES (1,1,0,'text',?,1)""", ("x" * 100,))
    con.execute("""INSERT INTO chunk(part_id,message_id,session_id,seq,text,vec_row,
                                     model_tag) VALUES (1,1,10,0,'x',0,'m')""")
    con.commit()
    assert metrics.index_health(con)["state"] == "unrecorded"


def test_an_empty_archive_really_has_never_been_indexed(tmp_path):
    from llm_archive.stats import metrics

    path = tmp_path / "a.db"
    make_v3(path)
    assert metrics.index_health(db.connect(path))["state"] == "never"


def make_v7(path: Path) -> None:
    """A database as it looked before turns were told apart from tool steps.

    Built by stripping the two v8 columns back out of SCHEMA, so the ALTERs in
    MIGRATIONS[8] really have to run rather than being swallowed as duplicates.
    """
    schema = db.SCHEMA
    for line in ("  turn_count        INTEGER NOT NULL DEFAULT 0,\n",
                 "  is_turn          INTEGER NOT NULL DEFAULT 1,\n"):
        assert line in schema, "SCHEMA changed; this fixture no longer strips v8"
        schema = schema.replace(line, "")
    schema = schema.replace(
        "  -- messages that carry said-or-shown content; <= msg_count, "
        "the rest are tool steps\n", "")
    schema = schema.replace(
        "  -- a conversational turn rather than a bare tool call/result; "
        "see models.TURN_KINDS\n", "")

    con = sqlite3.connect(path)
    con.executescript(schema)
    con.execute("INSERT INTO meta(key,value) VALUES ('schema_version','7')")
    con.execute("INSERT INTO source(id,kind,label,surface) "
                "VALUES (1,'claude_code','Claude Code','cli')")
    con.execute("""INSERT INTO session(id,source_id,native_id,started_at,msg_count,
                                       raw_path,raw_hash,ingested_at)
                   VALUES (1,1,'s1',1,4,'x','h',1)""")
    # one prompt, one reply, a tool call and its result, plus an abandoned prompt
    rows = [(1, "user", 1), (2, "assistant", 1), (3, "assistant", 1),
            (4, "user", 1), (5, "user", 0)]
    con.executemany("""INSERT INTO message(id,session_id,native_id,seq,on_active_path,
                                           role,created_at)
                       VALUES (?,1,'m'||?,?,?,?,1)""",
                    [(i, i, i, active, role) for i, role, active in rows])
    con.executemany("INSERT INTO part(message_id,seq,kind,text) VALUES (?,0,?,?)",
                    [(1, "text", "hello"), (2, "text", "hi"),
                     (3, "tool_use", "ls"), (4, "tool_result", "out"),
                     (5, "text", "rewound")])
    con.commit()
    con.close()


def test_v8_backfills_is_turn_from_the_parts_already_stored(tmp_path):
    path = tmp_path / "a.db"
    make_v7(path)
    con = db.connect(path)

    turns = dict(con.execute("SELECT native_id, is_turn FROM message").fetchall())
    assert turns == {"m1": 1, "m2": 1, "m3": 0, "m4": 0, "m5": 1}


def test_v8_backfills_turn_count_without_touching_msg_count(tmp_path):
    """Re-ingesting the archive to get this column would cost minutes and a re-index."""
    path = tmp_path / "a.db"
    make_v7(path)
    con = db.connect(path)

    row = con.execute("SELECT msg_count, turn_count FROM session").fetchone()
    assert row["msg_count"] == 4, "the volume figure must survive the migration"
    assert row["turn_count"] == 2, "the abandoned prompt is not a turn that happened"


def test_v8_backfill_is_safe_to_re_run(tmp_path):
    path = tmp_path / "a.db"
    make_v7(path)
    db.connect(path).close()
    con = db.connect(path)
    assert con.execute("SELECT SUM(turn_count) FROM session").fetchone()[0] == 2
    assert con.execute("SELECT SUM(is_turn) FROM message").fetchone()[0] == 3


def test_v8_backfill_uses_the_same_kinds_as_the_ingest_path(tmp_path):
    """The migration writes SQL and ingest writes Python; they must agree, or an
    archive ends up with old rows and new rows classified differently."""
    from llm_archive.core.models import TURN_KINDS

    assert db._TURN_KIND_SQL == ", ".join(f"'{k}'" for k in sorted(TURN_KINDS))
    for kind in TURN_KINDS:
        assert f"'{kind}'" in db.MIGRATIONS[8][2]


def make_v9(path: Path) -> None:
    """A database as it looked before session grouping.

    Built by stripping the v10 additions back out of SCHEMA, so the ALTERs and the two
    CREATE TABLEs in MIGRATIONS[10] really have to run rather than being swallowed as
    duplicates or as `IF NOT EXISTS` no-ops.
    """
    schema = db.SCHEMA
    for line in ("  continues_session_id INTEGER REFERENCES session(id),\n",
                 "  continues_overlap    INTEGER,\n",
                 "  superseded       INTEGER NOT NULL DEFAULT 0,\n"):
        assert line in schema, "SCHEMA changed; this fixture no longer strips v10"
        schema = schema.replace(line, "")
    for start in ("CREATE TABLE IF NOT EXISTS topic",
                  "CREATE TABLE IF NOT EXISTS session_topic"):
        i = schema.index(start)
        schema = schema[:i] + schema[schema.index(";", i) + 1:]

    con = sqlite3.connect(path)
    con.executescript(schema)
    con.execute("INSERT INTO meta(key,value) VALUES ('schema_version','9')")
    con.execute("INSERT INTO source(id,kind,label,surface) "
                "VALUES (1,'claude_code','Claude Code','cli')")
    con.execute("""INSERT INTO session(id,source_id,native_id,started_at,msg_count,
                                       raw_path,raw_hash,ingested_at)
                   VALUES (1,1,'s1',1,2,'x','h',1)""")
    con.executemany("""INSERT INTO message(id,session_id,native_id,seq,on_active_path,
                                           role,created_at)
                       VALUES (?,1,?,?,1,?,1)""",
                    [(1, "u1", 0, "user"), (2, "u2", 1, "assistant")])
    con.commit()
    con.close()


def test_v10_adds_the_grouping_columns_and_tables(tmp_path):
    path = tmp_path / "a.db"
    make_v9(path)
    con = db.connect(path)

    session_cols = {r[1] for r in con.execute("PRAGMA table_info(session)")}
    assert {"continues_session_id", "continues_overlap"} <= session_cols
    assert "superseded" in {r[1] for r in con.execute("PRAGMA table_info(message)")}
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"topic", "session_topic"} <= tables
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'"
                       ).fetchone()[0] == str(db.SCHEMA_VERSION)


def test_v10_leaves_existing_rows_counted(tmp_path):
    """`superseded` defaults to 0, so nothing an existing archive holds silently stops
    counting the moment the migration runs."""
    path = tmp_path / "a.db"
    make_v9(path)
    con = db.connect(path)

    assert con.execute("SELECT COUNT(*) FROM message WHERE superseded = 0"
                       ).fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM session "
                       "WHERE continues_session_id IS NULL").fetchone()[0] == 1
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v10_is_safe_to_re_run(tmp_path):
    path = tmp_path / "a.db"
    make_v9(path)
    db.connect(path).close()
    con = db.connect(path)
    assert con.execute("SELECT COUNT(*) FROM message").fetchone()[0] == 2


def test_v10_ddl_is_shared_between_fresh_and_migrated(tmp_path):
    """A new table has to be written into SCHEMA *and* MIGRATIONS or one kind of
    database never gets it. Sharing the constant is what makes that impossible."""
    assert db.TOPIC_DDL in db.MIGRATIONS[10]
    assert db.SESSION_TOPIC_DDL in db.MIGRATIONS[10]
    assert db.TOPIC_DDL.strip() in db.SCHEMA
    assert db.SESSION_TOPIC_DDL.strip() in db.SCHEMA


def test_fresh_and_migrated_databases_agree(tmp_path):
    """The two paths must produce the same schema, or a bug only reproduces on one."""
    migrated = tmp_path / "old.db"
    make_v9(migrated)
    con_migrated = db.connect(migrated)
    con_fresh = db.connect(tmp_path / "new.db")

    def shape(con, table):
        return {(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({table})")}

    for table in ("session", "message", "topic", "session_topic"):
        assert shape(con_migrated, table) == shape(con_fresh, table), table
