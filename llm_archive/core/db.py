"""SQLite schema and idempotent upsert.

Idempotence is a Phase 1 requirement, not a nicety (PLAN.md §8.7): exports are
point-in-time snapshots, so re-dropping a fresh one must update rather than duplicate.
Every session is keyed by (source, native_id) and carries the hash of the bytes it was
parsed from, so an unchanged session is skipped without re-reading it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .models import TURN_KINDS, Session

# Imported rather than duplicated: the redaction rules and the columns they write
# belong together, and the fresh-database DDL below has to create exactly what
# MIGRATIONS[6] adds to an existing one.
from .redact import MIGRATION as REDACTION_DDL

SCHEMA_VERSION = 9

# v9. Which export files the archive has taken in, keyed by content.
#
# Two jobs. First, `llma add` needs to answer "have I already got this?" about a file
# that may arrive under a different name every time — the same Mistral chat re-exported
# is `chat-export-<a new epoch ms>.zip`, and the same claude.ai archive downloaded twice
# is `conversations-000.zip` and `conversations-000 (1).zip`. sha256 is the only stable
# identity. Second, once a drop's sessions are stored the file moves to `_archive/`, and
# `path` is what still points at it — `session.raw_path` is updated to match, because
# `freshness` stats that path to date the export and would otherwise fall back to the
# ingest clock, quietly changing what the report promises.
#
# `kind` is NULL for a file that sniffed as nothing. Those are recorded too: "I looked
# at this and it was not an export" is the answer that stops a 50 MB PDF in ~/Downloads
# being re-sniffed on every scan.
DROP_DDL = """
CREATE TABLE IF NOT EXISTS drop_file (
  id          INTEGER PRIMARY KEY,
  sha256      TEXT NOT NULL UNIQUE,
  name        TEXT NOT NULL,
  kind        TEXT,
  bytes       INTEGER NOT NULL,
  exported_at INTEGER,
  added_at    INTEGER NOT NULL,
  ingested_at INTEGER,
  sessions    INTEGER NOT NULL DEFAULT 0,
  path        TEXT NOT NULL,
  origin      TEXT
)"""

INDEX_RUN_DDL = """
CREATE TABLE IF NOT EXISTS index_run (
  id           INTEGER PRIMARY KEY,
  started_at   INTEGER NOT NULL,
  finished_at  INTEGER NOT NULL,
  fts_rows     INTEGER NOT NULL DEFAULT 0,
  chunks       INTEGER NOT NULL DEFAULT 0,
  vectors      INTEGER NOT NULL DEFAULT 0,
  model_tag    TEXT,
  with_vectors INTEGER NOT NULL DEFAULT 1,
  seconds      REAL,
  warnings     TEXT
)"""


def _widen_surface(con: sqlite3.Connection) -> None:
    """Add 'editor_panel' to source.surface's CHECK.

    SQLite cannot alter a CHECK constraint, so the table is rebuilt. `session` has a
    foreign key onto `source`, so keys are off for the swap and `legacy_alter_table`
    keeps the rename from re-parsing a schema that momentarily has no `source` table.
    """
    con.commit()
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("PRAGMA legacy_alter_table=ON")
    try:
        con.executescript("""
            CREATE TABLE source_v4 (
              id      INTEGER PRIMARY KEY,
              kind    TEXT NOT NULL UNIQUE,
              label   TEXT NOT NULL,
              surface TEXT NOT NULL CHECK (surface IN ('cli','web','editor_panel'))
            );
            INSERT INTO source_v4(id,kind,label,surface)
                 SELECT id,kind,label,surface FROM source;
            DROP TABLE source;
            ALTER TABLE source_v4 RENAME TO source;
        """)
    finally:
        con.execute("PRAGMA legacy_alter_table=OFF")
        con.execute("PRAGMA foreign_keys=ON")


# Rendered from the Python constant rather than retyped: a migration that backfills
# `is_turn` with a different set of kinds than `Message.is_turn` uses would produce an
# archive whose old rows and new rows disagree.
_TURN_KIND_SQL = ", ".join(f"'{k}'" for k in sorted(TURN_KINDS))

# v2: `session.host`. Neither Claude Code's JSONL nor Codex's session_meta records the
# machine a session ran on, so it cannot be recovered from a copied tree — it has to be
# supplied at ingest time (`--host`) and stored. Without it, sessions from three laptops
# are indistinguishable in every statistic.
MIGRATIONS = {
    2: ["ALTER TABLE session ADD COLUMN host TEXT",
        "CREATE INDEX IF NOT EXISTS idx_session_host ON session(host)"],
    # v3: search. `remove_diacritics 2` is required for Slovene (č/š/ž); FTS5 has no
    # Slovene stemmer, so prefix matching does the rest.
    3: ["""CREATE VIRTUAL TABLE IF NOT EXISTS part_fts USING fts5(
              text, content='part', content_rowid='id',
              tokenize="unicode61 remove_diacritics 2")""",
        """CREATE TABLE IF NOT EXISTS chunk (
              id         INTEGER PRIMARY KEY,
              part_id    INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE,
              message_id INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
              session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
              seq        INTEGER NOT NULL,
              text       TEXT NOT NULL,
              vec_row    INTEGER NOT NULL,
              model_tag  TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_chunk_model ON chunk(model_tag, vec_row)",
        "CREATE INDEX IF NOT EXISTS idx_chunk_session ON chunk(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunk_part ON chunk(part_id)"],
    # v4: a third surface. VS Code chat is neither a CLI nor a web app — it is a panel
    # inside an editor, and calling it 'cli' made the surface-split metric read as
    # terminal work that never happened.
    4: [_widen_surface,
        "UPDATE source SET surface='editor_panel' WHERE kind='vscode_chat'"],
    # v5: record index builds. Ingesting renumbers part ids, which silently invalidates
    # both indexes — keyword search then matches stale rowids and vector search loses
    # the re-ingested sessions entirely. Nothing recorded when the index was last built,
    # so nothing could say search had gone quietly wrong.
    5: [INDEX_RUN_DDL],
    # v6: opt-in secret redaction (§8.6). `part.redacted` counts what was replaced in
    # that row, and doubles as the "not scanned yet" marker — a re-ingested session's
    # parts are rewritten and come back at 0, which is what makes redaction survive an
    # ingest that re-reads the plaintext source.
    6: REDACTION_DDL,
    # v7: tool call duration. A handful of sources timestamp the call and its result
    # separately (Claude Code, Codex, opencode) — everything else leaves it NULL rather
    # than guessing.
    7: ["ALTER TABLE part ADD COLUMN duration_ms INTEGER"],
    # v8: turns vs tool steps. `msg_count` counts whatever the source called a record,
    # and the sources disagree wildly (see models.TURN_KINDS), so it could not be
    # compared across them. `message.is_turn` marks the messages that carry actual
    # said-or-shown content and `session.turn_count` sums them, leaving `msg_count`
    # untouched as the volume measure. Backfilled from `part` so an existing archive
    # does not have to be re-ingested; the UPDATEs are pure functions of `part` and are
    # safe to re-run.
    8: ["ALTER TABLE message ADD COLUMN is_turn INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE session ADD COLUMN turn_count INTEGER NOT NULL DEFAULT 0",
        f"""UPDATE message SET is_turn = EXISTS(
              SELECT 1 FROM part p
               WHERE p.message_id = message.id AND p.kind IN ({_TURN_KIND_SQL}))""",
        """UPDATE session SET turn_count = (
              SELECT COUNT(*) FROM message m
               WHERE m.session_id = session.id
                 AND m.on_active_path = 1 AND m.is_turn = 1)"""],
    # v9: taking data IN. Three columns and a table, all serving the same promise —
    # a later export of an account can only ever ADD to what is stored.
    #
    # `session.exported_at` dates the snapshot a session was parsed from, which is what
    # lets ingest refuse to apply an older export over a newer one. Before this, drops
    # were discovered in filename order and the last one processed won: Grok names its
    # ZIP after a bare uuid and OpenRouter after the chat title, so "last" was
    # effectively random and a stale drop could overwrite a conversation that had grown.
    #
    # `message.absent_since` is the other half. `upsert_session` used to delete a
    # session's whole message tree and rewrite it from the incoming snapshot, which is
    # correct only while every new export is a superset of the last. It is not: a chat
    # deleted server-side, a provider pruning history, a Takeout window that has rolled
    # past its retention — each of those made the next ingest silently drop messages the
    # archive already held. Now a stored message the snapshot does not mention is kept
    # and stamped with the moment it first went missing.
    #
    # NULL on both columns means "not known", not "zero": every row that predates v9 was
    # ingested before either clock existed. exported_at is left NULL rather than
    # backfilled from ingested_at, because "when you exported" and "when you last ran
    # ingest" are different promises and freshness.py already refuses to blur them.
    9: ["ALTER TABLE session ADD COLUMN exported_at INTEGER",
        "ALTER TABLE message ADD COLUMN absent_since INTEGER",
        DROP_DDL,
        "CREATE INDEX IF NOT EXISTS idx_drop_kind ON drop_file(kind)",
        "CREATE INDEX IF NOT EXISTS idx_session_raw_path ON session(raw_path)"],
}


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS source (
  id      INTEGER PRIMARY KEY,
  kind    TEXT NOT NULL UNIQUE,
  label   TEXT NOT NULL,
  surface TEXT NOT NULL CHECK (surface IN ('cli','web','editor_panel'))
);

-- key is a CASEFOLDED project root, never a raw cwd: one Claude Code project directory
-- holds up to 6 distinct cwds, two differing only by drive-letter case. See
-- docs/phase0-findings.md §3d.
CREATE TABLE IF NOT EXISTS workspace (
  id         INTEGER PRIMARY KEY,
  source_id  INTEGER NOT NULL REFERENCES source(id),
  key        TEXT NOT NULL,
  label      TEXT,
  git_remote TEXT,
  UNIQUE (source_id, key)
);

CREATE TABLE IF NOT EXISTS session (
  id                INTEGER PRIMARY KEY,
  source_id         INTEGER NOT NULL REFERENCES source(id),
  workspace_id      INTEGER REFERENCES workspace(id),
  native_id         TEXT NOT NULL,
  parent_session_id INTEGER REFERENCES session(id),
  host              TEXT,
  title             TEXT,
  title_source      TEXT,
  model_primary     TEXT,
  started_at        INTEGER NOT NULL,
  ended_at          INTEGER,
  msg_count         INTEGER NOT NULL DEFAULT 0,
  -- messages that carry said-or-shown content; <= msg_count, the rest are tool steps
  turn_count        INTEGER NOT NULL DEFAULT 0,
  tok_in INTEGER, tok_out INTEGER,
  tok_cache_read INTEGER, tok_cache_write INTEGER,
  cost_usd          REAL,
  raw_path          TEXT NOT NULL,
  raw_hash          TEXT NOT NULL,
  ingested_at       INTEGER NOT NULL,
  -- when the snapshot this was parsed from was TAKEN, not when it was ingested.
  -- What lets a re-ingest refuse to apply an older export over a newer one.
  exported_at       INTEGER,
  meta              TEXT,
  UNIQUE (source_id, native_id)
);

CREATE TABLE IF NOT EXISTS message (
  id               INTEGER PRIMARY KEY,
  session_id       INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
  native_id        TEXT,
  parent_native_id TEXT,
  seq              INTEGER NOT NULL,
  on_active_path   INTEGER NOT NULL DEFAULT 1,
  is_sidechain     INTEGER NOT NULL DEFAULT 0,
  -- a conversational turn rather than a bare tool call/result; see models.TURN_KINDS
  is_turn          INTEGER NOT NULL DEFAULT 1,
  role             TEXT NOT NULL,
  model            TEXT,
  created_at       INTEGER NOT NULL,
  tok_in INTEGER, tok_out INTEGER,
  -- set when a re-ingest no longer found this message in the source. The row stays:
  -- an export that has lost history must not take it out of the archive too.
  absent_since     INTEGER,
  meta             TEXT,
  UNIQUE (session_id, native_id)
);

CREATE TABLE IF NOT EXISTS part (
  id             INTEGER PRIMARY KEY,
  message_id     INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  seq            INTEGER NOT NULL,
  kind           TEXT NOT NULL,
  text           TEXT,
  blob_id        INTEGER REFERENCES blob(id),
  tool_name      TEXT,
  tool_ok        INTEGER,
  bytes          INTEGER NOT NULL DEFAULT 0,
  embed_eligible INTEGER NOT NULL DEFAULT 0,
  -- how many secrets were replaced in `text`; 0 also means "not scanned yet"
  redacted       INTEGER NOT NULL DEFAULT 0,
  -- tool_use only: wall-clock time to the matching tool_result, where the source
  -- timestamps both ends. NULL means "not timed", not "instant".
  duration_ms    INTEGER
);

CREATE TABLE IF NOT EXISTS blob (
  id     INTEGER PRIMARY KEY,
  sha256 TEXT NOT NULL UNIQUE,
  bytes  INTEGER NOT NULL,
  path   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tag (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, color TEXT
);
CREATE TABLE IF NOT EXISTS session_tag (
  session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tag(id) ON DELETE CASCADE,
  PRIMARY KEY (session_id, tag_id)
);

CREATE TABLE IF NOT EXISTS ingest_run (
  id INTEGER PRIMARY KEY,
  source_kind TEXT, started_at INTEGER, finished_at INTEGER,
  files_seen INTEGER, sessions_new INTEGER, sessions_updated INTEGER,
  sessions_skipped INTEGER, messages INTEGER, parts INTEGER,
  stats TEXT
);

-- When the search indexes were last built, so staleness can be told from freshness.
CREATE TABLE IF NOT EXISTS index_run (
  id           INTEGER PRIMARY KEY,
  started_at   INTEGER NOT NULL,
  finished_at  INTEGER NOT NULL,
  fts_rows     INTEGER NOT NULL DEFAULT 0,
  chunks       INTEGER NOT NULL DEFAULT 0,
  vectors      INTEGER NOT NULL DEFAULT 0,
  model_tag    TEXT,
  with_vectors INTEGER NOT NULL DEFAULT 1,
  seconds      REAL,
  warnings     TEXT
);

-- Search tables live here as well as in MIGRATIONS[3]: migrations only run for an
-- EXISTING database, so a fresh archive would otherwise be created without them.
CREATE VIRTUAL TABLE IF NOT EXISTS part_fts USING fts5(
  text, content='part', content_rowid='id',
  tokenize="unicode61 remove_diacritics 2");

CREATE TABLE IF NOT EXISTS chunk (
  id         INTEGER PRIMARY KEY,
  part_id    INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE,
  message_id INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
  seq        INTEGER NOT NULL,
  text       TEXT NOT NULL,
  vec_row    INTEGER NOT NULL,
  model_tag  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_model   ON chunk(model_tag, vec_row);
CREATE INDEX IF NOT EXISTS idx_chunk_session ON chunk(session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_part    ON chunk(part_id);

-- What the opt-in redaction pass replaced, by fingerprint rather than by value.
CREATE TABLE IF NOT EXISTS redaction (
  id          INTEGER PRIMARY KEY,
  part_id     INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE,
  session_id  INTEGER,
  rule        TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  hits        INTEGER NOT NULL DEFAULT 1,
  at          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_redaction_rule ON redaction(rule);
CREATE INDEX IF NOT EXISTS idx_redaction_part ON redaction(part_id);

CREATE TABLE IF NOT EXISTS drop_file (
  id          INTEGER PRIMARY KEY,
  sha256      TEXT NOT NULL UNIQUE,
  name        TEXT NOT NULL,
  kind        TEXT,
  bytes       INTEGER NOT NULL,
  exported_at INTEGER,
  added_at    INTEGER NOT NULL,
  ingested_at INTEGER,
  sessions    INTEGER NOT NULL DEFAULT 0,
  path        TEXT NOT NULL,
  origin      TEXT
);
CREATE INDEX IF NOT EXISTS idx_drop_kind ON drop_file(kind);

CREATE INDEX IF NOT EXISTS idx_session_raw_path ON session(raw_path);
CREATE INDEX IF NOT EXISTS idx_session_time   ON session(started_at);
CREATE INDEX IF NOT EXISTS idx_session_source ON session(source_id, started_at);
CREATE INDEX IF NOT EXISTS idx_session_ws     ON session(workspace_id);
CREATE INDEX IF NOT EXISTS idx_msg_session    ON message(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_msg_active     ON message(session_id, on_active_path);
CREATE INDEX IF NOT EXISTS idx_part_msg       ON part(message_id, seq);
CREATE INDEX IF NOT EXISTS idx_part_tool      ON part(tool_name) WHERE tool_name IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_part_embed     ON part(embed_eligible) WHERE embed_eligible = 1;
"""


# Runs AFTER the migration ladder, for both fresh and existing databases.
#
# SCHEMA is executed against an existing archive too, so nothing in it may reference a
# column a migration adds — `CREATE INDEX ... ON part(redacted)` sitting in SCHEMA made
# every command fail with "no such column: redacted" on a v5 database, because the index
# was created before the ALTER that adds the column. Anything with that dependency
# belongs here instead.
POST_SCHEMA = [
    "CREATE INDEX IF NOT EXISTS idx_part_redacted ON part(redacted) WHERE redacted > 0",
    "CREATE INDEX IF NOT EXISTS idx_msg_turn ON message(session_id, is_turn)",
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)

    if fresh:
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('schema_version',?)",
                    (str(SCHEMA_VERSION),))
    else:
        row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 1
        for version in sorted(v for v in MIGRATIONS if v > current):
            for statement in MIGRATIONS[version]:
                try:
                    # a step can be a callable when one ALTER cannot express it
                    if callable(statement):
                        statement(con)
                    else:
                        con.execute(statement)
                except sqlite3.OperationalError as exc:
                    # a re-run over an already-migrated database is not an error
                    if "duplicate column" not in str(exc).lower():
                        raise
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('schema_version',?)",
                    (str(SCHEMA_VERSION),))

    for statement in POST_SCHEMA:
        con.execute(statement)
    con.commit()
    return con


def source_id(con: sqlite3.Connection, kind: str, label: str, surface: str) -> int:
    con.execute("INSERT OR IGNORE INTO source(kind,label,surface) VALUES (?,?,?)",
                (kind, label, surface))
    return con.execute("SELECT id FROM source WHERE kind=?", (kind,)).fetchone()["id"]


def workspace_id(con: sqlite3.Connection, src: int, key: str | None,
                 label: str | None) -> int | None:
    if not key:
        return None
    con.execute("INSERT OR IGNORE INTO workspace(source_id,key,label) VALUES (?,?,?)",
                (src, key, label))
    row = con.execute("SELECT id,label FROM workspace WHERE source_id=? AND key=?",
                      (src, key)).fetchone()
    if label and not row["label"]:
        con.execute("UPDATE workspace SET label=? WHERE id=?", (label, row["id"]))
    return row["id"]


def blob_id(con: sqlite3.Connection, sha: str, size: int, path: str) -> int:
    con.execute("INSERT OR IGNORE INTO blob(sha256,bytes,path) VALUES (?,?,?)",
                (sha, size, path))
    return con.execute("SELECT id FROM blob WHERE sha256=?", (sha,)).fetchone()["id"]


def existing_hash(con: sqlite3.Connection, src: int, native_id: str) -> str | None:
    row = con.execute("SELECT raw_hash FROM session WHERE source_id=? AND native_id=?",
                      (src, native_id)).fetchone()
    return row["raw_hash"] if row else None


@dataclass(frozen=True)
class SessionState:
    """What is already stored for one (source, native_id), as far as ingest cares.

    Two questions in one read: has this session changed since last time (`raw_hash`),
    and is what we are about to apply actually newer than what is stored
    (`exported_at`). Fetching them together keeps ingest to one query per session
    rather than two.
    """
    id: int
    raw_hash: str
    exported_at: int | None


def session_state(con: sqlite3.Connection, src: int,
                  native_id: str) -> SessionState | None:
    row = con.execute(
        "SELECT id, raw_hash, exported_at FROM session "
        " WHERE source_id=? AND native_id=?", (src, native_id)).fetchone()
    if row is None:
        return None
    return SessionState(row["id"], row["raw_hash"], row["exported_at"])


@dataclass
class MergeResult:
    """What one upsert did to a session's message tree."""
    session_id: int = 0
    was_new: bool = False
    appended: int = 0     # messages the snapshot added
    rewritten: int = 0    # messages the snapshot supplied that were already stored
    retained: int = 0     # stored messages this snapshot no longer mentions
    wholesale: bool = False   # the merge key was unusable; the tree was rebuilt

    def __iter__(self):
        """Unpacks as (session_id, was_new), which is what callers used to get back."""
        return iter((self.session_id, self.was_new))


def _mergeable(sess: Session) -> bool:
    """Can this session's messages be matched against stored rows one by one?

    The merge key is `UNIQUE (session_id, native_id)`, so it only works when every
    incoming message actually has a native_id. Three adapters cannot promise that:
    Codex synthesises `None` for records with no call_id (codex.py:203,282) and T3 Chat
    falls back to an empty string. Those sessions take the old wholesale rewrite —
    which is safe for all three, because each is a self-contained local file that is
    re-read in full every time, never a partial snapshot of a remote account.

    Blank is treated as missing: '' would collide every unnamed message onto one row.
    """
    return all(m.native_id for m in sess.messages)


def upsert_session(con: sqlite3.Connection, src: int, sess: Session) -> MergeResult:
    """Insert or merge one session and its message tree.

    Returns a MergeResult, which unpacks as the (session_id, was_new) pair this used to
    return so existing callers keep working.

    **Merge, not replace.** This used to delete the whole message tree and rewrite it
    from the incoming snapshot. That is correct only while every new export is a
    superset of the last one, and exports are not: a chat deleted server-side, a
    provider pruning old turns, a Takeout window that has rolled past its 18-month
    retention — each of those made the next ingest quietly remove messages the archive
    already held, which is the one thing an archive must never do.

    So a stored message the snapshot does not mention is kept and stamped
    `absent_since`. A message the snapshot does supply is rewritten in place, parts and
    all — parts have no unique key of their own, and rewriting them per message keeps
    parse changes from leaving stale rows behind, which is what the wholesale delete
    was really buying.
    """
    ws = workspace_id(con, src, sess.workspace_key, sess.workspace_label)
    now = int(time.time() * 1000)

    parent_id = None
    if sess.parent_native_id:
        prow = con.execute("SELECT id FROM session WHERE source_id=? AND native_id=?",
                           (src, sess.parent_native_id)).fetchone()
        parent_id = prow["id"] if prow else None

    row = con.execute("SELECT id FROM session WHERE source_id=? AND native_id=?",
                      (src, sess.native_id)).fetchone()
    was_new = row is None

    values = (
        src, ws, sess.native_id, parent_id, sess.host, sess.title, sess.title_source,
        sess.model_primary, sess.started_at, sess.ended_at, sess.msg_count,
        sess.turn_count, sess.tok_in, sess.tok_out, sess.tok_cache_read,
        sess.tok_cache_write,
        sess.cost_usd, sess.raw_path, sess.raw_hash, now, sess.exported_at,
        json.dumps(sess.meta, ensure_ascii=False) if sess.meta else None,
    )

    res = MergeResult(was_new=was_new)

    if was_new:
        cur = con.execute(
            "INSERT INTO session(source_id,workspace_id,native_id,parent_session_id,"
            "host,title,title_source,model_primary,started_at,ended_at,msg_count,"
            "turn_count,tok_in,"
            "tok_out,tok_cache_read,tok_cache_write,cost_usd,raw_path,raw_hash,"
            "ingested_at,exported_at,meta) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values)
        sid = cur.lastrowid
    else:
        sid = row["id"]
        con.execute(
            "UPDATE session SET workspace_id=?,parent_session_id=?,host=?,title=?,"
            "title_source=?,model_primary=?,started_at=?,ended_at=?,msg_count=?,"
            "turn_count=?,tok_in=?,tok_out=?,tok_cache_read=?,tok_cache_write=?,"
            "cost_usd=?,"
            "raw_path=?,raw_hash=?,ingested_at=?,exported_at=?,meta=? WHERE id=?",
            (ws, parent_id, sess.host, sess.title, sess.title_source, sess.model_primary,
             sess.started_at, sess.ended_at, sess.msg_count, sess.turn_count,
             sess.tok_in, sess.tok_out,
             sess.tok_cache_read, sess.tok_cache_write, sess.cost_usd, sess.raw_path,
             sess.raw_hash, now, sess.exported_at,
             json.dumps(sess.meta, ensure_ascii=False) if sess.meta else None, sid))
        if not _mergeable(sess):
            con.execute("DELETE FROM message WHERE session_id=?", (sid,))
            res.wholesale = True

    res.session_id = sid

    # Stored rows to match the snapshot against. Empty for a new session and for the
    # wholesale fallback, which makes the loop below a plain insert in both cases.
    stored: dict[str, int] = {}
    if not was_new and not res.wholesale:
        stored = {r["native_id"]: r["id"] for r in con.execute(
            "SELECT id, native_id FROM message "
            " WHERE session_id=? AND native_id IS NOT NULL", (sid,))}

    for msg in sess.messages:
        mid = stored.pop(msg.native_id, None) if stored else None
        if mid is None:
            cur = con.execute(
                "INSERT INTO message(session_id,native_id,parent_native_id,seq,"
                "on_active_path,is_sidechain,is_turn,role,model,created_at,tok_in,"
                "tok_out,meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, msg.native_id, msg.parent_native_id, msg.seq,
                 int(msg.on_active_path), int(msg.is_sidechain), int(msg.is_turn),
                 msg.role, msg.model,
                 msg.created_at, msg.tok_in, msg.tok_out,
                 json.dumps(msg.meta, ensure_ascii=False) if msg.meta else None))
            mid = cur.lastrowid
            if not was_new and not res.wholesale:
                res.appended += 1
        else:
            # Present in both. Rewritten in place rather than left alone: the snapshot
            # may have corrected the text, and `absent_since` has to be cleared for a
            # message that has come back.
            con.execute(
                "UPDATE message SET parent_native_id=?,seq=?,on_active_path=?,"
                "is_sidechain=?,is_turn=?,role=?,model=?,created_at=?,tok_in=?,"
                "tok_out=?,absent_since=NULL,meta=? WHERE id=?",
                (msg.parent_native_id, msg.seq, int(msg.on_active_path),
                 int(msg.is_sidechain), int(msg.is_turn), msg.role, msg.model,
                 msg.created_at, msg.tok_in, msg.tok_out,
                 json.dumps(msg.meta, ensure_ascii=False) if msg.meta else None, mid))
            con.execute("DELETE FROM part WHERE message_id=?", (mid,))
            res.rewritten += 1

        for part in msg.parts:
            bid = None
            if part.blob_sha and part.blob_path:
                bid = blob_id(con, part.blob_sha, part.bytes, part.blob_path)
            con.execute(
                "INSERT INTO part(message_id,seq,kind,text,blob_id,tool_name,tool_ok,"
                "bytes,embed_eligible,duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mid, part.seq, part.kind, part.text, bid, part.tool_name,
                 None if part.tool_ok is None else int(part.tool_ok),
                 part.bytes, int(part.embed_eligible), part.duration_ms))

    # Whatever is left in `stored` is a message the archive holds and this snapshot did
    # not mention. It stays. `absent_since` is stamped only the first time it goes
    # missing, so the column dates the loss rather than the most recent ingest.
    if stored:
        con.executemany(
            "UPDATE message SET absent_since=? WHERE id=? AND absent_since IS NULL",
            [(now, mid) for mid in stored.values()])
        res.retained = len(stored)

    # `sess.msg_count` describes the SNAPSHOT; after a merge the stored tree can hold
    # rows the snapshot never supplied, and the two would disagree. Recount from the
    # table so the session's own numbers describe the session's own rows.
    if res.retained:
        counts = con.execute(
            "SELECT COUNT(*) AS msgs, "
            "       COALESCE(SUM(is_turn), 0) AS turns "
            "  FROM message WHERE session_id=? AND on_active_path=1", (sid,)).fetchone()
        con.execute("UPDATE session SET msg_count=?, turn_count=? WHERE id=?",
                    (counts["msgs"], counts["turns"], sid))

    return res


def record_index_run(con: sqlite3.Connection, started: int, result) -> None:
    """Log one index build. Read back by `metrics.index_health` to date the indexes."""
    con.execute(
        "INSERT INTO index_run(started_at,finished_at,fts_rows,chunks,vectors,"
        "model_tag,with_vectors,seconds,warnings) VALUES (?,?,?,?,?,?,?,?,?)",
        (started, int(time.time() * 1000), result.fts_rows, result.chunks,
         result.vectors, result.model_tag, 0 if result.skipped_vectors else 1,
         result.seconds,
         json.dumps(result.warnings, ensure_ascii=False) if result.warnings else None))
    con.commit()


def record_run(con: sqlite3.Connection, kind: str, started: int, counts: dict) -> None:
    con.execute(
        "INSERT INTO ingest_run(source_kind,started_at,finished_at,files_seen,"
        "sessions_new,sessions_updated,sessions_skipped,messages,parts,stats) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (kind, started, int(time.time() * 1000), counts.get("files", 0),
         counts.get("new", 0), counts.get("updated", 0), counts.get("skipped", 0),
         counts.get("messages", 0), counts.get("parts", 0),
         json.dumps(counts, ensure_ascii=False)))
    con.commit()
