"""Sessions that stopped without finishing: the archive as a to-do list you already own.

`--abandoned` everywhere else in this codebase means a BRANCH left behind by a rewind or
a regeneration (`message.on_active_path = 0`). This is a different question, asked of
whole sessions, and it gets a different word so the two never blur: a session is
*unfinished* when its live transcript ends in a state that was waiting on you.

Three shapes, each carried as a `reason` so the list can be read and filtered:

* `unanswered` — the last turn is yours. You asked and nothing came back: the tab was
  closed, the export was taken mid-reply, the panel dropped the request.
* `cut-off` — the agent was mid-task when it stopped: the last message is a tool step
  with nothing after it, or you interrupted it and never picked it back up.
* `asked-you` — the agent's last message is a question to you, and nobody answered.
  "Want me to add that?" left hanging is work that was offered and never decided.

Two of the three had to be measured before they could be trusted.

**`asked-you` is agent-only.** Of the 75 sessions in this archive whose final reply ends
with `?`, 50 are web chats, and nearly every one of those is the model's sign-off —
"Would you like me to explain any part in more detail?" — which is not a question that
waits on anyone. The agent surfaces (CLI, editor panel) are different: their closing
questions propose concrete work, and the ones without a reply are decisions never made.
So the rule applies to `source.surface != 'web'`, and to the last SENTENCE only, which
is what lets a closing "Anything else?" be recognised as filler rather than a question.

**`cut-off` is calibrated per session.** A trailing tool step is a stop only if this
source records tool results at all. VS Code chat never stores one (155 of 155 tool
messages here carry none), T3 Chat's image generations are a bare tool_use by design,
and claude.ai's artifact creates leave no result in the export. So a trailing tool step
counts only when the same session also holds a tool result filed as a message of its
own — the shape Claude Code and Codex write, where a result that never came is a missing
row rather than an ambiguous one.

Everything is computed at read time from one query over the last live message of each
session; no column is added and nothing has to be rebuilt after an ingest. Left out
before the rules even run: subagent transcripts (not yours to answer), sessions a later
one resumed (you did come back — `session.continues_session_id` says so), sessions
whose last message is younger than `min_age` (still open on another screen), and
sessions tagged `dismissed`, which is how an item leaves the list without anything
being deleted.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

REASONS = ("unanswered", "cut-off", "asked-you")

# An ordinary tag, so the session page shows it and `browse` can facet on it. Dismissal
# is the only write this module makes, and it is undone by removing the tag.
DISMISS_TAG = "dismissed"

# How long since the last message before a session counts as stopped rather than as
# open on another screen. An hour is a lunch break, not a decision.
DEFAULT_MIN_AGE_MS = 60 * 60 * 1000

EXCERPT_CHARS = 200

# Claude Code writes this as the user's own turn when Esc cuts off a reply or a tool
# call; the person did not ask anything, so it is a stop rather than a question.
_INTERRUPTED = re.compile(r"^\s*\[request interrupted by user", re.IGNORECASE)

# A question mark, then only closing markup: `**...?**`, `...?)`, a fenced tail.
_ENDS_IN_QUESTION = re.compile(r"\?[\s*_`\"')\]]*$")

# Sign-offs that end in a question mark without asking for a decision.
_FILLER = re.compile(
    r"\b(anything else|let me know|any (other |further )?questions?|is there anything)\b",
    re.IGNORECASE)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Unfinished:
    session_id: int
    reason: str
    # the last live message's timestamp: how long this has been waiting
    last_at: int
    # one line of what it stopped on: your prompt, its question, or the tool step
    excerpt: str
    title: str | None
    source: str
    workspace: str
    host: str | None
    started_at: int


def last_sentence(text: str) -> str:
    """The final sentence of a reply, for the question rule and for the excerpt."""
    tail = (text or "").rstrip()
    # the last non-empty line first: a question is rarely split across a paragraph
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return ""
    pieces = _SENTENCE_BREAK.split(lines[-1])
    return pieces[-1].strip() if pieces else lines[-1]


def asks_a_question(text: str) -> bool:
    """Does the reply end by asking something that wants an answer?"""
    tail = (text or "").rstrip()
    if not tail or not _ENDS_IN_QUESTION.search(tail):
        return False
    return not _FILLER.search(last_sentence(tail))


def one_line(text: str | None, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


# The last message on each session's live path, with the few facts the rules need.
# `is_sidechain = 0` keeps a subagent's closing message inside a Claude Code transcript
# from standing in for the main conversation's.
_LAST_SQL = """
WITH last AS (
  SELECT m.id, m.session_id, m.role, m.is_turn, m.created_at,
         ROW_NUMBER() OVER (PARTITION BY m.session_id ORDER BY m.seq DESC) AS rn
    FROM message m
   WHERE m.on_active_path = 1 AND m.superseded = 0 AND m.is_sidechain = 0)
SELECT s.id AS session_id, s.title, s.started_at, s.host,
       src.kind AS source, src.surface, COALESCE(w.label, '') AS workspace,
       l.id AS message_id, l.role, l.is_turn, l.created_at AS last_at,
       EXISTS (SELECT 1 FROM part p WHERE p.message_id = l.id
                  AND p.kind IN ('tool_use', 'tool_result')) AS tool_step,
       (SELECT p.text FROM part p
         WHERE p.message_id = l.id AND p.kind IN ('tool_use', 'tool_result')
         ORDER BY p.seq DESC LIMIT 1) AS tool_line
  FROM last l
  JOIN session s   ON s.id = l.session_id
  JOIN source src  ON src.id = s.source_id
  LEFT JOIN workspace w ON w.id = s.workspace_id
 WHERE l.rn = 1
   AND s.parent_session_id IS NULL
   AND NOT EXISTS (SELECT 1 FROM session c WHERE c.continues_session_id = s.id)
"""

# Does this session file tool results as messages of their own? See the docstring:
# only then does a trailing tool step mean the loop stopped.
_RESULTS_APART_SQL = """
SELECT EXISTS (
  SELECT 1 FROM message m JOIN part p ON p.message_id = m.id
   WHERE m.session_id = ? AND p.kind = 'tool_result'
     AND NOT EXISTS (SELECT 1 FROM part q
                      WHERE q.message_id = m.id AND q.kind = 'tool_use'))
"""


def _results_apart(con: sqlite3.Connection, session_id: int) -> bool:
    return bool(con.execute(_RESULTS_APART_SQL, (session_id,)).fetchone()[0])


def message_text(con: sqlite3.Connection, message_id: int) -> str:
    """Every text part of one message, in order, as one string.

    Joined with nothing between: VS Code chat stores a streamed reply as dozens of
    fragments -- ", or", "to set", and a closing "?" on its own -- so the last part is
    a shard, not a sentence, and only the whole message can say how the reply ends.
    """
    rows = con.execute("SELECT text FROM part WHERE message_id = ? AND kind = 'text' "
                       "ORDER BY seq", (message_id,)).fetchall()
    return "".join(r["text"] or "" for r in rows)


def classify(con: sqlite3.Connection, row: sqlite3.Row) -> tuple[str, str] | None:
    """(reason, excerpt) for a session whose last live message leaves it unfinished.

    The text is read only for the rows whose rule needs it -- the assistant turns on
    web surfaces are the bulk of the archive and never do.
    """
    if row["role"] == "user" and row["is_turn"]:
        text = message_text(con, row["message_id"])
        reason = "cut-off" if _INTERRUPTED.match(text) else "unanswered"
        return reason, one_line(text)
    if not row["is_turn"] and row["tool_step"]:
        # asked only when it matters: one extra query per trailing tool step
        if _results_apart(con, row["session_id"]):
            return "cut-off", one_line(row["tool_line"])
        return None
    if row["role"] == "assistant" and row["is_turn"] and row["surface"] != "web":
        text = message_text(con, row["message_id"])
        if asks_a_question(text):
            return "asked-you", one_line(last_sentence(text))
    return None


def find(con: sqlite3.Connection, *, since: int | None = None, until: int | None = None,
         workspace: str | None = None, sources: tuple[str, ...] = (),
         reasons: tuple[str, ...] = (), min_age_ms: int = DEFAULT_MIN_AGE_MS,
         include_dismissed: bool = False, now_ms: int | None = None,
         limit: int | None = None) -> list[Unfinished]:
    """Every unfinished session, most recently stopped first."""
    for reason in reasons:
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {', '.join(REASONS)}; got {reason!r}")

    now = now_ms if now_ms is not None else int(time.time() * 1000)
    clauses, params = [], []
    if since:
        clauses.append("s.started_at >= ?")
        params.append(since)
    if until:
        clauses.append("s.started_at <= ?")
        params.append(until)
    if workspace:
        clauses.append("LOWER(COALESCE(w.label, '')) LIKE ?")
        params.append(f"%{workspace.lower()}%")
    if sources:
        clauses.append(f"src.kind IN ({','.join('?' * len(sources))})")
        params.extend(sources)
    if min_age_ms:
        clauses.append("l.created_at <= ?")
        params.append(now - min_age_ms)
    if not include_dismissed:
        clauses.append("""s.id NOT IN (SELECT st.session_id FROM session_tag st
                                        JOIN tag t ON t.id = st.tag_id
                                       WHERE t.name = ?)""")
        params.append(DISMISS_TAG)

    sql = _LAST_SQL + "".join(f" AND {c}" for c in clauses) + " ORDER BY l.created_at DESC"
    found: list[Unfinished] = []
    for row in con.execute(sql, params):
        verdict = classify(con, row)
        if verdict is None or (reasons and verdict[0] not in reasons):
            continue
        reason, excerpt = verdict
        found.append(Unfinished(
            session_id=row["session_id"], reason=reason, last_at=row["last_at"],
            excerpt=excerpt, title=row["title"], source=row["source"],
            workspace=row["workspace"], host=row["host"], started_at=row["started_at"]))
        if limit is not None and len(found) >= limit:
            break
    return found


def counts(items: list[Unfinished]) -> dict[str, int]:
    """How many of each reason, in REASONS order, zeros included."""
    out = {reason: 0 for reason in REASONS}
    for item in items:
        out[item.reason] += 1
    return out


def dismiss(con: sqlite3.Connection, session_ids: list[int]) -> int:
    """Tag sessions `dismissed`. Returns how many existed and were newly tagged."""
    con.execute("INSERT OR IGNORE INTO tag(name) VALUES (?)", (DISMISS_TAG,))
    tag_id = con.execute("SELECT id FROM tag WHERE name = ?", (DISMISS_TAG,)).fetchone()["id"]
    done = 0
    for sid in session_ids:
        if con.execute("SELECT 1 FROM session WHERE id = ?", (sid,)).fetchone() is None:
            continue
        cur = con.execute("INSERT OR IGNORE INTO session_tag(session_id, tag_id) "
                          "VALUES (?, ?)", (sid, tag_id))
        done += cur.rowcount
    con.commit()
    return done


def restore(con: sqlite3.Connection, session_ids: list[int]) -> int:
    """Take the `dismissed` tag off again. Returns how many rows it came off."""
    done = 0
    for sid in session_ids:
        cur = con.execute("""DELETE FROM session_tag WHERE session_id = ? AND tag_id =
                             (SELECT id FROM tag WHERE name = ?)""", (sid, DISMISS_TAG))
        done += cur.rowcount
    # the same housekeeping the web UI's untag does: a tag nothing carries is noise
    con.execute("DELETE FROM tag WHERE id NOT IN (SELECT tag_id FROM session_tag)")
    con.commit()
    return done
