"""One session compacted into a context primer for a fresh agent session.

A transcript is the wrong thing to hand a new session. The one this module was first
run on is 443 messages: 165 tool calls, their 165 results (293 KB of file reads and
test output), 109 assistant turns and four from the person. What the next session
needs from it is what was asked, what was decided, what was changed and where it
stopped -- which is under 3% of the bytes, and none of the tool traffic.

Extractive and deterministic. Nothing is paraphrased; every line is a span of the
archive, chosen by rule:

* **Your turns are kept whole.** They are the spine of the session and a tenth of a
  percent of its bytes. The first one is the goal and gets its own heading.
* **The assistant's turns are trimmed to what carries decisions**: the first paragraph
  of each, plus any paragraph that argues -- `because`, `instead`, `trade-off`, `root
  cause`, `won't` -- and short code fences. The rest is replaced by a count of what
  was cut. The final reply is kept whole under its own heading, because that is
  where an agent writes what it did and what is left.
* **Tool calls are dropped, except the ones that failed.** A failure is the one tool
  step worth carrying forward -- it is what the next session would otherwise
  rediscover -- so it keeps the call's one-line intent and the head and tail of the
  error. Successful reads, edits and runs are summarised by the outcome instead.
* **The outcome comes from the derived tables**, which no transcript has: the files
  the session wrote (project files first, the scratch and memory writes counted), the
  commits it made with their messages, and the last test command and how it ended.
  If the session is unfinished in the inbox's sense (core.unfinished), the primer
  says so and why, since a fresh session is most often started to pick it back up.

**The budget is the document.** `chars` caps the rendered primer, not the prose in
it, for the reason api.session_payload gives: the reader has a context window and is
asking how much of it this will spend. The goal, the outcome and the last reply are
paid for first; the middle is filled in order until the budget runs out, and the cut
is reported rather than hidden.

Secrets are redacted on the way out with the same rules an export uses. A primer is
the one output whose whole purpose is to be pasted into another model's context, so
it is the last place a leaked key should travel through.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from .. import api
from ..core import lineage, unfinished
from ..core.redact import active_rules, redact_text

DEFAULT_CHARS = 12_000
GOAL_CHARS = 2_500          # the first user turn
USER_CHARS = 1_500          # any later user turn
ASSISTANT_CHARS = 700       # a trimmed assistant turn
LAST_REPLY_CHARS = 3_000    # the final assistant turn, kept whole up to this
FAILURE_CHARS = 400
FILES_LISTED = 15
COMMITS_LISTED = 10
FENCE_LINES = 8             # a code block longer than this is dropped from a trimmed turn

# Paragraphs that argue a choice rather than narrate a step.
_DECISION = re.compile(
    r"\b(decid\w*|instead|because|trade-?offs?|root cause|the fix|won'?t|will not|"
    r"chose|rather than|turns out|the reason|caveat|gotcha|important|note that|"
    r"the catch|which means|so that)\b", re.IGNORECASE)

_FENCE = re.compile(r"^```")

# A lead-in: a short assistant line that ends in a colon and introduces the tool call
# that follows -- "Now the tests. Writing via heredoc:". Narration, not content.
_LEAD_IN_CHARS = 160

# What counts as a test run, on the parsed command rather than its text: the text of
# a heredoc that *writes* a test file mentions pytest too.
_TEST_PROGRAMS = {"pytest", "jest", "vitest", "mocha", "rspec", "tox", "nox", "phpunit"}
_TEST_SUBCOMMANDS = {
    "python": {"pytest", "unittest"}, "python3": {"pytest", "unittest"},
    "py": {"pytest", "unittest"}, "uv": {"run"},
    "npm": {"test", "t"}, "pnpm": {"test", "t"}, "yarn": {"test"}, "bun": {"test"},
    "cargo": {"test"}, "go": {"test"}, "mvn": {"test"}, "gradle": {"test"},
    "dotnet": {"test"}, "make": {"test", "check"},
}
_COMMIT_M = re.compile(r"""-m\s+(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""", re.S)
# The terminator line is required: a heredoc that never closes is a quoted script
# that *mentions* one, and its "body" is whatever code came after.
_COMMIT_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1\b", re.S)


@dataclass
class Primer:
    session_id: int
    markdown: str
    # what the compaction did, so the reader can tell a thin session from a deep cut
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------ text rules

def _paragraphs(text: str) -> list[str]:
    """Blank-line paragraphs, with a fenced code block kept as one paragraph."""
    out: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
            buf.append(line)
            if not in_fence:
                out.append("\n".join(buf))
                buf = []
            continue
        if in_fence:
            buf.append(line)
        elif line.strip():
            buf.append(line)
        elif buf:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def trim_reply(text: str, limit: int = ASSISTANT_CHARS) -> tuple[str, int]:
    """An assistant turn reduced to what carries decisions. Returns (kept, cut chars).

    The first paragraph is always kept: it is the sentence that says what this turn
    is about. Fenced code longer than FENCE_LINES goes -- the change is in the files,
    and the outcome section names them.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text, 0
    paras = _paragraphs(text)
    kept: list[str] = []
    spent = 0
    for i, para in enumerate(paras):
        is_fence = para.startswith("```")
        if is_fence and para.count("\n") > FENCE_LINES:
            continue
        # headings are kept for the shape they give the paragraphs under them
        is_heading = para.lstrip().startswith("#") and "\n" not in para.strip()
        if i == 0 or is_heading or is_fence and kept or _DECISION.search(para):
            if spent + len(para) > limit and kept:
                break
            kept.append(para)
            spent += len(para)
    body = "\n\n".join(kept) if kept else text[:limit]
    cut = len(text) - len(body)
    return body, max(cut, 0)


def _clip(text: str, limit: int) -> tuple[str, int]:
    text = (text or "").strip()
    if len(text) <= limit:
        return text, 0
    return text[:limit].rstrip() + " …", len(text) - limit


def _failure_excerpt(text: str, limit: int = FAILURE_CHARS) -> str:
    """The head and the tail of an error: the exit line and the exception."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    picked = lines[:1] + (lines[-2:] if len(lines) > 3 else lines[1:])
    body = "\n".join(picked)
    return body if len(body) <= limit else body[:limit].rstrip() + " …"


def _is_test_run(argv0: str | None, subcommand: str | None, text: str | None) -> bool:
    prog = (argv0 or "").casefold()
    sub = (subcommand or "").casefold()
    if prog in _TEST_PROGRAMS:
        return True
    if sub in _TEST_SUBCOMMANDS.get(prog, ()):
        # `uv run` is a test run only when it runs one
        return prog != "uv" or bool(re.search(r"\buv\s+run\s+(pytest|python\s+-m\s+pytest)",
                                              text or ""))
    return False


def is_lead_in(text: str) -> bool:
    """A short line that ends in a colon: an announcement of the tool call after it."""
    flat = (text or "").strip()
    return 0 < len(flat) <= _LEAD_IN_CHARS and flat.endswith(":") and "\n" not in flat


def _first_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return None


def commit_message(command: str) -> str | None:
    """The message out of a `git commit` line that starts `command`: `-m "..."`,
    `-m "$(cat <<'EOF' ...)"`, or the first line of a heredoc fed to `-F -`.

    Everything is anchored to the commit's own line. A later `<<'PYEOF'` on the same
    Bash call belongs to some other stage, and its first line is not a message.
    """
    command = command or ""
    head = command.split("\n", 1)[0]
    m = _COMMIT_M.search(command)
    if m and m.start() < len(head):
        value = m.group(1) or m.group(2) or ""
        if value.lstrip().startswith("$(cat"):
            body = value.split("\n", 1)[1] if "\n" in value else ""
            return _first_line(body)
        return _first_line(value)
    if "<<" in head:
        m = _COMMIT_HEREDOC.search(command)
        if m and m.start() < len(head):
            return _first_line(m.group(2))
    return None


def commits_in(command: str) -> list[str]:
    """Every commit message a command line makes, in order.

    Found by stage with `api._GIT_COMMIT`, the same rule `llma blame` uses to name
    the session that ran a commit -- not by `command.argv0`, which names the first
    real stage of the line, and the commit is usually the second, after `git add`.
    """
    found = []
    for m in api._GIT_COMMIT.finditer(command or ""):
        msg = commit_message(command[m.start():])
        if msg and msg not in found:
            found.append(msg)
    return found


# ------------------------------------------------------------------ the build

def _outcome(con: sqlite3.Connection, session_id: int) -> dict:
    """Files written, commits made, the last test run -- from the derived tables."""
    try:
        files = con.execute("""
            SELECT COALESCE(tf.rel, tf.norm) AS key, tf.rel IS NOT NULL AS inside,
                   SUM(CASE WHEN tf.action IN ('write','edit','delete') THEN 1 ELSE 0 END)
                       AS writes,
                   COUNT(*) AS calls
              FROM touched_file tf
             WHERE tf.session_id = ? AND tf.ok IS NOT 0
             GROUP BY key HAVING writes > 0
             ORDER BY inside DESC, writes DESC, calls DESC""", (session_id,)).fetchall()
        commands = con.execute("""
            SELECT argv0, subcommand, text, ok, at FROM command
             WHERE session_id = ? ORDER BY at""", (session_id,)).fetchall()
        indexed = con.execute("SELECT 1 FROM touched_file LIMIT 1").fetchone() is not None
    except sqlite3.OperationalError:          # pre-v11 archive
        files, commands, indexed = [], [], False

    inside = [dict(f) for f in files if f["inside"]]
    outside = sum(1 for f in files if not f["inside"])
    commits = []
    last_test = None
    for c in commands:
        for msg in commits_in(c["text"]):
            if msg not in commits:
                commits.append(msg)
        if _is_test_run(c["argv0"], c["subcommand"], c["text"]):
            last_test = {"text": " ".join((c["text"] or "").split())[:160],
                         "ok": None if c["ok"] is None else bool(c["ok"])}
    return {"files": inside[:FILES_LISTED], "more_files": max(0, len(inside) - FILES_LISTED),
            "outside_files": outside, "commits": commits[-COMMITS_LISTED:],
            "last_test": last_test, "indexed": indexed}


def _rows(con: sqlite3.Connection, session_id: int):
    return con.execute("""
        SELECT m.id AS mid, m.role, m.seq, m.is_turn,
               p.kind, p.text, p.tool_name, p.tool_ok
          FROM message m JOIN part p ON p.message_id = m.id
         WHERE m.session_id = ? AND m.on_active_path = 1 AND m.is_sidechain = 0
         ORDER BY m.seq, p.seq""", (session_id,)).fetchall()


def build(con: sqlite3.Connection, session_id: int, *, chars: int = DEFAULT_CHARS,
          redact: bool = True) -> Primer | None:
    """The primer, or None when there is no such session."""
    row = api.session_row(con, session_id)
    if row is None:
        return None
    brief = api.session_brief(row)
    chain = lineage.chain(con, session_id)
    outcome = _outcome(con, session_id)
    open_item = next((it for it in unfinished.find(con, min_age_ms=0)
                      if it.session_id == session_id), None)

    # ---- one pass over the transcript, into typed items
    # each: ("user", text) | ("assistant", text) | ("failure", tool, intent, excerpt)
    items: list[tuple] = []
    stats = {"messages": 0, "tool_calls": 0, "failures": 0, "assistant_turns": 0,
             "user_turns": 0, "thinking_dropped": 0, "sidechain_dropped": 0}
    last_mid = None
    pending: dict[str, list[str]] = {}
    last_intent: dict[str, str] = {}
    for r in _rows(con, session_id):
        if r["mid"] != last_mid:
            stats["messages"] += 1
            last_mid = r["mid"]
        kind = r["kind"]
        if kind == "thinking":
            stats["thinking_dropped"] += 1
            continue
        if kind == "tool_use":
            stats["tool_calls"] += 1
            last_intent[r["tool_name"] or ""] = " ".join((r["text"] or "").split())
            continue
        if kind == "tool_result":
            if r["tool_ok"] == 0:
                stats["failures"] += 1
                tool = r["tool_name"] or ""
                items.append(("failure", tool, last_intent.get(tool, ""),
                              _failure_excerpt(r["text"])))
            continue
        if kind in ("image", "attachment"):
            text = f"[{kind}]"
        else:
            text = (r["text"] or "").strip()
            if not text:
                continue
        # consecutive text parts of one message are one turn
        if items and items[-1][0] == r["role"] and items[-1][2] == r["mid"]:
            items[-1] = (r["role"], items[-1][1] + "\n\n" + text, r["mid"])
        else:
            items.append((r["role"], text, r["mid"]))
    stats["sidechain_dropped"] = con.execute(
        "SELECT COUNT(*) FROM message WHERE session_id = ? AND is_sidechain = 1",
        (session_id,)).fetchone()[0]

    turns = [it for it in items if it[0] in ("user", "assistant")]
    stats["user_turns"] = sum(1 for it in turns if it[0] == "user")
    stats["assistant_turns"] = sum(1 for it in turns if it[0] == "assistant")

    goal = next((it for it in items if it[0] == "user"), None)
    last_reply = next((it for it in reversed(items) if it[0] == "assistant"), None)
    middle = [it for it in items if it is not goal and it is not last_reply]
    # "Reading the file now:" before a tool call says nothing once the call is gone
    lead_ins = sum(1 for it in middle if it[0] == "assistant" and is_lead_in(it[1]))
    middle = [it for it in middle if not (it[0] == "assistant" and is_lead_in(it[1]))]
    stats["lead_ins_dropped"] = lead_ins

    # ---- fixed sections first: they are paid for before the middle
    head = _header(brief, chain, stats)
    goal_md, goal_cut = _clip(goal[1], GOAL_CHARS) if goal else ("", 0)
    tail = _outcome_md(outcome, open_item, last_reply, stats["tool_calls"])
    fixed = len(head) + len(goal_md) + len(tail) + 200
    left = max(chars - fixed, 0)

    # ---- the middle. First at the normal caps. If that does not fit, again with
    # every turn cut to its opening, so the whole arc survives rather than the first
    # sixty percent of it -- a late decision is the one the next session needs most.
    # If it fits with room to spare, again with the trimmed turns allowed more: a
    # 700-character cap on a design reply keeps its first paragraph and cuts the
    # argument, and unspent budget is no use to anyone.
    fill = _fill(middle, left, ASSISTANT_CHARS, USER_CHARS)
    if fill.omitted and middle:
        per_turn = max(120, left // len(middle))
        tight = _fill(middle, left, min(ASSISTANT_CHARS, per_turn),
                      min(USER_CHARS, max(per_turn * 2, 300)))
        if tight.omitted < fill.omitted:
            fill = tight
    elif fill.trimmed and left - fill.used > 500:
        wider = _fill(middle, left, ASSISTANT_CHARS + (left - fill.used) // fill.trimmed,
                      USER_CHARS)
        if not wider.omitted:
            fill = wider

    parts = [head, "", "## Goal", "", _quote(goal_md) if goal_md else "_(no user turn)_"]
    if fill.body or fill.omitted:
        parts += ["", "## What happened", ""]
        parts.append("\n\n".join(fill.body))
        if fill.omitted:
            parts.append(f"\n_(… {fill.omitted} turn{'s' if fill.omitted != 1 else ''} "
                         f"omitted for the budget; `llma show {session_id}` has them all)_")
    parts += ["", tail]
    stats.update({"assistant_trimmed": fill.trimmed, "turns_omitted": fill.omitted,
                  "chars_cut": goal_cut + fill.cut})
    parts += ["", "---", _footer(session_id, stats)]

    text = "\n".join(parts).rstrip() + "\n"
    if redact:
        text, _ = redact_text(text, active_rules())
    stats["chars"] = len(text)
    return Primer(session_id=session_id, markdown=text, stats=stats)


@dataclass
class _Fill:
    body: list[str] = field(default_factory=list)
    omitted: int = 0
    trimmed: int = 0
    cut: int = 0
    used: int = 0


def _fill(middle: list[tuple], left: int, assistant_limit: int, user_limit: int) -> _Fill:
    """The middle of the transcript, in order, until `left` characters are spent."""
    out = _Fill()
    for it in middle:
        if it[0] == "failure":
            line = (f"✗ **{it[1] or 'tool'}** `{it[2][:160]}`" if it[2]
                    else f"✗ **{it[1] or 'tool'}**")
            if it[3]:
                line += "\n" + _indent(it[3])
        elif it[0] == "user":
            text, cut = _clip(it[1], user_limit)
            out.cut += cut
            line = f"**you:** {text}"
        else:
            text, cut = trim_reply(it[1], assistant_limit)
            if cut:
                out.trimmed += 1
                out.cut += cut
                text += f"\n_(… {cut:,} chars trimmed)_"
            line = f"**assistant:** {text}"
        if len(line) + 2 > left:
            out.omitted += 1
            continue
        out.body.append(line)
        left -= len(line) + 2
        out.used += len(line) + 2
    return out


# ------------------------------------------------------------------ pieces

def _header(brief: dict, chain: dict, stats: dict) -> str:
    when = (brief["started_at"] or "")[:10]
    ended = (brief["ended_at"] or "")[:10]
    span = when if not ended or ended == when else f"{when} → {ended}"
    bits = [f"#{brief['session_id']}", brief["source"], brief.get("workspace"),
            span, brief.get("model"),
            f"{brief['turns']} turns" if brief.get("turns") else None]
    lines = [f"# Primer: {brief['title'] or '(untitled)'}",
             " · ".join(str(b) for b in bits if b)]
    if chain.get("continues"):
        c = chain["continues"]
        lines.append(f"Continues #{c['id']} “{c['title'] or '(untitled)'}” — the first "
                     f"{c['overlap']} messages replay it; prime that one for the start.")
    if chain.get("continued_by"):
        c = chain["continued_by"]
        lines.append(f"Continued by #{c['id']} “{c['title'] or '(untitled)'}” — the "
                     f"conversation went on there; prime that one for the latest state.")
    if stats["sidechain_dropped"]:
        lines.append(f"{stats['sidechain_dropped']} subagent messages left out.")
    return "\n".join(lines)


def _outcome_md(outcome: dict, open_item, last_reply, tool_calls: int) -> str:
    lines = ["## Outcome", ""]
    if outcome["files"]:
        names = ", ".join(f"`{f['key']}`" + (f" ({f['writes']})" if f["writes"] > 1 else "")
                          for f in outcome["files"])
        extra = []
        if outcome["more_files"]:
            extra.append(f"+{outcome['more_files']} more")
        if outcome["outside_files"]:
            extra.append(f"{outcome['outside_files']} outside the project")
        lines.append(f"**Files changed:** {names}" + (f" ({', '.join(extra)})" if extra else ""))
    elif outcome["outside_files"]:
        lines.append(f"**Files changed:** {outcome['outside_files']} outside the project")
    elif not outcome["indexed"]:
        lines.append("_(no file or command history: the archive has not been indexed "
                     "— `llma index`)_")
    elif tool_calls:
        # an agent that called tools and wrote nothing is worth saying; a web chat
        # with no tools at all is not
        lines.append("**Files changed:** none")
    if outcome["commits"]:
        lines.append("**Commits:**")
        lines += [f"- {m}" for m in outcome["commits"]]
    if outcome["last_test"]:
        t = outcome["last_test"]
        verdict = {True: "passed", False: "FAILED", None: "outcome not recorded"}[t["ok"]]
        lines.append(f"**Last test run:** `{t['text']}` — {verdict}")
    if open_item is not None:
        why = {"unanswered": "your last message got no reply",
               "cut-off": "the agent was stopped mid-task",
               "asked-you": "it asked and nobody answered"}[open_item.reason]
        lines.append(f"**Unfinished** ({open_item.reason}): {why}"
                     + (f" — “{open_item.excerpt}”" if open_item.excerpt else ""))
    if last_reply is not None:
        text, cut = _clip(last_reply[1], LAST_REPLY_CHARS)
        lines += ["", "### Last reply", "", text]
        if cut:
            lines.append(f"_(… {cut:,} chars)_")
    return "\n".join(lines)


def _footer(session_id: int, stats: dict) -> str:
    bits = [f"{stats['messages']} messages"]
    if stats["tool_calls"]:
        bits.append(f"{stats['tool_calls']} tool calls dropped")
    if stats["failures"]:
        bits.append(f"{stats['failures']} failure{'s' if stats['failures'] != 1 else ''} kept")
    if stats["assistant_trimmed"]:
        bits.append(f"{stats['assistant_trimmed']} of {stats['assistant_turns']} "
                    f"assistant turns trimmed")
    if stats.get("lead_ins_dropped"):
        bits.append(f"{stats['lead_ins_dropped']} lead-in lines dropped")
    if stats["thinking_dropped"]:
        bits.append(f"{stats['thinking_dropped']} thinking blocks dropped")
    if stats["chars_cut"]:
        bits.append(f"{stats['chars_cut']:,} chars cut")
    return (f"_Primer compacted from {', '.join(bits)}. Secrets redacted. "
            f"Full transcript: `llma show {session_id} --tools`._")


def _quote(text: str) -> str:
    return "\n".join(f"> {ln}" if ln.strip() else ">" for ln in text.splitlines())


def _indent(text: str) -> str:
    return "\n".join(f"    {ln}" for ln in text.splitlines())
