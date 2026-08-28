"""Export freshness — which sources have gone stale, and which ones cannot.

Ten sources, three completely different decay behaviours, and only one of them is
answered by "when was the newest session?":

* **live** — Claude Code, Codex, opencode, VS Code chat. The store is on this disk and
  every ingest re-reads it, so these are never stale *as exports*. What can go stale is
  the ingest itself, which is what `llma schedule` exists for; freshness reports their
  last ingest and stops there.
* **bulk** — claude.ai, ChatGPT, DeepSeek, T3 Chat, Gemini, Grok. One request returns
  the whole account, so "re-export every N days" is a coherent instruction and these
  are the only sources worth nagging about.
* **per_chat** — OpenRouter, Mistral, Copilot share links. Exporting means one file per
  conversation (§2.1). There is no action that makes such a source current, so an
  interval warning here is pure noise: they are listed with what is held and never
  marked overdue.

**The measurement that makes this honest.** Ranking bulk sources by newest session
answers the wrong question. A source whose newest chat is three months old is either an
export you have neglected *or* a service you have stopped using, and those want
opposite responses. The two are separable, because `session.raw_path` names the drop
the rows were parsed from and that file's mtime is when the export was taken:

    exported_at older than the interval  ->  overdue, go and re-export
    exported_at fresh, newest_session old ->  current; you just have not used it

Only the first is a warning. The second is reported as a quiet observation, so the list
does not cry wolf about the four accounts that are genuinely near-empty (§2.3, §2.5,
§2.6) and train you to skim past the one that matters.

A source in the table with **no rows at all** is the loudest case of all and the reason
the table is not derived from `source`: a bulk source that was never ingested cannot
appear in a query over sessions, so ChatGPT — still the outstanding Phase 3 gap — would
be silently omitted from the one report whose job is to notice missing exports.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DAY_MS = 86_400_000
DEFAULT_INTERVAL_DAYS = 30

LIVE = "live"
BULK = "bulk"
PER_CHAT = "per_chat"

# Cap on how many distinct raw files get stat()ed per source. Bulk web sources have
# one or two; claude_code has 67 and is `live`, so it never reaches this path anyway.
MAX_RAW_STATS = 250


@dataclass(frozen=True)
class ExportRoute:
    """How a source is kept current, in the user's hands rather than the parser's."""
    mode: str
    label: str              # used only when the source has no row in `source` yet
    how: str = ""           # where the export button actually is
    note: str | None = None       # context, shown when the source needs attention
    # A hard deadline after which data is gone rather than merely stale. Kept apart
    # from `note` because it is worth printing on a source that is perfectly current:
    # "exported 1d ago" and "deletes itself in 18 months" are both true, and only one
    # of them stops being true if you look away.
    retention: str | None = None


ROUTES: dict[str, ExportRoute] = {
    # --- live local stores: re-read on every ingest -------------------------
    "claude_code": ExportRoute(LIVE, "Claude Code", "~/.claude/projects"),
    "codex": ExportRoute(LIVE, "Codex", "~/.codex/sessions"),
    "opencode": ExportRoute(LIVE, "opencode", "~/.local/share/opencode/storage"),
    "vscode_chat": ExportRoute(LIVE, "VS Code chat", "VS Code globalStorage"),

    # --- bulk exports: one request returns the whole account ----------------
    "claude_web": ExportRoute(
        BULK, "claude.ai", "Settings -> Privacy -> Export data (arrives by email)"),
    "chatgpt": ExportRoute(
        BULK, "ChatGPT",
        "Settings -> Data controls -> Export data (arrives by email)",
        note="the export ZIP also carries the images you uploaded; drop the whole ZIP, "
             "not just conversations.json"),
    "deepseek": ExportRoute(
        BULK, "DeepSeek", "Profile -> Settings -> Data -> Export data",
        note="user.json in the drop holds account identity and is never read (§8.6)"),
    "t3chat": ExportRoute(
        BULK, "T3 Chat", "Settings -> History & Sync -> Export",
        note="the cheapest source here to keep current — one click, whole account"),
    "gemini": ExportRoute(
        BULK, "Gemini", "takeout.google.com -> My Activity -> Gemini Apps (HTML)",
        retention="My Activity auto-deletes on a rolling window (18 months by "
                  "default) — a missed export here loses conversations permanently, "
                  "not just temporarily"),
    "grok": ExportRoute(
        BULK, "Grok", "Settings -> Data Controls -> Export your data",
        retention="the emailed bundle sits under x.ai's own ttl/30d marker — "
                  "download it before the link ages out, not before the next reminder"),

    # --- per-chat only: no single action makes these current ----------------
    "openrouter": ExportRoute(
        PER_CHAT, "OpenRouter", "chat -> gear cog -> Export Chat, once per chat",
        note="local-first in browser storage; §2.1's opt-in IndexedDB reader is the "
             "only route that would not be one click per conversation"),
    "mistral": ExportRoute(
        PER_CHAT, "Mistral", "chat -> ... menu -> Export chat, once per chat"),
    "copilot_web": ExportRoute(
        PER_CHAT, "GitHub Copilot (web)", "Share link, captured from the network panel",
        note="there is no export at all (§2.7) — every conversation is a manual capture"),
}


def _newest_export_ms(con: sqlite3.Connection, kind: str) -> tuple[int | None, str]:
    """When the newest drop this source was parsed from was written.

    Falls back to `ingested_at` when the file has been moved or deleted — the drop is
    the better clock, but a missing drop is not a reason to report nothing. The second
    element says which clock answered, because "when you exported" and "when you last
    ran ingest" are different promises and the report should not blur them.
    """
    rows = con.execute("""
        SELECT DISTINCT s.raw_path FROM session s
        JOIN source src ON src.id = s.source_id
        WHERE src.kind = ? LIMIT ?""", (kind, MAX_RAW_STATS)).fetchall()

    newest = None
    for row in rows:
        try:
            mtime = Path(row["raw_path"]).stat().st_mtime
        except (OSError, ValueError):
            continue
        ms = int(mtime * 1000)
        newest = ms if newest is None else max(newest, ms)
    if newest is not None:
        return newest, "file"

    row = con.execute("""
        SELECT MAX(s.ingested_at) t FROM session s
        JOIN source src ON src.id = s.source_id
        WHERE src.kind = ?""", (kind,)).fetchone()
    return (row["t"], "ingest") if row and row["t"] else (None, "none")


def _age_days(ms: int | None, now: int) -> float | None:
    return None if ms is None else max(0.0, (now - ms) / DAY_MS)


def report(con: sqlite3.Connection, interval_days: int = DEFAULT_INTERVAL_DAYS,
           now_ms: int | None = None) -> list[dict]:
    """One row per known source, ordered worst-first.

    `state` is one of:
      missing   a bulk source with nothing ingested — never exported, or never parsed
      overdue   bulk, and the newest export is older than `interval_days`
      due       bulk, and the export is within a week of the interval
      current   bulk, exported recently
      manual    per-chat; reported, never nagged
      live      a local store, refreshed by ingest rather than by exporting
    """
    now = now_ms if now_ms is not None else int(time.time() * 1000)

    present = {
        r["kind"]: r for r in con.execute("""
            SELECT src.kind, src.label, src.surface, COUNT(*) sessions,
                   MAX(s.started_at) newest_session,
                   MAX(s.ingested_at) last_ingest
            FROM session s JOIN source src ON src.id = s.source_id
            GROUP BY src.id""").fetchall()
    }

    rows: list[dict] = []
    for kind, route in ROUTES.items():
        have = present.get(kind)
        sessions = have["sessions"] if have else 0
        exported_at, clock = _newest_export_ms(con, kind) if have else (None, "none")

        row = {
            "kind": kind,
            "label": have["label"] if have else route.label,
            "mode": route.mode,
            "how": route.how,
            "note": route.note,
            "retention": route.retention,
            "sessions": sessions,
            "newest_session": have["newest_session"] if have else None,
            "last_ingest": have["last_ingest"] if have else None,
            "exported_at": exported_at,
            "exported_clock": clock,
            "export_age_days": _age_days(exported_at, now),
            "content_age_days": _age_days(have["newest_session"] if have else None, now),
            "interval_days": interval_days if route.mode == BULK else None,
        }

        if route.mode == LIVE:
            row["state"] = "live"
            row["reason"] = ("never ingested" if not sessions else
                             f"re-read on every ingest; last run "
                             f"{_age_days(row['last_ingest'], now):.0f}d ago")
        elif route.mode == PER_CHAT:
            row["state"] = "manual"
            row["reason"] = (f"{sessions} chat(s) exported by hand"
                             if sessions else "nothing exported yet")
        elif not sessions:
            row["state"] = "missing"
            row["reason"] = "no export has ever been ingested"
        else:
            age = row["export_age_days"] or 0.0
            if age > interval_days:
                row["state"] = "overdue"
                row["reason"] = f"last export {age:.0f}d ago (interval {interval_days}d)"
            elif age > interval_days - 7:
                row["state"] = "due"
                row["reason"] = f"last export {age:.0f}d ago, interval {interval_days}d"
            else:
                row["state"] = "current"
                row["reason"] = f"exported {age:.0f}d ago"

            # Said separately and never as a warning: an account you have stopped
            # using looks exactly like a neglected export in every count except this
            # one. Conflating them is how a reminder list becomes noise.
            gap = (row["content_age_days"] or 0) - age
            if row["state"] == "current" and gap > interval_days:
                row["idle"] = (f"newest chat is {row['content_age_days']:.0f}d old — "
                               f"the export is fresh, the account is idle")

        rows.append(row)

    order = {"missing": 0, "overdue": 1, "due": 2, "current": 3, "manual": 4, "live": 5}
    rows.sort(key=lambda r: (order[r["state"]],
                             -(r["export_age_days"] or 0)))
    return rows


def summary(rows: list[dict]) -> dict:
    """Counts for the banner. `needs_action` is what a reminder should be loud about."""
    by_state: dict[str, int] = {}
    for row in rows:
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
    return {
        "by_state": by_state,
        "needs_action": [r for r in rows if r["state"] in ("missing", "overdue")],
        "due_soon": [r for r in rows if r["state"] == "due"],
    }
