"""Opt-in secret redaction over `part.text` (§8.6).

Ten sources of chat logs and tool output concentrated into one searchable file is the
most valuable thing on this laptop, and the leaks in it are not hypothetical: a
`cat .env`, a `printenv`, a curl with an `Authorization:` header, a key pasted into a
prompt to ask why it is rejected. All of that lands in `part.text`, which is exactly the
column that FTS indexes, the embedder reads and the web UI prints.

**What this does and does not promise.**

* It rewrites `part.text` in the database and records, per part, that it happened.
* It does **not** touch `data/blobs/` — those files are content-addressed by sha256, so
  rewriting one invalidates the hash every referring row was deduplicated on. Blobs are
  *scanned and reported* (`scan_blobs`), never edited. §8.6 already says `data/drops/`
  needs the same handling as the database; the blob store is the third member of that
  set, and this pass is not what secures it.
* It does **not** touch the original files under `raw_path`. That is the important
  caveat, because ingest re-reads them: a one-off redaction is undone by the next
  `llma ingest`, which is why redaction is a **stored setting** rather than a command
  you remember to re-run. With it enabled, every ingest and every scheduled sync
  applies it before the indexes are built.

**Fingerprints instead of deletion.** A match is replaced by
`[redacted:<rule>:<8 hex>]`, where the hex is the head of the sha256 of the secret
itself. That is enough to tell two leaked keys apart, to confirm months later whether
the key that leaked is the one you rotated, and to count how far one key spread — none
of which is possible if every secret becomes the same opaque marker, and all of which
would otherwise require keeping the plaintext.

**Two rulesets.** The default rules match tokens with issuer-specific prefixes and
fixed shapes (`sk-ant-`, `ghp_`, `AKIA…`) — those are near-zero false positive, because
nothing else looks like them. The `wide` rules match *shapes* instead (`password = …`,
a bearer header, a URL with credentials in it) and will occasionally redact a
placeholder or an example from documentation. They are off by default, since a search
archive that has eaten a paragraph of prose to protect the string `password = hunter2`
in a tutorial is worse at its job for no gain.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SETTING = "redact_enabled"
PLACEHOLDER_RE = re.compile(r"\[redacted:[a-z0-9_]+:[0-9a-f]{8}\]")

MIGRATION = [
    "ALTER TABLE part ADD COLUMN redacted INTEGER NOT NULL DEFAULT 0",
    """CREATE TABLE IF NOT EXISTS redaction (
          id         INTEGER PRIMARY KEY,
          part_id    INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE,
          session_id INTEGER,
          rule       TEXT NOT NULL,
          fingerprint TEXT NOT NULL,
          hits       INTEGER NOT NULL DEFAULT 1,
          at         INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_redaction_rule ON redaction(rule)",
    "CREATE INDEX IF NOT EXISTS idx_redaction_part ON redaction(part_id)",
    "CREATE INDEX IF NOT EXISTS idx_part_redacted ON part(redacted) WHERE redacted > 0",
]


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    group: int = 0        # the span actually replaced; 0 = the whole match
    wide: bool = False
    why: str = ""
    token: bool = True    # match must start a word — enforced in scan_text, see below


# Issuer-prefix rules must start a word, and the first dry run over the real archive
# is why: `sk-[A-Za-z0-9_-]{20,}` matched the tail of
# `MozillaBackgroundTask-308046B0AF4A39CB-backgroundupdate` — the `sk-` came out of
# "Ta**sk-**" — and reported four OpenAI keys inside a Firefox profile listing. A
# prefix rule is specific only because the prefix *starts* a token; mid-word it is two
# letters and a dash, which is nothing.
#
# The check lives in `scan_text` rather than as a `(?<![A-Za-z0-9_\-])` prefix on every
# pattern, because that costs 10x. `re` scans for a pattern's leading literal with a
# fast substring search and only then runs the engine; a lookbehind in front leaves no
# leading literal, so every one of the 20,825 parts gets walked character by character
# instead of skipped. Measured over this archive: 1.3s -> 15.5s for one pass, for a
# guard that is one character comparison after a match that has already been found.
WORD_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _r(name: str, pattern: str, group: int = 0, wide: bool = False,
       why: str = "", flags: int = 0, token: bool = True) -> Rule:
    """`token=True` requires the match to start a word.

    Off for the wide rules, which deliberately match a phrase (`password = ...`,
    a bearer header) rather than a token with an issuer prefix.
    """
    return Rule(name, re.compile(pattern, flags), group, wide, why, token)


# Order matters: the specific prefixes must be tried before anything generic, so an
# Anthropic key is reported as `anthropic_key` and not as a bare `openai_key`.
RULES: list[Rule] = [
    # The {100,} body is not decoration: BEGIN immediately followed by END is prose
    # *about* key files, not a key. The first dry run found exactly that — a
    # 65-character "private key" inside a sentence explaining `ssh-keygen -y -f id_rsa`.
    _r("private_key",
       r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
       r"[\s\S]{100,8000}?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?"
       r"PRIVATE KEY-----",
       why="a whole key block with a body, not just a pair of markers"),
    _r("anthropic_key", r"sk-ant-(?:api|admin)?[A-Za-z0-9_\-]{20,}"),
    _r("openrouter_key", r"sk-or-v1-[a-f0-9]{48,}"),
    # 32 and not 20: an OpenAI key's random tail is 48 characters, and the looser
    # bound was short enough to swallow ordinary hyphenated identifiers off a web page.
    _r("openai_key", r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{32,}"),
    _r("github_token", r"gh[pousr]_[A-Za-z0-9]{30,}"),
    _r("github_pat", r"github_pat_[A-Za-z0-9_]{40,}"),
    # Fixed-length ids need a closing guard as well, or a longer random string is
    # reported as a key that happens to be its first 20 characters.
    _r("aws_access_key",
       r"(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    _r("google_api_key", r"AIza[0-9A-Za-z_\-]{35}(?![A-Za-z0-9_\-])"),
    _r("slack_token", r"xox[abprse]-[A-Za-z0-9\-]{10,}"),
    _r("stripe_key", r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    _r("huggingface_token", r"hf_[A-Za-z0-9]{30,}"),
    _r("xai_key", r"xai-[A-Za-z0-9]{20,}"),
    _r("groq_key", r"gsk_[A-Za-z0-9]{40,}"),
    _r("npm_token", r"npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])"),
    _r("pypi_token", r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}"),
    _r("telegram_token", r"\d{8,10}:AA[A-Za-z0-9_\-]{32,}"),
    # A JWT is only interesting when it is a real three-segment token; the `eyJ` head
    # is base64 for `{"`, which is what makes this specific rather than shape-guessing.
    _r("jwt", r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{20,}"),

    # ---- wide: shapes, not issuers. Opt-in, and expected to misfire sometimes. ----
    _r("url_credentials",
       r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|ftp|https?)://"
       r"[^\s:@/]{1,64}:([^\s@/]{4,})@",
       group=1, wide=True, token=False,
       why="only the password half of a connection string is replaced, so the host "
           "the session was talking to survives for search"),
    _r("bearer_header",
       r"(?i)(?:authorization|proxy-authorization)\s*:\s*(?:bearer|basic|token)\s+"
       r"([A-Za-z0-9_\-\.=+/]{16,})",
       group=1, wide=True, token=False),
    # `(?<![A-Za-z0-9])` and not `\b` in front of the name: `\b` fails on the single
    # commonest shape in a .env dump, `DATABASE_PASSWORD=`, because the character
    # before `PASSWORD` is an underscore and therefore already a word character.
    # Underscore-prefixed is allowed; letter-prefixed (`MYPASSWORD`) is not.
    _r("assigned_secret",
       r"(?i)(?<![A-Za-z0-9])(?:api[_-]?key|secret[_-]?key|client[_-]?secret|"
       r"access[_-]?token|auth[_-]?token|password|passwd|pwd)\b\s*[:=]\s*[\"']?"
       r"([A-Za-z0-9_\-\.+/=]{12,})[\"']?",
       group=1, wide=True, token=False,
       why="catches .env dumps and printenv output; also catches documentation"),
]


def active_rules(wide: bool = False) -> list[Rule]:
    return [r for r in RULES if wide or not r.wide]


def fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8", "replace")).hexdigest()[:8]


def scan_text(text: str, rules: list[Rule]) -> list[tuple[Rule, int, int, str]]:
    """Every match, as (rule, start, end, secret), non-overlapping and left to right.

    Rules are applied in list order and a later rule may not claim a span an earlier
    one already took — that is what keeps `openai_key`'s deliberately loose `sk-` tail
    from re-reporting a string `anthropic_key` has already identified.
    """
    taken: list[tuple[int, int]] = []
    found: list[tuple[Rule, int, int, str]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < end and start < b for start, end in taken)

    for rule in rules:
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            if start < 0 or start == end or overlaps(start, end):
                continue
            # The word-start guard, applied to the *whole match* rather than to the
            # captured group: a wide rule's group 1 legitimately begins mid-string
            # (after `password=`), while a prefix rule's does not.
            if rule.token and match.start() > 0 \
                    and text[match.start() - 1] in WORD_CHARS:
                continue
            secret = match.group(rule.group)
            # Never redact a redaction: re-running the pass must be a no-op, and the
            # placeholder's own hex tail is exactly the shape some rules look for.
            if PLACEHOLDER_RE.fullmatch(secret) or secret.startswith("[redacted:"):
                continue
            taken.append((start, end))
            found.append((rule, start, end, secret))

    found.sort(key=lambda f: f[1])
    return found


def redact_text(text: str, rules: list[Rule]) -> tuple[str, list[tuple[str, str]]]:
    """Return the rewritten text and the (rule name, fingerprint) pairs applied."""
    hits = scan_text(text, rules)
    if not hits:
        return text, []

    out, cursor, applied = [], 0, []
    for rule, start, end, secret in hits:
        fp = fingerprint(secret)
        out.append(text[cursor:start])
        out.append(f"[redacted:{rule.name}:{fp}]")
        cursor = end
        applied.append((rule.name, fp))
    out.append(text[cursor:])
    return "".join(out), applied


@dataclass
class RedactResult:
    parts_scanned: int = 0
    parts_changed: int = 0
    secrets: int = 0
    by_rule: dict = None
    dry_run: bool = False
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.by_rule is None:
            self.by_rule = {}


def is_enabled(con: sqlite3.Connection) -> bool:
    # startswith, not ==: the value is "on" or "on+wide", and an equality test here
    # meant `--enable --wide` reported success and left redaction switched off.
    row = con.execute("SELECT value FROM meta WHERE key=?", (SETTING,)).fetchone()
    return bool(row) and str(row[0]).startswith("on")


def set_enabled(con: sqlite3.Connection, on: bool, wide: bool = False) -> None:
    """Persisted, because ingest undoes redaction by re-reading the source files.

    Stored as `on`/`on+wide`/`off` in one key rather than two: the ruleset is part of
    what "enabled" means, and a separate flag could drift out of step with it.
    """
    value = ("on+wide" if wide else "on") if on else "off"
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (SETTING, value))
    con.commit()


def enabled_wide(con: sqlite3.Connection) -> bool:
    row = con.execute("SELECT value FROM meta WHERE key=?", (SETTING,)).fetchone()
    return bool(row) and row[0] == "on+wide"


def apply(con: sqlite3.Connection, wide: bool = False, dry_run: bool = False,
          only_new: bool = False) -> RedactResult:
    """Scan every stored part and rewrite the ones holding secrets.

    `only_new` limits the pass to parts never scanned, which is what the automated path
    wants — but note that a re-ingested session's parts are deleted and re-inserted, so
    they come back with `redacted = 0` and are picked up again. That is the mechanism
    keeping redaction sticky across ingests, not an accident of the flag.
    """
    t0 = time.perf_counter()
    rules = active_rules(wide)
    result = RedactResult(dry_run=dry_run)
    now = int(time.time() * 1000)

    where = "p.text IS NOT NULL" + (" AND p.redacted = 0" if only_new else "")
    rows = con.execute(f"""
        SELECT p.id, p.text, m.session_id
        FROM part p JOIN message m ON m.id = p.message_id
        WHERE {where}""").fetchall()

    for row in rows:
        result.parts_scanned += 1
        new_text, applied = redact_text(row["text"], rules)
        if not applied:
            continue
        result.parts_changed += 1
        result.secrets += len(applied)
        for name, _ in applied:
            result.by_rule[name] = result.by_rule.get(name, 0) + 1
        if dry_run:
            continue

        con.execute("UPDATE part SET text=?, redacted=redacted+? WHERE id=?",
                    (new_text, len(applied), row["id"]))
        # One row per distinct secret in this part, not per occurrence: the useful
        # question is "which keys leaked and where", and a key repeated 40 times in one
        # .env dump is one leak.
        counts: dict[tuple[str, str], int] = {}
        for pair in applied:
            counts[pair] = counts.get(pair, 0) + 1
        for (name, fp), hits in counts.items():
            con.execute(
                "INSERT INTO redaction(part_id,session_id,rule,fingerprint,hits,at) "
                "VALUES (?,?,?,?,?,?)",
                (row["id"], row["session_id"], name, fp, hits, now))

    if not dry_run:
        con.commit()
    result.seconds = time.perf_counter() - t0
    return result


def scan_blobs(blob_dir: Path, wide: bool = False,
               limit_bytes: int = 8 << 20) -> list[dict]:
    """Report-only pass over the blob store. Nothing is rewritten — see the docstring.

    Blobs are the overflow half of exactly the parts most likely to hold a secret (tool
    results: file reads, command output), so a redaction report that covered only the
    inline text would understate the exposure and read as an all-clear.
    """
    rules = active_rules(wide)
    findings: list[dict] = []
    if not blob_dir.exists():
        return findings

    for path in sorted(blob_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > limit_bytes:
                continue
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        hits = scan_text(text, rules)
        if hits:
            by_rule: dict[str, int] = {}
            for rule, _, _, _ in hits:
                by_rule[rule.name] = by_rule.get(rule.name, 0) + 1
            findings.append({"path": str(path), "hits": len(hits), "by_rule": by_rule})
    return findings


def summary(con: sqlite3.Connection) -> dict:
    """What has already been redacted, for the stats page and `llma redact --status`."""
    try:
        total = con.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT fingerprint) k, COUNT(DISTINCT session_id) s"
            " FROM redaction").fetchone()
        by_rule = con.execute(
            "SELECT rule, COUNT(*) n, COUNT(DISTINCT fingerprint) k FROM redaction "
            "GROUP BY rule ORDER BY n DESC").fetchall()
    except sqlite3.OperationalError:
        return {"enabled": False, "rows": 0, "distinct": 0, "sessions": 0, "by_rule": []}
    return {
        "enabled": is_enabled(con),
        "wide": enabled_wide(con),
        "rows": total["n"], "distinct": total["k"], "sessions": total["s"],
        "by_rule": [dict(r) for r in by_rule],
    }
