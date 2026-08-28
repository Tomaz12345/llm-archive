"""Structural validation of the persisted message DAG.

Every DAG-shaped adapter (claude_code, claude_web, copilot_web, deepseek, grok,
openrouter) resolves one active path and persists every node, flagging the rest via
`message.on_active_path`. That resolution happens once, at ingest, over the adapter's
own raw graph -- and a bug in it does not raise anything (docs/phase0-findings.md §4b:
an earlier resolver shattered one 764-node session into 33 false roots and collapsed
its active path to a single message, silently discarding ~11,000 messages archive-wide).
This module re-derives the DAG's shape from what is actually in `message`, independent
of ingest, so that class of regression is caught by asking the database rather than by
trusting the adapter that just ran.

`message.parent_native_id` is not reliably a foreign key into `message.native_id` within
a session, and for three sources that is by design, not corruption:
  - claude_code: parent chains hop through bookkeeping records (file-history-snapshot,
    etc.) that are never persisted as message rows -- every session shows this.
  - claude_web: a rare dangling parent points at a turn that produced zero content
    parts and was dropped before persistence.
  - grok: the export never includes the conversation root; a dangling parent there is
    the root, not corruption.
  - gemini: not a DAG at all -- an activity log. `_messages()` sets `parent_native_id`
    only to pair one reply with its own prompt within a single activity cell; cells are
    never chained to each other, so a session legitimately has as many local roots as
    it has cells.
Five other sources (codex, opencode, vscode_chat, t3chat, mistral) never persist
`parent_native_id` at all -- they are naturally linear exports with nothing to resolve --
so the tree-shape checks below are skipped for a session with no non-null parent link at
all, rather than hardcoding which sources are "DAG sources".
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby

ERROR, WARNING, INFO = "error", "warning", "info"

# Sources with a documented, expected reason `parent_native_id` won't always resolve
# and their active path can show more than one local root. Anything not listed here
# is expected to resolve cleanly, and a session that doesn't is a regression, not a
# footnote.
KNOWN_PARENT_GAPS: dict[str, str] = {
    "claude_code": "parent chains hop through bookkeeping records (file-history-"
                   "snapshot, etc.) never persisted as message rows",
    "claude_web": "a rare dangling parent points at a turn that produced zero content "
                  "parts and was dropped before persistence",
    "grok": "the export never includes the conversation root -- a dangling parent "
            "there is the root, not corruption",
    "gemini": "not a DAG -- an activity log where parent_native_id only pairs a reply "
              "with its own prompt inside one cell; cells are never chained",
}

# Below this many non-sidechain messages, an active/total ratio is too noisy to mean
# anything (a 3-message session with 1 active message is unremarkable).
ACTIVE_COLLAPSE_MIN_TOTAL = 20
# The historical bug collapsed 764 nodes to 1 active (~0.1%); this floor is deliberately
# loose so it fires on that class of regression without flagging heavily-edited sessions.
ACTIVE_COLLAPSE_MAX_RATIO = 0.10


@dataclass(frozen=True)
class Finding:
    severity: str        # error | warning | info
    code: str
    session_id: int
    source_kind: str
    native_id: str
    detail: str


def check_session(con: sqlite3.Connection, session_id: int) -> list[Finding]:
    """Structural findings for one session, straight from `message`."""
    head = con.execute(
        "SELECT s.native_id, src.kind FROM session s "
        "JOIN source src ON src.id = s.source_id WHERE s.id = ?",
        (session_id,)).fetchone()
    if head is None:
        raise ValueError(f"no session #{session_id}")
    rows = con.execute(
        "SELECT native_id, parent_native_id, on_active_path, is_sidechain "
        "FROM message WHERE session_id = ?", (session_id,)).fetchall()
    return _evaluate(session_id, head["native_id"], head["kind"], rows)


def check_all(con: sqlite3.Connection, source_kind: str | None = None) -> list[Finding]:
    """Findings across every session, one query rather than one per session."""
    rows = con.execute("""
        SELECT m.session_id sid, s.native_id snid, src.kind kind,
               m.native_id, m.parent_native_id, m.on_active_path, m.is_sidechain
        FROM message m JOIN session s ON s.id = m.session_id
                       JOIN source src ON src.id = s.source_id
        WHERE (? IS NULL OR src.kind = ?)
        ORDER BY m.session_id""", (source_kind, source_kind)).fetchall()

    findings: list[Finding] = []
    for sid, group in groupby(rows, key=lambda r: r["sid"]):
        group = list(group)
        findings.extend(_evaluate(sid, group[0]["snid"], group[0]["kind"], group))
    return findings


def summarize(findings: list[Finding]) -> dict:
    """Counts for `llma validate` / `llma doctor --dag`."""
    by_source: dict[str, Counter] = defaultdict(Counter)
    for f in findings:
        by_source[f.source_kind][f.severity] += 1
    return {
        "total": len(findings),
        "by_severity": dict(Counter(f.severity for f in findings)),
        "by_code": dict(Counter(f.code for f in findings)),
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


# -- the checks themselves --------------------------------------------------

def _evaluate(session_id: int, native_id: str, source_kind: str, rows) -> list[Finding]:
    out: list[Finding] = []
    if not rows:
        return out

    def find(severity, code, detail):
        out.append(Finding(severity, code, session_id, source_kind, native_id, detail))

    # 1. cycles -- hard, universal, cheap even when there is nothing to find
    culprit = _find_cycle(rows)
    if culprit is not None:
        find(ERROR, "cycle", f"parent_native_id cycle reachable from {culprit!r}")

    nonside = [r for r in rows if not r["is_sidechain"]]
    active = [r for r in nonside if r["on_active_path"]]

    # 2. a session cannot have messages and zero of them active
    if nonside and not active:
        find(ERROR, "no_active_path",
             f"{len(nonside)} message(s), none on the active path")

    # 3. the historical-bug detector: active/total collapses on a session big enough
    #    for that ratio to be meaningful
    if len(nonside) >= ACTIVE_COLLAPSE_MIN_TOTAL:
        ratio = len(active) / len(nonside)
        if ratio < ACTIVE_COLLAPSE_MAX_RATIO:
            find(WARNING, "active_path_collapse",
                 f"active path is {len(active)}/{len(nonside)} messages ({ratio:.1%}) "
                 "-- check for a DAG-resolution regression (docs/phase0-findings.md §4b)")

    # Tree-shape checks only apply when this session actually declares a parent chain;
    # six adapters never set parent_native_id at all, and every message would trivially
    # look like its own root/leaf there.
    if any(r["parent_native_id"] is not None for r in nonside):
        all_roots, _, _ = _shape(nonside)
        if len(all_roots) > 1:
            find(INFO, "multi_root_session",
                 f"{len(all_roots)} distinct root(s) among all persisted messages")

        if active:
            act_roots, act_leaves, act_children = _shape(active)
            # A shared parent with >1 active child is real forking -- always wrong,
            # since "active path" means a path, not a subtree. A plain leaf-count of
            # >1 is NOT checked here: with single-parent pointers, disjoint simple
            # chains (leaf count == root count, no branch points) are structurally the
            # same phenomenon as multi_root_active_path below, and get that check's
            # source-aware severity instead of double-flagging as an unconditional
            # error (claude_code's interior bookkeeping hops routinely split one
            # logical active path into several disjoint on_active_path=1 segments).
            branch_points = [p for p, n in act_children.items() if n > 1]
            if branch_points:
                find(ERROR, "active_path_branches",
                     f"{len(act_leaves)} leaf/leaves, {len(branch_points)} branch "
                     "point(s) among on_active_path=1 rows -- expected one linear chain")

            if len(act_roots) > 1:
                if source_kind in KNOWN_PARENT_GAPS:
                    find(INFO, "multi_root_active_path",
                         f"{len(act_roots)} root(s) in the active path -- expected for "
                         f"{source_kind} ({KNOWN_PARENT_GAPS[source_kind]})")
                else:
                    find(ERROR, "multi_root_active_path",
                         f"{len(act_roots)} root(s) in the active path on a source with "
                         "no known parent-resolution gap")
    return out


def _shape(rows) -> tuple[set, set, dict]:
    """Roots, leaves and per-parent child counts over one set of message rows."""
    ids = {r["native_id"] for r in rows if r["native_id"] is not None}
    children: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["parent_native_id"] in ids:
            children[r["parent_native_id"]] += 1
    roots = {r["native_id"] for r in rows if r["parent_native_id"] not in ids}
    leaves = {r["native_id"] for r in rows if children.get(r["native_id"], 0) == 0}
    return roots, leaves, children


def _find_cycle(rows) -> str | None:
    """A node must never be its own ancestor. O(n): each node visited at most twice.

    Same walk-up shape as `_tree.resolve_active_path`, generalised across every node
    rather than starting from a single leaf, with visited nodes marked done so a session
    with many independent chains is still linear time.
    """
    parent_of = {r["native_id"]: r["parent_native_id"]
                for r in rows if r["native_id"] is not None}
    done: set = set()
    for start in parent_of:
        if start in done:
            continue
        path, cur = [], start
        while cur is not None and cur in parent_of and cur not in done:
            if cur in path:
                return cur
            path.append(cur)
            cur = parent_of[cur]
        done.update(path)
    return None
