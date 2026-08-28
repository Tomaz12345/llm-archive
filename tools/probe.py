"""Phase 0 format probe.

Walks every local session store, and reports:
  * every record `type` seen, with counts
  * the distinct top-level key-sets per type (a "shape")
  * one truncated example per shape, written to data/fixtures/
  * DAG statistics for the tree-structured sources (risk R1)
  * resume/fork linkage (risk R2)

Design rule, inherited from R5: this must never crash on an unknown shape.
Anything unrecognised is counted and reported, never fatal.

Usage:
    python tools/probe.py                 # all sources
    python tools/probe.py claude_code     # one source
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"
REPORTS = ROOT / "docs" / "formats"

MAX_STR = 200  # truncate long strings in dumped examples (also blunts secret leakage)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def truncate(obj, depth: int = 0):
    """Recursively shorten a JSON value so an example stays readable."""
    if depth > 6:
        return "<deep>"
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"...<+{len(obj) - MAX_STR} chars>"
    if isinstance(obj, list):
        head = [truncate(x, depth + 1) for x in obj[:3]]
        if len(obj) > 3:
            head.append(f"<+{len(obj) - 3} more items>")
        return head
    if isinstance(obj, dict):
        return {k: truncate(v, depth + 1) for k, v in obj.items()}
    return obj


def keyset(d) -> str:
    return ",".join(sorted(d.keys())) if isinstance(d, dict) else f"<{type(d).__name__}>"


def iter_jsonl(path: Path):
    """Yield (lineno, parsed) for each line; bad lines yield (lineno, None)."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError:
                    yield i, None
    except OSError as exc:
        print(f"  ! cannot read {path}: {exc}")


class Probe:
    """Accumulates shapes, examples and free-form notes for one source."""

    def __init__(self, name: str):
        self.name = name
        self.types = Counter()
        self.shapes = defaultdict(Counter)      # type -> keyset -> count
        self.examples = {}                      # (type, keyset) -> example
        self.notes = {}                         # heading -> Counter/dict/str
        self.errors = Counter()
        self.files = 0
        self.records = 0

    def record(self, rec, rtype: str | None = None):
        self.records += 1
        if rec is None:
            self.errors["json_decode_error"] += 1
            return
        t = rtype if rtype is not None else str(rec.get("type", "<no type field>"))
        ks = keyset(rec)
        self.types[t] += 1
        self.shapes[t][ks] += 1
        self.examples.setdefault((t, ks), truncate(rec))

    def note(self, heading: str, value):
        self.notes[heading] = value


# --------------------------------------------------------------------------
# claude code
# --------------------------------------------------------------------------

def probe_claude_code() -> Probe | None:
    root = HOME / ".claude" / "projects"
    if not root.exists():
        return None
    p = Probe("claude_code")

    versions = Counter()
    types_by_version = defaultdict(Counter)
    content_blocks = Counter()
    tool_names = Counter()
    roles = Counter()
    session_stats = []
    projects = Counter()
    entrypoints = Counter()
    sidechain = Counter()
    bridge_examples = []

    for path in sorted(root.rglob("*.jsonl")):
        p.files += 1
        projects[path.parent.name] += 1

        parents: dict[str, str | None] = {}
        seen_uuids: set[str] = set()
        n_records = 0
        ts_min = ts_max = None

        for _, rec in iter_jsonl(path):
            if rec is None:
                p.record(None)
                continue
            p.record(rec)
            n_records += 1

            v = rec.get("version")
            if v:
                versions[v] += 1
                types_by_version[v][str(rec.get("type"))] += 1
            if rec.get("entrypoint"):
                entrypoints[rec["entrypoint"]] += 1
            if "isSidechain" in rec:
                sidechain[bool(rec["isSidechain"])] += 1

            ts = rec.get("timestamp")
            if isinstance(ts, str):
                ts_min = ts if ts_min is None or ts < ts_min else ts_min
                ts_max = ts if ts_max is None or ts > ts_max else ts_max

            # DAG edges
            uuid = rec.get("uuid")
            if isinstance(uuid, str):
                seen_uuids.add(uuid)
                parents[uuid] = rec.get("parentUuid")

            if rec.get("type") == "bridge-session" and len(bridge_examples) < 3:
                bridge_examples.append(truncate(rec))

            # message internals
            msg = rec.get("message")
            if isinstance(msg, dict):
                if msg.get("role"):
                    roles[msg["role"]] += 1
                content = msg.get("content")
                if isinstance(content, str):
                    content_blocks["<bare string>"] += 1
                elif isinstance(content, list):
                    for blk in content:
                        if not isinstance(blk, dict):
                            content_blocks[f"<{type(blk).__name__}>"] += 1
                            continue
                        bt = str(blk.get("type"))
                        content_blocks[bt] += 1
                        p.shapes[f"content:{bt}"][keyset(blk)] += 1
                        p.examples.setdefault((f"content:{bt}", keyset(blk)), truncate(blk))
                        if bt == "tool_use":
                            tool_names[str(blk.get("name"))] += 1

        # per-session DAG shape
        children = defaultdict(list)
        for uid, par in parents.items():
            children[par].append(uid)
        roots = [u for u, par in parents.items() if par is None or par not in seen_uuids]
        leaves = [u for u in parents if not children.get(u)]
        branch_points = [u for u, kids in children.items() if u is not None and len(kids) > 1]

        session_stats.append({
            "file": path.name,
            "project": path.parent.name,
            "records": n_records,
            "nodes": len(parents),
            "roots": len(roots),
            "leaves": len(leaves),
            "branch_points": len(branch_points),
            "started": ts_min,
            "ended": ts_max,
        })

    branched = [s for s in session_stats if s["leaves"] > 1]
    multiroot = [s for s in session_stats if s["roots"] > 1]

    # Sidecar directories the *.jsonl glob never sees: externalised tool output and
    # per-project memory files.
    persisted = list(root.glob("*/*/tool-results/*"))
    memories = list(root.glob("*/memory/*"))
    cwd_per_project: dict[str, Counter] = {}
    for path in root.rglob("*.jsonl"):
        c = cwd_per_project.setdefault(path.parent.name, Counter())
        for _, rec in iter_jsonl(path):
            if rec and rec.get("cwd"):
                c[rec["cwd"]] += 1
    multi_cwd = {k: dict(v) for k, v in cwd_per_project.items() if len(v) > 1}

    p.note("persisted tool-results (sidecar files)", {
        "dirs": len({f.parent for f in persisted}),
        "files": len(persisted),
        "bytes": sum(f.stat().st_size for f in persisted if f.is_file()),
        "note": "referenced from tool_result via a <persisted-output> marker; "
                "resolve relative to the session dir, not the embedded absolute path",
    })
    p.note("per-project memory files", [str(f.relative_to(root)) for f in memories])
    p.note("R-workspace — project dirs with >1 cwd", multi_cwd)

    p.note("cli versions seen", versions)
    p.note("record types by version", {v: dict(c) for v, c in sorted(types_by_version.items())})
    p.note("message content block types", content_blocks)
    p.note("message roles", roles)
    p.note("tool_use names", tool_names)
    p.note("entrypoints", entrypoints)
    p.note("isSidechain values", {str(k): v for k, v in sidechain.items()})
    p.note("files per project", projects)
    p.note("R1 — DAG shape", {
        "sessions": len(session_stats),
        "sessions with >1 leaf (real branching)": len(branched),
        "sessions with >1 root (orphaned records)": len(multiroot),
        "total branch points": sum(s["branch_points"] for s in session_stats),
        "total leaves": sum(s["leaves"] for s in session_stats),
        "total DAG nodes": sum(s["nodes"] for s in session_stats),
    })
    p.note("R1 — most-branched sessions", sorted(
        branched, key=lambda s: s["leaves"], reverse=True)[:10])
    p.note("R2 — bridge-session examples", bridge_examples)
    p.note("session inventory", session_stats)
    return p


# --------------------------------------------------------------------------
# codex
# --------------------------------------------------------------------------

def probe_codex() -> Probe | None:
    root = HOME / ".codex" / "sessions"
    if not root.exists():
        return None
    p = Probe("codex")

    payload_types = Counter()
    originators = Counter()
    cli_versions = Counter()
    providers = Counter()
    response_roles = Counter()
    token_examples = []

    for path in sorted(root.rglob("*.jsonl")):
        p.files += 1
        for _, rec in iter_jsonl(path):
            if rec is None:
                p.record(None)
                continue
            outer = str(rec.get("type"))
            payload = rec.get("payload")
            ptype = payload.get("type") if isinstance(payload, dict) else None
            label = f"{outer}/{ptype}" if ptype else outer
            p.record(rec, rtype=label)
            payload_types[label] += 1

            if isinstance(payload, dict):
                p.shapes[f"payload:{label}"][keyset(payload)] += 1
                p.examples.setdefault(
                    (f"payload:{label}", keyset(payload)), truncate(payload))
                if outer == "session_meta":
                    originators[str(payload.get("originator"))] += 1
                    cli_versions[str(payload.get("cli_version"))] += 1
                    providers[str(payload.get("model_provider"))] += 1
                if ptype == "token_count" and len(token_examples) < 3:
                    token_examples.append(truncate(payload))
                if outer == "response_item" and payload.get("role"):
                    response_roles[str(payload["role"])] += 1

    index = HOME / ".codex" / "session_index.jsonl"
    idx_rows = 0
    idx_example = None
    if index.exists():
        for _, rec in iter_jsonl(index):
            if rec is None:
                continue
            idx_rows += 1
            if idx_example is None:
                idx_example = truncate(rec)

    p.note("envelope/payload types", payload_types)
    p.note("originators", originators)
    p.note("cli versions", cli_versions)
    p.note("model providers", providers)
    p.note("response_item roles", response_roles)
    p.note("token_count payload examples", token_examples)
    p.note("session_index.jsonl", {"rows": idx_rows, "example": idx_example})
    return p


# --------------------------------------------------------------------------
# opencode
# --------------------------------------------------------------------------

def probe_opencode() -> Probe | None:
    candidates = [
        HOME / ".local" / "share" / "opencode" / "storage",
        Path(os.environ.get("APPDATA", "")) / "opencode" / "storage",
    ]
    root = next((c for c in candidates if c.exists()), None)
    if root is None:
        return None
    p = Probe("opencode")

    part_types = Counter()
    roles = Counter()
    models = Counter()
    agents = Counter()
    providers = Counter()
    entity_counts = Counter()

    for entity_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        entity = entity_dir.name
        for path in sorted(entity_dir.rglob("*.json")):
            p.files += 1
            entity_counts[entity] += 1
            try:
                rec = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError):
                p.record(None)
                continue
            if isinstance(rec, list):
                # todo/ and session_diff/ store JSON arrays, not objects.
                p.types[f"{entity}[]"] += 1
                p.shapes[f"{entity}[]"][keyset(rec[0]) if rec else "<empty>"] += 1
                if rec:
                    p.examples.setdefault(
                        (f"{entity}[]", keyset(rec[0])), truncate(rec))
                p.records += 1
                continue
            if not isinstance(rec, dict):
                p.errors[f"{entity}:non-dict"] += 1
                continue
            p.record(rec, rtype=entity)

            if entity == "part":
                pt = str(rec.get("type"))
                part_types[pt] += 1
                p.shapes[f"part:{pt}"][keyset(rec)] += 1
                p.examples.setdefault((f"part:{pt}", keyset(rec)), truncate(rec))
            elif entity == "message":
                roles[str(rec.get("role"))] += 1
                if rec.get("agent"):
                    agents[str(rec["agent"])] += 1
                model = rec.get("model")
                if isinstance(model, dict):
                    models[str(model.get("modelID"))] += 1
                    providers[str(model.get("providerID"))] += 1

    p.note("entity file counts", entity_counts)
    p.note("part types", part_types)
    p.note("message roles", roles)
    p.note("models used", models)
    p.note("providers", providers)
    p.note("agents", agents)
    return p


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def write_report(p: Probe) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (FIXTURES / p.name).mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Format probe — `{p.name}`",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"by `tools/probe.py`.",
        "",
        f"- files scanned: **{p.files}**",
        f"- records parsed: **{p.records}**",
        f"- parse errors: **{sum(p.errors.values())}**"
        + (f" — {dict(p.errors)}" if p.errors else ""),
        "",
        "## Record types",
        "",
        "| type | count | distinct key-sets |",
        "|---|---:|---:|",
    ]
    for t, n in p.types.most_common():
        lines.append(f"| `{t}` | {n} | {len(p.shapes.get(t, {}))} |")

    lines += ["", "## Shapes", ""]
    for t in sorted(p.shapes):
        lines.append(f"### `{t}`")
        lines.append("")
        for ks, n in p.shapes[t].most_common():
            pretty = ", ".join(f"`{k}`" for k in ks.split(",")) if ks else "_(empty)_"
            lines.append(f"- **{n}×** — {pretty}")
        lines.append("")

    if p.notes:
        lines += ["## Observations", ""]
        for heading, value in p.notes.items():
            lines.append(f"### {heading}")
            lines.append("")
            if isinstance(value, Counter):
                for k, n in value.most_common():
                    lines.append(f"- `{k}` — {n}")
            elif isinstance(value, (dict, list)):
                lines.append("```json")
                lines.append(json.dumps(value, indent=2, ensure_ascii=False)[:6000])
                lines.append("```")
            else:
                lines.append(str(value))
            lines.append("")

    report = REPORTS / f"{p.name}.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    # fixtures: one example per shape
    fixture = {f"{t}::{ks}": ex for (t, ks), ex in sorted(p.examples.items())}
    (FIXTURES / p.name / "shapes.json").write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


PROBES = {
    "claude_code": probe_claude_code,
    "codex": probe_codex,
    "opencode": probe_opencode,
}


def main() -> int:
    wanted = sys.argv[1:] or list(PROBES)
    unknown = [w for w in wanted if w not in PROBES]
    if unknown:
        print(f"unknown source(s): {unknown}. known: {list(PROBES)}")
        return 2

    summary = []
    for name in wanted:
        print(f"\n=== {name} ===")
        probe = PROBES[name]()
        if probe is None:
            print("  store not present on this machine — skipped")
            summary.append((name, "absent", 0, 0, 0))
            continue
        report = write_report(probe)
        print(f"  files={probe.files}  records={probe.records}  "
              f"types={len(probe.types)}  shapes={sum(len(s) for s in probe.shapes.values())}  "
              f"errors={sum(probe.errors.values())}")
        print(f"  report   -> {report.relative_to(ROOT)}")
        print(f"  fixtures -> {(FIXTURES / name / 'shapes.json').relative_to(ROOT)}")
        summary.append((name, "ok", probe.files, probe.records,
                        sum(probe.errors.values())))

    print("\n--- summary ---")
    for name, status, files, records, errors in summary:
        print(f"  {name:14s} {status:8s} files={files:<5} records={records:<7} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
