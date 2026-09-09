"""Does Claude Code's `--resume` fork a new file or append to the old one?

PLAN.md.bak §8.2 downgraded resumed-session handling on a measurement of ~2.5%
cross-file message duplication, i.e. "resume does not replay history, so a resumed
session is a genuinely new one". That measurement predates several Claude Code releases
and the answer decides whether §5 of the import plan is real work or a docs note:

* **fork** — a resume writes a NEW <uuid>.jsonl whose leading records carry the SAME
  message uuids as an earlier file. Two session rows describe one conversation, the
  shared prefix is counted twice in every statistic, and the pair needs linking through
  `session.parent_session_id`.
* **append** — a resume keeps writing into the original file. Nothing to link; the
  merge upsert already handles it, because the file's raw_hash changes and the extra
  messages arrive as an append to the same native_id.

**Answered: it forks** — as of 2026-09-09, over 14 projects / 154 files / 36,586 records.
An earlier run over 9 / 93 / 23,735 found no forks at all, so this is a fact with a date
on it: Claude Code changed, and it may change again.

The linkage that answer called for is `llm_archive/core/lineage.py`, which asks the same
question of the archive instead of the file tree — over `message.native_id`, so it covers
all twelve sources rather than just this one, and it can write the answer down
(`session.continues_session_id` plus `message.superseded`). Run `llma lineage` for that.
This probe stays because it needs no database and can be pointed at any `.claude` tree,
including one copied off another machine before it has ever been ingested.

Read-only. Writes nothing, touches no database.

    uv run python tools/probe_resume.py [--root ~/.claude/projects] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# A file whose first records duplicate another's, but only barely, is coincidence
# (both sessions opened on the same file-history snapshot). This many shared leading
# uuids is not coincidence.
MIN_PREFIX = 3


def uuids(path: Path) -> list[str]:
    """Message uuids in file order, every record type included.

    The DAG spans all record types — see the note at claude_code.py:_parse_file — so a
    probe that looked only at conversational records would miss exactly the
    file-history-snapshot records a resume replays first.
    """
    out: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("uuid"):
            out.append(rec["uuid"])
    return out


def shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def probe(root: Path) -> dict:
    projects = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []

    files_total = 0
    pairs_examined = 0
    forks: list[dict] = []
    appended_in_place = 0        # files that grew but stayed one file: not visible here
    overlap_any: list[dict] = [] # shared uuids anywhere, not necessarily a prefix
    per_project: Counter = Counter()

    for proj in projects:
        paths = sorted(proj.glob("*.jsonl"))
        if not paths:
            continue
        seqs = {p: uuids(p) for p in paths}
        files_total += len(paths)

        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                sa, sb = seqs[a], seqs[b]
                if not sa or not sb:
                    continue
                pairs_examined += 1

                pre = shared_prefix(sa, sb)
                if pre >= MIN_PREFIX:
                    # Whichever file is longer is the continuation: the resume replays
                    # the prefix and then keeps going.
                    parent, child = (a, b) if len(sa) < len(sb) else (b, a)
                    forks.append({
                        "project": proj.name,
                        "parent": parent.stem,
                        "child": child.stem,
                        "shared_prefix": pre,
                        "parent_len": len(seqs[parent]),
                        "child_len": len(seqs[child]),
                        "prefix_is_whole_parent": pre == len(seqs[parent]),
                    })
                    per_project[proj.name] += 1
                    continue

                common = len(set(sa) & set(sb))
                if common:
                    overlap_any.append({
                        "project": proj.name, "a": a.stem, "b": b.stem,
                        "shared": common, "a_len": len(sa), "b_len": len(sb),
                    })

    dup_msgs = sum(f["shared_prefix"] for f in forks)
    all_msgs = sum(len(uuids(p)) for proj in projects for p in proj.glob("*.jsonl"))

    return {
        "root": str(root),
        "projects": len(projects),
        "files": files_total,
        "pairs_examined": pairs_examined,
        "forks": forks,
        "fork_count": len(forks),
        "overlap_without_prefix": overlap_any,
        "duplicated_messages": dup_msgs,
        "total_messages": all_msgs,
        "duplicated_pct": round(100 * dup_msgs / all_msgs, 2) if all_msgs else 0.0,
        "appended_in_place": appended_in_place,
        "busiest_projects": per_project.most_common(10),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    ap.add_argument("--json", action="store_true", help="emit the raw findings")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"no such root: {args.root}")
        return 1

    res = probe(args.root)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"root            {res['root']}")
    print(f"projects        {res['projects']}")
    print(f"files           {res['files']}")
    print(f"pairs examined  {res['pairs_examined']}")
    print(f"total messages  {res['total_messages']}")
    print()
    print(f"FORKS (>= {MIN_PREFIX} shared leading uuids): {res['fork_count']}")
    print(f"  duplicated messages: {res['duplicated_messages']} "
          f"({res['duplicated_pct']}% of the corpus)")
    whole = sum(1 for f in res["forks"] if f["prefix_is_whole_parent"])
    print(f"  of those, {whole} replay the parent file in full "
          f"(a clean continuation; the rest branch mid-session)")
    for f in res["forks"][:15]:
        mark = "=" if f["prefix_is_whole_parent"] else "~"
        print(f"    {mark} {f['project'][:40]:40} {f['parent'][:8]}({f['parent_len']})"
              f" -> {f['child'][:8]}({f['child_len']})  shared {f['shared_prefix']}")
    if res["fork_count"] > 15:
        print(f"    ... and {res['fork_count'] - 15} more")
    print()
    print(f"overlap without a shared prefix: {len(res['overlap_without_prefix'])}")
    for o in res["overlap_without_prefix"][:8]:
        print(f"    {o['project'][:40]:40} {o['a'][:8]} / {o['b'][:8]}  "
              f"shared {o['shared']}")
    print()
    if res["fork_count"]:
        print("VERDICT: resume FORKS. Continuations need linking via "
              "session.parent_session_id (plan §5.2).")
    else:
        print("VERDICT: no forks found. Resume appends in place; the merge upsert "
              "(plan §4c) already covers it and §5 reduces to a docs note.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
