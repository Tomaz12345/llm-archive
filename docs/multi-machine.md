# Sessions from other machines

Claude Code, Codex, opencode and VS Code all store transcripts **locally**, and none of
them records which machine produced them. So two things are needed: get the files here,
and say where they came from.

```bash
llma ingest --source claude_code --root D:/from-laptop/.claude --host laptop
llma ingest --source codex       --root D:/from-laptop/.codex  --host laptop
llma ingest --source opencode    --root D:/from-laptop/opencode-storage --host laptop
llma ingest --source vscode_chat --root D:/from-laptop/Code/User --host laptop
```

`--host` is not cosmetic. Nothing in any of these formats carries a hostname, so without
it every machine's sessions collapse into one indistinguishable pile in the statistics.
The command refuses `--root` without it.

That is a one-off read of a folder you copied. For a machine you cannot reach and will
want sessions from again, use a [bundle](#a-machine-you-cannot-reach-bundles) instead.

## Some of it is already here

Before copying anything, check what you have. **VS Code stores chats from remote SSH
sessions on the local disk**, tagged with a `vscode-remote://ssh-remote+<host>/…` folder
URI. The adapter reads that authority and attributes the session to the machine it
actually ran on:

```
BY MACHINE
  DESKTOP-DQMQ75J     84 sessions   10,722 msgs
  (web)               53 sessions      316 msgs
  devbox                 26 sessions      180 msgs     <- remote, already local
  vicos-proxy-30055    9 sessions       78 msgs     <- remote, already local
```

35 sessions from two other machines, captured without copying a thing. WSL and
dev-container workspaces are attributed the same way (`wsl:Ubuntu`, `dev-container:…`).

This does **not** extend to Claude Code. Its transcripts are JSONL in `~/.claude/projects`
on whichever machine ran it, including the VS Code extension over Remote-SSH, which runs
on the remote. Nothing of it reaches this disk.

## A machine you cannot reach: bundles

No SSH route back — a laptop, a work box, the one Claude Code was tried out on — and
sessions you will want from it more than once. Pack them there, carry the zip, add it here:

```bash
llma pack --script E:/            # once: put pack-machine.ps1 on the USB stick
```

```powershell
# on the other machine: Windows PowerShell 5.1, nothing to install. Zip lands on the Desktop.
powershell -ExecutionPolicy Bypass -File pack-machine.ps1
```

```bash
llma add E:/llma-machine-laptop-7q2-20261006-1530.zip
llma machines                     # what each holds, and when its next bundle is due
```

Where llma is installed on the other machine, `llma pack` there does what the script does.

What makes this more than `--root`:

- **The bundle names its machine.** The packer writes the machine's own hostname into
  the zip (`llma-machine.json`), so there is nothing to type and nothing to type
  differently next month. Filed under another name once with `llma add --host`, the
  machine keeps that name: later bundles find it by the hostname they carry.
- **It is merged, and kept.** A bundle is merged into `data/machines/<host>/`: new
  transcripts are added, grown ones replaced, and a transcript the bundle no longer
  carries is kept. That last rule is the point. Claude Code deletes a transcript 30 days
  after its last activity by default, so a later bundle has forgotten sessions that the
  archive must not. Files only move forward in time, so an older bundle found and added
  late cannot overwrite a newer copy.
- **Every `llma sync` re-reads it** under its own host, like a local store. The zip
  itself is not kept; the merged tree is the copy, and `raw_path` points into it.
- **Only transcripts travel.** The packer reads exactly the files the adapters read
  (`RULES` in `core/machines.py`): not `.credentials.json`, `auth.json`, Claude's
  `memory/`, or the rest of those folders, and of `settings.json` only
  `cleanupPeriodDays`. The unpacker applies the same list, so a zip that carries more
  still lands only the transcripts. The path rules are written out twice, in Python and
  in the PowerShell script, and `tests/test_machines.py` runs the script to check that
  the two still agree.

**When the next one is due.** A week before Claude Code there would start deleting:
every 23 days at its default retention, and at most every 30 whatever it is set to. The
nightly sync logs `BUNDLE DUE` for a machine that is late. When retention is at its
default, the packer says so on the other machine, which is the only place it can be
changed:

```json
// ~/.claude/settings.json on the other machine
{ "cleanupPeriodDays": 365 }
```

**No script?** Zip the other machine's `.claude` folder by hand and name the machine
when adding it: `llma add claude.zip --host laptop`. Only `projects/` is read from it,
but the zip itself then carries `.credentials.json` on whatever it travels on — which
is what the script exists to avoid.

**This machine's own bundle is refused**, whatever name it is given. Its sessions are
read live, and a second copy under another host would flip between the two on every
sync.

The same caution as below applies to how the zip travels. A USB stick is fine. Mailing
it to yourself is cloud storage under another name, so encrypt it first, as for any
cloud folder (see R6 below).

## What to copy

| Source | Copy this directory |
|---|---|
| Claude Code | `~/.claude` (or just `~/.claude/projects`) |
| Codex | `~/.codex` — needs `sessions/` **and** `session_index.jsonl` for titles |
| opencode | `~/.local/share/opencode/storage` |
| VS Code | `%APPDATA%/Code/User` (or `~/.config/Code/User`) |

`--root` accepts either the parent or the inner directory for Claude Code and opencode;
it looks for `projects/` and `storage/` respectively.

**Do not copy `~/.claude/settings.json`, `~/.codex/auth.json`, or opencode's `auth.json`.**
Those are credentials, not transcripts, and the archive has no use for them.

## Getting the files across

Any of these work; the archive only cares that the tree lands somewhere readable:

- **USB drive** — simplest, and nothing touches a network.
- **`scp` / `rsync`** — `rsync -a laptop:~/.claude/ D:/from-laptop/.claude/`
- **Syncthing or a LAN share** — good if you do this regularly.

**Think before using cloud sync for the transfer.** Per risk R6, these files contain API
keys, `.env` contents and private code that leaked into tool output. If you must use
Dropbox/OneDrive/Drive, put the tree in an encrypted archive first
(`7z a -p -mhe=on sessions.7z .claude`) rather than syncing it in the clear.

## Re-running

Ingest is idempotent: sessions are keyed by `(source, native_id)` and skipped when the
source bytes are unchanged. Session ids are UUIDs, so **the same session imported twice
from two copies deduplicates instead of doubling.** Re-importing after new work on the
laptop only writes what changed.

One exception: if the *adapter* changes rather than the data, the hash short-circuit
would skip the fix. Use `--force` to re-parse everything:

```bash
llma ingest --source vscode_chat --force
llma index                       # NOT optional — see below
```

## Always re-index after ingesting

An upsert deletes a session's messages and parts and writes them again, so **part ids
move**. Both indexes are keyed on those ids and neither notices:

- **Keyword** — `part_fts` is external-content (`content='part'`), addressed by rowid.
  A stale index keeps matching old rowids against whatever rows now hold them, so search
  returns *wrong* passages rather than none. This happened: after one `--force`, a term
  present in a Copilot session returned zero hits, while unrelated sessions matched.
- **Vector** — every `chunk` row for a re-ingested session points at a part id that no
  longer exists, so those sessions silently drop out of semantic results.

Neither raises. `llma index` rebuilds both, and since v5 it records the build in
`index_run`, which is what lets `/stats` and the search page say whether the index still
matches the archive. The web UI shows:

| state | meaning |
|---|---|
| **Up to date** | built after the last ingest, with full vector coverage |
| **Out of date** | sessions ingested since the build, or embeddable parts with no chunk |
| **Never built** | no index at all — search returns nothing |
| **Unknown** | a real index with no recorded build date (any archive from before v5) |

A deliberate `--no-vectors` build reads as up to date with a caveat, not as stale: its
keyword index really is current, and crying wolf there would train you to ignore the
warning that matters.

Full rebuilds are not cheap — 10.5k chunks took 3.6 hours on CPU — so budget for it
before forcing a re-ingest of a large source.

## Workspaces across machines

Workspace keys are casefolded paths, so `C:\...\telemetry_analysis` and
`c:\...\telemetry_analysis` collapse into one project — that mattered even on a single
machine, where 4 of 9 Claude Code project directories held case-variant cwds.

Across machines the behaviour is deliberate but worth knowing:

- Same path on two machines → **one workspace**, sessions distinguished by `host`.
- Same project at different paths (`/home/alex/doc2html` on `devbox` versus
  `C:\...\work_dev\doc2html` locally) → **two workspaces** with the same label.

The second case is real in this archive. They are left separate because they genuinely
are different checkouts; group them by label in the UI when that is what you want.
