# Keeping the archive current

> Every command below is `llma`, the entry point declared in `pyproject.toml`. It only
> exists once the project itself is installed into the venv — installing the
> *dependencies* does not create it:
>
> ```
> uv pip install -e . --no-deps        # or: pip install -e .
> ```
>
> Without that, use `python -m llm_archive.cli ...` instead; it is the same code. The
> scheduled task calls the module form on purpose, so it keeps working whether or not
> the package is installed.

Phase 7. Three things decay once the archive works: the local stores drift ahead of the
last ingest, the web exports go stale, and the secrets that leaked into tool output stay
in a file that gets more valuable every week.

---

## 0. Adding an export

The local stores need nothing — `sync` re-reads them. Web exports are the manual half,
and `llma add` is all of it:

```
llma add ~/Downloads                    # scans, claims only what is an export
llma add ~/Downloads/<file>             # or one file
llma add --no-ingest <file>             # file it now, read it on the next sync
```

Files are identified by their **contents**, never their names — Grok's export is a bare
uuid, Mistral's is a timestamp, and claude.ai, DeepSeek and ChatGPT all ship a file
called `conversations.json`. Your downloads are left alone; the archive copies into
`data/drops/`. Adding a file you already hold is a no-op, because identity is the sha256.

`llma serve` → **Import** is the same thing with a drop target, and it also shows the
freshness table, so one page answers both "take this file" and "which export should I go
and fetch?".

Once a drop's sessions are stored it moves to `data/drops/_archive/<YYYY-MM>/`, and
`session.raw_path` is repointed at the new location. The raw bytes stay, so any session
can still be re-parsed from the file it came from.

### What happens when the same account is exported twice

This is the case the whole import path is built around, because getting it wrong is
silent. An export is a point-in-time snapshot (§8.7), so a later one overlaps everything
you already have, and a conversation you kept using has *grown* since.

* **A grown conversation appends.** Messages already stored are rewritten in place;
  messages the new export adds are inserted. Reported as `appended`.
* **An older export cannot overwrite a newer one.** Every session records `exported_at`
  — the drop's mtime, the same clock `llma freshness` uses. Re-dropping last month's
  export is refused and reported as `stale`. Before this, drops were discovered in
  filename order and whichever sorted last won, so a stale Grok or OpenRouter drop could
  silently truncate a conversation.
* **A shrinking export never removes history.** A chat deleted server-side, a provider
  pruning old turns, a Takeout window that has rolled past its 18-month retention — the
  stored messages stay, stamped `message.absent_since` with the moment they first went
  missing. Reported as `retained`. They are still searchable and still export; the web UI
  marks them as not present in the latest export.

`--force` lifts both the unchanged-bytes short-circuit and the staleness guard. It is
what you want after changing an adapter, and nothing else.

### OpenRouter without the per-chat clicking

```
llma browser              # what is on this machine
llma browser --enable     # read it on every ingest from now on
```

OpenRouter is local-first and has no bulk export, so capturing 50 chats means 50
separate "Export Chat" clicks (§2.1). The chats are already on this disk — the site
keeps them in its own IndexedDB store — and `llm_archive/core/idb.py` reads them.

**Firefox only.** Firefox stores IndexedDB as a plain SQLite file of Snappy-compressed
structured clones, all of which `sqlite3` plus two small decoders can read. Chromium
stores a LevelDB SSTable of V8 blobs, which needs two parsers that are not in the
standard library and are a bad thing to write against an undocumented format. The
original probe searched only Chromium and reported "not available here", which was
wrong on this machine — see the note at the top of `tools/probe_openrouter_idb.py`.

Off by default, because it opens a browser profile — the place session cookies and
saved passwords also live. Only the directory belonging to `openrouter.ai` is ever
opened, the store is copied before being read (a running browser holds a lock on the
live one, and a half-written `-wal` is how a profile gets corrupted), and nothing else
is touched.

Chats read this way **merge** with ones you exported by hand: both routes key a session
on the same root message id, so switching this on does not duplicate anything. Which
route last wrote a session is in `session.meta.origin`.

A record the decoder does not recognise is counted as
`openrouter:browser-record-undecodable` and shows up in `llma doctor` — never skipped.
Skipping one would mean guessing its width, and a wrong guess desynchronises the rest
of the record into plausible nonsense rather than failing.

**On this account it found 3 conversations — exactly the 3 already exported by hand.**
So it adds nothing retroactively; its value is that every OpenRouter chat from here on
lands without a single export click.

---

## 1. Scheduled sync

```
llma schedule install                 # daily at 21:00
llma schedule install --at 06:30
llma schedule install --dry-run       # print the task XML, register nothing
llma schedule status
llma schedule run                     # start it now, the way the scheduler would
llma schedule remove
```

The task runs `llma sync`, which is **ingest + index in one command, deliberately**.
Ingesting a session deletes and rewrites its parts, so part ids move, and both search
indexes are keyed on those ids: FTS5 is external-content and keeps matching the old
rowids against new rows, and every `chunk` for a re-ingested session points at a part
that no longer exists. Neither failure raises anything. An automated ingest that did not
re-index would quietly degrade search every night, and the only symptom would be worse
results.

### The settings that decide whether it actually runs

Registered through PowerShell's `Register-ScheduledTask` rather than `schtasks.exe`,
because `schtasks /Query` prints localised field names and `/TR` needs nested quoting
that breaks on any path with a space — which is every path here.

| Setting | Value | Why |
|---|---|---|
| `StartWhenAvailable` | `true` | The laptop is asleep at 21:00 most nights. The default is to skip the run entirely, and the archive stops updating with nothing to say so. |
| `DisallowStartIfOnBatteries` | `false` | **The default is `true`.** On a laptop that means the task exists, reports "Ready" forever, and never runs. |
| `MultipleInstancesPolicy` | `IgnoreNew` | A full embed pass is minutes. Two of them writing `chunk` at once is the one way this job corrupts its own index. |
| Command | `pythonw.exe` | A nightly `python -m` flashes a console window and steals focus. |

### What a run actually costs, measured

The first real `llma sync` on this machine: 7s to ingest all twelve sources (12 new
Claude Code sessions, 7 updated, 60 unchanged), then **~30 minutes wall / ~15 minutes
CPU** to rebuild the index.

That asymmetry is not a bug but it is a decision you should make deliberately.
`index.build` is a **full rebuild** — it drops every chunk for the model tag and
re-embeds the whole corpus, however little changed. Nineteen changed sessions cost the
same as nineteen thousand. Since Claude Code is used most days, "nightly" here means
half an hour of CPU most nights.

Three ways to live with it, in order of how much they give up:

1. **Leave it.** It runs at 21:00 on a machine you are not using. This is the default
   for a reason.
2. **Split it: nightly keyword, weekly vectors.** This is what is installed here.
   `llma schedule install --no-vectors` gives a nightly job that finishes in seconds,
   and a second task runs the full `sync` once a week. Keyword search is exactly right
   every night; semantic search is at most a week behind. `/stats` reports the nightly
   build as `--no-vectors` rather than as stale, because it is not.
3. Make the indexer incremental. Out of scope here — it changes what `llma index` means
   for every caller, not just the scheduled one.

### The weekly task, and the bug that made it necessary

`llma schedule` only builds daily tasks, so the weekly one is registered directly from
`schedule.make_plan` + `build_xml` with the `ScheduleByDay` trigger swapped for
`ScheduleByWeek` — reusing them rather than hand-writing a definition, because the
laptop settings in the table above are the entire reason that XML looks the way it does.

| | `LLM Archive Sync` | `LLM Archive Reindex` |
|---|---|---|
| when | daily 21:00 | Sundays 22:00 |
| runs | `sync --no-vectors` | `sync` |
| costs | seconds | ~30 min |

It runs `sync`, not `index`: `index` has no `--log`, and under `pythonw` that means a
half-hour job whose only record is nothing at all. `sync` also does the right thing when
the nightly runs have already ingested everything — it re-indexes when something changed
**or when the vector index does not cover the archive**, and after six keyword-only
nights the second condition is what fires.

Nothing else on the hour: at 22:00 the 21:00 job is long finished, and if the two ever
did overlap the lock makes one stand down.

**`--no-vectors` used to destroy the vector index.** `index._build` ran
`DELETE FROM chunk` *before* testing `with_vectors`, then returned without re-inserting
anything — so a keyword-only build wiped whatever the last full build had embedded, and
this split schedule would have left semantic search dead six nights in seven. The delete
now happens only on a build that is about to re-insert. Keeping the chunks is sound:
`chunk.part_id` is `ON DELETE CASCADE` and ingest deletes a re-ingested session's parts,
so chunks whose text moved are already gone, and every surviving chunk's `vec_row` still
addresses the same row of the untouched `.npy`. What is left is coverage missing for new
sessions — exactly what `selection.unindexed_count` reports and what the weekly run
repairs. The exception is `redact --apply`, which rewrites `part.text` in place without
moving ids; that is why it tells you to run a full `llma index` afterwards.

Whichever you pick, the job logs a progress heartbeat every 25%. A scheduled run has no
console, and a log that goes silent for twenty minutes is indistinguishable from a hang.

### Where the output goes

`data/sync.log`, timestamped, truncated from the front past 1 MB. `pythonw` has no
console, so this is the only record a failed scheduled run leaves. `llma schedule
status` prints `LastTaskResult` — `0` is success, anything else means read the log.

### The lock

`data/.sync.lock`. `IgnoreNew` stops the task overlapping itself but nothing stops a
nightly run landing on top of a manual `llma index`. A held lock makes the second run
stand down; a skipped run is logged and exits 0, because it is not a failure.

Staleness is decided two ways, and it needs both:

- **Liveness** releases an orphaned lock immediately. A run killed mid-embed never
  reaches its cleanup — this happened on the very first real sync here, and the lock it
  left would otherwise have blocked the archive for two hours. The check is
  `OpenProcess(SYNCHRONIZE)` via ctypes and **never** `os.kill(pid, 0)`: on Windows that
  call does not test liveness the way it does on POSIX — it routes to
  `TerminateProcess`, so the check would kill the run it was asking about.
- **Age** covers what liveness cannot: a holder that is alive but hung, and a recycled
  pid that makes a dead holder look alive. The 2-hour limit matches the task's own
  `ExecutionTimeLimit`.

### Repairing an interrupted build

`sync` re-indexes when something was ingested **or when the vector index does not cover
the archive**, and the second half is not redundant. A build killed part-way commits its
`DELETE FROM chunk` and dies before the re-insert, leaving an empty index; nothing is
then ingested on the next run, so a "did anything change?" test alone would leave the
archive unsearchable indefinitely. The coverage question is asked through
`search.selection`, the same module `/stats` reports staleness with, so the two can
never disagree about what "indexed" means.

### Starting the web UI at logon

`llma schedule` builds the nightly *sync* task. The UI is a separate, long-running
process, so it gets its own task — registered by hand, from
`tools/serve_windowless.py`:

| | |
|---|---|
| task | `LLM Archive Serve` |
| trigger | at logon, `Delay PT1M` — the archive should not compete with the login storm |
| runs | `.venv\Scripts\pythonw.exe tools\serve_windowless.py` |
| log | `data/serve.log`, same front-truncation as `sync.log` |

Two settings are load-bearing, and both were found by the task failing silently:

* **`ExecutionTimeLimit` must be `PT0S`** (unlimited). The default is three days, after
  which the scheduler would kill a perfectly healthy server.
* **`Priority` must be 5, not the 7 the sync task uses.** 7 is `BELOW_NORMAL`, and on a
  machine with a browser and an editor open that starves the startup imports badly
  enough to look like a hang: `llm_archive.web.app` alone took **31 seconds** to import,
  and the process sat at 0.6s of CPU for four minutes without ever binding the port. At
  priority 5 the same task serves in **30 seconds**. Below-normal is right for a nightly
  batch job nobody is waiting on; it is wrong for the thing you are about to open.

`MultipleInstancesPolicy` stays `IgnoreNew`, so a second logon cannot start a second
server against the same port — but note the trap it sets during debugging: while an
instance is stuck, `Start-ScheduledTask` is silently a no-op, which reads exactly like
the task refusing to run.

**Why a launcher script and not `pythonw -m llm_archive.cli serve`.** That form exits 1
before it binds anything. pythonw gives the process no console, so `sys.stdout` and
`sys.stderr` are `None`, and uvicorn's logging config calls `sys.stderr.isatty()` while
deciding whether to colourise. The nightly sync gets away with pythonw only because
`sync --log` writes through `core.sync.logger` and never builds a logging config.
`tools/serve_windowless.py` points both streams at `data/serve.log` first, which fixes
the crash and means a failed start — a port already taken is the likely one — leaves a
traceback instead of an exit code and nothing.

```
schtasks /Run /TN "LLM Archive Serve"      # start it now
Get-ScheduledTask "LLM Archive Serve" | Get-ScheduledTaskInfo
Unregister-ScheduledTask "LLM Archive Serve" -Confirm:$false
```

### Other platforms

`llma schedule` is Windows-only and says so. The equivalent is a crontab line:

```
0 21 * * *  /path/to/python -m llm_archive.cli sync --log ~/llm-archive-sync.log
```

---

## 2. Export freshness

```
llma freshness              # bulk sources only
llma freshness --all        # plus per-chat and local sources
llma freshness --days 14
```

Also a panel on `/stats`, a block on `/import`, and — since Phase 7 — **the tail of
every sync**, which is the part that makes it automatic rather than something you have
to remember to run:

```
  EXPORT DUE  ChatGPT: no export has ever been ingested (last never)
              Settings -> Data controls -> Export data (arrives by email)
  retention   Gemini: My Activity auto-deletes on a rolling window (18 months by
              default) — a missed export here loses conversations permanently
```

Overdue sources are named with the instruction for fixing them; everything current
collapses to one line, so a healthy nightly run stays short enough to keep reading.
A **retention** warning is printed even for a source that is perfectly up to date —
Gemini's My Activity window and Grok's 30-day link are deadlines that run whether or
not you have exported recently, and they are the one warning that expires by itself.
`llma sync` repeats the overdue list on the console even when the detail went to a log
file: an export you have not taken is the one thing a sync cannot fix for you.

Ten sources decay three different ways, and only one of them is answered by "when was
the newest session?":

- **live** — Claude Code, Codex, opencode, VS Code chat. Re-read on every ingest, so
  never stale *as exports*. The scheduler above is what keeps these current.
- **bulk** — claude.ai, ChatGPT, DeepSeek, T3 Chat, Gemini, Grok. One request returns
  the whole account, so "re-export every N days" is a real instruction. These are the
  only sources ranked.
- **per_chat** — OpenRouter, Mistral, Copilot. One file per conversation (§2.1). No
  single action makes them current, so an interval warning is pure noise; they are
  listed and never marked overdue.

### The measurement that keeps it honest

Ranking by newest session answers the wrong question. A source whose newest chat is
three months old is *either* a neglected export *or* a service you stopped using, and
those want opposite responses. They are separable because `session.raw_path` names the
drop the rows came from, and that file's mtime is when the export was taken:

```
export old                    -> overdue: go and re-export
export fresh, newest chat old -> current: you just have not used it
```

Only the first is a warning. Four of these accounts are genuinely near-empty (§2.3,
§2.5, §2.6) and would otherwise dominate the list forever, training you to skim past the
one that matters.

### Two things the report says that a query over `session` cannot

- **A source with no rows at all.** ChatGPT has no adapter yet, so it appears in no
  join — and "you have never ingested ChatGPT" is precisely what this report is for. The
  source table is a constant in `core/freshness.py`, not derived from the database.
- **Retention.** Gemini's My Activity auto-deletes on a rolling window (18 months by
  default) and Grok's bundle sits under x.ai's own `ttl/30d` marker. Those are true of a
  perfectly current export, so they print regardless of state.

---

## 3. Secret redaction (opt-in, off by default)

```
llma redact                          # dry run, default rules
llma redact --wide                   # add shape-based rules
llma redact --blobs                  # also scan data/blobs/, report only
llma redact --apply                  # rewrite part.text
llma redact --apply --enable         # ...and keep doing it on every ingest
llma redact --status
llma redact --disable
```

Replacement is `[redacted:<rule>:<8 hex of the secret's sha256>]`. The fingerprint is
the point: it tells two leaked keys apart, says how far one key spread, and lets you
confirm months later whether the key that leaked is the one you rotated — none of which
works if every secret becomes the same opaque marker, and all of which would otherwise
mean keeping the plaintext.

### What it covers, and what it does not

| | Covered |
|---|---|
| `part.text` in `archive.db` | **yes** — rewritten in place |
| `data/blobs/` | **no** — scanned and reported only. Blobs are content-addressed by sha256; rewriting one invalidates the hash every referring row was deduplicated on. |
| `data/drops/` and the original `raw_path` files | **no** — and this is the important one. |

Ingest re-reads the originals, so a one-off redaction is undone by the next run. That
is why `--enable` exists: it stores the setting, and every subsequent ingest redacts
**before** anything is indexed or embedded. §8.6 already says `data/drops/` needs the
same handling as the database; this pass is not what secures either it or the blob
store.

### Two rulesets

Default rules match issuer prefixes with fixed shapes — `sk-ant-`, `ghp_`, `AKIA…`.
Near-zero false positive, because nothing else looks like them.

`--wide` adds shape rules: `password = …`, bearer headers, credentials inside a URL.
More coverage, and it will occasionally eat an example out of documentation. A search
archive that has destroyed a paragraph of prose to protect the string `password =
hunter2` in a tutorial is worse at its job for no gain, so this is opt-in.

### What the first real dry run over this archive taught

Both of these were found by running it, not by reading the regexes:

- **`sk-[A-Za-z0-9_-]{20,}` matched `MozillaBackgroundTask-308046B0AF4A39CB-…`** — the
  `sk-` came out of "Ta**sk-**". Four "OpenAI keys" inside a Firefox profile listing. A
  prefix rule is specific only because the prefix *starts a token*; mid-word it is two
  letters and a dash. Every prefix rule now requires a word start.
- **The word-start check must not be a regex lookbehind.** `re` scans for a pattern's
  leading literal with a fast substring search and only then runs the engine; a
  lookbehind in front leaves no leading literal, so all 20,825 parts get walked
  character by character instead of skipped. Measured: **1.3s → 15.5s** for one pass,
  for a guard that is one character comparison after a match already found. It lives in
  `scan_text` instead.
- **A `BEGIN PRIVATE KEY` / `END PRIVATE KEY` pair with nothing between them** is prose
  about key files. The rule now requires a 100-character body.

After those three fixes: 12 reported → 5, and all 5 look real.

Redaction rewrites text the keyword index has already read, so run `llma index`
afterwards. `llma sync` and `llma ingest` order it correctly on their own.

---

## 4. OpenRouter browser storage — built, Firefox only

§2.1's last open item, now closed. Usage is in §0 above (`llma browser --enable`); this
records how the decision went, because the answer reversed itself.

```
python tools/probe_openrouter_idb.py
```

**The probe was looking in the wrong place.** It searched only the Chromium family and
reported "not available here" — on this machine the store is in **Firefox**. That is not
a detail, it is the whole decision:

- **Chromium** keeps IndexedDB values as V8 structured-clone blobs inside a LevelDB
  SSTable. A reader needs an SSTable parser *and* a V8 decoder, neither in the standard
  library. Still not built, and still not worth it.
- **Firefox** keeps them as Snappy-compressed structured clones inside a plain SQLite
  file. `sqlite3` is in the standard library; the two decoders are ~150 lines and fully
  determined by the bytes. Built, in `llm_archive/core/idb.py`.

**The §8.5 mitigation the old note said was missing turned out to exist.** Every record
is keyed `/v3:<type>:<id>` inside a store named `openrouter:playground:v3`. A schema
change bumps that, the reader finds no rooms, and it says
`openrouter:browser-store-without-rooms` in `llma doctor` instead of mis-parsing a shape
it has never seen. That is a real version field, not an absent one.

**What it found here: 3 conversations, 6 messages, 26 records, all decodable** — exactly
the three already exported by hand. By the old note's own criterion ("three chats you
have already exported by hand is not worth it") that argues against building it; it was
built anyway because the value is forward-looking, and the measurement above is what to
re-run if that ever needs re-deciding.

Unknown tags are **refused, never skipped**. Skipping means guessing a width, and a
wrong guess does not raise — it desynchronises the stream so everything after it decodes
into plausible nonsense. A record that will not decode is counted and dropped, loudly.
