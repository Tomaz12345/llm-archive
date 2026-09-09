# Phase 0 — findings

What the probe and benchmark actually found, versus what `PLAN.md` assumed.
Generated from `tools/probe.py`, `tools/build_evalset.py`, `tools/bench_embed.py`.

Run order:

```bash
python tools/probe.py                    # -> docs/formats/*.md, data/fixtures/*/shapes.json
python tools/build_evalset.py            # -> data/fixtures/evalset.json
.venv/Scripts/python tools/bench_embed.py  # -> docs/formats/bench_embed.md
```

---

## Summary of corrections to PLAN.md

| # | Plan said | Reality | Effect |
|---|---|---|---|
| R1 | Trees not lists — hardest problem | **Confirmed, worse than assumed.** 47 of 66 sessions branch. | Severity up. Still the hardest problem. |
| R2 | 1,227 `bridge-session` records mean constant resuming | **Wrong mechanism.** Bridge records are periodic cloud-sync checkpoints. Real duplication is ~2.5% of messages and trivial. | Downgraded from Phase 1 requirement to a cheap assertion. |
| §5.3 | Embed thinking blocks | **Not possible.** Claude Code stores `thinking: ""` with only a signature. | Removed from the chunking plan. |
| R5 | Formats will drift | **Confirmed, live.** 12 CLI versions; `user` records have 26 distinct key-sets. | Unchanged, validated. |
| §1.1 | ~240K tokens of conversation | **Confirmed.** 1,719 extractable docs, 0.86 MB, ~216K tokens. | Unchanged. |
| §1.4 | Slovene content needs a multilingual model | **Refined.** Slovene is ~14% by volume across 13 of 70 sessions, but *every* session title is English. | The real task is partly **cross-lingual**: English query → Slovene passage. Slice too small to validate — see §6.2c. |
| §5.1 | Hybrid BM25 + dense, fused with RRF | **Validated, wider margin than expected.** On realistic paraphrase queries BM25's MRR collapses 66% and its R@1 falls to 0.114; hybrid ranks best. | Keep hybrid. Tune RRF weights in Phase 4. |
| §1.2 | 118M model indexes the corpus "in minutes" | **Wrong.** Measured 3.4–4.1 docs/s on this CPU. | Full index is 1.5–3.5 h, not minutes. Viable only because indexing is incremental. |
| §9 Q3 | Probe VS Code storage? | **Yes — 110 sessions, 59.7 MB.** | New source, added to Phase 2. |

---

## 1. Format probe

Zero parse errors across 20,324 records from three sources.

| source | files | records | record types | distinct shapes | errors |
|---|---:|---:|---:|---:|---:|
| claude_code | 66 | 19,586 | 17 | 78 | 0 |
| codex | 3 | 414 | 17 | 38 | 0 |
| opencode | 328 | 324 | 4 | 21 | 0 |

### 1.1 Claude Code record types

Five types were not visible in the initial hand sample, which is exactly why the probe
exists: `agent-name`, `atis-latch`, `artifact-autoreact-ledger`, `frame-link`,
`artifact-comment-monitor`.

| type | count | key-sets |
|---|---:|---:|
| `assistant` | 7,862 | 16 |
| `user` | 4,537 | **26** |
| `ai-title` | 1,399 | 1 |
| `last-prompt` | 1,314 | 2 |
| `attachment` | 1,287 | 5 |
| `bridge-session` | 1,235 | 2 |
| `queue-operation` | 664 | 2 |
| `file-history-delta` | 262 | 1 |
| `atis-latch` | 241 | 1 |
| `file-history-snapshot` | 226 | 1 |
| `mode` | 201 | 1 |
| `permission-mode` | 179 | 1 |
| `system` | 88 | 6 |
| `agent-name` | 82 | 1 |
| `artifact-autoreact-ledger` | 5 | 1 |
| `frame-link` | 3 | 2 |
| `artifact-comment-monitor` | 1 | 1 |

Twenty-six distinct key-sets for a single `user` type, across 12 CLI versions
(2.1.173 → 2.1.241), is the concrete form of R5. The adapter must key off presence,
never off an assumed shape.

### 1.2 Message content blocks

| block | count |
|---|---:|
| `tool_use` | 4,219 |
| `tool_result` | 4,218 |
| `thinking` | 2,214 |
| `text` | 1,720 |
| bare string content | 135 |
| `image` | 8 |
| `document` | 5 |

**`thinking` is stored empty.** Every block is `{"type":"thinking","thinking":"","signature":"<~900 chars>"}`.
The signature is opaque. So reasoning traces are not searchable, not embeddable, and
contribute nothing but row count. Drop them at the adapter boundary.

---

## 2. R1 — the tree problem, quantified

```
sessions                                66
sessions with >1 leaf (real branching)  47      <- 71%
sessions with >1 root (orphans)          0
total branch points                    373
total leaves                           439
total DAG nodes                     13,774
```

Worst case is a single Invoice-integration session spanning 13–18 Aug: **799 nodes,
139 leaves, 138 branch points** in one file. Read that file top-to-bottom and you get
139 conversational dead ends presented as one continuous thread.

Zero sessions have more than one root, which is good news: every `parentUuid` resolves
within its own file, so path resolution never has to reach across files.

**Verdict: keep R1 exactly as planned, and treat it as Phase 1's main risk.**

## 3. R2 — mostly a non-issue

Three checks, all pointing the same way:

1. **`bridge-session` is not a resume marker.** 53 files contain them; 52 have exactly
   *one* distinct `bridgeSessionId`, written repeatedly (median 14×, max 143×). That is a
   periodic cloud-sync checkpoint, not a fork record.
2. **No cross-file parent links.** Zero sessions have an unresolvable `parentUuid`.
3. **Almost no duplicated content.** Of 278 distinct user messages, 7 appear in more than
   one session file (2.5%), and the single shared "first prompt" is the slash command
   `/mcp` — harness noise, not conversation.

So resumed sessions do not replay history into a new file; each file is self-contained.

**Verdict: drop the planned `parent_session_id` linkage work from Phase 1.** Keep the
column in the schema (it costs nothing and other sources may need it), and add a cheap
ingest-time assertion that flags if cross-file duplication ever rises above a few percent.

> **Correction (2026-09-09): this no longer holds.** The finding above was accurate when
> taken and has since been overtaken by a change in Claude Code itself. Re-running
> `tools/probe_resume.py` against a corpus grown from 9 projects / 93 files / 23,735
> records to 14 / 154 / 36,586 turns the verdict over: one file now replays another's
> leading message uuids in full, so a resume *can* fork. The linkage work was done after
> all — as `session.continues_session_id`, detected by `core/lineage.py` over
> `message.native_id` rather than over the raw files, which covers every source instead
> of just Claude Code. `parent_session_id` was left alone and still means "subagent
> transcript of". The "cheap assertion" suggested here is what `llma lineage` became.
>
> The measurement was right both times. Treat the *verdict* as dated, not the numbers.

## 3b. A ninth source found — VS Code chat (answers open question Q3)

`PLAN.md §9 Q3` asked whether VS Code workspace storage was worth probing. It is:

```
%APPDATA%\Code\User\workspaceStorage\<hash>\chatSessions\<uuid>.json
  110 files, 59.7 MB          <- comparable to Claude Code's 66 sessions
%APPDATA%\Code\User\globalStorage\emptyWindowChatSessions\
  33 GitHub.copilot-chat workspace dirs
  15 AndrePimenta.claude-code-chat workspace dirs
```

Session shape — and it is **flat, not a DAG**, so it is *easier* to parse than Claude Code:

```jsonc
{
  "sessionId": "...", "customTitle": "...", "mode": "...", "selectedModel": "...",
  "creationDate": ..., "lastMessageDate": ...,
  "requests": [
    { "requestId": "...", "message": {...}, "response": [...],
      "agent": {...}, "timestamp": ..., "modelId": "...",
      "contentReferences": [...], "codeCitations": [...] }
  ]
}
```

It also carries `customTitle`, so it plugs straight into the eval-set trick in §5.

**Verdict: in scope.** Higher value per unit of effort than DeepSeek or OpenRouter — it is
already on disk, needs no export request, has real volume, and has the simplest structure of
any source so far. Suggest slotting it into **Phase 2** alongside Codex and opencode.

### One store, several assistants (added after Phase 2)

The panel is shared ground. Across the 115 files here:

| who answered | requests | sessions |
|---|---:|---:|
| `GitHub.copilot-chat` | 178 | 41 |
| `ms-vscode-remote.remote-ssh` | 9 | 5 |

So "VS Code chat" as a source label hides the thing you actually want to ask about — how
much of this was Copilot. It is **not** split into a second source kind: the adapter owns a
*format*, and this is one format, with one id space and one dedupe key. Splitting it would
also strand every session that names no assistant.

Instead each session records `meta.participant` (`copilot`, `remote-ssh`, …) and
`meta.participant_label`, resolved in this order:

1. `requests[].agent.extensionId.value` — the only signal that survives everywhere.
   `selectedModel` is absent on 34 of the 46 ingested sessions, so vendor metadata alone
   would have found barely a quarter of the Copilot use.
2. `responderUsername` (`"GitHub Copilot"`) for sessions written before `agent` existed.
3. `selectedModel.metadata.vendor`, last resort.

An unmapped extension keeps its own id as the key, so a newly installed chat extension
appears as its own participant with no code change.

This is the same rule the archive already applies elsewhere: **the source is the transcript
format; how you reached it is an attribute.** Claude Code run from the VS Code extension is
still `claude_code`, marked by `entrypoint: claude-vscode` (13,001 records against 2,083
from a terminal), and Codex likewise by `originator: codex_vscode`.

Surface follows: `vscode_chat` is `editor_panel`, not `cli` (schema v4). Calling it `cli`
made the surface-split metric read as terminal work that never happened. With three real
surfaces the split is finally worth drawing, so `/stats` renders it as **Where you work**:
Terminal 72, Editor panel 46, Web chat 53 across 14 months.

### Response blocks: one more that carries content

`markdownVuln` is assistant prose the panel decided not to make click-to-run — a shell
command it wants you to copy deliberately. It is the answer, and it nests one level
deeper than plain prose (`content.value`, not `value`), so it reads as chrome and was
being dropped. `progressTaskSerialized`, `elicitation` and `notebookEditGroup` really are
chrome and now sit on the ignore list rather than showing up in `llma doctor` as drift.

---

## 3c. Real export files — what actually arrived

Probed with `tools/probe_exports.py --stage ~/Downloads`.

| kind | files | status |
|---|---:|---|
| `openrouter_chat` | 3 | **staged into `data/drops/`** |
| `claude_manifest` | 1 | manifest only — **the actual data is not downloaded** |

### The Claude export is not an export — it is a manifest

`manifest-774a08cc-…json` (1.2 KB) contains no conversations. It lists four ZIPs behind
signed URLs:

```
light_metadata-000.zip
projects-000.zip
memories-000.zip
conversations-000.zip     <- the one that matters
```

with the instruction: **"Each export URL can only be used once."**

None of the four have been downloaded. They were deliberately left untouched: a failed or
partial fetch burns the single-use URL and costs a fresh export request plus the wait.
**Download them from the browser, which is already authenticated.**

The org UUID in the filename (`774a08cc-…`) matches the `ownerOrganizationUuid` seen in
Claude Code's `bridge-session` records — same account, as expected.

### OpenRouter — format fully decoded

Schema `orpg.3.0`. One file per conversation, and normalised rather than nested:

```jsonc
{
  "version": "orpg.3.0",
  "title": "...",
  "characters": { "char-…": { "model": "minimax/minimax-m3:free", "modelInfo": {...} } },
  "messages":   { "msg-…":  { "characterId": "USER" | "char-…", "createdAt": "ISO",
                              "type": "user", "isEdited": false, "isRetrying": false,
                              "items": [ { "id": "item-…" } ] } },
  "items":      { "item-…": { "messageId": "msg-…",
                              "data": { "role": "user",
                                        "content": [ { "type": "input_text", "text": "..." } ] } } },
  "artifacts": {}, "artifactFiles": {}, "artifactVersions": {}, "artifactFileContents": {}
}
```

Adapter paths:

| field | path |
|---|---|
| text | `items[*].data.content[*].text` |
| role | `items[*].data.role` (`characterId == "USER"` marks the user) |
| model | `characters[ messages[*].characterId ].model` |
| time | `messages[*].createdAt` — ISO-8601 |

`isEdited` and `isRetrying` flags exist, so OpenRouter may carry branching too — check
before assuming the message list is linear.

---

## 3d. Two schema-affecting discoveries in the Claude Code store

Both found by following up on the `Invoice_integration/admin_dashboard` pointer.

### Tool output is *already* externalised — and the probe missed it

Sessions have sibling directories the `*.jsonl` glob never saw:

```
<project-slug>/
  <sessionId>.jsonl
  <sessionId>/tool-results/<id>.txt     <- 11 dirs, 36 files, 3.6 MB
  memory/MEMORY.md + project_*.md
```

When output is too large, Claude Code writes it to disk and leaves a marker in the
`tool_result` block:

```jsonc
{ "type": "tool_result", "tool_use_id": "toolu_…",
  "content": "<persisted-output>\nOutput too large (348.2KB). Full output saved to:
              C:\\Users\\…\\<sessionId>\\tool-results\\b859uybcu.txt\n\nPreview (first 2KB):\n…",
  "persistedOutputSize": 356536 }
```

This **validates the blob design in PLAN.md §1.1** — Claude Code independently arrived at
the same head-preview-plus-offload split. But the adapter must handle it:

- Detect `<persisted-output>` and parse out the path.
- **Resolve relative to the session directory, not the embedded absolute path** — that path
  is baked in at write time and breaks if `~/.claude` ever moves.
- Ingest the file as a `blob`, keep the 2 KB preview as `part.text`, and set
  `part.bytes` from `persistedOutputSize` rather than from the preview length.

Without this, 3.6 MB of real tool output is silently reduced to previews.

### One project directory holds many working directories — including case-variants

`cwd` is not constant within a project slug:

```
c--Users-alex-Documents-Projekti-Invoice-integration  -> 6 distinct cwds
   1402  c:\Users\alex\Documents\Projekti\Invoice_integration
   1129  C:\Users\alex\Documents\Projekti\Invoice_integration\admin_dashboard
    915  C:\Users\alex\Documents\Projekti\Invoice_integration        <- same as #1, different case
     98  …\admin_dashboard\jobs\templates\jobs
     14  …\output\render_batch
      6  …\admin_dashboard\jobs
```

Four of nine project directories show this. `telemetry_analysis` has both
`C:\…\telemetry_analysis` (6,916 records) and `c:\…\telemetry_analysis` (947) — the same
folder, since Windows paths are case-insensitive.

**Consequence for `PLAN.md §4`.** Deriving `workspace.key` from raw `cwd` would create
duplicate workspaces that differ only by drive-letter case, and would split one project
into six. Two fixes:

1. **Normalise the key**: casefold the whole path on Windows, and unify separators.
2. **Key the workspace on the project root** — the directory Claude Code slugified — and
   keep the specific `cwd` per session in `session.meta`. You think of
   `Invoice_integration` as one project, not six.

Without both, every per-project statistic is silently wrong.

---

## 4. Other corrections

- **opencode `todo/` and `session_diff/` store JSON arrays**, not objects. The probe
  originally counted 4 of these as parse errors; now handled as an `entity[]` shape.
- **Codex `session_meta.base_instructions`** embeds the entire system prompt in every
  rollout file. It is large, identical across sessions, and must not be ingested as
  conversation.

---

## 4b. Phase 1 — what building the adapter uncovered

Three more things the probe had not caught, found only by writing real code against real data.

### The DAG spans every record type — a silent, halving bug

The first adapter built the parent graph over conversational records only. But
`parentUuid` chains hop *through* bookkeeping records: a `user` turn's parent is routinely
a `file-history-snapshot`. Filtering first shatters the tree.

Measured on one 764-node session:

| | with content-only graph | with full graph |
|---|---:|---:|
| roots | **33** | 1 |
| active path length | **1 node** | full conversation |
| orphaned across archive | **11,107** | **491** |

Nothing crashed. It would simply have discarded half the archive and reported success.
`tests/test_claude_code.py::test_parent_links_hop_through_bookkeeping_records` pins it.

### Subagent transcripts live one directory deeper

```
<project>/<sessionId>/subagents/agent-*.jsonl
```

A non-recursive `*.jsonl` glob misses them entirely — 59 records, an entire subagent
session, silently absent. This is the third sidecar location after `tool-results/` and
`memory/`, and it gives `session.parent_session_id` a genuine use after all: linking a
subagent transcript to the session that spawned it. (R2 removed its original purpose.)

### More harness noise than R8 found

Beyond the IDE tags, text parts carry `<task-notification>`, `<task-id>`,
`<tool-use-id>`, `<output-file>`, `<local-command-caveat>` and friends. **91 records are
nothing but a task notification** — they clean to empty and are correctly dropped.

### Reconciliation

```
raw files                    68      DB sessions              68
raw content records      13,249      DB messages          10,704
  no usable content       2,443        + 91 pure task-notification blocks
                                     ----------------------------
                                     99.9% accounted for
orphan rate                 4.6%
embed corpus    1.59 MB of 16.8 MB stored  =  9.5%  (~397K tokens)
```

That last line is PLAN.md §1.1 measured inside the running system rather than estimated:
the archive stores 16.8 MB and embeds 1.59 MB of it.

**Idempotence verified:** a second `llma ingest` reports `new 0, updated 0, skipped 68`.

---

## 4c. Phase 4 — search, and an evaluation that contaminated itself

### Measured on the live archive

10,528 chunks × 384 dims (16.2 MB), built in 63.5 minutes on this CPU. `tools/tune_fusion.py`
drives the same `hybrid.search()` the CLI uses, so these are real end-to-end numbers.

**Paraphrase queries (n=35) — the trustworthy set:**

| retriever | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| keyword only | 0.200 | 0.343 | 0.543 | 0.277 |
| **semantic only** | **0.457** | **0.714** | 0.771 | **0.560** |
| 1.0 / 1.0 (plain RRF) | 0.343 | 0.686 | 0.771 | 0.469 |
| 1.0 / 1.6 | 0.371 | 0.686 | **0.800** | 0.499 |
| 1.0 / 3.0 | 0.400 | 0.686 | **0.800** | 0.514 |

Plain RRF is again worse than its better half (0.469 vs 0.560), a third independent
confirmation of the Phase 0 warning. Default set to **(1.0, 3.0)**, the best fusion.

### The titles set is contaminated — discard it

Scoring the title query set produced semantic MRR **0.981** against keyword 0.820,
reversing the Phase 0 offline result where BM25 won on titles. That reversal is an
artifact of my own chunker:

```
[claude_code · alex · 2026-08-24 · Project ideas file]
Hey, can you find file with listed project ideas ...
```

`context_header()` prepends the session **title** to every embedded chunk. Querying by
title then matches the chunk's own header. FTS indexes `part.text` and never sees the
header, so the advantage accrues to the semantic side alone.

The header stays — it is genuine context for genuine queries, and prepending document
context to passages is standard contextual-retrieval practice. But **titles can never
again be used as queries against it.** Weights are fitted on the paraphrase set only.

### Why BM25 is kept even though it loses

Semantic alone still beats fusion on MRR (0.560 vs 0.514). BM25 stays for something this
benchmark structurally cannot measure: **62% of the archive is tool output, which is
never embedded** (§1.1). Vector search physically cannot find the session that ran a
given command or hit a given stack trace — only FTS reaches that text. The eval scores
conversational recall, so it understates keyword's actual job.

### Two bugs found by measuring

- **A fresh archive had no search at all.** The FTS and chunk tables were only in
  `MIGRATIONS[3]`, which runs exclusively for *existing* databases. The live archive
  worked because it upgraded from v2; a clean install would have crashed on first index.
  Now in both the base schema and the migration.
- **The model reloaded on every query.** `hybrid._semantic()` constructed a fresh
  `Embedder` per search, paying ONNX load each time: **4.88 s per search, versus 272 ms
  once cached.** Invisible in a one-off CLI call, crippling for the tuner (280 searches)
  and fatal for a web UI. Now a process-wide cache.

### Throughput correction, again

The first full index ran at **2.8 chunks/s** with `batch=64`, against 4.1 chunks/s when
Phase 0's benchmark passed the whole list at once — small batches re-enter fastembed's
pipeline and lose its internal batching. Default batch raised to 512. My "~30 minutes"
estimate for this corpus was wrong; the measured figure is **63.5 minutes**.

---

## 5. Evaluation set

Built with no manual labelling, by exploiting a fact all three sources share: each one
already stores a model-written title per session.

```
query  = the session title      ("Investigate slow inference times in telemetry analysis")
target = the session it names
metric = does the retriever surface that session in the top k?
```

| | |
|---|---|
| docs | 1,719 (0.86 MB, ~216K tokens) |
| queries | 69 |
| sessions | 70 |
| by source | claude_code 1,666 · opencode 30 · codex 23 |
| Slovene-bearing sessions | 13 of 70 (~14% of corpus by volume) |
| query language | **100% English** |

Scoring is session-level, matching what the UI returns: a document from the right session
anywhere in the top 10 counts as a hit.

The all-English-titles result is the interesting one. It means the honest retrieval task
is not "Slovene query → Slovene doc" but **English query → Slovene passage**, which is a
harder, cross-lingual problem, and one an English-only model cannot do at all. The
benchmark reports a separate `SL R@10` column over the 13 Slovene-bearing sessions.

### Caveat on this eval set

Titles are generated from the conversation, so they share vocabulary with it. That gives
BM25 a genuine advantage it might not have against a query typed from memory months later.
Read the BM25 row as an *optimistic* baseline, and weight the dense models' advantage on
the cross-lingual slice more heavily than their overall margin.

---

## 6. Embedding benchmark

See `docs/formats/bench_embed.md` for the generated tables.

`intfloat/multilingual-e5-small`, the model PLAN.md §5.2 recommended, is not in
fastembed's catalogue. The closest available equivalent in the same size class is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (118M params, 384 dim).
The benchmark attempts e5-small anyway and skips it cleanly if it cannot load.

### 6.1 The first run invalidated its own eval set

First result, on the auto-built title queries:

| retriever | R@1 | R@10 | MRR | SL R@10 | encode |
|---|---:|---:|---:|---:|---:|
| BM25 / FTS5 | 0.855 | **0.957** | **0.895** | 1.0 | 0.1 s |
| pm-MiniLM-L12 | 0.754 | 0.928 | 0.828 | 0.923 | 509 s |
| HYBRID (RRF) | 0.812 | 0.957 | 0.875 | 0.923 | — |

Keyword search beat the embedding model on every metric, and fusing the two was *worse*
than BM25 alone on MRR. Taken at face value that kills the dense half of the design.

It should not be taken at face value. Measuring query/document lexical overlap:

```
mean overlap of query terms with target session text   0.837
high (>=0.8)   44 queries
mid  (0.5-0.8) 21 queries
LOW  (<0.5)     4 queries      <- the only genuine semantic tests
```

Titles are *generated from* the conversations they name, so they reuse their vocabulary.
The eval set was measuring "find a session you can already name" — a real task, but the
easy one, and one keyword search should win. It cannot measure the case hybrid search
exists for: searching months later in words you never used.

**A benchmark that cannot distinguish the two candidates is not evidence for either.**

### 6.2 Second query set

`data/fixtures/paraphrase_queries.json` holds hand-authored queries describing the same
sessions in deliberately different words — "why was the match video taking so long to
process" for *Investigate slow inference times in telemetry analysis*. Both retrievers get
identical queries, and the benchmark prints the measured overlap for each set so the
disjointness is verifiable rather than asserted.

Authored from titles and first-message snippets, never from retrieval output.

### 6.2b Result — the second query set reverses the first conclusion

Same corpus (1,632 docs), same model, same code. Only the queries differ.

**`titles` — 64 queries, overlap 0.828 (the easy case)**

| retriever | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 / FTS5 | **0.812** | 0.922 | **0.953** | **0.863** |
| pm-MiniLM-L12 | 0.766 | 0.906 | 0.922 | 0.830 |
| HYBRID (RRF) | 0.797 | **0.938** | **0.953** | 0.860 |

**`paraphrase` — 35 queries, overlap 0.484 (the realistic case)**

| retriever | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 / FTS5 | 0.114 | 0.514 | 0.714 | 0.295 |
| pm-MiniLM-L12 | 0.286 | 0.686 | **0.886** | 0.446 |
| HYBRID (RRF) | **0.343** | **0.714** | 0.829 | **0.492** |

What changes when the words stop matching:

- **BM25 collapses.** MRR 0.863 → 0.295, a 66% fall. R@1 goes from 0.812 to **0.114** —
  it puts the right session first for one query in nine.
- **Dense degrades gracefully.** MRR 0.830 → 0.446, a 46% fall. It finds the right session
  in the top ten 89% of the time.
- **Hybrid is the best ranker.** Top MRR (0.492) and top R@1 (0.343 — three times BM25's)
  on realistic queries, while matching BM25 exactly on the easy set (R@10 0.953).

**Verdict: PLAN.md §5.1 hybrid + RRF is validated, and by a wider margin than expected.**
Neither retriever alone is adequate: BM25 owns the case where you remember your own words,
dense owns the case where you do not, and you cannot know which case a query is in advance.

One wrinkle worth carrying into Phase 4: on the paraphrase set hybrid *beats* dense on R@1
and MRR but *loses* on R@10 (0.829 vs 0.886). Fusing in a retriever that is nearly random
for a given query pushes some correct deep-tail results out. Plain unweighted RRF is
therefore not the final answer — Phase 4 should try weighting the dense side higher, or
adapting the weights to the query.

### 6.2d Which model — decision

`intfloat/multilingual-e5-small`, the model PLAN.md §5.2 named as the recommended starting
point, **is not available in fastembed** (`ValueError: Model ... is not supported in
TextEmbedding`). Using it would mean pulling in `sentence-transformers` and torch, or
registering a custom ONNX build. Not worth it given the results below.

Two candidates actually measured, on the same 1,632-doc corpus:

| model | params | dim | docs/s | full-index projection | paraphrase MRR | paraphrase R@10 |
|---|---:|---:|---:|---|---:|---:|
| **pm-MiniLM-L12** | 118 M | 384 | **4.1** | **1.5–3.5 h** | 0.446 | **0.886** |
| pm-mpnet-base | 278 M | 768 | 0.7 | **10–24 h** | **0.544** | 0.829 |

mpnet ranks better (MRR 0.544 vs 0.446) but **finds less** (R@10 0.829 vs 0.886), and costs
**6× the compute** — 41 minutes for this small corpus, projecting to overnight-plus for the
real one on a machine with no GPU.

**Decision: `paraphrase-multilingual-MiniLM-L12-v2`.**

- R@10 is what matters most for a UI that shows ten results, and MiniLM wins it.
- The MRR gap is 0.098 on n=35 — inside the noise band for this sample size.
- 6× throughput is not a marginal difference on this hardware; it is the difference between
  a one-time index you can run over lunch and one you run overnight.

Keep mpnet-base as the documented upgrade if ranking quality proves insufficient in real use.
Vectors are cached per model under `data/vectors/<label>.<corpus-hash>.npy`, so swapping
models later costs one encode pass and nothing else.

### 6.2e RRF is not free — a second warning

For mpnet, fusing with BM25 made results **worse**, not better:

| | dense alone | hybrid (RRF) |
|---|---:|---:|
| pm-mpnet-base, paraphrase MRR | **0.544** | 0.467 |
| pm-MiniLM-L12, paraphrase MRR | 0.446 | **0.492** |

The stronger the dense retriever, the more damage unweighted RRF does by mixing in a BM25
ranking that is near-random for paraphrase queries. This is now two independent signals
pointing the same way, so treat it as established rather than incidental:

**Phase 4 must tune the fusion, not just implement it.** Options: weight the dense side
higher, or estimate per-query whether BM25 is likely to be informative (for example from
whether the query terms appear in the index at all) and down-weight it when it is not.

### 6.2c Caveats on these numbers

- **35 paraphrase queries is a small sample.** Differences of a few points are noise;
  the BM25-vs-dense gap here is large enough to act on, the hybrid-vs-dense gap is not.
- **The queries are hand-authored by Claude**, from titles and first-message snippets and
  never from retrieval output. Measured overlap (0.484 vs 0.828) makes the disjointness
  verifiable, but authorship bias cannot be fully excluded. Re-run with your own queries
  before treating the margin as exact.
- **`SL R@10` should not be trusted.** Only 13 of 70 sessions carry Slovene diacritics, and
  a first attempt at a stricter word-based detector produced obvious false positives (the
  Slovene function words `in`, `so`, `se` are also English words). The slice is too small
  to validate the multilingual choice. Multilingual remains right because it costs nothing
  against an English-only model of the same size — not because this data proved it.
- **fastembed warns** that this model now uses mean pooling instead of CLS embedding,
  which may understate its quality versus its published scores. Pinning fastembed 0.5.1 or
  registering it via `add_custom_model` is worth trying in Phase 4.

### 6.3 A real bug found along the way

Every message in an editor session carried `<ide_opened_file>` / `<ide_selection>` tags
containing the open file's absolute path. Dozens of unrelated sessions therefore shared
the same path tokens, inflating keyword scores and blurring embeddings.

Stripping them removed 87 of 1,719 docs (5%). **This is an adapter requirement for Phase 1,
not just an eval-set fix** — the same noise would otherwise land in `archive.db`.

### 6.4 Throughput — the plan was too optimistic

Measured: **3.4 docs/s** for a 118M-param fp32 ONNX model on this CPU. PLAN.md §1.2 claimed
"minutes" for the full corpus; the honest projection is **1.5–3.5 hours** for a 2–5M-token
index. Survivable only because indexing is incremental — see the correction in §1.2.
