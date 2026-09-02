# LLM-archive

One local, searchable archive of every LLM conversation you have — web chats and CLI
coding agents alike — in a single SQLite database that never leaves your machine.

Thirteen sources, one index. Ask *"the chat where I worked out the offside rule"* and get
it back, whether you had it in ChatGPT eight months ago or in Claude Code last Tuesday.

```
$ llma search "why did the migration deadlock"

 1. Postgres advisory locks in the batch job
    2026-04-12 · Claude Code · payments-api · dell · hybrid · 0.0281  [#412]
    user      we keep deadlocking when two workers run the migration at once
    assistant Both workers take the same two row locks in opposite order …
```

---

## Features

**Thirteen sources, one schema**

| Read in place | Bulk export (drop the ZIP) |
|---|---|
| Claude Code · Codex · opencode · VS Code chat | ChatGPT · Claude.ai · Gemini · Grok · DeepSeek · Mistral · T3 Chat · OpenRouter · GitHub Copilot |

Every source is normalised into the same message DAG — branches, retries, subagents, tool
calls, thinking blocks, attachments and token counts included.

**Hybrid search.** SQLite FTS5 (BM25) over *everything*, plus dense vectors over the part
that is worth embedding, fused by weighted RRF. Both are needed: on paraphrased queries —
the reason this project exists — BM25 alone finds the right session first once in nine
tries. Multilingual model, so an English query finds a Slovene conversation.

**Runs entirely offline,** with one deliberate exception. The embedding model is local
(118M params, ONNX, CPU). No API keys, no cloud, no telemetry. The web UI binds to
`127.0.0.1` and vendors nothing from a CDN. The exception is `fetch-images`: T3 Chat
archives a generated image as a link and no bytes, so `serve` backfills those in the
background as it starts. It opens no socket when none are pending, and
`serve --no-fetch-images` keeps it strictly offline. A page still never hotlinks — the
bytes are in the blob store before anything renders.

**Nothing is scraped.** Official exports and stores already on your disk. No site
credentials are stored and no session is fetched on your behalf.

**Local web UI.** `llma serve` — faceted browse by source / workspace / model / machine /
tag, full transcript view, and a statistics dashboard with token and cost breakdowns.

**Export back out.** Any session, or a filtered batch, as self-contained HTML or Markdown
with assets bundled. Secrets are redacted by default on the way out.

**Secret redaction.** Finds API keys, tokens and `.env` contents that leaked into tool
output and replaces them with `[redacted:<rule>:<hash>]` — so you can still tell which key
leaked without keeping the plaintext. Opt in to run it on every ingest.

**Multi-machine.** Copy another machine's store here and tag it with `--host`; nothing in
these formats records a hostname, so without it three laptops collapse into one pile.

**Stays current on its own.** `llma schedule install` registers a nightly re-ingest and
re-index; `llma freshness` tells you which web exports have gone stale.

---

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/<you>/llm-archive.git
cd llm-archive

uv venv && uv pip install -e ".[web,embed]"

# or with plain pip, inside an activated virtualenv:
#   pip install -e ".[web,embed]"
```

`web` adds the local UI, `embed` adds the embedding model for semantic search. With
neither, the CLI still works and search falls back to keyword-only (`llma index
--no-vectors`).

The `llma` command exists only once the project itself is installed. Without that, use
`python -m llm_archive.cli …` — it is the same code.

---

## Usage

### 1. Get your conversations in

Local agent stores are found automatically:

```bash
llma ingest                       # reads Claude Code, Codex, opencode, VS Code chat
```

Web chats come from each provider's official export. Download them, then point `llma add`
at the folder — it identifies each file by its contents, not its name:

```bash
llma add ~/Downloads              # scans, claims only what is an export
llma add ~/Downloads/conversations-000.zip
```

Re-adding a file you already have does nothing; identity is the sha256. Which export to
request from where is in [docs/export-requests.md](docs/export-requests.md).

### 2. Index and search

```bash
llma index                        # keyword + vectors; first run is slow on CPU
llma index --no-vectors           # keyword only, seconds

llma search "postgres deadlock"
llma search "kako naredim migracijo" --source claude_code --since 2026-01-01
llma search "the offside thing" --mode semantic -n 20
```

Filters: `--source` (repeatable), `--workspace`, `--participant`, `--host`,
`--since` / `--until`, `--abandoned`.

### 3. Read, browse, export

```bash
llma show 412 --tools             # one session as a transcript
llma serve                        # web UI on http://127.0.0.1:8787
llma serve --no-fetch-images      # ... without the startup image backfill

llma export 412 --format html,md
llma export --workspace payments-api --zip --out ./out
```

### 4. Keep it current

```bash
llma sync                         # ingest every source, then re-index
llma schedule install --at 21:00  # nightly, via Windows Task Scheduler
llma freshness                    # which web exports have gone stale
llma stats                        # what's in the archive
```

### Other commands

| Command | |
|---|---|
| `llma redact` | find secrets in stored text; `--apply` to rewrite, `--enable` for every future ingest |
| `llma validate` | check the message DAG for structural corruption |
| `llma doctor` | report record types no adapter handled |
| `llma browser` | opt-in: read OpenRouter's chats from its own browser IndexedDB store |
| `llma fetch-images` | download images archived as a URL only; `llma serve` also does this at startup |

More: [automation](docs/automation.md) · [multi-machine](docs/multi-machine.md) ·
[per-source format notes](docs/formats/)

---

## Privacy

`data/` is the highest-value secret on the machine — chat logs, tool output, private code,
and any API key that leaked into them, all concentrated into one searchable place. It is
in `.gitignore` and must never reach a remote. Do not expose `llma serve` beyond localhost.

## Development

```bash
uv pip install -e ".[web,embed,dev]"
pytest        # 545 tests, all offline — every fixture is inline
ruff check .
```
