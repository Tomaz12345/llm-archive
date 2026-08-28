# Web export request checklist

**This is the one Phase 0 task that needs your hands, and it blocks Phase 3.**
ChatGPT and Claude.ai deliver by email hours after you ask, so request them first and
let them arrive while the local adapters get built.

**You no longer file these by hand.** Point `llma add` at whatever you downloaded and it
works out what each file is by sniffing its contents, never its name:

```
llma add ~/Downloads                    # scans the folder, claims only the exports
llma add ~/Downloads/conversations-000.zip
```

Your downloads are left where they are; the archive takes its own copy into
`data/drops/`. Re-adding a file you already hold does nothing — identity is the sha256,
so the same export downloaded twice under two names is recognised as one. The same thing
is available in the browser at `llma serve` → **Import**, which also lists which exports
are overdue and where each provider's export button is.

Once a drop's sessions are stored it moves to `data/drops/_archive/<YYYY-MM>/`. The raw
bytes stay there so any session can be re-parsed from the file it came from.

`data/drops/` is inside `data/`, which `.gitignore` already excludes (risk R6).

---

## Do these two first — they have a delay

### 1. ChatGPT
`Settings → Data controls → Export data → Export`
Arrives by email as a download link, typically within a few hours. Link expires — grab it.

**Add the whole ZIP, not just `conversations.json`.** The images you uploaded and the
ones the model generated ship as real files alongside it (`file-<id>-<name>.png`), and
they are only recoverable while they are in the same archive as the chats that point at
them. See `docs/formats/chatgpt.md`.

### 2. Claude.ai
`Settings → Privacy → Export data`
Also emailed. Contains `conversations.json` and `projects.json`.

---

## Then these six — immediate

### 3. DeepSeek
`Settings → Profile/Data → Export data` (exact wording unconfirmed).
**Highest-risk source** — this is the least documented export of the five. If you cannot
find the option, say so and we will skip DeepSeek rather than build a scraper.

> Note: DeepSeek exports have been reported to include personal data such as phone numbers.
> It stays in `data/`, which never leaves the machine.

### 4. T3 Chat — **done, and it is the easy one**
`Settings → History & Sync → Export` → `threads-export-<instant>.json`.

There *is* a bulk option: one file for the whole account, no per-chat clicking and no
IndexedDB detour. 170 threads / 1,430 messages verified and ingested; `adapters/t3chat.py`
reads it. Re-export whenever you want a refresh — the adapter hashes each thread separately,
so a fresh export only rewrites the chats that actually changed.

### 5. Gemini — **done, and the only one with a deadline**
`takeout.google.com` → **Deselect all** → tick **My Activity** → *Multiple formats* →
set **Activity records: HTML** → *All activity data included* → keep only **Gemini Apps**
→ Export once, .zip, send download link.

Arrives by email, usually within minutes. Drop the whole `takeout-*.zip` in — unzipping
it loses the attachments, which sit beside the HTML and are pulled into the blob store.

**Re-export every few months.** This is the one source where waiting costs you data:
Google auto-deletes My Activity on a rolling retention window (18 months by default, and
it can be set to 3), so anything older than the window is gone from Google as well as from
here. The current export spans 2025-06-09 → 2026-08-21. Re-dropping is safe — sessions are
keyed on the conversation link and messages on their timestamps, so a fresh export
rewrites only what changed and a *shorter* one does not disturb what is already stored.

> Two things this export does not contain, so do not go looking for them in the archive:
> which Gemini model answered, and any token or cost figure. Google records neither.
> Conversations are not exported either — what you get is an activity log that
> `adapters/gemini.py` regroups back into threads (PLAN §2.4).
>
> Pick HTML, not JSON: the adapter reads HTML today, and a JSON drop is not discovered.

### 6. Grok — **done, and the export names itself after nothing**
`Settings → Data Controls → Export your data`
Arrives by email as a link to a ZIP named after a bare uuid —
`5f4c8d58-c8d3-4b68-9256-13f01162dcb6.zip`. Do not rename it; the adapter identifies it by
the member inside, at `ttl/30d/export_data/<user id>/prod-grok-backend.json`.

The same ZIP also carries `prod-mc-auth-mgmt-api.json`, which is **the most sensitive file
any of these ten sources produces**: account email, linked Google email, birth date, and
per-session IP address, city and latitude/longitude. `adapters/grok.py` never opens it, and
neither does the probe — but the file is still sitting in `data/drops/`, so keep that
folder where it is. It never leaves the machine.

> Three things this export does not contain: token counts, cost, and any reasoning text
> beyond a one-line header per step. `model` on each turn is the UI picker (`build`), not
> the model — the real one is `metadata.request_metadata.resolved_model`.

### 7. Mistral (Le Chat) — **done, and per-chat like OpenRouter**
`Open a chat → ⋯ menu → Export chat` → `chat-export-<epoch ms>.zip`.

**Per chat. There is no bulk export.** Drop the ZIP in unrenamed — the name is the moment
you exported, not the chat, so the adapter identifies it by the member inside
(`chat-<chatId>.json`). Unzipping it is harmless; a bare `chat-<id>.json` is discovered too.

Same advice as OpenRouter below: export by hand the handful you actually care about rather
than grinding through the sidebar.

> Three things this export does not contain: **which model answered** (nothing anywhere
> records it — Le Chat is the second source after Gemini with no model at all), **prompt
> tokens**, and **cost**. Output tokens are there, per answer.
>
> Two more it half-contains: attachments are named but their bytes are not shipped, and
> canvas documents come through as an empty array. Both are counted as unknowns rather
> than guessed at, so `llma doctor` will say the moment a real one appears.

### 8. OpenRouter — the awkward one
`Open a chat → gear cog (bottom-left) → Export Chat → .json`

**Per chat. There is no bulk export**, because OpenRouter's chat is local-first: history
lives in your browser, on that device only.

Do not grind through fifty of these. Instead:

- **Export by hand only the handful you actually care about.**
- For everything else, get statistics without content: `openrouter.ai/activity` has
  per-generation usage and cost, which feeds the dashboard even with zero chat text.
- If you want the full history, Phase 7's opt-in IndexedDB reader gets it in one pass.
  That reads your own browser profile on your own disk — not their site.

### 9. GitHub Copilot on github.com — **done, and the only source with no export at all**

`Open the conversation → Share → copy link` gives you
`github.com/copilot/share/<uuid>`. That link is **not a public page**: fetched without a
browser session it redirects to `/login`, and a `gh` OAuth token does not open it either,
because the route authenticates on session cookies rather than on a bearer token. So
nothing here can download it for you — and per §0 nothing here is going to hold your
GitHub cookies to try.

**Ctrl+S does not work here, and it fails quietly.** Probed on a real save
(`GitHub Copilot.htm`, 73 KB): the share route server-renders nothing but a mount point.
`react-app.embeddedData.payload` is `{}`, the whole document holds **448 characters** of
visible text — cookie banners and "Uh oh! There was an error while loading" — and the
messages are fetched client-side afterwards from `api.individual.githubcopilot.com`
(api version `2025-05-01`). A *Webpage, HTML only* save re-requests that empty shell, so
you get a 73 KB file with no conversation in it and nothing that looks like an error.

You have to capture the traffic, or the page **after** it has rendered.

**B. The API response — use this one.** It is the richest artefact (it carries whatever
model, timestamp and usage fields the rendered page throws away), and the HAR form is the
only capture where you cannot pick the wrong request.

1. Open the share link, press **F12**, click **Network** (*Omrežje*)
2. Tick **Disable cache** (*Onemogoči predpomnilnik*), then **reload the page** — the
   fetch happens on load, so a panel opened afterwards shows nothing, and a cached
   response can export with an empty body
3. Export the whole capture:
   - **Firefox** — right-click any row → **Save All As HAR** (*Shrani vse kot HAR*),
     also on the ⚙ menu at the right of the toolbar
   - **Chrome / Edge** — the **⭳ download icon** → *Export HAR (with content)*;
     "with content" matters, a HAR without response bodies is just a list of URLs
4. Save it into `data/drops/`

> **A HAR of a logged-in session contains live credentials** — your GitHub session cookie
> and a Copilot bearer token sit in the *request* headers. `data/` is gitignored and never
> leaves the machine (§8.6), and the probe reads response bodies only, never request
> headers — but this is the one drop worth deleting once the adapter is built. It is a
> live key, which is worse than the PII in the Grok export.
>
> Prefer not to have it on disk at all? Capture just the one response instead: filter the
> Network panel on `api.individual.githubcopilot.com`, click the request whose **Response**
> holds your messages, then right-click it → **Copy → Copy response**, and paste that into
> `data/drops/copilot-share.json`. No headers, no tokens — you just have to pick the right
> request yourself.

**A. The rendered DOM** — fallback if the network panel is not cooperating.
`F12 → Elements → right-click the top <html> line → Copy → Copy outerHTML`, paste into
`data/drops/copilot-share.html`. Unlike Ctrl+S this is what is actually on screen.

Then shape-probe whichever you captured:

```bash
python tools/probe_copilot_share.py                   # scans data/drops/
python tools/probe_copilot_share.py ~/Downloads/x.har # or point it at one file
```

It reads both containers, tells you where the turns are, and writes a fixture the adapter
gets built against — the same "verify before parsing" step every other adapter here went
through. If you hand it a shell save by mistake it says so explicitly rather than
reporting an empty result.

**Delete the HAR once it is ingested.** `adapters/copilot_web.py` hashes the transcript
rather than the container, so the extracted response body ingests to a byte-identical
session — verified. Pull it out and drop the capture:

```bash
python tools/probe_copilot_share.py data/drops/<capture>.har   # writes the fixture
```

The ingested conversation here came in that way: the HAR was replaced by
`copilot-share-805e1228.json` (130 KB, no headers) and re-ingesting it reports `skipped 1`,
not `updated`.

> Verified on one conversation — 2 messages, 20 parts, 141,844 in / 5,414 out, workspace
> `github.com/example-user/maze_agent`. Three things this capture does not contain:
> **file contents** (every `file` reference ships `content: ""` — the endpoint strips the
> bytes), **cost** (Copilot bills premium requests, not tokens), and any **cache token
> split**.

> **This is not the VS Code Copilot Chat you already have.** Those 110 sessions come off
> local disk via `adapters/vscode_chat.py` and need no export. This entry is the separate
> github.com web surface, which shares nothing with that store.
>
> One saved page is **one conversation**. There is no bulk option, so — as with
> OpenRouter and Mistral — save the handful you actually care about rather than grinding
> through the whole sidebar.

---

## When they land

```bash
python tools/probe_exports.py        # (Phase 3) shape-probe whatever is in data/drops/
```

Then tell me, and I will write the adapters against the real shapes rather than the
assumed ones. Every "unverified" row in `PLAN.md §2` becomes verified at that point.

---

## What is already handled, no action needed

| Source | Status |
|---|---|
| Claude Code | on disk — 66 sessions, probed |
| Codex | on disk — 3 sessions, probed |
| opencode | on disk — 2 sessions, probed |
| **VS Code chat** | on disk — **110 sessions, 59.7 MB**, found in Phase 0, no export needed |
| **Gemini** | ingested — 745 activity cells → 124 sessions, 1,410 messages; re-export before the retention window closes |
| **Grok** | ingested — 1 conversation, 2 messages, 12 parts; re-export whenever, the adapter hashes each chat separately |
| **GitHub Copilot (web)** | ingested — 1 conversation, 2 messages, 20 parts; no export exists, so re-capture per chat (§9) |
| **Mistral** | ingested — 1 chat, 2 messages, 3 parts; per-chat export, so re-drop each chat you want refreshed |
