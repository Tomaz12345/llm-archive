# ChatGPT export

`Settings → Data controls → Export data`. Arrives by email as a link to a ZIP.

```
<export>.zip
  ├── conversations.json    every chat in the account
  ├── chat.html             the same thing rendered; not read
  ├── user.json             account identity; not read (§8.6, as with DeepSeek)
  ├── message_feedback.json thumbs up/down; not read
  └── file-<id>-<name>.png  images you uploaded, and ones the model generated
```

Add the **whole ZIP** — `llma add ~/Downloads/<export>.zip`. Extracting
`conversations.json` on its own loses every image in the account, because the pointers
inside it only resolve against members of the same archive.

## Shape

```jsonc
[ { "id": "...", "conversation_id": "...", "title": "...",
    "create_time": 1699999999.123,        // float epoch SECONDS, not ms
    "update_time": 1699999999.9,
    "current_node": "<node id>",          // the leaf that survived
    "default_model_slug": "gpt-5",
    "mapping": {
      "<node id>": {
        "id": "<node id>", "parent": "<node id>|null", "children": ["..."],
        "message": {
          "id": "...",
          "author": { "role": "system|user|assistant|tool", "name": null },
          "create_time": 1699999999.1,
          "weight": 1.0,
          "recipient": "all",             // or a tool name
          "content": { "content_type": "...", ... },
          "metadata": { "model_slug": "gpt-5", ... }
        } | null } } } ]
```

Structurally this is the format DeepSeek's export imitates, so `adapters/chatgpt.py`
follows `adapters/deepseek.py` throughout — same walk, same per-conversation hashing.

## The seven things that decide how it is read

**1. `current_node` states the answer DeepSeek has to guess.** DeepSeek's identical tree
has no such pointer, so it falls back to "newest leaf wins" (§8.1). Here the active path
is walked up from `current_node`, and the shared resolver is only the fallback for a
pointer that is missing or dangles — which happens in an export taken while a response
was still streaming. This matters more here than anywhere else: ChatGPT's regenerate
button gets used, and a guessed leaf files the abandoned answer as the real one.

**2. `content_type` is nine shapes, not one.** Reading `parts` unconditionally — the
obvious implementation — silently drops every reasoning block and every tool call:

| `content_type` | where the text is | stored as |
|---|---|---|
| `text` | `parts: [str, …]` | text — or **tool_use** when `recipient != "all"` |
| `multimodal_text` | `parts: [pointer\|str, …]` | image + text |
| `code` | `text` | tool_use |
| `execution_output` | `text` | tool_result |
| `thoughts` | `thoughts: [{summary, content}]` | thinking |
| `reasoning_recap` | `content` | thinking |
| `tether_browsing_display` | `result` / `text` | tool_result |
| `tether_quote` | `title`, `url`, `text` | tool_result |
| `user_editable_context` | `user_profile`, `user_instructions` | text |

Anything else is counted in `unknown_types` and shows up in `llma doctor`, rather than
being guessed at.

**3. Hidden system messages are structure, not content.** A conversation's root is
normally an empty `system` message, and custom-instruction turns carry
`metadata.is_visually_hidden_from_conversation`. They are dropped as messages but kept in
the graph — they are what joins the first real turn to the root, and filtering before
building the tree shatters the chain (the §8.1 trap, first hit in Claude Code).

**4. `recipient` is what makes a message a tool call.** The role stays `assistant` when
it calls `python` or `browser`. Without reading `recipient`, generated code reads as
something the model said out loud and `is_turn` counts a tool step as a conversational
turn — which is exactly the cross-source comparability `turn_count` exists to protect.

**5. `weight: 0` marks a turn the model was told to forget.** Kept, off the active path,
for the same reason abandoned branches are kept: it is real history.

**6. Images ship as real bytes** — as in Gemini and unlike every other web export. The
pointer is `file-service://file-<id>` and the member is `file-<id>-<name>.<ext>`, joined
on that id, and the bytes go to the blob store. When the member is missing (DALL·E output
ages out of the export) the part records the reference and the size the export claims,
rather than nothing.

**7. No usage anywhere.** No token counts, no cost, on any record. Those columns are a
real gap for this source, not a zero. `model_slug` names the model, not what it spent.

## Discovery

Three sources ship a file called `conversations.json`: claude.ai, DeepSeek and ChatGPT.
`ChatGPTAdapter.claims()` requires `"mapping"` and `"author"`, and rejects
`"chat_messages"` (claude.ai) and `"inserted_at"` (DeepSeek).

`author` is the positive marker rather than the more obvious `current_node`, which does
not work: the sniff window is 64 KB off the front of the file, `current_node` is written
*after* the whole of the first conversation's mapping, and one long chat pushes it far
past that. `author` appears inside the first message node, a few hundred bytes in.
`tests/test_chatgpt.py::test_identified_without_current_node_in_the_sniff_window` pins
this.

## Timestamps

`create_time` is a float epoch **second**, unlike every other source here, and is `null`
on system nodes and on messages that were still streaming when the export ran.
