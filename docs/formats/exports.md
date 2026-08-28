# Export probe

Generated 2026-08-26T22:26:37+00:00 from `C:\Users\alex\Documents\Projekti\LLM_sessions_grouping\data\drops` by `tools/probe_exports.py`.

Files are identified by **sniffing their structure**, never by filename — that is how the ingest watcher will work too.

| kind | files |
|---|---:|
| `openrouter_chat` | 3 |
| `grok_export` | 1 |
| `mistral_chat` | 1 |
| `claude_export` | 1 |
| `deepseek_export` | 1 |
| `gemini_activity` | 1 |
| `t3chat_export` | 1 |

## `grok_export`

**5f4c8d58-c8d3-4b68-9256-13f01162dcb6.zip::ttl/30d/export_data/c2f0fd62-8d42-489a-863d-c03521fc8558/prod-grok-backend.json**

```json
{
  "conversations": 1,
  "responses": 2,
  "senders": {
    "human": 1,
    "assistant": 1
  },
  "models": {
    "": 1,
    "grok-chat-app-builder-free": 1
  },
  "pickers": {
    "": 1,
    "build": 1
  },
  "steps": 4,
  "step_tags": {
    "header": 2,
    "tool_usage_card": 2,
    "raw_function_result": 2
  },
  "tool_calls": {
    "ReadFile": 1,
    "WebSearch": 1,
    "InitTerminalSession": 1,
    "BrowsePage": 2
  },
  "tool_results": 3,
  "calls_without_result": 2,
  "search_hits_aggregate": 9,
  "branched_conversations": 0,
  "conversations_without_leaf_in_export": 0,
  "text_chars": 4291,
  "side_collections": {
    "projects": 0,
    "tasks": 0,
    "media_posts": 0
  },
  "token_counts": false,
  "text_path": "conversations[*].responses[*].response.message",
  "role_path": "response.sender: human=user, assistant=assistant",
  "model_path": "response.metadata.request_metadata.resolved_model (response.model is the UI picker)",
  "join_path": "response.parent_response_id -> response._id; conversation.leaf_response_id names the surviving leaf"
}
```

## `mistral_chat`

**chat-export-1787777208552.zip::chat-bda873f6-e352-4b55-9d77-398dd4ba6446.json**

```json
{
  "chats": 1,
  "messages": 2,
  "roles": {
    "user": 1,
    "assistant": 1
  },
  "chunk_types": {
    "text": 2
  },
  "chunk_contexts": {
    "reasoning": 1,
    "-": 1
  },
  "answers_duplicated_in_content": 1,
  "reasoning_chars": 840,
  "text_chars": 3671,
  "versions": {
    "0": 2
  },
  "tok_out": 1266,
  "models_recorded": 0,
  "canvas_entries": 0,
  "files": 0
}
```

## `claude_export`

**conversations-000.zip::conversations.json**

```json
{
  "top_level": "array",
  "n": 53,
  "item_keys": [
    "account",
    "chat_messages",
    "created_at",
    "name",
    "summary",
    "updated_at",
    "uuid"
  ]
}
```

## `deepseek_export`

**deepseek_data-2026-08-27.zip::conversations.json**

```json
{
  "conversations": 1,
  "nodes": 3,
  "roles": {
    "user": 1,
    "assistant": 1
  },
  "fragment_types": {
    "REQUEST": 1,
    "SEARCH": 1,
    "RESPONSE": 1
  },
  "models": {
    "deepseek-chat": 2
  },
  "branched_conversations": 0,
  "backwards_timestamps": 1,
  "search_hits": 12,
  "text_chars": 3361,
  "current_node": false,
  "token_counts": false,
  "text_path": "mapping[*].message.fragments[*].content",
  "role_path": "fragment type: REQUEST=user, RESPONSE/SEARCH=assistant",
  "model_path": "mapping[*].message.model  (set on prompts too; selected, not speaker)"
}
```

## `openrouter_chat`

**OpenRouter Chat Tue Aug 25 2026(1).json**

```json
{
  "title": "Hey, can you tell me something about pro",
  "schema_version": "orpg.3.0",
  "messages": 2,
  "items": 2,
  "characters": {
    "char-1787652449-Ziw0r3rWdahbWNNgVrrz": "minimax/minimax-m3:free"
  },
  "roles": {
    "user": 1,
    "assistant": 1
  },
  "text_chars": 3249,
  "artifacts": 0,
  "edited_msgs": 0,
  "retried_msgs": 0,
  "text_path": "items[*].data.content[*].text",
  "role_path": "items[*].data.role  (characterId=='USER' for user)",
  "model_path": "characters[messages[*].characterId].model"
}
```

**OpenRouter Chat Tue Aug 25 2026(2).json**

```json
{
  "title": "Hey, can you tell me what is the weather",
  "schema_version": "orpg.3.0",
  "messages": 2,
  "items": 3,
  "characters": {
    "char-1787652549-gGf1Y4T9gHLo6hzrhpHN": "minimax/minimax-m3:free",
    "char-1787652564-yfy6nZQkcGt7UnxaRHQJ": "dots-studio/dots-3-note-preview:free",
    "char-1787652574-uOA5ncaCsNV7CPYrgxy3": "google/gemma-4-31b-it:free"
  },
  "roles": {
    "user": 1,
    "None": 1,
    "assistant": 1
  },
  "text_chars": 710,
  "artifacts": 0,
  "edited_msgs": 0,
  "retried_msgs": 0,
  "text_path": "items[*].data.content[*].text",
  "role_path": "items[*].data.role  (characterId=='USER' for user)",
  "model_path": "characters[messages[*].characterId].model"
}
```

**OpenRouter Chat Tue Aug 25 2026.json**

```json
{
  "title": "Hey, can you tell me some things about p",
  "schema_version": "orpg.3.0",
  "messages": 2,
  "items": 3,
  "characters": {
    "char-1787652480-XhVPeLClN7LyVC5phy8c": "nvidia/nemotron-3.5-lightning:free"
  },
  "roles": {
    "user": 1,
    "None": 1,
    "assistant": 1
  },
  "text_chars": 6639,
  "artifacts": 0,
  "edited_msgs": 0,
  "retried_msgs": 0,
  "text_path": "items[*].data.content[*].text",
  "role_path": "items[*].data.role  (characterId=='USER' for user)",
  "model_path": "characters[messages[*].characterId].model"
}
```

## `gemini_activity`

**takeout-20260826T202708Z-1-001.zip::Moja_dejavnost.html**

```json
{
  "cells": 745,
  "threads": 84,
  "canvases": 52,
  "prompts": 680,
  "markers_without_content": 13,
  "attachments": 15,
  "activity_labels": {
    "Poslali ste poziv:": 680,
    "Ustvarjeno platno Gemini z naslovom": 52,
    "Uporabljena je bila funkcija Pomočnika": 12,
    "Izbran je prednostni osnutek": 1
  },
  "biggest_thread": 58,
  "timezones": {
    "CEST": 745
  },
  "span": [
    "2025-04-28",
    "2026-08-21"
  ],
  "text_chars": 3276476,
  "unparsed": {},
  "roles": false,
  "model": false,
  "token_counts": false,
  "text_path": "outer-cell > content-cell.body-1, split on the localised stamp",
  "role_path": "before the stamp = you, after it = Gemini; NBSP+colon marks a prompt",
  "join_path": "caption > gemini.google.com/app/<id>  (absent on canvases)"
}
```

## `t3chat_export`

**threads-export-2026-08-26T15_25_39.018Z.json**

```json
{
  "schema_version": "11.0.1",
  "threads": 170,
  "messages": 1430,
  "roles": {
    "user": 715,
    "assistant": 715
  },
  "statuses": {
    "done": 1422,
    "cancelled": 3,
    "error": 5
  },
  "models": {
    "gemini-2.5-flash": 402,
    "gpt-5.2-instant": 364,
    "claude-4.5-opus": 304,
    "claude-4.5-sonnet": 96,
    "gpt-5.5": 62,
    "gemini-3-pro-image-preview": 44,
    "gemini-3-flash-thinking": 40,
    "gpt-5.2-reasoning": 24
  },
  "part_types": {
    "text": 693,
    "tool_call": 60,
    "reasoning": 284
  },
  "tools": {
    "webSearch": 38,
    "image_generation": 22
  },
  "usage_blocks": {
    "google": 232,
    "anthropic": 203,
    "openai": 206,
    "openrouter": 45
  },
  "text_chars": 3519605,
  "threads_without_messages": 0,
  "msgs_without_thread": 0,
  "attachment_refs": 90,
  "parent_pointers": 0,
  "text_path": "messages[*].parts[*].text  (fallback messages[*].content)",
  "join_path": "messages[*].threadId -> threads[*].threadId",
  "model_path": "messages[*].model  (threads[*].model is only the last pick)"
}
```
