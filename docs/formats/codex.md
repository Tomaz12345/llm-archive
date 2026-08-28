# Format probe — `codex`

Generated 2026-08-25T08:38:34+00:00 by `tools/probe.py`.

- files scanned: **3**
- records parsed: **414**
- parse errors: **0**

## Record types

| type | count | distinct key-sets |
|---|---:|---:|
| `response_item/reasoning` | 96 | 1 |
| `event_msg/token_count` | 89 | 1 |
| `response_item/function_call` | 63 | 1 |
| `response_item/function_call_output` | 63 | 1 |
| `response_item/message` | 21 | 1 |
| `response_item/custom_tool_call` | 17 | 1 |
| `response_item/custom_tool_call_output` | 17 | 1 |
| `response_item/web_search_call` | 15 | 1 |
| `event_msg/task_started` | 6 | 1 |
| `turn_context` | 6 | 1 |
| `event_msg/user_message` | 6 | 1 |
| `event_msg/task_complete` | 5 | 1 |
| `event_msg/agent_message` | 4 | 1 |
| `session_meta` | 3 | 1 |
| `event_msg/thread_rolled_back` | 1 | 1 |
| `event_msg/item_completed` | 1 | 1 |
| `event_msg/turn_aborted` | 1 | 1 |

## Shapes

### `event_msg/agent_message`

- **4×** — `payload`, `timestamp`, `type`

### `event_msg/item_completed`

- **1×** — `payload`, `timestamp`, `type`

### `event_msg/task_complete`

- **5×** — `payload`, `timestamp`, `type`

### `event_msg/task_started`

- **6×** — `payload`, `timestamp`, `type`

### `event_msg/thread_rolled_back`

- **1×** — `payload`, `timestamp`, `type`

### `event_msg/token_count`

- **89×** — `payload`, `timestamp`, `type`

### `event_msg/turn_aborted`

- **1×** — `payload`, `timestamp`, `type`

### `event_msg/user_message`

- **6×** — `payload`, `timestamp`, `type`

### `payload:event_msg/agent_message`

- **3×** — `message`, `phase`, `type`
- **1×** — `memory_citation`, `message`, `phase`, `type`

### `payload:event_msg/item_completed`

- **1×** — `item`, `thread_id`, `turn_id`, `type`

### `payload:event_msg/task_complete`

- **5×** — `last_agent_message`, `turn_id`, `type`

### `payload:event_msg/task_started`

- **6×** — `collaboration_mode_kind`, `model_context_window`, `turn_id`, `type`

### `payload:event_msg/thread_rolled_back`

- **1×** — `num_turns`, `type`

### `payload:event_msg/token_count`

- **89×** — `info`, `rate_limits`, `type`

### `payload:event_msg/turn_aborted`

- **1×** — `reason`, `turn_id`, `type`

### `payload:event_msg/user_message`

- **6×** — `images`, `local_images`, `message`, `text_elements`, `type`

### `payload:response_item/custom_tool_call`

- **17×** — `call_id`, `input`, `name`, `status`, `type`

### `payload:response_item/custom_tool_call_output`

- **17×** — `call_id`, `output`, `type`

### `payload:response_item/function_call`

- **63×** — `arguments`, `call_id`, `name`, `type`

### `payload:response_item/function_call_output`

- **63×** — `call_id`, `output`, `type`

### `payload:response_item/message`

- **21×** — `content`, `role`, `type`

### `payload:response_item/reasoning`

- **96×** — `content`, `encrypted_content`, `summary`, `type`

### `payload:response_item/web_search_call`

- **15×** — `action`, `status`, `type`

### `payload:session_meta`

- **1×** — `base_instructions`, `cli_version`, `cwd`, `id`, `model_provider`, `originator`, `source`, `timestamp`
- **1×** — `base_instructions`, `cli_version`, `cwd`, `git`, `id`, `model_provider`, `originator`, `source`, `timestamp`
- **1×** — `base_instructions`, `cli_version`, `cwd`, `dynamic_tools`, `id`, `model_provider`, `originator`, `source`, `timestamp`

### `payload:turn_context`

- **5×** — `approval_policy`, `collaboration_mode`, `current_date`, `cwd`, `effort`, `model`, `personality`, `realtime_active`, `sandbox_policy`, `summary`, `timezone`, `truncation_policy`, `turn_id`, `user_instructions`
- **1×** — `approval_policy`, `collaboration_mode`, `current_date`, `cwd`, `developer_instructions`, `effort`, `model`, `personality`, `realtime_active`, `sandbox_policy`, `summary`, `timezone`, `truncation_policy`, `turn_id`

### `response_item/custom_tool_call`

- **17×** — `payload`, `timestamp`, `type`

### `response_item/custom_tool_call_output`

- **17×** — `payload`, `timestamp`, `type`

### `response_item/function_call`

- **63×** — `payload`, `timestamp`, `type`

### `response_item/function_call_output`

- **63×** — `payload`, `timestamp`, `type`

### `response_item/message`

- **21×** — `payload`, `timestamp`, `type`

### `response_item/reasoning`

- **96×** — `payload`, `timestamp`, `type`

### `response_item/web_search_call`

- **15×** — `payload`, `timestamp`, `type`

### `session_meta`

- **3×** — `payload`, `timestamp`, `type`

### `turn_context`

- **6×** — `payload`, `timestamp`, `type`

## Observations

### envelope/payload types

- `response_item/reasoning` — 96
- `event_msg/token_count` — 89
- `response_item/function_call` — 63
- `response_item/function_call_output` — 63
- `response_item/message` — 21
- `response_item/custom_tool_call` — 17
- `response_item/custom_tool_call_output` — 17
- `response_item/web_search_call` — 15
- `event_msg/task_started` — 6
- `turn_context` — 6
- `event_msg/user_message` — 6
- `event_msg/task_complete` — 5
- `event_msg/agent_message` — 4
- `session_meta` — 3
- `event_msg/thread_rolled_back` — 1
- `event_msg/item_completed` — 1
- `event_msg/turn_aborted` — 1

### originators

- `codex_vscode` — 2
- `Codex Desktop` — 1

### cli versions

- `0.108.0-alpha.12` — 1
- `0.115.0-alpha.11` — 1
- `0.119.0-alpha.11` — 1

### model providers

- `openai` — 3

### response_item roles

- `user` — 11
- `developer` — 6
- `assistant` — 4

### token_count payload examples

```json
[
  {
    "type": "token_count",
    "info": {
      "total_token_usage": {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0
      },
      "last_token_usage": {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 8356
      },
      "model_context_window": 258400
    },
    "rate_limits": null
  },
  {
    "type": "token_count",
    "info": {
      "total_token_usage": {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0
      },
      "last_token_usage": {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 8356
      },
      "model_context_window": 258400
    },
    "rate_limits": {
      "limit_id": "codex",
      "limit_name": null,
      "primary": {
        "used_percent": 1.0,
        "window_minutes": 10080,
        "resets_at": 1773319899
      },
      "secondary": null,
      "credits": null,
      "plan_type": "free"
    }
  },
  {
    "type": "token_count",
    "info": {
      "total_token_usage": {
        "input_tokens": 18049,
        "cached_input_tokens": 8704,
        "output_tokens": 91,
        "reasoning_output_tokens": 0,
        "total_tokens": 18140
      },
      "last_token_usage": {
        "input_tokens": 18049,
        "cached_input_tokens": 8704,
        "output_tokens": 91,
        "reasoning_output_tokens": 0,
        "total_tokens": 18140
      },
      "model_context_window": 258400
    },
    "rate_limits": {
      "limit_id": "codex",
      "limit_name": null,
      "primary": {
        "used_percent": 1.0,
        "window_minutes": 10080,
        "resets_at": 1773319899
      },
      "secondary": null,
      "credits": null,
      "plan_type": "free"
    }
  }
]
```

### session_index.jsonl

```json
{
  "rows": 3,
  "example": {
    "id": "019cd7c7-70c5-73c1-b229-e510458ae87f",
    "thread_name": "Plan multilingual Slovene app",
    "updated_at": "2026-03-10T12:50:03.3988815Z"
  }
}
```
