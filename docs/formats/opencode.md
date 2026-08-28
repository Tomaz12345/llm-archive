# Format probe — `opencode`

Generated 2026-08-25T10:30:20+00:00 by `tools/probe.py`.

- files scanned: **328**
- records parsed: **328**
- parse errors: **0**

## Record types

| type | count | distinct key-sets |
|---|---:|---:|
| `part` | 261 | 7 |
| `message` | 58 | 2 |
| `project` | 3 | 3 |
| `session` | 2 | 1 |
| `session_diff[]` | 2 | 1 |
| `todo[]` | 2 | 1 |

## Shapes

### `message`

- **49×** — `agent`, `cost`, `finish`, `id`, `mode`, `modelID`, `parentID`, `path`, `providerID`, `role`, `sessionID`, `time`, `tokens`
- **9×** — `agent`, `id`, `model`, `role`, `sessionID`, `summary`, `time`

### `part`

- **84×** — `callID`, `id`, `messageID`, `sessionID`, `state`, `tool`, `type`
- **70×** — `id`, `messageID`, `sessionID`, `text`, `time`, `type`
- **37×** — `id`, `messageID`, `sessionID`, `type`
- **37×** — `cost`, `id`, `messageID`, `reason`, `sessionID`, `tokens`, `type`
- **12×** — `id`, `messageID`, `sessionID`, `snapshot`, `type`
- **12×** — `cost`, `id`, `messageID`, `reason`, `sessionID`, `snapshot`, `tokens`, `type`
- **9×** — `id`, `messageID`, `sessionID`, `text`, `type`

### `part:reasoning`

- **49×** — `id`, `messageID`, `sessionID`, `text`, `time`, `type`

### `part:step-finish`

- **37×** — `cost`, `id`, `messageID`, `reason`, `sessionID`, `tokens`, `type`
- **12×** — `cost`, `id`, `messageID`, `reason`, `sessionID`, `snapshot`, `tokens`, `type`

### `part:step-start`

- **37×** — `id`, `messageID`, `sessionID`, `type`
- **12×** — `id`, `messageID`, `sessionID`, `snapshot`, `type`

### `part:text`

- **21×** — `id`, `messageID`, `sessionID`, `text`, `time`, `type`
- **9×** — `id`, `messageID`, `sessionID`, `text`, `type`

### `part:tool`

- **84×** — `callID`, `id`, `messageID`, `sessionID`, `state`, `tool`, `type`

### `project`

- **1×** — `icon`, `id`, `sandboxes`, `time`, `vcs`, `worktree`
- **1×** — `id`, `sandboxes`, `time`, `vcs`, `worktree`
- **1×** — `id`, `sandboxes`, `time`, `worktree`

### `session`

- **2×** — `directory`, `id`, `projectID`, `slug`, `summary`, `time`, `title`, `version`

### `session_diff[]`

- **2×** — `<empty>`

### `todo[]`

- **2×** — `content`, `id`, `priority`, `status`

## Observations

### entity file counts

- `part` — 261
- `message` — 58
- `project` — 3
- `session` — 2
- `session_diff` — 2
- `todo` — 2

### part types

- `tool` — 84
- `step-start` — 49
- `reasoning` — 49
- `step-finish` — 49
- `text` — 30

### message roles

- `assistant` — 49
- `user` — 9

### models used

- `kimi-k2.5-free` — 9

### providers

- `opencode` — 9

### agents

- `build` — 34
- `plan` — 24
