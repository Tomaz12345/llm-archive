# Format probe — `claude_code`

Generated 2026-08-25T10:30:20+00:00 by `tools/probe.py`.

- files scanned: **68**
- records parsed: **20459**
- parse errors: **0**

## Record types

| type | count | distinct key-sets |
|---|---:|---:|
| `assistant` | 8177 | 16 |
| `user` | 4704 | 26 |
| `ai-title` | 1449 | 1 |
| `attachment` | 1428 | 5 |
| `last-prompt` | 1360 | 2 |
| `bridge-session` | 1283 | 2 |
| `queue-operation` | 682 | 2 |
| `atis-latch` | 289 | 1 |
| `file-history-delta` | 268 | 1 |
| `file-history-snapshot` | 230 | 1 |
| `mode` | 201 | 1 |
| `permission-mode` | 179 | 1 |
| `system` | 88 | 6 |
| `agent-name` | 82 | 1 |
| `frame-link` | 30 | 2 |
| `artifact-autoreact-ledger` | 7 | 1 |
| `artifact-comment-monitor` | 2 | 1 |

## Shapes

### `agent-name`

- **82×** — `agentName`, `sessionId`, `type`

### `ai-title`

- **1449×** — `aiTitle`, `sessionId`, `type`

### `artifact-autoreact-ledger`

- **7×** — `accountUuid`, `artifacts`, `sessionId`, `type`, `v`

### `artifact-comment-monitor`

- **2×** — `artifacts`, `sessionId`, `type`, `v`

### `assistant`

- **5334×** — `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1230×** — `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **434×** — `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `session_id`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **250×** — `attributionMcpServer`, `attributionMcpTool`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `session_id`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **246×** — `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `session_id`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **231×** — `attributionMcpServer`, `attributionMcpTool`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `session_id`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **191×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **166×** — `attributionSkill`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **34×** — `agentId`, `attributionAgent`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **33×** — `attributionMcpServer`, `attributionMcpTool`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **16×** — `attributionSkill`, `cwd`, `effort`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **5×** — `apiErrorStatus`, `cwd`, `entrypoint`, `error`, `gitBranch`, `isApiErrorMessage`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **4×** — `attributionMcpServer`, `attributionMcpTool`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `apiErrorStatus`, `cwd`, `entrypoint`, `error`, `gitBranch`, `isApiErrorMessage`, `isSidechain`, `message`, `parentUuid`, `sessionId`, `session_id`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `apiErrorStatus`, `cwd`, `entrypoint`, `error`, `gitBranch`, `isApiErrorMessage`, `isSidechain`, `message`, `parentUuid`, `requestId`, `sessionId`, `session_id`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isApiErrorMessage`, `isSidechain`, `message`, `parentUuid`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`

### `atis-latch`

- **289×** — `atis`, `sessionId`, `type`

### `attachment`

- **1134×** — `attachment`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `parentUuid`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **155×** — `attachment`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `parentUuid`, `sessionId`, `session_id`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **74×** — `attachment`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `parentUuid`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **63×** — `attachment`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `parentUuid`, `sessionId`, `session_id`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **2×** — `agentId`, `attachment`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `parentUuid`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`

### `bridge-session`

- **1282×** — `bridgeSessionId`, `lastSequenceNum`, `ownerAccountUuid`, `ownerOrganizationUuid`, `sessionId`, `type`
- **1×** — `bridgeSessionId`, `lastSequenceNum`, `sessionId`, `type`

### `content:document`

- **3×** — `source`, `type`
- **2×** — `source`, `title`, `type`

### `content:image`

- **8×** — `source`, `type`

### `content:text`

- **1788×** — `text`, `type`

### `content:thinking`

- **2308×** — `signature`, `thinking`, `type`

### `content:tool_result`

- **2350×** — `content`, `is_error`, `tool_use_id`, `type`
- **2028×** — `content`, `tool_use_id`, `type`

### `content:tool_use`

- **4379×** — `caller`, `id`, `input`, `name`, `type`

### `file-history-delta`

- **268×** — `backup`, `messageId`, `snapshotMessageId`, `timestamp`, `trackingPath`, `type`

### `file-history-snapshot`

- **230×** — `isSnapshotUpdate`, `messageId`, `snapshot`, `type`

### `frame-link`

- **28×** — `artifactCount`, `sessionId`, `timestamp`, `type`
- **2×** — `artifactCount`, `frameUrl`, `path`, `sessionId`, `timestamp`, `title`, `type`

### `last-prompt`

- **1358×** — `lastPrompt`, `leafUuid`, `sessionId`, `type`
- **2×** — `leafUuid`, `sessionId`, `type`

### `mode`

- **201×** — `mode`, `sessionId`, `type`

### `permission-mode`

- **179×** — `permissionMode`, `sessionId`, `type`

### `queue-operation`

- **411×** — `operation`, `sessionId`, `timestamp`, `type`
- **271×** — `content`, `operation`, `sessionId`, `timestamp`, `type`

### `system`

- **29×** — `cwd`, `durationMs`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `messageCount`, `parentUuid`, `sessionId`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **18×** — `cwd`, `entrypoint`, `error`, `gitBranch`, `isSidechain`, `level`, `maxRetries`, `parentUuid`, `retryAttempt`, `retryInMs`, `sessionId`, `source`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **14×** — `cwd`, `durationMs`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `messageCount`, `parentUuid`, `sessionId`, `slug`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **13×** — `content`, `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `parentUuid`, `sessionId`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **8×** — `content`, `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `parentUuid`, `sessionId`, `slug`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **6×** — `content`, `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `level`, `parentUuid`, `sessionId`, `subtype`, `timestamp`, `type`, `userType`, `uuid`, `version`

### `user`

- **2965×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **823×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **245×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `origin`, `parentUuid`, `permissionMode`, `promptId`, `promptSource`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **195×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **134×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `slug`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **100×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `mcpMeta`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **98×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `mcpMeta`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `slug`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **26×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `mcpMeta`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **24×** — `agentId`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `sourceToolAssistantUUID`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **22×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **22×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `origin`, `parentUuid`, `permissionMode`, `promptId`, `promptSource`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **14×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **10×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `permissionMode`, `promptId`, `promptSource`, `sessionId`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **6×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `timestamp`, `turnCompanion`, `type`, `userType`, `uuid`, `version`
- **4×** — `classifierMetaLines`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **3×** — `classifierMetaLines`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **2×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `sessionId`, `session_id`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **2×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `mcpMeta`, `message`, `parentUuid`, `sessionId`, `session_id`, `sourceToolAssistantUUID`, `timestamp`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **2×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolAssistantUUID`, `timestamp`, `toolDenialKind`, `toolUseResult`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `session_id`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolUseID`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `sourceToolUseID`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `agentId`, `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `slug`, `timestamp`, `type`, `userType`, `uuid`, `version`
- **1×** — `cwd`, `entrypoint`, `gitBranch`, `isMeta`, `isSidechain`, `message`, `parentUuid`, `promptId`, `sessionId`, `sourceToolUseID`, `timestamp`, `turnCompanion`, `type`, `userType`, `uuid`, `version`

## Observations

### metadata-only session files

Some `.jsonl` files hold no conversation at all — a `last-prompt` leaf pointer, a `mode`,
a `permission-mode` and a `system`/`local_command` record, and nothing else. The records
*are* content types, so a `if not nodes` guard does not catch them; they simply build no
message.

Stored, such a session lands with `msg_count = 0` and `started_at = 0`: it counts under
`by_source` but drops out of every chart keyed on time, so the two disagree by one for no
reason. One file on this machine
(`C--Users-alex/72ef7346-….jsonl`) was exactly this. The adapter now returns `None` on an
empty message list, matching Codex, opencode and VS Code chat.

### persisted tool-results (sidecar files)

```json
{
  "dirs": 11,
  "files": 36,
  "bytes": 3587982,
  "note": "referenced from tool_result via a <persisted-output> marker; resolve relative to the session dir, not the embedded absolute path"
}
```

### per-project memory files

```json
[
  "c--Users-alex-Documents-Projekti-Quiz-app-core-quiz-app-api\\memory\\MEMORY.md",
  "c--Users-alex-Documents-Projekti-Invoice-integration\\memory\\project_notes.md",
  "c--Users-alex-Documents-Projekti-Invoice-integration\\memory\\MEMORY.md",
  "c--Users-alex-Documents-Projekti-Invoice-integration\\memory\\project_invoice_manager.md",
  "c--Users-alex-Documents-Projekti-Invoice-integration\\memory\\project_admin_dashboard.md",
  "c--Users-alex-Documents-Projekti-Invoice-integration\\memory\\team_conventions.md",
  "c--Users-alex-Documents-Projekti-telemetry-analysis\\memory\\hold-commits-until-done.md",
  "c--Users-alex-Documents-Projekti-telemetry-analysis\\memory\\MEMORY.md",
  "c--Users-alex-Documents-Projekti-telemetry-analysis\\memory\\python-env.md",
  "c--Users-alex-Documents-Projekti-telemetry-analysis\\memory\\sample-video-source.md",
  "C--Users-alex-Documents-Projekti-Photo-gallery\\memory\\MEMORY.md",
  "C--Users-alex-Documents-Projekti-Photo-gallery\\memory\\port-8000-taken-by-vscode.md"
]
```

### R-workspace — project dirs with >1 cwd

```json
{
  "C--Users-alex": {
    "C:\\Users\\alex": 1856,
    "C:\\Users\\alex\\OneDrive\\Desktop\\UsefulDocs": 12,
    "C:\\Users\\alex\\OneDrive\\Desktop\\Misc_random": 38
  },
  "c--Users-alex-Documents-Projekti-traffic-analysis": {
    "c:\\Users\\alex\\Documents\\Projekti\\traffic_analysis": 124,
    "C:\\Users\\alex\\Documents\\Projekti\\traffic_analysis": 158
  },
  "c--Users-alex-Documents-Projekti-Invoice-integration": {
    "c:\\Users\\alex\\Documents\\Projekti\\Invoice_integration": 1402,
    "C:\\Users\\alex\\Documents\\Projekti\\Invoice_integration": 915,
    "C:\\Users\\alex\\Documents\\Projekti\\Invoice_integration\\admin_dashboard": 1151,
    "C:\\Users\\alex\\Documents\\Projekti\\Invoice_integration\\admin_dashboard\\jobs\\templates\\jobs": 98,
    "C:\\Users\\alex\\Documents\\Projekti\\Invoice_integration\\admin_dashboard\\jobs": 6,
    "C:\\Users\\alex\\Documents\\Projekti\\Invoice_integration\\output\\render_batch": 14
  },
  "c--Users-alex-Documents-Projekti-telemetry-analysis": {
    "c:\\Users\\alex\\Documents\\Projekti\\telemetry_analysis": 947,
    "C:\\Users\\alex\\Documents\\Projekti\\telemetry_analysis": 6916,
    "C:\\Users\\alex\\Documents\\Projekti\\telemetry_analysis\\telemetry_analysis": 83,
    "C:\\Users\\alex\\Documents\\Projekti\\telemetry_analysis\\input_videos": 38
  },
  "c--Users-alex-Documents-Projekti-LLM-sessions-grouping": {
    "c:\\Users\\alex\\Documents\\Projekti\\LLM_sessions_grouping": 199,
    "C:\\Users\\alex\\Documents\\Projekti\\LLM_sessions_grouping": 379
  }
}
```

### cli versions seen

- `2.1.233` — 7868
- `2.1.241` — 2307
- `2.1.228` — 1438
- `2.1.220` — 1084
- `2.1.235` — 701
- `2.1.239` — 241
- `2.1.229` — 219
- `2.1.237` — 186
- `2.1.209` — 130
- `2.1.173` — 109
- `2.1.207` — 106
- `2.1.177` — 8

### record types by version

```json
{
  "2.1.173": {
    "user": 40,
    "attachment": 10,
    "assistant": 59
  },
  "2.1.177": {
    "system": 2,
    "user": 6
  },
  "2.1.207": {
    "user": 32,
    "attachment": 10,
    "assistant": 64
  },
  "2.1.209": {
    "user": 46,
    "attachment": 11,
    "assistant": 73
  },
  "2.1.220": {
    "user": 400,
    "attachment": 66,
    "assistant": 618
  },
  "2.1.228": {
    "system": 44,
    "user": 441,
    "attachment": 97,
    "assistant": 856
  },
  "2.1.229": {
    "user": 57,
    "attachment": 52,
    "assistant": 105,
    "system": 5
  },
  "2.1.233": {
    "user": 2769,
    "attachment": 296,
    "assistant": 4785,
    "system": 18
  },
  "2.1.235": {
    "user": 181,
    "attachment": 192,
    "assistant": 319,
    "system": 9
  },
  "2.1.237": {
    "user": 52,
    "attachment": 51,
    "assistant": 83
  },
  "2.1.239": {
    "user": 58,
    "attachment": 50,
    "assistant": 123,
    "system": 10
  },
  "2.1.241": {
    "user": 622,
    "attachment": 593,
    "assistant": 1092
  }
}
```

### message content block types

- `tool_use` — 4379
- `tool_result` — 4378
- `thinking` — 2308
- `text` — 1788
- `<bare string>` — 138
- `image` — 8
- `document` — 5

### message roles

- `assistant` — 8177
- `user` — 4704

### tool_use names

- `Bash` — 2122
- `Edit` — 634
- `Read` — 495
- `mcp__blender__execute_blender_code` — 217
- `PowerShell` — 213
- `Write` — 190
- `WebFetch` — 139
- `WebSearch` — 121
- `TaskOutput` — 37
- `ToolSearch` — 35
- `SendUserFile` — 32
- `Grep` — 29
- `mcp__blender__get_viewport_screenshot` — 26
- `AskUserQuestion` — 22
- `mcp__claude_ai_Gmail__get_message` — 19
- `Glob` — 8
- `TodoWrite` — 8
- `EnterPlanMode` — 5
- `ExitPlanMode` — 5
- `Monitor` — 3
- `Skill` — 3
- `mcp__claude_ai_Gmail__search_threads` — 3
- `mcp__claude_ai_Gmail__get_thread` — 2
- `NotebookEdit` — 2
- `Artifact` — 2
- `mcp__blender__get_scene_info` — 1
- `mcp__blender__get_object_info` — 1
- `mcp__claude_ai_Gmail__create_draft` — 1
- `mcp__claude_ai_Google_Drive__search_files` — 1
- `mcp__claude_ai_Google_Drive__download_file_content` — 1
- `TaskStop` — 1
- `Agent` — 1

### entrypoints

- `claude-vscode` — 12314
- `cli` — 2083

### isSidechain values

```json
{
  "False": 14336,
  "True": 61
}
```

### files per project

- `c--Users-alex-Documents-Projekti-telemetry-analysis` — 38
- `c--Users-alex-Documents-Projekti-Invoice-integration` — 19
- `C--Users-alex` — 4
- `c--Users-alex-Documents-Projekti-traffic-analysis` — 4
- `c--Users-alex-Documents-Projekti-LLM-sessions-grouping` — 2
- `subagents` — 1

### R1 — DAG shape

```json
{
  "sessions": 68,
  "sessions with >1 leaf (real branching)": 49,
  "sessions with >1 root (orphaned records)": 0,
  "total branch points": 391,
  "total leaves": 459,
  "total DAG nodes": 14397
}
```

### R1 — most-branched sessions

```json
[
  {
    "file": "7bc30e2d-af70-487a-8dbd-cf41ecca51c9.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 1036,
    "nodes": 799,
    "roots": 1,
    "leaves": 139,
    "branch_points": 138,
    "started": "2026-08-13T18:18:00.936Z",
    "ended": "2026-08-18T20:15:33.220Z"
  },
  {
    "file": "6dc4c437-9b3d-4ba4-bfab-5121834c8244.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 423,
    "nodes": 300,
    "roots": 1,
    "leaves": 25,
    "branch_points": 24,
    "started": "2026-08-24T20:48:42.231Z",
    "ended": "2026-08-24T21:54:00.375Z"
  },
  {
    "file": "ccd7bee1-e514-49c7-a75c-ca8adf015138.jsonl",
    "project": "c--Users-alex-Documents-Projekti-telemetry-analysis",
    "records": 996,
    "nodes": 716,
    "roots": 1,
    "leaves": 20,
    "branch_points": 19,
    "started": "2026-08-19T13:58:28.563Z",
    "ended": "2026-08-19T19:34:57.778Z"
  },
  {
    "file": "41490e51-0938-4f17-a250-c6c8b6b6f86f.jsonl",
    "project": "c--Users-alex-Documents-Projekti-LLM-sessions-grouping",
    "records": 731,
    "nodes": 506,
    "roots": 1,
    "leaves": 20,
    "branch_points": 19,
    "started": "2026-08-24T23:32:11.132Z",
    "ended": "2026-08-25T10:30:14.365Z"
  },
  {
    "file": "c6d911b3-19ec-46be-b2e5-b644cc1333ab.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 523,
    "nodes": 338,
    "roots": 1,
    "leaves": 14,
    "branch_points": 13,
    "started": "2026-08-19T20:27:03.583Z",
    "ended": "2026-08-23T20:51:01.613Z"
  },
  {
    "file": "7f352c8a-ff78-49bb-aff8-d00a1b2b3623.jsonl",
    "project": "c--Users-alex-Documents-Projekti-telemetry-analysis",
    "records": 503,
    "nodes": 361,
    "roots": 1,
    "leaves": 13,
    "branch_points": 12,
    "started": "2026-08-20T20:21:25.387Z",
    "ended": "2026-08-20T21:37:22.058Z"
  },
  {
    "file": "0995dd27-a367-45a8-8c54-2438cded98b1.jsonl",
    "project": "c--Users-alex-Documents-Projekti-telemetry-analysis",
    "records": 356,
    "nodes": 242,
    "roots": 1,
    "leaves": 10,
    "branch_points": 9,
    "started": "2026-08-20T22:04:52.224Z",
    "ended": "2026-08-20T23:09:45.279Z"
  },
  {
    "file": "15903fd7-b7fa-4dc2-8f19-f5f36fdf577f.jsonl",
    "project": "C--Users-alex",
    "records": 339,
    "nodes": 241,
    "roots": 1,
    "leaves": 9,
    "branch_points": 8,
    "started": "2026-08-24T12:54:20.674Z",
    "ended": "2026-08-24T23:20:02.336Z"
  },
  {
    "file": "3fc42211-893a-4a28-8df4-229f60ad9ff2.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 235,
    "nodes": 165,
    "roots": 1,
    "leaves": 9,
    "branch_points": 8,
    "started": "2026-08-25T10:13:22.216Z",
    "ended": "2026-08-25T10:30:13.023Z"
  },
  {
    "file": "23d9b3e5-aad8-45a3-87a4-f10706be51f7.jsonl",
    "project": "c--Users-alex-Documents-Projekti-telemetry-analysis",
    "records": 364,
    "nodes": 234,
    "roots": 1,
    "leaves": 9,
    "branch_points": 8,
    "started": "2026-08-21T08:23:47.162Z",
    "ended": "2026-08-21T12:02:31.982Z"
  }
]
```

### R2 — bridge-session examples

```json
[
  {
    "type": "bridge-session",
    "sessionId": "15903fd7-b7fa-4dc2-8f19-f5f36fdf577f",
    "bridgeSessionId": "cse_013rHEfkFP7qr3P7d61d5pWD",
    "lastSequenceNum": 0,
    "ownerAccountUuid": "8a876296-755d-4f73-9b30-d31b4524dde6",
    "ownerOrganizationUuid": "774a08cc-0631-40c0-bf7f-7ad2c54b967f"
  },
  {
    "type": "bridge-session",
    "sessionId": "15903fd7-b7fa-4dc2-8f19-f5f36fdf577f",
    "bridgeSessionId": "cse_013rHEfkFP7qr3P7d61d5pWD",
    "lastSequenceNum": 0,
    "ownerAccountUuid": "8a876296-755d-4f73-9b30-d31b4524dde6",
    "ownerOrganizationUuid": "774a08cc-0631-40c0-bf7f-7ad2c54b967f"
  },
  {
    "type": "bridge-session",
    "sessionId": "15903fd7-b7fa-4dc2-8f19-f5f36fdf577f",
    "bridgeSessionId": "cse_013rHEfkFP7qr3P7d61d5pWD",
    "lastSequenceNum": 0,
    "ownerAccountUuid": "8a876296-755d-4f73-9b30-d31b4524dde6",
    "ownerOrganizationUuid": "774a08cc-0631-40c0-bf7f-7ad2c54b967f"
  }
]
```

### session inventory

```json
[
  {
    "file": "15903fd7-b7fa-4dc2-8f19-f5f36fdf577f.jsonl",
    "project": "C--Users-alex",
    "records": 339,
    "nodes": 241,
    "roots": 1,
    "leaves": 9,
    "branch_points": 8,
    "started": "2026-08-24T12:54:20.674Z",
    "ended": "2026-08-24T23:20:02.336Z"
  },
  {
    "file": "5710f584-f4fa-4d17-941e-c17ba1c8684d.jsonl",
    "project": "C--Users-alex",
    "records": 288,
    "nodes": 219,
    "roots": 1,
    "leaves": 8,
    "branch_points": 7,
    "started": "2026-08-19T10:10:31.024Z",
    "ended": "2026-08-19T13:47:08.891Z"
  },
  {
    "file": "72ef7346-aba7-436c-b903-8433e1f0cd20.jsonl",
    "project": "C--Users-alex",
    "records": 16,
    "nodes": 8,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-12T20:23:55.404Z",
    "ended": "2026-08-12T20:27:31.454Z"
  },
  {
    "file": "cc75614e-ffc1-4589-8621-a4b5e531bd8e.jsonl",
    "project": "C--Users-alex",
    "records": 2320,
    "nodes": 1438,
    "roots": 1,
    "leaves": 3,
    "branch_points": 2,
    "started": "2026-08-12T20:28:23.369Z",
    "ended": "2026-08-17T21:10:36.659Z"
  },
  {
    "file": "563be980-43e1-4124-9e72-280f637fd35b.jsonl",
    "project": "c--Users-alex-Documents-Projekti-traffic-analysis",
    "records": 21,
    "nodes": 11,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-20T23:50:42.562Z",
    "ended": "2026-08-20T23:51:00.040Z"
  },
  {
    "file": "5fef813a-3716-4ad0-90a5-38476a70c6cf.jsonl",
    "project": "c--Users-alex-Documents-Projekti-traffic-analysis",
    "records": 129,
    "nodes": 94,
    "roots": 1,
    "leaves": 2,
    "branch_points": 1,
    "started": "2026-08-19T16:32:05.563Z",
    "ended": "2026-08-20T22:46:40.330Z"
  },
  {
    "file": "69343b74-53f2-49f9-89f0-17d37cffff10.jsonl",
    "project": "c--Users-alex-Documents-Projekti-traffic-analysis",
    "records": 151,
    "nodes": 104,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-20T20:00:57.478Z",
    "ended": "2026-08-20T20:22:38.636Z"
  },
  {
    "file": "f62d41cd-321e-4aeb-abb3-059d59c1c8d5.jsonl",
    "project": "c--Users-alex-Documents-Projekti-traffic-analysis",
    "records": 99,
    "nodes": 73,
    "roots": 1,
    "leaves": 2,
    "branch_points": 1,
    "started": "2026-08-20T22:44:34.002Z",
    "ended": "2026-08-20T23:14:24.290Z"
  },
  {
    "file": "0c52b5ef-6e0e-4970-b45c-67598c0ac358.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 148,
    "nodes": 109,
    "roots": 1,
    "leaves": 2,
    "branch_points": 1,
    "started": "2026-06-24T11:01:27.716Z",
    "ended": "2026-06-24T11:17:49.789Z"
  },
  {
    "file": "243c6123-1c0a-4873-96f2-c55efee66666.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 47,
    "nodes": 31,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-24T20:46:24.267Z",
    "ended": "2026-08-24T20:47:02.080Z"
  },
  {
    "file": "25513848-9cfa-41c6-bdbb-5873c7b47cc6.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 139,
    "nodes": 106,
    "roots": 1,
    "leaves": 7,
    "branch_points": 6,
    "started": "2026-07-13T21:31:17.047Z",
    "ended": "2026-07-13T21:47:05.480Z"
  },
  {
    "file": "3b5fe9b7-b75e-4bed-931f-f45437a39be0.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 11,
    "nodes": 6,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-17T09:09:33.547Z",
    "ended": "2026-08-17T09:10:13.310Z"
  },
  {
    "file": "3fc42211-893a-4a28-8df4-229f60ad9ff2.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 235,
    "nodes": 165,
    "roots": 1,
    "leaves": 9,
    "branch_points": 8,
    "started": "2026-08-25T10:13:22.216Z",
    "ended": "2026-08-25T10:30:13.023Z"
  },
  {
    "file": "6a7ad225-9eb5-442c-afe9-41e15538b235.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 18,
    "nodes": 8,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-17T09:06:09.955Z",
    "ended": "2026-08-17T09:07:34.146Z"
  },
  {
    "file": "6dc4c437-9b3d-4ba4-bfab-5121834c8244.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 423,
    "nodes": 300,
    "roots": 1,
    "leaves": 25,
    "branch_points": 24,
    "started": "2026-08-24T20:48:42.231Z",
    "ended": "2026-08-24T21:54:00.375Z"
  },
  {
    "file": "75cef013-8210-4a6f-b414-9ec464973e93.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 1100,
    "nodes": 743,
    "roots": 1,
    "leaves": 1,
    "branch_points": 0,
    "started": "2026-08-24T12:45:26.441Z",
    "ended": "2026-08-24T20:41:41.279Z"
  },
  {
    "file": "7bc30e2d-af70-487a-8dbd-cf41ecca51c9.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 1036,
    "nodes": 799,
    "roots": 1,
    "leaves": 139,
    "branch_points": 138,
    "started": "2026-08-13T18:18:00.936Z",
    "ended": "2026-08-18T20:15:33.220Z"
  },
  {
    "file": "883d30f6-d664-4643-8676-f486ef495e08.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 41,
    "nodes": 30,
    "roots": 1,
    "leaves": 4,
    "branch_points": 3,
    "started": "2026-08-17T10:01:32.535Z",
    "ended": "2026-08-17T10:03:45.786Z"
  },
  {
    "file": "8cf9958b-0ad3-4613-8f84-100ca3e00076.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integration",
    "records": 172,
    "nodes": 130,
    "roots": 1,
    "leaves": 4,
    "branch_points": 3,
    "started": "2026-07-28T14:05:14.394Z",
    "ended": "2026-07-28T14:27:53.721Z"
  },
  {
    "file": "a14b835a-e946-4258-a10e-557522aab0ce.jsonl",
    "project": "c--Users-alex-Documents-Projekti-Invoice-integra
```
