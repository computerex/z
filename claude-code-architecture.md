# Claude Code Architecture: CLAUDE.md, Memory, and Hooks

> Based on analysis of `C:\projects\claude-code` (the Anthropic Claude Code CLI application).

---

## Table of Contents

1. [CLAUDE.md Inheritance & Context Injection](#1-claudemd-inheritance--context-injection)
2. [The Memory System (memdir)](#2-the-memory-system-memdir)
3. [The Hooks System](#3-the-hooks-system)
4. [How the Three Systems Interact](#4-how-the-three-systems-interact)

---

## 1. CLAUDE.md Inheritance & Context Injection

### File: `utils/claudemd.ts`

### Priority / Load Order

Files are loaded in this exact order (reverse priority — latest wins):

| Order | Type | Path | Description |
|-------|------|------|-------------|
| 1 | **Managed** | `/etc/claude-code/CLAUDE.md` | Global instructions for all users (admin-managed) |
| 2 | **User** | `~/.claude/CLAUDE.md` | Private global instructions for all projects |
| 3 | **Project** | `CLAUDE.md` in each directory up to root | Instructions checked into the codebase |
| 4 | **Local** | `CLAUDE.local.md` in each directory up to root | Private project-specific instructions (gitignored) |
| — | **AutoMem** | `MEMORY.md` in memdir | Separate memory system (see §2) |
| — | **TeamMem** | Team MEMORY.md | Shared team memory (feature-gated) |

The model receives a combined block preceded by:

```
Codebase and user instructions are shown below. Be sure to adhere to these instructions.
IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.
```

### Discovery Algorithm (`getMemoryFiles`, line 790)

1. **Managed** → read from managed path, then load `.claude/rules/*.md`
2. **User** → if `userSettings` enabled: read `~/.claude/CLAUDE.md`, then `~/.claude/rules/*.md`
3. **Project + Local** → walk from CWD **up to root**, collecting:
   - `CLAUDE.md` (if `projectSettings` enabled)
   - `.claude/CLAUDE.md` (if `projectSettings` enabled)
   - `.claude/rules/*.md` (unconditional rules — no `paths:` frontmatter)
   - `CLAUDE.local.md` (if `localSettings` enabled)
4. **Additional dirs** → if `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD` env is truthy
5. **AutoMem** → `MEMORY.md` entrypoint (if auto-memory enabled)
6. **TeamMem** → Team MEMORY.md (if feature flag `TEAMMEM` enabled)

Results are **memoized** (`memoize` from lodash). Invalidated via:
- `clearMemoryFileCaches()` — no hook fire (for /memory dialog, worktree enter/exit)
- `resetGetMemoryFilesCache(reason)` — fires `InstructionsLoaded` hook with reason (`'session_start'`, `'compact'`, etc.)

### `@include` Directive

Memory files can include other files using `@path` syntax:

```markdown
@./docs/api-reference.md
@~/other/file.md
@/absolute/path.md
```

- Works in **leaf text nodes** only (not inside code blocks or code strings)
- Included files are added **before** the including file (so context reads top-down)
- Non-existent files are silently ignored
- Max depth: 5 (prevents cycles)
- Only text file extensions allowed (whitelist: `.md`, `.txt`, `.json`, `.py`, etc.)
- External includes (outside CWD) require user approval (`hasClaudeMdExternalIncludesApproved`)

### Conditional Rules (`paths:` frontmatter)

Files in `.claude/rules/` can have YAML frontmatter with a `paths:` field:

```yaml
---
paths: src/**/*.ts
---
```

- Files **with** `paths:` → **conditional rules**: only loaded when the model touches a matching file path
- Files **without** `paths:` → **unconditional rules**: loaded eagerly at session start
- Glob matching uses the `ignore` library (gitignore-style)
- Managed and User rules are resolved relative to the original CWD
- Project rules are resolved relative to the directory containing `.claude`

### Nested Memory (Lazy Loading)

When the model reads/writes a file **outside CWD** (e.g., `src/components/Button.tsx`), the system:

1. **Phase 1**: Load Managed + User conditional rules matching the target path
2. **Phase 2**: Walk from CWD **down to target directory**, loading each nested directory's:
   - `CLAUDE.md` + `.claude/CLAUDE.md` (if `projectSettings`)
   - `CLAUDE.local.md` (if `localSettings`)
   - `.claude/rules/*.md` (unconditional → conditional)
3. **Phase 3**: Walk from root **to CWD**, loading **only conditional rules** (unconditional already loaded eagerly)

This is implemented in `getNestedMemoryAttachmentsForFile()` in `utils/attachments.ts` (line 1792).

### Content Processing

Each memory file undergoes:
1. **Frontmatter strip** — YAML frontmatter parsed, removed from content
2. **HTML comment strip** — block-level `<!-- -->` removed (inline preserved)
3. **MEMORY.md truncation** — line cap (200) + byte cap (25KB) for AutoMem/TeamMem entrypoints
4. **@include resolution** — referenced files loaded as separate entries

### Cache and Deduplication

- `processedPaths` Set prevents re-inclusion
- `readFileState` LRU cache (100 entries) prevents re-injection on eviction cycles
- `loadedNestedMemoryPaths` used as a non-evicting companion Set
- File state entries from injection are marked `isPartialView: true` when content differs from disk

---

## 2. The Memory System (memdir)

### Files: `memdir/`

| File | Purpose |
|------|---------|
| `memdir.ts` | Prompt builder, entrypoint reader, directory creation |
| `memoryTypes.ts` | Type taxonomy, section templates for the system prompt |
| `paths.ts` | Auto-memory path resolution |
| `findRelevantMemories.ts` | Semantic recall: select relevant memories for a query |
| `memoryScan.ts` | Directory scanning: read frontmatter of all `.md` files |
| `memoryAge.ts` | Age-based filtering for recall |
| `teamMemPaths.ts` | Team memory directory paths (feature-gated) |
| `teamMemPrompts.ts` | Combined (auto + team) prompt builder |

### Architecture

The memory system is a **file-based, typed, index-driven** persistent storage:

```
~/.claude/projects/<project-slug>/memory/
├── MEMORY.md              # Index file (always loaded into context)
├── user_role.md           # User-type memory
├── feedback_testing.md    # Feedback-type memory
└── ...
```

### Memory Types (closed taxonomy)

| Type | Description |
|------|-------------|
| `user` | User's role, goals, responsibilities, knowledge |
| `feedback` | Guidance from user — what to do/avoid |
| `project` | Ongoing work, goals, initiatives, bugs |
| `reference` | Pointers to external systems |

### What NOT to Save

- Code patterns, architecture, file paths, project structure
- Git history, recent changes
- Debugging solutions or fix recipes
- Anything already documented in CLAUDE.md
- Ephemeral task details

### The Index File (MEMORY.md)

- **Always loaded** into the system prompt (or user context)
- Contains one-line entries pointing to topic files:
  ```markdown
  - [User is a data scientist](user_role.md) — focused on observability
  - [Integration tests](feedback_testing.md) — must hit real database
  ```
- Capped at 200 lines / 25KB (truncated with warning)
- Model writes new topics as separate `.md` files, then adds a pointer to MEMORY.md

### Memory Recall (`findRelevantMemories`)

1. **Scan** `memoryDir` for all `.md` files (excluding `MEMORY.md`), read frontmatter
2. **Filter** out files already surfaced in prior turns (`alreadySurfaced` set)
3. **Select** up to 5 relevant memories by asking Sonnet (via `sideQuery`):
   - Sends a manifest (filename + description + timestamp) to Sonnet
   - Sonnet returns JSON with `{selected_memories: string[]}`
   - The query is the user's current input + recently used tools
4. **Return** matching file paths + mtimes to be injected as attachments

### Auto-Memory vs. Agent Memory

- **Auto Memory** (`loadMemoryPrompt` in `memdir.ts`): Built into the **system prompt** for the main session. The model is instructed to save/read memories over time. `MEMORY.md` loaded automatically.
- **Agent Memory** (separate, not in `memdir.ts`): When the `Memory` tool is used (for sub-agents), memories are scoped to the agent's own directory. Same format but separate directory.

### Integration with Context

- `loadMemoryPrompt()` generates the system-prompt section explaining the memory system
- `getMemoryFiles()` in `claudemd.ts` loads `MEMORY.md` as an `AutoMem` entry into the memory files list
- In the system prompt, the model sees instructions about *how* to use memory (types, when to save, when to access)
- The actual MEMORY.md content is appended to that section
- When `tengu_moth_copse` feature is on, `MEMORY.md` is excluded from system prompt injection — the `findRelevantMemories` prefetch surfaces memories via attachments instead

---

## 3. The Hooks System

### File: `utils/hooks.ts` (5,022 lines)

### What Are Hooks?

Hooks are **user-defined shell commands, HTTP calls, or LLM prompts** that execute at various points in Claude Code's lifecycle. They are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          { "type": "command", "command": "echo 'Read: $ARGUMENTS'", "shell": "bash" }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          { "type": "http", "url": "http://localhost:8080/hook", "headers": { "Authorization": "Bearer $MY_TOKEN" } }
        ]
      }
    ]
  }
}
```

### Hook Types

| Type | Description |
|------|-------------|
| `command` | Execute a shell command. Gets $ARGUMENTS as JSON on stdin. |
| `prompt` | Evaluate with an LLM. Gets $ARGUMENTS placeholder for input. |
| `agent` | Run an agentic verifier with a prompt. Gets $ARGUMENTS placeholder. |
| `http` | POST to a URL with hook input as JSON body. |
| `callback` | Internal JavaScript callback (not user-configurable). |
| `function` | Internal JS function (not user-configurable, only for Stop hooks). |

### Hook Events (all 23)

| Event | Timing | Matcher Field |
|-------|--------|---------------|
| `PreToolUse` | Before tool execution | `tool_name` |
| `PostToolUse` | After tool execution | `tool_name` |
| `PostToolUseFailure` | After tool execution fails | `tool_name` |
| `PermissionRequest` | When permission required | `tool_name` |
| `PermissionDenied` | After auto-classifier denies | `tool_name` |
| `Notification` | Notifications sent | `notification_type` |
| `UserPromptSubmit` | User submits a prompt | (none) |
| `SessionStart` | New session starts | `source` |
| `SessionEnd` | Session ends/clears | `reason` |
| `Stop` | Concludes response | (none) |
| `StopFailure` | Turn ends due to API error | `error` |
| `SubagentStart` | Agent tool starts | `agent_type` |
| `SubagentStop` | Agent tool completes | `agent_type` |
| `TeammateIdle` | Teammate is idle | (none) |
| `TaskCreated` | Task created | (none) |
| `TaskCompleted` | Task completed | (none) |
| `Setup` | Setup requested | `trigger` |
| `PreCompact` | Before context compaction | `trigger` |
| `PostCompact` | After context compaction | `trigger` |
| `ConfigChange` | Config was changed | `source` |
| `CwdChanged` | Working directory changed | (none) |
| `FileChanged` | File changed | `basename(file_path)` |
| `InstructionsLoaded` | Instruction file loaded | `load_reason` |
| `Elicitation` | MCP elicitation prompt | `mcp_server_name` |
| `ElicitationResult` | MCP elicitation result | `mcp_server_name` |

### Hook Configuration Sources (priority order)

Sources are merged with later sources overriding earlier ones:

1. **`policySettings`** — managed/admin settings (`.claude/settings.json` managed by IT)
2. **`userSettings`** — `~/.claude/settings.json`
3. **`projectSettings`** — `.claude/settings.json` (in project root)
4. **`localSettings`** — `.claude/settings.local.json`
5. **`pluginHook`** — from plugins (loaded via `loadPluginHooks.ts`)
6. **`sessionHook`** — in-memory, registered per-session (e.g., skill/agent frontmatter hooks)
7. **`builtinHook`** — internal registered callbacks

### Hook Filtering by `if` condition

Hooks can specify a condition using permission rule syntax:

```json
{
  "matcher": "Write",
  "hooks": [
    { "type": "command", "command": "echo hook", "if": "Write(*.ts)" }
  ]
}
```

The `if` condition is evaluated against the tool call. If it doesn't match, the hook is skipped without spawning a process.

### Hook Execution Flow (`executeHooksOutsideREPL`, line 3003)

1. **Gate check**: skip if `CLAUDE_CODE_SIMPLE`, `disableAllHooks`, or no workspace trust
2. **Match hooks**: call `getMatchingHooks()` which filters matchers by `matchQuery` (e.g., tool name) and `if` conditions
3. **Deduplicate**: remove duplicate hook commands/prompts/URLs within same source
4. **Execute in parallel**: all matching hooks run concurrently with individual timeouts (default 10 min)
5. **Collect results**: each hook returns `{succeeded, output, blocked}`

### InstructionsLoaded Hook (line 4335)

**Purpose**: Audit/observability only — fires when instruction files are loaded. Does NOT support blocking.

**Dispatch sites**:
- **Eager load** at session start (from `getMemoryFiles()` in `claudemd.ts`)
- **Eager reload** after compaction (cache clearance triggers reload with reason `'compact'`)
- **Lazy load** when Claude touches a file triggering nested CLAUDE.md or conditional rules with `paths:` frontmatter (from `memoryFilesToAttachments()` in `attachments.ts`)

**Input**:
```typescript
{
  hook_event_name: 'InstructionsLoaded',
  file_path: string,         // Path to the loaded instruction file
  memory_type: 'User' | 'Project' | 'Local' | 'Managed',
  load_reason: 'session_start' | 'compact' | 'include' | 'nested_traversal' | 'path_glob_match',
  globs?: string[],          // Conditional rule glob patterns (if applicable)
  trigger_file_path?: string, // File path that triggered nested loading
  parent_file_path?: string   // Parent file path (if loaded via @include)
}
```

**One-shot flag**: The hook fires only on the first eager load after cache reset. Uses `nextEagerLoadReason` and `shouldFireHook` to ensure it fires with the correct reason.

### Hooks in Skill/Agent Frontmatter

Skills and agents (`.md` files with frontmatter) can define hooks directly:

```markdown
---
name: verify
hooks:
  PostToolUse:
    - matcher: Write
      hooks:
        - type: command
          command: npm test
          timeout: 30
---

## verify

Run tests after editing files.
```

These are parsed by `parseHooksFromFrontmatter()` in `loadSkillsDir.ts` and registered as session hooks when the skill is invoked.

### Security

- **All hooks require workspace trust** (defense-in-depth against RCE)
- `allowManagedHooksOnly` policy restricts to admin-managed hooks only
- `disableAllHooks` managed setting kills all hooks
- `disableAllHooks` user setting only kills non-managed hooks
- `strictPluginOnlyCustomization` blocks user/project/local hooks but allows plugin hooks
- HTTP hook header values can reference env vars, but only via `allowedEnvVars` explicit allowlist

---

## 4. How the Three Systems Interact

### CLAUDE.md + Hooks

The `InstructionsLoaded` hook fires when CLAUDE.md files are loaded:
- At **session start**: eager load from `getMemoryFiles()` with reason `'session_start'`
- On **compaction**: cache reset triggers reload with reason `'compact'`
- On **nested traversal**: when model touches a file, lazy-loaded rules fire with reason `'nested_traversal'` or `'path_glob_match'`

### Memory + Context

- Auto-memory `MEMORY.md` is loaded as an `AutoMem` type file in `getMemoryFiles()` (claudemd.ts)
- The system prompt includes memory instructions via `loadMemoryPrompt()` (memdir.ts)
- When `tengu_moth_copse` is active, `MEMORY.md` is excluded from the system prompt; instead, `findRelevantMemories` prefetches relevant topic files on each turn
- The `findRelevantMemories` system uses a **separate Sonnet query** to classify which memories are relevant to the current user input
- Selected memories are injected as `nested_memory` attachments (same mechanism as nested CLAUDE.md files)

### Skills + Hooks

- Skills can declare hooks in their frontmatter
- When a skill is invoked, its hooks are registered as session hooks
- Session hooks are scoped to the agent/session that registered them (hooks don't leak between agents)

### Hooks Plugin System

- Plugins can register hooks via `hooks.json` in their plugin directory
- Plugin hooks use the `PluginHookMatcher` type which includes `pluginRoot` and `pluginId`
- Plugin hooks bypass `strictPluginOnlyCustomization` but respect `allowManagedHooksOnly`

### Key Architectural Patterns

1. **Memoization with explicit invalidation**: Both memory files and hooks config are memoized and invalidated through specific APIs
2. **Snapshot-based hooks config**: Hooks config is snapshotted at startup to avoid mid-session config changes causing inconsistency
3. **Fire-and-forget for audit hooks**: `InstructionsLoaded` is intentionally fire-and-forget (no blocking) to avoid impacting session startup time
4. **One-shot flags**: Critical events (like first load after compaction) use one-shot flags to ensure correct behavior even with caching
5. **Two-tier memory**: Index file (MEMORY.md) + topic files — the index is always in context, topic files are fetched on-demand
