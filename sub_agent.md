# Sub-Agent System — Design & Implementation Plan

## 1. Overview

Allow the main (parent) agent to spawn independent sub-agents via a new `create_agent` tool.
Each sub-agent is a fully independent `ClineAgent` instance with its own conversation history,
context container, todo list, streaming client, and session file — completely isolated from
the parent and from sibling sub-agents.

## 2. Key Decisions (from user feedback)

| Question | Decision |
|----------|----------|
| Blocking or non-blocking `create_agent`? | **Non-blocking** — returns immediately, sub-agent runs in background for parallel work |
| Model override per sub-agent? | **No** — same model as parent |
| Nested sub-agents? | **No** — flat only |
| Persist sub-agent sessions? | **Save for debugging** (sub-agent session files saved) but **no restore on restart** |
| Real-time streaming when switched? | **Yes** — when user switches to sub-agent via `/agent <name>`, output streams live |
| Max concurrent sub-agents? | **No limit** |
| Ctrl+C behavior? | **Interrupt only the focused agent** — parent or sub |
| Session isolation | Sub-agents use **separate session paths** (`_sub_<name>.json`) — parent sessions never touched |
| Completion notification | When sub-agent completes, harness injects a message into **parent's conversation** so parent knows to check |
| Getting output | Parent uses `send_agent_input` tool to communicate and retrieve results |

## 3. Architecture

### 3.1 Key Insight: Capturing Sub-Agent Output

The sub-agent's `_run_loop` writes output via:
- `sys.stdout.write()` — streaming content (on_chunk, on_reasoning)
- `self.console.print()` — tool execution headers, results, panels

To capture both for buffering + real-time display:
- Replace `sys.stdout` with a **Tee** stream that writes to both a buffer and (optionally) real stdout
- Give sub-agent a custom `Console` that writes through the same Tee stream
- When focused → Tee passes through to terminal + buffer
- When not focused → Tee only buffers, no terminal output

```python
class TeeWriter:
    """Writes to both a buffer and optionally the real stdout."""
    def __init__(self, real_stdout):
        self.buffer = io.StringIO()
        self.real_stdout = real_stdout
        self.active = False  # If True, also writes to real stdout
    
    def write(self, text):
        self.buffer.write(text)
        if self.active:
            self.real_stdout.write(text)
            self.real_stdout.flush()
    
    def flush(self):
        if self.active:
            self.real_stdout.flush()
    
    def getvalue(self):
        return self.buffer.getvalue()
```

### 3.2 `SubAgentManager` (new class, in `sub_agent_manager.py`)

Central registry owned by `main()` in `harness.py`. Manages all sub-agents:

```python
class SubAgentManager:
    def __init__(self, config: Config, console: Console, workspace: str, session_path: Path):
        self._agents: Dict[str, SubAgentInstance] = {}
        self._config = config
        self._console = console
        self._workspace = workspace
        self._parent_session_path = session_path  # For completion notification

    def create(self, name: str, task_prompt: str) -> str:
        """Create a new sub-agent. Returns the sub-agent name."""

    async def run(self, name: str, input_text: str = "") -> str:
        """Start or continue a sub-agent. If not running, starts background task.
        Waits for output and returns it."""

    async def _run_sub_agent(self, instance: SubAgentInstance, initial_input: str):
        """Background coroutine that runs the sub-agent loop.
        Manages TeeWriter activation based on focus."""

    def pause(self, name: str) -> bool:
        """Pause a sub-agent (cancel task, save session)."""

    def delete(self, name: str) -> bool:
        """Delete a sub-agent entirely."""

    def list(self) -> List[dict]:
        """Return status of all sub-agents."""

    def get(self, name: str) -> Optional[SubAgentInstance]:
        """Get instance by name."""

    def set_focused(self, name: Optional[str]):
        """Set which sub-agent is focused (for TeeWriter activation)."""

    def check_completed(self) -> Optional[str]:
        """Return name of any newly completed sub-agent (for notification injection)."""

    def save_all_sessions(self):
        """Save all sub-agent sessions."""
```

### 3.3 `SubAgentInstance` (new data class)

```python
@dataclass
class SubAgentInstance:
    name: str
    agent: ClineAgent
    task: Optional[asyncio.Task]  # Background asyncio task
    status: str  # "running", "completed", "error"
    output: str  # Accumulated output text
    tee: TeeWriter  # Captures all sub-agent output
    session_path: Path  # ~/.z/sessions/<hash>/_sub_<name>.json
    created_at: float
    completed_at: Optional[float] = None
    completion_notified: bool = False  # Whether parent has been notified
```

### 3.4 Session Isolation

- Parent sessions: `~/.z/sessions/<workspace_hash>/<session_name>.json`
- Sub-agent sessions: `~/.z/sessions/<workspace_hash>/_sub_<agent_name>.json`
- The underscore prefix prevents name collision and makes them easy to distinguish
- Sub-agent session files are ONLY written for debugging — never restored on restart
- Parent's `save_session()` / `load_session()` are never called by sub-agent code

### 3.5 Lifecycle Flow

#### Creating a Sub-Agent

1. Parent model calls `create_agent(name="foo", task="analyze auth")` tool
2. `SubAgentManager.create()`:
   - Validates name is unique
   - Creates a `Config` copy (same model/api_key/api_url as parent)
   - Creates a `ClineAgent` with fresh everything (messages=[], context=[], todos=[])
   - Creates a `TeeWriter` for the sub-agent (initially NOT focused → buffered only)
   - Creates a sub-agent Console using the TeeWriter
   - Stores `SubAgentInstance`
3. Tool returns `"Created sub-agent 'foo'. It is running in background."`
4. `SubAgentManager._run_sub_agent()` starts in background:
   - Replaces `sys.stdout` with TeeWriter
   - Calls `agent.run_message(task)` 
   - When complete: sets status to "completed", records completion time
   - Restores `sys.stdout`

#### Parent Checks on Sub-Agent

5. After `create_agent` tool result, harness checks `check_completed()` 
6. If sub-agent completed, harness injects a user message to parent agent:
   `"[SYSTEM: Sub-agent 'foo' has completed its task. Use send_agent_input(name='foo') to retrieve its output.]"`
7. Parent agent continues its loop, sees the notification, can call `send_agent_input` to get the output

#### Multi-turn with a Sub-Agent

8. Parent calls `send_agent_input(name="foo", input="check SQL injection too")` 
9. `SubAgentManager.run("foo", "check SQL injection too")`:
   - Appends the input to sub-agent's messages
   - Runs `agent.run_message(input_text)` (awaits completion)
   - Returns the sub-agent's response
10. Parent gets the result as a tool response

#### User Switching to a Sub-Agent

11. User types `/agent foo` in the CLI
12. Harness sets `focused_agent = "foo"` and calls `sub_agent_manager.set_focused("foo")`
13. `set_focused` activates the TeeWriter (writes to buffer + real stdout)
14. Prompt bar changes to: `workspace model [agent:foo] ❯`
15. User input goes to sub-agent via `SubAgentManager.run("foo", input)`
16. Sub-agent's streaming output appears in real-time on terminal
17. User types `/agent-back` to return to parent focus

## 4. Completion Notification Mechanism

The critical flow for making the main agent aware of sub-agent completion:

```python
# After each create_agent tool result in the main loop:
if agent_completed := sub_agent_manager.check_completed():
    # Inject a user message into parent agent's conversation
    msg = f"[SYSTEM: Sub-agent '{agent_completed}' has completed its task. Use send_agent_input(name='{agent_completed}') to retrieve its output.]"
    agent.messages.append(StreamingMessage(role="user", content=msg))
```

This makes the notification appear as a user message in the parent's conversation history,
which the model will see on the next API call and can act upon.

## 5. Tool Definitions

### 5.1 `create_agent`

```python
{
    "name": "create_agent",
    "description": "Create an independent sub-agent to work on a task concurrently. The sub-agent runs in background. You'll be notified when it completes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique name for the sub-agent"},
            "task": {"type": "string", "description": "The full task description to assign to the sub-agent"},
        },
        "required": ["name", "task"],
    },
}
```

Tool returns: `"Created sub-agent '<name>'. Running in background."`

### 5.2 `send_agent_input`

```python
{
    "name": "send_agent_input",
    "description": "Send input/text to a sub-agent and get its response. If the sub-agent is still running, it will receive this as a new instruction. If it has completed, this will start a new conversation turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the sub-agent"},
            "input": {"type": "string", "description": "Input to send to the sub-agent"},
        },
        "required": ["name", "input"],
    },
}
```

Tool returns: The sub-agent's full response text.

### 5.3 `list_agents`

```python
{
    "name": "list_agents",
    "description": "List all sub-agents with their current status (running/completed/error).",
    "input_schema": {"type": "object", "properties": {}},
}
```

### 5.4 `pause_agent` / `delete_agent`

```python
{
    "name": "pause_agent",
    "description": "Pause a running sub-agent. Its state is saved.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {...}},
        "required": ["name"],
    },
}
```

## 6. Changes Required

### 6.1 New Files

| File | Description |
|------|-------------|
| `src/harness/sub_agent_manager.py` | `SubAgentManager`, `SubAgentInstance`, `TeeWriter` classes |

### 6.2 Modified Files

| File | Changes |
|------|---------|
| `src/harness/tool_registry.py` | Register `create_agent`, `send_agent_input`, `list_agents`, `pause_agent`, `delete_agent` tool definitions |
| `src/harness/tool_handlers.py` | Add `sub_agent_manager` property; add handler methods for all 5 tools |
| `src/harness/cline_agent.py` | Add `_dispatch_tool` entries for the 5 new tools |
| `harness.py` | Add `SubAgentManager` instance; `/agents`, `/agent`, `/agent-back` slash commands; prompt bar modification; focused agent dispatch; completion notification injection |

### 6.3 Implementation Order

1. **`sub_agent_manager.py`** — New file with `TeeWriter`, `SubAgentInstance`, `SubAgentManager`
2. **`tool_registry.py`** — Add tool definitions 
3. **`tool_handlers.py`** — Add handler methods + `sub_agent_manager` plumbing
4. **`cline_agent.py`** — Add `_dispatch_tool` entries
5. **`harness.py`** — CLI UX: commands, prompt bar, focus, notifications

## 7. UX Flow — Complete Walkthrough

### Creating and Interacting

```
model ❯ Investigate the error handling in the API layer

[Parent agent starts working...]
[Parent agent decides to spawn a sub-agent]
create_agent(name="error-audit", task="Audit all error handling in src/harness/...")

  ✓ Created sub-agent 'error-audit'. Running in background.

[Parent agent continues working on other things...]
[Sub-agent error-audit completes...]
[Harness injects notification into parent conversation]

  [SYSTEM: Sub-agent 'error-audit' has completed. 
   Use send_agent_input(name='error-audit') to retrieve its output.]

[Parent agent sees this in next API call]
send_agent_input(name="error-audit", input="What did you find?")

  [error-audit response streams...]
  Found 3 issues:
  1. ...
```

### Manual User Interaction

```
model ❯ /agents
  Sub-agents:
    - error-audit: completed

model ❯ /agent error-audit
Switched to sub-agent 'error-audit'

model [agent:error-audit] ❯ Show me the full report
[Sub-agent shows its findings...]

model [agent:error-audit] ❯ /agent-back
Switched back to parent agent

model ❯ 
```

### Keybinding

| Key | Action |
|-----|--------|
| Ctrl+E | Toggle focus: cycle through available sub-agents (like Alt+Tab for agents) |
