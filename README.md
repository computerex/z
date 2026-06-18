# Z — The Agentic Coding Harness

**Stop context-switching. Start creating.**

---

## The Ordinary Way

Every day, millions of developers sit down at their terminals knowing exactly what they want to build. They reach for an AI coding assistant — and immediately hit a wall.

_What model do I use? What provider? What API key? Does it support tool calling? Does it have vision? Can it search my codebase? Will it stay within context limits? Will it lose track of what I asked it to do five minutes ago?_

There are a hundred tools, a thousand models, and a million configuration options. But none of them do one simple thing: **get out of your way and let you build.**

Existing coding agents are fragile. They lose context mid-session. They can't handle interruptions. They forget what they were doing after a compaction. They lock you into one provider, one model, one way of working. They treat thinking/reasoning models as an afterthought. They're designed for demos, not for the relentless complexity of real-world engineering.

You've felt it. That moment when the assistant starts a long chain of actions, context fills up, and you know — you *know* — it's about to lose the plot. You sit there watching tokens stream, hoping it remembers the architecture decision you discussed three turns ago.

There has to be a better way.

---

## The Z Way

Z was built from the ground up to be **the agentic coding harness that doesn't break.**

It's not another chat UI taped onto an API. It's a production-grade engineering partner that:

- **Survives context compaction** — When token limits hit, Z preserves your todos, your goals, and your tool results. It leaves breadcrumbs. It knows what it was doing and picks right back up after truncation.
- **Speaks every provider fluently** — Switch effortlessly between Z.AI, MiniMax, Anthropic, OpenAI, DeepSeek, Groq, Together, Ollama (local or cloud), Bedrock, GitHub Copilot OAuth, or any OpenAI-compatible API. **26 providers. One interface. Zero friction.**
- **Handles thinking models correctly** — DeepSeek-R1, QwQ, gpt-oss with reasoning/reasoning_content, these are first-class citizens, not afterthoughts. Z extracts thinking, reasoning_content, reasoning_details — whatever the model returns — and shows it to you.
- **Never drops a tool call** — Every tool has error recovery. Every result is stored for later retrieval. Every background process is tracked. If Z needs to introspect, it has a dedicated deep-thinking tool that makes a separate API call with zero tool interference.
- **Checkpoints everything** — Git-backed snapshots before every agent turn. `/undo` and `/redo` for both files AND conversation. Ctrl+Z on your AI. Yes.
- **Pays attention to cost** — Real-time token and dollar tracking per model, per session. You're never surprised by the bill.

---

## What Makes Z Different

### One Terminal, Every Model

```
model: glm-4.7  ·  reasoning: high  ❯
```

Select from **26 providers** in the interactive setup — or switch on the fly with `/providers use`. Z auto-detects which provider you're talking to and normalizes the model name so LiteLLM routes correctly every time. Local Ollama, cloud Ollama, Z.AI's GLM-4.7 with coding plan, Anthropic, DeepSeek — all accessible from the same prompt.

### Context That Survives

Context compaction doesn't mean context loss. Z's `SmartContextManager` uses multiple strategies (half, quarter, lastTwo) to trim conversation while preserving your system prompt, your original request, your todo list, and any results you explicitly mark as important. After compaction, it injects a **recovery notice** so Z knows exactly what was trimmed and can re-read files or re-run commands if needed.

### Tool Results That Don't Vanish

Compact away a `read_file` result? Z stores it. Later, the `retrieve_tool_result` tool pulls it back. No re-reads. No lost data. Tool results survive across compactions, across sessions, across model switches.

### Todo Lists That Ground Every Action

```
○ [1] Implement payment webhook handler
◐ [2] Add idempotency key to Stripe events
● [3] Write integration tests
```

The `manage_todos` tool creates a persistent, structured plan that survives context compaction, model changes, and even crashes. Z checks it after every truncation to reorient itself. You see it at the top of every session. It's always there. Always accurate.

### Deep Thinking, No Distractions

The `introspect` tool is not just another tool. It spawns a **separate API call** with no tools available — just the model and your thoughts. Pure reasoning without the temptation to reach for a tool. For complex architecture decisions, debugging mysteries, or planning multi-step refactors. The result comes back as part of the conversation.

### Undo/Redo for AI Turns

Every agent turn snapshots your workspace via git. `/undo` reverts both files AND conversation to before the turn. `/redo` replays it. Entirely optional, entirely transparent, entirely free (binary files are excluded automatically).

### Plugins That Extend Everything

```python
def register(api):
    api.add_tool(
        name="deploy",
        description="Deploy the current service to staging",
        params={"env": {"required": True, "description": "staging or production"}},
        handler=lambda params: run_deploy(params["env"]),
    )
    api.on("pre_turn", my_hook)
```

Plugins can add tools, hook into lifecycle events (pre_turn, post_tool, on_compact), and extend the system prompt. Drop a `.py` file in `~/.z/plugins/` — Z discovers it automatically.

### Vision That Works When the Model Can See

Z auto-detects whether your model supports vision (via LiteLLM's registry or models.dev metadata). If it does, you can paste images directly from clipboard or type image paths. If not, the `analyze_image` tool routes to Z.AI's multimodal model. Z never sends images to a model that can't handle them.

### Web Search, MCP, and Beyond

Built-in `web_search` uses a configurable search API. `mcp_search_tools`, `mcp_list_tools`, `mcp_call_tools` give you access to the Model Context Protocol ecosystem — databases, APIs, documentation servers, anything with an MCP server.

---

## Quick Start

```bash
# Install
cd harness
pip install -e ".[dev]"

# Configure
python -m harness.cli --install

# Or configure headlessly
python -m harness.cli --api-url http://localhost:11434/v1 --api-key ollama --model gpt-oss:20b

# Run
python -m harness.cli
```

### Workspace & Sessions

```bash
# Work in a specific project
python -m harness.cli -w /path/to/project

# Name your session
python -m harness.cli -w . -s my-feature-branch

# Run one command and exit
python -m harness.cli -m "Refactor the auth module to use async/await"

# List and resume past sessions
python -m harness.cli --list
```

### The Interactive Prompt

| Command | What it does |
|---------|--------------|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/compact` | Force context compaction |
| `/undo` | Undo last agent turn (files + conversation) |
| `/redo` | Redo last undone turn |
| `/save [name]` | Save session snapshot |
| `/sessions` | List all sessions |
| `/session <name>` | Switch to a session |
| `/providers` | List/manage LLM providers |
| `/providers setup` | Add or edit a provider profile |
| `/providers use <name>` | Switch provider on the fly |
| `/model list` | List available models |
| `/model search <query>` | Search models across providers |
| `/cost` | Show token usage and cost summary |
| `/cost reset` | Reset cost tracking |
| `/plugins` | List loaded plugins and their tools |
| `/todo` | Show todo panel |
| `!command` | Execute a shell command directly |
| Ctrl+T | Cycle reasoning effort |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+V | Paste image from clipboard |

### Configuration

Z stores its configuration in `~/.z.json`:

```json
{
  "api_url": "https://api.z.ai/api/paas/v4/",
  "api_key": "your-api-key",
  "model": "glm-4.7",
  "max_tokens": 128000,
  "temperature": 0.7,
  "reasoning_effort": "high",
  "compaction_threshold": 0.85
}
```

Provider profiles are stored alongside — switch between them without editing files.

---

## Provider Support

| # | Provider | Model |
|---|----------|-------|
| 1 | Z.AI Coding | glm-4.7 |
| 2 | Z.AI Standard | glm-4.7 |
| 3 | MiniMax | MiniMax-M2.1 |
| 4 | Amazon Bedrock | qwen.qwen3-32b-v1:0 |
| 5 | Together AI | Llama-3.3-70B-Instruct-Turbo |
| 6 | Anthropic | claude-3-5-sonnet-latest |
| 7 | OpenRouter | anthropic/claude-3.5-sonnet |
| 8 | OpenAI | gpt-4o |
| 9 | Groq | llama-3.3-70b-versatile |
| 10 | DeepSeek | deepseek-chat |
| 11 | Mistral AI | mistral-large-latest |
| 12 | Cohere | command-r-plus |
| 13 | Fireworks AI | llama-v3p1-70b-instruct |
| 14 | Perplexity | llama-3.1-sonar-large-128k-online |
| 15 | AI21 | jamba-1.5-large |
| 16 | xAI (Grok) | grok-2-latest |
| 17 | Google Gemini | gemini-1.5-pro-latest |
| 18 | Cerebras | llama3.1-70b |
| 19 | Databricks | (workspace URL) |
| 20 | Replicate | meta-llama-3-70b-instruct |
| 21 | Anyscale | Llama-3.1-70B-Instruct |
| 22 | Ollama Cloud | llama3.1 |
| 23 | OpenAI Subscription (OAuth) | gpt-4o |
| 24 | GitHub Copilot (OAuth) | gpt-4o |
| 25 | Custom OpenAI-compatible API | (any) |
| **26** | **Local Ollama** | **(auto-detected)** |

### Local Ollama

Z detects Ollama running on `localhost:11434` and auto-discovers your installed models. Thinking/reasoning models (like `gpt-oss:20b`) work correctly — Z routes through Ollama's OpenAI-compatible `/v1/chat/completions` endpoint, which properly separates `content` from `reasoning`.

```bash
# Before using Local Ollama, make sure it's running:
ollama run gpt-oss:20b

# Then in Z:
# Select option 26 in the provider setup, or use:
/providers setup ollama
```

---

## How It Works

```
                    ┌─────────────────────────────┐
                    │      Your Terminal           │
                    │   (prompt_toolkit REPL)      │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │         ClineAgent           │
                    │  ┌──────────────────────┐   │
                    │  │   SmartContextManager │   │
                    │  │  • Compaction        │   │
                    │  │  • Tool result store │   │
                    │  │  • Recovery notices  │   │
                    │  └──────────────────────┘   │
                    │  ┌──────────────────────┐   │
                    │  │   TodoManager        │   │
                    │  │  • Persistent goals  │   │
                    │  │  • Progress tracking │   │
                    │  └──────────────────────┘   │
                    │  ┌──────────────────────┐   │
                    │  │   CheckpointManager  │   │
                    │  │  • Git snapshots     │   │
                    │  │  • Undo/Redo         │   │
                    │  └──────────────────────┘   │
                    │  ┌──────────────────────┐   │
                    │  │   PluginManager      │   │
                    │  │  • Custom tools      │   │
                    │  │  • Lifecycle hooks   │   │
                    │  └──────────────────────┘   │
                    │  ┌──────────────────────┐   │
                    │  │   CostTracker        │   │
                    │  │  • Per-model pricing │   │
                    │  │  • Session totals    │   │
                    │  └──────────────────────┘   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │      StreamingClient        │
                    │   (LiteLLM unified API)     │
                    │                              │
                    │  ┌────────────────────────┐ │
                    │  │  Provider routing      │ │
                    │  │  • Model normalization │ │
                    │  │  • OAuth tokens        │ │
                    │  │  • Reasoning detection │ │
                    │  │  • Vision detection    │ │
                    │  └────────────────────────┘ │
                    └──────────┬──────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    ┌────▼────┐          ┌─────▼─────┐         ┌────▼────┐
    │  Z.AI   │          │  OpenAI   │         │ Ollama  │
    │  GLM-4.7│          │  GPT-4o   │         │  Local  │
    └─────────┘          └───────────┘         └─────────┘
    ... 23 more providers ...
```

---

## The Tool System

Z exposes **19 built-in tools** through OpenAI function calling — the model sees them as native API capabilities:

| Category | Tools |
|----------|-------|
| **File operations** | `read_file`, `write_to_file`, `replace_in_file`, `replace_between_anchors` |
| **Shell/Process** | `execute_command`, `list_background_processes`, `check_background_process`, `stop_background_process` |
| **Search** | `list_files`, `search_files` |
| **External** | `analyze_image`, `web_search`, `mcp_search_tools`, `mcp_list_tools`, `mcp_call_tool` |
| **Context** | `retrieve_tool_result` |
| **Agent meta** | `manage_todos`, `introspect`, `attempt_completion` |

All tools share a single, unified schema in `tool_registry.py` — add one tool there and it propagates to parsing, detection, dispatch, and system prompts automatically.

---

## Running Tests

```bash
# All unit tests
pytest

# With coverage
pytest --cov=harness

# Integration tests (requires API key)
pytest -m integration

# Specific test file
pytest tests/test_file_tools.py
```

---

## Architecture File Map

```
harness/
├── harness.py                 # Entry point: CLI, REPL, install wizard, session management
├── pyproject.toml             # Project metadata and dependencies
├── README.md                  # You are here
└── src/harness/
    ├── __init__.py            # Package exports
    ├── __main__.py            # python -m harness.cli
    ├── cli.py                 # CLI entry point (loads harness.py)
    ├── cline_agent.py         # Agent core: conversation loop, tool dispatch
    ├── config.py              # Config loading (~/.z.json)
    ├── streaming_client.py    # LiteLLM unified streaming client
    ├── streaming_client_litellm.py  # Alternative LiteLLM client
    ├── tool_registry.py       # Single source of truth for all tool definitions
    ├── tool_handlers.py       # Tool execution implementations
    ├── prompts.py             # System prompt generator
    ├── smart_context.py       # Context compaction, tool result storage
    ├── context_management.py  # Token counting, conversation truncation
    ├── checkpoint.py          # Git-backed undo/redo snapshots
    ├── todo_manager.py        # Structured, persistent goal tracking
    ├── cost_tracker.py        # Per-model API cost tracking
    ├── plugin_manager.py      # Lightweight plugin system
    ├── model_capabilities.py  # Vision detection via LiteLLM + remote metadata
    ├── workspace_index.py     # Codebase indexing for search
    ├── logger.py              # Structured logging
    ├── prompts.py             # System prompt generation
    ├── image_utils.py         # Image processing utilities
    ├── instruction_loader.py  # Custom instruction loading
    ├── interrupt.py           # Graceful interrupt handling
    ├── status_line.py         # Terminal status line display
    ├── usage_report.py        # Session usage reporting
    ├── codex_models.py        # ChatGPT Codex model definitions
    ├── codex_oauth_client.py  # OAuth client for ChatGPT Plus/Pro
    ├── copilot_oauth_client.py# OAuth client for GitHub Copilot
    ├── oauth.py               # OAuth authentication manager
    ├── bedrock_provider.py    # Amazon Bedrock provider support
    ├── context_replay.py      # Context quality replay for evaluation
    ├── context_quality_benchmark.py  # Context quality benchmarking
    ├── context_benchmark_pack.py     # Benchmark data pack
    └── ...
```

---

## The Philosophy

Z is built on three convictions:

1. **The model is a tool, not the product.** The product is what you ship. Z abstracts away provider complexity, context management, and cost tracking so you can focus on building.

2. **Context should never be lost.** Every compaction leaves breadcrumbs. Every tool result is stored for retrieval. Every todo survives truncation. Z treats context as a precious resource and manages it ruthlessly.

3. **You should never be locked in.** 26 providers, one interface. Switch models mid-session. Run the same task across multiple providers and compare results. Use local models for prototyping and cloud models for production. Z adapts to your workflow, not the other way around.

---

## License

MIT
