# z — Agentic Coding Harness

- Entry point: `z` (or `python -m harness`)
- Config: `~/.z.json`
- Tests: `pytest tests/` (95 passing)
- Build: `python build.py`
- Package: `src/harness/` — everything lives here

## Architecture

- `main.py` — CLI, argparser, REPL loop, install wizard, session management
- `cline_agent.py` — Core agent loop: streaming, tool dispatch, context management
- `tool_handlers.py` — All tool implementations (file ops, shell, search, MCP, etc.)
- `streaming_client.py` — LiteLLM unified client, provider routing, OAuth
- `config.py` — Configuration loading from `~/.z.json`
- `smart_context.py` — Context compaction, tool result storage, content classification
- `hooks.py` — Lifecycle hooks (PreToolUse, PostToolUse, SessionStart, etc.)
- `memdir.py` / `claude_md.py` — Memory system and CLAUDE.md inheritance
- `output_protocol.py` — Structured JSON output, NDJSON progress, schema validation

## Style

- Relative imports within `src/harness/`
- No dead code — run `python -m vulture src/harness/` to verify
- Boot timing gated behind `HARNESS_TIMING=1`
- Single source of truth for config paths in `config.py`
