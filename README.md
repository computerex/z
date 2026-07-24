# z

An agentic coding harness I built because I got tired of context-window anxiety and provider lock-in.

I use my own API keys. GLM 4.7, GLM 5.2, DeepSeek V4 Flash, DeepSeek V4 Pro — hot-swap mid-session with `/providers use`. No subscriptions, no rate limits I didn't sign up for.

## What makes it different

**Constant streaming.** Every token streams to the terminal as it arrives. Thinking/reasoning content visible in real time — nothing hidden, no "loading..." spinners. Background tool output streams transparently too.

**Telegram remote.** Fire up `z --telegram <token>` and drive your agent from your phone. Walking the dog, taking a dump, whatever — you can still ship.

**Context survives compaction.** Half, quarter, lastTwo strategies with breadcrumbs. Todos persist. Tool results are cached for retrieval. If z already read it, it doesn't re-read it.

**Undo/redo for AI turns.** Git snapshots before every agent action. `/undo` reverts files *and* conversation. `/redo` replays.

**26 providers, one `z`.** Z.AI, DeepSeek, Anthropic, OpenAI, Ollama, Bedrock, Copilot OAuth — all through LiteLLM. Normalizes model names automatically.

## Quick start

```bash
pip install -e ".[dev]"
z --install
z
```

Config lives in `~/.z.json`:

```json
{ "api_url": "https://api.z.ai/api/paas/v4/", "api_key": "sk-...", "model": "glm-4.7" }
```

Telegram:

```bash
z --telegram <bot-token> --telegram-username your_username
```

## Commands

| Command | What |
|---|---|
| `/providers use <name>` | Switch models mid-session |
| `/undo`, `/redo` | Undo/redo AI turns |
| `/compact` | Force context compaction |
| `/cost` | Token usage and spend |
| `/todo` | Todo panel |
| `!cmd` | Run shell command |

19 built-in tools: file ops, shell, search, web search, image analysis, MCP, background processes, sub-agents.

Borrows Claude Code semantics where they make sense — CLAUDE.md, hooks (PreToolUse, PostToolUse), persistent memory, scheduled tasks (cron).

Plugins: drop a `.py` in `~/.z/plugins/`.

MIT
