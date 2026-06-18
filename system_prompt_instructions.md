## CLAUDE.md, Memory & Hooks

**CLAUDE.md inheritance** — Place `CLAUDE.md` (project, committed) or `CLAUDE.local.md` (private, gitignored) in any directory. They're loaded from CWD up to root; deeper paths win. `.claude/rules/*.md` are loaded as unconditional rules. To make a rule conditional on which file the model touches, add `paths:` frontmatter (e.g., `paths: src/**/*.ts`). Use `@path/to/file.md` to include other files (max depth 5, text-only). Block HTML comments (`<!-- -->`) are stripped.

**Memory system** — Persistent file-based memory at `~/.claude/projects/<slug>/memory/`. Four types: `user` (user's role/preferences), `feedback` (guidance on approach), `project` (ongoing work context), `reference` (pointers to external systems). Each memory is a `.md` file with frontmatter: `---\nname: ...\ndescription: ...\ntype: ...\n---`. Index entries go in `MEMORY.md` (one-line pointers like `- [Title](file.md) — hook`). MEMORY.md is always loaded; topic files are fetched on-demand via relevance ranking.

**Hooks** — Configured in `.claude/settings.json`:
```json
{"hooks":{"PostToolUse":[{"matcher":"Read","hooks":[{"type":"command","command":"echo '$ARGUMENTS'","shell":"bash"}]}]}}
```
Events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `InstructionsLoaded`. Hook types: `command` (shell), `prompt` (LLM), `agent` (verifier), `http` (POST). Use `"if":"Bash(git *)"` to only fire when the tool call matches. Use `"async":true` for fire-and-forget, `"asyncRewake":true` to wake the model on error. Exit code 2 blocks the tool call; 0 = success; other = error shown to user.
