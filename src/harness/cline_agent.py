"""Streaming agent with native tool calling via litellm."""

import asyncio
import sys
import os
import re
import time
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.markdown import Markdown
from rich.panel import Panel

import time as _time_mod_agent

_agent_import_t0 = _time_mod_agent.perf_counter()

from .streaming_client import StreamingJSONClient, StreamingMessage
from .config import Config
from .prompts import get_system_prompt
from .cost_tracker import get_global_tracker
from .logger import debug_print
from .interrupt import (
    is_interrupted,
    is_background_requested,
    reset_interrupt,
    get_interrupt_state,
    start_monitoring,
    stop_monitoring,
)
from .context_management import (
    estimate_tokens,
    estimate_messages_tokens,
    get_model_limits,
    truncate_conversation,
    truncate_output,
    truncate_file_content,
    DuplicateDetector,
)

_agent_t1 = _time_mod_agent.perf_counter()

from .tool_handlers import ToolHandlers

_agent_t2 = _time_mod_agent.perf_counter()

from .todo_manager import TodoManager, TodoStatus
from .smart_context import SmartContextManager

_agent_t3 = _time_mod_agent.perf_counter()

from .status_line import StatusLine
from .workspace_index import WorkspaceIndex
from .instruction_loader import load_instruction_hierarchy, load_subdirectory_instructions
from .tool_registry import (
    get_tool_names,
    get_metrics,
    tool_defs_to_openai_tools,
)
from .logger import get_logger, log_exception, truncate as log_truncate

_agent_t4 = _time_mod_agent.perf_counter()

import logging as _logging_mod

_boot_logger = _logging_mod.getLogger("harness.agent.boot")
_boot_logger.info(
    "cline_agent import breakdown: core=%.0fms tool_handlers=%.0fms smart_context=%.0fms rest=%.0fms total=%.0fms",
    (_agent_t1 - _agent_import_t0) * 1000,
    (_agent_t2 - _agent_t1) * 1000,
    (_agent_t3 - _agent_t2) * 1000,
    (_agent_t4 - _agent_t3) * 1000,
    (_agent_t4 - _agent_import_t0) * 1000,
)

log = get_logger("agent")


def _normalize_display_text(text: str) -> str:
    """Decode common escaped sequences for prettier terminal rendering.

    Some providers occasionally return JSON-style escaped text (e.g. ``\\u2501``
    or ``\\n``) in the assistant message body. Decode it for display only.
    Keep this heuristic conservative to avoid mangling legitimate code samples.
    """
    if not text:
        return text

    has_unicode_escapes = bool(re.search(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}", text))
    many_line_escapes = text.count("\\n") >= 3 and "\n" not in text
    if not (has_unicode_escapes or many_line_escapes):
        return text

    # Some providers double-escape JSON strings (e.g. "\\u2192" instead of
    # "\u2192"). Collapse common escape prefixes once before decoding.
    out = text
    out = out.replace("\\\\u", "\\u").replace("\\\\U", "\\U")
    out = out.replace("\\\\n", "\\n").replace("\\\\r", "\\r").replace("\\\\t", "\\t")
    out = out.replace("\\\\x", "\\x")

    def _u_replace(m: re.Match[str]) -> str:
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)

    def _U_replace(m: re.Match[str]) -> str:
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)

    out = re.sub(r"\\u([0-9a-fA-F]{4})", _u_replace, out)
    out = re.sub(r"\\U([0-9a-fA-F]{8})", _U_replace, out)
    out = re.sub(r"\\x([0-9a-fA-F]{2})", _u_replace, out)
    # Decode common control escapes when the payload appears escape-heavy.
    out = out.replace("\\r\\n", "\n").replace("\\n", "\n")
    out = out.replace("\\t", "\t")
    return out


@dataclass
class ContextItem:
    """An item in the agent's context container."""

    id: int
    type: str  # 'file', 'fragment', 'command_output', 'search_result'
    source: str  # path or command
    content: str
    added_at: float = field(default_factory=time.time)
    line_range: Optional[Tuple[int, int]] = None  # for file fragments

    def summary(self) -> str:
        """Return a short summary of this item."""
        lines = len(self.content.splitlines())
        size = len(self.content)
        age = int(time.time() - self.added_at)
        age_str = f"{age}s" if age < 60 else f"{age // 60}m"

        if self.type == "file":
            if self.line_range:
                return f"[{self.id}] file: {self.source} (L{self.line_range[0]}-{self.line_range[1]}, {lines}L, {size}B, {age_str} ago)"
            return f"[{self.id}] file: {self.source} ({lines}L, {size}B, {age_str} ago)"
        elif self.type == "command_output":
            cmd_short = (
                self.source[:40] + "..." if len(self.source) > 40 else self.source
            )
            return f"[{self.id}] cmd: {cmd_short} ({lines}L, {size}B, {age_str} ago)"
        elif self.type == "search_result":
            return f"[{self.id}] search: {self.source} ({lines} matches, {age_str} ago)"
        else:
            return f"[{self.id}] {self.type}: {self.source} ({lines}L, {age_str} ago)"


class ContextContainer:
    """Manages the agent's working context."""

    def __init__(self):
        self._items: Dict[int, ContextItem] = {}
        self._next_id = 1

    def add(
        self,
        type: str,
        source: str,
        content: str,
        line_range: Optional[Tuple[int, int]] = None,
    ) -> int:
        """Add an item to context. Returns the item ID."""
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = ContextItem(
            id=item_id, type=type, source=source, content=content, line_range=line_range
        )
        return item_id

    def remove(self, item_id: int) -> bool:
        """Remove an item from context."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def get(self, item_id: int) -> Optional[ContextItem]:
        return self._items.get(item_id)

    def list_items(self) -> List[ContextItem]:
        return list(self._items.values())

    def total_size(self) -> int:
        """Total character count of all context items."""
        return sum(len(item.content) for item in self._items.values())

    def summary(self) -> str:
        """Return a summary of all context items."""
        if not self._items:
            return "Context is empty."

        lines = [f"Context ({len(self._items)} items, {self.total_size():,} chars):"]
        for item in sorted(self._items.values(), key=lambda x: x.added_at):
            lines.append(f"  {item.summary()}")
        return "\n".join(lines)

    def clear(self):
        self._items.clear()
        self._next_id = 1


@dataclass
class ParsedToolCall:
    """Parsed tool call from model response."""

    name: str
    parameters: Dict[str, str]
    tool_call_id: Optional[str] = None


class ClineAgent:
    """Agent using native tool calling with streaming via litellm."""

    def __init__(
        self,
        config: Config,
        console: Optional[Console] = None,
        max_iterations: int = 500,
        providers: Optional[Dict[str, dict]] = None,
    ):
        self.config = config
        self.console = console or Console()
        self.max_iterations = max_iterations
        self.workspace_path = str(Path.cwd().resolve())
        self.cost_tracker = get_global_tracker()

        # Provider management
        self.providers = providers or {}

        # Conversation history
        self.messages: List[StreamingMessage] = []
        self._initialized = False

        # Context container for managing loaded content
        self.context = ContextContainer()

        # Duplicate file detection
        self._duplicate_detector = DuplicateDetector()

        # Todo list for tracking goals/objectives
        self.todo_manager = TodoManager()

        # Smart context manager for intelligent eviction/compaction
        _t0_sc = time.perf_counter()
        self.smart_context = SmartContextManager(self.todo_manager)
        log.info(
            "SmartContextManager init: %.1fms", (time.perf_counter() - _t0_sc) * 1000
        )

        # Token tracking
        self._last_token_count = 0

        # Persistent bottom status line
        self.status = StatusLine(enabled=sys.stdin.isatty())
        # Wrap console so prints automatically clear/restore status line
        self.console = self.status.wrap_console(self.console)

        # Thrash detection: track consecutive edit failures per file
        # {filepath: {"failures": int, "last_error": str}}
        self._edit_failures: Dict[str, dict] = {}

        self._active_client = None  # set during _run_loop for introspect tool access

        # Track which instruction files have been loaded (for on-demand subdir loading)
        self._loaded_instruction_paths: set = set()

        # Tool handlers - delegates all tool execution logic
        self.tool_handlers = ToolHandlers(
            config=self.config,
            console=self.console,
            workspace_path=self.workspace_path,
            context=self.context,
            duplicate_detector=self._duplicate_detector,
            context_manager=self.smart_context,
        )
        # Workspace index — built once at startup
        _t0_idx = time.perf_counter()
        self.workspace_index = WorkspaceIndex(self.workspace_path).build()
        log.info(
            "WorkspaceIndex.build(): %.1fms (%d files)",
            (time.perf_counter() - _t0_idx) * 1000,
            len(self.workspace_index.files),
        )

        # Plugin system — discover and load plugins
        from .plugin_manager import PluginManager
        from .tool_registry import TOOL_NAMES, register_plugin_tools
        plugin_paths = (self.config.plugins or []) if hasattr(self.config, 'plugins') else []
        plugin_configs = (self.config.plugin_config or {}) if hasattr(self.config, 'plugin_config') else {}
        self.plugin_manager = PluginManager(
            workspace_path=self.workspace_path,
            plugin_configs=plugin_configs,
        )
        self.plugin_manager.discover_and_load(
            extra_paths=plugin_paths,
            builtin_tool_names=set(TOOL_NAMES),
        )
        # Merge plugin tools into the global registry
        plugin_tool_defs = self.plugin_manager.get_tool_defs()
        if plugin_tool_defs:
            register_plugin_tools(plugin_tool_defs)
            log.info("Registered %d plugin tool(s) into global registry", len(plugin_tool_defs))

    def _load_instruction_files(self) -> str:
        """Load the full CLAUDE.md inheritance hierarchy.

        Mirrors Claude Code's model: walks up directory tree, loads
        ~/.claude/CLAUDE.md, project CLAUDE.md, .claude/rules/,
        agent.md, and resolves @path imports.
        """
        text, loaded_paths = load_instruction_hierarchy(self.workspace_path)
        self._loaded_instruction_paths = loaded_paths
        return text

    def _check_subdirectory_instructions(self, file_path: str) -> None:
        """Check for CLAUDE.md in subdirectories when reading a file.

        If new instruction files are found, inject them into the system
        prompt by refreshing it.
        """
        new_text = load_subdirectory_instructions(
            self.workspace_path, file_path, self._loaded_instruction_paths
        )
        if not new_text:
            return
        # Inject into the system message
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content += "\n\n====\n\nSUBDIRECTORY INSTRUCTIONS\n\n" + new_text
            log.info("Loaded subdirectory instructions for %s", file_path)

    def _system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        debug_print("_system_prompt: getting workspace summary...")
        index_summary = (
            self.workspace_index.summary() if self.workspace_index.files else ""
        )
        debug_print("_system_prompt: calling get_system_prompt...")
        base = get_system_prompt(self.workspace_path, project_map=index_summary)
        # Keep MCP server inventory visible in the active system prompt so the
        # model can use newly added servers without restarting the harness.
        debug_print("_system_prompt: loading MCP servers...")
        servers = {}
        try:
            servers = self.tool_handlers._load_mcp_servers()
            debug_print("_system_prompt: MCP servers loaded")
        except Exception as e:
            debug_print(f"_system_prompt: MCP load failed: {e}")
            servers = {}
        debug_print("_system_prompt: building result...")
        if isinstance(servers, dict) and servers:
            lines = ["MCP servers (from global config):"]
            for name in sorted(servers.keys()):
                cfg = servers.get(name, {}) or {}
                enabled = bool(cfg.get("enabled", True))
                stype = str(cfg.get("type", "local"))
                lines.append(
                    f"- {name} [{stype}] ({'enabled' if enabled else 'disabled'})"
                )
            debug_print("_system_prompt: DONE with servers")
            result = base + "\n\n====\n\n" + "\n".join(lines)
        else:
            debug_print("_system_prompt: DONE no servers")
            result = base + "\n\n====\n\nMCP servers (from global config): none configured."

        # Append plugin tool documentation
        plugin_docs = self.plugin_manager.get_tool_prompt_docs()
        if plugin_docs:
            result += "\n\n====\n\nPLUGIN TOOLS\n\n" + plugin_docs

        # Append workspace instruction files (CLAUDE.md, agent.md)
        instructions = self._load_instruction_files()
        if instructions:
            result += "\n\n====\n\nWORKSPACE INSTRUCTIONS\n\n" + instructions

        return result

    def refresh_system_prompt(self) -> bool:
        """Refresh the in-memory system prompt from current runtime config.

        Returns True if the system message was updated.
        """
        if not self._initialized:
            return False
        new_prompt = self._system_prompt()
        if self.messages and self.messages[0].role == "system":
            old = self.messages[0].content
            if old != new_prompt:
                self.messages[0] = StreamingMessage(role="system", content=new_prompt)
                return True
            return False
        self.messages.insert(0, StreamingMessage(role="system", content=new_prompt))
        return True

    @staticmethod
    def _thinking_system_prompt() -> str:
        """System prompt for deep analysis mode (introspect tool) — no tool definitions."""
        return (
            "You are a highly skilled software engineer in deep analysis mode. "
            "You've been working on a coding task and have gathered information. "
            "Your job right now is to THINK DEEPLY — not to act.\n\n"
            "CRITICAL: No tools are available in this mode. "
            "Write ONLY natural language prose.\n\n"
            "Write an extended reasoning monologue — multiple paragraphs. This is your "
            "working memory. Use it to organize your understanding, trace through "
            "details, identify patterns, catch issues, and plan your approach.\n\n"
            "Think like you're talking to yourself while solving a hard problem:\n"
            "- Trace through what you've seen: code structure, data flow, dependencies\n"
            "- Make connections between different pieces of information\n"
            "- Self-correct: 'Wait, actually...', 'Hmm, that contradicts...'\n"
            "- Consider alternatives, tradeoffs, and edge cases\n"
            "- Identify what you don't know yet and what matters most\n"
            "- Plan concrete next steps with clear reasoning for each\n\n"
            "Write several paragraphs minimum. Go deep. The more thorough your "
            "analysis, the better your subsequent actions will be."
        )

    def _build_introspect_prompt(self, focus: str = "") -> str:
        """Build the user message for an introspect tool call."""
        parts = []

        if focus:
            parts.append(f"FOCUS: {focus}")

        # Active task context
        todo_state = self.todo_manager.format_list(include_completed=False)
        if todo_state and "empty" not in todo_state.lower():
            parts.append(f"\nACTIVE TASKS:\n{todo_state}")

        parts.append(
            "\n[DEEP ANALYSIS MODE — No tools available. "
            "Analyze everything you've learned from the conversation above. "
            "Write an extended monologue — trace through the code, make connections, "
            "identify issues, plan your approach. "
            "The full conversation history above contains all the information you've gathered.]"
        )

        return "\n".join(parts)

    _INTROSPECT_MIN_CHARS = 600  # minimum chars for a useful introspect pass
    _INTROSPECT_MAX_CONTINUATIONS = 2  # max times to push for more depth

    async def _execute_introspect(self, focus: str = "") -> str:
        """Execute the introspect tool: a dedicated API call with no tools available.

        Makes a separate API call using a system prompt WITHOUT tool definitions,
        so the model can ONLY produce reasoning. If the initial output is too short,
        we push the model to continue (up to _INTROSPECT_MAX_CONTINUATIONS times).
        The thinking output is returned as the tool result.
        """
        log.info("Introspect tool called: focus=%r", focus[:80] if focus else "(none)")

        self.console.print(
            "  [dim]•[/dim] [magenta]Introspect[/magenta]"
            + (f" [dim]{focus[:80]}[/dim]" if focus else "")
        )

        # Build thinking messages: swap system prompt, keep conversation, add thinking prompt
        thinking_messages = list(self.messages)  # shallow copy
        thinking_messages[0] = StreamingMessage(
            role="system", content=self._thinking_system_prompt()
        )
        thinking_messages.append(
            StreamingMessage(role="user", content=self._build_introspect_prompt(focus))
        )

        # Filter images for non-vision models even in introspect mode
        from .image_utils import filter_messages_for_non_vision_model
        from .model_capabilities import supports_vision

        if not supports_vision(self.config.model, api_url=self.config.api_url):
            thinking_messages = filter_messages_for_non_vision_model(thinking_messages)

        full_thinking = ""
        api_t0 = time.time()
        client = self._active_client

        # Build a set of tag patterns to suppress from the live stream.
        # The thinking model sometimes emits XML tool tags despite being told not to.
        _suppress_tags = set(get_tool_names(model=self.config.model))

        for attempt in range(1 + self._INTROSPECT_MAX_CONTINUATIONS):
            first_token = True
            chunk_text = ""
            _tag_buf = ""  # buffer for potential XML tag detection
            _in_tag = False  # currently inside a < ... > sequence
            _suppress = False  # current tag should be suppressed

            def _flush_tag_buf():
                """Flush buffered tag to stdout (it wasn't a tool tag)."""
                nonlocal _tag_buf
                if _tag_buf:
                    sys.stdout.write(_tag_buf)
                    sys.stdout.flush()
                    _tag_buf = ""

            def on_chunk(c: str):
                nonlocal chunk_text, first_token, _tag_buf, _in_tag, _suppress
                if first_token:
                    self.status.clear()
                    first_token = False
                chunk_text += c

                # Stream filtering: suppress XML tool tags from display
                if c == "<" and not _in_tag:
                    _flush_tag_buf()
                    _in_tag = True
                    _suppress = False
                    _tag_buf = c
                    return

                if _in_tag:
                    _tag_buf += c
                    if c == ">":
                        # Tag complete — check if it's a tool tag
                        # Extract tag name: <tagname ...> or </tagname>
                        m = re.match(r"</?(\w+)", _tag_buf)
                        if m and m.group(1) in _suppress_tags:
                            # Suppress this tag (don't display it)
                            _tag_buf = ""
                        else:
                            _flush_tag_buf()
                        _in_tag = False
                        return
                    # Safety: if tag buffer gets too long, it's not a real tag
                    if len(_tag_buf) > 80:
                        _flush_tag_buf()
                        _in_tag = False
                    return

                self.status.clear()
                sys.stdout.write(c)
                sys.stdout.flush()

            self.status.update(
                "Thinking deeply..." if attempt == 0 else "Continuing analysis...",
                StatusLine.SENDING,
            )
            response = await client.chat_stream_raw(
                messages=thinking_messages,
                on_content=on_chunk,
                check_interrupt=is_interrupted,
                status_line=self.status,
            )
            _flush_tag_buf()  # flush any trailing buffered content
            self.status.clear()

            full_thinking += chunk_text

            # Track cost for this sub-call
            input_tokens = estimate_messages_tokens(thinking_messages)
            output_tokens = estimate_tokens(chunk_text)
            self.cost_tracker.record_call(
                model=self.config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=0,
                tool_calls=0,
                finish_reason=response.finish_reason,
                extra_usage=response.usage
                if getattr(response, "usage", None)
                else None,
            )

            # Check if output is deep enough
            if (
                len(full_thinking) >= self._INTROSPECT_MIN_CHARS
                or response.interrupted
                or response.is_truncated
            ):
                break

            # Output too short — push for more depth
            log.info(
                "Introspect attempt %d too short (%d chars, need %d). Continuing.",
                attempt + 1,
                len(full_thinking),
                self._INTROSPECT_MIN_CHARS,
            )

            # Add the short response and a continuation prompt
            thinking_messages.append(
                StreamingMessage(role="assistant", content=chunk_text)
            )
            thinking_messages.append(
                StreamingMessage(
                    role="user",
                    content=(
                        f"You've only written {len(full_thinking.split())} words. "
                        "Go much deeper. Trace through specific code paths you read. "
                        "Identify concrete bugs with file names and line numbers. "
                        "Consider race conditions, resource leaks, error handling gaps. "
                        "What are the non-obvious issues? What could break under load? "
                        "Continue your analysis."
                    ),
                )
            )

        print()  # newline after stream
        api_elapsed = time.time() - api_t0

        # Strip XML tool tags that the thinking model sometimes emits
        # despite being told not to (it pattern-matches from conversation history).
        _tool_names_pattern = "|".join(re.escape(n) for n in get_tool_names(model=self.config.model))
        full_thinking = re.sub(
            rf"</?(?:{_tool_names_pattern})\b[^>]*>", "", full_thinking
        ).strip()

        word_count = len(full_thinking.split())
        log.info(
            "Introspect complete: %d words, %d chars, %.1fs",
            word_count,
            len(full_thinking),
            api_elapsed,
        )

        result = (
            f"[Deep analysis complete — {word_count} words, {api_elapsed:.1f}s]\n\n"
            f"{full_thinking}"
        )
        return result

    async def cleanup_background_procs_async(self) -> None:
        """Terminate all background processes safely (async)."""
        return await self.tool_handlers.cleanup_background_procs_async()

    def cleanup_background_procs(self) -> None:
        """Terminate all background processes safely (sync wrapper)."""
        return self.tool_handlers.cleanup_background_procs()

    def list_background_procs(self) -> List[dict]:
        """List all background processes with their status."""
        return self.tool_handlers.list_background_procs()

    def save_session(self, path: str) -> None:
        """Save conversation history and context."""
        import json

        def sanitize_string(s) -> str:
            """Remove unicode surrogates that can't be encoded."""
            if isinstance(s, str):
                return s.encode("utf-8", errors="replace").decode("utf-8")
            return s

        def sanitize_content(content):
            """Sanitize message content (can be string or list for vision)."""
            if isinstance(content, str):
                return sanitize_string(content)
            elif isinstance(content, list):
                result = []
                for item in content:
                    if isinstance(item, dict):
                        sanitized = {}
                        for k, v in item.items():
                            sanitized[k] = (
                                sanitize_string(v) if isinstance(v, str) else v
                            )
                        result.append(sanitized)
                    else:
                        result.append(
                            sanitize_string(item) if isinstance(item, str) else item
                        )
                return result
            return content

        def sanitize_provider_blocks(blocks):
            if not isinstance(blocks, list):
                return None
            out = []
            for b in blocks:
                if isinstance(b, dict):
                    clean = {}
                    for k, v in b.items():
                        if isinstance(v, str):
                            clean[k] = sanitize_string(v)
                        elif isinstance(v, dict):
                            clean[k] = {
                                kk: sanitize_string(vv) if isinstance(vv, str) else vv
                                for kk, vv in v.items()
                            }
                        else:
                            clean[k] = v
                    out.append(clean)
            return out or None

        # Ensure session always persists a valid system prompt at index 0.
        if not self.messages or self.messages[0].role != "system":
            self.messages.insert(
                0, StreamingMessage(role="system", content=self._system_prompt())
            )

        # Serialize context items
        context_items = []
        for item in self.context.list_items():
            context_items.append(
                {
                    "id": item.id,
                    "type": item.type,
                    "source": sanitize_string(item.source),
                    "content": sanitize_string(item.content),
                    "added_at": item.added_at,
                    "line_range": item.line_range,
                }
            )

        data = {
            "workspace": self.workspace_path,
            "messages": [
                {
                    "role": m.role,
                    "content": sanitize_content(m.content),
                    **(
                        {
                            "provider_blocks": sanitize_provider_blocks(
                                getattr(m, "provider_blocks", None)
                            )
                        }
                        if getattr(m, "provider_blocks", None)
                        else {}
                    ),
                    **(
                        {"tool_calls": m.tool_calls}
                        if getattr(m, "tool_calls", None)
                        else {}
                    ),
                    **(
                        {"tool_call_id": m.tool_call_id}
                        if getattr(m, "tool_call_id", None)
                        else {}
                    ),
                    **(
                        {"name": m.name}
                        if getattr(m, "name", None)
                        else {}
                    ),
                }
                for m in self.messages
            ],
            "context": context_items,
            "context_next_id": self.context._next_id,
            "todos": self.todo_manager.to_dict(),
            "smart_context": self.smart_context.to_dict(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.debug(
            "Session saved: path=%s messages=%d context_items=%d",
            path,
            len(self.messages),
            len(context_items),
        )

    def load_session(self, path: str, inject_resume: bool = True) -> bool:
        """Load conversation history and context.

        Args:
            path: Path to session file
            inject_resume: If True, adds a resume context message to help the model
        """
        import json

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.messages = [
                StreamingMessage(
                    role=m["role"],
                    content=m["content"],
                    provider_blocks=m.get("provider_blocks"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                )
                for m in data["messages"]
            ]

            # CRITICAL: Verify system prompt integrity after load
            # If messages[0] is not a system prompt, inject a fresh one
            if not self.messages or self.messages[0].role != "system":
                self.console.print(
                    "[yellow][!] Session missing system prompt - injecting fresh one[/yellow]"
                )
                system_msg = StreamingMessage(
                    role="system", content=self._system_prompt()
                )
                self.messages.insert(0, system_msg)
            else:
                # Verify system prompt has tool definitions (not truncated/evicted)
                sys_content = self.messages[0].content
                if (
                    not isinstance(sys_content, str)
                    or "TOOL USE" not in sys_content
                ):
                    self.console.print(
                        "[yellow][!] System prompt appears corrupted - replacing with fresh one[/yellow]"
                    )
                    self.messages[0] = StreamingMessage(
                        role="system", content=self._system_prompt()
                    )

            # Load context if present
            if "context" in data:
                self.context.clear()
                for item_data in data["context"]:
                    item = ContextItem(
                        id=item_data["id"],
                        type=item_data["type"],
                        source=item_data["source"],
                        content=item_data["content"],
                        added_at=item_data.get("added_at", time.time()),
                        line_range=tuple(item_data["line_range"])
                        if item_data.get("line_range")
                        else None,
                    )
                    self.context._items[item.id] = item
                self.context._next_id = data.get(
                    "context_next_id", max(self.context._items.keys(), default=0) + 1
                )

            # Load todos if present
            if "todos" in data:
                self.todo_manager = TodoManager.from_dict(data["todos"])
                # Reconnect smart context manager to the loaded todo manager
                self.smart_context = SmartContextManager(self.todo_manager)

            # Load smart context state if present
            if "smart_context" in data:
                self.smart_context.load_dict(data["smart_context"])

            # Inject a resume context message to orient the model
            if inject_resume and len(self.messages) > 1:
                resume_msg = self._build_resume_context()
                if resume_msg:
                    self.messages.append(
                        StreamingMessage(role="user", content=resume_msg)
                    )

            self._initialized = True
            log.info(
                "Session loaded: path=%s messages=%d context_items=%d",
                path,
                len(self.messages),
                len(self.context.list_items()),
            )
            return True
        except Exception as e:
            log_exception(log, f"Failed to load session from {path}", e)
            self.console.print(f"[red][!] Failed to load session: {e}[/red]")
            return False

    def _build_resume_context(self) -> str:
        """Build a resume context message from conversation history."""
        # Find the last few exchanges to summarize
        summary_parts = []

        # Get last user request (not tool results)
        last_user_request = None
        for msg in reversed(self.messages):
            text = msg.content if isinstance(msg.content, str) else ""
            if (
                msg.role == "user"
                and text
                and not text.startswith("[")
                and not text.startswith("<")
            ):
                last_user_request = text[:200]
                break

        # Get last assistant action
        last_action = None
        for msg in reversed(self.messages):
            text = msg.content if isinstance(msg.content, str) else ""
            if msg.role == "assistant":
                # Check tool_calls attribute for native tool calling
                tc_list = getattr(msg, 'tool_calls', None)
                if tc_list:
                    tc_names = [tc.get('function', {}).get('name', '') for tc in tc_list if isinstance(tc, dict)]
                    if any(n in ('read_file',) for n in tc_names):
                        last_action = "reading files"
                    elif any(n in ('write_to_file',) for n in tc_names):
                        last_action = "writing files"
                    elif any(n in ('execute_command',) for n in tc_names):
                        last_action = "executing commands"
                    elif any(n in ('search_files',) for n in tc_names):
                        last_action = "searching code"
                    else:
                        last_action = tc_names[0] if tc_names else "responding"
                else:
                    last_action = "responding"
                break

        if last_user_request:
            summary_parts.append(f"Last request: {last_user_request}...")
        if last_action:
            summary_parts.append(f"Last action: {last_action}")

        if not summary_parts:
            return ""

        # Include todo list state if available
        todo_state = self.todo_manager.format_list(include_completed=False)
        if todo_state and "empty" not in todo_state.lower():
            summary_parts.append(f"\n{todo_state}")

        # Include eviction recovery notice
        recovery = self.smart_context.build_context_recovery_notice()
        if recovery:
            summary_parts.append(f"\n{recovery}")

        heuristic = " ".join(summary_parts)
        return f"[Session resumed. {heuristic}. Continue where you left off or ask what you need to know.]"

    def clear_history(self) -> None:
        """Clear conversation history and context."""
        debug_print("clear_history: START")
        self.messages = [StreamingMessage(role="system", content=self._system_prompt())]
        debug_print("clear_history: after messages")
        self.context.clear()
        debug_print("clear_history: after context.clear()")
        self._duplicate_detector.clear()
        debug_print("clear_history: after duplicate_detector.clear()")
        self.todo_manager.clear()
        debug_print("clear_history: after todo_manager.clear()")
        self.smart_context = SmartContextManager(self.todo_manager)
        debug_print("clear_history: after SmartContextManager")
        self._last_token_count = 0
        self._initialized = True
        debug_print("clear_history: DONE")

    def get_token_count(self) -> int:
        """Get estimated token count of current conversation."""
        return estimate_messages_tokens(self.messages)

    def get_context_stats(self) -> dict:
        """Get context statistics for display."""
        context_window, max_allowed = get_model_limits(self.config.model, api_url=self.config.api_url)
        tokens = self.get_token_count()
        todos = self.todo_manager.list_all()
        active_todos = self.todo_manager.list_active()
        return {
            "tokens": tokens,
            "max_allowed": max_allowed,
            "context_window": context_window,
            "percent": (tokens / context_window * 100) if context_window > 0 else 0,
            "messages": len(self.messages),
            "context_items": len(self.context.list_items()),
            "context_chars": self.context.total_size(),
            "todos_total": len(todos),
            "todos_active": len(active_todos),
            "todos_completed": len(todos) - len(active_todos),
            "evictions": len(self.smart_context.compaction_traces),
        }

    def get_token_breakdown(self) -> dict:
        """Get detailed breakdown of where tokens are going."""
        system_tokens = 0
        conv_tokens = 0
        message_sizes = []

        for i, msg in enumerate(self.messages):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tokens = estimate_tokens(content)
            if msg.role == "system":
                system_tokens += tokens
            else:
                conv_tokens += tokens
                # Track largest messages for diagnosis
                preview = content[:100].replace("\n", " ")
                message_sizes.append(
                    {"index": i, "role": msg.role, "tokens": tokens, "preview": preview}
                )

        # Sort by size, get top 5
        message_sizes.sort(key=lambda x: x["tokens"], reverse=True)
        largest = message_sizes[:5]

        return {
            "system": system_tokens,
            "conversation": conv_tokens,
            "total": system_tokens + conv_tokens,
            "message_count": len(self.messages) - 1,  # minus system
            "largest_messages": largest,
        }

    def dump_context(self, path: Optional[str] = None, reason: str = "") -> str:
        """Dump the full model context (all messages) to a JSON log file.

        This writes EXACTLY what would be sent to the API, making it possible
        to trace what the model actually sees.

        Args:
            path: Output file path. If None, auto-generates timestamped path.
            reason: Optional label for why this dump was triggered.

        Returns:
            Path to the dump file.
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.workspace_path, f".harness_context_{ts}.json")

        # Build the exact payload that would be sent to the API
        messages_data = []
        for i, msg in enumerate(self.messages):
            content = msg.content
            content_str = content if isinstance(content, str) else str(content)
            tokens = estimate_tokens(content_str)
            char_count = len(content_str)
            messages_data.append(
                {
                    "index": i,
                    "role": msg.role,
                    "tokens_est": tokens,
                    "chars": char_count,
                    "content": content,
                }
            )

        total_tokens = estimate_messages_tokens(self.messages)
        _, max_allowed = get_model_limits(self.config.model, api_url=self.config.api_url)

        # System prompt analysis
        sys_msg = (
            self.messages[0]
            if self.messages and self.messages[0].role == "system"
            else None
        )
        sys_analysis = {}
        if sys_msg:
            sys_content = (
                sys_msg.content
                if isinstance(sys_msg.content, str)
                else str(sys_msg.content)
            )
            sys_analysis = {
                "chars": len(sys_content),
                "tokens_est": estimate_tokens(sys_content),
                "has_read_file": "read_file" in sys_content,
                "has_write_to_file": "write_to_file" in sys_content,
                "has_execute_command": "execute_command" in sys_content,
                "has_replace_in_file": "replace_in_file" in sys_content,
                "has_manage_todos": "manage_todos" in sys_content,
                "has_TOOL_USE_section": "TOOL USE" in sys_content,
                "has_RULES_section": "RULES" in sys_content,
                "first_200": sys_content[:200],
                "last_200": sys_content[-200:],
            }

        dump = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "model": self.config.model,
            "api_url": self.config.api_url,
            "workspace": self.workspace_path,
            "summary": {
                "total_messages": len(self.messages),
                "total_tokens_est": total_tokens,
                "max_allowed": max_allowed,
                "percent_used": f"{(total_tokens / max_allowed * 100):.1f}%"
                if max_allowed > 0
                else "N/A",
                "system_prompt_analysis": sys_analysis,
            },
            "todos": self.todo_manager.to_dict(),
            "compaction_traces": [
                t.format_notice() for t in self.smart_context.compaction_traces
            ],
            "messages": messages_data,
        }

        Path(path).write_text(
            json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def compact_history(self, strategy: str = "half") -> int:
        """Compact conversation history by removing older messages.

        Strategies:
        - 'half': Remove first half of messages
        - 'quarter': Remove first quarter of messages
        - 'last2': Keep only last 2 exchanges

        Returns: number of tokens removed
        """
        before = estimate_messages_tokens(self.messages)

        # Find messages (skip system prompt at index 0)
        if len(self.messages) <= 1:
            return 0

        # Map strategy names
        strat_map = {"last2": "lastTwo"}
        strat = strat_map.get(strategy, strategy)

        # Apply truncation to messages (preserves system prompt)
        result = truncate_conversation(self.messages, strat)
        self.messages = result.messages

        after = estimate_messages_tokens(self.messages)
        return before - after

    async def run_message(
        self,
        user_content: Union[str, List[Dict[str, Any]]],
        enable_interrupt: bool = True,
        user_label: Optional[str] = None,
    ) -> str:
        """Run the agent with a user message (text or multimodal content)."""
        if isinstance(user_content, str):
            log_input = user_content
        else:
            log_input = user_label or "[multimodal message]"
        log.info(
            "agent.run START interrupt=%s msg_count=%d input=%s",
            enable_interrupt,
            len(self.messages),
            log_truncate(log_input, 150),
        )

        # Initialize system prompt if first run
        if not self._initialized:
            self.messages = [
                StreamingMessage(role="system", content=self._system_prompt()),
            ]
            self._initialized = True
            log.debug(
                "System prompt initialised (%d chars)", len(self.messages[0].content)
            )
        else:
            # Keep MCP/provider config changes reflected without requiring restart.
            self.refresh_system_prompt()

        # Seamless multimodal handling:
        # - Track attached images in the context container.
        # - For vision models, pass image blocks through to the API.
        # - For non-vision models, strip image blocks and replace them with
        #   placeholders so the API never receives raw images.
        if not isinstance(user_content, str) and user_content:
            from .image_utils import (
                content_has_image_blocks,
                adapt_content_for_non_vision_model,
                extract_image_paths_from_content,
            )
            from .model_capabilities import supports_vision

            image_paths = extract_image_paths_from_content(user_content)
            for p in image_paths:
                self.context.add("image", str(p), f"[Attached image: {p.name}]")

            if content_has_image_blocks(user_content):
                if supports_vision(self.config.model, api_url=self.config.api_url):
                    log.debug("Model supports vision; passing image blocks through.")
                else:
                    log.info(
                        "Model %s does not support vision; stripping image blocks.",
                        self.config.model,
                    )
                    user_content = adapt_content_for_non_vision_model(user_content)
                    log_input = user_label or "[image blocks omitted]"

        # Add user message
        self.messages.append(StreamingMessage(role="user", content=user_content))

        # Capture original request for todo grounding (first real user message)
        if (
            isinstance(user_content, str)
            and not self.todo_manager.original_request
            and not user_content.startswith("[")
        ):
            self.todo_manager.set_original_request(user_content[:500])

        # Start keyboard monitoring for escape/Ctrl+C.
        # Keep this enabled even in non-tty wrappers so SIGINT soft-cancel works.
        if enable_interrupt:
            start_monitoring()

        self.status.set_turn_start()
        try:
            result = await self._run_loop()
            log.info("agent.run DONE result_len=%d", len(result or ""))
            return result
        except Exception as exc:
            log_exception(log, "agent.run FAILED", exc)
            raise
        finally:
            self.status.clear()
            self.status.clear_turn()
            if enable_interrupt:
                stop_monitoring()

    async def run(self, user_input: str, enable_interrupt: bool = True) -> str:
        """Run the agent with a plain text user request."""
        return await self.run_message(user_input, enable_interrupt=enable_interrupt)

    async def _run_loop(self) -> str:
        """Main agent loop."""
        # Client is created inside the loop so that reasoning-mode switches
        # (which change self.config) take effect on the very next API call.
        client: Optional[StreamingJSONClient] = None
        _client_model: Optional[str] = (
            None  # track which config the client was built for
        )

        async def _ensure_client():
            """Create or recreate the streaming client when config changes."""
            nonlocal client, _client_model
            config_key = f"{self.config.api_url}|{self.config.model}"
            if client is not None and _client_model == config_key:
                return  # already up-to-date
            # Close old client first, then clear the reference so the finally
            # block won't try to double-close if __aenter__ fails below.
            if client is not None:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
                log.info(
                    "Closed old client (was %s), opening new one for %s",
                    _client_model,
                    config_key,
                )
            client = None
            _client_model = None
            new_client = StreamingJSONClient(
                api_key=self.config.api_key,
                base_url=self.config.api_url,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            new_client.reasoning_effort = getattr(self.config, "reasoning_effort", "high")
            await new_client.__aenter__()
            client = new_client
            self._active_client = client
            _client_model = config_key

        # Build native tool definitions for the API tools parameter
        native_tools = tool_defs_to_openai_tools(
            model=self.config.model, api_url=self.config.api_url
        )

        # For non-vision models, filter image blocks from all messages
        # before sending to API. Original messages are preserved in history.
        from .image_utils import filter_messages_for_non_vision_model
        from .model_capabilities import supports_vision

        api_messages = self.messages
        if not supports_vision(self.config.model, api_url=self.config.api_url):
            api_messages = filter_messages_for_non_vision_model(self.messages)
            log.debug("Filtered messages for non-vision model")

        try:
            _hidden_only_retry_count = 0
            _hidden_only_retry_max = 2
            _empty_nudge_count = 0
            _EMPTY_NUDGE_MAX = 3  # max consecutive empty-response nudges
            for iteration in range(self.max_iterations):
                log.info("[WATCHDOG] Starting iteration %d", iteration + 1)

                # (Re)create client if needed
                log.debug("[WATCHDOG] Ensuring client...")
                await _ensure_client()
                log.debug("[WATCHDOG] Client ready")

                # Sync reasoning_effort (user may toggle mid-conversation)
                if client:
                    client.reasoning_effort = getattr(self.config, "reasoning_effort", "high")

                _, max_allowed = get_model_limits(self.config.model, api_url=self.config.api_url)

                log.debug(
                    "_run_loop iteration=%d/%d tokens=%d msgs=%d",
                    iteration + 1,
                    self.max_iterations,
                    estimate_messages_tokens(self.messages),
                    len(self.messages),
                )
                self.status.set_iterations(iteration + 1, self.max_iterations)

                # Check for interrupt BEFORE starting new iteration
                # This ensures user can stop between iterations
                if is_interrupted():
                    log.info("Interrupted before iteration %d", iteration + 1)
                    self.status.clear()
                    self.console.print("\n[yellow][STOP] Interrupted by user[/yellow]")
                    return "[Interrupted - session preserved. Type to continue or start new request]"

                # Reset only background flag (not interrupt) for this iteration
                # User needs to explicitly continue after interrupt

                # Check if we need to compact/truncate conversation
                # Wrap in try/timeout to prevent hangs from corrupt context
                try:
                    import asyncio

                    self._last_token_count = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: estimate_messages_tokens(self.messages)
                        ),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    log.error("Token estimation timed out - context may be corrupt")
                    self.console.print("[red]Error: Token estimation timed out[/red]")
                    # Fallback: truncate aggressively
                    from .context_management import truncate_conversation

                    result = truncate_conversation(self.messages, strategy="lastTwo")
                    self.messages = result.messages
                    self._last_token_count = estimate_messages_tokens(self.messages)

                # Check interrupt before expensive operations
                if is_interrupted():
                    log.info("Interrupted before token check")
                    self.status.clear()
                    return "[Interrupted]"

                # Warn if context is getting large — user should use /compact
                compact_threshold = int(max_allowed * self.config.compaction_threshold)
                if self._last_token_count > compact_threshold:
                    pct = (self._last_token_count / max_allowed) * 100
                    log.warning(
                        "Context at %d/%d tokens (%.0f%%) — use /compact to truncate",
                        self._last_token_count,
                        max_allowed,
                        pct,
                    )
                    self.console.print(
                        f"[dim][~] Context at {pct:.0f}% capacity "
                        f"({self._last_token_count:,}/{max_allowed:,} tok). "
                        f"Use [bold]/compact[/bold] to truncate.[/dim]"
                    )

                # Auto-dump context before API call if debug enabled
                if os.environ.get("HARNESS_DEBUG_CONTEXT"):
                    dump_path = self.dump_context(
                        reason=f"pre_api_call_iter_{iteration}"
                    )
                    self.console.print(
                        f"[dim][DEBUG] Context dumped to {dump_path}[/dim]"
                    )

                # Show status: sending request
                token_info = f"{self._last_token_count // 1000}k tokens"
                self.status.update(f"Sending to LLM ({token_info})", StatusLine.SENDING)

                full_content = ""
                first_token = True
                _defer_markdown_render = bool(sys.stdout.isatty())

                # ── Stream filter ────────────────────────────────────
                # With native tool calling, tool invocations come via
                # structured tool_calls, not in content text. The filter
                # now only handles <thinking>/<think> tags.
                _sf_thinking_tags = {
                    "thinking", "think"
                }  # Both variants: models use <think> or <thinking> for reasoning
                _sf_thinking_suppress = False  # eating content inside <thinking>/<think>
                _sf_native_reasoning_active = False  # True if native reasoning_content is being used
                _sf_tag_buf = ""
                _sf_in_tag = False
                _sf_had_visible = False

                def _sf_flush():
                    """Flush buffered tag content to stdout."""
                    nonlocal _sf_tag_buf, _sf_in_tag, _sf_had_visible
                    if _sf_tag_buf:
                        if not _defer_markdown_render:
                            self.status.clear()
                            sys.stdout.write(_sf_tag_buf)
                            sys.stdout.flush()
                            _sf_had_visible = True
                        _sf_tag_buf = ""
                        _sf_in_tag = False

                def _sf_char(c: str):
                    """Process a single character through the filter.

                    Only handles <thinking>/<think> tag extraction now.
                    Tool calls are handled via native tool_calls, not content text.
                    """
                    nonlocal \
                        _sf_thinking_suppress, \
                        _sf_tag_buf, \
                        _sf_in_tag, \
                        _sf_had_visible

                    # ── THINKING_SUPPRESS: route content to reasoning display ──
                    if _sf_thinking_suppress:
                        _sf_tag_buf += c
                        for tag in _sf_thinking_tags:
                            close = f"</{tag}>"
                            if _sf_tag_buf.endswith(close):
                                text = _sf_tag_buf[: -len(close)]
                                if text and not _sf_native_reasoning_active:
                                    on_reasoning(text)
                                _sf_thinking_suppress = False
                                _sf_tag_buf = ""
                                return
                        if len(_sf_tag_buf) > 200:
                            if not _sf_native_reasoning_active:
                                on_reasoning(_sf_tag_buf[:-20])
                            _sf_tag_buf = _sf_tag_buf[-20:]
                        return

                    # ── DETECT_TAG: buffering after '<' ──
                    if _sf_in_tag:
                        _sf_tag_buf += c
                        if c == ">":
                            m = re.match(r"</?(\w+)", _sf_tag_buf)
                            if m:
                                tag = m.group(1)
                                is_close = _sf_tag_buf.startswith("</")
                                if tag in _sf_thinking_tags:
                                    # Skip <thinking> tag extraction if native reasoning is active
                                    # to avoid double-printing thinking content
                                    if _sf_native_reasoning_active:
                                        # Suppress the tags but don't route to on_reasoning
                                        if not is_close:
                                            _sf_thinking_suppress = True
                                            _sf_tag_buf = ""
                                        _sf_in_tag = False
                                        return
                                    if is_close:
                                        # Orphaned close tag outside a thinking block
                                        # — display it as-is (may be user content)
                                        _sf_flush()
                                    else:
                                        _sf_thinking_suppress = True
                                        _sf_tag_buf = ""
                                    _sf_in_tag = False
                                    return
                            _sf_flush()
                            _sf_in_tag = False
                            return
                        if len(_sf_tag_buf) > 80:
                            _sf_flush()
                            _sf_in_tag = False
                        return

                    # ── NORMAL ──
                    if c == "<":
                        _sf_in_tag = True
                        _sf_tag_buf = c
                        return

                    # In deferred/TTY mode, regular content is rendered via
                    # Markdown after the full response; only thinking content
                    # (routed through on_reasoning above) is shown live.
                    if not _defer_markdown_render:
                        self.status.clear()
                        sys.stdout.write(c)
                        sys.stdout.flush()
                        _sf_had_visible = True

                def on_chunk(chunk: str):
                    """Handle a (possibly multi-char) streaming chunk."""
                    nonlocal full_content, first_token
                    full_content += chunk
                    if len(full_content) % 5000 == 0:
                        log.debug(
                            "on_chunk: accumulated content_len=%d chunk_len=%d",
                            len(full_content),
                            len(chunk),
                        )
                    if first_token:
                        # If native reasoning already started, don't repaint the
                        # status line here; \r-based status updates can overwrite
                        # the currently streaming thinking line.
                        if _defer_markdown_render and not _thinking_started:
                            self.status.update(
                                "Receiving response...", StatusLine.STREAMING
                            )
                        else:
                            self.status.clear()
                        first_token = False
                    # Scan chars through the filter for <thinking>/<think> extraction
                    for c in chunk:
                        _sf_char(c)

                full_reasoning = ""
                _thinking_started = False
                _thinking_line_start = True
                _thinking_line_has_text = False
                _tty = sys.stdout.isatty()
                _ansi_dim = "\033[2m" if _tty else ""
                _ansi_italic = "\033[3m" if _tty else ""
                _ansi_reset = "\033[0m" if _tty else ""
                _thinking_prefix = f"{_ansi_dim}  {_ansi_reset}" if _tty else "  "
                _thinking_header = (
                    f"\n{_ansi_dim}{_ansi_italic}Thinking:{_ansi_reset}\n"
                    if _tty
                    else "\nThinking:\n"
                )

                def on_reasoning(chunk: str):
                    """Display native reasoning stream (reasoning_content)."""
                    nonlocal \
                        full_reasoning, \
                        _thinking_started, \
                        _thinking_line_start, \
                        _thinking_line_has_text, \
                        _sf_native_reasoning_active
                    if not chunk:
                        return
                    # Mark that we're receiving native reasoning - skip <thinking> tag extraction
                    _sf_native_reasoning_active = True
                    full_reasoning += chunk
                    if len(full_reasoning) % 3000 == 0:
                        log.debug(
                            "on_reasoning: accumulated reasoning_len=%d chunk_len=%d",
                            len(full_reasoning),
                            len(chunk),
                        )
                    if not _thinking_started and not chunk.strip():
                        return
                    if not _thinking_started:
                        self.status.clear()
                        sys.stdout.write(_thinking_header)
                        sys.stdout.flush()
                        _thinking_started = True
                        _thinking_line_start = True
                    # Clear status before writing thinking content
                    self.status.clear()
                    for c in chunk:
                        if _thinking_line_start:
                            # Skip leading blank lines so the block starts tight.
                            if c == "\n" and not _thinking_line_has_text:
                                continue
                            sys.stdout.write(_thinking_prefix)
                            _thinking_line_start = False
                        if _tty and c != "\n":
                            sys.stdout.write(f"{_ansi_dim}{c}{_ansi_reset}")
                        else:
                            sys.stdout.write(c)
                        if c not in ("\n", "\r", "\t", " "):
                            _thinking_line_has_text = True
                        if c == "\n":
                            _thinking_line_start = True
                            _thinking_line_has_text = False
                    sys.stdout.flush()

                # Stream the response - using raw mode (no JSON parsing)
                # Web search disabled by default to avoid unnecessary searches
                # Use /search command for explicit web searches
                api_t0 = time.time()
                log.info(
                    "API request: model=%s tokens_in=%d iter=%d",
                    self.config.model,
                    self._last_token_count,
                    iteration + 1,
                )

                # Wrap API call in timeout to prevent indefinite hangs.
                # Also run an interrupt-watcher so Ctrl+C / Escape cancel the
                # blocked await immediately (not only after the first chunk).
                import asyncio

                async def _interrupt_watcher(task: asyncio.Task):
                    """Cancel *task* as soon as the interrupt flag is set."""
                    while not task.done():
                        if is_interrupted():
                            task.cancel()
                            return
                        await asyncio.sleep(0.15)

                # Retry loop for throttling errors (Bedrock quota limits, rate limits, etc.)
                _throttle_max_retries = 10
                _throttle_base_wait = 30  # Start with 30 seconds for quota errors
                _throttle_attempt = 0
                response = None

                while _throttle_attempt <= _throttle_max_retries:
                    # Reset content buffers before each attempt so partial content
                    # from a failed mid-stream attempt doesn't get duplicated.
                    if _throttle_attempt > 0:
                        full_content = ""
                        full_reasoning = ""
                        first_token = True
                        _thinking_started = False
                        _thinking_line_start = True
                        _thinking_line_has_text = False
                        # Reset stream filter state
                        _sf_thinking_suppress = False
                        _sf_native_reasoning_active = False
                        _sf_tag_buf = ""
                        _sf_in_tag = False
                        _sf_had_visible = False
                        api_t0 = time.time()

                    try:
                        api_task = asyncio.ensure_future(
                            client.chat_stream_raw(
                                messages=self.messages,
                                on_content=on_chunk,
                                on_reasoning=on_reasoning,
                                check_interrupt=is_interrupted,
                                enable_web_search=False,
                                status_line=self.status,
                                tools=native_tools,
                            )
                        )
                        watcher = asyncio.ensure_future(_interrupt_watcher(api_task))
                        try:
                            # No total timeout — httpx per-read timeout (120s idle)
                            # handles stalled connections; _interrupt_watcher handles
                            # user interrupts.  The old asyncio.wait_for(timeout=120)
                            # was killing long-but-healthy streams because Python 3.12
                            # returns the CancelledError-caught result instead of
                            # raising TimeoutError, causing false "interrupted" flags.
                            response = await api_task
                        finally:
                            watcher.cancel()
                        break  # Success — exit retry loop

                    except asyncio.CancelledError:
                        # This happens when user presses Ctrl+C - client was closed
                        log.info("API call cancelled by user interrupt")
                        self.console.print("\n  [yellow]Interrupted by user[/yellow]")
                        return "[Interrupted]"

                    except RuntimeError as e:
                        err_lower = str(e).lower()
                        if "closed" in err_lower:
                            # Client was closed due to interrupt
                            log.info("HTTP client closed due to interrupt")
                            self.console.print("\n  [yellow]Interrupted[/yellow]")
                            return "[Interrupted]"
                        # Treat read timeouts as transient — retry instead of crashing
                        if "timeout" in err_lower or "timed out" in err_lower:
                            raise  # Fall through to Exception handler below
                        raise

                    except Exception as e:
                        err_str = str(e).lower()
                        err_type = type(e).__name__

                        # Check for throttling/rate limit errors OR transient timeouts
                        is_throttle = any(kw in err_str for kw in (
                            "throttling", "throttled", "rate limit", "rate_limit",
                            "too many requests", "too many tokens", "quota",
                            "overloaded", "serviceunavailable",
                        )) or "429" in err_str or err_type in (
                            "ThrottlingException", "ServiceUnavailableException",
                        )

                        # Read timeouts and connection errors are transient —
                        # the server may have dropped the connection or been
                        # temporarily unavailable.  Retry with a short backoff.

                        # For Bedrock token quotas, toggle between us./global. profiles
                        # Each profile has its own independent token bucket that refills gradually.
                        if (is_throttle
                                and "tokens per day" in err_str
                                and hasattr(client, '_bedrock_client')
                                and client._bedrock_client):
                            current_model = getattr(client._bedrock_client, 'model', '')
                            # Toggle: us. → global. or global. → us.
                            if current_model.startswith("us."):
                                alt_model = "global." + current_model[3:]
                            elif current_model.startswith("global."):
                                alt_model = "us." + current_model[7:]
                            else:
                                alt_model = None

                            if alt_model and not getattr(self, '_bedrock_alt_tried_this_attempt', False):
                                log.warning(
                                    "Token quota hit for %s, switching to %s",
                                    current_model, alt_model
                                )
                                self.console.print(
                                    f"  [yellow]Token quota hit for {current_model}, trying {alt_model}[/yellow]"
                                )
                                client._bedrock_client.model = alt_model
                                self._bedrock_alt_tried_this_attempt = True
                                # Reset buffers and retry immediately (no backoff needed)
                                full_content = ""
                                full_reasoning = ""
                                first_token = True
                                _thinking_started = False
                                _thinking_line_start = True
                                _thinking_line_has_text = False
                                _sf_thinking_suppress = False
                                _sf_native_reasoning_active = False
                                _sf_tag_buf = ""
                                _sf_in_tag = False
                                _sf_had_visible = False
                                api_t0 = time.time()
                                continue  # Retry with alternate profile immediately
                            else:
                                # Both profiles exhausted — reset flag for next retry cycle
                                self._bedrock_alt_tried_this_attempt = False

                        if is_throttle and _throttle_attempt < _throttle_max_retries:
                            _throttle_attempt += 1
                            # Tokens refill gradually (~350 tokens/minute observed).
                            # Use moderate backoff: 30s, 60s, 90s, 120s capped at 2 min.
                            wait_time = min(_throttle_base_wait + (30 * (_throttle_attempt - 1)), 120)
                            log.warning(
                                "Throttling detected (attempt %d/%d): %s. Waiting %ds...",
                                _throttle_attempt, _throttle_max_retries, str(e)[:100], wait_time
                            )
                            # Wait with interrupt checking and countdown display
                            _wait_remaining = wait_time
                            for _wi in range(int(wait_time * 10)):
                                if is_interrupted():
                                    self.status.clear()
                                    self.console.print("\n  [yellow]Interrupted during retry wait[/yellow]")
                                    return "[Interrupted]"
                                # Update countdown every second
                                _wait_remaining = wait_time - (_wi * 0.1)
                                if _wi % 10 == 0:
                                    self.status.set_retry(
                                        _throttle_attempt, _throttle_max_retries,
                                        max(0, _wait_remaining),
                                        reason="quota/rate limit"
                                    )
                                await asyncio.sleep(0.1)
                            self.status.update("Retrying request...", state=StatusLine.SENDING)
                            continue  # Retry
                        else:
                            # Not a throttle error or max retries exceeded — re-raise
                            raise

                if response is None:
                    self.console.print(
                        f"[red]Error: Max retries ({_throttle_max_retries}) exceeded for throttling[/red]"
                    )
                    return "[Error - API throttling. Try again later.]"
                api_elapsed = time.time() - api_t0
                log.info(
                    "API response: finish_reason=%s interrupted=%s elapsed=%.1fs content_len=%d reasoning_len=%d",
                    response.finish_reason,
                    response.interrupted,
                    api_elapsed,
                    len(full_content),
                    len(full_reasoning),
                )
                log.debug(
                    "Response object details: raw_json_len=%s thinking_len=%s usage=%s",
                    len(response.raw_json) if response.raw_json else 0,
                    len(response.thinking) if response.thinking else 0,
                    response.usage,
                )

                _sf_flush()  # Flush any trailing buffered content
                # If a <think> block was never closed (model truncated mid-thinking),
                # flush whatever is buffered in the thinking suppress state.
                if _sf_thinking_suppress and _sf_tag_buf:
                    on_reasoning(_sf_tag_buf)
                    _sf_tag_buf = ""
                if _thinking_started:
                    if not _thinking_line_start:
                        sys.stdout.write("\n")
                    sys.stdout.flush()
                self.status.clear()
                if _sf_had_visible:
                    print()  # Newline after visible stream content

                # Display web search results if any
                if response.has_web_search:
                    self.console.print(
                        f"\n[dim][Web Search: {len(response.web_search_results)} results][/dim]"
                    )
                    for result in response.web_search_results[:3]:  # Show top 3
                        self.console.print(
                            f"[dim]  - {result.title[:60]}... ({result.media})[/dim]"
                        )

                # Handle interrupt
                if response.interrupted:
                    _int_flag, _int_reason = get_interrupt_state().snapshot()
                    log.warning(
                        "Stream interrupted: flag=%s reason=%r",
                        _int_flag, _int_reason,
                    )
                    self.status.clear()
                    if _int_flag:
                        print(f"\n[STOP] Interrupted ({_int_reason})")
                    else:
                        # Interrupted without InterruptState being set —
                        # likely a task cancellation or unexpected condition.
                        log.warning("response.interrupted=True but InterruptState not set")
                        print("\n[STOP] Interrupted")
                    # Save partial response to history
                    if full_content.strip():
                        self.messages.append(
                            StreamingMessage(
                                role="assistant",
                                content=full_content + "\n[interrupted by user]",
                            )
                        )
                    return "[Interrupted - press Enter to continue or type new request]"

                full_content = response.content or full_content
                full_reasoning = response.thinking or full_reasoning

                # Save the actual visible text from model BEFORE thinking wrapping.
                # Used below to detect hidden-only responses (reasoning but no text/tool calls).
                _response_visible_text = full_content

                # Strip XML tool tags that the model sometimes emits despite using
                # native tool calling.  The old plugin docs taught XML format via
                # <tagname>...</tagname> usage examples in the system prompt; this
                # is defense-in-depth until that history fully flushes out.
                if _response_visible_text.strip():
                    _tool_names_pattern = "|".join(
                        re.escape(n) for n in get_tool_names(model=self.config.model)
                    )
                    _response_visible_text = re.sub(
                        rf"</?(?:{_tool_names_pattern})\b[^>]*>",
                        "",
                        _response_visible_text,
                    ).strip()
                    # Also strip from full_content so history stays clean
                    full_content = _response_visible_text

                # Wrap reasoning into full_content so the pipeline can see it
                if full_reasoning.strip():
                    full_content = (
                        f"<thinking>\n{full_reasoning}\n</thinking>\n{full_content}"
                    )

                # Track usage - estimate if API didn't return it
                if response.usage:
                    input_tokens = response.usage.get("prompt_tokens", 0)
                    output_tokens = response.usage.get("completion_tokens", 0)
                    # Fallback to estimation if API returns zeros
                    if input_tokens == 0 and output_tokens == 0:
                        input_tokens = estimate_messages_tokens(self.messages)
                        output_tokens = estimate_tokens(full_content)
                else:
                    # Estimate tokens
                    input_tokens = estimate_messages_tokens(self.messages)
                    output_tokens = estimate_tokens(full_content)

                self.cost_tracker.record_call(
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=0,
                    tool_calls=0,
                    finish_reason=response.finish_reason,
                    extra_usage=response.usage
                    if getattr(response, "usage", None)
                    else None,
                )
                log.debug(
                    "Usage: in=%d out=%d finish_reason=%s",
                    input_tokens,
                    output_tokens,
                    response.finish_reason,
                )

                # Detect truncated output: model hit max_tokens mid-response.
                if response.is_truncated and not response.tool_calls:
                    log.warning(
                        "Output truncated at iteration %d (content_len=%d, no tool_calls)",
                        iteration + 1,
                        len(full_content),
                    )
                    # Save the partial response and ask the model to finish
                    self.messages.append(
                        StreamingMessage(
                            role="assistant",
                            content=full_content,
                            provider_blocks=getattr(
                                response, "provider_content_blocks", None
                            ),
                        )
                    )
                    self.messages.append(
                        StreamingMessage(
                            role="user",
                            content=(
                                "[SYSTEM: Your output was truncated before completing the tool call. "
                                "Please continue from where you left off.]"
                            ),
                        )
                    )
                    continue  # next iteration will get the continuation

                # Convert native tool_calls from response to ParsedToolCall objects
                all_tool_calls = []
                if response.tool_calls:
                    for _tc_raw in response.tool_calls:
                        try:
                            _tc_name = _tc_raw["function"]["name"]
                            _tc_args_str = _tc_raw["function"].get("arguments", "{}")
                            _tc_args = json.loads(_tc_args_str) if _tc_args_str else {}
                            all_tool_calls.append(ParsedToolCall(
                                name=_tc_name,
                                parameters=_tc_args,
                                tool_call_id=_tc_raw.get("id"),
                            ))
                        except (KeyError, json.JSONDecodeError) as e:
                            log.warning("Failed to parse tool call: %s — %s", _tc_raw, e)
                tool_call = all_tool_calls[-1] if all_tool_calls else None

                if all_tool_calls:
                    _hidden_only_retry_count = 0
                    _empty_nudge_count = 0
                    log.info(
                        "Tools parsed: %d call(s) — %s",
                        len(all_tool_calls),
                        ", ".join(tc.name for tc in all_tool_calls),
                    )
                else:
                    log.debug(
                        "No tool call parsed from response (len=%d)", len(full_content)
                    )

                # Handle attempt_completion — agent signals it is done.
                # If attempt_completion appears alongside other tool calls,
                # execute those tools FIRST, then process the completion.
                # This prevents silent skipping of tool calls when the model
                # emits tools + attempt_completion in the same response.
                completion_call = next(
                    (tc for tc in all_tool_calls if tc.name == "attempt_completion"),
                    None,
                )
                if completion_call:
                    # Filter out attempt_completion — execute only the FIRST remaining tool (1-per-turn policy)
                    actionable_tools = [tc for tc in all_tool_calls if tc.name != "attempt_completion"]
                    if actionable_tools:
                        tc = actionable_tools[0]
                        ignored_pre = actionable_tools[1:]
                        log.info(
                            "attempt_completion found alongside %d other tool(s) — executing first tool only: %s",
                            len(actionable_tools),
                            tc.name,
                        )
                        if ignored_pre:
                            log.info(
                                "1-tool-per-turn policy (pre-completion): ignoring %d extra call(s): %s",
                                len(ignored_pre),
                                ", ".join(t.name for t in ignored_pre),
                            )
                        tool_results_combined = []
                        self.status.update(
                            f"Executing: {tc.name}",
                            StatusLine.TOOL_EXEC,
                        )
                        tool_t0 = time.time()
                        log.info("Tool exec START (pre-completion): %s", tc.name)
                        tc_result = await self._execute_tool(tc)
                        tool_elapsed = time.time() - tool_t0
                        log.info(
                            "Tool exec DONE (pre-completion): %s elapsed=%.1fs result_len=%d",
                            tc.name,
                            tool_elapsed,
                            len(tc_result or ""),
                        )
                        if is_interrupted():
                            self.console.print(
                                "\n[yellow][STOP] Interrupted by user[/yellow]"
                            )
                            self.messages.append(
                                StreamingMessage(
                                    role="assistant",
                                    content=full_content,
                                    provider_blocks=getattr(
                                        response, "provider_content_blocks", None
                                    ),
                                )
                            )
                            return "[Interrupted - session preserved. Type to continue or start new request]"
                        if tc.name not in self.tool_handlers._NO_SPILL_TOOLS:
                            tc_result = self.tool_handlers.spill_output_to_file(
                                tc_result, tc.name
                            )
                        tool_results_combined.append(
                            f"[{tc.name} result (1/1)]:\n{tc_result}"
                        )
                        if ignored_pre:
                            skipped_names = ", ".join(t.name for t in ignored_pre)
                            tool_results_combined.append(
                                f"[SYSTEM: Only 1 tool call is executed per turn. "
                                f"{len(ignored_pre)} additional tool call(s) were NOT executed: {skipped_names}. "
                                f"The attempt_completion was also deferred. Re-issue remaining tools one at a time.]"
                            )
                            # Feed the tool result + system notice back and
                            # continue the loop so the model can re-issue the
                            # dropped tools before completing.
                            log.info(
                                "Deferring attempt_completion — %d tool(s) were dropped, feeding result back",
                                len(ignored_pre),
                            )
                            tool_result = "\n\n".join(tool_results_combined)
                            _raw_tool_calls = response.tool_calls if response.tool_calls else None
                            self.messages.append(
                                StreamingMessage(
                                    role="assistant",
                                    content=full_content,
                                    tool_calls=_raw_tool_calls,
                                    provider_blocks=getattr(
                                        response, "provider_content_blocks", None
                                    ),
                                )
                            )
                            self.messages.append(
                                StreamingMessage(
                                    role="tool",
                                    content=tool_result,
                                    tool_call_id=tc.tool_call_id,
                                    name=tc.name,
                                )
                            )
                            # Synthetic results for all other ignored tool calls
                            for _ign in ignored_pre:
                                self.messages.append(
                                    StreamingMessage(
                                        role="tool",
                                        content="[Not executed — re-issue this tool call.]",
                                        tool_call_id=_ign.tool_call_id or f"fallback_{_ign.name}",
                                        name=_ign.name,
                                    )
                                )
                            # Synthetic result for the deferred attempt_completion
                            self.messages.append(
                                StreamingMessage(
                                    role="tool",
                                    content="[Deferred — complete remaining tools first.]",
                                    tool_call_id=completion_call.tool_call_id or f"fallback_{completion_call.name}",
                                    name="attempt_completion",
                                )
                            )
                            continue

                    result_text = completion_call.parameters.get("result", "").strip()
                    command = completion_call.parameters.get("command", "").strip()
                    log.info(
                        "attempt_completion received (result_len=%d, has_command=%s)",
                        len(result_text),
                        bool(command),
                    )
                    _raw_tool_calls = response.tool_calls if response.tool_calls else None
                    self.messages.append(
                        StreamingMessage(
                            role="assistant",
                            content=full_content,
                            tool_calls=_raw_tool_calls,
                            provider_blocks=getattr(
                                response, "provider_content_blocks", None
                            ),
                        )
                    )
                    # Add a synthetic tool result so every tool_calls has a
                    # matching role="tool" message.  Without this, providers
                    # like Copilot reject the next request with 400 Bad Request.
                    if _raw_tool_calls:
                        self.messages.append(
                            StreamingMessage(
                                role="tool",
                                content=result_text or "[Task completed]",
                                tool_call_id=completion_call.tool_call_id or f"fallback_{completion_call.name}",
                                name="attempt_completion",
                            )
                        )
                    # Render the result — stream filter suppressed it during streaming
                    if result_text:
                        if _defer_markdown_render:
                            self.console.print()
                            self.console.print(Panel(Markdown(result_text), border_style="dim", padding=(0, 1)))
                        else:
                            print(result_text)
                        if command:
                            self.console.print(f"\n  [dim]$ {command}[/dim]")
                    return full_content

                # Debug output - save full content for analysis
                if os.environ.get("HARNESS_DEBUG"):
                    debug_file = os.path.join(self.workspace_path, ".harness_debug.txt")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(
                            f"=== Full Model Output ({len(full_content)} chars) ===\n"
                        )
                        f.write(full_content)
                        f.write(
                            f"\n\n=== Parsed Tool Calls ({len(all_tool_calls)}) ===\n"
                        )
                        for tc in all_tool_calls:
                            f.write(f"  tool={tc.name}, params={tc.parameters}\n")
                    print(f"[DEBUG] Saved model output to {debug_file}")
                    print(
                        f"[DEBUG] native tool_calls returned: {len(all_tool_calls)} calls"
                    )

                if not tool_call:
                    # Display the VISIBLE model text only — never the <thinking>
                    # wrapper that was spliced into full_content for the internal
                    # pipeline/history. Rich's Markdown treats <thinking> as an
                    # HTML block and silently swallows it, producing an empty box.
                    display_text = _response_visible_text.strip()

                    if _defer_markdown_render:
                        display_text = _normalize_display_text(display_text)
                        if display_text:
                            self.console.print()
                            self.console.print(Panel(Markdown(display_text), border_style="dim", padding=(0, 1)))

                    # Check for hidden-only or empty response and auto-retry with guidance.
                    # Use _response_visible_text (pre-thinking-wrap) to detect the case
                    # where the model produced reasoning but no actual visible output or
                    # tool calls — full_content may be non-empty due to <thinking> wrapping
                    # but the Panel would render as blank.
                    if not _response_visible_text.strip():
                        thinking_len = (
                            len(response.thinking) if response.thinking else 0
                        )

                        if thinking_len > 50 and _hidden_only_retry_count < _hidden_only_retry_max:
                            # Hidden-only: model produced reasoning but no visible output.
                            # Nudge it to either call a tool or emit a visible completion.
                            _hidden_only_retry_count += 1
                            log.warning(
                                "Hidden-only response (thinking_len=%d, retry=%d/%d) — nudging for visible output",
                                thinking_len,
                                _hidden_only_retry_count,
                                _hidden_only_retry_max,
                            )
                            self.messages.append(
                                StreamingMessage(
                                    role="assistant",
                                    content=full_content,
                                    provider_blocks=getattr(
                                        response, "provider_content_blocks", None
                                    ),
                                )
                            )
                            self.messages.append(
                                StreamingMessage(
                                    role="user",
                                    content=(
                                        "Your response contained only internal reasoning with no visible output. "
                                        "Please proceed: either use a tool to continue the task, or use "
                                        "attempt_completion to present your final result."
                                    ),
                                )
                            )
                            self.console.print(
                                "\n  [dim]→ Hidden-only response. Nudging for visible output...[/dim]\n"
                            )
                            continue

                        elif _empty_nudge_count < _EMPTY_NUDGE_MAX:
                            # Truly empty response (no thinking either).
                            _empty_nudge_count += 1
                            log.warning(
                                "Model returned empty response (thinking_len=%d, content_len=%d, nudge=%d/%d) - sending guidance nudge",
                                thinking_len,
                                len(full_content),
                                _empty_nudge_count,
                                _EMPTY_NUDGE_MAX,
                            )

                            # Add the assistant's response (even if empty/thinking-only) to history
                            self.messages.append(
                                StreamingMessage(
                                    role="assistant",
                                    content=full_content,
                                    provider_blocks=getattr(
                                        response, "provider_content_blocks", None
                                    ),
                                )
                            )

                            # Add a guidance message to nudge the model
                            guidance_msg = (
                                "Your previous response was incomplete or contained only internal thinking. "
                                "Please provide a clear, written response to complete the user's request. "
                                "If you need to use a tool, call the appropriate tool."
                            )
                            self.messages.append(
                                StreamingMessage(role="user", content=guidance_msg)
                            )
                            self.console.print(
                                "\n  [dim]→ Model response was empty. Sending guidance nudge...[/dim]\n"
                            )

                            # Continue to next iteration to try again
                            continue

                    # No tool call — final response to user
                    self.messages.append(
                        StreamingMessage(
                            role="assistant",
                            content=full_content,
                            provider_blocks=getattr(
                                response, "provider_content_blocks", None
                            ),
                        )
                    )
                    return full_content

                # Execute only the FIRST tool call per turn (1 tool per turn policy).
                # If the model emitted multiple tool calls, only the first is executed;
                # the rest are reported back so the model can re-issue them.
                tc = all_tool_calls[0]
                ignored_tools = all_tool_calls[1:]

                if ignored_tools:
                    log.info(
                        "1-tool-per-turn policy: executing %s, ignoring %d extra call(s): %s",
                        tc.name,
                        len(ignored_tools),
                        ", ".join(t.name for t in ignored_tools),
                    )

                self.status.update(
                    f"Executing: {tc.name}",
                    StatusLine.TOOL_EXEC,
                )
                tool_t0 = time.time()
                log.info("Tool exec START: %s", tc.name)
                tc_result = await self._execute_tool(tc)
                tool_elapsed = time.time() - tool_t0
                log.info(
                    "Tool exec DONE: %s elapsed=%.1fs result_len=%d",
                    tc.name,
                    tool_elapsed,
                    len(tc_result or ""),
                )

                # Check for interrupt
                if is_interrupted():
                    self.console.print(
                        "\n[yellow][STOP] Interrupted by user[/yellow]"
                    )
                    self.messages.append(
                        StreamingMessage(
                            role="assistant",
                            content=full_content,
                            provider_blocks=getattr(
                                response, "provider_content_blocks", None
                            ),
                        )
                    )
                    return "[Interrupted - session preserved. Type to continue or start new request]"

                if tc.name not in self.tool_handlers._NO_SPILL_TOOLS:
                    tc_result = self.tool_handlers.spill_output_to_file(
                        tc_result, tc.name
                    )

                tool_results_combined = [tc_result]

                # If extra tool calls were ignored, inform the model
                if ignored_tools:
                    skipped_names = ", ".join(t.name for t in ignored_tools)
                    tool_results_combined.append(
                        f"[SYSTEM: Only 1 tool call is executed per turn. "
                        f"The following {len(ignored_tools)} tool call(s) were NOT executed: {skipped_names}. "
                        f"Please re-issue them one at a time in subsequent turns.]"
                    )

                tool_result = "\n\n".join(tool_results_combined)

                # Add assistant message with tool_calls to history
                _raw_tool_calls = response.tool_calls if response.tool_calls else None
                self.messages.append(
                    StreamingMessage(
                        role="assistant",
                        content=full_content,
                        tool_calls=_raw_tool_calls,
                        provider_blocks=getattr(
                            response, "provider_content_blocks", None
                        ),
                    )
                )

                # Build tool result message as role="tool" with tool_call_id
                header_parts = []

                # Active todos at top for grounding
                active_todos = self.todo_manager.list_active()
                if active_todos:
                    in_progress = [
                        t for t in active_todos if t.status.value == "in-progress"
                    ]
                    not_started = [
                        t for t in active_todos if t.status.value == "not-started"
                    ]
                    todo_hint = "[ACTIVE TODOS]"
                    if in_progress:
                        todo_hint += "\n  In progress: " + "; ".join(
                            f"[{t.id}] {t.title}" for t in in_progress
                        )
                    if not_started:
                        todo_hint += "\n  Remaining: " + "; ".join(
                            f"[{t.id}] {t.title}" for t in not_started[:5]
                        )
                        if len(not_started) > 5:
                            todo_hint += f" (+{len(not_started) - 5} more)"
                    header_parts.append(todo_hint)

                header = "\n".join(header_parts)
                if header:
                    result_content = f"{header}\n\n{tool_result}"
                else:
                    result_content = tool_result

                self.messages.append(
                    StreamingMessage(
                        role="tool",
                        content=result_content,
                        tool_call_id=tc.tool_call_id,
                        name=tc.name,
                    )
                )

                # Add synthetic tool results for ignored tool calls so every
                # tool_calls entry has a matching role="tool" message.
                # Without this, APIs like Copilot reject with 400.
                for _ignored_tc in ignored_tools:
                    self.messages.append(
                        StreamingMessage(
                            role="tool",
                            content="[Not executed — 1 tool per turn policy. Re-issue this tool call.]",
                            tool_call_id=_ignored_tc.tool_call_id or f"fallback_{_ignored_tc.name}",
                            name=_ignored_tc.name,
                        )
                    )

            log.warning("Max iterations reached (%d)", self.max_iterations)
            self.status.clear()
            return "Max iterations reached."
        finally:
            self._active_client = None
            if client is not None:
                await client.__aexit__(None, None, None)

    async def _execute_tool(self, tool: ParsedToolCall) -> str:
        """Execute a tool call and return the result."""
        # Clear status line before any console output to prevent
        # ghost text from status line interleaving with tool output
        self.status.clear()
        _metrics = get_metrics()
        _t0 = time.time()
        _success = True
        _error_msg = None

        try:
            result = await self._dispatch_tool(tool)
            return result
        except Exception as e:
            _success = False
            _error_msg = str(e)
            log_exception(log, f"Tool execution failed: {tool.name}", e)
            self.console.print(f"[red][X] {tool.name}: {rich_escape(str(e))}[/red]")
            return f"Error: {str(e)}"
        finally:
            _elapsed = (time.time() - _t0) * 1000
            _result_size = 0
            # result is local to the try block; capture it safely
            try:
                _result_size = len(result) if "result" in dir() else 0
            except Exception:
                pass
            _metrics.record(tool.name, _elapsed, _success, _error_msg, _result_size)

    async def _dispatch_tool(self, tool: ParsedToolCall) -> str:
        """Dispatch a parsed tool call to its handler. Returns result string."""
        try:
            if tool.name == "read_file":
                path = tool.parameters.get("path", "")
                result = await self.tool_handlers.read_file(tool.parameters)
                # On-demand subdirectory CLAUDE.md loading
                self._check_subdirectory_instructions(path)
                lines = result.count("\n") + 1
                start_line = tool.parameters.get("start_line", "")
                end_line = tool.parameters.get("end_line", "")
                line_range_info = ""
                if start_line or end_line:
                    line_range_info = f" L{start_line or '1'}-{end_line or 'end'}"
                self.console.print(
                    f"  [dim]•[/dim] [cyan]Read[/cyan] [dim]{rich_escape(path)}{line_range_info}  ({lines} lines)[/dim]"
                )

            elif tool.name == "write_to_file":
                path = tool.parameters.get("path", "")
                content = tool.parameters.get("content", "")
                # Capture old content for diff (if file exists)
                resolved = self.tool_handlers._resolve_path(path)
                old_content = None
                if resolved.exists():
                    try:
                        old_content = resolved.read_text(encoding="utf-8")
                    except Exception:
                        pass
                self.console.print(
                    f"  [dim]•[/dim] [green]Write[/green] [dim]{rich_escape(path)}[/dim]"
                )
                result = await self.tool_handlers.write_file(tool.parameters)
                # Show diff for overwrites, or creation summary for new files
                if old_content is not None and not result.startswith("Error:"):
                    self._show_write_diff(old_content, content, path)
                elif old_content is None and not result.startswith("Error:"):
                    n_lines = content.count("\n") + 1
                    self.console.print(
                        f"    [green]+ {n_lines} lines[/green] [dim](new file)[/dim]"
                    )

            elif tool.name == "replace_in_file":
                path = tool.parameters.get("path", "")
                # Backward compat: if model sends old <diff> format, parse it
                if "diff" in tool.parameters and "old_text" not in tool.parameters:
                    diff_text = tool.parameters["diff"]
                    import re as _re
                    m = _re.search(
                        r'<{7}\s*SEARCH\s*\n(.*?)\n={7}\s*\n(.*?)\n>{7}\s*REPLACE',
                        diff_text, _re.DOTALL
                    )
                    if m:
                        tool.parameters["old_text"] = m.group(1)
                        tool.parameters["new_text"] = m.group(2)
                    else:
                        # Treat entire diff as old_text (best effort)
                        tool.parameters["old_text"] = diff_text
                        tool.parameters["new_text"] = ""
                self.console.print(
                    f"  [dim]•[/dim] [yellow]Edit[/yellow] [dim]{rich_escape(path)}[/dim]"
                )
                result = await self.tool_handlers.replace_in_file(tool.parameters)

                # Show pretty diff on success
                old_text = tool.parameters.get("old_text", "")
                new_text = tool.parameters.get("new_text", "")
                if not result.startswith("Error:") and (old_text or new_text):
                    self._render_udiff(old_text, new_text)

                # Thrash detection: track consecutive failures
                norm_path = str(self.tool_handlers._resolve_path(path))
                if result.startswith("Error:"):
                    entry = self._edit_failures.setdefault(
                        norm_path, {"failures": 0, "last_error": ""}
                    )
                    entry["failures"] += 1
                    entry["last_error"] = result[:200]
                    n = entry["failures"]
                    if n >= 3:
                        self.console.print(
                            f"    [bold red]{n} consecutive edit failures on this file![/bold red]"
                        )
                        result += (
                            f"\n\n[REPEATED FAILURE — {n} consecutive failed edits on this file]\n"
                            f"You are stuck in a loop. STOP and try a DIFFERENT approach:\n"
                            f"  1. Re-read the file (read_file with start_line/end_line around the target area)\n"
                            f"  2. Copy the EXACT text from the file into your <old_text> block\n"
                            f"  3. If the file is badly broken, use write_to_file to rewrite the entire file\n"
                            f"  4. Make SMALLER edits — change only a few lines at a time\n"
                            f"Do NOT retry the same edit pattern. Change your approach."
                        )
                    elif n >= 2:
                        result += (
                            f"\n\nNote: This is the {n}nd consecutive failed edit on this file. "
                            f"Re-read the file to get the exact content before retrying."
                        )
                else:
                    # Success — reset failure counter
                    self._edit_failures.pop(norm_path, None)

            elif tool.name == "replace_between_anchors":
                path = tool.parameters.get("path", "")
                self.console.print(
                    f"  [dim]•[/dim] [yellow]Rewrite[/yellow] [dim]{rich_escape(path)}[/dim]"
                )
                result = await self.tool_handlers.replace_between_anchors(
                    tool.parameters
                )
                if not result.startswith("Error:"):
                    norm_path = str(self.tool_handlers._resolve_path(path))
                    self._edit_failures.pop(norm_path, None)

            elif tool.name == "execute_command":
                result = await self.tool_handlers.execute_command(tool.parameters)

            elif tool.name == "list_files":
                path = tool.parameters.get("path", "")
                result = await self.tool_handlers.list_files(tool.parameters)
                count = len(result.splitlines())
                self.console.print(
                    f"  [dim]•[/dim] [blue]Listed[/blue] [dim]{rich_escape(path)}  ({count} items)[/dim]"
                )

            elif tool.name == "search_files":
                regex = tool.parameters.get("regex", "")
                result = await self.tool_handlers.search_files(tool.parameters)
                matches = len(result.splitlines()) if result != "(no matches)" else 0
                self.console.print(
                    f"  [dim]•[/dim] [magenta]Searched[/magenta] [dim]{rich_escape(regex)}  ({matches} matches)[/dim]"
                )

            elif tool.name == "check_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(
                    f"  [dim]•[/dim] [cyan]Check process[/cyan] [dim]{bg_id or 'all'}[/dim]"
                )
                result = await self.tool_handlers.check_background_process(
                    tool.parameters
                )

            elif tool.name == "stop_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(
                    f"  [dim]•[/dim] [red]Stop process[/red] [dim]{bg_id}[/dim]"
                )
                result = await self.tool_handlers.stop_background_process(
                    tool.parameters
                )

            elif tool.name == "list_background_processes":
                self.console.print(f"  [dim]•[/dim] [cyan]List processes[/cyan]")
                result = await self.tool_handlers.list_background_processes(
                    tool.parameters
                )

            elif tool.name == "retrieve_tool_result":
                self.console.print(f"  [dim]•[/dim] [cyan]Retrieve result[/cyan]")
                result = await self.tool_handlers.retrieve_tool_result(tool.parameters)

            elif tool.name == "analyze_image":
                path = tool.parameters.get("path", "")
                question = tool.parameters.get(
                    "question", "Describe this image in detail."
                )
                self.console.print(
                    f"  [dim]•[/dim] [magenta]Analyze image[/magenta] [dim]{rich_escape(path)}[/dim]"
                )
                result = await self.tool_handlers.analyze_image(tool.parameters)

            elif tool.name == "web_search":
                query = tool.parameters.get("query", "")
                self.console.print(
                    f"  [dim]*[/dim] [cyan]Web search[/cyan] [dim]{rich_escape(query)}[/dim]"
                )
                result = await self.tool_handlers.web_search(tool.parameters)

            elif tool.name == "mcp_search_tools":
                server = tool.parameters.get("server", "")
                query = tool.parameters.get("query", "")
                self.console.print(
                    f"  [dim]*[/dim] [cyan]MCP search[/cyan] [dim]{rich_escape(server)} - {rich_escape(query)}[/dim]"
                )
                result = await self.tool_handlers.mcp_search_tools(tool.parameters)

            elif tool.name == "mcp_list_tools":
                server = tool.parameters.get("server", "")
                self.console.print(
                    f"  [dim]*[/dim] [cyan]MCP list[/cyan] [dim]{rich_escape(server)}[/dim]"
                )
                result = await self.tool_handlers.mcp_list_tools(tool.parameters)

            elif tool.name == "mcp_call_tool":
                server = tool.parameters.get("server", "")
                tname = tool.parameters.get("tool", "")
                self.console.print(
                    f"  [dim]*[/dim] [cyan]MCP call[/cyan] [dim]{rich_escape(server)}::{rich_escape(tname)}[/dim]"
                )
                result = await self.tool_handlers.mcp_call_tool(tool.parameters)

            elif tool.name == "manage_todos":
                action = tool.parameters.get("action", "list")
                self.console.print(
                    f"  [dim]*[/dim] [blue]Todo[/blue] [dim]{action}[/dim]"
                )
                result = self._handle_manage_todos(tool.parameters)
                # Render live todo panel after any todo change
                self.todo_manager.print_todo_panel(self.console)

            elif tool.name == "introspect":
                focus = tool.parameters.get("focus", "")
                result = await self._execute_introspect(focus)

            else:
                # Try plugin tools before giving up
                plugin_result = await self.plugin_manager.dispatch_tool(tool.name, tool.parameters)
                if plugin_result is not None:
                    ptool = self.plugin_manager.tools.get(tool.name)
                    label = (ptool.console_label if ptool and ptool.console_label
                             else f"[cyan]{tool.name}[/cyan]")
                    self.console.print(f"  [dim]•[/dim] {label}")
                    result = plugin_result
                else:
                    result = f"Unknown tool: {tool.name}"

            return result

        except Exception as e:
            raise  # Re-raise — _execute_tool handles logging + metrics

    # ── Pretty-printed diff display ─────────────────────────────────
    _DIFF_CONTEXT = 2  # Context lines around each change
    _DIFF_MAX_LINES = 20  # Max total diff lines before collapsing

    def _render_udiff(self, old_text: str, new_text: str) -> None:
        """Render a unified diff between two strings, showing only real changes."""
        import difflib

        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, n=self._DIFF_CONTEXT))

        if not diff:
            return

        shown = 0
        for line in diff[2:]:  # skip --- / +++ headers
            if shown >= self._DIFF_MAX_LINES:
                remaining = len(diff) - 2 - shown
                if remaining > 0:
                    self.console.print(f"    [dim]… {remaining} more diff lines[/dim]")
                break
            text = line.rstrip("\n")
            if line.startswith("+"):
                self.console.print(f"    [green]{rich_escape(text)}[/green]")
            elif line.startswith("-"):
                self.console.print(f"    [red]{rich_escape(text)}[/red]")
            elif line.startswith("@@"):
                self.console.print(f"    [cyan]{rich_escape(text)}[/cyan]")
            else:
                self.console.print(f"    [dim]{rich_escape(text)}[/dim]")
            shown += 1

    def _show_write_diff(self, old_content: str, new_content: str, path: str) -> None:
        """Display a unified diff for write_to_file overwrites."""
        self._render_udiff(old_content, new_content)

    def _handle_manage_todos(self, params: Dict[str, str]) -> str:
        """Handle the manage_todos tool."""
        action = params.get("action", "list").lower()

        if action == "add":
            title = params.get("title", "")
            if not title:
                return "Error: 'title' is required for add action."
            description = params.get("description", "")
            parent_id = None
            if params.get("parent_id"):
                try:
                    parent_id = int(params["parent_id"])
                except ValueError:
                    return f"Error: Invalid parent_id: {params['parent_id']}"
            context_refs = []
            if params.get("context_refs"):
                context_refs = [
                    r.strip() for r in params["context_refs"].split(",") if r.strip()
                ]

            item = self.todo_manager.add(
                title=title,
                description=description,
                parent_id=parent_id,
                context_refs=context_refs,
            )

            # If this is the first todo and we don't have an original request, set it
            if (
                not self.todo_manager.original_request
                and len(self.todo_manager.list_all()) == 1
            ):
                # Find the original user message
                for msg in self.messages:
                    _text = msg.content if isinstance(msg.content, str) else ""
                    if msg.role == "user" and _text and not _text.startswith("["):
                        self.todo_manager.set_original_request(_text[:500])
                        break

            return f"Added todo [{item.id}]: {item.title}\n\n{self.todo_manager.format_list()}"

        elif action == "update":
            item_id_str = params.get("id", "")
            if not item_id_str:
                return "Error: 'id' is required for update action."
            try:
                item_id = int(item_id_str)
            except ValueError:
                return f"Error: Invalid id: {item_id_str}"

            context_refs = None
            if params.get("context_refs"):
                context_refs = [
                    r.strip() for r in params["context_refs"].split(",") if r.strip()
                ]

            item = self.todo_manager.update(
                item_id=item_id,
                title=params.get("title"),
                status=params.get("status"),
                parent_id=params.get("parent_id"),
                description=params.get("description"),
                notes=params.get("notes"),
                context_refs=context_refs,
            )

            if not item:
                return f"Error: Todo [{item_id}] not found."

            return f"Updated todo [{item.id}]: {item.title} ({item.status.value})\n\n{self.todo_manager.format_list()}"

        elif action == "remove":
            item_id_str = params.get("id", "")
            if not item_id_str:
                return "Error: 'id' is required for remove action."
            try:
                item_id = int(item_id_str)
            except ValueError:
                return f"Error: Invalid id: {item_id_str}"

            if self.todo_manager.remove(item_id):
                return f"Removed todo [{item_id}].\n\n{self.todo_manager.format_list()}"
            else:
                return f"Error: Todo [{item_id}] not found."

        elif action == "list":
            return self.todo_manager.format_list()

        else:
            return (
                f"Error: Unknown action '{action}'. Use add, update, remove, or list."
            )
