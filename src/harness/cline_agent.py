"""Streaming agent using Cline-style XML tool format."""

import asyncio
import sys
import os
import re
import time
import json
import datetime
import html
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
from .tool_registry import (
    get_tool_names,
    get_complex_content_tools,
    get_metrics,
    get_tool_def,
    get_param_names,
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

    def remove_by_source(self, source: str) -> int:
        """Remove all items with matching source. Returns count removed."""
        to_remove = [id for id, item in self._items.items() if source in item.source]
        for id in to_remove:
            del self._items[id]
        return len(to_remove)

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
    """Parsed tool call from XML format."""

    name: str
    parameters: Dict[str, str]


def strip_thinking_blocks(content: str) -> str:
    """Remove <thinking>/<think> reasoning blocks and orphaned </thinking> tags from content."""
    content = re.sub(r"<thinking>.*?</thinking>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"</thinking>\s*", "", content)
    # Also strip <think>...</think> blocks (used by many open-source reasoning models)
    # but do NOT strip orphaned </think> — those may be legitimate user content
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return content


def _normalize_tool_xml(content: str) -> str:
    """Fix common model hallucinations in tool call XML.

    Handles the <tool_call>tool_name> hybrid format where the model wraps
    a standard tool call in <tool_call> instead of using <tool_name> directly.
    Converts: <tool_call>tool_name>..params..</tool_name>
    Into:     <tool_name>..params..</tool_name>
    """
    tool_names = get_tool_names()
    tool_names_alt = "|".join(re.escape(n) for n in tool_names)
    pattern = rf"<tool_call>\s*({tool_names_alt})\s*>"
    content = re.sub(pattern, lambda m: f"<{m.group(1)}>", content)
    return content


def parse_xml_tool(content: str) -> Optional[ParsedToolCall]:
    """Parse Cline-style XML tool call from content.

    Uses smart matching to handle XML examples embedded in content.
    For tools with complex content (write_to_file, replace_in_file), uses
    greedy matching to get the full content including any nested examples.
    """
    content = strip_thinking_blocks(content)
    content = _normalize_tool_xml(content)

    tool_names = get_tool_names()

    # Compatibility parser for shorthand style:
    # <tool_call>list_files path="." recursive="true" />
    shorthand_matches = []
    shorthand_pattern = (
        r"<tool_call>\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([^<]*?)(?:/>\s*|>\s*</tool_call>)"
    )
    for m in re.finditer(shorthand_pattern, content, re.DOTALL):
        tool_name = m.group(1)
        if tool_name not in tool_names:
            continue

        attr_blob = m.group(2) or ""
        params: Dict[str, str] = {}
        for attr in re.finditer(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'=<>`]+))',
            attr_blob,
            re.DOTALL,
        ):
            key = attr.group(1)
            value = attr.group(2) or attr.group(3) or attr.group(4) or ""
            params[key] = html.unescape(value)

        shorthand_matches.append(
            (m.end(), ParsedToolCall(name=tool_name, parameters=params))
        )

    if shorthand_matches:
        shorthand_matches.sort(key=lambda x: x[0], reverse=True)
        return shorthand_matches[0][1]

    # Tools that may have nested XML examples in their content
    complex_content_tools = set(get_complex_content_tools())

    # Find ALL tool matches across ALL tool types, track by end position
    all_matches = []  # (end_pos, tool_name, match)

    for tool_name in tool_names:
        if tool_name in complex_content_tools:
            # For complex tools with nested XML examples in content:
            # Find FIRST opening tag and LAST closing tag to get the outermost block
            open_tag = f"<{tool_name}>"
            close_tag = f"</{tool_name}>"

            # Find the FIRST opening tag (the real tool call, not an inner example)
            first_open = content.find(open_tag)
            if first_open == -1:
                continue

            # Find the LAST closing tag (the real closing, not an inner example)
            last_close = content.rfind(close_tag)
            if last_close == -1 or last_close < first_open:
                continue

            # Create a match-like object
            inner_start = first_open + len(open_tag)
            inner_content = content[inner_start:last_close]

            class FakeMatch:
                def __init__(self, start_pos, end_pos, inner):
                    self._start = start_pos
                    self._end = end_pos
                    self._inner = inner

                def end(self):
                    return self._end

                def group(self, n):
                    return self._inner if n == 1 else content[self._start : self._end]

            fake_match = FakeMatch(
                first_open, last_close + len(close_tag), inner_content
            )
            all_matches.append((fake_match.end(), tool_name, fake_match))
        else:
            # For simple tools, use non-greedy matching
            pattern = rf"<{tool_name}>(.*?)</{tool_name}>"
            for match in re.finditer(pattern, content, re.DOTALL):
                all_matches.append((match.end(), tool_name, match))

    if not all_matches:
        return None

    # Use the match with the highest end position (last in content)
    all_matches.sort(key=lambda x: x[0], reverse=True)
    _, tool_name, match = all_matches[0]

    inner = match.group(1)
    params = {}

    # Parse parameters - order matters because large payload params may contain nested XML.
    # Use canonical tool definitions so parser stays aligned with the registry.
    tool_def = get_tool_def(tool_name)
    param_names = [p.name for p in tool_def.params] if tool_def else []
    complex_param_names = {"content", "diff", "replacement", "arguments", "old_text", "new_text"}
    complex_params = [p for p in param_names if p in complex_param_names]
    simple_params = [p for p in param_names if p not in complex_param_names]

    # For tools with complex content (write_to_file, replace_in_file),
    # extract the content/diff FIRST, then only search for other params
    # in the portion BEFORE <content> or <diff> starts
    search_area = inner  # default: search entire inner block

    # Find where complex params start (to avoid extracting example XML from content)
    for cp in complex_params:
        open_tag = f"<{cp}>"
        start = inner.find(open_tag)
        if start != -1:
            # Only search for simple params BEFORE the complex param starts
            search_area = inner[:start]
            break

    for param_name in simple_params:
        open_tag = f"<{param_name}>"
        close_tag = f"</{param_name}>"
        start = search_area.find(open_tag)
        if start == -1:
            continue
        # Use FIRST closing tag for simple params
        end = search_area.find(close_tag, start)
        if end == -1:
            continue
        value = search_area[start + len(open_tag) : end]
        value = value.strip("\n")
        params[param_name] = value

    for param_name in complex_params:
        open_tag = f"<{param_name}>"
        close_tag = f"</{param_name}>"
        start = inner.find(open_tag)
        if start == -1:
            continue
        # Use LAST closing tag for complex params (content may have nested examples)
        end = inner.rfind(close_tag)
        if end == -1 or end < start:
            continue
        value = inner[start + len(open_tag) : end]
        # Only strip leading/trailing newlines for complex content, preserve internal structure
        if param_name in complex_param_names:
            # For complex content like file content, preserve internal structure but clean edges
            value = value.strip("\n")
        else:
            # For simple params, strip all whitespace
            value = value.strip()
        params[param_name] = value

    return ParsedToolCall(name=tool_name, parameters=params)


def parse_all_xml_tools(content: str) -> List[ParsedToolCall]:
    """Parse ALL tool calls from content, in the order they appear.

    Unlike parse_xml_tool (which returns only the last match),
    this returns every non-overlapping tool call so that the agent
    can execute them sequentially. This is essential for batched
    operations like adding multiple todo items in one response.
    """
    stripped = strip_thinking_blocks(content)
    stripped = _normalize_tool_xml(stripped)

    tool_names = get_tool_names()

    # Find all <tool_name>...</tool_name> blocks with their positions
    matches = []  # (start_pos, end_pos, tool_name)
    for tool_name in tool_names:
        pattern = rf"<{tool_name}>(.*?)</{tool_name}>"
        for m in re.finditer(pattern, stripped, re.DOTALL):
            matches.append((m.start(), m.end(), tool_name, m.group(0)))

    # Sort by start position (order of appearance)
    matches.sort(key=lambda x: x[0])

    # Remove overlapping matches (keep earliest)
    filtered = []
    last_end = -1
    for start, end, tool_name, raw in matches:
        if start >= last_end:
            filtered.append((start, end, tool_name, raw))
            last_end = end

    # Parse each match into a ParsedToolCall using the existing parser
    results = []
    for start, end, tool_name, raw in filtered:
        parsed = parse_xml_tool(raw)
        if parsed:
            results.append(parsed)

    return results


class ClineAgent:
    """Agent using Cline-style XML tool format with streaming."""

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

        # Thrash detection: track consecutive edit failures per file
        # {filepath: {"failures": int, "last_error": str}}
        self._edit_failures: Dict[str, dict] = {}

        self._active_client = None  # set during _run_loop for introspect tool access

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
            "CRITICAL: No tools are available. Do NOT output any XML tags like "
            "<introspect>, <read_file>, <execute_command>, etc. "
            "Write ONLY natural language prose. Any XML tags will be stripped.\n\n"
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

        full_thinking = ""
        api_t0 = time.time()
        client = self._active_client

        # Build a set of tag patterns to suppress from the live stream.
        # The thinking model sometimes emits XML tool tags despite being told not to.
        _suppress_tags = set(get_tool_names())

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
        _tool_names_pattern = "|".join(re.escape(n) for n in get_tool_names())
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
                    or "read_file" not in sys_content
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
                # Extract tool name if any
                if "<read_file>" in text:
                    last_action = "reading files"
                elif "<write_to_file>" in text:
                    last_action = "writing files"
                elif "<execute_command>" in text:
                    last_action = "executing commands"
                elif "<search_files>" in text:
                    last_action = "searching code"
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
        _, max_allowed = get_model_limits(self.config.model)
        tokens = self.get_token_count()
        todos = self.todo_manager.list_all()
        active_todos = self.todo_manager.list_active()
        return {
            "tokens": tokens,
            "max_allowed": max_allowed,
            "percent": (tokens / max_allowed * 100) if max_allowed > 0 else 0,
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
        _, max_allowed = get_model_limits(self.config.model)

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

                _, max_allowed = get_model_limits(self.config.model)

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
                    log.info("Interrupted before semantic maintenance")
                    self.status.clear()
                    return "[Interrupted]"

                # Continuous semantic maintenance (lightweight) every turn.
                # Keeps stale guidance/noise trimmed even before hard thresholds.
                try:
                    self.messages, _maint_freed, _maint_report = (
                        self.smart_context.semantic_maintenance_tick(
                            self.messages,
                            max_allowed,
                            current_tokens=self._last_token_count,
                        )
                    )
                except Exception as e:
                    log.error("Semantic maintenance failed: %s", e)
                    _maint_freed = 0
                    _maint_report = "error"
                if _maint_freed > 0:
                    self._last_token_count = estimate_messages_tokens(self.messages)
                    log.info(
                        "Semantic maintenance freed %d tokens. Report: %s",
                        _maint_freed,
                        _maint_report,
                    )
                    self.console.print(
                        f"[dim][~] Semantic maintenance: freed {_maint_freed:,} tokens ({_maint_report})[/dim]"
                    )

                compact_threshold = int(max_allowed * self.config.compaction_threshold)
                if self._last_token_count > compact_threshold:
                    log.info(
                        "Context compaction triggered: %d tokens > %d threshold",
                        self._last_token_count,
                        compact_threshold,
                    )
                    self.status.update("Compacting context...", StatusLine.COMPACTING)
                    # Smart compaction: dedup → compact low-priority → evict lowest-scored
                    self.messages, freed, report = self.smart_context.compact_context(
                        self.messages,
                        max_allowed,
                        current_tokens=self._last_token_count,
                    )
                    if freed > 0:
                        self._last_token_count = estimate_messages_tokens(self.messages)
                        log.info(
                            "Compaction freed %d tokens, now %d tokens. Report: %s",
                            freed,
                            self._last_token_count,
                            report,
                        )
                        self.console.print(
                            f"[dim][!] Smart compaction: freed {freed:,} tokens ({report})[/dim]"
                        )

                        # Inject reorientation so the agent knows context changed
                        todo_state = self.todo_manager.format_list(
                            include_completed=False
                        )
                        recovery = self.smart_context.build_context_recovery_notice()

                        reorientation = "[CONTEXT COMPACTED - Re-orienting]\n"
                        if todo_state and "empty" not in todo_state.lower():
                            reorientation += f"\n{todo_state}\n"
                        if recovery:
                            reorientation += f"\n{recovery}\n"
                        reorientation += "\nCheck your todos and continue working on the current in-progress item."

                        self.messages.append(
                            StreamingMessage(role="user", content=reorientation)
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
                _payload_progress_step = 32 * 1024
                _payload_next_report = _payload_progress_step
                _payload_announced = False
                _stream_payload_enabled = (
                    os.environ.get("HARNESS_STREAM_CODE_CHANGES", "1") != "0"
                )
                _stream_tool_param = {
                    "write_to_file": "content",
                    "replace_in_file": "old_text",
                    "replace_between_anchors": "replacement",
                }
                _stream_tool_stack: List[str] = []
                _stream_in_tag = False
                _stream_tag_buf = ""
                _stream_in_payload = False
                _stream_payload_line = ""
                _stream_payload_header_printed = False
                _stream_payload_lines = 0
                _stream_payload_line_limit = 220

                def _stream_process_tag(raw_tag: str):
                    nonlocal \
                        _stream_tool_stack, \
                        _stream_in_payload, \
                        _stream_payload_line
                    tag = raw_tag.strip()
                    m = re.match(r"</?\s*([a-zA-Z_][a-zA-Z0-9_]*)", tag)
                    if not m:
                        return
                    name = m.group(1)
                    is_close = tag.startswith("</")
                    is_self_closing = tag.endswith("/>")

                    if name in _stream_tool_param:
                        if is_close:
                            if _stream_tool_stack and _stream_tool_stack[-1] == name:
                                _stream_tool_stack.pop()
                        elif not is_self_closing:
                            _stream_tool_stack.append(name)
                        return

                    if not _stream_tool_stack:
                        return

                    active_tool = _stream_tool_stack[-1]
                    expected_param = _stream_tool_param.get(active_tool)
                    if not expected_param or name != expected_param:
                        return

                    if is_close:
                        _stream_emit_payload_line(force=True)
                        _stream_in_payload = False
                    elif not is_self_closing:
                        _stream_in_payload = True
                        _stream_payload_line = ""

                def _stream_emit_payload_line(force: bool = False):
                    nonlocal _stream_payload_line, _stream_payload_lines
                    nonlocal _stream_payload_header_printed

                    if not _stream_payload_enabled:
                        _stream_payload_line = ""
                        return

                    if not force and "\n" not in _stream_payload_line:
                        return

                    line = _stream_payload_line
                    if not force:
                        line, _stream_payload_line = line.split("\n", 1)
                    else:
                        _stream_payload_line = ""

                    line = line.rstrip("\r")
                    if not line.strip():
                        return

                    if not _stream_payload_header_printed:
                        self.console.print()
                        self.console.print("  [dim]• Streaming changes…[/dim]")
                        _stream_payload_header_printed = True

                    preview = line
                    if len(preview) > _stream_payload_line_limit:
                        preview = preview[:_stream_payload_line_limit] + " ..."
                    self.console.print(f"    [dim]{rich_escape(preview)}[/dim]")
                    _stream_payload_lines += 1

                def _stream_process_chunk(chunk: str):
                    nonlocal _stream_in_tag, _stream_tag_buf
                    nonlocal _stream_in_payload, _stream_payload_line
                    if not _stream_payload_enabled:
                        return
                    for c in chunk:
                        if _stream_in_tag:
                            _stream_tag_buf += c
                            if c == ">":
                                _stream_process_tag(_stream_tag_buf)
                                _stream_tag_buf = ""
                                _stream_in_tag = False
                            continue

                        if c == "<":
                            _stream_in_tag = True
                            _stream_tag_buf = "<"
                            continue

                        if _stream_in_payload:
                            _stream_payload_line += c
                            if c == "\n":
                                _stream_emit_payload_line()
                            elif (
                                len(_stream_payload_line)
                                >= _stream_payload_line_limit * 2
                            ):
                                _stream_emit_payload_line(force=True)

                # ── XML stream filter ────────────────────────────────────
                # Suppresses raw XML tool tags from the live terminal while
                # full_content still accumulates everything for parsing.
                # IMPORTANT: chat_stream_raw passes MULTI-CHARACTER chunks
                # to on_chunk, so we iterate char-by-char internally.
                _sf_tool_names = set(get_tool_names()) | {"tool_call"}
                _sf_thinking_tags = {
                    "thinking", "think"
                }  # Both variants: models use <think> or <thinking> for reasoning
                _sf_suppressing: Optional[str] = None  # tool block being eaten
                _sf_thinking_suppress = False  # eating content inside <thinking>/<think>
                _sf_tag_buf = ""
                _sf_in_tag = False
                _sf_had_visible = False

                def _sf_flush():
                    nonlocal _sf_tag_buf, _sf_had_visible
                    if _sf_tag_buf:
                        # In deferred mode only thinking content is shown live;
                        # non-thinking buffered tags are rendered via Markdown later.
                        if not _defer_markdown_render:
                            sys.stdout.write(_sf_tag_buf)
                            sys.stdout.flush()
                            _sf_had_visible = True
                        _sf_tag_buf = ""

                def _sf_char(c: str):
                    """Process a single character through the filter."""
                    nonlocal \
                        _sf_suppressing, \
                        _sf_thinking_suppress, \
                        _sf_tag_buf, \
                        _sf_in_tag, \
                        _sf_had_visible

                    # ── SUPPRESS_BLOCK: eat everything until </tool_name> ──
                    if _sf_suppressing:
                        _sf_tag_buf += c
                        close = f"</{_sf_suppressing}>"
                        if _sf_tag_buf.endswith(close):
                            _sf_suppressing = None
                            _sf_tag_buf = ""
                        elif len(_sf_tag_buf) > len(close) + 30:
                            _sf_tag_buf = _sf_tag_buf[-(len(close) + 10) :]
                        return

                    # ── THINKING_SUPPRESS: route content to reasoning display ──
                    if _sf_thinking_suppress:
                        _sf_tag_buf += c
                        for tag in _sf_thinking_tags:
                            close = f"</{tag}>"
                            if _sf_tag_buf.endswith(close):
                                text = _sf_tag_buf[: -len(close)]
                                if text:
                                    on_reasoning(text)
                                _sf_thinking_suppress = False
                                _sf_tag_buf = ""
                                return
                        if len(_sf_tag_buf) > 200:
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
                                if tag in _sf_tool_names:
                                    if not is_close:
                                        _sf_suppressing = tag
                                    _sf_tag_buf = ""
                                    _sf_in_tag = False
                                    return
                                elif tag in _sf_thinking_tags:
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
                        sys.stdout.write(c)
                        sys.stdout.flush()
                        _sf_had_visible = True

                def on_chunk(chunk: str):
                    """Handle a (possibly multi-char) streaming chunk."""
                    nonlocal \
                        full_content, \
                        first_token, \
                        _payload_next_report, \
                        _payload_announced
                    full_content += chunk
                    if len(full_content) % 5000 == 0:
                        log.debug(
                            "on_chunk: accumulated content_len=%d chunk_len=%d",
                            len(full_content),
                            len(chunk),
                        )
                    _stream_process_chunk(chunk)
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
                    if _defer_markdown_render:
                        # Always scan every char through the filter so that:
                        # 1. <think>/<thinking> tags in content are extracted
                        #    and forwarded to on_reasoning for live display.
                        # 2. In deferred mode, _sf_char's NORMAL path suppresses
                        #    stdout writes, so regular content is not doubled.
                        for c in chunk:
                            _sf_char(c)
                        # Payload progress tracking — skip while thinking is live
                        # to avoid status-line noise interrupting the reasoning display.
                        if not _thinking_started:
                            active_tool = None
                            payload_bytes = 0
                            for tname in (
                                "write_to_file",
                                "replace_in_file",
                                "replace_between_anchors",
                            ):
                                open_tag = f"<{tname}>"
                                close_tag = f"</{tname}>"
                                open_pos = full_content.rfind(open_tag)
                                if open_pos == -1:
                                    continue
                                close_pos = full_content.rfind(close_tag)
                                if close_pos > open_pos:
                                    continue
                                active_tool = tname
                                payload_bytes = len(full_content) - open_pos
                                break

                            if active_tool and payload_bytes >= _payload_next_report:
                                if not _payload_announced:
                                    self.console.print(
                                        f"  [dim]• Receiving {active_tool}…[/dim]"
                                    )
                                    _payload_announced = True
                                kb = max(1, payload_bytes // 1024)
                                self.status.update(
                                    f"Receiving {active_tool} payload ({kb} KB)...",
                                    StatusLine.STREAMING,
                                )
                                _payload_next_report += _payload_progress_step
                        return
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
                        _thinking_line_has_text
                    if not chunk:
                        return
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
                        _payload_announced = False
                        _payload_next_report = _payload_progress_step
                        _thinking_started = False
                        _thinking_line_start = True
                        _thinking_line_has_text = False
                        # Reset stream filter state
                        _sf_suppressing = None
                        _sf_thinking_suppress = False
                        _sf_tag_buf = ""
                        _sf_in_tag = False
                        _sf_had_visible = False
                        # Reset payload stream state
                        _stream_tool_stack.clear()
                        _stream_in_tag = False
                        _stream_tag_buf = ""
                        _stream_in_payload = False
                        _stream_payload_line = ""
                        _stream_payload_header_printed = False
                        _stream_payload_lines = 0
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
                            )
                        )
                        watcher = asyncio.ensure_future(_interrupt_watcher(api_task))
                        try:
                            response = await asyncio.wait_for(api_task, timeout=120.0)
                        finally:
                            watcher.cancel()
                        break  # Success — exit retry loop

                    except asyncio.TimeoutError:
                        log.error("API call timed out after 120s")
                        self.console.print(
                            "[red]Error: API call timed out after 2 minutes[/red]"
                        )
                        return "[Error - API timeout. Check connection and try again]"

                    except asyncio.CancelledError:
                        # This happens when user presses Ctrl+C - client was closed
                        log.info("API call cancelled by user interrupt")
                        self.console.print("\n  [yellow]Interrupted by user[/yellow]")
                        return "[Interrupted]"

                    except RuntimeError as e:
                        if "closed" in str(e).lower():
                            # Client was closed due to interrupt
                            log.info("HTTP client closed due to interrupt")
                            self.console.print("\n  [yellow]Interrupted[/yellow]")
                            return "[Interrupted]"
                        raise

                    except Exception as e:
                        err_str = str(e).lower()
                        err_type = type(e).__name__

                        # Check for throttling/rate limit errors
                        is_throttle = any(kw in err_str for kw in (
                            "throttling", "throttled", "rate limit", "rate_limit",
                            "too many requests", "too many tokens", "quota",
                            "overloaded", "serviceunavailable",
                        )) or "429" in err_str or err_type in (
                            "ThrottlingException", "ServiceUnavailableException",
                        )

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
                                _payload_announced = False
                                _payload_next_report = _payload_progress_step
                                _thinking_started = False
                                _thinking_line_start = True
                                _thinking_line_has_text = False
                                _sf_suppressing = None
                                _sf_thinking_suppress = False
                                _sf_tag_buf = ""
                                _sf_in_tag = False
                                _sf_had_visible = False
                                _stream_tool_stack.clear()
                                _stream_in_tag = False
                                _stream_tag_buf = ""
                                _stream_in_payload = False
                                _stream_payload_line = ""
                                _stream_payload_header_printed = False
                                _stream_payload_lines = 0
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
                _stream_emit_payload_line(force=True)
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
                    self.status.clear()
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

                # Detect truncated output: the model hit max_tokens mid-response.
                # Only trigger continuation if there's an UNCLOSED tool tag —
                # meaning the model was genuinely cut off mid-tool-call.
                # If finish_reason is "length" but all tags are closed, the
                # response is usable; don't waste iterations on continuation.
                has_unclosed = self._has_unclosed_tool_tag(full_content)

                if has_unclosed:
                    log.warning(
                        "Output truncated (unclosed tool tag) at iteration %d (content_len=%d)",
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
                                "Please continue from where you left off. Do NOT repeat what you already wrote — "
                                "just output the remaining XML to complete the tool call.]"
                            ),
                        )
                    )
                    continue  # next iteration will get the continuation
                elif response.is_truncated:
                    # finish_reason was "length" but all tool tags are closed.
                    # The model finished its work; just log it and proceed.
                    log.info(
                        "finish_reason=length but all tool tags closed — proceeding normally (content_len=%d)",
                        len(full_content),
                    )

                # Parse ALL XML tool calls from content (not just the last one)
                all_tool_calls = parse_all_xml_tools(full_content)
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
                # Must be checked before the regular tool-execution loop so we
                # can return cleanly without routing through _dispatch_tool.
                completion_call = next(
                    (tc for tc in all_tool_calls if tc.name == "attempt_completion"),
                    None,
                )
                if completion_call:
                    result_text = completion_call.parameters.get("result", "").strip()
                    command = completion_call.parameters.get("command", "").strip()
                    log.info(
                        "attempt_completion received (result_len=%d, has_command=%s)",
                        len(result_text),
                        bool(command),
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
                        f"[DEBUG] parse_all_xml_tools returned: {len(all_tool_calls)} calls"
                    )

                if not tool_call:
                    # Detect invalid direct MCP tool-tag usage (e.g. <browser_navigate>).
                    # MCP tools must be invoked via mcp_call_tool, not direct tags.
                    tool_names_set = set(get_tool_names())
                    param_names_set = set(get_param_names())
                    invalid_tag_names: List[str] = []
                    for m in re.finditer(r"<([a-zA-Z_][a-zA-Z0-9_]*)>", full_content):
                        tag = m.group(1)
                        if tag in tool_names_set or tag in param_names_set:
                            continue
                        if tag in ("thinking", "think", "tool_call"):
                            continue
                        if "_" in tag:
                            invalid_tag_names.append(tag)

                    if invalid_tag_names:
                        bad = ", ".join(sorted(set(invalid_tag_names))[:5])
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
                                    "[SYSTEM: Invalid tool format. Do NOT emit direct tool tags like "
                                    f"{bad}. MCP tools must be called via <mcp_call_tool> with "
                                    "<server>, <tool>, and <arguments> JSON. Re-issue only valid harness tool XML.]"
                                ),
                            )
                        )
                        continue

                    display_text = strip_thinking_blocks(full_content).strip()

                    if _defer_markdown_render:
                        display_text = _normalize_display_text(display_text)
                        if display_text:
                            self.console.print()
                            self.console.print(Panel(Markdown(display_text), border_style="dim", padding=(0, 1)))

                    # Check for truly empty response and auto-retry with guidance.
                    # Only nudge when display_text is completely empty — short but
                    # non-empty responses (e.g. "yes", "42", "repeat this") are valid.
                    if not display_text:
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
                                "If you need to use a tool, emit the appropriate tool call in XML format."
                            )
                            self.messages.append(
                                StreamingMessage(role="user", content=guidance_msg)
                            )
                            self.console.print(
                                "\n  [dim]→ Model response was empty. Sending guidance nudge...[/dim]\n"
                            )

                            # Continue to next iteration to try again
                            continue

                    # If the model tried native function calling (tool_calls finish_reason)
                    # but no XML tool calls were found, nudge it to use XML format.
                    if response.finish_reason == "tool_calls" and not all_tool_calls:
                        log.warning(
                            "Model used native tool_calls instead of XML — sending format nudge"
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
                                    "[SYSTEM: You used native function calling, but this system requires XML tool format. "
                                    "Re-issue your tool call as XML. Example:\n"
                                    "<read_file>\n<path>filename</path>\n</read_file>"
                                ),
                            )
                        )
                        self.console.print(
                            "\n  [dim]→ Model used native tool calling. Nudging to use XML format...[/dim]\n"
                        )
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

                # Execute tool(s) — if multiple calls found, run them all
                tool_results_combined = []
                _mode_switched = False
                for tc_idx, tc in enumerate(all_tool_calls):
                    self.status.update(
                        f"Executing: {tc.name}"
                        + (
                            f" ({tc_idx + 1}/{len(all_tool_calls)})"
                            if len(all_tool_calls) > 1
                            else ""
                        ),
                        StatusLine.TOOL_EXEC,
                    )
                    tool_t0 = time.time()
                    log.info(
                        "Tool exec START: %s (%d/%d)",
                        tc.name,
                        tc_idx + 1,
                        len(all_tool_calls),
                    )
                    tc_result = await self._execute_tool(tc)
                    tool_elapsed = time.time() - tool_t0
                    log.info(
                        "Tool exec DONE: %s elapsed=%.1fs result_len=%d",
                        tc.name,
                        tool_elapsed,
                        len(tc_result or ""),
                    )

                    # Check for interrupt between tool calls
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

                    # Spill each individual result BEFORE combining, so that
                    # small results stay inline even when there are many calls.
                    tc_result = self.tool_handlers.spill_output_to_file(
                        tc_result, tc.name
                    )

                    if len(all_tool_calls) > 1:
                        tool_results_combined.append(
                            f"[{tc.name} result ({tc_idx + 1}/{len(all_tool_calls)})]:\n{tc_result}"
                        )
                    else:
                        tool_results_combined.append(tc_result)

                tool_result = "\n\n".join(tool_results_combined)

                # Add to history
                self.messages.append(
                    StreamingMessage(
                        role="assistant",
                        content=full_content,
                        provider_blocks=getattr(
                            response, "provider_content_blocks", None
                        ),
                    )
                )

                # Build tool result message
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
                # Label: for multi-tool batches show count; for single tool
                # use the actual tool name (tc is the last *executed* tool).
                if len(tool_results_combined) > 1:
                    _result_label = f"tool results ({len(tool_results_combined)} calls)"
                else:
                    _result_label = f"{tc.name} result"
                if header:
                    result_content = f"{header}\n\n[{_result_label}]\n{tool_result}"
                else:
                    result_content = f"[{_result_label}]\n{tool_result}"

                self.messages.append(
                    StreamingMessage(role="user", content=result_content)
                )

            log.warning("Max iterations reached (%d)", self.max_iterations)
            self.status.clear()
            return "Max iterations reached."
        finally:
            self._active_client = None
            if client is not None:
                await client.__aexit__(None, None, None)

    # Tool tag names — derived from the single registry
    _TOOL_TAGS = get_tool_names()

    @staticmethod
    def _has_unclosed_tool_tag(content: str) -> bool:
        """Detect if the content has an opening tool XML tag without a matching close.

        This indicates the model's output was truncated mid-tool-call,
        even if finish_reason wasn't set to 'length' by the API.
        """
        for tag in ClineAgent._TOOL_TAGS:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            if open_tag in content and close_tag not in content:
                return True
        return False

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
                lines = result.count("\n") + 1
                self.console.print(
                    f"  [dim]•[/dim] [cyan]Read[/cyan] [dim]{rich_escape(path)}  ({lines} lines)[/dim]"
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

    def _show_diff_blocks(self, blocks: List[Tuple[str, str]], path: str) -> None:
        """Display SEARCH/REPLACE blocks as a unified diff (only real changes)."""
        for search, replace in blocks:
            self._render_udiff(search, replace)

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

            parent_id_str = params.get("parent_id")

            item = self.todo_manager.update(
                item_id=item_id,
                title=params.get("title"),
                status=params.get("status"),
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
