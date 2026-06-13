"""Single source of truth for all tool definitions.

Every tool known to the harness is defined ONCE here.  The parser,
the unclosed-tag detector, the system prompt, and the dispatch table
all derive their tool lists from this module.

Adding a new tool?  Add it here and it automatically propagates everywhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import time
import threading
import json
from .logger import get_logger

log = get_logger("registry")



@dataclass
class ToolParam:
    """Metadata for a single tool parameter."""
    name: str
    required: bool = False
    description: str = ""


@dataclass
class ToolDef:
    """Canonical definition of a tool.

    Everything the system needs to know about a tool lives here:
    name, parameters, and its category.
    """
    name: str
    category: str = "general"         # file, shell, search, context, meta, agent
    params: List[ToolParam] = field(default_factory=list)
    description: str = ""



TOOL_DEFS: List[ToolDef] = [
    # --- File operations ---
    ToolDef("read_file",       category="file",
            description="Read the contents of a file at the specified path. "
                        "For large files (over 2000 lines), only the first 300 lines are returned "
                        "unless you specify a line range with start_line/end_line. "
                        "Read before editing.",
            params=[ToolParam("path", required=True,
                              description="The path of the file to read (relative to the working directory)"),
                    ToolParam("start_line",
                              description="1-based line number to start reading from"),
                    ToolParam("end_line",
                              description="1-based line number to stop reading at (inclusive)")]),
    ToolDef("write_to_file",   category="file", 
            description="Write content to a file at the specified path. If the file exists it will "
                        "be overwritten. Creates directories as needed. Use for NEW files. "
                        "Prefer replace_in_file for modifying existing files. "
                        "ALWAYS provide the COMPLETE intended content no truncation. "
                        "MUST read existing files before overwriting.",
            params=[ToolParam("path", required=True,
                              description="The path of the file to write to"),
                    ToolParam("content", required=True,
                              description="The full content to write to the file")]),
    ToolDef("replace_in_file", category="file", 
            description="Perform exact string replacement in a file. You MUST read the file first. "
                        "old_text must match the file content EXACTLY character-for-character "
                        "including whitespace, indentation, and line endings. "
                        "Only the FIRST occurrence is replaced. Use multiple calls for multiple changes. "
                        "Keep replacements concise include just the changing lines and a few surrounding for uniqueness.",
            params=[ToolParam("path", required=True,
                              description="The path of the file to modify"),
                    ToolParam("old_text", required=True,
                              description="The exact text to find in the file (must match character-for-character)"),
                    ToolParam("new_text", required=True,
                              description="The replacement text (use empty string to delete code)")]),
    ToolDef("replace_between_anchors", category="file", 
            description="Replace everything BETWEEN two exact anchor strings in an existing file. "
                        "The anchors themselves are preserved. "
                        "Use when replace_in_file is brittle (delimiter collisions, large corrupted regions).",
            params=[ToolParam("path", required=True,
                              description="The path of the file to modify"),
                    ToolParam("start_anchor", required=True,
                              description="Exact string marking the start of the region to replace (preserved)"),
                    ToolParam("end_anchor", required=True,
                              description="Exact string marking the end of the region to replace (preserved)"),
                    ToolParam("replacement", required=True,
                              description="The new content to place between the anchors")]),

    # --- Shell / process ---
    ToolDef("execute_command",          category="shell",
            description="Execute a CLI command on the system. "
                        "Avoid using for find/grep/cat/sed use dedicated tools instead. "
                        "Use background=true for servers and long-running processes.",
            params=[ToolParam("command", required=True,
                              description="The CLI command to execute"),
                    ToolParam("background",
                              description="Set to 'true' to run as a background process")]),
    ToolDef("list_background_processes", category="shell",
            description="List all background processes (ID, PID, status, elapsed, command)."),
    ToolDef("check_background_process", category="shell",
            description="Check status and recent logs of a background process. "
                        "Don't poll in a loop check once, do other work, check later.",
            params=[ToolParam("id", required=True,
                              description="The ID of the background process to check"),
                    ToolParam("lines",
                              description="Number of recent log lines to return (default: 50)")]),
    ToolDef("stop_background_process",  category="shell",
            description="Terminate a background process. ONLY use when the user explicitly asks.",
            params=[ToolParam("id", required=True,
                              description="The ID of the background process to stop")]),

    # --- Search / exploration ---
    ToolDef("list_files",   category="search",
            description="Fast file listing tool that works with any codebase size. "
                        "Use when you need to find files by directory structure.",
            params=[ToolParam("path", required=True,
                              description="The directory path to list"),
                    ToolParam("recursive",
                              description="Set to 'true' for recursive listing")]),
    ToolDef("search_files", category="search",
            description="A powerful content search tool built on ripgrep. "
                        "ALWAYS use this for content search tasks. NEVER invoke grep or rg "
                        "as an execute_command command. Supports full regex syntax.",
            params=[ToolParam("path", required=True,
                              description="The directory to search in (recursive)"),
                    ToolParam("regex", required=True,
                              description="The regular expression pattern to search for (Python regex syntax)"),
                    ToolParam("file_pattern",
                              description="Glob pattern to filter files (e.g., *.py)")]),

    # --- External / vision / web ---
    ToolDef("analyze_image", category="external",
            description="Analyze an image file (jpg, png, gif, webp) using a vision model.",
            params=[ToolParam("path", required=True,
                              description="The path of the image file to analyze"),
                    ToolParam("question",
                              description="A specific question about the image")]),
    ToolDef("web_search",    category="external",
            description="Search the web for real-time information. Use for up-to-date information "
                        "that might not be available in training data.",
            params=[ToolParam("query", required=True,
                              description="The search query"),
                    ToolParam("count",
                              description="Number of results to return (default: 5)")]),
    ToolDef("mcp_search_tools", category="external",
            description="Semantically search tools exposed by a configured MCP server. "
                        "Use this FIRST when you do not know the exact tool name.",
            params=[ToolParam("server", required=True,
                              description="The name of the MCP server to search"),
                    ToolParam("query", required=True,
                              description="A natural language query describing what you want to do"),
                    ToolParam("limit",
                              description="Maximum number of results to return")]),
    ToolDef("mcp_list_tools", category="external",
            description="List all tools exposed by a configured MCP server, "
                        "including required fields. Use to confirm tool input schema before calling.",
            params=[ToolParam("server", required=True,
                              description="The name of the MCP server to list tools for")]),
    ToolDef("mcp_call_tool", category="external", 
            description="Call a specific tool on a configured MCP server. "
                        "Always invoke MCP tools via this tool.",
            params=[ToolParam("server", required=True,
                              description="The name of the MCP server"),
                    ToolParam("tool", required=True,
                              description="The name of the tool to call"),
                    ToolParam("arguments", required=True,
                              description="JSON object with the tool's input arguments")]),

    # --- Context management ---
    ToolDef("retrieve_tool_result", category="context",
            description="Retrieve a stored tool result by its ID. Use this to access "
                       "the output of previous tool executions that have been stored in the context.",
            params=[ToolParam("result_id", required=True,
                    description="The ID of the stored tool result (e.g., res_abc123_456)")]),

    # --- Agent meta ---
    ToolDef("manage_todos",     category="agent",
            description="Track goals and progress with a structured task list. "
                        "Persists across context compaction your permanent memory. "
                        "For complex tasks: break into todos FIRST.",
            params=[ToolParam("action", required=True,
                              description="One of: add, update, remove, list"),
                    ToolParam("id", description="The todo item ID (required for update and remove)"),
                    ToolParam("title", description="The title of the todo item (required for add)"),
                    ToolParam("description", description="Optional longer description"),
                    ToolParam("status", description="One of: not-started, in-progress, completed, blocked"),
                    ToolParam("parent_id", description="ID of a parent todo to create a subtask"),
                    ToolParam("notes", description="Freeform notes to attach to a todo"),
                    ToolParam("context_refs", description="Comma-separated list of context references")]),
    ToolDef("introspect",          category="agent",
            description="Dedicated deep-thinking tool. Makes a separate API call with no tools "
                        "available so you can reason freely without constraints. "
                        "Use when facing complex decisions, debugging tricky issues, or planning multi-step approaches.",
            params=[ToolParam("focus",
                              description="A description of what to focus your thinking on")]),
    ToolDef("attempt_completion", category="agent",
            description="Signal that the task is complete and present the result to the user. "
                        "Use ONLY after confirming all previous tool uses succeeded. "
                        "Do NOT end the result with questions or offers for further assistance.",
            params=[ToolParam("result", required=True,
                              description="The final result of the task must be complete and not require further input"),
                    ToolParam("command",
                              description="A CLI command to demonstrate the result (e.g., open a browser, run a script)")]),
]

# Derived lookups (computed once at import time)
TOOL_NAMES: List[str] = [t.name for t in TOOL_DEFS]
TOOL_BY_NAME: Dict[str, ToolDef] = {t.name: t for t in TOOL_DEFS}


def rebuild_lookups() -> None:
    """Rebuild derived lookups after tools are added dynamically (e.g. plugins)."""
    global TOOL_NAMES, TOOL_BY_NAME
    TOOL_NAMES = [t.name for t in TOOL_DEFS]
    TOOL_BY_NAME = {t.name: t for t in TOOL_DEFS}


def register_plugin_tools(tool_defs: List[ToolDef]) -> None:
    """Register plugin-contributed tools into the global registry."""
    for td in tool_defs:
        if td.name not in TOOL_BY_NAME:
            TOOL_DEFS.append(td)
    rebuild_lookups()


def get_tool_names() -> List[str]:
    """Return canonical list of tool names."""
    return TOOL_NAMES


def tool_defs_to_openai_tools(tool_defs: Optional[List[ToolDef]] = None) -> List[Dict[str, Any]]:
    """Convert ToolDef list to OpenAI function-calling tool schema.

    Returns a list of dicts suitable for the ``tools`` parameter of
    litellm.acompletion / openai.chat.completions.create.
    """
    defs = tool_defs if tool_defs is not None else TOOL_DEFS
    tools: List[Dict[str, Any]] = []
    for td in defs:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for p in td.params:
            properties[p.name] = {"type": "string"}
            if p.description:
                properties[p.name]["description"] = p.description
            if p.required:
                required.append(p.name)

        func: Dict[str, Any] = {"name": td.name}
        if td.description:
            func["description"] = td.description
        params_schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            params_schema["required"] = required
        func["parameters"] = params_schema

        tools.append({"type": "function", "function": func})
    return tools


@dataclass
class ToolCallRecord:
    """A single tool invocation record."""
    tool_name: str
    started_at: float
    elapsed_ms: float
    success: bool
    error: Optional[str] = None
    result_size: int = 0


class ToolMetrics:
    """Thread-safe observability for tool execution.

    Tracks per-tool: call count, total time, error count, last N calls.
    Queryable at any time for dashboards / logging.
    """

    def __init__(self, history_size: int = 200):
        self._lock = threading.Lock()
        self._history_size = history_size
        self._calls: List[ToolCallRecord] = []
        self._per_tool: Dict[str, dict] = {}  # name -> {count, total_ms, errors}
        self._start_time = time.time()

    def record(self, tool_name: str, elapsed_ms: float, success: bool,
               error: Optional[str] = None, result_size: int = 0) -> None:
        """Record a tool call."""
        rec = ToolCallRecord(
            tool_name=tool_name,
            started_at=time.time(),
            elapsed_ms=elapsed_ms,
            success=success,
            error=error,
            result_size=result_size,
        )
        with self._lock:
            self._calls.append(rec)
            if len(self._calls) > self._history_size:
                self._calls = self._calls[-self._history_size:]

            entry = self._per_tool.setdefault(tool_name, {
                "count": 0, "total_ms": 0.0, "errors": 0,
                "min_ms": float("inf"), "max_ms": 0.0,
            })
            entry["count"] += 1
            entry["total_ms"] += elapsed_ms
            entry["min_ms"] = min(entry["min_ms"], elapsed_ms)
            entry["max_ms"] = max(entry["max_ms"], elapsed_ms)
            if not success:
                entry["errors"] += 1

        log.debug("tool_metric: %s elapsed=%.1fms success=%s result_size=%d",
                  tool_name, elapsed_ms, success, result_size)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for logging or display."""
        with self._lock:
            total_calls = sum(e["count"] for e in self._per_tool.values())
            total_errors = sum(e["errors"] for e in self._per_tool.values())
            uptime = time.time() - self._start_time

            per_tool = {}
            for name, e in self._per_tool.items():
                avg_ms = e["total_ms"] / e["count"] if e["count"] else 0
                per_tool[name] = {
                    "count": e["count"],
                    "avg_ms": round(avg_ms, 1),
                    "min_ms": round(e["min_ms"], 1) if e["min_ms"] != float("inf") else 0,
                    "max_ms": round(e["max_ms"], 1),
                    "errors": e["errors"],
                    "error_rate": round(e["errors"] / e["count"] * 100, 1) if e["count"] else 0,
                }

            return {
                "uptime_s": round(uptime, 1),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate_pct": round(total_errors / total_calls * 100, 1) if total_calls else 0,
                "per_tool": per_tool,
            }

    def recent(self, n: int = 20) -> List[dict]:
        """Return last N call records as dicts."""
        with self._lock:
            return [
                {
                    "tool": r.tool_name,
                    "elapsed_ms": round(r.elapsed_ms, 1),
                    "success": r.success,
                    "error": r.error,
                    "result_size": r.result_size,
                }
                for r in self._calls[-n:]
            ]



# Global metrics instance
_global_metrics = ToolMetrics()


def get_metrics() -> ToolMetrics:
    """Return the global ToolMetrics instance."""
    return _global_metrics
