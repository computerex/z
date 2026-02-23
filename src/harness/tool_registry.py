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


# ── Tool Definition ──────────────────────────────────────────────

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
    name, parameters (for parser hints), whether its content block
    may contain nested XML (complex_content), and its category.
    """
    name: str
    category: str = "general"         # file, shell, search, context, meta, agent
    complex_content: bool = False     # uses greedy XML matching (write/replace)
    params: List[ToolParam] = field(default_factory=list)
    description: str = ""


# ── The Registry ─────────────────────────────────────────────────

TOOL_DEFS: List[ToolDef] = [
    # --- File operations ---
    ToolDef("read_file",       category="file",
            params=[ToolParam("path", required=True),
                    ToolParam("start_line"), ToolParam("end_line")]),
    ToolDef("write_to_file",   category="file", complex_content=True,
            params=[ToolParam("path", required=True),
                    ToolParam("content", required=True)]),
    ToolDef("replace_in_file", category="file", complex_content=True,
            params=[ToolParam("path", required=True),
                    ToolParam("diff", required=True)]),

    # --- Shell / process ---
    ToolDef("execute_command",          category="shell",
            params=[ToolParam("command", required=True),
                    ToolParam("background")]),
    ToolDef("list_background_processes", category="shell"),
    ToolDef("check_background_process", category="shell",
            params=[ToolParam("id", required=True), ToolParam("lines")]),
    ToolDef("stop_background_process",  category="shell",
            params=[ToolParam("id", required=True)]),

    # --- Search / exploration ---
    ToolDef("list_files",   category="search",
            params=[ToolParam("path", required=True), ToolParam("recursive")]),
    ToolDef("search_files", category="search",
            params=[ToolParam("path", required=True),
                    ToolParam("regex", required=True),
                    ToolParam("file_pattern")]),

    # --- External / vision / web ---
    ToolDef("analyze_image", category="external",
            params=[ToolParam("path", required=True), ToolParam("question")]),
    ToolDef("web_search",    category="external",
            params=[ToolParam("query", required=True), ToolParam("count")]),

    # --- Context management ---
    ToolDef("list_context",        category="context"),
    ToolDef("remove_from_context", category="context",
            params=[ToolParam("id"), ToolParam("source")]),

    # --- Agent meta ---
    ToolDef("manage_todos",     category="agent",
            params=[ToolParam("action", required=True),
                    ToolParam("id"), ToolParam("title"),
                    ToolParam("description"), ToolParam("status"),
                    ToolParam("parent_id"), ToolParam("notes"),
                    ToolParam("context_refs")]),
    ToolDef("set_reasoning_mode", category="agent",
            params=[ToolParam("mode", required=True)]),
    ToolDef("create_plan",        category="agent",
            params=[ToolParam("prompt", required=True)]),
    ToolDef("update_agent_rules", category="agent",
            params=[ToolParam("rule", required=True),
                    ToolParam("category")]),
    ToolDef("introspect",          category="agent",
            params=[ToolParam("focus")]),
]

# Derived lookups (computed once at import time)
TOOL_NAMES: List[str] = [t.name for t in TOOL_DEFS]
TOOL_BY_NAME: Dict[str, ToolDef] = {t.name: t for t in TOOL_DEFS}
COMPLEX_CONTENT_TOOLS: set = {t.name for t in TOOL_DEFS if t.complex_content}
# All parameter names across all tools (for XML suppression)
PARAM_NAMES: set = set()
for tool in TOOL_DEFS:
    PARAM_NAMES.update(p.name for p in tool.params)


def get_tool_names() -> List[str]:
    """Return canonical list of tool names."""
    return TOOL_NAMES


def get_param_names() -> set:
    """Return set of all parameter names across all tools."""
    return PARAM_NAMES


def get_complex_content_tools() -> set:
    """Return tool names that may contain nested XML in their content."""
    return COMPLEX_CONTENT_TOOLS


def get_tool_def(name: str) -> Optional[ToolDef]:
    """Return tool definition by name, or None if unknown."""
    return TOOL_BY_NAME.get(name)


# ── Observability: ToolMetrics ───────────────────────────────────

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

    def to_json(self) -> str:
        """Serialize full metrics to JSON."""
        return json.dumps({"summary": self.summary(), "recent": self.recent()}, indent=2)


# Global metrics instance
_global_metrics = ToolMetrics()


def get_metrics() -> ToolMetrics:
    """Return the global ToolMetrics instance."""
    return _global_metrics
