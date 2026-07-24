"""Smart context compaction — keeps important messages, drops duplicates, preserves structure."""

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING

from .context_management import estimate_tokens, estimate_messages_tokens, truncate_conversation
from ..logger import get_logger


if TYPE_CHECKING:
    from ..todo_manager import TodoManager

log = get_logger("smart_context")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Message indices that must NEVER be removed or altered.
# 0 = system prompt, 1 = first user message, 2 = first assistant response.
PROTECTED_INDICES = frozenset({0, 1, 2})

# Unicode marker identifying already-compacted messages.
COMPACT_MARKER = "\u25c7"  # ◇


# ---------------------------------------------------------------------------
# Compaction trace — breadcrumb left when content is compacted
# ---------------------------------------------------------------------------


@dataclass
class CompactionTrace:
    """Breadcrumb left when content is compacted, enabling recovery."""

    original_type: str
    source: str
    summary: str
    tokens_freed: int = 0
    compacted_at: float = field(default_factory=time.time)

    def format_notice(self) -> str:
        """Format as an inline notice the agent can act on."""
        if self.original_type == "file_read":
            return (
                f"[{COMPACT_MARKER} File '{self.source}' was read here "
                f"({self.tokens_freed} tok freed). Re-read if needed.]"
            )
        elif self.original_type == "duplicate_read":
            return (
                f"[{COMPACT_MARKER} Duplicate read of '{self.source}' — "
                f"latest version exists later in conversation.]"
            )
        elif self.original_type == "command_output":
            return (
                f"[{COMPACT_MARKER} Output of '{self.source}' was here "
                f"({self.tokens_freed} tok freed). Re-run if needed.]"
            )
        elif self.original_type == "search_result":
            return (
                f"[{COMPACT_MARKER} Search results were here "
                f"({self.tokens_freed} tok freed). Re-search if needed.]"
            )
        elif self.original_type == "assistant":
            return (
                f"[{COMPACT_MARKER} Previous analysis: {self.summary} "
                f"({self.tokens_freed} tok freed)]"
            )
        else:
            return (
                f"[{COMPACT_MARKER} {self.original_type}: {self.summary} "
                f"({self.tokens_freed} tok freed)]"
            )


# ---------------------------------------------------------------------------
# Tool result storage — for retrieving compacted tool results
# ---------------------------------------------------------------------------


@dataclass
class StoredToolResult:
    """A stored tool result that can be retrieved after compaction."""

    result_id: str
    tool_name: str
    original_content: str
    timestamp: float
    message_index: int
    tokens: int


class ToolResultStorage:
    """Storage for tool results that have been compacted.

    Allows the agent to retrieve full tool results after they've been
    abbreviated in the conversation context.
    """

    # Maximum total bytes to store (100MB)
    MAX_TOTAL_BYTES = 100 * 1024 * 1024

    # Maximum age for results (1 hour)
    MAX_AGE_SECONDS = 3600

    def __init__(self, max_results: int = 100):
        self._results: Dict[str, StoredToolResult] = {}
        self._max_results = max_results
        self._access_order: List[str] = []  # For LRU eviction
        self._total_bytes = 0  # Track total bytes stored
        self._counter = 0  # For unique result IDs

    def _evict_if_needed(self) -> None:
        """Evict old results if we exceed limits."""
        # Evict by count
        while len(self._results) > self._max_results:
            if not self._access_order:
                break
            rid = self._access_order.pop(0)
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

        # Evict by size
        while self._total_bytes > self.MAX_TOTAL_BYTES and self._access_order:
            rid = self._access_order.pop(0)
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

    def _cleanup_old_results(self) -> int:
        """Clean up results older than MAX_AGE_SECONDS."""
        now = time.time()
        to_remove = [
            rid
            for rid, result in self._results.items()
            if now - result.timestamp > self.MAX_AGE_SECONDS
        ]
        if not to_remove:
            return 0

        to_remove_set = set(to_remove)

        self._access_order = [
            rid for rid in self._access_order if rid not in to_remove_set
        ]

        for rid in to_remove:
            result = self._results.pop(rid, None)
            if result:
                self._total_bytes -= len(
                    result.original_content.encode("utf-8", errors="replace")
                )

        return len(to_remove)


    def get_result(self, result_id: str) -> Optional[StoredToolResult]:
        """Retrieve a stored tool result by ID."""
        if not result_id or not isinstance(result_id, str):
            return None
        if not re.match(r"^res_[a-f0-9]{8}_\d+$", result_id):
            return None

        result = self._results.get(result_id)
        if result:
            if result_id in self._access_order:
                self._access_order.remove(result_id)
            self._access_order.append(result_id)
        return result



# ---------------------------------------------------------------------------
# SmartContextManager — simplified, no semantic scoring
# ---------------------------------------------------------------------------


class SmartContextManager:
    """Manages context using simple naive truncation strategies.

    No semantic scoring, no embedding model — just Cline-style
    half/quarter/lastTwo truncation.  User triggers compaction via /compact.
    """

    def __init__(self, todo_manager: "TodoManager"):
        self.todo_manager = todo_manager
        self.compaction_traces: List[CompactionTrace] = []

        # Storage for tool results that have been compacted
        self.result_storage = ToolResultStorage(max_results=100)

    def semantic_maintenance_tick(
        self,
        messages: List[Any],
        max_tokens: int,
        current_tokens: Optional[int] = None,
    ) -> Tuple[List[Any], int, str]:
        """No-op: semantic maintenance is disabled.

        Use /compact to trigger truncation manually.
        """
        return messages, 0, ""

    def build_context_recovery_notice(self) -> str:
        """Build a reorientation notice after context compaction/truncation."""
        if not self.compaction_traces:
            return ""

        recent = self.compaction_traces[-10:]

        # Group by type
        by_type: Dict[str, List[CompactionTrace]] = {}
        for t in recent:
            by_type.setdefault(t.original_type, []).append(t)

        lines = [
            "[CONTEXT COMPACTION NOTICE]",
            f"Some content was removed to stay within context limits. "
            f"Total compactions this session: {len(self.compaction_traces)}",
            "",
        ]

        for typ, traces in by_type.items():
            sources = list(dict.fromkeys(t.source for t in traces))[:5]
            lines.append(f"  {typ}: {', '.join(sources)}")

        todo_state = self.todo_manager.format_list(include_completed=False)
        if todo_state and "empty" not in todo_state.lower():
            lines.append("")
            lines.append(todo_state)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "compaction_traces": [
                {
                    "original_type": t.original_type,
                    "source": t.source,
                    "summary": t.summary,
                    "tokens_freed": t.tokens_freed,
                    "compacted_at": t.compacted_at,
                }
                for t in self.compaction_traces
            ],
        }

    def load_dict(self, data: dict):
        raw = data.get("compaction_traces", data.get("eviction_traces", []))
        self.compaction_traces = []
        for t in raw:
            tokens = t.get("tokens_freed", t.get("token_count", 0))
            self.compaction_traces.append(
                CompactionTrace(
                    original_type=t["original_type"],
                    source=t["source"],
                    summary=t["summary"],
                    tokens_freed=tokens,
                    compacted_at=t.get("compacted_at", t.get("evicted_at", 0)),
                )
            )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(content: str, role: str) -> Tuple[str, str]:
        """Classify a message into ``(type, source_label)``."""

        if content.startswith("[GUIDANCE_NUDGE"):
            m = re.search(r"type=([a-zA-Z0-9._-]+)", content)
            nudge_type = m.group(1) if m else "guidance"
            return "guidance_nudge", nudge_type

        if role == "user":
            lower = content.lower()
            if "without using introspect" in lower:
                return "guidance_nudge", "introspect"
            if "consider using the introspect tool" in lower:
                return "guidance_nudge", "introspect-tip"
            if "result contains errors" in lower:
                return "guidance_nudge", "error-analysis"

        tool_match = re.match(r"\[(\w+) result(?:\s*-[^\]]+)?\]", content)
        if tool_match:
            tool_name = tool_match.group(1)

            if tool_name == "read_file":
                lines = content.split("\n", 2)
                path = lines[1].strip() if len(lines) > 1 else "unknown"
                return "file_read", path

            if tool_name == "execute_command":
                cmd_match = re.search(r"\$\s*(.+?)(?:\n|$)", content)
                cmd = cmd_match.group(1).strip()[:80] if cmd_match else "command"
                return "command_output", cmd

            if tool_name in ("search_files", "grep_search"):
                return "search_result", "search"

            if tool_name == "manage_todos":
                return "todo_result", "todos"

            if tool_name in ("list_context", "remove_from_context"):
                return "context_result", "context"

            if tool_name == "introspect":
                return "introspect_result", "introspect"

            return "other_tool_result", tool_name

        if content.startswith("[Deep analysis complete"):
            return "introspect_result", "introspect"

        if role == "assistant":
            return "assistant_analysis", "analysis"

        return "other", ""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_content(msg: Any) -> str:
    """Extract string content from a message (object or dict)."""
    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
    return content if isinstance(content, str) else ""


def _set_content(msg: Any, content: str) -> None:
    """Set string content on a message (object or dict)."""
    if hasattr(msg, "content"):
        msg.content = content
    else:
        msg["content"] = content



def _normalize_path(path: str) -> str:
    """Normalize a file path for dedup comparison."""
    p = path.replace("\\", "/").strip()
    if os.name == "nt":
        p = p.lower()
    p = p.rstrip("/")
    return p


def _message_fingerprint(role: str, content: str) -> str:
    """Stable fingerprint for message-level metadata tracking."""
    base = f"{role}\n{content[:4000]}"
    return hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()
