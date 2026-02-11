"""Agentic coding harness optimized for Z.AI GLM-4.7."""

from .cline_agent import ClineAgent
from .streaming_agent import StreamingAgent
from .streaming_client import StreamingJSONClient
from .config import Config
from .cost_tracker import CostTracker, get_global_tracker
from .prompts import get_system_prompt
from .todo_manager import TodoManager, TodoItem, TodoStatus
from .smart_context import SmartContextManager, SemanticScorer, CompactionTrace, PROTECTED_INDICES, COMPACT_MARKER

__version__ = "0.1.0"
__all__ = [
    "ClineAgent",
    "StreamingAgent", 
    "StreamingJSONClient", 
    "Config", 
    "CostTracker", 
    "get_global_tracker",
    "get_system_prompt",
    "TodoManager",
    "TodoItem",
    "TodoStatus",
    "SmartContextManager",
    "SemanticScorer",
    "CompactionTrace",
    "PROTECTED_INDICES",
    "COMPACT_MARKER",
]
