"""Context & memory — compaction, token management, CLAUDE.md, memory system, replay, cross-encoder."""
from .context_management import (
    DuplicateDetector,
    estimate_messages_tokens,
    estimate_tokens,
    get_model_limits,
    get_remote_model_data,
    sanitize_tool_call_groups,
    truncate_conversation,
    truncate_file_content,
    truncate_output,
)
from .context_replay import run_replay
from .smart_context import CompactionTrace, SmartContextManager
from .claude_md import parse_frontmatter, parse_memory_file_content
from .memdir import find_relevant_memories, get_memory_dir, load_memory_prompt
from .cross_encoder import MemoryCandidate, rank_memories
