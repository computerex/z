"""Context management utilities - Cline-style truncation and optimization."""

from typing import List, Tuple, Optional
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate token count. Rough approximation: ~4 chars per token."""
    return len(text) // 4


def estimate_messages_tokens(messages: List[dict]) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        total += estimate_tokens(content) + 4  # +4 for message overhead
    return total


# Model context windows and safe limits (leave room for response)
MODEL_LIMITS = {
    "glm-4.7": (128_000, 98_000),      # (context_window, max_allowed)
    "glm-4-plus": (128_000, 98_000),
    "deepseek-chat": (64_000, 37_000),
    "gpt-4": (128_000, 98_000),
    "claude-3": (200_000, 160_000),
}

DEFAULT_LIMIT = (128_000, 98_000)


def get_model_limits(model: str) -> Tuple[int, int]:
    """Get (context_window, max_allowed) for a model."""
    for key, limits in MODEL_LIMITS.items():
        if key in model.lower():
            return limits
    return DEFAULT_LIMIT


@dataclass
class TruncationResult:
    """Result of a truncation operation."""
    messages: List
    removed_count: int
    notice: Optional[str] = None


def truncate_conversation(
    messages: List,
    strategy: str = "half",  # "none", "lastTwo", "half", "quarter"
) -> TruncationResult:
    """
    Truncate conversation history Cline-style.
    
    Always preserves:
    - Index 0: System prompt
    - Index 1-2: First user-assistant pair (original task)
    - Recent messages based on strategy
    
    Strategies:
    - "none": Keep only first pair
    - "lastTwo": Keep first pair + last user-assistant pair
    - "half": Keep first pair + 50% of remaining
    - "quarter": Keep first pair + 25% of remaining
    """
    if len(messages) <= 4:
        return TruncationResult(messages=messages, removed_count=0)
    
    # Preserve system (0), first user (1), first assistant (2)
    preserved_start = messages[:3]
    remaining = messages[3:]
    
    if strategy == "none":
        # Keep only first pair
        keep_count = 0
    elif strategy == "lastTwo":
        # Keep last user-assistant pair
        keep_count = min(2, len(remaining))
    elif strategy == "half":
        keep_count = len(remaining) // 2
    elif strategy == "quarter":
        keep_count = len(remaining) // 4
    else:
        keep_count = len(remaining) // 2
    
    # Ensure we keep pairs (user + assistant)
    keep_count = max(0, keep_count - (keep_count % 2))
    
    if keep_count >= len(remaining):
        return TruncationResult(messages=messages, removed_count=0)
    
    kept_messages = remaining[-keep_count:] if keep_count > 0 else []
    removed_count = len(remaining) - keep_count
    
    # Add truncation notice
    notice = (
        f"[CONTEXT NOTICE: {removed_count} messages from conversation history have been removed "
        f"to maintain optimal context window length. The original task and recent context are preserved.]"
    )
    
    # Insert notice as a user message after first pair
    notice_msg = type(messages[0])(role="user", content=notice)
    
    result = preserved_start + [notice_msg] + kept_messages
    
    return TruncationResult(
        messages=result,
        removed_count=removed_count,
        notice=notice
    )


def truncate_output(
    text: str,
    max_lines: int = 200,
    keep_start: int = 50,
    keep_end: int = 50,
) -> str:
    """
    Truncate long output, keeping start and end.
    Like Cline's terminal output handling.
    """
    lines = text.splitlines()
    
    if len(lines) <= max_lines:
        return text
    
    start = lines[:keep_start]
    end = lines[-keep_end:]
    removed = len(lines) - keep_start - keep_end
    
    return "\n".join(start) + f"\n\n... ({removed} lines truncated) ...\n\n" + "\n".join(end)


def truncate_file_content(
    content: str,
    max_bytes: int = 400_000,  # ~100k tokens
) -> str:
    """Truncate file content if too large."""
    if len(content) <= max_bytes:
        return content
    
    # Keep start, add truncation notice
    truncated = content[:max_bytes]
    remaining = len(content) - max_bytes
    
    return truncated + f"\n\n... ({remaining:,} bytes truncated) ..."


class DuplicateDetector:
    """Detects duplicate file reads in conversation."""
    
    DUPLICATE_NOTICE = (
        "[NOTE: This file was read earlier in the conversation. "
        "Refer to the most recent read for current contents.]"
    )
    
    def __init__(self):
        self._read_files: dict = {}  # path -> message_index
    
    def record_read(self, path: str, message_index: int):
        """Record that a file was read at a given message index."""
        self._read_files[path] = message_index
    
    def was_read_before(self, path: str) -> Optional[int]:
        """Check if file was read before. Returns previous index or None."""
        return self._read_files.get(path)
    
    def clear(self):
        self._read_files.clear()
    
    @classmethod
    def replace_old_reads(cls, messages: List, path: str, new_index: int) -> int:
        """
        Replace older reads of the same file with a notice.
        Returns count of replacements made.
        """
        count = 0
        for i, msg in enumerate(messages):
            if i >= new_index:
                break
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            
            # Check if this is a tool result with this file path
            if f"[read_file result]" in content and path in content:
                # Replace with notice
                if hasattr(msg, 'content'):
                    msg.content = f"[read_file result]\n{path}\n{cls.DUPLICATE_NOTICE}"
                else:
                    msg['content'] = f"[read_file result]\n{path}\n{cls.DUPLICATE_NOTICE}"
                count += 1
        
        return count
