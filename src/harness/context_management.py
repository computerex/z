"""Context management utilities - Cline-style truncation and optimization."""

from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass


def estimate_tokens(text: Union[str, List, Any]) -> int:
    """Estimate token count. Rough approximation: ~4 chars per token.
    
    Handles both string content and list content (for vision messages).
    """
    if isinstance(text, str):
        return len(text) // 4
    elif isinstance(text, list):
        # Vision format: list of content blocks
        total = 0
        for block in text:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += len(block.get("text", "")) // 4
                elif block.get("type") == "image_url":
                    total += 1000  # Rough estimate for image tokens
                else:
                    total += 50  # Unknown block type
            else:
                total += 50
        return total
    else:
        return 50  # Fallback for unknown types


def estimate_messages_tokens(messages: List[dict]) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        total += estimate_tokens(content) + 4  # +4 for message overhead
    return total


# Model context windows and safe limits (leave room for response)
# Format: (context_window, max_allowed_input)
# max_allowed_input leaves headroom for the model's output tokens.
# These are fetched from https://models.dev/api.json at first use.
# This minimal hardcoded set is used as fallback if the remote fetch fails.
_FALLBACK_LIMITS = {
    "glm-4.7": (200_000, 128_000),
    "glm-4-plus": (128_000, 98_000),
    "gpt-4": (128_000, 98_000),
    "claude": (200_000, 160_000),
    "MiniMax-M2": (1_000_000, 200_000),
}

DEFAULT_LIMIT = (128_000, 98_000)

_remote_limits: dict | None = None
_remote_providers: dict | None = None  # raw provider data keyed by domain
_remote_models_raw: dict | None = None  # raw model metadata keyed by model_id
_remote_load_attempted = False


def get_remote_model_data(model: str, api_url: str = "") -> Optional[dict]:
    """Return raw model metadata from models.dev/api.json if available.

    Prefers a provider-specific match when ``api_url`` is provided, then
    falls back to a generic model-id match.
    """
    global _remote_models_raw
    _load_remote_limits()
    if not _remote_models_raw:
        return None

    model_lower = model.lower()

    # Provider-specific match: use the domain to narrow the provider first.
    if api_url:
        domain = _extract_domain(api_url)
        if domain and _remote_providers:
            provider_models = _remote_providers.get(domain)
            if provider_models:
                for key in sorted(provider_models, key=len, reverse=True):
                    if key in model_lower:
                        return _remote_models_raw.get(key)

    # Generic match across all models
    if isinstance(_remote_models_raw, dict):
        for key in sorted(_remote_models_raw.keys(), key=len, reverse=True):
            if key in model_lower:
                data = _remote_models_raw[key]
                return data if isinstance(data, dict) else None

    return None

def _extract_domain(url: str) -> str:
    """Extract the hostname from a URL for provider matching."""
    if not url:
        return ""
    url = url.strip()
    # Strip protocol
    if "://" in url:
        url = url.split("://", 1)[1]
    # Strip path/port
    url = url.split("/")[0]
    # Strip port
    if ":" in url:
        url = url.split(":")[0]
    return url.lower()


def _load_remote_limits() -> dict:
    """Fetch model limits from models.dev API. Cached after first call."""
    global _remote_limits, _remote_load_attempted, _remote_providers, _remote_models_raw
    if _remote_load_attempted:
        return _remote_limits or {}

    _remote_load_attempted = True
    import logging as _logging
    _log = _logging.getLogger("harness.context_management")
    try:
        import httpx
        resp = httpx.get(
            "https://models.dev/api.json",
            timeout=10,
            headers={"User-Agent": "Harness/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _log.warning("Failed to fetch model limits from models.dev: %s", exc)
        _remote_limits = {}
        _remote_models_raw = {}
        return _remote_limits

    # Build two lookups:
    #   1) flat limits dict: model_id -> (context, max_input)
    #   2) provider index: domain -> { model_id -> (context, max_input) }
    limits: dict = {}
    provider_by_domain: dict = {}

    for provider_id, provider_data in data.items():
        models = provider_data.get("models", {})
        provider_api = provider_data.get("api", "")
        domain = _extract_domain(provider_api)

        # Build per-domain provider model lookup
        provider_models: dict = {}
        for model_id, model in models.items():
            ctx = model.get("limit", {}).get("context")
            out = model.get("limit", {}).get("output")
            if ctx and out:
                # Reserve headroom for output: 20% of context, capped at 128K.
                # This is more consistent than subtracting the full output limit,
                # which can be unrealistically large (e.g. 384K for DeepSeek V4).
                headroom = min(max(ctx // 5, 8_000), 128_000)
                max_input = max(2_000, ctx - headroom)
                limits[model_id] = (ctx, max_input)
                if "/" in model_id:
                    short = model_id.rsplit("/", 1)[1]
                    if short not in limits:
                        limits[short] = (ctx, max_input)
                provider_models[model_id] = (ctx, max_input)
                if "/" in model_id:
                    short = model_id.rsplit("/", 1)[1]
                    if short not in provider_models:
                        provider_models[short] = (ctx, max_input)

        if domain:
            provider_by_domain[domain] = provider_models

    _remote_limits = limits
    _remote_providers = provider_by_domain
    # Also keep a flat registry of raw model metadata for capability lookups.
    _remote_models_raw = {}
    for provider_data in data.values():
        if not isinstance(provider_data, dict):
            continue
        for model_id, model in provider_data.get("models", {}).items():
            if isinstance(model, dict):
                _remote_models_raw[model_id.lower()] = model
                if "/" in model_id:
                    _remote_models_raw[model_id.rsplit("/", 1)[1].lower()] = model

    _log.info("Loaded %d model limits from models.dev", len(limits))
    return limits


def get_model_limits(
    model: str,
    api_url: str = "",
) -> Tuple[int, int]:
    """Get (context_window, max_allowed) for a model.

    Uses remote models.dev API on first call, cached thereafter.
    If ``api_url`` is provided, prefers the provider whose API URL
    domain matches (e.g. ``api.deepseek.com`` → deepseek models).
    Falls back to hardcoded limits, then DEFAULT_LIMIT.
    """
    model_lower = model.lower()

    remote = _load_remote_limits()

    # Provider-specific match: if we know the API URL, check that
    # provider's models first for a more accurate limit.
    if api_url:
        domain = _extract_domain(api_url)
        if domain and _remote_providers:
            provider_models = _remote_providers.get(domain)
            if provider_models:
                for key in sorted(provider_models, key=len, reverse=True):
                    if key in model_lower:
                        return provider_models[key]

    # Generic match across all models — sort by key length descending
    # so more specific matches win over accidental substrings.
    for key in sorted(remote, key=len, reverse=True):
        if key in model_lower:
            return remote[key]

    # Fall back to hardcoded limits
    for key, limits in _FALLBACK_LIMITS.items():
        if key in model_lower:
            return limits

    return DEFAULT_LIMIT


@dataclass
class TruncationResult:
    """Result of a truncation operation."""
    messages: List
    removed_count: int
    notice: Optional[str] = None


def _msg_role(msg) -> str:
    """Get the role of a message, handling both objects and dicts."""
    if hasattr(msg, 'role'):
        return msg.role
    return msg.get('role', '')


def _msg_tool_calls(msg):
    """Get tool_calls from a message, handling both objects and dicts."""
    if hasattr(msg, 'tool_calls'):
        return msg.tool_calls or []
    return msg.get('tool_calls', []) or []


def _is_tool(msg) -> bool:
    """Check if a message is a tool result message."""
    return _msg_role(msg) == "tool"


def _has_tool_calls(msg) -> bool:
    """Check if an assistant message has tool_calls entries."""
    if _msg_role(msg) != "assistant":
        return False
    return bool(_msg_tool_calls(msg))


def _tc_count(msg) -> int:
    """Count the number of tool_calls in an assistant message."""
    return len(_msg_tool_calls(msg))


def _strip_tool_calls(msg) -> None:
    """Remove tool_calls from an assistant message (mutates in place)."""
    if hasattr(msg, 'tool_calls'):
        msg.tool_calls = None
    else:
        msg.pop('tool_calls', None)


def sanitize_tool_call_groups(messages: List, logger=None) -> int:
    """Walk messages and fix orphaned tool_calls / tool-result groups.

    Ensures:
      * Every ``assistant`` message with ``tool_calls`` has enough
        following ``role="tool"`` messages to satisfy each call.
      * Every ``role="tool"`` message has a preceding
        ``assistant(tool_calls)`` that claimed it.

    Fixes by:
      * Stripping ``tool_calls`` from orphaned assistant messages
        (sets them to ``None`` so the provider doesn't reject them).
      * Removing orphaned ``role="tool"`` messages (they have no
        matching ``assistant(tool_calls)`` parent).

    Returns the number of messages modified or removed.
    """
    if not messages:
        return 0

    modified = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = _msg_role(msg)

        if role == "assistant" and _has_tool_calls(msg):
            need = _tc_count(msg)
            have = 0
            # Count consecutive tool messages following this assistant
            j = i + 1
            while j < len(messages) and _is_tool(messages[j]):
                have += 1
                j += 1

            if have < need:
                # Orphaned — strip tool_calls so the API doesn't 400
                _strip_tool_calls(msg)
                modified += 1
                if logger:
                    logger.debug(
                        "sanitize: stripped tool_calls from msg[%d] "
                        "(needed=%d, have=%d)",
                        i, need, have,
                    )
            # If have >= need, the group is fine as-is.
            # (Extra tool messages will be handled as orphans below.)
            i = j  # skip past the tool messages we just counted
            continue

        elif role == "tool":
            # Orphaned tool message with no preceding assistant(tool_calls).
            # Check: is there an assistant(tool_calls) somewhere before this?
            found_parent = False
            for k in range(i - 1, -1, -1):
                if _has_tool_calls(messages[k]):
                    # Need to verify this assistant actually has enough
                    # tool results including this one.
                    need = _tc_count(messages[k])
                    # Count tool messages between k+1 and after us
                    parent_tools = 0
                    for t in range(k + 1, len(messages)):
                        if _is_tool(messages[t]):
                            parent_tools += 1
                        else:
                            break
                    if parent_tools >= need:
                        # This tool belongs to a valid group — keep it
                        found_parent = True
                    break
                elif _msg_role(messages[k]) == "assistant":
                    # Another assistant without tool_calls broke the chain
                    break
                elif _msg_role(messages[k]) == "user":
                    # User message broke the chain
                    break

            if not found_parent:
                # Remove orphaned tool message
                if logger:
                    logger.debug(
                        "sanitize: removing orphaned tool msg[%d]",
                        i,
                    )
                messages.pop(i)
                modified += 1
                continue  # don't increment i

        i += 1

    return modified


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
    
    Integrity guarantee:
    ``assistant`` messages with ``tool_calls`` and their corresponding
    ``role="tool"`` results are kept as atomic groups.  The truncation
    boundary is adjusted so that:
    
    * No ``role="tool"`` message survives without its preceding
      ``assistant(tool_calls)``.
    * No ``assistant(tool_calls)`` message survives without all of its
      matching ``role="tool"`` results.
    
    If the required ``role="tool"`` messages were removed, they are
    pulled back from the removed region if possible; otherwise the
    orphaned ``assistant(tool_calls)`` is also dropped.
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
    
    # ──────────────────────────────────────────────────────────────────
    # Fix: Preserve tool_calls/tool message group integrity
    #
    # The naive "even pair" assumption breaks when tool_calls are in
    # play because the message sequence can be:
    #
    #   assistant(tool_calls=[a,b]) → tool(a) → tool(b) → assistant(...)
    #
    # A truncation boundary through the middle of such a group produces
    # orphaned tool or assistant(tool_calls) messages, which cause
    # providers like DeepSeek to reject the request with 400.
    #
    # The fix combines two steps:
    #   1. Find the first non-tool message in kept_messages.
    #   2. If it has tool_calls, verify all its tool results are present.
    #      Tools at the start of kept_messages may belong to this
    #      assistant — they are NOT blindly dropped.
    # ──────────────────────────────────────────────────────────────────
    
    # Locate the first non-tool message in the kept tail
    first_non_tool = 0
    while first_non_tool < len(kept_messages) and _is_tool(kept_messages[first_non_tool]):
        first_non_tool += 1
    
    if first_non_tool < len(kept_messages) and _has_tool_calls(kept_messages[first_non_tool]):
        # This assistant has tool_calls.  Count how many tool results
        # are available (both before and after it in kept_messages).
        need = _tc_count(kept_messages[first_non_tool])
        tools_before = first_non_tool  # tools at the start of kept
        tools_after = 0
        for m in kept_messages[first_non_tool + 1:]:
            if _is_tool(m):
                tools_after += 1
            else:
                break
        total_have = tools_before + tools_after
        missing = need - total_have
        
        if missing > 0:
            # Pull missing tool messages from the removed region
            # (messages immediately before the truncation boundary).
            removed_idx = len(remaining) - len(kept_messages)
            predecessor_slice = remaining[:removed_idx]
            pulled = []
            for m in reversed(predecessor_slice):
                if _is_tool(m) and len(pulled) < missing:
                    pulled.insert(0, m)
                else:
                    break
            
            if len(pulled) == missing:
                # Success — prepend pulled tools to kept_messages
                kept_messages = pulled + kept_messages
            else:
                # Can't satisfy — drop the entire orphaned group
                # (assistant + all its tool results) to prevent API 400.
                drop_end = first_non_tool + 1 + tools_after
                kept_messages = kept_messages[drop_end:]
        elif missing < 0:
            # More tools than needed — drop excess from the front
            excess = -missing
            kept_messages = kept_messages[excess:]
        # else missing == 0: perfect match, keep as-is
    elif first_non_tool > 0:
        # Tools at the start, but the first non-tool message does NOT
        # have tool_calls — these tools are truly orphaned. Drop them.
        kept_messages = kept_messages[first_non_tool:]
    
    # ──────────────────────────────────────────────────────────────────
    
    removed_count = len(remaining) - len(kept_messages)
    
    # Add truncation notice
    notice = (
        f"[CONTEXT NOTICE: {removed_count} messages from conversation history have been removed "
        f"to maintain optimal context window length. The original task and recent context are preserved.]"
    )
    
    # Insert notice as a user message after first pair
    notice_msg = type(messages[0])(role="user", content=notice)
    
    # ── Final safety net: verify ALL tool_calls groups ──────────────
    # The group-integrity check above only validates the FIRST
    # tool_calls group in kept_messages.  Additional groups and the
    # preserved-start assistant (index 2) are unchecked.
    #
    # Scan from left to right stripping any orphaned assistant(tool_calls)
    # and removing orphaned role="tool" messages.
    if _has_tool_calls(preserved_start[-1]):
        _strip_tool_calls(preserved_start[-1])
    
    _i = 0
    while _i < len(kept_messages):
        if _has_tool_calls(kept_messages[_i]):
            _need = _tc_count(kept_messages[_i])
            _have = 0
            _j = _i + 1
            while _j < len(kept_messages) and _is_tool(kept_messages[_j]):
                _have += 1
                _j += 1
            if _have < _need:
                _strip_tool_calls(kept_messages[_i])
            _i = _j  # skip past tool messages
        else:
            _i += 1
    
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
    max_bytes: int = 120_000,  # ~30k tokens
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
    
    # Protected message indices that must never be modified
    PROTECTED_INDICES = {0, 1, 2}  # system prompt, first user-assistant pair

