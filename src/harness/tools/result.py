"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logger import get_logger

log = get_logger("tools")

async def retrieve_tool_result(self, params: Dict[str, str]) -> str:
    """Retrieve the full content of a previously compacted tool result.

    When tool results are compacted to save context space, they're stored
    with a unique ID. Use this tool to retrieve the full result when needed.

    Args:
        result_id: The ID of the stored result (e.g., res_abc123_456)

    Returns:
        The full tool result content, or an error message if not found
    """
    result_id = params.get("result_id", "").strip()
    if not result_id:
        return "Error: result_id is required. Example: res_abc123_456"

    # Check if context_manager is available
    if not self._context_manager:
        return "Error: Context manager not available for result retrieval"

    # Retrieve the stored result
    stored = self._context_manager.result_storage.get_result(result_id)
    if not stored:
        return (
            f"Error: Result {result_id} not found. "
            f"It may have been evicted due to age or memory limits."
        )

    # Format the result with metadata
    age_seconds = time.time() - stored.timestamp
    age_str = f"{age_seconds:.0f}s" if age_seconds < 60 else f"{age_seconds/60:.0f}m"

    result = (
        f"[Retrieved tool result: {stored.tool_name}]\n"
        f"Result ID: {result_id}\n"
        f"Age: {age_str} ago\n"
        f"Size: {stored.tokens:,} tokens (~{len(stored.original_content):,} chars)\n"
        f"{'='*60}\n"
        f"{stored.original_content}"
    )

    log.info("retrieve_tool_result: result_id=%s tool=%s tokens=%d age=%s",
             result_id, stored.tool_name, stored.tokens, age_str)

    return result

