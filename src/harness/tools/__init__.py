"""Tool system — ToolHandlers dispatcher with methods from sub-modules."""
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from rich.console import Console

from ._base import (
    _track_write,
    kill_process_tree,
    sanitize_terminal_output,
    _decode_powershell_clixml,
    _decode_clixml_in_text,
    _detect_log_file_encoding,
    HAS_MCP_SDK,
    log,
)


class ToolHandlers:
    """Thin dispatcher — all tool implementations are in separate modules."""

    # -- Class-level constants ------------------------------------------------

    OUTPUT_SPILL_TOKEN_THRESHOLD = 8000
    _NO_SPILL_TOOLS = frozenset({"read_file", "search_files"})
    OUTPUT_INLINE_PREVIEW_TOKENS = 300
    LARGE_WRITE_FEEDBACK_THRESHOLD = 256 * 1024
    LARGE_WRITE_CHUNK_SIZE = 128 * 1024
    MAX_FULL_READ_LINES = 2000
    _MAX_LIVE_DISPLAY = 10

    def __init__(
        self,
        config,
        console: Console,
        workspace_path: str,
        context,
        duplicate_detector,
        context_manager=None,
        sub_agent_manager=None,
    ):
        self.config = config
        self.console = console
        self.workspace_path = workspace_path
        self.context = context
        self._duplicate_detector = duplicate_detector
        self._context_manager = context_manager
        self.sub_agent_manager = sub_agent_manager

        self._background_procs: Dict[int, dict] = {}
        self._next_bg_id = 1
        self._next_cmd_id = 1

        self._output_dir = os.path.join(workspace_path, ".harness_output")
        os.makedirs(self._output_dir, exist_ok=True)

        self._mcp_sessions: Dict[str, Dict[str, Any]] = {}
        self._mcp_locks: Dict[str, asyncio.Lock] = {}


# -- Attach methods from sub-modules ------------------------------------------

from . import mcp as _mcp
for _name in dir(_mcp):
    if _name in ("cleanup_background_procs", "cleanup_background_procs_async", "list_background_procs"):
        continue  # defined in background.py
    if not _name.startswith("_") or _name == "__init__":
        setattr(ToolHandlers, _name, getattr(_mcp, _name))

from . import file_ops as _fo
for _name in dir(_fo):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_fo, _name))

from . import shell as _sh
for _name in dir(_sh):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_sh, _name))

from . import filesystem as _fs
for _name in dir(_fs):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_fs, _name))

from . import background as _bg
for _name in dir(_bg):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_bg, _name))

from . import image as _img
for _name in dir(_img):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_img, _name))

from . import web as _web
for _name in dir(_web):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_web, _name))

from . import result as _res
for _name in dir(_res):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_res, _name))

from . import cron_tools as _ct
for _name in dir(_ct):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_ct, _name))

from . import subagent as _sa
for _name in dir(_sa):
    if not _name.startswith("__"):
        setattr(ToolHandlers, _name, getattr(_sa, _name))
