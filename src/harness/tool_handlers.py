"""Tool implementations — delegates to tools/ package."""
from .tools import ToolHandlers
from .tools._base import (
    _track_write,
    kill_process_tree,
    sanitize_terminal_output,
    _decode_powershell_clixml,
    _decode_clixml_in_text,
    _detect_log_file_encoding,
    HAS_MCP_SDK,
    log,
)
from .tools._fuzzy import (
    normalize_trailing,
    normalize_unicode,
    unescape_text,
    strip_indent,
    block_anchor_match,
    find_best_fuzzy_match,
    build_diagnostic,
)
