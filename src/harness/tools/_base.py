"""Base helpers and imports for tool implementations."""

import asyncio
import base64
import html
import json
import os
import platform
import re
import signal
import time
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console
import psutil

import logging
from ..context import truncate_file_content, truncate_output
from ..logger import get_logger, log_exception, truncate as log_truncate
from ..streaming_client import desanitize_think_tokens

try:
    from mcp import ClientSession, StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore
    from mcp.client.sse import sse_client  # type: ignore
    HAS_MCP_SDK = True
except Exception:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamablehttp_client = None
    sse_client = None
    HAS_MCP_SDK = False

log = get_logger("tools")


# ── Output protocol file tracking ──────────────────────────────────────

def _track_write(path: Path) -> None:
    """Track a file write for the output protocol (--json mode)."""
    try:
        from ..output_protocol import track_file_written, emit_progress
        track_file_written(str(path))
        emit_progress("writing_file", file=str(path), action="write")
    except Exception:
        pass


def kill_process_tree(pid: int, timeout: float = 3.0) -> None:
    """Kill a process and all its descendants, cross-platform.
    
    Uses psutil to walk the process tree and kill children first,
    then the parent. Works on Windows, Linux, and macOS.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    
    # Collect all children recursively before killing anything
    children = []
    try:
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    # Kill children first (leaf-to-root order)
    for child in reversed(children):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Kill the parent
    try:
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    # Wait for all to die
    _, alive = psutil.wait_procs(children + [parent], timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# Regex to strip ANSI escape sequences that could trigger terminal responses
# Matches: CSI sequences (\x1b[...), OSC sequences (\x1b]...), and other escape sequences
_ANSI_ESCAPE_RE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')

# Control chars that should not be printed to terminal (keeps tab, newline, carriage return)
_DANGEROUS_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]')


def sanitize_terminal_output(text: str) -> str:
    """Strip ANSI escape sequences and dangerous control characters from text.
    
    This prevents binary/garbage output from commands from being interpreted
    by the terminal as query sequences (e.g. \x1b[6n Device Status Report),
    which would cause the terminal to echo response bytes back into the console
    input buffer. The keyboard monitor (msvcrt.getch on Windows) would then
    read those bytes and misinterpret them as user keystrokes (Escape = interrupt).
    """
    text = _ANSI_ESCAPE_RE.sub('', text)
    text = _DANGEROUS_CTRL_CHARS.sub('\ufffd', text)
    return text


_PS_CLIXML_STR_RE = re.compile(r'<S(?:\s+S="[^"]+")?>(.*?)</S>', re.DOTALL)
_PS_CLIXML_HEX_ESCAPE_RE = re.compile(r'_x([0-9A-Fa-f]{4})_')
# Match an entire CLIXML block: #< CLIXML\n<Objs ...>...</Objs>
_PS_CLIXML_BLOCK_RE = re.compile(
    r'#< CLIXML\s*\n\s*<Objs[^>]*>.*?</Objs>',
    re.DOTALL,
)


def _decode_powershell_clixml(text: str) -> str:
    """Best-effort decode of PowerShell CLIXML error/progress output to plain text.

    Handles both single-line fragments and multi-line CLIXML blocks.
    """
    if not text:
        return text
    t = text
    if "#< CLIXML" in t:
        t = t.replace("#< CLIXML", "")
    if "http://schemas.microsoft.com/powershell/2004/04" not in t and "<Objs" not in t:
        return t.strip("\r\n")

    parts = _PS_CLIXML_STR_RE.findall(t)
    if not parts:
        return t.strip("\r\n")

    decoded = "".join(parts)
    decoded = html.unescape(decoded)
    decoded = _PS_CLIXML_HEX_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), decoded)
    return decoded.strip("\r\n")


def _decode_clixml_in_text(text: str) -> str:
    """Replace all CLIXML blocks in *text* with their decoded plain-text form.

    Non-CLIXML content passes through unchanged.  This is the main entry point
    for cleaning up raw log files that may contain a mix of normal output and
    CLIXML error/progress fragments.
    """
    if not text or "CLIXML" not in text:
        return text
    return _PS_CLIXML_BLOCK_RE.sub(
        lambda m: _decode_powershell_clixml(m.group(0)),
        text,
    )


def _detect_log_file_encoding(log_path: str) -> str:
    """Detect the encoding of a log file, defaulting to utf-8.

    Checks for UTF-16LE BOM (0xFF 0xFE) or null byte patterns that indicate
    UTF-16LE encoding (common in PowerShell output on Windows).

    Returns:
        "utf-16-le" if UTF-16LE is detected, "utf-8" otherwise.
    """
    try:
        with open(log_path, "rb") as f:
            # Read first 100 bytes to check for BOM or encoding patterns
            header = f.read(100)
            if not header:
                return "utf-8"

            # Check for UTF-16LE BOM (0xFF 0xFE)
            if header[:2] == b"\xff\xfe":
                return "utf-16-le"

            # Check for UTF-16BE BOM (0xFE 0xFF)
            if header[:2] == b"\xfe\xff":
                return "utf-16-be"

            # Check for null byte pattern that indicates UTF-16LE:
            # UTF-16LE stores null bytes at even positions for ASCII text
            # If we see many null bytes at even positions, it's likely UTF-16LE
            null_count_even = 0
            null_count_odd = 0
            for i in range(min(len(header), 50)):
                if header[i] == 0:
                    if i % 2 == 0:
                        null_count_even += 1
                    else:
                        null_count_odd += 1

            # If we have multiple null bytes at even positions (e.g., >2),
            # it's likely UTF-16LE (ASCII text would have nulls at odd positions)
            if null_count_even >= 2 and null_count_even > null_count_odd:
                return "utf-16-le"

            # Default to UTF-8
            return "utf-8"
    except Exception:
        return "utf-8"
