"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
from ..logger import get_logger
log = get_logger("tools")

async def list_files(self, params: Dict[str, str]) -> str:
    """List files in a directory."""
    from .mcp import _resolve_path
    path = _resolve_path(self, params.get("path", "."))
    recursive = params.get("recursive", False); recursive = bool(recursive) if isinstance(recursive, (bool, int)) else str(recursive).lower() == "true"

    if not path.exists():
        return f"Error: Directory not found: {path}"

    # Skip directories starting with . or common junk, unless user explicitly requested them
    user_path = params.get("path", ".")
    user_requested_hidden = user_path.startswith(".") and user_path != "."
    skip_dirs = {'node_modules', '__pycache__', 'venv', 'dist', 'build', 'target', 'vendor', 'obj', 'bin'}

    def should_skip(p: Path) -> bool:
        if user_requested_hidden:
            return False
        for part in p.relative_to(path).parts:
            # Skip dotfiles/dotdirs (except current dir)
            if part.startswith('.') and part != '.':
                return True
            if part in skip_dirs:
                return True
        return False

    items = []
    truncated = False
    max_items = 100 if recursive else 50

    try:
        if recursive:
            for p in sorted(path.rglob("*")):
                if len(items) >= max_items:
                    truncated = True
                    break
                if should_skip(p):
                    continue
                rel = p.relative_to(path)
                suffix = "/" if p.is_dir() else ""
                items.append(f"{rel}{suffix}")
        else:
            for p in sorted(path.iterdir())[:max_items]:
                suffix = "/" if p.is_dir() else ""
                items.append(f"{p.name}{suffix}")
            if len(list(path.iterdir())) > max_items:
                truncated = True
    except PermissionError:
        return "Error: Permission denied"

    result = "\n".join(items) or "(empty directory)"
    if truncated:
        result += f"\n\n... (truncated at {max_items} items, use more specific path)"

    return result


async def search_files(self, params: Dict[str, str]) -> str:
    """Search for patterns in files."""
    from .mcp import _resolve_path
    path = _resolve_path(self, params.get("path", "."))
    regex = params.get("regex", "")
    file_pattern = params.get("file_pattern", "*")
    log.debug("search_files: path=%s regex=%s file_pattern=%s", path, regex, file_pattern)

    if not path.exists():
        return f"Error: Directory not found: {path}"

    try:
        pattern = re.compile(regex, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex: {e}"

    # Skip directories starting with . or common junk, unless user explicitly requested
    user_path = params.get("path", ".")
    user_requested_hidden = user_path.startswith(".") and user_path != "."
    skip_dirs = {'node_modules', '__pycache__', 'venv', 'dist', 'build', 'target', 'vendor', 'obj', 'bin'}

    def should_skip(p: Path) -> bool:
        if user_requested_hidden:
            return False
        for part in p.relative_to(path).parts:
            if part.startswith('.') and part != '.':
                return True
            if part in skip_dirs:
                return True
        return False

    results = []
    files_scanned = 0
    max_files = 2000  # Safety limit

    for file in path.rglob(file_pattern):
        if should_skip(file):
            continue
        if file.is_file():
            files_scanned += 1
            if files_scanned > max_files:
                break
            # Skip large files (>1MB)
            try:
                if file.stat().st_size > 1024 * 1024:
                    continue
                content = file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern.search(line):
                        rel = file.relative_to(path)
                        results.append(f"{rel}:{i}: {line[:150]}")
                        if len(results) >= 100:
                            break
            except:
                pass
        if len(results) >= 100:
            break

    if not results:
        return "(no matches)"

    result = "\n".join(results)
    # Add to context if significant results
    if len(results) > 5:
        ctx_id = self.context.add("search_result", regex, result)
        return f"[Context ID: {ctx_id}]\n{result}"
    return result

