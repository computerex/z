"""File operation handlers — read, write, replace."""

import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List

from ..streaming_client import desanitize_think_tokens
from ..logger import get_logger
from ._base import log, _track_write
from ._fuzzy import (
    normalize_trailing,
    normalize_unicode,
    unescape_text,
    strip_indent,
    block_anchor_match,
    find_best_fuzzy_match,
    build_diagnostic,
)

# Maximum lines allowed for a full-file read without line range params.
# Files exceeding this return an error telling the model to use start_line/end_line.
MAX_FULL_READ_LINES = 2000


def _display_path(self, path: Path) -> str:
    try:
        return str(path.relative_to(Path(self.workspace_path)))
    except Exception:
        return str(path)


async def _write_text_with_feedback(self, path: Path, content: str, action: str = "Writing") -> None:
    """Write text with visible progress for large payloads."""
    total = len(content)
    if total < self.LARGE_WRITE_FEEDBACK_THRESHOLD:
        path.write_text(content, encoding="utf-8")
        _track_write(path)
        return

    self.console.print(f"[dim]   {action} {_display_path(self, path)} ({total:,} chars)[/dim]")
    next_report = 10
    written = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for i in range(0, total, self.LARGE_WRITE_CHUNK_SIZE):
            chunk = content[i:i + self.LARGE_WRITE_CHUNK_SIZE]
            f.write(chunk)
            written += len(chunk)
            pct = int((written * 100) / max(total, 1))
            if pct >= next_report and pct < 100:
                self.console.print(f"[dim]     ... {pct}%[/dim]")
                next_report += 10
                await asyncio.sleep(0)
    self.console.print("[dim]     ... 100%[/dim]")
    _track_write(path)


async def read_file(self, params: Dict[str, str]) -> str:
    """Read a file and return its contents, with optional line range.

    If the file exceeds MAX_FULL_READ_LINES and no start_line/end_line
    are provided, returns an error instructing the model to use line
    range parameters instead.

    For image files, adds the image to context and returns a reference.
    """
    from ..image_utils import is_image_file, encode_image_to_data_uri
    from .mcp import _resolve_path
    from ..context import truncate_file_content

    path = _resolve_path(self, params.get("path", ""))
    log.debug("read_file: path=%s start_line=%s end_line=%s",
              path, params.get("start_line"), params.get("end_line"))

    if not path.exists():
        log.warning("read_file: file not found: %s", path)
        return f"Error: File not found: {path}"

    if is_image_file(path):
        # Add image to context for persistence across model switches
        data_uri = encode_image_to_data_uri(path)
        ctx_id = self.context.add("image", str(path), data_uri)
        return f"[Image: {path.name} added to context (ID: {ctx_id})]"

    rel_path = str(path.relative_to(self.workspace_path)) if str(path).startswith(self.workspace_path) else str(path)
    
    # Parse optional line range parameters (1-based, inclusive)
    # Accept aliases: offset→start_line, limit→end_line (some models prefer these)
    start_line = params.get("start_line") or params.get("offset")
    end_line = params.get("end_line") or params.get("limit")
    has_range = start_line is not None or end_line is not None

    if start_line is not None:
        try:
            start_line = int(start_line)
        except (ValueError, TypeError):
            return f"Error: start_line must be an integer, got '{start_line}'"
    if end_line is not None:
        try:
            end_line = int(end_line)
        except (ValueError, TypeError):
            return f"Error: end_line must be an integer, got '{end_line}'"

    # Track file reads for duplicate reporting
    # (Actual dedup is handled by SmartContextManager.consolidate_duplicates)
    # Only flag as duplicate if reading the SAME range (or full file twice).
    # Different line ranges of the same file are NOT duplicates.
    range_key = f"{rel_path}:{start_line or ''}-{end_line or ''}"
    prev_index = self._duplicate_detector.was_read_before(range_key)
    if prev_index is not None:
        self.console.print(f"[dim]   (duplicate read - will be consolidated during compaction)[/dim]")
    self._duplicate_detector.record_read(range_key, 0)
    
    content = path.read_text(encoding="utf-8", errors="replace")
    all_lines = content.splitlines()
    total_lines = len(all_lines)

    # If file is too large and no line range was specified, return the
    # first chunk so the model gets real content (imports, class names,
    # top-level structure) instead of a dead-end error message.
    LARGE_FILE_PREVIEW_LINES = 300
    if not has_range and total_lines > self.MAX_FULL_READ_LINES:
        preview = all_lines[:LARGE_FILE_PREVIEW_LINES]
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(preview)]
        result = "\n".join(numbered)
        result = truncate_file_content(result)
        ctx_id = self.context.add("file", rel_path, result)
        return (
            f"[Context ID: {ctx_id}]\n"
            f"(Showing first {LARGE_FILE_PREVIEW_LINES} of {total_lines:,} lines. "
            f"Use start_line/end_line to read other sections.)\n\n"
            f"{result}"
        )

    # Apply line range if specified
    if has_range:
        start_idx = max(0, (start_line - 1)) if start_line else 0
        end_idx = min(total_lines, end_line) if end_line else total_lines
        if start_idx >= total_lines:
            return f"Error: start_line {start_line} is beyond end of file ({total_lines} lines)"
        selected = all_lines[start_idx:end_idx]
        # Number lines with their actual position in the file
        numbered = [f"{start_idx + i + 1:4d} | {line}" for i, line in enumerate(selected)]
    else:
        # Small file — return entire contents
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(all_lines)]
    
    result = "\n".join(numbered)
    
    # Still apply byte-level truncation as a safety net
    result = truncate_file_content(result)
    
    # Add to context container
    ctx_id = self.context.add("file", rel_path, result)
    
    return f"[Context ID: {ctx_id}]\n{result}"


async def write_file(self, params: Dict[str, str]) -> str:
    """Write content to a new file."""
    from .mcp import _resolve_path

    path = _resolve_path(self, params.get("path", ""))
    content = desanitize_think_tokens(params.get("content", ""))
    log.info("write_file: path=%s content_len=%d", path, len(content))
    
    # Clean up invalid backtick escapes in Go files
    # Models sometimes generate \` or \`\`\` which are invalid in Go raw strings
    if path.suffix == '.go':
        original_len = len(content)
        # Remove escaped backticks like \` (invalid in Go)
        content = re.sub(r'\\`', '`', content)
        # Remove triple-backtick markdown fences that might be in raw strings
        # These often appear as ```go or ``` which break Go compilation
        content = re.sub(r'```\w*\n?', '', content)
        if len(content) != original_len:
            self.console.print(f"[dim]   (cleaned {original_len - len(content)} invalid backtick chars)[/dim]")
    
    # Warn if overwriting existing file (should use replace_in_file instead)
    was_overwrite = path.exists()
    if was_overwrite:
        old_size = path.stat().st_size
        self.console.print(f"[yellow]Warning: Overwriting existing file ({old_size} bytes). Consider replace_in_file for edits.[/yellow]")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    await _write_text_with_feedback(self, path, content, action="Writing")
    
    # DEBUG: Verify write
    if os.environ.get("HARNESS_DEBUG"):
        actual_size = path.stat().st_size
        print(f"[DEBUG write_file] written! actual_size={actual_size}", flush=True)
    
    if was_overwrite:
        return f"Successfully wrote to {path}\nNote: This file already existed. For future edits to existing files, please use replace_in_file instead of write_to_file."
    return f"Successfully wrote to {path}"


async def replace_between_anchors(self, params: Dict[str, str]) -> str:
    """Replace content between two exact anchors, preserving the anchors.

    Useful for large-block replacements where you want to keep the
    boundary lines intact and replace everything between them.
    """
    from .mcp import _resolve_path

    path = _resolve_path(self, params.get("path", ""))
    start_anchor = params.get("start_anchor", "")
    end_anchor = params.get("end_anchor", "")
    replacement = params.get("replacement", "")
    log.info(
        "replace_between_anchors: path=%s start_len=%d end_len=%d repl_len=%d",
        path, len(start_anchor), len(end_anchor), len(replacement)
    )

    if not path.exists():
        return f"Error: File not found: {path}"
    if not start_anchor:
        return "Error: start_anchor is required."
    if not end_anchor:
        return "Error: end_anchor is required."

    content = path.read_text(encoding="utf-8", errors="replace")

    start_count = content.count(start_anchor)
    end_count = content.count(end_anchor)
    if start_count == 0:
        return "Error: start_anchor not found in file."
    if end_count == 0:
        return "Error: end_anchor not found in file."
    if start_count > 1:
        return f"Error: start_anchor matched {start_count} times. Use a more specific anchor."
    if end_count > 1:
        return f"Error: end_anchor matched {end_count} times. Use a more specific anchor."

    start_idx = content.find(start_anchor)
    end_idx = content.find(end_anchor)
    if end_idx <= start_idx:
        return "Error: end_anchor occurs before start_anchor."

    body_start = start_idx + len(start_anchor)
    old_segment = content[body_start:end_idx]
    new_content = content[:body_start] + replacement + content[end_idx:]
    await _write_text_with_feedback(self, path, new_content, action="Writing updated file")

    old_lines = old_segment.count("\n") + (1 if old_segment else 0)
    new_lines = replacement.count("\n") + (1 if replacement else 0)
    return (
        f"Successfully replaced content between anchors in {path}\n"
        f"Anchors preserved. Replaced ~{old_lines} line(s) with ~{new_lines} line(s)."
    )


async def replace_in_file(self, params: Dict[str, str]) -> str:
    """Replace a section of text in an existing file.
    
    Matching strategy (in order):
    1. Exact string match
    2. Trailing-whitespace-normalized match
    3. Unicode-normalized match (smart quotes, en-dashes, NBSP → ASCII)
    4. Escape-sequence-normalized match (literal \\n → newline, etc.)
    5. Indentation-agnostic match (strip leading whitespace, compare content)
    6. Block-anchor match (first+last line exact, fuzzy middle — from opencode)
    7. Fuzzy best-match (difflib) — if similarity ≥ 0.6, apply with a warning
    8. Fail with a helpful diagnostic showing the closest section in the file
    """
    from .mcp import _resolve_path

    path = _resolve_path(self, params.get("path", ""))
    search = desanitize_think_tokens(params.get("old_text", ""))
    replace = desanitize_think_tokens(params.get("new_text", ""))
    log.info("replace_in_file: path=%s old_len=%d new_len=%d", path, len(search), len(replace))
    
    if not path.exists():
        log.warning("replace_in_file: file not found: %s", path)
        return f"Error: File not found: {path}"
    
    if not search:
        return "Error: old_text is required (the text to find and replace)."
    
    raw_content = path.read_text(encoding="utf-8")
    content = raw_content
    
    # Strategy 1: Exact match
    if search in content:
        content = content.replace(search, replace, 1)
        await _write_text_with_feedback(self, path, content, action="Writing updated file")
        return f"Successfully replaced text in {path}"
    
    # Strategy 2: Trailing-whitespace-normalized match
    norm_content = normalize_trailing(content)
    norm_search = normalize_trailing(search)
    
    if norm_search in norm_content:
        search_lines = norm_search.split('\n')
        content_lines = content.replace('\r\n', '\n').split('\n')
        
        for i in range(len(content_lines) - len(search_lines) + 1):
            match = True
            for j, search_line in enumerate(search_lines):
                if content_lines[i + j].rstrip() != search_line:
                    match = False
                    break
            if match:
                replace_lines = replace.replace('\r\n', '\n').split('\n')
                content_lines = content_lines[:i] + replace_lines + content_lines[i + len(search_lines):]
                content = '\n'.join(content_lines)
                await _write_text_with_feedback(self, path, content, action="Writing updated file")
                return f"Successfully replaced text in {path}"

    # Strategy 3: Unicode-normalized match
    # Handles LLM substitutions: smart quotes → ASCII, en-dash → hyphen, NBSP → space
    uni_content = normalize_unicode(content)
    uni_search = normalize_unicode(search)
    uni_replace = normalize_unicode(replace)
    # Only apply if normalization changed search or content (otherwise S1/S2 already handled it)
    if (uni_search != search or uni_content != content) and uni_search in uni_content:
        idx = uni_content.index(uni_search)
        # Apply replacement inside unicode-normalized content, then write
        new_uni = uni_content[:idx] + uni_replace + uni_content[idx + len(uni_search):]
        await _write_text_with_feedback(self, path, new_uni, action="Writing updated file")
        content = new_uni
        self.console.print(f"[dim]   (matched with Unicode normalization)[/dim]")
        return f"Successfully replaced text in {path}"

    # Strategy 4: Escape-sequence-normalized match
    # Handles LLM outputting literal \n instead of actual newline characters
    esc_content = unescape_text(content)
    esc_search = unescape_text(search)
    # Only apply if unescaping changed search (otherwise S1 should have caught it)
    if esc_search != search and esc_search in esc_content:
        idx = esc_content.index(esc_search)
        esc_replace = unescape_text(replace)
        new_content = esc_content[:idx] + esc_replace + esc_content[idx + len(esc_search):]
        await _write_text_with_feedback(self, path, new_content, action="Writing updated file")
        content = new_content
        self.console.print(f"[dim]   (matched with escape-sequence normalization)[/dim]")
        return f"Successfully replaced text in {path}"

    # Strategy 5: Indentation-agnostic match
    search_stripped, _ = strip_indent(search)
    content_lines_raw = content.replace('\r\n', '\n').split('\n')
    content_stripped = [l.lstrip() for l in content_lines_raw]
    
    for i in range(len(content_stripped) - len(search_stripped) + 1):
        if all(content_stripped[i + j] == search_stripped[j]
               for j in range(len(search_stripped))):
            file_indent = content_lines_raw[i][:len(content_lines_raw[i]) - len(content_lines_raw[i].lstrip())]
            search_indent = search.replace('\r\n', '\n').split('\n')[0]
            search_indent = search_indent[:len(search_indent) - len(search_indent.lstrip())]
            
            replace_lines_raw = replace.replace('\r\n', '\n').split('\n')
            adjusted_replace = []
            for rl in replace_lines_raw:
                if rl.startswith(search_indent):
                    adjusted_replace.append(file_indent + rl[len(search_indent):])
                else:
                    adjusted_replace.append(rl)
            
            content_lines_raw = content_lines_raw[:i] + adjusted_replace + content_lines_raw[i + len(search_stripped):]
            content = '\n'.join(content_lines_raw)
            await _write_text_with_feedback(self, path, content, action="Writing updated file")
            self.console.print(f"[dim]   (matched with indentation adjustment)[/dim]")
            return f"Successfully replaced text in {path}"

    # Strategy 6: Block-anchor match (first+last line exact, fuzzy middle)
    # Ported from opencode's BlockAnchorReplacer.  Handles cases where the
    # model's old_text has slightly-off middle lines but correct boundaries.
    content_lines_raw = content.replace('\r\n', '\n').split('\n')
    search_lines_raw = search.replace('\r\n', '\n').split('\n')
    anchor = block_anchor_match(content_lines_raw, search_lines_raw)
    if anchor:
        start, end = anchor
        replace_lines = replace.replace('\r\n', '\n').split('\n')
        content_lines_raw = content_lines_raw[:start] + replace_lines + content_lines_raw[end:]
        content = '\n'.join(content_lines_raw)
        await _write_text_with_feedback(self, path, content, action="Writing updated file")
        self.console.print(f"[dim]   (matched with block-anchor at lines {start+1}-{end})[/dim]")
        return f"Successfully replaced text in {path}"

    # Strategy 7: Fuzzy match — apply if similarity ≥ 0.6
    # Guard: also reject if ANY single line diverges too much from the
    # corresponding file line (catches reasoning-text contamination where
    # the model's thinking leaks into old_text string arguments).
    import difflib
    fuzzy = find_best_fuzzy_match(content, search)
    if fuzzy and fuzzy[2] >= 0.6:
        start, end, ratio = fuzzy
        content_lines_raw = content.replace('\r\n', '\n').split('\n')
        search_lines_check = search.replace('\r\n', '\n').split('\n')
        candidate_lines = content_lines_raw[start:end]

        # Per-line similarity floor: every aligned line pair must be ≥ 0.5
        min_line_ratio = 1.0
        for sl, cl in zip(search_lines_check, candidate_lines):
            lr = difflib.SequenceMatcher(None, sl.strip(), cl.strip()).ratio()
            min_line_ratio = min(min_line_ratio, lr)

        if min_line_ratio >= 0.5:
            replace_lines = replace.replace('\r\n', '\n').split('\n')
            content_lines_raw = content_lines_raw[:start] + replace_lines + content_lines_raw[end:]
            content = '\n'.join(content_lines_raw)
            await _write_text_with_feedback(self, path, content, action="Writing updated file")
            self.console.print(f"[dim]   (fuzzy match {ratio:.0%} at lines {start+1}-{end})[/dim]")
            return f"Successfully replaced text in {path}"
        else:
            log.info("replace_in_file: fuzzy match rejected — per-line min ratio %.2f < 0.5 (possible contamination)", min_line_ratio)
    
    # Strategy 8: Fail with helpful diagnostic
    return build_diagnostic(content, search)
