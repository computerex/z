"""Tool handlers for ClineAgent - manages all tool execution logic."""

import asyncio
import base64
import os
import platform
import re
import signal
import time
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
import psutil

from .context_management import truncate_file_content, truncate_output
from .logger import get_logger, log_exception, truncate as log_truncate

log = get_logger("tools")


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
    gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
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


def parse_search_replace_blocks(diff: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from diff string."""
    blocks = []
    
    # Normalize line endings first
    diff = diff.replace('\r\n', '\n').replace('\r', '\n')
    
    # Pattern: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
    # Allow optional whitespace and be flexible with newlines
    pattern = r'<{7}\s*SEARCH\s*\n(.*?)\n={7}\s*\n(.*?)\n>{7}\s*REPLACE'
    
    for m in re.finditer(pattern, diff, re.DOTALL):
        search, replace = m.groups()
        blocks.append((search, replace))
    
    return blocks


class ToolHandlers:
    """Handles execution of all tools for ClineAgent."""
    
    def __init__(
        self,
        config,
        console: Console,
        workspace_path: str,
        context,
        duplicate_detector,
        context_manager=None
    ):
        """Initialize tool handlers with required dependencies.
        
        Args:
            config: Config object with API settings
            console: Rich console for output
            workspace_path: Path to workspace directory
            context: ContextContainer for managing loaded content
            duplicate_detector: DuplicateDetector for tracking file reads
            context_manager: SmartContextManager for accessing stored tool results
        """
        self.config = config
        self.console = console
        self.workspace_path = workspace_path
        self.context = context
        self._duplicate_detector = duplicate_detector
        self._context_manager = context_manager
        
        # Background processes: {id: {"proc": Process, "command": str, "started": float, "log_file": str, "task": Task}}
        self._background_procs: Dict[int, dict] = {}
        self._next_bg_id = 1
        self._next_cmd_id = 1  # For unique command log files
        
        # Directory for spilled command output files
        self._output_dir = os.path.join(workspace_path, ".harness_output")
        os.makedirs(self._output_dir, exist_ok=True)
    
    # -- Output spill helpers --------------------------------------------------
    
    # Threshold in estimated tokens above which output is spilled to a file.
    OUTPUT_SPILL_TOKEN_THRESHOLD = 3000  # ~12,000 chars
    
    # Maximum tokens to include inline when output is spilled.
    OUTPUT_INLINE_PREVIEW_TOKENS = 300   # ~1,200 chars — enough for LLM to understand
    
    def spill_output_to_file(self, output: str, label: str) -> str:
        """Write large output to a file and return a compact reference.
        
        Returns the original output unchanged if it's below threshold,
        otherwise writes to .harness_output/ and returns a truncated preview
        with the file path so the model can read_file to inspect details.
        """
        est_tokens = len(output) // 4
        if est_tokens <= self.OUTPUT_SPILL_TOKEN_THRESHOLD:
            return output
        
        # Write full output to a file
        import hashlib
        safe_label = re.sub(r'[^\w\-.]', '_', label)[:60]
        ts = int(time.time())
        filename = f"{safe_label}_{ts}.txt"
        os.makedirs(self._output_dir, exist_ok=True)
        filepath = os.path.join(self._output_dir, filename)
        Path(filepath).write_text(output, encoding="utf-8")
        log.info("Output spilled to file: %s (%d tokens, %d chars)", filepath, est_tokens, len(output))
        
        # Build compact inline result
        lines = output.splitlines()
        total_lines = len(lines)
        preview_chars = self.OUTPUT_INLINE_PREVIEW_TOKENS * 4
        head = output[:preview_chars // 2]
        tail = output[-(preview_chars // 2):]
        
        return (
            f"[OUTPUT SPILLED TO FILE — {est_tokens:,} tokens, {total_lines} lines]\n"
            f"Full output saved to: {filepath}\n"
            f"Use read_file to inspect specific sections.\n\n"
            f"--- First lines ---\n{head}\n\n"
            f"--- Last lines ---\n{tail}"
        )
    
    def _get_bg_log_path(self, proc_id: int) -> str:
        """Get the log file path for a background process."""
        os.makedirs(self._output_dir, exist_ok=True)
        return os.path.join(self._output_dir, f"bg_process_{proc_id}.log")
    
    def _get_cmd_log_path(self) -> str:
        """Get a unique log file path for a foreground command."""
        os.makedirs(self._output_dir, exist_ok=True)
        cmd_id = self._next_cmd_id
        self._next_cmd_id += 1
        return os.path.join(self._output_dir, f"cmd_{cmd_id}.log")
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.workspace_path) / p
        return p.resolve()
    
    async def _background_log_tailer(self, bg_id: int, proc: asyncio.subprocess.Process,
                                      log_path: str):
        """Continuously tail a log file for a background process.
        
        Since the process output is shell-redirected to log_path, this task
        just keeps reading new content and caching it in memory.
        """
        info = self._background_procs.get(bg_id)
        if not info:
            return
        
        file_pos = 0
        try:
            while proc.returncode is None:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(file_pos)
                        new_data = f.read()
                        file_pos = f.tell()
                    if new_data:
                        for line in new_data.splitlines():
                            info["logs"].append(line)
                            if len(info["logs"]) > 200:
                                info["logs"] = info["logs"][-200:]
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            
            # Post-exit grace period: keep reading in case a detached child
            # process (e.g. GUI app) is still writing to the log file via
            # inherited file handles.  Stop after 5s of no new content.
            idle_elapsed = 0.0
            while idle_elapsed < 5.0:
                await asyncio.sleep(0.5)
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(file_pos)
                        new_data = f.read()
                        file_pos = f.tell()
                    if new_data:
                        for line in new_data.splitlines():
                            info["logs"].append(line)
                            if len(info["logs"]) > 200:
                                info["logs"] = info["logs"][-200:]
                        idle_elapsed = 0.0  # reset — still getting output
                    else:
                        idle_elapsed += 0.5
                except Exception:
                    idle_elapsed += 0.5
        except Exception:
            pass
    
    async def cleanup_background_procs_async(self) -> None:
        """Async version - properly waits for processes to terminate."""
        for pid, info in list(self._background_procs.items()):
            try:
                proc = info["proc"]
                if proc.returncode is None:
                    kill_process_tree(proc.pid)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3.0)
                    except (asyncio.TimeoutError, Exception):
                        pass
                if "task" in info and info["task"]:
                    try:
                        info["task"].cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        self._background_procs.clear()
    
    def cleanup_background_procs(self) -> None:
        """Terminate all background processes safely (sync wrapper)."""
        for pid, info in list(self._background_procs.items()):
            try:
                proc = info["proc"]
                if proc.returncode is None:
                    kill_process_tree(proc.pid)
                if "task" in info and info["task"]:
                    try:
                        info["task"].cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        self._background_procs.clear()
    
    def list_background_procs(self) -> list:
        """List all background processes with their status."""
        result = []
        for bg_id, info in self._background_procs.items():
            proc = info["proc"]
            # Poll the process to update returncode if it has finished
            if proc.returncode is None:
                try:
                    # asyncio.subprocess.Process doesn't have poll(), but
                    # checking the underlying transport can update returncode
                    if proc._transport is not None and proc._transport.get_returncode() is not None:
                        proc._returncode = proc._transport.get_returncode()
                except Exception:
                    pass
            elapsed = time.time() - info["started"]
            status = "running" if proc.returncode is None else f"exited ({proc.returncode})"
            result.append({
                "id": bg_id,
                "pid": proc.pid,
                "command": info["command"][:50],
                "elapsed": elapsed,
                "status": status
            })
        return result
    
    # Maximum lines allowed for a full-file read without line range params.
    # Files exceeding this return an error telling the model to use start_line/end_line.
    MAX_FULL_READ_LINES = 2000

    async def read_file(self, params: Dict[str, str]) -> str:
        """Read a file and return its contents, with optional line range.
        
        If the file exceeds MAX_FULL_READ_LINES and no start_line/end_line
        are provided, returns an error instructing the model to use line
        range parameters instead.
        """
        path = self._resolve_path(params.get("path", ""))
        log.debug("read_file: path=%s start_line=%s end_line=%s",
                  path, params.get("start_line"), params.get("end_line"))
        
        if not path.exists():
            log.warning("read_file: file not found: %s", path)
            return f"Error: File not found: {path}"
        
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

        # If file is too large and no line range was specified, reject with guidance
        if not has_range and total_lines > self.MAX_FULL_READ_LINES:
            return (
                f"Error: File is too large to read in full ({total_lines:,} lines, "
                f"~{len(content) // 4:,} tokens). Use start_line and end_line "
                f"parameters to read specific sections.\n"
                f"Example: <read_file><path>{rel_path}</path>"
                f"<start_line>1</start_line><end_line>100</end_line></read_file>\n"
                f"Total lines in file: {total_lines}"
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
        path = self._resolve_path(params.get("path", ""))
        content = params.get("content", "")
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
        path.write_text(content, encoding="utf-8")
        
        # DEBUG: Verify write
        if os.environ.get("HARNESS_DEBUG"):
            actual_size = path.stat().st_size
            print(f"[DEBUG write_file] written! actual_size={actual_size}", flush=True)
        
        if was_overwrite:
            return f"Successfully wrote to {path}\nNote: This file already existed. For future edits to existing files, please use replace_in_file instead of write_to_file."
        return f"Successfully wrote to {path}"

    async def replace_between_anchors(self, params: Dict[str, str]) -> str:
        """Replace content between two exact anchors, preserving the anchors.

        Designed for cases where `replace_in_file` is brittle, especially when
        the file contains SEARCH/REPLACE delimiters like `=======` literally.
        """
        path = self._resolve_path(params.get("path", ""))
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
        path.write_text(new_content, encoding="utf-8")

        old_lines = old_segment.count("\n") + (1 if old_segment else 0)
        new_lines = replacement.count("\n") + (1 if replacement else 0)
        return (
            f"Successfully replaced content between anchors in {path}\n"
            f"Anchors preserved. Replaced ~{old_lines} line(s) with ~{new_lines} line(s)."
        )
    
    async def replace_in_file(self, params: Dict[str, str]) -> str:
        """Replace sections of content in an existing file.
        
        Matching strategy (in order):
        1. Exact string match
        2. Trailing-whitespace-normalized match
        3. Indentation-agnostic match (strip leading whitespace, compare content)
        4. Fuzzy best-match (difflib) — if similarity ≥ 0.6, apply with a warning
        5. Fail with a helpful diagnostic showing the closest section in the file
        """
        path = self._resolve_path(params.get("path", ""))
        diff = params.get("diff", "")
        log.info("replace_in_file: path=%s diff_len=%d", path, len(diff))
        
        if not path.exists():
            log.warning("replace_in_file: file not found: %s", path)
            return f"Error: File not found: {path}"
        
        content = path.read_text(encoding="utf-8")
        blocks = parse_search_replace_blocks(diff)
        
        if not blocks:
            return "Error: No valid SEARCH/REPLACE blocks found"
        
        def normalize_trailing(s: str) -> str:
            """Normalize line endings and trailing whitespace."""
            lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            return '\n'.join(line.rstrip() for line in lines)
        
        def strip_indent(s: str) -> list:
            """Return (stripped_lines, indent_per_line) for indentation-agnostic compare."""
            lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            stripped = [line.lstrip() for line in lines]
            indents = [line[:len(line) - len(line.lstrip())] for line in lines]
            return stripped, indents
        
        def find_best_fuzzy_match(content_text: str, search_text: str):
            """Find the best fuzzy match for search_text within content_text.
            
            Returns (start_line_idx, end_line_idx, similarity_ratio) or None.
            """
            import difflib
            search_lines = search_text.replace('\r\n', '\n').split('\n')
            content_lines = content_text.replace('\r\n', '\n').split('\n')
            search_len = len(search_lines)
            
            if search_len == 0 or len(content_lines) == 0:
                return None
            
            best_ratio = 0.0
            best_start = 0
            
            # Slide a window of size search_len (±30%) across content_lines
            min_window = max(1, int(search_len * 0.7))
            max_window = int(search_len * 1.3) + 1
            
            for window_size in range(min_window, min(max_window, len(content_lines) + 1)):
                for i in range(len(content_lines) - window_size + 1):
                    candidate = content_lines[i:i + window_size]
                    # Quick pre-filter: at least some lines must share content
                    shared = sum(1 for a, b in zip(search_lines, candidate)
                                 if a.strip() == b.strip())
                    if shared < min(3, search_len * 0.3):
                        continue
                    
                    ratio = difflib.SequenceMatcher(
                        None,
                        '\n'.join(search_lines),
                        '\n'.join(candidate),
                    ).ratio()
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_start = i
                        best_window = window_size
            
            if best_ratio > 0.4:
                return best_start, best_start + best_window, best_ratio
            return None
        
        def build_diagnostic(content_text: str, search_text: str) -> str:
            """Build a helpful error message showing the closest match."""
            match_info = find_best_fuzzy_match(content_text, search_text)
            
            search_lines = search_text.replace('\r\n', '\n').split('\n')
            content_lines = content_text.replace('\r\n', '\n').split('\n')
            
            msg_parts = [
                f"Error: SEARCH block not found in file.",
                f"",
                f"SEARCH block ({len(search_lines)} lines):",
                f"  {chr(10).join('  ' + l for l in search_lines[:5])}",
            ]
            if len(search_lines) > 5:
                msg_parts.append(f"  ... ({len(search_lines) - 5} more lines)")
            
            if match_info:
                start, end, ratio = match_info
                msg_parts.append(f"")
                msg_parts.append(f"Closest match in file (lines {start+1}-{end}, {ratio:.0%} similar):")
                for i in range(start, min(end, start + 10)):
                    msg_parts.append(f"  {i+1:4d} | {content_lines[i]}")
                if end - start > 10:
                    msg_parts.append(f"  ... ({end - start - 10} more lines)")
                msg_parts.append(f"")
                msg_parts.append(f"Tip: Re-read the file around lines {start+1}-{end} and retry with the exact content.")
            else:
                # Show first few lines of each search line in context
                first_search = search_lines[0].strip() if search_lines else ""
                if first_search:
                    near = [i for i, l in enumerate(content_lines) if first_search in l]
                    if near:
                        msg_parts.append(f"")
                        msg_parts.append(f"First search line found near line(s): {', '.join(str(n+1) for n in near[:5])}")
                        ctx_start = max(0, near[0] - 2)
                        ctx_end = min(len(content_lines), near[0] + 5)
                        for i in range(ctx_start, ctx_end):
                            msg_parts.append(f"  {i+1:4d} | {content_lines[i]}")
                msg_parts.append(f"")
                msg_parts.append(f"Tip: Use read_file with start_line/end_line to see the exact content, then retry.")
            
            return '\n'.join(msg_parts)
        
        changes = 0
        for search, replace in blocks:
            # Strategy 1: Exact match
            if search in content:
                content = content.replace(search, replace, 1)
                changes += 1
                continue
            
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
                        changes += 1
                        break
                else:
                    return build_diagnostic(content, search)
                continue
            
            # Strategy 3: Indentation-agnostic match
            search_stripped, _ = strip_indent(search)
            content_lines_raw = content.replace('\r\n', '\n').split('\n')
            content_stripped = [l.lstrip() for l in content_lines_raw]
            indent_match_found = False
            
            for i in range(len(content_stripped) - len(search_stripped) + 1):
                if all(content_stripped[i + j] == search_stripped[j]
                       for j in range(len(search_stripped))):
                    # Content matches but indentation differs.
                    # Determine the indentation offset and apply it to the replacement.
                    file_indent = content_lines_raw[i][:len(content_lines_raw[i]) - len(content_lines_raw[i].lstrip())]
                    search_indent = search.replace('\r\n', '\n').split('\n')[0]
                    search_indent = search_indent[:len(search_indent) - len(search_indent.lstrip())]
                    
                    replace_lines_raw = replace.replace('\r\n', '\n').split('\n')
                    # Adjust each replacement line: remove search's base indent, add file's base indent
                    adjusted_replace = []
                    for rl in replace_lines_raw:
                        if rl.startswith(search_indent):
                            adjusted_replace.append(file_indent + rl[len(search_indent):])
                        else:
                            adjusted_replace.append(rl)
                    
                    content_lines_raw = content_lines_raw[:i] + adjusted_replace + content_lines_raw[i + len(search_stripped):]
                    content = '\n'.join(content_lines_raw)
                    changes += 1
                    indent_match_found = True
                    self.console.print(f"[dim]   (matched with indentation adjustment)[/dim]")
                    break
            
            if indent_match_found:
                continue
            
            # Strategy 4: Fuzzy match — apply if similarity ≥ 0.6
            fuzzy = find_best_fuzzy_match(content, search)
            if fuzzy and fuzzy[2] >= 0.6:
                start, end, ratio = fuzzy
                content_lines_raw = content.replace('\r\n', '\n').split('\n')
                replace_lines = replace.replace('\r\n', '\n').split('\n')
                content_lines_raw = content_lines_raw[:start] + replace_lines + content_lines_raw[end:]
                content = '\n'.join(content_lines_raw)
                changes += 1
                self.console.print(f"[dim]   (fuzzy match {ratio:.0%} at lines {start+1}-{end})[/dim]")
                continue
            
            # Strategy 5: Fail with helpful diagnostic
            return build_diagnostic(content, search)
        
        path.write_text(content, encoding="utf-8")
        return f"Successfully made {changes} replacement(s) in {path}"
    
    async def execute_command(self, params: Dict[str, str]) -> str:
        """Execute a shell command with live output display and interrupt support.
        
        All commands are launched with stdout/stderr redirected to a log file
        (not piped to Python).  An async tail loop reads the log file for live
        console output.  This unified approach means GUI applications can create
        windows (stdout is not captured via pipe) while CLI tools still get
        their output displayed and recorded.
        """
        from .interrupt import is_interrupted, is_background_requested, reset_background
        
        command = params.get("command", "")
        background = params.get("background", "").lower() == "true"
        timeout_secs = 120  # Auto-background after this many seconds
        if not command.strip():
            return "Error: execute_command requires a non-empty <command> parameter."
        log.info("execute_command: cmd=%s bg=%s", log_truncate(command, 200), background)
        
        # Show command being executed
        mode_indicator = "[bg] " if background else ""
        cmd_short = command.split('\n')[0][:120]  # First line, truncated
        self.console.print(f"\n  [dim]•[/dim] Running [dim]{mode_indicator}{cmd_short}[/dim]")
        
        if background:
            return await self._run_background_command(command)
        
        # ── Launch with file redirect ──────────────────────────────────
        cmd_log_path = self._get_cmd_log_path()

        # Truncate the log file before launching so the tail loop never
        # reads stale content from a previous session (the cmd_id counter
        # resets on each harness start, so filenames are reused).
        Path(cmd_log_path).write_text("", encoding="utf-8")

        # Shell-level redirect: stdout+stderr go to the log file.
        # The process itself runs without Python pipes so GUI windows work.
        # NOTE: create_subprocess_shell already invokes the platform shell
        # (cmd.exe on Windows, /bin/sh on Unix), so we must NOT wrap with
        # an extra "cmd /c" — that breaks commands containing double quotes.
        wrapped = self._wrap_shell_command(command, cmd_log_path)

        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.workspace_path,
        )
        log.info("Process launched: PID=%d log=%s", proc.pid, cmd_log_path)
        
        # ── Tail loop: read log file for live output ───────────────────
        output_lines: List[str] = []
        start_time = time.time()
        hint_shown = False
        file_pos = 0
        
        try:
            while proc.returncode is None:
                elapsed = time.time() - start_time
                
                # Check for interrupt (Esc)
                if is_interrupted():
                    self._kill_proc(proc)
                    self.console.print(f"\n    [yellow]⏹ Command interrupted[/yellow]")
                    raw_output = self._read_log_file(cmd_log_path)
                    output = self.spill_output_to_file(
                        raw_output, f"interrupted_{command.split()[0] if command else 'cmd'}")
                    return f"Command interrupted after {elapsed:.0f}s.\nOutput captured:\n{output}" if output else "Command interrupted (no output)"
                
                # Check for background request (Ctrl+B)
                if is_background_requested():
                    reset_background()
                    self.console.print(f"\n    [cyan]→ Sending to background...[/cyan]")
                    return self._promote_to_background(proc, command, start_time, cmd_log_path, output_lines)
                
                # Show hint after 5 seconds
                if elapsed > 5 and not hint_shown:
                    self.console.print(f"    [dim](Ctrl+B to background, Esc to stop)[/dim]")
                    hint_shown = True
                
                # Auto-background after timeout.  Use a shorter timeout for
                # commands producing no output — likely GUI apps (e.g. notepad)
                # that block cmd.exe but have no console output to tail.
                effective_timeout = 10 if not output_lines else timeout_secs
                if elapsed > effective_timeout:
                    self.console.print(f"\n    [cyan]→ No output for {elapsed:.0f}s, sending to background...[/cyan]")
                    return self._promote_to_background(proc, command, start_time, cmd_log_path, output_lines)
                
                # Read new content from log file
                file_pos = self._tail_log_file(cmd_log_path, file_pos, output_lines)
                
                # Poll process exit (non-blocking)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.15)
                except asyncio.TimeoutError:
                    pass
            
            # Process has exited — do one final read to catch any trailing output
            await asyncio.sleep(0.1)  # Brief pause for OS to flush file buffers
            file_pos = self._tail_log_file(cmd_log_path, file_pos, output_lines)

            # Detached GUI app detection: if the shell exited very quickly with
            # little or no output, the actual app may have detached (e.g. Windows
            # GUI subsystem) and is still writing to the log file via inherited
            # handles.  Wait briefly, then check if the log is growing — if so,
            # promote to background so the standard log tailer keeps monitoring.
            elapsed_so_far = time.time() - start_time
            if elapsed_so_far < 2.0 and len(output_lines) < 3:
                log.info("Fast exit with little output (%.1fs, %d lines) — "
                         "checking for detached GUI app", elapsed_so_far, len(output_lines))
                await asyncio.sleep(1.0)
                file_pos = self._tail_log_file(cmd_log_path, file_pos, output_lines)
                if output_lines:
                    # Log file is growing — a detached app is still running.
                    # Promote to background so the log tailer keeps reading.
                    self.console.print(f"    [cyan]→ Detached process detected, sending to background...[/cyan]")
                    return self._promote_to_background(proc, command, start_time, cmd_log_path, output_lines)

            exit_code = proc.returncode
            elapsed_cmd = time.time() - start_time
            log.info("execute_command finished: cmd=%s exit=%d elapsed=%.1fs output_lines=%d",
                     log_truncate(command, 80), exit_code, elapsed_cmd, len(output_lines))
            # Show collapsed line count if output was truncated
            n_lines = len(output_lines)
            if n_lines > self._MAX_LIVE_DISPLAY:
                self.console.print(f"    [dim]… +{n_lines - self._MAX_LIVE_DISPLAY} lines[/dim]")
            
            if exit_code == 0:
                self.console.print(f"    [dim](exit 0, {elapsed_cmd:.1f}s)[/dim]")
            else:
                log.warning("Command failed: exit=%d cmd=%s", exit_code, log_truncate(command, 120))
                self.console.print(f"    [red]✗ exit {exit_code}[/red] [dim]({elapsed_cmd:.1f}s)[/dim]")
            
            # Build raw output and spill to file if huge
            raw_output = self._read_log_file(cmd_log_path) or "(no output)"
            output = truncate_output(raw_output, max_lines=300, keep_start=80, keep_end=80)
            output = self.spill_output_to_file(output, f"cmd_{command.split()[0] if command else 'cmd'}")
            
            # Add to context if significant output
            if len(output_lines) > 3:
                ctx_id = self.context.add("command_output", command, output)
                return f"[Context ID: {ctx_id}]\n{output}"
            return output
            
        except Exception as e:
            log_exception(log, f"execute_command exception: {log_truncate(command, 80)}", e)
            self._kill_proc(proc)
            raw_output = self._read_log_file(cmd_log_path)
            output = self.spill_output_to_file(raw_output, "cmd_error") if raw_output else ""
            return f"Error: {str(e)}\nOutput captured:\n{output}" if output else f"Error: {str(e)}"
    
    # ── Helpers for file-redirect execution ─────────────────────────────
    
    # Maximum lines to show live before collapsing (show first N, then summarize)
    _MAX_LIVE_DISPLAY = 10

    def _wrap_shell_command(self, command: str, log_path: str) -> str:
        """Build platform shell wrapper for a command with file redirection.
        
        On Windows, default to PowerShell execution for deterministic behavior
        with PowerShell syntax (the system prompt advertises PowerShell).
        Set HARNESS_WINDOWS_SHELL=cmd to force legacy cmd.exe behavior.
        """
        if platform.system() == "Windows":
            win_shell = os.environ.get("HARNESS_WINDOWS_SHELL", "powershell").strip().lower()
            if win_shell != "cmd":
                # Use EncodedCommand to avoid quote/escape issues through cmd.exe.
                ps_command = "$ProgressPreference='SilentlyContinue'; " + command
                encoded = base64.b64encode(ps_command.encode("utf-16le")).decode("ascii")
                launcher = (
                    "powershell -NoProfile -NonInteractive "
                    "-InputFormat Text -OutputFormat Text "
                    "-ExecutionPolicy Bypass "
                    f"-EncodedCommand {encoded}"
                )
                return f'{launcher} > "{log_path}" 2>&1'
        return f'{command} > "{log_path}" 2>&1'

    def _tail_log_file(self, log_path: str, file_pos: int, output_lines: List[str]) -> int:
        """Read new content from a log file starting at file_pos.
        
        Displays new lines in the console and appends to output_lines.
        After _MAX_LIVE_DISPLAY lines, suppresses further live display —
        the final summary is printed by execute_command on completion.
        Returns the updated file position.
        """
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(file_pos)
                new_data = f.read()
                new_pos = f.tell()
            if new_data:
                for line in new_data.splitlines():
                    output_lines.append(line)
                    n = len(output_lines)
                    if n <= self._MAX_LIVE_DISPLAY:
                        safe_line = sanitize_terminal_output(line)
                        self.console.print(f"    [dim]{safe_line}[/dim]")
                    elif n == self._MAX_LIVE_DISPLAY + 1:
                        self.console.print(f"    [dim]… +more lines (running)[/dim]")
            return new_pos
        except FileNotFoundError:
            return file_pos
        except Exception:
            return file_pos
    
    def _read_log_file(self, log_path: str) -> str:
        """Read the entire contents of a log file."""
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    
    def _kill_proc(self, proc: asyncio.subprocess.Process) -> None:
        """Kill a process and its entire process tree."""
        kill_process_tree(proc.pid)
    
    def _promote_to_background(self, proc, command: str, start_time: float,
                                log_path: str, output_lines: List[str]) -> str:
        """Promote a foreground process to a tracked background process."""
        proc_id = self._next_bg_id
        self._next_bg_id += 1
        
        self._background_procs[proc_id] = {
            "proc": proc,
            "command": command,
            "started": start_time,
            "logs": output_lines.copy()[-200:],
            "log_file": log_path,
            "task": asyncio.create_task(self._background_log_tailer(proc_id, proc, log_path)),
        }
        self.console.print(f"    [green]→ Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
        recent = "\n".join(output_lines[-30:])
        return (
            f"Command sent to background (ID: {proc_id}, PID: {proc.pid}).\n"
            f"Log file: {log_path}\n"
            f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
            f"Output so far:\n{recent}"
        )

    async def create_plan(self, params: Dict[str, str], context_summary: str = "") -> str:
        """Delegate a complex reasoning task to Claude CLI (Opus 4.6).
        
        Auto-attaches context summary (todos, recent files, workspace info).
        
        NOTE: This does NOT go through execute_command.  The prompt may contain
        newlines, quotes, pipes, angle brackets, etc. that would be mangled by
        shell quoting / redirect.  Instead we launch the claude CLI directly
        via create_subprocess_exec and pipe the prompt through a temp file
        passed with the -p flag reading from file, or via stdin.
        """
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return "Error: 'prompt' is required for create_plan."
        
        # Build full prompt with context
        full_prompt = prompt
        if context_summary:
            full_prompt = f"CONTEXT:\n{context_summary}\n\nTASK:\n{prompt}"
        
        # Write prompt to a temp file to avoid all shell quoting issues.
        # Claude CLI reads -p from the command line, but we can pipe via stdin
        # using the '-' convention or just pass a sanitized file reference.
        # Safest approach: write to file, pass via stdin with --pipe flag.
        import tempfile
        prompt_file = None
        try:
            # Write prompt to temp file
            prompt_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.txt', prefix='harness_plan_',
                dir=self._output_dir, delete=False, encoding='utf-8'
            )
            prompt_file.write(full_prompt)
            prompt_file.close()
            prompt_path = prompt_file.name
            
            # Build args: claude reads prompt from stdin via -p "$(cat file)"
            # Actually simplest: use -p @file or just pipe stdin.
            # Claude CLI supports: echo "prompt" | claude -p -
            # But safest: claude -p with the file contents piped via stdin
            args = ["claude", "-p", "-", "--dangerously-skip-permissions"]
            
            # Use configured model if available
            claude_model = getattr(self, '_claude_model', None)
            if claude_model:
                args.extend(["--model", claude_model])
            
            log.info("create_plan: launching Claude CLI with %d-char prompt via stdin", len(full_prompt))
            self.console.print(f"[bold yellow]{'─' * 4} create_plan (Claude CLI) {'─' * 25}[/bold yellow]")
            self.console.print(f"[dim]  Prompt: {len(full_prompt)} chars | Model: {claude_model or 'default'}[/dim]")
            
            # Launch directly (no shell) with stdin pipe for the prompt
            # and stdout/stderr piped for capture
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.workspace_path,
            )
            
            # Send prompt via stdin and collect output
            output_chunks = []
            start_time = time.time()
            
            # Write prompt to stdin, then close it
            proc.stdin.write(full_prompt.encode('utf-8'))
            await proc.stdin.drain()
            proc.stdin.close()
            
            # Read output line by line with live display
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    # Check for interrupt
                    from .interrupt import is_interrupted
                    if is_interrupted():
                        self._kill_proc(proc)
                        self.console.print(f"\n[yellow][STOP] create_plan interrupted[/yellow]")
                        partial = "\n".join(output_chunks)
                        return f"create_plan interrupted after {time.time() - start_time:.0f}s.\nPartial output:\n{partial}"
                    continue
                
                if not line:
                    break
                
                decoded = line.decode('utf-8', errors='replace').rstrip()
                output_chunks.append(decoded)
                safe_line = sanitize_terminal_output(decoded)
                self.console.print(f"[dim]  {safe_line}[/dim]")
            
            await proc.wait()
            elapsed = time.time() - start_time
            exit_code = proc.returncode
            
            self.console.print(f"[bold yellow]{'─' * 4} create_plan done {'─' * 32}[/bold yellow]")
            
            raw_output = "\n".join(output_chunks)
            
            if exit_code != 0:
                log.warning("create_plan failed: exit=%d elapsed=%.1fs output_len=%d",
                           exit_code, elapsed, len(raw_output))
                self.console.print(f"[red][X] Claude CLI exit code: {exit_code}[/red]")
                return f"Error: Claude CLI failed (exit code {exit_code}).\nOutput:\n{raw_output}" if raw_output else f"Error: Claude CLI failed (exit code {exit_code}, no output)"
            
            log.info("create_plan success: elapsed=%.1fs output_len=%d", elapsed, len(raw_output))
            
            if not raw_output.strip():
                return "Error: Claude CLI returned empty output."
            
            # Save plan to persistent file for audit trail
            plan_dir = os.path.join(self._output_dir, "plans")
            os.makedirs(plan_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            plan_path = os.path.join(plan_dir, f"plan_{ts}.md")
            Path(plan_path).write_text(
                f"# create_plan — {ts}\n\n"
                f"**Model**: {claude_model or 'default'}\n"
                f"**Elapsed**: {elapsed:.1f}s\n\n"
                f"## Prompt\n\n{full_prompt}\n\n"
                f"## Response\n\n{raw_output}\n",
                encoding='utf-8'
            )
            log.info("Plan saved to: %s", plan_path)
            self.console.print(f"[dim]  Plan saved: {plan_path}[/dim]")
            
            # Truncate for context if very large
            output = truncate_output(raw_output, max_lines=300, keep_start=80, keep_end=80)
            output = self.spill_output_to_file(output, "create_plan")

            # Prepend plan file path so the agent can read the full output
            output = f"Plan saved to: {plan_path}\nRead this file to see exactly what the planner did before taking any action.\n\n{output}"

            # Add to context
            if len(output_chunks) > 3:
                ctx_id = self.context.add("command_output", "create_plan", output)
                return f"[Context ID: {ctx_id}]\n{output}"
            return output
            
        finally:
            # Clean up temp prompt file
            if prompt_file and os.path.exists(prompt_file.name):
                try:
                    os.unlink(prompt_file.name)
                except Exception:
                    pass
    
    async def _run_background_command(self, command: str) -> str:
        """Run a command in background with output redirected to a log file."""
        log.info("_run_background_command: cmd=%s", log_truncate(command, 120))
        if not command.strip():
            return "Error: execute_command requires a non-empty <command> parameter."
        
        proc_id = self._next_bg_id
        self._next_bg_id += 1
        log_path = self._get_bg_log_path(proc_id)

        # Truncate before launch to avoid stale content from previous sessions
        Path(log_path).write_text("", encoding="utf-8")

        # Shell-level redirect to log file (no cmd /c wrapper — see execute_command)
        wrapped = self._wrap_shell_command(command, log_path)
        
        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.workspace_path,
        )
        
        # Brief pause to capture initial output
        await asyncio.sleep(0.5)
        initial_output = []
        try:
            data = Path(log_path).read_text(encoding="utf-8", errors="replace")
            initial_output = data.splitlines()[-10:]
            for line in initial_output:
                safe_line = sanitize_terminal_output(line)
                self.console.print(f"[dim]  {safe_line}[/dim]")
        except Exception:
            pass
        
        self._background_procs[proc_id] = {
            "proc": proc,
            "command": command,
            "started": time.time(),
            "logs": initial_output.copy(),
            "log_file": log_path,
            "task": asyncio.create_task(self._background_log_tailer(proc_id, proc, log_path)),
        }
        
        self.console.print(f"    [green]→ Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
        return (
            f"Command started in background (ID: {proc_id}, PID: {proc.pid}).\n"
            f"Log file: {log_path}\n"
            f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
            f"Use check_background_process for status and recent output.\n"
            f"Initial output:\n" + "\n".join(initial_output)
        )
    
    
    async def list_files(self, params: Dict[str, str]) -> str:
        """List files in a directory."""
        path = self._resolve_path(params.get("path", "."))
        recursive = params.get("recursive", "").lower() == "true"
        
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
        path = self._resolve_path(params.get("path", "."))
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
    
    async def check_background_process(self, params: Dict[str, str]) -> str:
        """Check status and logs of a background process."""
        bg_id_str = params.get("id", "")
        lines = int(params.get("lines", "50"))
        
        try:
            bg_id = int(bg_id_str)
        except ValueError:
            # List all if no ID given
            procs = self.list_background_procs()
            if not procs:
                return "No background processes running."
            result = "Background processes:\n"
            for p in procs:
                elapsed_min = p['elapsed'] / 60
                result += f"  [{p['id']}] PID {p['pid']} - {p['status']} - {elapsed_min:.1f}m - {p['command']}\n"
            result += "\nUse check_background_process with id parameter to see logs."
            return result
        
        if bg_id not in self._background_procs:
            return f"Error: No background process with ID {bg_id}"
        
        info = self._background_procs[bg_id]
        proc = info["proc"]
        elapsed = time.time() - info["started"]
        # Poll to get updated returncode
        if proc.returncode is None:
            try:
                if proc._transport is not None and proc._transport.get_returncode() is not None:
                    proc._returncode = proc._transport.get_returncode()
            except Exception:
                pass
        status = "running" if proc.returncode is None else f"exited (code {proc.returncode})"
        logs = info.get("logs", [])
        
        # Get last N lines
        recent_logs = logs[-lines:] if logs else []
        
        log_file = info.get("log_file", "")
        
        result = f"Background Process [{bg_id}]\n"
        result += f"Command: {info['command']}\n"
        result += f"PID: {proc.pid}\n"
        result += f"Status: {status}\n"
        result += f"Running time: {elapsed:.0f}s\n"
        result += f"Total log lines (in memory): {len(logs)}\n"
        if log_file:
            result += f"Full log file: {log_file}\n"
            result += f"(Use read_file on this path to inspect the full output at any time)\n"
        result += f"\n--- Last {len(recent_logs)} lines ---\n"
        result += "\n".join(recent_logs) if recent_logs else "(no output yet)"
        
        # Add guidance to prevent spam checking
        if proc.returncode is None:
            if not recent_logs or len(logs) == info.get('_last_check_lines', 0):
                result += "\n\n[!] Process still running with no new output. Continue with other tasks instead of re-checking immediately."
            info['_last_check_lines'] = len(logs)
        
        return result
    
    async def stop_background_process(self, params: Dict[str, str]) -> str:
        """Stop a background process by ID."""
        bg_id_str = params.get("id", "")
        
        try:
            bg_id = int(bg_id_str)
        except ValueError:
            return "Error: ID must be a number"
        
        if bg_id not in self._background_procs:
            return f"Error: No background process with ID {bg_id}"
        
        info = self._background_procs[bg_id]
        proc = info["proc"]
        
        if proc.returncode is not None:
            return f"Process [{bg_id}] already exited with code {proc.returncode}"
        
        kill_process_tree(proc.pid)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass
        
        # Cancel log tailer
        if "task" in info and info["task"]:
            info["task"].cancel()
        
        return f"Stopped background process [{bg_id}] (PID: {proc.pid})"
    
    async def list_background_processes(self, params: Dict[str, str]) -> str:
        """List all background processes."""
        procs = self.list_background_procs()
        if not procs:
            return "No background processes."
        
        result = "Background processes:\n"
        for p in procs:
            elapsed_min = p['elapsed'] / 60
            result += f"  [{p['id']}] PID {p['pid']} - {p['status']} - {elapsed_min:.1f}m - {p['command']}\n"
        return result
    
    async def analyze_image(self, params: Dict[str, str]) -> str:
        """Analyze an image using GLM-4.6V vision model via coding endpoint."""
        import base64
        import httpx
        from urllib.parse import urlparse
        
        path_str = params.get("path", "")
        question = params.get("question", "Describe this image in detail. Note any text, UI elements, errors, or important visual details.")
        
        path = self._resolve_path(path_str)
        if not path.exists():
            return f"Error: Image not found: {path}"
        
        # Check file extension
        ext = path.suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return f"Error: Unsupported image format: {ext}. Use jpg, png, gif, or webp."
        
        # Read and encode image as base64
        try:
            img_data = path.read_bytes()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            return f"Error reading image: {e}"
        
        # Determine mime type
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', 
                    '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/png')
        
        # Use the Coding endpoint which properly supports vision with base64
        # https://api.z.ai/api/coding/paas/v4/chat/completions
        parsed = urlparse(self.config.api_url)
        vision_url = f"{parsed.scheme}://{parsed.netloc}/api/coding/paas/v4/chat/completions"
        
        # OpenAI format with data URI base64
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
        
        payload = {
            "model": "glm-4.6v",  # Vision model
            "messages": vision_messages,
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(vision_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract OpenAI response format
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content:
                        # Add to context
                        ctx_id = self.context.add("image_analysis", str(path), content)
                        return f"[Context ID: {ctx_id}]\n\nImage: {path_str}\n\n{content}"
                    return "Vision model returned empty response."
                return f"Unexpected response format: {data}"
                
        except httpx.HTTPStatusError as e:
            return f"Error calling vision API: {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            return f"Error analyzing image: {e}"
    
    async def web_search(self, params: Dict[str, str]) -> str:
        """Search the web using Z.AI's built-in web search via chat completions."""
        import httpx
        from urllib.parse import urlparse
        
        query = params.get("query", "")
        if not query:
            return "Error: search query is required"
        
        count = int(params.get("count", "5"))
        count = max(1, min(10, count))  # Clamp to 1-10
        
        # Use chat completion with web_search tool enabled
        parsed = urlparse(self.config.api_url)
        search_url = f"{parsed.scheme}://{parsed.netloc}/api/coding/paas/v4/chat/completions"
        
        payload = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": f"Search the web for: {query}"}],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": False,
            "tools": [{
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search-prime",
                    "search_result": True,
                    "count": str(count),
                }
            }]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept-Language": "en-US,en"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(search_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("web_search", [])
                content = ""
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")
                
                if not results and not content:
                    return f"No results found for: {query}"
                
                # Format results
                output = [f"Web Search Results for: {query}\n"]
                
                if results:
                    output.append(f"Found {len(results)} sources:\n")
                    for i, r in enumerate(results, 1):
                        title = r.get("title", "No title")
                        link = r.get("link", "")
                        media = r.get("media", "")
                        date = r.get("publish_date", "")
                        snippet = r.get("content", "")[:200]
                        
                        output.append(f"[{i}] {title}")
                        if media:
                            output.append(f"    Source: {media}")
                        if date:
                            output.append(f"    Date: {date}")
                        if snippet:
                            output.append(f"    {snippet}...")
                        if link:
                            output.append(f"    URL: {link}")
                        output.append("")
                
                if content:
                    output.append(f"\nSummary:\n{content}")
                
                result_text = "\n".join(output)
                
                # Add to context
                ctx_id = self.context.add("web_search", query, result_text)
                return f"[Context ID: {ctx_id}]\n\n{result_text}"
                
        except httpx.HTTPStatusError as e:
            return f"Error calling search API: {e.response.status_code} - {e.response.text[:200]}"
        except httpx.TimeoutException:
            return f"Error: Search request timed out after 120 seconds"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"Error searching web: {type(e).__name__}: {e}\n{tb}"
    
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
