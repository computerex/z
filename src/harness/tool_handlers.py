"""Tool handlers for ClineAgent - manages all tool execution logic."""

import asyncio
import os
import platform
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console

from .context_management import truncate_file_content, truncate_output


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
        duplicate_detector
    ):
        """Initialize tool handlers with required dependencies.
        
        Args:
            config: Config object with API settings
            console: Rich console for output
            workspace_path: Path to workspace directory
            context: ContextContainer for managing loaded content
            duplicate_detector: DuplicateDetector for tracking file reads
        """
        self.config = config
        self.console = console
        self.workspace_path = workspace_path
        self.context = context
        self._duplicate_detector = duplicate_detector
        
        # Background processes: {id: {"proc": Process, "command": str, "started": float, "log_file": str, "task": Task}}
        self._background_procs: Dict[int, dict] = {}
        self._next_bg_id = 1
        
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
        """Get the log file path for a background process.
        
        Ensures the output directory exists (it may have been removed
        between __init__ and now, e.g. by a clean command or .gitignore).
        """
        os.makedirs(self._output_dir, exist_ok=True)
        return os.path.join(self._output_dir, f"bg_process_{proc_id}.log")
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.workspace_path) / p
        return p.resolve()
    
    async def _background_log_reader(self, bg_id: int, proc: asyncio.subprocess.Process):
        """Continuously read output from a background process and append to log file."""
        info = self._background_procs.get(bg_id)
        if not info:
            return
        
        log_path = info.get("log_file", self._get_bg_log_path(bg_id))
        
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_fh:
                while True:
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                    except asyncio.TimeoutError:
                        if proc.returncode is not None:
                            break
                        continue
                    
                    if not line:
                        break
                    
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    log_fh.write(decoded + "\n")
                    log_fh.flush()
                    # Keep last 200 lines in memory for quick checks
                    info["logs"].append(decoded)
                    if len(info["logs"]) > 200:
                        info["logs"] = info["logs"][-200:]
        except Exception:
            pass
        
        # Ensure returncode is populated once stdout closes
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass
    
    async def cleanup_background_procs_async(self) -> None:
        """Async version - properly waits for processes to terminate."""
        for pid, info in list(self._background_procs.items()):
            try:
                proc = info["proc"]
                if proc.returncode is None:
                    # On Windows, use taskkill to kill the entire process tree
                    if platform.system() == "Windows":
                        try:
                            os.system(f'taskkill /F /T /PID {proc.pid} >nul 2>&1')
                        except:
                            pass
                    try:
                        proc.terminate()
                    except:
                        pass
                    # Wait for process to finish (properly closes transports)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        try:
                            proc.kill()
                            await asyncio.wait_for(proc.wait(), timeout=1.0)
                        except:
                            pass
                # Cancel the log reader task
                if "task" in info and info["task"]:
                    try:
                        info["task"].cancel()
                    except:
                        pass
            except Exception:
                pass
        self._background_procs.clear()
    
    def cleanup_background_procs(self) -> None:
        """Terminate all background processes safely (sync wrapper)."""
        # Try to use async cleanup if event loop is available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule async cleanup
                asyncio.create_task(self.cleanup_background_procs_async())
            else:
                # Run async cleanup
                loop.run_until_complete(self.cleanup_background_procs_async())
        except RuntimeError:
            # No event loop - do basic sync cleanup
            for pid, info in list(self._background_procs.items()):
                try:
                    proc = info["proc"]
                    if proc.returncode is None:
                        if platform.system() == "Windows":
                            os.system(f'taskkill /F /T /PID {proc.pid} >nul 2>&1')
                        try:
                            proc.terminate()
                        except:
                            pass
                    if "task" in info and info["task"]:
                        try:
                            info["task"].cancel()
                        except:
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
        
        if not path.exists():
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
        
        # DEBUG: Log write operation
        if os.environ.get("HARNESS_DEBUG"):
            print(f"[DEBUG write_file] path={path}, content_len={len(content)}", flush=True)
        
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
        
        if not path.exists():
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
        """Execute a shell command with live output display and interrupt support."""
        from .interrupt import is_interrupted, is_background_requested, reset_background
        
        import subprocess
        command = params.get("command", "")
        background = params.get("background", "").lower() == "true"
        timeout_secs = 120  # Auto-background after this many seconds
        
        # Show command being executed
        print()
        mode_indicator = "[bg] " if background else ""
        self.console.print(f"[dim]$ {mode_indicator}{command}[/dim]")
        
        if background:
            return await self._run_background_command(command)
        
        # Windows: create new process group to isolate from child process chaos
        # This prevents multiprocessing fork bombs from taking down the harness
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        # Foreground execution with interrupt/background/timeout support
        # IMPORTANT: stdin=DEVNULL prevents interactive commands from stealing
        # keyboard input, which would make Esc/Ctrl+C not work for interrupting.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.workspace_path,
            creationflags=creationflags,
        )
        
        output_lines = []
        start_time = time.time()
        hint_shown = False
        consecutive_timeouts = 0  # Track idle readline timeouts after process exit
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                # Check for interrupt (Esc)
                if is_interrupted():
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                    self.console.print(f"\n[yellow][STOP] Command interrupted[/yellow]")
                    raw_output = "\n".join(output_lines)
                    output = self.spill_output_to_file(raw_output, f"interrupted_{command.split()[0] if command else 'cmd'}")
                    return f"Command interrupted after {elapsed:.0f}s.\nOutput captured:\n{output}" if output else "Command interrupted (no output)"
                
                # Check for background request (Ctrl+B)
                if is_background_requested():
                    reset_background()  # Reset flag so next command doesn't also go to background
                    self.console.print(f"\n[cyan]-> Sending to background...[/cyan]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    log_path = self._get_bg_log_path(proc_id)
                    # Write existing output to log file
                    Path(log_path).write_text("\n".join(output_lines) + "\n", encoding="utf-8")
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "log_file": log_path,
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    output = "\n".join(output_lines[-30:])
                    return (
                        f"Command sent to background (ID: {proc_id}, PID: {proc.pid}).\n"
                        f"Log file: {log_path}\n"
                        f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
                        f"Output so far:\n{output}"
                    )
                
                # Show hint after 5 seconds
                if elapsed > 5 and not hint_shown:
                    self.console.print(f"[dim]  (Ctrl+B to background, Esc to stop)[/dim]")
                    hint_shown = True
                
                # Auto-background after timeout
                if elapsed > timeout_secs:
                    self.console.print(f"\n[yellow][TIME] Command running for {timeout_secs}s - auto-backgrounding[/yellow]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    log_path = self._get_bg_log_path(proc_id)
                    # Write existing output to log file
                    Path(log_path).write_text("\n".join(output_lines) + "\n", encoding="utf-8")
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "log_file": log_path,
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    raw_output = "\n".join(output_lines)
                    output = self.spill_output_to_file(raw_output, f"autobg_{command.split()[0] if command else 'cmd'}")
                    return (
                        f"Command auto-backgrounded after {timeout_secs}s (ID: {proc_id}, PID: {proc.pid}).\n"
                        f"Log file: {log_path}\n"
                        f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
                        f"Output captured:\n{output}"
                    ) if output else f"Command auto-backgrounded (ID: {proc_id}, no output yet).\nLog file: {log_path}"
                
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                    consecutive_timeouts = 0  # Reset on successful read
                except asyncio.TimeoutError:
                    # On Windows with piped commands (e.g. cmd1 | findstr),
                    # the pipe may never deliver EOF even after the process
                    # exits, causing this loop to spin forever.  Detect this
                    # by counting consecutive timeouts after the process has
                    # already exited and breaking out.
                    if proc.returncode is not None:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= 3:  # 3 × 0.1s = 0.3s grace
                            # Drain any remaining buffered data
                            try:
                                remaining = await asyncio.wait_for(
                                    proc.stdout.read(65536), timeout=0.2
                                )
                                if remaining:
                                    for chunk_line in remaining.decode("utf-8", errors="replace").splitlines():
                                        output_lines.append(chunk_line)
                                        safe_line = sanitize_terminal_output(chunk_line)
                                        self.console.print(f"[dim]  {safe_line}[/dim]")
                            except (asyncio.TimeoutError, Exception):
                                pass
                            break
                    else:
                        # Process still running — just poll in case it exited
                        # between the readline timeout and now.
                        try:
                            ret = proc.returncode  # non-blocking check
                            if ret is None and hasattr(proc, '_transport') and proc._transport:
                                rc = proc._transport.get_returncode()
                                if rc is not None:
                                    proc._returncode = rc
                        except Exception:
                            pass
                    continue
                    
                if not line:
                    break
                    
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                # Sanitize before printing to prevent terminal escape sequences
                # in binary output from triggering terminal responses that the
                # keyboard monitor would misinterpret as user input (e.g. Escape)
                safe_line = sanitize_terminal_output(decoded)
                self.console.print(f"[dim]  {safe_line}[/dim]")
            
            await proc.wait()
            exit_code = proc.returncode
            
            if exit_code == 0:
                self.console.print(f"[green][OK] Exit code: {exit_code}[/green]")
            else:
                self.console.print(f"[red][X] Exit code: {exit_code}[/red]")
            
            # Build raw output and spill to file if huge
            raw_output = "\n".join(output_lines) or "(no output)"
            
            # First apply line-level truncation for sanity
            output = truncate_output(raw_output, max_lines=300, keep_start=80, keep_end=80)
            
            # Then spill to file if still too large for context
            output = self.spill_output_to_file(output, f"cmd_{command.split()[0] if command else 'cmd'}")
            
            # Add to context if significant output
            if len(output_lines) > 3:
                ctx_id = self.context.add("command_output", command, output)
                return f"[Context ID: {ctx_id}]\n{output}"
            return output
            
        except Exception as e:
            proc.kill()
            raw_output = "\n".join(output_lines)
            output = self.spill_output_to_file(raw_output, "cmd_error") if raw_output else ""
            return f"Error: {str(e)}\nOutput captured:\n{output}" if output else f"Error: {str(e)}"
    
    async def _run_background_command(self, command: str) -> str:
        """Run a command in background with output streamed to a log file."""
        import subprocess
        
        # Windows: create new process group to isolate from child process chaos
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.workspace_path,
            creationflags=creationflags,
        )
        
        # Store background process with log buffer
        proc_id = self._next_bg_id
        self._next_bg_id += 1
        
        # Create log file for this background process
        log_path = self._get_bg_log_path(proc_id)
        
        # Wait briefly to capture initial output
        output_lines = []
        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                for _ in range(20):
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    output_lines.append(decoded)
                    log_fh.write(decoded + "\n")
                    safe_line = sanitize_terminal_output(decoded)
                    self.console.print(f"[dim]  {safe_line}[/dim]")
        except asyncio.TimeoutError:
            pass
        
        # Store process and start log reader
        self._background_procs[proc_id] = {
            "proc": proc, 
            "command": command, 
            "started": time.time(),
            "logs": output_lines.copy(),
            "log_file": log_path,
            "task": None
        }
        # Start background log reader (appends to log file)
        task = asyncio.create_task(self._background_log_reader(proc_id, proc))
        self._background_procs[proc_id]["task"] = task
        
        self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
        return (
            f"Command started in background (ID: {proc_id}, PID: {proc.pid}).\n"
            f"Log file: {log_path}\n"
            f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
            f"Use check_background_process for status and recent output.\n"
            f"Initial output:\n" + "\n".join(output_lines[-10:])
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
        
        try:
            proc.terminate()
            # Wait briefly for graceful termination
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
        
        # Cancel log reader
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
