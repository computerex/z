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
        
        # Background processes: {id: {"proc": Process, "command": str, "started": float, "logs": list, "task": Task}}
        self._background_procs: Dict[int, dict] = {}
        self._next_bg_id = 1
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.workspace_path) / p
        return p.resolve()
    
    async def _background_log_reader(self, bg_id: int, proc: asyncio.subprocess.Process):
        """Continuously read output from a background process."""
        info = self._background_procs.get(bg_id)
        if not info:
            return
        
        try:
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
                # Keep last 500 lines
                info["logs"].append(decoded)
                if len(info["logs"]) > 500:
                    info["logs"] = info["logs"][-500:]
        except Exception:
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
    
    async def read_file(self, params: Dict[str, str]) -> str:
        """Read a file and return its contents."""
        path = self._resolve_path(params.get("path", ""))
        
        if not path.exists():
            return f"Error: File not found: {path}"
        
        rel_path = str(path.relative_to(self.workspace_path)) if str(path).startswith(self.workspace_path) else str(path)
        
        # Check for duplicate reads - replace older ones with notice
        prev_index = self._duplicate_detector.was_read_before(rel_path)
        if prev_index is not None:
            from .context_management import DuplicateDetector
            replaced = DuplicateDetector.replace_old_reads([], rel_path, 0)
            if replaced > 0:
                self.console.print(f"[dim]   (replaced {replaced} older read(s) with notice)[/dim]")
        
        # Record this read
        self._duplicate_detector.record_read(rel_path, 0)
        
        content = path.read_text(encoding="utf-8", errors="replace")
        
        # Truncate if file is too large
        content = truncate_file_content(content)
        
        # Add line numbers
        lines = content.splitlines()
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        result = "\n".join(numbered)
        
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
        """Replace sections of content in an existing file."""
        path = self._resolve_path(params.get("path", ""))
        diff = params.get("diff", "")
        
        if not path.exists():
            return f"Error: File not found: {path}"
        
        content = path.read_text(encoding="utf-8")
        blocks = parse_search_replace_blocks(diff)
        
        if not blocks:
            return "Error: No valid SEARCH/REPLACE blocks found"
        
        def normalize_whitespace(s: str) -> str:
            """Normalize line endings and trailing whitespace for matching."""
            lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            return '\n'.join(line.rstrip() for line in lines)
        
        changes = 0
        for search, replace in blocks:
            # Try exact match first
            if search in content:
                content = content.replace(search, replace, 1)
                changes += 1
            else:
                # Try with normalized whitespace
                norm_content = normalize_whitespace(content)
                norm_search = normalize_whitespace(search)
                
                if norm_search in norm_content:
                    # Find position in normalized content, apply to original
                    # Replace line by line with whitespace tolerance
                    search_lines = norm_search.split('\n')
                    content_lines = content.replace('\r\n', '\n').split('\n')
                    
                    # Find starting line
                    for i in range(len(content_lines) - len(search_lines) + 1):
                        match = True
                        for j, search_line in enumerate(search_lines):
                            if content_lines[i + j].rstrip() != search_line:
                                match = False
                                break
                        if match:
                            # Found it - replace those lines
                            replace_lines = replace.replace('\r\n', '\n').split('\n')
                            content_lines = content_lines[:i] + replace_lines + content_lines[i + len(search_lines):]
                            content = '\n'.join(content_lines)
                            changes += 1
                            break
                    else:
                        return f"Error: SEARCH block not found in file (even with whitespace normalization):\n{search[:200]}..."
                else:
                    # Show helpful diff for debugging
                    lines = search.split('\n')[:3]
                    return f"Error: SEARCH block not found in file:\n{chr(10).join(lines)}...\n\nTip: Check for whitespace/indentation differences."
        
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
                    output = "\n".join(output_lines)[:15000]
                    return f"Command interrupted after {elapsed:.0f}s.\nOutput captured:\n{output}" if output else "Command interrupted (no output)"
                
                # Check for background request (Ctrl+B)
                if is_background_requested():
                    reset_background()  # Reset flag so next command doesn't also go to background
                    self.console.print(f"\n[cyan]-> Sending to background...[/cyan]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    output = "\n".join(output_lines[-30:])
                    return f"Command sent to background (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nOutput so far:\n{output}"
                
                # Show hint after 5 seconds
                if elapsed > 5 and not hint_shown:
                    self.console.print(f"[dim]  (Ctrl+B to background, Esc to stop)[/dim]")
                    hint_shown = True
                
                # Auto-background after timeout
                if elapsed > timeout_secs:
                    self.console.print(f"\n[yellow][TIME] Command running for {timeout_secs}s - auto-backgrounding[/yellow]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    output = "\n".join(output_lines)[:15000]
                    return f"Command auto-backgrounded after {timeout_secs}s (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nOutput captured:\n{output}" if output else f"Command auto-backgrounded (ID: {proc_id}, PID: {proc.pid}, no output yet)"
                
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                    
                if not line:
                    break
                    
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                self.console.print(f"[dim]  {decoded}[/dim]")
            
            await proc.wait()
            exit_code = proc.returncode
            
            if exit_code == 0:
                self.console.print(f"[green][OK] Exit code: {exit_code}[/green]")
            else:
                self.console.print(f"[red][X] Exit code: {exit_code}[/red]")
            
            # Truncate long output (keep start and end)
            raw_output = "\n".join(output_lines) or "(no output)"
            output = truncate_output(raw_output, max_lines=200, keep_start=50, keep_end=50)
            
            # Add to context if significant output
            if len(output_lines) > 3:
                ctx_id = self.context.add("command_output", command, output)
                return f"[Context ID: {ctx_id}]\n{output}"
            return output
            
        except Exception as e:
            proc.kill()
            output = truncate_output("\n".join(output_lines), max_lines=150)
            return f"Error: {str(e)}\nOutput captured:\n{output}" if output else f"Error: {str(e)}"
    
    async def _run_background_command(self, command: str) -> str:
        """Run a command in background."""
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
        
        # Wait briefly to capture initial output
        output_lines = []
        try:
            for _ in range(20):
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                self.console.print(f"[dim]  {decoded}[/dim]")
        except asyncio.TimeoutError:
            pass
        
        # Store process and start log reader
        self._background_procs[proc_id] = {
            "proc": proc, 
            "command": command, 
            "started": time.time(),
            "logs": output_lines.copy(),
            "task": None
        }
        # Start background log reader
        task = asyncio.create_task(self._background_log_reader(proc_id, proc))
        self._background_procs[proc_id]["task"] = task
        
        self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
        return f"Command started in background (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nInitial output:\n" + "\n".join(output_lines[-10:])
    
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
        status = "running" if proc.returncode is None else f"exited (code {proc.returncode})"
        logs = info.get("logs", [])
        
        # Get last N lines
        recent_logs = logs[-lines:] if logs else []
        
        result = f"Background Process [{bg_id}]\n"
        result += f"Command: {info['command']}\n"
        result += f"PID: {proc.pid}\n"
        result += f"Status: {status}\n"
        result += f"Running time: {elapsed:.0f}s\n"
        result += f"Total log lines: {len(logs)}\n"
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