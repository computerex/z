"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import subprocess
import signal
import platform

async def execute_command(self, params: Dict[str, str]) -> str:
    """Execute a shell command with live output display and interrupt support.

    All commands are launched with stdout/stderr redirected to a log file
    (not piped to Python).  An async tail loop reads the log file for live
    console output.  This unified approach means GUI applications can create
    windows (stdout is not captured via pipe) while CLI tools still get
    their output displayed and recorded.
    """
    from ..interrupt import is_interrupted, is_background_requested, reset_background

    command = params.get("command", "")
    background = params.get("background", False); background = bool(background) if isinstance(background, (bool, int)) else str(background).lower() == "true"
    timeout_secs = 120  # Auto-background after this many seconds
    if not command.strip():
        return "Error: execute_command requires a non-empty <command> parameter."
    log.info("execute_command: cmd=%s bg=%s", log_truncate(command, 200), background)

    mode_indicator = "[bg] " if background else ""
    cmd_short = command.split('\n')[0][:120]  # First line, truncated
    self.console.print(f"  [dim]•[/dim] [bold]Running[/bold] [dim]{mode_indicator}{cmd_short}[/dim]")

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
                self.console.print(f"    [yellow]interrupted[/yellow]")
                raw_output = self._read_log_file(cmd_log_path)
                output = self.spill_output_to_file(
                    raw_output, f"interrupted_{command.split()[0] if command else 'cmd'}")
                return f"Command interrupted after {elapsed:.0f}s.\nOutput captured:\n{output}" if output else "Command interrupted (no output)"

            # Check for background request (Ctrl+B)
            if is_background_requested():
                reset_background()
                self.console.print(f"    [cyan]→ background[/cyan]")
                return self._promote_to_background(proc, command, start_time, cmd_log_path, output_lines)

            # Show hint after 5 seconds
            if elapsed > 5 and not hint_shown:
                self.console.print(f"    [dim]Ctrl+B background · Esc stop[/dim]")
                hint_shown = True

            # Auto-background after timeout.  Use a shorter timeout for
            # commands producing no output — likely GUI apps (e.g. notepad)
            # that block cmd.exe but have no console output to tail.
            effective_timeout = 10 if not output_lines else timeout_secs
            if elapsed > effective_timeout:
                self.console.print(f"    [cyan]→ background[/cyan] [dim](no output for {elapsed:.0f}s)[/dim]")
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

        # Detached GUI app detection: a detached child (e.g. a Windows GUI
        # subsystem app launched via cmd.exe) makes the shell exit fast while
        # producing NO console output, then may write to the log via inherited
        # handles.  We only enter this check when the fast exit produced *zero*
        # output so far; if new lines then appear during a brief wait, a
        # detached child is alive and we promote to background.  A normal CLI
        # script that printed its result and exited (e.g. `python calc.py` →
        # "5") already has output and must NOT be misclassified.
        elapsed_so_far = time.time() - start_time
        if elapsed_so_far < 2.0 and len(output_lines) == 0:
            log.info("Fast exit with no output (%.1fs) — checking for detached GUI app",
                     elapsed_so_far)
            await asyncio.sleep(1.0)
            file_pos = self._tail_log_file(cmd_log_path, file_pos, output_lines)
            if output_lines:
                # New output appeared after the shell exited — a detached app
                # is still running.  Promote so the log tailer keeps reading.
                self.console.print(f"    [cyan]→ background[/cyan] [dim](detached process)[/dim]")
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
            # Force UTF-8 output to avoid UTF-16LE encoding issues in log files
            # (especially for WSL commands which output UTF-8 that PowerShell
            # would otherwise encode as UTF-16LE).
            ps_command = (
                "$ProgressPreference='SilentlyContinue'; "
                "$OutputEncoding = [System.Text.Encoding]::UTF8; "
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                + command
            )
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
        # Auto-detect encoding: check for UTF-16LE BOM or null byte patterns
        encoding = _detect_log_file_encoding(log_path)
        with open(log_path, "r", encoding=encoding, errors="replace") as f:
            f.seek(file_pos)
            new_data = f.read()
            new_pos = f.tell()
        if new_data:
            # Decode any CLIXML blocks before splitting into lines so
            # multi-line XML fragments are cleaned up as a unit.
            new_data = _decode_clixml_in_text(new_data)
            for line in new_data.splitlines():
                decoded_line = _decode_powershell_clixml(line)
                if not decoded_line.strip():
                    continue
                output_lines.append(decoded_line)
                n = len(output_lines)
                if n <= self._MAX_LIVE_DISPLAY:
                    safe_line = sanitize_terminal_output(decoded_line)
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
        encoding = _detect_log_file_encoding(log_path)
        raw = Path(log_path).read_text(encoding=encoding, errors="replace")
        cleaned = _decode_clixml_in_text(raw)
        decoded = _decode_powershell_clixml(cleaned)
        return decoded or raw
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
    self.console.print(f"    [green]→ background[/green] [dim](ID: {proc_id}, PID: {proc.pid})[/dim]")
    recent = "\n".join(output_lines[-30:])
    return (
        f"Command sent to background (ID: {proc_id}, PID: {proc.pid}).\n"
        f"Log file: {log_path}\n"
        f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
        f"Output so far:\n{recent}"
    )


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
        bg_encoding = _detect_log_file_encoding(log_path)
        data = Path(log_path).read_text(encoding=bg_encoding, errors="replace")
        data = _decode_clixml_in_text(data)
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

    self.console.print(f"    [green]→ background[/green] [dim](ID: {proc_id}, PID: {proc.pid})[/dim]")
    return (
        f"Command started in background (ID: {proc_id}, PID: {proc.pid}).\n"
        f"Log file: {log_path}\n"
        f"Use read_file on the log file to inspect stdout/stderr at any time.\n"
        f"Use check_background_process for status and recent output.\n"
        f"Initial output:\n" + "\n".join(initial_output)
    )


