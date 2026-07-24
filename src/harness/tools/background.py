"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from ._base import kill_process_tree, _detect_log_file_encoding
from .mcp import _close_mcp_session

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
    encoding = _detect_log_file_encoding(log_path)
    try:
        while proc.returncode is None:
            try:
                with open(log_path, "r", encoding=encoding, errors="replace") as f:
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
                with open(log_path, "r", encoding=encoding, errors="replace") as f:
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

    # Close persistent MCP sessions on shutdown.
    for sname in list(self._mcp_sessions.keys()):
        try:
            await _close_mcp_session(self, sname)
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

