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
                        # Monotonic counter — info["logs"] is capped at 200,
                        # so len(info["logs"]) cannot tell the incremental
                        # checker how many lines arrived since last delivery.
                        info["_total_lines"] = int(info.get("_total_lines", 0)) + 1
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
                        info["_total_lines"] = int(info.get("_total_lines", 0)) + 1
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
                # asyncio-owned child: let the loop's child watcher reap it
                kill_process_tree(proc.pid, reap_parent=False)
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
                # asyncio-owned children are reaped by the loop's child
                # watcher; a psutil reap here races with it (rc 255).
                kill_process_tree(proc.pid, reap_parent=False)
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


# Long-poll defaults for check_background_process(wait=...)
_BG_WAIT_DEFAULT_SECS = 120.0
_BG_WAIT_MAX_SECS = 300.0


def _parse_wait_secs(value: Any) -> float:
    """Tolerantly parse the ``wait`` param into seconds, clamped to [0, MAX].

    Accepts numbers, numeric strings, or bool-ish values ("true" → default).
    Anything unparsable means "don't wait" so a malformed call degrades to
    the plain (non-blocking) check instead of erroring.
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return _BG_WAIT_DEFAULT_SECS if value else 0.0
    try:
        return max(0.0, min(float(str(value).strip()), _BG_WAIT_MAX_SECS))
    except (TypeError, ValueError):
        return _BG_WAIT_DEFAULT_SECS if str(value).strip().lower() in ("true", "yes") else 0.0


async def check_background_process(self, params: Dict[str, str]) -> str:
    """Check status and output of a background process.

    Repeat checks are INCREMENTAL: only lines appended since the previous
    check are returned, and a check with nothing new returns a compact
    one-screen status instead of re-sending the log tail — repeated
    polling is nearly free instead of re-delivering ~50 identical lines.

    ``wait`` (seconds, optional) long-polls: the call blocks until new
    output arrives, the process exits, or the deadline expires.  This is
    the mechanism for "wait for stuff" — one blocking check instead of a
    tight poll loop.  The wait is interruptible (Esc / Ctrl+B) and capped
    at _BG_WAIT_MAX_SECS.

    Explicitly passing ``lines`` preserves the legacy full-tail behavior
    (re-send the last N lines regardless of the incremental cursor).
    """
    bg_id_str = params.get("id", "")
    explicit_lines = str(params.get("lines", "")).strip() != ""
    try:
        lines = int(float(params.get("lines", "50")))
    except (TypeError, ValueError):
        lines = 50
    lines = max(1, min(lines, 500))
    wait_secs = _parse_wait_secs(params.get("wait"))

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

    # Defensive cursor init (covers creation paths that predate tracking):
    # "_total_lines" = lines ever appended; "_delivered" = lines already
    # shown to the model.  New output = total - delivered.
    total_lines = int(info.get("_total_lines", len(info.get("logs", []))))
    info["_total_lines"] = total_lines
    delivered = int(info.setdefault("_delivered", total_lines))

    # ── Long-poll: block until there is something new to report ─────
    # Only waits when a plain check would return "no new output" (i.e. the
    # cursor is caught up and the process is still running); otherwise it
    # returns immediately with the pending content.
    if wait_secs > 0 and delivered >= total_lines and proc.returncode is None:
        from ..interrupt import is_interrupted, is_background_requested, reset_background
        from .shell import _owner_interrupts, _own_signal
        self.console.print(
            f"  [dim]•[/dim] [cyan]Waiting on process[/cyan] [dim]{bg_id} (up to {wait_secs:.0f}s)[/dim]")
        deadline = time.time() + wait_secs
        while time.time() < deadline and proc.returncode is None:
            await asyncio.sleep(0.5)
            total_lines = int(info.get("_total_lines", total_lines))
            if total_lines > delivered:
                break  # new output arrived — go report it
            # Escape hatches mirror the shell tool's gated interrupt checks
            # (global flags must not leak into background sub-agents).
            if _own_signal(self, "interrupt") or (
                _owner_interrupts(self) and is_interrupted()
            ):
                break
            if _owner_interrupts(self) and is_background_requested():
                reset_background()
                break
        total_lines = int(info.get("_total_lines", total_lines))

    elapsed = time.time() - info["started"]
    running = proc.returncode is None
    status = "running" if running else f"exited (code {proc.returncode})"
    logs = info.get("logs", [])
    log_file = info.get("log_file", "")

    # ── Legacy full-tail mode: explicit lines param re-sends the tail ──
    if explicit_lines:
        recent_logs = logs[-lines:] if logs else []
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
        info["_delivered"] = total_lines
        return result

    new_count = max(0, total_lines - delivered)

    # ── New output: deliver only the delta ──────────────────────────
    if new_count > 0:
        # logs holds at most the last 200 lines; slicing clamps to what
        # memory retained when output outpaced the checks.
        new_logs = logs[-new_count:]
        dropped = new_count - len(new_logs)
        result = (f"Background Process [{bg_id}] — {status}, {elapsed:.0f}s elapsed. "
                  f"{new_count} new line(s) since last check.\n")
        if log_file:
            result += f"Full log file: {log_file}\n"
        if dropped > 0:
            result += (f"(+{dropped} earlier line(s) not retained in memory — "
                       f"read the full log file for history)\n")
        result += "\n--- new output ---\n" + "\n".join(new_logs)
        info["_delivered"] = total_lines
        return result

    # ── Nothing new: keep the response tiny — never re-send the tail ──
    if running:
        return (
            f"Background Process [{bg_id}] — still running, {elapsed:.0f}s elapsed. "
            f"No new output since last check ({total_lines} line(s) so far; tail unchanged). "
            f"Do not re-check in a tight loop. To block until output arrives or the "
            f"process exits, call again with \"wait\": <seconds> (e.g. 120)."
        )
    # Exit is a status change, i.e. news — even with zero new lines.
    return (
        f"Background Process [{bg_id}] — exited (code {proc.returncode}) after "
        f"{elapsed:.0f}s. No new output since last check."
    )


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

    # asyncio-owned child: the loop's child watcher reaps it (see
    # kill_process_tree for the double-reap race this avoids)
    kill_process_tree(proc.pid, reap_parent=False)
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

