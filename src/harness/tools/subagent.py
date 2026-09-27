"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SUBAGENT_TOOLS_UNAVAILABLE = (
    "Error: Sub-agent tools are not available from this context — "
    "this agent cannot manage sub-agents."
)

async def create_agent(self, params: dict) -> str:
    """Create a sub-agent and start it in the background."""
    name = params.get("name", "").strip()
    task = params.get("task", "").strip()
    if not name or not task:
        return "Error: Both 'name' (unique identifier) and 'task' (description) are required."
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    try:
        self.sub_agent_manager.create(name, task)
        self.console.print(
            f"  [green]\u2713[/green] Created sub-agent [bold]{name}[/bold]"
        )
        warning = ""
        try:
            live = len([
                1 for i in (self.sub_agent_manager.list())
                if i["status"] in ("running", "created")
            ])
            if live >= 8:
                warning = (
                    f" Note: {live} agents are now running; each consumes "
                    f"context and cost independently — consider whether all are needed."
                )
        except Exception:
            pass  # cost warning is best-effort
        return (
            f"Created sub-agent '{name}'. It is running in the background. "
            f"You will be notified when it completes.{warning}"
        )
    except ValueError as e:
        return f"Error: {e}"


async def send_agent_input(self, params: dict) -> str:
    """Send input to a sub-agent (multi-turn conversation).

    Idle agent → runs a turn and returns its response (synchronous chat).
    Busy agent → by default enqueues and returns a queued-ack immediately
    (never freezes the parent on a tool call); the result arrives via the
    completion notification → get_agent_output cycle. Pass wait=true to block
    until the queued entry's own turn completes.

    To just read a completed agent's output without starting a new turn, use
    get_agent_output(name).
    """
    name = params.get("name", "").strip()
    input_text = params.get("input", "").strip()
    wait = params.get("wait", False)
    if isinstance(wait, str):
        wait = wait.lower() in ("true", "1", "yes")
    wait = bool(wait)
    if not name or input_text is None:
        return "Error: Both 'name' and 'input' are required."
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    try:
        inst = self.sub_agent_manager.get(name)
        if not inst:
            return f"Error: Sub-agent '{name}' not found."

        self.console.print(
            f"  [dim]\u2192[/dim] Sending input to [bold]{name}[/bold]..."
        )
        result = await self.sub_agent_manager.run(name, input_text, wait=wait)
        return result
    except KeyError:
        return f"Error: Sub-agent '{name}' not found."
    except Exception as e:
        return f"Error communicating with sub-agent '{name}': {e}"


async def list_agents(self, params: dict) -> str:
    """List all sub-agents with current output."""
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    agents = self.sub_agent_manager.list()
    if not agents:
        return "No sub-agents running."

    # Optional name filter
    filter_name = params.get("name", "").strip()
    if filter_name:
        agents = [a for a in agents if a["name"] == filter_name]
        if not agents:
            return f"No sub-agent named '{filter_name}' found."

    lines = ["Sub-agents:"]
    for a in agents:
        elapsed = a["elapsed_seconds"]
        elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
        status_icon = {"running": "\u25b6", "completed": "\u2713", "error": "\u2717", "created": "\u25cb"}.get(a["status"], "?")
        lines.append(f"  {status_icon} {a['name']}: {a['status']} ({elapsed_str})")
        # Include output snippet if available
        output_snippet = a.get("output", "")
        if output_snippet:
            if a["status"] == "completed":
                # Full output for completed agents — parent needs the result
                display = output_snippet.replace("\n", "\n    ")
                lines.append(f"    Output:\n    {display}")
            else:
                # Truncated snippet for running agents
                display = output_snippet[:600].replace("\n", "\n    ")
                if len(output_snippet) > 600:
                    display += "\n    [...]"
                lines.append(f"    Output:\n    {display}")
    return "\n".join(lines)


async def pause_agent(self, params: dict) -> str:
    """Pause a running sub-agent."""
    name = params.get("name", "").strip()
    if not name:
        return "Error: 'name' is required."
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    if self.sub_agent_manager.pause(name):
        self.console.print(f"  [yellow]\u23f8[/yellow] Paused sub-agent [bold]{name}[/bold]")
        return f"Sub-agent '{name}' paused."
    return f"Error: Sub-agent '{name}' not found."


async def get_agent_output(self, params: dict) -> str:
    """Retrieve a completed sub-agent's full output without sending new input."""
    name = params.get("name", "").strip()
    if not name:
        return "Error: 'name' is required."
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    try:
        inst = self.sub_agent_manager.get(name)
        if not inst:
            return f"Error: Sub-agent '{name}' not found."

        # Consume-on-fetch: drain any results recorded since the last fetch
        # first, so overlapping turn completions are all retrievable.
        fetch = getattr(self.sub_agent_manager, "fetch_new_results", None)
        new_results = fetch(name) if fetch else []
        if new_results:
            parts = []
            for r in new_results:
                if getattr(r, "status", "completed") == "error":
                    parts.append(
                        f"[Turn {r.turn_id} FAILED: {r.error}]\n{r.text}"
                    )
                elif len(new_results) > 1:
                    parts.append(f"[Turn {r.turn_id} result]:\n{r.text}")
                else:
                    parts.append(r.text)
            return "\n\n".join(parts)

        # Terminal states (completed or error): return the cached verdict.
        if inst.status in ("completed", "error"):
            if inst.status == "error" and inst.last_error:
                err_line = f"Sub-agent '{name}' failed: {inst.last_error}"
                if inst.final_result:
                    return f"{err_line}\n{inst.final_result}"
                return err_line
            # Return the completed task's final result, not the diagnostic
            # terminal transcript. This preserves the agent's verdict even
            # when the verbose stream output is huge or contains terminal
            # control sequences.
            if inst.final_result:
                return inst.final_result
            output_text = inst.output or ""
            if not output_text and inst.tee:
                output_text = inst.tee.getvalue() or ""
            if not output_text:
                return f"Sub-agent '{name}' completed but produced no output."
            return output_text

        if inst.task and inst.task.done():
            # Race condition: the background task may have finished writing
            # to the TeeWriter but instance.status hasn't been updated yet.
            output_text = inst.tee.getvalue() if inst.tee else ""
            if output_text:
                # Task is done, output exists — return it
                return output_text
        # Still running: return a readable tail of the partial output so
        # the caller can check live progress (stripped of terminal
        # styling — the tee capture contains ANSI codes).
        partial = ""
        if inst.tee:
            try:
                from ..sub_agent_manager import strip_ansi
                partial = strip_ansi(inst.tee.getvalue() or "")
            except Exception:
                partial = ""
        if partial:
            partial = partial[-1500:].strip()
            return (
                f"Sub-agent '{name}' is still {inst.status}. "
                f"Recent output (tail):\n{partial}\n"
                f"[Check again later for the final result.]"
            )
        return f"Sub-agent '{name}' is still {inst.status} (no output yet). Use list_agents(name='{name}') to check its progress."
    except Exception as e:
        return f"Error retrieving output from '{name}': {e}"


async def delete_agent(self, params: dict) -> str:
    """Delete a sub-agent completely."""
    name = params.get("name", "").strip()
    if not name:
        return "Error: 'name' is required."
    if not self.sub_agent_manager:
        return _SUBAGENT_TOOLS_UNAVAILABLE
    if self.sub_agent_manager.delete(name):
        self.console.print(f"  [red]\u2717[/red] Deleted sub-agent [bold]{name}[/bold]")
        return f"Sub-agent '{name}' deleted."
    return f"Error: Sub-agent '{name}' not found."
