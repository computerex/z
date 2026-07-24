"""Tool implementations — see tools/__init__.py for the ToolHandlers class."""
import asyncio
import os
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

async def create_agent(self, params: dict) -> str:
    """Create a sub-agent and start it in the background."""
    name = params.get("name", "").strip()
    task = params.get("task", "").strip()
    if not name or not task:
        return "Error: Both 'name' (unique identifier) and 'task' (description) are required."
    if not self.sub_agent_manager:
        return "Error: Sub-agent system not initialized."
    try:
        self.sub_agent_manager.create(name, task)
        self.console.print(
            f"  [green]\u2713[/green] Created sub-agent [bold]{name}[/bold]"
        )
        return f"Created sub-agent '{name}'. It is running in the background. You will be notified when it completes."
    except ValueError as e:
        return f"Error: {e}"


async def send_agent_input(self, params: dict) -> str:
    """Send input to a sub-agent and get its response.

    If the sub-agent has already completed (no longer running), returns the
    cached output directly without re-running. To start a new conversation
    turn with a completed sub-agent, provide meaningful new input.
    """
    name = params.get("name", "").strip()
    input_text = params.get("input", "").strip()
    if not name or input_text is None:
        return "Error: Both 'name' and 'input' are required."
    if not self.sub_agent_manager:
        return "Error: Sub-agent system not initialized."
    try:
        inst = self.sub_agent_manager.get(name)
        if not inst:
            return f"Error: Sub-agent '{name}' not found."

        # If agent is completed and has output, return cached output
        # without re-running (avoids unnecessary API calls for retrieval).
        if inst.status == "completed" and inst.task and inst.task.done() and inst.output:
            return inst.output

        self.console.print(
            f"  [dim]\u2192[/dim] Sending input to [bold]{name}[/bold]..."
        )
        result = await self.sub_agent_manager.run(name, input_text)
        return result
    except KeyError:
        return f"Error: Sub-agent '{name}' not found."
    except Exception as e:
        return f"Error communicating with sub-agent '{name}': {e}"


async def list_agents(self, params: dict) -> str:
    """List all sub-agents with current output."""
    if not self.sub_agent_manager:
        return "Sub-agent system not available."
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
        return "Error: Sub-agent system not initialized."
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
        return "Error: Sub-agent system not initialized."
    try:
        inst = self.sub_agent_manager.get(name)
        if not inst:
            return f"Error: Sub-agent '{name}' not found."
        if inst.status != "completed":
            # Race condition: the background task may have finished writing
            # to the TeeWriter but instance.status hasn't been set to
            # "completed" yet. Check if the task actually finished.
            if inst.task and inst.task.done():
                output_text = inst.tee.getvalue() if inst.tee else ""
                if output_text:
                    # Task is done, output exists — return it
                    return output_text
            return f"Sub-agent '{name}' is still {inst.status}. Use list_agents(name='{name}') to check its progress."
        # Return the full cached output
        output_text = inst.output or ""
        if not output_text and inst.tee:
            output_text = inst.tee.getvalue() or ""
        if not output_text:
            return f"Sub-agent '{name}' completed but produced no output."
        return output_text
    except Exception as e:
        return f"Error retrieving output from '{name}': {e}"


async def delete_agent(self, params: dict) -> str:
    """Delete a sub-agent completely."""
    name = params.get("name", "").strip()
    if not name:
        return "Error: 'name' is required."
    if not self.sub_agent_manager:
        return "Error: Sub-agent system not initialized."
    if self.sub_agent_manager.delete(name):
        self.console.print(f"  [red]\u2717[/red] Deleted sub-agent [bold]{name}[/bold]")
        return f"Sub-agent '{name}' deleted."
    return f"Error: Sub-agent '{name}' not found."
