"""Sub-agent lifecycle manager.

Each sub-agent is a fully independent ClineAgent with its own conversation history,
context container, todo list, streaming client, and session path — completely isolated
from the parent agent and sibling sub-agents.
"""

import asyncio
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console

from .cline_agent import ClineAgent
from .config import Config
from .logger import get_logger, log_exception

log = get_logger("sub_agent")


class TeeWriter:
    """A file-like writer that buffers all output and optionally passes through
    to the real stdout.

    Used to capture sub-agent output for both buffering (when agent runs in
    background) and real-time display (when the user switches focus to it).
    """

    def __init__(self, real_stdout):
        self.buffer = io.StringIO()
        self.real_stdout = real_stdout
        self.active = False  # If True, also writes to real stdout

    def write(self, text: str) -> None:
        self.buffer.write(text)
        if self.active:
            try:
                self.real_stdout.write(text)
                self.real_stdout.flush()
            except Exception:
                pass

    def flush(self) -> None:
        if self.active:
            try:
                self.real_stdout.flush()
            except Exception:
                pass

    def getvalue(self) -> str:
        return self.buffer.getvalue()

    def clear(self) -> None:
        self.buffer = io.StringIO()


@dataclass
class SubAgentInstance:
    """Holds the state for a single sub-agent."""

    name: str
    agent: ClineAgent
    task: Optional[asyncio.Task] = None
    status: str = "created"  # created, running, completed, error
    output: str = ""
    tee: Optional[TeeWriter] = None
    session_path: Optional[Path] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    completion_notified: bool = False
    last_error: Optional[str] = None


class SubAgentManager:
    """Central registry for all sub-agents. Owned by main() in harness.py."""

    def __init__(
        self,
        config: Config,
        console: Console,
        workspace: str,
        get_session_path_fn,
    ):
        self._agents: Dict[str, SubAgentInstance] = {}
        self._config = config
        self._console = console
        self._workspace = workspace
        self._get_session_path = get_session_path_fn
        self._focused_name: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def create(self, name: str, task_prompt: str) -> str:
        """Create a new sub-agent and start it running in the background.

        Returns the sub-agent name immediately (non-blocking).
        The sub-agent's background task is started as an asyncio.Task.
        """
        if not name or not task_prompt:
            raise ValueError("Both 'name' and 'task_prompt' are required.")
        if name in self._agents:
            raise ValueError(f"Sub-agent '{name}' already exists.")

        # Validate name is safe for filenames
        safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        if not safe_name:
            raise ValueError("Invalid sub-agent name.")

        # Build session path: underscore prefix prevents collision with parent sessions
        session_path = self._get_session_path(self._workspace, f"_sub_{safe_name}")

        # Create a TeeWriter to capture all output
        real_stdout = getattr(sys, "__stdout__", sys.stdout)
        tee = TeeWriter(real_stdout)

        # Create sub-agent Console that writes through the Tee
        sub_console = Console(file=tee)

        # Clone config (same model/provider as parent)
        sub_config = Config(
            api_key=self._config.api_key,
            api_url=self._config.api_url,
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        # Copy additional attributes
        for attr in ("reasoning_effort", "compaction_threshold", "plugins", "plugin_config"):
            if hasattr(self._config, attr):
                setattr(sub_config, attr, getattr(self._config, attr))

        # Create a fresh ClineAgent — no messages, context, todos inherited
        sub_agent = ClineAgent(
            config=sub_config,
            console=sub_console,
            output_stream=tee,
            enable_status_line=False,
        )

        instance = SubAgentInstance(
            name=name,
            agent=sub_agent,
            tee=tee,
            session_path=session_path,
        )
        self._agents[name] = instance

        # Start background task
        instance.task = asyncio.create_task(
            self._run_agent_task(instance, task_prompt)
        )
        instance.status = "running"
        log.info("Sub-agent '%s' created and started.", name)
        return name

    async def run(self, name: str, input_text: str) -> str:
        """Send input to a sub-agent and wait for its response.

        If the sub-agent is currently running (e.g. still processing a previous
        task), this method waits for it to finish first, then starts a new turn
        with the given input.

        Returns the sub-agent's full response text.
        """
        instance = self._get(name)

        # Wait for any currently running task
        if instance.task and not instance.task.done():
            try:
                await instance.task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.warning("Sub-agent '%s' task raised: %s", name, e)

        # Start a new turn with the given input
        instance.task = asyncio.create_task(
            self._run_agent_task(instance, input_text)
        )
        instance.status = "running"
        result = await instance.task
        return result

    def pause(self, name: str) -> bool:
        """Pause a running sub-agent. Cancels its background task."""
        instance = self._agents.get(name)
        if not instance:
            return False
        if instance.task and not instance.task.done():
            instance.task.cancel()
        instance.status = "paused"
        self._save_session(instance)
        log.info("Sub-agent '%s' paused.", name)
        return True

    def delete(self, name: str) -> bool:
        """Delete a sub-agent: cancel task, remove from registry."""
        instance = self._agents.pop(name, None)
        if not instance:
            return False
        if instance.task and not instance.task.done():
            instance.task.cancel()
        self._save_session(instance)
        if self._focused_name == name:
            self._focused_name = None
        log.info("Sub-agent '%s' deleted.", name)
        return True

    def list(self) -> List[Dict[str, Any]]:
        """Return info for all sub-agents."""
        result = []
        for name, inst in self._agents.items():
            elapsed = time.time() - inst.created_at
            result.append({
                "name": name,
                "status": inst.status,
                "elapsed_seconds": int(elapsed),
                "has_output": bool(inst.output),
                "completed": inst.status == "completed",
            })
        return result

    def get(self, name: str) -> Optional[SubAgentInstance]:
        return self._agents.get(name)

    def set_focused(self, name: Optional[str]) -> None:
        """Set which sub-agent is focused.

        When focused, the sub-agent's TeeWriter passes through to the real
        terminal so the user sees output in real-time.
        """
        # Deactivate previous focus
        if self._focused_name and self._focused_name in self._agents:
            prev = self._agents[self._focused_name]
            if prev.tee:
                prev.tee.active = False

        self._focused_name = name

        # Activate new focus
        if name and name in self._agents:
            inst = self._agents[name]
            if inst.tee:
                inst.tee.active = True

    def get_focused(self) -> Optional[str]:
        return self._focused_name

    def check_completed(self) -> Optional[str]:
        """Return the name of a sub-agent that just completed and hasn't been
        notified about yet. Returns None if nothing new."""
        for name, inst in self._agents.items():
            if inst.status == "completed" and not inst.completion_notified:
                inst.completion_notified = True
                return name
        return None

    def save_all_sessions(self) -> None:
        """Save all sub-agent sessions for debugging."""
        for inst in self._agents.values():
            self._save_session(inst)

    def cleanup(self) -> None:
        """Cancel all background tasks and save sessions."""
        for inst in self._agents.values():
            if inst.task and not inst.task.done():
                inst.task.cancel()
            self._save_session(inst)
        self._agents.clear()
        self._focused_name = None

    # ── Internals ─────────────────────────────────────────────────────

    def _get(self, name: str) -> SubAgentInstance:
        inst = self._agents.get(name)
        if not inst:
            raise KeyError(f"Sub-agent '{name}' not found.")
        return inst

    async def _run_agent_task(self, instance: SubAgentInstance, input_text: str) -> str:
        """Background task: run the sub-agent with the given input.

        All stdout output goes through the TeeWriter (buffered; optionally
        passed through to terminal if focused).
        """
        try:
            result = await instance.agent.run_message(
                input_text,
                enable_interrupt=False,  # Interrupt handled by main loop
            )
            instance.output = instance.tee.getvalue() if instance.tee else ""
            instance.status = "completed"
            instance.completed_at = time.time()
            instance.completion_notified = False  # Reset for notification cycle
            self._save_session(instance)
            log.info(
                "Sub-agent '%s' completed. output_len=%d",
                instance.name,
                len(instance.output),
            )
            return result
        except asyncio.CancelledError:
            instance.status = "paused"
            instance.output = instance.tee.getvalue() if instance.tee else ""
            self._save_session(instance)
            log.info("Sub-agent '%s' cancelled.", instance.name)
            return "[Sub-agent paused]"
        except Exception as e:
            instance.status = "error"
            instance.last_error = str(e)
            instance.output = instance.tee.getvalue() if instance.tee else ""
            log_exception(log, f"Sub-agent '{instance.name}' failed", e)
            return f"[Sub-agent error: {e}]"

    def _save_session(self, instance: SubAgentInstance) -> None:
        """Save sub-agent session for debugging purposes."""
        try:
            if instance.session_path and instance.agent:
                instance.agent.save_session(str(instance.session_path))
        except Exception as e:
            log.debug("Failed to save sub-agent session '%s': %s", instance.name, e)

    def _append_output(self, instance: SubAgentInstance) -> None:
        """Update instance.output from the tee buffer."""
        if instance.tee:
            instance.output = instance.tee.getvalue()

