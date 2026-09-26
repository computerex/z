"""Sub-agent lifecycle manager.

Each sub-agent is a fully independent ClineAgent with its own conversation history,
context container, todo list, streaming client, and session path — completely isolated
from the parent agent and sibling sub-agents.
"""

import asyncio
import concurrent.futures
import io
import sys
import threading
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

    def __init__(self, real_stdout, dynamic: bool = False):
        self.buffer = io.StringIO()
        self.real_stdout = real_stdout
        self.dynamic = dynamic  # Resolve sys.stdout at write time (prompt proxy)
        self.active = False  # If True, also writes to real stdout

    def write(self, text: str) -> None:
        self.buffer.write(text)
        if self.active:
            # real_stdout is the patch_stdout proxy in interactive sessions, so
            # these writes render above the prompt while it is active and pass
            # straight through otherwise.
            self._write_real(text)

    def _write_real(self, text: str) -> None:
        try:
            # Resolve the destination at write time: while the prompt is on
            # screen, sys.stdout is the patch_stdout proxy (render above the
            # prompt); during turns it is the real stdout.
            out = sys.stdout if self.dynamic else self.real_stdout
            out.write(text)
            out.flush()
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

    def replay_to_real(self, max_chars: int = 16000) -> int:
        """Replay buffered output when a user focuses this agent.

        Background output is intentionally buffered. Without replay, switching
        focus only shows output generated after the switch, which makes a
        running agent appear silent if it already produced its first turn.
        """
        text = self.getvalue()
        if not text:
            return 0
        if len(text) > max_chars:
            text = "[... earlier output omitted ...]\n" + text[-max_chars:]
        try:
            def _replay() -> None:
                self._write_real(text)
                if not text.endswith("\n"):
                    self._write_real("\n")

            _replay()
        except Exception:
            return 0
        return len(text)


@dataclass
class SubAgentInstance:
    """Holds the state for a single sub-agent."""

    name: str
    agent: ClineAgent
    task: Optional[concurrent.futures.Future] = None  # Thread-safe future on the background loop
    status: str = "created"  # created, running, completed, error
    output: str = ""  # Rendered terminal transcript (diagnostic only)
    final_result: str = ""  # Return value from the sub-agent's completed turn
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

        # Dedicated background event loop thread.  Sub-agent coroutines are
        # scheduled onto this loop so they progress regardless of whether the
        # main thread is blocked at the prompt, rendering, or idle.
        self._bg_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_ready = threading.Event()

    # ── Background loop management ────────────────────────────────────

    def _ensure_bg_loop(self) -> asyncio.AbstractEventLoop:
        """Start (once) and return the dedicated sub-agent event loop."""
        if self._bg_loop is not None and not self._bg_loop.is_closed():
            return self._bg_loop

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._bg_loop = loop
            self._bg_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._bg_thread = threading.Thread(
            target=_runner, daemon=True, name="subagent-loop"
        )
        self._bg_thread.start()
        if not self._bg_ready.wait(timeout=10):
            raise RuntimeError("Sub-agent background loop failed to start")
        return self._bg_loop

    def _schedule_bg(self, coro) -> "concurrent.futures.Future":
        """Schedule *coro* on the background loop; returns a thread-safe Future."""
        loop = self._ensure_bg_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

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

        # Create a TeeWriter to capture all output.  dynamic=True so live
        # writes resolve sys.stdout at write time — the patch_stdout proxy
        # while the prompt is active, real stdout during parent turns.
        tee = TeeWriter(sys.stdout, dynamic=True)

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

        # Start background task on the dedicated sub-agent loop so it runs
        # independently of the main REPL's event loop.
        instance.task = self._schedule_bg(
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
                await asyncio.wrap_future(instance.task)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.warning("Sub-agent '%s' task raised: %s", name, e)

        # Start a new turn with the given input
        instance.task = self._schedule_bg(
            self._run_agent_task(instance, input_text)
        )
        instance.status = "running"
        result = await asyncio.wrap_future(instance.task)
        return result

    def pause(self, name: str) -> bool:
        """Pause a running sub-agent. Cancels its background future."""
        instance = self._agents.get(name)
        if not instance:
            return False
        if instance.task and not instance.task.done():
            instance.task.cancel()
            # Give the background loop a moment to process the cancellation so
            # status/output are updated before the caller inspects them.
            try:
                instance.task.result(timeout=5)
            except concurrent.futures.CancelledError:
                pass
            except Exception:
                pass
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
            try:
                instance.task.result(timeout=5)
            except concurrent.futures.CancelledError:
                pass
            except Exception:
                pass
        self._save_session(instance)
        if self._focused_name == name:
            self._focused_name = None
        log.info("Sub-agent '%s' deleted.", name)
        return True

    def list(self) -> List[Dict[str, Any]]:
        """Return info for all sub-agents, including a snippet of current output."""
        result = []
        for name, inst in self._agents.items():
            # Freeze elapsed time at completion so the counter doesn't keep ticking
            end_time = inst.completed_at if inst.status == "completed" else time.time()
            elapsed = end_time - inst.created_at

            # Read live output from the TeeWriter buffer
            output_text = ""
            if inst.tee:
                try:
                    output_text = inst.tee.getvalue() or ""
                except Exception:
                    output_text = ""
            if not output_text:
                output_text = inst.output or ""

            # Truncate to a manageable snippet
            output_snippet = ""
            if output_text:
                _MAX_SNIPPET = 500
                # For completed agents, expose the concise final result rather
                # than a potentially huge ANSI-decorated terminal transcript.
                if inst.status == "completed" and inst.final_result:
                    output_snippet = inst.final_result
                elif len(output_text) > _MAX_SNIPPET:
                    output_snippet = "[...] " + output_text[-_MAX_SNIPPET:]
                else:
                    output_snippet = output_text

            result.append({
                "name": name,
                "status": inst.status,
                "elapsed_seconds": int(elapsed),
                "has_output": bool(output_text),
                "completed": inst.status == "completed",
                "output": output_snippet,
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
                # Show output produced while the agent was in the background,
                # then continue streaming new output live.
                inst.tee.replay_to_real()
                inst.tee.active = True

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
        """Cancel all background tasks, stop the background loop, save sessions."""
        for inst in self._agents.values():
            if inst.task and not inst.task.done():
                inst.task.cancel()
            self._save_session(inst)
        self._agents.clear()
        self._focused_name = None

        # Drain cancelled tasks, then stop the dedicated loop so the thread
        # can exit cleanly instead of dying with "Task was destroyed".
        if self._bg_loop is not None and not self._bg_loop.is_closed():
            async def _drain_and_stop():
                await asyncio.sleep(0.1)
                asyncio.get_running_loop().stop()

            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _drain_and_stop(), self._bg_loop
                )
                fut.result(timeout=5)
            except Exception:
                try:
                    self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
                except Exception:
                    pass
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5)
            self._bg_thread = None
        self._bg_loop = None

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
            # The terminal transcript may be enormous, ANSI-decorated, or
            # truncated when later passed through a tool result. Keep it for
            # diagnostics, but retain the agent's final return value separately
            # as the authoritative completed-task report.
            instance.final_result = result or ""
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


