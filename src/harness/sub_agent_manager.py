"""Sub-agent lifecycle manager.

Each sub-agent is a fully independent ClineAgent with its own conversation history,
context container, todo list, streaming client, and session path — completely isolated
from the parent agent and sibling sub-agents.

Persistence: a registry (.sessions/_subagents.json) survives restarts. On boot the
manager restores registry metadata only (lazy hydration — ClineAgent construction is
expensive); the actual agent + session history load on first interaction.
"""

import asyncio
import concurrent.futures
import io
import json
import os
import re
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

# Strip ANSI/VT100 escape sequences from captured output before using it as
# plain-text snippets (list_agents progress, partial-output tails). The
# sub-agent transcript is a live terminal capture and contains styling codes
# (e.g. per-chunk dim spans around streamed thinking text).
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI/VT100 escape sequences and collapse runs of blank lines."""
    if not text:
        return ""
    text = _ANSI_ESCAPE_RE.sub("", text)
    # The per-chunk dim styling leaves "\x1b[0m\x1b[2m" boundaries that render
    # as no space in a terminal; after stripping, lines may contain stray
    # artifacts. Collapse 3+ consecutive blank lines for snippet readability.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


class TeeWriter:
    """A file-like writer that buffers all output and optionally passes through
    to the real stdout.

    Used to capture sub-agent output for both buffering (when agent runs in
    background) and real-time display (when the user switches focus to it).
    """

    # Cap the in-memory transcript so long-running background agents can't
    # balloon memory. The buffer keeps the TAIL (most recent output); the
    # head is dropped with a marker once the cap is exceeded. Disk transcript
    # tails (_save_transcript_tail) preserve the full picture within its own
    # cap for restore/replay.
    MAX_BUFFER_CHARS = 2 * 1024 * 1024  # ~2MB

    def __init__(self, real_stdout, dynamic: bool = False):
        self.buffer = io.StringIO()
        self.real_stdout = real_stdout
        self.dynamic = dynamic  # Resolve sys.stdout at write time (prompt proxy)
        self.active = False  # If True, also writes to real stdout
        self._pending = ""  # Partial line held back for line-buffered passthrough
        self._truncated = False  # Head was dropped (cap exceeded)
        self.last_write_at = 0.0  # Timestamp of last write (activity/hung indicator)

    def write(self, text: str) -> None:
        if text:
            self.last_write_at = time.time()
        self.buffer.write(text)
        self._maybe_trim()
        if self.active:
            # Line-buffered passthrough: hold back partial lines and flush
            # only complete lines as a single write. Rich emits styled text
            # in many small chunks (per ANSI span), so writing per-chunk
            # through the patch_stdout proxy redraws the prompt after every
            # chunk — the prompt visibly jumps, partial lines gain extra
            # newlines, and ANSI sequences get split mid-escape. One write
            # per complete line avoids all three.
            self._pending += text
            if "\n" in self._pending:
                lines = self._pending.split("\n")
                self._pending = lines.pop()  # keep the trailing partial line
                self._write_real("\n".join(lines) + "\n")

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
            # Release any held-back partial line (rare — rich prints are
            # newline-terminated, so _pending is normally empty at flush).
            if self._pending:
                text = self._pending
                self._pending = ""
                self._write_real(text)
            try:
                self.real_stdout.flush()
            except Exception:
                pass

    def getvalue(self) -> str:
        value = self.buffer.getvalue()
        if self._truncated:
            return "[... earlier output omitted ...]\n" + value
        return value

    def clear(self) -> None:
        self.buffer = io.StringIO()
        self._truncated = False

    def _maybe_trim(self) -> None:
        """Drop the buffer head once it exceeds the cap (checked periodically,
        not per-write — buffer.tell() is cheap but not free)."""
        if self._truncated:
            return
        try:
            if self.buffer.tell() > self.MAX_BUFFER_CHARS:
                text = self.buffer.getvalue()
                self.buffer = io.StringIO()
                self.buffer.write(text[len(text) - self.MAX_BUFFER_CHARS // 2:])
                self._truncated = True
        except Exception:
            pass

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
class AgentSignals:
    """Per-agent keyboard signals, set by the focused-agent UI controls.

    Unlike the process-global InterruptState (owned by the parent), these are
    scoped to one agent instance: a signal can never leak to a sibling, and
    consume-on-read semantics mean the next shell command / loop iteration
    that observes it owns the action. The queue drainer clears them at turn
    end, bounding "background the next command" to the current turn.
    """

    interrupt: bool = False
    background: bool = False

    def consume_interrupt(self) -> bool:
        """Return and clear the interrupt flag (consume-on-read)."""
        if self.interrupt:
            self.interrupt = False
            return True
        return False

    def consume_background(self) -> bool:
        """Return and clear the background flag (consume-on-read)."""
        if self.background:
            self.background = False
            return True
        return False

    def clear(self) -> None:
        """Clear both signals (turn boundary)."""
        self.interrupt = False
        self.background = False


@dataclass
class TurnResult:
    """A single completed turn's result, kept in a monotonic per-agent log.

    Replaces the old single-`final_result` overwrite semantics: overlapping
    turn completions each retain their own result until fetched.
    """

    turn_id: int
    status: str  # "completed" | "error"
    text: str  # agent's return value (completed) or error text (error)
    error: Optional[str] = None


@dataclass
class QueueEntry:
    """A pending input for a busy sub-agent.

    source: "user" (REPL-typed while focused) — consecutive entries coalesce
            into one turn at drain time; or "model" (send_agent_input) —
            each entry gets its own turn and its future resolved with that
            turn's result.
    """

    source: str  # "user" | "model"
    text: str
    future: Any = None  # asyncio.Future for model entries


@dataclass
class SubAgentInstance:
    """Holds the state for a single sub-agent."""

    name: str
    agent: ClineAgent
    task: Any = None  # asyncio.Task on the main loop, or thread-safe Future fallback
    status: str = "created"  # created, running, completed, paused, error
    output: str = ""  # Rendered terminal transcript (diagnostic only)
    final_result: str = ""  # Latest turn's return value (display/summary use)
    result_log: List[TurnResult] = field(default_factory=list)  # Unfetched results
    turn_seq: int = 0  # Monotonic per-agent turn counter
    signals: AgentSignals = field(default_factory=AgentSignals)  # Focused-agent keyboard signals
    pending: Any = None  # Turn queue (List[QueueEntry]); built lazily
    turn_started_at: Optional[float] = None  # Current turn's start (prompt-bar elapsed)
    tee: Optional[TeeWriter] = None
    session_path: Optional[Path] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    completion_notified: bool = False
    last_error: Optional[str] = None
    # ── Persistence / restore ─────────────────────────────────────────
    # A restored instance is created WITHOUT a live ClineAgent (lazy
    # hydration: construction is expensive — WorkspaceIndex, SmartContext,
    # TodoManager). The agent is built + load_session()'d on first real
    # interaction (focus, send_agent_input, a turn).
    hydrated: bool = True  # False for restored-but-not-yet-loaded instances
    safe_name: str = ""  # Filename-safe variant ("" → derive from name)
    restored: bool = False  # True for instances adopted from a previous run

    def ensure_safe_name(self) -> str:
        if not self.safe_name:
            self.safe_name = self.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return self.safe_name


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

        # Registry persistence (.sessions/_subagents.json) — survives restarts.
        self._registry_path = Path(workspace) / ".sessions" / "_subagents.json"
        self._restored_count = 0  # Agents adopted from a previous run

        # Dedicated background event loop thread.  Sub-agent coroutines are
        # scheduled onto this loop so they progress regardless of whether the
        # main thread is blocked at the prompt, rendering, or idle.
        self._bg_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_ready = threading.Event()

    # ── Registry persistence ──────────────────────────────────────────

    def _write_registry(self) -> None:
        """Atomically persist the sub-agent registry (temp + os.replace).

        Windows note: os.replace fails if the target is open — retry briefly.
        """
        entries = []
        for inst in self._agents.values():
            try:
                safe = inst.ensure_safe_name()
            except Exception:
                safe = str(getattr(inst, "safe_name", "") or getattr(inst, "name", ""))
            entries.append({
                "name": inst.name,
                "safe_name": safe,
                "status": inst.status,
                "session_path": str(inst.session_path) if inst.session_path else None,
                "created_at": getattr(inst, "created_at", None),
                "completed_at": getattr(inst, "completed_at", None),
                "final_result": getattr(inst, "final_result", "") or "",
                "last_error": getattr(inst, "last_error", None),
                "restored": getattr(inst, "restored", False),
            })
        payload = json.dumps({"agents": entries}, ensure_ascii=False, indent=1)
        tmp = self._registry_path.with_suffix(".json.tmp")
        try:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(payload, encoding="utf-8")
            last_err = None
            for _ in range(5):  # Windows: target may be open by a concurrent reader
                try:
                    os.replace(tmp, self._registry_path)
                    return
                except PermissionError as e:
                    last_err = e
                    time.sleep(0.05)
            try:
                os.replace(tmp, self._registry_path)
            except Exception as e:
                log.debug("Registry write failed: %s / %s", e, last_err)
        except Exception as e:
            log.debug("Registry write failed: %s", e)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def registry_path(self) -> Path:
        return self._registry_path

    def _registry_entry_for(self, inst: SubAgentInstance) -> None:
        """Mark a mid-turn instance as interrupted in the registry before a
        save-and-exit so restore classifies it correctly."""
        task = getattr(inst, "task", None)
        if inst.status == "running" and (task is None or not task.done()):
            inst.status = "interrupted"

    def restore(self) -> int:
        """Adopt sub-agents from a previous run's registry (metadata only).

        Lazy hydration: instances are registered WITHOUT a live ClineAgent;
        the agent is constructed + load_session()'d on first interaction
        (focus, input). Returns the number of restored agents.
        """
        if not self._registry_path.exists():
            return 0
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Sub-agent registry unreadable (%s) — skipping restore", e)
            return 0
        entries = data.get("agents", []) if isinstance(data, dict) else []
        restored = 0
        for entry in entries:
            name = entry.get("name") or ""
            status = entry.get("status") or "completed"
            if not name or name in self._agents:
                # Live-created names win; never overwrite them.
                continue
            session_path = entry.get("session_path")
            if session_path and not Path(session_path).exists():
                session_path = None
            inst = SubAgentInstance(
                name=name,
                agent=None,  # lazy: hydrated on first interaction
                status=status,
                final_result=entry.get("final_result", "") or "",
                session_path=Path(session_path) if session_path else None,
                created_at=entry.get("created_at") or time.time(),
                completed_at=entry.get("completed_at"),
                last_error=entry.get("last_error"),
                # Completion happened in a previous process. Only new
                # completions should wake the parent prompt.
                completion_notified=True,
                hydrated=False,
                safe_name=entry.get("safe_name", "") or "",
                restored=True,
            )
            inst.ensure_safe_name()
            self._agents[name] = inst
            restored += 1
        self._restored_count = restored
        if restored:
            log.info("Restored %d sub-agent(s) from previous run", restored)
        return restored

    def _hydrate(self, inst: SubAgentInstance) -> bool:
        """Construct the ClineAgent for a restored instance and load its
        session history. Returns True on success."""
        if getattr(inst, "hydrated", True):
            return True
        if inst.agent is not None:
            inst.hydrated = True
            return True
        try:
            tee = TeeWriter(sys.stdout, dynamic=True)
            sub_console = Console(file=tee)
            sub_config = Config(
                api_key=self._config.api_key,
                api_url=self._config.api_url,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            for attr in ("reasoning_effort", "compaction_threshold", "plugins", "plugin_config"):
                if hasattr(self._config, attr):
                    setattr(sub_config, attr, getattr(self._config, attr))
            sub_agent = ClineAgent(
                config=sub_config,
                console=sub_console,
                output_stream=tee,
                enable_status_line=False,
            )
            inst.tee = tee
            inst.agent = sub_agent
            sub_agent.agent_signals = inst.signals

            # Load history if a session file survived
            if inst.session_path and Path(inst.session_path).exists():
                loaded = sub_agent.load_session(str(inst.session_path), inject_resume=False)
                if not loaded:
                    log.warning("Sub-agent '%s' session failed to load — fresh agent", inst.name)
                else:
                    self._stitch_dangling_tool_calls(sub_agent)
                # run_message must not prepend a second system prompt
                sub_agent._initialized = True

            inst.hydrated = True
            return True
        except Exception as e:
            log_exception(log, f"Hydration of sub-agent '{inst.name}' failed", e)
            return False

    @staticmethod
    def _stitch_dangling_tool_calls(agent: ClineAgent) -> None:
        """Repair history that was cut mid-turn by a kill/exit.

        An agent killed between an assistant tool_calls message and its tool
        results leaves history that providers reject with 400s. Inject a
        synthetic result for each dangling tool_call_id, and drop orphan tool
        results whose assistant message is gone.
        """
        try:
            msgs = agent.messages
            answered: set = set()
            for m in msgs:
                if m.role == "tool" and m.tool_call_id:
                    answered.add(m.tool_call_id)
            # Find the LAST assistant message with tool_calls; only the tail
            # can be dangling (mid-turn kills happen at the end of history).
            for i in range(len(msgs) - 1, -1, -1):
                m = msgs[i]
                if m.role != "assistant":
                    continue
                raw_calls = m.tool_calls or []
                if not raw_calls:
                    break  # no dangling tail
                from .streaming_client import StreamingMessage  # local import

                for tc in raw_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id and tc_id not in answered:
                        tc_name = tc.get("function", {}).get("name", "unknown")
                        msgs.append(
                            StreamingMessage(
                                role="tool",
                                content=(
                                    "[Interrupted before execution — this tool call "
                                    "was not executed. Re-issue it if still needed.]"
                                ),
                                tool_call_id=tc_id,
                                name=tc_name,
                            )
                        )
                break  # only the tail assistant message can dangle
            # Drop orphan tool results (their assistant message was evicted)
            keep = []
            call_ids = set()
            for m in msgs:
                if m.role == "assistant":
                    for tc in (m.tool_calls or []):
                        if isinstance(tc, dict) and tc.get("id"):
                            call_ids.add(tc["id"])
            for m in msgs:
                if m.role == "tool" and m.tool_call_id and m.tool_call_id not in call_ids:
                    continue  # orphan
                keep.append(m)
            agent.messages = keep
        except Exception as e:
            log.debug("Tool-call stitching failed (non-fatal): %s", e)

    def _save_transcript_tail(self, inst: SubAgentInstance) -> None:
        """Write a plain-text transcript tail for restored replay (~64KB)."""
        try:
            if inst.tee is None or not inst.session_path:
                return
            from .sub_agent_manager import strip_ansi  # self-import safe

            text = strip_ansi(inst.tee.getvalue() or "")
            if not text:
                return
            tail_path = inst.session_path.with_suffix(".log")
            keep = 64 * 1024
            if len(text) > keep:
                text = "[... earlier output omitted ...]\n" + text[-keep:]
            tail_path.write_text(text, encoding="utf-8", errors="replace")
        except Exception as e:
            log.debug("Transcript tail save failed for '%s': %s", inst.name, e)

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

    def _schedule_bg(self, coro):
        """Schedule *coro* for execution.

        Prefers the caller's running event loop (the main loop): asyncio
        subprocess child watchers only work on the main thread's loop on
        Unix, so routing sub-agent shell commands through any other loop
        breaks exit-status delivery ("exit status already read", rc 255).
        The main loop keeps running while the user types (prompt_async),
        so main-loop scheduling still executes independently of typing.
        Falls back to the dedicated background loop only when called from
        a context with no running loop (tests, embedders).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            return loop.create_task(coro)
        bg = self._ensure_bg_loop()
        return asyncio.run_coroutine_threadsafe(coro, bg)

    async def _await_task(self, task) -> str:
        """Await an asyncio Task or a thread-safe concurrent Future."""
        if isinstance(task, concurrent.futures.Future):
            return await asyncio.wrap_future(task)
        return await task

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
            safe_name=safe_name,
        )
        sub_agent.agent_signals = instance.signals
        self._agents[name] = instance

        # Persist the registry on every state transition
        self._write_registry()

        # Start background task on the dedicated sub-agent loop so it runs
        # independently of the main REPL's event loop.
        self._start_turn(instance, task_prompt)
        log.info("Sub-agent '%s' created and started.", name)
        return name

    # ── Turn scheduling & queue ───────────────────────────────────────

    def _start_turn(self, instance: SubAgentInstance, text: str, future: Any = None):
        """Schedule a turn on the appropriate loop; returns the task/future."""
        try:
            instance._active_future = future
        except Exception:
            pass
        try:
            instance.turn_started_at = time.time()
        except Exception:
            pass
        task = self._schedule_bg(self._run_agent_task(instance, text))
        instance.task = task
        instance.status = "running"
        return task

    def _enqueue(self, instance: SubAgentInstance, entry: QueueEntry) -> None:
        pending = getattr(instance, "pending", None)
        if pending is None:
            pending = []
            instance.pending = pending
        pending.append(entry)

    def _start_next_from_queue(self, instance: SubAgentInstance) -> None:
        """Drain the queue after a turn completes.

        Consecutive user-source entries coalesce into one turn (they were all
        authored before the turn ran — conversational fragments); each
        model-source entry gets its own turn with its future resolved with
        that turn's result.
        """
        pending = getattr(instance, "pending", None)
        if not pending:
            return
        entry = pending.pop(0)
        if entry.source == "user":
            texts = [entry.text]
            while pending and pending[0].source == "user":
                texts.append(pending.pop(0).text)
            self._start_turn(instance, "\n\n".join(texts))
        else:
            self._start_turn(instance, entry.text, future=getattr(entry, "future", None))

    def _resolve_future(self, future: Any, value: str) -> None:
        """Thread-safe future resolution (futures may live on another loop)."""
        if future is None:
            return
        try:
            if future.done():
                return
        except Exception:
            return
        try:
            future.get_loop().call_soon_threadsafe(
                self._set_future_result, future, value
            )
        except Exception:
            # Loop closed or not an asyncio future — resolve directly.
            self._set_future_result(future, value)

    @staticmethod
    def _set_future_result(future: Any, value: str) -> None:
        try:
            if not future.done():
                future.set_result(value)
        except Exception:
            pass

    def _flush_queue(self, instance: SubAgentInstance, message: str) -> None:
        """Drop all pending queue entries (pause/delete), resolving any
        waiting futures so send_agent_input(wait=true) callers don't hang."""
        pending = getattr(instance, "pending", None) or []
        instance.pending = []
        for entry in pending:
            self._resolve_future(getattr(entry, "future", None), message)
        self._resolve_future(getattr(instance, "_active_future", None), message)
        try:
            instance._active_future = None
        except Exception:
            pass

    async def queue_input(self, name: str, input_text: str) -> str:
        """REPL focused-agent path: non-blocking input delivery.

        If the agent is idle, start a turn immediately; if busy, queue the
        input (never block the prompt). Returns "started", "queued", or
        "dropped" (self-notification).
        """
        instance = self._get(name)

        # Defensive: never queue an agent's own [SYSTEM: ...] notification
        # (the REPL routes system input to the parent, but be safe).
        if input_text.strip().startswith("[SYSTEM:") and f"'{name}'" in input_text:
            log.info("Dropping self-notification for sub-agent '%s'", name)
            return "dropped"

        # Restored instances hydrate on first interaction
        self._hydrate(instance)

        task = getattr(instance, "task", None)
        busy = task is not None and not task.done()
        if not busy:
            self._start_turn(instance, input_text)
            return "started"
        self._enqueue(instance, QueueEntry(source="user", text=input_text))
        n = len(instance.pending)
        if self._console:
            try:
                self._console.print(
                    f"  [yellow]\u2957[/yellow] '{name}' busy — input queued ({n} pending)"
                )
            except Exception:
                pass
        return "queued"

    async def run(self, name: str, input_text: str, wait: bool = True) -> str:
        """Send input to a sub-agent (the model-facing path).

        Idle agent → start a turn and return its response (synchronous chat).
        Busy agent → enqueue; if wait=True block until this entry's turn
        completes and return its result, else return a queued-ack immediately
        (the parent learns the result via the completion notification →
        get_agent_output cycle).
        """
        instance = self._get(name)

        # Never feed an agent its own completion/error notification as a new
        # turn. The REPL routes input to the focused agent, and if a
        # notification ("[SYSTEM: Sub-agent 'X' has completed...]" or the
        # FAILED variant) is routed back into the same agent, it starts a new
        # turn, completes (or errors) again, resets completion_notified, and
        # the notification fires again — an infinite notify→run loop. Return
        # the cached result instead.
        if (
            instance.status in ("completed", "error")
            and input_text.strip().startswith("[SYSTEM:")
            and f"'{name}'" in input_text
        ):
            log.info(
                "Refusing to feed completion notification back into sub-agent '%s'",
                name,
            )
            return instance.final_result or instance.output or ""

        # Restored instances hydrate on first interaction
        self._hydrate(instance)

        task = getattr(instance, "task", None)
        busy = task is not None and not task.done()
        if not busy:
            started = self._start_turn(instance, input_text)
            return await self._await_task(started)

        # Busy: enqueue this input
        try:
            fut = asyncio.get_running_loop().create_future()
        except Exception:
            fut = None
        self._enqueue(instance, QueueEntry(source="model", text=input_text, future=fut))
        if fut is not None and wait:
            return await fut
        position = len(getattr(instance, "pending", None) or [])
        return (
            f"Queued (position {position}). Sub-agent '{name}' is busy with its "
            f"current task. You'll be notified when it completes — then use "
            f"get_agent_output(name='{name}') to retrieve the result."
        )

    def pause(self, name: str) -> bool:
        """Pause a running sub-agent. Cancels its background future."""
        instance = self._agents.get(name)
        if not instance:
            return False
        self._flush_queue(instance, "[Sub-agent paused]")
        if instance.task and not instance.task.done():
            instance.task.cancel()
            # Only thread-safe futures can be awaited from this sync context.
            # asyncio Tasks process their cancellation next loop iteration.
            if isinstance(instance.task, concurrent.futures.Future):
                try:
                    instance.task.result(timeout=5)
                except concurrent.futures.CancelledError:
                    pass
                except Exception:
                    pass
        instance.status = "paused"
        self._save_session(instance)
        self._save_transcript_tail(instance)
        self._write_registry()
        log.info("Sub-agent '%s' paused.", name)
        return True

    def delete(self, name: str) -> bool:
        """Delete a sub-agent: cancel task, remove from registry. Destroy
        means destroy — registry entry and session files are removed too."""
        instance = self._agents.pop(name, None)
        if not instance:
            return False
        self._flush_queue(instance, "[Sub-agent deleted]")
        if instance.task and not instance.task.done():
            instance.task.cancel()
            if isinstance(instance.task, concurrent.futures.Future):
                try:
                    instance.task.result(timeout=5)
                except concurrent.futures.CancelledError:
                    pass
                except Exception:
                    pass
        self._save_session(instance)
        if self._focused_name == name:
            self._focused_name = None
        # Remove persisted artifacts (registry entry + session/transcript files)
        try:
            self._write_registry()  # entry gone now that instance was popped
            if instance.session_path:
                instance.session_path.unlink(missing_ok=True)
                instance.session_path.with_suffix(".log").unlink(missing_ok=True)
        except Exception as e:
            log.debug("Cleanup of sub-agent '%s' files failed: %s", name, e)
        log.info("Sub-agent '%s' deleted.", name)
        return True

    def list(self) -> List[Dict[str, Any]]:
        """Return info for all sub-agents, including a snippet of current output."""
        result = []
        for name, inst in self._agents.items():
            # Freeze elapsed time at completion so the counter doesn't keep ticking
            completed_at = getattr(inst, "completed_at", None)
            end_time = completed_at if inst.status == "completed" and completed_at else time.time()
            created_at = getattr(inst, "created_at", None)
            if not created_at:
                created_at = time.time()
            elapsed = max(0, end_time - created_at)

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
                else:
                    # Running agents: strip terminal styling from the live
                    # transcript so the progress snippet stays readable as
                    # plain text (the tee capture contains ANSI codes).
                    output_text = strip_ansi(output_text)
                    if len(output_text) > _MAX_SNIPPET:
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
                "new_results": len(getattr(inst, "result_log", None) or []),
                "last_output_age": (
                    max(0, int(time.time() - inst.tee.last_write_at))
                    if inst.tee and getattr(inst.tee, "last_write_at", 0)
                    else None
                ),
                "tokens": dict(getattr(inst.agent, "agent_usage", {}) or {}),
            })
        return result

    def get(self, name: str) -> Optional[SubAgentInstance]:
        return self._agents.get(name)

    def signal(self, name: str, kind: str) -> bool:
        """Set a per-agent keyboard signal on the named agent.

        kind: "interrupt" or "background". Returns True if the agent exists
        and has a signals object. Used by the focused-agent UI controls
        (prompt Ctrl+B binding, staged Ctrl+C).
        """
        inst = self._agents.get(name)
        signals = getattr(inst, "signals", None) if inst else None
        if signals is None:
            return False
        if kind == "interrupt":
            signals.interrupt = True
        elif kind == "background":
            signals.background = True
        else:
            return False
        return True

    def active_count(self) -> int:
        """Number of agents with a live task (running or queued) — used by the
        staged Ctrl+C to decide whether exiting needs a warning."""
        n = 0
        for inst in self._agents.values():
            task = getattr(inst, "task", None)
            if task is not None and not task.done():
                n += 1
            elif getattr(inst, "pending", None):
                n += 1
        return n

    def stats(self) -> Dict[str, int]:
        """Cheap status counts for the prompt bar — NO output snippets.

        list() reads and ANSI-strips tee buffers, which is far too heavy to
        run per keystroke prompt render; this walks the dict only.
        """
        running = 0
        unread = 0
        for inst in self._agents.values():
            task = getattr(inst, "task", None)
            if (task is not None and not task.done()) or getattr(inst, "pending", None):
                running += 1
            if inst.status in ("completed", "error") and (getattr(inst, "result_log", None) or []):
                unread += 1
        return {"running": running, "unread": unread, "total": len(self._agents)}

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
            # Restored instances hydrate on focus (prints the transcript tail)
            if not getattr(inst, "hydrated", True):
                self._hydrate(inst)
                if self._console is not None:
                    try:
                        self._console.print(
                            f"  [dim](restored session — live transcript not retained"
                            f" from the previous run)[/dim]"
                        )
                    except Exception:
                        pass
                if inst.session_path is not None:
                    tail = inst.session_path.with_suffix(".log")
                    if tail.exists():
                        try:
                            text = tail.read_text(encoding="utf-8", errors="replace")
                            if text:
                                keep = 4000
                                if len(text) > keep:
                                    text = "[... earlier output omitted ...]\n" + text[-keep:]
                                if self._console is not None:
                                    self._console.print(text)
                        except Exception:
                            pass
            if inst.tee:
                # Show output produced while the agent was in the background,
                # then continue streaming new output live.
                inst.tee.replay_to_real()
                inst.tee.active = True

    def purge_restored(self) -> int:
        """Remove all restored (non-live) agents and their session files.
        Returns the number purged. `/agents --purge` entry point."""
        purged = 0
        for name in list(self._agents.keys()):
            inst = self._agents.get(name)
            if inst is not None and getattr(inst, "restored", False):
                self.delete(name)
                purged += 1
        return purged

    def next_focus(self, current: Optional[str]) -> Optional[str]:
        """Next agent in focus-cycle order: parent → agent1 → … → agentN →
        parent. Order: focused first, then running, then completed/
        interrupted, then error — i.e. the same order /agents displays.
        Returns None when the registry is empty (no-op)."""
        names = self._ordered_names()
        if not names:
            return None
        if current is None:
            return names[0]
        try:
            idx = names.index(current)
        except ValueError:
            return names[0]
        # Wrap back to the parent (None) after the last agent
        return names[(idx + 1) % len(names)] if idx + 1 < len(names) else None

    def _ordered_names(self) -> List[str]:
        """Agent names in display order: focused, running, completed/
        interrupted, error."""

        def _rank(inst: SubAgentInstance) -> int:
            name = getattr(inst, "name", "")
            if name == self._focused_name:
                return 0
            task = getattr(inst, "task", None)
            if (task is not None and not task.done()) or getattr(inst, "pending", None):
                return 1
            if getattr(inst, "status", "") in ("completed", "interrupted"):
                return 2
            return 3

        names = [inst.name for inst in self._agents.values()]
        return sorted(names, key=lambda n: _rank(self._agents[n]))

    def check_completed(self) -> Optional[str]:
        """Return the name of a sub-agent that just completed (or errored) and
        hasn't been notified about yet. Returns None if nothing new.

        Errored sub-agents MUST notify too — otherwise the parent waits forever
        on a zombie agent that will never produce a result.
        """
        for name, inst in self._agents.items():
            if (
                inst.status in ("completed", "error")
                and not inst.completion_notified
            ):
                inst.completion_notified = True
                return name
        return None

    def peek_completed(self) -> Optional[str]:
        """Non-consuming variant of check_completed().

        Used by the prompt-side watcher to detect a completion while the
        REPL sits inside prompt_async() without claiming the notification —
        the consuming check_completed() call that follows the prompt still
        owns injecting it.
        """
        for name, inst in self._agents.items():
            if (
                inst.status in ("completed", "error")
                and not inst.completion_notified
            ):
                return name
        return None

    def notification_text(self, name: str) -> str:
        """Canonical [SYSTEM: ...] notification for a completed/errored agent.

        Single source of truth for the message injected into the parent's
        conversation (all injection sites use this).
        """
        inst = self._get(name)
        if getattr(inst, "status", None) == "error":
            err = getattr(inst, "last_error", None) or "unknown error"
            return (
                f"[SYSTEM: Sub-agent '{name}' has FAILED with error: {err} "
                f"Use get_agent_output(name='{name}') to see details, "
                f"or list_agents() to see all agents.]"
            )
        return (
            f"[SYSTEM: Sub-agent '{name}' has completed its task. "
            f"Use get_agent_output(name='{name}') to retrieve its full output, "
            f"or list_agents(name='{name}') to see a summary. "
            f"Use send_agent_input(name='{name}', input='...') to start a new turn.]"
        )

    # ── Result log (consume-on-fetch) ─────────────────────────────────

    def _record_result(
        self, instance: SubAgentInstance, status: str, text: str, error: Optional[str] = None
    ) -> None:
        """Append a turn's result to the agent's monotonic result log.

        Defensive against foreign instance objects (tests use SimpleNamespace).
        """
        seq = getattr(instance, "turn_seq", 0) + 1
        try:
            instance.turn_seq = seq
        except Exception:
            pass
        log = getattr(instance, "result_log", None)
        if log is None:
            try:
                instance.result_log = []
                log = instance.result_log
            except Exception:
                return
        log.append(TurnResult(turn_id=seq, status=status, text=text or "", error=error))

    def fetch_new_results(self, name: str) -> List[TurnResult]:
        """Return and clear all results recorded since the last fetch.

        Consume-on-fetch semantics: the caller (parent) is responsible for
        retaining anything it needs. A second call returns [] until a new
        turn completes.
        """
        inst = self._get(name)
        log = getattr(inst, "result_log", None) or []
        new = list(log)
        if log:
            log.clear()
        return new

    def requeue_notification(self, name: str) -> None:
        """Undo a check_completed() consumption so the notification fires
        again on the next idle REPL cycle.

        Used when a completion arrives while the user is mid-typing: their
        typed input must not be discarded in favor of the notification, so
        the completion is requeued and delivered once the user's turn is
        done.
        """
        inst = self._agents.get(name)
        if inst is not None:
            inst.completion_notified = False

    def save_all_sessions(self) -> None:
        """Save all sub-agent sessions for debugging."""
        for inst in self._agents.values():
            self._save_session(inst)

    def cleanup(self) -> None:
        """Cancel all background tasks, stop the background loop, save sessions."""
        for inst in self._agents.values():
            # Classify mid-turn agents as interrupted BEFORE cancelling so
            # the registry/restore knows they were cut (not finished).
            self._registry_entry_for(inst)
            if inst.task and not inst.task.done():
                inst.task.cancel()
            self._save_session(inst)
            self._save_transcript_tail(inst)
        self._write_registry()
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
            self._record_result(instance, "completed", result or "")
            self._save_session(instance)
            self._save_transcript_tail(instance)
            self._write_registry()
            # Per-turn signal scoping: clear any unconsumed keyboard signals
            # so they cannot leak into a later turn.
            _signals = getattr(instance, "signals", None)
            if _signals is not None:
                _signals.clear()
            log.info(
                "Sub-agent '%s' completed. output_len=%d",
                instance.name,
                len(instance.output),
            )
            self._resolve_future(getattr(instance, "_active_future", None), result or "")
            self._start_next_from_queue(instance)
            return result
        except asyncio.CancelledError:
            instance.status = "paused"
            instance.output = instance.tee.getvalue() if instance.tee else ""
            _signals = getattr(instance, "signals", None)
            if _signals is not None:
                _signals.clear()
            self._save_session(instance)
            log.info("Sub-agent '%s' cancelled.", instance.name)
            self._resolve_future(
                getattr(instance, "_active_future", None), "[Sub-agent paused]"
            )
            return "[Sub-agent paused]"
        except Exception as e:
            instance.status = "error"
            instance.last_error = str(e)
            instance.output = instance.tee.getvalue() if instance.tee else ""
            instance.completion_notified = False  # Reset so the error notifies
            self._record_result(instance, "error", f"[Sub-agent error: {e}]", error=str(e))
            self._save_session(instance)
            self._save_transcript_tail(instance)
            self._write_registry()
            _signals = getattr(instance, "signals", None)
            if _signals is not None:
                _signals.clear()
            log_exception(log, f"Sub-agent '{instance.name}' failed", e)
            self._resolve_future(
                getattr(instance, "_active_future", None), f"[Sub-agent error: {e}]"
            )
            self._start_next_from_queue(instance)
            return f"[Sub-agent error: {e}]"

    def _save_session(self, instance: SubAgentInstance) -> None:
        """Save sub-agent session for debugging purposes."""
        try:
            if instance.session_path and instance.agent:
                instance.agent.save_session(str(instance.session_path))
        except Exception as e:
            log.debug("Failed to save sub-agent session '%s': %s", instance.name, e)


