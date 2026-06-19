"""Base class for remote input/output providers.

Two polling models are supported:

1. **Async polling** (default): override `_poll()` as an async generator.
   The provider must be started/stopped from an asyncio event loop.
   Suitable when the REPL loop is also asyncio-based.

2. **Thread polling**: override `_poll_thread()` as a synchronous
   generator.  The base class runs it in a daemon thread and bridges
   messages into an asyncio.Queue via `loop.call_soon_threadsafe`.
   This is REQUIRED when the REPL loop blocks on prompt_toolkit
   (because prompt blocks the main thread and no asyncio task can run).

Set ``thread_based=True`` in __init__ to use thread polling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging
import queue as _queue
import threading

log = logging.getLogger(__name__)


@dataclass
class RemoteMessage:
    """A message received from a remote messaging platform."""

    provider: str  # e.g. "telegram", "slack"
    text: str
    sender_id: str  # platform-specific user/chat ID
    chat_id: str  # platform-specific chat/thread ID
    message_id: Optional[str] = None  # platform-specific message ID for replies
    raw: dict = field(default_factory=dict)  # full raw payload


class RemoteProvider(ABC):
    """Base class for a remote messaging provider.

    Subclasses implement:
      - _poll()           → async generator yielding RemoteMessage objects
                           (only when thread_based=False)
      - _poll_thread()    → synchronous generator yielding RemoteMessage objects
                           (only when thread_based=True)
      - send_message()    → send text to a specific chat
      - send_chunk()      → send or update a streaming chunk
      - start/stop        → lifecycle management
    """

    def __init__(self, config: dict, *, thread_based: bool = False):
        self.config = config
        self._thread_based = thread_based
        self._aq: "asyncio.Queue[RemoteMessage]" = asyncio.Queue()
        self._squeue: "_queue.Queue[Optional[RemoteMessage]]" = _queue.Queue()
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'telegram')."""
        ...

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        """Start polling. Idempotent."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        if self._thread_based:
            self._poll_thread = threading.Thread(
                target=self._thread_poll_loop,
                daemon=True,
                name=f"poll-{self.name}",
            )
            self._poll_thread.start()
        else:
            self._poll_task = asyncio.create_task(self._async_poll_loop())
        log.info("Remote provider '%s' started (thread=%s)", self.name, self._thread_based)

    async def stop(self):
        """Stop polling. Idempotent."""
        if not self._running:
            return
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._poll_thread:
            # Push a sentinel to unblock the thread's queue.get()
            self._squeue.put(None)
            self._poll_thread.join(timeout=5)
            self._poll_thread = None
        log.info("Remote provider '%s' stopped", self.name)

    # ── Message reading ────────────────────────────────────────────────

    async def next_message(self) -> Optional[RemoteMessage]:
        """Get the next queued message, or None if empty."""
        try:
            return await asyncio.wait_for(self._aq.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    # ── Poll loops ─────────────────────────────────────────────────────

    async def _async_poll_loop(self):
        """Continuously poll for new messages (asyncio-based)."""
        while self._running:
            try:
                async for msg in self._poll():
                    await self._aq.put(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Poll error in provider '%s'", self.name)
                await asyncio.sleep(5)

    def _thread_poll_loop(self):
        """Continuously poll for new messages in a daemon thread.

        Bridges messages into the asyncio queue via
        ``loop.call_soon_threadsafe``.
        """
        while self._running:
            try:
                for msg in self._poll_thread():
                    if msg is None:
                        return  # sentinel
                    # Bridge from thread → asyncio queue
                    if self._loop and self._loop.is_running():
                        self._loop.call_soon_threadsafe(self._aq.put_nowait, msg)
                    # Wake up prompt_toolkit so the message gets seen
                    self._wake_prompt_toolkit()
            except Exception:
                log.exception("Thread poll error in provider '%s'", self.name)
                import time
                time.sleep(5)

    @staticmethod
    def _wake_prompt_toolkit():
        """Unblock prompt_toolkit if it is currently waiting for input."""
        try:
            from prompt_toolkit.application.current import get_app
            app = get_app()
            if app and app.is_running:
                app.exit(result='')
        except Exception:
            pass

    # ── Subclass hooks ─────────────────────────────────────────────────

    def _poll(self):
        """Async generator yielding RemoteMessage objects.

        Override this when ``thread_based=False`` (default).
        """
        raise NotImplementedError("_poll() not implemented")
        yield  # make this an async generator

    def _poll_thread(self):
        """Synchronous generator yielding RemoteMessage objects.

        Override this when ``thread_based=True``.
        """
        raise NotImplementedError("_poll_thread() not implemented")
        yield  # make this a generator

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to the given chat/thread."""
        ...

    @abstractmethod
    async def send_chunk(
        self, chat_id: str, text: str, message_id: Optional[str] = None
    ) -> Optional[str]:
        """Send or update a streaming chunk.

        If *message_id* is provided, update that existing message.
        Returns the message_id of the message (for subsequent edits).
        """
        ...
