"""Base class for remote input/output providers.

Two polling models are supported:

1. **Async polling** (default): override `_poll()` as an async generator.
   The provider must be started/stopped from an asyncio event loop.
   Suitable when the REPL loop is also asyncio-based.

2. **Thread polling**: override `_poll_thread()` as a synchronous
   generator.  The base class runs it in a daemon thread and stores
   messages in a thread-safe deque.
   This is REQUIRED when the REPL loop blocks on prompt_toolkit
   (because prompt blocks the main thread and no asyncio task can run).

Set ``thread_based=True`` in __init__ to use thread polling.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging
import threading

log = logging.getLogger(__name__)

# ── Cross-thread prompt_toolkit Application reference ───────────────
#
# The main thread captures the Application object and stores it here so
# the background polling thread can call app.exit('') to unblock the
# prompt without needing get_app() (which uses thread-local storage).

_pt_app: Optional["Application"] = None  # type: ignore[name-defined]


def set_pt_app(app: Optional["Application"]) -> None:  # type: ignore[name-defined]
    """Store a reference to the running prompt_toolkit Application.

    Called from the main thread after the first ``prompt()`` call so
    the background polling thread can wake prompt_toolkit.
    """
    global _pt_app
    _pt_app = app


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

        # Thread-safe message buffer: the polling thread (or async loop)
        # appends messages here; next_message() pops them.
        self._buf: deque[RemoteMessage] = deque()
        self._buf_lock: threading.Lock = threading.Lock()

        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._poll_thr: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

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
        self._stop_event.clear()
        if self._thread_based:
            self._poll_thr = threading.Thread(
                target=self._thread_poll_loop,
                daemon=True,
                name=f"poll-{self.name}",
            )
            self._poll_thr.start()
        else:
            self._poll_task = asyncio.create_task(self._async_poll_loop())
        log.info("Remote provider '%s' started (thread=%s)", self.name, self._thread_based)

    async def stop(self):
        """Stop polling. Idempotent."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()  # unblock thread if sleeping
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._poll_thr:
            self._poll_thr.join(timeout=5)
            self._poll_thr = None
        log.info("Remote provider '%s' stopped", self.name)

    # ── Message reading ────────────────────────────────────────────────

    async def next_message(self) -> Optional[RemoteMessage]:
        """Get the next queued message, or None if empty."""
        with self._buf_lock:
            if self._buf:
                return self._buf.popleft()
            return None

    def _enqueue_message(self, msg: RemoteMessage) -> None:
        """Thread-safe append to the message buffer."""
        with self._buf_lock:
            self._buf.append(msg)

    # ── Poll loops ─────────────────────────────────────────────────────

    async def _async_poll_loop(self):
        """Continuously poll for new messages (asyncio-based)."""
        while self._running:
            try:
                async for msg in self._poll():
                    self._enqueue_message(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Poll error in provider '%s'", self.name)
                await asyncio.sleep(5)

    def _thread_poll_loop(self):
        """Continuously poll for new messages in a daemon thread.

        Messages are stored directly into a thread-safe deque (no asyncio
        bridging needed). prompt_toolkit is woken via the stored
        Application reference instead of ``get_app()`` (which uses
        thread-local storage and doesn't work from background threads).
        """
        while self._running:
            try:
                for msg in self._poll_thread():
                    if msg is None:
                        return  # sentinel
                    # Thread-safe: store directly into deque
                    self._enqueue_message(msg)
                    # Wake prompt_toolkit from here (direct call, not
                    # call_soon_threadsafe — the event loop may not be
                    # running when the REPL is blocked on prompt()).
                    self._wake_prompt_toolkit()
            except Exception:
                log.exception("Thread poll error in provider '%s'", self.name)
                # Wait with interruptible sleep so stop() is responsive
                if self._stop_event.wait(timeout=5):
                    break

    @staticmethod
    def _wake_prompt_toolkit():
        """Unblock prompt_toolkit if it is currently waiting for input.

        Uses the module-level ``_pt_app`` reference set by the main thread
        rather than ``get_app()``, because ``get_app()`` uses thread-local
        storage and cannot find the Application from a background thread.
        """
        global _pt_app
        if _pt_app is not None:
            try:
                if hasattr(_pt_app, 'is_running') and _pt_app.is_running:
                    _pt_app.exit(result='')
            except Exception:
                pass

    # ── Subclass hooks ─────────────────────────────────────────────────

    def _poll(self):
        """Async generator yielding RemoteMessage objects.

        Override this when ``thread_based=False`` (default).
        """
        raise NotImplementedError("_poll() not implemented")

    def _poll_thread(self):
        """Synchronous generator yielding RemoteMessage objects.

        Override this when ``thread_based=True``.
        """
        raise NotImplementedError("_poll_thread() not implemented")

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
