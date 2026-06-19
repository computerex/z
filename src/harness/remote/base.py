"""Base class for remote input/output providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging

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
      - _poll()        → yield RemoteMessage objects (one per incoming message)
      - send_message() → send text to a specific chat
      - start/stop     → lifecycle management
    """

    def __init__(self, config: dict):
        self.config = config
        self._queue: "asyncio.Queue[RemoteMessage]" = asyncio.Queue()
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'telegram')."""
        ...

    async def start(self):
        """Start polling. Idempotent."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("Remote provider '%s' started", self.name)

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
        log.info("Remote provider '%s' stopped", self.name)

    async def next_message(self) -> Optional[RemoteMessage]:
        """Get the next queued message, or None if empty."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def _poll_loop(self):
        """Continuously poll for new messages."""
        while self._running:
            try:
                async for msg in self._poll():
                    await self._queue.put(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Poll error in provider '%s'", self.name)
                await asyncio.sleep(5)  # backoff on error

    @abstractmethod
    def _poll(self):
        """Async generator yielding RemoteMessage objects."""
        ...

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to the given chat/thread."""
        ...

    @abstractmethod
    async def send_chunk(self, chat_id: str, text: str, message_id: Optional[str] = None) -> Optional[str]:
        """Send or update a streaming chunk.

        If *message_id* is provided, update that existing message.
        Returns the message_id of the message (for subsequent edits).
        """
        ...
