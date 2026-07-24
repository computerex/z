"""RemoteInputManager — coordinates multiple remote input/output providers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .base import RemoteMessage, RemoteProvider

log = logging.getLogger(__name__)


class RemoteInputManager:
    """Manages one or more remote messaging providers.

    Handles:
      - Starting/stopping providers
      - Queuing incoming messages from all providers
      - Routing output back to the correct provider/chat
      - Authorization flow
    """

    def __init__(self):
        self._providers: dict[str, RemoteProvider] = {}
        self._queue: asyncio.Queue[RemoteMessage] = asyncio.Queue()
        self._started = False

    def register(self, provider: RemoteProvider) -> None:
        """Register a provider (idempotent on name)."""
        name = provider.name
        if name in self._providers:
            log.warning("Provider '%s' already registered, skipping", name)
            return
        self._providers[name] = provider
        log.info("Registered remote provider '%s'", name)

    async def start_all(self) -> None:
        """Start all registered providers."""
        if self._started:
            return
        errors = []
        for name, provider in self._providers.items():
            try:
                await provider.start()
            except Exception as e:
                log.error("Failed to start provider '%s': %s", name, e)
                errors.append((name, str(e)))
        self._started = True
        if errors:
            raise RuntimeError(
                f"Failed to start {len(errors)} provider(s): {errors}"
            )

    async def stop_all(self) -> None:
        """Stop all registered providers."""
        for provider in self._providers.values():
            try:
                await provider.stop()
            except Exception:
                log.exception("Error stopping provider '%s'", provider.name)

    async def get_pending_messages(self) -> list[RemoteMessage]:
        """Drain all queued messages from all providers.

        This should be called from the main loop before waiting for user input.
        """
        messages = []

        # Check each provider for new messages
        for provider in self._providers.values():
            while True:
                msg = await provider.next_message()
                if msg is None:
                    break
                messages.append(msg)

        # Also drain our own queue (for messages queued via observer pattern)
        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break

        return messages

    async def send_message(self, provider: str, chat_id: str, text: str) -> None:
        """Send a text message via the specified provider."""
        prov = self._providers.get(provider)
        if prov is None:
            log.warning("Unknown provider '%s'", provider)
            return
        await prov.send_message(chat_id, text)

    async def send_chunk(
        self,
        provider: str,
        chat_id: str,
        text: str,
        message_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send or update a streaming chunk via the specified provider."""
        prov = self._providers.get(provider)
        if prov is None:
            log.warning("Unknown provider '%s'", provider)
            return None
        return await prov.send_chunk(chat_id, text, message_id)

    def get_provider(self, name: str) -> Optional[RemoteProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def has_providers(self) -> bool:
        """Check if any providers are registered."""
        return len(self._providers) > 0
