"""Telegram remote input/output provider.

Uses the Telegram Bot API via HTTP (no third-party library required beyond httpx).
Polling-based: fetches new messages via getUpdates every second.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, AsyncIterator, Optional

import httpx

from .base import RemoteMessage, RemoteProvider

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot"
_MAX_MESSAGE_LEN = 4096  # Telegram's max message length
_CHUNK_INTERVAL = 0.3  # seconds between sending streaming chunks


def _split_long_message(text: str, max_len: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split a long message into multiple chunks at sensible boundaries."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # Try to split at a newline, otherwise at a space, otherwise hard-cut
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1 or split_at < max_len // 2:
            split_at = max_len
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    return parts


class TelegramProvider(RemoteProvider):
    """Remote provider that connects to the Telegram Bot API."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._token: str = config["token"]
        self._allowed_chat_ids: set[str] = set()
        self._http: httpx.AsyncClient | None = None

        # Parse allowed users from config
        raw_allowed = config.get("allowed_chat_ids", "")
        if raw_allowed:
            for cid in raw_allowed.split(","):
                cid = cid.strip()
                if cid:
                    self._allowed_chat_ids.add(cid)

        # Track last processed update_id to avoid duplicates
        self._last_update_id: int = 0

        # Track streaming message IDs per chat
        self._stream_msg_ids: dict[str, str] = {}

        # Pending authentication requests: chat_id -> asyncio.Event
        self._pending_auth: dict[str, asyncio.Event] = {}

    @property
    def name(self) -> str:
        return "telegram"

    async def _ensure_client(self):
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)

    async def _api(self, method: str, **kwargs) -> dict[str, Any]:
        """Call a Telegram Bot API method."""
        await self._ensure_client()
        url = f"{_TELEGRAM_API}{self._token}/{method}"
        try:
            r = await self._http.post(url, json=kwargs)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                log.warning("Telegram API error: %s", data.get("description", "unknown"))
            return data
        except httpx.HTTPStatusError as e:
            log.error("Telegram API HTTP error: %s", e)
            return {"ok": False, "description": str(e)}
        except httpx.TimeoutException:
            log.warning("Telegram API timeout")
            return {"ok": False, "description": "timeout"}

    async def _poll(self) -> AsyncIterator[RemoteMessage]:
        """Poll getUpdates, yielding messages from allowed users."""
        await self._ensure_client()

        params: dict[str, Any] = {
            "timeout": 10,
            "allowed_updates": ["message"],
        }
        if self._last_update_id:
            params["offset"] = self._last_update_id + 1

        try:
            r = await self._http.get(
                f"{_TELEGRAM_API}{self._token}/getUpdates",
                params=params,
                timeout=12.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.debug("Telegram poll error: %s", e)
            return

        if not data.get("ok"):
            return

        for update in data.get("result", []):
            upd_id = update.get("update_id", 0)
            if upd_id > self._last_update_id:
                self._last_update_id = upd_id

            msg = update.get("message")
            if not msg:
                continue

            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if not chat_id:
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            sender_id = str(msg.get("from", {}).get("id", chat_id))

            yield RemoteMessage(
                provider="telegram",
                text=text,
                sender_id=sender_id,
                chat_id=chat_id,
                message_id=str(msg.get("message_id", "")),
                raw=msg,
            )

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to the given chat."""
        for part in _split_long_message(text):
            await self._api("sendMessage", chat_id=chat_id, text=part)

    async def send_chunk(self, chat_id: str, text: str, message_id: Optional[str] = None) -> Optional[str]:
        """Send or update a streaming chunk message.

        If *message_id* is provided, edit that existing message.
        Returns the message_id of the message (for subsequent edits).
        """
        if message_id:
            # Edit existing message
            resp = await self._api(
                "editMessageText",
                chat_id=chat_id,
                message_id=int(message_id),
                text=text[:_MAX_MESSAGE_LEN],
            )
            if resp.get("ok"):
                result = resp.get("result", {})
                return str(result.get("message_id", message_id))
            # If edit fails (e.g. message too old), send a new one
            log.debug("Telegram edit failed, sending new message")
            return await self.send_chunk(chat_id, text, message_id=None)
        else:
            # Send new message
            resp = await self._api(
                "sendMessage",
                chat_id=chat_id,
                text=text[:_MAX_MESSAGE_LEN],
            )
            if resp.get("ok"):
                result = resp.get("result", {})
                return str(result.get("message_id", ""))
            return None

    async def start(self):
        """Start the provider and verify the bot token first."""
        await self._ensure_client()
        # Verify token
        me = await self._api("getMe")
        if not me.get("ok"):
            log.error(
                "Telegram bot token is invalid. "
                "Get a valid token from @BotFather on Telegram."
            )
            raise RuntimeError("Invalid Telegram bot token")
        bot_user = me.get("result", {})
        bot_name = bot_user.get("first_name", "?")
        bot_username = bot_user.get("username", "?")
        log.info(
            "Telegram bot '%s' (@%s) connected",
            bot_name,
            bot_username,
        )
        # If no allowed_chat_ids configured, print a warning
        if not self._allowed_chat_ids:
            log.warning(
                "No allowed_chat_ids configured! Any user who finds your bot can interact with it. "
                "Set TELEGRAM_ALLOWED_CHAT_IDS env var or pass via --telegram-allow-list."
            )
        await super().start()

    def is_chat_allowed(self, chat_id: str) -> bool:
        """Check if a chat is authorized to interact with the bot."""
        if not self._allowed_chat_ids:
            return True  # No restrictions
        return chat_id in self._allowed_chat_ids

    def authorize_chat(self, chat_id: str) -> None:
        """Add a chat to the allowed list (after manual approval)."""
        self._allowed_chat_ids.add(chat_id)

    async def send_auth_request(self, chat_id: str) -> bool:
        """Send an authentication request and wait for approval."""
        await self._api(
            "sendMessage",
            chat_id=chat_id,
            text=(
                "⚠️ *Unauthorized*\n\n"
                "This bot requires authorization. "
                "Run this command in the harness terminal to approve:\n\n"
                f"`/telegram-auth {chat_id}`"
            ),
            parse_mode="Markdown",
        )
        # Wait for approval (up to 5 minutes)
        event = asyncio.Event()
        self._pending_auth[chat_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=300)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending_auth.pop(chat_id, None)

    def approve_pending(self, chat_id: str) -> bool:
        """Approve a pending authorization request. Called from /telegram-auth command."""
        event = self._pending_auth.get(chat_id)
        if event:
            event.set()
            return True
        return False
