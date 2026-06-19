"""Telegram remote input/output provider.

Uses the Telegram Bot API via HTTP calls.  Polling runs in a **background
thread** (``thread_based=True``) so that incoming messages are discovered even
when the REPL loop is blocked on ``prompt_toolkit``.

When a new message arrives, the polling thread:
  1. Bridges the message into the asyncio queue via ``loop.call_soon_threadsafe``
  2. Calls ``app.exit('')`` to unblock prompt_toolkit (same pattern as cron)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import httpx

from .base import RemoteMessage, RemoteProvider

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot"
_MAX_MESSAGE_LEN = 4096  # Telegram's max message length


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


def _tg_api_call(token: str, method: str, **kwargs) -> dict[str, Any]:
    """Synchronous call to Telegram Bot API.

    Used by the background polling thread.  Not the asyncio path.
    """
    url = f"{_TELEGRAM_API}{token}/{method}"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=kwargs)
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
    except Exception as e:
        log.debug("Telegram API call failed: %s", e)
        return {"ok": False, "description": str(e)}


class TelegramProvider(RemoteProvider):
    """Remote provider that connects to the Telegram Bot API.

    Polling runs in a daemon thread so it works even when the REPL is
    blocked on prompt_toolkit.
    """

    def __init__(self, config: dict):
        super().__init__(config, thread_based=True)
        self._token: str = config["token"]
        self._allowed_chat_ids: set[str] = set()
        self._allowed_username: Optional[str] = None
        self._http_sync: httpx.Client | None = None

        # Parse allowed users from config
        raw_allowed = config.get("allowed_chat_ids", "")
        if raw_allowed:
            for cid in raw_allowed.split(","):
                cid = cid.strip()
                if cid:
                    self._allowed_chat_ids.add(cid)

        # Parse allowed Telegram username (e.g. "computerex_1992")
        raw_username = config.get("allowed_username", "")
        if raw_username:
            self._allowed_username = raw_username.lstrip("@").strip()

        # Track last processed update_id to avoid duplicates
        self._last_update_id: int = 0

        # Pending authentication requests: chat_id -> threading.Event
        self._pending_auth: dict[str, threading.Event] = {}

    @property
    def name(self) -> str:
        return "telegram"

    # ── Async API calls (used by send_message / send_chunk) ────────────

    def _sync_api(self, method: str, **kwargs) -> dict[str, Any]:
        """Synchronous call to Telegram Bot API.

        Used both by the polling thread and internally for sends.
        """
        return _tg_api_call(self._token, method, **kwargs)

    # ── Thread-based polling ───────────────────────────────────────────

    def _poll_thread(self):
        """Synchronous generator — runs in a background daemon thread.

        Long-polls ``getUpdates`` with a 10-second timeout, then yields
        any messages found.
        """
        while True:
            params: dict[str, Any] = {
                "timeout": 10,
                "allowed_updates": ["message"],
            }
            if self._last_update_id:
                params["offset"] = self._last_update_id + 1

            data = _tg_api_call(self._token, "getUpdates", **params)
            if not data.get("ok"):
                time.sleep(5)
                continue

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

                # ── Username-based access control ──────────────────────
                if self._allowed_username:
                    sender_username = msg.get("from", {}).get("username", "")
                    if sender_username and sender_username.startswith("@"):
                        sender_username = sender_username[1:]
                    if sender_username.lower() != self._allowed_username.lower():
                        # Silently drop messages from other users
                        log.debug(
                            "Ignoring message from @%s (allowed: @%s)",
                            sender_username or "?",
                            self._allowed_username,
                        )
                        continue

                # ── Chat-ID-based access control (fallback) ────────────
                if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
                    log.debug(
                        "Ignoring message from chat %s (not in allow list)",
                        chat_id,
                    )
                    continue

                yield RemoteMessage(
                    provider="telegram",
                    text=text,
                    sender_id=sender_id,
                    chat_id=chat_id,
                    message_id=str(msg.get("message_id", "")),
                    raw=msg,
                )

    # ── Sending messages (called from asyncio side) ────────────────────

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a text message to the given chat."""
        for part in _split_long_message(text):
            self._sync_api("sendMessage", chat_id=chat_id, text=part)

    async def send_chunk(
        self, chat_id: str, text: str, message_id: Optional[str] = None
    ) -> Optional[str]:
        """Send or update a streaming chunk message.

        If *message_id* is provided, edit that existing message.
        Returns the message_id (for subsequent edits).
        """
        if message_id:
            # Edit existing message
            resp = self._sync_api(
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
            resp = self._sync_api(
                "sendMessage",
                chat_id=chat_id,
                text=text[:_MAX_MESSAGE_LEN],
            )
            if resp.get("ok"):
                result = resp.get("result", {})
                return str(result.get("message_id", ""))
            return None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        """Verify the bot token, then start the polling thread."""
        # Verify token (synchronous — quick check)
        me = _tg_api_call(self._token, "getMe")
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
        # Warn if neither username nor chat-ID filter is configured
        if not self._allowed_username and not self._allowed_chat_ids:
            log.warning(
                "No access restrictions configured! Any user who finds your bot "
                "can interact with it. "
                "Use --telegram-username <your_username> to restrict by username."
            )
        await super().start()

    # ── Authorization ─────────────────────────────────────────────────

    def is_chat_allowed(self, chat_id: str, username: Optional[str] = None) -> bool:
        """Check if a user/chat is authorized to interact with the bot."""
        if self._allowed_username:
            if username:
                clean = username.lstrip("@")
                if clean.lower() == self._allowed_username.lower():
                    return True
            return False
        if self._allowed_chat_ids:
            return chat_id in self._allowed_chat_ids
        return True  # No restrictions

    def authorize_chat(self, chat_id: str) -> None:
        """Add a chat to the allowed list (after manual approval)."""
        self._allowed_chat_ids.add(chat_id)

    async def send_auth_request(self, chat_id: str) -> bool:
        """Send an authentication request and wait for approval."""
        self._sync_api(
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
        event = threading.Event()
        self._pending_auth[chat_id] = event
        try:
            # Wait in a thread so we don't block the event loop
            import asyncio
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: event.wait(timeout=300)
            )
        finally:
            self._pending_auth.pop(chat_id, None)

    def approve_pending(self, chat_id: str) -> bool:
        """Approve a pending authorization request from /telegram-auth."""
        event = self._pending_auth.get(chat_id)
        if event:
            event.set()
            return True
        return False
