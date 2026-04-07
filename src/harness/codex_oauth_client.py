"""OpenAI Codex OAuth API client.

This client handles OAuth tokens for ChatGPT Plus/Pro subscriptions.
Unlike standard OpenAI API keys, OAuth tokens access the ChatGPT web backend directly.

Based on opencode's implementation:
- Uses https://chatgpt.com/backend-api/codex/responses endpoint
- Requires special headers (ChatGPT-Account-Id, originator)
- Handles token refresh automatically
- Supports streaming responses
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator, Union
from dataclasses import dataclass
import aiohttp

from .oauth import OAuthToken, get_oauth_manager

from .codex_models import ALLOWED_CODEX_MODELS, get_codex_models, is_codex_model

# OpenAI OAuth constants (must match oauth.py)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"

CODEX_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


@dataclass
class CodexMessage:
    """Message for Codex API."""

    role: str
    content: Union[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        # Convert multimodal content from Chat Completions format to
        # Responses API format (input_text / input_image instead of
        # text / image_url).
        converted: List[Dict[str, Any]] = []
        for block in self.content:
            btype = block.get("type", "")
            if btype == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif btype == "image_url":
                url = block.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                converted.append({"type": "input_image", "image_url": url})
            else:
                # Pass through unknown block types as-is
                converted.append(block)
        return {"role": self.role, "content": converted}


@dataclass
class CodexResponse:
    """Response from Codex API."""

    content: str
    thinking: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: str = "stop"
    interrupted: bool = False


class CodexOAuthClient:
    """Client for OpenAI Codex API using OAuth tokens."""

    def __init__(
        self,
        oauth_token: OAuthToken,
        model: str = "gpt-5.3-codex",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        """Initialize Codex OAuth client.

        Args:
            oauth_token: OAuth token from authentication
            model: Model name (must be a Codex model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
        """
        self.oauth_token = oauth_token
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_token_valid(self) -> str:
        """Ensure OAuth token is valid, refresh if needed.

        Returns:
            Valid access token
        """
        if self.oauth_token.is_expired():
            # Refresh token using aiohttp (non-blocking)
            session = self._session or aiohttp.ClientSession()
            close_after = self._session is None
            try:
                async with session.post(
                    f"{ISSUER}/oauth/token",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.oauth_token.refresh_token,
                        "client_id": CLIENT_ID,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(f"Token refresh failed: {response.status}")

                    tokens = await response.json()
            finally:
                if close_after:
                    await session.close()

            self.oauth_token.access_token = tokens["access_token"]
            self.oauth_token.refresh_token = tokens.get(
                "refresh_token", self.oauth_token.refresh_token
            )
            self.oauth_token.expires_at = time.time() + tokens.get("expires_in", 3600)

            # Update stored token
            oauth_manager = get_oauth_manager()
            oauth_manager._tokens["openai"] = self.oauth_token
            oauth_manager._save_token("openai", self.oauth_token)

        return self.oauth_token.access_token

    def _build_headers(self, access_token: str) -> Dict[str, str]:
        """Build headers for Codex API request.

        Args:
            access_token: OAuth access token

        Returns:
            Headers dict
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "originator": "harness",
            "User-Agent": "harness/1.0.0",
        }

        # Add ChatGPT-Account-Id if available
        if self.oauth_token.account_id:
            headers["ChatGPT-Account-Id"] = self.oauth_token.account_id

        return headers

    def _build_request_body(
        self, messages: List[CodexMessage], stream: bool = True, system_prompt: str = ""
    ) -> Dict[str, Any]:
        """Build request body for Codex API (OpenAI Responses API format).

        Args:
            messages: List of messages
            stream: Whether to stream response
            system_prompt: System instructions (required for Codex)

        Returns:
            Request body dict
        """
        # Convert messages to input format (Responses API uses 'input' not 'messages')
        input_messages = []
        for msg in messages:
            if (
                msg.role != "system"
            ):  # System messages go in 'instructions', not 'input'
                input_messages.append(msg.to_dict())

        # Ensure we have at least one message - Codex API requires non-empty input
        if not input_messages:
            # Add a dummy user message if none provided
            input_messages = [{"role": "user", "content": "Hello"}]

        # Note: Codex models support large context windows (272K tokens input)
        # No artificial truncation needed - the API will handle context limits

        # Build request body - Responses API format
        # Note: The 'instructions' field is REQUIRED and must be a non-empty string
        # Note: 'store' must be set to false for Codex API
        # Note: 'temperature' and 'max_output_tokens' are NOT supported by Codex models
        body: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "stream": stream,
            "store": False,  # Required: Codex doesn't store conversations
            "instructions": system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else "You are a helpful coding assistant.",
        }

        # Note: Codex models don't support temperature or max_tokens parameters
        # They use fixed settings optimized for coding tasks

        return body

    async def chat_stream(
        self,
        messages: List[CodexMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
        system_prompt: str = "",
    ) -> CodexResponse:
        """Stream chat response from Codex API.

        Args:
            messages: List of messages
            on_content: Callback for content chunks
            on_reasoning: Callback for reasoning/thinking content
            check_interrupt: Callback to check if interrupted

        Returns:
            CodexResponse with full content
        """
        access_token = await self._ensure_token_valid()
        headers = self._build_headers(access_token)
        body = self._build_request_body(
            messages, stream=True, system_prompt=system_prompt
        )

        if not self._session:
            self._session = aiohttp.ClientSession()

        full_content = ""
        full_reasoning = ""
        usage = {}
        finish_reason = "stop"
        interrupted = False

        try:
            async with self._session.post(
                CODEX_API_ENDPOINT,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Codex API error {response.status}: {error_text}"
                    )

                # Process SSE stream
                async for line in response.content:
                    # Check for interrupt
                    if check_interrupt and check_interrupt():
                        interrupted = True
                        finish_reason = "interrupted"
                        break

                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Handle different event types
                    event_type = event.get("type")

                    if event_type == "response.output_text.delta":
                        # Content delta
                        delta = event.get("delta", "")
                        if delta:
                            full_content += delta
                            if on_content:
                                on_content(delta)

                    elif event_type == "response.reasoning":
                        # Reasoning/thinking content
                        reasoning = event.get("reasoning", "")
                        if reasoning:
                            full_reasoning += reasoning
                            if on_reasoning:
                                on_reasoning(reasoning)

                    elif event_type == "response.reasoning_summary_text.delta":
                        # Alternative reasoning format
                        delta = event.get("delta", "")
                        if delta:
                            full_reasoning += delta
                            if on_reasoning:
                                on_reasoning(delta)

                    elif event_type == "response.completed":
                        # Response completed
                        output = event.get("output", [])
                        if output:
                            # Extract usage from first output
                            first_output = output[0]
                            if "usage" in first_output:
                                usage = {
                                    "prompt_tokens": first_output["usage"].get(
                                        "input_tokens", 0
                                    ),
                                    "completion_tokens": first_output["usage"].get(
                                        "output_tokens", 0
                                    ),
                                    "total_tokens": first_output["usage"].get(
                                        "total_tokens", 0
                                    ),
                                }

                    elif event_type == "response.incomplete":
                        # Response incomplete (e.g., max tokens reached)
                        finish_reason = "length"

                    elif event_type == "error":
                        error_msg = event.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        raise RuntimeError(f"Codex API error: {error_msg}")

        except asyncio.CancelledError:
            interrupted = True
            finish_reason = "interrupted"
        except Exception as e:
            raise RuntimeError(f"Codex streaming error: {e}")

        return CodexResponse(
            content=full_content,
            thinking=full_reasoning or None,
            usage=usage if usage else None,
            finish_reason=finish_reason,
            interrupted=interrupted,
        )

    async def chat(
        self,
        messages: List[CodexMessage],
    ) -> CodexResponse:
        """Non-streaming chat (for simple requests).

        Args:
            messages: List of messages

        Returns:
            CodexResponse
        """
        return await self.chat_stream(messages)


def is_oauth_token(api_key: str) -> bool:
    """Check if API key is an OAuth token.

    Args:
        api_key: API key string

    Returns:
        True if it's an OAuth token
    """
    return api_key.startswith("oauth:")


def extract_oauth_token(api_key: str) -> Optional[str]:
    """Extract OAuth token from API key string.

    Args:
        api_key: API key string (format: "oauth:<token>")

    Returns:
        OAuth token or None if not OAuth
    """
    if is_oauth_token(api_key):
        return api_key[6:]  # Remove "oauth:" prefix
    return None


# get_codex_models and is_codex_model are now imported from codex_models module
