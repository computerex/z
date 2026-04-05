"""GitHub Copilot API client for OAuth tokens.

This client handles OAuth tokens for GitHub Copilot subscriptions.
Unlike standard API keys, Copilot OAuth tokens access GitHub's API directly.

Based on opencode's implementation:
- Uses https://api.githubcopilot.com or enterprise endpoints
- Requires special headers (x-initiator, Copilot-Vision-Request)
- Supports streaming responses with reasoning/thinking tokens
- Properly handles reasoning_opaque for multi-turn conversations
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import aiohttp

from .oauth import OAuthToken

log = logging.getLogger(__name__)

# GitHub Copilot API endpoints
COPILOT_API_BASE = "https://api.githubcopilot.com"
COPILOT_CHAT_ENDPOINT = "/chat/completions"

# Allowed Copilot models (from models.dev - https://models.dev/api.json)
ALLOWED_COPILOT_MODELS = {
    # Claude models
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-41",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    # GPT models
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-copilot",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4",
    # Gemini models
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    # Grok
    "grok-code-fast-1",
}


@dataclass
class CopilotMessage:
    """Message for Copilot API with optional reasoning context."""

    role: str
    content: Union[str, List[Dict[str, Any]]]
    # For assistant messages, store reasoning_opaque for multi-turn context
    reasoning_opaque: Optional[str] = None
    # Store reasoning text for display
    reasoning_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}

        # For assistant messages with reasoning, include provider metadata
        if self.role == "assistant" and self.reasoning_opaque:
            msg["reasoning_opaque"] = self.reasoning_opaque

        return msg


@dataclass
class CopilotResponse:
    """Response from Copilot API with reasoning support."""

    content: str
    thinking: Optional[str] = None
    # Encrypted reasoning token for multi-turn context (CRITICAL)
    reasoning_opaque: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: str = "stop"
    interrupted: bool = False


class CopilotOAuthClient:
    """Client for GitHub Copilot API using OAuth tokens."""

    def __init__(
        self,
        oauth_token: OAuthToken,
        model: str = "gpt-4o-copilot",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        enterprise_url: Optional[str] = None,
    ):
        """Initialize Copilot OAuth client.

        Args:
            oauth_token: OAuth token from authentication
            model: Model name (must be a Copilot model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
            enterprise_url: GitHub Enterprise URL (optional)
        """
        self.oauth_token = oauth_token
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort: str = "high"
        self._session: Optional[aiohttp.ClientSession] = None

        # Set base URL
        if enterprise_url:
            domain = (
                enterprise_url.replace("https://", "")
                .replace("http://", "")
                .rstrip("/")
            )
            self.base_url = f"https://copilot-api.{domain}"
        else:
            self.base_url = COPILOT_API_BASE

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _build_headers(self) -> Dict[str, str]:
        """Build headers for Copilot API request."""
        return {
            "Authorization": f"Bearer {self.oauth_token.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "harness/1.0.0",
            "x-initiator": "agent",
            "Openai-Intent": "conversation-edits",
        }

    def _build_request_body(
        self, messages: List[CopilotMessage], stream: bool = True
    ) -> Dict[str, Any]:
        """Build request body for Copilot API with reasoning support."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }

        # Enable reasoning (returns reasoning_text in SSE deltas)
        if self.reasoning_effort and self.reasoning_effort != "none":
            body["reasoning_effort"] = self.reasoning_effort

        # Add optional parameters
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens

        return body

    def _has_vision_content(self, messages: List[CopilotMessage]) -> bool:
        """Check if any message contains image/vision content."""
        for m in messages:
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    async def chat_stream(
        self,
        messages: List[CopilotMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> CopilotResponse:
        """Stream chat response from Copilot API with reasoning support."""
        headers = self._build_headers()
        if self._has_vision_content(messages):
            headers["Copilot-Vision-Request"] = "true"
        body = self._build_request_body(messages, stream=True)

        if not self._session:
            self._session = aiohttp.ClientSession()

        full_content = ""
        full_reasoning = ""
        reasoning_opaque = None
        usage = {}
        finish_reason = "stop"
        interrupted = False

        try:
            url = f"{self.base_url}{COPILOT_CHAT_ENDPOINT}"
            async with self._session.post(
                url,
                headers=headers,
                json=body,
                # Use sock_read (idle timeout) instead of total timeout.
                # total= kills long-but-active streams; sock_read= only
                # fires when no data arrives for N seconds.
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=30,
                    sock_read=self.timeout,
                ),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Copilot API error {response.status}: {error_text}"
                    )

                # Process SSE stream
                async for line in response.content:
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

                    # Handle chat completion format (most common)
                    if "choices" in event and event["choices"]:
                        choice = event["choices"][0]
                        delta = choice.get("delta", {})

                        # Log delta keys on first chunk for debugging reasoning issues
                        if delta and not full_content and not full_reasoning:
                            log.debug("copilot SSE first delta keys: %s", list(delta.keys()))

                        # Regular content
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            if on_content:
                                on_content(content)

                        # Reasoning content (for Claude models through Copilot)
                        # Check both field names: API may use reasoning_content or reasoning_text
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning_text")
                            or delta.get("reasoning")
                            or ""
                        )
                        if reasoning:
                            full_reasoning += reasoning
                            if on_reasoning:
                                on_reasoning(reasoning)

                        # Store reasoning_opaque for multi-turn context (CRITICAL)
                        # This must be passed back in subsequent requests
                        message = choice.get("message", {})
                        if message.get("reasoning_opaque"):
                            reasoning_opaque = message["reasoning_opaque"]

                        # Also check delta for reasoning_opaque
                        if delta.get("reasoning_opaque"):
                            reasoning_opaque = delta["reasoning_opaque"]

                        # Finish reason
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                    # Extract usage if present
                    if "usage" in event:
                        usage = {
                            "prompt_tokens": event["usage"].get("prompt_tokens", 0),
                            "completion_tokens": event["usage"].get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": event["usage"].get("total_tokens", 0),
                        }

        except asyncio.CancelledError:
            interrupted = True
            finish_reason = "interrupted"
        except asyncio.TimeoutError:
            # Idle timeout — no data from API for self.timeout seconds.
            # If we have partial content, return it as interrupted rather
            # than crashing.
            if full_content:
                interrupted = True
                finish_reason = "interrupted"
            else:
                raise RuntimeError(
                    f"Copilot streaming error: read timeout after {self.timeout}s with no data"
                )
        except Exception as e:
            raise RuntimeError(f"Copilot streaming error: {type(e).__name__}: {e}")

        return CopilotResponse(
            content=full_content,
            thinking=full_reasoning if full_reasoning else None,
            reasoning_opaque=reasoning_opaque,
            usage=usage if usage else None,
            finish_reason=finish_reason,
            interrupted=interrupted,
        )

    async def chat(self, messages: List[CopilotMessage]) -> CopilotResponse:
        """Non-streaming chat (for simple requests)."""
        return await self.chat_stream(messages)


def is_oauth_token(api_key: str) -> bool:
    """Check if API key is an OAuth token."""
    return api_key.startswith("oauth:")


def extract_oauth_token(api_key: str) -> Optional[str]:
    """Extract OAuth token from API key string."""
    if is_oauth_token(api_key):
        return api_key[6:]
    return None


def get_copilot_models() -> List[str]:
    """Get list of allowed Copilot models."""
    return sorted(ALLOWED_COPILOT_MODELS)


def is_copilot_model(model: str) -> bool:
    """Check if model is a Copilot model."""
    return model in ALLOWED_COPILOT_MODELS
