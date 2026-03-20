"""LiteLLM-based streaming client for harness.

This replaces all provider-specific code with a unified LiteLLM API.
Supports 100+ providers including Bedrock, Anthropic, OpenAI, Together, etc.
"""

import os
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass

# Import LiteLLM
try:
    import litellm
    from litellm import completion, acompletion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    completion = None
    acompletion = None


@dataclass
class StreamingMessage:
    """Message format for chat."""

    role: str
    content: Union[str, List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": self.content}

    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM message format."""
        return self.to_dict()


@dataclass
class StreamingChatResponse:
    """Response from streaming chat."""

    content: str
    thinking: Optional[str] = None
    raw_json: str = ""
    usage: Optional[Dict[str, int]] = None
    finish_reason: str = "stop"
    interrupted: bool = False
    provider_content_blocks: Optional[List[Dict[str, Any]]] = None


def _normalize_model_name(model: str, base_url: str = "") -> str:
    """Normalize model name for LiteLLM.

    LiteLLM expects format: provider/model_name
    e.g., "bedrock/qwen.qwen3-32b-v1:0", "openai/gpt-4o"
    """
    # If already has provider prefix, return as-is
    if "/" in model and not model.startswith("http"):
        return model

    # Detect provider from base_url
    url = (base_url or "").lower()

    if "bedrock" in url and "amazonaws.com" in url:
        return f"bedrock/{model}"
    elif "anthropic.com" in url:
        return f"anthropic/{model}"
    elif "openrouter.ai" in url:
        return f"openrouter/{model}"
    elif "together.xyz" in url:
        return f"together_ai/{model}"
    elif "minimax" in url:
        return f"minimax/{model}"
    elif "api.z.ai" in url:
        # Z.AI uses OpenAI-compatible format
        return f"openai/{model}"
    elif "api.openai.com" in url:
        return f"openai/{model}"

    # Default to openai format for OpenAI-compatible APIs
    return f"openai/{model}"


class StreamingJSONClient:
    """LLM Client using LiteLLM for unified provider support."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is required. Install with: pip install litellm")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        # Normalize model name for LiteLLM
        self.litellm_model = _normalize_model_name(model, base_url)

    def _provider_kind(self) -> str:
        """Infer provider from model name or URL."""
        # Check model name first
        if self.litellm_model.startswith("bedrock/"):
            return "litellm_bedrock"
        elif self.litellm_model.startswith("anthropic/"):
            return "litellm_anthropic"
        elif self.litellm_model.startswith("openrouter/"):
            return "litellm_openrouter"
        elif self.litellm_model.startswith("together_ai/"):
            return "litellm_together"

        # Fall back to URL detection
        url = (self.base_url or "").lower()
        if "bedrock" in url and "amazonaws.com" in url:
            return "litellm_bedrock"
        elif "anthropic.com" in url:
            return "litellm_anthropic"
        elif "openrouter.ai" in url:
            return "litellm_openrouter"
        elif "together.xyz" in url:
            return "litellm_together"

        return "litellm_openai_compat"

    async def __aenter__(self):
        """Async context manager entry."""
        # LiteLLM doesn't need client initialization
        # It handles connections internally
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        # LiteLLM handles cleanup internally
        pass

    async def _chat_stream_raw_litellm(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream chat using LiteLLM."""
        full_content = ""
        full_reasoning = ""
        usage = {}
        finish_reason = "stop"
        interrupted = False

        # Convert messages to LiteLLM format
        litellm_messages = [m.to_litellm_format() for m in messages]

        # Build request kwargs
        kwargs = {
            "model": self.litellm_model,
            "messages": litellm_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Add base_url if provided (for OpenAI-compatible APIs)
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # Enable detailed streaming for reasoning content
        litellm.set_verbose = False  # Set to True for debugging

        try:
            # Get streaming response
            response = await acompletion(**kwargs)

            # Process stream
            async for chunk in response:
                # Check for interrupt
                if check_interrupt and check_interrupt():
                    interrupted = True
                    finish_reason = "interrupted"
                    break

                # Extract content from chunk
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Regular content
                    if hasattr(delta, "content") and delta.content:
                        content = delta.content
                        full_content += content
                        if on_content:
                            on_content(content)

                    # Reasoning/thinking content (for models that support it)
                    if hasattr(delta, "provider_specific_fields"):
                        psf = delta.provider_specific_fields or {}
                        if "thinking" in psf:
                            thinking = psf.get("thinking", "")
                            if thinking:
                                full_reasoning += thinking
                                if on_reasoning:
                                    on_reasoning(thinking)

                # Extract usage info
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            chunk.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                    }

                # Extract finish reason
                if chunk.choices and len(chunk.choices) > 0:
                    finish_reason = (
                        getattr(chunk.choices[0], "finish_reason", finish_reason)
                        or finish_reason
                    )

            return StreamingChatResponse(
                content=full_content,
                thinking=full_reasoning or None,
                raw_json=full_content,
                usage=usage if usage else None,
                finish_reason=finish_reason,
                interrupted=interrupted,
            )

        except asyncio.CancelledError:
            interrupted = True
            finish_reason = "interrupted"
            return StreamingChatResponse(
                content=full_content,
                thinking=full_reasoning or None,
                raw_json=full_content,
                usage=usage if usage else None,
                finish_reason=finish_reason,
                interrupted=interrupted,
            )
        except Exception as e:
            # Re-raise as RuntimeError for consistency
            raise RuntimeError(f"LiteLLM error: {e}")

    async def chat_stream_raw(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream raw text response (no JSON parsing).

        This is the main entry point for streaming chat.
        """
        return await self._chat_stream_raw_litellm(
            messages=messages,
            on_content=on_content,
            on_reasoning=on_reasoning,
            check_interrupt=check_interrupt,
        )

    async def chat_stream(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        max_retries: int = 5,
    ) -> StreamingChatResponse:
        """Stream a chat response with real-time content display.

        This wraps chat_stream_raw with retry logic.
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.chat_stream_raw(
                    messages=messages,
                    on_content=on_content,
                    on_reasoning=on_thinking,
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = min(2**attempt, 30)
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {max_retries + 1} attempts: {e}")

        # Should not reach here, but just in case
        raise RuntimeError(f"Unexpected error: {last_error}")


def get_litellm_model_list() -> List[str]:
    """Get list of all models supported by LiteLLM.

    Returns a list of model names in format "provider/model_name".
    """
    if not LITELLM_AVAILABLE:
        return []

    try:
        # LiteLLM maintains a list of supported models
        from litellm import model_list

        return sorted(model_list, key=str.lower)
    except Exception:
        return []


def search_litellm_models(query: str) -> List[str]:
    """Search for models matching a query.

    Args:
        query: Search string (case-insensitive)

    Returns:
        List of matching model names
    """
    all_models = get_litellm_model_list()
    q = query.lower()

    matches = [m for m in all_models if q in m.lower()]

    # Sort by relevance (exact matches first)
    matches.sort(
        key=lambda m: (
            0 if m.lower() == q else 1,
            0 if m.lower().startswith(q) else 1,
            m.lower(),
        )
    )

    return matches
