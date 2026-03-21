"""Unified streaming client - uses LiteLLM for all providers.

This replaces all provider-specific code with LiteLLM's unified API.
Supports 100+ providers: Bedrock, Anthropic, OpenAI, Together, OpenRouter, etc.

Special handling for OAuth tokens (ChatGPT Plus/Pro subscriptions):
- Detects OAuth tokens (starts with "oauth:")
- Uses custom CodexOAuthClient instead of LiteLLM
- Handles token refresh automatically
"""

import os
import re
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass

# Import OAuth client
from .codex_oauth_client import (
    CodexOAuthClient,
    CodexMessage,
    is_oauth_token,
    extract_oauth_token,
)
from .codex_models import get_codex_models
from .oauth import get_oauth_manager

# Think-token sanitization (for tool_handlers compatibility)
_THINK_ZWS = "\u200b"
_THINK_TAG_RE = re.compile(r"<(/?)think>")


def _sanitize_think_tokens(text: str) -> str:
    """Escape <think> and </think> with zero-width space."""
    return _THINK_TAG_RE.sub(lambda m: f"<{_THINK_ZWS}{m.group(1)}think>", text)


def desanitize_think_tokens(text: str) -> str:
    """Reverse ZWS escaping."""
    return text.replace(f"<{_THINK_ZWS}/think>", "</think>").replace(
        f"<{_THINK_ZWS}think>", "<think>"
    )


# Import LiteLLM - lazy import to speed up startup
LITELLM_AVAILABLE = False
_litellm = None
_acompletion = None


def _import_litellm():
    """Lazy import litellm to avoid slow startup."""
    global _litellm, _acompletion, LITELLM_AVAILABLE
    if _litellm is not None:
        return _litellm, _acompletion
    try:
        import litellm
        from litellm import acompletion

        _litellm = litellm
        _acompletion = acompletion
        LITELLM_AVAILABLE = True
    except ImportError:
        LITELLM_AVAILABLE = False
        _litellm = None
        _acompletion = None
    return _litellm, _acompletion


# For backwards compatibility - will be set after first import
litellm = None
acompletion = None


class _SSEFallback(Exception):
    """Raised by _chat_stream_raw_sse to signal fallback to LiteLLM."""
    pass


class LiteLLMNotInstalled:
    def __init__(self, *args, **kwargs):
        raise ImportError("LiteLLM required. Install: pip install litellm")


@dataclass
@dataclass
class StreamingMessage:
    """Message format for chat."""

    role: str
    content: Union[str, List[Dict[str, Any]]]
    provider_blocks: Optional[List[Dict[str, Any]]] = (
        None  # For compatibility with cline_agent
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": self.content}


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
    has_web_search: bool = False  # For compatibility with cline_agent
    is_truncated: bool = False  # For compatibility with cline_agent
    web_search_results: Optional[List[Dict[str, Any]]] = (
        None  # For compatibility with cline_agent
    )


def _extract_reasoning_details_text(value: Any) -> str:
    """Flatten MiniMax/OpenAI-compatible `reasoning_details` blocks into text."""
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "reasoning_details" in value:
            return _extract_reasoning_details_text(value.get("reasoning_details"))
        parts: List[str] = []
        for k in ("summary", "content", "items"):
            if k in value:
                t = _extract_reasoning_details_text(value.get(k))
                if t:
                    parts.append(t)
        return "".join(parts)
    if not isinstance(value, list):
        return ""
    parts: List[str] = []
    for item in value:
        t = _extract_reasoning_details_text(item)
        if t:
            parts.append(t)
    return "".join(parts)


def _extract_reasoning_from_delta(delta: Any) -> Optional[str]:
    """Extract reasoning/thinking content from a LiteLLM delta object.

    Works with both attribute-style objects (LiteLLM Delta) and plain dicts.
    Checks all known field names:
    - reasoning_content (OpenAI, DeepSeek)
    - reasoning (NanoGPT, some OpenRouter models)
    - thinking (Anthropic via LiteLLM)
    - provider_specific_fields.{thinking,reasoning,reasoning_content}
    - reasoning_details (ZAI/MiniMax)
    """
    _is_dict = isinstance(delta, dict)

    def _get(key: str) -> Any:
        return delta.get(key) if _is_dict else getattr(delta, key, None)

    rc = _get("reasoning_content") or _get("reasoning") or _get("thinking")

    if not rc:
        psf = _get("provider_specific_fields") or {}
        for key in ("thinking", "reasoning", "reasoning_content"):
            if key in psf and psf[key]:
                rc = psf[key]
                break

    if not rc:
        rd = _get("reasoning_details")
        if rd:
            rc = _extract_reasoning_details_text(rd) or None

    return rc or None


def _normalize_model_name(model: str, base_url: str = "") -> str:
    """Normalize model name for LiteLLM.

    LiteLLM format: provider/model_name
    Examples:
      - bedrock/qwen.qwen3-32b-v1:0
      - anthropic/claude-3-5-sonnet-20241022
      - openai/gpt-4o
      - together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo
    """
    # If already has provider prefix, return as-is
    if model.count("/") >= 1 and not model.startswith("http"):
        # Check if it looks like a LiteLLM model name
        parts = model.split("/")
        if len(parts) >= 2:
            provider = parts[0]
            # Common LiteLLM providers
            known_providers = [
                "bedrock",
                "anthropic",
                "openai",
                "together_ai",
                "openrouter",
                "minimax",
                "deepseek",
                "groq",
                "ollama",
            ]
            if provider.lower() in known_providers:
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
    elif "minimax.io" in url:
        return f"minimax/{model}"
    elif "api.deepseek.com" in url:
        return f"deepseek/{model}"
    elif "api.groq.com" in url:
        return f"groq/{model}"
    elif "localhost:11434" in url or "localhost:11434" in url:
        return f"ollama/{model}"

    # Default to openai format for OpenAI-compatible APIs
    return f"openai/{model}"


class StreamingJSONClient:
    """LLM Client using LiteLLM for unified provider support.

    This client automatically handles:
    - Provider detection from URL/model name
    - Authentication for each provider
    - Streaming responses
    - Error handling and retries
    - Token usage tracking
    - OAuth tokens (ChatGPT Plus/Pro) via custom CodexOAuthClient

    Example:
        async with StreamingJSONClient(
            api_key="your-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            model="qwen.qwen3-32b-v1:0"
        ) as client:
            response = await client.chat_stream_raw(messages)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize LiteLLM client.

        Args:
            api_key: API key for the provider
            base_url: Base URL for the API (optional, auto-detected for most providers)
            model: Model name (e.g. "gpt-4o", "bedrock/qwen.qwen3-32b-v1:0")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort: str = "high"  # Set externally by caller

        # Check if using OAuth token
        self._is_oauth = is_oauth_token(api_key)
        self._oauth_provider: Optional[str] = None
        self._codex_client: Optional[CodexOAuthClient] = None
        self._copilot_client: Optional["CopilotOAuthClient"] = None  # type: ignore
        self._copilot_reasoning_opaque: Optional[str] = None  # Store reasoning context

        if self._is_oauth:
            # OAuth token - determine provider and create appropriate client
            oauth_token_str = extract_oauth_token(api_key)
            if oauth_token_str:
                # Get full token info from OAuth manager
                oauth_manager = get_oauth_manager()

                # Check API URL to determine provider
                is_copilot_url = (
                    "githubcopilot" in base_url.lower() or "copilot" in base_url.lower()
                )

                if is_copilot_url:
                    # GitHub Copilot
                    self._oauth_provider = "github-copilot"
                    oauth_token = oauth_manager.get_token("github-copilot")
                    if oauth_token:
                        from .copilot_oauth_client import CopilotOAuthClient

                        self._copilot_client = CopilotOAuthClient(
                            oauth_token=oauth_token,
                            model=model if model else "gpt-4o-copilot",
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            enterprise_url=oauth_token.enterprise_url,
                        )
                        self.litellm_model = model if model else "gpt-4o-copilot"
                    else:
                        raise RuntimeError(
                            "GitHub Copilot OAuth token not found. Please authenticate first."
                        )
                else:
                    # OpenAI Codex
                    self._oauth_provider = "openai"
                    oauth_token = oauth_manager.get_token("openai")
                    if oauth_token:
                        self._codex_client = CodexOAuthClient(
                            oauth_token=oauth_token,
                            model=model if model else "gpt-5.3-codex",
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                        )
                        self.litellm_model = model if model else "gpt-5.3-codex"
                    else:
                        raise RuntimeError(
                            "OpenAI OAuth token not found. Please authenticate first."
                        )

            self._is_bedrock = False
            self._bedrock_client = None
        else:
            # Normalize model name for LiteLLM
            self.litellm_model = _normalize_model_name(model, base_url)

            # Detect if using Bedrock (needs custom provider for bearer token auth)
            self._is_bedrock = "bedrock" in self.litellm_model.lower() or (
                "bedrock" in base_url.lower() and "amazonaws.com" in base_url.lower()
            )

            if self._is_bedrock:
                # Use custom Bedrock provider for bearer token auth
                try:
                    from .bedrock_provider import BedrockClient

                    # Bedrock models typically have 32K context, leave room for input
                    bedrock_max_tokens = min(max_tokens, 16000)  # Conservative default
                    self._bedrock_client = BedrockClient(
                        api_key=api_key,
                        model=model,
                        temperature=temperature,
                        max_tokens=bedrock_max_tokens,
                    )
                except ImportError:
                    raise ImportError(
                        "boto3 required for Bedrock. Install: pip install boto3"
                    )
            elif "together" in base_url.lower():
                # Together AI has 131K context limit
                self.max_tokens = min(max_tokens, 120000)  # Leave room for input
                self._bedrock_client = None
            else:
                self._bedrock_client = None

            # Set LiteLLM configuration (lazy import)
            _litellm, _ = _import_litellm()
            if _litellm:
                _litellm.set_verbose = False  # Set True for debugging
                _litellm.drop_params = True  # Drop unsupported params
                _litellm.success_callback = []  # Disable callbacks
                _litellm.failure_callback = []

    def _provider_kind(self) -> str:
        """Get provider type from model name or URL."""
        model_lower = self.litellm_model.lower()

        if model_lower.startswith("bedrock/"):
            return "litellm_bedrock"
        elif model_lower.startswith("anthropic/"):
            return "litellm_anthropic"
        elif model_lower.startswith("openrouter/"):
            return "litellm_openrouter"
        elif model_lower.startswith("together_ai/"):
            return "litellm_together"
        elif model_lower.startswith("minimax/"):
            return "litellm_minimax"
        elif model_lower.startswith("deepseek/"):
            return "litellm_deepseek"
        elif model_lower.startswith("groq/"):
            return "litellm_groq"
        elif model_lower.startswith("ollama/"):
            return "litellm_ollama"
        else:
            return "litellm_openai_compat"

    def _is_openai_compat_proxy(self) -> bool:
        """Check if this is an OpenAI-compatible proxy (not a native provider).

        Returns True when we should use raw SSE to preserve non-standard
        fields like ``reasoning`` that LiteLLM's stream processor drops.
        """
        if not self.base_url:
            return False
        url = self.base_url.lower()
        # These are native provider endpoints where LiteLLM handles
        # reasoning correctly (or where raw SSE would break):
        native_urls = [
            "api.anthropic.com",
            "api.deepseek.com",
            "api.groq.com",
            "localhost:11434",  # ollama
        ]
        # Bedrock is handled separately via _is_bedrock
        for native in native_urls:
            if native in url:
                return False
        return True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        # LiteLLM handles cleanup internally
        pass

    async def _chat_stream_raw_sse(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream chat via raw SSE for OpenAI-compatible APIs.

        Bypasses LiteLLM's stream processing which drops non-standard fields
        like ``reasoning`` (used by NanoGPT / kimi).  Falls back to the
        LiteLLM path on any transport-level error so that provider-specific
        quirks are still handled.
        """
        import httpx
        import json as _json

        full_content = ""
        full_reasoning = ""
        usage: Dict[str, int] = {}
        finish_reason = "stop"
        interrupted = False

        url = self.base_url.rstrip("/") + "/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,  # Use original model name, not LiteLLM-normalized
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort and self.reasoning_effort != "none":
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as http:
                async with http.stream(
                    "POST", url, json=body, headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        # Non-200 → let LiteLLM handle it (better error messages)
                        raise _SSEFallback(f"HTTP {resp.status_code}")

                    async for raw_line in resp.aiter_lines():
                        if check_interrupt and check_interrupt():
                            interrupted = True
                            finish_reason = "interrupted"
                            break

                        line = raw_line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break

                        try:
                            obj = _json.loads(payload)
                        except _json.JSONDecodeError:
                            continue

                        choices = obj.get("choices") or []
                        if not choices:
                            # Usage-only chunk (some providers send usage after [DONE]-like choices)
                            u = obj.get("usage")
                            if u:
                                usage = {
                                    "prompt_tokens": u.get("prompt_tokens", 0),
                                    "completion_tokens": u.get("completion_tokens", 0),
                                    "total_tokens": u.get("total_tokens", 0),
                                }
                            continue

                        delta = choices[0].get("delta", {})

                        # -- reasoning (check all known field names) --
                        r_text = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("thinking")
                        )
                        if not r_text:
                            psf = delta.get("provider_specific_fields") or {}
                            for _k in ("thinking", "reasoning", "reasoning_content"):
                                if psf.get(_k):
                                    r_text = psf[_k]
                                    break
                        if r_text:
                            full_reasoning += r_text
                            if on_reasoning:
                                on_reasoning(r_text)

                        # -- content --
                        c_text = delta.get("content")
                        if c_text:
                            full_content += c_text
                            if on_content:
                                on_content(c_text)

                        # -- debug --
                        if os.environ.get("HARNESS_DEBUG_THINKING") and (r_text or c_text):
                            import pathlib
                            dbg_path = pathlib.Path.home() / ".harness_thinking_debug.txt"
                            with open(dbg_path, "a", encoding="utf-8") as _f:
                                if r_text:
                                    _f.write(f"[REASONING] {r_text!r}\n")
                                if c_text:
                                    _f.write(f"[CONTENT]   {c_text!r}\n")

                        # -- usage / finish_reason --
                        u = obj.get("usage")
                        if u:
                            usage = {
                                "prompt_tokens": u.get("prompt_tokens", 0),
                                "completion_tokens": u.get("completion_tokens", 0),
                                "total_tokens": u.get("total_tokens", 0),
                            }
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

            return StreamingChatResponse(
                content=full_content,
                thinking=full_reasoning or None,
                raw_json=full_content,
                usage=usage if usage else None,
                finish_reason=finish_reason,
                interrupted=interrupted,
            )

        except _SSEFallback:
            raise  # Re-raise to trigger LiteLLM fallback in caller
        except asyncio.CancelledError:
            return StreamingChatResponse(
                content=full_content,
                thinking=full_reasoning or None,
                raw_json=full_content,
                usage=usage if usage else None,
                finish_reason="interrupted",
                interrupted=True,
            )
        except Exception as e:
            # Connection errors, timeouts, etc → fall back to LiteLLM
            raise _SSEFallback(str(e))

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
        litellm_messages = [m.to_dict() for m in messages]

        # Build request kwargs
        kwargs = {
            "model": self.litellm_model,
            "messages": litellm_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # Get usage in stream
        }

        # Enable LiteLLM's thinking/reasoning stream extraction.
        # NOTE: Do NOT set kwargs["reasoning"] here — it is not a standard
        # OpenAI parameter and LiteLLM passes it through verbatim for
        # unknown providers, causing deserialization errors (e.g. Ollama's
        # Go backend expects an object, not a boolean).  Models that
        # support reasoning will emit thinking tokens regardless; LiteLLM's
        # ``enable_thinking`` flag makes it surface them in the stream.
        kwargs["enable_thinking"] = True

        # ZAI native reasoning stream (Claude/Cursor-like visible thinking).
        # Safe no-op for providers that ignore unknown fields.
        if (
            self.base_url
            and "z.ai" in self.base_url.lower()
            and os.environ.get("HARNESS_ENABLE_NATIVE_THINKING", "1") != "0"
        ):
            kwargs["thinking"] = {"type": "enabled"}
            kwargs["tool_stream"] = True
            kwargs["clear_thinking"] = False

        # Add API key
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # For Bedrock, also set env var for LiteLLM to pick up
        if "bedrock" in self.litellm_model.lower():
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key

        # Add base_url for OpenAI-compatible APIs
        if self.base_url and "bedrock" not in self.litellm_model.lower():
            kwargs["api_base"] = self.base_url

        # Add timeout
        kwargs["timeout"] = self.timeout

        try:
            # Get streaming response (lazy import litellm)
            _, _acompletion = _import_litellm()
            if _acompletion is None:
                raise RuntimeError("LiteLLM not available")
            response = await _acompletion(**kwargs)

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
                    reasoning_content = _extract_reasoning_from_delta(delta)

                    if reasoning_content:
                        full_reasoning += reasoning_content
                        if on_reasoning:
                            on_reasoning(reasoning_content)

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
            raise RuntimeError(f"LiteLLM error: {e}")

    async def _chat_stream_bedrock(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream using custom Bedrock provider for bearer token auth."""
        import concurrent.futures
        from .bedrock_provider import BedrockMessage

        # Convert StreamingMessage to BedrockMessage
        bedrock_messages = [
            BedrockMessage(role=m.role, content=m.content) for m in messages
        ]

        def stream_bedrock():
            return self._bedrock_client.chat_stream(
                messages=bedrock_messages,
                on_content=on_content,
                on_thinking=on_reasoning,
            )

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, stream_bedrock)

        full_content = result.get("content", "")
        full_reasoning = result.get("thinking", "")

        # Note: callbacks were already invoked during streaming in bedrock_provider.py
        # Do NOT call them again here or content will be displayed twice

        return StreamingChatResponse(
            content=full_content,
            thinking=full_reasoning or None,
            raw_json=full_content,
            usage=result.get("usage", {}),
            finish_reason=result.get("finish_reason", "stop"),
            interrupted=False,
        )

    async def _chat_stream_codex(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream using Codex OAuth client."""
        if not self._codex_client:
            raise RuntimeError("Codex OAuth client not initialized")

        # Extract system messages for instructions
        system_parts = []
        chat_messages = []

        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if m.role == "system":
                system_parts.append(content)
            else:
                chat_messages.append(CodexMessage(role=m.role, content=content))

        system_prompt = "\n\n".join(system_parts) if system_parts else ""

        # Ensure we have at least one message
        if not chat_messages:
            raise RuntimeError(
                "No chat messages to send to Codex API (only system messages provided)"
            )

        # Use Codex client with system prompt
        async with self._codex_client:
            response = await self._codex_client.chat_stream(
                messages=chat_messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
                system_prompt=system_prompt,
            )

        return StreamingChatResponse(
            content=response.content,
            thinking=response.thinking,
            raw_json=response.content,
            usage=response.usage,
            finish_reason=response.finish_reason,
            interrupted=response.interrupted,
        )

    async def _chat_stream_copilot(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Stream using GitHub Copilot OAuth client."""
        if not self._copilot_client:
            raise RuntimeError("Copilot OAuth client not initialized")

        from .copilot_oauth_client import CopilotMessage

        # Convert StreamingMessage to CopilotMessage with reasoning context
        copilot_messages = []
        for i, m in enumerate(messages):
            msg = CopilotMessage(
                role=m.role,
                content=m.content,
            )
            # For assistant messages, check if this is the last assistant message
            # and if we have a stored reasoning_opaque from the last response
            if (
                m.role == "assistant"
                and i == len(messages) - 1
                and self._copilot_reasoning_opaque
            ):
                msg.reasoning_opaque = self._copilot_reasoning_opaque
            copilot_messages.append(msg)

        # Use Copilot client
        self._copilot_client.reasoning_effort = self.reasoning_effort
        async with self._copilot_client:
            response = await self._copilot_client.chat_stream(
                messages=copilot_messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
            )

        # Store reasoning_opaque for multi-turn context
        # This is critical for maintaining reasoning context across turns
        if response.reasoning_opaque:
            # Store it in the client for future use
            self._copilot_reasoning_opaque = response.reasoning_opaque

        return StreamingChatResponse(
            content=response.content,
            thinking=response.thinking,
            raw_json=response.content,
            usage=response.usage,
            finish_reason=response.finish_reason,
            interrupted=response.interrupted,
        )

    async def chat_stream_raw(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
        enable_web_search: bool = False,
        status_line: Optional[Any] = None,
    ) -> StreamingChatResponse:
        """Stream raw text response (no JSON parsing).

        This is the main entry point for streaming chat.

        Args:
            messages: List of chat messages
            on_content: Callback for content chunks (real-time display)
            on_reasoning: Callback for reasoning/thinking content
            check_interrupt: Callback to check if operation should be interrupted
            enable_web_search: Enable web search (ignored for compatibility)
            status_line: Status line object (ignored for compatibility)

        Returns:
            StreamingChatResponse with full content and metadata
        """
        if self._is_oauth and self._copilot_client:
            # Use Copilot OAuth client for GitHub Copilot tokens
            return await self._chat_stream_copilot(
                messages=messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
            )
        elif self._is_oauth and self._codex_client:
            # Use Codex OAuth client for ChatGPT Plus/Pro tokens
            return await self._chat_stream_codex(
                messages=messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
            )
        elif self._is_bedrock and self._bedrock_client:
            # Use custom Bedrock provider for bearer token auth
            return await self._chat_stream_bedrock(
                messages=messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
            )
        else:
            # For OpenAI-compatible proxy APIs (NanoGPT, OpenRouter, etc.),
            # use raw SSE streaming to preserve non-standard fields like
            # `reasoning` that LiteLLM's stream processor drops.
            # Falls back to LiteLLM on any transport error.
            if self._is_openai_compat_proxy():
                try:
                    return await self._chat_stream_raw_sse(
                        messages=messages,
                        on_content=on_content,
                        on_reasoning=on_reasoning,
                        check_interrupt=check_interrupt,
                    )
                except _SSEFallback:
                    pass  # Fall through to LiteLLM
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
        max_retries: Optional[int] = None,
    ) -> StreamingChatResponse:
        """Stream a chat response with retry logic.

        Args:
            messages: List of chat messages
            on_content: Callback for content chunks
            on_thinking: Callback for thinking content
            max_retries: Override default max_retries

        Returns:
            StreamingChatResponse with full content and metadata
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_error = None

        for attempt in range(retries + 1):
            try:
                return await self.chat_stream_raw(
                    messages=messages,
                    on_content=on_content,
                    on_reasoning=on_thinking,
                )
            except Exception as e:
                last_error = e
                if attempt < retries:
                    # Exponential backoff
                    wait_time = min(2**attempt, 30)
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {retries + 1} attempts: {e}")

        # Should not reach here
        raise RuntimeError(f"Unexpected error: {last_error}")


# Model discovery functions
def get_litellm_model_list() -> List[str]:
    """Get list of all models supported by LiteLLM.

    Returns:
        List of model names in format "provider/model_name"
    """
    _litellm, _ = _import_litellm()
    if not LITELLM_AVAILABLE or not _litellm:
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
        List of matching model names, sorted by relevance
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


# Convenience function for harness.py
def fetch_models_litellm() -> List[str]:
    """Fetch model list for harness /model command.

    Returns:
        List of model names
    """
    return get_litellm_model_list()


def fetch_models_for_provider(
    api_key: str, api_url: str = "", oauth_provider: Optional[str] = None
) -> List[str]:
    """Fetch models for a provider, handling OAuth specially.

    Args:
        api_key: API key (may be OAuth token)
        api_url: API base URL
        oauth_provider: OAuth provider ID ("openai" or "github-copilot")

    Returns:
        List of model names
    """
    if is_oauth_token(api_key):
        # OAuth tokens - determine which models to show based on provider
        if oauth_provider == "github-copilot":
            from .copilot_oauth_client import get_copilot_models

            return get_copilot_models()
        else:
            # Default to Codex models (OpenAI)
            return get_codex_models()
    else:
        # Use LiteLLM for regular API keys
        return get_litellm_model_list()
