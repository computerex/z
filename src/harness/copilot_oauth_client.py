"""OAuth client for GitHub Copilot API access."""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import aiohttp

from .oauth import OAuthToken

log = logging.getLogger("harness.copilot")

# GitHub Copilot API endpoints
COPILOT_API_BASE = "https://api.githubcopilot.com"
COPILOT_CHAT_ENDPOINT = "/chat/completions"

# Allowed Copilot models — sourced from the Copilot API's available model list.
# This is a hardcoded cache; update when the API adds or removes models.
ALLOWED_COPILOT_MODELS = {
    # Claude models
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    # GPT models
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-copilot",
    "gpt-4-o-preview",
    "gpt-4o-japanwest",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5-mini",
    "copilot-preview-4o-mini-a1cfd608",
    "copilot-preview-gpt4-centralus",
    "copilot-preview-gpt4o-centralus",
    # Gemini models
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    # Other
    "goldeneye-secondary",
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
    # Native tool calling fields
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        msg: Dict[str, Any] = {"role": self.role}

        # For assistant messages with tool_calls, content can be null
        if self.role == "assistant" and self.tool_calls:
            msg["content"] = self.content if self.content else None
            msg["tool_calls"] = self.tool_calls
        elif self.role == "tool":
            # Tool result messages: content must be a string
            msg["content"] = str(self.content) if self.content else ""
            # tool_call_id is REQUIRED on tool messages — without it the
            # Copilot API returns 400 Bad Request.
            if self.tool_call_id:
                msg["tool_call_id"] = self.tool_call_id
            else:
                # Generate a fallback ID so the message is still valid
                import hashlib
                fallback_id = "call_" + hashlib.md5(
                    (str(self.content)[:100] + (self.name or "")).encode()
                ).hexdigest()[:24]
                msg["tool_call_id"] = fallback_id
                log.warning("Tool message missing tool_call_id, generated fallback: %s", fallback_id)
        else:
            msg["content"] = self.content

        # For assistant messages with reasoning, include provider metadata
        if self.role == "assistant" and self.reasoning_opaque:
            msg["reasoning_opaque"] = self.reasoning_opaque

        # Tool name for tool result messages
        if self.name and self.role == "tool":
            msg["name"] = self.name

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
    # Native tool calling
    tool_calls: Optional[List[Dict[str, Any]]] = None


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

    def _build_headers(self, messages: Optional[List["CopilotMessage"]] = None) -> Dict[str, str]:
        """Build headers for Copilot API request.

        The x-initiator header is set dynamically based on the last message role,
        matching opencode's behavior: 'agent' when last message is not from user.
        """
        # Determine if this is an agent-initiated request (tool loop, not direct user message)
        is_agent = True
        if messages:
            last = messages[-1] if messages else None
            if last and last.role == "user":
                is_agent = False
        return {
            "Authorization": f"Bearer {self.oauth_token.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "harness/1.0.0",
            "x-initiator": "agent" if is_agent else "user",
            "Openai-Intent": "conversation-edits",
            "Copilot-Integration-Id": "vscode-chat",
        }

    def _build_request_body(
        self,
        messages: List[CopilotMessage],
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
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

        # Native tool calling
        if tools:
            body["tools"] = tools

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
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> CopilotResponse:
        """Stream chat response from Copilot API with reasoning support."""
        headers = self._build_headers(messages)
        if self._has_vision_content(messages):
            headers["Copilot-Vision-Request"] = "true"
        body = self._build_request_body(messages, stream=True, tools=tools)

        # Copilot API enforces max 40-char tool_call IDs (opencode issue #12653).
        # Truncate any oversized IDs in message history to prevent 400 errors.
        for msg_dict in body.get("messages", []):
            for tc in msg_dict.get("tool_calls", []) or []:
                if isinstance(tc.get("id"), str) and len(tc["id"]) > 40:
                    tc["id"] = tc["id"][:40]
            # Also truncate tool_call_id in tool result messages
            tcid = msg_dict.get("tool_call_id")
            if isinstance(tcid, str) and len(tcid) > 40:
                msg_dict["tool_call_id"] = tcid[:40]

        if not self._session:
            self._session = aiohttp.ClientSession()

        full_content = ""
        full_reasoning = ""
        reasoning_opaque = None
        usage = {}
        finish_reason = "stop"
        interrupted = False
        _tool_call_accum: Dict[int, Dict[str, Any]] = {}

        try:
            url = f"{self.base_url}{COPILOT_CHAT_ENDPOINT}"

            # Debug: log the request body structure for diagnosing 400 errors
            if log.isEnabledFor(logging.DEBUG):
                _msg_summary = []
                for _m in body.get("messages", []):
                    _role = _m.get("role", "?")
                    _has_tc = "tool_calls" in _m
                    _has_tcid = "tool_call_id" in _m
                    _content_type = type(_m.get("content")).__name__
                    _content_len = len(str(_m.get("content", "")))
                    _msg_summary.append(
                        f"{_role}(content={_content_type}:{_content_len}"
                        f"{',tool_calls' if _has_tc else ''}"
                        f"{',tool_call_id=' + _m['tool_call_id'] if _has_tcid else ''})"
                    )
                log.debug(
                    "Copilot request: model=%s tools=%d messages=[%s]",
                    body.get("model"),
                    len(body.get("tools", [])),
                    ", ".join(_msg_summary),
                )

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
                    # Try to parse JSON error for more detail
                    error_detail = error_text
                    try:
                        err_json = json.loads(error_text)
                        if isinstance(err_json, dict):
                            # OpenAI-style: {"error": {"message": "...", "type": "..."}}
                            inner = err_json.get("error", {})
                            if isinstance(inner, dict) and inner.get("message"):
                                error_detail = f"{inner.get('type', 'error')}: {inner['message']}"
                            elif err_json.get("message"):
                                error_detail = err_json["message"]
                    except (json.JSONDecodeError, KeyError):
                        pass
                    # Dump request body for debugging 400 errors
                    if response.status == 400:
                        import pathlib
                        dbg_path = pathlib.Path.home() / ".harness_copilot_400_debug.json"
                        try:
                            debug_obj = {
                                "error_status": response.status,
                                "error_detail": error_detail,
                                "error_raw": error_text,
                                "request_body": body,
                            }
                            dbg_path.write_text(json.dumps(debug_obj, indent=2, default=str), encoding="utf-8")
                            log.error("400 Bad Request — full debug dumped to %s", dbg_path)
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Copilot API error {response.status}: {error_detail}"
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

                        # Regular content
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                            if on_content:
                                on_content(content)

                        # Reasoning content (for Claude models through Copilot)
                        reasoning = delta.get("reasoning_text", "") or delta.get("reasoning_content", "")
                        if reasoning:
                            full_reasoning += reasoning
                            if on_reasoning:
                                on_reasoning(reasoning)

                        # Tool calls (streaming — chunks arrive indexed)
                        delta_tcs = delta.get("tool_calls")
                        if delta_tcs:
                            for tc_chunk in delta_tcs:
                                idx = tc_chunk.get("index", 0)
                                if idx not in _tool_call_accum:
                                    _tool_call_accum[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                acc = _tool_call_accum[idx]
                                if tc_chunk.get("id"):
                                    acc["id"] = tc_chunk["id"]
                                if tc_chunk.get("type"):
                                    acc["type"] = tc_chunk["type"]
                                fn = tc_chunk.get("function") or {}
                                if fn.get("name"):
                                    acc["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    acc["function"]["arguments"] += fn["arguments"]

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
                        # Propagate any extra usage fields (cache tokens etc.)
                        for _extra in (
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                            "prompt_cached_tokens",
                        ):
                            _val = event["usage"].get(_extra)
                            if _val is not None:
                                usage[_extra] = _val

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

        _final_tool_calls = None
        if _tool_call_accum:
            _final_tool_calls = [_tool_call_accum[i] for i in sorted(_tool_call_accum.keys())]

        return CopilotResponse(
            content=full_content,
            thinking=full_reasoning if full_reasoning else None,
            reasoning_opaque=reasoning_opaque,
            usage=usage if usage else None,
            finish_reason=finish_reason,
            interrupted=interrupted,
            tool_calls=_final_tool_calls,
        )


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
