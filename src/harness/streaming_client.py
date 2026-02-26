"""JSON-based streaming LLM client - true token-by-token streaming."""

import json
import asyncio
import os
import time
import base64
import httpx
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field

from .logger import get_logger, log_exception

try:
    from anthropic import AsyncAnthropic  # type: ignore
except Exception:
    AsyncAnthropic = None

_log = get_logger("streaming")


def _normalize_anthropic_base_url(url: str) -> str:
    """Anthropic SDK expects API root, not a URL ending in /v1."""
    u = (url or "").rstrip("/")
    if u.lower().endswith("/v1"):
        return u[:-3]
    return u


@dataclass 
class StreamingMessage:
    """A message for the streaming client.
    
    Content can be:
    - str: Simple text message
    - list: Multi-part content for vision (OpenAI-compatible format)
      e.g., [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
    """
    role: str
    content: Union[str, List[Dict[str, Any]]]
    provider_blocks: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class StreamingToolCall:
    """A parsed tool call from the JSON response."""
    name: str
    parameters: Dict[str, Any]


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    content: str
    link: str
    media: str = ""
    icon: str = ""
    refer: str = ""
    publish_date: str = ""


@dataclass
class StreamingChatResponse:
    """Response from the streaming JSON approach."""
    content: Optional[str] = None
    thinking: Optional[str] = None
    tool_call: Optional[StreamingToolCall] = None
    message: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw_json: str = ""
    provider_content_blocks: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    interrupted: bool = False
    web_search_results: List[WebSearchResult] = field(default_factory=list)
    
    @property
    def is_truncated(self) -> bool:
        """Check if response was cut off due to length."""
        return self.finish_reason == "length"
    
    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None
    
    @property
    def has_web_search(self) -> bool:
        return len(self.web_search_results) > 0


class ContentExtractor:
    """Extract content/code fields from streaming JSON in real-time."""
    
    # Fields to stream (parameter values that contain code/content)
    STREAM_FIELDS = ('"content": "', '"content":"', 
                     '"old_string": "', '"old_string":"', 
                     '"new_string": "', '"new_string":"',
                     '"command": "', '"command":"',
                     '"pattern": "', '"pattern":"')
    
    def __init__(self):
        self.buffer = ""
        self.in_content = False
        self.escape_next = False
        self.current_field = None
        
    def feed(self, text: str, on_content: Optional[Callable[[str], None]] = None) -> None:
        """Process incoming text, call on_content for each char in streamable fields."""
        for char in text:
            self.buffer += char
            
            # Check for streamable field patterns
            if not self.in_content:
                for field_pattern in self.STREAM_FIELDS:
                    if self.buffer.endswith(field_pattern):
                        self.in_content = True
                        self.escape_next = False
                        self.current_field = field_pattern
                        break
                continue
            else:
                # Inside the string value
                if self.escape_next:
                    self.escape_next = False
                    if on_content:
                        if char == 'n':
                            on_content('\n')
                        elif char == 't':
                            on_content('\t')
                        elif char == 'r':
                            on_content('\r')
                        elif char == '"':
                            on_content('"')
                        elif char == '\\':
                            on_content('\\')
                        else:
                            on_content(char)
                    continue
                    
                if char == '\\':
                    self.escape_next = True
                    continue
                    
                if char == '"':
                    self.in_content = False
                    self.current_field = None
                    continue
                    
                if on_content:
                    on_content(char)
    
    def get_json(self) -> Optional[Dict[str, Any]]:
        """Parse the complete JSON."""
        try:
            return json.loads(self.buffer)
        except json.JSONDecodeError:
            return None


class StreamingJSONClient:
    """LLM Client using JSON response format for true streaming."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.z.ai/api/paas/v4",
        model: str = "glm-4.7",
        temperature: float = 0.7,
        max_tokens: int = 128000,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None
        self._anthropic_client = None

    def _make_anthropic_client(self):
        if AsyncAnthropic is None:
            return None
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": 600.0,
        }
        if self.base_url:
            kwargs["base_url"] = _normalize_anthropic_base_url(self.base_url)
        return AsyncAnthropic(**kwargs)
        
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0)
        )
        if self._provider_kind() == "anthropic" and AsyncAnthropic is not None:
            self._anthropic_client = self._make_anthropic_client()
        return self
        
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
        self._client = None
        self._anthropic_client = None

    def _provider_kind(self) -> str:
        """Infer provider protocol from URL/model.

        Returns one of: ``anthropic``, ``openrouter``, ``openai``, ``openai_compat``.
        """
        url = (self.base_url or "").lower()
        model = (self.model or "").lower()
        if "anthropic.com" in url or model.startswith("claude-"):
            return "anthropic"
        if "openrouter.ai" in url:
            return "openrouter"
        if "api.openai.com" in url:
            return "openai"
        return "openai_compat"

    @staticmethod
    def _data_uri_to_anthropic_image_block(url: str) -> Optional[Dict[str, Any]]:
        """Convert data URI image_url to Anthropic image block."""
        if not isinstance(url, str) or not url.startswith("data:"):
            return None
        try:
            header, b64 = url.split(",", 1)
            if ";base64" not in header:
                return None
            media_type = header[5:].split(";", 1)[0] or "image/png"
            # Validate base64 payload without altering it
            base64.b64decode(b64, validate=True)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        except Exception:
            return None

    @staticmethod
    def _to_anthropic_text_blocks(
        content: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert harness message content to Anthropic content blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    img_block = StreamingJSONClient._data_uri_to_anthropic_image_block(url)
                    if img_block:
                        blocks.append(img_block)
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks

    @staticmethod
    def _serialize_anthropic_block_obj(block_obj: Any) -> Optional[Dict[str, Any]]:
        """Best-effort conversion of Anthropic SDK content blocks to dicts."""
        if block_obj is None:
            return None
        if isinstance(block_obj, dict):
            return block_obj
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(block_obj, method_name, None)
            if callable(method):
                try:
                    data = method()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
        # Fallback by introspecting common attributes
        data: Dict[str, Any] = {}
        for attr in ("type", "text", "thinking", "signature", "redacted_thinking", "source"):
            if hasattr(block_obj, attr):
                try:
                    val = getattr(block_obj, attr)
                    if val is not None:
                        if hasattr(val, "model_dump"):
                            val = val.model_dump()
                        data[attr] = val
                except Exception:
                    pass
        return data or None

    def _build_anthropic_messages(
        self,
        messages: List[StreamingMessage],
        enable_prompt_caching: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert OpenAI-style messages to Anthropic Messages API format."""
        system_blocks: List[Dict[str, Any]] = []
        anth_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "assistant" and msg.provider_blocks:
                blocks = [dict(b) for b in msg.provider_blocks]
            else:
                blocks = self._to_anthropic_text_blocks(msg.content)
            if msg.role == "system":
                system_blocks.extend(blocks)
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            anth_messages.append({"role": role, "content": blocks})

        if enable_prompt_caching:
            ttl = os.environ.get("HARNESS_ANTHROPIC_CACHE_TTL", "").strip().lower()
            cache_ctl: Dict[str, Any] = {"type": "ephemeral"}
            if ttl in ("1h", "1hr", "1hour"):
                cache_ctl["ttl"] = "1h"

            def _mark_last_cacheable(blocks: List[Dict[str, Any]]) -> bool:
                # Anthropic cache breakpoints go on content blocks. Avoid marking
                # thinking/redacted_thinking blocks when replaying prior assistant turns.
                for i in range(len(blocks) - 1, -1, -1):
                    b = blocks[i]
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") in ("thinking", "redacted_thinking"):
                        continue
                    # Anthropic prompt caching does not allow empty text blocks.
                    if b.get("type") == "text" and not str(b.get("text", "")).strip():
                        continue
                    blocks[i] = {**b, "cache_control": dict(cache_ctl)}
                    return True
                return False

            # System prompt breakpoint (highest ROI)
            if system_blocks:
                _mark_last_cacheable(system_blocks)

            # Add periodic conversation breakpoints so long histories remain
            # cache-addressable (Anthropic cache lookup is limited from markers).
            for idx, msg in enumerate(anth_messages):
                if idx % 18 == 0 or idx == len(anth_messages) - 3:
                    _mark_last_cacheable(msg.get("content") or [])
        return system_blocks, anth_messages
            
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # OpenRouter recommends these headers; most other providers ignore them,
        # but keep them scoped to OpenRouter to minimize compatibility risk.
        if self._provider_kind() == "openrouter":
            headers["HTTP-Referer"] = "https://cline.bot"
            headers["X-Title"] = "Cline"
        return headers

    @staticmethod
    def _normalize_openai_usage(usage: Dict[str, Any]) -> Dict[str, int]:
        """Normalize OpenAI/OpenRouter/OpenAI-compatible usage payloads.

        Keeps canonical keys (`prompt_tokens`, `completion_tokens`) and preserves
        useful detail fields (e.g. cached/reasoning tokens) when present.
        """
        if not isinstance(usage, dict):
            return {}
        out: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage and usage[key] is not None:
                try:
                    out[key] = int(usage[key])
                except Exception:
                    pass

        # OpenAI nested details
        pdet = usage.get("prompt_tokens_details")
        if isinstance(pdet, dict):
            for k in ("cached_tokens", "audio_tokens"):
                if pdet.get(k) is not None:
                    try:
                        out[f"prompt_{k}"] = int(pdet[k])
                    except Exception:
                        pass

        cdet = usage.get("completion_tokens_details")
        if isinstance(cdet, dict):
            for k in ("reasoning_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"):
                if cdet.get(k) is not None:
                    try:
                        out[f"completion_{k}"] = int(cdet[k])
                    except Exception:
                        pass

        # OpenRouter sometimes flattens or renames details; keep common extras if present.
        for k in (
            "reasoning_tokens",
            "cached_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "input_tokens",
            "output_tokens",
        ):
            if k in usage and usage[k] is not None:
                try:
                    out[k] = int(usage[k])
                except Exception:
                    pass

        # Some providers return input/output instead of prompt/completion.
        if "prompt_tokens" not in out and "input_tokens" in out:
            out["prompt_tokens"] = out["input_tokens"]
        if "completion_tokens" not in out and "output_tokens" in out:
            out["completion_tokens"] = out["output_tokens"]
        if "total_tokens" not in out and ("prompt_tokens" in out or "completion_tokens" in out):
            out["total_tokens"] = int(out.get("prompt_tokens", 0)) + int(out.get("completion_tokens", 0))
        return out

    async def _chat_stream_raw_anthropic(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        check_interrupt: Optional[Callable[[], bool]] = None,
    ) -> StreamingChatResponse:
        """Anthropic-native streaming (Messages API) with prompt caching."""
        if AsyncAnthropic is None:
            raise RuntimeError(
                "Anthropic SDK not installed. Install 'anthropic' to use Claude models."
            )
        if self._anthropic_client is None:
            self._anthropic_client = self._make_anthropic_client()

        enable_cache = os.environ.get("HARNESS_ANTHROPIC_PROMPT_CACHING", "1") != "0"
        system_blocks, anth_messages = self._build_anthropic_messages(
            messages, enable_prompt_caching=enable_cache
        )
        _log.info(
            "chat_stream_raw[anthropic]: model=%s msgs=%d system_blocks=%d cache=%s",
            self.model, len(anth_messages), len(system_blocks), enable_cache
        )

        full_content = ""
        full_reasoning = ""
        _thinking_sigs: Dict[int, str] = {}
        usage: Dict[str, int] = {}
        finish_reason = "stop"

        stream_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anth_messages,
        }
        if os.environ.get("HARNESS_ENABLE_ANTHROPIC_THINKING", "0") != "1":
            stream_kwargs["temperature"] = self.temperature
        if system_blocks:
            stream_kwargs["system"] = system_blocks

        # Anthropic "thinking" is optional and model-specific; only enable if asked.
        if os.environ.get("HARNESS_ENABLE_ANTHROPIC_THINKING", "0") == "1":
            stream_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        interrupted = False

        final_msg = None
        async with self._anthropic_client.messages.stream(**stream_kwargs) as stream:
            async for event in stream:
                if check_interrupt and check_interrupt():
                    interrupted = True
                    finish_reason = "interrupted"
                    try:
                        _close = getattr(stream, "close", None)
                        if _close:
                            _res = _close()
                            if asyncio.iscoroutine(_res):
                                await _res
                    except Exception:
                        pass
                    break

                event_type = getattr(event, "type", "")
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    block_index = int(getattr(event, "index", 0) or 0)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", None)
                        if text:
                            full_content += text
                            if on_content:
                                on_content(text)
                    elif delta_type == "thinking_delta":
                        thinking_text = getattr(delta, "thinking", None)
                        if thinking_text:
                            full_reasoning += thinking_text
                            if on_reasoning:
                                on_reasoning(thinking_text)
                    elif delta_type == "signature_delta":
                        sig = getattr(delta, "signature", None)
                        if sig:
                            _thinking_sigs[block_index] = sig
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None)
                    if stop_reason:
                        finish_reason = "length" if stop_reason == "max_tokens" else "stop"

            # SDK helper accumulates final message + usage.
            try:
                final_msg = await stream.get_final_message()
                msg_usage = getattr(final_msg, "usage", None)
                if msg_usage is not None:
                    usage = {
                        "prompt_tokens": int(getattr(msg_usage, "input_tokens", 0) or 0),
                        "completion_tokens": int(getattr(msg_usage, "output_tokens", 0) or 0),
                    }
                    # Preserve cache metrics when available.
                    cache_create = getattr(msg_usage, "cache_creation_input_tokens", None)
                    cache_read = getattr(msg_usage, "cache_read_input_tokens", None)
                    if cache_create is not None:
                        usage["cache_creation_input_tokens"] = int(cache_create or 0)
                    if cache_read is not None:
                        usage["cache_read_input_tokens"] = int(cache_read or 0)
            except Exception as e:
                _log.debug("Anthropic final message unavailable: %s", e)

        provider_blocks = None
        if final_msg is not None:
            try:
                content_blocks = getattr(final_msg, "content", None) or []
                provider_blocks = []
                for i, b in enumerate(content_blocks):
                    bd = self._serialize_anthropic_block_obj(b)
                    if not bd:
                        continue
                    # If streaming emitted signature deltas but serialization omitted
                    # them, patch signatures back in for thinking blocks.
                    if bd.get("type") == "thinking" and "signature" not in bd and i in _thinking_sigs:
                        bd["signature"] = _thinking_sigs[i]
                    provider_blocks.append(bd)
                if not provider_blocks:
                    provider_blocks = None
            except Exception as e:
                _log.debug("Anthropic content block serialization failed: %s", e)
        return StreamingChatResponse(
            content=full_content,
            thinking=full_reasoning or None,
            raw_json=full_content,
            usage=usage,
            provider_content_blocks=provider_blocks,
            finish_reason=finish_reason,
            interrupted=interrupted,
        )
    
    async def chat_stream(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        max_retries: int = 5,
    ) -> StreamingChatResponse:
        """Stream a chat response with real-time content display."""
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                extractor = ContentExtractor()
                usage = {}
                finish_reason = "stop"
                
                async with self._client.stream(
                    "POST", url, headers=self._get_headers(), json=payload
                ) as response:
                    response.raise_for_status()
                    
                    line_buffer = ""
                    async for chunk in response.aiter_bytes():
                        line_buffer += chunk.decode('utf-8', errors='ignore')
                        
                        while '\n' in line_buffer:
                            line, line_buffer = line_buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line or not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            if "usage" in data:
                                usage = self._normalize_openai_usage(data["usage"])
                            
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            
                            # Track finish reason
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            
                            if content:
                                extractor.feed(content, on_content)
                
                # Parse final result
                result = StreamingChatResponse(raw_json=extractor.buffer, usage=usage, finish_reason=finish_reason)
                parsed = extractor.get_json()
                
                if parsed:
                    result.thinking = parsed.get("thinking")
                    result.message = parsed.get("message")
                    
                    tool_name = parsed.get("tool")
                    if tool_name and tool_name != "null" and parsed.get("parameters"):
                        result.tool_call = StreamingToolCall(
                            name=tool_name,
                            parameters=parsed["parameters"]
                        )
                
                return result
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503):
                    if attempt < max_retries:
                        wait = min(2 ** (attempt + 1), 60)
                        print(f"\n⚠️  HTTP {e.response.status_code}. Retry in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                raise RuntimeError(f"API error: {e}")
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = min(2 ** (attempt + 1), 60)
                    print(f"\n⚠️  Connection error. Retry in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Request failed: {e}")
        
        raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")

    async def chat_stream_raw(
        self,
        messages: List[StreamingMessage],
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        max_retries: int = 5,
        check_interrupt: Optional[Callable[[], bool]] = None,
        enable_web_search: bool = False,
        web_search_count: int = 5,
        status_line: Optional[object] = None,
    ) -> StreamingChatResponse:
        """Stream raw text response (no JSON parsing) - for XML tool format.
        
        Args:
            enable_web_search: If True, adds built-in web search tool to the request.
                The model will automatically decide when to search and incorporate results.
            web_search_count: Number of web search results to return (1-50, default 5).
        """
        
        if self._provider_kind() == "anthropic":
            return await self._chat_stream_raw_anthropic(
                messages=messages,
                on_content=on_content,
                on_reasoning=on_reasoning,
                check_interrupt=check_interrupt,
            )

        url = f"{self.base_url}/chat/completions"
        msg_count = len(messages)
        est_chars = sum(len(m.content) if isinstance(m.content, str) else 0 for m in messages)
        _log.info("chat_stream_raw: url=%s model=%s msgs=%d est_chars=%d web_search=%s",
                  url, self.model, msg_count, est_chars, enable_web_search)
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        # ZAI native reasoning stream (Claude/Cursor-like visible thinking).
        # Safe no-op for providers that ignore unknown fields.
        if ("z.ai" in self.base_url.lower()
                and os.environ.get("HARNESS_ENABLE_NATIVE_THINKING", "1") != "0"):
            payload["thinking"] = {"type": "enabled"}
            payload["tool_stream"] = True
            payload["clear_thinking"] = False
        
        # Add built-in web search tool if enabled
        if enable_web_search:
            payload["tools"] = [{
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search-prime",
                    "search_result": True,
                    "count": str(web_search_count),
                }
            }]
        
        last_error = None
        debug_log = os.environ.get("HARNESS_DEBUG_API")
        _req_t0 = time.time()
        
        for attempt in range(max_retries + 1):
            try:
                full_content = ""
                full_reasoning = ""
                usage = {}
                finish_reason = "stop"
                interrupted = False
                web_search_data = []  # Collect web search results
                raw_chunks = []  # For debug logging
                
                async with self._client.stream(
                    "POST", url, headers=self._get_headers(), json=payload
                ) as response:
                    response.raise_for_status()
                    
                    line_buffer = ""
                    # Use polling loop instead of 'async for' so we can
                    # check for interrupt every 300ms even when the server
                    # is slow to emit tokens (e.g. during initial thinking).
                    aiter = response.aiter_bytes().__aiter__()
                    pending_read = None
                    while True:
                        # Check for interrupt
                        if check_interrupt and check_interrupt():
                            if pending_read and not pending_read.done():
                                pending_read.cancel()
                                try:
                                    await pending_read
                                except (asyncio.CancelledError, StopAsyncIteration):
                                    pass
                            interrupted = True
                            finish_reason = "interrupted"
                            break
                        
                        # Start a new read if we don't have one pending
                        if pending_read is None:
                            pending_read = asyncio.ensure_future(aiter.__anext__())
                        
                        # Wait for chunk with timeout so we can check interrupts
                        done, _ = await asyncio.wait({pending_read}, timeout=0.3)
                        
                        if not done:
                            continue  # Timeout — loop back to check interrupt
                        
                        try:
                            chunk = pending_read.result()
                        except StopAsyncIteration:
                            break
                        pending_read = None
                        
                        if debug_log:
                            raw_chunks.append(chunk)
                        
                        line_buffer += chunk.decode('utf-8', errors='ignore')
                        
                        while '\n' in line_buffer:
                            line, line_buffer = line_buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line or not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            if "usage" in data:
                                usage = self._normalize_openai_usage(data["usage"])
                            
                            # Capture web_search results from response
                            if "web_search" in data:
                                web_search_data = data["web_search"]
                            
                            choices = data.get("choices", [])
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if isinstance(content, list):
                                # OpenRouter/other providers may emit content parts.
                                joined = []
                                for part in content:
                                    if isinstance(part, dict):
                                        txt = part.get("text")
                                        if isinstance(txt, str):
                                            joined.append(txt)
                                content = "".join(joined)
                            reasoning = (
                                delta.get("reasoning_content", "")
                                or delta.get("reasoning", "")
                            )
                            
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            
                            if reasoning:
                                full_reasoning += reasoning
                                if on_reasoning:
                                    on_reasoning(reasoning)
                            
                            if content:
                                full_content += content
                                if on_content:
                                    on_content(content)
                
                # Parse web search results
                web_search_results = []
                for item in web_search_data:
                    web_search_results.append(WebSearchResult(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        link=item.get("link", ""),
                        media=item.get("media", ""),
                        icon=item.get("icon", ""),
                        refer=item.get("refer", ""),
                        publish_date=item.get("publish_date", ""),
                    ))
                
                # Write debug log if enabled
                if debug_log and raw_chunks:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = f"harness_api_debug_{timestamp}.log"
                    with open(debug_path, "wb") as f:
                        for chunk in raw_chunks:
                            f.write(chunk)
                    print(f"\n[DEBUG] Raw API response saved to: {debug_path}")
                
                _req_elapsed = time.time() - _req_t0
                _log.info("chat_stream_raw complete: finish=%s interrupted=%s "
                          "content_len=%d reasoning_len=%d elapsed=%.1fs usage=%s",
                          finish_reason, interrupted, len(full_content),
                          len(full_reasoning), _req_elapsed, usage)
                return StreamingChatResponse(
                    content=full_content,
                    thinking=full_reasoning or None,
                    raw_json=full_content,
                    usage=usage,
                    finish_reason=finish_reason,
                    interrupted=interrupted,
                    web_search_results=web_search_results,
                )
                
            except httpx.HTTPStatusError as e:
                last_error = e
                _log.warning("HTTP error %d on attempt %d/%d: %s",
                             e.response.status_code, attempt + 1, max_retries, e)
                if e.response.status_code in (429, 500, 502, 503):
                    if attempt < max_retries:
                        wait = min(2 ** (attempt + 1), 60)
                        reason = f"HTTP {e.response.status_code}"
                        _log.info("Retrying in %ds (attempt %d/%d) reason=%s",
                                  wait, attempt + 1, max_retries, reason)
                        print(f"\n⚠️  {reason}. Retry {attempt+1}/{max_retries} in {wait}s...")
                        if status_line and hasattr(status_line, 'set_retry'):
                            status_line.set_retry(attempt + 1, max_retries, wait, reason)
                        # Interruptible sleep — check every 300ms
                        _remaining = wait
                        while _remaining > 0:
                            await asyncio.sleep(min(0.3, _remaining))
                            _remaining -= 0.3
                            if check_interrupt and check_interrupt():
                                return StreamingChatResponse(
                                    content=full_content, raw_json=full_content,
                                    finish_reason="interrupted", interrupted=True,
                                )
                        continue
                raise RuntimeError(f"API error: {e}")
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                _log.warning("Connection error on attempt %d/%d: %s: %s",
                             attempt + 1, max_retries, type(e).__name__, e)
                if attempt < max_retries:
                    wait = min(2 ** (attempt + 1), 60)
                    reason = type(e).__name__
                    _log.info("Retrying in %ds (attempt %d/%d) reason=%s",
                              wait, attempt + 1, max_retries, reason)
                    print(f"\n⚠️  {reason}. Retry {attempt+1}/{max_retries} in {wait}s...")
                    if status_line and hasattr(status_line, 'set_retry'):
                        status_line.set_retry(attempt + 1, max_retries, wait, reason)
                    # Interruptible sleep — check every 300ms
                    _remaining = wait
                    while _remaining > 0:
                        await asyncio.sleep(min(0.3, _remaining))
                        _remaining -= 0.3
                        if check_interrupt and check_interrupt():
                            return StreamingChatResponse(
                                content=full_content, raw_json=full_content,
                                finish_reason="interrupted", interrupted=True,
                            )
                    continue
                raise RuntimeError(f"Request failed after {max_retries} retries: {e}")
        
        raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")
