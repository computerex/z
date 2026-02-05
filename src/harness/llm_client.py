"""LLM client for Z.AI GLM-4.7 API."""

import json
import time
import httpx
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from .config import Config
from .cost_tracker import CostTracker, get_global_tracker


@dataclass
class Message:
    """A chat message."""
    
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result = {"role": self.role}
        
        if self.content is not None:
            result["content"] = self.content
        
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        if self.name:
            result["name"] = self.name
        
        return result


@dataclass
class ChatResponse:
    """Response from the LLM API."""
    
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


class LLMClient:
    """Client for Z.AI GLM-4.7 API with tool calling support."""
    
    def __init__(self, config: Config, cost_tracker: Optional[CostTracker] = None):
        self.config = config
        self.cost_tracker = cost_tracker or get_global_tracker()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(7200.0, connect=60.0),  # 2 hour timeout for long responses
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://cline.bot",
            "X-Title": "Cline",
        }
    
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        max_retries: int = 5,
    ) -> ChatResponse:
        """Send a chat completion request with automatic retry on rate limits.
        
        Args:
            messages: List of chat messages.
            tools: Optional list of tool definitions.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            stream: Whether to stream the response.
            max_retries: Maximum number of retries on rate limit errors.
        
        Returns:
            ChatResponse with the LLM's response.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.api_url.rstrip('/')}/chat/completions"
        
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        start_time = time.perf_counter()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                
                data = response.json()
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                result = self._parse_response(data)
                
                # Track costs
                usage = result.usage
                if usage and self.cost_tracker:
                    tool_call_count = len(result.tool_calls) if result.tool_calls else 0
                    self.cost_tracker.record_call(
                        model=self.config.model,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        duration_ms=duration_ms,
                        tool_calls=tool_call_count,
                        finish_reason=result.finish_reason,
                    )
                
                return result
            
            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code
                error_detail = ""
                try:
                    error_detail = e.response.text
                except:
                    pass
                
                # Retry on rate limit (429) or server errors (5xx)
                if status_code == 429 or status_code >= 500:
                    if attempt < max_retries:
                        # Exponential backoff: 2, 4, 8, 16, 32 seconds
                        wait_time = min(2 ** (attempt + 1), 60)
                        import asyncio
                        print(f"\n⚠️  Rate limited (HTTP {status_code}). Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                
                raise RuntimeError(
                    f"API request failed with status {status_code}: {error_detail}"
                )
            
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt + 1), 60)
                    import asyncio
                    print(f"\n⚠️  Request timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"API request timed out after {max_retries} retries")
            
            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt + 1), 60)
                    import asyncio
                    print(f"\n⚠️  Request error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"API request failed after {max_retries} retries: {str(e)}")
        
        # Should not reach here, but just in case
        raise RuntimeError(f"API request failed after {max_retries} retries: {last_error}")
    
    def _parse_response(self, data: Dict[str, Any]) -> ChatResponse:
        """Parse API response into ChatResponse."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        # Extract tool calls if present
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
        
        return ChatResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            raw_response=data,
        )
    
    async def chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_content: Optional[callable] = None,
        on_tool_start: Optional[callable] = None,
        on_tool_content: Optional[callable] = None,
        max_retries: int = 5,
    ) -> ChatResponse:
        """Stream a chat completion response with tool call support."""
        import sys
        
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.config.api_url.rstrip('/')}/chat/completions"
        
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # May help with streaming
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        start_time = time.perf_counter()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                content_parts = []
                tool_calls_data = {}
                finish_reason = ""
                usage = {}
                current_tool_name = ""
                current_tool_idx = -1
                in_content_field = False
                content_buffer = ""
                content_finished = False
                
                async with self._client.stream(
                    "POST",
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=httpx.Timeout(300.0, connect=30.0),
                ) as response:
                    response.raise_for_status()
                    
                    line_buffer = ""
                    done = False
                    
                    async for raw_chunk in response.aiter_bytes():
                        if done:
                            break
                            
                        line_buffer += raw_chunk.decode('utf-8', errors='ignore')
                        
                        while '\n' in line_buffer:
                            line, line_buffer = line_buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                done = True
                                break
                            
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            # Content tokens - stream immediately
                            if "content" in delta and delta["content"]:
                                chunk = delta["content"]
                                content_parts.append(chunk)
                                if on_content:
                                    on_content(chunk)
                            
                            # Tool calls - accumulate and stream content fields
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_data:
                                        tool_calls_data[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc.get("id"):
                                        tool_calls_data[idx]["id"] = tc["id"]
                                    if "function" in tc:
                                        fn = tc["function"]
                                        if fn.get("name"):
                                            # New tool starting
                                            current_tool_name = fn["name"]
                                            current_tool_idx = idx
                                            tool_calls_data[idx]["function"]["name"] = fn["name"]
                                            if on_tool_start:
                                                on_tool_start(current_tool_name)
                                            in_content_field = False
                                            content_buffer = ""
                                            content_finished = False
                                        if fn.get("arguments"):
                                            args_chunk = fn["arguments"]
                                            tool_calls_data[idx]["function"]["arguments"] += args_chunk
                                            
                                            # Stream content/new_string fields for live display
                                            if on_tool_content and not content_finished:
                                                content_fields = ['"content":"', '"content": "', 
                                                                  '"new_string":"', '"new_string": "']
                                                
                                                if not in_content_field:
                                                    content_buffer += args_chunk
                                                    for pat in content_fields:
                                                        if pat in content_buffer:
                                                            in_content_field = True
                                                            start_idx = content_buffer.find(pat) + len(pat)
                                                            rest = content_buffer[start_idx:]
                                                            content_buffer = ""
                                                            if rest:
                                                                rest = rest.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                                                                for end_pat in ['"}', '","']:
                                                                    if end_pat in rest:
                                                                        rest = rest[:rest.find(end_pat)]
                                                                        in_content_field = False
                                                                        content_finished = True
                                                                        break
                                                                if rest:
                                                                    on_tool_content(rest)
                                                            break
                                                elif in_content_field:
                                                    display = args_chunk
                                                    for end_pat in ['"}', '","']:
                                                        if end_pat in display:
                                                            display = display[:display.find(end_pat)]
                                                            in_content_field = False
                                                            content_finished = True
                                                            break
                                                    display = display.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                                                    if display:
                                                        on_tool_content(display)
                            
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            if "usage" in data:
                                usage = data["usage"]
                
                duration_ms = (time.perf_counter() - start_time) * 1000
                content = "".join(content_parts) if content_parts else None
                tool_calls = list(tool_calls_data.values()) if tool_calls_data else None
                
                result = ChatResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                
                if usage and self.cost_tracker:
                    self.cost_tracker.record_call(
                        model=self.config.model,
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        duration_ms=duration_ms,
                        tool_calls=len(tool_calls) if tool_calls else 0,
                        finish_reason=finish_reason,
                    )
                
                return result
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503):
                    if attempt < max_retries:
                        wait_time = min(2 ** (attempt + 1), 60)
                        print(f"\n⚠️  HTTP {e.response.status_code}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                raise RuntimeError(f"API request failed: {e}")
                
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt + 1), 60)
                    print(f"\n⚠️  Connection error. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"API request failed: {e}")
        
        raise RuntimeError(f"API request failed after {max_retries} retries: {last_error}")
