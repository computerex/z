"""JSON-based streaming LLM client - true token-by-token streaming."""

import json
import asyncio
import httpx
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass 
class StreamingMessage:
    """A message for the streaming client."""
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class StreamingToolCall:
    """A parsed tool call from the JSON response."""
    name: str
    parameters: Dict[str, Any]


@dataclass
class StreamingChatResponse:
    """Response from the streaming JSON approach."""
    thinking: Optional[str] = None
    tool_call: Optional[StreamingToolCall] = None
    message: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw_json: str = ""
    finish_reason: str = "stop"
    
    @property
    def is_truncated(self) -> bool:
        """Check if response was cut off due to length."""
        return self.finish_reason == "length"
    
    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None


class ContentExtractor:
    """Extract content/code fields from streaming JSON in real-time."""
    
    # Fields to stream (parameter values that contain code/content)
    STREAM_FIELDS = ('"content": "', '"content":"', '"old_string": "', '"old_string":"', 
                     '"new_string": "', '"new_string":"')
    
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
        max_tokens: int = 16384,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0)
        )
        return self
        
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://cline.bot",
            "X-Title": "Cline",
        }
    
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
                                usage = data["usage"]
                            
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
