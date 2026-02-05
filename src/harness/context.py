"""Context management for the harness."""

import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .llm_client import Message


@dataclass
class ContextWindow:
    """Manages the context window with compaction."""
    
    max_tokens: int = 32000
    reserve_tokens: int = 4000  # Reserve for response
    model: str = "gpt-4"  # For tokenizer (tiktoken approximation)
    
    _messages: List[Message] = field(default_factory=list)
    _system_message: Optional[Message] = None
    _encoder: Any = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize tokenizer."""
        try:
            self._encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self._encoder.encode(text))
    
    def count_message_tokens(self, message: Message) -> int:
        """Count tokens in a message."""
        tokens = 4  # Message overhead
        
        if message.content:
            tokens += self.count_tokens(message.content)
        
        if message.tool_calls:
            tokens += self.count_tokens(str(message.tool_calls))
        
        if message.name:
            tokens += self.count_tokens(message.name)
        
        return tokens
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self.max_tokens - self.reserve_tokens - self.current_tokens
    
    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        total = 0
        if self._system_message:
            total += self.count_message_tokens(self._system_message)
        
        for msg in self._messages:
            total += self.count_message_tokens(msg)
        
        return total
    
    def set_system_message(self, content: str) -> None:
        """Set the system message."""
        self._system_message = Message(role="system", content=content)
    
    def add_message(self, message: Message) -> bool:
        """Add a message to the context.
        
        Returns:
            True if message was added, False if context is full.
        """
        msg_tokens = self.count_message_tokens(message)
        
        if msg_tokens > self.available_tokens:
            # Try compaction first
            self.compact()
            if msg_tokens > self.available_tokens:
                return False
        
        self._messages.append(message)
        return True
    
    def add_user_message(self, content: str) -> bool:
        """Add a user message."""
        return self.add_message(Message(role="user", content=content))
    
    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add an assistant message."""
        return self.add_message(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ))
    
    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        name: str = "",
    ) -> bool:
        """Add a tool result message."""
        return self.add_message(Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        ))
    
    def get_messages(self) -> List[Message]:
        """Get all messages including system message."""
        messages = []
        if self._system_message:
            messages.append(self._system_message)
        messages.extend(self._messages)
        return messages
    
    def compact(self, target_reduction: float = 0.3) -> int:
        """Compact the context to free up tokens.
        
        Uses strategies:
        1. Truncate long tool outputs
        2. Summarize older messages
        3. Remove older messages (keeping important ones)
        
        Args:
            target_reduction: Target reduction as fraction of current tokens.
        
        Returns:
            Number of tokens freed.
        """
        initial_tokens = self.current_tokens
        target_tokens = int(initial_tokens * (1 - target_reduction))
        
        # Strategy 1: Truncate long tool outputs
        for msg in self._messages:
            if msg.role == "tool" and msg.content:
                tokens = self.count_tokens(msg.content)
                if tokens > 1000:
                    # Truncate to first and last 400 tokens
                    encoded = self._encoder.encode(msg.content)
                    if len(encoded) > 800:
                        truncated = encoded[:400] + encoded[-400:]
                        msg.content = (
                            self._encoder.decode(encoded[:400])
                            + "\n...[truncated]...\n"
                            + self._encoder.decode(encoded[-400:])
                        )
        
        # Strategy 2: Remove older messages if still over target
        while self.current_tokens > target_tokens and len(self._messages) > 4:
            # Keep the most recent messages and first message
            # Find the oldest non-essential message
            for i in range(1, len(self._messages) - 2):
                msg = self._messages[i]
                # Don't remove important messages (tool calls in progress)
                if msg.role == "assistant" and msg.tool_calls:
                    # Check if there's a corresponding tool result
                    has_result = any(
                        m.role == "tool" and 
                        m.tool_call_id in [tc.get("id") for tc in msg.tool_calls]
                        for m in self._messages[i+1:]
                    )
                    if not has_result:
                        continue
                
                self._messages.pop(i)
                break
            else:
                break  # No more messages to remove
        
        return initial_tokens - self.current_tokens
    
    def clear(self) -> None:
        """Clear all messages except system message."""
        self._messages = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "total_messages": len(self._messages),
            "current_tokens": self.current_tokens,
            "available_tokens": self.available_tokens,
            "max_tokens": self.max_tokens,
            "has_system_message": self._system_message is not None,
        }


class ContextManager:
    """High-level context management for the agent."""
    
    def __init__(self, max_tokens: int = 32000):
        self.window = ContextWindow(max_tokens=max_tokens)
        self._turn_count = 0
    
    def initialize(self, system_prompt: str) -> None:
        """Initialize with a system prompt."""
        self.window.set_system_message(system_prompt)
    
    def add_user_turn(self, content: str) -> bool:
        """Add a user turn."""
        self._turn_count += 1
        return self.window.add_user_message(content)
    
    def add_assistant_turn(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Add an assistant turn."""
        return self.window.add_assistant_message(content, tool_calls)
    
    def add_tool_result(
        self,
        tool_call_id: str,
        result: str,
        name: str = "",
    ) -> bool:
        """Add a tool result."""
        return self.window.add_tool_result(tool_call_id, result, name)
    
    def get_messages(self) -> List[Message]:
        """Get all messages for API call."""
        return self.window.get_messages()
    
    def needs_compaction(self, threshold: float = 0.85) -> bool:
        """Check if context needs compaction."""
        usage = self.window.current_tokens / (self.window.max_tokens - self.window.reserve_tokens)
        return usage > threshold
    
    def compact_if_needed(self) -> int:
        """Compact context if needed."""
        if self.needs_compaction():
            return self.window.compact()
        return 0
    
    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        return self._turn_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            **self.window.get_summary(),
            "turn_count": self._turn_count,
        }
