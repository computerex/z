"""Streaming agent using Cline-style XML tool format."""

import asyncio
import sys
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.markup import escape as rich_escape

from .streaming_client import StreamingJSONClient, StreamingMessage
from .config import Config
from .prompts import get_system_prompt
from .cost_tracker import get_global_tracker
from .interrupt import is_interrupted, is_background_requested, reset_interrupt, start_monitoring, stop_monitoring
from .context_management import (
    estimate_tokens, estimate_messages_tokens, get_model_limits,
    truncate_conversation, truncate_output, truncate_file_content,
    DuplicateDetector
)
from .tool_handlers import ToolHandlers


@dataclass
class ContextItem:
    """An item in the agent's context container."""
    id: int
    type: str  # 'file', 'fragment', 'command_output', 'search_result'
    source: str  # path or command
    content: str
    added_at: float = field(default_factory=time.time)
    line_range: Optional[Tuple[int, int]] = None  # for file fragments
    
    def summary(self) -> str:
        """Return a short summary of this item."""
        lines = len(self.content.splitlines())
        size = len(self.content)
        age = int(time.time() - self.added_at)
        age_str = f"{age}s" if age < 60 else f"{age//60}m"
        
        if self.type == 'file':
            if self.line_range:
                return f"[{self.id}] file: {self.source} (L{self.line_range[0]}-{self.line_range[1]}, {lines}L, {size}B, {age_str} ago)"
            return f"[{self.id}] file: {self.source} ({lines}L, {size}B, {age_str} ago)"
        elif self.type == 'command_output':
            cmd_short = self.source[:40] + '...' if len(self.source) > 40 else self.source
            return f"[{self.id}] cmd: {cmd_short} ({lines}L, {size}B, {age_str} ago)"
        elif self.type == 'search_result':
            return f"[{self.id}] search: {self.source} ({lines} matches, {age_str} ago)"
        else:
            return f"[{self.id}] {self.type}: {self.source} ({lines}L, {age_str} ago)"


class ContextContainer:
    """Manages the agent's working context."""
    
    def __init__(self):
        self._items: Dict[int, ContextItem] = {}
        self._next_id = 1
    
    def add(self, type: str, source: str, content: str, line_range: Optional[Tuple[int, int]] = None) -> int:
        """Add an item to context. Returns the item ID."""
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = ContextItem(
            id=item_id,
            type=type,
            source=source,
            content=content,
            line_range=line_range
        )
        return item_id
    
    def remove(self, item_id: int) -> bool:
        """Remove an item from context."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False
    
    def remove_by_source(self, source: str) -> int:
        """Remove all items with matching source. Returns count removed."""
        to_remove = [id for id, item in self._items.items() if source in item.source]
        for id in to_remove:
            del self._items[id]
        return len(to_remove)
    
    def get(self, item_id: int) -> Optional[ContextItem]:
        return self._items.get(item_id)
    
    def list_items(self) -> List[ContextItem]:
        return list(self._items.values())
    
    def total_size(self) -> int:
        """Total character count of all context items."""
        return sum(len(item.content) for item in self._items.values())
    
    def summary(self) -> str:
        """Return a summary of all context items."""
        if not self._items:
            return "Context is empty."
        
        lines = [f"Context ({len(self._items)} items, {self.total_size():,} chars):"]
        for item in sorted(self._items.values(), key=lambda x: x.added_at):
            lines.append(f"  {item.summary()}")
        return "\n".join(lines)
    
    def clear(self):
        self._items.clear()
        self._next_id = 1


@dataclass
class ParsedToolCall:
    """Parsed tool call from XML format."""
    name: str
    parameters: Dict[str, str]


def strip_thinking_blocks(content: str) -> str:
    """Remove <thinking> blocks from content (MiniMax reasoning)."""
    return re.sub(r'<thinking>.*?</thinking>\s*', '', content, flags=re.DOTALL)


def parse_xml_tool(content: str) -> Optional[ParsedToolCall]:
    """Parse Cline-style XML tool call from content.
    
    Uses smart matching to handle XML examples embedded in content.
    For tools with complex content (write_to_file, replace_in_file), uses
    greedy matching to get the full content including any nested examples.
    """
    # Strip thinking blocks first (MiniMax outputs these)
    content = strip_thinking_blocks(content)
    
    tool_names = [
        'read_file', 'write_to_file', 'replace_in_file',
        'execute_command', 'list_files', 'search_files',
        'check_background_process', 'stop_background_process', 'list_background_processes',
        'list_context', 'remove_from_context', 'analyze_image', 'web_search',
        'attempt_completion'
    ]
    
    # Tools that may have nested XML examples in their content
    complex_content_tools = {'write_to_file', 'replace_in_file'}
    
    # Find ALL tool matches across ALL tool types, track by end position
    all_matches = []  # (end_pos, tool_name, match)
    
    for tool_name in tool_names:
        if tool_name in complex_content_tools:
            # For complex tools with nested XML examples in content:
            # Find FIRST opening tag and LAST closing tag to get the outermost block
            open_tag = f'<{tool_name}>'
            close_tag = f'</{tool_name}>'
            
            # Find the FIRST opening tag (the real tool call, not an inner example)
            first_open = content.find(open_tag)
            if first_open == -1:
                continue
            
            # Find the LAST closing tag (the real closing, not an inner example)
            last_close = content.rfind(close_tag)
            if last_close == -1 or last_close < first_open:
                continue
            
            # Create a match-like object
            inner_start = first_open + len(open_tag)
            inner_content = content[inner_start:last_close]
            
            class FakeMatch:
                def __init__(self, start_pos, end_pos, inner):
                    self._start = start_pos
                    self._end = end_pos
                    self._inner = inner
                def end(self):
                    return self._end
                def group(self, n):
                    return self._inner if n == 1 else content[self._start:self._end]
            
            fake_match = FakeMatch(first_open, last_close + len(close_tag), inner_content)
            all_matches.append((fake_match.end(), tool_name, fake_match))
        else:
            # For simple tools, use non-greedy matching
            pattern = rf'<{tool_name}>(.*?)</{tool_name}>'
            for match in re.finditer(pattern, content, re.DOTALL):
                all_matches.append((match.end(), tool_name, match))
    
    if not all_matches:
        return None
    
    # Use the match with the highest end position (last in content)
    all_matches.sort(key=lambda x: x[0], reverse=True)
    _, tool_name, match = all_matches[0]
    
    inner = match.group(1)
    params = {}
    
    # Parse parameters - order matters because content/diff may contain nested XML
    # Simple params (path, command, etc.) use FIRST closing tag
    # Complex params (content, diff) use LAST closing tag
    simple_params = ['path', 'command', 'background', 'regex', 'file_pattern', 
                    'result', 'query', 'image_path', 'process_id', 'item', 
                    'search_term', 'count', 'recursive', 'id', 'lines', 'question']
    complex_params = ['content', 'diff']
    
    # For tools with complex content (write_to_file, replace_in_file),
    # extract the content/diff FIRST, then only search for other params
    # in the portion BEFORE <content> or <diff> starts
    search_area = inner  # default: search entire inner block
    
    # Find where complex params start (to avoid extracting example XML from content)
    for cp in complex_params:
        open_tag = f'<{cp}>'
        start = inner.find(open_tag)
        if start != -1:
            # Only search for simple params BEFORE the complex param starts
            search_area = inner[:start]
            break
    
    for param_name in simple_params:
        open_tag = f'<{param_name}>'
        close_tag = f'</{param_name}>'
        start = search_area.find(open_tag)
        if start == -1:
            continue
        # Use FIRST closing tag for simple params
        end = search_area.find(close_tag, start)
        if end == -1:
            continue
        value = search_area[start + len(open_tag):end]
        value = value.strip('\n')
        params[param_name] = value
    
    for param_name in complex_params:
        open_tag = f'<{param_name}>'
        close_tag = f'</{param_name}>'
        start = inner.find(open_tag)
        if start == -1:
            continue
        # Use LAST closing tag for complex params (content may have nested examples)
        end = inner.rfind(close_tag)
        if end == -1 or end < start:
            continue
        value = inner[start + len(open_tag):end]
        value = value.strip('\n')
        params[param_name] = value
    
    return ParsedToolCall(name=tool_name, parameters=params)


class ClineAgent:
    """Agent using Cline-style XML tool format with streaming."""
    
    def __init__(
        self,
        config: Config,
        console: Optional[Console] = None,
        max_iterations: int = 30,
    ):
        self.config = config
        self.console = console or Console()
        self.max_iterations = max_iterations
        self.workspace_path = str(Path.cwd().resolve())
        self.cost_tracker = get_global_tracker()
        
        # Conversation history
        self.messages: List[StreamingMessage] = []
        self._initialized = False
        
        # Context container for managing loaded content
        self.context = ContextContainer()
        
        # Duplicate file detection
        self._duplicate_detector = DuplicateDetector()
        
        # Token tracking
        self._last_token_count = 0
        
        # Tool handlers - delegates all tool execution logic
        self.tool_handlers = ToolHandlers(
            config=self.config,
            console=self.console,
            workspace_path=self.workspace_path,
            context=self.context,
            duplicate_detector=self._duplicate_detector
        )
    
    async def cleanup_background_procs_async(self) -> None:
        """Terminate all background processes safely (async)."""
        return await self.tool_handlers.cleanup_background_procs_async()
    
    def cleanup_background_procs(self) -> None:
        """Terminate all background processes safely (sync wrapper)."""
        return self.tool_handlers.cleanup_background_procs()
    
    def list_background_procs(self) -> List[dict]:
        """List all background processes with their status."""
        return self.tool_handlers.list_background_procs()
    
    def save_session(self, path: str) -> None:
        """Save conversation history and context."""
        import json
        
        def sanitize_string(s) -> str:
            """Remove unicode surrogates that can't be encoded."""
            if isinstance(s, str):
                return s.encode('utf-8', errors='replace').decode('utf-8')
            return s
        
        def sanitize_content(content):
            """Sanitize message content (can be string or list for vision)."""
            if isinstance(content, str):
                return sanitize_string(content)
            elif isinstance(content, list):
                result = []
                for item in content:
                    if isinstance(item, dict):
                        sanitized = {}
                        for k, v in item.items():
                            sanitized[k] = sanitize_string(v) if isinstance(v, str) else v
                        result.append(sanitized)
                    else:
                        result.append(sanitize_string(item) if isinstance(item, str) else item)
                return result
            return content
        
        # Serialize context items
        context_items = []
        for item in self.context.list_items():
            context_items.append({
                "id": item.id,
                "type": item.type,
                "source": sanitize_string(item.source),
                "content": sanitize_string(item.content),
                "added_at": item.added_at,
                "line_range": item.line_range
            })
        
        data = {
            "workspace": self.workspace_path,
            "messages": [{"role": m.role, "content": sanitize_content(m.content)} for m in self.messages],
            "context": context_items,
            "context_next_id": self.context._next_id
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load_session(self, path: str, inject_resume: bool = True) -> bool:
        """Load conversation history and context.
        
        Args:
            path: Path to session file
            inject_resume: If True, adds a resume context message to help the model
        """
        import json
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.messages = [StreamingMessage(role=m["role"], content=m["content"]) for m in data["messages"]]
            
            # Load context if present
            if "context" in data:
                self.context.clear()
                for item_data in data["context"]:
                    item = ContextItem(
                        id=item_data["id"],
                        type=item_data["type"],
                        source=item_data["source"],
                        content=item_data["content"],
                        added_at=item_data.get("added_at", time.time()),
                        line_range=tuple(item_data["line_range"]) if item_data.get("line_range") else None
                    )
                    self.context._items[item.id] = item
                self.context._next_id = data.get("context_next_id", max(self.context._items.keys(), default=0) + 1)
            
            # Inject a resume context message to orient the model
            if inject_resume and len(self.messages) > 1:
                resume_msg = self._build_resume_context()
                if resume_msg:
                    self.messages.append(StreamingMessage(role="user", content=resume_msg))
            
            self._initialized = True
            return True
        except:
            return False
    
    def _build_resume_context(self) -> str:
        """Build a resume context message from conversation history."""
        # Find the last few exchanges to summarize
        summary_parts = []
        
        # Get last user request (not tool results)
        last_user_request = None
        for msg in reversed(self.messages):
            if msg.role == "user" and not msg.content.startswith("[") and not msg.content.startswith("<"):
                last_user_request = msg.content[:200]
                break
        
        # Get last assistant action
        last_action = None
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                # Extract tool name if any
                if "<read_file>" in msg.content:
                    last_action = "reading files"
                elif "<write_to_file>" in msg.content:
                    last_action = "writing files"
                elif "<execute_command>" in msg.content:
                    last_action = "executing commands"
                elif "<search_files>" in msg.content:
                    last_action = "searching code"
                elif "<attempt_completion>" in msg.content:
                    last_action = "completing the task"
                else:
                    last_action = "responding"
                break
        
        if last_user_request:
            summary_parts.append(f"Last request: {last_user_request}...")
        if last_action:
            summary_parts.append(f"Last action: {last_action}")
        
        if not summary_parts:
            return ""
        
        return f"[Session resumed. {' '.join(summary_parts)}. Continue where you left off or ask what you need to know.]"
    
    def clear_history(self) -> None:
        """Clear conversation history and context."""
        self.messages = []
        self.context.clear()
        self._duplicate_detector.clear()
        self._last_token_count = 0
        self._initialized = False
    
    def get_token_count(self) -> int:
        """Get estimated token count of current conversation."""
        return estimate_messages_tokens(self.messages)
    
    def get_context_stats(self) -> dict:
        """Get context statistics for display."""
        _, max_allowed = get_model_limits(self.config.model)
        tokens = self.get_token_count()
        return {
            "tokens": tokens,
            "max_allowed": max_allowed,
            "percent": (tokens / max_allowed * 100) if max_allowed > 0 else 0,
            "messages": len(self.messages),
            "context_items": len(self.context.list_items()),
            "context_chars": self.context.total_size(),
        }
    
    def get_token_breakdown(self) -> dict:
        """Get detailed breakdown of where tokens are going."""
        system_tokens = 0
        conv_tokens = 0
        message_sizes = []
        
        for i, msg in enumerate(self.messages):
            tokens = estimate_tokens(msg.content)
            if msg.role == "system":
                system_tokens += tokens
            else:
                conv_tokens += tokens
                # Track largest messages for diagnosis
                preview = msg.content[:100].replace('\n', ' ')
                message_sizes.append({
                    "index": i,
                    "role": msg.role,
                    "tokens": tokens,
                    "preview": preview
                })
        
        # Sort by size, get top 5
        message_sizes.sort(key=lambda x: x["tokens"], reverse=True)
        largest = message_sizes[:5]
        
        return {
            "system": system_tokens,
            "conversation": conv_tokens,
            "total": system_tokens + conv_tokens,
            "message_count": len(self.messages) - 1,  # minus system
            "largest_messages": largest
        }
    
    def compact_history(self, strategy: str = "half") -> int:
        """Compact conversation history by removing older messages.
        
        Strategies:
        - 'half': Remove first half of messages
        - 'quarter': Remove first quarter of messages
        - 'last2': Keep only last 2 exchanges
        
        Returns: number of tokens removed
        """
        before = estimate_messages_tokens(self.messages)
        
        # Find messages (skip system prompt at index 0)
        if len(self.messages) <= 1:
            return 0
        
        # Map strategy names
        strat_map = {'last2': 'lastTwo'}
        strat = strat_map.get(strategy, strategy)
        
        # Apply truncation to messages (preserves system prompt)
        result = truncate_conversation(self.messages, strat)
        self.messages = result.messages
        
        after = estimate_messages_tokens(self.messages)
        return before - after
    
    async def run(self, user_input: str, enable_interrupt: bool = True) -> str:
        """Run the agent with streaming output."""
        
        # Initialize system prompt if first run
        if not self._initialized:
            self.messages = [
                StreamingMessage(role="system", content=get_system_prompt(self.workspace_path)),
            ]
            self._initialized = True
        
        # Add user message
        self.messages.append(StreamingMessage(role="user", content=user_input))
        
        # Start keyboard monitoring for escape key
        if enable_interrupt and sys.stdin.isatty():
            start_monitoring()
        
        try:
            return await self._run_loop()
        finally:
            if enable_interrupt:
                stop_monitoring()
    
    async def _run_loop(self) -> str:
        """Main agent loop."""
        async with StreamingJSONClient(
            api_key=self.config.api_key,
            base_url=self.config.api_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ) as client:
            
            # Get model limits for auto-truncation
            _, max_allowed = get_model_limits(self.config.model)
            
            for iteration in range(self.max_iterations):
                # Check for interrupt BEFORE starting new iteration
                # This ensures user can stop between iterations
                if is_interrupted():
                    self.console.print("\n[yellow][STOP] Interrupted by user[/yellow]")
                    return "[Interrupted - session preserved. Type to continue or start new request]"
                
                # Reset only background flag (not interrupt) for this iteration
                # User needs to explicitly continue after interrupt
                
                # Check if we need to truncate conversation
                self._last_token_count = estimate_messages_tokens(self.messages)
                if self._last_token_count > max_allowed:
                    # Determine aggressiveness based on how far over we are
                    if self._last_token_count > max_allowed * 1.5:
                        strategy = "quarter"  # 75% removal
                    else:
                        strategy = "half"  # 50% removal
                    
                    result = truncate_conversation(self.messages, strategy)
                    self.messages = result.messages
                    self._last_token_count = estimate_messages_tokens(self.messages)
                    self.console.print(f"[dim][!] Context truncated: removed {result.removed_count} messages ({strategy})[/dim]")
                
                print("", end="", flush=True)
                
                full_content = ""
                
                def on_chunk(c: str):
                    nonlocal full_content
                    full_content += c
                    sys.stdout.write(c)
                    sys.stdout.flush()
                
                # Stream the response - using raw mode (no JSON parsing)
                # Web search disabled by default to avoid unnecessary searches
                # Use /search command for explicit web searches
                response = await client.chat_stream_raw(
                    messages=self.messages,
                    on_content=on_chunk,
                    check_interrupt=is_interrupted,
                    enable_web_search=False,
                )
                
                print()  # Newline after stream
                
                # Display web search results if any
                if response.has_web_search:
                    self.console.print(f"\n[dim][Web Search: {len(response.web_search_results)} results][/dim]")
                    for result in response.web_search_results[:3]:  # Show top 3
                        self.console.print(f"[dim]  - {result.title[:60]}... ({result.media})[/dim]")
                
                # Handle interrupt
                if response.interrupted:
                    print("\n[STOP] Interrupted")
                    # Save partial response to history
                    if full_content.strip():
                        self.messages.append(StreamingMessage(
                            role="assistant", 
                            content=full_content + "\n[interrupted by user]"
                        ))
                    return "[Interrupted - press Enter to continue or type new request]"
                
                full_content = response.content or full_content
                
                # Track usage - estimate if API didn't return it
                if response.usage:
                    input_tokens = response.usage.get("prompt_tokens", 0)
                    output_tokens = response.usage.get("completion_tokens", 0)
                    # Fallback to estimation if API returns zeros
                    if input_tokens == 0 and output_tokens == 0:
                        input_tokens = estimate_messages_tokens(self.messages)
                        output_tokens = estimate_tokens(full_content)
                else:
                    # Estimate tokens
                    input_tokens = estimate_messages_tokens(self.messages)
                    output_tokens = estimate_tokens(full_content)
                
                self.cost_tracker.record_call(
                    model=self.config.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_ms=0,
                    tool_calls=0,
                    finish_reason="stop",
                )
                
                # Parse XML tool call from content
                tool_call = parse_xml_tool(full_content)
                
                # Debug output - save full content for analysis
                if os.environ.get("HARNESS_DEBUG"):
                    debug_file = os.path.join(self.workspace_path, ".harness_debug.txt")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(f"=== Full Model Output ({len(full_content)} chars) ===\n")
                        f.write(full_content)
                        f.write(f"\n\n=== Parsed Tool Call ===\n")
                        f.write(f"{tool_call}\n")
                        if tool_call:
                            f.write(f"tool={tool_call.name}, params={tool_call.parameters}\n")
                    print(f"[DEBUG] Saved model output to {debug_file}")
                    print(f"[DEBUG] parse_xml_tool returned: {tool_call}")
                    if tool_call:
                        print(f"[DEBUG] tool={tool_call.name}, params keys={list(tool_call.parameters.keys())}")
                
                if not tool_call:
                    # Check for attempt_completion
                    if "<attempt_completion>" in full_content:
                        # Extract result
                        match = re.search(r'<result>(.*?)</result>', full_content, re.DOTALL)
                        result = match.group(1).strip() if match else full_content
                        self.messages.append(StreamingMessage(role="assistant", content=full_content))
                        return result
                    
                    # No tool call - final response
                    self.messages.append(StreamingMessage(role="assistant", content=full_content))
                    return full_content
                
                # Execute the tool
                tool_result = await self._execute_tool(tool_call)
                
                # Check for interrupt after tool execution
                if is_interrupted():
                    self.console.print("\n[yellow][STOP] Interrupted by user[/yellow]")
                    # Still save the partial conversation
                    self.messages.append(StreamingMessage(role="assistant", content=full_content))
                    if tool_result:
                        self.messages.append(StreamingMessage(
                            role="user",
                            content=f"[{tool_call.name} result - interrupted]\n{tool_result[:500]}..."
                        ))
                    return "[Interrupted - session preserved. Type to continue or start new request]"
                
                # Add to history
                self.messages.append(StreamingMessage(role="assistant", content=full_content))
                self.messages.append(StreamingMessage(
                    role="user",
                    content=f"[{tool_call.name} result]\n{tool_result}"
                ))
            
            return "Max iterations reached."
    
    async def _execute_tool(self, tool: ParsedToolCall) -> str:
        """Execute a tool call and return the result."""
        
        try:
            if tool.name == "read_file":
                path = tool.parameters.get("path", "")
                self.console.print(f"[cyan]> Reading:[/cyan] {path}")
                result = await self.tool_handlers.read_file(tool.parameters)
                lines = result.count('\n') + 1
                self.console.print(f"[dim]   ({lines} lines)[/dim]")
                
            elif tool.name == "write_to_file":
                path = tool.parameters.get("path", "")
                content = tool.parameters.get("content", "")
                self.console.print(f"[green]> Writing:[/green] {path} ({len(content)} bytes)")
                result = await self.tool_handlers.write_file(tool.parameters)
                
            elif tool.name == "replace_in_file":
                path = tool.parameters.get("path", "")
                self.console.print(f"[yellow]> Editing:[/yellow] {path}")
                result = await self.tool_handlers.replace_in_file(tool.parameters)
                
            elif tool.name == "execute_command":
                # execute_command handles its own display
                result = await self.tool_handlers.execute_command(tool.parameters)
                
            elif tool.name == "list_files":
                path = tool.parameters.get("path", "")
                self.console.print(f"[blue]> Listing:[/blue] {path}")
                result = await self.tool_handlers.list_files(tool.parameters)
                count = len(result.splitlines())
                self.console.print(f"[dim]   ({count} items)[/dim]")
                
            elif tool.name == "search_files":
                regex = tool.parameters.get("regex", "")
                self.console.print(f"[magenta]> Searching:[/magenta] {regex}")
                result = await self.tool_handlers.search_files(tool.parameters)
                matches = len(result.splitlines()) if result != "(no matches)" else 0
                self.console.print(f"[dim]   ({matches} matches)[/dim]")
                
            elif tool.name == "check_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(f"[cyan]> Checking background process:[/cyan] {bg_id or 'all'}")
                result = await self.tool_handlers.check_background_process(tool.parameters)
                
            elif tool.name == "stop_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(f"[red]> Stopping background process:[/red] {bg_id}")
                result = await self.tool_handlers.stop_background_process(tool.parameters)
                
            elif tool.name == "list_background_processes":
                self.console.print(f"[cyan]> Listing background processes[/cyan]")
                result = await self.tool_handlers.list_background_processes(tool.parameters)
                
            elif tool.name == "list_context":
                self.console.print(f"[blue]> Listing context[/blue]")
                result = self.context.summary()
                self.console.print(f"[dim]   ({len(self.context.list_items())} items, {self.context.total_size():,} chars)[/dim]")
                
            elif tool.name == "remove_from_context":
                item_id = tool.parameters.get("id", "")
                source = tool.parameters.get("source", "")
                self.console.print(f"[yellow]> Removing from context:[/yellow] {item_id or source}")
                result = self._remove_from_context(tool.parameters)
                
            elif tool.name == "analyze_image":
                path = tool.parameters.get("path", "")
                question = tool.parameters.get("question", "Describe this image in detail.")
                self.console.print(f"[magenta]> Analyzing image:[/magenta] {path}")
                result = await self.tool_handlers.analyze_image(tool.parameters)
                
            elif tool.name == "web_search":
                query = tool.parameters.get("query", "")
                self.console.print(f"[cyan]> Web Search:[/cyan] {query}")
                self.console.print("[dim]  (searching... this may take up to 2 minutes)[/dim]")
                result = await self.tool_handlers.web_search(tool.parameters)
                
            elif tool.name == "attempt_completion":
                self.console.print("[green]> Task complete[/green]")
                result = "Task completed."
                
            else:
                result = f"Unknown tool: {tool.name}"
            
            return result
            
        except Exception as e:
            self.console.print(f"[red][X] {tool.name}: {rich_escape(str(e))}[/red]")
            return f"Error: {str(e)}"
    
    def _remove_from_context(self, params: Dict[str, str]) -> str:
        """Remove items from context by ID(s) or source pattern."""
        item_id = params.get("id", "")
        source = params.get("source", "")
        
        if item_id:
            # Handle multiple IDs (comma-separated or from multiple <id> tags)
            id_strs = [s.strip() for s in item_id.replace('\n', ',').split(',') if s.strip()]
            removed = []
            errors = []
            
            for id_str in id_strs:
                try:
                    id_int = int(id_str)
                    if self.context.remove(id_int):
                        removed.append(id_int)
                    else:
                        errors.append(f"{id_int} (not found)")
                except ValueError:
                    errors.append(f"{id_str} (invalid)")
            
            result = []
            if removed:
                result.append(f"Removed context items: {', '.join(map(str, removed))}")
            if errors:
                result.append(f"Errors: {', '.join(errors)}")
            return "\n".join(result) if result else "No items removed."
        elif source:
            count = self.context.remove_by_source(source)
            if count > 0:
                return f"Removed {count} context item(s) matching '{source}'."
            else:
                return f"No context items matching '{source}'."
        else:
            return "Error: Provide either 'id' or 'source' parameter."


# Alias for backward compatibility
StreamingAgent = ClineAgent