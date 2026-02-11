"""Streaming agent using Cline-style XML tool format."""

import asyncio
import sys
import os
import re
import time
import json
import datetime
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
from .todo_manager import TodoManager, TodoStatus
from .smart_context import SmartContextManager
from .status_line import StatusLine


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
        'manage_todos', 'attempt_completion'
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
                    'search_term', 'count', 'recursive', 'id', 'lines', 'question',
                    'action', 'title', 'status', 'parent_id', 'notes', 'context_refs',
                    'start_line', 'end_line', 'offset', 'limit']
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
        
        # Todo list for tracking goals/objectives
        self.todo_manager = TodoManager()
        
        # Smart context manager for intelligent eviction/compaction
        self.smart_context = SmartContextManager(self.todo_manager)
        
        # Token tracking
        self._last_token_count = 0
        
        # Persistent bottom status line
        self.status = StatusLine(enabled=sys.stdin.isatty())
        
        # Thrash detection: track consecutive edit failures per file
        # {filepath: {"failures": int, "last_error": str}}
        self._edit_failures: Dict[str, dict] = {}
        
        # Reasoning checkpoint: track consecutive tool calls with low reasoning
        # When threshold is hit, inject a "stop and think" checkpoint
        self._consecutive_low_reasoning = 0
        self._force_reasoning_next = False
        
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
            "context_next_id": self.context._next_id,
            "todos": self.todo_manager.to_dict(),
            "smart_context": self.smart_context.to_dict(),
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
            
            # CRITICAL: Verify system prompt integrity after load
            # If messages[0] is not a system prompt, inject a fresh one
            if not self.messages or self.messages[0].role != "system":
                self.console.print("[yellow][!] Session missing system prompt - injecting fresh one[/yellow]")
                system_msg = StreamingMessage(role="system", content=get_system_prompt(self.workspace_path))
                self.messages.insert(0, system_msg)
            else:
                # Verify system prompt has tool definitions (not truncated/evicted)
                sys_content = self.messages[0].content
                if not isinstance(sys_content, str) or "TOOL USE" not in sys_content or "read_file" not in sys_content:
                    self.console.print("[yellow][!] System prompt appears corrupted - replacing with fresh one[/yellow]")
                    self.messages[0] = StreamingMessage(role="system", content=get_system_prompt(self.workspace_path))
            
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
            
            # Load todos if present
            if "todos" in data:
                self.todo_manager = TodoManager.from_dict(data["todos"])
                # Reconnect smart context manager to the loaded todo manager
                self.smart_context = SmartContextManager(self.todo_manager)
            
            # Load smart context state if present
            if "smart_context" in data:
                self.smart_context.load_dict(data["smart_context"])
            
            # Inject a resume context message to orient the model
            if inject_resume and len(self.messages) > 1:
                resume_msg = self._build_resume_context()
                if resume_msg:
                    self.messages.append(StreamingMessage(role="user", content=resume_msg))
            
            self._initialized = True
            return True
        except Exception as e:
            self.console.print(f"[red][!] Failed to load session: {e}[/red]")
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
        
        # Include todo list state if available
        todo_state = self.todo_manager.format_list(include_completed=False)
        if todo_state and "empty" not in todo_state.lower():
            summary_parts.append(f"\n{todo_state}")
        
        # Include eviction recovery notice
        recovery = self.smart_context.build_context_recovery_notice()
        if recovery:
            summary_parts.append(f"\n{recovery}")
        
        return f"[Session resumed. {' '.join(summary_parts)}. Continue where you left off or ask what you need to know.]"
    
    def clear_history(self) -> None:
        """Clear conversation history and context."""
        self.messages = []
        self.context.clear()
        self._duplicate_detector.clear()
        self.todo_manager.clear()
        self.smart_context = SmartContextManager(self.todo_manager)
        self._last_token_count = 0
        self._initialized = False
    
    def get_token_count(self) -> int:
        """Get estimated token count of current conversation."""
        return estimate_messages_tokens(self.messages)
    
    def get_context_stats(self) -> dict:
        """Get context statistics for display."""
        _, max_allowed = get_model_limits(self.config.model)
        tokens = self.get_token_count()
        todos = self.todo_manager.list_all()
        active_todos = self.todo_manager.list_active()
        return {
            "tokens": tokens,
            "max_allowed": max_allowed,
            "percent": (tokens / max_allowed * 100) if max_allowed > 0 else 0,
            "messages": len(self.messages),
            "context_items": len(self.context.list_items()),
            "context_chars": self.context.total_size(),
            "todos_total": len(todos),
            "todos_active": len(active_todos),
            "todos_completed": len(todos) - len(active_todos),
            "evictions": len(self.smart_context.compaction_traces),
        }
    
    def get_token_breakdown(self) -> dict:
        """Get detailed breakdown of where tokens are going."""
        system_tokens = 0
        conv_tokens = 0
        message_sizes = []
        
        for i, msg in enumerate(self.messages):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tokens = estimate_tokens(content)
            if msg.role == "system":
                system_tokens += tokens
            else:
                conv_tokens += tokens
                # Track largest messages for diagnosis
                preview = content[:100].replace('\n', ' ')
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
    
    def dump_context(self, path: Optional[str] = None, reason: str = "") -> str:
        """Dump the full model context (all messages) to a JSON log file.
        
        This writes EXACTLY what would be sent to the API, making it possible
        to trace what the model actually sees.
        
        Args:
            path: Output file path. If None, auto-generates timestamped path.
            reason: Optional label for why this dump was triggered.
            
        Returns:
            Path to the dump file.
        """
        if path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.workspace_path, f".harness_context_{ts}.json")
        
        # Build the exact payload that would be sent to the API
        messages_data = []
        for i, msg in enumerate(self.messages):
            content = msg.content
            content_str = content if isinstance(content, str) else str(content)
            tokens = estimate_tokens(content_str)
            char_count = len(content_str)
            messages_data.append({
                "index": i,
                "role": msg.role,
                "tokens_est": tokens,
                "chars": char_count,
                "content": content,
            })
        
        total_tokens = estimate_messages_tokens(self.messages)
        _, max_allowed = get_model_limits(self.config.model)
        
        # System prompt analysis
        sys_msg = self.messages[0] if self.messages and self.messages[0].role == "system" else None
        sys_analysis = {}
        if sys_msg:
            sys_content = sys_msg.content if isinstance(sys_msg.content, str) else str(sys_msg.content)
            sys_analysis = {
                "chars": len(sys_content),
                "tokens_est": estimate_tokens(sys_content),
                "has_read_file": "read_file" in sys_content,
                "has_write_to_file": "write_to_file" in sys_content,
                "has_execute_command": "execute_command" in sys_content,
                "has_replace_in_file": "replace_in_file" in sys_content,
                "has_manage_todos": "manage_todos" in sys_content,
                "has_attempt_completion": "attempt_completion" in sys_content,
                "has_TOOL_USE_section": "TOOL USE" in sys_content,
                "has_RULES_section": "RULES" in sys_content,
                "first_200": sys_content[:200],
                "last_200": sys_content[-200:],
            }
        
        dump = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "model": self.config.model,
            "api_url": self.config.api_url,
            "workspace": self.workspace_path,
            "summary": {
                "total_messages": len(self.messages),
                "total_tokens_est": total_tokens,
                "max_allowed": max_allowed,
                "percent_used": f"{(total_tokens / max_allowed * 100):.1f}%" if max_allowed > 0 else "N/A",
                "system_prompt_analysis": sys_analysis,
            },
            "todos": self.todo_manager.to_dict(),
            "compaction_traces": [t.format_notice() for t in self.smart_context.compaction_traces],
            "messages": messages_data,
        }
        
        Path(path).write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    
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
        
        # Capture original request for todo grounding (first real user message)
        if not self.todo_manager.original_request and not user_input.startswith("["):
            self.todo_manager.set_original_request(user_input[:500])
        
        # Start keyboard monitoring for escape key
        if enable_interrupt and sys.stdin.isatty():
            start_monitoring()
        
        try:
            return await self._run_loop()
        finally:
            self.status.clear()
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
                self.status.set_iterations(iteration + 1, self.max_iterations)
                
                # Check for interrupt BEFORE starting new iteration
                # This ensures user can stop between iterations
                if is_interrupted():
                    self.status.clear()
                    self.console.print("\n[yellow][STOP] Interrupted by user[/yellow]")
                    return "[Interrupted - session preserved. Type to continue or start new request]"
                
                # Reset only background flag (not interrupt) for this iteration
                # User needs to explicitly continue after interrupt
                
                # Check if we need to compact/truncate conversation
                self._last_token_count = estimate_messages_tokens(self.messages)
                compact_threshold = int(max_allowed * 0.85)
                if self._last_token_count > compact_threshold:
                    self.status.update("Compacting context...", StatusLine.COMPACTING)
                    # Smart compaction: dedup → compact low-priority → evict lowest-scored
                    self.messages, freed, report = self.smart_context.compact_context(
                        self.messages, max_allowed, current_tokens=self._last_token_count
                    )
                    if freed > 0:
                        self._last_token_count = estimate_messages_tokens(self.messages)
                        self.console.print(f"[dim][!] Smart compaction: freed {freed:,} tokens ({report})[/dim]")
                        
                        # Inject reorientation so the agent knows context changed
                        todo_state = self.todo_manager.format_list(include_completed=False)
                        recovery = self.smart_context.build_context_recovery_notice()
                        
                        reorientation = "[CONTEXT COMPACTED - Re-orienting]\n"
                        if todo_state and "empty" not in todo_state.lower():
                            reorientation += f"\n{todo_state}\n"
                        if recovery:
                            reorientation += f"\n{recovery}\n"
                        reorientation += "\nCheck your todos and continue working on the current in-progress item."
                        
                        self.messages.append(StreamingMessage(
                            role="user", content=reorientation
                        ))
                
                # Auto-dump context before API call if debug enabled
                if os.environ.get("HARNESS_DEBUG_CONTEXT"):
                    dump_path = self.dump_context(
                        reason=f"pre_api_call_iter_{iteration}"
                    )
                    self.console.print(f"[dim][DEBUG] Context dumped to {dump_path}[/dim]")
                
                # Show status: sending request
                token_info = f"{self._last_token_count // 1000}k tokens"
                self.status.update(f"Sending to LLM ({token_info})", StatusLine.SENDING)
                
                full_content = ""
                first_token = True
                
                def on_chunk(c: str):
                    nonlocal full_content, first_token
                    if first_token:
                        # First token arrived — clear status line before writing
                        # stream content to avoid ghost text interleaving
                        self.status.clear()
                        first_token = False
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
                    status_line=self.status,
                )
                
                self.status.clear()
                print()  # Newline after stream
                
                # Display web search results if any
                if response.has_web_search:
                    self.console.print(f"\n[dim][Web Search: {len(response.web_search_results)} results][/dim]")
                    for result in response.web_search_results[:3]:  # Show top 3
                        self.console.print(f"[dim]  - {result.title[:60]}... ({result.media})[/dim]")
                
                # Handle interrupt
                if response.interrupted:
                    self.status.clear()
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
                    finish_reason=response.finish_reason,
                )
                
                # Detect truncated output: the model hit max_tokens mid-response.
                # This commonly happens after compaction frees space, the model
                # generates a long response, and the closing XML tag gets cut off.
                output_truncated = response.is_truncated or self._has_unclosed_tool_tag(full_content)
                
                if output_truncated:
                    self.console.print("[dim][!] Output truncated — asking model to continue[/dim]")
                    # Save the partial response and ask the model to finish
                    self.messages.append(StreamingMessage(role="assistant", content=full_content))
                    self.messages.append(StreamingMessage(
                        role="user",
                        content=(
                            "[SYSTEM: Your output was truncated before completing the tool call. "
                            "Please continue from where you left off. Do NOT repeat what you already wrote — "
                            "just output the remaining XML to complete the tool call.]"
                        ),
                    ))
                    continue  # next iteration will get the continuation
                
                # Parse XML tool call from content
                tool_call = parse_xml_tool(full_content)
                
                # Reasoning checkpoint enforcement: if we previously injected
                # a "stop and think" checkpoint, strictly require reasoning now.
                if tool_call and self._force_reasoning_next and tool_call.name not in ('manage_todos', 'attempt_completion', 'list_context'):
                    tag_pos = full_content.find(f'<{tool_call.name}>')
                    if tag_pos >= 0:
                        reasoning_text = full_content[:tag_pos].strip()
                        reasoning_text = re.sub(r'<thinking>.*?</thinking>', '', reasoning_text, flags=re.DOTALL).strip()
                        if len(reasoning_text) < 80:
                            self.console.print("[dim][!] Reasoning checkpoint — asking model to think before continuing[/dim]")
                            self.messages.append(StreamingMessage(role="assistant", content=full_content))
                            self.messages.append(StreamingMessage(
                                role="user",
                                content=(
                                    "[SYSTEM: REASONING CHECKPOINT. You were asked to stop and think but jumped straight to a tool call. "
                                    "Before continuing, you MUST write a short paragraph summarizing: "
                                    "(1) what you have accomplished so far, (2) what your current objective is, "
                                    "(3) what remains to be done, and (4) why this specific next step is correct. "
                                    "Write your reasoning first, then repeat the tool call.]"
                                ),
                            ))
                            continue  # force reasoning before allowing tool call
                
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
                    # No tool call — model produced text. Reset reasoning counters.
                    self._consecutive_low_reasoning = 0
                    self._force_reasoning_next = False
                    
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
                self.status.update(f"Executing: {tool_call.name}", StatusLine.TOOL_EXEC)
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
                
                # Spill any large tool result to a file, keeping only a
                # compact preview inline.  This applies uniformly to ALL tools
                # (read_file, search_files, execute_command, etc.) so that no
                # single tool result can blow up the context window.
                tool_result = self.tool_handlers.spill_output_to_file(
                    tool_result, tool_call.name
                )
                
                # Track reasoning quality for checkpoint detection
                _is_exempt = tool_call.name in ('manage_todos', 'attempt_completion', 'list_context')
                if not _is_exempt:
                    tag_pos = full_content.find(f'<{tool_call.name}>')
                    reasoning_text = ''
                    if tag_pos >= 0:
                        reasoning_text = full_content[:tag_pos].strip()
                        reasoning_text = re.sub(r'<thinking>.*?</thinking>', '', reasoning_text, flags=re.DOTALL).strip()
                    
                    if len(reasoning_text) < 40:
                        self._consecutive_low_reasoning += 1
                    else:
                        # Decent reasoning — reset the counter and clear force flag
                        self._consecutive_low_reasoning = 0
                        self._force_reasoning_next = False
                
                # Add to history
                self.messages.append(StreamingMessage(role="assistant", content=full_content))
                
                # Build tool result message
                header_parts = []
                
                # Active todos at top for grounding
                active_todos = self.todo_manager.list_active()
                if active_todos:
                    in_progress = [t for t in active_todos if t.status.value == "in-progress"]
                    not_started = [t for t in active_todos if t.status.value == "not-started"]
                    todo_hint = "[ACTIVE TODOS]"
                    if in_progress:
                        todo_hint += "\n  In progress: " + "; ".join(f"[{t.id}] {t.title}" for t in in_progress)
                    if not_started:
                        todo_hint += "\n  Remaining: " + "; ".join(f"[{t.id}] {t.title}" for t in not_started[:5])
                        if len(not_started) > 5:
                            todo_hint += f" (+{len(not_started) - 5} more)"
                    header_parts.append(todo_hint)
                
                # Reasoning checkpoint: if too many consecutive low-reasoning calls,
                # inject a "stop and think" interleaving checkpoint
                CHECKPOINT_THRESHOLD = 3
                if self._consecutive_low_reasoning >= CHECKPOINT_THRESHOLD and not _is_exempt:
                    self.console.print(f"[dim][!] {self._consecutive_low_reasoning} consecutive tool calls without reasoning — injecting checkpoint[/dim]")
                    header_parts.append(
                        f"[REASONING CHECKPOINT — You have made {self._consecutive_low_reasoning} consecutive tool calls "
                        "without substantive reasoning. STOP. Before making ANY more tool calls, "
                        "write a paragraph summarizing: (1) what you have accomplished so far, "
                        "(2) what your current objective is, (3) what remains to be done, "
                        "and (4) your plan for the next steps. This is mandatory.]"
                    )
                    self._force_reasoning_next = True
                    self._consecutive_low_reasoning = 0  # reset after checkpoint
                
                header = "\n".join(header_parts)
                if header:
                    result_content = f"{header}\n\n[{tool_call.name} result]\n{tool_result}"
                else:
                    result_content = f"[{tool_call.name} result]\n{tool_result}"
                
                self.messages.append(StreamingMessage(
                    role="user",
                    content=result_content
                ))
            
            self.status.clear()
            return "Max iterations reached."
    
    # Tool tag names that the model emits as XML tool calls
    _TOOL_TAGS = [
        'read_file', 'write_to_file', 'replace_in_file',
        'execute_command', 'list_files', 'search_files',
        'check_background_process', 'stop_background_process', 'list_background_processes',
        'list_context', 'remove_from_context', 'analyze_image', 'web_search',
        'manage_todos', 'attempt_completion',
    ]

    @staticmethod
    def _has_unclosed_tool_tag(content: str) -> bool:
        """Detect if the content has an opening tool XML tag without a matching close.
        
        This indicates the model's output was truncated mid-tool-call,
        even if finish_reason wasn't set to 'length' by the API.
        """
        for tag in ClineAgent._TOOL_TAGS:
            open_tag = f'<{tag}>'
            close_tag = f'</{tag}>'
            if open_tag in content and close_tag not in content:
                return True
        return False

    async def _execute_tool(self, tool: ParsedToolCall) -> str:
        """Execute a tool call and return the result."""
        # Clear status line before any console output to prevent
        # ghost text from status line interleaving with tool output
        self.status.clear()
        
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
                
                # Thrash detection: track consecutive failures
                norm_path = str(self.tool_handlers._resolve_path(path))
                if result.startswith("Error:"):
                    entry = self._edit_failures.setdefault(norm_path, {"failures": 0, "last_error": ""})
                    entry["failures"] += 1
                    entry["last_error"] = result[:200]
                    n = entry["failures"]
                    if n >= 3:
                        self.console.print(f"[bold red]   ⚠ {n} consecutive edit failures on this file![/bold red]")
                        result += (
                            f"\n\n[REPEATED FAILURE — {n} consecutive failed edits on this file]\n"
                            f"You are stuck in a loop. STOP and try a DIFFERENT approach:\n"
                            f"  1. Re-read the file (read_file with start_line/end_line around the target area)\n"
                            f"  2. Copy the EXACT text from the file into your SEARCH block\n"
                            f"  3. If the file is badly broken, use write_to_file to rewrite the entire file\n"
                            f"  4. Make SMALLER edits — change only a few lines at a time\n"
                            f"Do NOT retry the same edit pattern. Change your approach."
                        )
                    elif n >= 2:
                        result += (
                            f"\n\nNote: This is the {n}nd consecutive failed edit on this file. "
                            f"Re-read the file to get the exact content before retrying."
                        )
                else:
                    # Success — reset failure counter
                    self._edit_failures.pop(norm_path, None)
                
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
                
            elif tool.name == "manage_todos":
                action = tool.parameters.get("action", "list")
                self.console.print(f"[blue]> Todo:[/blue] {action}")
                result = self._handle_manage_todos(tool.parameters)
                
            elif tool.name == "attempt_completion":
                self.console.print("[green]> Task complete[/green]")
                result = "Task completed."
                
            else:
                result = f"Unknown tool: {tool.name}"
            
            return result
            
        except Exception as e:
            self.console.print(f"[red][X] {tool.name}: {rich_escape(str(e))}[/red]")
            return f"Error: {str(e)}"
    
    def _handle_manage_todos(self, params: Dict[str, str]) -> str:
        """Handle the manage_todos tool."""
        action = params.get("action", "list").lower()
        
        if action == "add":
            title = params.get("title", "")
            if not title:
                return "Error: 'title' is required for add action."
            description = params.get("description", "")
            parent_id = None
            if params.get("parent_id"):
                try:
                    parent_id = int(params["parent_id"])
                except ValueError:
                    return f"Error: Invalid parent_id: {params['parent_id']}"
            context_refs = []
            if params.get("context_refs"):
                context_refs = [r.strip() for r in params["context_refs"].split(",") if r.strip()]
            
            item = self.todo_manager.add(
                title=title,
                description=description,
                parent_id=parent_id,
                context_refs=context_refs,
            )
            
            # If this is the first todo and we don't have an original request, set it
            if not self.todo_manager.original_request and len(self.todo_manager.list_all()) == 1:
                # Find the original user message
                for msg in self.messages:
                    if msg.role == "user" and not msg.content.startswith("["):
                        self.todo_manager.set_original_request(msg.content[:500])
                        break
            
            return f"Added todo [{item.id}]: {item.title}\n\n{self.todo_manager.format_list()}"
        
        elif action == "update":
            item_id_str = params.get("id", "")
            if not item_id_str:
                return "Error: 'id' is required for update action."
            try:
                item_id = int(item_id_str)
            except ValueError:
                return f"Error: Invalid id: {item_id_str}"
            
            context_refs = None
            if params.get("context_refs"):
                context_refs = [r.strip() for r in params["context_refs"].split(",") if r.strip()]
            
            parent_id_str = params.get("parent_id")
            
            item = self.todo_manager.update(
                item_id=item_id,
                title=params.get("title"),
                status=params.get("status"),
                description=params.get("description"),
                notes=params.get("notes"),
                context_refs=context_refs,
            )
            
            if not item:
                return f"Error: Todo [{item_id}] not found."
            
            return f"Updated todo [{item.id}]: {item.title} ({item.status.value})\n\n{self.todo_manager.format_list()}"
        
        elif action == "remove":
            item_id_str = params.get("id", "")
            if not item_id_str:
                return "Error: 'id' is required for remove action."
            try:
                item_id = int(item_id_str)
            except ValueError:
                return f"Error: Invalid id: {item_id_str}"
            
            if self.todo_manager.remove(item_id):
                return f"Removed todo [{item_id}].\n\n{self.todo_manager.format_list()}"
            else:
                return f"Error: Todo [{item_id}] not found."
        
        elif action == "list":
            return self.todo_manager.format_list()
        
        else:
            return f"Error: Unknown action '{action}'. Use add, update, remove, or list."
    
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