"""Streaming agent using Cline-style XML tool format."""

import asyncio
import sys
import platform
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from rich.console import Console

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


def parse_xml_tool(content: str) -> Optional[ParsedToolCall]:
    """Parse Cline-style XML tool call from content."""
    tool_names = [
        'read_file', 'write_to_file', 'replace_in_file',
        'execute_command', 'list_files', 'search_files',
        'check_background_process', 'stop_background_process', 'list_background_processes',
        'list_context', 'remove_from_context', 'analyze_image',
        'attempt_completion'
    ]
    
    for tool_name in tool_names:
        pattern = rf'<{tool_name}>(.*?)</{tool_name}>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            inner = match.group(1)
            params = {}
            
            # Parse each parameter tag - collect multiple values for same param
            param_pattern = r'<(\w+)>(.*?)</\1>'
            for m in re.finditer(param_pattern, inner, re.DOTALL):
                name, value = m.groups()
                # Clean up value - strip outer newlines but preserve internal structure
                value = value.strip('\n')
                # If param already exists, append as comma-separated
                if name in params:
                    params[name] = params[name] + "," + value
                else:
                    params[name] = value
            
            return ParsedToolCall(name=tool_name, parameters=params)
    
    return None


def parse_search_replace_blocks(diff: str) -> List[Tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from diff string."""
    blocks = []
    
    # Pattern: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
    pattern = r'<{7}\s*SEARCH\n(.*?)\n={7}\n(.*?)\n>{7}\s*REPLACE'
    
    for m in re.finditer(pattern, diff, re.DOTALL):
        search, replace = m.groups()
        blocks.append((search, replace))
    
    return blocks


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
        
        # Background processes: {id: {"proc": Process, "command": str, "started": float, "logs": list, "task": Task}}
        self._background_procs: Dict[int, dict] = {}
        self._next_bg_id = 1
    
    async def _background_log_reader(self, bg_id: int, proc: asyncio.subprocess.Process):
        """Continuously read output from a background process."""
        info = self._background_procs.get(bg_id)
        if not info:
            return
        
        try:
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    continue
                
                if not line:
                    break
                
                decoded = line.decode("utf-8", errors="replace").rstrip()
                # Keep last 500 lines
                info["logs"].append(decoded)
                if len(info["logs"]) > 500:
                    info["logs"] = info["logs"][-500:]
        except Exception:
            pass
    
    def cleanup_background_procs(self) -> None:
        """Terminate all background processes safely."""
        for pid, info in list(self._background_procs.items()):
            try:
                proc = info["proc"]
                if proc.returncode is None:
                    # On Windows, use taskkill to kill the entire process tree
                    # This prevents zombie multiprocessing children
                    if platform.system() == "Windows":
                        try:
                            os.system(f'taskkill /F /T /PID {proc.pid} >nul 2>&1')
                        except:
                            try:
                                proc.terminate()
                            except:
                                pass
                    else:
                        try:
                            proc.terminate()
                        except:
                            pass
                # Cancel the log reader task
                if "task" in info and info["task"]:
                    try:
                        info["task"].cancel()
                    except:
                        pass
            except Exception:
                pass
        self._background_procs.clear()
    
    def list_background_procs(self) -> List[dict]:
        """List all background processes with their status."""
        import time
        result = []
        for bg_id, info in self._background_procs.items():
            proc = info["proc"]
            elapsed = time.time() - info["started"]
            status = "running" if proc.returncode is None else f"exited ({proc.returncode})"
            result.append({
                "id": bg_id,
                "pid": proc.pid,
                "command": info["command"][:50],
                "elapsed": elapsed,
                "status": status
            })
        return result
    
    def save_session(self, path: str) -> None:
        """Save conversation history and context."""
        import json
        # Serialize context items
        context_items = []
        for item in self.context.list_items():
            context_items.append({
                "id": item.id,
                "type": item.type,
                "source": item.source,
                "content": item.content,
                "added_at": item.added_at,
                "line_range": item.line_range
            })
        
        data = {
            "workspace": self.workspace_path,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
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
                response = await client.chat_stream_raw(
                    messages=self.messages,
                    on_content=on_chunk,
                    check_interrupt=is_interrupted,
                )
                
                print()  # Newline after stream
                
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
                    # DEBUG: Check if API returns zeros
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
                result = await self._read_file(tool.parameters)
                lines = result.count('\n') + 1
                self.console.print(f"[dim]   ({lines} lines)[/dim]")
                
            elif tool.name == "write_to_file":
                path = tool.parameters.get("path", "")
                self.console.print(f"[green]> Writing:[/green] {path}")
                result = await self._write_file(tool.parameters)
                
            elif tool.name == "replace_in_file":
                path = tool.parameters.get("path", "")
                self.console.print(f"[yellow]✏️  Editing:[/yellow] {path}")
                result = await self._replace_in_file(tool.parameters)
                
            elif tool.name == "execute_command":
                # execute_command handles its own display
                result = await self._execute_command(tool.parameters)
                
            elif tool.name == "list_files":
                path = tool.parameters.get("path", "")
                self.console.print(f"[blue]> Listing:[/blue] {path}")
                result = await self._list_files(tool.parameters)
                count = len(result.splitlines())
                self.console.print(f"[dim]   ({count} items)[/dim]")
                
            elif tool.name == "search_files":
                regex = tool.parameters.get("regex", "")
                self.console.print(f"[magenta]> Searching:[/magenta] {regex}")
                result = await self._search_files(tool.parameters)
                matches = len(result.splitlines()) if result != "(no matches)" else 0
                self.console.print(f"[dim]   ({matches} matches)[/dim]")
                
            elif tool.name == "check_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(f"[cyan]> Checking background process:[/cyan] {bg_id or 'all'}")
                result = await self._check_background_process(tool.parameters)
                
            elif tool.name == "stop_background_process":
                bg_id = tool.parameters.get("id", "")
                self.console.print(f"[red]> Stopping background process:[/red] {bg_id}")
                result = await self._stop_background_process(tool.parameters)
                
            elif tool.name == "list_background_processes":
                self.console.print(f"[cyan]> Listing background processes[/cyan]")
                result = await self._list_background_processes(tool.parameters)
                
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
                result = await self._analyze_image(tool.parameters)
                
            elif tool.name == "attempt_completion":
                self.console.print("[green]> Task complete[/green]")
                result = "Task completed."
                
            else:
                result = f"Unknown tool: {tool.name}"
            
            return result
            
        except Exception as e:
            self.console.print(f"[red][X] {tool.name}: {e}[/red]")
            return f"Error: {str(e)}"
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.workspace_path) / p
        return p.resolve()
    
    async def _read_file(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        
        if not path.exists():
            return f"Error: File not found: {path}"
        
        rel_path = str(path.relative_to(self.workspace_path)) if str(path).startswith(self.workspace_path) else str(path)
        
        # Check for duplicate reads - replace older ones with notice
        prev_index = self._duplicate_detector.was_read_before(rel_path)
        if prev_index is not None:
            replaced = DuplicateDetector.replace_old_reads(self.messages, rel_path, len(self.messages))
            if replaced > 0:
                self.console.print(f"[dim]   (replaced {replaced} older read(s) with notice)[/dim]")
        
        # Record this read
        self._duplicate_detector.record_read(rel_path, len(self.messages))
        
        content = path.read_text(encoding="utf-8", errors="replace")
        
        # Truncate if file is too large
        content = truncate_file_content(content)
        
        # Add line numbers
        lines = content.splitlines()
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        result = "\n".join(numbered)
        
        # Add to context container
        ctx_id = self.context.add("file", rel_path, result)
        
        return f"[Context ID: {ctx_id}]\n{result}"
    
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
    
    async def _write_file(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        content = params.get("content", "")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        return f"Successfully wrote to {path}"
    
    async def _replace_in_file(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", ""))
        diff = params.get("diff", "")
        
        if not path.exists():
            return f"Error: File not found: {path}"
        
        content = path.read_text(encoding="utf-8")
        blocks = parse_search_replace_blocks(diff)
        
        if not blocks:
            return "Error: No valid SEARCH/REPLACE blocks found"
        
        changes = 0
        for search, replace in blocks:
            if search in content:
                content = content.replace(search, replace, 1)
                changes += 1
            else:
                return f"Error: SEARCH block not found in file:\n{search[:200]}..."
        
        path.write_text(content, encoding="utf-8")
        return f"Successfully made {changes} replacement(s) in {path}"
    
    async def _execute_command(self, params: Dict[str, str]) -> str:
        """Execute a shell command with live output display and interrupt support."""
        import time
        import subprocess
        command = params.get("command", "")
        background = params.get("background", "").lower() == "true"
        timeout_secs = 120  # Auto-background after this many seconds
        
        # Show command being executed
        print()
        mode_indicator = "[bg] " if background else ""
        self.console.print(f"[dim]$ {mode_indicator}{command}[/dim]")
        
        if background:
            return await self._run_background_command(command)
        
        # Windows: create new process group to isolate from child process chaos
        # This prevents multiprocessing fork bombs from taking down the harness
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        # Foreground execution with interrupt/background/timeout support
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.workspace_path,
            creationflags=creationflags,
        )
        
        output_lines = []
        start_time = time.time()
        hint_shown = False
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                # Check for interrupt (Esc)
                if is_interrupted():
                    proc.terminate()
                    self.console.print(f"\n[yellow][STOP] Command interrupted[/yellow]")
                    output = "\n".join(output_lines)[:15000]
                    return f"Command interrupted after {elapsed:.0f}s.\nOutput captured:\n{output}" if output else "Command interrupted (no output)"
                
                # Check for background request (Ctrl+B)
                if is_background_requested():
                    self.console.print(f"\n[cyan]-> Sending to background...[/cyan]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    output = "\n".join(output_lines[-30:])
                    return f"Command sent to background (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nOutput so far:\n{output}"
                
                # Show hint after 5 seconds
                if elapsed > 5 and not hint_shown:
                    self.console.print(f"[dim]  (Ctrl+B to background, Esc to stop)[/dim]")
                    hint_shown = True
                
                # Auto-background after timeout
                if elapsed > timeout_secs:
                    self.console.print(f"\n[yellow][TIME] Command running for {timeout_secs}s - auto-backgrounding[/yellow]")
                    proc_id = self._next_bg_id
                    self._next_bg_id += 1
                    self._background_procs[proc_id] = {
                        "proc": proc, 
                        "command": command, 
                        "started": start_time,
                        "logs": output_lines.copy(),
                        "task": None
                    }
                    task = asyncio.create_task(self._background_log_reader(proc_id, proc))
                    self._background_procs[proc_id]["task"] = task
                    self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
                    output = "\n".join(output_lines)[:15000]
                    return f"Command auto-backgrounded after {timeout_secs}s (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nOutput captured:\n{output}" if output else f"Command auto-backgrounded (ID: {proc_id}, PID: {proc.pid}, no output yet)"
                
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                    
                if not line:
                    break
                    
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                self.console.print(f"[dim]  {decoded}[/dim]")
            
            await proc.wait()
            exit_code = proc.returncode
            
            if exit_code == 0:
                self.console.print(f"[green][OK] Exit code: {exit_code}[/green]")
            else:
                self.console.print(f"[red][X] Exit code: {exit_code}[/red]")
            
            # Truncate long output (keep start and end)
            raw_output = "\n".join(output_lines) or "(no output)"
            output = truncate_output(raw_output, max_lines=200, keep_start=50, keep_end=50)
            
            # Add to context if significant output
            if len(output_lines) > 3:
                ctx_id = self.context.add("command_output", command, output)
                return f"[Context ID: {ctx_id}]\n{output}"
            return output
            
        except Exception as e:
            proc.kill()
            output = truncate_output("\n".join(output_lines), max_lines=150)
            return f"Error: {str(e)}\nOutput captured:\n{output}" if output else f"Error: {str(e)}"
    
    async def _run_background_command(self, command: str) -> str:
        """Run a command in background."""
        import time
        import subprocess
        
        # Windows: create new process group to isolate from child process chaos
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.workspace_path,
            creationflags=creationflags,
        )
        
        # Store background process with log buffer
        proc_id = self._next_bg_id
        self._next_bg_id += 1
        
        # Wait briefly to capture initial output
        output_lines = []
        try:
            for _ in range(20):
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.1)
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(decoded)
                self.console.print(f"[dim]  {decoded}[/dim]")
        except asyncio.TimeoutError:
            pass
        
        # Store process and start log reader
        self._background_procs[proc_id] = {
            "proc": proc, 
            "command": command, 
            "started": time.time(),
            "logs": output_lines.copy(),
            "task": None
        }
        # Start background log reader
        task = asyncio.create_task(self._background_log_reader(proc_id, proc))
        self._background_procs[proc_id]["task"] = task
        
        self.console.print(f"[green]-> Running in background (ID: {proc_id}, PID: {proc.pid})[/green]")
        return f"Command started in background (ID: {proc_id}, PID: {proc.pid}).\nUse check_background_process to see logs.\nInitial output:\n" + "\n".join(output_lines[-10:])
    
    async def _list_files(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", "."))
        recursive = params.get("recursive", "").lower() == "true"
        
        if not path.exists():
            return f"Error: Directory not found: {path}"
        
        items = []
        truncated = False
        max_items = 100 if recursive else 50  # Smaller limits
        
        try:
            if recursive:
                for p in sorted(path.rglob("*")):
                    if len(items) >= max_items:
                        truncated = True
                        break
                    # Skip common junk directories
                    parts = p.parts
                    if any(skip in parts for skip in ['node_modules', '.git', '__pycache__', '.venv', 'venv', 'dist', 'build']):
                        continue
                    rel = p.relative_to(path)
                    suffix = "/" if p.is_dir() else ""
                    items.append(f"{rel}{suffix}")
            else:
                for p in sorted(path.iterdir())[:max_items]:
                    suffix = "/" if p.is_dir() else ""
                    items.append(f"{p.name}{suffix}")
                if len(list(path.iterdir())) > max_items:
                    truncated = True
        except PermissionError:
            return "Error: Permission denied"
        
        result = "\n".join(items) or "(empty directory)"
        if truncated:
            result += f"\n\n... (truncated at {max_items} items, use more specific path)"
        
        return result
    
    async def _search_files(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", "."))
        regex = params.get("regex", "")
        file_pattern = params.get("file_pattern", "*")
        
        if not path.exists():
            return f"Error: Directory not found: {path}"
        
        try:
            pattern = re.compile(regex, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex: {e}"
        
        results = []
        for file in path.rglob(file_pattern):
            if file.is_file():
                try:
                    content = file.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            rel = file.relative_to(path)
                            results.append(f"{rel}:{i}: {line[:150]}")
                            if len(results) >= 100:
                                break
                except:
                    pass
            if len(results) >= 100:
                break
        
        if not results:
            return "(no matches)"
        
        result = "\n".join(results)
        # Add to context if significant results
        if len(results) > 5:
            ctx_id = self.context.add("search_result", regex, result)
            return f"[Context ID: {ctx_id}]\n{result}"
        return result
    
    async def _check_background_process(self, params: Dict[str, str]) -> str:
        """Check status and logs of a background process."""
        import time
        bg_id_str = params.get("id", "")
        lines = int(params.get("lines", "50"))
        
        try:
            bg_id = int(bg_id_str)
        except ValueError:
            # List all if no ID given
            procs = self.list_background_procs()
            if not procs:
                return "No background processes running."
            result = "Background processes:\n"
            for p in procs:
                elapsed_min = p['elapsed'] / 60
                result += f"  [{p['id']}] PID {p['pid']} - {p['status']} - {elapsed_min:.1f}m - {p['command']}\n"
            result += "\nUse check_background_process with id parameter to see logs."
            return result
        
        if bg_id not in self._background_procs:
            return f"Error: No background process with ID {bg_id}"
        
        info = self._background_procs[bg_id]
        proc = info["proc"]
        elapsed = time.time() - info["started"]
        status = "running" if proc.returncode is None else f"exited (code {proc.returncode})"
        logs = info.get("logs", [])
        
        # Get last N lines
        recent_logs = logs[-lines:] if logs else []
        
        result = f"Background Process [{bg_id}]\n"
        result += f"Command: {info['command']}\n"
        result += f"PID: {proc.pid}\n"
        result += f"Status: {status}\n"
        result += f"Running time: {elapsed:.0f}s\n"
        result += f"Total log lines: {len(logs)}\n"
        result += f"\n--- Last {len(recent_logs)} lines ---\n"
        result += "\n".join(recent_logs) if recent_logs else "(no output yet)"
        
        # Add guidance to prevent spam checking
        if proc.returncode is None:
            if not recent_logs or len(logs) == info.get('_last_check_lines', 0):
                result += "\n\n[!] Process still running with no new output. Continue with other tasks instead of re-checking immediately."
            info['_last_check_lines'] = len(logs)
        
        return result
    
    async def _stop_background_process(self, params: Dict[str, str]) -> str:
        """Stop a background process by ID."""
        bg_id_str = params.get("id", "")
        
        try:
            bg_id = int(bg_id_str)
        except ValueError:
            return "Error: ID must be a number"
        
        if bg_id not in self._background_procs:
            return f"Error: No background process with ID {bg_id}"
        
        info = self._background_procs[bg_id]
        proc = info["proc"]
        
        if proc.returncode is not None:
            return f"Process [{bg_id}] already exited with code {proc.returncode}"
        
        try:
            proc.terminate()
            # Wait briefly for graceful termination
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
        
        # Cancel log reader
        if "task" in info and info["task"]:
            info["task"].cancel()
        
        return f"Stopped background process [{bg_id}] (PID: {proc.pid})"
    
    async def _list_background_processes(self, params: Dict[str, str]) -> str:
        """List all background processes."""
        procs = self.list_background_procs()
        if not procs:
            return "No background processes."
        
        result = "Background processes:\n"
        for p in procs:
            elapsed_min = p['elapsed'] / 60
            result += f"  [{p['id']}] PID {p['pid']} - {p['status']} - {elapsed_min:.1f}m - {p['command']}\n"
        return result
    
    async def _analyze_image(self, params: Dict[str, str]) -> str:
        """Analyze an image using GLM-4.6V vision model via coding endpoint."""
        import base64
        import httpx
        from urllib.parse import urlparse
        
        path_str = params.get("path", "")
        question = params.get("question", "Describe this image in detail. Note any text, UI elements, errors, or important visual details.")
        
        path = self._resolve_path(path_str)
        if not path.exists():
            return f"Error: Image not found: {path}"
        
        # Check file extension
        ext = path.suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return f"Error: Unsupported image format: {ext}. Use jpg, png, gif, or webp."
        
        # Read and encode image as base64
        try:
            img_data = path.read_bytes()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            return f"Error reading image: {e}"
        
        # Determine mime type
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', 
                    '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/png')
        
        # Use the Coding endpoint which properly supports vision with base64
        # https://api.z.ai/api/coding/paas/v4/chat/completions
        parsed = urlparse(self.config.api_url)
        vision_url = f"{parsed.scheme}://{parsed.netloc}/api/coding/paas/v4/chat/completions"
        
        # OpenAI format with data URI base64
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
        
        payload = {
            "model": "glm-4.6v",  # Vision model
            "messages": vision_messages,
            "max_tokens": 2048,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(vision_url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract OpenAI response format
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content:
                        # Add to context
                        ctx_id = self.context.add("image_analysis", str(path), content)
                        return f"[Context ID: {ctx_id}]\n\nImage: {path_str}\n\n{content}"
                    return "Vision model returned empty response."
                return f"Unexpected response format: {data}"
                
        except httpx.HTTPStatusError as e:
            return f"Error calling vision API: {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            return f"Error analyzing image: {e}"


# Alias for backward compatibility
StreamingAgent = ClineAgent
