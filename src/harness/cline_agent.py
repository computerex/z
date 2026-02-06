"""Streaming agent using Cline-style XML tool format."""

import asyncio
import sys
import platform
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console

from .streaming_client import StreamingJSONClient, StreamingMessage
from .config import Config
from .prompts import get_system_prompt
from .cost_tracker import get_global_tracker
from .interrupt import is_interrupted, reset_interrupt, start_monitoring, stop_monitoring


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
        'attempt_completion'
    ]
    
    for tool_name in tool_names:
        pattern = rf'<{tool_name}>(.*?)</{tool_name}>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            inner = match.group(1)
            params = {}
            
            # Parse each parameter tag
            param_pattern = r'<(\w+)>(.*?)</\1>'
            for m in re.finditer(param_pattern, inner, re.DOTALL):
                name, value = m.groups()
                # Clean up value - strip outer newlines but preserve internal structure
                value = value.strip('\n')
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
    
    def save_session(self, path: str) -> None:
        """Save conversation history."""
        import json
        data = {
            "workspace": self.workspace_path,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load_session(self, path: str) -> bool:
        """Load conversation history."""
        import json
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.messages = [StreamingMessage(role=m["role"], content=m["content"]) for m in data["messages"]]
            self._initialized = True
            return True
        except:
            return False
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self._initialized = False
    
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
            
            for iteration in range(self.max_iterations):
                # Reset interrupt state for this iteration
                reset_interrupt()
                
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
                    print("\n⏹️  Interrupted")
                    # Save partial response to history
                    if full_content.strip():
                        self.messages.append(StreamingMessage(
                            role="assistant", 
                            content=full_content + "\n[interrupted by user]"
                        ))
                    return "[Interrupted - press Enter to continue or type new request]"
                
                full_content = response.content or full_content
                
                # Track usage
                if response.usage:
                    self.cost_tracker.record_call(
                        model=self.config.model,
                        input_tokens=response.usage.get("prompt_tokens", 0),
                        output_tokens=response.usage.get("completion_tokens", 0),
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
                
                # Add to history
                self.messages.append(StreamingMessage(role="assistant", content=full_content))
                self.messages.append(StreamingMessage(
                    role="user",
                    content=f"[{tool_call.name} result]\n{tool_result}"
                ))
            
            return "Max iterations reached."
    
    async def _execute_tool(self, tool: ParsedToolCall) -> str:
        """Execute a tool call and return the result."""
        print(f"🔧 {tool.name}", end="", flush=True)
        
        try:
            if tool.name == "read_file":
                result = await self._read_file(tool.parameters)
            elif tool.name == "write_to_file":
                result = await self._write_file(tool.parameters)
            elif tool.name == "replace_in_file":
                result = await self._replace_in_file(tool.parameters)
            elif tool.name == "execute_command":
                result = await self._execute_command(tool.parameters)
            elif tool.name == "list_files":
                result = await self._list_files(tool.parameters)
            elif tool.name == "search_files":
                result = await self._search_files(tool.parameters)
            elif tool.name == "attempt_completion":
                result = "Task completed."
            else:
                result = f"Unknown tool: {tool.name}"
            
            print(" ✓")
            return result
            
        except Exception as e:
            print(f" ✗ {e}")
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
        
        content = path.read_text(encoding="utf-8", errors="replace")
        
        # Add line numbers
        lines = content.splitlines()
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)
    
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
        command = params.get("command", "")
        
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace_path,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace")
            return output[:15000] or "(no output)"
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: Command timed out after 120s"
    
    async def _list_files(self, params: Dict[str, str]) -> str:
        path = self._resolve_path(params.get("path", "."))
        recursive = params.get("recursive", "").lower() == "true"
        
        if not path.exists():
            return f"Error: Directory not found: {path}"
        
        items = []
        try:
            if recursive:
                for p in sorted(path.rglob("*"))[:500]:
                    rel = p.relative_to(path)
                    suffix = "/" if p.is_dir() else ""
                    items.append(f"{rel}{suffix}")
            else:
                for p in sorted(path.iterdir())[:200]:
                    suffix = "/" if p.is_dir() else ""
                    items.append(f"{p.name}{suffix}")
        except PermissionError:
            return "Error: Permission denied"
        
        return "\n".join(items) or "(empty directory)"
    
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
        
        return "\n".join(results) or "(no matches)"


# Alias for backward compatibility
StreamingAgent = ClineAgent
