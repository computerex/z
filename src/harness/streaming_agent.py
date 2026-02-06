"""Streaming agent using JSON response format for true token-by-token streaming."""

import json
import asyncio
import sys
import time
import platform
import shutil
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel

from .streaming_client import StreamingJSONClient, StreamingMessage, StreamingChatResponse
from .config import Config
from .tools import file_tools, shell_tools, search_tools
from .cost_tracker import CostTracker, get_global_tracker


def get_tools_prompt(workspace_path: str) -> str:
    """Generate system prompt with tools defined inline."""
    os_name = platform.system()
    os_version = platform.release()
    
    if os_name == 'Windows':
        shell_name = 'PowerShell'
    else:
        shell_name = os.path.basename(os.environ.get('SHELL', 'bash'))
    
    return f"""You are an expert AI coding assistant. You MUST always respond with valid JSON.

ENVIRONMENT:
  Platform: {os_name} {os_version}
  Shell: {shell_name}
  Working Directory: {workspace_path}

AVAILABLE TOOLS:

1. write_file - Create a NEW file only. DO NOT use for existing files!
   Parameters: {{"file_path": "absolute path", "content": "file content"}}

2. read_file - Read content from a file
   Parameters: {{"file_path": "absolute path", "start_line": optional int, "end_line": optional int}}

3. edit_file - REQUIRED for modifying existing files. Replace a specific chunk of text.
   Parameters: {{"file_path": "absolute path", "old_string": "exact text to find (include 3-5 lines context)", "new_string": "replacement text"}}
   IMPORTANT: Only emit the chunk being changed, NOT the whole file. old_string must be unique in the file.
   For multiple changes, call edit_file multiple times - one change per call.

4. list_directory - List contents of a directory
   Parameters: {{"directory_path": "absolute path", "recursive": optional bool, "pattern": optional glob}}

5. file_search - Search for files matching a glob pattern
   Parameters: {{"directory": "root dir", "pattern": "glob like **/*.py", "max_results": optional int}}

6. grep_search - Search file contents with regex
   Parameters: {{"directory": "root dir", "pattern": "regex pattern", "file_pattern": optional glob}}

7. run_command - Execute a shell command
   Parameters: {{"command": "shell command", "working_directory": optional path}}

RESPONSE FORMAT - Always respond with this exact JSON structure:
{{
  "thinking": "Brief reasoning about what to do (1-2 sentences)",
  "tool": "tool_name" or null,
  "parameters": {{"param": "value"}} or null,
  "message": "Message to user, or empty if using a tool"
}}

RULES:
- ALWAYS use absolute paths starting with {workspace_path}
- For NEW files: use write_file with full content
- For EXISTING files: ALWAYS use edit_file, NEVER write_file. Only output the changed chunk.
- You can make multiple edit_file calls for multiple changes - this is more efficient than rewriting
- When done with a task, set tool to null and put your response in message
- One tool call per response
- After a tool result, continue working or provide final message
"""


class StreamingAgent:
    """Agent with true streaming using JSON response format."""
    
    def __init__(
        self,
        config: Config,
        console: Optional[Console] = None,
        max_iterations: int = 20,
    ):
        self.config = config
        self.console = console or Console()
        self.max_iterations = max_iterations
        self.workspace_path = str(Path.cwd().resolve())
        self.cost_tracker = get_global_tracker()
        
        # Conversation history for session persistence
        self.messages: List[StreamingMessage] = []
        self._initialized = False
        
        # Tool implementations
        self.tools = {
            "write_file": self._tool_write_file,
            "read_file": self._tool_read_file,
            "edit_file": self._tool_edit_file,
            "list_directory": self._tool_list_directory,
            "file_search": self._tool_file_search,
            "grep_search": self._tool_grep_search,
            "run_command": self._tool_run_command,
        }
    
    def save_session(self, path: str) -> None:
        """Save conversation history to a file."""
        import json
        data = {
            "workspace": self.workspace_path,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load_session(self, path: str) -> bool:
        """Load conversation history from a file."""
        import json
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.messages = [StreamingMessage(role=m["role"], content=m["content"]) for m in data["messages"]]
            self._initialized = True
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self._initialized = False
    
    async def run(self, user_input: str) -> str:
        """Run the agent with streaming output. Maintains conversation history."""
        # Initialize system prompt if first run
        if not self._initialized:
            self.messages = [
                StreamingMessage(role="system", content=get_tools_prompt(self.workspace_path)),
            ]
            self._initialized = True
        
        # Add user's new message
        self.messages.append(StreamingMessage(role="user", content=user_input))
        
        async with StreamingJSONClient(
            api_key=self.config.api_key,
            base_url=self.config.api_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ) as client:
            
            for iteration in range(self.max_iterations):
                print("⏳ ", end="", flush=True)
                
                def print_char(c):
                    sys.stdout.write(c)
                    sys.stdout.flush()
                
                response = await client.chat_stream(
                    messages=self.messages,
                    on_content=print_char,
                )
                
                print()  # Newline after streaming
                
                # Handle truncation - continue generating
                accumulated_json = response.raw_json
                while response.is_truncated:
                    print("⏳ (continuing...)", end="", flush=True)
                    
                    # Ask to continue
                    continuation_messages = self.messages + [
                        StreamingMessage(role="assistant", content=accumulated_json),
                        StreamingMessage(role="user", content="Continue from where you stopped. Output ONLY the remaining JSON, nothing else.")
                    ]
                    
                    response = await client.chat_stream(
                        messages=continuation_messages,
                        on_content=print_char,
                    )
                    print()
                    accumulated_json += response.raw_json
                
                # Re-parse the accumulated JSON if we had continuations
                if accumulated_json != response.raw_json:
                    response.raw_json = accumulated_json
                    try:
                        import json
                        parsed = json.loads(accumulated_json)
                        response.thinking = parsed.get("thinking")
                        response.message = parsed.get("message")
                        tool_name = parsed.get("tool")
                        if tool_name and tool_name != "null" and parsed.get("parameters"):
                            from .streaming_client import StreamingToolCall
                            response.tool_call = StreamingToolCall(
                                name=tool_name,
                                parameters=parsed["parameters"]
                            )
                    except json.JSONDecodeError:
                        print("⚠️  Failed to parse accumulated JSON")
                
                # Track usage
                if response.usage:
                    self.cost_tracker.record_call(
                        model=self.config.model,
                        input_tokens=response.usage.get("prompt_tokens", 0),
                        output_tokens=response.usage.get("completion_tokens", 0),
                        duration_ms=0,
                        tool_calls=1 if response.has_tool_call else 0,
                        finish_reason="tool_calls" if response.has_tool_call else "stop",
                    )
                
                # No tool call - final response
                if not response.has_tool_call:
                    # Add assistant's final response to history
                    self.messages.append(StreamingMessage(
                        role="assistant",
                        content=response.raw_json
                    ))
                    return response.message or ""
                
                # Execute tool
                tool_name = response.tool_call.name
                tool_params = response.tool_call.parameters
                
                tool_fn = self.tools.get(tool_name)
                if not tool_fn:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        result = await tool_fn(tool_params)
                        print(f"✅ {tool_name}")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        print(f"❌ {tool_name}: {e}")
                
                # Add assistant response and tool result to history
                self.messages.append(StreamingMessage(
                    role="assistant",
                    content=response.raw_json
                ))
                self.messages.append(StreamingMessage(
                    role="user",
                    content=f"Tool result for {tool_name}:\n{result}"
                ))
            
            return "Max iterations reached."
    
    # Tool implementations
    async def _tool_write_file(self, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path", "")
        content = params.get("content", "")
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        return f"Created {file_path} ({len(content)} chars)"
    
    async def _tool_read_file(self, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path", "")
        start_line = params.get("start_line")
        end_line = params.get("end_line")
        
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        content = path.read_text(encoding="utf-8")
        
        if start_line or end_line:
            lines = content.splitlines()
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            content = "\n".join(lines[start:end])
        
        return content
    
    async def _tool_edit_file(self, params: Dict[str, Any]) -> str:
        file_path = params.get("file_path", "")
        old_string = params.get("old_string", "")
        new_string = params.get("new_string", "")
        
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        content = path.read_text(encoding="utf-8")
        
        if old_string not in content:
            return f"Error: String not found in file"
        
        count = content.count(old_string)
        if count > 1:
            return f"Error: String found {count} times, must be unique"
        
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")
        
        return f"Edited {file_path}"
    
    async def _tool_list_directory(self, params: Dict[str, Any]) -> str:
        directory = params.get("directory_path", "")
        recursive = params.get("recursive", False)
        pattern = params.get("pattern", "*")
        
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        
        if recursive:
            items = list(path.rglob(pattern))
        else:
            items = list(path.glob(pattern))
        
        result = []
        for item in sorted(items)[:100]:
            rel = item.relative_to(path)
            suffix = "/" if item.is_dir() else ""
            result.append(f"{rel}{suffix}")
        
        return "\n".join(result) or "(empty)"
    
    async def _tool_file_search(self, params: Dict[str, Any]) -> str:
        directory = params.get("directory", "")
        pattern = params.get("pattern", "*")
        max_results = params.get("max_results", 100)
        
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        
        matches = list(path.rglob(pattern))[:max_results]
        return "\n".join(str(m) for m in matches) or "(no matches)"
    
    async def _tool_grep_search(self, params: Dict[str, Any]) -> str:
        import re
        
        directory = params.get("directory", "")
        pattern = params.get("pattern", "")
        file_pattern = params.get("file_pattern", "*")
        
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        for file in path.rglob(file_pattern):
            if file.is_file():
                try:
                    content = file.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{file}:{i}: {line[:100]}")
                            if len(results) >= 50:
                                break
                except:
                    pass
            if len(results) >= 50:
                break
        
        return "\n".join(results) or "(no matches)"
    
    async def _tool_run_command(self, params: Dict[str, Any]) -> str:
        command = params.get("command", "")
        working_dir = params.get("working_directory", self.workspace_path)
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            output = stdout.decode("utf-8", errors="ignore")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="ignore")
            
            return output[:10000] or "(no output)"
        except asyncio.TimeoutError:
            return "Error: Command timed out after 60s"
        except Exception as e:
            return f"Error: {str(e)}"
