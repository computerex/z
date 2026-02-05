"""Main agent implementation for the coding harness."""

import json
import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

from .config import Config
from .llm_client import LLMClient, Message, ChatResponse
from .context import ContextManager
from .tools.registry import ToolRegistry, Tool, ToolResult
from .tools import file_tools, shell_tools, search_tools


def get_system_prompt(workspace_path: str) -> str:
    """Generate system prompt with platform and shell info."""
    import platform
    import os
    import shutil
    
    # Detect platform
    os_name = platform.system()  # Windows, Linux, Darwin
    os_version = platform.release()
    
    # Detect shell
    shell = os.environ.get('SHELL', '')
    comspec = os.environ.get('COMSPEC', '')
    
    if os_name == 'Windows':
        # Check for PowerShell vs CMD
        if 'powershell' in os.environ.get('PSModulePath', '').lower() or shutil.which('pwsh'):
            shell_name = 'PowerShell'
            shell_examples = """
Shell Examples (PowerShell):
  - List files: Get-ChildItem or ls
  - Current directory: Get-Location or pwd
  - Change directory: Set-Location path or cd path
  - Create directory: New-Item -ItemType Directory -Path "name"
  - Remove file: Remove-Item file.txt
  - Copy file: Copy-Item src.txt dest.txt
  - Environment variable: $env:VARNAME
  - Run executable: & "path\\to\\exe"
  - Path separator: Use backslash (\\) for paths"""
        else:
            shell_name = 'CMD'
            shell_examples = """
Shell Examples (CMD):
  - List files: dir
  - Current directory: cd
  - Change directory: cd path
  - Create directory: mkdir name
  - Remove file: del file.txt
  - Copy file: copy src.txt dest.txt
  - Environment variable: %VARNAME%
  - Path separator: Use backslash (\\) for paths"""
    else:
        shell_name = os.path.basename(shell) if shell else 'bash'
        shell_examples = """
Shell Examples (bash/zsh):
  - List files: ls -la
  - Current directory: pwd
  - Change directory: cd path
  - Create directory: mkdir -p name
  - Remove file: rm file.txt
  - Copy file: cp src.txt dest.txt
  - Environment variable: $VARNAME
  - Path separator: Use forward slash (/) for paths"""
    
    return f"""You are an expert AI coding assistant with access to tools for file operations, code editing, shell commands, and code search.

ENVIRONMENT:
  Platform: {os_name} {os_version}
  Shell: {shell_name}
  Working Directory: {workspace_path}

CRITICAL: Always create files in the working directory ({workspace_path}) unless the user specifies a different location. 
NEVER use /tmp or other Unix paths on Windows. Always use the working directory as the base for relative paths.

{shell_examples}

When working on coding tasks:
1. First understand the request and gather context using search and file reading tools.
2. Plan your approach before making changes.
3. Make precise, targeted edits using the edit_file tool.
4. Verify your changes work by reading the results or running tests.
5. Report clearly what you did and any issues encountered.

Always use tools when needed - don't just describe what you would do. Execute the actions.

For file edits, always include enough context (3-5 lines before and after) to uniquely identify the location.
When running shell commands, use {shell_name}-compatible syntax and prefer full paths.
Create files in the working directory: {workspace_path}
"""


class Agent:
    """Agentic coding assistant with tool calling capabilities."""
    
    def __init__(
        self,
        config: Config,
        console: Optional[Console] = None,
        max_iterations: int = 20,
    ):
        self.config = config
        self.console = console or Console()
        self.max_iterations = max_iterations
        
        self.context = ContextManager(max_tokens=config.max_context_tokens)
        self.tools = ToolRegistry()
        self._client: Optional[LLMClient] = None
        
        # Register default tools
        self._register_default_tools()
    
    async def _chat_with_streaming(self, client: LLMClient, messages, tools) -> ChatResponse:
        """Make API call with streaming - show content as it arrives."""
        import sys
        
        start_time = time.time()
        
        print("⏳ Waiting for response...", flush=True)
        
        response = await client.chat_stream(
            messages=messages,
            tools=tools,
            on_content=lambda c: print(c, end="", flush=True),
            on_tool_start=lambda name: print(f"\n📝 Generating {name}...", flush=True),
            on_tool_content=lambda c: print(c, end="", flush=True),
        )
        
        elapsed = int(time.time() - start_time)
        print(f"\n✓ {elapsed}s", flush=True)
        
        return response
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        
        # File reading
        self.tools.register(Tool(
            name="read_file",
            description="Read content from a file. Can optionally read specific line ranges.",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based start line number.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-based end line number (inclusive).",
                },
            },
            function=file_tools.read_file,
            required_params=["file_path"],
        ))
        
        # File writing
        self.tools.register(Tool(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed.",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            function=file_tools.write_file,
            required_params=["file_path", "content"],
        ))
        
        # File editing
        self.tools.register(Tool(
            name="edit_file",
            description="Edit a file by replacing a specific string. The old_string must match exactly and uniquely. Include 3-5 lines of context before and after the target text.",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to replace (include context for unique matching).",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string.",
                },
            },
            function=file_tools.edit_file,
            required_params=["file_path", "old_string", "new_string"],
        ))
        
        # Directory listing
        self.tools.register(Tool(
            name="list_directory",
            description="List contents of a directory.",
            parameters={
                "directory_path": {
                    "type": "string",
                    "description": "Absolute path to the directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter results.",
                },
            },
            function=file_tools.list_directory,
            required_params=["directory_path"],
        ))
        
        # File search
        self.tools.register(Tool(
            name="file_search",
            description="Search for files matching a glob pattern.",
            parameters={
                "directory": {
                    "type": "string",
                    "description": "Root directory to search from.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py' for all Python files).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 100).",
                },
            },
            function=file_tools.file_search,
            required_params=["directory", "pattern"],
        ))
        
        # Shell commands
        self.tools.register(Tool(
            name="run_command",
            description="Execute a shell command and return the output.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 60).",
                },
            },
            function=shell_tools.run_shell_command,
            required_params=["command"],
        ))
        
        # Lexical search
        self.tools.register(Tool(
            name="lexical_search",
            description="Search for text patterns in files using exact or regex matching.",
            parameters={
                "directory": {
                    "type": "string",
                    "description": "Root directory to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (text or regex pattern).",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter (e.g., '*.py'). Default: '*'.",
                },
                "is_regex": {
                    "type": "boolean",
                    "description": "Whether query is a regex pattern (default: false).",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive (default: false).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default: 50).",
                },
            },
            function=search_tools.lexical_search,
            required_params=["directory", "query"],
        ))
        
        # Semantic search
        self.tools.register(Tool(
            name="semantic_search",
            description="Search for code using natural language. Finds semantically similar code chunks.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search (will index if needed).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results (default: 10).",
                },
                "rebuild_index": {
                    "type": "boolean",
                    "description": "Force rebuild of search index (default: false).",
                },
            },
            function=search_tools.semantic_search,
            required_params=["query"],
        ))
    
    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool call."""
        tool_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        
        # Try to repair truncated JSON for write_file
        if tool_name == "write_file" and arguments_str:
            # If JSON is truncated, try to extract file_path and content
            if not arguments_str.rstrip().endswith("}"):
                print(f"  ⚠️  Truncated JSON detected ({len(arguments_str)} chars), attempting repair...", flush=True)
                try:
                    # Extract file_path
                    import re
                    fp_match = re.search(r'"file_path"\s*:\s*"([^"]+)"', arguments_str)
                    file_path = fp_match.group(1) if fp_match else None
                    
                    # Extract content - find start and take everything until last valid point
                    content_match = re.search(r'"content"\s*:\s*"', arguments_str)
                    if content_match and file_path:
                        content_start = content_match.end()
                        # Get content, handling escape sequences
                        raw_content = arguments_str[content_start:]
                        # Remove trailing incomplete escape or quote
                        if raw_content.endswith('\\'):
                            raw_content = raw_content[:-1]
                        # Unescape
                        content = raw_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                        arguments = {"file_path": file_path, "content": content}
                        print(f"  ✓ Repaired: {file_path} ({len(content)} chars)", flush=True)
                except Exception as repair_err:
                    print(f"  ✗ Repair failed: {repair_err}", flush=True)
                    arguments = {}
            else:
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}
        else:
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                # Show what we got for debugging
                preview = arguments_str[:200] + "..." if len(arguments_str) > 200 else arguments_str
                print(f"  ⚠️  JSON parse error: {e}", flush=True)
                print(f"  ⚠️  Raw args ({len(arguments_str)} chars): {preview}", flush=True)
                return ToolResult(
                    success=False,
                    error=f"Failed to parse tool arguments: {e}\nRaw: {preview}",
                )
        
        # Check if we have required args
        if tool_name == "write_file" and (not arguments.get("file_path") or not arguments.get("content")):
            return ToolResult(
                success=False,
                error=f"write_file missing required arguments. Got: {list(arguments.keys())}",
            )
        
        # Show what tool is being called with key info
        if tool_name == "write_file":
            file_path = arguments.get("file_path", "?")
            content_len = len(arguments.get("content", ""))
            print(f"  📝 Writing: {file_path} ({content_len} chars)", flush=True)
        elif tool_name == "read_file":
            print(f"  📖 Reading: {arguments.get('file_path', '?')}", flush=True)
        elif tool_name == "edit_file":
            print(f"  ✏️  Editing: {arguments.get('file_path', '?')}", flush=True)
        elif tool_name == "run_command":
            cmd = arguments.get("command", "?")
            print(f"  💻 Running: {cmd[:60]}...", flush=True) if len(cmd) > 60 else print(f"  💻 Running: {cmd}", flush=True)
        elif tool_name == "list_directory":
            print(f"  📁 Listing: {arguments.get('directory_path', '?')}", flush=True)
        else:
            print(f"  🔧 {tool_name}", flush=True)
        
        result = await self.tools.execute(tool_name, **arguments)
        
        if result.success:
            # Show brief result
            output = str(result.output) if result.output else ""
            if len(output) > 100:
                output = output[:100] + "..."
            print(f"     ✓ {output if output else 'Done'}", flush=True)
        else:
            print(f"     ✗ {result.error}", flush=True)
        
        return result
    
    async def _process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Message]:
        """Process multiple tool calls and return result messages."""
        results = []
        
        for tool_call in tool_calls:
            call_id = tool_call.get("id", "")
            tool_name = tool_call.get("function", {}).get("name", "")
            
            result = await self._execute_tool_call(tool_call)
            
            # Add result to context
            result_message = result.to_message()
            
            # Truncate very long results
            if len(result_message) > 10000:
                result_message = result_message[:5000] + "\n...[truncated]...\n" + result_message[-5000:]
            
            results.append(Message(
                role="tool",
                content=result_message,
                tool_call_id=call_id,
                name=tool_name,
            ))
        
        return results
    
    async def run(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run the agent with a user message.
        
        Args:
            user_message: The user's request.
            system_prompt: Optional custom system prompt.
        
        Returns:
            The agent's final response.
        """
        self.config.validate()
        
        # Initialize context with dynamic system prompt including CWD and shell info
        default_prompt = get_system_prompt(str(self.config.workspace_path))
        self.context.initialize(system_prompt or default_prompt)
        self.context.add_user_turn(user_message)
        
        self.console.print(Panel(user_message, title="User Request", border_style="blue"))
        
        async with LLMClient(self.config) as client:
            self._client = client
            
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                self.console.print(f"\n[bold cyan]--- Iteration {iteration} ---[/bold cyan]")
                
                # Compact context if needed
                freed = self.context.compact_if_needed()
                if freed > 0:
                    self.console.print(f"[dim]Compacted context, freed {freed} tokens[/dim]")
                
                # Get LLM response with streaming
                try:
                    response = await self._chat_with_streaming(
                        client,
                        messages=self.context.get_messages(),
                        tools=self.tools.to_openai_schema(),
                    )
                except asyncio.TimeoutError:
                    self.console.print(f"[red]API Error: Request timed out after 2 hours[/red]")
                    return "Error: API request timed out after 2 hours. The model may be overloaded."
                except Exception as e:
                    error_str = str(e)
                    self.console.print(f"[red]API Error: {error_str}[/red]")
                    if "timeout" in error_str.lower():
                        self.console.print("[yellow]Hint: The API may be slow. Try a simpler request.[/yellow]")
                    return f"Error communicating with LLM: {e}"
                
                # Show model's thinking if any
                if response.content:
                    self.console.print(Panel(response.content, title="Model", border_style="dim"))
                
                # Process response
                if response.has_tool_calls:
                    print(f"🔧 Executing {len(response.tool_calls)} tool(s)...", flush=True)
                    
                    # Add assistant message with tool calls
                    self.context.add_assistant_turn(
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    
                    # Execute tool calls
                    tool_results = await self._process_tool_calls(response.tool_calls)
                    
                    # Add tool results to context
                    for result_msg in tool_results:
                        self.context.window.add_message(result_msg)
                    
                    # Continue loop for next LLM response
                    continue
                
                # No tool calls - this is the final response
                if response.content:
                    self.console.print(Panel(
                        response.content,
                        title="Assistant Response",
                        border_style="green",
                    ))
                    return response.content
                
                self.console.print("[red]Warning: Model returned empty response[/red]")
                return "Agent completed without a response."
            
            return f"Agent reached maximum iterations ({self.max_iterations})"
    
    async def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.console.print("[bold]Agentic Coding Harness[/bold]")
        self.console.print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                user_input = self.console.input("[bold blue]You:[/bold blue] ")
            except (KeyboardInterrupt, EOFError):
                break
            
            if user_input.lower() in ("exit", "quit"):
                break
            
            if not user_input.strip():
                continue
            
            await self.run(user_input)
            self.console.print()


async def main():
    """Main entry point for the agent."""
    config = Config.from_env()
    agent = Agent(config)
    await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
