"""Command-line interface for the harness with headless and interactive modes."""

import asyncio
import argparse
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.prompt import Prompt

from .config import Config
from .agent import Agent
from .cost_tracker import CostTracker, CostSummary, get_global_tracker, reset_global_tracker


class OutputFormat(Enum):
    """Output format options."""
    HUMAN = "human"
    JSON = "json"
    JSONL = "jsonl"


@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolUse:
    """Record of a tool invocation."""
    name: str
    arguments: Dict[str, Any]
    result: str
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    """Structured result from a run - Claude Code style."""
    session_id: str
    input: str
    output: str
    messages: List[Dict[str, Any]]
    tool_uses: List[Dict[str, Any]]
    cost: Dict[str, Any]
    duration_ms: float
    success: bool
    error: Optional[str] = None
    iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "input": self.input,
            "output": self.output,
            "messages": self.messages,
            "tool_uses": self.tool_uses,
            "cost": self.cost,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error,
            "iterations": self.iterations,
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class HeadlessAgent(Agent):
    """Agent that collects structured output for headless mode."""
    
    def __init__(self, *args, output_format: OutputFormat = OutputFormat.HUMAN, **kwargs):
        # Create a silent console for headless mode
        if output_format != OutputFormat.HUMAN:
            kwargs['console'] = Console(quiet=True)
        super().__init__(*args, **kwargs)
        
        self.output_format = output_format
        self.messages: List[Message] = []
        self.tool_uses: List[ToolUse] = []
        self._iteration_count = 0
    
    async def run_headless(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> RunResult:
        """Run in headless mode with structured output."""
        import time
        import uuid
        
        session_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        reset_global_tracker()
        
        self.messages = []
        self.tool_uses = []
        self._iteration_count = 0
        
        self.messages.append(Message(role="user", content=user_message))
        
        try:
            output = await self._run_tracked(user_message, system_prompt)
            success = True
            error = None
            
            self.messages.append(Message(role="assistant", content=output))
            
        except Exception as e:
            output = ""
            success = False
            error = str(e)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        cost_summary = get_global_tracker().get_summary()
        
        return RunResult(
            session_id=session_id,
            input=user_message,
            output=output,
            messages=[m.to_dict() for m in self.messages],
            tool_uses=[t.to_dict() for t in self.tool_uses],
            cost=cost_summary.to_dict(),
            duration_ms=duration_ms,
            success=success,
            error=error,
            iterations=self._iteration_count,
        )
    
    async def _run_tracked(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run with tracking for structured output."""
        from .llm_client import LLMClient, Message as LLMMessage
        from .agent import get_system_prompt
        
        self.config.validate()
        
        # Initialize context with dynamic system prompt
        default_prompt = get_system_prompt(str(self.config.workspace_path))
        self.context.initialize(system_prompt or default_prompt)
        self.context.add_user_turn(user_message)
        
        async with LLMClient(self.config) as client:
            self._client = client
            
            while self._iteration_count < self.max_iterations:
                self._iteration_count += 1
                
                # Compact context if needed
                self.context.compact_if_needed()
                
                # Get LLM response (use parent's streaming method if not quiet)
                if self.output_format == OutputFormat.HUMAN:
                    response = await self._chat_with_streaming(
                        client,
                        messages=self.context.get_messages(),
                        tools=self.tools.to_openai_schema(),
                    )
                else:
                    # Headless JSON mode - still use streaming for reliability
                    response = await client.chat_stream(
                        messages=self.context.get_messages(),
                        tools=self.tools.to_openai_schema(),
                        on_token=None,  # No output for JSON mode
                    )
                
                # Process response
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    self.context.add_assistant_turn(
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                    
                    # Execute tool calls with tracking
                    tool_results = await self._process_tool_calls_tracked(response.tool_calls)
                    
                    # Add tool results to context
                    for result_msg in tool_results:
                        self.context.window.add_message(result_msg)
                    
                    continue
                
                # No tool calls - this is the final response
                if response.content:
                    return response.content
                
                return "Agent completed without a response."
            
            return f"Agent reached maximum iterations ({self.max_iterations})"
    
    async def _process_tool_calls_tracked(
        self,
        tool_calls: List[Dict[str, Any]],
    ):
        """Process tool calls with tracking."""
        from .llm_client import Message as LLMMessage
        import json
        import sys
        
        results = []
        
        for tool_call in tool_calls:
            call_id = tool_call.get("id", "")
            tool_name = tool_call.get("function", {}).get("name", "")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}
            
            # Show what's happening (even in headless mode for human output)
            if self.output_format == OutputFormat.HUMAN:
                if tool_name == "write_file":
                    fp = arguments.get("file_path", "?")
                    cl = len(arguments.get("content", ""))
                    print(f"📝 Writing: {fp} ({cl} chars)", flush=True)
                elif tool_name == "read_file":
                    print(f"📖 Reading: {arguments.get('file_path', '?')}", flush=True)
                elif tool_name == "edit_file":
                    print(f"✏️  Editing: {arguments.get('file_path', '?')}", flush=True)
                elif tool_name == "run_command":
                    cmd = arguments.get("command", "?")[:60]
                    print(f"💻 Running: {cmd}", flush=True)
                else:
                    print(f"🔧 {tool_name}", flush=True)
            
            result = await self.tools.execute(tool_name, **arguments)
            
            if self.output_format == OutputFormat.HUMAN:
                if result.success:
                    print(f"  ✓ Done", flush=True)
                else:
                    print(f"  ✗ {result.error}", flush=True)
            
            # Track tool use
            self.tool_uses.append(ToolUse(
                name=tool_name,
                arguments=arguments,
                result=str(result.output) if result.success else (result.error or ""),
                success=result.success,
            ))
            
            result_message = result.to_message()
            
            # Truncate very long results
            if len(result_message) > 10000:
                result_message = result_message[:5000] + "\n...[truncated]...\n" + result_message[-5000:]
            
            results.append(LLMMessage(
                role="tool",
                content=result_message,
                tool_call_id=call_id,
                name=tool_name,
            ))
        
        return results


class InteractiveSession:
    """Interactive REPL session with rich UI."""
    
    def __init__(
        self,
        config: Config,
        max_iterations: int = 20,
    ):
        self.config = config
        self.max_iterations = max_iterations
        self.console = Console()
        self.cost_tracker = CostTracker(
            on_update=self._on_cost_update,
            report_interval=3,
        )
        self.agent: Optional[Agent] = None
        self.session_start = datetime.now()
        self.total_requests = 0
    
    def _on_cost_update(self, summary: CostSummary) -> None:
        """Callback when costs are updated."""
        self.console.print(
            f"[dim]💰 Cost update: ${summary.total_cost:.4f} "
            f"({summary.total_tokens:,} tokens, {summary.total_calls} calls)[/dim]"
        )
    
    def _show_status_bar(self) -> Panel:
        """Create status bar with cost info."""
        summary = self.cost_tracker.get_summary()
        elapsed = (datetime.now() - self.session_start).total_seconds()
        
        status = Table.grid(expand=True)
        status.add_column(justify="left")
        status.add_column(justify="center")
        status.add_column(justify="right")
        
        status.add_row(
            f"[bold]{self.config.model}[/bold]",
            f"Requests: {self.total_requests} | Tokens: {summary.total_tokens:,}",
            f"[green]${summary.total_cost:.4f}[/green] | {elapsed:.0f}s"
        )
        
        return Panel(status, style="dim")
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold blue]Agentic Coding Harness[/bold blue]\n"
            f"Model: [cyan]{self.config.model}[/cyan]\n"
            f"Workspace: [dim]{self.config.workspace_path}[/dim]",
            border_style="blue",
        ))
        self.console.print()
        self.console.print("[dim]Commands: /help, /cost, /clear, /exit[/dim]")
        self.console.print()
    
    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]/help[/cyan]     Show this help message
  [cyan]/cost[/cyan]     Show current session cost summary
  [cyan]/clear[/cyan]    Clear the conversation context
  [cyan]/exit[/cyan]     Exit the session (or Ctrl+C)
  [cyan]/stats[/cyan]    Show detailed session statistics

[bold]Tips:[/bold]

  • The agent can read, write, and edit files
  • It can run shell commands and search code
  • Be specific about file paths for best results
        """
        self.console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))
    
    def _show_cost(self) -> None:
        """Show current cost summary."""
        summary = self.cost_tracker.get_summary()
        self.console.print(Panel(
            summary.format_human(),
            title="Session Cost Summary",
            border_style="green",
        ))
    
    def _show_stats(self) -> None:
        """Show detailed session statistics."""
        summary = self.cost_tracker.get_summary()
        elapsed = (datetime.now() - self.session_start).total_seconds()
        
        stats = f"""
[bold]Session Statistics[/bold]

Session Duration: {elapsed:.1f} seconds
Total Requests: {self.total_requests}

[bold]Token Usage[/bold]
  Input Tokens:  {summary.total_input_tokens:,}
  Output Tokens: {summary.total_output_tokens:,}
  Total Tokens:  {summary.total_tokens:,}

[bold]Costs[/bold]
  Input Cost:  ${summary.total_input_cost:.6f}
  Output Cost: ${summary.total_output_cost:.6f}
  Total Cost:  ${summary.total_cost:.6f}

[bold]API Calls[/bold]
  Total Calls:  {summary.total_calls}
  Tool Calls:   {summary.total_tool_calls}
  Avg Tokens/Call: {summary.total_tokens / max(1, summary.total_calls):.0f}
  Avg Cost/Call: ${summary.total_cost / max(1, summary.total_calls):.6f}
        """
        self.console.print(Panel(stats.strip(), title="Detailed Statistics", border_style="cyan"))
    
    async def run(self) -> None:
        """Run interactive session."""
        from .llm_client import LLMClient
        
        self._show_welcome()
        
        # Create agent with our console and cost tracker
        self.agent = Agent(
            config=self.config,
            console=self.console,
            max_iterations=self.max_iterations,
        )
        
        # Set up cost tracking
        reset_global_tracker()
        
        while True:
            try:
                # Show status and get input
                self.console.print(self._show_status_bar())
                import sys
                sys.stdout.write("You: ")
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye![/dim]")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/exit", "/quit", "/q"):
                    self.console.print("[dim]Goodbye![/dim]")
                    break
                elif cmd == "/help":
                    self._show_help()
                    continue
                elif cmd == "/cost":
                    self._show_cost()
                    continue
                elif cmd == "/stats":
                    self._show_stats()
                    continue
                elif cmd == "/clear":
                    self.agent.context.clear()
                    self.console.print("[dim]Context cleared.[/dim]")
                    continue
                else:
                    self.console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                    continue
            
            # Run agent
            self.total_requests += 1
            
            try:
                result = await self.agent.run(user_input)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
            
            self.console.print()
        
        # Show final cost summary
        self.console.print()
        self._show_cost()


def run_headless(args: argparse.Namespace) -> int:
    """Run in headless mode."""
    # Load configuration
    env_path = Path(args.env)
    if env_path.exists():
        config = Config.from_env(env_path)
    else:
        config = Config.from_env()
    
    config.workspace_path = Path(args.workspace).resolve()
    config.max_context_tokens = args.max_tokens
    
    try:
        config.validate()
    except ValueError as e:
        error_result = {
            "success": False,
            "error": f"Configuration error: {e}",
        }
        if args.output_format == "json":
            print(json.dumps(error_result, indent=2))
        else:
            print(json.dumps(error_result))
        return 1
    
    # Determine output format
    output_format = OutputFormat.JSON if args.print else OutputFormat.HUMAN
    if hasattr(args, 'output_format') and args.output_format:
        output_format = OutputFormat(args.output_format)
    
    # Create agent
    agent = HeadlessAgent(
        config,
        max_iterations=args.max_iterations,
        output_format=output_format,
    )
    
    # Get input
    if args.message:
        message = args.message
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            message = f.read().strip()
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        print("Error: No input provided. Use -m, --input-file, or pipe input.", file=sys.stderr)
        return 1
    
    # Run
    result = asyncio.run(agent.run_headless(message))
    
    # Output
    if output_format == OutputFormat.JSON:
        print(result.to_json(indent=2))
    elif output_format == OutputFormat.JSONL:
        print(result.to_json(indent=None))
    else:
        # Human readable
        console = Console()
        console.print(Panel(result.output, title="Response", border_style="green"))
        console.print()
        summary = CostSummary(**{k: v for k, v in result.cost.items() 
                                 if k in CostSummary.__dataclass_fields__})
        console.print(Panel(summary.format_human(), title="Cost", border_style="cyan"))
    
    return 0 if result.success else 1


def run_interactive(args: argparse.Namespace) -> int:
    """Run in interactive mode."""
    # Load configuration
    env_path = Path(args.env)
    if env_path.exists():
        config = Config.from_env(env_path)
    else:
        config = Config.from_env()
    
    config.workspace_path = Path(args.workspace).resolve()
    config.max_context_tokens = args.max_tokens
    
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    
    session = InteractiveSession(
        config=config,
        max_iterations=args.max_iterations,
    )
    
    asyncio.run(session.run())
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic coding harness optimized for Z.AI GLM-4.7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode in current directory
  harness .
  
  # Interactive mode in specific directory
  harness /path/to/project
  
  # Headless mode with structured JSON output
  harness . -p -m "Fix the bug in main.py"
  
  # Pipe input
  echo "Explain this code" | harness . -p
  
  # Read from file
  harness . -p --input-file prompt.txt
  
  # Output formats
  harness . --output-format json -m "List files"
        """
    )
    
    # Positional argument for workspace
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace directory (default: current directory)"
    )
    
    # Mode selection
    parser.add_argument(
        "-p", "--print",
        action="store_true",
        help="Print structured JSON output and exit (headless mode)"
    )
    parser.add_argument(
        "--output-format",
        choices=["human", "json", "jsonl"],
        default=None,
        help="Output format (default: human for interactive, json for --print)"
    )
    
    # Input options
    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Input message for headless mode"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Read input from file"
    )
    
    # Configuration
    parser.add_argument(
        "-e", "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations per request (default: 20)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum context tokens (default: 32000)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.print or args.message or args.input_file or not sys.stdin.isatty():
        sys.exit(run_headless(args))
    else:
        sys.exit(run_interactive(args))


if __name__ == "__main__":
    main()
