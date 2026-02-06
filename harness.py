#!/usr/bin/env python3
"""Streaming harness entry point - true token-by-token streaming."""

import sys
import os

# Force unbuffered output BEFORE any imports
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(write_through=True)

import asyncio
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config
from harness.cline_agent import ClineAgent
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from rich.console import Console
from rich.panel import Panel


def get_session_path(workspace: str) -> Path:
    """Get session file path for a workspace."""
    harness_dir = Path(__file__).parent
    sessions_dir = harness_dir / ".sessions"
    sessions_dir.mkdir(exist_ok=True)
    workspace_hash = hashlib.md5(workspace.encode()).hexdigest()[:12]
    return sessions_dir / f"{workspace_hash}.json"


async def run_single(agent: ClineAgent, user_input: str, console: Console):
    """Run a single user request."""
    result = await agent.run(user_input)
    
    # Show final response
    if result:
        console.print()
        console.print(Panel(result, title="Response", border_style="green"))
    
    # Show cost
    cost = get_global_tracker().get_summary()
    console.print(Panel(
        f"API Calls: {cost.total_calls}\n"
        f"Tokens: {cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out\n"
        f"Cost: ${cost.total_cost:.4f}",
        title="Cost",
        border_style="blue"
    ))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Harness")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace directory")
    parser.add_argument("--new", action="store_true", help="Start fresh session (ignore saved)")
    args = parser.parse_args()
    
    # Resolve workspace
    if args.workspace == ".":
        workspace = os.getcwd()
    else:
        workspace = os.path.abspath(args.workspace)
    
    os.chdir(workspace)
    
    # Load config
    harness_dir = Path(__file__).parent
    config = Config.from_env(harness_dir / ".env")
    config.validate()
    
    console = Console()
    
    # Create agent
    agent = ClineAgent(config)
    
    # Session management
    session_path = get_session_path(workspace)
    
    # Try to resume session unless --new
    if not args.new and session_path.exists():
        if agent.load_session(str(session_path)):
            msg_count = len(agent.messages) - 1  # minus system prompt
            console.print(f"[dim]Resumed session ({msg_count} messages)[/dim]")
    
    # Get input from stdin or interactive prompt
    if not sys.stdin.isatty():
        # Piped input - single request
        user_input = sys.stdin.read().strip()
        if not user_input:
            print("No input provided.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_single(agent, user_input, console))
        agent.save_session(str(session_path))
    else:
        # Interactive mode
        console.print("[bold blue]Harness[/bold blue] - Type your request. Commands: /clear, /save, /exit")
        console.print(f"[dim]Workspace: {workspace}[/dim]\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd in ('/exit', '/quit', '/q'):
                        agent.save_session(str(session_path))
                        console.print("[dim]Session saved.[/dim]")
                        break
                    elif cmd == '/clear':
                        agent.clear_history()
                        reset_global_tracker()
                        console.print("[dim]History cleared.[/dim]")
                        continue
                    elif cmd == '/save':
                        agent.save_session(str(session_path))
                        console.print(f"[dim]Session saved to {session_path}[/dim]")
                        continue
                    elif cmd == '/history':
                        console.print(f"[dim]Messages: {len(agent.messages)}[/dim]")
                        continue
                    else:
                        console.print("[dim]Unknown command. Use /clear, /save, /history, /exit[/dim]")
                        continue
                
                asyncio.run(run_single(agent, user_input, console))
                print()  # Blank line between requests
                
            except KeyboardInterrupt:
                agent.save_session(str(session_path))
                console.print("\n[dim]Session saved. Exiting...[/dim]")
                break
            except EOFError:
                agent.save_session(str(session_path))
                break


if __name__ == "__main__":
    main()
