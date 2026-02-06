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
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config
from harness.cline_agent import ClineAgent
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from rich.console import Console
from rich.panel import Panel


def get_sessions_dir(workspace: str) -> Path:
    """Get sessions directory for a workspace."""
    harness_dir = Path(__file__).parent
    sessions_dir = harness_dir / ".sessions"
    workspace_hash = hashlib.md5(workspace.encode()).hexdigest()[:12]
    workspace_sessions = sessions_dir / workspace_hash
    workspace_sessions.mkdir(parents=True, exist_ok=True)
    return workspace_sessions


def get_session_path(workspace: str, session_name: str = "default") -> Path:
    """Get session file path for a workspace and session name."""
    return get_sessions_dir(workspace) / f"{session_name}.json"


def list_sessions(workspace: str) -> list[tuple[str, datetime, int]]:
    """List all sessions for a workspace. Returns [(name, modified_time, message_count), ...]"""
    import json
    sessions_dir = get_sessions_dir(workspace)
    sessions = []
    
    for f in sessions_dir.glob("*.json"):
        name = f.stem
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            msg_count = len(data.get("messages", [])) - 1  # minus system prompt
        except:
            msg_count = 0
        sessions.append((name, mtime, max(0, msg_count)))
    
    # Sort by most recently modified
    sessions.sort(key=lambda x: x[1], reverse=True)
    return sessions


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
    parser.add_argument("--new", action="store_true", help="Start fresh session")
    parser.add_argument("--session", "-s", default="default", help="Session name (default: 'default')")
    parser.add_argument("--list", "-l", action="store_true", help="List all sessions")
    args = parser.parse_args()
    
    # Resolve workspace
    if args.workspace == ".":
        workspace = os.getcwd()
    else:
        workspace = os.path.abspath(args.workspace)
    
    # List sessions mode
    if args.list:
        sessions = list_sessions(workspace)
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Sessions for {workspace}:\n")
            for name, mtime, count in sessions:
                print(f"  {name:20s}  {mtime:%Y-%m-%d %H:%M}  ({count} msgs)")
        return
    
    os.chdir(workspace)
    
    # Load config
    harness_dir = Path(__file__).parent
    config = Config.from_env(harness_dir / ".env")
    config.validate()
    
    console = Console()
    
    # Create agent
    agent = ClineAgent(config)
    
    # Session management
    current_session = args.session
    session_path = get_session_path(workspace, current_session)
    
    # Try to resume session unless --new
    if not args.new and session_path.exists():
        if agent.load_session(str(session_path)):
            msg_count = len(agent.messages) - 1  # minus system prompt
            console.print(f"[dim]Resumed session '{current_session}' ({msg_count} messages)[/dim]")
    else:
        console.print(f"[dim]New session '{current_session}'[/dim]")
    
    # Get input from stdin or interactive prompt
    if not sys.stdin.isatty():
        # Piped input - single request
        user_input = sys.stdin.read().strip()
        if not user_input:
            print("No input provided.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_single(agent, user_input, console))
        agent.cleanup_background_procs()
        agent.save_session(str(session_path))
    else:
        # Interactive mode
        console.print("[bold blue]Harness[/bold blue] - Press [bold]Esc[/bold] to interrupt | Commands: /sessions, /session, /clear, /exit")
        console.print(f"[dim]Workspace: {workspace}[/dim]\n")
        
        while True:
            try:
                prompt = f"[{current_session}]> "
                user_input = input(prompt).strip()
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    cmd_arg = parts[1] if len(parts) > 1 else ""
                    
                    if cmd in ('/exit', '/quit', '/q'):
                        agent.cleanup_background_procs()
                        agent.save_session(str(session_path))
                        console.print("[dim]Session saved.[/dim]")
                        break
                    
                    elif cmd == '/sessions':
                        sessions = list_sessions(workspace)
                        if not sessions:
                            console.print("[dim]No sessions.[/dim]")
                        else:
                            console.print("[dim]Sessions:[/dim]")
                            for name, mtime, count in sessions:
                                marker = "*" if name == current_session else " "
                                console.print(f"[dim] {marker} {name:20s} ({count} msgs)[/dim]")
                        continue
                    
                    elif cmd == '/session':
                        if not cmd_arg:
                            console.print(f"[dim]Current: {current_session}[/dim]")
                            continue
                        # Save current session first
                        agent.save_session(str(session_path))
                        # Switch to new session
                        new_name = cmd_arg.strip()
                        current_session = new_name
                        session_path = get_session_path(workspace, current_session)
                        if session_path.exists():
                            agent.load_session(str(session_path))
                            msg_count = len(agent.messages) - 1
                            console.print(f"[dim]Switched to '{current_session}' ({msg_count} msgs)[/dim]")
                        else:
                            agent.clear_history()
                            console.print(f"[dim]Created new session '{current_session}'[/dim]")
                        continue
                    
                    elif cmd == '/clear':
                        agent.clear_history()
                        reset_global_tracker()
                        console.print("[dim]History cleared.[/dim]")
                        continue
                    
                    elif cmd == '/save':
                        agent.save_session(str(session_path))
                        console.print(f"[dim]Saved '{current_session}'[/dim]")
                        continue
                    
                    elif cmd == '/delete':
                        if not cmd_arg:
                            console.print("[dim]Usage: /delete <session_name>[/dim]")
                            continue
                        target = cmd_arg.strip()
                        if target == current_session:
                            console.print("[dim]Cannot delete active session.[/dim]")
                            continue
                        target_path = get_session_path(workspace, target)
                        if target_path.exists():
                            target_path.unlink()
                            console.print(f"[dim]Deleted '{target}'[/dim]")
                        else:
                            console.print(f"[dim]Session '{target}' not found.[/dim]")
                        continue
                    
                    elif cmd == '/history':
                        console.print(f"[dim]Messages: {len(agent.messages)}[/dim]")
                        continue
                    
                    else:
                        console.print("[dim]Commands: /sessions, /session <name>, /delete <name>, /clear, /save, /exit[/dim]")
                        continue
                
                asyncio.run(run_single(agent, user_input, console))
                print()  # Blank line between requests
                
            except KeyboardInterrupt:
                agent.cleanup_background_procs()
                agent.save_session(str(session_path))
                console.print("\n[dim]Session saved. Exiting...[/dim]")
                break
            except EOFError:
                agent.cleanup_background_procs()
                agent.save_session(str(session_path))
                break


if __name__ == "__main__":
    main()
