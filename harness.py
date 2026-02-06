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

# Multiline input support
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


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


def create_prompt_session(history_file: Path) -> "PromptSession":
    """Create a prompt session with multiline support.
    
    Keybindings:
    - Enter: Submit input
    - Ctrl+Enter: Insert newline (for multiline input)
    - Paste: Multiline paste works automatically
    """
    if not HAS_PROMPT_TOOLKIT:
        return None
    
    bindings = KeyBindings()
    
    # Ctrl+Enter inserts newline (Escape+Enter as fallback for terminals that don't support Ctrl+Enter)
    @bindings.add(Keys.Escape, Keys.Enter)
    def _(event):
        """Escape+Enter: insert newline (fallback)."""
        event.current_buffer.insert_text('\n')
    
    @bindings.add('c-j')  # Ctrl+J = Ctrl+Enter in most terminals
    def _(event):
        """Ctrl+Enter: insert newline."""
        event.current_buffer.insert_text('\n')
    
    # Create session with history
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    return PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=False,  # Enter submits, Ctrl+Enter for newline
    )


async def run_single(agent: ClineAgent, user_input: str, console: Console):
    """Run a single user request."""
    try:
        result = await agent.run(user_input)
    except asyncio.CancelledError:
        console.print("\n[yellow][STOP] Cancelled[/yellow]")
        result = "[Interrupted]"
    except KeyboardInterrupt:
        console.print("\n[yellow][STOP] Interrupted[/yellow]")
        result = "[Interrupted]"
    
    # Show final response
    if result:
        console.print()
        console.print(Panel(result, title="Response", border_style="green"))
    
    # Show cost and context stats
    cost = get_global_tracker().get_summary()
    stats = agent.get_context_stats()
    
    console.print(Panel(
        f"API Calls: {cost.total_calls}\n"
        f"Tokens: {cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out\n"
        f"Cost: ${cost.total_cost:.4f}\n"
        f"───────────────\n"
        f"Context: {stats['tokens']:,} tokens ({stats['percent']:.0f}% of {stats['max_allowed']:,} limit)\n"
        f"Messages: {stats['messages']} | Items: {stats['context_items']} ({stats['context_chars']:,} chars)",
        title="Stats",
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
    
    # Create a single persistent event loop for the entire session
    # This prevents "Event loop is closed" errors when subprocesses are cleaned up
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
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
        try:
            loop.run_until_complete(run_single(agent, user_input, console))
        finally:
            agent.cleanup_background_procs()
            agent.save_session(str(session_path))
            loop.close()
    else:
        # Interactive mode
        console.print("[bold blue]Harness[/bold blue] - [bold]Esc[/bold] interrupt | [bold]Ctrl+B[/bold] background | /help for commands")
        if HAS_PROMPT_TOOLKIT:
            console.print("[dim]Ctrl+Enter for newline | Workspace: " + workspace + "[/dim]\n")
        else:
            console.print(f"[dim]Workspace: {workspace}[/dim]\n")
        
        # Create prompt session for multiline input
        history_file = get_sessions_dir(workspace) / ".history"
        prompt_session = create_prompt_session(history_file) if HAS_PROMPT_TOOLKIT else None
        
        while True:
            try:
                # Show token count in prompt if significant
                stats = agent.get_context_stats()
                token_info = f" [{stats['tokens']//1000}k]" if stats['tokens'] > 1000 else ""
                prompt_text = f"[{current_session}]{token_info}> "
                
                # Get input (multiline with prompt_toolkit, or simple input)
                if prompt_session:
                    try:
                        user_input = prompt_session.prompt(prompt_text).strip()
                    except KeyboardInterrupt:
                        continue  # Ctrl+C cancels current input
                else:
                    user_input = input(prompt_text).strip()
                
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
                    
                    elif cmd == '/bg':
                        procs = agent.list_background_procs()
                        if not procs:
                            console.print("[dim]No background processes.[/dim]")
                        else:
                            console.print("[dim]Background processes:[/dim]")
                            for p in procs:
                                elapsed_min = p['elapsed'] / 60
                                console.print(f"[dim]  [{p['id']}] PID {p['pid']} - {p['status']} - {elapsed_min:.1f}m - {p['command']}[/dim]")
                        continue
                    
                    elif cmd == '/ctx':
                        stats = agent.get_context_stats()
                        console.print(f"[dim]Context: {stats['tokens']:,} tokens ({stats['percent']:.0f}% of {stats['max_allowed']:,} limit)[/dim]")
                        console.print(f"[dim]Messages: {stats['messages']} | Context items: {stats['context_items']}[/dim]")
                        console.print(f"[dim]{agent.context.summary()}[/dim]")
                        continue
                    
                    elif cmd in ('/help', '/?'):
                        help_text = """[dim]Commands:
  /sessions          - List all sessions
  /session <name>    - Switch to session (creates if new)
  /delete <name>     - Delete a session
  /clear             - Clear conversation history
  /save              - Save current session
  /history           - Show message count
  /bg                - List background processes
  /ctx               - Show context container
  /exit              - Save and exit

Keys during commands:
  Esc                - Stop/interrupt
  Ctrl+B             - Send to background"""
                        if HAS_PROMPT_TOOLKIT:
                            help_text += """

Input:
  Ctrl+Enter         - Insert newline (multiline input)
  Enter              - Submit input
  Up/Down            - Browse history
  Paste              - Multiline paste supported"""
                        help_text += "[/dim]"
                        console.print(help_text)
                        continue
                    
                    else:
                        console.print("[dim]Type /help for commands[/dim]")
                        continue
                
                try:
                    loop.run_until_complete(run_single(agent, user_input, console))
                except KeyboardInterrupt:
                    console.print("\n[yellow][STOP] Interrupted - Ctrl+C again to exit[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
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
        
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass


if __name__ == "__main__":
    main()
