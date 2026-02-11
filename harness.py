#!/usr/bin/env python3
"""Streaming harness entry point - true token-by-token streaming."""

import sys
import os

# Force unbuffered output and UTF-8 encoding BEFORE any imports
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', write_through=True)
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass  # May fail on some systems

import asyncio
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config
from harness.cline_agent import ClineAgent
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from harness.logger import init_logging, get_logger, log_exception, truncate
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape as rich_escape

log = get_logger("main")

# Multiline input support
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.formatted_text import ANSI
    HAS_PROMPT_TOOLKIT = True
    
    class SafeFileHistory(FileHistory):
        """FileHistory that handles unicode surrogates gracefully."""
        def store_string(self, string: str) -> None:
            # Remove unicode surrogates that can't be encoded
            safe_string = string.encode('utf-8', errors='replace').decode('utf-8')
            super().store_string(safe_string)
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    SafeFileHistory = None

# Clipboard image support
try:
    from PIL import ImageGrab
    HAS_CLIPBOARD_IMAGE = True
except ImportError:
    HAS_CLIPBOARD_IMAGE = False


def run_install(api_url: str = None, api_key: str = None, model: str = None, global_config: bool = True):
    """Setup wizard for API configuration. Supports headless mode with CLI args.
    
    Args:
        api_url: API base URL (headless mode)
        api_key: API key (headless mode)
        model: Model name (headless mode)
        global_config: If True, save to ~/.z.json, else workspace/.z/.z.json
    """
    from pathlib import Path
    import json
    
    # Headless mode - all params provided
    if api_url and api_key:
        config_data = {
            "api_url": api_url.rstrip('/') + '/',
            "api_key": api_key,
            "model": model or "glm-4.7",
        }
        
        if global_config:
            config_path = Path.home() / ".z.json"
        else:
            config_dir = Path.cwd() / ".z"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / ".z.json"
        
        config_path.write_text(json.dumps(config_data, indent=2))
        print(f"Configuration saved to: {config_path}")
        print(f"  URL:   {api_url}")
        print(f"  Model: {config_data['model']}")
        print(f"  Key:   {api_key[:10]}...")
        return
    
    # Interactive mode
    print("\n" + "="*60)
    print("  LLM Harness Setup")
    print("="*60 + "\n")
    
    # Choose provider
    print("Select your LLM provider:\n")
    print("  [1] Z.AI Coding Plan (recommended)")
    print("      - https://api.z.ai/api/coding/paas/v4/\n")
    print("  [2] Z.AI Standard API")
    print("      - https://api.z.ai/api/paas/v4/\n")
    print("  [3] MiniMax")
    print("      - https://api.minimax.io/v1/\n")
    print("  [4] Custom OpenAI-compatible API")
    print("      - Enter your own URL\n")
    
    while True:
        choice = input("Enter choice [1/2/3/4]: ").strip()
        if choice == "1":
            base_url = "https://api.z.ai/api/coding/paas/v4/"
            provider = "Z.AI Coding"
            default_model = "glm-4.7"
            break
        elif choice == "2":
            base_url = "https://api.z.ai/api/paas/v4/"
            provider = "Z.AI Standard"
            default_model = "glm-4.7"
            break
        elif choice == "3":
            base_url = "https://api.minimax.io/v1/"
            provider = "MiniMax"
            default_model = "MiniMax-M2.1"
            break
        elif choice == "4":
            base_url = input("Enter API base URL: ").strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Custom"
            default_model = input("Enter default model name: ").strip() or "gpt-4"
            break
        else:
            print("Please enter 1, 2, 3, or 4.")
    
    print(f"\nUsing {provider}: {base_url}")
    
    # Get API key
    print("\nEnter your API key:\n")
    
    api_key = ""
    while not api_key:
        api_key = input("API Key: ").strip()
        if not api_key:
            print("API key is required.")
    
    # Model
    model_input = input(f"\nModel name (default: {default_model}): ").strip()
    model = model_input or default_model
    
    # Build config
    config_data = {
        "api_url": base_url,
        "api_key": api_key,
        "model": model,
    }
    
    # Choose where to save
    print("\nWhere to save configuration?\n")
    print("  [1] Global (~/.z.json) - applies to all projects")
    print("  [2] Workspace (.z/.z.json) - this project only\n")
    
    save_choice = input("Enter choice [1/2] (default: 1): ").strip() or "1"
    
    if save_choice == "2":
        config_dir = Path.cwd() / ".z"
        config_path = config_dir / ".z.json"
        location = "workspace"
    else:
        config_dir = Path.home()
        config_path = config_dir / ".z.json"
        location = "global"
    
    # Create directory if needed
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config
    config_path.write_text(json.dumps(config_data, indent=2))
    
    print(f"\n" + "="*60)
    print("  Setup Complete!")
    print("="*60)
    print(f"\nConfiguration saved to: {config_path}")
    print(f"  Location: {location}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Key:      {api_key[:10]}...")
    print(f"\nRun 'python harness.py' to start.\n")


def get_clipboard_image() -> tuple[Path | None, str]:
    """Get image from clipboard if available.
    
    Returns:
        (temp_file_path, error_message) - path is None on error
    """
    if not HAS_CLIPBOARD_IMAGE:
        return None, "PIL not installed. Run: pip install Pillow"
    
    try:
        img = ImageGrab.grabclipboard()
        if img is None:
            return None, "No image in clipboard"
        
        # Check if it's a list of file paths (copied files)
        if isinstance(img, list):
            # Filter for image files
            image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
            for path_str in img:
                p = Path(path_str)
                if p.suffix.lower() in image_exts:
                    return p, ""
            return None, "Clipboard contains files but no images"
        
        # It's a PIL Image - save to temp file
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "harness_clipboard"
        temp_dir.mkdir(exist_ok=True)
        
        # Use timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = temp_dir / f"clipboard_{timestamp}.png"
        
        img.save(temp_path, "PNG")
        return temp_path, ""
        
    except Exception as e:
        return None, f"Error getting clipboard: {e}"


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



def load_providers(workspace: str) -> Dict[str, dict]:
    """Load provider configs from .z/models.json.
    
    Supports both old key names (low/orchestrator) and new (fast/normal).
    Returns a normalised dict with 'fast' and 'normal' keys.
    """
    import json
    
    models_path = Path(workspace) / ".z" / "models.json"
    if not models_path.exists():
        return {}
    data = json.loads(models_path.read_text(encoding="utf-8-sig"))
    providers = data.get("providers", {})
    
    # Normalise old key names
    if "low" in providers and "fast" not in providers:
        providers["fast"] = providers.pop("low")
    if "orchestrator" in providers and "normal" not in providers:
        providers["normal"] = providers.pop("orchestrator")
    
    return providers


def load_claude_cli_config(workspace: str) -> dict:
    """Load optional claude_cli config from .z/models.json."""
    import json
    models_path = Path(workspace) / ".z" / "models.json"
    if not models_path.exists():
        return {}
    data = json.loads(models_path.read_text(encoding="utf-8-sig"))
    return data.get("claude_cli", {})


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
        history=SafeFileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=False,  # Enter submits, Ctrl+Enter for newline
    )


async def run_single(agent: ClineAgent, user_input: str, console: Console) -> str:
    """Run a single user request."""
    start_time = time.time()
    log.debug("run_single START mode=%s input=%s",
              agent.reasoning_mode, truncate(user_input, 120))
    try:
        result = await agent.run(user_input)
    except asyncio.CancelledError:
        log.warning("run_single cancelled after %.1fs", time.time() - start_time)
        console.print("\n[yellow][STOP] Cancelled[/yellow]")
        result = "[Interrupted]"
    except KeyboardInterrupt:
        log.warning("run_single KeyboardInterrupt after %.1fs", time.time() - start_time)
        console.print("\n[yellow][STOP] Interrupted[/yellow]")
        result = "[Interrupted]"
    
    elapsed = time.time() - start_time
    mode_tag = agent.reasoning_mode.upper()
    log.info("run_single DONE mode=%s elapsed=%.1fs result_len=%d",
             agent.reasoning_mode, elapsed, len(result or ""))
    
    # Show final response
    if result:
        console.print()
        console.print(Panel(
            result,
            title=f"[bold blue]{mode_tag}[/bold blue] Response",
            border_style="blue",
        ))
    
    # Show cost and context stats
    cost = get_global_tracker().get_summary()
    stats = agent.get_context_stats()
    breakdown = agent.get_token_breakdown()
    
    todo_line = ""
    if stats.get('todos_total', 0) > 0:
        todo_line = f"\nTodos: {stats['todos_completed']}/{stats['todos_total']} done, {stats['todos_active']} active"
    
    eviction_line = ""
    if stats.get('evictions', 0) > 0:
        eviction_line = f"\nEvictions: {stats['evictions']} context items evicted"
    
    # Format elapsed time
    if elapsed < 60:
        elapsed_str = f"{elapsed:.1f}s"
    else:
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins}m{secs:02d}s"
    
    console.print(Panel(
        f"API Calls: {cost.total_calls}  |  Elapsed: {elapsed_str}  |  Mode: {agent.reasoning_mode}\n"
        f"Tokens: {cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out\n"
        f"Cost: ${cost.total_cost:.4f}\n"
        f"---------------\n"
        f"Context: {stats['tokens']:,} tokens ({stats['percent']:.0f}% of {stats['max_allowed']:,} limit)\n"
        f"System: {breakdown['system']:,}t | Conv: {breakdown['conversation']:,}t | Messages: {stats['messages']}"
        f"{todo_line}{eviction_line}",
        title=f"[bold blue]{mode_tag}[/bold blue] Stats",
        border_style="blue",
    ))
    
    return result or ""


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Harness")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace directory")
    parser.add_argument("--new", action="store_true", help="Start fresh session")
    parser.add_argument("--session", "-s", default="default", help="Session name (default: 'default')")
    parser.add_argument("--list", "-l", action="store_true", help="List all sessions")
    parser.add_argument("--install", action="store_true", help="Run setup wizard (interactive or headless)")
    parser.add_argument("--api-url", help="API base URL (headless install)")
    parser.add_argument("--api-key", help="API key (headless install)")
    parser.add_argument("--model", help="Model name (headless install)")
    parser.add_argument("--workspace-config", action="store_true", help="Save config to workspace instead of global")
    args = parser.parse_args()
    
    # Install mode - run setup wizard
    if args.install or args.api_url or args.api_key:
        run_install(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            global_config=not args.workspace_config
        )
        return
    
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
    
    # Initialise logging FIRST so every subsequent action is captured
    init_logging(workspace=workspace, session_id=args.session)
    log.info("=== Harness starting === workspace=%s session=%s new=%s",
             workspace, args.session, args.new)
    
    # Load providers from models.json
    providers = load_providers(workspace)
    claude_cli_config = load_claude_cli_config(workspace)
    
    if not providers:
        log.warning("No providers found in .z/models.json — falling back to default config")
    
    # Determine starting config — use 'normal' provider if available, else fall back to .z.json
    if "normal" in providers:
        p = providers["normal"]
        config = Config.from_json(workspace=Path(workspace), overrides={
            "api_url": p["api_url"],
            "api_key": p["api_key"],
            "model": p["model"],
        })
    else:
        config = Config.from_json(workspace=Path(workspace))
    config.validate()
    log.info("Config loaded: api_url=%s model=%s max_tokens=%d providers=%s",
             config.api_url, config.model, config.max_tokens, list(providers.keys()))
    
    # Create a single persistent event loop for the entire session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    console = Console()
    
    # Create single agent with providers for mode switching
    agent = ClineAgent(
        config,
        providers=providers,
        claude_cli_config=claude_cli_config,
    )
    log.info("Agent created: reasoning_mode=%s", agent.reasoning_mode)
    
    # Session management
    current_session = args.session
    session_path = get_session_path(workspace, current_session)
    
    # Try to resume session unless --new
    if not args.new and session_path.exists():
        if agent.load_session(str(session_path)):
            msg_count = len(agent.messages) - 1  # minus system prompt
            log.info("Resumed session '%s' (%d messages) from %s",
                     current_session, msg_count, session_path)
            console.print(f"[dim]Resumed session '{current_session}' ({msg_count} messages)[/dim]")
    else:
        log.info("New session '%s'", current_session)
        console.print(f"[dim]New session '{current_session}'[/dim]")

    def save_session():
        log.debug("Saving session")
        agent.save_session(str(session_path))
        log.debug("Session saved")

    def cleanup_and_save():
        loop.run_until_complete(agent.cleanup_background_procs_async())
        save_session()
    
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
            cleanup_and_save()
            loop.close()
    else:
        # Interactive mode
        console.print("[bold blue]Harness[/bold blue] - [bold]Esc[/bold] interrupt | [bold]Ctrl+B[/bold] background | /help")
        if HAS_PROMPT_TOOLKIT:
            console.print("[dim]Ctrl+Enter for newline | Workspace: " + workspace + "[/dim]\n")
        else:
            console.print(f"[dim]Workspace: {workspace}[/dim]\n")
        
        # Create prompt session for multiline input
        history_file = get_sessions_dir(workspace) / ".history"
        prompt_session = create_prompt_session(history_file) if HAS_PROMPT_TOOLKIT else None
        
        last_interrupt_time = 0  # Track time of last Ctrl+C for double-tap exit
        
        while True:
            try:
                # Show token count and reasoning mode in prompt
                stats = agent.get_context_stats()
                token_info = f" [{stats['tokens']//1000}k]" if stats['tokens'] > 1000 else ""
                mode_tag = agent.reasoning_mode
                prompt_text = f"\x1b[34m[{mode_tag}:{current_session}]{token_info}>\x1b[0m "
                
                # Get input (multiline with prompt_toolkit, or simple input)
                if prompt_session:
                    try:
                        user_input = prompt_session.prompt(ANSI(prompt_text)).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print("\n[dim]Ctrl+C again to exit[/dim]")
                        continue
                else:
                    try:
                        user_input = input(prompt_text).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print("\n[dim]Ctrl+C again to exit[/dim]")
                        continue
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    cmd_arg = parts[1] if len(parts) > 1 else ""
                    
                    if cmd in ('/exit', '/quit', '/q'):
                        cleanup_and_save()
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
                    
                    elif cmd == '/mode':
                        if not cmd_arg:
                            console.print(f"[dim]Current reasoning mode: {agent.reasoning_mode}[/dim]")
                            console.print(f"[dim]Available: {', '.join(providers.keys())}[/dim]")
                        else:
                            mode = cmd_arg.strip().lower()
                            if mode in providers:
                                result = agent._handle_set_reasoning_mode({"mode": mode})
                                console.print(f"[dim]{result}[/dim]")
                            else:
                                console.print(f"[dim]Unknown mode '{mode}'. Available: {', '.join(providers.keys())}[/dim]")
                        continue
                    
                    elif cmd == '/ctx':
                        stats = agent.get_context_stats()
                        console.print(f"[dim]Context: {stats['tokens']:,} tokens ({stats['percent']:.0f}% of {stats['max_allowed']:,} limit)[/dim]")
                        console.print(f"[dim]Messages: {stats['messages']} | Context items: {stats['context_items']}[/dim]")
                        console.print(f"[dim]{agent.context.summary()}[/dim]")
                        continue
                    
                    elif cmd == '/tokens':
                        breakdown = agent.get_token_breakdown()
                        console.print(f"[dim]Token breakdown:[/dim]")
                        console.print(f"[dim]  System prompt: {breakdown['system']:,} tokens[/dim]")
                        console.print(f"[dim]  Conversation:  {breakdown['conversation']:,} tokens ({breakdown['message_count']} messages)[/dim]")
                        console.print(f"[dim]  Total:         {breakdown['total']:,} tokens[/dim]")
                        if breakdown['largest_messages']:
                            console.print(f"[dim]  Largest messages:[/dim]")
                            for msg in breakdown['largest_messages']:
                                role = msg['role']
                                tokens = msg['tokens']
                                preview = msg['preview'][:60] + '...' if len(msg['preview']) > 60 else msg['preview']
                                console.print(f"[dim]    [{role}] {tokens:,}t: {preview}[/dim]")
                        continue
                    
                    elif cmd == '/compact':
                        strategy = cmd_arg.strip() if cmd_arg else 'half'
                        before = agent.get_token_count()
                        removed = agent.compact_history(strategy)
                        after = agent.get_token_count()
                        console.print(f"[dim]Compacted: {before:,} -> {after:,} tokens (-{removed:,})[/dim]")
                        continue
                    
                    elif cmd == '/todo':
                        if not cmd_arg:
                            agent.todo_manager.print_todo_panel(console)
                        elif cmd_arg.startswith("add "):
                            title = cmd_arg[4:].strip()
                            item = agent.todo_manager.add(title=title)
                            agent.todo_manager.print_todo_panel(console)
                        elif cmd_arg.startswith("done "):
                            try:
                                item_id = int(cmd_arg[5:].strip())
                                item = agent.todo_manager.update(item_id, status="completed")
                                if not item:
                                    console.print(f"[dim]Todo [{item_id}] not found[/dim]")
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("[dim]Usage: /todo done <id>[/dim]")
                        elif cmd_arg.startswith("rm "):
                            try:
                                item_id = int(cmd_arg[3:].strip())
                                if not agent.todo_manager.remove(item_id):
                                    console.print(f"[dim]Todo [{item_id}] not found[/dim]")
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("[dim]Usage: /todo rm <id>[/dim]")
                        elif cmd_arg.strip() == "clear":
                            agent.todo_manager.clear()
                            console.print("[dim]All todos cleared.[/dim]")
                        else:
                            console.print("[dim]Usage: /todo [add <title>|done <id>|rm <id>|clear][/dim]")
                        continue
                    
                    elif cmd == '/smart':
                        from harness.context_management import get_model_limits as _gml, estimate_messages_tokens as _emt
                        _, max_allowed = _gml(agent.config.model)
                        total_tokens = _emt(agent.messages)
                        pct = total_tokens / max_allowed * 100 if max_allowed else 0
                        over = total_tokens > int(max_allowed * 0.85)
                        active_todos = agent.todo_manager.list_active()
                        traces = agent.smart_context.compaction_traces
                        
                        console.print(f"[dim]Context: {total_tokens:,} / {max_allowed:,} tokens ({pct:.0f}%)[/dim]")
                        console.print(f"[dim]Over compaction threshold (85%): {'YES' if over else 'no'}[/dim]")
                        console.print(f"[dim]Messages: {len(agent.messages)}  |  Active todos: {len(active_todos)}[/dim]")
                        console.print(f"[dim]Context items: {len(agent.context.list_items())}  ({agent.context.total_size():,} chars)[/dim]")
                        if traces:
                            console.print(f"[dim]Compactions this session: {len(traces)}[/dim]")
                            for t in traces[-5:]:
                                console.print(f"[dim]  - {t.format_notice()}[/dim]")
                        else:
                            console.print(f"[dim]No compactions yet this session.[/dim]")
                        continue
                    
                    elif cmd == '/dump':
                        dump_path = agent.dump_context(reason="user_requested")
                        console.print(f"[dim]Context dumped to: {dump_path}[/dim]")
                        breakdown = agent.get_token_breakdown()
                        console.print(f"[dim]  System prompt: {breakdown['system']:,} tokens ({len(agent.messages[0].content) if agent.messages else 0:,} chars)[/dim]")
                        console.print(f"[dim]  Conversation:  {breakdown['conversation']:,} tokens ({breakdown['message_count']} messages)[/dim]")
                        console.print(f"[dim]  Total:         {breakdown['total']:,} tokens[/dim]")
                        if agent.messages and agent.messages[0].role == "system":
                            sys_content = agent.messages[0].content
                            expected_tools = [
                                'read_file', 'write_to_file', 'replace_in_file',
                                'execute_command', 'manage_todos', 'attempt_completion',
                                'set_reasoning_mode', 'create_plan',
                            ]
                            tools_present = [t for t in expected_tools if t in sys_content]
                            tools_missing = [t for t in expected_tools if t not in sys_content]
                            if tools_missing:
                                console.print(f"[red]  WARNING: System prompt MISSING tools: {', '.join(tools_missing)}[/red]")
                            else:
                                console.print(f"[dim]  System prompt OK: all {len(tools_present)} core tools present[/dim]")
                        else:
                            console.print("[red]  WARNING: No system message found![/red]")
                        continue
                    
                    elif cmd == '/config':
                        key_preview = agent.config.api_key[:8] + "..." if agent.config.api_key else "(not set)"
                        console.print(f"[dim]API URL: {agent.config.api_url}[/dim]")
                        console.print(f"[dim]Model:   {agent.config.model}[/dim]")
                        console.print(f"[dim]API Key: {key_preview}[/dim]")
                        console.print(f"[dim]Max tokens: {agent.config.max_tokens:,}[/dim]")
                        console.print(f"[dim]Reasoning mode: {agent.reasoning_mode}[/dim]")
                        console.print(f"[dim]Max iterations: {agent.max_iterations}[/dim]")
                        continue
                    
                    elif cmd == '/iter':
                        if not cmd_arg:
                            console.print(f"[dim]Max iterations: {agent.max_iterations}[/dim]")
                            continue
                        try:
                            new_val = int(cmd_arg.strip())
                            if new_val < 1:
                                console.print("[dim]Must be at least 1[/dim]")
                                continue
                            agent.max_iterations = new_val
                            console.print(f"[dim]Max iterations set to {new_val}[/dim]")
                        except ValueError:
                            console.print("[dim]Usage: /iter <number>[/dim]")
                        continue
                    
                    elif cmd == '/clip':
                        img_path, error = get_clipboard_image()
                        if error:
                            console.print(f"[red]{error}[/red]")
                            continue
                        
                        console.print(f"[dim]Clipboard image: {img_path}[/dim]")
                        question = cmd_arg.strip() if cmd_arg else "Describe this image in detail."
                        user_input = f"Analyze this image: {img_path}\n\nQuestion: {question}"
                        # Fall through to process this request
                    
                    elif cmd == '/log':
                        log_file = os.path.join(workspace, ".harness_output", "harness.log")
                        if not os.path.exists(log_file):
                            console.print("[dim]No log file yet.[/dim]")
                        else:
                            size_kb = os.path.getsize(log_file) / 1024
                            console.print(f"[dim]Log file: {log_file} ({size_kb:.0f} KB)[/dim]")
                            if cmd_arg.strip():
                                try:
                                    n = int(cmd_arg.strip())
                                except ValueError:
                                    n = 30
                            else:
                                n = 30
                            with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                                lines = fh.readlines()
                            for ln in lines[-n:]:
                                console.print(f"[dim]{ln.rstrip()}[/dim]")
                        continue
                    
                    elif cmd in ('/help', '/?'):
                        help_text = """[dim]Commands:
  /sessions          - List all sessions
  /session <name>    - Switch to session (creates if new)
  /delete <name>     - Delete a session
  /clear             - Clear conversation history
  /compact [strat]   - Remove older messages (half/quarter/last2)
  /tokens            - Show token breakdown  
  /config            - Show current API configuration
  /mode [fast|normal] - Show or switch reasoning mode
  /iter [n]          - Show or set max iterations
  /clip [question]   - Analyze image from clipboard
  /todo              - Show todo list
  /todo add <title>  - Add a todo
  /todo done <id>    - Mark todo as completed
  /todo rm <id>      - Remove a todo
  /todo clear        - Clear all todos
  /smart             - Show smart context analysis
  /dump              - Dump full model context to JSON file
  /save              - Save current session
  /history           - Show message count
  /bg                - List background processes
  /ctx               - Show context container
  /log [n]           - Show last n lines of harness.log (default 30)
  /exit              - Save and exit

Logging: full observability log at .harness_output/harness.log (always on)

Keys during agent work:
  Esc                - Stop/interrupt (responds within ~300ms)
  Ctrl+B             - Send running command to background
  Ctrl+C             - Interrupt (press twice within 2s to exit)"""
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
                
                log.info("User input [mode=%s]: %s", agent.reasoning_mode, truncate(user_input, 200))
                try:
                    result = loop.run_until_complete(run_single(agent, user_input, console))
                    log.info("run_single completed, result_len=%d", len(result or ""))
                    # Show live todo panel if there are any todos
                    agent.todo_manager.print_todo_panel(console)
                except KeyboardInterrupt:
                    log.warning("KeyboardInterrupt during run_single")
                    console.print("\n[yellow][STOP] Interrupted - Ctrl+C again to exit[/yellow]")
                    last_interrupt_time = time.time()
                except Exception as e:
                    log_exception(log, "run_single failed", e)
                    console.print(f"[red]Error: {rich_escape(str(e))}[/red]")
                
                # Auto-save session after each exchange
                agent.save_session(str(session_path))
                print()  # Blank line between requests
                
            except KeyboardInterrupt:
                cleanup_and_save()
                console.print("\n[dim]Session saved. Exiting...[/dim]")
                break
            except EOFError:
                cleanup_and_save()
                break
        
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass


if __name__ == "__main__":
    main()
