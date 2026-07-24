"""Session listing and path helpers."""
import os
from pathlib import Path

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


class HarnessCompleter(Completer):
    """Tab completer for commands, file paths, and shell commands."""

    COMMANDS = [
        "/sessions",
        "/session",
        "/new",
        "/delete",
        "/fork",
        "/agents",
        "/agent",
        "/agent-back",
        "/clear",
        "/save",
        "/history",
        "/bg",
        "/ctx",
        "/tokens",
        "/compact",
        "/cost",
        "/usage",
        "/maxctx",
        "/todo",
        "/smart",
        "/dump",
        "/config",
        "/providers",
        "/model",
        "/iter",
        "/clip",
        "/index",
        "/log",
        "/mcp",
        "/compactthresh",
        "/help",
        "/-",
        "/exit",
        "/quit",
        "/q",
    ]

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        # Shell command completion with ! prefix
        if text.startswith("!"):
            cmd_part = text[1:].strip()
            if not cmd_part:
                # Just typed !, show common shell commands
                common_cmds = [
                    "ls",
                    "cd",
                    "pwd",
                    "git",
                    "npm",
                    "pip",
                    "python",
                    "node",
                    "cat",
                    "grep",
                    "find",
                ]
                for cmd in common_cmds:
                    yield Completion(
                        f"!{cmd}",
                        start_position=-len(text),
                        display=cmd,
                    )
            else:
                # Complete file paths for shell commands
                if " " in cmd_part:
                    # Completing a path argument
                    prefix = cmd_part.split()[-1]
                    try:
                        import glob

                        # Handle glob patterns
                        if "*" in prefix or "-" in prefix:
                            matches = glob.glob(prefix, recursive=False)
                        else:
                            # Complete from current directory
                            matches = glob.glob(prefix + "*", recursive=False)
                        for match in sorted(matches):
                            display = match + ("/" if os.path.isdir(match) else "")
                            yield Completion(
                                f"!{cmd_part.rsplit(' ', 1)[0]} {match}",
                                start_position=-len(prefix),
                                display=display,
                            )
                    except Exception:
                        pass
                else:
                    # Complete the command itself
                    common_cmds = [
                        "ls",
                        "cd",
                        "pwd",
                        "git",
                        "npm",
                        "pip",
                        "python",
                        "node",
                        "cat",
                        "grep",
                        "find",
                        "rm",
                        "cp",
                        "mv",
                        "mkdir",
                        "touch",
                        "echo",
                        "clear",
                    ]
                    for cmd in common_cmds:
                        if cmd.startswith(cmd_part):
                            yield Completion(
                                f"!{cmd}",
                                start_position=-len(text),
                                display=cmd,
                            )
            return

        # Note: history-based ghost text is handled by AutoSuggestFromHistory
        # (the inline gray suggestion). Don't duplicate it here in the completer
        # because completer dropdown menus suppress the ghost text display.

        # Complete commands
        if text.startswith("/"):
            # Get the partial command
            parts = text.split()
            if len(parts) == 1:
                # Completing the command itself
                prefix = text
                for cmd in self.COMMANDS:
                    if cmd.startswith(prefix):
                        yield Completion(
                            cmd,
                            start_position=-len(prefix),
                            display=cmd,
                        )
            elif parts[0] in ["/session", "/delete", "/fork", "/agent"]:
                # Complete session names
                sessions_dir = get_sessions_dir(str(self.workspace))
                if sessions_dir.exists():
                    prefix = parts[-1]
                    for session_file in sessions_dir.glob("*.json"):
                        session_name = session_file.stem
                        if session_name.startswith(prefix):
                            yield Completion(
                                session_name,
                                start_position=-len(prefix),
                                display=session_name,
                            )
            elif parts[0] == "/todo" and len(parts) == 2:
                # Complete todo subcommands
                subcommands = ["add", "done", "rm", "clear"]
                prefix = parts[1]
                for sub in subcommands:
                    if sub.startswith(prefix):
                        yield Completion(
                            sub,
                            start_position=-len(prefix),
                            display=sub,
                        )
            elif parts[0] == "/compact":
                # LLM-based compaction — no strategies needed
                return
            elif parts[0] == "/index":
                # Complete index subcommands
                subcommands = ["rebuild", "tree"]
                prefix = parts[-1]
                for sub in subcommands:
                    if sub.startswith(prefix):
                        yield Completion(
                            sub,
                            start_position=-len(prefix),
                            display=sub,
                        )
        else:
            # Complete file paths
            # Get the last word (potential file path)
            words = text.split()
            if words:
                last_word = words[-1]
                # Check if it looks like a file path (contains / or \ or .)
                if "/" in last_word or "\\" in last_word or "." in last_word:
                    # Try to complete as a path
                    try:
                        # Handle both Unix and Windows paths
                        if "\\" in last_word:
                            # Windows path
                            parts = last_word.rsplit("\\", 1)
                            dir_part = parts[0] if len(parts) > 1 else "."
                            prefix = parts[1] if len(parts) > 1 else last_word
                            sep = "\\"
                        else:
                            # Unix path or relative
                            parts = last_word.rsplit("/", 1)
                            dir_part = parts[0] if len(parts) > 1 else "."
                            prefix = parts[1] if len(parts) > 1 else last_word
                            sep = "/"

                        # Resolve directory relative to workspace
                        try:
                            search_dir = (self.workspace / dir_part).resolve()
                        except:
                            search_dir = Path.cwd()

                        if search_dir.exists() and search_dir.is_dir():
                            for item in sorted(search_dir.iterdir()):
                                if item.name.startswith(prefix):
                                    display_name = item.name
                                    if item.is_dir():
                                        display_name += sep
                                    yield Completion(
                                        item.name,
                                        start_position=-len(prefix),
                                        display=display_name,
                                    )
                    except Exception:
                        pass

