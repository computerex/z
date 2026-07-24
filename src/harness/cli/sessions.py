"""Session listing and path helpers."""
import os
from datetime import datetime
from pathlib import Path


def get_sessions_dir(workspace: str) -> Path:
    """Get the sessions directory for a workspace."""
    return Path(workspace) / ".sessions"


def get_session_path(workspace: str, session_name: str = "default") -> Path:
    """Get session file path for a workspace and session name."""
    return get_sessions_dir(workspace) / f"{session_name}.json"


def list_sessions(workspace: str) -> list:
    """List all sessions for a workspace."""
    sessions_dir = get_sessions_dir(workspace)
    if not sessions_dir.exists():
        return []

    sessions = []
    for f in sorted(sessions_dir.glob("*.json"), reverse=True):
        if f.name.startswith("_"):
            continue
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            import json
            data = json.loads(f.read_text(encoding="utf-8"))
            count = len(data.get("messages", []))
        except Exception:
            count = 0
        name = f.stem
        sessions.append((name, mtime, count))

    return sessions
