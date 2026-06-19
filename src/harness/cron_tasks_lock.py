"""PID-based scheduler lock to prevent double-firing.

When two harness sessions share the same project directory, the first to
acquire the lock runs the scheduled-task checker.  The other session
probes periodically and takes over if the owner's PID is dead.
"""

import json
import os
import signal
import time
from pathlib import Path
from typing import Optional

LOCK_FILE_REL = Path(".claude", "scheduler.lock")


def get_lock_file_path(project_dir: Optional[str] = None) -> Path:
    base = Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
    return base / LOCK_FILE_REL


def try_acquire_scheduler_lock(
    project_dir: Optional[str] = None,
    identity: Optional[str] = None,
) -> bool:
    """Try to acquire the per-project scheduler lock.

    Writes ``{"pid": <pid>, "identity": <identity>, "acquiredAt": <epoch_ms>}``
    to the lock file.

    Succeeds when:
    - No lock file exists.
    - The PID in the lock file is no longer alive.
    - The PID matches our own PID (re-entrant).

    Returns ``True`` if the lock is now owned by this process.
    """
    path = get_lock_file_path(project_dir)
    my_pid = os.getpid()

    try:
        if path.exists():
            raw = path.read_text("utf-8")
            data = json.loads(raw)
            owner_pid = int(data.get("pid", 0))
            if owner_pid == my_pid:
                return True  # Already own it
            if _pid_is_alive(owner_pid):
                return False  # Someone else holds it
        # Owner is dead or no lock file — take ownership
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "pid": my_pid,
                "identity": identity or f"pid:{my_pid}",
                "acquiredAt": int(time.time() * 1000),
            }),
            "utf-8",
        )
        return True
    except Exception:
        return False


def release_scheduler_lock(project_dir: Optional[str] = None) -> None:
    """Remove the lock file if we own it."""
    path = get_lock_file_path(project_dir)
    try:
        if path.exists():
            raw = path.read_text("utf-8")
            data = json.loads(raw)
            if int(data.get("pid", 0)) == os.getpid():
                path.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform PID liveness check."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = test existence
        return True
    except OSError:
        return False
    except Exception:
        return False
