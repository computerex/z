"""Centralized observability logger for the harness.

Writes a structured, always-on log to .harness_output/harness.log.
The log captures every significant event so that when something goes
wrong, the full timeline can be reconstructed after the fact.

Usage in any module:
    from .logger import get_logger
    log = get_logger(__name__)
    log.info("something happened", extra={"key": "value"})

The log file rotates at 5 MB and keeps the last 5 files.
"""

import logging
import os
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Singleton state ──────────────────────────────────────────

_initialized = False
_log_dir: Optional[Path] = None
_session_id: Optional[str] = None


def _ensure_log_dir() -> Path:
    """Return (and create) the log directory."""
    global _log_dir
    if _log_dir is not None:
        return _log_dir
    # Default: workspace-relative .harness_output/
    # Caller can override via init_logging()
    _log_dir = Path.cwd() / ".harness_output"
    _log_dir.mkdir(parents=True, exist_ok=True)
    return _log_dir


def init_logging(
    workspace: Optional[str] = None,
    session_id: Optional[str] = None,
    level: int = logging.DEBUG,
) -> None:
    """Initialise the file logger.  Safe to call more than once."""
    global _initialized, _log_dir, _session_id

    if workspace:
        _log_dir = Path(workspace) / ".harness_output"
        _log_dir.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_log_dir()

    _session_id = session_id

    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("harness")
    root.setLevel(level)

    # Avoid duplicate handlers if init is called twice
    if root.handlers:
        return

    log_path = _log_dir / "harness.log"

    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    root.addHandler(handler)

    # Also log to stderr if HARNESS_DEBUG is set (useful during development)
    if os.environ.get("HARNESS_DEBUG"):
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.DEBUG)
        stderr_handler.setFormatter(fmt)
        root.addHandler(stderr_handler)

    root.info(
        "=== Logging initialised === pid=%d python=%s log=%s",
        os.getpid(),
        sys.version.split()[0],
        log_path,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'harness' namespace.

    Automatically initialises logging on first call so that even
    imports before init_logging() still get a working logger.
    """
    if not _initialized:
        # Lazy init with defaults — will be re-configured later
        init_logging()
    return logging.getLogger(f"harness.{name}")


# ── Convenience helpers ──────────────────────────────────────

def log_exception(logger: logging.Logger, msg: str, exc: BaseException) -> None:
    """Log an exception with full traceback."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error("%s: %s\n%s", msg, exc, "".join(tb))


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate a string for log readability."""
    if not text:
        return "(empty)"
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...[{len(text)} chars]"
