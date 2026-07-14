"""
Structured output protocol for machine-readable consumption.

Features:
- --json flag: emit structured JSON on stdout per turn
- NDJSON progress events on stderr
- Structured error protocol (error.json)
- Partial results on timeout (SIGTERM handler)
- Schema-driven output validation
"""

import json
import sys
import os
import time
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .logger import get_logger

log = get_logger("output_protocol")

# ── Global state ────────────────────────────────────────────────────────
_json_mode: bool = False
_output_meta: Dict[str, Any] = {}
_files_written: List[str] = []
_error_info: Optional[Dict[str, Any]] = None
_partial_state: Optional[Dict[str, Any]] = None
_sigterm_received: bool = False
_start_time: float = 0.0
_iteration_count: int = 0
_workspace: str = "."


# ── Initialisation ──────────────────────────────────────────────────────

def init_output_protocol(
    *,
    json_mode: bool = False,
    workspace: str = ".",
    model: str = "",
) -> None:
    """Initialise the output protocol module for a session."""
    global _json_mode, _workspace, _output_meta, _files_written
    global _error_info, _partial_state, _sigterm_received, _start_time
    global _iteration_count

    _json_mode = json_mode
    _workspace = workspace
    _output_meta = {"model": model}
    _files_written = []
    _error_info = None
    _partial_state = None
    _sigterm_received = False
    _start_time = time.time()
    _iteration_count = 0

    if _json_mode:
        _install_signal_handlers()


# ── Progress events (stderr NDJSON) ─────────────────────────────────────

def _emit_event(data: dict) -> None:
    """Write a single NDJSON event line to stderr."""
    try:
        line = json.dumps(data, ensure_ascii=False, default=str)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass  # Never let progress events crash the main process


def emit_progress(event: str, **kwargs) -> None:
    """Emit a progress event on stderr.

    Args:
        event: Event name (phase_start, phase_end, writing_file, error, etc.)
        **kwargs: Additional fields to include in the event.
    """
    payload = {"event": event, "timestamp": time.time()}
    payload.update(kwargs)
    _emit_event(payload)


# ── File tracking ───────────────────────────────────────────────────────

def track_file_written(path: str) -> None:
    """Record that a file was written during this turn."""
    if path not in _files_written:
        _files_written.append(path)


def get_files_written() -> List[str]:
    """Return list of files written this turn."""
    return list(_files_written)


def reset_files_written() -> None:
    """Reset file tracking for a new turn."""
    _files_written.clear()
    _error_info = None
    _partial_state = None
    _iteration_count = 0


# ── Iteration tracking ──────────────────────────────────────────────────

def set_iteration_count(n: int) -> None:
    """Set the current iteration count."""
    global _iteration_count
    _iteration_count = n


def get_iteration_count() -> int:
    """Get the current iteration count."""
    return _iteration_count


# ── Structured JSON output ──────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result of a single agent turn."""
    status: str  # "completed", "partial", "error"
    files_written: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


def emit_json_result(
    status: str,
    *,
    result_text: str = "",
    files_written: Optional[List[str]] = None,
    tokens_used: int = 0,
    iterations: int = 0,
    wall_ms: int = 0,
    model: str = "",
    error: Optional[Dict[str, Any]] = None,
    partial_iterations: int = 0,
    partial_reason: str = "",
) -> None:
    """Emit a final structured JSON result on stdout.

    Only emits when --json mode is enabled.  Writes a single JSON object
    per turn to stdout, terminated with a newline.
    """
    if not _json_mode:
        return

    if files_written is None:
        files_written = list(_files_written)

    if wall_ms == 0:
        wall_ms = int((time.time() - _start_time) * 1000)

    if model == "":
        model = _output_meta.get("model", "")

    result: Dict[str, Any] = {"status": status}

    if files_written:
        result["files_written"] = files_written

    if result_text:
        # Try to parse result_text as JSON, fall back to string
        try:
            parsed = json.loads(result_text)
            result["result"] = parsed
        except (json.JSONDecodeError, TypeError):
            result["result"] = result_text

    meta: Dict[str, Any] = {}
    if tokens_used:
        meta["tokens_used"] = tokens_used
    if iterations:
        meta["iterations"] = iterations
    if model:
        meta["model"] = model
    meta["wall_ms"] = wall_ms
    if partial_iterations:
        meta["completed_iterations"] = partial_iterations
    if partial_reason:
        meta["reason"] = partial_reason
    result["meta"] = meta

    if error:
        result["error"] = error

    try:
        line = json.dumps(result, ensure_ascii=False, default=str)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass  # Never let output formatting crash


def emit_error(error_type: str, **kwargs) -> None:
    """Emit a structured error for the current turn.

    Args:
        error_type: One of 'api_rate_limit', 'schema_violation',
                    'model_failure', 'bad_input', 'internal'
        **kwargs: Additional error fields (retryable, retry_after_seconds,
                  partial_result, message, detail)
    """
    global _error_info
    _error_info = {
        "error_type": error_type,
        "retryable": kwargs.get("retryable", False),
        "retry_after_seconds": kwargs.get("retry_after_seconds", 0),
        "message": kwargs.get("message", ""),
    }
    if "partial_result" in kwargs:
        _error_info["partial_result"] = kwargs["partial_result"]
    if "detail" in kwargs:
        _error_info["detail"] = kwargs["detail"]

    # Always write error.json in workspace if possible
    _write_error_json()


def _write_error_json() -> None:
    """Write error.json in the workspace directory."""
    if not _error_info or not _workspace:
        return
    try:
        wp = Path(_workspace)
        if not wp.exists():
            return
        error_path = wp / "error.json"
        error_path.write_text(
            json.dumps(_error_info, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def classify_api_error(exc: Exception) -> Dict[str, Any]:
    """Classify an API exception into structured error info.

    Args:
        exc: The exception to classify.

    Returns:
        Dict with error_type, retryable, retry_after_seconds, message, detail.
    """
    msg = str(exc)
    msg_lower = msg.lower()

    # Rate limit detection
    if any(phrase in msg_lower for phrase in (
        "rate limit", "too many requests", "429", "quota", "throttl",
        "try again later",
    )):
        # Try to extract retry-after from common patterns
        retry_after = 30
        import re
        m = re.search(r"retry[_-]?after[:\s]+(\d+)", msg_lower)
        if m:
            retry_after = int(m.group(1))
        else:
            m = re.search(r"(\d+)\s*seconds?", msg_lower)
            if m:
                retry_after = int(m.group(1))

        return {
            "error_type": "api_rate_limit",
            "retryable": True,
            "retry_after_seconds": retry_after,
            "message": msg[:500],
            "detail": _format_traceback(exc),
        }

    # Authentication errors
    if any(phrase in msg_lower for phrase in (
        "unauthorized", "401", "forbidden", "403", "auth", "invalid api key",
        "token", "credential",
    )):
        return {
            "error_type": "bad_input",
            "retryable": False,
            "retry_after_seconds": 0,
            "message": msg[:500],
            "detail": _format_traceback(exc),
        }

    # Model failures / bad responses
    if any(phrase in msg_lower for phrase in (
        "model", "stream", "parse", "json", "tool_call", "empty response",
        "invalid", "timeout", "timed out",
    )):
        return {
            "error_type": "model_failure",
            "retryable": True,
            "retry_after_seconds": 5,
            "message": msg[:500],
            "detail": _format_traceback(exc),
        }

    # Default: internal error
    return {
        "error_type": "internal",
        "retryable": False,
        "retry_after_seconds": 0,
        "message": msg[:500],
        "detail": _format_traceback(exc),
    }


def _format_traceback(exc: Exception) -> str:
    """Format exception traceback as string (limited length)."""
    import traceback
    tb = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return tb[:2000]


# ── SIGTERM / timeout handling ──────────────────────────────────────────

def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM by writing partial results."""
    global _sigterm_received, _partial_state

    _sigterm_received = True
    _partial_state = {
        "status": "partial",
        "completed_iterations": _iteration_count,
        "reason": "timeout",
        "files_written": list(_files_written),
        "timestamp": time.time(),
    }

    # Write partial result immediately
    try:
        if _workspace:
            wp = Path(_workspace)
            if wp.exists():
                partial_path = wp / "partial_result.json"
                partial_path.write_text(
                    json.dumps(_partial_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    except Exception:
        pass

    # Also emit via --json if enabled
    if _json_mode:
        emit_json_result(
            status="partial",
            files_written=list(_files_written),
            partial_iterations=_iteration_count,
            partial_reason="timeout",
        )

    # Re-raise KeyboardInterrupt to allow graceful shutdown
    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers for partial result on timeout."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _sigterm_handler)
        except (AttributeError, ValueError, OSError):
            pass  # Not available on all platforms (e.g. Windows threads)

    # On Windows, SIGTERM is not available but we can use atexit
    import atexit

    def _atexit_handler():
        global _partial_state, _files_written, _iteration_count
        if _partial_state is None and _files_written:
            _partial_state = {
                "status": "partial",
                "completed_iterations": _iteration_count,
                "reason": "process_exit",
                "files_written": list(_files_written),
                "timestamp": time.time(),
            }
            try:
                if _workspace:
                    wp = Path(_workspace)
                    if wp.exists():
                        partial_path = wp / "partial_result.json"
                        partial_path.write_text(
                            json.dumps(_partial_state, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
            except Exception:
                pass

    atexit.register(_atexit_handler)


def is_sigterm_received() -> bool:
    """Check if SIGTERM/SIGINT was received."""
    return _sigterm_received


def get_partial_state() -> Optional[Dict[str, Any]]:
    """Get the partial state if a timeout occurred."""
    return _partial_state


# ── Schema validation ───────────────────────────────────────────────────

def load_schema(workspace: str) -> Optional[Dict[str, Any]]:
    """Load schema.json from the workspace if it exists.

    Returns:
        Schema dict or None if no schema file found.
    """
    schema_path = Path(workspace) / "schema.json"
    if not schema_path.exists():
        return None
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        log.warning("Failed to load schema.json: %s", e)
        return None


def validate_against_schema(
    data: Any,
    schema: Dict[str, Any],
    schema_path: str = "",
) -> Tuple[bool, List[str]]:
    """Validate output data against a JSON schema.

    Supports a simplified schema format with:
      - "type": "object"
      - "properties": {...}
      - "required": [...]
      - "additionalProperties": bool

    Args:
        data: The data to validate (dict or parsed JSON).
        schema: The schema definition.
        schema_path: Current path in the schema (for nested validation).

    Returns:
        Tuple of (is_valid, list_of_missing_or_error_fields).
    """
    issues: List[str] = []

    if not isinstance(data, dict) or schema.get("type") != "object":
        return True, []

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in data or data[field] is None:
            field_path = f"{schema_path}.{field}" if schema_path else field
            issues.append(f"Missing required field: {field_path}")

    # Check property types
    for prop_name, prop_schema in (properties or {}).items():
        if prop_name not in data:
            continue
        value = data[prop_name]
        prop_type = prop_schema.get("type", "")

        if prop_type == "object" and isinstance(value, dict):
            sub_valid, sub_issues = validate_against_schema(
                value,
                prop_schema,
                schema_path=f"{schema_path}.{prop_name}" if schema_path else prop_name,
            )
            issues.extend(sub_issues)
        elif prop_type == "array" and isinstance(value, list):
            items_schema = prop_schema.get("items", {})
            if items_schema.get("type") == "object":
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        sub_valid, sub_issues = validate_against_schema(
                            item,
                            items_schema,
                            schema_path=f"{schema_path}.{prop_name}[{i}]"
                            if schema_path
                            else f"{prop_name}[{i}]",
                        )
                        issues.extend(sub_issues)
        elif prop_type == "string" and not isinstance(value, str):
            field_path = (
                f"{schema_path}.{prop_name}" if schema_path else prop_name
            )
            issues.append(
                f"Type mismatch for {field_path}: expected {prop_type}, got {type(value).__name__}"
            )
        elif prop_type == "number" and not isinstance(value, (int, float)):
            field_path = (
                f"{schema_path}.{prop_name}" if schema_path else prop_name
            )
            issues.append(
                f"Type mismatch for {field_path}: expected {prop_type}, got {type(value).__name__}"
            )

    return len(issues) == 0, issues


def fill_schema_gaps(
    data: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill missing required fields with diagnostic defaults.

    When schema validation finds gaps, this fills them with reasonable
    defaults so callers get a complete structure.

    Args:
        data: The original data dict.
        schema: The schema definition.

    Returns:
        Data dict with gaps filled.
    """
    import copy
    result = copy.deepcopy(data)

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in result or result[field] is None:
            prop_schema = properties.get(field, {})
            default = prop_schema.get("default", None)
            prop_type = prop_schema.get("type", "string")

            # Generate diagnostic default
            if default is not None:
                result[field] = default
            elif prop_type == "string":
                result[field] = f"[NOT ASSESSED: {field}]"
            elif prop_type == "number":
                result[field] = -1
            elif prop_type == "array":
                result[field] = []
            elif prop_type == "object":
                result[field] = {}
            elif prop_type == "boolean":
                result[field] = False
            else:
                result[field] = None

    return result
