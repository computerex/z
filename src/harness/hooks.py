"""Hook system — PreToolUse, PostToolUse, SessionStart, SessionEnd, and other lifecycle events."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook events
# ---------------------------------------------------------------------------


class HookEvent(str, Enum):
    PreToolUse = "PreToolUse"
    PostToolUse = "PostToolUse"
    PostToolUseFailure = "PostToolUseFailure"
    SessionStart = "SessionStart"
    SessionEnd = "SessionEnd"
    UserPromptSubmit = "UserPromptSubmit"
    InstructionsLoaded = "InstructionsLoaded"


# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------


class HookCommandType(str, Enum):
    command = "command"
    prompt = "prompt"
    agent = "agent"
    http = "http"
    callback = "callback"
    function = "function"


@dataclass
class HookCommand:
    """A single hook command configuration."""

    type: HookCommandType

    # For command hooks
    command: Optional[str] = None

    # For prompt/agent hooks
    prompt: Optional[str] = None

    # For HTTP hooks
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    allowed_env_vars: Optional[List[str]] = None

    # Common fields
    shell: Optional[str] = None  # "bash" or "powershell"
    timeout: Optional[int] = None  # seconds
    async_: Optional[bool] = None  # "async" is a Python keyword
    async_rewake: Optional[bool] = None
    if_: Optional[str] = None  # Permission rule syntax for filtering

    # For callback hooks (internal)
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.shell is None and self.type == HookCommandType.command:
            self.shell = "bash"


@dataclass
class HookMatcher:
    """A hook matcher: pattern + list of hook commands."""

    matcher: Optional[str]  # Pattern to match (e.g., tool name like "Write")
    hooks: List[HookCommand]


@dataclass
class HookResult:
    """Result of a single hook execution."""

    command: str
    succeeded: bool
    output: str
    blocked: bool = False
    exit_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _find_settings_files() -> List[Tuple[str, Path]]:
    """Find all .claude/settings.json files.

    Returns list of (source_name, path) tuples in priority order (lowest first).
    """
    files: List[Tuple[str, Path]] = []

    # User settings
    user_settings = Path.home() / ".claude" / "settings.json"
    if user_settings.exists():
        files.append(("user", user_settings))

    # Project settings: walk from CWD up to root
    current = Path.cwd().resolve()
    while True:
        project_settings = current / ".claude" / "settings.json"
        if project_settings.exists():
            files.append(("project", project_settings))
        parent = current.parent
        if parent == current:
            break
        current = parent

    return files


def _load_hooks_config() -> Dict[str, List[HookMatcher]]:
    """Load and merge hooks configuration from all settings files.

    Later files override earlier ones.
    """
    merged: Dict[str, List[HookMatcher]] = {}

    for source_name, settings_path in _find_settings_files():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Failed to load hooks from %s: %s", settings_path, e)
            continue

        hooks_data = data.get("hooks", {})
        if not hooks_data:
            continue

        for event_key, matchers_list in hooks_data.items():
            if event_key not in merged:
                merged[event_key] = []

            for matcher_data in matchers_list:
                matcher_pattern = matcher_data.get("matcher")
                hooks_list = matcher_data.get("hooks", [])

                parsed_hooks = []
                for h in hooks_list:
                    hook = _parse_hook_command(h)
                    if hook:
                        parsed_hooks.append(hook)

                if parsed_hooks:
                    merged[event_key].append(HookMatcher(
                        matcher=matcher_pattern,
                        hooks=parsed_hooks,
                    ))

    return merged


def _parse_hook_command(data: dict) -> Optional[HookCommand]:
    """Parse a hook command from config data."""
    hook_type = data.get("type", "command")
    if hook_type not in ("command", "prompt", "agent", "http"):
        logger.debug("Unknown hook type: %s", hook_type)
        return None

    try:
        ht = HookCommandType(hook_type)
    except ValueError:
        return None

    kwargs: dict = {
        "type": ht,
        "shell": data.get("shell"),
        "timeout": data.get("timeout"),
        "async_": data.get("async"),
        "async_rewake": data.get("asyncRewake"),
        "if_": data.get("if"),
    }

    if ht == HookCommandType.command:
        kwargs["command"] = data.get("command", "")
    elif ht in (HookCommandType.prompt, HookCommandType.agent):
        kwargs["prompt"] = data.get("prompt", "")
        kwargs["model"] = data.get("model")
    elif ht == HookCommandType.http:
        kwargs["url"] = data.get("url", "")
        kwargs["headers"] = data.get("headers")
        kwargs["allowed_env_vars"] = data.get("allowedEnvVars")

    return HookCommand(**{k: v for k, v in kwargs.items() if v is not None})


# ---------------------------------------------------------------------------
# Hook matching
# ---------------------------------------------------------------------------


def _tool_name_matches_pattern(tool_name: str, pattern: Optional[str]) -> bool:
    """Check if a tool name matches a matcher pattern.

    Uses fnmatch-style matching. Pattern=None means "match all".
    """
    if pattern is None or pattern == "":
        return True

    import fnmatch
    return fnmatch.fnmatch(tool_name.lower(), pattern.lower())


def _parse_if_condition(if_str: str) -> Tuple[str, str]:
    """Parse an 'if' condition like 'Bash(git *)' into (tool_name, pattern).

    Returns (tool_name, arg_pattern).
    """
    match = re.match(r"^(\w+)\((.+)\)$", if_str.strip())
    if match:
        return match.group(1), match.group(2)
    return if_str.strip(), "*"


def _check_if_condition(
    if_str: Optional[str],
    tool_name: str,
    tool_input: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if an 'if' condition matches the current tool call.

    The 'if' condition uses syntax like:
      "Bash(git *)"  — matches only when tool is "Bash" and command starts with "git"
      "Write(*.ts)"   — matches only when tool is "Write" and file path ends with .ts
      "Read"          — matches only when tool is "Read" (any args)
    """
    if if_str is None or if_str.strip() == "":
        return True

    import fnmatch

    cond_tool, cond_pattern = _parse_if_condition(if_str)

    # Check tool name
    if not fnmatch.fnmatch(tool_name.lower(), cond_tool.lower()):
        return False

    # Check argument pattern if we have tool_input
    if cond_pattern != "*" and tool_input:
        # Join all string values in tool_input to check against pattern
        all_values = " ".join(str(v) for v in tool_input.values() if isinstance(v, (str, list)))
        if not fnmatch.fnmatch(all_values.lower(), cond_pattern.lower()):
            return False

    return True


def get_matching_hooks(
    event: HookEvent,
    match_query: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_input: Optional[Dict[str, Any]] = None,
) -> List[HookCommand]:
    """Get all hooks matching an event and optional query.

    Args:
        event: The hook event.
        match_query: The value to match against matcher patterns.
        tool_name: The tool name (for 'if' condition matching).
        tool_input: The tool input (for 'if' condition matching).

    Returns:
        List of matched HookCommand objects.
    """
    config = _load_hooks_config()
    matchers = config.get(event.value, [])

    matched: List[HookCommand] = []

    for matcher in matchers:
        # Check matcher pattern against match_query
        if match_query is not None:
            if not _tool_name_matches_pattern(match_query, matcher.matcher):
                continue

        for hook in matcher.hooks:
            # Check 'if' condition
            if not _check_if_condition(hook.if_, tool_name or match_query or "", tool_input):
                continue

            matched.append(hook)

    return matched


def has_hook_for_event(event: HookEvent) -> bool:
    """Check if any hooks are configured for the given event."""
    config = _load_hooks_config()
    return event.value in config and len(config[event.value]) > 0


# ---------------------------------------------------------------------------
# Hook execution
# ---------------------------------------------------------------------------

DEFAULT_HOOK_TIMEOUT_MS = 10 * 60 * 1000  # 10 minutes


async def execute_hooks(
    event: HookEvent,
    hook_input: Dict[str, Any],
    match_query: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_input: Optional[Dict[str, Any]] = None,
    timeout_ms: int = DEFAULT_HOOK_TIMEOUT_MS,
) -> List[HookResult]:
    """Execute all hooks matching an event.

    Hooks run in parallel with individual timeouts.

    Args:
        event: The hook event.
        hook_input: JSON-serializable input to pass to hooks.
        match_query: Value to match against matcher patterns.
        tool_name: Tool name for 'if' condition filtering.
        tool_input: Tool input for 'if' condition filtering.
        timeout_ms: Default timeout for each hook.

    Returns:
        List of HookResult objects.
    """
    hooks = get_matching_hooks(event, match_query, tool_name, tool_input)
    if not hooks:
        return []

    json_input = json.dumps(hook_input, default=str)

    tasks = []
    for i, hook in enumerate(hooks):
        task = _execute_single_hook(hook, json_input, event, timeout_ms)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append(HookResult(
                command=str(hooks[i].command or hooks[i].prompt or hooks[i].url or ""),
                succeeded=False,
                output=str(result),
                blocked=False,
            ))
        else:
            final_results.append(result)

    return final_results


async def _execute_single_hook(
    hook: HookCommand,
    json_input: str,
    event: HookEvent,
    timeout_ms: int,
) -> HookResult:
    """Execute a single hook command."""
    # Handle async hooks — fire and forget
    if hook.async_ is True or hook.async_rewake is True:
        return await _execute_async_hook(hook, json_input, event)

    hook_timeout = (hook.timeout or 0) * 1000 or timeout_ms

    if hook.type == HookCommandType.command:
        return await _exec_command_hook(hook, json_input, hook_timeout)
    elif hook.type == HookCommandType.prompt:
        return await _exec_prompt_hook(hook, json_input, hook_timeout)
    elif hook.type == HookCommandType.agent:
        return await _exec_agent_hook(hook, json_input, hook_timeout)
    elif hook.type == HookCommandType.http:
        return await _exec_http_hook(hook, json_input, hook_timeout)
    elif hook.type == HookCommandType.callback:
        return await _exec_callback_hook(hook, json_input, hook_timeout)
    elif hook.type == HookCommandType.function:
        return HookResult(
            command="function",
            succeeded=False,
            output="Function hooks not supported outside REPL context",
            blocked=False,
        )

    return HookResult(
        command=str(hook.command or hook.prompt or hook.url or ""),
        succeeded=False,
        output=f"Unknown hook type: {hook.type}",
        blocked=False,
    )


def _prepare_command(cmd: str, json_input: str, shell: str = "bash") -> List[str]:
    """Prepare a shell command with $ARGUMENTS substitution."""
    cmd = cmd.replace("$ARGUMENTS", json_input)

    if shell == "powershell":
        return ["powershell", "-Command", cmd]
    else:
        # For bash, use -c
        return ["bash", "-c", cmd]


async def _exec_command_hook(
    hook: HookCommand,
    json_input: str,
    timeout_ms: int,
) -> HookResult:
    """Execute a command-type hook."""
    if not hook.command:
        return HookResult(command="", succeeded=False, output="No command specified", blocked=False)

    try:
        args = _prepare_command(hook.command, json_input, hook.shell or "bash")
        timeout_s = timeout_ms / 1000

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            return HookResult(
                command=hook.command,
                succeeded=False,
                output=f"Hook timed out after {timeout_s}s",
                blocked=False,
            )

        out_text = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace")
        combined = (out_text + err_text).strip()

        exit_code = proc.returncode or 0

        # Exit code semantics:
        #   0 → success (output not shown)
        #   2 → blocking error (show stderr to model)
        #   other → error (show to user only)
        blocked = exit_code == 2

        return HookResult(
            command=hook.command,
            succeeded=exit_code == 0,
            output=combined,
            blocked=blocked,
            exit_code=exit_code,
        )

    except Exception as e:
        return HookResult(
            command=hook.command,
            succeeded=False,
            output=str(e),
            blocked=False,
        )




_rewake_hooks: List[Dict[str, Any]] = []


def _track_rewake_hook(
    proc: asyncio.subprocess.Process,
    hook: HookCommand,
    event: HookEvent,
):
    """Track an asyncRewake hook so its exit code can be checked later."""
    hook_id = str(uuid.uuid4())[:8]
    _rewake_hooks.append({
        "id": hook_id,
        "proc": proc,
        "hook": hook,
        "event": event,
        "started_at": time.time(),
    })

    # Background task to check exit code
    async def _check():
        try:
            stdout, stderr = await proc.communicate()
            if proc.returncode == 2:
                out_text = stdout.decode("utf-8", errors="replace")
                err_text = stderr.decode("utf-8", errors="replace")
                logger.info(
                    "asyncRewake hook '%s' returned exit code 2: %s",
                    hook.command, err_text or out_text,
                )
        except Exception:
            pass
        finally:
            # Remove from tracking
            for i, h in enumerate(_rewake_hooks):
                if h["id"] == hook_id:
                    _rewake_hooks.pop(i)
                    break

    asyncio.ensure_future(_check())


# ---------------------------------------------------------------------------
# Prompt hook execution
# ---------------------------------------------------------------------------


async def _exec_prompt_hook(
    hook: HookCommand,
    json_input: str,
    timeout_ms: int,
) -> HookResult:
    """Execute a prompt-type hook using an LLM call.

    This requires the hook to have a configured provider/model.
    """
    return HookResult(
        command=hook.prompt or "",
        succeeded=False,
        output="Prompt hooks are not yet supported outside REPL context",
        blocked=False,
    )


# ---------------------------------------------------------------------------
# Agent hook execution
# ---------------------------------------------------------------------------


async def _exec_agent_hook(
    hook: HookCommand,
    json_input: str,
    timeout_ms: int,
) -> HookResult:
    """Execute an agent-type hook (spawn sub-agent)."""
    return HookResult(
        command=hook.prompt or "",
        succeeded=False,
        output="Agent hooks are not yet supported outside REPL context",
        blocked=False,
    )


# ---------------------------------------------------------------------------
# HTTP hook execution
# ---------------------------------------------------------------------------


async def _exec_http_hook(
    hook: HookCommand,
    json_input: str,
    timeout_ms: int,
) -> HookResult:
    """Execute an HTTP-type hook (POST to URL)."""
    if not hook.url:
        return HookResult(command="", succeeded=False, output="No URL specified", blocked=False)

    import urllib.request
    import urllib.error

    headers = dict(hook.headers or {})
    headers.setdefault("Content-Type", "application/json")

    # Interpolate env vars in header values
    allowed = set(hook.allowed_env_vars or [])
    if allowed:
        for key, val in list(headers.items()):
            for env_name in allowed:
                env_val = os.environ.get(env_name, "")
                if env_val:
                    headers[key] = headers[key].replace(f"${env_name}", env_val)
                    headers[key] = headers[key].replace(f"${{{env_name}}}", env_val)

    timeout_s = (hook.timeout or 30)

    try:
        data = json_input.encode("utf-8")
        req = urllib.request.Request(hook.url, data=data, headers=headers, method="POST")

        loop = asyncio.get_event_loop()

        def _do_request():
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    return resp.status, body, None
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                return e.code, body, None
            except Exception as e:
                return 0, "", str(e)

        status, body, error = await loop.run_in_executor(None, _do_request)

        if error:
            return HookResult(
                command=hook.url,
                succeeded=False,
                output=error,
                blocked=False,
            )

        ok = 200 <= status < 300
        return HookResult(
            command=hook.url,
            succeeded=ok,
            output=body if not ok else "",
            blocked=False,
        )

    except Exception as e:
        return HookResult(
            command=hook.url,
            succeeded=False,
            output=str(e),
            blocked=False,
        )


# ---------------------------------------------------------------------------
# Callback hook execution
# ---------------------------------------------------------------------------


async def _exec_callback_hook(
    hook: HookCommand,
    json_input: str,
    timeout_ms: int,
) -> HookResult:
    """Execute a callback-type hook (internal Python callback)."""
    if not hook.callback:
        return HookResult(
            command="callback",
            succeeded=False,
            output="No callback provided",
            blocked=False,
        )

    try:
        result = hook.callback(json_input)
        if asyncio.iscoroutine(result):
            result = await result

        return HookResult(
            command="callback",
            succeeded=True,
            output=result or "",
            blocked=False,
        )
    except Exception as e:
        return HookResult(
            command="callback",
            succeeded=False,
            output=str(e),
            blocked=False,
        )


# ---------------------------------------------------------------------------
# Specific hook dispatchers
# ---------------------------------------------------------------------------


async def execute_pre_tool_use_hooks(
    tool_name: str,
    tool_input: Dict[str, Any],
) -> List[HookResult]:
    """Execute PreToolUse hooks."""
    return await execute_hooks(
        HookEvent.PreToolUse,
        hook_input={
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
        match_query=tool_name,
        tool_name=tool_name,
        tool_input=tool_input,
    )


async def execute_post_tool_use_hooks(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_response: str,
) -> List[HookResult]:
    """Execute PostToolUse hooks."""
    return await execute_hooks(
        HookEvent.PostToolUse,
        hook_input={
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "response": tool_response,
        },
        match_query=tool_name,
        tool_name=tool_name,
        tool_input=tool_input,
    )


async def execute_post_tool_use_failure_hooks(
    tool_name: str,
    tool_input: Dict[str, Any],
    error: str,
) -> List[HookResult]:
    """Execute PostToolUseFailure hooks."""
    return await execute_hooks(
        HookEvent.PostToolUseFailure,
        hook_input={
            "hook_event_name": "PostToolUseFailure",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": error,
        },
        match_query=tool_name,
        tool_name=tool_name,
        tool_input=tool_input,
    )


async def execute_session_start_hooks(source: str = "startup"):
    """Execute SessionStart hooks."""
    return await execute_hooks(
        HookEvent.SessionStart,
        hook_input={
            "hook_event_name": "SessionStart",
            "source": source,
        },
        match_query=source,
    )


async def execute_session_end_hooks(reason: str = "exit"):
    """Execute SessionEnd hooks."""
    return await execute_hooks(
        HookEvent.SessionEnd,
        hook_input={
            "hook_event_name": "SessionEnd",
            "reason": reason,
        },
        match_query=reason,
    )


async def execute_user_prompt_submit_hooks(user_input: str) -> List[HookResult]:
    """Execute UserPromptSubmit hooks."""
    return await execute_hooks(
        HookEvent.UserPromptSubmit,
        hook_input={
            "hook_event_name": "UserPromptSubmit",
            "user_prompt": user_input,
        },
    )

