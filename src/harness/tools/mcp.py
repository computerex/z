"""MCP (Model Context Protocol) handlers — server discovery, session management, tool dispatch."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..logger import get_logger
from ._base import HAS_MCP_SDK, log, _track_write

# ── Class-level constants ──────────────────────────────────────────────

OUTPUT_SPILL_TOKEN_THRESHOLD = 8000  # ~32,000 chars

# Tools that manage their own output size and must NOT be spilled.
# Spilling read_file results creates an infinite loop: the model reads
# the spill file which gets spilled again, making content invisible.
# search_files is capped at 100 matches internally — no need to spill.
_NO_SPILL_TOOLS = frozenset({"read_file", "search_files"})

# Maximum tokens to include inline when output is spilled.
OUTPUT_INLINE_PREVIEW_TOKENS = 300   # ~1,200 chars — enough for LLM to understand
LARGE_WRITE_FEEDBACK_THRESHOLD = 256 * 1024  # chars
LARGE_WRITE_CHUNK_SIZE = 128 * 1024  # chars


def __init__(
    self,
    config,
    console,
    workspace_path: str,
    context,
    duplicate_detector,
    context_manager=None,
    sub_agent_manager=None,
):
    """Initialize tool handlers with required dependencies.
    
    Args:
        config: Config object with API settings
        console: Rich console for output
        workspace_path: Path to workspace directory
        context: ContextContainer for managing loaded content
        duplicate_detector: DuplicateDetector for tracking file reads
        context_manager: SmartContextManager for accessing stored tool results
        sub_agent_manager: SubAgentManager for creating/managing sub-agents
    """
    self.config = config
    self.console = console
    self.workspace_path = workspace_path
    self.context = context
    self._duplicate_detector = duplicate_detector
    self._context_manager = context_manager
    self.sub_agent_manager = sub_agent_manager
    
    # Background processes: {id: {"proc": Process, "command": str, "started": float, "log_file": str, "task": Task}}
    self._background_procs: Dict[int, dict] = {}
    self._next_bg_id = 1
    self._next_cmd_id = 1  # For unique command log files
    
    # Directory for spilled command output files
    self._output_dir = os.path.join(workspace_path, ".harness_output")
    os.makedirs(self._output_dir, exist_ok=True)

    # Persistent MCP sessions (server name -> session bundle).
    # Keeps browser/tool state across turns (critical for Playwright MCP refs).
    self._mcp_sessions: Dict[str, Dict[str, Any]] = {}
    self._mcp_locks: Dict[str, asyncio.Lock] = {}


# -- Output spill helpers --------------------------------------------------

def spill_output_to_file(self, output: str, label: str) -> str:
    """Write large output to a file and return a compact reference.
    
    Returns the original output unchanged if it's below threshold,
    otherwise writes to .harness_output/ and returns a truncated preview
    with the file path so the model can read_file to inspect details.
    """
    est_tokens = len(output) // 4
    if est_tokens <= self.OUTPUT_SPILL_TOKEN_THRESHOLD:
        return output
    
    # Write full output to a file
    import hashlib
    import re as _re
    import time as _time
    safe_label = _re.sub(r'[^\w\-.]', '_', label)[:60]
    ts = int(_time.time())
    filename = f"{safe_label}_{ts}.txt"
    os.makedirs(self._output_dir, exist_ok=True)
    filepath = os.path.join(self._output_dir, filename)
    Path(filepath).write_text(output, encoding="utf-8")
    log.info("Output spilled to file: %s (%d tokens, %d chars)", filepath, est_tokens, len(output))
    
    # Build compact inline result
    lines = output.splitlines()
    total_lines = len(lines)
    preview_chars = self.OUTPUT_INLINE_PREVIEW_TOKENS * 4
    head = output[:preview_chars // 2]
    tail = output[-(preview_chars // 2):]
    
    return (
        f"[OUTPUT SPILLED TO FILE — {est_tokens:,} tokens, {total_lines} lines]\n"
        f"Full output saved to: {filepath}\n"
        f"Use read_file to inspect specific sections.\n\n"
        f"--- First lines ---\n{head}\n\n"
        f"--- Last lines ---\n{tail}"
    )


def _get_bg_log_path(self, proc_id: int) -> str:
    """Get the log file path for a background process."""
    os.makedirs(self._output_dir, exist_ok=True)
    return os.path.join(self._output_dir, f"bg_process_{proc_id}.log")


def _get_cmd_log_path(self) -> str:
    """Get a unique log file path for a foreground command."""
    os.makedirs(self._output_dir, exist_ok=True)
    cmd_id = self._next_cmd_id
    self._next_cmd_id += 1
    return os.path.join(self._output_dir, f"cmd_{cmd_id}.log")


def _resolve_path(self, path: str) -> Path:
    """Resolve a path relative to workspace."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(self.workspace_path) / p
    return p.resolve()


def _load_mcp_servers(self) -> Dict[str, dict]:
    cfg_path = Path.home() / ".z.json"
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        mcp = data.get("mcp", {})
        return mcp if isinstance(mcp, dict) else {}
    except Exception:
        return {}


def _mcp_server_cfg(self, name: str) -> Tuple[dict, str]:
    servers = _load_mcp_servers(self)
    cfg = servers.get(name)
    if not isinstance(cfg, dict):
        return {}, f"MCP server '{name}' not found in ~/.z.json mcp config."
    if cfg.get("enabled", True) is False:
        return {}, f"MCP server '{name}' is disabled."
    stype = str(cfg.get("type", "local")).lower()
    if stype in ("local", "stdio"):
        cmd = cfg.get("command")
        if not isinstance(cmd, list) or not cmd:
            return {}, f"MCP server '{name}' has invalid command configuration."
    elif stype in ("http", "streamable_http", "sse"):
        url = cfg.get("url")
        if not isinstance(url, str) or not url.strip():
            return {}, f"MCP server '{name}' has invalid URL configuration."
    else:
        return {}, f"MCP server '{name}' has unsupported type '{stype}'."
    return cfg, ""


def _normalize_mcp_command(self, cmd: List[str]) -> List[str]:
    """Rewrite known noisy MCP launchers into protocol-safe equivalents."""
    if len(cmd) >= 2 and cmd[0] == "uvx" and cmd[1] == "minimax-coding-plan-mcp":
        # minimax-coding-plan-mcp prints a banner to stdout before protocol
        # frames, which breaks MCP SDK parsing. Run its module directly.
        return [
            "uvx",
            "--from",
            "minimax-coding-plan-mcp",
            "python",
            "-c",
            "from minimax_mcp.server import mcp; mcp.run()",
        ]
    return cmd


async def _close_mcp_session(self, name: str) -> None:
    entry = self._mcp_sessions.pop(name, None)
    if not entry:
        return
    try:
        session_cm = entry.get("session_cm")
        if session_cm is not None:
            await session_cm.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        cm = entry.get("cm")
        if cm is not None:
            await cm.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        errlog = entry.get("errlog")
        if errlog is not None:
            errlog.close()
    except Exception:
        pass


_mcp_sse_suppressor_installed = False


def _mcp_suppress_reader_errors(self):
    """Install a permanent asyncio exception handler that filters out
    ``sse_reader`` errors from the MCP SDK's internal background task.

    The MCP SDK starts internal background reader tasks (``sse_reader``)
    that aren't properly awaited.  When the server closes the SSE connection,
    the task crashes with a traceback printed by Python's asyncio event loop.
    The handler is installed once and stays for the event loop's lifetime
    (the sse_reader task outlives any single session creation context).
    """
    # Idempotent: install once.
    global _mcp_sse_suppressor_installed
    if _mcp_sse_suppressor_installed:
        return
    loop = asyncio.get_running_loop()
    _orig = loop.get_exception_handler()

    def _handler(loop, context):
        task = context.get("task")
        msg = context.get("message", "")
        if task is not None and "sse_reader" in str(task):
            return
        if "sse_reader" in msg:
            return
        if _orig is not None:
            _orig(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    _mcp_sse_suppressor_installed = True


async def _get_or_create_mcp_session(self, name: str, cfg: dict):
    # Suppress noisy non-JSON-line errors from MCP SDK's background readers
    # (e.g. when an MCP server prints timestamped log lines to stdout/SSE).
    # Our retry logic handles transient failures transparently.
    for _mcp_logger in ("mcp.client.stdio", "mcp.client.sse", "httpx_sse"):
        logging.getLogger(_mcp_logger).setLevel(logging.ERROR)

    if not HAS_MCP_SDK:
        raise RuntimeError(
            "MCP SDK is not installed. Install package 'mcp' to enable MCP server execution."
        )
    cfg_key = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    existing = self._mcp_sessions.get(name)
    if existing and existing.get("cfg_key") == cfg_key:
        return existing.get("session")
    if existing:
        await _close_mcp_session(self, name)

    from ._base import StdioServerParameters, ClientSession, stdio_client, sse_client, streamablehttp_client

    stype = str(cfg.get("type", "local")).lower()

    if stype in ("local", "stdio"):
        cmd = [str(x) for x in (cfg.get("command", []) or [])]
        if not cmd:
            raise RuntimeError("MCP server command is empty.")
        cmd = _normalize_mcp_command(self, cmd)
        env_cfg = cfg.get("environment", {})
        env = os.environ.copy()
        if isinstance(env_cfg, dict):
            env.update({str(k): str(v) for k, v in env_cfg.items()})

        server_params = StdioServerParameters(
            command=cmd[0],
            args=cmd[1:],
            env=env,
        )

        _errlog = open(os.devnull, "w", encoding="utf-8")
        cm = stdio_client(server_params, errlog=_errlog)
        read_stream, write_stream = await cm.__aenter__()
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        try:
            await asyncio.wait_for(session.initialize(), timeout=15)
        except Exception:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                _errlog.close()
            except Exception:
                pass
            raise
        self._mcp_sessions[name] = {
            "cfg_key": cfg_key,
            "stype": stype,
            "cm": cm,
            "session_cm": session_cm,
            "session": session,
            "errlog": _errlog,
        }
        return session

    if stype in ("http", "streamable_http", "sse"):
        url = str(cfg.get("url", "") or "").strip()
        if not url:
            raise RuntimeError("MCP HTTP server URL is empty.")
        headers_cfg = cfg.get("headers", {})
        headers: Dict[str, str] = {}
        if isinstance(headers_cfg, dict):
            headers = {str(k): str(v) for k, v in headers_cfg.items()}

        if stype == "sse":
            _mcp_suppress_reader_errors(self)
            cm = sse_client(url, headers=headers, timeout=20, sse_read_timeout=300)
            read_stream, write_stream = await cm.__aenter__()
            session_cm = ClientSession(read_stream, write_stream)
            session = await session_cm.__aenter__()
            try:
                await asyncio.wait_for(session.initialize(), timeout=20)
            except Exception:
                try:
                    await session_cm.__aexit__(None, None, None)
                except Exception:
                    pass
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass
                raise
            self._mcp_sessions[name] = {
                "cfg_key": cfg_key,
                "stype": stype,
                "cm": cm,
                "session_cm": session_cm,
                "session": session,
            }
            return session

        cm = streamablehttp_client(url, headers=headers, timeout=20, sse_read_timeout=300)
        read_stream, write_stream, _get_session_id = await cm.__aenter__()
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        try:
            await asyncio.wait_for(session.initialize(), timeout=20)
        except Exception:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise
        self._mcp_sessions[name] = {
            "cfg_key": cfg_key,
            "stype": stype,
            "cm": cm,
            "session_cm": session_cm,
            "session": session,
        }
        return session

    raise RuntimeError(f"Unsupported MCP server type: {stype}")


async def _mcp_with_sdk_session(self, name: str, cfg: dict, fn):
    lock = self._mcp_locks.setdefault(name, asyncio.Lock())
    async with lock:
        session = await _get_or_create_mcp_session(self, name, cfg)
        try:
            return await asyncio.wait_for(fn(session), timeout=45)
        except Exception:
            # Reset broken sessions so next call recreates cleanly.
            await _close_mcp_session(self, name)
            # Retry exactly once — the session may have been a zombie
            # whose background sse_reader crashed (e.g. server closed the
            # SSE connection).  Recreating it transparently recovers.
            try:
                session = await _get_or_create_mcp_session(self, name, cfg)
                return await asyncio.wait_for(fn(session), timeout=45)
            except Exception:
                await _close_mcp_session(self, name)
                raise


async def mcp_list_tools(self, params: Dict[str, str]) -> str:
    import difflib
    import re as _re

    name = (params.get("server") or "").strip()
    if not name:
        return "Error: mcp_list_tools requires <server>."
    cfg, err = _mcp_server_cfg(self, name)
    if err:
        return f"Error: {err}"
    try:
        async def _do(session):
            return await session.list_tools()

        result = await _mcp_with_sdk_session(self, name, cfg, _do)
        tools = getattr(result, "tools", None)
        if tools is None and isinstance(result, dict):
            tools = result.get("tools", [])
        if not isinstance(tools, list):
            tools = []
        if not tools:
            return f"MCP server '{name}' returned no tools."
        lines = [f"MCP tools on '{name}':"]
        for t in tools[:200]:
            tn = str(getattr(t, "name", "") or (t.get("name", "") if isinstance(t, dict) else ""))
            td = str(getattr(t, "description", "") or (t.get("description", "") if isinstance(t, dict) else "")).strip()
            req_fields: List[str] = []
            schema = getattr(t, "inputSchema", None)
            if schema is None and isinstance(t, dict):
                schema = t.get("inputSchema")
            if isinstance(schema, dict):
                req = schema.get("required", [])
                if isinstance(req, list):
                    req_fields = [str(x) for x in req if isinstance(x, str)]
            req_txt = f" [required: {', '.join(req_fields)}]" if req_fields else ""
            lines.append(f"- {tn}" + (f": {td}" if td else "") + req_txt)
        return "\n".join(lines)
    except Exception as e:
        return f"Error: MCP list failed for '{name}': {e}"


async def mcp_search_tools(self, params: Dict[str, str]) -> str:
    import re as _re

    name = (params.get("server") or "").strip()
    query = (params.get("query") or "").strip()
    try:
        limit = int(params.get("limit", 8) or 8)
    except Exception:
        limit = 8
    limit = max(1, min(limit, 20))
    if not name or not query:
        return "Error: mcp_search_tools requires <server> and <query>."

    cfg, err = _mcp_server_cfg(self, name)
    if err:
        return f"Error: {err}"
    try:
        async def _do_mcp_search(session):
            return await session.list_tools()

        result = await _mcp_with_sdk_session(self, name, cfg, _do_mcp_search)
        tools = getattr(result, "tools", None)
        if tools is None and isinstance(result, dict):
            tools = result.get("tools", [])
        if not isinstance(tools, list) or not tools:
            return f"MCP server '{name}' returned no tools."

        import difflib
        q = query.lower()
        q_tokens = {t for t in _re.split(r"[^a-z0-9]+", q) if t}

        scored: List[Tuple[float, dict]] = []
        for t in tools:
            tn = str(getattr(t, "name", "") or (t.get("name", "") if isinstance(t, dict) else ""))
            td = str(getattr(t, "description", "") or (t.get("description", "") if isinstance(t, dict) else ""))
            hay = f"{tn} {td}".lower()
            hay_tokens = {x for x in _re.split(r"[^a-z0-9]+", hay) if x}
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & hay_tokens) / len(q_tokens)
            ratio_name = difflib.SequenceMatcher(None, q, tn.lower()).ratio()
            ratio_hay = difflib.SequenceMatcher(None, q, hay).ratio()
            score = overlap * 2.0 + ratio_name * 1.5 + ratio_hay
            if q in hay:
                score += 1.5
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for s, t in scored[:limit] if s > 0.1]
        if not top:
            return f"No relevant MCP tools found on '{name}' for query: {query}"

        lines = [f"Top MCP tools on '{name}' for '{query}' (limit={limit}):"]
        for t in top:
            tn = str(getattr(t, "name", "") or (t.get("name", "") if isinstance(t, dict) else ""))
            td = str(getattr(t, "description", "") or (t.get("description", "") if isinstance(t, dict) else "")).strip()
            req_fields: List[str] = []
            schema = getattr(t, "inputSchema", None)
            if schema is None and isinstance(t, dict):
                schema = t.get("inputSchema")
            if isinstance(schema, dict):
                req = schema.get("required", [])
                if isinstance(req, list):
                    req_fields = [str(x) for x in req if isinstance(x, str)]
            req_txt = f" [required: {', '.join(req_fields)}]" if req_fields else ""
            lines.append(f"- {tn}" + (f": {td[:180]}" if td else "") + req_txt)
        lines.append("Use mcp_call_tool with the exact chosen tool name.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: MCP search failed for '{name}': {e}"


async def mcp_call_tool(self, params: Dict[str, str]) -> str:
    name = (params.get("server") or "").strip()
    tool = (params.get("tool") or "").strip()
    args_raw = (params.get("arguments") or "").strip()
    if not name or not tool:
        return "Error: mcp_call_tool requires <server> and <tool>."
    if not args_raw:
        args_raw = "{}"
    try:
        arguments = json.loads(args_raw)
        if not isinstance(arguments, dict):
            return "Error: <arguments> must be a JSON object."
    except Exception as e:
        return f"Error: invalid JSON in <arguments>: {e}"

    cfg, err = _mcp_server_cfg(self, name)
    if err:
        return f"Error: {err}"
    try:
        async def _do_mcp_call(session):
            # Schema-aware argument aliasing: if caller provided a generic
            # "query" field but the MCP tool requires e.g. "search_query",
            # map it automatically to reduce model-side friction.
            try:
                listed = await session.list_tools()
                listed_tools = getattr(listed, "tools", None)
                if listed_tools is None and isinstance(listed, dict):
                    listed_tools = listed.get("tools", [])
                if isinstance(listed_tools, list):
                    match = None
                    for t in listed_tools:
                        tname = str(getattr(t, "name", "") or (t.get("name", "") if isinstance(t, dict) else ""))
                        if tname == tool:
                            match = t
                            break
                    if match is not None:
                        schema = getattr(match, "inputSchema", None)
                        if schema is None and isinstance(match, dict):
                            schema = match.get("inputSchema")
                        if isinstance(schema, dict):
                            required = schema.get("required", [])
                            if isinstance(required, list):
                                missing = [r for r in required if isinstance(r, str) and r not in arguments]
                                if "query" in arguments:
                                    query_val = arguments.get("query")
                                    for r in missing:
                                        if isinstance(query_val, str) and r.endswith("query"):
                                            arguments[r] = query_val
            except Exception:
                pass

            return await session.call_tool(tool, arguments)

        result = await _mcp_with_sdk_session(self, name, cfg, _do_mcp_call)
        text_parts: List[str] = []
        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = result.get("content", [])
        if isinstance(content, list):
            for part in content:
                txt = getattr(part, "text", None)
                if txt is None and isinstance(part, dict):
                    txt = part.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
        if text_parts:
            return "\n".join(text_parts)
        try:
            if hasattr(result, "model_dump"):
                return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            pass
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return f"Error: MCP tool call failed on '{name}/{tool}': {e}"
