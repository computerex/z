"""MCP server management — /mcp slash command handler."""
import asyncio
import json, os, shlex, subprocess, time
from typing import Any, Dict, List

from ..config import get_global_config_path, load_json_config, save_json_config
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape as rich_escape


def run_mcp_manager(console: Console, cmd_arg: str = "") -> str:
    """Manage generic MCP server entries in ~/.z.json.

    Schema:
      "mcp": {
        "<name>": {
          "type": "local",
          "command": ["uvx", "pkg", "-y"],
          "environment": {"KEY": "VALUE"},
          "enabled": true
        }
      }
    """
    try:
        parts = shlex.split(cmd_arg) if cmd_arg else []
    except Exception:
        parts = cmd_arg.split()
    sub = parts[0].lower() if parts else "list"

    data = load_json_config(get_global_config_path())
    mcp = dict(data.get("mcp", {}) or {})

    if sub in ("list", "ls"):
        if not mcp:
            return "No MCP servers configured. Use /mcp add <name> <command...>"
        tbl = Table(show_header=True, box=None, padding=(0, 2), pad_edge=False)
        tbl.add_column("Name", style="bold")
        tbl.add_column("Enabled", width=8)
        tbl.add_column("Type", width=8)
        tbl.add_column("Command/URL", style="dim")
        for name in sorted(mcp.keys()):
            cfg = mcp.get(name, {}) or {}
            enabled = bool(cfg.get("enabled", True))
            stype = str(cfg.get("type", "local"))
            cmd = cfg.get("command", [])
            cmd_text = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else ""
            if stype in ("http", "streamable_http", "sse"):
                cmd_text = str(cfg.get("url", "") or "")
            tbl.add_row(
                name,
                "[green]yes[/green]" if enabled else "[yellow]no[/yellow]",
                stype,
                rich_escape(cmd_text[:140] + ("..." if len(cmd_text) > 140 else "")),
            )
        console.print()
        console.print(
            Panel(
                tbl,
                title="[bold]MCP Servers[/bold]",
                border_style="dim",
                padding=(1, 2),
            )
        )
        console.print(
            "  [dim]Use [white]/mcp show <name>[/white], [white]/mcp test <name>[/white], [white]/mcp enable|disable <name>[/white], [white]/mcp remove <name>[/white][/dim]"
        )
        console.print()
        return ""

    if sub == "show":
        if len(parts) < 2:
            return "Usage: /mcp show <name>"
        name = parts[1]
        cfg = mcp.get(name)
        if not isinstance(cfg, dict):
            return f"MCP server '{name}' not found."
        pretty = json.dumps(cfg, indent=2, ensure_ascii=False)
        return f"{name}:\n{pretty}"

    if sub == "add":
        if len(parts) < 3:
            return (
                "Usage: /mcp add <name> <command...> [--type local|http|sse] [--url URL] [--env KEY=VALUE] [--header KEY=VALUE] [--disabled]\n"
                "Examples:\n"
                "  /mcp add MiniMax uvx minimax-coding-plan-mcp -y --env MINIMAX_API_HOST=https://api.minimax.io\n"
                '  /mcp add web-search-prime --type http --url https://api.z.ai/api/mcp/web_search_prime/mcp --header Authorization="Bearer <key>"'
            )
        name = parts[1]
        if any(ch.isspace() for ch in name):
            return "MCP server name must not contain spaces."
        enabled = True
        mcp_type = "local"
        url = ""
        env: Dict[str, str] = {}
        headers: Dict[str, str] = {}
        cmd: List[str] = []
        i = 2
        while i < len(parts):
            token = parts[i]
            if token == "--disabled":
                enabled = False
                i += 1
                continue
            if token == "--type":
                if i + 1 >= len(parts):
                    return "Usage error: --type requires local|http|sse."
                mcp_type = str(parts[i + 1]).lower().strip()
                i += 2
                continue
            if token.startswith("--type="):
                mcp_type = token[len("--type="):].lower().strip()
                i += 1
                continue
            if token == "--url":
                if i + 1 >= len(parts):
                    return "Usage error: --url requires a value."
                url = parts[i + 1]
                i += 2
                continue
            if token.startswith("--url="):
                url = token[len("--url="):]
                i += 1
                continue
            if token == "--env":
                if i + 1 >= len(parts):
                    return "Usage error: --env requires KEY=VALUE."
                kv = parts[i + 1]
                if "=" not in kv:
                    return "Usage error: --env value must be KEY=VALUE."
                k, v = kv.split("=", 1)
                env[k] = v
                i += 2
                continue
            if token.startswith("--env="):
                kv = token[len("--env="):]
                if "=" not in kv:
                    return "Usage error: --env value must be KEY=VALUE."
                k, v = kv.split("=", 1)
                env[k] = v
                i += 1
                continue
            if token == "--header":
                if i + 1 >= len(parts):
                    return "Usage error: --header requires KEY=VALUE."
                kv = parts[i + 1]
                if "=" not in kv:
                    return "Usage error: --header value must be KEY=VALUE."
                k, v = kv.split("=", 1)
                headers[k] = v
                i += 2
                continue
            if token.startswith("--header="):
                kv = token[len("--header="):]
                if "=" not in kv:
                    return "Usage error: --header value must be KEY=VALUE."
                k, v = kv.split("=", 1)
                headers[k] = v
                i += 1
                continue
            cmd.append(token)
            i += 1
        if mcp_type not in ("local", "http", "streamable_http", "sse"):
            return "Usage error: --type must be local, http, streamable_http, or sse."
        if mcp_type == "local" and not cmd:
            return "Usage error: missing MCP command."
        if mcp_type in ("http", "streamable_http", "sse") and not url:
            return "Usage error: --url is required for HTTP/SSE MCP servers."

        entry: Dict[str, Any] = {"type": mcp_type, "enabled": enabled}
        if mcp_type == "local":
            entry["command"] = cmd
            entry["environment"] = env
        else:
            entry["url"] = url
            entry["headers"] = headers
        mcp[name] = entry
        data["mcp"] = mcp
        path = save_json_config(get_global_config_path(), data)
        return f"Saved MCP server '{name}' to {path}"

    if sub in ("remove", "rm", "delete"):
        if len(parts) < 2:
            return "Usage: /mcp remove <name>"
        name = parts[1]
        if name not in mcp:
            return f"MCP server '{name}' not found."
        mcp.pop(name, None)
        data["mcp"] = mcp
        path = save_json_config(get_global_config_path(), data)
        return f"Removed MCP server '{name}' from {path}"

    if sub in ("enable", "disable"):
        if len(parts) < 2:
            return f"Usage: /mcp {sub} <name>"
        name = parts[1]
        cfg = dict(mcp.get(name, {}) or {})
        if not cfg:
            return f"MCP server '{name}' not found."
        cfg["enabled"] = sub == "enable"
        mcp[name] = cfg
        data["mcp"] = mcp
        path = save_json_config(get_global_config_path(), data)
        return f"{'Enabled' if cfg['enabled'] else 'Disabled'} MCP server '{name}' in {path}"

    if sub == "test":
        if len(parts) < 2:
            return "Usage: /mcp test <name>"
        name = parts[1]
        cfg = dict(mcp.get(name, {}) or {})
        if not cfg:
            return f"MCP server '{name}' not found."
        mcp_type = str(cfg.get("type", "local")).lower()
        if cfg.get("enabled", True) is False:
            return f"MCP server '{name}' is disabled. Use /mcp enable {name} first."
        console.print(f"  [dim]Testing MCP server '{name}'...[/dim]")
        if mcp_type == "local":
            cmd_list = cfg.get("command", [])
            if not isinstance(cmd_list, list) or not cmd_list:
                return f"MCP server '{name}' has invalid command config."
            env_cfg = dict(cfg.get("environment", {}) or {})
            env_vars = os.environ.copy()
            env_vars.update({str(k): str(v) for k, v in env_cfg.items()})
            try:
                proc = subprocess.Popen(
                    cmd_list,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd(),
                    env=env_vars,
                )
            except FileNotFoundError:
                return f"Failed to start '{name}': command not found ({cmd_list[0]})."
            except Exception as e:
                return f"Failed to start '{name}': {e}"

            time.sleep(2.0)
            rc = proc.poll()
            output = ""
            if proc.stdout:
                try:
                    output = proc.stdout.read(4000) if rc is not None else ""
                except Exception:
                    output = ""
            if rc is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return f"MCP server '{name}' started successfully (command responds and stays alive)."
            snippet = (output or "").strip()
            if snippet:
                snippet = snippet[:500]
                return (
                    f"MCP server '{name}' exited early (code={rc}). Output:\n{snippet}"
                )
            return f"MCP server '{name}' exited early (code={rc}) with no output."

        if mcp_type in ("http", "streamable_http", "sse"):
            url = str(cfg.get("url", "") or "").strip()
            headers = dict(cfg.get("headers", {}) or {})
            if not url:
                return f"MCP server '{name}' has invalid URL config."
            try:
                from mcp import ClientSession  # type: ignore
                from mcp.client.streamable_http import streamablehttp_client  # type: ignore
                from mcp.client.sse import sse_client  # type: ignore
            except Exception:
                return "MCP SDK not installed; cannot test HTTP MCP server."

            async def _probe():
                if mcp_type == "sse":
                    async with sse_client(
                        url, headers=headers, timeout=15, sse_read_timeout=120
                    ) as (r, w):
                        async with ClientSession(r, w) as s:
                            await asyncio.wait_for(s.initialize(), timeout=20)
                            await asyncio.wait_for(s.list_tools(), timeout=20)
                            return
                async with streamablehttp_client(
                    url, headers=headers, timeout=15, sse_read_timeout=120
                ) as (r, w, _sid):
                    async with ClientSession(r, w) as s:
                        await asyncio.wait_for(s.initialize(), timeout=20)
                        await asyncio.wait_for(s.list_tools(), timeout=20)

            try:
                asyncio.run(_probe())
                return f"MCP server '{name}' is reachable and responded to initialize/list_tools."
            except Exception as e:
                return f"MCP server '{name}' test failed: {e}"

        return f"MCP server '{name}' has unsupported type '{mcp_type}'."

    if sub == "setenv":
        if len(parts) < 4:
            return "Usage: /mcp setenv <name> <KEY> <VALUE>"
        name, key, value = parts[1], parts[2], parts[3]
        cfg = dict(mcp.get(name, {}) or {})
        if not cfg:
            return f"MCP server '{name}' not found."
        env_vars = dict(cfg.get("environment", {}) or {})
        env_vars[key] = value
        cfg["environment"] = env_vars
        mcp[name] = cfg
        data["mcp"] = mcp
        save_json_config(get_global_config_path(), data)
        return f"Set env var '{key}' on MCP server '{name}'."

    if sub == "unsetenv":
        if len(parts) < 3:
            return "Usage: /mcp unsetenv <name> <KEY>"
        name, key = parts[1], parts[2]
        cfg = dict(mcp.get(name, {}) or {})
        if not cfg:
            return f"MCP server '{name}' not found."
        env_vars = dict(cfg.get("environment", {}) or {})
        env_vars.pop(key, None)
        cfg["environment"] = env_vars
        mcp[name] = cfg
        data["mcp"] = mcp
        save_json_config(get_global_config_path(), data)
        return f"Removed env var '{key}' from MCP server '{name}'."

    return (
        "Usage: /mcp [list|show <name>|add <name> <command...> [--env KEY=VALUE] [--disabled]|"
        "remove <name>|enable <name>|disable <name>|test <name>|setenv <name> <KEY> <VALUE>|unsetenv <name> <KEY>]"
    )
