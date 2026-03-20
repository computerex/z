"""Lightweight plugin system for the harness.

Plugins are Python files that export a `register(api)` function.
The function receives a PluginAPI instance and uses it to register
tools, hooks, and prompt sections.

Discovery (in priority order):
1. Project-local:  <workspace>/.z/plugins/*.py
2. Global:         ~/.z/plugins/*.py
3. Explicit:       "plugins" list in ~/.z.json

Example plugin (~/.z/plugins/hello.py):

    def register(api):
        api.add_tool(
            name="hello",
            description="Say hello",
            params={"name": {"required": True, "description": "Who to greet"}},
            handler=lambda params: f"Hello, {params.get('name', 'world')}!",
        )
"""

import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .logger import get_logger
from .tool_registry import ToolDef, ToolParam

log = get_logger("plugins")


# ── Types ────────────────────────────────────────────────────────

# Handler: sync or async callable(params: dict) -> str
ToolHandler = Callable[[Dict[str, Any]], Any]

# Hook: sync or async callable(*args) -> optional result
HookFn = Callable[..., Any]


@dataclass
class PluginToolDef:
    """A tool contributed by a plugin."""
    name: str
    description: str
    params: Dict[str, dict]  # {param_name: {"required": bool, "description": str}}
    handler: ToolHandler
    category: str = "plugin"
    complex_content: bool = False
    console_label: Optional[str] = None  # e.g. "[cyan]Hello[/cyan]"


@dataclass
class LoadedPlugin:
    """Metadata about a successfully loaded plugin."""
    name: str
    path: str
    tools: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)


# ── Hook names ───────────────────────────────────────────────────
# Lifecycle hooks that plugins can subscribe to.
HOOK_PRE_TOOL = "pre_tool"        # (tool_name, params) -> optional modified params
HOOK_POST_TOOL = "post_tool"      # (tool_name, params, result) -> optional modified result
HOOK_PRE_TURN = "pre_turn"        # (messages) -> None
HOOK_POST_TURN = "post_turn"      # (messages, response) -> None
HOOK_ON_COMPACT = "on_compact"    # (freed_tokens, report) -> None
HOOK_SYSTEM_PROMPT = "system_prompt"  # () -> str (appended to system prompt)

VALID_HOOKS = {
    HOOK_PRE_TOOL, HOOK_POST_TOOL, HOOK_PRE_TURN, HOOK_POST_TURN,
    HOOK_ON_COMPACT, HOOK_SYSTEM_PROMPT,
}


# ── PluginAPI (passed to each plugin's register function) ────────

class PluginAPI:
    """API surface exposed to plugins during registration."""

    def __init__(self, manager: "PluginManager", plugin_name: str):
        self._manager = manager
        self._plugin_name = plugin_name

    @property
    def workspace_path(self) -> str:
        return self._manager.workspace_path

    def add_tool(
        self,
        name: str,
        description: str,
        params: Optional[Dict[str, dict]] = None,
        handler: Optional[ToolHandler] = None,
        category: str = "plugin",
        complex_content: bool = False,
        console_label: Optional[str] = None,
    ) -> None:
        """Register a new tool.

        Args:
            name: Tool name (must be unique across builtins and plugins).
            description: What the tool does (shown to the LLM).
            params: {param_name: {"required": bool, "description": str}}.
            handler: Callable(params_dict) -> str. Can be sync or async.
            category: Tool category for grouping.
            complex_content: True if tool content may contain nested XML.
            console_label: Rich markup for console display, e.g. "[cyan]Greet[/cyan]".
        """
        if handler is None:
            raise ValueError(f"Plugin '{self._plugin_name}': tool '{name}' needs a handler")
        tool = PluginToolDef(
            name=name,
            description=description,
            params=params or {},
            handler=handler,
            category=category,
            complex_content=complex_content,
            console_label=console_label,
        )
        self._manager._register_tool(tool, self._plugin_name)

    def on(self, hook_name: str, fn: HookFn) -> None:
        """Subscribe to a lifecycle hook.

        Valid hooks: pre_tool, post_tool, pre_turn, post_turn,
                     on_compact, system_prompt.
        """
        if hook_name not in VALID_HOOKS:
            log.warning(
                "Plugin '%s' registered unknown hook '%s' (ignored). Valid: %s",
                self._plugin_name, hook_name, ", ".join(sorted(VALID_HOOKS)),
            )
            return
        self._manager._register_hook(hook_name, fn, self._plugin_name)

    def get_config(self) -> dict:
        """Return any per-plugin config from ~/.z.json  plugin_config.<name>."""
        return self._manager.plugin_configs.get(self._plugin_name, {})


# ── PluginManager ────────────────────────────────────────────────

class PluginManager:
    """Discovers, loads, and manages plugins."""

    def __init__(self, workspace_path: str = "", plugin_configs: Optional[dict] = None):
        self.workspace_path = workspace_path
        self.plugin_configs: Dict[str, dict] = plugin_configs or {}

        # Plugin tool registry: name -> PluginToolDef
        self.tools: Dict[str, PluginToolDef] = {}
        # Hook registry: hook_name -> [(fn, plugin_name)]
        self.hooks: Dict[str, List[tuple]] = {h: [] for h in VALID_HOOKS}
        # Loaded plugin metadata
        self.loaded: List[LoadedPlugin] = []
        # Track which names are taken (to prevent collisions)
        self._tool_owners: Dict[str, str] = {}

    # ── Registration (called via PluginAPI) ──────────────────────

    def _register_tool(self, tool: PluginToolDef, plugin_name: str) -> None:
        if tool.name in self._tool_owners:
            owner = self._tool_owners[tool.name]
            log.warning(
                "Plugin '%s' tried to register tool '%s' already owned by '%s' — skipped",
                plugin_name, tool.name, owner,
            )
            return
        self.tools[tool.name] = tool
        self._tool_owners[tool.name] = plugin_name
        log.info("Registered plugin tool: %s (from %s)", tool.name, plugin_name)

    def _register_hook(self, hook_name: str, fn: HookFn, plugin_name: str) -> None:
        self.hooks[hook_name].append((fn, plugin_name))
        log.info("Registered hook: %s (from %s)", hook_name, plugin_name)

    # ── Discovery & loading ──────────────────────────────────────

    def discover_and_load(
        self,
        extra_paths: Optional[List[str]] = None,
        builtin_tool_names: Optional[set] = None,
    ) -> None:
        """Discover and load all plugins.

        Args:
            extra_paths: Additional plugin file paths from config.
            builtin_tool_names: Set of built-in tool names to prevent collisions.
        """
        # Reserve built-in names
        if builtin_tool_names:
            for name in builtin_tool_names:
                self._tool_owners[name] = "__builtin__"

        paths = self._discover_plugin_paths(extra_paths)
        for path in paths:
            self._load_plugin(path)

        if self.loaded:
            names = ", ".join(p.name for p in self.loaded)
            tools = ", ".join(self.tools.keys()) or "(none)"
            log.info("Loaded %d plugin(s): %s | tools: %s", len(self.loaded), names, tools)

    def _discover_plugin_paths(self, extra_paths: Optional[List[str]] = None) -> List[Path]:
        """Find plugin .py files from standard locations + explicit paths."""
        found: List[Path] = []
        seen: set = set()

        def _scan_dir(d: Path, label: str) -> None:
            if not d.is_dir():
                return
            for f in sorted(d.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
                    log.debug("Discovered plugin [%s]: %s", label, resolved)

        # 1. Project-local
        ws = Path(self.workspace_path) if self.workspace_path else Path.cwd()
        _scan_dir(ws / ".z" / "plugins", "project")

        # 2. Global
        _scan_dir(Path.home() / ".z" / "plugins", "global")

        # 3. Explicit paths from config
        for p in (extra_paths or []):
            resolved = Path(p).resolve()
            if resolved.suffix == ".py" and resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
                log.debug("Discovered plugin [config]: %s", resolved)
            elif resolved.is_dir():
                _scan_dir(resolved, "config-dir")

        return found

    def _load_plugin(self, path: Path) -> None:
        """Load a single plugin file."""
        plugin_name = path.stem
        try:
            spec = importlib.util.spec_from_file_location(
                f"harness_plugin_{plugin_name}", str(path)
            )
            if spec is None or spec.loader is None:
                log.error("Cannot load plugin '%s': invalid module spec", path)
                return

            module = importlib.util.module_from_spec(spec)
            # Don't pollute sys.modules with plugin modules
            spec.loader.exec_module(module)

            register_fn = getattr(module, "register", None)
            if register_fn is None:
                log.warning("Plugin '%s' has no register() function — skipped", path)
                return
            if not callable(register_fn):
                log.warning("Plugin '%s': register is not callable — skipped", path)
                return

            api = PluginAPI(self, plugin_name)
            register_fn(api)

            loaded = LoadedPlugin(
                name=plugin_name,
                path=str(path),
                tools=[n for n, t in self.tools.items() if self._tool_owners.get(n) == plugin_name],
                hooks=[h for h, fns in self.hooks.items() if any(pn == plugin_name for _, pn in fns)],
            )
            self.loaded.append(loaded)
            log.info("Loaded plugin: %s from %s (tools: %s, hooks: %s)",
                     plugin_name, path, loaded.tools, loaded.hooks)

        except Exception as e:
            log.error("Failed to load plugin '%s': %s", path, e, exc_info=True)

    # ── Hook execution ───────────────────────────────────────────

    async def run_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Run all registered handlers for a hook.

        For hooks that return values (system_prompt, post_tool), results
        are collected. For post_tool, the last non-None return wins.
        """
        handlers = self.hooks.get(hook_name, [])
        if not handlers:
            return None

        last_result = None
        for fn, plugin_name in handlers:
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                if result is not None:
                    last_result = result
            except Exception as e:
                log.error("Hook %s from plugin '%s' failed: %s", hook_name, plugin_name, e)
        return last_result

    async def run_system_prompt_hooks(self) -> str:
        """Collect all system_prompt hook contributions."""
        parts = []
        for fn, plugin_name in self.hooks.get(HOOK_SYSTEM_PROMPT, []):
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn()
                else:
                    result = fn()
                if result:
                    parts.append(str(result))
            except Exception as e:
                log.error("system_prompt hook from '%s' failed: %s", plugin_name, e)
        return "\n\n".join(parts)

    # ── Tool dispatch ────────────────────────────────────────────

    async def dispatch_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """Dispatch a tool call to a plugin handler. Returns None if not a plugin tool."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return None

        handler = tool.handler
        if inspect.iscoroutinefunction(handler):
            return await handler(params)
        else:
            return handler(params)

    # ── ToolDef generation (for tool_registry integration) ───────

    def get_tool_defs(self) -> List[ToolDef]:
        """Convert plugin tools to ToolDef objects for the registry."""
        defs = []
        for tool in self.tools.values():
            params = [
                ToolParam(
                    name=pname,
                    required=pinfo.get("required", False),
                    description=pinfo.get("description", ""),
                )
                for pname, pinfo in tool.params.items()
            ]
            defs.append(ToolDef(
                name=tool.name,
                category=tool.category,
                complex_content=tool.complex_content,
                params=params,
                description=tool.description,
            ))
        return defs

    # ── Prompt documentation ─────────────────────────────────────

    def get_tool_prompt_docs(self) -> str:
        """Generate XML-formatted tool documentation for plugin tools (for system prompt)."""
        if not self.tools:
            return ""

        sections = []
        for tool in self.tools.values():
            lines = [f"## {tool.name}"]
            lines.append(f"Description: {tool.description}")
            if tool.params:
                lines.append("Parameters:")
                for pname, pinfo in tool.params.items():
                    req = "(required)" if pinfo.get("required") else "(optional)"
                    desc = pinfo.get("description", "")
                    lines.append(f"- {pname}: {req} {desc}")
            lines.append("Usage:")
            lines.append(f"<{tool.name}>")
            for pname in tool.params:
                lines.append(f"<{pname}>value</{pname}>")
            lines.append(f"</{tool.name}>")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # ── Info ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of loaded plugins."""
        if not self.loaded:
            return "No plugins loaded."
        lines = ["Loaded plugins:"]
        for p in self.loaded:
            tools_str = ", ".join(p.tools) if p.tools else "none"
            hooks_str = ", ".join(p.hooks) if p.hooks else "none"
            lines.append(f"  {p.name} ({p.path})")
            lines.append(f"    tools: {tools_str}")
            lines.append(f"    hooks: {hooks_str}")
        return "\n".join(lines)
