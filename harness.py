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
import base64
import hashlib
import time
import json
import mimetypes
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
import httpx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config
from harness.cline_agent import ClineAgent
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from harness.logger import init_logging, get_logger, log_exception, truncate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape as rich_escape
from rich import box

log = get_logger("main")

try:
    from anthropic import Anthropic  # type: ignore
except Exception:
    Anthropic = None

# Multiline input support
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.document import Document
    from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog
    try:
        from prompt_toolkit.application import run_in_terminal as pt_run_in_terminal
    except Exception:
        pt_run_in_terminal = None
    HAS_PROMPT_TOOLKIT = True
    HAS_PT_DIALOGS = True
    
    class SafeFileHistory(FileHistory):
        """FileHistory that handles unicode surrogates gracefully."""
        def store_string(self, string: str) -> None:
            # Remove unicode surrogates that can't be encoded
            safe_string = string.encode('utf-8', errors='replace').decode('utf-8')
            super().store_string(safe_string)
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    HAS_PT_DIALOGS = False
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
    con = Console()
    con.print()
    con.print(Panel(
        "[bold]Welcome to Harness[/bold]\n\n[dim]Let's configure your LLM provider.[/dim]",
        border_style="bright_blue",
        padding=(1, 3),
        width=50,
    ))
    con.print()
    con.print("  Select your LLM provider:\n")
    con.print("  [cyan][1][/cyan] Z.AI Coding Plan [dim](recommended)[/dim]")
    con.print("  [cyan][2][/cyan] Z.AI Standard API")
    con.print("  [cyan][3][/cyan] MiniMax")
    con.print("  [cyan][4][/cyan] Anthropic")
    con.print("  [cyan][5][/cyan] OpenRouter")
    con.print("  [cyan][6][/cyan] OpenAI")
    con.print("  [cyan][7][/cyan] Custom OpenAI-compatible API")
    con.print()
    
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
            base_url = "https://api.anthropic.com/v1/"
            provider = "Anthropic"
            default_model = "claude-3-5-sonnet-latest"
            break
        elif choice == "5":
            base_url = "https://openrouter.ai/api/v1/"
            provider = "OpenRouter"
            default_model = "anthropic/claude-3.5-sonnet"
            break
        elif choice == "6":
            base_url = "https://api.openai.com/v1/"
            provider = "OpenAI"
            default_model = "gpt-4o"
            break
        elif choice == "7":
            base_url = input("Enter API base URL: ").strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Custom"
            default_model = input("Enter default model name: ").strip() or "gpt-4"
            break
        else:
            print("Please enter 1, 2, 3, 4, 5, 6, or 7.")
    
    con.print(f"\n  [green]\u2713[/green] Using [bold]{provider}[/bold]")
    con.print(f"    [dim]{base_url}[/dim]\n")
    
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
    
    con.print("\n  Where to save configuration?\n")
    con.print("  [cyan][1][/cyan] Global [dim](~/.z.json \u2014 all projects)[/dim]")
    con.print("  [cyan][2][/cyan] Workspace [dim](.z/.z.json \u2014 this project only)[/dim]")
    con.print()
    
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
    
    tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
    tbl.add_column("label", style="dim", width=10, justify="right")
    tbl.add_column("value")
    tbl.add_row("Saved to", str(config_path))
    tbl.add_row("Location", location)
    tbl.add_row("Provider", provider)
    tbl.add_row("Model", model)
    tbl.add_row("Key", api_key[:10] + "...")
    con.print()
    con.print(Panel(tbl, title="[bold green] Setup Complete [/bold green]", border_style="green", padding=(1, 2)))
    con.print("\n  [dim]Run [white]python harness.py[/white] to start.[/dim]\n")


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


CLIPBOARD_IMAGE_MARKER_RE = re.compile(r"\[\[clipboard_image:(.+?)\]\]")


def _supports_multimodal_input(api_url: str, model: str) -> bool:
    """Best-effort heuristic for vision-capable chat models/providers."""
    u = (api_url or "").lower()
    m = (model or "").lower()
    if "anthropic.com" in u and m.startswith("claude"):
        return True
    vision_markers = (
        "gpt-4o", "gpt-4.1", "o4", "vision", "claude", "glm-4.6v",
        "gemini", "llava", "qwen-vl"
    )
    return any(tok in m for tok in vision_markers)


def _image_path_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _extract_clipboard_image_markers(text: str) -> tuple[str, List[Path]]:
    """Extract image markers inserted by Ctrl+V and return cleaned text + paths."""
    paths: List[Path] = []

    def _repl(match: re.Match) -> str:
        raw = match.group(1).strip()
        p = Path(raw)
        paths.append(p)
        return ""

    cleaned = CLIPBOARD_IMAGE_MARKER_RE.sub(_repl, text)
    # Normalize extra blank lines after marker removal
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, paths


def _build_multimodal_user_content(text: str, image_paths: List[Path]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    prompt_text = text.strip() if text and text.strip() else "Please analyze the pasted image."
    blocks.append({"type": "text", "text": prompt_text})
    for p in image_paths:
        blocks.append({
            "type": "image_url",
            "image_url": {"url": _image_path_to_data_uri(p)}
        })
    return blocks


def get_sessions_dir(workspace: str) -> Path:
    """Get sessions directory for a workspace."""
    harness_dir = Path(__file__).parent
    sessions_dir = harness_dir / ".sessions"
    # Normalise the workspace path so that case differences on Windows
    # (e.g. C:\Projects\evoke vs c:\projects\evoke) map to the same hash.
    normalised = os.path.normcase(os.path.normpath(workspace))
    workspace_hash = hashlib.md5(normalised.encode()).hexdigest()[:12]
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


PROVIDER_PRESETS = {
    "zai-coding": ("Z.AI Coding", "https://api.z.ai/api/coding/paas/v4/", "glm-4.7"),
    "zai-standard": ("Z.AI Standard", "https://api.z.ai/api/paas/v4/", "glm-4.7"),
    "minimax": ("MiniMax", "https://api.minimax.io/v1/", "MiniMax-M2.1"),
    "anthropic": ("Anthropic", "https://api.anthropic.com/v1/", "claude-3-5-sonnet-latest"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1/", "anthropic/claude-3.5-sonnet"),
    "openai": ("OpenAI", "https://api.openai.com/v1/", "gpt-4o"),
}
_MODEL_FETCH_CACHE: Dict[str, tuple[float, List[str]]] = {}
_MODEL_FETCH_CACHE_TTL_SECS = 300
_LAST_MODEL_SEARCH_RESULTS: List[Dict[str, Any]] = []
_LAST_MODEL_SEARCH_QUERY: str = ""


def _provider_family_for_url(api_url: str) -> str:
    u = (api_url or "").lower()
    if "api.anthropic.com" in u:
        return "anthropic"
    if "openrouter.ai" in u:
        return "openrouter"
    if "api.openai.com" in u:
        return "openai"
    return "openai_compat"


def _detect_provider_label(api_url: str) -> str:
    """Return a human-readable provider name based on the API URL."""
    u = (api_url or "").lower()
    if "api.z.ai" in u:
        if "/coding/" in u:
            return "Z.AI Coding"
        return "Z.AI"
    if "minimax" in u:
        return "MiniMax"
    if "api.anthropic.com" in u:
        return "Anthropic"
    if "openrouter.ai" in u:
        return "OpenRouter"
    if "api.openai.com" in u:
        return "OpenAI"
    if "api.deepseek.com" in u:
        return "DeepSeek"
    if "api.groq.com" in u:
        return "Groq"
    if "api.together" in u:
        return "Together AI"
    if "api.mistral.ai" in u:
        return "Mistral"
    return "Custom"


def _fetch_models_openai_compatible(api_url: str, api_key: str) -> List[str]:
    """Fetch model IDs from OpenAI/OpenRouter/OpenAI-compatible /models endpoint."""
    url = api_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in api_url.lower():
        headers["HTTP-Referer"] = "https://cline.bot"
        headers["X-Title"] = "Cline"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    ids = []
    for item in data.get("data", []):
        mid = item.get("id")
        if isinstance(mid, str) and mid.strip():
            ids.append(mid.strip())
    return sorted(set(ids), key=str.lower)


def _fetch_models_anthropic(api_url: str, api_key: str) -> List[str]:
    """Fetch Anthropic model IDs using the Anthropic SDK."""
    if Anthropic is None:
        raise RuntimeError("Anthropic SDK not installed. Install dependencies first.")
    kwargs = {"api_key": api_key}
    if api_url:
        base = api_url.rstrip("/")
        if base.lower().endswith("/v1"):
            base = base[:-3]
        kwargs["base_url"] = base
    client = Anthropic(**kwargs)
    page = client.models.list()
    ids: List[str] = []

    def _extract(page_obj):
        # SDK may return iterable pages or objects with .data
        if hasattr(page_obj, "data"):
            for item in getattr(page_obj, "data") or []:
                mid = getattr(item, "id", None)
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
        else:
            try:
                for item in page_obj:
                    mid = getattr(item, "id", None)
                    if isinstance(mid, str) and mid.strip():
                        ids.append(mid.strip())
            except TypeError:
                pass

    _extract(page)
    # Best-effort pagination support if SDK exposes has_next_page/get_next_page
    try:
        while True:
            has_next = getattr(page, "has_next_page", False)
            if callable(has_next):
                has_next = has_next()
            if not has_next:
                break
            page = page.get_next_page()
            _extract(page)
    except Exception:
        pass
    return sorted(set(ids), key=str.lower)


def _fetch_provider_model_ids(api_url: str, api_key: str) -> List[str]:
    family = _provider_family_for_url(api_url)
    if family == "anthropic":
        return _fetch_models_anthropic(api_url, api_key)
    if family in ("openai", "openrouter", "openai_compat"):
        return _fetch_models_openai_compatible(api_url, api_key)
    return []


def _cache_key_for_models(api_url: str, api_key: str) -> str:
    key_hash = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12]
    return f"{api_url.rstrip('/').lower()}|{key_hash}"


def _fetch_provider_model_ids_cached(api_url: str, api_key: str, refresh: bool = False) -> List[str]:
    cache_key = _cache_key_for_models(api_url, api_key)
    now = time.time()
    if not refresh and cache_key in _MODEL_FETCH_CACHE:
        ts, ids = _MODEL_FETCH_CACHE[cache_key]
        if now - ts <= _MODEL_FETCH_CACHE_TTL_SECS:
            return ids
    ids = _fetch_provider_model_ids(api_url, api_key)
    _MODEL_FETCH_CACHE[cache_key] = (now, ids)
    return ids


def _interactive_model_picker(current_model: str, model_ids: List[str]) -> str:
    """Interactive selector for model IDs with optional filtering."""
    if not model_ids:
        return current_model

    models = model_ids[:]
    while True:
        print(f"\n  Fetched {len(models):,} model(s).")
        if len(models) > 40:
            flt = input("  Filter by substring (blank = skip): ").strip().lower()
            if flt:
                filtered = [m for m in models if flt in m.lower()]
                if filtered:
                    models = filtered
                else:
                    print("  No matches for that filter.")
                    continue

        shown = models[:40]
        print()
        for i, mid in enumerate(shown, 1):
            print(f"  [{i:2d}] {mid}")
        if len(models) > len(shown):
            print(f"  ... ({len(models) - len(shown)} more)")
        prompt = f"\n  Choose number, type model id, or Enter to keep [{current_model}]: "
        choice = input(prompt).strip()
        if not choice:
            return current_model
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(shown):
                return shown[idx - 1]
            print("Invalid number.")
            continue
        # exact or custom free text
        return choice


def _apply_selected_provider_model(
    workspace: str,
    agent: ClineAgent,
    providers: Dict[str, dict],
    profile: str,
    cfg: Dict[str, Any],
    chosen_model: str,
) -> str:
    """Apply selected provider+model as active config and persist."""
    agent.config.api_url = cfg.get("api_url", agent.config.api_url)
    agent.config.api_key = cfg.get("api_key", agent.config.api_key)
    agent.config.model = chosen_model
    if "max_tokens" in cfg:
        try:
            agent.config.max_tokens = int(cfg["max_tokens"])
        except Exception:
            pass
    if "temperature" in cfg:
        try:
            agent.config.temperature = float(cfg["temperature"])
        except Exception:
            pass
    agent.tool_handlers.config = agent.config

    if profile in providers:
        providers[profile]["model"] = chosen_model
        _save_provider_profile_fields(workspace, providers, profile, {"model": chosen_model})
    cfg_path = _save_active_config_fields(workspace, {
        "api_url": agent.config.api_url,
        "api_key": agent.config.api_key,
        "model": agent.config.model,
        "max_tokens": agent.config.max_tokens,
        "temperature": agent.config.temperature,
    })
    return f"\u2713 Switched to [bold]{chosen_model}[/bold] via {profile}"


def _build_searchable_providers(agent: ClineAgent, providers: Dict[str, dict]) -> tuple[List[tuple[str, dict]], Optional[str]]:
    searchable: List[tuple[str, dict]] = []
    for name in sorted(providers.keys()):
        cfg = dict(providers.get(name, {}))
        if cfg.get("api_url") and cfg.get("api_key"):
            searchable.append((name, cfg))
    active_name = _infer_active_provider_profile(agent, providers)
    if not active_name and agent.config.api_url and agent.config.api_key:
        searchable.insert(0, ("active", {
            "api_url": agent.config.api_url,
            "api_key": agent.config.api_key,
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
        }))
    return searchable, active_name


def _provider_display_name(profile: str, cfg: Dict[str, Any]) -> str:
    """Human-friendly provider label for search results."""
    if profile != "active":
        return profile
    fam = _provider_family_for_url(str(cfg.get("api_url", "")))
    return f"active/{fam}"


def _save_active_config_fields(workspace: str, updates: dict) -> Path:
    cfg_path = Path(workspace) / ".z" / ".z.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
    data.update(updates)
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cfg_path


def _save_provider_profile_fields(workspace: str, providers: Dict[str, dict], profile: str, updates: dict) -> Path:
    models_path = Path(workspace) / ".z" / "models.json"
    models_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if models_path.exists():
        try:
            data = json.loads(models_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
    data.setdefault("providers", {})
    profile_cfg = dict(data["providers"].get(profile, {}))
    profile_cfg.update(updates)
    data["providers"][profile] = profile_cfg
    models_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    providers[profile] = profile_cfg
    return models_path


def run_model_switch_wizard(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    cmd_arg: str = "",
) -> str:
    """Cross-provider model switcher (no nested prompts).

    Usage:
      /model search <query>  -> search models across all configured providers
      /model <query>         -> shorthand for search
      /model use <n>         -> use result #n from last search
      /model refresh <query> -> refresh model lists, then search
      /model list            -> list current provider models only
    """
    global _LAST_MODEL_SEARCH_RESULTS, _LAST_MODEL_SEARCH_QUERY
    parts = [p for p in cmd_arg.split() if p.strip()]
    verb = parts[0].lower() if parts else ""

    if verb == "use":
        if len(parts) < 2 or not parts[1].isdigit():
            return "Usage: /model use <number> (use a result number from the last /model search)"
        idx = int(parts[1])
        if idx < 1 or idx > len(_LAST_MODEL_SEARCH_RESULTS):
            return "Invalid selection. Run /model <query> first."
        row = _LAST_MODEL_SEARCH_RESULTS[idx - 1]
        return _apply_selected_provider_model(
            workspace, agent, providers,
            row["profile"], dict(row["cfg"]), row["model_id"]
        )

    if verb == "list":
        api_url = agent.config.api_url
        api_key = agent.config.api_key
        if not api_url or not api_key:
            return "No active provider configured."
        try:
            console.print("  [dim]Fetching models...[/dim]")
            mids = _fetch_provider_model_ids_cached(api_url, api_key, refresh=False)
        except Exception as e:
            return f"Model fetch failed: {e}"
        shown = mids[:100]
        console.print(f"\n  [bold]Models[/bold] [dim]({len(mids)} total)[/dim]\n")
        for m in shown:
            marker = "[cyan]\u25cf[/cyan]" if m == agent.config.model else " "
            console.print(f"  {marker} {m}")
        if len(mids) > len(shown):
            console.print(f"  [dim]... and {len(mids) - len(shown)} more[/dim]")
        console.print()
        return ""

    refresh = (verb == "refresh")
    if verb in ("search", "refresh"):
        query = " ".join(parts[1:]).strip()
    else:
        query = " ".join(parts).strip()

    if not query:
        if _LAST_MODEL_SEARCH_RESULTS:
            console.print(f"\n  [bold]Last Search[/bold] [dim]'{_LAST_MODEL_SEARCH_QUERY}' \u2014 {len(_LAST_MODEL_SEARCH_RESULTS)} results[/dim]\n")
            tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
            tbl.add_column(width=2)
            tbl.add_column("num", style="dim", width=4)
            tbl.add_column("model", style="bold")
            tbl.add_column("provider", style="cyan")
            for i, row in enumerate(_LAST_MODEL_SEARCH_RESULTS[:20], 1):
                mark = "[cyan]\u25cf[/cyan]" if row["model_id"] == agent.config.model and row["cfg"].get("api_url") == agent.config.api_url else " "
                display_provider = row.get("provider_display") or row["profile"]
                tbl.add_row(mark, f"[{i}]", row["model_id"], display_provider)
            console.print(tbl)
            console.print(f"\n  [dim]Use [white]/model use <n>[/white] or [white]/model <query>[/white][/dim]\n")
            return ""
        return "Usage: /model <query> to search, then /model use <n>"

    searchable, active_name = _build_searchable_providers(agent, providers)
    if not searchable:
        return "No configured providers. Use /providers setup <name> first."
    if len(searchable) == 1:
        only_name, only_cfg = searchable[0]
        console.print(
            f"[dim]Searching only one configured provider: {_provider_display_name(only_name, only_cfg)}. "
            "Add more via /providers setup <name> to compare across providers.[/dim]"
        )

    aggregate: List[tuple[str, str, str, dict]] = []  # (profile, provider_display, model_id, cfg)
    failures: List[str] = []
    for profile, cfg in searchable:
        api_url = cfg.get("api_url", "")
        api_key = cfg.get("api_key", "")
        provider_display = _provider_display_name(profile, cfg)
        if not api_url or not api_key:
            continue
        try:
            console.print(f"  [dim]Fetching: {provider_display}...[/dim]")
            mids = _fetch_provider_model_ids_cached(api_url, api_key, refresh=refresh)
            for mid in mids:
                aggregate.append((profile, provider_display, mid, cfg))
        except Exception as e:
            failures.append(f"{provider_display}: {e}")

    if failures:
        for f in failures[:5]:
            console.print(f"  [yellow]\u26a0 {rich_escape(f)}[/yellow]")
        if not aggregate:
            return "Model fetch failed for all providers."

    if not aggregate:
        return "No models found from configured providers."
    q = query.lower().strip()
    matches = [row for row in aggregate if (q in row[2].lower() or q in row[1].lower() or q in row[0].lower())]
    if not matches:
        return f"No models matched '{query}'."

    # Deduplicate by (profile, model) while preserving order.
    seen = set()
    deduped = []
    for row in matches:
        k = (row[0], row[2])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)
    matches = deduped

    # Sort exact/startswith hits first for model id.
    if q:
        matches.sort(key=lambda r: (
            0 if r[2].lower() == q else 1,
            0 if r[2].lower().startswith(q) else 1,
            0 if q in r[2].lower() else 1,
            r[0].lower(),
            r[2].lower(),
        ))

    shown = matches[:60]
    _LAST_MODEL_SEARCH_QUERY = query
    _LAST_MODEL_SEARCH_RESULTS = [
        {"profile": profile, "provider_display": provider_display, "model_id": mid, "cfg": dict(cfg)}
        for (profile, provider_display, mid, cfg) in shown
    ]

    console.print(f"\n  [bold]Model Search[/bold] [dim]'{query}' \u2014 {len(matches)} match(es), showing {len(shown)}[/dim]\n")
    tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    tbl.add_column(width=2)
    tbl.add_column("num", style="dim", width=4)
    tbl.add_column("model", style="bold")
    tbl.add_column("provider", style="cyan")
    for i, (profile, provider_display, mid, cfg) in enumerate(shown, 1):
        active_mark = "[cyan]\u25cf[/cyan]" if ((profile == active_name or profile == "active") and mid == agent.config.model) else " "
        tbl.add_row(active_mark, f"[{i}]", mid, provider_display)
    console.print(tbl)

    # QoL: if query resolves cleanly, switch immediately (provider + model).
    exact_matches = [row for row in matches if row[2].lower() == q]
    if len(exact_matches) == 1:
        profile, _provider_display, chosen_model, cfg = exact_matches[0]
        return _apply_selected_provider_model(workspace, agent, providers, profile, dict(cfg), chosen_model)
    if len(matches) == 1:
        profile, _provider_display, chosen_model, cfg = matches[0]
        return _apply_selected_provider_model(workspace, agent, providers, profile, dict(cfg), chosen_model)

    console.print("\n  [dim]Use [white]/model use <n>[/white] to switch to a result.[/dim]\n")
    return ""


def _choose_provider_preset_interactive(current_api_url: str, current_model: str) -> tuple[str, str, str, str]:
    """Prompt user for provider preset.

    Returns (preset_key, label, api_url, default_model).
    preset_key is the PROVIDER_PRESETS key (e.g. "zai-coding") or "custom".
    """
    presets = [
        ("1", "zai-coding"),
        ("2", "zai-standard"),
        ("3", "minimax"),
        ("4", "anthropic"),
        ("5", "openrouter"),
        ("6", "openai"),
        ("7", "custom"),
    ]
    con = Console()
    con.print("\n  [bold]Select provider:[/bold]\n")
    for num, key in presets:
        if key == "custom":
            con.print(f"  [cyan][{num}][/cyan] Custom URL")
        else:
            label, url, model = PROVIDER_PRESETS[key]
            con.print(f"  [cyan][{num}][/cyan] [bold]{label}[/bold]  [dim]{model}  ·  {url}[/dim]")
    con.print()
    while True:
        choice = input("  Enter choice [1-7]: ").strip() or "6"
        selected = dict(presets).get(choice)
        if not selected:
            print("  Please enter 1-7.")
            continue
        if selected == "custom":
            api_url = input(f"  API URL [{current_api_url or 'https://api.example.com/v1/'}]: ").strip() or current_api_url or "https://api.example.com/v1/"
            return "custom", "Custom", api_url.rstrip("/") + "/", current_model or "gpt-4o"
        label, api_url, default_model = PROVIDER_PRESETS[selected]
        return selected, label, api_url, default_model


def run_in_app_config_wizard(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    scope_arg: str = "",
) -> str:
    """Interactive config editor inside the app.

    scope_arg:
      - "" / "active" -> saves .z/.z.json and updates current agent config
      - any other name -> saves a provider profile in .z/models.json
    """
    scope = (scope_arg or "active").strip()
    if not scope:
        scope = "active"
    scope_key = scope.lower()
    if any(ch.isspace() for ch in scope):
        return "Usage: /providers setup [active|<profile_name>] (no spaces in profile name)"

    is_active = scope_key == "active"
    is_new_profile = not is_active and scope not in providers

    target_existing = (
        providers.get(scope, {}) if not is_active else {
            "api_url": agent.config.api_url,
            "api_key": agent.config.api_key,
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
        }
    )
    current_url = target_existing.get("api_url", "")
    current_model = target_existing.get("model", "")

    console.print(f"\n  [bold]{'Configure active provider' if is_active else f'Provider profile: {scope}'}[/bold] [dim](Enter to keep current values)[/dim]")
    preset_key, label, api_url, preset_model = _choose_provider_preset_interactive(current_url, current_model)

    # Auto-suggest a profile name for new profiles based on the preset chosen
    if is_new_profile and scope == "default" and preset_key != "custom":
        suggested_name = preset_key
        entered = input(f"  Profile name [{suggested_name}]: ").strip()
        scope = entered or suggested_name
        # Validate: warn if the profile name conflicts with a different provider
        if scope in providers:
            existing_url = providers[scope].get("api_url", "")
            if existing_url and existing_url != api_url:
                overwrite = input(f"  Profile '{scope}' already exists ({_detect_provider_label(existing_url)}). Overwrite? [y/N]: ").strip().lower()
                if overwrite not in ("y", "yes"):
                    return "Cancelled."

    api_key_current = target_existing.get("api_key", "")
    model_current = target_existing.get("model", "") or preset_model
    max_tokens_current = int(target_existing.get("max_tokens", getattr(agent.config, "max_tokens", 128000)) or 128000)
    temp_current = float(target_existing.get("temperature", getattr(agent.config, "temperature", 0.7)) or 0.7)

    api_key = input(f"  API key [{'***' + api_key_current[-4:] if len(api_key_current) > 4 else ('set' if api_key_current else 'not set')}]: ").strip() or api_key_current
    if not api_key:
        return "Cancelled: API key is required."

    model = model_current
    family = _provider_family_for_url(api_url)
    if family in ("anthropic", "openai", "openrouter", "openai_compat"):
        fetch_now = input(f"  Fetch available models? [Y/n]: ").strip().lower()
        if fetch_now in ("", "y", "yes"):
            try:
                model_ids = _fetch_provider_model_ids(api_url, api_key)
                if model_ids:
                    model = _interactive_model_picker(model_current, model_ids)
                else:
                    console.print("  [dim]No models returned — using manual entry[/dim]")
            except Exception as e:
                console.print(f"  [yellow]\u26a0 Model fetch failed: {rich_escape(str(e))}[/yellow]")
    if not model:
        model = input(f"  Model [{model_current or preset_model}]: ").strip() or model_current or preset_model
    else:
        manual_override = input(f"  Model [{model}] (Enter to keep, or type to change): ").strip()
        if manual_override:
            model = manual_override

    max_tokens_in = input(f"  Max tokens [{max_tokens_current}]: ").strip()
    temp_in = input(f"  Temperature [{temp_current}]: ").strip()
    try:
        max_tokens = int(max_tokens_in) if max_tokens_in else max_tokens_current
        temperature = float(temp_in) if temp_in else temp_current
    except ValueError:
        return "Cancelled: invalid numeric value for max_tokens or temperature."

    detected = _detect_provider_label(api_url)
    config_data = {
        "api_url": api_url,
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if is_active:
        save_scope = input("  Save to [1] workspace or [2] global? [1/2]: ").strip() or "1"
        config_path = (Path.home() / ".z.json") if save_scope == "2" else (Path(workspace) / ".z" / ".z.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        agent.config.api_url = api_url
        agent.config.api_key = api_key
        agent.config.model = model
        agent.config.max_tokens = max_tokens
        agent.config.temperature = temperature
        agent.tool_handlers.config = agent.config
        return f"\u2713 Active config saved \u2014 {detected} / {model}"

    models_path = Path(workspace) / ".z" / "models.json"
    models_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if models_path.exists():
        try:
            data = json.loads(models_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
    data.setdefault("providers", {})
    data["providers"][scope] = config_data
    models_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    providers[scope] = data["providers"][scope]
    agent.providers = providers
    return f"\u2713 Saved profile [bold]{scope}[/bold] \u2014 {detected} / {model}"


def _infer_active_provider_profile(agent: ClineAgent, providers: Dict[str, dict]) -> Optional[str]:
    for name, p in providers.items():
        if (
            p.get("api_url") == agent.config.api_url
            and p.get("api_key") == agent.config.api_key
            and p.get("model") == agent.config.model
        ):
            return name
    for name, p in providers.items():
        if p.get("api_url") == agent.config.api_url and p.get("api_key") == agent.config.api_key:
            return name
    return None


def run_provider_manager(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    cmd_arg: str = "",
) -> str:
    """Manage saved provider profiles with a simple UX."""
    parts = [p for p in cmd_arg.split() if p.strip()]
    sub = (parts[0].lower() if parts else "list")

    if sub in ("list", "ls"):
        if not providers:
            return "No provider profiles saved yet. Use /providers setup."
        active_name = _infer_active_provider_profile(agent, providers)
        _render_providers_table(console, providers, active_name, show_numbers=False)
        console.print()
        return ""

    if sub == "setup":
        profile = parts[1] if len(parts) > 1 else "default"
        return run_in_app_config_wizard(workspace, console, agent, providers, profile)

    if sub == "use":
        if len(parts) < 2:
            return "Usage: /providers use <profile_name>"
        profile = parts[1]
        # Support numeric shorthand: /providers use 1
        if profile.isdigit():
            names = sorted(providers.keys())
            idx = int(profile)
            if 1 <= idx <= len(names):
                profile = names[idx - 1]
            else:
                return f"No provider at index {idx}. Use /providers to see the list."
        p = providers.get(profile)
        if not p:
            return f"Provider profile '{profile}' not found. Use /providers to see the list."
        agent.config.api_url = p.get("api_url", agent.config.api_url)
        agent.config.api_key = p.get("api_key", agent.config.api_key)
        agent.config.model = p.get("model", agent.config.model)
        if "max_tokens" in p:
            try:
                agent.config.max_tokens = int(p["max_tokens"])
            except Exception:
                pass
        if "temperature" in p:
            try:
                agent.config.temperature = float(p["temperature"])
            except Exception:
                pass
        agent.tool_handlers.config = agent.config
        _save_active_config_fields(workspace, {
            "api_url": agent.config.api_url,
            "api_key": agent.config.api_key,
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
        })
        detected = _detect_provider_label(agent.config.api_url)
        return f"\u2713 Switched to [bold]{profile}[/bold] — {detected} / {agent.config.model}"

    if sub in ("remove", "rm", "delete"):
        if len(parts) < 2:
            return "Usage: /providers remove <profile_name>"
        profile = parts[1]
        if profile not in providers:
            return f"Provider profile '{profile}' not found."
        models_path = Path(workspace) / ".z" / "models.json"
        data = {}
        if models_path.exists():
            try:
                data = json.loads(models_path.read_text(encoding="utf-8-sig"))
            except Exception:
                data = {}
        data.setdefault("providers", {})
        data["providers"].pop(profile, None)
        models_path.parent.mkdir(parents=True, exist_ok=True)
        models_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        providers.pop(profile, None)
        if getattr(agent, "providers", None) is not None:
            agent.providers = providers
        return f"Removed provider profile '{profile}'."

    if sub in ("current", "show"):
        active_name = _infer_active_provider_profile(agent, providers)
        if active_name:
            return f"Current provider profile: {active_name}"
        return "Current provider is active config (not matching a saved profile)."

    return "Usage: /providers [list|current|setup <name>|use <name|#>|remove <name>]"


def _render_providers_table(console: Console, providers: Dict[str, dict], active_name: Optional[str], show_numbers: bool = True) -> None:
    """Render a clear providers table showing profile name, detected provider, model, and URL."""
    if not providers:
        console.print("\n  [dim]No providers configured yet.[/dim]")
        return
    tbl = Table(show_header=True, box=None, padding=(0, 2), pad_edge=False)
    tbl.add_column("", width=2)
    if show_numbers:
        tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Profile", style="bold", min_width=10)
    tbl.add_column("Provider", min_width=12)
    tbl.add_column("Model", style="cyan")
    tbl.add_column("URL", style="dim")
    names = sorted(providers.keys())
    for i, name in enumerate(names, 1):
        p = providers[name]
        marker = "[cyan]\u25cf[/cyan]" if name == active_name else " "
        detected = _detect_provider_label(p.get("api_url", ""))
        model = p.get("model", "")
        url = p.get("api_url", "")
        row = [marker]
        if show_numbers:
            row.append(f"[{i}]")
        row.extend([name, detected, model, url])
        tbl.add_row(*row)
    console.print()
    console.print(Panel(tbl, title="[bold]Providers[/bold]", border_style="dim", padding=(1, 2)))


def run_providers_hub(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    cmd_arg: str = "",
) -> str:
    """Simple provider UX hub.

    `/providers` with no args shows configured providers and offers a quick action.
    Supports `/providers setup/use/remove/list/current ...` as aliases.
    """
    parts = [p for p in cmd_arg.split() if p.strip()]
    if parts:
        if len(parts) == 1 and parts[0].isdigit():
            names = sorted(providers.keys())
            idx = int(parts[0])
            if 1 <= idx <= len(names):
                return run_provider_manager(workspace, console, agent, providers, f"use {names[idx - 1]}")
        return run_provider_manager(workspace, console, agent, providers, cmd_arg)

    active_name = _infer_active_provider_profile(agent, providers)
    _render_providers_table(console, providers, active_name, show_numbers=True)

    console.print()
    console.print("  [dim][white]/providers setup[/white]         Add or edit a provider[/dim]")
    console.print("  [dim][white]/providers use <name|#>[/white]  Switch to a provider[/dim]")
    console.print("  [dim][white]/providers remove <name>[/white] Remove a provider[/dim]")
    console.print()
    return ""


def _ui_choose_from_list(
    title: str,
    text: str,
    values: List[tuple[str, str]],
) -> Optional[str]:
    """Prompt-toolkit radio-list dialog picker (returns selected value)."""
    if not (HAS_PROMPT_TOOLKIT and HAS_PT_DIALOGS):
        return None
    try:
        return radiolist_dialog(
            title=title,
            text=text,
            values=values,
            ok_text="Select",
            cancel_text="Cancel",
        ).run()
    except Exception:
        return None


def run_provider_picker_ui(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
) -> str:
    """Interactive provider picker UI (no command memorization needed)."""
    if not providers:
        return "No provider profiles yet. Use /provider setup <name> once, then use F2/F3."
    active = _infer_active_provider_profile(agent, providers)
    values: List[tuple[str, str]] = []
    for name in sorted(providers.keys()):
        p = providers[name]
        detected = _detect_provider_label(p.get("api_url", ""))
        model = p.get("model", "")
        label = f"{name}  |  {detected}"
        if model:
            label += f"  |  {model}"
        values.append((name, label))
    picked = _ui_choose_from_list(
        title="Select Provider",
        text="Choose a saved provider profile to use now.",
        values=values,
    )
    if not picked:
        return "Provider selection cancelled."
    return run_provider_manager(workspace, console, agent, providers, f"use {picked}")


def run_model_picker_ui(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    refresh: bool = False,
) -> str:
    """Interactive model picker UI for the current provider."""
    api_url = agent.config.api_url
    api_key = agent.config.api_key
    current_model = agent.config.model
    if not api_url or not api_key:
        return "No active provider configured. Use /provider setup <name> then /provider use <name>."
    try:
        model_ids = _fetch_provider_model_ids_cached(api_url, api_key, refresh=refresh)
    except Exception as e:
        return f"Model fetch failed: {e}"
    if not model_ids:
        return "Provider returned no models."

    display_ids = model_ids
    if len(model_ids) > 200 and HAS_PROMPT_TOOLKIT and HAS_PT_DIALOGS:
        flt = input_dialog(
            title="Filter Models",
            text=f"{len(model_ids)} models found. Enter a search substring (optional):",
        ).run()
        if flt:
            filtered = [m for m in model_ids if flt.lower() in m.lower()]
            if filtered:
                display_ids = filtered
            else:
                return f"No models matched filter '{flt}'."
    # Keep dialog usable
    shown = display_ids[:200]
    if len(shown) == 0:
        return "No models available."
    values = [(mid, mid) for mid in shown]
    picked = _ui_choose_from_list(
        title="Select Model",
        text=f"Current: {current_model}\nShowing {len(shown)} of {len(display_ids)} model(s)",
        values=values,
    )
    if not picked:
        return "Model selection cancelled."
    return run_model_switch_wizard(workspace, console, agent, providers, picked)


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
    """Tab completer for commands, file paths, and history."""
    
    COMMANDS = [
        '/sessions', '/session', '/delete', '/clear', '/save',
        '/history', '/bg', '/ctx', '/tokens', '/compact', '/cost', '/maxctx',
        '/todo', '/smart', '/dump', '/config', '/providers', '/model', '/iter', '/clip',
        '/index', '/log', '/help', '/?', '/exit', '/quit', '/q',
    ]
    
    def __init__(self, workspace: Path, history: SafeFileHistory = None):
        self.workspace = workspace
        self.history = history
        self._history_cache = []
        self._load_history()
    
    def _load_history(self):
        """Load history strings for completion."""
        if self.history:
            try:
                self._history_cache = list(self.history.load_history_strings())
            except Exception:
                self._history_cache = []
    
    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        
        # Shell command completion with ! prefix
        if text.startswith('!'):
            cmd_part = text[1:].strip()
            if not cmd_part:
                # Just typed !, show common shell commands
                common_cmds = ['ls', 'cd', 'pwd', 'git', 'npm', 'pip', 'python', 'node', 'cat', 'grep', 'find']
                for cmd in common_cmds:
                    yield Completion(
                        f"!{cmd}",
                        start_position=-len(text),
                        display=cmd,
                    )
            else:
                # Complete file paths for shell commands
                if ' ' in cmd_part:
                    # Completing a path argument
                    prefix = cmd_part.split()[-1]
                    try:
                        import glob
                        # Handle glob patterns
                        if '*' in prefix or '?' in prefix:
                            matches = glob.glob(prefix, recursive=False)
                        else:
                            # Complete from current directory
                            matches = glob.glob(prefix + '*', recursive=False)
                        for match in sorted(matches):
                            display = match + ('/' if os.path.isdir(match) else '')
                            yield Completion(
                                f"!{cmd_part.rsplit(' ', 1)[0]} {match}",
                                start_position=-len(prefix),
                                display=display,
                            )
                    except Exception:
                        pass
                else:
                    # Complete the command itself
                    common_cmds = ['ls', 'cd', 'pwd', 'git', 'npm', 'pip', 'python', 'node', 'cat', 'grep', 'find', 'rm', 'cp', 'mv', 'mkdir', 'touch', 'echo', 'clear']
                    for cmd in common_cmds:
                        if cmd.startswith(cmd_part):
                            yield Completion(
                                f"!{cmd}",
                                start_position=-len(text),
                                display=cmd,
                            )
            return
        
        # First, try history completion (if text doesn't start with /)
        if not text.startswith('/') and text.strip():
            prefix = text
            for hist_item in self._history_cache:
                if hist_item.startswith(prefix) and hist_item != prefix:
                    yield Completion(
                        hist_item,
                        start_position=-len(prefix),
                        display=hist_item[:40] + '...' if len(hist_item) > 40 else hist_item,
                    )
            # Only show history if we have matches, otherwise continue
            if any(h.startswith(prefix) for h in self._history_cache):
                return
        
        # Complete commands
        if text.startswith('/'):
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
            elif parts[0] in ['/session', '/delete']:
                # Complete session names
                sessions_dir = self.workspace / ".forge" / "sessions"
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
            elif parts[0] == '/todo' and len(parts) == 2:
                # Complete todo subcommands
                subcommands = ['add', 'done', 'rm', 'clear']
                prefix = parts[1]
                for sub in subcommands:
                    if sub.startswith(prefix):
                        yield Completion(
                            sub,
                            start_position=-len(prefix),
                            display=sub,
                        )
            elif parts[0] == '/compact':
                # Complete compact strategies
                strategies = ['half', 'quarter', 'last2']
                prefix = parts[-1]
                for strat in strategies:
                    if strat.startswith(prefix):
                        yield Completion(
                            strat,
                            start_position=-len(prefix),
                            display=strat,
                        )
            elif parts[0] == '/index':
                # Complete index subcommands
                subcommands = ['rebuild', 'tree']
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
                if '/' in last_word or '\\' in last_word or '.' in last_word:
                    # Try to complete as a path
                    try:
                        # Handle both Unix and Windows paths
                        if '\\' in last_word:
                            # Windows path
                            parts = last_word.rsplit('\\', 1)
                            dir_part = parts[0] if len(parts) > 1 else '.'
                            prefix = parts[1] if len(parts) > 1 else last_word
                            sep = '\\'
                        else:
                            # Unix path or relative
                            parts = last_word.rsplit('/', 1)
                            dir_part = parts[0] if len(parts) > 1 else '.'
                            prefix = parts[1] if len(parts) > 1 else last_word
                            sep = '/'
                        
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


def create_prompt_session(
    history_file: Path,
    workspace: Path,
    on_paste_image_marker: Optional[Callable[[], Optional[str]]] = None,
    on_open_provider_picker: Optional[Callable[[], None]] = None,
    on_open_model_picker: Optional[Callable[[], None]] = None,
) -> "PromptSession":
    """Create a prompt session with multiline support.
    
    Keybindings:
    - Enter: Submit input
    - Ctrl+Enter: Insert newline (for multiline input)
    - Paste: Multiline paste works automatically
    - Tab: Complete commands and file paths
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

    @bindings.add('c-v')
    def _(event):
        """Ctrl+V: paste clipboard image marker when available."""
        if on_paste_image_marker:
            try:
                marker = on_paste_image_marker()
            except Exception:
                marker = None
            if marker:
                event.current_buffer.insert_text(marker)
                return
        # Fallback: let users still get a visible character rather than no-op.
        event.current_buffer.insert_text('\x16')

    @bindings.add('f2')
    def _(event):
        """F2: open provider picker UI."""
        if on_open_provider_picker:
            if pt_run_in_terminal:
                pt_run_in_terminal(on_open_provider_picker)
            else:
                on_open_provider_picker()

    @bindings.add('f3')
    def _(event):
        """F3: open model picker UI."""
        if on_open_model_picker:
            if pt_run_in_terminal:
                pt_run_in_terminal(on_open_model_picker)
            else:
                on_open_model_picker()
    
    # Create session with history
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create history instance first
    history = SafeFileHistory(str(history_file))
    
    # Create completer with history
    completer = HarnessCompleter(workspace, history)
    
    return PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=False,  # Enter submits, Ctrl+Enter for newline
        completer=completer,
    )


def _render_cost_report(console: Console) -> None:
    tracker = get_global_tracker()
    summary = tracker.get_summary()

    tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
    tbl.add_column("label", style="dim", width=15, justify="right")
    tbl.add_column("value")
    tbl.add_row("API Calls", str(summary.total_calls))
    tbl.add_row("Input", f"{summary.total_input_tokens:,} tokens  [dim]${summary.total_input_cost:.4f}[/dim]")
    tbl.add_row("Output", f"{summary.total_output_tokens:,} tokens  [dim]${summary.total_output_cost:.4f}[/dim]")
    tbl.add_row("Total", f"[bold]{summary.total_tokens:,} tokens  ${summary.total_cost:.4f}[/bold]")

    if summary.extra_usage_totals:
        tbl.add_row("", "")
        for key in (
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "prompt_cached_tokens",
            "completion_reasoning_tokens",
            "reasoning_tokens",
        ):
            if key in summary.extra_usage_totals:
                label = key.replace("_", " ").replace("tokens", "").strip()
                tbl.add_row(label, f"{summary.extra_usage_totals[key]:,}")

    console.print()
    console.print(Panel(tbl, title="[bold]Session Costs[/bold]", border_style="dim", padding=(1, 2), width=52))

    by_model = tracker.get_cost_by_model()
    if by_model and len(by_model) > 1:
        mtbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
        mtbl.add_column("Model", style="cyan")
        mtbl.add_column("Calls", justify="right")
        mtbl.add_column("Tokens", justify="right")
        mtbl.add_column("Cost", justify="right", style="bold")
        for model, row in sorted(by_model.items(), key=lambda kv: kv[1]["total_cost"], reverse=True):
            mtbl.add_row(
                model,
                str(int(row["calls"])),
                f"{int(row['total_tokens']):,}",
                f"${row['total_cost']:.4f}",
            )
        console.print(mtbl)
    console.print()


def _parse_token_limit_input(raw: str) -> Optional[int]:
    s = (raw or "").strip().lower().replace(",", "")
    if not s:
        return None
    mult = 1
    if s.endswith("k"):
        mult = 1000
        s = s[:-1]
    elif s.endswith("m"):
        mult = 1_000_000
        s = s[:-1]
    try:
        val = int(float(s) * mult)
    except ValueError:
        return None
    return val if val > 0 else None


async def run_single(
    agent: ClineAgent,
    user_input: Union[str, List[Dict[str, Any]]],
    console: Console,
    user_label: Optional[str] = None,
) -> str:
    """Run a single user request."""
    start_time = time.time()
    log.debug("run_single START mode=%s input=%s",
              agent.reasoning_mode,
              truncate(user_label or (user_input if isinstance(user_input, str) else "[multimodal]"), 120))
    try:
        if isinstance(user_input, str):
            result = await agent.run(user_input)
        else:
            result = await agent.run_message(user_input, user_label=user_label or "[multimodal]")
    except asyncio.CancelledError:
        log.warning("run_single cancelled after %.1fs", time.time() - start_time)
        console.print("\n  [yellow]Cancelled[/yellow]")
        result = "[Interrupted]"
    except KeyboardInterrupt:
        log.warning("run_single KeyboardInterrupt after %.1fs", time.time() - start_time)
        console.print("\n  [yellow]Interrupted[/yellow]")
        result = "[Interrupted]"
    
    elapsed = time.time() - start_time
    mode_tag = agent.reasoning_mode.upper()
    log.info("run_single DONE mode=%s elapsed=%.1fs result_len=%d",
             agent.reasoning_mode, elapsed, len(result or ""))
    
    cost = get_global_tracker().get_summary()
    stats = agent.get_context_stats()
    
    if elapsed < 60:
        elapsed_str = f"{elapsed:.1f}s"
    else:
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins}m{secs:02d}s"
    
    ctx_k = stats['tokens'] // 1000
    max_k = stats['max_allowed'] // 1000
    pct = stats['percent']
    
    bar_width = 10
    filled = int(bar_width * pct / 100)
    bar_color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    
    status = Text("  ")
    status.append(agent.config.model, style="bold dim")
    status.append("  ", style="dim")
    status.append("\u2501" * filled, style=bar_color)
    status.append("\u2500" * (bar_width - filled), style="dim")
    status.append(f" {ctx_k}k/{max_k}k", style="dim")
    status.append(f"  {elapsed_str}", style="dim")
    if cost.total_cost > 0:
        status.append(f"  ${cost.total_cost:.4f}", style="dim")
    console.print(status)
    
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
    
    if args.list:
        sessions = list_sessions(workspace)
        con = Console()
        if not sessions:
            con.print("  [dim]No sessions found.[/dim]")
        else:
            tbl = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 2))
            tbl.add_column("Session", style="bold")
            tbl.add_column("Messages", justify="right")
            tbl.add_column("Modified", style="dim")
            for name, mtime, count in sessions:
                tbl.add_row(name, str(count), f"{mtime:%Y-%m-%d %H:%M}")
            con.print()
            con.print(tbl)
            con.print()
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
    
    # Determine starting config from the active config file (.z/.z.json or ~/.z.json).
    # Provider profiles are available via /provider use and /provider setup.
    config = Config.from_json(workspace=Path(workspace))
    if (not config.api_url or not config.api_key) and providers:
        first_name = sorted(providers.keys())[0]
        p = providers[first_name]
        config = Config.from_json(workspace=Path(workspace), overrides={
            "api_url": p.get("api_url", ""),
            "api_key": p.get("api_key", ""),
            "model": p.get("model", config.model),
            "max_tokens": p.get("max_tokens", config.max_tokens),
            "temperature": p.get("temperature", config.temperature),
        })
        log.info("No active config found; bootstrapping from provider profile '%s'", first_name)
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
            console.print(f"  [dim]Resumed [white]{current_session}[/white] ({msg_count} messages)[/dim]")
    else:
        log.info("New session '%s'", current_session)
        console.print(f"  [dim]New session [white]{current_session}[/white][/dim]")

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
        # Interactive mode — clean startup banner
        stats = agent.get_context_stats()
        ws_short = os.path.basename(workspace) or workspace
        banner_table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
        banner_table.add_column(style="dim", width=11, justify="right")
        banner_table.add_column()
        banner_table.add_row("model", f"[bold]{config.model}[/bold]")
        banner_table.add_row("context", f"[bold]{stats['max_allowed']:,}[/bold] [dim]tokens[/dim]")
        banner_table.add_row("workspace", f"[cyan]{ws_short}[/cyan]")
        banner_table.add_row("session", current_session)
        console.print()
        console.print(Panel(
            banner_table,
            title="[bold bright_blue] harness [/bold bright_blue]",
            border_style="bright_blue",
            padding=(1, 3),
            width=50,
        ))
        console.print(
            "  [dim]Type a message to chat."
            " [white]/help[/white] for commands,"
            " [white]!cmd[/white] for shell.[/dim]\n"
        )
        
        # Create prompt session for multiline input
        def _prompt_paste_image_marker() -> Optional[str]:
            img_path, error = get_clipboard_image()
            if error or not img_path:
                return None
            return f" [[clipboard_image:{img_path}]] "

        def _open_provider_picker_ui() -> None:
            console.print("  [dim]Use /providers[/dim]")

        def _open_model_picker_ui() -> None:
            console.print("  [dim]Use /model[/dim]")

        history_file = get_sessions_dir(workspace) / ".history"
        prompt_session = (
            create_prompt_session(
                history_file,
                Path(workspace),
                on_paste_image_marker=_prompt_paste_image_marker,
                                on_open_provider_picker=_open_provider_picker_ui,
                                on_open_model_picker=_open_model_picker_ui,
            )
            if HAS_PROMPT_TOOLKIT else None
        )
        
        last_interrupt_time = 0  # Track time of last Ctrl+C for double-tap exit
        
        while True:
            try:
                prompt_text = "\x1b[38;5;75mharness\x1b[0m \x1b[38;5;243m\u276f\x1b[0m "
                
                # Get input (multiline with prompt_toolkit, or simple input)
                if prompt_session:
                    try:
                        user_input = prompt_session.prompt(ANSI(prompt_text)).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print("\n  [dim]Press [white]Ctrl+C[/white] again to exit[/dim]")
                        continue
                else:
                    try:
                        user_input = input(prompt_text).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print("\n  [dim]Press [white]Ctrl+C[/white] again to exit[/dim]")
                        continue
                
                if not user_input:
                    continue
                
                # Handle shell commands with ! prefix
                if user_input.startswith("!"):
                    shell_cmd = user_input[1:].strip()
                    if shell_cmd:
                        console.print(f"  [dim]\u25b6 {rich_escape(shell_cmd)}[/dim]")
                        try:
                            result = loop.run_until_complete(
                                agent.tool_handlers.execute_command({"command": shell_cmd})
                            )
                            console.print(result)
                        except Exception as e:
                            console.print(f"  [red]\u2717 {rich_escape(str(e))}[/red]")
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    cmd_arg = parts[1] if len(parts) > 1 else ""
                    
                    if cmd in ('/exit', '/quit', '/q'):
                        cleanup_and_save()
                        console.print("  [dim]Session saved. Goodbye![/dim]")
                        break
                    
                    elif cmd == '/sessions':
                        sessions = list_sessions(workspace)
                        if not sessions:
                            console.print("  [dim]No sessions yet.[/dim]")
                        else:
                            tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
                            tbl.add_column(width=2)
                            tbl.add_column("name", style="bold")
                            tbl.add_column("msgs", justify="right", style="dim")
                            tbl.add_column("modified", style="dim")
                            for name, mtime, count in sessions:
                                marker = "[cyan]\u25cf[/cyan]" if name == current_session else " "
                                tbl.add_row(marker, name, f"{count} msgs", f"{mtime:%b %d, %H:%M}")
                            console.print()
                            console.print(Panel(tbl, title="[bold]Sessions[/bold]", border_style="dim", padding=(1, 2)))
                            console.print()
                        continue
                    
                    elif cmd == '/session':
                        if not cmd_arg:
                            console.print(f"  [dim]Current session:[/dim] [bold]{current_session}[/bold]")
                            continue
                        agent.save_session(str(session_path))
                        new_name = cmd_arg.strip()
                        current_session = new_name
                        session_path = get_session_path(workspace, current_session)
                        if session_path.exists():
                            agent.load_session(str(session_path))
                            msg_count = len(agent.messages) - 1
                            console.print(f"  [green]\u2713[/green] Switched to [bold]{current_session}[/bold] [dim]({msg_count} messages)[/dim]")
                        else:
                            agent.clear_history()
                            console.print(f"  [green]\u2713[/green] Created new session [bold]{current_session}[/bold]")
                        continue
                    
                    elif cmd == '/clear':
                        agent.clear_history()
                        reset_global_tracker()
                        console.print("  [green]\u2713[/green] History cleared")
                        continue
                    
                    elif cmd == '/save':
                        agent.save_session(str(session_path))
                        console.print(f"  [green]\u2713[/green] Session [bold]{current_session}[/bold] saved")
                        continue
                    
                    elif cmd == '/delete':
                        if not cmd_arg:
                            console.print("  [dim]Usage: /delete <session_name>[/dim]")
                            continue
                        target = cmd_arg.strip()
                        if target == current_session:
                            console.print("  [yellow]\u26a0[/yellow] Cannot delete the active session")
                            continue
                        target_path = get_session_path(workspace, target)
                        if target_path.exists():
                            target_path.unlink()
                            console.print(f"  [green]\u2713[/green] Deleted session [bold]{target}[/bold]")
                        else:
                            console.print(f"  [dim]Session '{target}' not found[/dim]")
                        continue
                    
                    elif cmd == '/history':
                        console.print(f"  [dim]{len(agent.messages)} messages in conversation[/dim]")
                        continue

                    elif cmd == '/cost':
                        _render_cost_report(console)
                        continue
                    
                    elif cmd == '/bg':
                        procs = agent.list_background_procs()
                        if not procs:
                            console.print("  [dim]No background processes running[/dim]")
                        else:
                            tbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
                            tbl.add_column("ID", style="bold")
                            tbl.add_column("PID", style="dim")
                            tbl.add_column("Status")
                            tbl.add_column("Time", justify="right")
                            tbl.add_column("Command", style="dim")
                            for p in procs:
                                elapsed_min = p['elapsed'] / 60
                                status_style = "green" if p['status'] == "running" else "yellow"
                                tbl.add_row(
                                    str(p['id']),
                                    str(p['pid']),
                                    f"[{status_style}]{p['status']}[/{status_style}]",
                                    f"{elapsed_min:.1f}m",
                                    p['command'][:50],
                                )
                            console.print()
                            console.print(Panel(tbl, title="[bold]Background Processes[/bold]", border_style="dim", padding=(0, 1)))
                            console.print()
                        continue
                    
                    elif cmd == '/mode':
                        console.print("  [dim]/mode is deprecated. Use [white]/providers[/white] and [white]/model[/white] instead.[/dim]")
                        continue
                    
                    elif cmd == '/ctx':
                        stats = agent.get_context_stats()
                        pct = stats['percent']
                        bar_width = 24
                        filled = int(bar_width * pct / 100)
                        bar_color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
                        bar = Text()
                        bar.append("\u2501" * filled, style=bar_color)
                        bar.append("\u2500" * (bar_width - filled), style="dim")
                        console.print()
                        console.print("  [bold]Context Usage[/bold]")
                        ctx_line = Text("  ")
                        ctx_line.append_text(bar)
                        ctx_line.append(f"  {stats['tokens']:,} / {stats['max_allowed']:,} tokens ({pct:.0f}%)", style="dim")
                        console.print(ctx_line)
                        console.print(f"  [dim]{stats['messages']} messages \u00b7 {stats['context_items']} context items[/dim]")
                        console.print()
                        continue
                    
                    elif cmd == '/tokens':
                        breakdown = agent.get_token_breakdown()
                        tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
                        tbl.add_column("label", style="dim", width=14, justify="right")
                        tbl.add_column("value")
                        tbl.add_row("System", f"{breakdown['system']:,} tokens")
                        tbl.add_row("Conversation", f"{breakdown['conversation']:,} tokens [dim]({breakdown['message_count']} msgs)[/dim]")
                        tbl.add_row("Total", f"[bold]{breakdown['total']:,} tokens[/bold]")
                        console.print()
                        console.print(Panel(tbl, title="[bold]Token Breakdown[/bold]", border_style="dim", padding=(1, 2)))
                        if breakdown['largest_messages']:
                            console.print("  [dim]Largest messages:[/dim]")
                            for msg in breakdown['largest_messages']:
                                role = msg['role']
                                tokens = msg['tokens']
                                preview = msg['preview'][:50] + '...' if len(msg['preview']) > 50 else msg['preview']
                                console.print(f"    [dim]{role:>10}  {tokens:>6,}t  {rich_escape(preview)}[/dim]")
                        console.print()
                        continue
                    
                    elif cmd == '/compact':
                        strategy = cmd_arg.strip() if cmd_arg else 'half'
                        before = agent.get_token_count()
                        removed = agent.compact_history(strategy)
                        after = agent.get_token_count()
                        console.print(f"  [green]\u2713[/green] Compacted [bold]{before:,}[/bold] \u2192 [bold]{after:,}[/bold] tokens [dim](-{removed:,})[/dim]")
                        continue
                    
                    elif cmd == '/todo':
                        if not cmd_arg:
                            agent.todo_manager.print_todo_panel(console)
                        elif cmd_arg.startswith("add "):
                            title = cmd_arg[4:].strip()
                            agent.todo_manager.add(title=title)
                            console.print(f"  [green]\u2713[/green] Added: {title}")
                            agent.todo_manager.print_todo_panel(console)
                        elif cmd_arg.startswith("done "):
                            try:
                                item_id = int(cmd_arg[5:].strip())
                                item = agent.todo_manager.update(item_id, status="completed")
                                if not item:
                                    console.print(f"  [dim]Todo [{item_id}] not found[/dim]")
                                else:
                                    console.print(f"  [green]\u2713[/green] Completed: {item.title}")
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("  [dim]Usage: /todo done <id>[/dim]")
                        elif cmd_arg.startswith("rm "):
                            try:
                                item_id = int(cmd_arg[3:].strip())
                                if not agent.todo_manager.remove(item_id):
                                    console.print(f"  [dim]Todo [{item_id}] not found[/dim]")
                                else:
                                    console.print(f"  [green]\u2713[/green] Removed")
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("  [dim]Usage: /todo rm <id>[/dim]")
                        elif cmd_arg.strip() == "clear":
                            agent.todo_manager.clear()
                            console.print("  [green]\u2713[/green] All todos cleared")
                        else:
                            console.print("  [dim]Usage: /todo [add <title> | done <id> | rm <id> | clear][/dim]")
                        continue
                    
                    elif cmd == '/smart':
                        from harness.context_management import get_model_limits as _gml, estimate_messages_tokens as _emt
                        _, max_allowed = _gml(agent.config.model)
                        total_tokens = _emt(agent.messages)
                        pct = total_tokens / max_allowed * 100 if max_allowed else 0
                        over = total_tokens > int(max_allowed * 0.85)
                        active_todos = agent.todo_manager.list_active()
                        traces = agent.smart_context.compaction_traces

                        bar_width = 24
                        filled = int(bar_width * pct / 100)
                        bar_color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
                        bar = Text()
                        bar.append("\u2501" * filled, style=bar_color)
                        bar.append("\u2500" * (bar_width - filled), style="dim")

                        tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
                        tbl.add_column("label", style="dim", width=14, justify="right")
                        tbl.add_column("value")
                        ctx_val = Text()
                        ctx_val.append_text(bar)
                        ctx_val.append(f"  {total_tokens:,} / {max_allowed:,} ({pct:.0f}%)")
                        tbl.add_row("Context", ctx_val)
                        threshold_style = "red bold" if over else "green"
                        tbl.add_row("Compaction", f"[{threshold_style}]{'OVER 85% threshold' if over else 'below threshold'}[/{threshold_style}]")
                        tbl.add_row("Messages", str(len(agent.messages)))
                        tbl.add_row("Active todos", str(len(active_todos)))
                        tbl.add_row("Context items", f"{len(agent.context.list_items())}  [dim]({agent.context.total_size():,} chars)[/dim]")
                        console.print()
                        console.print(Panel(tbl, title="[bold]Smart Context[/bold]", border_style="dim", padding=(1, 2)))
                        if traces:
                            console.print(f"  [dim]Compactions this session: {len(traces)}[/dim]")
                            for t in traces[-5:]:
                                console.print(f"    [dim]\u2192 {t.format_notice()}[/dim]")
                        else:
                            console.print("  [dim]No compactions yet.[/dim]")
                        console.print()
                        continue
                    
                    elif cmd == '/dump':
                        dump_path = agent.dump_context(reason="user_requested")
                        breakdown = agent.get_token_breakdown()
                        console.print(f"  [green]\u2713[/green] Context dumped to [cyan]{rich_escape(str(dump_path))}[/cyan]")
                        console.print(f"  [dim]System: {breakdown['system']:,}t \u00b7 Conversation: {breakdown['conversation']:,}t ({breakdown['message_count']} msgs) \u00b7 Total: {breakdown['total']:,}t[/dim]")
                        if agent.messages and agent.messages[0].role == "system":
                            sys_content = agent.messages[0].content
                            expected_tools = [
                                'read_file', 'write_to_file', 'replace_in_file',
                                'execute_command', 'manage_todos', 'create_plan',
                            ]
                            tools_missing = [t for t in expected_tools if t not in sys_content]
                            if tools_missing:
                                console.print(f"  [red]\u2717 System prompt missing tools: {', '.join(tools_missing)}[/red]")
                            else:
                                console.print(f"  [green]\u2713[/green] [dim]All {len(expected_tools)} core tools present in system prompt[/dim]")
                        else:
                            console.print("  [red]\u2717 No system message found![/red]")
                        continue
                    
                    elif cmd == '/config':
                        subparts = cmd_arg.split()
                        if subparts and subparts[0].lower() == "setup":
                            scope = subparts[1].lower() if len(subparts) > 1 else "active"
                            try:
                                result = run_in_app_config_wizard(workspace, console, agent, providers, scope)
                                console.print(f"  [dim]{result}[/dim]")
                            except KeyboardInterrupt:
                                console.print("\n  [dim]Config wizard cancelled.[/dim]")
                            continue

                        key_preview = agent.config.api_key[:8] + "..." if agent.config.api_key else "(not set)"
                        tbl = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
                        tbl.add_column("label", style="dim", width=13, justify="right")
                        tbl.add_column("value")
                        tbl.add_row("API URL", f"[cyan]{rich_escape(agent.config.api_url)}[/cyan]")
                        tbl.add_row("Model", f"[bold]{rich_escape(agent.config.model)}[/bold]")
                        tbl.add_row("API Key", key_preview)
                        tbl.add_row("Max Tokens", f"{agent.config.max_tokens:,}")
                        tbl.add_row("Temperature", str(agent.config.temperature))
                        tbl.add_row("Max Iters", str(agent.max_iterations))
                        if providers:
                            tbl.add_row("Providers", ", ".join(sorted(providers.keys())))
                        console.print()
                        console.print(Panel(tbl, title="[bold]Configuration[/bold]", border_style="dim", padding=(1, 2)))
                        console.print("  [dim][white]/providers[/white] to manage providers, [white]/model[/white] to switch models.[/dim]")
                        console.print()
                        continue

                    elif cmd in ('/provider', '/providers'):
                        try:
                            if cmd == '/providers':
                                result = run_providers_hub(workspace, console, agent, providers, cmd_arg)
                            else:
                                result = run_provider_manager(workspace, console, agent, providers, cmd_arg)
                            if result:
                                console.print(f"  [dim]{result}[/dim]")
                        except KeyboardInterrupt:
                            console.print("\n  [dim]Cancelled.[/dim]")
                        continue

                    elif cmd == '/model':
                        try:
                            result = run_model_switch_wizard(workspace, console, agent, providers, cmd_arg)
                            if result:
                                console.print(f"  [dim]{result}[/dim]")
                        except KeyboardInterrupt:
                            console.print("\n  [dim]Cancelled.[/dim]")
                        continue

                    elif cmd == '/maxctx':
                        if not cmd_arg.strip():
                            console.print(f"  [dim]Max tokens:[/dim] [bold]{agent.config.max_tokens:,}[/bold]")
                            console.print("  [dim]Usage: /maxctx <tokens>  e.g. /maxctx 8000, /maxctx 32k[/dim]")
                            continue
                        parsed = _parse_token_limit_input(cmd_arg)
                        if not parsed:
                            console.print("  [dim]Invalid value. Examples: /maxctx 8000, /maxctx 32k, /maxctx 0.5m[/dim]")
                            continue
                        if parsed < 256:
                            console.print("  [yellow]\u26a0[/yellow] Minimum allowed is 256 tokens")
                            continue
                        agent.config.max_tokens = parsed
                        agent.tool_handlers.config = agent.config
                        _save_active_config_fields(workspace, {"max_tokens": parsed})
                        active_profile = _infer_active_provider_profile(agent, providers)
                        if active_profile and active_profile in providers:
                            _save_provider_profile_fields(workspace, providers, active_profile, {"max_tokens": parsed})
                            providers[active_profile]["max_tokens"] = parsed
                        console.print(f"  [green]\u2713[/green] Max tokens set to [bold]{parsed:,}[/bold]")
                        continue
                    
                    elif cmd == '/iter':
                        if not cmd_arg:
                            console.print(f"  [dim]Max iterations:[/dim] [bold]{agent.max_iterations}[/bold]")
                            continue
                        try:
                            new_val = int(cmd_arg.strip())
                            if new_val < 1:
                                console.print("  [yellow]\u26a0[/yellow] Must be at least 1")
                                continue
                            agent.max_iterations = new_val
                            console.print(f"  [green]\u2713[/green] Max iterations set to [bold]{new_val}[/bold]")
                        except ValueError:
                            console.print("  [dim]Usage: /iter <number>[/dim]")
                        continue
                    
                    elif cmd == '/clip':
                        img_path, error = get_clipboard_image()
                        if error:
                            console.print(f"  [red]\u2717 {error}[/red]")
                            continue
                        
                        console.print(f"  [dim]Clipboard image: {img_path}[/dim]")
                        question = cmd_arg.strip() if cmd_arg else "Describe this image in detail."
                        user_input = f"Analyze this image: {img_path}\n\nQuestion: {question}"
                        # Fall through to process this request
                    
                    elif cmd == '/index':
                        idx = agent.workspace_index
                        if cmd_arg.strip().lower() == 'rebuild':
                            console.print("  [dim]Rebuilding workspace index...[/dim]")
                            idx.build()
                            console.print(f"  [green]\u2713[/green] Index rebuilt: {len(idx.files)} files in {idx._build_time:.2f}s")
                        elif cmd_arg.strip().lower() == 'tree':
                            console.print(f"[dim]{idx.compact_tree()}[/dim]")
                        else:
                            console.print(f"  [dim]{idx.summary()}[/dim]")
                            console.print("  [dim][white]/index rebuild[/white]  re-scan workspace[/dim]")
                            console.print("  [dim][white]/index tree[/white]     show file tree[/dim]")
                        continue
                    
                    elif cmd == '/log':
                        log_file = os.path.join(workspace, ".harness_output", "harness.log")
                        if not os.path.exists(log_file):
                            console.print("  [dim]No log file yet.[/dim]")
                        else:
                            size_kb = os.path.getsize(log_file) / 1024
                            console.print(f"  [dim]Log: {log_file} ({size_kb:.0f} KB)[/dim]")
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
                                console.print(f"  [dim]{rich_escape(ln.rstrip())}[/dim]")
                        continue
                    
                    elif cmd in ('/help', '/?'):
                        console.print()
                        console.print("  [bold]Chat[/bold]")
                        console.print("  [cyan]!command[/cyan]             [dim]Run a shell command[/dim]")
                        console.print("  [cyan]/clear[/cyan]               [dim]Clear conversation history[/dim]")
                        console.print("  [cyan]/compact[/cyan] [dim][strategy][/dim]   [dim]Compact context (half/quarter/last2)[/dim]")
                        console.print()
                        console.print("  [bold]Sessions[/bold]")
                        console.print("  [cyan]/sessions[/cyan]            [dim]List all sessions[/dim]")
                        console.print("  [cyan]/session[/cyan] [dim]<name>[/dim]      [dim]Switch or create session[/dim]")
                        console.print("  [cyan]/delete[/cyan] [dim]<name>[/dim]       [dim]Delete a session[/dim]")
                        console.print("  [cyan]/save[/cyan]                [dim]Save current session[/dim]")
                        console.print()
                        console.print("  [bold]Models & Providers[/bold]")
                        console.print("  [cyan]/model[/cyan] [dim]<query>[/dim]       [dim]Search and switch models[/dim]")
                        console.print("  [cyan]/model list[/cyan]          [dim]List models from current provider[/dim]")
                        console.print("  [cyan]/providers[/cyan]           [dim]Manage provider profiles[/dim]")
                        console.print("  [cyan]/config[/cyan]              [dim]Show/edit API configuration[/dim]")
                        console.print("  [cyan]/maxctx[/cyan] [dim]<n>[/dim]          [dim]Set max token cap (e.g. 8k, 32k)[/dim]")
                        console.print("  [cyan]/iter[/cyan] [dim]<n>[/dim]            [dim]Set max agent iterations[/dim]")
                        console.print()
                        console.print("  [bold]Tools[/bold]")
                        console.print("  [cyan]/todo[/cyan]                [dim]Manage todo list (add/done/rm/clear)[/dim]")
                        console.print("  [cyan]/bg[/cyan]                  [dim]List background processes[/dim]")
                        console.print("  [cyan]/clip[/cyan] [dim][question][/dim]     [dim]Analyze clipboard image[/dim]")
                        console.print()
                        console.print("  [bold]Diagnostics[/bold]")
                        console.print("  [cyan]/ctx[/cyan]                 [dim]Show context usage[/dim]")
                        console.print("  [cyan]/tokens[/cyan]              [dim]Token breakdown by message[/dim]")
                        console.print("  [cyan]/cost[/cyan]                [dim]API usage and cost totals[/dim]")
                        console.print("  [cyan]/smart[/cyan]               [dim]Smart context analysis[/dim]")
                        console.print("  [cyan]/dump[/cyan]                [dim]Dump full context to JSON[/dim]")
                        console.print("  [cyan]/index[/cyan] [dim][rebuild|tree][/dim] [dim]Project file index[/dim]")
                        console.print("  [cyan]/log[/cyan] [dim][n][/dim]             [dim]Show last n log lines[/dim]")
                        console.print()
                        console.print("  [bold]Keys[/bold]")
                        console.print("  [cyan]Esc[/cyan]                  [dim]Stop / interrupt agent[/dim]")
                        console.print("  [cyan]Ctrl+C[/cyan]               [dim]Interrupt (press twice to exit)[/dim]")
                        console.print("  [cyan]Ctrl+B[/cyan]               [dim]Send command to background[/dim]")
                        if HAS_PROMPT_TOOLKIT:
                            console.print("  [cyan]Ctrl+Enter[/cyan]           [dim]Insert newline[/dim]")
                            console.print("  [cyan]Ctrl+V[/cyan]               [dim]Paste clipboard image[/dim]")
                        console.print()
                        continue
                    
                    else:
                        console.print(f"  [dim]Unknown command. Type [white]/help[/white] for available commands.[/dim]")
                        continue
                
                multimodal_content: Optional[List[Dict[str, Any]]] = None
                multimodal_label: Optional[str] = None
                original_user_input_for_cleanup = user_input
                if isinstance(user_input, str) and "[[clipboard_image:" in user_input:
                    text_part, image_paths = _extract_clipboard_image_markers(user_input)
                    image_paths = [p for p in image_paths if p.exists()]
                    if image_paths:
                        if _supports_multimodal_input(agent.config.api_url, agent.config.model):
                            multimodal_content = _build_multimodal_user_content(text_part, image_paths)
                            multimodal_label = text_part or f"[pasted {len(image_paths)} image(s)]"
                            console.print(f"  [dim]Attached {len(image_paths)} image(s)[/dim]")
                        else:
                            fallback_q = text_part or "Describe this image in detail."
                            user_input = f"Analyze this image: {image_paths[0]}\n\nQuestion: {fallback_q}"
                            console.print("  [dim]Model may not support images \u2014 using fallback mode[/dim]")
                    else:
                        user_input, _ = _extract_clipboard_image_markers(user_input)

                log.info("User input [mode=%s]: %s", agent.reasoning_mode, truncate(multimodal_label or user_input, 200))
                try:
                    result = loop.run_until_complete(
                        run_single(
                            agent,
                            multimodal_content if multimodal_content is not None else user_input,
                            console,
                            user_label=multimodal_label,
                        )
                    )
                    log.info("run_single completed, result_len=%d", len(result or ""))
                    # Show live todo panel if there are any todos
                    agent.todo_manager.print_todo_panel(console)
                except KeyboardInterrupt:
                    log.warning("KeyboardInterrupt during run_single")
                    console.print("\n  [yellow]Interrupted[/yellow] [dim]- press Ctrl+C again to exit[/dim]")
                    last_interrupt_time = time.time()
                except Exception as e:
                    log_exception(log, "run_single failed", e)
                    console.print(f"  [red]\u2717 {rich_escape(str(e))}[/red]")
                
                # Auto-save session after each exchange
                agent.save_session(str(session_path))
                if isinstance(original_user_input_for_cleanup, str) and "[[clipboard_image:" in original_user_input_for_cleanup:
                    _, _tmp_paths = _extract_clipboard_image_markers(original_user_input_for_cleanup)
                    for _p in _tmp_paths:
                        try:
                            if "harness_clipboard" in str(_p).lower():
                                _p.unlink(missing_ok=True)
                        except Exception:
                            pass
                print()  # Blank line between requests
                
            except KeyboardInterrupt:
                cleanup_and_save()
                console.print("\n  [dim]Session saved. Goodbye![/dim]")
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
