#!/usr/bin/env python3
"""Streaming harness entry point - true token-by-token streaming."""

import sys
import os
import time as _time_mod
import warnings

# Suppress Pydantic serialization warnings from LiteLLM
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

_BOOT_T0 = _time_mod.perf_counter()
_boot_marks: list = []  # [(label, elapsed_since_boot)]


def _mark(label: str) -> None:
    _boot_marks.append((label, _time_mod.perf_counter() - _BOOT_T0))


def _print_boot_timing(console=None) -> None:
    """Print a table of startup phase durations."""
    if not _boot_marks:
        return
    try:
        from rich.table import Table as _T
        from rich.console import Console as _C

        con = console or _C()
        tbl = _T(
            title="[bold]startup timing[/bold]",
            show_header=True,
            box=None,
            padding=(0, 2),
            expand=False,
        )
        tbl.add_column("phase", style="dim")
        tbl.add_column("delta", justify="right")
        tbl.add_column("cumulative", justify="right", style="bold")
        prev = 0.0
        for label, cumul in _boot_marks:
            delta = cumul - prev
            style = "red" if delta > 1.0 else ("yellow" if delta > 0.3 else "")
            tbl.add_row(
                label,
                f"[{style}]{delta * 1000:8.0f}ms[/{style}]"
                if style
                else f"{delta * 1000:8.0f}ms",
                f"{cumul * 1000:8.0f}ms",
            )
            prev = cumul
        con.print()
        con.print(tbl)
        con.print()
    except Exception:
        prev = 0.0
        print("\n--- startup timing ---")
        for label, cumul in _boot_marks:
            delta = cumul - prev
            flag = " ***" if delta > 1.0 else ""
            print(
                f"  {label:25s}  delta={delta * 1000:8.1f}ms  cumul={cumul * 1000:8.1f}ms{flag}"
            )
            prev = cumul
        print()


# Force unbuffered output and UTF-8 encoding BEFORE any imports
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", write_through=True)
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # May fail on some systems

_mark("env_setup")

# Enable faulthandler to dump Python traceback on SIGSEGV, SIGFPE, SIGABRT, SIGBUS
# and on user signal (Ctrl+\ on Unix, or programmatically). This helps debug freezes.
import faulthandler
import signal

faulthandler.enable()
# Also enable traceback on SIGINT (Ctrl+C) - helps if Python is stuck in C code
# Note: this may not work on Windows for all cases, but helps on Unix
try:
    faulthandler.register(signal.SIGINT)
except (AttributeError, RuntimeError, OSError):
    pass  # Not available on all platforms

import asyncio
import base64
import hashlib
import time
import json
import mimetypes
import re
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
import concurrent.futures
import httpx

_mark("stdlib_imports")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from harness.config import Config

_mark("import_config")
from harness.cline_agent import ClineAgent

_mark("import_cline_agent")
from harness.checkpoint import CheckpointManager
from harness.cost_tracker import get_global_tracker, reset_global_tracker
from harness.logger import (
    init_logging,
    get_logger,
    log_exception,
    truncate,
    enable_debug,
    debug_print,
)

_mark("import_harness_core")
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape as rich_escape
from rich import box

_mark("import_rich")

# Initialize logger
log = get_logger("main")

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
            safe_string = string.encode("utf-8", errors="replace").decode("utf-8")
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


def run_install(
    api_url: str = None,
    api_key: str = None,
    model: str = None,
    global_config: bool = True,
):
    """Setup wizard for API configuration. Supports headless mode with CLI args.

    Args:
        api_url: API base URL (headless mode)
        api_key: API key (headless mode)
        model: Model name (headless mode)
        global_config: Deprecated; config is always saved to ~/.z.json
    """
    from pathlib import Path
    import json

    # Headless mode - all params provided
    if api_url and api_key:
        config_data = {
            "api_url": api_url.rstrip("/") + "/",
            "api_key": api_key,
            "model": model or "glm-4.7",
        }

        config_path = Path.home() / ".z.json"

        config_path.write_text(json.dumps(config_data, indent=2))
        print(f"Configuration saved to: {config_path}")
        print(f"  URL:   {api_url}")
        print(f"  Model: {config_data['model']}")
        print(f"  Key:   {api_key[:10]}...")
        return

    # Interactive mode
    con = Console()
    con.print()
    con.print(
        Panel(
            "[bold]Welcome to Harness[/bold]\n\n[dim]Let's configure your LLM provider.[/dim]",
            border_style="bright_blue",
            padding=(1, 3),
            width=50,
        )
    )
    con.print()
    con.print("  Select your LLM provider:\n")
    con.print("  [cyan][1][/cyan] Z.AI Coding Plan [dim](recommended)[/dim]")
    con.print("  [cyan][2][/cyan] Z.AI Standard API")
    con.print("  [cyan][3][/cyan] MiniMax")
    con.print("  [cyan][4][/cyan] Amazon Bedrock")
    con.print("  [cyan][5][/cyan] Together AI")
    con.print("  [cyan][6][/cyan] Anthropic")
    con.print("  [cyan][7][/cyan] OpenRouter")
    con.print("  [cyan][8][/cyan] OpenAI")
    con.print("  [cyan][9][/cyan] Groq")
    con.print("  [cyan][10][/cyan] DeepSeek")
    con.print("  [cyan][11][/cyan] Mistral AI")
    con.print("  [cyan][12][/cyan] Cohere")
    con.print("  [cyan][13][/cyan] Fireworks AI")
    con.print("  [cyan][14][/cyan] Perplexity")
    con.print("  [cyan][15][/cyan] AI21")
    con.print("  [cyan][16][/cyan] xAI (Grok)")
    con.print("  [cyan][17][/cyan] Google Gemini")
    con.print("  [cyan][18][/cyan] Cerebras")
    con.print("  [cyan][19][/cyan] Databricks")
    con.print("  [cyan][20][/cyan] Replicate")
    con.print("  [cyan][21][/cyan] Anyscale")
    con.print("  [cyan][22][/cyan] Ollama Cloud")
    con.print("  [cyan][23][/cyan] OpenAI Subscription (OAuth)")
    con.print("  [cyan][24][/cyan] GitHub Copilot (OAuth)")
    con.print("  [cyan][25][/cyan] Custom OpenAI-compatible API")
    con.print()

    while True:
        choice = input("Enter choice [1-25]: ").strip()
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
            base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
            provider = "Amazon Bedrock"
            default_model = "qwen.qwen3-32b-v1:0"
            break
        elif choice == "5":
            base_url = "https://api.together.xyz/v1/"
            provider = "Together AI"
            default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            break
        elif choice == "6":
            base_url = "https://api.anthropic.com/v1/"
            provider = "Anthropic"
            default_model = "claude-3-5-sonnet-latest"
            break
        elif choice == "7":
            base_url = "https://openrouter.ai/api/v1/"
            provider = "OpenRouter"
            default_model = "anthropic/claude-3.5-sonnet"
            break
        elif choice == "8":
            base_url = "https://api.openai.com/v1/"
            provider = "OpenAI"
            default_model = "gpt-4o"
            break
        elif choice == "9":
            base_url = "https://api.groq.com/openai/v1/"
            provider = "Groq"
            default_model = "llama-3.3-70b-versatile"
            break
        elif choice == "10":
            base_url = "https://api.deepseek.com/v1/"
            provider = "DeepSeek"
            default_model = "deepseek-chat"
            break
        elif choice == "11":
            base_url = "https://api.mistral.ai/v1/"
            provider = "Mistral AI"
            default_model = "mistral-large-latest"
            break
        elif choice == "12":
            base_url = "https://api.cohere.ai/v1/"
            provider = "Cohere"
            default_model = "command-r-plus"
            break
        elif choice == "13":
            base_url = "https://api.fireworks.ai/inference/v1/"
            provider = "Fireworks AI"
            default_model = "accounts/fireworks/models/llama-v3p1-70b-instruct"
            break
        elif choice == "14":
            base_url = "https://api.perplexity.ai/"
            provider = "Perplexity"
            default_model = "llama-3.1-sonar-large-128k-online"
            break
        elif choice == "15":
            base_url = "https://api.ai21.com/studio/v1/"
            provider = "AI21"
            default_model = "jamba-1.5-large"
            break
        elif choice == "16":
            base_url = "https://api.x.ai/v1/"
            provider = "xAI (Grok)"
            default_model = "grok-2-latest"
            break
        elif choice == "17":
            base_url = "https://generativelanguage.googleapis.com/v1beta/"
            provider = "Google Gemini"
            default_model = "gemini-1.5-pro-latest"
            break
        elif choice == "18":
            base_url = "https://api.cerebras.ai/v1/"
            provider = "Cerebras"
            default_model = "llama3.1-70b"
            break
        elif choice == "19":
            base_url = input(
                "Databricks workspace URL (e.g., https://my-workspace.cloud.databricks.com/serving-endpoints/): "
            ).strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Databricks"
            default_model = "databricks-meta-llama-3-1-70b-instruct"
            break
        elif choice == "20":
            base_url = "https://api.replicate.com/v1/"
            provider = "Replicate"
            default_model = "meta/meta-llama-3-70b-instruct"
            break
        elif choice == "21":
            base_url = "https://api.endpoints.anyscale.com/v1/"
            provider = "Anyscale"
            default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
            break
        elif choice == "22":
            base_url = "https://ollama.com/v1/"
            provider = "Ollama Cloud"
            default_model = "llama3.1"
            break
        elif choice == "23":
            base_url = "https://api.openai.com/v1/"
            provider = "OpenAI Subscription (OAuth)"
            default_model = "gpt-4o"
            break
        elif choice == "24":
            base_url = "https://api.githubcopilot.com/"
            provider = "GitHub Copilot (OAuth)"
            default_model = "gpt-4o-copilot"
            break
        elif choice == "25":
            base_url = input("Enter API base URL: ").strip()
            if not base_url:
                print("URL is required.")
                continue
            provider = "Custom"
            default_model = input("Enter default model name: ").strip() or "gpt-4"
            break
        else:
            print("Please enter 1-25.")

    con.print(f"\n  [green]\u2713[/green] Using [bold]{provider}[/bold]")
    con.print(f"    [dim]{base_url}[/dim]\n")

    # Check if OAuth provider
    is_oauth = "(OAuth)" in provider

    if is_oauth:
        # OAuth flow
        con.print("  [dim]This provider uses OAuth authentication.[/dim]")

        # Import OAuth manager
        try:
            from harness.oauth import get_oauth_manager

            oauth_manager = get_oauth_manager()

            # Map provider name to OAuth provider ID
            if "OpenAI" in provider:
                oauth_provider_id = "openai"
            else:
                oauth_provider_id = "github-copilot"

            # For OpenAI, let user choose method
            oauth_method = "browser"
            enterprise_url = None
            if "OpenAI" in provider:
                con.print("\n  Select OAuth method:")
                con.print("  [1] Browser-based (opens browser for authorization)")
                con.print("  [2] Device code (headless, enter code manually)")
                method_choice = input("\n  Enter choice [1/2]: ").strip()
                oauth_method = "device" if method_choice == "2" else "browser"
            elif "GitHub Copilot" in provider:
                # GitHub Copilot only supports device code flow
                con.print("\n  GitHub Copilot uses device code authentication.")

                # Ask about GitHub Enterprise
                is_enterprise = (
                    input("  Is this GitHub Enterprise? [y/N]: ").strip().lower()
                )
                if is_enterprise in ("y", "yes"):
                    enterprise_url = input(
                        "  Enter GitHub Enterprise domain (e.g., company.ghe.com): "
                    ).strip()

            con.print("\n  Opening browser for authentication...\n")

            # Trigger OAuth flow with selected method
            token = oauth_manager.authenticate(
                oauth_provider_id,
                method=oauth_method,
                timeout=300,
                enterprise_url=enterprise_url,
            )
            if token:
                api_key = f"oauth:{token.access_token}"
                con.print(f"  [green]✓[/green] OAuth authentication successful!\n")
            else:
                con.print("  [red]✗[/red] OAuth authentication failed.\n")
                return
        except Exception as e:
            con.print(f"  [red]✗[/red] OAuth error: {e}\n")
            return
    else:
        # API Key flow
        api_key = ""
        while not api_key:
            api_key = input("API Key: ").strip()
            if not api_key:
                print("API key is required.")

    # Model
    if is_oauth:
        from harness.codex_models import get_codex_models
        from harness.copilot_oauth_client import get_copilot_models

        if "GitHub Copilot" in provider:
            print(
                "  [dim]Note: GitHub Copilot OAuth tokens access Copilot models directly.[/dim]"
            )
            copilot_models = get_copilot_models()
            print(f"\n  Available Copilot models:")
            for i, m in enumerate(copilot_models, 1):
                marker = "●" if m == default_model else " "
                print(f"    {marker} [{i}] {m}")

            print(
                f"\n  Select model [1-{len(copilot_models)}] or enter name (default: {default_model}): ",
                end="",
                flush=True,
            )
            model_choice = input().strip()
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(copilot_models):
                model = copilot_models[int(model_choice) - 1]
            else:
                model = model_choice or default_model
        else:
            print(
                "  [dim]Note: OAuth tokens access ChatGPT Codex models directly.[/dim]"
            )

            # Show available Codex models (hardcoded for instant display)
            codex_models = get_codex_models()
            print(f"\n  Available Codex models:")
            for i, m in enumerate(codex_models, 1):
                marker = "●" if m == default_model else " "
                print(f"    {marker} [{i}] {m}")

            print(
                f"\n  Select model [1-{len(codex_models)}] or enter name (default: {default_model}): ",
                end="",
                flush=True,
            )
            model_choice = input().strip()
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(codex_models):
                model = codex_models[int(model_choice) - 1]
            else:
                model = model_choice or default_model
    else:
        model_input = input(f"\nModel name (default: {default_model}): ").strip()
        model = model_input or default_model

    # Build config
    config_data = {
        "api_url": base_url,
        "api_key": api_key,
        "model": model,
    }

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
    con.print(
        Panel(
            tbl,
            title="[bold green] Setup Complete [/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
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
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
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

# Regex to find image file paths typed directly in user input.
# Matches absolute paths (C:\... or /...) and relative paths ending with image ext.
_IMAGE_PATH_RE = re.compile(
    r"""(?:(?:[A-Za-z]:[\\\/]|[\/\\])  # drive letter or leading slash
        (?:[^\s"<>|*?]+)               # path characters
        \.(?:jpg|jpeg|png|gif|webp|bmp) # image extension
    )
    |(?:\.\.?[\/\\]\S+\.(?:jpg|jpeg|png|gif|webp|bmp))  # relative ./foo.png ../bar.jpg
    """,
    re.IGNORECASE | re.VERBOSE,
)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _extract_image_paths_from_text(text: str) -> tuple[str, List[Path]]:
    """Detect image file paths typed directly in user input.

    Returns (cleaned_text, list_of_existing_image_paths).
    The cleaned text has the raw paths removed so the model sees only
    the user's question/instruction.
    """
    found: List[Path] = []
    spans_to_remove: List[tuple[int, int]] = []
    for m in _IMAGE_PATH_RE.finditer(text):
        raw = m.group(0).strip()
        p = Path(raw)
        if p.exists() and p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            found.append(p)
            spans_to_remove.append((m.start(), m.end()))
    if not spans_to_remove:
        return text, []
    # Remove matched spans from text (reverse order to preserve indices)
    chars = list(text)
    for start, end in reversed(spans_to_remove):
        del chars[start:end]
    cleaned = "".join(chars)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, found


def _supports_multimodal_input(api_url: str, model: str) -> bool:
    """Best-effort heuristic for vision-capable chat models/providers.

    Most modern LLMs accept image input via the OpenAI multipart content
    format.  We default to True and only return False for models we
    positively know cannot do vision.
    """
    # Almost all current-gen models support vision.  Return True broadly;
    # the worst case is the provider silently ignoring the image block.
    return True


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


def _build_multimodal_user_content(
    text: str, image_paths: List[Path]
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    prompt_text = (
        text.strip() if text and text.strip() else "Please analyze the pasted image."
    )
    blocks.append({"type": "text", "text": prompt_text})
    for p in image_paths:
        blocks.append(
            {"type": "image_url", "image_url": {"url": _image_path_to_data_uri(p)}}
        )
    return blocks


def get_sessions_dir(workspace: str) -> Path:
    """Get sessions directory for a workspace.
    
    Sessions are stored globally in ~/.z/sessions/<workspace_hash>/ so that
    both main harness and safe mode can access the same sessions.
    """
    # Use global ~/.z/sessions/ directory (not relative to harness installation)
    # This ensures safe mode and main harness share the same sessions
    sessions_dir = Path.home() / ".z" / "sessions"
    # Normalise the workspace path so that case differences on Windows
    # (e.g. C:\Projects\evoke vs c:\projects\evoke) map to the same hash.
    normalised = os.path.normcase(os.path.normpath(workspace))
    workspace_hash = hashlib.md5(normalised.encode()).hexdigest()[:12]
    workspace_sessions = sessions_dir / workspace_hash
    workspace_sessions.mkdir(parents=True, exist_ok=True)
    
    # Migrate from old location (harness_dir/.sessions/) if needed
    old_sessions_dir = Path(__file__).parent / ".sessions" / workspace_hash
    if old_sessions_dir.exists() and old_sessions_dir != workspace_sessions:
        import shutil
        for old_session in old_sessions_dir.glob("*.json"):
            new_session = workspace_sessions / old_session.name
            if not new_session.exists():
                try:
                    shutil.copy2(old_session, new_session)
                except Exception:
                    pass  # Ignore migration errors
    
    return workspace_sessions


def get_session_path(workspace: str, session_name: str = "default") -> Path:
    """Get session file path for a workspace and session name."""
    return get_sessions_dir(workspace) / f"{session_name}.json"


def _get_global_config_path() -> Path:
    return Path.home() / ".z.json"


def _get_legacy_global_models_path() -> Path:
    return Path.home() / ".z" / "models.json"


def load_providers(workspace: str) -> Dict[str, dict]:
    """Load provider configs from ~/.z.json (single-file config)."""
    import json

    cfg_path = _get_global_config_path()
    data = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}

    providers = dict(data.get("providers", {}) or {})

    # One-time migration from legacy models file if providers are missing.
    if not providers:
        legacy_path = _get_legacy_global_models_path()
        if legacy_path.exists():
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8-sig"))
                providers = dict(legacy.get("providers", {}) or {})
                if providers:
                    data["providers"] = providers
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    return providers


PROVIDER_PRESETS = {
    "zai-coding": ("Z.AI Coding", "https://api.z.ai/api/coding/paas/v4/", "glm-4.7"),
    "zai-standard": ("Z.AI Standard", "https://api.z.ai/api/paas/v4/", "glm-4.7"),
    "minimax": ("MiniMax", "https://api.minimax.io/v1/", "MiniMax-M2.1"),
    "bedrock": (
        "Amazon Bedrock",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        "qwen.qwen3-32b-v1:0",
    ),
    "together": (
        "Together AI",
        "https://api.together.xyz/v1/",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "anthropic": (
        "Anthropic",
        "https://api.anthropic.com/v1/",
        "claude-3-5-sonnet-latest",
    ),
    "openrouter": (
        "OpenRouter",
        "https://openrouter.ai/api/v1/",
        "anthropic/claude-3.5-sonnet",
    ),
    "openai": ("OpenAI", "https://api.openai.com/v1/", "gpt-4o"),
    "groq": (
        "Groq",
        "https://api.groq.com/openai/v1/",
        "llama-3.3-70b-versatile",
    ),
    "deepseek": (
        "DeepSeek",
        "https://api.deepseek.com/v1/",
        "deepseek-chat",
    ),
    "mistral": (
        "Mistral AI",
        "https://api.mistral.ai/v1/",
        "mistral-large-latest",
    ),
    "cohere": (
        "Cohere",
        "https://api.cohere.ai/v1/",
        "command-r-plus",
    ),
    "fireworks": (
        "Fireworks AI",
        "https://api.fireworks.ai/inference/v1/",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    ),
    "perplexity": (
        "Perplexity",
        "https://api.perplexity.ai/",
        "llama-3.1-sonar-large-128k-online",
    ),
    "ai21": (
        "AI21",
        "https://api.ai21.com/studio/v1/",
        "jamba-1.5-large",
    ),
    "xai": (
        "xAI (Grok)",
        "https://api.x.ai/v1/",
        "grok-2-latest",
    ),
    "gemini": (
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/",
        "gemini-1.5-pro-latest",
    ),
    "cerebras": (
        "Cerebras",
        "https://api.cerebras.ai/v1/",
        "llama3.1-70b",
    ),
    "databricks": (
        "Databricks",
        "https://<your-workspace>.cloud.databricks.com/serving-endpoints/",
        "databricks-meta-llama-3-1-70b-instruct",
    ),
    "replicate": (
        "Replicate",
        "https://api.replicate.com/v1/",
        "meta/meta-llama-3-70b-instruct",
    ),
    "anyscale": (
        "Anyscale",
        "https://api.endpoints.anyscale.com/v1/",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    "ollama-cloud": (
        "Ollama Cloud",
        "https://ollama.com/v1/",
        "llama3.1",
    ),
    "openai-subscription": (
        "OpenAI Subscription (OAuth)",
        "https://api.openai.com/v1/",
        "gpt-4o",
    ),
    "github-copilot": (
        "GitHub Copilot (OAuth)",
        "https://api.githubcopilot.com/",
        "gpt-4o",
    ),
}
_MODEL_FETCH_CACHE: Dict[str, tuple[float, List[str]]] = {}
_MODEL_FETCH_CACHE_TTL_SECS = 300
_LAST_MODEL_SEARCH_RESULTS: List[Dict[str, Any]] = []
_LAST_MODEL_SEARCH_QUERY: str = ""
_MODEL_HISTORY_MAX = 20


def _load_model_history() -> List[Dict[str, str]]:
    """Load model history from ~/.z.json (most recent first)."""
    cfg_path = Path.home() / ".z.json"
    if not cfg_path.exists():
        return []
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        return list(data.get("model_history", []))
    except Exception:
        return []


def _record_model_history(model: str, profile: str) -> None:
    """Record a model switch in the MRU history (persisted to ~/.z.json)."""
    cfg_path = Path.home() / ".z.json"
    data = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
    history: List[dict] = list(data.get("model_history", []))
    entry = {"model": model, "profile": profile}
    # Remove existing entry for same model+profile to move it to front
    history = [
        h
        for h in history
        if not (h.get("model") == model and h.get("profile") == profile)
    ]
    history.insert(0, entry)
    history = history[:_MODEL_HISTORY_MAX]
    data["model_history"] = history
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _provider_family_for_url(api_url: str) -> str:
    u = (api_url or "").lower()
    if "bedrock" in u and "amazonaws.com" in u:
        return "bedrock"
    if "api.anthropic.com" in u:
        return "anthropic"
    if "openrouter.ai" in u:
        return "openrouter"
    if "api.openai.com" in u:
        return "openai"
    if "together.xyz" in u:
        return "together"
    if "minimax" in u:
        return "minimax"
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


def _fetch_provider_model_ids(api_url: str, api_key: str) -> List[str]:
    """Fetch model IDs from a provider using LiteLLM."""
    # Check if OAuth token FIRST, before importing streaming_client
    # This avoids the slow LiteLLM import for OAuth providers
    if api_key and api_key.startswith("oauth:"):
        # Check URL to determine which OAuth provider
        url_lower = (api_url or "").lower()
        if "githubcopilot" in url_lower or "copilot" in url_lower:
            # GitHub Copilot OAuth - return Copilot models
            from harness.copilot_oauth_client import get_copilot_models

            return get_copilot_models()
        else:
            # OpenAI OAuth - return Codex models
            from harness.codex_models import get_codex_models

            return get_codex_models()

    from harness.streaming_client import search_litellm_models

    # Extract model prefix from URL
    url = (api_url or "").lower()

    if "bedrock" in url and "amazonaws.com" in url:
        # Query actual AWS Bedrock API for available models
        from harness.bedrock_provider import list_bedrock_models

        # Extract region from URL
        region = "us-east-1"
        if ".amazonaws.com" in url:
            parts = url.split(".")
            if len(parts) >= 2:
                potential_region = parts[1]
                if potential_region.startswith("us-") or potential_region.startswith(
                    "eu-"
                ):
                    region = potential_region

        models = list_bedrock_models(api_key, region)
        if models:
            return models
        # Fallback to LiteLLM list if API query fails
        return search_litellm_models("bedrock/")
    elif "anthropic.com" in url:
        return search_litellm_models("anthropic/")
    elif "openrouter.ai" in url:
        # Query OpenRouter's actual API (LiteLLM registry may be stale)
        models = _fetch_models_from_provider_api(api_url, api_key)
        if models:
            # Prefix with openrouter/ for LiteLLM routing
            return [f"openrouter/{m}" if not m.startswith("openrouter/") else m for m in models]
        return search_litellm_models("openrouter/")
    elif "together.xyz" in url:
        # Query Together's actual API (LiteLLM registry may be empty/stale)
        models = _fetch_models_from_provider_api(api_url, api_key)
        if models:
            return models
        return search_litellm_models("together_ai/")
    elif "minimax" in url:
        return search_litellm_models("minimax/")
    elif "api.deepseek.com" in url:
        return search_litellm_models("deepseek/")
    elif "api.groq.com" in url:
        return search_litellm_models("groq/")

    # For custom/unknown providers, query their API directly
    return _fetch_models_from_provider_api(api_url, api_key)


def _fetch_models_from_provider_api(api_url: str, api_key: str) -> List[str]:
    """Query provider's /v1/models endpoint (OpenAI-compatible format).

    For custom providers, we query their actual API instead of using LiteLLM's
    global model registry. This ensures we only show models actually available.
    """
    import requests

    try:
        headers = {}
        if api_key and not api_key.startswith("oauth:"):
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(
            f"{api_url.rstrip('/')}/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Extract model IDs - handle both OpenAI format {"data": [...]}
        # and plain list format [...] (e.g. Together AI)
        items = data if isinstance(data, list) else data.get("data", [])
        models = []
        for item in items:
            if isinstance(item, dict) and "id" in item:
                models.append(item["id"])

        return sorted(models)
    except Exception:
        # On any error, return empty list to trigger manual entry
        return []


def _cache_key_for_models(api_url: str, api_key: str) -> str:
    key_hash = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:12]
    return f"{api_url.rstrip('/').lower()}|{key_hash}"


def _fetch_provider_model_ids_cached(
    api_url: str, api_key: str, refresh: bool = False
) -> List[str]:
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
        prompt = (
            f"\n  Choose number, type model id, or Enter to keep [{current_model}]: "
        )
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
        _save_provider_profile_fields(
            workspace, providers, profile, {"model": chosen_model}
        )
    cfg_path = _save_active_config_fields(
        workspace,
        {
            "api_url": agent.config.api_url,
            "api_key": agent.config.api_key,
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
        },
    )
    _record_model_history(chosen_model, profile)
    return f"\u2713 Switched to [bold]{chosen_model}[/bold] via {profile}"


def _build_searchable_providers(
    agent: ClineAgent, providers: Dict[str, dict]
) -> tuple[List[tuple[str, dict]], Optional[str]]:
    searchable: List[tuple[str, dict]] = []
    for name in sorted(providers.keys()):
        cfg = dict(providers.get(name, {}))
        if cfg.get("api_url") and cfg.get("api_key"):
            searchable.append((name, cfg))
    active_name = _infer_active_provider_profile(agent, providers)
    # Only add "active" synthetic provider as fallback when no saved providers exist
    # This avoids confusing duplicates when user has configured provider profiles
    if (
        not searchable
        and not active_name
        and agent.config.api_url
        and agent.config.api_key
    ):
        searchable.insert(
            0,
            (
                "active",
                {
                    "api_url": agent.config.api_url,
                    "api_key": agent.config.api_key,
                    "model": agent.config.model,
                    "max_tokens": agent.config.max_tokens,
                    "temperature": agent.config.temperature,
                },
            ),
        )
    return searchable, active_name


def _provider_display_name(profile: str, cfg: Dict[str, Any]) -> str:
    """Human-friendly provider label for search results."""
    if profile != "active":
        return profile
    fam = _provider_family_for_url(str(cfg.get("api_url", "")))
    return f"active/{fam}"


def _save_active_config_fields(workspace: str, updates: dict) -> Path:
    cfg_path = Path.home() / ".z.json"
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


def _save_provider_profile_fields(
    workspace: str, providers: Dict[str, dict], profile: str, updates: dict
) -> Path:
    models_path = _get_global_config_path()
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

    # /model <number> — switch from last displayed list (same as /model use <n>)
    if verb and verb.isdigit() and len(parts) == 1:
        idx = int(verb)
        if not _LAST_MODEL_SEARCH_RESULTS:
            return (
                "No model list displayed yet. Use /model to see history or /model <query> to search first."
            )
        if idx < 1 or idx > len(_LAST_MODEL_SEARCH_RESULTS):
            return f"Invalid selection. List has {len(_LAST_MODEL_SEARCH_RESULTS)} item(s)."
        row = _LAST_MODEL_SEARCH_RESULTS[idx - 1]
        return _apply_selected_provider_model(
            workspace,
            agent,
            providers,
            row["profile"],
            dict(row["cfg"]),
            row["model_id"],
        )

    if verb == "use":
        if len(parts) < 2 or not parts[1].isdigit():
            return "Usage: /model use <number> (pick from the last displayed /model list)"
        idx = int(parts[1])
        if not _LAST_MODEL_SEARCH_RESULTS:
            return "No model list displayed yet. Use /model to see history or /model <query> to search first."
        if idx < 1 or idx > len(_LAST_MODEL_SEARCH_RESULTS):
            return f"Invalid selection. List has {len(_LAST_MODEL_SEARCH_RESULTS)} item(s)."
        row = _LAST_MODEL_SEARCH_RESULTS[idx - 1]
        return _apply_selected_provider_model(
            workspace,
            agent,
            providers,
            row["profile"],
            dict(row["cfg"]),
            row["model_id"],
        )

    if verb == "list":
        api_url = agent.config.api_url
        api_key = agent.config.api_key
        if not api_url or not api_key:
            return "No active provider configured."
        try:
            console.print("  [dim]Fetching models...[/dim]")
            # Always refresh for list command to get latest models
            mids = _fetch_provider_model_ids_cached(api_url, api_key, refresh=True)
        except Exception as e:
            return f"Model fetch failed: {e}"
        shown = mids
        console.print(f"\n  [bold]Models[/bold] [dim]({len(mids)} total)[/dim]\n")
        for m in shown:
            marker = "[cyan]\u25cf[/cyan]" if m == agent.config.model else " "
            console.print(f"  {marker} {m}")
        console.print()
        return ""

    refresh = verb == "refresh"
    if verb in ("search", "refresh"):
        query = " ".join(parts[1:]).strip()
    else:
        query = " ".join(parts).strip()

    if not query:
        # Show MRU model history
        history = _load_model_history()
        if history:
            shown = history[:10]
            # Populate _LAST_MODEL_SEARCH_RESULTS so /model use <n> works
            # with the displayed MRU list
            _LAST_MODEL_SEARCH_RESULTS = []
            for entry in shown:
                m_profile = entry.get("profile", "")
                p_cfg = dict(providers.get(m_profile, {}))
                _LAST_MODEL_SEARCH_RESULTS.append({
                    "profile": m_profile,
                    "provider_display": m_profile,
                    "model_id": entry.get("model", ""),
                    "cfg": p_cfg,
                })
            console.print()
            tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
            tbl.add_column(width=2)
            tbl.add_column("num", style="dim", width=4)
            tbl.add_column("model", style="bold")
            tbl.add_column("provider", style="cyan")
            for i, entry in enumerate(shown, 1):
                m_model = entry.get("model", "")
                m_profile = entry.get("profile", "")
                mark = (
                    "[cyan]\u25cf[/cyan]"
                    if m_model == agent.config.model
                    and m_profile == _infer_active_provider_profile(agent, providers)
                    else " "
                )
                tbl.add_row(mark, f"[{i}]", m_model, m_profile)
            console.print(tbl)
            console.print(
                "\n  [dim]Use [white]/model <n>[/white] to switch, or [white]/model <query>[/white] to search.[/dim]\n"
            )
            return ""
        return "No model history yet. Use /model <query> to search for models."

    searchable, active_name = _build_searchable_providers(agent, providers)
    if not searchable:
        return "No configured providers. Use /providers setup <name> first."
    if len(searchable) == 1:
        only_name, only_cfg = searchable[0]
        console.print(
            f"[dim]Searching only one configured provider: {_provider_display_name(only_name, only_cfg)}. "
            "Add more via /providers setup <name> to compare across providers.[/dim]"
        )

    aggregate: List[
        tuple[str, str, str, dict]
    ] = []  # (profile, provider_display, model_id, cfg)
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
        # No models found from any provider (e.g., all are Bedrock/MiniMax)
        # If query looks like a model ID, offer to switch directly
        if "." in query or "/" in query:
            console.print(
                f"  [dim]No searchable providers found. Switching to model '{query}' on current provider...[/dim]"
            )
            current_cfg = (
                searchable[0][1]
                if searchable
                else {
                    "api_url": agent.config.api_url,
                    "api_key": agent.config.api_key,
                }
            )
            return _apply_selected_provider_model(
                workspace, agent, providers, active_name or "active", current_cfg, query
            )
        return "No models found from configured providers."

    q = query.lower().strip()
    # Normalize spaces to hyphens so "sonnet 4.6" matches "claude-sonnet-4.6"
    q_hyphen = q.replace(" ", "-")
    q_words = q.split()
    matches = [
        row
        for row in aggregate
        if (
            q in row[2].lower()
            or q in row[1].lower()
            or q in row[0].lower()
            or q_hyphen in row[2].lower()
            or (len(q_words) > 1 and all(w in row[2].lower() for w in q_words))
        )
    ]
    if not matches:
        # If query looks like a model ID and current provider doesn't support listing,
        # just switch to it directly
        if "." in query or "/" in query:
            current_cfg = (
                searchable[0][1]
                if searchable
                else {
                    "api_url": agent.config.api_url,
                    "api_key": agent.config.api_key,
                }
            )
            console.print(
                f"  [dim]Model not in searchable list. Switching to '{query}'...[/dim]"
            )
            return _apply_selected_provider_model(
                workspace, agent, providers, active_name or "active", current_cfg, query
            )
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
        matches.sort(
            key=lambda r: (
                0 if r[2].lower() == q else 1,
                0 if r[2].lower().startswith(q) else 1,
                0 if q in r[2].lower() else 1,
                r[0].lower(),
                r[2].lower(),
            )
        )

    shown = matches
    _LAST_MODEL_SEARCH_QUERY = query
    _LAST_MODEL_SEARCH_RESULTS = [
        {
            "profile": profile,
            "provider_display": provider_display,
            "model_id": mid,
            "cfg": dict(cfg),
        }
        for (profile, provider_display, mid, cfg) in shown
    ]

    console.print(
        f"\n  [bold]Model Search[/bold] [dim]'{query}' \u2014 {len(matches)} match(es)[/dim]\n"
    )
    tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    tbl.add_column(width=2)
    tbl.add_column("num", style="dim", width=4)
    tbl.add_column("model", style="bold")
    tbl.add_column("provider", style="cyan")
    for i, (profile, provider_display, mid, cfg) in enumerate(shown, 1):
        active_mark = (
            "[cyan]\u25cf[/cyan]"
            if (
                (profile == active_name or profile == "active")
                and mid == agent.config.model
            )
            else " "
        )
        tbl.add_row(active_mark, f"[{i}]", mid, provider_display)
    console.print(tbl)

    # QoL: if query resolves cleanly, switch immediately (provider + model).
    exact_matches = [row for row in matches if row[2].lower() == q]
    if len(exact_matches) == 1:
        profile, _provider_display, chosen_model, cfg = exact_matches[0]
        return _apply_selected_provider_model(
            workspace, agent, providers, profile, dict(cfg), chosen_model
        )
    if len(matches) == 1:
        profile, _provider_display, chosen_model, cfg = matches[0]
        return _apply_selected_provider_model(
            workspace, agent, providers, profile, dict(cfg), chosen_model
        )

    console.print(
        "\n  [dim]Use [white]/model use <n>[/white] to switch to a result.[/dim]\n"
    )
    return ""


def _choose_provider_preset_interactive(
    current_api_url: str, current_model: str
) -> tuple[str, str, str, str, str]:
    """Prompt user for provider preset.

    Returns (preset_key, label, api_url, default_model, profile_name).
    preset_key is the PROVIDER_PRESETS key (e.g. "zai-coding") or "custom".
    profile_name is the user-defined name for custom providers.
    """
    presets = [
        ("1", "zai-coding"),
        ("2", "zai-standard"),
        ("3", "minimax"),
        ("4", "bedrock"),
        ("5", "together"),
        ("6", "anthropic"),
        ("7", "openrouter"),
        ("8", "openai"),
        ("9", "groq"),
        ("10", "deepseek"),
        ("11", "mistral"),
        ("12", "cohere"),
        ("13", "fireworks"),
        ("14", "perplexity"),
        ("15", "ai21"),
        ("16", "xai"),
        ("17", "gemini"),
        ("18", "cerebras"),
        ("19", "databricks"),
        ("20", "replicate"),
        ("21", "anyscale"),
        ("22", "ollama-cloud"),
        ("23", "openai-subscription"),
        ("24", "github-copilot"),
        ("25", "custom"),
    ]
    con = Console()
    con.print("\n  [bold]Select provider:[/bold]\n")
    for num, key in presets:
        if key == "custom":
            con.print(f"  [cyan][{num}][/cyan] Custom URL")
        else:
            label, url, model = PROVIDER_PRESETS[key]
            con.print(
                f"  [cyan][{num}][/cyan] [bold]{label}[/bold]  [dim]{model}  ·  {url}[/dim]"
            )
    con.print()
    while True:
        choice = input("  Enter choice [1-25]: ").strip() or "8"
        selected = dict(presets).get(choice)
        if not selected:
            print("  Please enter 1-25.")
            continue
        if selected == "custom":
            api_url = (
                input(
                    f"  API URL [{current_api_url or 'https://api.example.com/v1/'}]: "
                ).strip()
                or current_api_url
                or "https://api.example.com/v1/"
            )
            # Prompt for custom profile name
            profile_name = input("  Profile name [default]: ").strip() or "default"
            return (
                "custom",
                "Custom",
                api_url.rstrip("/") + "/",
                current_model or "gpt-4o",
                profile_name,
            )
        label, api_url, default_model = PROVIDER_PRESETS[selected]
        return selected, label, api_url, default_model, ""


def run_in_app_config_wizard(
    workspace: str,
    console: Console,
    agent: ClineAgent,
    providers: Dict[str, dict],
    scope_arg: str = "",
) -> str:
    """Interactive config editor inside the app.

    scope_arg:
      - "" / "active" -> saves ~/.z.json and updates current agent config
      - any other name -> saves a provider profile in ~/.z.json
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
        providers.get(scope, {})
        if not is_active
        else {
            "api_url": agent.config.api_url,
            "api_key": agent.config.api_key,
            "model": agent.config.model,
            "max_tokens": agent.config.max_tokens,
            "temperature": agent.config.temperature,
        }
    )
    current_url = target_existing.get("api_url", "")
    current_model = target_existing.get("model", "")

    console.print(
        f"\n  [bold]{'Configure active provider' if is_active else f'Provider profile: {scope}'}[/bold] [dim](Enter to keep current values)[/dim]"
    )
    preset_key, label, api_url, preset_model, custom_profile_name = (
        _choose_provider_preset_interactive(current_url, current_model)
    )

    # Auto-suggest a profile name for new profiles based on the preset chosen
    if is_new_profile and scope == "default":
        if preset_key == "custom" and custom_profile_name:
            # Use the custom profile name provided by user
            scope = custom_profile_name
        elif preset_key != "custom":
            suggested_name = preset_key
            entered = input(f"  Profile name [{suggested_name}]: ").strip()
            scope = entered or suggested_name
        # Validate: warn if the profile name conflicts with a different provider
        if scope in providers:
            existing_url = providers[scope].get("api_url", "")
            if existing_url and existing_url != api_url:
                overwrite = (
                    input(
                        f"  Profile '{scope}' already exists ({_detect_provider_label(existing_url)}). Overwrite- [y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if overwrite not in ("y", "yes"):
                    return "Cancelled."

    api_key_current = target_existing.get("api_key", "")
    model_current = target_existing.get("model", "") or preset_model
    max_tokens_current = int(
        target_existing.get("max_tokens", getattr(agent.config, "max_tokens", 128000))
        or 128000
    )
    temp_current = float(
        target_existing.get("temperature", getattr(agent.config, "temperature", 0.7))
        or 0.7
    )

    # Check if OAuth provider
    is_oauth = "(OAuth)" in label

    if is_oauth:
        # OAuth flow
        console.print("  [dim]This provider uses OAuth authentication.[/dim]")

        # Import OAuth manager
        try:
            from harness.oauth import get_oauth_manager

            oauth_manager = get_oauth_manager()

            # Map provider name to OAuth provider ID
            if "OpenAI" in label:
                oauth_provider_id = "openai"
            else:
                oauth_provider_id = "github-copilot"

            # For OpenAI, let user choose method
            oauth_method = "browser"
            enterprise_url = None
            if "OpenAI" in label:
                console.print("\n  Select OAuth method:")
                console.print("  [1] Browser-based (opens browser for authorization)")
                console.print("  [2] Device code (headless, enter code manually)")
                method_choice = input("\n  Enter choice [1/2]: ").strip()
                oauth_method = "device" if method_choice == "2" else "browser"
            elif "GitHub Copilot" in label:
                # GitHub Copilot only supports device code flow
                console.print("\n  GitHub Copilot uses device code authentication.")

                # Ask about GitHub Enterprise
                is_enterprise = (
                    input("  Is this GitHub Enterprise? [y/N]: ").strip().lower()
                )
                if is_enterprise in ("y", "yes"):
                    enterprise_url = input(
                        "  Enter GitHub Enterprise domain (e.g., company.ghe.com): "
                    ).strip()

            console.print("\n  Opening browser for authentication...\n")

            # Trigger OAuth flow with selected method
            token = oauth_manager.authenticate(
                oauth_provider_id,
                method=oauth_method,
                timeout=300,
                enterprise_url=enterprise_url,
            )
            if token:
                api_key = f"oauth:{token.access_token}"
                console.print(f"  [green]✓[/green] OAuth authentication successful!\n")
            else:
                return "Cancelled: OAuth authentication failed."
        except Exception as e:
            return f"Cancelled: OAuth error: {e}"
    else:
        # API Key flow
        api_key = (
            input(
                f"  API key [{'***' + api_key_current[-4:] if len(api_key_current) > 4 else ('set' if api_key_current else 'not set')}]: "
            ).strip()
            or api_key_current
        )
        if not api_key:
            return "Cancelled: API key is required."

    model = model_current
    family = _provider_family_for_url(api_url)

    # Skip model fetching for OAuth providers (OAuth tokens are for ChatGPT web, not standard API)
    if is_oauth:
        from harness.codex_models import get_codex_models

        if "GitHub Copilot" in label:
            from harness.copilot_oauth_client import get_copilot_models

            console.print(
                "  [dim]Note: GitHub Copilot OAuth tokens access Copilot models directly.[/dim]"
            )

            # Show available Copilot models
            copilot_models = get_copilot_models()
            console.print(f"\n  [bold]Available Copilot models:[/bold]")
            for i, m in enumerate(copilot_models, 1):
                marker = (
                    "[cyan]●[/cyan]" if m == (model_current or preset_model) else " "
                )
                console.print(f"    {marker} [{i}] {m}")

            if not model:
                model_choice = input(
                    f"\n  Select model [1-{len(copilot_models)}] or enter name: "
                ).strip()
                if model_choice.isdigit() and 1 <= int(model_choice) <= len(
                    copilot_models
                ):
                    model = copilot_models[int(model_choice) - 1]
                else:
                    model = model_choice or model_current or preset_model
        else:
            console.print(
                "  [dim]Note: OAuth tokens access ChatGPT Codex models directly.[/dim]"
            )

            # Show available Codex models
            codex_models = get_codex_models()
            console.print(f"\n  [bold]Available Codex models:[/bold]")
            for i, m in enumerate(codex_models, 1):
                marker = (
                    "[cyan]●[/cyan]" if m == (model_current or preset_model) else " "
                )
                console.print(f"    {marker} [{i}] {m}")

            if not model:
                model_choice = input(
                    f"\n  Select model [1-{len(codex_models)}] or enter name: "
                ).strip()
                if model_choice.isdigit() and 1 <= int(model_choice) <= len(
                    codex_models
                ):
                    model = codex_models[int(model_choice) - 1]
                else:
                    model = model_choice or model_current or preset_model
    elif family in ("anthropic", "openai", "openrouter", "openai_compat"):
        fetch_now = input(f"  Fetch available models- [Y/n]: ").strip().lower()
        if fetch_now in ("", "y", "yes"):
            try:
                model_ids = _fetch_provider_model_ids(api_url, api_key)
                if model_ids:
                    model = _interactive_model_picker(model_current, model_ids)
                else:
                    console.print(
                        "  [dim]No models returned â€” using manual entry[/dim]"
                    )
            except Exception as e:
                console.print(
                    f"  [yellow]âš Model fetch failed: {rich_escape(str(e))}[/yellow]"
                )
            except Exception as e:
                console.print(
                    f"  [yellow]\u26a0 Model fetch failed: {rich_escape(str(e))}[/yellow]"
                )
    if not model:
        model = (
            input(f"  Model [{model_current or preset_model}]: ").strip()
            or model_current
            or preset_model
        )
    else:
        manual_override = input(
            f"  Model [{model}] (Enter to keep, or type to change): "
        ).strip()
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
        _save_active_config_fields(workspace, config_data)

        agent.config.api_url = api_url
        agent.config.api_key = api_key
        agent.config.model = model
        agent.config.max_tokens = max_tokens
        agent.config.temperature = temperature
        agent.tool_handlers.config = agent.config
        return f"\u2713 Active config saved - {detected} / {model}"

    models_path = _get_global_config_path()
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
    return f"\u2713 Saved profile [bold]{scope}[/bold] - {detected} / {model}"


def _infer_active_provider_profile(
    agent: ClineAgent, providers: Dict[str, dict]
) -> Optional[str]:
    for name, p in providers.items():
        if (
            p.get("api_url") == agent.config.api_url
            and p.get("api_key") == agent.config.api_key
            and p.get("model") == agent.config.model
        ):
            return name
    for name, p in providers.items():
        if (
            p.get("api_url") == agent.config.api_url
            and p.get("api_key") == agent.config.api_key
        ):
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
    sub = parts[0].lower() if parts else "list"

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
        _save_active_config_fields(
            workspace,
            {
                "api_url": agent.config.api_url,
                "api_key": agent.config.api_key,
                "model": agent.config.model,
                "max_tokens": agent.config.max_tokens,
                "temperature": agent.config.temperature,
            },
        )
        detected = _detect_provider_label(agent.config.api_url)
        _record_model_history(agent.config.model, profile)
        return f"\u2713 Switched to [bold]{profile}[/bold] - {detected} / {agent.config.model}"

    if sub in ("remove", "rm", "delete"):
        if len(parts) < 2:
            return "Usage: /providers remove <profile_name>"
        profile = parts[1]
        if profile not in providers:
            return f"Provider profile '{profile}' not found."
        models_path = _get_global_config_path()
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

    # Fuzzy match: treat the entire cmd_arg as a provider name query
    fuzzy_q = cmd_arg.strip().lower()
    if fuzzy_q and providers:
        names = sorted(providers.keys())
        # Exact substring match first
        fuzzy_matches = [n for n in names if fuzzy_q in n.lower()]
        if not fuzzy_matches:
            # Try each word matching any part of the name
            fuzzy_words = fuzzy_q.split()
            fuzzy_matches = [
                n for n in names if all(w in n.lower() for w in fuzzy_words)
            ]
        if len(fuzzy_matches) == 1:
            return run_provider_manager(
                workspace, console, agent, providers, f"use {fuzzy_matches[0]}"
            )
        if len(fuzzy_matches) > 1:
            match_list = ", ".join(fuzzy_matches)
            return f"Multiple providers match '{cmd_arg.strip()}': {match_list}. Be more specific."

    return "Usage: /providers [list|current|setup <name>|use <name|#>|remove <name>]"


def _load_global_config_json() -> dict:
    cfg_path = _get_global_config_path()
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _save_global_config_json(data: dict) -> Path:
    cfg_path = _get_global_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cfg_path


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

    data = _load_global_config_json()
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
                mcp_type = token[len("--type=") :].lower().strip()
                i += 1
                continue
            if token == "--url":
                if i + 1 >= len(parts):
                    return "Usage error: --url requires a value."
                url = parts[i + 1]
                i += 2
                continue
            if token.startswith("--url="):
                url = token[len("--url=") :]
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
                kv = token[len("--env=") :]
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
                kv = token[len("--header=") :]
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
        path = _save_global_config_json(data)
        return f"Saved MCP server '{name}' to {path}"

    if sub in ("remove", "rm", "delete"):
        if len(parts) < 2:
            return "Usage: /mcp remove <name>"
        name = parts[1]
        if name not in mcp:
            return f"MCP server '{name}' not found."
        mcp.pop(name, None)
        data["mcp"] = mcp
        path = _save_global_config_json(data)
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
        path = _save_global_config_json(data)
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
            cmd = cfg.get("command", [])
            if not isinstance(cmd, list) or not cmd:
                return f"MCP server '{name}' has invalid command config."
            env_cfg = dict(cfg.get("environment", {}) or {})
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in env_cfg.items()})
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd(),
                    env=env,
                )
            except FileNotFoundError:
                return f"Failed to start '{name}': command not found ({cmd[0]})."
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
        env = dict(cfg.get("environment", {}) or {})
        env[key] = value
        cfg["environment"] = env
        mcp[name] = cfg
        data["mcp"] = mcp
        _save_global_config_json(data)
        return f"Set env var '{key}' on MCP server '{name}'."

    if sub == "unsetenv":
        if len(parts) < 3:
            return "Usage: /mcp unsetenv <name> <KEY>"
        name, key = parts[1], parts[2]
        cfg = dict(mcp.get(name, {}) or {})
        if not cfg:
            return f"MCP server '{name}' not found."
        env = dict(cfg.get("environment", {}) or {})
        env.pop(key, None)
        cfg["environment"] = env
        mcp[name] = cfg
        data["mcp"] = mcp
        _save_global_config_json(data)
        return f"Removed env var '{key}' from MCP server '{name}'."

    return (
        "Usage: /mcp [list|show <name>|add <name> <command...> [--env KEY=VALUE] [--disabled]|"
        "remove <name>|enable <name>|disable <name>|test <name>|setenv <name> <KEY> <VALUE>|unsetenv <name> <KEY>]"
    )


def _render_providers_table(
    console: Console,
    providers: Dict[str, dict],
    active_name: Optional[str],
    show_numbers: bool = True,
) -> None:
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
    console.print(
        Panel(tbl, title="[bold]Providers[/bold]", border_style="dim", padding=(1, 2))
    )


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
                return run_provider_manager(
                    workspace, console, agent, providers, f"use {names[idx - 1]}"
                )
        return run_provider_manager(workspace, console, agent, providers, cmd_arg)

    active_name = _infer_active_provider_profile(agent, providers)
    _render_providers_table(console, providers, active_name, show_numbers=True)

    console.print()
    console.print(
        "  [dim][white]/providers setup[/white]         Add or edit a provider[/dim]"
    )
    console.print(
        "  [dim][white]/providers use <name|#>[/white]  Switch to a provider[/dim]"
    )
    console.print(
        "  [dim][white]/providers remove <name>[/white] Remove a provider[/dim]"
    )
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
        return (
            "No provider profiles yet. Use /provider setup <name> once, then use F2/F3."
        )
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
    """Tab completer for commands, file paths, and shell commands."""

    COMMANDS = [
        "/sessions",
        "/session",
        "/new",
        "/delete",
        "/clear",
        "/save",
        "/history",
        "/bg",
        "/ctx",
        "/tokens",
        "/compact",
        "/cost",
        "/maxctx",
        "/todo",
        "/smart",
        "/dump",
        "/policyeval",
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
        "/undo",
        "/redo",
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
            elif parts[0] in ["/session", "/delete"]:
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
                # Complete compact strategies
                strategies = ["half", "quarter", "last2"]
                prefix = parts[-1]
                for strat in strategies:
                    if strat.startswith(prefix):
                        yield Completion(
                            strat,
                            start_position=-len(prefix),
                            display=strat,
                        )
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


def create_prompt_session(
    history_file: Path,
    workspace: Path,
    on_paste_image_marker: Optional[Callable[[], Optional[str]]] = None,
    on_open_provider_picker: Optional[Callable[[], None]] = None,
    on_open_model_picker: Optional[Callable[[], None]] = None,
    on_undo: Optional[Callable[[], None]] = None,
    on_redo: Optional[Callable[[], None]] = None,
    on_toggle_reasoning: Optional[Callable[[], str]] = None,
) -> "PromptSession":
    """Create a prompt session with multiline support.

    Keybindings:
    - Enter: Submit input
    - Ctrl+Enter: Insert newline (for multiline input)
    - Paste: Multiline paste works automatically
    - Tab: Complete commands and file paths
    - Ctrl+Z: Undo last turn (when prompt is empty)
    - Ctrl+Y: Redo (when prompt is empty)
    - Ctrl+T: Toggle reasoning effort
    """
    if not HAS_PROMPT_TOOLKIT:
        return None

    bindings = KeyBindings()

    # Ctrl+Enter inserts newline (Escape+Enter as fallback for terminals that don't support Ctrl+Enter)
    @bindings.add(Keys.Escape, Keys.Enter)
    def _(event):
        """Escape+Enter: insert newline (fallback)."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")  # Ctrl+J = Ctrl+Enter in most terminals
    def _(event):
        """Ctrl+Enter: insert newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-v")
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
        event.current_buffer.insert_text("\x16")

    @bindings.add("f2")
    def _(event):
        """F2: open provider picker UI."""
        if on_open_provider_picker:
            if pt_run_in_terminal:
                pt_run_in_terminal(on_open_provider_picker)
            else:
                on_open_provider_picker()

    @bindings.add("f3")
    def _(event):
        """F3: open model picker UI."""
        if on_open_model_picker:
            if pt_run_in_terminal:
                pt_run_in_terminal(on_open_model_picker)
            else:
                on_open_model_picker()

    @bindings.add("c-z")
    def _(event):
        """Ctrl+Z: undo last turn (when prompt is empty)."""
        if event.current_buffer.text.strip() == "" and on_undo:
            # Submit /undo as the command
            event.current_buffer.text = "/undo"
            event.current_buffer.validate_and_handle()
        else:
            # Let normal undo work when there's text being edited
            event.current_buffer.undo()

    @bindings.add("c-y")
    def _(event):
        """Ctrl+Y: redo last turn (when prompt is empty)."""
        if event.current_buffer.text.strip() == "" and on_redo:
            event.current_buffer.text = "/redo"
            event.current_buffer.validate_and_handle()

    @bindings.add("c-t", eager=True)
    def _(event):
        """Ctrl+T: cycle reasoning effort."""
        if on_toggle_reasoning:
            new_level = on_toggle_reasoning()
            if pt_run_in_terminal:
                def _show():
                    print(f"  reasoning effort → {new_level}")
                pt_run_in_terminal(_show)
            event.app.invalidate()

    # Create session with history
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Create history instance first
    history = SafeFileHistory(str(history_file))

    # Create completer for commands and file paths
    completer = HarnessCompleter(workspace)

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
    tbl.add_row(
        "Input",
        f"{summary.total_input_tokens:,} tokens  [dim]${summary.total_input_cost:.4f}[/dim]",
    )
    tbl.add_row(
        "Output",
        f"{summary.total_output_tokens:,} tokens  [dim]${summary.total_output_cost:.4f}[/dim]",
    )
    tbl.add_row(
        "Total",
        f"[bold]{summary.total_tokens:,} tokens  ${summary.total_cost:.4f}[/bold]",
    )

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
    console.print(
        Panel(
            tbl,
            title="[bold]Session Costs[/bold]",
            border_style="dim",
            padding=(1, 2),
            width=52,
        )
    )

    by_model = tracker.get_cost_by_model()
    if by_model and len(by_model) > 1:
        mtbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
        mtbl.add_column("Model", style="cyan")
        mtbl.add_column("Calls", justify="right")
        mtbl.add_column("Tokens", justify="right")
        mtbl.add_column("Cost", justify="right", style="bold")
        for model, row in sorted(
            by_model.items(), key=lambda kv: kv[1]["total_cost"], reverse=True
        ):
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
    log.debug(
        "run_single START input=%s",
        truncate(
            user_label
            or (user_input if isinstance(user_input, str) else "[multimodal]"),
            120,
        ),
    )
    try:
        if isinstance(user_input, str):
            result = await agent.run(user_input)
        else:
            result = await agent.run_message(
                user_input, user_label=user_label or "[multimodal]"
            )
    except asyncio.CancelledError:
        log.warning("run_single cancelled after %.1fs", time.time() - start_time)
        console.print("\n  [yellow]Cancelled[/yellow]")
        result = "[Interrupted]"
    except KeyboardInterrupt:
        log.warning(
            "run_single KeyboardInterrupt after %.1fs", time.time() - start_time
        )
        console.print("\n  [yellow]Interrupted[/yellow]")
        result = "[Interrupted]"

    elapsed = time.time() - start_time
    log.info("run_single DONE elapsed=%.1fs result_len=%d", elapsed, len(result or ""))

    cost = get_global_tracker().get_summary()
    stats = agent.get_context_stats()

    if elapsed < 60:
        elapsed_str = f"{elapsed:.1f}s"
    else:
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        elapsed_str = f"{mins}m{secs:02d}s"

    ctx_k = stats["tokens"] // 1000
    max_k = stats["max_allowed"] // 1000
    pct = stats["percent"]

    bar_width = 10
    filled = int(bar_width * pct / 100)
    bar_color = "green" if pct < 60 else "yellow" if pct < 85 else "red"

    console.print()
    status = Text("  ")
    status.append(agent.config.model, style="dim")
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
    parser.add_argument(
        "--session", "-s", default="default", help="Session name (default: 'default')"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List all sessions")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run setup wizard (interactive or headless)",
    )
    parser.add_argument("--api-url", help="API base URL (headless install)")
    parser.add_argument("--api-key", help="API key (headless install)")
    parser.add_argument("--model", help="Model name (headless install)")
    parser.add_argument(
        "--workspace-config",
        action="store_true",
        help="Save config to workspace instead of global",
    )
    parser.add_argument(
        "--policy-eval",
        action="append",
        help="Run context policy replay on dump JSON path (repeatable)",
    )
    parser.add_argument(
        "--policy-eval-out",
        default="",
        help="Write policy replay JSON report to this path",
    )
    parser.add_argument(
        "--policy-no-train",
        action="store_true",
        help="Policy replay: skip classifier training",
    )
    parser.add_argument(
        "--policy-embed-backend",
        default="auto",
        help="Policy replay embedding backend: auto | semantic_scorer | hash | hf:<model-id>",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Run frozen safe mode harness (for self-repair when main harness breaks)",
    )
    parser.add_argument(
        "--freeze-safe",
        action="store_true",
        help="Freeze current harness as the new safe mode version",
    )
    args = parser.parse_args()

    # Safe mode handling
    SAFE_MODE_DIR = Path.home() / ".z" / "safe"
    
    if args.freeze_safe:
        # Freeze current harness as safe mode
        import shutil
        con = Console()
        con.print("\n  [bold]Freezing current harness as safe mode...[/bold]\n")
        
        # Get the source directory (where this script lives)
        src_dir = Path(__file__).parent.absolute()
        
        # Remove old safe mode if exists
        if SAFE_MODE_DIR.exists():
            shutil.rmtree(SAFE_MODE_DIR)
        
        # Create safe mode directory
        SAFE_MODE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Copy harness.py
        shutil.copy2(src_dir / "harness.py", SAFE_MODE_DIR / "harness.py")
        con.print(f"  [green]✓[/green] Copied harness.py")
        
        # Copy src/harness/ directory
        src_harness = src_dir / "src" / "harness"
        dst_harness = SAFE_MODE_DIR / "src" / "harness"
        if src_harness.exists():
            shutil.copytree(src_harness, dst_harness)
            con.print(f"  [green]✓[/green] Copied src/harness/ ({len(list(dst_harness.rglob('*.py')))} files)")
        
        con.print(f"\n  [bold green]Safe mode frozen to:[/bold green] {SAFE_MODE_DIR}")
        con.print(f"  [dim]Run with:[/dim] z --safe-mode")
        con.print()
        return
    
    if args.safe_mode:
        # Run the frozen safe mode harness
        safe_harness = SAFE_MODE_DIR / "harness.py"
        
        if not safe_harness.exists():
            con = Console()
            con.print("\n  [red]✗[/red] Safe mode not found!")
            con.print(f"  [dim]Expected at:[/dim] {safe_harness}")
            con.print(f"\n  [yellow]Run first:[/yellow] z --freeze-safe")
            con.print()
            sys.exit(1)
        
        # Re-exec with the frozen harness, passing through all other args
        # Remove --safe-mode from args to avoid infinite loop
        other_args = [a for a in sys.argv[1:] if a != "--safe-mode"]
        
        # Add safe mode's src to path
        safe_src = SAFE_MODE_DIR / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(safe_src) + os.pathsep + env.get("PYTHONPATH", "")
        # Mark that we're in safe mode (so the frozen harness knows)
        env["HARNESS_SAFE_MODE"] = "1"
        
        print(f"\n  [SAFE MODE] Running frozen harness from {SAFE_MODE_DIR}\n")
        
        # Use subprocess.run with inherited stdin/stdout/stderr for proper terminal handling
        result = subprocess.run(
            [sys.executable, str(safe_harness)] + other_args,
            env=env,
            cwd=os.getcwd(),
        )
        sys.exit(result.returncode)

    # Install mode - run setup wizard
    if args.install or args.api_url or args.api_key:
        run_install(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            global_config=not args.workspace_config,
        )
        return

    if args.policy_eval:
        from harness.context_replay import run_replay

        con = Console()
        dump_paths = [Path(p).expanduser().resolve() for p in args.policy_eval]
        result = run_replay(
            dump_paths,
            train=not args.policy_no_train,
            embedding_backend=args.policy_embed_backend,
        )
        text = json.dumps(result, indent=2)
        if args.policy_eval_out:
            out_path = Path(args.policy_eval_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            con.print(
                f"  [green]\u2713[/green] Policy replay report: [cyan]{rich_escape(str(out_path))}[/cyan]"
            )
        else:
            con.print(text)
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
    _mark("pre_init_logging")
    init_logging(workspace=workspace, session_id=args.session)
    if args.debug:
        enable_debug()
    _mark("init_logging")
    log.info(
        "=== Harness starting === workspace=%s session=%s new=%s",
        workspace,
        args.session,
        args.new,
    )

    # Load providers from global config (~/.z.json)
    providers = load_providers(workspace)
    _mark("load_providers")

    if not providers:
        log.warning(
            "No providers found in ~/.z.json — falling back to active global config"
        )

    # Determine starting config from the active global config file (~/.z.json).
    # Provider profiles are available via /provider use and /provider setup.
    config = Config.from_json(workspace=Path(workspace))
    if (not config.api_url or not config.api_key) and providers:
        first_name = sorted(providers.keys())[0]
        p = providers[first_name]
        config = Config.from_json(
            workspace=Path(workspace),
            overrides={
                "api_url": p.get("api_url", ""),
                "api_key": p.get("api_key", ""),
                "model": p.get("model", config.model),
                "max_tokens": p.get("max_tokens", config.max_tokens),
                "temperature": p.get("temperature", config.temperature),
            },
        )
        log.info(
            "No active config found; bootstrapping from provider profile '%s'",
            first_name,
        )
    config.validate()
    if not providers and config.api_url and config.api_key:
        providers = {
            "active": {
                "api_url": config.api_url,
                "api_key": config.api_key,
                "model": config.model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }
        }
        log.info("Bootstrapped in-memory provider profile 'active' from current config")
    _mark("config_loaded")
    log.info(
        "Config loaded: api_url=%s model=%s max_tokens=%d providers=%s",
        config.api_url,
        config.model,
        config.max_tokens,
        list(providers.keys()),
    )

    # Create a single persistent event loop for the entire session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    console = Console()

    # Create single agent with providers for mode switching
    agent = ClineAgent(
        config,
        providers=providers,
    )
    _mark("agent_created")
    log.info("Agent created")

    # Checkpoint manager for undo/redo
    checkpoint_mgr = CheckpointManager(workspace)
    checkpoint_mgr._workspace_index = agent.workspace_index
    agent.checkpoint_mgr = checkpoint_mgr
    if not checkpoint_mgr._check_git():
        console.print("  [dim]git not found — /undo and /redo disabled[/dim]")
    _mark("checkpoint_mgr")

    # Session management
    current_session = args.session
    session_path = get_session_path(workspace, current_session)

    # Try to resume session unless --new
    if not args.new and session_path.exists():
        if agent.load_session(str(session_path)):
            msg_count = len(agent.messages) - 1  # minus system prompt
            log.info(
                "Resumed session '%s' (%d messages) from %s",
                current_session,
                msg_count,
                session_path,
            )
            console.print(
                f"  [dim]Resumed [white]{current_session}[/white] ({msg_count} messages)[/dim]"
            )
    else:
        log.info("New session '%s'", current_session)
        console.print(f"  [dim]New session [white]{current_session}[/white][/dim]")
    _mark("session_loaded")

    def save_session():
        log.debug("Saving session")
        agent.save_session(str(session_path))
        log.debug("Session saved")

    def cleanup_and_save():
        loop.run_until_complete(agent.cleanup_background_procs_async())
        save_session()

    _mark("ready")
    if os.environ.get("HARNESS_BOOT_TIMING", ""):
        _print_boot_timing(console)

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
        # Interactive mode â€” clean startup banner
        stats = agent.get_context_stats()
        ws_short = os.path.basename(workspace) or workspace
        banner_table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
        banner_table.add_column(style="dim", width=11, justify="right")
        banner_table.add_column()
        _banner_prov = _infer_active_provider_profile(agent, providers) or ""
        _banner_model = f"[bold]{config.model}[/bold]"
        if _banner_prov:
            _banner_model += f"  [dim]{_banner_prov}[/dim]"
        banner_table.add_row("model", _banner_model)
        banner_table.add_row(
            "context", f"[bold]{stats['max_allowed']:,}[/bold] [dim]tokens[/dim]"
        )
        banner_table.add_row("workspace", f"[cyan]{ws_short}[/cyan]")
        banner_table.add_row("session", current_session)
        _banner_effort = getattr(config, "reasoning_effort", "high")
        banner_table.add_row("reasoning", f"[bold]{_banner_effort}[/bold] [dim]Ctrl+T to toggle[/dim]")
        console.print()
        # Check if running in safe mode
        _is_safe_mode = os.environ.get("HARNESS_SAFE_MODE") == "1"
        _banner_title = "[bold yellow] SAFE MODE [/bold yellow]" if _is_safe_mode else "[bold bright_blue] harness [/bold bright_blue]"
        _banner_style = "yellow" if _is_safe_mode else "bright_blue"
        console.print(
            Panel(
                banner_table,
                title=_banner_title,
                border_style=_banner_style,
                padding=(1, 3),
                width=50,
            )
        )
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

        _REASONING_LEVELS = ["high", "medium", "low", "none"]

        def _toggle_reasoning_effort() -> str:
            """Cycle reasoning effort: high → medium → low → none → high."""
            current = getattr(agent.config, "reasoning_effort", "high")
            try:
                idx = _REASONING_LEVELS.index(current)
            except ValueError:
                idx = 0
            new_level = _REASONING_LEVELS[(idx + 1) % len(_REASONING_LEVELS)]
            agent.config.reasoning_effort = new_level
            return new_level

        history_file = get_sessions_dir(workspace) / ".history"
        prompt_session = (
            create_prompt_session(
                history_file,
                Path(workspace),
                on_paste_image_marker=_prompt_paste_image_marker,
                on_open_provider_picker=_open_provider_picker_ui,
                on_open_model_picker=_open_model_picker_ui,
                on_undo=lambda: True,  # Ctrl+Z submits "/undo" text
                on_redo=lambda: True,  # Ctrl+Y submits "/redo" text
                on_toggle_reasoning=_toggle_reasoning_effort,
            )
            if HAS_PROMPT_TOOLKIT
            else None
        )

        last_interrupt_time = 0  # Track time of last Ctrl+C for double-tap exit

        def _build_prompt_text():
            """Build prompt text dynamically so Ctrl+T updates are visible immediately."""
            model_short = agent.config.model.split("/")[-1] if "/" in agent.config.model else agent.config.model
            active_prov = _infer_active_provider_profile(agent, providers) or ""
            effort = getattr(agent.config, "reasoning_effort", "high")
            effort_colors = {"high": "208", "medium": "214", "low": "243", "none": "240"}
            effort_color = effort_colors.get(effort, "243")
            info_parts = []
            if active_prov:
                info_parts.append(f"\x1b[38;5;243m{active_prov}\x1b[0m")
            if effort != "none":
                info_parts.append(f"\x1b[38;5;{effort_color}m{effort}\x1b[0m")
            info_str = f" \x1b[38;5;243m·\x1b[0m ".join(info_parts)
            return ANSI(f"\x1b[1m{model_short}\x1b[0m {info_str} \x1b[38;5;243m\u276f\x1b[0m ")

        while True:
            try:
                # Get input (multiline with prompt_toolkit, or simple input)
                if prompt_session:
                    try:
                        user_input = prompt_session.prompt(_build_prompt_text).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print(
                            "\n  [dim]Press [white]Ctrl+C[/white] again to exit[/dim]"
                        )
                        continue
                else:
                    try:
                        _pt = _build_prompt_text()
                        # ANSI object -> raw string for plain input()
                        _raw = f"{agent.config.model} \u276f "
                        user_input = input(_raw).strip()
                    except KeyboardInterrupt:
                        now = time.time()
                        if now - last_interrupt_time < 2.0:
                            raise
                        last_interrupt_time = now
                        console.print(
                            "\n  [dim]Press [white]Ctrl+C[/white] again to exit[/dim]"
                        )
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
                                agent.tool_handlers.execute_command(
                                    {"command": shell_cmd}
                                )
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

                    if cmd in ("/exit", "/quit", "/q"):
                        cleanup_and_save()
                        console.print("  [dim]Session saved. Goodbye![/dim]")
                        break

                    elif cmd == "/sessions":
                        sessions = list_sessions(workspace)
                        if not sessions:
                            console.print("  [dim]No sessions yet.[/dim]")
                        else:
                            tbl = Table(
                                show_header=False,
                                box=None,
                                padding=(0, 2),
                                pad_edge=False,
                            )
                            tbl.add_column(width=2)
                            tbl.add_column("name", style="bold")
                            tbl.add_column("msgs", justify="right", style="dim")
                            tbl.add_column("modified", style="dim")
                            for name, mtime, count in sessions:
                                marker = (
                                    "[cyan]\u25cf[/cyan]"
                                    if name == current_session
                                    else " "
                                )
                                tbl.add_row(
                                    marker,
                                    name,
                                    f"{count} msgs",
                                    f"{mtime:%b %d, %H:%M}",
                                )
                            console.print()
                            console.print(
                                Panel(
                                    tbl,
                                    title="[bold]Sessions[/bold]",
                                    border_style="dim",
                                    padding=(1, 2),
                                )
                            )
                            console.print()
                        continue

                    elif cmd == "/session":
                        if not cmd_arg:
                            console.print(
                                f"  [dim]Current session:[/dim] [bold]{current_session}[/bold]"
                            )
                            continue
                        agent.save_session(str(session_path))
                        new_name = cmd_arg.strip()
                        current_session = new_name
                        session_path = get_session_path(workspace, current_session)
                        if session_path.exists():
                            agent.load_session(str(session_path))
                            msg_count = len(agent.messages) - 1
                            console.print(
                                f"  [green]\u2713[/green] Switched to [bold]{current_session}[/bold] [dim]({msg_count} messages)[/dim]"
                            )
                        else:
                            agent.clear_history()
                            console.print(
                                f"  [green]\u2713[/green] Created new session [bold]{current_session}[/bold]"
                            )
                        continue

                    elif cmd == "/new":
                        agent.save_session(str(session_path))
                        new_name = (
                            cmd_arg.strip()
                            if cmd_arg.strip()
                            else datetime.now().strftime("%Y%m%d-%H%M%S")
                        )
                        current_session = new_name
                        session_path = get_session_path(workspace, current_session)
                        agent.clear_history()
                        reset_global_tracker()
                        console.print(
                            f"  [green]\u2713[/green] Started fresh session [bold]{current_session}[/bold]"
                        )
                        continue

                    elif cmd == "/clear":
                        debug_print("/clear: calling clear_history...")
                        agent.clear_history()
                        debug_print(
                            "/clear: clear_history returned, resetting tracker..."
                        )
                        reset_global_tracker()
                        debug_print("/clear: resetting done, printing...")
                        console.print("  [green]\u2713[/green] History cleared")
                        debug_print("/clear: about to continue...")
                        continue

                    elif cmd == "/save":
                        agent.save_session(str(session_path))
                        console.print(
                            f"  [green]\u2713[/green] Session [bold]{current_session}[/bold] saved"
                        )
                        continue

                    elif cmd == "/delete":
                        if not cmd_arg:
                            console.print("  [dim]Usage: /delete <session_name>[/dim]")
                            continue
                        target = cmd_arg.strip()
                        if target == current_session:
                            console.print(
                                "  [yellow]\u26a0[/yellow] Cannot delete the active session"
                            )
                            continue
                        target_path = get_session_path(workspace, target)
                        if target_path.exists():
                            target_path.unlink()
                            console.print(
                                f"  [green]\u2713[/green] Deleted session [bold]{target}[/bold]"
                            )
                        else:
                            console.print(f"  [dim]Session '{target}' not found[/dim]")
                        continue

                    elif cmd == "/history":
                        console.print(
                            f"  [dim]{len(agent.messages)} messages in conversation[/dim]"
                        )
                        continue

                    elif cmd == "/cost":
                        _render_cost_report(console)
                        continue

                    elif cmd == "/bg":
                        procs = agent.list_background_procs()
                        if not procs:
                            console.print(
                                "  [dim]No background processes running[/dim]"
                            )
                        else:
                            tbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1))
                            tbl.add_column("ID", style="bold")
                            tbl.add_column("PID", style="dim")
                            tbl.add_column("Status")
                            tbl.add_column("Time", justify="right")
                            tbl.add_column("Command", style="dim")
                            for p in procs:
                                elapsed_min = p["elapsed"] / 60
                                status_style = (
                                    "green" if p["status"] == "running" else "yellow"
                                )
                                tbl.add_row(
                                    str(p["id"]),
                                    str(p["pid"]),
                                    f"[{status_style}]{p['status']}[/{status_style}]",
                                    f"{elapsed_min:.1f}m",
                                    p["command"][:50],
                                )
                            console.print()
                            console.print(
                                Panel(
                                    tbl,
                                    title="[bold]Background Processes[/bold]",
                                    border_style="dim",
                                    padding=(0, 1),
                                )
                            )
                            console.print()
                        continue

                    elif cmd == "/mode":
                        console.print(
                            "  [dim]/mode is deprecated. Use [white]/providers[/white] and [white]/model[/white] instead.[/dim]"
                        )
                        continue

                    elif cmd == "/ctx":
                        stats = agent.get_context_stats()
                        pct = stats["percent"]
                        bar_width = 24
                        filled = int(bar_width * pct / 100)
                        bar_color = (
                            "green" if pct < 60 else "yellow" if pct < 85 else "red"
                        )
                        bar = Text()
                        bar.append("\u2501" * filled, style=bar_color)
                        bar.append("\u2500" * (bar_width - filled), style="dim")
                        console.print()
                        console.print("  [bold]Context Usage[/bold]")
                        ctx_line = Text("  ")
                        ctx_line.append_text(bar)
                        ctx_line.append(
                            f"  {stats['tokens']:,} / {stats['max_allowed']:,} tokens ({pct:.0f}%)",
                            style="dim",
                        )
                        console.print(ctx_line)
                        console.print(
                            f"  [dim]{stats['messages']} messages \u00b7 {stats['context_items']} context items[/dim]"
                        )
                        console.print()
                        continue

                    elif cmd == "/tokens":
                        breakdown = agent.get_token_breakdown()
                        tbl = Table(
                            show_header=False, box=None, padding=(0, 2), pad_edge=False
                        )
                        tbl.add_column("label", style="dim", width=14, justify="right")
                        tbl.add_column("value")
                        tbl.add_row("System", f"{breakdown['system']:,} tokens")
                        tbl.add_row(
                            "Conversation",
                            f"{breakdown['conversation']:,} tokens [dim]({breakdown['message_count']} msgs)[/dim]",
                        )
                        tbl.add_row(
                            "Total", f"[bold]{breakdown['total']:,} tokens[/bold]"
                        )
                        console.print()
                        console.print(
                            Panel(
                                tbl,
                                title="[bold]Token Breakdown[/bold]",
                                border_style="dim",
                                padding=(1, 2),
                            )
                        )
                        if breakdown["largest_messages"]:
                            console.print("  [dim]Largest messages:[/dim]")
                            for msg in breakdown["largest_messages"]:
                                role = msg["role"]
                                tokens = msg["tokens"]
                                preview = (
                                    msg["preview"][:50] + "..."
                                    if len(msg["preview"]) > 50
                                    else msg["preview"]
                                )
                                console.print(
                                    f"    [dim]{role:>10}  {tokens:>6,}t  {rich_escape(preview)}[/dim]"
                                )
                        console.print()
                        continue

                    elif cmd == "/compact":
                        strategy = cmd_arg.strip() if cmd_arg else "half"
                        before = agent.get_token_count()
                        removed = agent.compact_history(strategy)
                        after = agent.get_token_count()
                        console.print(
                            f"  [green]\u2713[/green] Compacted [bold]{before:,}[/bold] \u2192 [bold]{after:,}[/bold] tokens [dim](-{removed:,})[/dim]"
                        )
                        continue

                    elif cmd == "/todo":
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
                                item = agent.todo_manager.update(
                                    item_id, status="completed"
                                )
                                if not item:
                                    console.print(
                                        f"  [dim]Todo [{item_id}] not found[/dim]"
                                    )
                                else:
                                    console.print(
                                        f"  [green]\u2713[/green] Completed: {item.title}"
                                    )
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("  [dim]Usage: /todo done <id>[/dim]")
                        elif cmd_arg.startswith("rm "):
                            try:
                                item_id = int(cmd_arg[3:].strip())
                                if not agent.todo_manager.remove(item_id):
                                    console.print(
                                        f"  [dim]Todo [{item_id}] not found[/dim]"
                                    )
                                else:
                                    console.print(f"  [green]\u2713[/green] Removed")
                                agent.todo_manager.print_todo_panel(console)
                            except ValueError:
                                console.print("  [dim]Usage: /todo rm <id>[/dim]")
                        elif cmd_arg.strip() == "clear":
                            agent.todo_manager.clear()
                            console.print("  [green]\u2713[/green] All todos cleared")
                        else:
                            console.print(
                                "  [dim]Usage: /todo [add <title> | done <id> | rm <id> | clear][/dim]"
                            )
                        continue

                    elif cmd == "/smart":
                        from harness.context_management import (
                            get_model_limits as _gml,
                            estimate_messages_tokens as _emt,
                        )

                        _, max_allowed = _gml(agent.config.model)
                        total_tokens = _emt(agent.messages)
                        pct = total_tokens / max_allowed * 100 if max_allowed else 0
                        over = total_tokens > int(max_allowed * 0.85)
                        active_todos = agent.todo_manager.list_active()
                        traces = agent.smart_context.compaction_traces

                        bar_width = 24
                        filled = int(bar_width * pct / 100)
                        bar_color = (
                            "green" if pct < 60 else "yellow" if pct < 85 else "red"
                        )
                        bar = Text()
                        bar.append("\u2501" * filled, style=bar_color)
                        bar.append("\u2500" * (bar_width - filled), style="dim")

                        tbl = Table(
                            show_header=False, box=None, padding=(0, 2), pad_edge=False
                        )
                        tbl.add_column("label", style="dim", width=14, justify="right")
                        tbl.add_column("value")
                        ctx_val = Text()
                        ctx_val.append_text(bar)
                        ctx_val.append(
                            f"  {total_tokens:,} / {max_allowed:,} ({pct:.0f}%)"
                        )
                        tbl.add_row("Context", ctx_val)
                        threshold_style = "red bold" if over else "green"
                        tbl.add_row(
                            "Compaction",
                            f"[{threshold_style}]{'OVER 85% threshold' if over else 'below threshold'}[/{threshold_style}]",
                        )
                        tbl.add_row("Messages", str(len(agent.messages)))
                        tbl.add_row("Active todos", str(len(active_todos)))
                        tbl.add_row(
                            "Context items",
                            f"{len(agent.context.list_items())}  [dim]({agent.context.total_size():,} chars)[/dim]",
                        )
                        console.print()
                        console.print(
                            Panel(
                                tbl,
                                title="[bold]Smart Context[/bold]",
                                border_style="dim",
                                padding=(1, 2),
                            )
                        )
                        if traces:
                            console.print(
                                f"  [dim]Compactions this session: {len(traces)}[/dim]"
                            )
                            for t in traces[-5:]:
                                console.print(
                                    f"    [dim]\u2192 {t.format_notice()}[/dim]"
                                )
                        else:
                            console.print("  [dim]No compactions yet.[/dim]")
                        console.print()
                        continue

                    elif cmd == "/dump":
                        dump_path = agent.dump_context(reason="user_requested")
                        breakdown = agent.get_token_breakdown()
                        console.print(
                            f"  [green]\u2713[/green] Context dumped to [cyan]{rich_escape(str(dump_path))}[/cyan]"
                        )
                        console.print(
                            f"  [dim]System: {breakdown['system']:,}t \u00b7 Conversation: {breakdown['conversation']:,}t ({breakdown['message_count']} msgs) \u00b7 Total: {breakdown['total']:,}t[/dim]"
                        )
                        if agent.messages and agent.messages[0].role == "system":
                            sys_content = agent.messages[0].content
                            expected_tools = [
                                "read_file",
                                "write_to_file",
                                "replace_in_file",
                                "execute_command",
                                "manage_todos",
                                "create_plan",
                            ]
                            tools_missing = [
                                t for t in expected_tools if t not in sys_content
                            ]
                            if tools_missing:
                                console.print(
                                    f"  [red]\u2717 System prompt missing tools: {', '.join(tools_missing)}[/red]"
                                )
                            else:
                                console.print(
                                    f"  [green]\u2713[/green] [dim]All {len(expected_tools)} core tools present in system prompt[/dim]"
                                )
                        else:
                            console.print(
                                "  [red]\u2717 No system message found![/red]"
                            )
                        continue

                    elif cmd == "/policyeval":
                        from harness.context_replay import run_replay

                        arg = cmd_arg.strip()
                        if not arg:
                            console.print(
                                "  [dim]Usage: /policyeval <dump.json> [--no-train] "
                                "[--out <report.json>] [--embed-backend <auto|hash|semantic_scorer|hf:model>][/dim]"
                            )
                            continue
                        try:
                            parts = shlex.split(arg)
                        except ValueError as e:
                            console.print(
                                f"  [red]\u2717 Invalid arguments:[/red] {rich_escape(str(e))}"
                            )
                            continue
                        no_train = "--no-train" in parts
                        out_path = ""
                        embed_backend = "auto"
                        if "--out" in parts:
                            oi = parts.index("--out")
                            if oi + 1 < len(parts):
                                out_path = parts[oi + 1]
                        if "--embed-backend" in parts:
                            bi = parts.index("--embed-backend")
                            if bi + 1 < len(parts):
                                embed_backend = parts[bi + 1]
                        skip_vals = {
                            "--no-train",
                            "--out",
                            out_path,
                            "--embed-backend",
                            embed_backend,
                        }
                        dump_tokens = [
                            p
                            for p in parts
                            if p not in skip_vals and not p.startswith("--")
                        ]
                        if not dump_tokens:
                            console.print("  [red]\u2717 Missing dump path.[/red]")
                            continue
                        dump_path = Path(dump_tokens[0]).expanduser()
                        if not dump_path.is_absolute():
                            dump_path = (Path(workspace) / dump_path).resolve()
                        if not dump_path.exists():
                            console.print(
                                f"  [red]\u2717 Dump not found:[/red] {rich_escape(str(dump_path))}"
                            )
                            continue
                        result = run_replay(
                            [dump_path],
                            train=not no_train,
                            embedding_backend=embed_backend,
                        )
                        report = json.dumps(result, indent=2)
                        if out_path:
                            op = Path(out_path).expanduser()
                            if not op.is_absolute():
                                op = (Path(workspace) / op).resolve()
                            op.parent.mkdir(parents=True, exist_ok=True)
                            op.write_text(report, encoding="utf-8")
                            console.print(
                                f"  [green]\u2713[/green] Policy replay report: [cyan]{rich_escape(str(op))}[/cyan]"
                            )
                        else:
                            console.print(report)
                        continue

                    elif cmd == "/config":
                        subparts = cmd_arg.split()
                        if subparts and subparts[0].lower() == "setup":
                            scope = (
                                subparts[1].lower() if len(subparts) > 1 else "active"
                            )
                            try:
                                result = run_in_app_config_wizard(
                                    workspace, console, agent, providers, scope
                                )
                                console.print(f"  [dim]{result}[/dim]")
                            except KeyboardInterrupt:
                                console.print("\n  [dim]Config wizard cancelled.[/dim]")
                            continue

                        key_preview = (
                            agent.config.api_key[:8] + "..."
                            if agent.config.api_key
                            else "(not set)"
                        )
                        tbl = Table(
                            show_header=False, box=None, padding=(0, 2), pad_edge=False
                        )
                        tbl.add_column("label", style="dim", width=13, justify="right")
                        tbl.add_column("value")
                        tbl.add_row(
                            "API URL",
                            f"[cyan]{rich_escape(agent.config.api_url)}[/cyan]",
                        )
                        tbl.add_row(
                            "Model", f"[bold]{rich_escape(agent.config.model)}[/bold]"
                        )
                        tbl.add_row("API Key", key_preview)
                        tbl.add_row("Max Tokens", f"{agent.config.max_tokens:,}")
                        threshold_pct = int(agent.config.compaction_threshold * 100)
                        threshold_tokens = int(
                            agent.config.max_tokens * agent.config.compaction_threshold
                        )
                        tbl.add_row(
                            "Compact At",
                            f"{threshold_pct}% (~{threshold_tokens:,} tokens)",
                        )
                        tbl.add_row("Temperature", str(agent.config.temperature))
                        tbl.add_row("Max Iters", str(agent.max_iterations))
                        if providers:
                            tbl.add_row(
                                "Providers", ", ".join(sorted(providers.keys()))
                            )
                        console.print()
                        console.print(
                            Panel(
                                tbl,
                                title="[bold]Configuration[/bold]",
                                border_style="dim",
                                padding=(1, 2),
                            )
                        )
                        console.print(
                            "  [dim][white]/providers[/white] to manage providers, [white]/model[/white] to switch models.[/dim]"
                        )
                        console.print()
                        continue

                    elif cmd in ("/provider", "/providers"):
                        try:
                            if cmd == "/providers":
                                result = run_providers_hub(
                                    workspace, console, agent, providers, cmd_arg
                                )
                            else:
                                result = run_provider_manager(
                                    workspace, console, agent, providers, cmd_arg
                                )
                            if result:
                                console.print(f"  [dim]{result}[/dim]")
                        except KeyboardInterrupt:
                            console.print("\n  [dim]Cancelled.[/dim]")
                        continue

                    elif cmd == "/model":
                        try:
                            result = run_model_switch_wizard(
                                workspace, console, agent, providers, cmd_arg
                            )
                            if result:
                                console.print(f"  [dim]{result}[/dim]")
                        except KeyboardInterrupt:
                            console.print("\n  [dim]Cancelled.[/dim]")
                        continue

                    elif cmd == "/mcp":
                        try:
                            result = run_mcp_manager(console, cmd_arg)
                            refreshed = agent.refresh_system_prompt()
                            if result:
                                console.print(f"  [dim]{result}[/dim]")
                            if refreshed:
                                console.print(
                                    "  [dim]System prompt refreshed (MCP config updated).[/dim]"
                                )
                        except KeyboardInterrupt:
                            console.print("\n  [dim]Cancelled.[/dim]")
                        continue

                    elif cmd == "/maxctx":
                        if not cmd_arg.strip():
                            console.print(
                                f"  [dim]Max tokens:[/dim] [bold]{agent.config.max_tokens:,}[/bold]"
                            )
                            console.print(
                                "  [dim]Usage: /maxctx <tokens>  e.g. /maxctx 8000, /maxctx 32k[/dim]"
                            )
                            continue
                        parsed = _parse_token_limit_input(cmd_arg)
                        if not parsed:
                            console.print(
                                "  [dim]Invalid value. Examples: /maxctx 8000, /maxctx 32k, /maxctx 0.5m[/dim]"
                            )
                            continue
                        if parsed < 256:
                            console.print(
                                "  [yellow]\u26a0[/yellow] Minimum allowed is 256 tokens"
                            )
                            continue
                        agent.config.max_tokens = parsed
                        agent.tool_handlers.config = agent.config
                        _save_active_config_fields(workspace, {"max_tokens": parsed})
                        active_profile = _infer_active_provider_profile(
                            agent, providers
                        )
                        if active_profile and active_profile in providers:
                            _save_provider_profile_fields(
                                workspace,
                                providers,
                                active_profile,
                                {"max_tokens": parsed},
                            )
                            providers[active_profile]["max_tokens"] = parsed
                        console.print(
                            f"  [green]\u2713[/green] Max tokens set to [bold]{parsed:,}[/bold]"
                        )
                        continue

                    elif cmd == "/compactthresh":
                        if not cmd_arg.strip():
                            current_pct = int(agent.config.compaction_threshold * 100)
                            threshold_tokens = int(
                                agent.config.max_tokens
                                * agent.config.compaction_threshold
                            )
                            console.print(
                                f"  [dim]Compaction threshold:[/dim] [bold]{current_pct}%[/bold] (~{threshold_tokens:,} tokens)"
                            )
                            console.print(
                                "  [dim]Context compaction starts when usage exceeds this threshold[/dim]"
                            )
                            console.print(
                                "  [dim]Usage: /compactthresh <percent|tokens>  e.g. /compactthresh 65, /compactthresh 50k, /compactthresh 25000[/dim]"
                            )
                            continue
                        try:
                            arg = cmd_arg.strip().lower().replace(",", "")

                            # Check if input is a token count (contains 'k' or number > 100)
                            is_token_count = False
                            token_value = None

                            if "k" in arg:
                                # Parse formats like "25k", "50k", "250k"
                                num_part = arg.replace("k", "").strip()
                                token_value = float(num_part) * 1000
                                is_token_count = True
                            elif "m" in arg:
                                # Parse formats like "0.1m" for 100k
                                num_part = arg.replace("m", "").strip()
                                token_value = float(num_part) * 1000000
                                is_token_count = True
                            else:
                                # Check if it's a plain number
                                num_val = float(arg.replace("%", "").strip())
                                if num_val > 100:
                                    # Treat as absolute token count
                                    token_value = num_val
                                    is_token_count = True
                                else:
                                    # Treat as percentage
                                    token_value = None
                                    is_token_count = False

                            if is_token_count and token_value is not None:
                                # Convert token count to threshold percentage
                                threshold = token_value / agent.config.max_tokens
                                if threshold < 0.1 or threshold > 0.95:
                                    console.print(
                                        f"  [yellow]\u26a0[/yellow] Token value must be between {int(agent.config.max_tokens * 0.1):,} and {int(agent.config.max_tokens * 0.95):,} for current max_tokens"
                                    )
                                    continue
                                agent.config.compaction_threshold = threshold
                                _save_active_config_fields(
                                    workspace, {"compaction_threshold": threshold}
                                )
                                pct = int(threshold * 100)
                                console.print(
                                    f"  [green]\u2713[/green] Compaction threshold set to [bold]{int(token_value):,} tokens[/bold] ({pct}%)"
                                )
                            else:
                                # Parse as percentage
                                parsed = float(arg.replace("%", "").strip())
                                if parsed < 10 or parsed > 95:
                                    console.print(
                                        "  [yellow]\u26a0[/yellow] Percentage must be between 10 and 95"
                                    )
                                    continue
                                agent.config.compaction_threshold = parsed / 100.0
                                _save_active_config_fields(
                                    workspace, {"compaction_threshold": parsed / 100.0}
                                )
                                threshold_tokens = int(
                                    agent.config.max_tokens
                                    * agent.config.compaction_threshold
                                )
                                console.print(
                                    f"  [green]\u2713[/green] Compaction threshold set to [bold]{int(parsed)}%[/bold]"
                                )
                                console.print(
                                    f"  [dim]Compaction will now start at ~{threshold_tokens:,} tokens[/dim]"
                                )
                        except ValueError:
                            console.print(
                                "  [dim]Usage: /compactthresh <percent|tokens>  e.g. /compactthresh 65, /compactthresh 50k, /compactthresh 25000[/dim]"
                            )
                        continue

                    elif cmd == "/iter":
                        if not cmd_arg:
                            console.print(
                                f"  [dim]Max iterations:[/dim] [bold]{agent.max_iterations}[/bold]"
                            )
                            continue
                        try:
                            new_val = int(cmd_arg.strip())
                            if new_val < 1:
                                console.print(
                                    "  [yellow]\u26a0[/yellow] Must be at least 1"
                                )
                                continue
                            agent.max_iterations = new_val
                            console.print(
                                f"  [green]\u2713[/green] Max iterations set to [bold]{new_val}[/bold]"
                            )
                        except ValueError:
                            console.print("  [dim]Usage: /iter <number>[/dim]")
                        continue

                    elif cmd == "/clip":
                        img_path, error = get_clipboard_image()
                        if error:
                            console.print(f"  [red]\u2717 {error}[/red]")
                            continue

                        console.print(f"  [dim]Clipboard image: {img_path}[/dim]")
                        question = (
                            cmd_arg.strip()
                            if cmd_arg
                            else "Describe this image in detail."
                        )
                        user_input = (
                            f"Analyze this image: {img_path}\n\nQuestion: {question}"
                        )
                        # Fall through to process this request

                    elif cmd == "/index":
                        idx = agent.workspace_index
                        if cmd_arg.strip().lower() == "rebuild":
                            console.print("  [dim]Rebuilding workspace index...[/dim]")
                            idx.build()
                            console.print(
                                f"  [green]\u2713[/green] Index rebuilt: {len(idx.files)} files in {idx._build_time:.2f}s"
                            )
                        elif cmd_arg.strip().lower() == "tree":
                            console.print(f"[dim]{idx.compact_tree()}[/dim]")
                        else:
                            console.print(f"  [dim]{idx.summary()}[/dim]")
                            console.print(
                                "  [dim][white]/index rebuild[/white]  re-scan workspace[/dim]"
                            )
                            console.print(
                                "  [dim][white]/index tree[/white]     show file tree[/dim]"
                            )
                        continue

                    elif cmd == "/log":
                        log_file = os.path.join(
                            workspace, ".harness_output", "harness.log"
                        )
                        if not os.path.exists(log_file):
                            console.print("  [dim]No log file yet.[/dim]")
                        else:
                            size_kb = os.path.getsize(log_file) / 1024
                            console.print(
                                f"  [dim]Log: {log_file} ({size_kb:.0f} KB)[/dim]"
                            )
                            if cmd_arg.strip():
                                try:
                                    n = int(cmd_arg.strip())
                                except ValueError:
                                    n = 30
                            else:
                                n = 30
                            with open(
                                log_file, "r", encoding="utf-8", errors="replace"
                            ) as fh:
                                lines = fh.readlines()
                            for ln in lines[-n:]:
                                console.print(
                                    f"  [dim]{rich_escape(ln.rstrip())}[/dim]"
                                )
                        continue

                    elif cmd == "/undo":
                        if not checkpoint_mgr.can_undo():
                            console.print("  [dim]Nothing to undo[/dim]")
                            continue
                        result = checkpoint_mgr.undo(agent.messages)
                        if result is None:
                            console.print("  [red]Undo failed[/red]")
                            continue
                        cp, diff, restored_msgs = result
                        agent.messages = restored_msgs
                        # Show summary
                        n_files = (
                            len(diff.files_modified)
                            + len(diff.files_added)
                            + len(diff.files_deleted)
                        )
                        parts = []
                        if diff.files_modified:
                            parts.append(f"{len(diff.files_modified)} modified")
                        if diff.files_added:
                            parts.append(f"{len(diff.files_added)} added")
                        if diff.files_deleted:
                            parts.append(f"{len(diff.files_deleted)} deleted")
                        file_summary = ", ".join(parts) if parts else "no file changes"
                        console.print(
                            f"  [green]\u21b6[/green] [bold]Undone[/bold] \u2014 reverted {file_summary}"
                        )
                        if (
                            diff.files_modified
                            or diff.files_added
                            or diff.files_deleted
                        ):
                            for f in (
                                diff.files_modified
                                + diff.files_added
                                + diff.files_deleted
                            )[:8]:
                                console.print(f"    [dim]{f}[/dim]")
                            remaining = n_files - 8
                            if remaining > 0:
                                console.print(f"    [dim]...and {remaining} more[/dim]")
                        console.print(f"  [dim]Request was: {cp.user_input}[/dim]")
                        if checkpoint_mgr.can_redo():
                            console.print(
                                "  [dim]Type [white]/redo[/white] to re-apply[/dim]"
                            )
                        continue

                    elif cmd == "/redo":
                        if not checkpoint_mgr.can_redo():
                            console.print("  [dim]Nothing to redo[/dim]")
                            continue
                        result = checkpoint_mgr.redo(agent.messages)
                        if result is None:
                            console.print("  [red]Redo failed[/red]")
                            continue
                        cp, diff, restored_msgs = result
                        agent.messages = restored_msgs
                        n_files = (
                            len(diff.files_modified)
                            + len(diff.files_added)
                            + len(diff.files_deleted)
                        )
                        parts = []
                        if diff.files_modified:
                            parts.append(f"{len(diff.files_modified)} modified")
                        if diff.files_added:
                            parts.append(f"{len(diff.files_added)} added")
                        if diff.files_deleted:
                            parts.append(f"{len(diff.files_deleted)} deleted")
                        file_summary = ", ".join(parts) if parts else "no file changes"
                        console.print(
                            f"  [green]\u21b7[/green] [bold]Redone[/bold] \u2014 restored {file_summary}"
                        )
                        if (
                            diff.files_modified
                            or diff.files_added
                            or diff.files_deleted
                        ):
                            for f in (
                                diff.files_modified
                                + diff.files_added
                                + diff.files_deleted
                            )[:8]:
                                console.print(f"    [dim]{f}[/dim]")
                        continue

                    elif cmd in ("/help", "/-"):
                        console.print()
                        console.print("  [bold]Chat[/bold]")
                        console.print(
                            "  [cyan]!command[/cyan]             [dim]Run a shell command[/dim]"
                        )
                        console.print(
                            "  [cyan]/clear[/cyan]               [dim]Clear conversation history[/dim]"
                        )
                        console.print(
                            "  [cyan]/compact[/cyan] [dim][strategy][/dim]   [dim]Compact context (half/quarter/last2)[/dim]"
                        )
                        console.print(
                            "  [cyan]/undo[/cyan]                [dim]Undo last turn (files + conversation)[/dim]"
                        )
                        console.print(
                            "  [cyan]/redo[/cyan]                [dim]Redo undone turn[/dim]"
                        )
                        console.print()
                        console.print("  [bold]Sessions[/bold]")
                        console.print(
                            "  [cyan]/sessions[/cyan]            [dim]List all sessions[/dim]"
                        )
                        console.print(
                            "  [cyan]/session[/cyan] [dim]<name>[/dim]      [dim]Switch or create session[/dim]"
                        )
                        console.print(
                            "  [cyan]/new[/cyan] [dim][name][/dim]          [dim]Start a fresh session[/dim]"
                        )
                        console.print(
                            "  [cyan]/delete[/cyan] [dim]<name>[/dim]       [dim]Delete a session[/dim]"
                        )
                        console.print(
                            "  [cyan]/save[/cyan]                [dim]Save current session[/dim]"
                        )
                        console.print()
                        console.print("  [bold]Models & Providers[/bold]")
                        console.print(
                            "  [cyan]/model[/cyan] [dim]<query>[/dim]       [dim]Search and switch models[/dim]"
                        )
                        console.print(
                            "  [cyan]/model list[/cyan]          [dim]List models from current provider[/dim]"
                        )
                        console.print(
                            "  [cyan]/providers[/cyan]           [dim]Manage provider profiles[/dim]"
                        )
                        console.print(
                            "  [cyan]/mcp[/cyan]                 [dim]Manage MCP servers (/mcp test <name>)[/dim]"
                        )
                        console.print(
                            "  [cyan]/config[/cyan]              [dim]Show/edit API configuration[/dim]"
                        )
                        console.print(
                            "  [cyan]/maxctx[/cyan] [dim]<n>[/dim]          [dim]Set max token cap (e.g. 8k, 32k)[/dim]"
                        )
                        console.print(
                            "  [cyan]/compactthresh[/cyan] [dim]<n>[/dim]      [dim]Set compaction threshold (e.g., 65%, 50k, 25000)[/dim]"
                        )
                        console.print(
                            "  [cyan]/iter[/cyan] [dim]<n>[/dim]            [dim]Set max agent iterations[/dim]"
                        )
                        console.print()
                        console.print("  [bold]Tools[/bold]")
                        console.print(
                            "  [cyan]/todo[/cyan]                [dim]Manage todo list (add/done/rm/clear)[/dim]"
                        )
                        console.print(
                            "  [cyan]/bg[/cyan]                  [dim]List background processes[/dim]"
                        )
                        console.print(
                            "  [cyan]/clip[/cyan] [dim][question][/dim]     [dim]Analyze clipboard image[/dim]"
                        )
                        console.print()
                        console.print("  [bold]Diagnostics[/bold]")
                        console.print(
                            "  [cyan]/ctx[/cyan]                 [dim]Show context usage[/dim]"
                        )
                        console.print(
                            "  [cyan]/tokens[/cyan]              [dim]Token breakdown by message[/dim]"
                        )
                        console.print(
                            "  [cyan]/cost[/cyan]                [dim]API usage and cost totals[/dim]"
                        )
                        console.print(
                            "  [cyan]/smart[/cyan]               [dim]Smart context analysis[/dim]"
                        )
                        console.print(
                            "  [cyan]/dump[/cyan]                [dim]Dump full context to JSON[/dim]"
                        )
                        console.print(
                            "  [cyan]/policyeval[/cyan] [dim]<dump.json> [--embed-backend <...>][/dim] [dim]Replay + classifier eval on a context dump[/dim]"
                        )
                        console.print(
                            "  [cyan]/index[/cyan] [dim][rebuild|tree][/dim] [dim]Project file index[/dim]"
                        )
                        console.print(
                            "  [cyan]/log[/cyan] [dim][n][/dim]             [dim]Show last n log lines[/dim]"
                        )
                        console.print()
                        console.print("  [bold]Keys[/bold]")
                        console.print(
                            "  [cyan]Esc[/cyan]                  [dim]Stop / interrupt agent[/dim]"
                        )
                        console.print(
                            "  [cyan]Ctrl+C[/cyan]               [dim]Interrupt (press twice to exit)[/dim]"
                        )
                        console.print(
                            "  [cyan]Ctrl+B[/cyan]               [dim]Send command to background[/dim]"
                        )
                        if HAS_PROMPT_TOOLKIT:
                            console.print(
                                "  [cyan]Ctrl+Enter[/cyan]           [dim]Insert newline[/dim]"
                            )
                            console.print(
                                "  [cyan]Ctrl+V[/cyan]               [dim]Paste clipboard image[/dim]"
                            )
                            console.print(
                                "  [cyan]Ctrl+Z[/cyan]               [dim]Undo last turn[/dim]"
                            )
                            console.print(
                                "  [cyan]Ctrl+Y[/cyan]               [dim]Redo[/dim]"
                            )
                            console.print(
                                "  [cyan]Ctrl+T[/cyan]               [dim]Toggle reasoning effort[/dim]"
                            )
                        console.print()
                        continue

                    else:
                        console.print(
                            f"  [dim]Unknown command. Type [white]/help[/white] for available commands.[/dim]"
                        )
                        continue

                multimodal_content: Optional[List[Dict[str, Any]]] = None
                multimodal_label: Optional[str] = None
                original_user_input_for_cleanup = user_input

                # --- Detect images: clipboard markers AND typed file paths ---
                image_paths: List[Path] = []
                text_part = user_input

                if isinstance(user_input, str) and "[[clipboard_image:" in user_input:
                    text_part, image_paths = _extract_clipboard_image_markers(
                        user_input
                    )
                    image_paths = [p for p in image_paths if p.exists()]
                    if not image_paths:
                        user_input, _ = _extract_clipboard_image_markers(user_input)
                        text_part = user_input

                # Also detect image file paths typed directly in the message
                if isinstance(text_part, str):
                    remaining_text, typed_images = _extract_image_paths_from_text(text_part)
                    if typed_images:
                        image_paths.extend(typed_images)
                        text_part = remaining_text

                if image_paths:
                    if _supports_multimodal_input(
                        agent.config.api_url, agent.config.model
                    ):
                        multimodal_content = _build_multimodal_user_content(
                            text_part, image_paths
                        )
                        multimodal_label = (
                            text_part or f"[attached {len(image_paths)} image(s)]"
                        )
                        img_names = ", ".join(p.name for p in image_paths)
                        console.print(
                            f"  [dim]Attached {len(image_paths)} image(s): {img_names}[/dim]"
                        )
                    else:
                        fallback_q = text_part or "Describe this image in detail."
                        user_input = f"Analyze this image: {image_paths[0]}\n\nQuestion: {fallback_q}"
                        console.print(
                            "  [dim]Model may not support images \u2014 using fallback mode[/dim]"
                        )

                log.info(
                    "User input: %s", truncate(multimodal_label or user_input, 200)
                )

                # Take checkpoint before each agent turn for undo/redo
                try:
                    _cp_input = multimodal_label or (
                        user_input if isinstance(user_input, str) else "[multimodal]"
                    )
                    checkpoint_mgr.take_snapshot(
                        _cp_input,
                        len(agent.messages),
                        model=agent.config.model,
                    )
                except Exception as _cp_err:
                    log.warning("Checkpoint snapshot failed: %s", _cp_err)

                # Start watchdog timer to detect hangs
                import threading
                import traceback

                _watchdog_start_time = time.time()
                _watchdog_last_activity = _watchdog_start_time
                _watchdog_main_thread = threading.current_thread()
                _watchdog_stop_event = threading.Event()
                _watchdog_debug_mode = (
                    os.environ.get("HARNESS_DEBUG_WATCHDOG", "0") == "1"
                )

                def _watchdog_timer():
                    """Print periodic status and stack traces to help diagnose freezes."""
                    _trace_dumped = False
                    while not _watchdog_stop_event.is_set():
                        # Wait 5 seconds or until stop signal
                        if _watchdog_stop_event.wait(5.0):
                            break  # Stop signal received
                        elapsed = time.time() - _watchdog_start_time
                        inactive = time.time() - _watchdog_last_activity
                        if elapsed > 10:  # Only start warning after 10 seconds
                            log.warning(
                                "WATCHDOG: Request running for %.1fs (inactive for %.1fs)",
                                elapsed,
                                inactive,
                            )
                            # Only print to console in debug mode
                            if _watchdog_debug_mode:
                                print(
                                    f"[WATCHDOG] Processing for {elapsed:.1f}s...",
                                    flush=True,
                                )
                                # Dump stack trace ONCE after 30s to diagnose hangs
                                if elapsed > 30 and not _trace_dumped:
                                    _trace_dumped = True
                                    print(
                                        "\n=== STACK TRACE (where the code is) ===",
                                        flush=True,
                                    )
                                    for (
                                        thread_id,
                                        frame,
                                    ) in sys._current_frames().items():
                                        thread_name = "Unknown"
                                        for t in threading.enumerate():
                                            if t.ident == thread_id:
                                                thread_name = t.name
                                                break
                                        print(
                                            f"\nThread: {thread_name} (ID: {thread_id})",
                                            flush=True,
                                        )
                                        traceback.print_stack(frame, file=sys.stdout)
                                    print("=== END STACK TRACE ===\n", flush=True)
                                    log.warning("WATCHDOG: Stack trace dumped once")

                _watchdog_thread = threading.Thread(target=_watchdog_timer, daemon=True)
                _watchdog_thread.start()

                result = None
                try:
                    try:
                        result = loop.run_until_complete(
                            run_single(
                                agent,
                                multimodal_content
                                if multimodal_content is not None
                                else user_input,
                                console,
                                user_label=multimodal_label,
                            )
                        )
                        _watchdog_last_activity = time.time()  # Mark activity
                    finally:
                        # Signal watchdog to stop immediately
                        _watchdog_stop_event.set()
                    log.info("run_single completed, result_len=%d", len(result or ""))
                    # Show live todo panel if there are any todos
                    agent.todo_manager.print_todo_panel(console)
                except KeyboardInterrupt:
                    log.warning("KeyboardInterrupt during run_single")
                    console.print(
                        "\n  [yellow]Interrupted[/yellow] [dim]- press Ctrl+C again to exit[/dim]"
                    )
                    last_interrupt_time = time.time()
                except Exception as e:
                    log_exception(log, "run_single failed", e)
                    console.print(f"  [red]\u2717 {rich_escape(str(e))}[/red]")

                # Auto-save session after each exchange
                agent.save_session(str(session_path))
                if (
                    isinstance(original_user_input_for_cleanup, str)
                    and "[[clipboard_image:" in original_user_input_for_cleanup
                ):
                    _, _tmp_paths = _extract_clipboard_image_markers(
                        original_user_input_for_cleanup
                    )
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
