"""Provider management — loading, switching, model picker, provider hub."""
import concurrent.futures
import json, os, hashlib, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config import get_global_config_path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape as rich_escape
from rich import box
import httpx

def _get_legacy_global_models_path() -> Path:
    return Path.home() / ".z" / "models.json"


def load_providers(workspace: str) -> Dict[str, dict]:
    """Load provider configs from ~/.z.json (single-file config)."""
    import json

    cfg_path = get_global_config_path()
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
    "ollama-local": (
        "Local Ollama",
        "http://localhost:11434/v1",
        "",
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


# ── Provider routing table ──────────────────────────────────────────
# Maps URL substrings to provider configuration for model listing.
# Each entry: (url_substring, litellm_prefix, flags)
#   litellm_prefix: str = prefix required for LiteLLM routing, None = bare IDs work
#   flags: set of strings
#     "custom"        — completely different API (Bedrock), handled separately
#     "x-api-key"     — uses x-api-key header instead of Authorization: Bearer
#     "registry-only" — no /v1/models endpoint; use LiteLLM registry only
_PROVIDER_ROUTES = [
    ("amazonaws.com",    None,        {"custom"}),       # Bedrock — AWS sigv4
    ("anthropic.com",    None,        {"x-api-key"}),    # Anthropic — x-api-key auth
    ("openrouter.ai",    "openrouter", set()),           # needs openrouter/ prefix
    ("together.xyz",     None,        set()),            # bare model IDs work
    ("api.deepseek.com", "deepseek",  set()),            # needs deepseek/ prefix
    ("minimax",          "minimax",   {"registry-only"}),# registry fallback only
    ("api.groq.com",     "groq",      {"registry-only"}),# registry fallback only
]


def _fetch_provider_model_ids(api_url: str, api_key: str) -> List[str]:
    """Fetch model IDs from a provider using LiteLLM.

    Uses a generic strategy:
      1. Look up the URL in the provider routing table.
      2. Try querying the provider's /v1/models endpoint (if available).
      3. Prefix model IDs for LiteLLM routing if needed.
      4. Fall back to LiteLLM's model registry if the API query fails.
    """
    # Check if OAuth token FIRST, before importing streaming_client
    # This avoids the slow LiteLLM import for OAuth providers
    if api_key and api_key.startswith("oauth:"):
        return _fetch_oauth_models(api_url)

    from ..streaming_client import search_litellm_models
    url = (api_url or "").lower()

    # ── Look up URL in routing table ────────────────────────────────
    prefix = None
    flags: set = set()
    for pattern, p, f in _PROVIDER_ROUTES:
        if pattern in url:
            prefix = p
            flags = f
            break

    # ── Custom provider (Bedrock) — completely different API ────────
    if "custom" in flags:
        return _fetch_bedrock_models(api_url, api_key)

    # ── Try querying the provider's /v1/models endpoint ─────────────
    api_models: List[str] = []
    if "registry-only" not in flags:
        if "x-api-key" in flags:
            # Anthropic: uses x-api-key header instead of Bearer
            api_models = _fetch_models_from_provider_api(
                api_url,
                api_key,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        else:
            api_models = _fetch_models_from_provider_api(api_url, api_key)

    if api_models:
        if prefix:
            return [f"{prefix}/{m}" if not m.startswith(f"{prefix}/") else m for m in api_models]
        return api_models

    # ── Fallback to LiteLLM registry ────────────────────────────────
    if prefix:
        return search_litellm_models(f"{prefix}/")

    # Anthropic fallback: LiteLLM registry contains many proxy/gateway
    # routes (deepinfra/anthropic/..., openrouter/anthropic/...) which
    # require a different API key.  Filter to only direct models.
    if "x-api-key" in flags:
        return _filter_anthropic_registry_models(search_litellm_models)

    # Unknown provider — try generic API query
    return _fetch_models_from_provider_api(api_url, api_key)


def _fetch_bedrock_models(api_url: str, api_key: str) -> List[str]:
    """Fetch model IDs from AWS Bedrock.  Uses Bedrock's custom API."""
    from ..streaming_client import search_litellm_models
    from ..providers.bedrock_provider import list_bedrock_models

    url = (api_url or "").lower()
    region = "us-east-1"
    if ".amazonaws.com" in url:
        parts = url.split(".")
        if len(parts) >= 2:
            potential_region = parts[1]
            if potential_region.startswith("us-") or potential_region.startswith("eu-"):
                region = potential_region

    models = list_bedrock_models(api_key, region)
    if models:
        return models
    return search_litellm_models("bedrock/")


def _filter_anthropic_registry_models(
    search_fn,
) -> List[str]:
    """Return only direct Anthropic models from LiteLLM's registry.

    Excludes proxy/gateway routes (deepinfra/anthropic/..., openrouter/anthropic/...)
    that require a different API key.
    """
    from litellm import model_cost as _anthropic_cost

    bare = [
        m
        for m, info in _anthropic_cost.items()
        if "/" not in m and info.get("litellm_provider") == "anthropic"
    ]
    litellm_matches = [m for m in search_fn("anthropic/") if m.lower().startswith("anthropic/")]
    return sorted(set(litellm_matches + bare))


def _fetch_oauth_models(api_url: str) -> List[str]:
    """Fetch model IDs for OAuth-based providers (GitHub Copilot, OpenAI Codex)."""
    url_lower = (api_url or "").lower()
    if "githubcopilot" in url_lower or "copilot" in url_lower:
        from ..providers.copilot_oauth_client import get_copilot_models
        return get_copilot_models()
    from ..providers.codex_models import get_codex_models
    return get_codex_models()


def _fetch_models_from_provider_api(
    api_url: str,
    api_key: str,
    headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Query provider's /v1/models endpoint (OpenAI-compatible format).

    By default uses ``Authorization: Bearer <key>`` auth.  Pass custom
    *headers* to override (e.g. ``{"x-api-key": ...}`` for Anthropic).
    """
    import requests

    try:
        if headers is None:
            headers = {}
            if api_key and not api_key.startswith("oauth:"):
                headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(
            f"{api_url.rstrip('/')}/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Handle both {"data": [...]} and plain [...] formats
        items = data if isinstance(data, list) else data.get("data", [])
        models = []
        for item in items:
            if isinstance(item, dict) and "id" in item:
                models.append(item["id"])

        return sorted(models)
    except Exception:
        # On any error, return empty list to trigger manual entry or fallback
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
    agent: "ClineAgent",
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
    agent: "ClineAgent", providers: Dict[str, dict]
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
    models_path = get_global_config_path()
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
    agent: "ClineAgent",
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

    # Collect work items (skip providers without credentials)
    work_items: List[tuple[str, str, str, str, dict]] = []
    for profile, cfg in searchable:
        api_url = cfg.get("api_url", "")
        api_key = cfg.get("api_key", "")
        if not api_url or not api_key:
            continue
        provider_display = _provider_display_name(profile, cfg)
        work_items.append((profile, provider_display, api_url, api_key, cfg))
        console.print(f"  [dim]Fetching: {provider_display}...[/dim]")

    if not work_items:
        return "No configured providers with valid API configuration."

    # Parallel fetch all providers — each call is I/O bound (HTTP request)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(10, len(work_items))
    ) as executor:
        future_map = {
            executor.submit(
                _fetch_provider_model_ids_cached, api_url, api_key, refresh
            ): (profile, provider_display, cfg)
            for (profile, provider_display, api_url, api_key, cfg) in work_items
        }
        for future in concurrent.futures.as_completed(future_map):
            profile, provider_display, cfg = future_map[future]
            try:
                mids = future.result()
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
        ("26", "ollama-local"),
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
            print("  Please enter 1-26.")
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
    agent: "ClineAgent",
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
            from ..providers.oauth import get_oauth_manager

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
        from ..providers.codex_models import get_codex_models

        if "GitHub Copilot" in label:
            from ..providers.copilot_oauth_client import get_copilot_models

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
                        "  [dim]No models returned using manual entry[/dim]"
                    )
            except Exception as e:
                console.print(
                    f"  [yellow]Model fetch failed: {rich_escape(str(e))}[/yellow]"
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

    models_path = get_global_config_path()
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
    agent: "ClineAgent", providers: Dict[str, dict]
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
    agent: "ClineAgent",
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
        models_path = get_global_config_path()
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

