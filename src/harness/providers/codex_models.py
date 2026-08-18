"""Model registry for Codex/OAuth-based providers."""

import time
from typing import List, Optional

# Fallback set — used only if models.dev/api.json is unreachable.
# Kept for backward compatibility and as an offline fallback.
ALLOWED_CODEX_MODELS = {
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.1-codex",
}

_MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_TTL_SECS = 3600  # refresh the cached list hourly

_remote_openai_models: Optional[List[str]] = None
_remote_fetched_at: float = 0.0


def _fetch_openai_models() -> List[str]:
    """Fetch agent-capable OpenAI model IDs from models.dev/api.json.

    Filters to chat/agent models only: they must support tool calling and
    produce text output. This excludes gpt-image-*, text-embedding-*, and
    gpt-realtime-* (audio) models that the Codex/ChatGPT OAuth endpoint
    cannot drive as a coding agent.
    """
    import httpx

    resp = httpx.get(
        _MODELS_DEV_URL,
        timeout=10,
        headers={"User-Agent": "Harness/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()

    models = data.get("openai", {}).get("models", {})
    ids: List[str] = []
    for model_id, meta in models.items():
        if not isinstance(meta, dict):
            continue
        if not meta.get("tool_call"):
            continue
        out = meta.get("modalities", {}).get("output")
        if out and out != ["text"]:
            continue
        ids.append(model_id)
    return sorted(ids)


def get_codex_models() -> List[str]:
    """Get available models for the OpenAI subscription (OAuth) flow.

    Resolves the list from models.dev/api.json (cached for an hour) so it
    tracks newly released models, falling back to a hardcoded set when the
    network is unavailable.
    """
    global _remote_openai_models, _remote_fetched_at

    now = time.time()
    if _remote_openai_models is not None and (now - _remote_fetched_at) <= _MODELS_DEV_TTL_SECS:
        return sorted(_remote_openai_models)

    try:
        fetched = _fetch_openai_models()
    except Exception:
        fetched = []

    if fetched:
        _remote_openai_models = fetched
        _remote_fetched_at = now
        return sorted(fetched)

    # Offline fallback: prefer any previously cached list, then hardcoded set.
    return sorted(_remote_openai_models if _remote_openai_models is not None else ALLOWED_CODEX_MODELS)


def is_codex_model(model: str) -> bool:
    """Check if *model* is available via the OpenAI subscription OAuth flow.

    Args:
        model: Model name

    Returns:
        True if it's a known Codex/OpenAI model
    """
    return model in get_codex_models() or "codex" in model.lower()
