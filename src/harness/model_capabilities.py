"""Model capability detection (vision, etc.).

Uses two sources, in order of preference:
1. LiteLLM's model info (``litellm.get_model_info``)
2. models.dev/api.json remote metadata

If neither source can determine the capability, the function returns False
(conservative default). No name-based heuristics are used.
"""

from typing import Optional
import logging

log = logging.getLogger("harness.model_capabilities")


def _get_litellm_model_info(model: str) -> Optional[dict]:
    """Return LiteLLM model info dict, or None if unavailable/unknown."""
    try:
        import litellm
        import io
        import sys

        # Redirect stdout/stderr to suppress LiteLLM's provider list messages
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            info = litellm.get_model_info(model)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        if isinstance(info, dict):
            return info
    except Exception:
        # LiteLLM not installed or model not in registry
        pass
    return None


def _remote_model_has_vision(model: str, api_url: str = "") -> Optional[bool]:
    """Check models.dev metadata for vision capability.

    Returns True/False if the remote data has a definitive signal,
    otherwise None.
    """
    try:
        from .context_management import get_remote_model_data

        data = get_remote_model_data(model, api_url=api_url)
        if not isinstance(data, dict):
            return None

        for key in ("supports_vision", "vision", "multimodal", "image"):
            val = data.get(key)
            if val is not None:
                return bool(val)

        caps = data.get("capabilities") or data.get("caps")
        if isinstance(caps, dict):
            for key in ("vision", "image", "multimodal"):
                val = caps.get(key)
                if val is not None:
                    return bool(val)

        modalities = data.get("modalities")
        if isinstance(modalities, dict):
            inputs = modalities.get("input", [])
            if isinstance(inputs, list):
                return any(str(m).lower() in {"image", "vision"} for m in inputs)
        if isinstance(modalities, list):
            return any(str(m).lower() in {"image", "vision"} for m in modalities)
    except Exception as exc:
        log.debug("Failed to read remote model capabilities: %s", exc)
    return None


def supports_vision(model: str, api_url: str = "") -> bool:
    """Return True if the model is believed to support image/vision input.

    Uses LiteLLM first, then models.dev. If neither source knows the model,
    returns False so images are never sent to a model that cannot handle them.
    """
    if not model:
        return False

    # 1. LiteLLM registry
    litellm_info = _get_litellm_model_info(model)
    if litellm_info and "supports_vision" in litellm_info:
        return bool(litellm_info["supports_vision"])

    # 2. models.dev remote metadata
    remote = _remote_model_has_vision(model, api_url=api_url)
    if remote is not None:
        return remote

    # Unknown model — conservative default.
    return False
