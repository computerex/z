"""Codex models constants - separate module to avoid importing LiteLLM.

This module provides hardcoded Codex model lists without importing
heavy dependencies like LiteLLM.
"""

from typing import List

# Allowed Codex models for OAuth (from opencode)
ALLOWED_CODEX_MODELS = {
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.1-codex",
}


def get_codex_models() -> List[str]:
    """Get list of allowed Codex models for OAuth.

    Returns:
        List of model IDs (hardcoded for instant access)
    """
    return sorted(ALLOWED_CODEX_MODELS)


def is_codex_model(model: str) -> bool:
    """Check if model is a Codex model.

    Args:
        model: Model name

    Returns:
        True if it's a Codex model
    """
    return model in ALLOWED_CODEX_MODELS or "codex" in model.lower()
