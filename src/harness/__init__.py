"""Agentic coding harness optimized for Z.AI GLM-4.7."""

# Note: Heavy imports (ClineAgent, streaming_client) are NOT imported here
# to keep startup fast. Import them directly when needed.
from .codex_models import get_codex_models, is_codex_model

__version__ = "0.1.0"
__all__ = [
    "get_codex_models",
    "is_codex_model",
]
