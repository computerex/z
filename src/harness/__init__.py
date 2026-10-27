"""Agentic coding harness optimized for Z.AI GLM-4.7."""

import os
import warnings

# urllib3/charset_normalizer version mismatch — harmless packaging noise
warnings.filterwarnings("ignore", message=r"urllib3.*doesn't match a supported version")

# LiteLLM (1.6x) dumps its own response models with ``.dict()`` on every
# streamed chunk and in its success-logging path (streaming_handler.py,
# litellm_logging.py).  Some nested objects (``Message``, ``Choices``) are
# created via ``model_construct``/unvalidated init, so they carry only a
# subset of the declared fields.  Pydantic >= 2.12 then emits a
# ``PydanticSerializationUnexpectedValue`` warning ("Pydantic serializer
# warnings: Expected 9 fields but got 6...") for each dump, even though the
# serialized payload is still correct.  It is pure noise on every turn and
# the warning is raised inside LiteLLM, not by the harness.
# Set HARNESS_PYDANTIC_WARNINGS=1 to see them again when debugging
# serialization mismatches.
if os.environ.get("HARNESS_PYDANTIC_WARNINGS") != "1":
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings")

# Note: Heavy imports (ClineAgent, streaming_client) are NOT imported here
# to keep startup fast. Import them directly when needed.
from .providers import get_codex_models, is_codex_model

__version__ = "0.1.0"
__all__ = [
    "get_codex_models",
    "is_codex_model",
]
