"""Configuration management for the harness."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def get_global_config_path() -> Path:
    """Get path to global config: ~/.z.json"""
    return Path.home() / ".z.json"


def get_workspace_config_path(workspace: Optional[Path] = None) -> Path:
    """Legacy helper kept for compatibility (workspace config is ignored)."""
    ws = workspace or Path.cwd()
    return ws / ".z" / ".z.json"


def load_json_config(path: Path) -> dict:
    """Load config from JSON file if it exists."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _is_empty_override(value) -> bool:
    """True when a config value should not override an existing non-empty value."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _merge_non_empty(base: dict, overlay: dict) -> dict:
    """Merge overlay into base, skipping None/empty-string values."""
    out = dict(base)
    for k, v in (overlay or {}).items():
        if _is_empty_override(v):
            continue
        out[k] = v
    return out


@dataclass
class Config:
    """Configuration for the agentic harness."""
    
    api_url: str = ""
    api_key: str = ""
    model: str = "glm-4.7"
    max_tokens: int = 128000
    temperature: float = 0.7
    workspace_path: Path = field(default_factory=lambda: Path.cwd())
    max_context_tokens: int = 32000
    embedding_model: str = "all-MiniLM-L6-v2"
    
    @classmethod
    def from_json(
        cls,
        workspace: Optional[Path] = None,
        overrides: Optional[dict] = None,
    ) -> "Config":
        """Load configuration from JSON files (global-only).

        Priority (later overrides earlier):
        1. ~/.z.json (global)
        2. explicit overrides (runtime)
        """
        # Start with defaults
        config_data = {}
        
        # Load global config
        global_config = load_json_config(get_global_config_path())
        config_data = _merge_non_empty(config_data, global_config)
        
        if overrides:
            config_data = _merge_non_empty(config_data, overrides)
        
        return cls(
            api_url=config_data.get("api_url", ""),
            api_key=config_data.get("api_key", ""),
            model=config_data.get("model", "glm-4.7"),
            max_tokens=int(config_data.get("max_tokens", 128000)),
            temperature=float(config_data.get("temperature", 0.7)),
            workspace_path=workspace or Path.cwd(),
            max_context_tokens=int(config_data.get("max_context_tokens", 32000)),
        )
    
    @classmethod
    def from_env(
        cls,
        env_path: Optional[Path] = None,
        workspace: Optional[Path] = None,
    ) -> "Config":
        """Load configuration from global JSON config file.

        The env_path parameter is accepted but ignored (legacy compat).
        """
        return cls.from_json(workspace)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.api_url:
            raise ValueError("API URL is required. Run 'python harness.py --install' to configure.")
        if not self.api_key:
            raise ValueError("API key is required. Run 'python harness.py --install' to configure.")
        return True
