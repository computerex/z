"""Configuration management for the harness."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


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
    def from_env(cls, env_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        return cls(
            api_url=os.getenv("LLM_API_URL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "glm-4.7"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "128000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            workspace_path=Path(os.getenv("WORKSPACE_PATH", str(Path.cwd()))),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "32000")),
        )
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.api_url:
            raise ValueError("LLM_API_URL is required")
        if not self.api_key:
            raise ValueError("LLM_API_KEY is required")
        return True
