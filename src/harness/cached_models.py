"""Cached model list manager for harness.

Provides efficient caching of model lists from providers to avoid
slow fetches every time. Models are cached locally and refreshed
periodically.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    provider: str
    context_length: int = 4096
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class ProviderModels:
    """Models for a specific provider."""

    provider: str
    models: List[ModelInfo]
    last_updated: float = 0

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if cache is stale."""
        if self.last_updated == 0:
            return True
        age = time.time() - self.last_updated
        return age > (max_age_hours * 3600)


class CachedModelList:
    """Manages cached model lists from providers."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.cache/harness
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "harness"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, ProviderModels] = {}

    def _get_cache_file(self, provider: str) -> Path:
        """Get cache file path for a provider."""
        return self.cache_dir / f"{provider}_models.json"

    def _load_from_disk(self, provider: str) -> Optional[ProviderModels]:
        """Load cached models from disk."""
        cache_file = self._get_cache_file(provider)
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            models = [ModelInfo(**m) for m in data.get("models", [])]
            return ProviderModels(
                provider=provider,
                models=models,
                last_updated=data.get("last_updated", 0),
            )
        except Exception:
            return None

    def _save_to_disk(self, provider_models: ProviderModels):
        """Save models to disk cache."""
        cache_file = self._get_cache_file(provider_models.provider)
        data = {
            "provider": provider_models.provider,
            "models": [asdict(m) for m in provider_models.models],
            "last_updated": provider_models.last_updated,
        }
        cache_file.write_text(json.dumps(data, indent=2))

    def get_models(
        self, provider: str, force_refresh: bool = False
    ) -> Optional[ProviderModels]:
        """Get cached models for a provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            force_refresh: Force refresh from API

        Returns:
            ProviderModels or None if not cached and not available
        """
        # Check memory cache first
        if provider in self._memory_cache and not force_refresh:
            cached = self._memory_cache[provider]
            if not cached.is_stale():
                return cached

        # Try disk cache
        if not force_refresh:
            disk_cached = self._load_from_disk(provider)
            if disk_cached and not disk_cached.is_stale():
                self._memory_cache[provider] = disk_cached
                return disk_cached

        return None

    def set_models(self, provider: str, models: List[ModelInfo]):
        """Cache models for a provider.

        Args:
            provider: Provider name
            models: List of ModelInfo objects
        """
        provider_models = ProviderModels(
            provider=provider, models=models, last_updated=time.time()
        )
        self._memory_cache[provider] = provider_models
        self._save_to_disk(provider_models)

    def search_models(
        self, query: str, provider: Optional[str] = None
    ) -> List[ModelInfo]:
        """Search cached models.

        Args:
            query: Search query
            provider: Optional provider to limit search

        Returns:
            List of matching ModelInfo objects
        """
        results = []
        query_lower = query.lower()

        providers_to_search = (
            [provider] if provider else list(self._memory_cache.keys())
        )

        for prov in providers_to_search:
            cached = self._memory_cache.get(prov)
            if not cached:
                cached = self._load_from_disk(prov)
                if cached:
                    self._memory_cache[prov] = cached

            if cached:
                for model in cached.models:
                    if (
                        query_lower in model.id.lower()
                        or query_lower in model.name.lower()
                    ):
                        results.append(model)

        return results

    def clear_cache(self, provider: Optional[str] = None):
        """Clear cache for a provider or all providers.

        Args:
            provider: Provider name or None to clear all
        """
        if provider:
            if provider in self._memory_cache:
                del self._memory_cache[provider]
            cache_file = self._get_cache_file(provider)
            if cache_file.exists():
                cache_file.unlink()
        else:
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("*_models.json"):
                cache_file.unlink()


# Global instance
_model_cache: Optional[CachedModelList] = None


def get_model_cache() -> CachedModelList:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = CachedModelList()
    return _model_cache
