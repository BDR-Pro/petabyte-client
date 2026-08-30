"""Model source adapters. Petabyte owns the higher-level UX; each provider only knows how to
search, resolve a revision, produce a normalized manifest, and hand back per-file download URLs."""
from .base import ModelProvider, SearchResult, ProviderError, GatedError
from .registry import get_provider, search_all

__all__ = ["ModelProvider", "SearchResult", "ProviderError", "GatedError",
           "get_provider", "search_all"]
