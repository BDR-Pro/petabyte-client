"""modelhub — Petabyte's provider-independent model discovery / download / management layer.

Hugging Face-grade convenience (`petabyte model pull publisher/model`) with a clean local layer:
content-addressed cache, resumable verified downloads, hardware-aware compatibility, and a provider
abstraction so models can come from Hugging Face, a mirror, or a future Petabyte registry without the
rest of Petabyte caring which.

Pure standard library at the core (urllib) so the CLI installs light and the whole thing is testable
offline against a local HTTP server.
"""
from .ids import ModelRef, parse, ModelIdError
from .manifest import Manifest, ModelFile
from .cache import Cache, default_home, CachePathError
from .manager import ModelManager, CompatibilityError, ConcurrentPullError
from .hardware import detect, compatibility
from . import download
from .providers import get_provider, search_all, ProviderError, GatedError

__all__ = [
    "ModelRef", "parse", "ModelIdError", "Manifest", "ModelFile",
    "Cache", "default_home", "CachePathError",
    "ModelManager", "CompatibilityError", "ConcurrentPullError",
    "detect", "compatibility", "download",
    "get_provider", "search_all", "ProviderError", "GatedError",
]
