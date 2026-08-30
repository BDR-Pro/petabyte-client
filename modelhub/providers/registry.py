"""registry.py — provider selection + the (optional) Petabyte model registry.

`get_provider(ref)` picks the adapter from the ref's source. The Petabyte registry maps short,
memorable aliases (llama3.1:8b) to the appropriate upstream open model, so a user can type
`petabyte model pull llama3.1:8b` without knowing the publisher. Metadata is normalized by Petabyte
while the actual blobs still come from upstream — we do NOT mirror giant third-party weights.

The alias table is static + offline by default; a future models.petabyte.market can extend it via
PETABYTE_REGISTRY_URL without changing callers.
"""
import os

from .base import ModelProvider, SearchResult, get_json
from .huggingface import HuggingFaceProvider
from .http import HttpProvider

# curated aliases -> upstream (source, id). Small, honest, and resolvable with no network of ours.
ALIASES = {
    "llama3.1:8b": ("hf", "meta-llama/Llama-3.1-8B-Instruct"),
    "llama3.1:70b": ("hf", "meta-llama/Llama-3.1-70B-Instruct"),
    "llama3.2:1b": ("hf", "meta-llama/Llama-3.2-1B-Instruct"),
    "llama3.2:3b": ("hf", "meta-llama/Llama-3.2-3B-Instruct"),
    "qwen3:8b": ("hf", "Qwen/Qwen3-8B"),
    "qwen2.5:7b": ("hf", "Qwen/Qwen2.5-7B-Instruct"),
    "mistral:7b": ("hf", "mistralai/Mistral-7B-Instruct-v0.3"),
    "phi3:mini": ("hf", "microsoft/Phi-3-mini-4k-instruct"),
    "gemma2:9b": ("hf", "google/gemma-2-9b-it"),
}


def _load_aliases():
    table = dict(ALIASES)
    url = os.environ.get("PETABYTE_REGISTRY_URL")
    if url:
        try:
            remote = get_json(url.rstrip("/") + "/aliases.json") or {}
            for k, v in remote.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    table[k] = (v[0], v[1])
        except Exception:  # noqa: BLE001 — registry is optional; static table still works offline
            pass
    return table


class PetabyteRegistryProvider(ModelProvider):
    """Resolves curated aliases to an upstream provider, then delegates."""
    name = "pt"

    def __init__(self):
        self.aliases = _load_aliases()

    def _upstream_ref(self, ref):
        from ..ids import parse
        key = f"{ref.name}:{ref.tag}" if ref.tag else ref.name
        target = self.aliases.get(key) or self.aliases.get(ref.name)
        if not target:
            raise KeyError(f"unknown Petabyte alias {key!r}")
        source, upstream_id = target
        up = parse(upstream_id, default_source=source)
        if ref.revision:
            up = up.with_revision(ref.revision)
        return up

    def resolve(self, ref):
        up = self._upstream_ref(ref)
        return get_provider(up).resolve(up)

    def manifest(self, ref, **kw):
        up = self._upstream_ref(ref)
        m = get_provider(up).manifest(up, **kw)
        m.extra["alias"] = str(ref)
        return m

    def auth_headers(self):
        return HuggingFaceProvider().auth_headers()

    def search(self, query, *, limit=25, filters=None):
        q = (query or "").lower()
        out = []
        for alias, (src, upstream) in sorted(self.aliases.items()):
            if q and q not in alias and q not in upstream.lower():
                continue
            out.append(SearchResult(alias, "pt", name=alias,
                                    architecture=upstream, pipeline="alias"))
        return out[:limit]


def get_provider(ref) -> ModelProvider:
    src = (ref.source or "hf").lower()
    if src in ("http", "https"):
        return HttpProvider()
    if src in ("pt", "petabyte"):
        return PetabyteRegistryProvider()
    if src in ("hf", "huggingface"):
        return HuggingFaceProvider()
    raise ValueError(f"no provider for source {src!r}")


def search_all(query, *, limit=25, filters=None, sources=("hf",)):
    """Aggregate search across sources (HF by default; add 'pt' for curated aliases)."""
    results, seen = [], set()
    for src in sources:
        try:
            prov = get_provider(_fake_ref(src))
            for r in prov.search(query, limit=limit, filters=filters):
                if r.id in seen:
                    continue
                seen.add(r.id)
                results.append(r)
        except Exception:  # noqa: BLE001 — one source failing must not kill the whole search
            continue
    return results[:limit]


class _fake_ref:
    def __init__(self, source):
        self.source = source
        self.name = self.publisher = self.tag = self.revision = self.url = None
