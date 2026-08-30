"""base.py — the provider abstraction the rest of Petabyte codes against.

The public API is deliberately tiny: search / resolve / manifest / (per-file) download URLs. Nothing
outside a provider should know whether a model came from Hugging Face, a mirror, or S3.
"""
import json
import urllib.error
import urllib.parse
import urllib.request


def require_http_url(url):
    """Reject non-http(s) URL schemes (file:, ftp:, gopher:, data:, ...). A remote manifest can name
    a file's URL, so this stops a `file:///etc/passwd` / `gopher://` fetch from ever leaving urllib.
    (Host/IP egress restriction for the SERVER is enforced upstream via the source allowlist in
    models_routes; the CLI legitimately reaches arbitrary public hosts.)"""
    scheme = (urllib.parse.urlparse(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ProviderError(f"unsupported URL scheme {scheme!r} (only http/https allowed)")


class ProviderError(Exception):
    pass


class GatedError(ProviderError):
    """The publisher requires accepting a license (or a token) before download."""

    def __init__(self, message, model_id=None, url=None):
        super().__init__(message)
        self.model_id = model_id
        self.url = url


class SearchResult:
    """A discovery row. Fields are best-effort — a source may not publish size or params up front."""
    __slots__ = ("id", "publisher", "name", "source", "parameters", "size", "format",
                 "license", "architecture", "downloads", "likes", "pipeline", "gated", "updated")

    def __init__(self, id, source, *, publisher=None, name=None, parameters=0, size=0,
                 format=None, license=None, architecture=None, downloads=0, likes=0,
                 pipeline=None, gated=False, updated=None):
        self.id = id
        self.publisher = publisher or (id.split("/")[0] if "/" in id else None)
        self.name = name or id.split("/")[-1]
        self.source = source
        self.parameters = int(parameters or 0)
        self.size = int(size or 0)
        self.format = format
        self.license = license
        self.architecture = architecture
        self.downloads = int(downloads or 0)
        self.likes = int(likes or 0)
        self.pipeline = pipeline
        self.gated = bool(gated)
        self.updated = updated

    def to_dict(self):
        return {"id": self.id, "publisher": self.publisher, "name": self.name,
                "source": self.source, "parameters": self.parameters, "size": self.size,
                "format": self.format, "license": self.license,
                "architecture": self.architecture, "downloads": self.downloads,
                "likes": self.likes, "pipeline": self.pipeline, "gated": self.gated,
                "updated": self.updated}


class ModelProvider:
    """Adapter interface. Concrete providers implement these four."""
    name = "base"

    def search(self, query, *, limit=25, filters=None):
        raise NotImplementedError

    def resolve(self, ref):
        """Return the concrete revision (commit sha / tag) the ref points at."""
        raise NotImplementedError

    def manifest(self, ref):
        """Return a normalized modelhub.manifest.Manifest for ref (with files + urls)."""
        raise NotImplementedError

    def auth_headers(self):
        """Per-request auth headers for downloads (never logged)."""
        return {}


# ---- shared HTTP helper (stdlib, no dependency) ----
def get_json(url, headers=None, timeout=20):
    require_http_url(url)
    req = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise GatedError(f"access denied ({e.code})", url=url) from e
        if e.code == 404:
            raise ProviderError(f"not found (404): {url}") from e
        raise ProviderError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise ProviderError(f"network error for {url}: {e}") from e
