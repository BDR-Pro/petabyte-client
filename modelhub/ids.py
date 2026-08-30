"""ids.py — Petabyte model identity.

One clean identifier the user types; everything downstream (providers, cache paths, manifests)
resolves from it. A user should never have to know a storage URL or an individual .safetensors file.

Grammar (all optional parts really optional):

    [source:]publisher/model[:tag][@revision]
    [source:]alias[:tag]                       # short alias, no publisher (Petabyte registry)

Examples:

    meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-8B:latest
    mistralai/Mistral-7B-Instruct-v0.3@e0bc86c
    hf:meta-llama/Llama-3.1-8B-Instruct
    llama3.1:8b                                # alias -> resolved by the registry provider
    https://mirror.example.com/models/foo.json # a direct manifest URL (source inferred)

`source` selects the provider ("hf" is the default). The parser is deliberately strict about the
charset so an id can be turned into a filesystem path with no traversal risk — see cache.py.
"""
import re

# publisher / model / tag / alias components. Conservative: letters, digits, dot, underscore, dash.
# (Hugging Face ids use exactly this set.) No slashes inside a component, no "..", no leading dot.
_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_COMPONENT_RE = re.compile(r"^" + _COMPONENT + r"$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")   # a git ref or commit sha
KNOWN_SOURCES = ("hf", "huggingface", "http", "https", "pt", "petabyte", "s3", "file")


class ModelIdError(ValueError):
    """Raised when a model identifier is malformed or unsafe."""


def _valid_component(s: str) -> bool:
    return bool(s) and s != "." and s != ".." and ".." not in s and _COMPONENT_RE.match(s) is not None


class ModelRef:
    """A parsed, validated model reference. Immutable-ish value object."""

    __slots__ = ("source", "publisher", "name", "tag", "revision", "url")

    def __init__(self, publisher, name, *, tag=None, revision=None, source="hf", url=None):
        self.source = source
        self.publisher = publisher            # may be None for a bare alias
        self.name = name
        self.tag = tag
        self.revision = revision
        self.url = url                        # set only for direct-URL refs

    # ---- identity ----
    @property
    def id(self) -> str:
        """Canonical publisher/name (or bare name for an alias). No tag/revision."""
        return f"{self.publisher}/{self.name}" if self.publisher else self.name

    def with_revision(self, revision: str) -> "ModelRef":
        return ModelRef(self.publisher, self.name, tag=self.tag, revision=revision,
                        source=self.source, url=self.url)

    def slug_parts(self):
        """Filesystem-safe path components (already charset-validated). Used by the cache."""
        return [p for p in ([self.publisher] if self.publisher else []) + [self.name] if p]

    def __str__(self):
        s = self.id
        if self.tag:
            s += ":" + self.tag
        if self.revision:
            s += "@" + self.revision
        return s

    def __repr__(self):
        return f"ModelRef({self.source}:{self!s})"

    def __eq__(self, other):
        return isinstance(other, ModelRef) and str(self) == str(other) and self.source == other.source

    def __hash__(self):
        return hash((self.source, str(self)))


def parse(identifier: str, *, default_source: str = "hf") -> ModelRef:
    """Parse a user-typed identifier into a validated ModelRef. Raises ModelIdError on anything
    malformed or unsafe (traversal, empty component, bad charset)."""
    if not identifier or not isinstance(identifier, str):
        raise ModelIdError("empty model identifier")
    raw = identifier.strip()
    if not raw:
        raise ModelIdError("empty model identifier")

    # A full URL is a direct-manifest / direct-repo reference.
    if raw.startswith(("http://", "https://")):
        source = "https" if raw.startswith("https://") else "http"
        # derive a stable-ish name from the last path segment for cache pathing
        tail = raw.rstrip("/").split("/")[-1] or "model"
        tail = re.sub(r"[^A-Za-z0-9._-]", "-", tail)
        return ModelRef(None, tail or "model", source=source, url=raw)

    source = default_source
    body = raw
    # explicit source prefix "hf:...", "https:..." only when it is a KNOWN source (so a bare
    # "publisher/model" is never mistaken for a scheme).
    if ":" in raw:
        head = raw.split(":", 1)[0].lower()
        if head in KNOWN_SOURCES and not (head in ("http", "https")):
            source, body = head, raw.split(":", 1)[1]

    # split off @revision
    revision = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)
        if not _REVISION_RE.match(revision or "") or ".." in revision:
            raise ModelIdError(f"invalid revision: {revision!r}")

    # split off :tag (tag has no slash)
    tag = None
    if ":" in body:
        body, tag = body.rsplit(":", 1)
        if not _valid_component(tag):
            raise ModelIdError(f"invalid tag: {tag!r}")

    body = body.strip("/")
    if "/" in body:
        parts = body.split("/")
        if len(parts) != 2:
            raise ModelIdError(f"model id must be 'publisher/model', got {body!r}")
        publisher, name = parts
        if not _valid_component(publisher):
            raise ModelIdError(f"invalid publisher: {publisher!r}")
        if not _valid_component(name):
            raise ModelIdError(f"invalid model name: {name!r}")
        return ModelRef(publisher, name, tag=tag, revision=revision, source=source)

    # bare alias (no publisher) — resolved by the registry provider
    if not _valid_component(body):
        raise ModelIdError(f"invalid model name: {body!r}")
    return ModelRef(None, body, tag=tag, revision=revision,
                    source=("pt" if source == "hf" else source))
