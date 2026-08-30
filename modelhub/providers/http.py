"""http.py — a direct HTTP(S) model source (self-hosted mirrors, community drops, CI fixtures).

The reference points at a JSON manifest. Two shapes are accepted:
  * a full normalized Petabyte manifest (has "schema_version"), or
  * a simple {"id","revision"?,"files":[{"path","url","size"?,"sha256"?}]}.
Relative file URLs are resolved against the manifest URL. This is the escape hatch that keeps
Petabyte provider-independent (and it's what the hermetic tests exercise end to end).
"""
from urllib.parse import urljoin

from .base import ModelProvider, ProviderError, get_json
from ..manifest import Manifest, ModelFile


class HttpProvider(ModelProvider):
    name = "http"

    def resolve(self, ref):
        doc = self._doc(ref)
        return ref.revision or doc.get("revision") or "latest"

    def manifest(self, ref, **_):
        doc = self._doc(ref)
        base = ref.url
        if doc.get("schema_version"):
            m = Manifest.from_dict(doc)
            for f in m.files:
                if f.url:
                    f.url = urljoin(base, f.url)
            m.source = ref.source
            return m
        files = [ModelFile(f["path"], f.get("size", 0), f.get("sha256"),
                           url=urljoin(base, f.get("url") or f["path"])) for f in doc.get("files", [])]
        if not files:
            raise ProviderError(f"manifest at {base} lists no files")
        return Manifest(
            id=doc.get("id") or ref.name, revision=doc.get("revision") or "latest",
            source=ref.source, name=doc.get("name") or ref.name,
            architecture=doc.get("architecture"), parameters=doc.get("parameters", 0),
            format=doc.get("format"), files=files, license=doc.get("license"),
            trust={"source_verified": False})            # a random URL is not a verified source

    def search(self, query, *, limit=25, filters=None):
        return []                                         # arbitrary URLs aren't searchable

    def _doc(self, ref):
        if not ref.url:
            raise ProviderError("direct-http reference has no URL")
        return get_json(ref.url)
