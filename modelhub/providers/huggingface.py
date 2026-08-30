"""huggingface.py — Hugging Face as a first-class source (not a shell around huggingface-cli).

We talk to HF's public HTTP API directly (stdlib urllib) and own the higher-level UX: file
selection, normalized manifests, hash verification via LFS oids, and understandable gated-model
errors. The endpoint is configurable (HF_ENDPOINT) so tests can point it at a local server.

Auth for gated/private models is read from the environment or ~/.petabyte/auth.json and is NEVER
printed or logged.
"""
import json
import os
import re

from .base import ModelProvider, SearchResult, ProviderError, GatedError, get_json
from ..manifest import Manifest, ModelFile

DEFAULT_ENDPOINT = "https://huggingface.co"

# Files a runtime always needs alongside the weights (tokenizer + config), regardless of format.
_ALWAYS = re.compile(
    r"(^|/)(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|"
    r"special_tokens_map\.json|vocab\.json|merges\.txt|preprocessor_config\.json|"
    r"chat_template\.jinja|.*\.model)$")
_WEIGHT_EXT = {"safetensors": (".safetensors",), "gguf": (".gguf",),
               "pytorch": (".bin",), "onnx": (".onnx",)}


def endpoint():
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def hf_token():
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or os.environ.get("HUGGINGFACE_TOKEN"))
    if tok:
        return tok.strip()
    try:
        from ..cache import default_home
        p = os.path.join(default_home(), "auth.json")
        if os.path.exists(p):
            return (json.load(open(p)).get("huggingface") or "").strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


def set_hf_token(token):
    """Persist a HF token to ~/.petabyte/auth.json (0600). Never echoed back."""
    from ..cache import default_home
    home = default_home()
    os.makedirs(home, exist_ok=True)
    p = os.path.join(home, "auth.json")
    data = {}
    if os.path.exists(p):
        try:
            data = json.load(open(p))
        except Exception:  # noqa: BLE001
            data = {}
    data["huggingface"] = token.strip()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)


def _available_formats(paths):
    fmts = []
    for fmt, exts in _WEIGHT_EXT.items():
        if any(p.endswith(exts) for p in paths):
            fmts.append(fmt)
    return fmts


def select_files(entries, fmt=None, quantization=None):
    """Choose the minimal file set for a format. `entries` is [{path,size,sha256}]. Returns
    (files, chosen_format, available_formats). Prefers safetensors; never mixes weight formats."""
    paths = [e["path"] for e in entries]
    available = _available_formats(paths)
    chosen = fmt
    if not chosen:
        for pref in ("safetensors", "gguf", "onnx", "pytorch"):
            if pref in available:
                chosen = pref
                break
    if chosen and chosen not in available:
        raise ProviderError(
            f"format {chosen!r} not available; this model offers: {', '.join(available) or 'none'}")
    exts = _WEIGHT_EXT.get(chosen, ())
    picked = []
    for e in entries:
        p = e["path"]
        keep = False
        if _ALWAYS.search(p):
            keep = True
        elif exts and p.endswith(exts):
            if chosen == "gguf" and quantization:
                keep = quantization.lower() in p.lower()
            else:
                keep = True
        # safetensors shard index
        if chosen == "safetensors" and p.endswith(".safetensors.index.json"):
            keep = True
        if keep:
            picked.append(e)
    # For gguf with a quant filter that matched nothing, surface it rather than silently pulling all.
    if chosen == "gguf" and quantization and not any(p["path"].endswith(".gguf") for p in picked):
        raise ProviderError(f"no gguf file matching quantization {quantization!r}")
    return picked, chosen, available


class HuggingFaceProvider(ModelProvider):
    name = "hf"

    def __init__(self, endpoint_url=None, token=None):
        self._endpoint = (endpoint_url or endpoint()).rstrip("/")
        self._token = token

    def _headers(self):
        tok = self._token or hf_token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def auth_headers(self):
        return self._headers()

    # ---- search ----
    def search(self, query, *, limit=25, filters=None):
        filters = filters or {}
        url = f"{self._endpoint}/api/models?search={_q(query)}&limit={int(limit)}&full=true&config=true"
        rows = get_json(url, self._headers()) or []
        out = []
        for r in rows:
            mid = r.get("id") or r.get("modelId")
            if not mid:
                continue
            tags = r.get("tags") or []
            lic = _license_from(r)
            arch = _arch_from(r)
            params = ((r.get("safetensors") or {}).get("total")) or 0
            out.append(SearchResult(
                mid, "hf", parameters=params, license=lic, architecture=arch,
                downloads=r.get("downloads", 0), likes=r.get("likes", 0),
                pipeline=r.get("pipeline_tag"), gated=bool(r.get("gated")),
                updated=r.get("lastModified")))
        # client-side filters the API doesn't take directly
        return _apply_filters(out, filters)

    # ---- resolve ----
    def resolve(self, ref):
        if ref.revision:
            return ref.revision
        rev = ref.tag or "main"
        info = self._model_info(ref.id, rev)
        return info.get("sha") or rev

    # ---- manifest ----
    def manifest(self, ref, *, fmt=None, quantization=None):
        rev = self.resolve(ref)
        info = self._model_info(ref.id, rev)
        if _is_gated(info) and not (self._token or hf_token()):
            raise GatedError(
                "gated model — accept its license on Hugging Face, then authenticate",
                model_id=ref.id, url=f"{self._endpoint}/{ref.id}")
        entries = self._tree(ref.id, rev)
        picked, chosen, available = select_files(entries, fmt=fmt, quantization=quantization)
        files = [ModelFile(e["path"], e.get("size", 0), e.get("sha256"),
                           url=f"{self._endpoint}/{ref.id}/resolve/{rev}/{_urlpath(e['path'])}")
                 for e in picked]
        cfg = info.get("config") or {}
        params = ((info.get("safetensors") or {}).get("total")) or 0
        pub, _, name = ref.id.partition("/")
        return Manifest(
            id=ref.id, revision=rev, source="hf", publisher=pub or None, name=name or ref.id,
            architecture=_arch_from(info), parameters=params,
            context_length=cfg.get("max_position_embeddings"),
            tokenizer=("tokenizer.json" if any(f.path.endswith("tokenizer.json") for f in files) else None),
            format=chosen, formats=available, quantization=quantization,
            files=files, license=_license_from(info), gated=_is_gated(info),
            extra={"pipeline": info.get("pipeline_tag"), "downloads": info.get("downloads", 0),
                   "likes": info.get("likes", 0), "available_formats": available})

    # ---- HF API calls ----
    def _model_info(self, model_id, rev):
        url = f"{self._endpoint}/api/models/{model_id}"
        if rev:
            url += f"/revision/{_urlpath(rev)}"
        try:
            return get_json(url, self._headers()) or {}
        except GatedError as e:
            raise GatedError("gated or private model — authentication required",
                             model_id=model_id, url=f"{self._endpoint}/{model_id}") from e

    def _tree(self, model_id, rev):
        url = f"{self._endpoint}/api/models/{model_id}/tree/{_urlpath(rev)}?recursive=true"
        raw = get_json(url, self._headers()) or []
        out = []
        for e in raw:
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            out.append({"path": e["path"],
                        "size": lfs.get("size") or e.get("size") or 0,
                        # LFS oid IS the sha256 of the content; a plain git oid is not, so skip it.
                        "sha256": (lfs.get("oid") if lfs else None)})
        if not out:
            raise ProviderError(f"no files found for {model_id}@{rev}")
        return out


def _q(s):
    from urllib.parse import quote
    return quote(str(s or ""))


def _urlpath(s):
    from urllib.parse import quote
    return quote(str(s), safe="/")


def _license_from(info):
    cd = info.get("cardData") or {}
    if cd.get("license"):
        return cd["license"]
    for t in info.get("tags") or []:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def _arch_from(info):
    cfg = info.get("config") or {}
    arch = cfg.get("architectures")
    if isinstance(arch, list) and arch:
        return arch[0]
    return cfg.get("model_type") or info.get("pipeline_tag")


def _is_gated(info):
    g = info.get("gated")
    return g in (True, "auto", "manual")


def _apply_filters(rows, filters):
    def ok(r):
        if filters.get("license") and (r.license or "").lower() != str(filters["license"]).lower():
            return False
        if filters.get("architecture") and (r.architecture or "").lower() != str(filters["architecture"]).lower():
            return False
        if filters.get("max_params") and r.parameters and r.parameters > int(filters["max_params"]):
            return False
        return True
    return [r for r in rows if ok(r)]
