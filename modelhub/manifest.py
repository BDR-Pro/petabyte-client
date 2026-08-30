"""manifest.py — the normalized Petabyte model manifest.

Every provider (Hugging Face, a direct mirror, the Petabyte registry) resolves a model down to ONE
shape so the rest of Petabyte — cache, downloader, hardware check, CLI, web UI, scheduler — never
has to care which backend hosted it. This is the provider-independent contract.
"""
import json

SCHEMA_VERSION = 1

# rough bytes-per-parameter by on-disk dtype, for size/requirement estimates when a provider
# doesn't give exact numbers.
_BYTES_PER_PARAM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "int8": 1.0, "q8": 1.06,
                    "q6": 0.82, "q5": 0.68, "q4": 0.56, "q3": 0.43, "q2": 0.34}


class ModelFile:
    """One artifact to fetch. `sha256` is set only when the source publishes a real content hash
    (e.g. Hugging Face LFS oids) — we verify when we can and say so when we can't."""
    __slots__ = ("path", "size", "sha256", "url")

    def __init__(self, path, size=0, sha256=None, url=None):
        self.path = path
        self.size = int(size or 0)
        self.sha256 = (sha256 or None)
        self.url = url

    def to_dict(self):
        d = {"path": self.path, "size": self.size}
        if self.sha256:
            d["sha256"] = self.sha256
        if self.url:
            d["url"] = self.url
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["path"], d.get("size", 0), d.get("sha256"), d.get("url"))


class Manifest:
    """Normalized model metadata + the exact file set to download."""

    def __init__(self, id, revision, *, source="hf", publisher=None, name=None,
                 architecture=None, parameters=0, context_length=None, tokenizer=None,
                 formats=None, format=None, quantization=None, dtype=None,
                 files=None, total_size=0, license=None, requirements=None,
                 trust=None, gated=False, extra=None):
        self.schema_version = SCHEMA_VERSION
        self.id = id
        self.revision = revision
        self.source = source
        self.publisher = publisher
        self.name = name
        self.architecture = architecture
        self.parameters = int(parameters or 0)
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.formats = list(formats or ([format] if format else []))
        self.format = format or (self.formats[0] if self.formats else None)
        self.quantization = quantization
        self.dtype = dtype
        self.files = list(files or [])
        self.total_size = int(total_size or sum(f.size for f in self.files))
        self.license = license
        self.requirements = dict(requirements or {})
        # trust/security surface: was the source itself verified, and can we hash-check the bytes?
        self.trust = dict(trust or {})
        self.trust.setdefault("source", source)
        self.trust.setdefault("source_verified", source in ("hf", "huggingface", "pt", "petabyte"))
        # "hashed" = the WEIGHT files are content-hash verifiable. Small config/tokenizer files often
        # ship without a publisher hash on HF; that shouldn't downgrade the trust of the real bytes.
        weights = [f for f in self.files if f.path.endswith((".safetensors", ".gguf", ".bin", ".onnx"))]
        self.trust.setdefault("hashed", bool(weights) and all(f.sha256 for f in weights))
        self.gated = bool(gated)
        self.extra = dict(extra or {})
        if not self.requirements:
            self.requirements = self.estimate_requirements()

    # ---- estimates ----
    def estimate_requirements(self) -> dict:
        """Best-effort RAM/VRAM/disk from parameter count + dtype/quant. Providers may override."""
        disk_gb = round((self.total_size or 0) / (1024 ** 3), 1)
        bpp = self._bytes_per_param()
        if self.parameters and bpp:
            weights_gb = self.parameters * bpp / (1024 ** 3)
        else:
            weights_gb = disk_gb or 0
        # inference working set: weights + a KV-cache/activation headroom (~20%).
        vram_gb = int(round(weights_gb * 1.2)) if weights_gb else 0
        ram_gb = max(vram_gb, int(round((disk_gb or weights_gb) * 1.1)) + 2)
        return {"disk_gb": disk_gb or int(round(weights_gb)),
                "vram_gb": vram_gb, "ram_gb": ram_gb}

    def _bytes_per_param(self) -> float:
        q = (self.quantization or "").lower()
        for key, val in _BYTES_PER_PARAM.items():
            if key in q:
                return val
        dt = (self.dtype or "").lower()
        if dt in _BYTES_PER_PARAM:
            return _BYTES_PER_PARAM[dt]
        fmt = (self.format or "").lower()
        if "gguf" in fmt:
            return _BYTES_PER_PARAM["q4"]      # gguf without a quant tag: assume ~q4
        return _BYTES_PER_PARAM["bf16"]        # modern default for safetensors

    # ---- serialization ----
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "id": self.id, "revision": self.revision,
            "source": self.source, "publisher": self.publisher, "name": self.name,
            "architecture": self.architecture, "parameters": self.parameters,
            "context_length": self.context_length, "tokenizer": self.tokenizer,
            "formats": self.formats, "format": self.format, "quantization": self.quantization,
            "dtype": self.dtype, "files": [f.to_dict() for f in self.files],
            "total_size": self.total_size, "license": self.license,
            "requirements": self.requirements, "trust": self.trust, "gated": self.gated,
            "extra": self.extra,
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d) -> "Manifest":
        return cls(
            id=d["id"], revision=d.get("revision"), source=d.get("source", "hf"),
            publisher=d.get("publisher"), name=d.get("name"),
            architecture=d.get("architecture"), parameters=d.get("parameters", 0),
            context_length=d.get("context_length"), tokenizer=d.get("tokenizer"),
            formats=d.get("formats"), format=d.get("format"),
            quantization=d.get("quantization"), dtype=d.get("dtype"),
            files=[ModelFile.from_dict(x) for x in d.get("files", [])],
            total_size=d.get("total_size", 0), license=d.get("license"),
            requirements=d.get("requirements"), trust=d.get("trust"),
            gated=d.get("gated", False), extra=d.get("extra"),
        )
