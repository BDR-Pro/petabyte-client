"""cache.py — a content-addressed local model cache (conceptually Git/Docker/HF).

Layout under PETABYTE_HOME (default ~/.petabyte):

    blobs/sha256/<hash>              deduplicated content store — one copy per distinct blob
    manifests/<pub>/<name>/<rev>.json   the normalized manifest we resolved
    refs/<pub>/<name>/<tag>          a small file naming the revision a tag points at
    models/<pub>/<name>/<rev>/...    the materialized tree a runtime consumes (links into blobs)
    tmp/                             .partial downloads + per-model locks

If two models/revisions share a blob (same sha256) it is stored once and every model dir links to
it. Removing a model never deletes a blob another model still references; `prune` reclaims only
blobs nothing points at.

Path safety is enforced everywhere a remote-supplied path becomes a filesystem path: a file path
from a manifest can never escape its model directory (no "..", no absolute paths, no drive letters,
no backslashes, no symlink games).
"""
import hashlib
import json
import os
import shutil


class CachePathError(ValueError):
    """A remote-supplied path tried to escape the cache."""


def default_home() -> str:
    return os.environ.get("PETABYTE_HOME") or os.path.join(os.path.expanduser("~"), ".petabyte")


def _safe_rel(rel: str) -> str:
    """Validate a manifest file path and return a normalized, forward-slash relative path that is
    guaranteed to stay inside a model directory. Raises CachePathError on anything unsafe."""
    if rel is None:
        raise CachePathError("empty path")
    p = str(rel).replace("\\", "/").strip()
    if not p or "\x00" in p:
        raise CachePathError(f"empty/invalid path: {rel!r}")
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):
        raise CachePathError(f"absolute path not allowed: {rel!r}")
    parts = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise CachePathError(f"path traversal not allowed: {rel!r}")
        parts.append(seg)
    if not parts:
        raise CachePathError(f"empty path: {rel!r}")
    return "/".join(parts)


class Cache:
    def __init__(self, home=None):
        self.home = os.path.abspath(home or default_home())
        self.blobs = os.path.join(self.home, "blobs", "sha256")
        self.manifests = os.path.join(self.home, "manifests")
        self.refs = os.path.join(self.home, "refs")
        self.models = os.path.join(self.home, "models")
        self.tmp = os.path.join(self.home, "tmp")

    def ensure(self):
        for d in (self.blobs, self.manifests, self.refs, self.models, self.tmp):
            os.makedirs(d, exist_ok=True)

    # ---- blobs ----
    def blob_path(self, sha256: str) -> str:
        sha = (sha256 or "").lower()
        if not (len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise CachePathError(f"bad sha256: {sha256!r}")
        return os.path.join(self.blobs, sha)

    def has_blob(self, sha256: str) -> bool:
        try:
            return os.path.exists(self.blob_path(sha256))
        except CachePathError:
            return False

    # ---- model dir / materialization ----
    def model_dir(self, ref, revision) -> str:
        parts = ref.slug_parts()
        rev = _safe_rel(revision or "main").replace("/", "_")
        d = os.path.join(self.models, *[_safe_rel(p) for p in parts], rev)
        # defence in depth: the resolved path must live under self.models
        if os.path.commonpath([os.path.abspath(d), self.models]) != self.models:
            raise CachePathError("model dir escaped cache root")
        return d

    def materialize(self, model_dir, rel_path, *, blob_sha=None, src_file=None):
        """Place one file into the model dir at rel_path. If blob_sha is given, link to the shared
        blob (symlink → hardlink → copy fallback). Otherwise move src_file into place. Returns the
        final absolute path."""
        rel = _safe_rel(rel_path)
        dest = os.path.join(model_dir, rel)
        # the joined path must stay within model_dir
        if os.path.commonpath([os.path.abspath(dest), os.path.abspath(model_dir)]) != os.path.abspath(model_dir):
            raise CachePathError(f"path escaped model dir: {rel_path!r}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.islink(dest) or os.path.exists(dest):
            os.remove(dest)
        if blob_sha is not None:
            blob = self.blob_path(blob_sha)
            rel_to_blob = os.path.relpath(blob, os.path.dirname(dest))
            try:
                os.symlink(rel_to_blob, dest)
            except (OSError, NotImplementedError):
                try:
                    os.link(blob, dest)
                except OSError:
                    shutil.copy2(blob, dest)
        elif src_file is not None:
            shutil.move(src_file, dest)
        return dest

    # ---- manifests + refs ----
    def _manifest_path(self, ref, revision) -> str:
        parts = [_safe_rel(p) for p in ref.slug_parts()]
        rev = _safe_rel(revision).replace("/", "_")
        return os.path.join(self.manifests, *parts, rev + ".json")

    def write_manifest(self, manifest):
        path = self._manifest_path_from_manifest(manifest)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(manifest.to_json())
        os.replace(tmp, path)

    def _manifest_path_from_manifest(self, manifest):
        pub = manifest.publisher
        name = manifest.name or manifest.id.split("/")[-1]
        parts = [_safe_rel(x) for x in ([pub] if pub else []) + [name]]
        rev = _safe_rel(manifest.revision).replace("/", "_")
        return os.path.join(self.manifests, *parts, rev + ".json")

    def read_manifest(self, ref, revision):
        from .manifest import Manifest
        path = self._manifest_path(ref, revision)
        if not os.path.exists(path):
            return None
        return Manifest.from_dict(json.load(open(path)))

    def write_ref(self, ref, revision):
        parts = [_safe_rel(p) for p in ref.slug_parts()]
        tag = _safe_rel(ref.tag or "latest")
        d = os.path.join(self.refs, *parts)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, tag), "w") as f:
            f.write(revision)

    def read_ref(self, ref):
        parts = [_safe_rel(p) for p in ref.slug_parts()]
        tag = _safe_rel(ref.tag or "latest")
        p = os.path.join(self.refs, *parts, tag)
        return open(p).read().strip() if os.path.exists(p) else None

    # ---- listing / inspection ----
    def installed(self) -> list:
        """Every installed (materialized + manifested) model revision."""
        from .manifest import Manifest
        out = []
        for root, _dirs, files in os.walk(self.manifests):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                try:
                    m = Manifest.from_dict(json.load(open(os.path.join(root, fn))))
                except Exception:  # noqa: BLE001
                    continue
                from .ids import ModelRef
                ref = ModelRef(m.publisher, m.name or m.id.split("/")[-1], source=m.source)
                mdir = self.model_dir(ref, m.revision)
                out.append({"id": m.id, "revision": m.revision, "source": m.source,
                            "format": m.format, "parameters": m.parameters,
                            "total_size": m.total_size, "license": m.license,
                            "path": mdir, "installed": os.path.isdir(mdir),
                            "requirements": m.requirements})
        out.sort(key=lambda x: x["id"])
        return out

    def _referenced_shas(self) -> set:
        """sha256 of every blob referenced by a manifest whose model dir still exists (live)."""
        from .manifest import Manifest
        from .ids import ModelRef
        live = set()
        for root, _dirs, files in os.walk(self.manifests):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                try:
                    m = Manifest.from_dict(json.load(open(os.path.join(root, fn))))
                except Exception:  # noqa: BLE001
                    continue
                ref = ModelRef(m.publisher, m.name or m.id.split("/")[-1], source=m.source)
                if os.path.isdir(self.model_dir(ref, m.revision)):
                    live.update(f.sha256 for f in m.files if f.sha256)
        return live

    def status(self) -> dict:
        total, blob_count, blob_bytes, reclaimable = 0, 0, 0, 0
        referenced = self._referenced_shas()
        ref_count = {}
        # count how many live manifests reference each sha (for "shared")
        from .manifest import Manifest
        from .ids import ModelRef
        for root, _dirs, files in os.walk(self.manifests):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                try:
                    m = Manifest.from_dict(json.load(open(os.path.join(root, fn))))
                except Exception:  # noqa: BLE001
                    continue
                r = ModelRef(m.publisher, m.name or m.id.split("/")[-1], source=m.source)
                if not os.path.isdir(self.model_dir(r, m.revision)):
                    continue
                for f in m.files:
                    if f.sha256:
                        ref_count[f.sha256] = ref_count.get(f.sha256, 0) + 1
        shared = 0
        if os.path.isdir(self.blobs):
            for fn in os.listdir(self.blobs):
                fp = os.path.join(self.blobs, fn)
                if not os.path.isfile(fp):
                    continue
                sz = os.path.getsize(fp)
                blob_count += 1
                blob_bytes += sz
                total += sz
                if fn not in referenced:
                    reclaimable += sz
                elif ref_count.get(fn, 0) > 1:
                    shared += 1
        # non-blob model files (configs/tokenizers stored directly)
        model_extra = 0
        for root, _dirs, files in os.walk(self.models):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.islink(fp):
                    continue
                try:
                    model_extra += os.path.getsize(fp)
                except OSError:
                    pass
        total += model_extra
        installed = self.installed()
        return {"home": self.home, "total_bytes": total, "blob_bytes": blob_bytes,
                "blob_count": blob_count, "shared_blobs": shared,
                "reclaimable_bytes": reclaimable, "model_extra_bytes": model_extra,
                "models": len(installed), "installed": installed}

    def prune(self, dry_run=False) -> dict:
        """Delete blobs nothing references. Never touches a referenced blob."""
        referenced = self._referenced_shas()
        removed, freed = [], 0
        if os.path.isdir(self.blobs):
            for fn in list(os.listdir(self.blobs)):
                fp = os.path.join(self.blobs, fn)
                if not os.path.isfile(fp) or fn in referenced:
                    continue
                freed += os.path.getsize(fp)
                removed.append(fn)
                if not dry_run:
                    os.remove(fp)
        return {"removed": removed, "removed_count": len(removed), "freed_bytes": freed,
                "dry_run": dry_run}

    def remove(self, ref, revision) -> dict:
        """Remove one model revision (model dir + manifest + tag ref). Blobs are left for prune."""
        removed = []
        mdir = self.model_dir(ref, revision)
        if os.path.isdir(mdir):
            shutil.rmtree(mdir)
            removed.append("model")
        mpath = self._manifest_path(ref, revision)
        if os.path.exists(mpath):
            os.remove(mpath)
            removed.append("manifest")
        # drop any tag ref pointing at this revision
        parts = [_safe_rel(p) for p in ref.slug_parts()]
        rdir = os.path.join(self.refs, *parts)
        if os.path.isdir(rdir):
            for tag in os.listdir(rdir):
                tp = os.path.join(rdir, tag)
                try:
                    if open(tp).read().strip() == revision:
                        os.remove(tp)
                        removed.append(f"ref:{tag}")
                except OSError:
                    pass
        return {"removed": removed, "id": ref.id, "revision": revision}

    @staticmethod
    def sha256_file(path, chunk=1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
