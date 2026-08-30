"""manager.py — the high-level model manager. One object the CLI, the API, and the agent all use.

It orchestrates: resolve a ref -> normalized manifest -> hardware/disk preflight -> download only the
blobs we don't already have (content-addressed dedup + resume) -> verify -> materialize a local model
dir a runtime can consume. Plus list / inspect / remove / cache status+prune.

Progress is surfaced through an optional `on_event(dict)` callback so both a live CLI progress bar and
a web download-job poller can render the same underlying events.
"""
import os
import time

from .ids import parse, ModelRef
from .cache import Cache
from .hardware import detect, compatibility
from .download import download_to, DownloadError
from .providers import get_provider, search_all


class CompatibilityError(Exception):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


class ConcurrentPullError(Exception):
    pass


class ModelManager:
    def __init__(self, home=None, provider_factory=get_provider):
        self.cache = Cache(home)
        self.cache.ensure()
        self._provider_for = provider_factory

    # ---- discovery ----
    def search(self, query, *, limit=25, filters=None, sources=("hf",)):
        return [r.to_dict() for r in search_all(query, limit=limit, filters=filters, sources=sources)]

    def _ref(self, ref_or_str):
        return ref_or_str if isinstance(ref_or_str, ModelRef) else parse(ref_or_str)

    def resolve_manifest(self, ref_or_str, *, fmt=None, quantization=None, revision=None):
        ref = self._ref(ref_or_str)
        if revision:
            ref = ref.with_revision(revision)
        prov = self._provider_for(ref)
        try:
            return prov.manifest(ref, fmt=fmt, quantization=quantization)
        except TypeError:
            return prov.manifest(ref)   # providers that don't take fmt kwargs

    def info(self, ref_or_str, *, fmt=None, quantization=None, revision=None, hw=None):
        m = self.resolve_manifest(ref_or_str, fmt=fmt, quantization=quantization, revision=revision)
        return {"manifest": m.to_dict(),
                "compatibility": compatibility(m, hw, cache_home=self.cache.home)}

    # ---- pull ----
    def pull(self, ref_or_str, *, fmt=None, quantization=None, revision=None, force=False,
             on_event=None, jobs=1, cancel=None):
        ref = self._ref(ref_or_str)
        _emit(on_event, {"event": "resolving", "id": ref.id})
        manifest = self.resolve_manifest(ref, fmt=fmt, quantization=quantization, revision=revision)
        ref = ref.with_revision(manifest.revision)
        _emit(on_event, {"event": "manifest", "id": manifest.id, "revision": manifest.revision,
                         "format": manifest.format, "files": len(manifest.files),
                         "total_size": manifest.total_size, "trust": manifest.trust})

        # hardware compatibility — warn/allow, block only the clearly-impossible unless --force
        report = compatibility(manifest, cache_home=self.cache.home)
        _emit(on_event, {"event": "compatibility", **report})
        if report["level"] == "insufficient" and not force:
            raise CompatibilityError(
                f"{manifest.id} needs ~{report['need'].get('vram_gb')} GB VRAM; "
                f"this machine can't run it. Re-run with force to download anyway.", report)

        model_dir = self.cache.model_dir(ref, manifest.revision)
        os.makedirs(model_dir, exist_ok=True)
        lock = _Lock(model_dir)
        if not lock.acquire():
            raise ConcurrentPullError(f"another pull of {manifest.id} is already running")
        try:
            return self._download_all(ref, manifest, model_dir, on_event, cancel)
        finally:
            lock.release()

    def _download_all(self, ref, manifest, model_dir, on_event, cancel):
        total = manifest.total_size or sum(f.size for f in manifest.files)
        overall = {"done": 0}
        cache_hits = 0
        for f in manifest.files:
            if cancel is not None and cancel.is_set():
                from .download import Canceled
                raise Canceled("pull cancelled")
            _emit(on_event, {"event": "file_start", "file": f.path, "size": f.size})

            def prog(dl, tot, _f=f):
                base = overall["done"]
                _emit(on_event, {"event": "file_progress", "file": _f.path,
                                 "downloaded": dl, "total": tot or _f.size,
                                 "overall_downloaded": base + dl, "overall_total": total})

            headers = self._provider_for(ref).auth_headers()
            if f.sha256:
                blob = self.cache.blob_path(f.sha256)
                # Cheap integrity guard: a content-addressed blob is verified when written, but a
                # truncated/corrupted file (wrong size) is caught here and re-fetched — without
                # re-hashing every GB on every pull.
                healthy = self.cache.has_blob(f.sha256) and (
                    not f.size or os.path.getsize(blob) == f.size)
                if healthy:
                    cache_hits += 1
                    prog(f.size, f.size)
                    _emit(on_event, {"event": "file_done", "file": f.path, "cache_hit": True})
                else:
                    if os.path.exists(blob):
                        os.remove(blob)           # drop the corrupt blob, then re-download + verify
                    _download_one(f.url, blob, f, headers, prog, cancel)
                    _emit(on_event, {"event": "file_done", "file": f.path, "cache_hit": False})
                self.cache.materialize(model_dir, f.path, blob_sha=f.sha256)
            else:
                # no publisher hash (usually tiny config/tokenizer) — fetch straight into place
                dest = os.path.join(model_dir, *_relparts(f.path))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                res = _download_one(f.url, dest, f, headers, prog, cancel, to_blob=False)
                _emit(on_event, {"event": "file_done", "file": f.path,
                                 "cache_hit": res.get("cache_hit", False)})
            overall["done"] += f.size

        self.cache.write_manifest(manifest)
        self.cache.write_ref(ref, manifest.revision)
        record = {"id": manifest.id, "revision": manifest.revision, "path": model_dir,
                  "total_size": manifest.total_size, "files": len(manifest.files),
                  "cache_hits": cache_hits, "format": manifest.format,
                  "verified": manifest.trust.get("hashed", False)}
        _emit(on_event, {"event": "done", **record})
        return record

    # ---- listing / inspection / removal ----
    def list_installed(self):
        return self.cache.installed()

    def inspect(self, ref_or_str, *, revision=None):
        ref = self._ref(ref_or_str)
        rev = revision or ref.revision or self.cache.read_ref(ref) or self._only_installed_rev(ref)
        if not rev:
            return None
        m = self.cache.read_manifest(ref, rev)
        if not m:
            return None
        mdir = self.cache.model_dir(ref, rev)
        files = []
        for f in m.files:
            fp = os.path.join(mdir, *_relparts(f.path))
            files.append({"path": f.path, "size": f.size, "sha256": f.sha256,
                          "present": os.path.exists(fp),
                          "shared": bool(f.sha256 and self.cache.has_blob(f.sha256))})
        return {"manifest": m.to_dict(), "path": mdir, "installed": os.path.isdir(mdir),
                "files": files}

    def local_path(self, ref_or_str, *, revision=None):
        """The materialized dir a runtime consumes — or None if not installed."""
        ref = self._ref(ref_or_str)
        rev = revision or ref.revision or self.cache.read_ref(ref) or self._only_installed_rev(ref)
        if not rev:
            return None
        d = self.cache.model_dir(ref, rev)
        return d if os.path.isdir(d) else None

    def is_installed(self, ref_or_str, *, revision=None):
        return self.local_path(ref_or_str, revision=revision) is not None

    def remove(self, ref_or_str, *, revision=None):
        ref = self._ref(ref_or_str)
        rev = revision or ref.revision or self.cache.read_ref(ref) or self._only_installed_rev(ref)
        if not rev:
            return {"removed": [], "id": ref.id, "revision": None}
        return self.cache.remove(ref, rev)

    def _only_installed_rev(self, ref):
        revs = [i["revision"] for i in self.cache.installed() if i["id"] == ref.id]
        return revs[0] if len(revs) == 1 else None

    # ---- cache ----
    def cache_status(self):
        return self.cache.status()

    def cache_prune(self, dry_run=False):
        return self.cache.prune(dry_run=dry_run)

    def cached_model_ids(self):
        """For the marketplace cache-locality signal: which model ids this node holds."""
        return sorted({i["id"] for i in self.cache.installed() if i["installed"]})


def _download_one(url, dest, f, headers, prog, cancel, to_blob=True):
    try:
        return download_to(url, dest, expected_sha256=f.sha256,
                           expected_size=(f.size or None), headers=headers,
                           on_progress=prog, cancel=cancel)
    except DownloadError:
        raise


def _relparts(path):
    from .cache import _safe_rel
    return _safe_rel(path).split("/")


def _emit(cb, ev):
    if cb:
        try:
            cb(ev)
        except Exception:  # noqa: BLE001 — a progress sink must never break a download
            pass


class _Lock:
    """Atomic per-model-dir lock (mkdir is atomic) to stop duplicate concurrent pulls. A lock older
    than STALE seconds is assumed abandoned (crashed process) and reclaimed."""
    STALE = 6 * 3600

    def __init__(self, model_dir):
        self.path = os.path.join(model_dir, ".pull.lock")

    def acquire(self):
        try:
            os.mkdir(self.path)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(self.path) > self.STALE:
                    os.rmdir(self.path)
                    os.mkdir(self.path)
                    return True
            except OSError:
                pass
            return False

    def release(self):
        try:
            os.rmdir(self.path)
        except OSError:
            pass
