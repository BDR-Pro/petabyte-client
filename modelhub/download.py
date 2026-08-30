"""download.py — treat downloading as infrastructure, not an HTTP GET.

Models are tens to hundreds of GB. A user who reaches 98% and loses their connection must NOT start
over. This module gives:
  * HTTP range requests + resume from a .partial file (re-hashing what's already on disk)
  * retry with exponential backoff on connection loss, 429 (honouring Retry-After), and 5xx
  * streaming sha256 verification + corrupted-file detection
  * atomic finalization (rename only after the bytes verify)
  * disk-space preflight and ENOSPC handling
  * safe cancellation via a threading.Event

Pure stdlib (urllib) so it adds no dependency and its behaviour is fully controllable in tests.
"""
import errno
import hashlib
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


class DownloadError(Exception):
    pass


class AuthError(DownloadError):
    """401/403 — gated or private model, or a bad/absent token."""


class NotFoundError(DownloadError):
    pass


class ChecksumError(DownloadError):
    pass


class DiskFullError(DownloadError):
    pass


class Canceled(DownloadError):
    pass


_TRANSIENT = (500, 502, 503, 504)
DEFAULT_CHUNK = 1024 * 1024


def _retry_after(hdrs, default):
    try:
        ra = hdrs.get("Retry-After") if hdrs else None
        return max(0.0, float(ra)) if ra and str(ra).strip().isdigit() else default
    except Exception:  # noqa: BLE001
        return default


def _free_bytes(path):
    import shutil
    try:
        return shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".").free
    except Exception:  # noqa: BLE001
        return None


def download_to(url, dest, *, expected_sha256=None, expected_size=None, headers=None,
                on_progress=None, max_retries=5, chunk=DEFAULT_CHUNK, cancel=None,
                timeout=30, disk_margin=64 * 1024 * 1024, backoff_base=0.5, backoff_cap=30.0,
                sleep=time.sleep):
    """Download `url` to `dest`, resumably and verified. Returns a small result dict.

    `on_progress(downloaded, total)` is called as bytes land. `cancel` is a threading.Event-like
    object with .is_set(); when set mid-flight we stop cleanly and leave the .partial for resume."""
    headers = dict(headers or {})
    # Reject non-http(s) schemes: a remote manifest controls file URLs, so this stops a
    # file:///etc/passwd or gopher:// entry from being fetched into the cache. (Server-side host/IP
    # egress is restricted upstream by the models_routes source allowlist.)
    _scheme = (urllib.parse.urlparse(url).scheme or "").lower()
    if _scheme not in ("http", "https"):
        raise DownloadError(f"unsupported URL scheme {_scheme!r} (only http/https allowed)")
    part = dest + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)

    # already finished (idempotent / duplicate-pull safe)
    if os.path.exists(dest) and (expected_size is None or os.path.getsize(dest) == expected_size):
        if expected_sha256 and _sha_of(dest) != expected_sha256:
            os.remove(dest)                                   # corrupted cache entry -> refetch
        else:
            total = os.path.getsize(dest)
            if on_progress:
                on_progress(total, total)
            return {"path": dest, "size": total, "resumed": False, "cache_hit": True,
                    "verified": bool(expected_sha256)}

    have = os.path.getsize(part) if os.path.exists(part) else 0
    hasher = hashlib.sha256()
    if have:
        # resume: fold the bytes already on disk into the running hash
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                hasher.update(block)
        have = os.path.getsize(part)

    attempt = 0
    while True:
        if cancel is not None and cancel.is_set():
            raise Canceled("download cancelled")
        req_headers = dict(headers)
        if have:
            req_headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=req_headers)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            status = resp.getcode()
            total = expected_size
            clen = resp.headers.get("Content-Length")
            crange = resp.headers.get("Content-Range")
            if crange and "/" in crange:
                try:
                    total = int(crange.rsplit("/", 1)[1])
                except ValueError:
                    pass
            elif clen and clen.isdigit():
                total = have + int(clen) if status == 206 else int(clen)

            if have and status == 200:
                # server ignored our Range — restart cleanly from zero
                have = 0
                hasher = hashlib.sha256()
                mode = "wb"
            elif status in (200, 206):
                mode = "ab" if (have and status == 206) else "wb"
                if mode == "wb":
                    have = 0
                    hasher = hashlib.sha256()
            elif status == 416:
                # requested range beyond EOF: the .partial is already complete
                resp.close()
                total = have
                break
            else:
                resp.close()
                raise DownloadError(f"unexpected status {status} for {url}")

            # disk preflight now that we know the size
            if total is not None:
                free = _free_bytes(dest)
                remaining = max(0, total - have)
                if free is not None and free < remaining + disk_margin:
                    resp.close()
                    raise DiskFullError(
                        f"need ~{remaining} bytes free, only {free} available")

            with open(part, mode) as out:
                while True:
                    if cancel is not None and cancel.is_set():
                        resp.close()
                        raise Canceled("download cancelled")
                    block = resp.read(chunk)
                    if not block:
                        break
                    try:
                        out.write(block)
                    except OSError as e:
                        if e.errno == errno.ENOSPC:
                            raise DiskFullError("no space left on device") from e
                        raise
                    hasher.update(block)
                    have += len(block)
                    if on_progress:
                        on_progress(have, total if total is not None else have)
            resp.close()

            if total is not None and have < total:
                raise urllib.error.URLError(f"short read: {have}/{total}")   # -> retry/resume
            break

        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthError(f"access denied ({e.code}) for {url}") from e
            if e.code == 404:
                raise NotFoundError(f"not found (404): {url}") from e
            if e.code == 416:
                total = have
                break
            if e.code == 429 or e.code in _TRANSIENT:
                attempt += 1
                if attempt > max_retries:
                    raise DownloadError(f"giving up after {max_retries} retries ({e.code})") from e
                wait = _retry_after(getattr(e, "headers", None),
                                    min(backoff_cap, backoff_base * (2 ** (attempt - 1))))
                have = os.path.getsize(part) if os.path.exists(part) else have
                sleep(wait)
                continue
            raise DownloadError(f"HTTP {e.code} for {url}") from e
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                raise DownloadError(f"network error after {max_retries} retries: {e}") from e
            have = os.path.getsize(part) if os.path.exists(part) else have
            sleep(min(backoff_cap, backoff_base * (2 ** (attempt - 1))))
            continue

    # verify + atomically finalize
    if expected_size is not None and os.path.getsize(part) != expected_size:
        raise ChecksumError(f"size mismatch: got {os.path.getsize(part)}, want {expected_size}")
    digest = hasher.hexdigest()
    if expected_sha256 and digest != expected_sha256:
        os.remove(part)
        raise ChecksumError(f"sha256 mismatch for {os.path.basename(dest)}")
    os.replace(part, dest)
    size = os.path.getsize(dest)
    if on_progress:
        on_progress(size, size)
    return {"path": dest, "size": size, "sha256": digest, "resumed": have != size,
            "cache_hit": False, "verified": bool(expected_sha256)}


def _sha_of(path, chunk=DEFAULT_CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
