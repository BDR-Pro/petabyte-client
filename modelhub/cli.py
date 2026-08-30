"""cli.py — the `petabyte model ...` command line. Meant to feel like ollama/huggingface-cli.

Colours, tables, live progress with speed/ETA, cache-hit indication, verification status, and
useful errors. Runs locally against the modelhub library (no Petabyte server needed to pull a model,
exactly like `huggingface-cli download` or `ollama pull`).

Exposed two ways:
  * `python -m modelhub ...`            (standalone)
  * embedded into the existing `petabyte` CLI via register()/handle().
"""
import argparse
import os
import sys
import time

from . import ModelManager, parse, ModelIdError
from .manager import CompatibilityError, ConcurrentPullError
from .providers import GatedError, ProviderError
from .download import DownloadError, AuthError
from .hardware import detect
from .providers.huggingface import set_hf_token

# ---- colour (respects NO_COLOR + non-tty) ----
_TTY = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(t, code):
    return f"\033[{code}m{t}\033[0m" if _TTY else t


def bold(t): return _c(t, "1")
def dim(t): return _c(t, "2")
def green(t): return _c(t, "38;5;42")
def amber(t): return _c(t, "38;5;214")
def cyan(t): return _c(t, "38;5;44")
def red(t): return _c(t, "38;5;203")


def human(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return (f"{n:.0f} {unit}" if unit == "B" or n >= 100 else f"{n:.1f} {unit}")
        n /= 1024


def _hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}" if sec >= 3600 \
        else f"{sec // 60:02d}:{sec % 60:02d}"


def params_h(n):
    n = int(n or 0)
    if not n:
        return "—"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B" if n % 1e9 == 0 else f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


# =============================================================================== progress reporter
class Reporter:
    """Consumes ModelManager.pull events and renders live progress."""

    def __init__(self, quiet=False):
        self.quiet = quiet
        self.start = time.monotonic()
        self.cur = None
        self.width = 24
        self.last_line = ""

    def __call__(self, ev):
        e = ev.get("event")
        if e == "resolving":
            self._say(f"Resolving {bold(ev['id'])}...")
        elif e == "manifest":
            tr = ev.get("trust", {})
            vmark = green("verifiable") if tr.get("hashed") else amber("unverified bytes")
            self._say(green("✓") + f" Manifest resolved  ·  {ev['format'] or '—'}  ·  "
                      f"{ev['files']} files  ·  {human(ev['total_size'])}  ·  {vmark}")
        elif e == "compatibility":
            self._compat(ev)
            if not self.quiet:
                sys.stderr.write("\n" + bold("Downloading") + "\n")
        elif e == "file_start":
            self._flush_line()
            self.cur = {"file": ev["file"], "size": ev.get("size", 0)}
        elif e == "file_progress":
            self._bar(ev)
        elif e == "file_done":
            if ev.get("cache_hit"):
                self._flush_line()
                self._say("  " + cyan("✓ cached ") + dim(ev["file"]))
            else:
                self._finish_bar()
        elif e == "done":
            self._flush_line()

    # ---- rendering ----
    def _say(self, s):
        if not self.quiet:
            sys.stderr.write(s + "\n")
            sys.stderr.flush()

    def _compat(self, rep):
        hw = rep.get("machine", {})
        mach = hw.get("gpu_name") or f"{hw.get('cpu_count', '?')} CPU"
        vram = f" · {hw.get('vram_gb')} GB VRAM" if hw.get("vram_gb") else ""
        icon = {"good": green("✓ Good"), "tight": amber("~ Tight"),
                "insufficient": red("✗ Insufficient"), "unknown": amber("? Unknown")}.get(rep["level"], "")
        self._say(dim(f"Machine: {mach}{vram}") + f"   Compatibility: {icon}")
        for r in rep.get("reasons", [])[:2]:
            self._say(dim("  " + r))

    def _bar(self, ev):
        if self.quiet or not _TTY:
            return
        dl, tot = ev["downloaded"], ev.get("total") or 1
        frac = min(1.0, dl / tot) if tot else 0
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = max(1e-6, time.monotonic() - self.start)
        speed = ev.get("overall_downloaded", dl) / elapsed
        remain = max(0, (ev.get("overall_total", tot) - ev.get("overall_downloaded", dl)))
        eta = remain / speed if speed > 0 else 0
        name = ev["file"]
        if len(name) > 34:
            name = name[:31] + "..."
        line = (f"  {name:<34} {cyan(bar)} {int(frac * 100):3d}%  "
                f"{human(dl)}/{human(tot)}  {dim(human(speed) + '/s · ETA ' + _hms(eta))}")
        sys.stderr.write("\r" + line + "\033[K")
        sys.stderr.flush()
        self.last_line = line

    def _finish_bar(self):
        if self.quiet:
            return
        if _TTY and self.last_line:
            sys.stderr.write("\r" + self.last_line + "  " + green("✓") + "\033[K\n")
        elif self.cur:
            sys.stderr.write("  " + green("✓ ") + self.cur["file"] + "\n")
        sys.stderr.flush()
        self.last_line = ""

    def _flush_line(self):
        if self.last_line:
            sys.stderr.write("\n")
            self.last_line = ""


# =============================================================================== command handlers
def _mgr(args):
    return ModelManager(getattr(args, "home", None) or os.environ.get("PETABYTE_HOME"))


def _sources(args):
    s = ["hf"]
    if getattr(args, "source", None):
        s = [args.source]
    elif getattr(args, "aliases", False):
        s = ["pt", "hf"]
    return tuple(s)


def cmd_search(args):
    mgr = _mgr(args)
    filters = {}
    for k in ("license", "architecture"):
        if getattr(args, k, None):
            filters[k] = getattr(args, k)
    if getattr(args, "max_params", None):
        filters["max_params"] = int(args.max_params)
    try:
        rows = mgr.search(args.query, limit=args.limit, filters=filters, sources=_sources(args))
    except Exception as e:  # noqa: BLE001
        return _fail(f"search failed: {e}")
    if not rows:
        print(dim("no models found"))
        return 0
    print(dim(f"  {'MODEL':<44} {'PARAMS':>7}  {'LICENSE':<14} {'PULLS':>9}  SRC"))
    for r in rows:
        gated = amber(" (gated)") if r.get("gated") else ""
        print(f"  {bold(r['id'][:44]):<44} {params_h(r['parameters']):>7}  "
              f"{(r.get('license') or '—')[:14]:<14} {r.get('downloads', 0):>9}  {r['source']}{gated}")
    print(dim(f"\n  {len(rows)} result(s) · pull with: petabyte model pull <MODEL>"))
    return 0


def cmd_info(args):
    mgr = _mgr(args)
    try:
        info = mgr.info(args.id, fmt=args.format, quantization=args.quantization, revision=args.revision)
    except GatedError as e:
        return _gated(e)
    except (ProviderError, ModelIdError) as e:
        return _fail(str(e))
    m, comp = info["manifest"], info["compatibility"]
    print(bold(m["id"]) + dim(f"  @ {m['revision'][:12]}"))
    line = [params_h(m["parameters"]) + " params" if m["parameters"] else None,
            m.get("architecture"), m.get("format"), m.get("license"),
            human(m["total_size"]) if m["total_size"] else None]
    print("  " + dim(" · ".join(x for x in line if x)))
    if m.get("context_length"):
        print(dim(f"  context length: {m['context_length']}"))
    req = m.get("requirements", {})
    print(f"  requirements: ~{req.get('vram_gb', '?')} GB VRAM · ~{req.get('ram_gb', '?')} GB RAM · "
          f"{req.get('disk_gb', '?')} GB disk")
    icon = {"good": green("✓ runs comfortably"), "tight": amber("~ tight fit"),
            "insufficient": red("✗ insufficient"), "unknown": amber("? unknown")}.get(comp["level"], "")
    hw = comp.get("machine", {})
    mach = hw.get("gpu_name") or "CPU"
    if hw.get("vram_gb"):
        mach += f" · {hw['vram_gb']} GB VRAM"
    print(f"  your machine: {dim(mach)}  →  {icon}")
    for alt in comp.get("alternatives", [])[:3]:
        print(dim(f"    alternative: {alt['quantization']} ~{alt['vram_gb']} GB"))
    tr = m.get("trust", {})
    vmark = green("verifiable (hashed)") if tr.get("hashed") else amber("unverified bytes")
    src = green("verified source") if tr.get("source_verified") else amber("unverified source")
    print(dim(f"  trust: {src} · {vmark}"))
    print(dim(f"  {len(m['files'])} files · pull: petabyte model pull {m['id']}"))
    return 0


def cmd_pull(args):
    mgr = _mgr(args)
    rep = Reporter(quiet=getattr(args, "quiet", False))
    try:
        rec = mgr.pull(args.id, fmt=args.format, quantization=args.quantization,
                       revision=args.revision, force=args.force, on_event=rep, jobs=args.jobs)
    except CompatibilityError as e:
        _fail(str(e))
        for alt in (e.report.get("alternatives") or [])[:3]:
            print(dim(f"  try: {alt['quantization']} (~{alt['vram_gb']} GB VRAM)"))
        return 2
    except GatedError as e:
        return _gated(e)
    except ConcurrentPullError as e:
        return _fail(str(e))
    except AuthError:
        return _fail("access denied — set a token with `petabyte auth huggingface` (gated/private model)")
    except (DownloadError, ProviderError, ModelIdError) as e:
        return _fail(str(e))
    vmark = green("✓ SHA256 verified") if rec["verified"] else amber("• bytes not hash-verified by source")
    print("\n" + dim("Cache:  ") + rec["path"])
    if rec["cache_hits"]:
        print(dim(f"        {rec['cache_hits']} file(s) already cached (deduplicated)"))
    print(vmark)
    print(green("✓ ") + bold(rec["id"]) + green(" ready"))
    return 0


def cmd_list(args):
    mgr = _mgr(args)
    rows = mgr.list_installed()
    if not rows:
        print(dim("no models installed — try: petabyte model pull Qwen/Qwen3-8B"))
        return 0
    print(dim(f"  {'MODEL':<44} {'SIZE':>9}  {'FORMAT':<12} STATUS"))
    for r in rows:
        status = green("Ready") if r["installed"] else amber("incomplete")
        print(f"  {bold(r['id'][:44]):<44} {human(r['total_size']):>9}  "
              f"{(r.get('format') or '—'):<12} {status}")
    return 0


def cmd_inspect(args):
    mgr = _mgr(args)
    insp = mgr.inspect(args.id, revision=args.revision)
    if not insp:
        return _fail(f"{args.id} is not installed")
    m = insp["manifest"]
    print(bold(m["id"]) + dim(f"  @ {m['revision'][:12]}  →  {insp['path']}"))
    print(dim(f"  {'FILE':<44} {'SIZE':>9}  SHARED"))
    for f in insp["files"]:
        shared = cyan("shared") if f["shared"] else ("" if f["present"] else red("missing"))
        print(f"  {f['path'][:44]:<44} {human(f['size']):>9}  {shared}")
    return 0


def cmd_remove(args):
    mgr = _mgr(args)
    res = mgr.remove(args.id, revision=args.revision)
    if not res["removed"]:
        return _fail(f"{args.id} is not installed")
    print(green("✓ removed ") + bold(res["id"]) + dim(f" ({res['revision'][:12]})"))
    print(dim("  blobs are kept for dedup — reclaim with: petabyte model cache prune"))
    return 0


def cmd_cache(args):
    mgr = _mgr(args)
    if args.cache_cmd == "prune":
        res = mgr.cache_prune(dry_run=args.dry_run)
        verb = "would reclaim" if args.dry_run else "reclaimed"
        print(green("✓ ") + f"{verb} {human(res['freed_bytes'])} from {res['removed_count']} blob(s)")
        return 0
    st = mgr.cache_status()
    print(bold("Petabyte model cache") + dim("  " + st["home"]))
    print(f"  models installed : {st['models']}")
    print(f"  total on disk    : {human(st['total_bytes'])}")
    print(f"  content blobs    : {st['blob_count']} ({human(st['blob_bytes'])})")
    print(f"  shared blobs     : {st['shared_blobs']}  " + dim("(referenced by >1 model)"))
    print(f"  reclaimable      : {amber(human(st['reclaimable_bytes']))}  " + dim("(unreferenced — prune)"))
    return 0


def cmd_auth(args):
    if args.provider not in ("huggingface", "hf"):
        return _fail(f"unknown auth provider {args.provider!r} (supported: huggingface)")
    token = args.token or os.environ.get("HF_TOKEN")
    if not token and sys.stdin and not sys.stdin.isatty():
        token = sys.stdin.readline().strip()
    if not token:
        try:
            import getpass
            token = getpass.getpass("Hugging Face token (input hidden): ").strip()
        except Exception:  # noqa: BLE001
            return _fail("no token provided")
    if not token:
        return _fail("no token provided")
    set_hf_token(token)
    print(green("✓ Hugging Face token saved") + dim(" to ~/.petabyte/auth.json (0600) — never printed"))
    return 0


def cmd_run(args):
    """Ensure a model is present (pull if missing) then hand it to a runtime. We never fabricate an
    inference engine: if a supported local engine is available we can launch it, otherwise we print
    the ready local path + the marketplace job spec a Petabyte node would consume."""
    mgr = _mgr(args)
    ref = None
    try:
        ref = parse(args.id)
    except ModelIdError as e:
        return _fail(str(e))
    if not mgr.is_installed(ref, revision=args.revision):
        print(dim("Model is not installed. Downloading ") + bold(args.id) + dim("..."))
        rc = cmd_pull(argparse.Namespace(id=args.id, format=args.format, quantization=args.quantization,
                                         revision=args.revision, force=args.force, jobs=1,
                                         quiet=False, home=getattr(args, "home", None)))
        if rc != 0:
            return rc
    path = mgr.local_path(ref, revision=args.revision)
    print(green("✓ ") + bold(args.id) + green(" ready") + dim(f"  →  {path}"))
    print(dim("  runtime input (marketplace job): ")
          + f'{{"model": "{ref.id}", "revision": "{args.revision or ""}"}}')
    print(dim("  a Petabyte compute node consumes this local path directly (vLLM / Ollama / custom)."))
    return 0


def _gated(e):
    print(red("This model is gated.") +
          " Accept its license on the publisher's page, then authenticate:")
    if getattr(e, "url", None):
        print(dim("  open:  ") + e.url)
    print(dim("  then:  ") + "petabyte auth huggingface")
    print(dim("  retry: ") + "petabyte model pull ...")
    return 3


def _fail(msg):
    print(red("error: ") + msg, file=sys.stderr)
    return 1


# =============================================================================== argparse wiring
def _add_common(p):
    p.add_argument("--home", help="cache root (default $PETABYTE_HOME or ~/.petabyte)")


def register(sub, include=("model", "pull", "run", "auth")):
    """Attach the `model` command group + optional `pull`/`run`/`auth` aliases to an existing
    subparsers object. `include` lets a host CLI (which may own its own `run`) opt out of aliases."""
    if "model" in include:
        _register_model(sub)
    if "pull" in include:
        p = sub.add_parser("pull", help="alias for `model pull`")
        _pull_args(p)
    if "run" in include:
        r = sub.add_parser("run", help="ensure a model is present, then hand it to a runtime")
        r.add_argument("id"); r.add_argument("--format"); r.add_argument("--quantization")
        r.add_argument("--revision"); r.add_argument("--force", action="store_true"); _add_common(r)
    if "auth" in include:
        a = sub.add_parser("auth", help="save credentials for gated/private models")
        a.add_argument("provider"); a.add_argument("--token"); _add_common(a)


def _register_model(sub):
    m = sub.add_parser("model", help="discover, download and manage AI models")
    ms = m.add_subparsers(dest="model_cmd", required=True)

    s = ms.add_parser("search", help="search for models")
    s.add_argument("query"); s.add_argument("--limit", type=int, default=25)
    s.add_argument("--license"); s.add_argument("--architecture"); s.add_argument("--max-params", dest="max_params")
    s.add_argument("--source", help="hf|pt"); s.add_argument("--aliases", action="store_true")
    _add_common(s)

    s = ms.add_parser("info", help="show a model's manifest + hardware fit")
    s.add_argument("id"); s.add_argument("--format"); s.add_argument("--quantization")
    s.add_argument("--revision"); _add_common(s)

    s = ms.add_parser("pull", help="download + verify + cache a model")
    _pull_args(s)

    s = ms.add_parser("list", help="list installed models")
    _add_common(s)

    s = ms.add_parser("inspect", help="show installed files + shared blobs")
    s.add_argument("id"); s.add_argument("--revision"); _add_common(s)

    s = ms.add_parser("remove", help="remove an installed model")
    s.add_argument("id"); s.add_argument("--revision"); _add_common(s)

    s = ms.add_parser("cache", help="cache status / prune")
    cs = s.add_subparsers(dest="cache_cmd", required=True)
    st = cs.add_parser("status"); _add_common(st)
    pr = cs.add_parser("prune"); pr.add_argument("--dry-run", action="store_true"); _add_common(pr)


def _pull_args(s):
    s.add_argument("id"); s.add_argument("--format"); s.add_argument("--quantization")
    s.add_argument("--revision"); s.add_argument("--force", action="store_true")
    s.add_argument("--jobs", type=int, default=1); s.add_argument("--quiet", action="store_true")
    _add_common(s)


_MODEL_DISPATCH = {"search": cmd_search, "info": cmd_info, "pull": cmd_pull, "list": cmd_list,
                   "inspect": cmd_inspect, "remove": cmd_remove, "cache": cmd_cache}


def handle(args) -> int:
    """Dispatch a parsed command. Returns an exit code, or None if this module doesn't own it."""
    cmd = getattr(args, "cmd", None)
    if cmd == "model":
        return _MODEL_DISPATCH[args.model_cmd](args)
    if cmd == "pull":
        return cmd_pull(args)
    if cmd == "run":
        return cmd_run(args)
    if cmd == "auth":
        return cmd_auth(args)
    return None


def build_parser():
    p = argparse.ArgumentParser(prog="petabyte", description="Petabyte model hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    register(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rc = handle(args)
    sys.exit(rc if rc is not None else 0)


if __name__ == "__main__":
    main()
