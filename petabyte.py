#!/usr/bin/env python3
"""Petabyte CLI — book GPU compute and run a notebook in one command.

Authentication: export an account API key — sign in on the web (Google), create an
'account'-scoped key, then `export PETABYTE_API_KEY=pk_…`. The CLI sends it on every
request; there is no password login.

  export PETABYTE_API_KEY=pk_...
  petabyte deposit 100
  petabyte specs
  petabyte launch ollama --hours 2
  petabyte run notebook.ipynb --gpu H100 --hours 1
  petabyte wallet
"""
import argparse
import json
import os
import sys
import time

import httpx
import os as _os

# The model hub (discover/pull/manage models) lives in the sibling `modelhub` package. Make it
# importable whether the CLI is run as `python cli/petabyte.py` or installed as `petabyte`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modelhub import cli as mh_cli
except Exception:  # noqa: BLE001 — model commands are optional; the compute CLI still works without
    mh_cli = None

_TTY = hasattr(__import__("sys").stdout, "isatty") and __import__("sys").stdout.isatty() and not _os.getenv("NO_COLOR")
def _c(txt, code):
    return f"\033[{code}m{txt}\033[0m" if _TTY else txt
def _amber(t): return _c(t, "38;5;214")
def _cyan(t): return _c(t, "38;5;44")
def _green(t): return _c(t, "38;5;42")
def _dim(t): return _c(t, "2")
def _bold(t): return _c(t, "1")

CONFIG = os.getenv("PETABYTE_CONFIG") or os.path.expanduser("~/.petabyte/cli.json")
DEFAULT_API = os.getenv("PETABYTE_API_URL", "http://localhost:8000")


def _cfg():
    try:
        return json.load(open(CONFIG))
    except Exception:
        return {"api_url": DEFAULT_API}


def _save(cfg):
    d = os.path.dirname(CONFIG)
    if d:                                   # a bare filename (e.g. PETABYTE_CONFIG=cli.json) has no dir
        os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(CONFIG, "w"))


def _api_key(cfg):
    """The account API key used to authenticate. Export PETABYTE_API_KEY (an 'account'-scoped
    key created while signed in on the web), or save it as `api_key` in your config file."""
    return os.getenv("PETABYTE_API_KEY") or cfg.get("api_key")


def _client(cfg, auth=True):
    headers = {}
    if auth:
        # Prefer a session token from browser `login` (or $PETABYTE_TOKEN for headless/CI) as a
        # Bearer; otherwise fall back to an 'account'-scoped API key as X-API-KEY. The API accepts
        # either for every buyer endpoint (deps.get_current_user).
        tok = cfg.get("token") or os.getenv("PETABYTE_TOKEN")
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        else:
            key = _api_key(cfg)
            if key:
                headers["X-API-KEY"] = key
    return httpx.Client(base_url=cfg["api_url"], headers=headers, timeout=30)


def _die(msg, r=None):
    if r is not None:
        msg += f" ({r.status_code}: {r.text[:200]})"
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# Password login + /register_user were removed server-side (OAuth + API keys only). The CLI
# authenticates with an exported PETABYTE_API_KEY (see the module docstring), or `login` runs the
# browser device flow below to mint a session token — no password ever touches the CLI.
def cmd_login(a, cfg):
    return _login_web(cfg)


def _login_web(cfg):
    """OAuth-device-style browser login: start a request, open the approval page, poll for a token.
    No password ever touches the CLI."""
    import time as _t
    import webbrowser
    with _client(cfg, auth=False) as c:
        r = c.post("/auth/cli/start")
    if r.status_code != 200:
        _die("could not start browser login", r)
    d = r.json()
    url = d.get("verification_uri_complete") or d.get("verification_uri")
    print("Open this URL to authorize the CLI:\n  " + _cyan(url) +
          "\nand confirm the code: " + _bold(d.get("user_code", "?")))
    try:
        webbrowser.open(url)
    except Exception:
        pass
    interval = max(2, int(d.get("interval", 3)))
    deadline = _t.time() + int(d.get("expires_in", 600))
    print(_dim("waiting for approval in your browser… (Ctrl-C to cancel)"))
    while _t.time() < deadline:
        _t.sleep(interval)
        try:
            with _client(cfg, auth=False) as c:
                pr = c.post("/auth/cli/poll", json={"device_code": d["device_code"]})
        except Exception:
            continue                              # transient network hiccup — keep polling
        if pr.status_code != 200:
            continue
        st = pr.json().get("status")
        if st == "approved":
            cfg["token"] = pr.json()["access_token"]
            _save(cfg)
            print(_green("✓ logged in"))
            return
        if st in ("denied", "expired"):
            _die(f"browser login {st} — run `petabyte login --web` again")
    _die("browser login timed out — run it again")


def cmd_deposit(a, cfg):
    with _client(cfg) as c:
        r = c.post("/deposit", json={"amount": a.amount})
    print(f"balance: ${r.json()['balance']}" if r.status_code == 200 else _die("deposit failed", r))


def cmd_wallet(a, cfg):
    with _client(cfg) as c:
        r = c.get("/wallet")
    if r.status_code != 200:
        _die("wallet failed", r)
    w = r.json()
    print(f"balance:  ${w['balance']}\nearnings: ${w['earnings']}")


def cmd_specs(a, cfg):
    with _client(cfg) as c:
        r = c.get("/specs")
    if r.status_code != 200:
        _die("specs failed", r)
    specs = r.json()["specs"]
    if not specs:
        print("no bookable GPUs available right now")
        return
    print(_dim(f"  {'ID':>3}  {'GPU':<10} {'$/HR':>7}  {'UNITS':>5}  {'REP':>3}  PROVIDER"))
    for sp in specs:
        rep = sp.get("reputation_score", sp.get("reputation"))
        tags = []
        if sp.get("confidential"): tags.append(_amber("confidential"))
        if sp.get("region_verified"): tags.append(_cyan("region\u2713"))
        line = (f"  {sp['spec_id']:>3}  {_bold(str(sp['gpu_model'] or 'CPU')):<10} "
                f"{_amber('$'+format(sp['price_per_hour'],'.2f')):>7}  "
                f"{sp['available_units']:>5}  {rep:>3}  {sp['provider']}")
        print(line + ("  " + " ".join(tags) if tags else ""))


def _read_code(path):
    if path.endswith(".ipynb"):
        nb = json.load(open(path))
        cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        return "\n\n".join("".join(c.get("source", [])) for c in cells)
    return open(path).read()


def _bundle_project(entry, max_bytes=25 * 1024 * 1024):
    """tar.gz the entry's project folder (siblings + subpackages), skipping junk and
    files >8MB, and return (base64, entry_relpath) — or None if it's a lone file or
    the bundled code exceeds max_bytes (mount/download big data separately)."""
    import tarfile, io, base64
    entry = os.path.abspath(entry)
    root = os.path.dirname(entry) or "."
    IGNORE = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache",
              ".ipynb_checkpoints", ".pytest_cache", "dist", "build", ".idea", ".vscode"}
    buf = io.BytesIO(); total = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in IGNORE and not d.startswith(".")]
            for fn in files:
                if fn.endswith((".pyc", ".pyo")) or fn.startswith("."):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                if sz > 8 * 1024 * 1024:            # skip big data — not source
                    continue
                total += sz
                if total > max_bytes:
                    return None
                tar.add(fp, arcname=os.path.relpath(fp, root))
    return base64.b64encode(buf.getvalue()).decode(), os.path.relpath(entry, root)


def _run_payload(a):
    """The notebook `code` payload for `run`: a JSON project bundle when the entry has
    local dependencies, else the plain single-file source (unchanged behaviour)."""
    code = _read_code(a.file)
    if getattr(a, "deps", None) is False:          # --no-deps: force single file
        return code
    root = os.path.dirname(os.path.abspath(a.file)) or "."
    try:
        siblings = [x for x in os.listdir(root)
                    if x.endswith(".py") and os.path.join(root, x) != os.path.abspath(a.file)]
    except OSError:
        siblings = []
    auto = bool(siblings) or os.path.exists(os.path.join(root, "requirements.txt"))
    if getattr(a, "deps", None) is True or auto:
        b = _bundle_project(a.file)
        if b is None:
            print(_amber("! project code exceeds 25MB — running the single file only "
                         "(mount or download large data from your script)"))
            return code
        b64, entry = b
        print(_dim(f"→ bundling project ({len(b64) // 1024} KB, entry {entry})"))
        return json.dumps({"bundle_b64": b64, "entry": entry, "gpu": bool(getattr(a, "gpu", None))})
    return code


def cmd_run(a, cfg):
    with _client(cfg) as c:
        # pick a spec
        spec_id = a.spec
        if not spec_id:
            sr = c.get("/specs")
            if sr.status_code != 200:       # e.g. 401/5xx — don't .json() a non-OK body
                _die("could not list GPUs", sr)
            specs = sr.json().get("specs", [])
            if a.gpu:
                specs = [s for s in specs if (s["gpu_model"] or "").lower() == a.gpu.lower()]
            if not specs:
                _die("no matching GPU available")
            spec_id = specs[0]["spec_id"]   # cheapest (list is price-sorted)
            print(_dim(f"→ selected spec {spec_id} ({specs[0]['gpu_model']} @ ${specs[0]['price_per_hour']}/hr)"))
        # book (optionally on a private WireGuard VPN — the buyer chooses)
        want_vpn = bool(getattr(a, "vpn", False))
        r = c.post("/request_vm", json={"spec_id": spec_id, "hours": a.hours, "vpn": want_vpn})
        if r.status_code != 200:
            _die("booking failed", r)
        bk = r.json()
        print(f"booked #{bk['booking_id']}  escrow ${bk['gross_amount']} "
              f"(fee ${bk['platform_fee']}, seller ${bk['seller_payout']})")
        if want_vpn and bk.get("vpn_config_url"):
            cr = c.get(bk["vpn_config_url"])
            if cr.status_code == 200:
                path = f"petabyte-{bk['booking_id']}.conf"
                with open(path, "w") as f:
                    f.write(cr.text)
                print(_green(f"✓ VPN config written to {path}") +
                      _dim(f"  → connect with:  sudo wg-quick up ./{path}"))
            else:
                print(_amber("! could not fetch VPN config (booking still active)"))
        # create task
        r = c.post("/create_task", json={"booking_id": bk["booking_id"],
                                         "task_type": "notebook", "code": _run_payload(a)})
        if r.status_code != 200:
            _die("task creation failed", r)
        tid = r.json()["task_id"]
        print(f"dispatched task #{tid} — waiting for a node to execute...")
        # poll
        deadline = time.time() + a.timeout
        while time.time() < deadline:
            t = c.get(f"/tasks/{tid}").json()
            if t["status"] in ("completed", "failed"):
                hdr = _green("\u2713 COMPLETED") if t["status"]=="completed" else _amber("\u2717 FAILED")
                print(f"\n{hdr}")
                print(t.get("result") or "(no output)")
                return
            time.sleep(2)
        print("timed out waiting for result", file=sys.stderr)


def cmd_launch(a, cfg):
    """Launch a ready-made template (ollama, jupyter, blender, minecraft, …) on the cheapest
    verified GPU that fits — the CLI twin of the web one-click launcher (`POST /launch`)."""
    body = {"template": a.template, "hours": a.hours}
    if getattr(a, "max_price", None) is not None:
        body["max_price_per_hour"] = a.max_price
    if getattr(a, "region", None):
        body["region"] = a.region
    if getattr(a, "spec", None):
        body["spec_id"] = str(a.spec)
    with _client(cfg) as c:
        r = c.post("/launch", json=body)
        if r.status_code != 200:
            _die("launch failed", r)
        d = r.json()
        print(_green("✓ launched ") + _bold(a.template) +
              _dim(f"  · {d.get('gpu_model', '?')} @ ${d.get('price_per_hour', '?')}/hr · {a.hours}h"))
        print(f"  booking #{d.get('booking_id')}   escrow ${d.get('gross_amount')}")
        if d.get("routing_explanation"):
            print("  " + _dim(d["routing_explanation"]))
        url = d.get("url")
        addr = url.get("http") if isinstance(url, dict) else url
        if addr:
            print("  address  " + _cyan(addr))
        if isinstance(url, dict) and url.get("ssh"):
            print("  ssh      " + _dim(url["ssh"]))
        if d.get("connect"):
            print("  " + _dim(d["connect"]))


def cmd_ask(a, cfg):
    """Send a prompt to the pay-per-token Inference API (OpenAI-compatible) and print the answer.

    Uses an `inference`-scoped API key (not the login token): --key, else $PETABYTE_API_KEY,
    else `api_key` in your saved config. Answer goes to stdout (pipeable); the token/cost line
    goes to stderr."""
    if getattr(a, "key_stdin", False):
        key = sys.stdin.readline().strip()      # key never touches argv / shell history / ps
    else:
        key = a.key or os.getenv("PETABYTE_API_KEY") or cfg.get("api_key")
    if not key:
        _die("no inference API key — create one with scope 'inference' at /keys, then pipe it "
             "in with --key-stdin, set PETABYTE_API_KEY, or save it as 'api_key' in your config")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(base_url=cfg["api_url"], headers=headers, timeout=120) as c:
        body = {"messages": [{"role": "user", "content": a.prompt}]}
        if getattr(a, "model", None):
            body["model"] = a.model
        try:
            r = c.post("/v1/chat/completions", json=body)
        except httpx.RequestError as e:
            _die(f"cannot reach the inference API at {cfg['api_url']} ({e.__class__.__name__})")
        if r.status_code != 200:
            _die("inference failed", r)
        try:
            j = r.json()
        except ValueError:
            _die("unexpected non-JSON response from the inference API", r)
        try:
            print(j["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            _die("unexpected response from the inference API", r)
        b = j.get("x_petabyte") or {}
        if b.get("tokens") is not None:
            note = f"{b.get('tokens')} tokens"
            if b.get("charged") is not None:
                note += f" · ${float(b.get('charged')):.6f}" + (" billed" if b.get("billed") else " (free)")
            print(_dim("  " + note), file=sys.stderr)


def cmd_vpn(a, cfg):
    """Download (or re-download) the WireGuard client config for a VPN-enabled booking."""
    with _client(cfg) as c:
        r = c.get(f"/vpn_config/{a.booking_id}")
        if r.status_code != 200:
            _die("no VPN config for that booking (was it booked with --vpn?)", r)
        path = a.out or f"petabyte-{a.booking_id}.conf"
        with open(path, "w") as f:
            f.write(r.text)
        print(_green(f"✓ VPN config written to {path}"))
        print(_dim(f"  connect:  sudo wg-quick up ./{path}      disconnect:  sudo wg-quick down ./{path}"))


def cmd_earnings(a, cfg):
    """Seller payout state: balance, withdrawable earnings, what's still clearing, recent payouts."""
    with _client(cfg) as c:
        w = c.get("/wallet")
        if w.status_code != 200:
            _die("wallet failed", w)
        w = w.json()
        pays = c.get("/wallet/payouts")
    print(_bold("Earnings"))
    print(f"  balance:      {_amber('$' + format(w['balance'], '.2f'))}")
    print(f"  earnings:     ${format(w['earnings'], '.2f')}")
    print(f"  withdrawable: {_green('$' + format(w['withdrawable'], '.2f'))}")
    print(f"  clearing:     ${format(w['clearing'], '.2f')}  " + _dim(f"(held {w['hold_hours']}h)"))
    print("  instant payout: " + (_green('eligible') if w.get('instant_eligible') else _dim('not yet')))
    if pays.status_code == 200 and pays.json().get("payouts"):
        print(_dim("\n  recent payouts:"))
        print(_dim(f"    {'AMOUNT':>10}  {'KIND':<12} {'STATUS':<10} WHEN"))
        for p in pays.json()["payouts"][:8]:
            print(f"    {'$' + format(p['amount_usd'], '.2f'):>10}  {str(p['kind'])[:12]:<12} "
                  f"{str(p['status'])[:10]:<10} {str(p['created_at'])[:10]}")
    else:
        print(_dim("\n  no payouts yet — add a payout method on the earnings page, then `withdraw`."))


def _node_status(a, cfg):
    sid = a.spec_id
    with _client(cfg) as c:
        dash = c.get("/seller/dashboard")
        if dash.status_code != 200:
            _die("dashboard failed", dash)
        dash = dash.json()
        models = c.get(f"/nodes/{sid}/models")
        disk = c.get(f"/nodes/{sid}/disk")
    node = next((n for n in dash.get("nodes", []) if n.get("spec_id") == sid), None)
    if not node:
        _die(f"node {sid} not found (is it one of yours?)")
    on = _green("online") if node["online"] else _amber("offline")
    att = _green("attested") if node["attested"] else _amber("unverified")
    print(_bold(f"Node {sid}") + f"  {node.get('gpu_model') or 'CPU'}  [{on} · {att}]")
    sug = _dim(f"  (suggested ${format(node['suggested_price'], '.2f')})") if node.get("suggested_price") else ""
    print(f"  price:        ${format(node['price_per_hour'], '.2f')}/hr" + sug)
    print(f"  units:        {node['units_busy']}/{node['units_total']} busy  ({node['utilization_pct']}% util)")
    succ = f"  ({node['success_rate']}% success)" if node.get("success_rate") is not None else ""
    print(f"  jobs:         {node['jobs_completed']} done · {node['jobs_failed']} failed" + succ)
    print(f"  reputation:   {node['reputation']}")
    print(f"  earned:       {_green('$' + format(node['earned_total'], '.2f'))}")
    print(f"  last seen:    {node.get('last_seen') or 'never'}")
    if models.status_code == 200:
        ms = models.json().get("models", [])
        if ms:
            print(f"  models cached: {len(ms)}" + _dim("  " + ", ".join(ms[:6]) + (" …" if len(ms) > 6 else "")))
        else:
            print("  models cached: 0" + _dim(f"  (run: petabyte node sync-models {sid})"))
    if disk.status_code == 200 and disk.json().get("enabled"):
        d = disk.json()
        print(f"  disk rental:  {d['provider']} up to {d['alloc_gb']} GB")
    for b in [x for x in dash.get("blockers", []) if x.get("node") == node.get("id")]:
        print(_amber("  ! " + b["issue"]) + _dim("  fix: " + b.get("fix", "")))


def _node_sync_models(a, cfg):
    """Scan the local ~/.petabyte model cache and report the ids to the marketplace, so the
    scheduler prefers THIS node for jobs that need a model it already holds."""
    try:
        from modelhub import ModelManager
    except Exception:  # noqa: BLE001
        _die("model hub not available on this machine (cannot scan the local cache)")
    ids = ModelManager().cached_model_ids()
    with _client(cfg) as c:
        r = c.post("/nodes/models", json={"spec_id": a.spec_id, "models": ids})
    if r.status_code != 200:
        _die("sync failed", r)
    n = r.json().get("cached_models", 0)
    print(_green(f"✓ reported {n} cached model(s)") + _dim(f" for node {a.spec_id}"))
    for m in ids[:12]:
        print(_dim("    " + m))
    if not ids:
        print(_dim("    (local cache is empty — pull a model first: petabyte model pull <id>)"))


def cmd_node(a, cfg):
    {"status": _node_status, "sync-models": _node_sync_models}[a.node_cmd](a, cfg)


_JOB_DONE = {"complete", "done", "ok", "stitched", "succeeded"}
_JOB_FAILED = {"failed", "error", "cancelled"}


def _parse_frames(spec):
    spec = (spec or "1-1").strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


def _upload_input(c, path):
    if not os.path.exists(path):
        _die(f"file not found: {path}")
    r = c.post("/uploads/url", json={"filename": os.path.basename(path)})
    if r.status_code >= 300:
        _die("could not get an upload URL", r)
    up = r.json()
    with open(path, "rb") as fh:
        pr = httpx.put(up["upload_url"], content=fh.read(), timeout=1800)
    if pr.status_code >= 300:
        _die(f"upload failed ({pr.status_code})")
    return up["ref"]


def _poll_job(c, job_id, label):
    seen = None
    while True:
        r = c.get(f"/jobs/manifest/{job_id}")
        if r.status_code >= 300:
            _die("could not read job status", r)
        m = r.json()
        segs = m.get("segments", [])
        done = sum(1 for x in segs if str(x.get("status", "")).lower() in _JOB_DONE)
        tot = m.get("total_segments") or len(segs) or 1
        key = (done, m.get("status"))
        if key != seen:
            print(f"  {label}: {done}/{tot} segment(s) — {m.get('status')}")
            seen = key
        st = str(m.get("status", "")).lower()
        if st in _JOB_DONE or st in _JOB_FAILED or (tot and done >= tot):
            return m
        time.sleep(4)


def _download_outputs(c, job_id, outdir):
    r = c.post("/jobs/output_url", json={"job_id": job_id})
    if r.status_code >= 300:
        _die("could not list outputs", r)
    outs = r.json().get("outputs", [])
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for o in outs:
        name = (o.get("output_ref", "").rstrip("/").split("/")[-1]) or f"seg{o.get('idx')}"
        d = httpx.get(o["download_url"], timeout=1800)
        if d.status_code < 300:
            dst = os.path.join(outdir, name)
            with open(dst, "wb") as fh:
                fh.write(d.content)
            saved.append(dst)
    return saved


def cmd_render(a, cfg):
    """Drop a .blend, render on the farm, pay only for render time, download the frames."""
    fs, fe = _parse_frames(a.frames)
    with _client(cfg) as c:
        print(_dim(f"uploading {os.path.basename(a.file)} …"))
        ref = _upload_input(c, a.file)
        r = c.post("/render", json={"blend_ref": ref, "frame_start": fs, "frame_end": fe,
                                    "samples": a.samples, "nodes": a.nodes, "hours": a.hours})
        if r.status_code >= 300:
            _die("render request failed", r)
        d = r.json()
        print(_green("✓ render started  ")
              + f"job #{d['job_id']} · {d['nodes']} node(s) · frames {fs}-{fe}")
        print("  " + _bold(f"~${d.get('estimated_cost')}")
              + _dim(" max — you're billed only for actual render time"))
        m = _poll_job(c, d["job_id"], "render")
        if str(m.get("status", "")).lower() in _JOB_FAILED:
            _die("render job failed")
        saved = _download_outputs(c, d["job_id"], a.out)
        print(_green(f"✓ {len(saved)} frame(s) → {a.out}") if saved
              else _amber("job finished but no outputs were ready — re-run to fetch later"))


def cmd_transcode(a, cfg):
    """Drop a video, GPU-transcode it (NVENC), download the result."""
    with _client(cfg) as c:
        print(_dim(f"uploading {os.path.basename(a.file)} …"))
        ref = _upload_input(c, a.file)
        body = {"input_ref": ref, "codec": a.codec, "container": a.container, "use_gpu": True}
        if a.resolution:
            body["resolution"] = a.resolution
        if a.crf is not None:
            body["crf"] = a.crf
        r = c.post("/transcode", json=body)
        if r.status_code >= 300:
            _die("transcode request failed", r)
        d = r.json()
        print(_green("✓ transcode started  ") + f"job #{d.get('job_id')}")
        if d.get("estimated_cost") is not None:
            print("  " + _bold(f"~${d.get('estimated_cost')}") + _dim(" max — billed for actual time"))
        m = _poll_job(c, d["job_id"], "transcode")
        if str(m.get("status", "")).lower() in _JOB_FAILED:
            _die("transcode job failed")
        saved = _download_outputs(c, d["job_id"], a.out)
        print(_green(f"✓ output → {a.out}") if saved else _amber("finished; no output ready yet"))


def main():
    p = argparse.ArgumentParser(prog="petabyte")
    p.add_argument("--api", help="API base URL (overrides saved config)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("login", help="authorize in the browser (device flow) — no password on "
                                     "the CLI; token also via $PETABYTE_TOKEN")
    s.add_argument("--web", action="store_true", help="(default) browser device-login")
    s = sub.add_parser("deposit");  s.add_argument("amount", type=float)
    sub.add_parser("wallet")
    sub.add_parser("specs")
    s = sub.add_parser("run", help="run a notebook/.py on a rented GPU, OR start a model runtime")
    s.add_argument("--deps", dest="deps", action="store_true", default=None,
                   help="bundle the whole project folder (auto when siblings/requirements.txt exist)")
    s.add_argument("--no-deps", dest="deps", action="store_false",
                   help="ship only the single file")
    s.add_argument("file", help="a .ipynb/.py file (compute job) OR a model id like Qwen/Qwen3-8B")
    s.add_argument("--spec", type=int); s.add_argument("--gpu")
    s.add_argument("--hours", type=int, default=1); s.add_argument("--timeout", type=int, default=120)
    s.add_argument("--vpn", action="store_true",
                   help="rent on a private WireGuard VPN and save the client config")
    s.add_argument("--revision"); s.add_argument("--format"); s.add_argument("--quantization")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("launch",
                       help="launch a ready-made template (ollama, jupyter, blender, minecraft…) on the cheapest verified GPU")
    s.add_argument("template", help="template name, e.g. ollama, jupyter, blender, minecraft")
    s.add_argument("--hours", type=int, default=2)
    s.add_argument("--region")
    s.add_argument("--max-price", type=float, dest="max_price", help="cap the $/hour you'll pay")
    s.add_argument("--spec", type=int, help="pin to a specific host spec id")
    s = sub.add_parser("vpn", help="download the WireGuard config for a VPN booking")
    s.add_argument("booking_id", type=int); s.add_argument("-o", "--out")
    s = sub.add_parser("ask", help="send a prompt to the pay-per-token Inference API and print the answer")
    s.add_argument("prompt", help="the prompt to send")
    s.add_argument("--model", help="model id (default: the server's default)")
    s.add_argument("--key", help="inference API key — lands in shell history and `ps`, so prefer "
                                 "--key-stdin / $PETABYTE_API_KEY / saved config; use only with "
                                 "short-lived keys (e.g. CI)")
    s.add_argument("--key-stdin", action="store_true", dest="key_stdin",
                   help="read the API key from stdin, keeping it out of argv and shell history")

    # seller: read node/payout state, and feed the model cache-locality signal
    sub.add_parser("earnings", help="your balance, withdrawable earnings and recent payouts")
    n = sub.add_parser("node", help="inspect a node you host")
    ns = n.add_subparsers(dest="node_cmd", required=True)
    st = ns.add_parser("status", help="node status: online/attested, utilization, jobs, earnings")
    st.add_argument("spec_id", type=int)
    sm = ns.add_parser("sync-models",
                       help="scan the local ~/.petabyte cache and report it to the marketplace")
    sm.add_argument("spec_id", type=int)

    s = sub.add_parser("render", help="render a .blend on the GPU farm — pay only for render time")
    s.add_argument("file", help="path to a .blend scene")
    s.add_argument("--frames", default="1-1", help="frame range, e.g. 1-120 or 5")
    s.add_argument("--samples", type=int, default=128)
    s.add_argument("--nodes", type=int, default=1, help="split the frame range across N nodes")
    s.add_argument("--hours", type=int, default=1, help="max hours to escrow (unused is refunded)")
    s.add_argument("--out", default="./renders", help="download frames here")

    s = sub.add_parser("transcode", help="GPU-transcode a video (NVENC) — drop a file, get it back")
    s.add_argument("file", help="path to a source video")
    s.add_argument("--codec", default="h264", help="h264|h265|av1|vp9")
    s.add_argument("--resolution", help="e.g. 1920x1080")
    s.add_argument("--crf", type=int, help="quality 0-51 (lower = better)")
    s.add_argument("--container", default="mp4", help="mp4|mkv|webm|mov|…")
    s.add_argument("--out", default="./transcoded", help="download output here")

    # model hub: discover/pull/manage AI models (Hugging Face-grade UX). Owns `model`, `pull`, `auth`;
    # `run` is shared with the compute flow above and dispatched smartly below.
    if mh_cli is not None:
        mh_cli.register(sub, include=("model", "pull", "auth"))

    a = p.parse_args()
    cfg = _cfg()
    if a.api:
        cfg["api_url"] = a.api

    if mh_cli is not None and a.cmd in ("model", "pull", "auth"):
        sys.exit(mh_cli.handle(a) or 0)
    if a.cmd == "run" and _is_model_ref(a.file):
        if mh_cli is None:
            _die("model runtime unavailable (modelhub not importable)")
        ns = __import__("argparse").Namespace(
            id=a.file, format=a.format, quantization=a.quantization, revision=a.revision,
            force=a.force, home=None)
        sys.exit(mh_cli.cmd_run(ns) or 0)
    {"deposit": cmd_deposit, "login": cmd_login,
     "wallet": cmd_wallet, "specs": cmd_specs, "run": cmd_run, "launch": cmd_launch, "vpn": cmd_vpn,
     "earnings": cmd_earnings, "node": cmd_node, "ask": cmd_ask,
     "render": cmd_render, "transcode": cmd_transcode}[a.cmd](a, cfg)


def _is_model_ref(arg):
    """`run` overloads a file path and a model id. A model id has a source/slug shape and is NOT an
    existing local file or a notebook/script."""
    if os.path.exists(arg) or arg.endswith((".ipynb", ".py")):
        return False
    return ("/" in arg) or (":" in arg) or arg.startswith(("hf:", "pt:", "http://", "https://"))


if __name__ == "__main__":
    main()
