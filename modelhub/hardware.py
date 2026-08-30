"""hardware.py — detect what the machine can actually run, and judge a model against it.

Before downloading a 100 GB model onto a 20 GB-VRAM box, tell the user. Detection is best-effort and
degrade-safe: anything we can't read comes back None and simply isn't used in the verdict. No new
dependency — /proc on Linux, sysctl on macOS, and `nvidia-smi`/`rocminfo` when present.
"""
import os
import re
import shutil
import subprocess


def _run(cmd, timeout=4):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout
    except Exception:  # noqa: BLE001 — tool missing / not permitted / timed out
        return None
    return None


def _system_ram_gb():
    try:
        if os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)   # kB -> GiB
    except Exception:  # noqa: BLE001
        pass
    out = _run(["sysctl", "-n", "hw.memsize"])                            # macOS
    if out and out.strip().isdigit():
        return round(int(out.strip()) / (1024 ** 3), 1)
    return None


def _gpus():
    """List of {name, vram_gb} via nvidia-smi (NVIDIA) or rocminfo (AMD). Empty if none/undetected."""
    gpus = []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
                gpus.append({"name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1),
                             "vendor": "nvidia"})
        if gpus:
            return gpus
    if shutil.which("rocminfo"):                                          # AMD ROCm (coarse)
        ro = _run(["rocminfo"]) or ""
        for m in re.finditer(r"Marketing Name:\s*(.+)", ro):
            gpus.append({"name": m.group(1).strip(), "vram_gb": None, "vendor": "amd"})
    return gpus


def detect(cache_home=None) -> dict:
    """A snapshot of the local machine: CPU, RAM, GPUs, accelerator runtime, free disk."""
    gpus = _gpus()
    vram = [g["vram_gb"] for g in gpus if g.get("vram_gb")]
    free_gb = None
    try:
        free_gb = round(shutil.disk_usage(cache_home or os.path.expanduser("~")).free / (1024 ** 3), 1)
    except Exception:  # noqa: BLE001
        pass
    accel = "cpu"
    if any(g.get("vendor") == "nvidia" for g in gpus) or shutil.which("nvidia-smi"):
        accel = "cuda"
    elif any(g.get("vendor") == "amd" for g in gpus) or shutil.which("rocminfo"):
        accel = "rocm"
    return {
        "cpu_count": os.cpu_count(),
        "ram_gb": _system_ram_gb(),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "gpu_name": gpus[0]["name"] if gpus else None,
        "vram_gb": max(vram) if vram else None,      # single-GPU working set (largest card)
        "vram_total_gb": round(sum(vram), 1) if vram else None,
        "accelerator": accel,
        "free_disk_gb": free_gb,
    }


def compatibility(manifest, hw=None, *, cache_home=None) -> dict:
    """Judge a manifest against a machine. Returns level good|tight|insufficient|unknown, human
    reasons, and (when short on VRAM) suggested lighter quantizations."""
    hw = hw or detect(cache_home=cache_home)
    req = manifest.requirements or {}
    need_vram = req.get("vram_gb") or 0
    need_ram = req.get("ram_gb") or 0
    need_disk = req.get("disk_gb") or round((manifest.total_size or 0) / (1024 ** 3), 1)
    reasons, blocking = [], []

    have_vram = hw.get("vram_gb")
    cpu_only = False
    if need_vram:
        if have_vram is None:
            # No GPU is not automatically a blocker: many models run on CPU (RAM), just slower.
            # Feasibility then hinges on RAM, checked below.
            reasons.append("no GPU detected — will run on CPU (slower)")
            cpu_only = True
        elif have_vram + 0.5 < need_vram:
            reasons.append(f"needs ~{need_vram} GB VRAM, GPU has {have_vram} GB")
            blocking.append("vram")
        elif have_vram < need_vram * 1.15:
            reasons.append(f"~{need_vram} GB VRAM needed, {have_vram} GB available (tight)")

    have_ram = hw.get("ram_gb")
    if need_ram and have_ram is not None:
        # On CPU the weights live in RAM, so RAM must cover them; on GPU, RAM is only a modest host
        # buffer, so compare against a lower bar.
        bar = need_ram if cpu_only else max(need_ram * 0.5, 8)
        if have_ram + 1 < bar:
            reasons.append(f"needs ~{round(bar)} GB RAM, system has {have_ram} GB")
            blocking.append("ram")

    free = hw.get("free_disk_gb")
    if need_disk and free is not None and free < need_disk:
        reasons.append(f"needs ~{need_disk} GB free disk, {free} GB free")
        blocking.append("disk")

    if blocking:
        level = "insufficient"
    elif cpu_only:
        level = "tight"                       # CPU-capable but slower than on a GPU
    elif any("tight" in r for r in reasons):
        level = "tight"
    else:
        level = "good"
        if not reasons:
            reasons.append("runs comfortably" if have_vram else "no GPU requirement / CPU-friendly")

    out = {"level": level, "reasons": reasons, "blocking": blocking,
           "need": {"vram_gb": need_vram, "ram_gb": need_ram, "disk_gb": need_disk}, "machine": hw}
    if "vram" in blocking and have_vram:
        out["alternatives"] = _lighter_variants(manifest, have_vram)
    return out


def _lighter_variants(manifest, have_vram):
    """Suggest quantizations that would fit the detected VRAM, with rough sizes."""
    from .manifest import _BYTES_PER_PARAM
    params = manifest.parameters
    out = []
    if not params:
        return out
    for label, bpp in (("Q5_K_M", _BYTES_PER_PARAM["q5"]), ("Q4_K_M", _BYTES_PER_PARAM["q4"]),
                       ("Q3_K_M", _BYTES_PER_PARAM["q3"]), ("Q2_K", _BYTES_PER_PARAM["q2"])):
        gb = round(params * bpp / (1024 ** 3) * 1.2, 1)
        if gb <= have_vram:
            out.append({"quantization": label, "vram_gb": gb})
    return out[:3]
