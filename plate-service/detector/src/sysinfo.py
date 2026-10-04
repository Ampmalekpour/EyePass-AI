"""
sysinfo.py
--------------------------------------------------------------------
Startup hardware report for the CPU pipeline: what this machine is, how
much of it the detector may use, what the measured capacity is, and what
N cameras would need. Printed once by engine 0 right after the real-time
calibration (capacity.py); also stored in the capacity profile
(Redis plate:internal:detector:capacity and /health).

collect() reads /proc and /sys only (Linux container); every field is
optional — a missing source is simply left out of the report.
--------------------------------------------------------------------
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict, List, Optional

import config
import capacity


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _count_cpulist(text: Optional[str]) -> Optional[int]:
    """'0-7,16-19' -> 12"""
    if not text:
        return None
    n = 0
    for part in text.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def _cpu_busy_percent(interval: float = 0.4) -> Optional[float]:
    def snap():
        line = (_read("/proc/stat") or "").splitlines()[0].split()[1:]
        v = [int(x) for x in line]
        return sum(v), v[3] + (v[4] if len(v) > 4 else 0)
    try:
        t0, i0 = snap()
        time.sleep(interval)
        t1, i1 = snap()
        return round(100.0 * (1 - (i1 - i0) / max(1, t1 - t0)), 1)
    except Exception:
        return None


def _kb(text: Optional[str]) -> Optional[float]:
    try:
        return int(text.split()[0]) / 1024.0 / 1024.0     # kB -> GiB
    except Exception:
        return None


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("openvino", "onnxruntime", "numpy", "cv2", "torch", "ultralytics"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?")).split("+")[0]
        except Exception:
            pass
    return out


def collect() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    cpuinfo = _read("/proc/cpuinfo") or ""
    flags, pairs, phys, model, mhz = set(), set(), set(), None, None
    cur_phys = cur_core = None
    for line in cpuinfo.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "model name" and not model:
            model = v
        elif k == "flags" and not flags:
            flags = set(v.split())
        elif k == "physical id":
            cur_phys = v
            phys.add(v)
        elif k == "core id":
            cur_core = v
            pairs.add((cur_phys, cur_core))
        elif k == "cpu MHz" and not mhz:
            mhz = v
    info["cpu"] = model or platform.processor() or platform.machine()
    info["logical_cores"] = os.cpu_count() or 0
    info["physical_cores"] = len(pairs) or None
    info["sockets"] = len(phys) or None
    try:
        info["usable_cores"] = len(os.sched_getaffinity(0))
    except Exception:
        info["usable_cores"] = info["logical_cores"]
    cur = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    info["max_mhz"] = round(int(cur) / 1000) if cur and cur.isdigit() else (round(float(mhz)) if mhz else None)
    info["l3_cache"] = _read("/sys/devices/system/cpu/cpu0/cache/index3/size")
    p_cores = _count_cpulist(_read("/sys/devices/cpu_core/cpus"))
    e_cores = _count_cpulist(_read("/sys/devices/cpu_atom/cpus"))
    if p_cores or e_cores:
        info["hybrid"] = {"p_threads": p_cores, "e_cores": e_cores}
    info["isa"] = [n for n, f in (("AVX2", "avx2"), ("AVX-512", "avx512f"), ("VNNI", "avx512_vnni"),
                                  ("AVX-VNNI", "avx_vnni"), ("AMX", "amx_tile")) if f in flags]
    info["cgroup_cpu_limit"] = capacity.device_info().get("cpu_limit_cores")
    mem = {l.split(":")[0]: l.split(":")[1] for l in (_read("/proc/meminfo") or "").splitlines() if ":" in l}
    info["mem_total_gib"] = _kb(mem.get("MemTotal"))
    info["mem_available_gib"] = _kb(mem.get("MemAvailable"))
    lim = _read("/sys/fs/cgroup/memory.max")
    info["mem_limit_gib"] = round(int(lim) / 2 ** 30, 1) if lim and lim.isdigit() else None
    info["virtualized"] = "microsoft" in (_read("/proc/version") or "").lower()
    try:
        info["load_avg_1m"] = round(os.getloadavg()[0], 2)
    except Exception:
        pass
    info["cpu_busy_percent"] = _cpu_busy_percent()
    info["versions"] = _versions()
    return info


def _fmt(v, unit="", nd=1):
    return "?" if v is None else (f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def render(profile: Dict[str, Any], info: Dict[str, Any]) -> List[str]:
    """The report, one string per log line."""
    L: List[str] = []
    rows = profile.get("table") or []
    fps = float(config.REALTIME_MIN_FPS)
    margin = float(config.CAPACITY_SAFETY_MARGIN)
    base_iv = float(max(1, int(config.DETECT_EVERY_N_FRAMES)))
    cam_fps = float(config.CAMERA_ASSUMED_FPS)
    max_iv = max(base_iv, cam_fps / float(config.DETECT_MIN_FPS))
    bar = "═" * 78

    def add(s=""):
        L.append(f"║ {s}")

    L.append("╔" + bar)
    add("🖥️  DETECTOR HARDWARE & CAPACITY REPORT  (CPU pipeline)")
    L.append("╠" + bar)

    # ---- machine
    add("💻 MACHINE")
    add(f"   CPU        : {info.get('cpu')}")
    topo = f"{_fmt(info.get('physical_cores'))} physical / {info.get('logical_cores')} logical cores"
    if info.get("sockets"):
        topo += f", {info['sockets']} socket(s)"
    if info.get("max_mhz"):
        topo += f", up to {info['max_mhz']} MHz"
    if info.get("l3_cache"):
        topo += f", L3 {info['l3_cache']}"
    add(f"   Topology   : {topo}")
    hy = info.get("hybrid")
    if hy:
        add(f"   Hybrid CPU : {hy.get('p_threads')} performance-core threads + {hy.get('e_cores')} efficiency cores "
            "(efficiency cores are slower — one reason loop times are uneven)")
    isa = info.get("isa") or []
    add(f"   Instr. sets: {', '.join(isa) if isa else '?'}"
        + ("   → INT8 runs on VNNI/AMX hardware" if any(x in isa for x in ("VNNI", "AVX-VNNI", "AMX"))
           else "   (no VNNI/AMX: INT8 gains are smaller)"))
    lim, usable, logical = info.get("cgroup_cpu_limit"), info.get("usable_cores"), info.get("logical_cores")
    avail = min(x for x in (lim, usable, logical) if x) if any((lim, usable, logical)) else None
    why = []
    if lim:
        why.append(f"container CPU limit {lim}")
    if usable and usable < (logical or usable):
        why.append(f"affinity {usable}")
    add(f"   Usable now : {_fmt(avail)} logical cores for this container"
        + (f"  ({'; '.join(why)})" if why else "  (no container limit)"))
    if info.get("virtualized"):
        add("   ⚠️  Runs inside a Windows (WSL2 / Docker Desktop) VM: the cores above are what the VM gives "
            "Docker (.wslconfig `processors=`), not necessarily the whole PC.")
    mt, ma, ml = info.get("mem_total_gib"), info.get("mem_available_gib"), info.get("mem_limit_gib")
    add(f"   Memory     : {_fmt(mt, ' GiB')} total, {_fmt(ma, ' GiB')} free"
        + (f", container limit {ml} GiB" if ml else ""))
    busy, la = info.get("cpu_busy_percent"), info.get("load_avg_1m")
    add(f"   Load now   : CPU {_fmt(busy, '%')} busy at startup, load average {_fmt(la, '', 2)}"
        + ("   ⚠️ the machine is already busy — measured capacity is lower than on an idle machine"
           if busy is not None and busy > 25 else ""))
    v = info.get("versions") or {}
    add("   Software   : " + ", ".join(f"{k} {x}" for k, x in v.items()))

    # ---- model and measured throughput
    L.append("╠" + bar)
    add("🧠 MODEL & MEASURED SPEED   (real inference path, 1080p dummy frames, nothing else running)")
    add(f"   Backend    : {profile.get('backend')}  | model {profile.get('variant')} | input {profile.get('input')} "
        f"| torch threads {config.CPU_TORCH_NUM_THREADS or 'default'} | cv2 threads {config.CV2_NUM_THREADS}")
    if rows:
        ceiling = max(r["n"] * 1000.0 / r["avg_ms"] for r in rows if r["avg_ms"] > 0)
        per_cam_ms = rows[-1]["avg_ms"] / rows[-1]["n"]
        usable_det = ceiling * margin
        add(f"   One camera : {rows[0]['avg_ms']} ms per frame alone ({1000.0 / rows[0]['avg_ms']:.0f} fps); "
            f"with {rows[-1]['n']} cameras {per_cam_ms:.1f} ms each (they share the cores)")
        add(f"   CEILING    : ≈ {ceiling:.0f} detections/s for the whole machine "
            f"(= {ceiling / fps:.1f} cameras × {fps:.0f} fps, before decode/OCR)")
        add(f"   SAFE USE   : ≈ {usable_det:.0f} detections/s after the {margin:.0%} safety margin — "
            f"the other {1 - margin:.0%} is reserved for RTSP decoding, tracking, JPEG/Redis, OCR hand-off")
        cap = profile.get("max_cameras", 0)
        add(f"   ✅ CAPACITY: {cap} camera(s) at the full ≥{fps:.0f} fps"
            + (" (at least — sweep limit reached)" if profile.get("max_is_lower_bound") else ""))

        # ---- planning table
        L.append("╠" + bar)
        add("📋 PLANNING TABLE — what N cameras need and what you get")
        add(f"   {'cameras':>7} │ {'needs':>10} │ {'of safe use':>11} │ {'loop (p95)':>10} │ detection per camera")
        top = max(10, min(16, int(profile.get("max_cameras", 0)) + 6))
        for n in sorted({*range(1, top + 1)}):
            need = n * fps
            pct = 100.0 * need / usable_det if usable_det else 0
            lm = capacity.loop_ms(profile, n)
            iv, _, fits = capacity.pick_detect_interval(profile, n, base_iv, max_iv)
            rate = cam_fps / iv
            if fits and iv <= base_iv + 1e-9:
                st = f"✅ {rate:4.1f} fps  full rate"
            elif fits:
                st = f"🐢 {rate:4.1f} fps  ({rate / cam_fps:.0%} of full, auto-degrade {'ON' if config.CAPACITY_AUTO_DEGRADE else 'OFF → frames will be missed'})"
            else:
                st = f"🚨 below the {config.DETECT_MIN_FPS:g} fps floor — too many cameras for this CPU"
            mark = "  ◄ measured" if n <= rows[-1]["n"] else ""
            add(f"   {n:>7} │ {need:>6.0f} /s │ {pct:>10.0f}% │ {lm:>8.1f} ms │ {st}{mark}")

        # ---- advice
        L.append("╠" + bar)
        add("💡 WHAT TO DO")
        for k in (4, 8, 12, 16):
            need_ceiling = k * fps / margin
            if need_ceiling > ceiling * 1.0:
                add(f"   • {k:>2} cameras at full {fps:.0f} fps need a ceiling of ≈ {need_ceiling:.0f} det/s — "
                    f"this machine has {ceiling:.0f} ({ceiling / need_ceiling:.0%}); i.e. ×{need_ceiling / ceiling:.1f} faster")
        if profile.get("variant") == "openvino_fp32":
            add("   • Try DETECTION_CPU_MODEL=openvino_int8: ≈1.6× faster here (check accuracy on your clips first).")
        if info.get("virtualized"):
            add("   • Give Docker more cores (WSL2 .wslconfig `processors=`), or run the service on a Linux host.")
        add("   • A GPU (DETECTION_DEVICE=gpu) serves many more cameras: it batches all frames in one call.")
        add(f"   • Auto-degrade is {'ON' if config.CAPACITY_AUTO_DEGRADE else 'OFF'} (CAPACITY_AUTO_DEGRADE in .env): "
            + ("over capacity every camera is detected at a lower rate instead of losing frames."
               if config.CAPACITY_AUTO_DEGRADE else "over capacity expect missed frames (warning only)."))
    L.append("╚" + bar)
    return L
