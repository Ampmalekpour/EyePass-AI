"""
capacity.py
--------------------------------------------------------------------
Real-time capacity of THIS machine for THIS model, measured instead of
guessed.

Rule (config.py "2b. REAL-TIME CAPACITY"): every camera must be served at
>= REALTIME_MIN_FPS, so one inference loop for n cameras has to finish
within

    budget_ms = DETECT_EVERY_N_FRAMES * 1000 / REALTIME_MIN_FPS * CAPACITY_SAFETY_MARGIN

Loop time is NOT linear in n (model instances compete for the cores), so
calibrate() runs the engine's real code path (backend.infer on n frames,
n = 1, 2, 3, ...) and records avg / p95 loop time per n. The capacity is
the largest n for which n and every smaller count fit the budget.

Engine 0 calibrates at startup, before any camera is attached, and
publishes the profile to Redis; check_camera() turns it into the
✅ / 🚨 [CAPACITY] line the bridge logs for every camera it attaches.
--------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import time
from typing import Any, Dict, List, Optional

import numpy as np

import config


def budget_ms() -> float:
    n = max(1, int(config.DETECT_EVERY_N_FRAMES))
    return n * 1000.0 / float(config.REALTIME_MIN_FPS) * float(config.CAPACITY_SAFETY_MARGIN)


def frame_period_ms() -> float:
    return max(1, int(config.DETECT_EVERY_N_FRAMES)) * 1000.0 / float(config.REALTIME_MIN_FPS)


def device_info() -> Dict[str, Any]:
    """CPU model, logical cores and the container's CPU limit (cgroup)."""
    model = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    limit = None
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as f:
            quota, period = f.read().split()[:2]
            if quota != "max":
                limit = round(int(quota) / int(period), 2)
    except Exception:
        pass
    return {"cpu": model, "logical_cores": os.cpu_count() or 0, "cpu_limit_cores": limit}


def _p95(v: List[float]) -> float:
    return float(np.percentile(v, 95))


def calibrate(backend, log, engine_id: int = 0) -> Dict[str, Any]:
    """Sweep n = 1.. cameras on `backend`, print the table (🧪), return the
    profile. Leaves the backend with its instances created for the largest
    n measured — the caller resets them (backend.reset_streams())."""
    budget = budget_ms()
    period = frame_period_ms()
    max_n = max(1, int(config.CAPACITY_MAX_CAMERAS_TESTED))
    h, w = config.CAPACITY_FRAME_SIZE
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    info = device_info()
    tag = f"engine={engine_id} {backend.label}"

    log.info("🧪 [CAPACITY] %s | calibrating real-time capacity on %s (%s logical cores%s)",
             tag, info["cpu"], info["logical_cores"],
             f", limit {info['cpu_limit_cores']}" if info["cpu_limit_cores"] else "")
    log.info("🧪 [CAPACITY] real-time = ≥%.0f fps per camera → one loop must finish in %.1f ms "
             "(%.1f ms frame period × %.2f safety margin). Measuring 1..%d cameras, "
             "%d timed loops each — no camera is running yet.",
             config.REALTIME_MIN_FPS, budget, period, config.CAPACITY_SAFETY_MARGIN, max_n,
             config.CAPACITY_ROUNDS)

    table: List[Dict[str, Any]] = []
    fails = 0
    t_start = time.perf_counter()
    for n in range(1, max_n + 1):
        backend.ensure_streams(n)
        frames = [frame] * n
        for _ in range(max(0, int(config.CAPACITY_WARMUP_ROUNDS))):
            backend.infer(frames)
        times = []
        for _ in range(max(3, int(config.CAPACITY_ROUNDS))):
            t0 = time.perf_counter()
            backend.infer(frames)
            times.append((time.perf_counter() - t0) * 1000.0)
        avg, p95 = float(np.mean(times)), _p95(times)
        ok = p95 <= budget
        table.append({"n": n, "avg_ms": round(avg, 1), "p95_ms": round(p95, 1),
                      "fps_per_camera": round(1000.0 / p95, 1) if p95 > 0 else 0.0, "ok": ok})
        log.info("🧪 [CAPACITY] %s | %2d camera%s: loop avg %6.1f ms  p95 %6.1f ms  → %5.1f fps/camera  %s",
                 tag, n, " " if n == 1 else "s", avg, p95, table[-1]["fps_per_camera"],
                 "✅ real-time" if ok else f"❌ too slow (budget {budget:.1f} ms)")
        fails = 0 if ok else fails + 1
        if fails >= max(1, int(config.CAPACITY_STOP_AFTER_FAILS)):
            break

    max_cameras = 0
    for row in table:
        if not row["ok"]:
            break
        max_cameras = row["n"]
    tested_up_to = table[-1]["n"]
    capped = max_cameras == tested_up_to == max_n

    profile = {
        "engine_id": engine_id, "backend": backend.label, "variant": getattr(backend.spec, "variant", ""),
        "input": getattr(backend, "imgsz_label", ""), **info,
        "target_fps": config.REALTIME_MIN_FPS, "safety_margin": config.CAPACITY_SAFETY_MARGIN,
        "detect_every_n": config.DETECT_EVERY_N_FRAMES, "budget_ms": round(budget, 1),
        "max_cameras": max_cameras, "max_is_lower_bound": capped, "tested_up_to": tested_up_to,
        "table": table, "calibration_seconds": round(time.perf_counter() - t_start, 1),
        "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if max_cameras == 0:
        log.error("🚨 [CAPACITY] %s | this machine CANNOT serve even 1 camera at ≥%.0f fps "
                  "(1 camera: p95 %.1f ms > budget %.1f ms). Use a faster model/variant, "
                  "more CPU, or lower REALTIME_MIN_FPS.", tag, config.REALTIME_MIN_FPS,
                  table[0]["p95_ms"], budget)
    else:
        log.info("🏁 [CAPACITY] %s | ✅ REAL-TIME CAPACITY: %s%d camera%s at ≥%.0f fps "
                 "(model %s, input %s; measured in %.0f s)",
                 tag, "at least " if capped else "", max_cameras, "" if max_cameras == 1 else "s",
                 config.REALTIME_MIN_FPS, profile["variant"] or backend.label, profile["input"],
                 profile["calibration_seconds"])
    return profile


def publish(bus, profile: Dict[str, Any]) -> None:
    try:
        bus.rt.set(bus.keys.detector_capacity, json.dumps(profile))
    except Exception:
        pass


def clear(bus) -> None:
    try:
        bus.rt.delete(bus.keys.detector_capacity)
    except Exception:
        pass


def load(bus) -> Optional[Dict[str, Any]]:
    try:
        raw = bus.rt.get(bus.keys.detector_capacity)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def check_camera(profile: Optional[Dict[str, Any]], cameras_after: int, camera_id: str):
    """(level, message) for the bridge to log when a camera is attached.
    level: 'info' (within capacity), 'warning' (over it), None (no profile)."""
    if not profile or "max_cameras" not in profile:
        return None, ""
    cap = int(profile["max_cameras"])
    what = f"{profile.get('variant') or profile.get('backend')} @ ≥{profile.get('target_fps', 25):.0f} fps"
    if cameras_after <= cap:
        return "info", (f"✅ [CAPACITY] camera {camera_id} attached: {cameras_after}/{cap} real-time cameras "
                        f"({what}) — {cap - cameras_after} more fit")
    if profile.get("max_is_lower_bound"):
        return "info", (f"ℹ️ [CAPACITY] camera {camera_id} attached: {cameras_after} cameras is above the range "
                        f"that was measured (all {cap} tested counts were real-time, {what}) — unverified")
    return "warning", (f"🚨 [CAPACITY] camera {camera_id} attached: {cameras_after} cameras > real-time capacity "
                       f"{cap} ({what}) — NOT real-time, expect missed frames on every camera. "
                       f"See the 🧪 table in the engine-0 startup log, or lower the load "
                       f"(fewer cameras / DETECT_EVERY_N_FRAMES / lighter model)")
