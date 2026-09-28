"""
cpu_topology.py
--------------------------------------------------------------------
CPU-only capacity planning for the detector's engine pool. Answers two
questions from os.cpu_count() plus the CPU_* knobs in config.py's
"CPU ENGINE TOPOLOGY" section:

  1. how many cameras can ONE engine process realistically batch
     through model.predict() per frame interval, at the configured
     DETECT_EVERY_N_FRAMES cadence, without falling behind the
     camera's own frame rate                 -> max_cameras_per_engine

  2. how many torch intra-op threads a single engine process should
     ask for                                 -> torch_num_threads

main.py calls plan() exactly once at startup, before EngineManager is
constructed — topology is fixed for the life of the process; change
CPU_ENGINE_MODE (or any of its inputs) and restart to re-plan.

This is a deliberately simple linear model, not a live profiler. CPU
inference cost scales roughly linearly with images-per-batch (unlike
GPU, there is no occupancy curve to fill), so

    engine_batch_ms ~= n_cameras * (CPU_MS_PER_CAMERA_FRAME / N)

is a reasonable first-order estimate for "auto" mode. CPU_MS_PER_CAMERA_FRAME
is a config value seeded from the detector's own [STATS] "infer avg" /
batch_size log line, not measured live in-process — recalibrate it after
changing DETECTION_IMG_SIZE, the model file, or the CPU generation.

Only ever consulted when the resolved device is "cpu"; GPU topology
(MAX_CAMERAS_PER_ENGINE, TORCH_NUM_THREADS as configured) is untouched
by anything in this file — see main.py's call site.
--------------------------------------------------------------------
"""

import logging
import math
import os

import config

logger = logging.getLogger("cpu_topology")


def usable_cores() -> int:
    """Logical cores left over after CPU_RESERVE_CORES is set aside for
    OS overhead and the per-camera RTSP capture/decode threads."""
    total = os.cpu_count() or 4
    return max(1, total - max(0, config.CPU_RESERVE_CORES))


def auto_max_cameras_per_engine() -> int:
    """How many cameras one engine can batch per loop iteration and
    still keep up with CAMERA_ASSUMED_FPS, at the current
    DETECT_EVERY_N_FRAMES. Never exceeds the operator's own
    MAX_CAMERAS_PER_ENGINE ceiling."""
    n = max(1, config.DETECT_EVERY_N_FRAMES)
    effective_ms_per_camera = max(1e-3, config.CPU_MS_PER_CAMERA_FRAME / n)
    frame_budget_ms = 1000.0 / max(1e-3, config.CAMERA_ASSUMED_FPS)
    margin = max(0.05, min(1.0, config.CPU_AUTO_SAFETY_MARGIN))
    budget_ms = frame_budget_ms * margin
    cap = max(1, math.floor(budget_ms / effective_ms_per_camera))
    return max(1, min(cap, max(1, config.MAX_CAMERAS_PER_ENGINE)))


def plan(device: str, log: logging.Logger = None) -> dict:
    """Returns the effective {mode, max_cameras_per_engine,
    torch_num_threads} for this box. `device` is the already-resolved
    "cpu"/"cuda[:N]" string (see engine.resolve_device) -- on anything
    but "cpu" this is a no-op passthrough of the existing GPU config,
    so calling this unconditionally at startup is safe."""
    log = log or logger

    if not str(device).startswith("cpu"):
        result = {
            "mode": "gpu",
            "max_cameras_per_engine": max(1, config.MAX_CAMERAS_PER_ENGINE),
            "torch_num_threads": config.TORCH_NUM_THREADS,
        }
        log.info("[CPU-TOPOLOGY] device=%s -> GPU batching unchanged "
                  "(max_cameras_per_engine=%d)", device, result["max_cameras_per_engine"])
        return result

    cores = os.cpu_count() or 4
    usable = usable_cores()
    mode = config.CPU_ENGINE_MODE

    if mode == "multi":
        result = {
            "mode": "multi",
            "max_cameras_per_engine": max(1, config.CPU_MULTI_CAMERAS_PER_ENGINE),
            "torch_num_threads": max(1, config.CPU_MULTI_TORCH_THREADS),
        }
    elif mode == "auto":
        result = {
            "mode": "auto",
            "max_cameras_per_engine": auto_max_cameras_per_engine(),
            # the common case is one engine active at a time under "auto"
            # (cameras <= max_cameras_per_engine) -- give its thread pool
            # everything not reserved for capture/OS.
            "torch_num_threads": config.TORCH_NUM_THREADS or usable,
        }
    else:
        if mode != "single":
            log.warning("[CPU-TOPOLOGY] unknown CPU_ENGINE_MODE=%r, falling back to 'single'", mode)
            mode = "single"
        result = {
            "mode": "single",
            "max_cameras_per_engine": max(1, config.MAX_CAMERAS_PER_ENGINE),
            "torch_num_threads": config.TORCH_NUM_THREADS or usable,
        }

    log.info(
        "[CPU-TOPOLOGY] mode=%s cores=%d reserved=%d usable=%d -> "
        "max_cameras_per_engine=%d torch_num_threads=%s "
        "(detect_every_n=%d ms_per_camera=%.1f assumed_fps=%.1f safety_margin=%.2f)",
        result["mode"], cores, config.CPU_RESERVE_CORES, usable,
        result["max_cameras_per_engine"], result["torch_num_threads"] or "auto(cpu_count-1)",
        config.DETECT_EVERY_N_FRAMES, config.CPU_MS_PER_CAMERA_FRAME,
        config.CAMERA_ASSUMED_FPS, config.CPU_AUTO_SAFETY_MARGIN,
    )
    return result
