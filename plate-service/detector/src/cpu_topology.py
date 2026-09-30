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
     ask for, GIVEN HOW MANY ENGINES ARE EXPECTED TO RUN AT THE SAME
     TIME                                     -> torch_num_threads

  3. for the OpenVINO / ONNX backends: how many parallel requests
     (streams) and how many inference threads each engine's model
     gets                        -> cpu_streams, cpu_infer_threads
     (see openvino_plan(); torch_num_threads is unused on that path)

main.py calls plan() exactly once at startup, before EngineManager is
constructed — topology is fixed for the life of the process; change
CPU_ENGINE_MODE (or any of its inputs) and restart to re-plan.

WHY torch_num_threads DIVIDES BY EXPECTED CONCURRENT ENGINES
------------------------------------------------------------------
The first version of this file computed torch_num_threads as "however
many cores are usable", on the unstated assumption that only one
engine process would ever be running at once. That assumption breaks
the instant camera count exceeds max_cameras_per_engine and a 2nd/3rd
engine spins up: every engine then tries to claim the FULL thread pool
simultaneously, and the box thrashes (measured: infer time going from
~40ms to 600-700ms the moment a 3rd single-camera engine started).
CPU_MAX_CONCURRENT_ENGINES/CPU_EXPECTED_CAMERAS (config.py) exist so
every engine's thread count is planned against how many peers it will
actually have, not against a best case that stops being true the
moment a second camera is added.

This is a deliberately simple linear model, not a live profiler. CPU
inference cost scales roughly linearly with images-per-batch (unlike
GPU, there is no occupancy curve to fill), so

    engine_batch_ms ~= n_cameras * (CPU_MS_PER_CAMERA_FRAME / N)

is a reasonable first-order estimate for "auto" mode's batching-capacity
side. CPU_MS_PER_CAMERA_FRAME is a config value seeded from the
detector's own [STATS] "infer avg" / batch_size log line, not measured
live in-process — recalibrate it after changing DETECTION_IMG_SIZE, the
model file, or the CPU generation.

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


def _ceil_div(a: int, b: int) -> int:
    return max(1, -(-max(1, a) // max(1, b)))


def total_cores() -> int:
    """CPU_CORES_OVERRIDE if set, else os.cpu_count(). Overriding
    matters because os.cpu_count() reports the box's/VM's full logical
    core count regardless of any Docker `--cpus` quota on this
    container -- see config.py's CPU_CORES_OVERRIDE comment for why an
    un-overridden planner under a quota reproduces the exact
    wild-latency oversubscription symptom this module exists to fix."""
    return int(config.CPU_CORES_OVERRIDE) or (os.cpu_count() or 4)


def usable_cores() -> int:
    """Logical cores left over after CPU_RESERVE_CORES is set aside for
    OS overhead and the per-camera RTSP capture/decode threads, out of
    total_cores() (the real/overridden count, not necessarily
    os.cpu_count())."""
    return max(1, total_cores() - max(0, config.CPU_RESERVE_CORES))


def auto_max_cameras_per_engine() -> int:
    """How many cameras one engine can batch per loop iteration and
    still keep up with CAMERA_ASSUMED_FPS, at the current
    DETECT_EVERY_N_FRAMES. Never exceeds the operator's own
    MAX_CAMERAS_PER_ENGINE ceiling. This is the batching-capacity
    question only -- it says nothing about how many such engines can
    run concurrently; see plan() for that half."""
    n = max(1, config.DETECT_EVERY_N_FRAMES)
    effective_ms_per_camera = max(1e-3, config.CPU_MS_PER_CAMERA_FRAME / n)
    frame_budget_ms = 1000.0 / max(1e-3, config.CAMERA_ASSUMED_FPS)
    margin = max(0.05, min(1.0, config.CPU_AUTO_SAFETY_MARGIN))
    budget_ms = frame_budget_ms * margin
    cap = max(1, math.floor(budget_ms / effective_ms_per_camera))
    return max(1, min(cap, max(1, config.MAX_CAMERAS_PER_ENGINE)))


def openvino_plan(expected_cameras: int, max_cams: int, expected_engines: int, backend: str) -> dict:
    """Parallel requests + threads for one engine's OpenVINO/ONNX model.

    streams: one per camera the engine carries (OpenVINO NUM_STREAMS /
      infer requests, or ONNX Runtime sessions) -- the engine starts one
      async request per camera frame and waits for all of them.
    threads: OpenVINO INFERENCE_NUM_THREADS / ORT intra_op_num_threads.
      openvino, one engine, no CPU_CORES_OVERRIDE -> 0 = let OpenVINO
      choose (forcing the full LOGICAL core count measured a single
      inference 2x slower: 15 ms vs 7.2 ms, work spread onto E-cores and
      hyper-threads). Several engines -> physical cores / engines, so
      they share the cores instead of each claiming all of them. Under a
      Docker quota (CPU_CORES_OVERRIDE) -> that budget / engines.
      onnx: sessions don't coordinate, so always cores / (engines * sessions)."""
    per_engine_cams = max(1, min(max_cams, _ceil_div(expected_cameras, expected_engines)))
    budget = int(config.CPU_CORES_OVERRIDE) or physical_cores()
    if backend == "onnx":
        threads = max(1, budget // max(1, expected_engines * per_engine_cams))
    elif config.CPU_CORES_OVERRIDE or expected_engines > 1:
        threads = max(1, budget // max(1, expected_engines))
    else:
        threads = 0
    return {"cpu_streams": per_engine_cams, "cpu_infer_threads": threads}


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return os.cpu_count() or 4


def plan(device: str, log: logging.Logger = None, backend: str = "pt") -> dict:
    """Returns the effective {mode, max_cameras_per_engine,
    torch_num_threads, expected_concurrent_engines, cpu_streams,
    cpu_infer_threads} for this box. `device` is the already-resolved
    "cpu"/"cuda[:N]" string and `backend` the CPU backend (see
    engine.resolve_runtime) -- on anything but "cpu" this is a no-op
    passthrough of the existing GPU config, so calling this
    unconditionally at startup is safe."""
    log = log or logger

    if not str(device).startswith("cpu"):
        result = {
            "mode": "gpu",
            "max_cameras_per_engine": max(1, config.MAX_CAMERAS_PER_ENGINE),
            "torch_num_threads": config.TORCH_NUM_THREADS,
            "expected_concurrent_engines": 1,
            "cpu_streams": 0,
            "cpu_infer_threads": 0,
        }
        log.info("[CPU-TOPOLOGY] device=%s -> GPU batching unchanged "
                  "(max_cameras_per_engine=%d)", device, result["max_cameras_per_engine"])
        return result

    host_cores = os.cpu_count() or 4
    cores = total_cores()
    usable = usable_cores()
    mode = config.CPU_ENGINE_MODE
    ceiling = max(1, config.CPU_MAX_CONCURRENT_ENGINES)
    expected_cameras = max(1, config.CPU_EXPECTED_CAMERAS)

    if mode == "multi":
        max_cams = max(1, config.CPU_MULTI_CAMERAS_PER_ENGINE)
        expected_engines = _ceil_div(expected_cameras, max_cams)
        if expected_engines > ceiling:
            log.warning(
                "[CPU-TOPOLOGY] mode=multi expects %d cameras / %d per engine = %d "
                "concurrent engines, above CPU_MAX_CONCURRENT_ENGINES=%d for this box "
                "-- dividing threads by %d anyway (each engine gets fewer threads than "
                "it would alone), but you will oversubscribe once that many engines are "
                "actually running. Consider CPU_ENGINE_MODE=auto instead, which folds "
                "batches together to stay within the concurrency ceiling.",
                expected_cameras, max_cams, expected_engines, ceiling, expected_engines,
            )
        threads = config.CPU_MULTI_TORCH_THREADS or max(1, usable // expected_engines)
        result = {
            "mode": "multi",
            "max_cameras_per_engine": max_cams,
            "torch_num_threads": threads,
            "expected_concurrent_engines": expected_engines,
        }
    elif mode == "auto":
        max_cams = auto_max_cameras_per_engine()
        expected_engines = _ceil_div(expected_cameras, max_cams)
        if expected_engines > ceiling:
            # Respect the concurrency ceiling over the pure batching-capacity
            # estimate: fold more cameras into each engine (bigger, slower
            # batches) rather than let more engines run than the box can
            # actually parallelize -- this is the fix for exactly the
            # collapse observed going from 2 to 3 concurrent engines.
            max_cams = _ceil_div(expected_cameras, ceiling)
            expected_engines = _ceil_div(expected_cameras, max_cams)
        threads = max(1, usable // expected_engines)
        result = {
            "mode": "auto",
            "max_cameras_per_engine": max_cams,
            "torch_num_threads": threads,
            "expected_concurrent_engines": expected_engines,
        }
    else:
        if mode != "single":
            log.warning("[CPU-TOPOLOGY] unknown CPU_ENGINE_MODE=%r, falling back to 'single'", mode)
        result = {
            "mode": "single",
            "max_cameras_per_engine": max(1, config.MAX_CAMERAS_PER_ENGINE),
            "torch_num_threads": config.TORCH_NUM_THREADS or usable,
            "expected_concurrent_engines": 1,
        }

    result.update(openvino_plan(expected_cameras, result["max_cameras_per_engine"],
                                result["expected_concurrent_engines"], backend))

    if config.CPU_CORES_OVERRIDE:
        log.info("[CPU-TOPOLOGY] using CPU_CORES_OVERRIDE=%d (host/container reports os.cpu_count()=%d "
                  "-- set this to match your Docker --cpus limit, they must agree)",
                  cores, host_cores)
    elif host_cores >= 8:
        log.warning("[CPU-TOPOLOGY] CPU_CORES_OVERRIDE is unset and os.cpu_count()=%d -- if this "
                    "container has a Docker --cpus limit (DETECTOR_CPU_LIMIT) below that, or shares "
                    "the host with other CPU-heavy services (OCR workers, publishers, backend/frontend), "
                    "set CPU_CORES_OVERRIDE to the real budget or this planner will oversize thread "
                    "pools and you'll see the same wild bursty [INFER-SLOW] latencies this exists to fix.",
                    host_cores)

    log.info(
        "[CPU-TOPOLOGY] mode=%s cores=%d reserved=%d usable=%d expected_cameras=%d "
        "concurrency_ceiling=%d -> max_cameras_per_engine=%d expected_engines=%d "
        "torch_num_threads=%s (detect_every_n=%d ms_per_camera=%.1f assumed_fps=%.1f "
        "safety_margin=%.2f)",
        result["mode"], cores, config.CPU_RESERVE_CORES, usable, expected_cameras, ceiling,
        result["max_cameras_per_engine"], result["expected_concurrent_engines"],
        result["torch_num_threads"] or "auto(cpu_count-1)",
        config.DETECT_EVERY_N_FRAMES, config.CPU_MS_PER_CAMERA_FRAME,
        config.CAMERA_ASSUMED_FPS, config.CPU_AUTO_SAFETY_MARGIN,
    )
    if backend != "pt":
        log.info("[CPU-TOPOLOGY] backend=%s -> per engine: streams=%s%s threads=%s%s "
                 "(torch_num_threads above is unused — PyTorch does not compute on this path)",
                 backend,
                 config.CPU_STREAMS or result["cpu_streams"], " (DETECTION_CPU_STREAMS)" if config.CPU_STREAMS else "",
                 config.CPU_INFER_THREADS or result["cpu_infer_threads"] or "auto(OpenVINO chooses)",
                 " (DETECTION_CPU_THREADS)" if config.CPU_INFER_THREADS else "")
        load = expected_cameras * config.CAMERA_ASSUMED_FPS / max(1, config.DETECT_EVERY_N_FRAMES)
        log.info("[CPU-TOPOLOGY] planned load: %d cameras x %.0f fps / DETECT_EVERY_N_FRAMES=%d = "
                 "%.0f inferences/s — keep this under ~60-70%% of the capacity you measured "
                 "(README 'CPU pipeline')", expected_cameras, config.CAMERA_ASSUMED_FPS,
                 max(1, config.DETECT_EVERY_N_FRAMES), load)
    return result
