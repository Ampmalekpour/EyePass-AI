"""
main.py (detector)
--------------------------------------------------------------------
Entry point. Wires RedisBus -> EngineManager -> ServiceLifecycle ->
DetectorBridge -> health endpoint, then self-heals: reads the last
known phase (idle/processing) and engine count from Redis and gets
itself back there, without waiting for the backend to resend anything.

Startup does NOT block on Redis being reachable forever — it retries
via bus.wait_until_available(), same discipline as the reference
alpr_api.py (bind the health port, come up idle, reconcile once Redis
answers).
--------------------------------------------------------------------
"""

import json
import logging
import os
import signal
import sys
import threading
import time

import config
from platecore.bus import RedisBus
from platecore.health import serve_health
from platecore.lifecycle import PHASE_STOPPED, ServiceLifecycle
from platecore.logging_setup import setup_logger

import cpu_topology
from backend_bridge import DetectorBridge
from engine import resolve_device
from engine_manager import EngineManager

logger = setup_logger("detector.main")


def main():
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # Resolved once, here, in the parent process, purely to plan engine
    # topology (a read-only torch.cuda.is_available() probe -- the
    # parent never builds a model or runs inference itself). Each engine
    # subprocess still calls resolve_device() again for itself in
    # Engine.__init__, which remains the source of truth for what
    # device IT actually runs on -- see config.py's DETECTION_DEVICE
    # comment for why this is safe with spawn-based children.
    # See cpu_topology.py: on a GPU box this is a no-op passthrough of
    # the existing MAX_CAMERAS_PER_ENGINE/TORCH_NUM_THREADS config; on a
    # CPU box, CPU_ENGINE_MODE (single/multi/auto) decides how many
    # cameras share one engine process and how many torch threads each
    # engine asks for.
    planned_device = resolve_device(config.DETECTION_DEVICE, config.STRICT_DEVICE, logger)
    topology = cpu_topology.plan(planned_device, logger)
    max_cameras_per_engine = topology["max_cameras_per_engine"]
    if topology["torch_num_threads"]:
        # Engine subprocesses are spawned (mp.get_context("spawn")) after
        # this point and inherit the parent's environment, so each one's
        # own `config.TORCH_NUM_THREADS` read in Engine.__init__ picks
        # this up without any change to the spawn call itself.
        os.environ["TORCH_NUM_THREADS"] = str(int(topology["torch_num_threads"]))

    logger.info(
        "Starting plate detector | device preference=%s resolved=%s cpu_engine_mode=%s | "
        "rtsp base=%s | max_cameras_per_engine=%d (was config default %d)",
        config.DETECTION_DEVICE, planned_device, config.CPU_ENGINE_MODE,
        config.MTX_RTSP_BASE_URL, max_cameras_per_engine, config.MAX_CAMERAS_PER_ENGINE,
    )

    bus = RedisBus(module=config.REDIS_MODULE)
    bus.wait_until_available()

    # `lifecycle` is referenced by on_topology_changed before it exists,
    # so it is created first and populated once the ServiceLifecycle
    # object is built.
    lifecycle_ref = {}

    def on_topology_changed():
        lc = lifecycle_ref.get("lifecycle")
        if lc is not None:
            lc.checkpoint_now()

    engine_manager = EngineManager(
        model_path=config.MODEL_PATH,
        imgsz=config.IMG_SIZE,
        conf=config.CONF_THRESHOLD,
        save_output=config.DEBUG_VIDEO_ENABLED,
        output_dir=config.DEBUG_VIDEO_DIR,
        class_labels=config.CLASS_LABELS,
        max_cameras_per_engine=max_cameras_per_engine,
        on_topology_changed=on_topology_changed,
        engine_shutdown_timeout=config.ENGINE_SHUTDOWN_TIMEOUT_SEC,
    )

    def on_demand_changed(is_processing: bool):
        # Drives the detector's OWN idle<->processing switch, and tells
        # the (camera-agnostic) OCR service to mirror it. Fires only
        # from a live activate/deactivate command — see backend_bridge's
        # handle_activated/handle_deactivated — never from boot.
        lc = lifecycle_ref.get("lifecycle")
        if lc is not None:
            if is_processing:
                lc.start_process()
            else:
                lc.stop_process()
        try:
            bus.rt.publish(bus.keys.demand_events, json.dumps({"processing": is_processing}))
        except Exception:
            logger.exception("failed to publish demand event")

    bridge = DetectorBridge(bus, engine_manager, on_demand_changed=on_demand_changed)

    def on_start_process():
        engine_manager.start_process()  # currently a no-op, kept for symmetry
        bridge.reconcile_on_startup()

    def on_stop_process():
        bridge.stop_all()
        engine_manager.stop_process()

    lifecycle = ServiceLifecycle(
        bus=bus,
        state_key=bus.keys.detector_state,
        on_start_idle=engine_manager.start_idle,
        on_stop_idle=engine_manager.stop_idle,
        on_start_process=on_start_process,
        on_stop_process=on_stop_process,
        get_unit_count=engine_manager.get_engine_count,
        default_count=config.DEFAULT_ENGINE_COUNT,
        unit_name="engine",
    )
    lifecycle_ref["lifecycle"] = lifecycle

    # ---- background threads -------------------------------------------
    threading.Thread(target=bridge.monitor_engine_status_loop, daemon=True, name="engine-status").start()
    threading.Thread(target=bridge.cmd_worker_loop, daemon=True, name="cmd-worker").start()
    bus.subscribe(bus.keys.cameras_events, bridge.on_camera_event, thread_name="camera-events")

    def rebalance_loop():
        while True:
            time.sleep(config.ENGINE_REBALANCE_INTERVAL_SEC)
            try:
                engine_manager.rebalance()
            except Exception:
                logger.exception("engine rebalance failed")

    threading.Thread(target=rebalance_loop, daemon=True, name="rebalance").start()

    def heartbeat_loop():
        while True:
            bus.heartbeat(bus.keys.detector_heartbeat, ttl_seconds=config.HEARTBEAT_TTL_SEC)
            time.sleep(config.HEARTBEAT_INTERVAL_SEC)

    threading.Thread(target=heartbeat_loop, daemon=True, name="heartbeat").start()

    def snapshot() -> dict:
        return {
            "status": "ok" if lifecycle.phase != PHASE_STOPPED else "starting",
            "phase": lifecycle.phase,
            "engines": engine_manager.get_engine_count(),
            "cameras_running": bridge.running_camera_count(),
            "topology": engine_manager.snapshot(),
            "camera_status": bridge.status_snapshot(),
            "device_preference": config.DETECTION_DEVICE,
        }

    serve_health(config.API_PORT, snapshot)

    # ---- self-healing: read the last known state and get back to it --
    lifecycle.self_heal()

    # A truly cold boot (no checkpoint, no cameras ever activated) stays
    # IDLE on purpose — nothing to process yet. But if the durable
    # active_cameras ledger already has entries (this container
    # restarted, or was redeployed fresh, while the backend still
    # considers cameras active) there IS demand right now regardless of
    # what self_heal() found, so promote — this also runs
    # reconcile_on_startup() to actually re-attach them. No-op if
    # self_heal() already got there first.
    if bridge.active_state.all_active():
        lifecycle.start_process()

    # ---- graceful shutdown ----------------------------------------------
    stop_flag = threading.Event()

    def handle_signal(signum, _frame):
        logger.info("received signal %s — shutting down", signum)
        stop_flag.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not stop_flag.is_set():
        time.sleep(1)

    logger.info("Shutting down — stopping engines (this flushes any open debug video writers)")
    lifecycle.stop_idle()
    logger.info("Detector shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
