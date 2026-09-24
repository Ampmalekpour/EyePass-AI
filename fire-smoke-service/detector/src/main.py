"""
main.py (detector)
--------------------------------------------------------------------
Entry point. Wires RedisBus -> EngineManager -> ServiceLifecycle ->
DetectorBridge -> health endpoint, then self-heals: reads the last
known phase (idle/processing) and engine count from Redis and gets
itself back there, without waiting for the backend to resend anything.

Ported from plate_detector's/face_detector's main.py — this module is
single-service (no paired OCR/recognizer worker to notify of demand
changes), so the plate/face module's `demand_events` publish was
dropped; everything else (self-healing, startup reconcile, rebalance,
heartbeat, graceful shutdown) is unchanged.
--------------------------------------------------------------------
"""

import logging
import signal
import sys
import threading
import time

import config
from firecore.bus import RedisBus
from firecore.health import serve_health
from firecore.lifecycle import PHASE_STOPPED, ServiceLifecycle
from firecore.logging_setup import setup_logger

from backend_bridge import DetectorBridge
from engine_manager import EngineManager

logger = setup_logger("detector.main")


def main():
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.info(
        "Starting fire/smoke detector | device preference=%s | rtsp base=%s | max_cameras_per_engine=%d",
        config.DETECTION_DEVICE, config.MTX_RTSP_BASE_URL, config.MAX_CAMERAS_PER_ENGINE,
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
        max_cameras_per_engine=config.MAX_CAMERAS_PER_ENGINE,
        on_topology_changed=on_topology_changed,
        engine_shutdown_timeout=config.ENGINE_SHUTDOWN_TIMEOUT_SEC,
    )

    def on_demand_changed(is_processing: bool):
        lc = lifecycle_ref.get("lifecycle")
        if lc is not None:
            if is_processing:
                lc.start_process()
            else:
                lc.stop_process()

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
    # IDLE on purpose. But if the durable active_cameras ledger already
    # has entries (this container restarted, or was redeployed fresh,
    # while the backend still considers cameras active) there IS demand
    # right now regardless of what self_heal() found, so promote — this
    # also runs reconcile_on_startup() to actually re-attach them. No-op
    # if self_heal() already got there first.
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
