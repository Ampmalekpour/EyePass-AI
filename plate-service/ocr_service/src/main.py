"""
main.py (ocr_service)
--------------------------------------------------------------------
Entry point. Wires RedisBus -> OcrPool -> ServiceLifecycle -> health
endpoint, then self-heals: reads the last known phase (idle/processing)
and worker count from Redis and gets itself back there, without
waiting on any external command.

Why the OCR service doesn't run its own {module}:cmd:ai:request
listener: that BRPOP queue is a single-consumer contract (the backend
pops-and-routes one activate/deactivate camera command per item), and
it is entirely about camera activation — the detector's concern. The
OCR service stays camera-agnostic; it only ever needs ONE bit of
information from the detector — "is any camera active right now" —
delivered two ways:

  * live: subscribed to RedisKeys.demand_events, a pub/sub channel the
    detector publishes to from backend_bridge's
    handle_activated/handle_deactivated (the exact moment
    active_cameras transitions between empty and non-empty).
  * at boot: pub/sub only fires on a transition, so an OCR service that
    starts up fresh WHILE cameras are already active would otherwise
    never hear about it — self_heal() below is followed by one direct
    read of the active_cameras ledger to cover exactly that case.

A cold boot with no checkpoint AND no active cameras stays IDLE on
purpose. self_heal() restoring a "processing" checkpoint from before a
restart, or an operator's own explicit pause, both still take
precedence the normal way self_heal() already handles.
--------------------------------------------------------------------
"""

import logging
import signal
import sys
import threading
import time

import config
from platecore.active_state import ActiveCameraState
from platecore.bus import RedisBus
from platecore.health import serve_health
from platecore.lifecycle import PHASE_STOPPED, ServiceLifecycle
from platecore.logging_setup import setup_logger

from pool import OcrPool

logger = setup_logger("ocr_service.main")


def main():
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.info(
        "Starting plate OCR service | models_dir=%s | use_gpu=%s | default_workers=%d",
        config.OCR_MODELS_DIR, config.OCR_USE_GPU, config.DEFAULT_WORKER_COUNT,
    )

    bus = RedisBus(module=config.REDIS_MODULE)
    bus.wait_until_available()

    lifecycle_ref = {}

    def on_topology_changed():
        lc = lifecycle_ref.get("lifecycle")
        if lc is not None:
            lc.checkpoint_now()

    pool = OcrPool(on_topology_changed=on_topology_changed)

    lifecycle = ServiceLifecycle(
        bus=bus,
        state_key=bus.keys.ocr_state,
        on_start_idle=pool.start_idle,
        on_stop_idle=pool.stop_idle,
        on_start_process=pool.start_process,
        on_stop_process=pool.stop_process,
        get_unit_count=pool.get_worker_count,
        default_count=config.DEFAULT_WORKER_COUNT,
        unit_name="worker",
    )
    lifecycle_ref["lifecycle"] = lifecycle

    def on_demand_event(payload: dict):
        is_processing = bool(payload.get("processing"))
        if is_processing:
            lifecycle.start_process()
        else:
            lifecycle.stop_process()

    bus.subscribe(bus.keys.demand_events, on_demand_event, thread_name="demand-events")

    def heartbeat_loop():
        while True:
            bus.heartbeat(bus.keys.ocr_heartbeat, ttl_seconds=config.HEARTBEAT_TTL_SEC)
            time.sleep(config.HEARTBEAT_INTERVAL_SEC)

    threading.Thread(target=heartbeat_loop, daemon=True, name="heartbeat").start()

    def snapshot() -> dict:
        return {
            "status": "ok" if lifecycle.phase != PHASE_STOPPED else "starting",
            "phase": lifecycle.phase,
            "workers": pool.get_worker_count(),
            "pool": pool.snapshot(),
            "models_dir": config.OCR_MODELS_DIR,
            "use_gpu": config.OCR_USE_GPU,
        }

    serve_health(config.API_PORT, snapshot)

    # ---- self-healing: read the last known state and get back to it --
    lifecycle.self_heal()

    # Reality check against the durable active_cameras ledger, same
    # reasoning as the detector: covers a fresh OCR service deployment
    # (or one that missed the live pub/sub event) starting up while
    # cameras are already active. No-op if self_heal() already got
    # there; never demotes on its own.
    if ActiveCameraState(bus).all_active():
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

    logger.info("Shutting down — stopping worker pool")
    lifecycle.stop_idle()
    logger.info("OCR service shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
