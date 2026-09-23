"""
main.py (recognizer)
--------------------------------------------------------------------
Entry point. Wires RedisBus -> RecognizerPool -> ServiceLifecycle ->
health endpoint, then self-heals: reads the last known phase
(idle/processing) and worker count from Redis and gets itself back
there, without waiting on any external command.

Why the recognizer doesn't run its own {module}:cmd:ai:request
listener: that BRPOP queue is a single-consumer contract (the backend
pops-and-routes one activate/deactivate camera command per item), and
it is entirely about camera activation — the detector's concern. The
recognizer stays camera-agnostic; it only ever needs ONE bit of
information from the detector — "is any camera active right now" —
delivered two ways:

  * live: subscribed to RedisKeys.demand_events, a pub/sub channel the
    detector publishes to from backend_bridge's
    handle_activated/handle_deactivated (the exact moment
    active_cameras transitions between empty and non-empty).
  * at boot: pub/sub only fires on a transition, so a recognizer that
    starts up fresh WHILE cameras are already active would otherwise
    never hear about it — self_heal() below is followed by one direct
    read of the active_cameras ledger to cover exactly that case.

A cold boot with no checkpoint AND no active cameras stays IDLE on
purpose. self_heal() restoring a "processing" checkpoint from before a
restart, or an operator's own explicit pause, both still take
precedence the normal way self_heal() already handles.

--------------------------------------------------------------------
ADD-FACE (2026-09): two additions, both independent of the
idle/processing lifecycle above (enrollment works whether or not any
camera is active — it never touches active_cameras/demand_events):

  * `EnrollCoordinator` (enroll.py) — the recognizer's OWN
    backend-contract listener, on `cmd:enroll:request` /
    `cmd:enroll:response`, exactly the way backend_bridge's
    cmd_worker_loop is the detector's listener on `cmd:ai:request` /
    `cmd:ai:response`. Started unconditionally, independent of
    lifecycle phase, same reasoning as the health endpoint below: an
    admin should be able to enroll someone whether or not any camera
    is currently streaming.

  * a subscription to `gallery:updated` — fired by EnrollCoordinator
    after a successful commit — that recycles the worker pool
    (pool.reload_gallery()) so every worker picks up the newly
    enrolled person. See pool.py's docstring for exactly what that
    does and costs.
--------------------------------------------------------------------
"""

import logging
import signal
import sys
import threading
import time

import config
from facecore.active_state import ActiveCameraState
from facecore.bus import RedisBus
from facecore.health import serve_health
from facecore.lifecycle import ServiceLifecycle
from facecore.logging_setup import setup_logger

from enroll import EnrollCoordinator
from pool import RecognizerPool

logger = setup_logger("recognizer.main")


def main():
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.info(
        "Starting face recognizer | model=%s | use_onnx=%s | default_workers=%d",
        config.MODEL_NAME, config.USE_ONNX, config.DEFAULT_WORKER_COUNT,
    )

    bus = RedisBus(module=config.REDIS_MODULE)
    bus.wait_until_available()

    lifecycle_ref = {}

    def on_topology_changed():
        lc = lifecycle_ref.get("lifecycle")
        if lc is not None:
            lc.checkpoint_now()

    pool = RecognizerPool(on_topology_changed=on_topology_changed)

    lifecycle = ServiceLifecycle(
        bus=bus,
        state_key=bus.keys.recognizer_state,
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

    # ---- ADD-FACE: gallery reload on a successful enrollment --------
    def on_gallery_updated(payload: dict):
        logger.info("gallery:updated received (personnelid=%s) — reloading pool", payload.get("personnelid"))
        try:
            pool.reload_gallery()
        except Exception:
            logger.exception("reload_gallery() failed after gallery:updated")

    bus.subscribe(bus.keys.gallery_updated, on_gallery_updated, thread_name="gallery-updated")

    def heartbeat_loop():
        while True:
            bus.heartbeat(bus.keys.recognizer_heartbeat, ttl_seconds=30)
            time.sleep(10)

    threading.Thread(target=heartbeat_loop, daemon=True, name="heartbeat").start()

    def snapshot() -> dict:
        return {
            "status": "ok" if lifecycle.phase != "stopped" else "starting",
            "phase": lifecycle.phase,
            "workers": pool.get_worker_count(),
            "pool": pool.snapshot(),
            "model": config.MODEL_NAME,
            "use_onnx": config.USE_ONNX,
        }

    serve_health(config.API_PORT, snapshot)

    # ---- self-healing: read the last known state and get back to it --
    lifecycle.self_heal()

    # Reality check against the durable active_cameras ledger, same
    # reasoning as the detector: covers a fresh recognizer deployment
    # (or one that missed the live pub/sub event) starting up while
    # cameras are already active. No-op if self_heal() already got
    # there; never demotes on its own.
    if ActiveCameraState(bus).all_active():
        lifecycle.start_process()

    # ---- ADD-FACE: enrollment listener, independent of camera phase --
    # A separate RedisBus instance: EnrollCoordinator's loop blocks on
    # BRPOP against cmd:enroll:request AND pop_result_bytes against a
    # per-request result list, both using the same underlying
    # connection pool as everything else in `bus` — giving it its own
    # RedisBus keeps its blocking calls from ever contending with the
    # heartbeat/subscribe threads' use of `bus.rt` for a connection.
    enroll_bus = RedisBus(module=config.REDIS_MODULE)
    enroll_coordinator = EnrollCoordinator(enroll_bus)
    enroll_coordinator.start()

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
    logger.info("Recognizer shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
