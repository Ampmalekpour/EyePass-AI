"""
api.py
--------------------------------------------------------------------
The application. Backend integration is entirely over Redis; the HTTP
endpoints here are for operations and debugging.

Carried over from the pre-existing standalone build's api.py (all of
its resilience properties are unchanged — durable desired state,
activation gating, graceful shutdown, one lock over `processes`,
nothing blocking startup on a dependency), restructured on top of
common/heatmapcore instead of this module's own redis_client.py /
ai_state.py:

  * RedisBus / ActiveCameraState come from heatmapcore, so this module
    shares its Redis key-naming and connection-retry code with the
    plate/face/fire modules instead of maintaining its own copy.
  * cameras:events is now the SINGULAR channel the bundled
    camera-service (the same one the other modules run) publishes.
  * ServiceLifecycle wraps engine startup so a self-healing checkpoint
    (`{module}:internal:detector:state`) is persisted, matching the
    other modules — the pre-existing build had no such checkpoint,
    relying solely on `ai:active` (camera-level desired state, unaffected
    by this addition) to resume after a restart.
  * per-camera `ai_status` is now written (engine.py), for parity with
    the Django-facing status field the other modules expose.
--------------------------------------------------------------------
"""

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import camera_registry
from config import (
    VIDEO_OUTPUT_DIR, SAVE_OUTPUT, SAVE_AS_VIDEO,
    CLASS_LABELS, MAX_CAMERAS_PER_ENGINE, HEARTBEAT_TIMEOUT_SEC,
    CAMERA_OFFLINE_GRACE_SECONDS, MTX_RTSP_BASE_URL,
    DETECTION_DEVICE, LOG_LEVEL, DEFAULT_ENGINE_COUNT,
    DETECTOR_DEBUG_VIDEO_ENABLED, DEBUG_VIDEO_DIR,
    DetectionConfig, StorageConfig, GridConfig,
)
from heatmapcore.bus import RedisBus
from heatmapcore.active_state import ActiveCameraState
from heatmapcore.lifecycle import ServiceLifecycle
from engine import EngineManager, CameraState

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_default_detection = DetectionConfig()
_default_storage = StorageConfig()
_default_grid = GridConfig()


# -----------------------------------------------------------------------------
# Process-local live state. Durable state lives in Redis ({module}:ai:active)
# and in camera_registry.json; this dict is only "what is running right now".
# Mutated from the Redis command thread, the camera-events thread and the HTTP
# handlers, so every access goes through _processes_lock.
# -----------------------------------------------------------------------------
processes: Dict[str, dict] = {}
_processes_lock = threading.RLock()

# camera_id -> threading.Timer for "went offline, waiting to see if it returns".
offline_timers: Dict[str, threading.Timer] = {}
_offline_timers_lock = threading.Lock()


class VideoConfig(BaseModel):
    video_path: str
    camera_id: int
    roi_x: float = 0
    roi_y: float = 0
    roi_width: float = 1
    roi_height: float = 1


class StopConfig(BaseModel):
    camera_id: int


def create_app() -> FastAPI:
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    if DETECTOR_DEBUG_VIDEO_ENABLED:
        os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Engine manager
    # The DEVICE PREFERENCE (auto|cpu|cuda|cuda:N) is passed down as a
    # string. It is resolved to a concrete torch device inside each
    # engine child process, so this parent never initialises CUDA.
    # ----------------------------------------------------------------
    engine_manager = EngineManager(
        model_path=_default_detection.model_path,
        imgsz=_default_detection.img_size,
        conf=_default_detection.conf_threshold,
        device=DETECTION_DEVICE,
        save_output=SAVE_OUTPUT,
        save_as_video=SAVE_AS_VIDEO,
        output_dir=VIDEO_OUTPUT_DIR,
        class_labels=CLASS_LABELS,
        max_cameras_per_engine=MAX_CAMERAS_PER_ENGINE,
        time_resolution_minutes=_default_storage.time_resolution_minutes,
        save_interval_seconds=_default_storage.save_interval_seconds,
        debug_video_enabled=DETECTOR_DEBUG_VIDEO_ENABLED,
        debug_video_dir=DEBUG_VIDEO_DIR,
    )

    bus = RedisBus()
    active_state = ActiveCameraState(bus)

    # Self-healing checkpoint: persists {phase, engine_count} to
    # {module}:internal:detector:state, matching the plate/face/fire
    # modules. Engine count here doesn't currently shrink/grow outside
    # camera churn (see engine.py's EngineManager), so on_start_idle
    # simply ensures at least `n` engines exist.
    lifecycle = ServiceLifecycle(
        bus=bus,
        state_key=bus.keys.detector_state,
        on_start_idle=lambda n: (engine_manager.start(initial_engines=n) or engine_manager.engine_count())
                                  if engine_manager.engine_count() == 0 else engine_manager.engine_count(),
        on_stop_idle=engine_manager.shutdown,
        on_start_process=lambda: None,   # camera attach is driven by ai:active reconcile below
        on_stop_process=lambda: None,
        get_unit_count=engine_manager.engine_count,
        default_count=DEFAULT_ENGINE_COUNT,
        unit_name="engine",
    )

    # Readiness flags surfaced by /health.
    ready = {"engines": False, "redis": False, "reconciled": False}

    # ================================================================
    # Helpers
    # ================================================================
    def _safe_set(camera_id: str, key: str, value: Any):
        with _processes_lock:
            if camera_id in processes:
                processes[camera_id][key] = value

    def _mark_status(camera_id: str, status: str):
        _safe_set(camera_id, "status", status)
        camera_registry.upsert_camera(camera_id, status=status)

    def _is_running(camera_id: str) -> bool:
        with _processes_lock:
            info = processes.get(camera_id)
            return bool(info and info.get("status") in ("running", "starting"))

    def _handle_msg(camera_id: str, msg: dict):
        if "frames_processed" in msg:
            _safe_set(camera_id, "frames_processed", msg["frames_processed"])

        if "last_frame_ts" in msg:
            _safe_set(camera_id, "last_frame_ts", msg["last_frame_ts"])
            _safe_set(camera_id, "last_heartbeat_ts", time.time())

        if "status" in msg:
            _mark_status(camera_id, msg["status"])
            mapping = {"running": "RUNNING", "stopped": "STOPPED", "error": "DEGRADED"}
            if msg["status"] in mapping:
                _safe_set(camera_id, "state_name", mapping[msg["status"]])

        if "error" in msg:
            _safe_set(camera_id, "last_error", msg["error"])

    def monitor_engine_status():
        q = engine_manager.status_queue
        while True:
            try:
                msg = q.get(timeout=0.5)
            except Exception:
                continue
            cam_id = str(msg.get("camera_id", "")).strip()
            if not cam_id:
                continue
            with _processes_lock:
                known = cam_id in processes
            if known:
                _handle_msg(cam_id, msg)

    def check_processes():
        """Heartbeat watchdog: flags a camera DEGRADED if the engine stops
        reporting frames. Does not stop anything — the offline path owns that."""
        while True:
            now = time.time()
            with _processes_lock:
                for camera_id, data in list(processes.items()):
                    last_hb = data.get("last_heartbeat_ts")
                    if data.get("status") == "running" and last_hb is not None:
                        if (now - float(last_hb)) > HEARTBEAT_TIMEOUT_SEC:
                            data["state_name"] = "DEGRADED"
                            data["last_error"] = data.get("last_error") or "No heartbeat"
            time.sleep(5)

    def _validate_source(video_source: str):
        if not video_source:
            raise HTTPException(status_code=400, detail="video_path is required")
        if not os.path.exists(video_source) and not video_source.startswith(("rtsp://", "http://", "https://")):
            raise HTTPException(status_code=400, detail=f"Video source not found or invalid: {video_source}")

    def _roi_tuple(roi_dict: Optional[dict]):
        roi_dict = roi_dict or {}
        return (
            float(roi_dict.get("x", 0)),
            float(roi_dict.get("y", 0)),
            float(roi_dict.get("w", 1)),
            float(roi_dict.get("h", 1)),
        )

    def _start_camera(camera_id: str, video_source: str, roi):
        task_id = str(uuid.uuid4())
        now = time.time()

        ret = engine_manager.add_camera(camera_id=camera_id, url=video_source, roi=roi)
        lifecycle.checkpoint_now()

        with _processes_lock:
            processes[camera_id] = {
                "proc": None,
                "state": CameraState(),
                "status": "running",
                "state_name": "RUNNING",
                "frames_processed": 0,
                "video_source": video_source,
                "task_id": task_id,
                "started_at": now,
                "last_heartbeat_ts": now,
                "last_frame_ts": None,
                "last_error": None,
                "engine_id": ret.get("engine_id"),
            }

        camera_registry.upsert_camera(
            camera_id,
            video_source=video_source,
            roi=list(roi),
            engine_id=ret.get("engine_id"),
            started_at=now,
            status="running",
        )

        logger.info("Started camera %s on engine %s (%s)", camera_id, ret.get("engine_id"), video_source)
        return task_id, ret

    def _stop_camera(camera_id: str) -> dict:
        with _processes_lock:
            info = processes.pop(camera_id, None)

        if not info:
            camera_registry.upsert_camera(camera_id, status="stopped")
            return {"camera_id": camera_id, "status": "not_running"}

        ret = engine_manager.remove_camera(camera_id)
        camera_registry.upsert_camera(camera_id, status="stopped")
        logger.info("Stopped camera %s", camera_id)
        return {"camera_id": camera_id, "status": ret.get("status", "stopped")}

    def _rtsp_url(camera_id: str) -> str:
        """Frames always come from the relay, never from the camera directly."""
        return f"{MTX_RTSP_BASE_URL.rstrip('/')}/{camera_id}"

    # ================================================================
    # Offline grace period
    # ================================================================
    def _cancel_pending_offline(camera_id: str):
        with _offline_timers_lock:
            timer = offline_timers.pop(camera_id, None)
        if timer:
            timer.cancel()

    def _handle_offline_expired(camera_id: str):
        with _offline_timers_lock:
            if offline_timers.get(camera_id) is None:
                return          # an online event already cancelled us
            offline_timers.pop(camera_id, None)

        logger.warning(
            "Camera %s stayed offline past the %.1fs grace period — stopping it "
            "(it stays in ai:active and will restart when it returns)",
            camera_id, CAMERA_OFFLINE_GRACE_SECONDS,
        )
        _stop_camera(camera_id)

    def _schedule_offline_stop(camera_id: str):
        with _offline_timers_lock:
            if camera_id in offline_timers:
                return          # already counting down
            timer = threading.Timer(CAMERA_OFFLINE_GRACE_SECONDS, _handle_offline_expired, args=(camera_id,))
            timer.daemon = True
            offline_timers[camera_id] = timer
        timer.start()

    # ================================================================
    # Redis command listener (backend -> us)
    # ================================================================
    def redis_command_listener():
        logger.info("Redis command listener started | request_key=%s", bus.keys.cmd_request)

        while True:
            request = bus.pop_request(timeout=2)
            if request is None:
                continue

            request_id = request.get("request_id")
            camera_id = str(request.get("camera_id", "")).strip()
            action = (request.get("action") or "").lower().strip()

            if not camera_id:
                logger.error("Request missing camera_id: %s", request)
                bus.send_response(request_id, status="ERROR")
                continue

            try:
                if action == "activated":
                    details = bus.get_camera_details(camera_id)
                    if not details or not details.get("address"):
                        logger.error("No cameras:details entry (or no address) for camera_id=%s", camera_id)
                        bus.send_response(request_id, status="ERROR")
                        continue

                    roi_dict = details.get("roi") or {}

                    # Record desired state FIRST. If we die between here and
                    # the engine starting, the startup reconcile picks it up.
                    active_state.mark_active(camera_id, roi_dict, request_id)

                    if _is_running(camera_id):
                        bus.send_response(request_id, status="OK")
                        continue

                    if not details.get("connected", True):
                        # Activated but the camera is down right now. That is
                        # a valid state, not an error: we wait for its online
                        # event. Answering OK is correct — we have accepted
                        # responsibility for the camera.
                        logger.info("Camera %s activated while offline — waiting for its online event", camera_id)
                        bus.send_response(request_id, status="OK")
                        continue

                    _start_camera(camera_id, _rtsp_url(camera_id), _roi_tuple(roi_dict))
                    bus.send_response(request_id, status="OK")

                elif action == "deactivated":
                    active_state.mark_inactive(camera_id)
                    _cancel_pending_offline(camera_id)
                    _stop_camera(camera_id)
                    bus.send_response(request_id, status="OK")

                else:
                    logger.warning("Unknown action %r in request: %s", action, request)
                    bus.send_response(request_id, status="ERROR")

            except Exception as e:
                logger.exception("Failed to handle request %s: %s", request, e)
                bus.send_response(request_id, status="ERROR")

    # ================================================================
    # Camera online/offline reaction (camera_stream -> us)
    # ================================================================
    def _on_camera_event(payload: dict):
        camera_id = str(payload.get("id", "")).strip()
        connected = payload.get("connected")

        if not camera_id:
            return

        logger.info("Camera event | id=%s connected=%s error=%s",
                    camera_id, connected, payload.get("error"))

        if connected:
            # Only cameras the backend actually activated may be started.
            # Without this check, a camera that was explicitly deactivated
            # comes back to life on its next online transition.
            if not active_state.is_active(camera_id):
                logger.debug("Camera %s is online but not activated — ignoring", camera_id)
                return

            _cancel_pending_offline(camera_id)

            if _is_running(camera_id):
                return

            roi_dict = payload.get("roi") or {}
            if not roi_dict:
                stored = active_state.all_active().get(camera_id) or {}
                roi_dict = stored.get("roi") or {}

            _start_camera(camera_id, _rtsp_url(camera_id), _roi_tuple(roi_dict))

        else:
            if not _is_running(camera_id):
                return
            _schedule_offline_stop(camera_id)

    # ================================================================
    # Startup reconciliation — the heart of "resume where we left off"
    # ================================================================
    def _reconcile_on_startup():
        bus.wait_until_available()
        ready["redis"] = True

        wanted = active_state.all_active()
        logger.info("Startup reconcile: %d camera(s) marked active", len(wanted))

        for camera_id, entry in wanted.items():
            try:
                details = bus.get_camera_details(camera_id) or {}
                if not details.get("connected"):
                    logger.info("Camera %s is active but currently offline — waiting for its online event", camera_id)
                    continue
                roi_dict = entry.get("roi") or details.get("roi") or {}
                _start_camera(camera_id, _rtsp_url(camera_id), _roi_tuple(roi_dict))
            except Exception as e:
                logger.error("Failed to restore camera %s: %s", camera_id, e)

        ready["reconciled"] = True
        logger.info("Startup reconcile complete")

    # ================================================================
    # Lifespan
    # ================================================================
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("Starting AI service | device preference=%s | rtsp base=%s",
                    DETECTION_DEVICE, MTX_RTSP_BASE_URL)

        lifecycle.self_heal()
        lifecycle.start_process()
        ready["engines"] = True

        threading.Thread(target=monitor_engine_status, daemon=True, name="engine-status").start()
        threading.Thread(target=check_processes, daemon=True, name="heartbeat-watchdog").start()
        threading.Thread(target=redis_command_listener, daemon=True, name="redis-commands").start()
        threading.Thread(target=_reconcile_on_startup, daemon=True, name="startup-reconcile").start()

        def _heartbeat_loop():
            while True:
                bus.heartbeat(bus.keys.detector_heartbeat, ttl_seconds=30)
                time.sleep(10)

        threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat").start()

        bus.subscribe_camera_events(_on_camera_event)

        yield

        # ---- shutdown -------------------------------------------------
        # Engine.cleanup() flushes every camera's cube to MinIO before the
        # process exits. stop_grace_period in compose must exceed
        # ENGINE_SHUTDOWN_TIMEOUT_SEC or Docker will SIGKILL mid-upload.
        logger.info("Shutting down — flushing cubes to MinIO")
        with _offline_timers_lock:
            for timer in offline_timers.values():
                timer.cancel()
            offline_timers.clear()
        try:
            lifecycle.stop_idle()
        except Exception as e:
            logger.error("Error during engine shutdown: %s", e)
        logger.info("Shutdown complete")

    app = FastAPI(title="Heatmap AI Service", lifespan=lifespan)

    # ================================================================
    # Endpoints
    # ================================================================
    @app.get("/health")
    async def health():
        """Liveness + readiness. Used by the container healthcheck, so it
        must stay cheap and must NOT fail just because a dependency is
        briefly unavailable — the service is designed to ride that out."""
        with _processes_lock:
            running = len(processes)
        return {
            "status": "ok" if ready["engines"] else "starting",
            "engines_started": ready["engines"],
            "engine_count": engine_manager.engine_count(),
            "phase": lifecycle.phase,
            "redis_reachable": bus.ping(),
            "startup_reconciled": ready["reconciled"],
            "cameras_running": running,
            "device_preference": DETECTION_DEVICE,
        }

    def _status_payload(camera_id: str, info: dict) -> Dict[str, Any]:
        alive = info.get("status") == "running"
        return {
            "camera_id": camera_id,
            "state": info.get("state_name") or ("RUNNING" if alive else "STOPPED"),
            "alive": alive,
            "engine_id": info.get("engine_id"),
            "status": info.get("status"),
            "frames_processed": info.get("frames_processed", 0),
            "video_source": info.get("video_source"),
            "started_at": info.get("started_at"),
            "last_heartbeat_ts": info.get("last_heartbeat_ts"),
            "last_frame_ts": info.get("last_frame_ts"),
            "last_error": info.get("last_error"),
        }

    @app.get("/status")
    async def status_all():
        with _processes_lock:
            return {cid: _status_payload(cid, info) for cid, info in processes.items()}

    @app.get("/status/{camera_id}")
    async def status_one(camera_id: str):
        with _processes_lock:
            info = processes.get(str(camera_id))
        if not info:
            return {"camera_id": str(camera_id), "state": "STOPPED", "alive": False}
        return _status_payload(str(camera_id), info)

    @app.get("/active")
    async def active_cameras():
        """What the backend has asked us to monitor (durable desired state),
        versus what is actually running. The first place to look when
        something is 'activated but not producing data'."""
        wanted = active_state.all_active()
        with _processes_lock:
            running = set(processes.keys())
        return {
            "activated": wanted,
            "running": sorted(running),
            "activated_but_not_running": sorted(set(wanted) - running),
            "running_but_not_activated": sorted(running - set(wanted)),
        }

    @app.get("/cameras")
    async def cameras_all():
        return camera_registry.all_cameras()

    @app.get("/cameras/{camera_id}")
    async def cameras_one(camera_id: str):
        entry = camera_registry.get_camera(camera_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"No registry entry for camera {camera_id}")
        return entry

    # ---- manual overrides, debugging only --------------------------
    @app.post("/start_video")
    async def start_processing(request: VideoConfig):
        camera_id = str(request.camera_id)
        video_source = (request.video_path or "").strip()
        roi = (float(request.roi_x), float(request.roi_y), float(request.roi_width), float(request.roi_height))

        _validate_source(video_source)

        if _is_running(camera_id):
            with _processes_lock:
                info = processes[camera_id]
            return {"camera_id": camera_id, "video_source": info.get("video_source"),
                    "status": "already_running", "engine_id": info.get("engine_id")}

        _cancel_pending_offline(camera_id)
        task_id, ret = _start_camera(camera_id, video_source, roi)
        return {"camera_id": camera_id, "video_source": video_source,
                "status": ret.get("status", "started"), "task_id": task_id,
                "engine_id": ret.get("engine_id")}

    @app.post("/stop_video")
    async def stop_video(cfg: StopConfig):
        camera_id = str(cfg.camera_id)
        _cancel_pending_offline(camera_id)
        ret = _stop_camera(camera_id)
        return {"camera_id": camera_id, "status": ret.get("status", "stopped")}

    @app.post("/reset_video")
    async def reset_video(request: VideoConfig):
        camera_id = str(request.camera_id)
        _cancel_pending_offline(camera_id)
        _stop_camera(camera_id)

        video_source = (request.video_path or "").strip() or _rtsp_url(camera_id)
        roi = (float(request.roi_x), float(request.roi_y), float(request.roi_width), float(request.roi_height))
        _validate_source(video_source)

        _, ret = _start_camera(camera_id, video_source, roi)
        return {"camera_id": camera_id, "video_source": video_source,
                "status": "started", "engine_id": ret.get("engine_id")}

    return app
