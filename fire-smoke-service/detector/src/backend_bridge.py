"""
backend_bridge.py
--------------------------------------------------------------------
Everything that talks to the backend Redis contract on the detector's
behalf: activate/deactivate commands, camera online/offline events,
startup reconciliation, the offline-grace-period bookkeeping, the
per-camera `ai_status` hash Django reads, and the bounded internal-
processing-error auto-restart.

Ported from plate_detector's/face_detector's backend_bridge.py — the
camera-lifecycle logic here is fully generic (durable active_state
ledger written before acting, offline-grace timer, demand-events
notification, cmd request/response worker, startup reconcile); trimmed
of the plate-only cross_line/stop_roi trigger-geometry fields this
module has no equivalent of. Fire/smoke cameras only need a ROI (the
2x2 spatial grid is computed inside that ROI — see engine.py).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import config
from firecore.active_state import ActiveCameraState
from firecore.bus import RedisBus
from firecore.logging_setup import setup_logger
from firecore.relay import relay_path_for

logger = setup_logger("backend_bridge")


def rtsp_url(camera_id: str, info: Optional[dict] = None) -> str:
    """Frames always come from the suite's shared MediaMTX relay, never
    from the camera directly. The relay path is the one camera_stream
    registered for this camera: `relay_path` from cameras:details when
    known, else derived from the camera's address with the same rule
    (firecore/relay.py) — one path per physical camera, shared by every
    module that uses it."""
    info = info or {}
    path = info.get("relay_path") or relay_path_for(camera_id, info.get("address"))
    return f"{config.MTX_RTSP_BASE_URL.rstrip('/')}/{path}"


@dataclass
class CameraJob:
    camera_id: str
    video_path: str
    roi: Tuple[float, float, float, float]


def build_camera_job(camera_id: str, cfg: dict) -> CameraJob:
    """cfg is exactly the JSON object stored per-camera in
    `fire:cameras:config` (roi: {x,y,w,h}). address comes from
    cameras:details (or from cfg itself if the backend already inlines
    it there)."""
    roi_d = cfg.get("roi") or {}
    return CameraJob(
        camera_id=str(camera_id),
        video_path=rtsp_url(camera_id, cfg),
        roi=(
            float(roi_d.get("x", 0)), float(roi_d.get("y", 0)),
            float(roi_d.get("w", 1)), float(roi_d.get("h", 1)),
        ),
    )


class DetectorBridge:
    def __init__(self, bus: RedisBus, engine_manager,
                 on_demand_changed: Optional[Callable[[bool], None]] = None):
        self.bus = bus
        self.engine_manager = engine_manager
        self.active_state = ActiveCameraState(bus)
        # Called with True the moment the first camera becomes active,
        # False the moment the last one is removed — drives this
        # service's own idle<->processing lifecycle (see main.py).
        self._on_demand_changed = on_demand_changed

        self._running_cameras: set = set()
        self._running_lock = threading.RLock()

        self._offline_timers: Dict[str, threading.Timer] = {}
        self._offline_lock = threading.Lock()

        self._camera_status: Dict[str, dict] = {}
        self._status_lock = threading.RLock()

        # ---- internal-processing-error auto-restart --------------------
        self._error_retry_state: Dict[str, Dict[str, Any]] = {}
        self._error_retry_lock = threading.Lock()

    # ================================================================
    # Attach / detach
    # ================================================================
    def is_running(self, camera_id: str) -> bool:
        with self._running_lock:
            return str(camera_id) in self._running_cameras

    def running_camera_count(self) -> int:
        with self._running_lock:
            return len(self._running_cameras)

    def _attach(self, camera_id: str, cfg: dict) -> bool:
        try:
            job = build_camera_job(camera_id, cfg)
        except Exception as e:
            logger.error("bad config for camera %s: %s", camera_id, e)
            self.bus.write_ai_status(camera_id, "stopped", error=f"bad config: {e}")
            return False

        self.engine_manager.add_camera(camera_id=job.camera_id, url=job.video_path, roi=job.roi)
        with self._running_lock:
            self._running_cameras.add(str(camera_id))
        # Written immediately, right after handing the camera to the
        # engine — Django only cares "did we accept responsibility for
        # this camera", not "has it produced output yet".
        self.bus.write_ai_status(camera_id, "running")
        logger.info("Attached camera %s -> %s", camera_id, job.video_path)
        return True

    def _detach(self, camera_id: str):
        self.engine_manager.remove_camera(camera_id)
        with self._running_lock:
            self._running_cameras.discard(str(camera_id))
        logger.info("Detached camera %s", camera_id)

    # ================================================================
    # Offline grace period
    # ================================================================
    def _cancel_pending_offline(self, camera_id: str):
        with self._offline_lock:
            timer = self._offline_timers.pop(camera_id, None)
        if timer:
            timer.cancel()

    def _handle_offline_expired(self, camera_id: str):
        with self._offline_lock:
            if self._offline_timers.get(camera_id) is None:
                return  # an online event already cancelled this
            self._offline_timers.pop(camera_id, None)

        logger.warning(
            "Camera %s stayed offline past the %.1fs grace period — detaching it "
            "(stays in active_cameras and restarts on its next online event)",
            camera_id, config.CAMERA_OFFLINE_GRACE_SECONDS,
        )
        self._detach(camera_id)
        self.bus.write_ai_status(camera_id, "stopped", error="Camera disconnected")

    def _schedule_offline_stop(self, camera_id: str):
        with self._offline_lock:
            if camera_id in self._offline_timers:
                return  # already counting down
            timer = threading.Timer(
                config.CAMERA_OFFLINE_GRACE_SECONDS, self._handle_offline_expired, args=(camera_id,)
            )
            timer.daemon = True
            self._offline_timers[camera_id] = timer
        timer.start()

    # ================================================================
    # Idle <-> processing demand
    # ================================================================
    def _notify_demand(self):
        if self._on_demand_changed is None:
            return
        any_active = bool(self.active_state.all_active())
        try:
            self._on_demand_changed(any_active)
        except Exception:
            logger.exception("on_demand_changed callback failed")

    # ================================================================
    # Backend cmd request/response (fire:cmd:ai:request/response)
    # ================================================================
    def handle_activated(self, camera_id: str, request_id: Optional[str]) -> bool:
        details = self.bus.get_camera_details(camera_id)
        cfg = self.bus.get_camera_config(camera_id) or details
        if not cfg:
            logger.error("activated: no cameras:config for camera %s", camera_id)
            return False
        roi_dict = cfg.get("roi") or {}

        # Durable desired state FIRST. If we die between here and the
        # engine actually starting, the next startup reconcile picks it
        # back up — this is the self-healing checkpoint for cameras.
        self.active_state.mark_active(camera_id, roi_dict, request_id, extra={"config": cfg})
        self._notify_demand()

        self._cancel_pending_offline(camera_id)
        self._reset_error_retry(camera_id)

        if self.is_running(camera_id):
            return True

        if details is not None and not details.get("connected", True):
            # Activated but the camera is down right now. That is a
            # valid state, not an error: we wait for its online event.
            logger.info("Camera %s activated while offline — waiting for its online event", camera_id)
            return True

        return self._attach(camera_id, cfg)

    def handle_deactivated(self, camera_id: str) -> bool:
        self.active_state.mark_inactive(camera_id)
        self._notify_demand()
        self._cancel_pending_offline(camera_id)
        self._reset_error_retry(camera_id)
        self._detach(camera_id)
        # Distinct status from an automatic stop (disconnect/error) —
        # Django's UI distinguishes "we turned it off" from "it fell
        # over on its own".
        self.bus.write_ai_status(camera_id, "stopped_by_user")
        return True

    def cmd_worker_loop(self):
        logger.info("cmd worker started | request_key=%s", self.bus.keys.cmd_request)
        while True:
            request = self.bus.pop_request(timeout=2)
            if request is None:
                continue

            request_id = request.get("request_id")
            camera_id = str(request.get("camera_id", "")).strip()
            action = (request.get("action") or "").lower().strip()

            if not camera_id:
                logger.error("request missing camera_id: %s", request)
                self.bus.send_response(request_id, "ERROR")
                continue

            try:
                if action == "activated":
                    ok = self.handle_activated(camera_id, request_id)
                elif action == "deactivated":
                    ok = self.handle_deactivated(camera_id)
                else:
                    logger.warning("unknown action %r for camera %s", action, camera_id)
                    ok = False
            except Exception:
                logger.exception("failed to handle request %s", request)
                ok = False

            self.bus.send_response(request_id, "OK" if ok else "ERROR")

    # ================================================================
    # camera_stream events (fire:camera:events — singular, see keys.py)
    # ================================================================
    def on_camera_event(self, payload: dict):
        camera_id = str(payload.get("id", payload.get("camera_id", ""))).strip()
        connected = payload.get("connected")
        if not camera_id:
            return

        logger.info("Camera event | id=%s connected=%s error=%s", camera_id, connected, payload.get("error"))

        if connected:
            # Only cameras the backend actually activated may be started
            # — otherwise a deactivated camera would come back to life
            # on its next online transition.
            if not self.active_state.is_active(camera_id):
                logger.debug("Camera %s is online but not activated — ignoring", camera_id)
                return
            self._cancel_pending_offline(camera_id)
            if self.is_running(camera_id):
                return
            cfg = self.bus.get_camera_config(camera_id)
            if not cfg:
                stored = self.active_state.get(camera_id) or {}
                cfg = stored.get("config") or {}
            self._attach(camera_id, cfg)
        else:
            if not self.is_running(camera_id):
                return
            self._schedule_offline_stop(camera_id)

    # ================================================================
    # Internal-processing-error auto-restart (distinct from the
    # offline-grace path above — this is for an Engine's own
    # status="error" report, e.g. a batch-inference exception or a
    # transient resource shortage, not a physical camera disconnect).
    # ================================================================
    def _reset_error_retry(self, camera_id: str):
        with self._error_retry_lock:
            self._error_retry_state.pop(camera_id, None)

    def handle_engine_error(self, camera_id: str, error: Optional[str]):
        now = time.time()

        with self._error_retry_lock:
            state = self._error_retry_state.get(camera_id)
            if state is not None and (now - state.get("last_attempt_ts", 0)) > config.ERROR_RESTART_COUNTER_RESET_AFTER_SEC:
                # Previous error is old enough that this looks like a
                # fresh problem — give it a full set of retries again.
                state = None
            if state is None:
                state = {"count": 0, "last_attempt_ts": now}
                self._error_retry_state[camera_id] = state

            if state["count"] >= config.ERROR_RESTART_MAX_RETRIES:
                logger.warning(
                    "[ERROR_RESTART] camera %s: max retries (%d) reached, staying stopped. last_error=%s",
                    camera_id, config.ERROR_RESTART_MAX_RETRIES, error,
                )
                self.bus.write_ai_status(camera_id, "stopped", error=error)
                return

            state["count"] += 1
            state["last_attempt_ts"] = now
            attempt_no = state["count"]

        # Detach immediately so the engine doesn't keep re-erroring on
        # this camera every tick while we wait to retry.
        self._cancel_pending_offline(camera_id)
        self._detach(camera_id)
        self.bus.write_ai_status(camera_id, "stopped", error=error)

        def _worker():
            time.sleep(config.ERROR_RESTART_DELAY_SEC)

            if not self.active_state.is_active(camera_id):
                # Deactivated while we were waiting — don't fight that.
                return

            cfg = self.bus.get_camera_config(camera_id)
            if not cfg:
                stored = self.active_state.get(camera_id) or {}
                cfg = stored.get("config") or {}
            if not cfg:
                logger.warning("[ERROR_RESTART] camera %s: config not found, cannot restart", camera_id)
                self.bus.write_ai_status(camera_id, "stopped", error=error or "config not found")
                return

            logger.info(
                "[ERROR_RESTART] camera %s: restart attempt %d/%d",
                camera_id, attempt_no, config.ERROR_RESTART_MAX_RETRIES,
            )
            if self._attach(camera_id, cfg):
                logger.info("[ERROR_RESTART] camera %s: restart succeeded", camera_id)
            else:
                logger.warning("[ERROR_RESTART] camera %s: restart failed", camera_id)

        threading.Thread(target=_worker, daemon=True, name=f"error-restart-{camera_id}").start()

    # ================================================================
    # Startup reconciliation — "resume where it left off"
    # ================================================================
    def reconcile_on_startup(self):
        self.bus.wait_until_available()
        wanted = self.active_state.all_active()
        logger.info("Startup reconcile: %d camera(s) marked active", len(wanted))

        for camera_id, entry in wanted.items():
            try:
                details = self.bus.get_camera_details(camera_id)
                if details is not None and not details.get("connected", True):
                    logger.info("Camera %s is active but currently offline — waiting for its online event", camera_id)
                    continue
                cfg = self.bus.get_camera_config(camera_id) or entry.get("config") or details
                if not cfg:
                    logger.warning("Camera %s is active but has no config anywhere — skipping", camera_id)
                    continue
                self._attach(camera_id, cfg)
            except Exception:
                logger.exception("Failed to restore camera %s", camera_id)

        logger.info("Startup reconcile complete")

    def stop_all(self):
        with self._offline_lock:
            for timer in self._offline_timers.values():
                timer.cancel()
            self._offline_timers.clear()
        with self._running_lock:
            camera_ids = list(self._running_cameras)
        for cid in camera_ids:
            self._detach(cid)

    # ================================================================
    # Status (informational — actual stop/restart is driven entirely by
    # the offline-grace path and handle_engine_error() above)
    # ================================================================
    def monitor_engine_status_loop(self):
        q = self.engine_manager.status_queue
        while True:
            try:
                msg = q.get(timeout=0.5)
            except Exception:
                continue
            cam_id = str(msg.get("camera_id", "")).strip()
            if not cam_id:
                continue
            with self._status_lock:
                entry = self._camera_status.setdefault(cam_id, {})
                if "frames_processed" in msg:
                    entry["frames_processed"] = msg["frames_processed"]
                if "last_frame_ts" in msg:
                    entry["last_frame_ts"] = msg["last_frame_ts"]
                    entry["last_heartbeat_ts"] = time.time()
                if "status" in msg:
                    entry["status"] = msg["status"]
                if "error" in msg:
                    entry["last_error"] = msg["error"]

            status = msg.get("status")
            if status == "running":
                self._reset_error_retry(cam_id)
            elif status == "error":
                try:
                    self.handle_engine_error(cam_id, msg.get("error"))
                except Exception:
                    logger.exception("handle_engine_error failed for camera %s", cam_id)

    def status_snapshot(self) -> Dict[str, dict]:
        now = time.time()
        with self._status_lock:
            out = {}
            for cid, entry in self._camera_status.items():
                e = dict(entry)
                last_hb = e.get("last_heartbeat_ts")
                e["degraded"] = bool(
                    e.get("status") == "running" and last_hb is not None
                    and (now - float(last_hb)) > config.HEARTBEAT_TIMEOUT_SEC
                )
                out[cid] = e
            return out
