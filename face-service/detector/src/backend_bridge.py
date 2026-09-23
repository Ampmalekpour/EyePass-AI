"""
backend_bridge.py
--------------------------------------------------------------------
Everything that talks to the backend Redis contract on the detector's
behalf: activate/deactivate commands, camera online/offline events,
startup reconciliation, and the offline-grace-period bookkeeping.
Ported from the heatmap reference's api.py (redis_command_listener,
_on_camera_event, _reconcile_on_startup, offline timers), adapted to
face's richer per-camera config (cross_line, stop_roi, per-trigger
flags) instead of heatmap's ROI-only config — see
alpr_api.py._build_video_config in the reference material for the
field mapping this follows.

EngineManager only knows how to place a camera once told to; this
module is what decides WHEN a camera should be attached or detached,
using the durable `active_cameras` ledger as the source of truth so a
restart (self-healing) replays exactly what was running before.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import config
from facecore.active_state import ActiveCameraState
from facecore.bus import RedisBus
from facecore.logging_setup import setup_logger

logger = setup_logger("backend_bridge")


def rtsp_url(camera_id: str) -> str:
    """Frames always come from the relay, never from the camera directly."""
    return f"{config.MTX_RTSP_BASE_URL.rstrip('/')}/{camera_id}"


def _bbox_to_points(bbox: Optional[dict]) -> Tuple[float, float, float, float, float, float, float, float]:
    if not bbox:
        return (0, 0, 0, 0, 0, 0, 0, 0)
    x, y, w, h = bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h")
    if None in (x, y, w, h):
        return (0, 0, 0, 0, 0, 0, 0, 0)
    x, y, w, h = float(x), float(y), float(w), float(h)
    return (x, y, x + w, y, x + w, y + h, x, y + h)


@dataclass
class CameraJob:
    camera_id: str
    video_path: str
    roi: Tuple[float, float, float, float]
    line_p1: Tuple[float, float]
    line_p2: Tuple[float, float]
    stop_roi: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    cond_per_trig: bool
    cross_line_trig: bool
    stop_roi_trig: bool
    leave_scene_trig: bool


def build_camera_job(camera_id: str, cfg: dict) -> CameraJob:
    roi_d = cfg.get("roi") or {}
    cl = cfg.get("cross_line") or {}
    cl_start = cl.get("start") or {}
    cl_end = cl.get("end") or {}
    sr = _bbox_to_points(cfg.get("stop_roi"))

    return CameraJob(
        camera_id=str(camera_id),
        video_path=rtsp_url(camera_id),
        roi=(
            float(roi_d.get("x", 0)), float(roi_d.get("y", 0)),
            float(roi_d.get("w", 1)), float(roi_d.get("h", 1)),
        ),
        line_p1=(float(cl_start.get("x", 0)), float(cl_start.get("y", 0))),
        line_p2=(float(cl_end.get("x", 0)), float(cl_end.get("y", 0))),
        stop_roi=((sr[0], sr[1]), (sr[2], sr[3]), (sr[4], sr[5]), (sr[6], sr[7])),
        cond_per_trig=True,
        cross_line_trig=bool(cfg.get("cross_line")),
        stop_roi_trig=bool(cfg.get("stop_roi")),
        leave_scene_trig=True,
    )


class DetectorBridge:
    def __init__(self, bus: RedisBus, engine_manager,
                 on_demand_changed: Optional[Callable[[bool], None]] = None):
        self.bus = bus
        self.engine_manager = engine_manager
        self.active_state = ActiveCameraState(bus)
        # Called with True the moment the first camera becomes active,
        # False the moment the last one is removed. main.py uses this to
        # drive the detector's OWN idle<->processing lifecycle and to
        # tell the (camera-agnostic) recognizer to do the same — see
        # RedisKeys.demand_events.
        self._on_demand_changed = on_demand_changed

        self._running_cameras: set = set()
        self._running_lock = threading.RLock()

        self._offline_timers: Dict[str, threading.Timer] = {}
        self._offline_lock = threading.Lock()

        self._camera_status: Dict[str, dict] = {}
        self._status_lock = threading.RLock()

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
            return False

        self.engine_manager.add_camera(
            camera_id=job.camera_id, url=job.video_path, roi=job.roi,
            cond_per_trig=job.cond_per_trig, cross_line_trig=job.cross_line_trig,
            stop_roi_trig=job.stop_roi_trig, leave_scene_trig=job.leave_scene_trig,
            line_p1_x=job.line_p1[0], line_p1_y=job.line_p1[1],
            line_p2_x=job.line_p2[0], line_p2_y=job.line_p2[1],
            stop_roi_p1_x=job.stop_roi[0][0], stop_roi_p1_y=job.stop_roi[0][1],
            stop_roi_p2_x=job.stop_roi[1][0], stop_roi_p2_y=job.stop_roi[1][1],
            stop_roi_p3_x=job.stop_roi[2][0], stop_roi_p3_y=job.stop_roi[2][1],
            stop_roi_p4_x=job.stop_roi[3][0], stop_roi_p4_y=job.stop_roi[3][1],
        )
        with self._running_lock:
            self._running_cameras.add(str(camera_id))
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
    # Idle <-> processing demand (driven purely by "is any camera
    # marked active", not by whether it's currently connected/attached
    # — a camera waiting to reconnect is still demand; see
    # RedisKeys.demand_events docstring).
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
    # Backend cmd request/response (face:cmd:ai:request/response)
    # ================================================================
    def handle_activated(self, camera_id: str, request_id: Optional[str]) -> bool:
        details = self.bus.get_camera_details(camera_id)
        if not details or not details.get("address"):
            logger.error("activated: no cameras:details (or no address) for camera %s", camera_id)
            return False

        cfg = self.bus.get_camera_config(camera_id) or details
        roi_dict = cfg.get("roi") or {}

        # Durable desired state FIRST. If we die between here and the
        # engine actually starting, the next startup reconcile picks it
        # back up — this is the self-healing checkpoint for cameras.
        self.active_state.mark_active(camera_id, roi_dict, request_id, extra={"config": cfg})
        self._notify_demand()

        self._cancel_pending_offline(camera_id)

        if self.is_running(camera_id):
            return True

        if not details.get("connected", True):
            # Activated but the camera is down right now. That is a
            # valid state, not an error: we wait for its online event.
            logger.info("Camera %s activated while offline — waiting for its online event", camera_id)
            return True

        return self._attach(camera_id, cfg)

    def handle_deactivated(self, camera_id: str) -> bool:
        self.active_state.mark_inactive(camera_id)
        self._notify_demand()
        self._cancel_pending_offline(camera_id)
        self._detach(camera_id)
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
    # camera_stream events (face:cameras:events)
    # ================================================================
    def on_camera_event(self, payload: dict):
        camera_id = str(payload.get("id", "")).strip()
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
    # Startup reconciliation — "resume where it left off"
    # ================================================================
    def reconcile_on_startup(self):
        self.bus.wait_until_available()
        wanted = self.active_state.all_active()
        logger.info("Startup reconcile: %d camera(s) marked active", len(wanted))

        for camera_id, entry in wanted.items():
            try:
                details = self.bus.get_camera_details(camera_id) or {}
                if not details.get("connected"):
                    logger.info("Camera %s is active but currently offline — waiting for its online event", camera_id)
                    continue
                cfg = self.bus.get_camera_config(camera_id) or entry.get("config") or details
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
    # Status (informational only — never drives stop/restart; that is
    # entirely the camera_stream event + offline-grace path above)
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
