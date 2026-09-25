"""
enroll.py (recognizer) — NEW, add-face
--------------------------------------------------------------------
`EnrollCoordinator` is the recognizer's counterpart to the detector's
`backend_bridge.cmd_worker_loop` — a single-threaded, sequential
BRPOP-and-reply loop against the backend contract, just against
`cmd:enroll:request`/`response` instead of `cmd:ai:request`/`response`.

It does NO image processing itself. Its whole job is coordination:

  * pull one enroll command off Redis
  * for a camera-capture request, grab exactly one frame from the
    MediaMTX relay (never the camera directly — same rule the
    detector follows)
  * dispatch the real work as a task onto the SAME `rec:tasks` queue
    every live recognition crop already goes through, using a
    synthetic engine_id (`enroll:{request_id}`) so the result routes
    back here and nowhere else
  * for a commit, hold the distributed gallery lock for the round trip
    and publish `gallery:updated` on success
  * reply on `cmd:enroll:response:{request_id}`

Runs on a daemon thread started from main.py, in the recognizer's MAIN
process — never inside a RecognitionWorker subprocess. It shares no
state with the worker pool beyond Redis itself.
--------------------------------------------------------------------
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import cv2

import config
from facecore.bus import RedisBus
from facecore.codec import decode_task, encode_task
from facecore.logging_setup import setup_logger
from facecore.relay import relay_path_for

logger = setup_logger("recognizer.enroll")


def rtsp_url(camera_id: str, info: Optional[dict] = None) -> str:
    """Frames always come from the shared relay, never the camera
    directly — identical rule to detector/src/backend_bridge.py::rtsp_url
    (facecore/relay.py)."""
    info = info or {}
    path = info.get("relay_path") or relay_path_for(camera_id, info.get("address"))
    return f"{config.MTX_RTSP_BASE_URL.rstrip('/')}/{path}"


def _grab_camera_snapshot(camera_id: str, info: Optional[dict] = None) -> bytes:
    """One-shot RTSP grab: open, read a frame, close. Deliberately NOT
    a persistent connection or a shared frame buffer — enrollment is
    infrequent enough that paying connection setup cost per call is
    the right trade against the complexity of keeping a live buffer
    warm for something this rare. Mirrors the reference's
    `capture_snapshot_with_retry`, minus the buffer-first fast path
    that no longer has anything to be a fast path in front of."""
    url = rtsp_url(camera_id, info)
    last_error = None
    for attempt in range(config.ENROLL_SNAPSHOT_RETRIES):
        cap = cv2.VideoCapture(url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not cap.isOpened():
                last_error = f"failed to open {url}"
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                last_error = f"failed to read a frame from {url}"
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                last_error = "failed to encode captured frame"
                continue
            return buf.tobytes()
        finally:
            cap.release()
    raise RuntimeError(f"snapshot capture failed for camera {camera_id} after "
                        f"{config.ENROLL_SNAPSHOT_RETRIES} attempts: {last_error}")


class EnrollCoordinator:
    def __init__(self, bus: RedisBus):
        self.bus = bus
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="enroll-coordinator")
        self._thread.start()
        logger.info("EnrollCoordinator started | request_key=%s", self.bus.keys.enroll_request)

    # ------------------------------------------------------------------
    def _loop(self):
        while True:
            raw = self.bus.pop_enroll_request(timeout=2)
            if raw is None:
                continue
            try:
                cmd = decode_task(raw)
            except Exception:
                logger.exception("failed to decode enroll request, dropping")
                continue

            request_id = cmd.get("request_id")
            action = (cmd.get("action") or "").strip()
            logger.info("enroll request | action=%s request_id=%s", action, request_id)

            try:
                if action == "verify_pose":
                    reply = self._handle_verify_pose(cmd)
                elif action == "commit":
                    reply = self._handle_commit(cmd)
                else:
                    reply = {"status": "error", "message": f"unknown action: {action!r}"}
            except Exception as e:
                logger.exception("enroll request %s failed", request_id)
                reply = {"status": "error", "message": str(e)}

            self.bus.send_enroll_response(request_id, reply)

    # ------------------------------------------------------------------
    # verify_pose — dispatch to a worker, wait for its verdict
    # ------------------------------------------------------------------
    def _handle_verify_pose(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        request_id = cmd.get("request_id")
        flag = cmd.get("flag")
        if flag not in config.ENROLL_YAW_WINDOWS:
            return {"status": "error", "message": f"invalid flag {flag!r}, expected one of {sorted(config.ENROLL_YAW_WINDOWS)}"}

        image_bytes = cmd.get("image_bytes")
        camera_id = cmd.get("camera_id")
        if image_bytes is None and camera_id is not None:
            try:
                image_bytes = _grab_camera_snapshot(str(camera_id), self.bus.get_camera_details(str(camera_id)))
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if image_bytes is None:
            return {"status": "error", "message": "request must supply image_bytes or camera_id"}

        engine_id = f"enroll:{request_id}"
        task = {"task_type": "enroll_pose_check", "engine_id": engine_id, "image_bytes": image_bytes, "flag": flag}
        self.bus.push_task(encode_task(task))

        raw_result = self.bus.pop_result_bytes(engine_id, timeout=int(config.ENROLL_TASK_TIMEOUT_SEC))
        if raw_result is None:
            return {"status": "error", "message": "pose-check timed out — no worker picked up the task in time"}

        verdict = decode_task(raw_result)
        verdict["status"] = "ok"
        return verdict

    # ------------------------------------------------------------------
    # commit — locked, dispatch to a worker, publish gallery:updated
    # ------------------------------------------------------------------
    def _handle_commit(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        request_id = cmd.get("request_id")
        person = cmd.get("person") or {}
        crops = cmd.get("crops") or []

        if len(crops) != config.ENROLL_IMAGES_PER_PERSON:
            return {
                "status": "error",
                "message": f"expected {config.ENROLL_IMAGES_PER_PERSON} approved crops, got {len(crops)}",
            }
        required_fields = ("name", "lastname", "section", "codeid", "personnelid")
        missing = [f for f in required_fields if not person.get(f)]
        if missing:
            return {"status": "error", "message": f"person is missing required field(s): {missing}"}

        lock = self.bus.gallery_lock(
            timeout=config.ENROLL_LOCK_TIMEOUT_SEC,
            blocking_timeout=config.ENROLL_LOCK_BLOCKING_TIMEOUT_SEC,
        )
        acquired = lock.acquire(blocking=True)
        if not acquired:
            return {"status": "error", "message": "gallery is busy with another enrollment — try again shortly"}

        try:
            engine_id = f"enroll:{request_id}"
            task = {"task_type": "enroll_commit", "engine_id": engine_id, "person": person, "crops": crops}
            self.bus.push_task(encode_task(task))

            raw_result = self.bus.pop_result_bytes(engine_id, timeout=int(config.ENROLL_COMMIT_TASK_TIMEOUT_SEC))
            if raw_result is None:
                return {"status": "error", "message": "commit timed out — no worker picked up the task in time"}

            result = decode_task(raw_result)
            if result.get("status") == "ok":
                self.bus.publish_gallery_updated({
                    "personnelid": result.get("personnelid"),
                    "committed_at": time.time(),
                })
                logger.info("gallery:updated published for personnelid=%s", result.get("personnelid"))
            return result
        finally:
            try:
                lock.release()
            except Exception:
                logger.warning("gallery lock release failed (already expired?) for request %s", request_id)
