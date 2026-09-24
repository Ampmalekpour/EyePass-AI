"""
engine.py
--------------------------------------------------------------------
The fire/smoke detection engine — one YOLO model per process, handling
however many cameras EngineManager assigns it via batched Ultralytics
inference. Ported from the pre-existing video_processor.py's `Engine`
class (already close to the plate/face module shape — RTSP reader
thread, background I/O worker, per-camera state) onto the shared
platform pattern used by plate_detector/face_detector:

  * resolve_device() / STRICT_DEVICE / TORCH_NUM_THREADS — same CPU/GPU
    resolution as the other two modules, instead of a hardcoded
    "cuda if available else cpu".
  * rtsp_reader.RTSPStreamReader — the shared, env-tunable reader
    (reconnect backoff, FFmpeg socket timeout) instead of the
    module-local reader class the pre-port version had.
  * results are pushed through firecore.bus.RedisBus.push_result()
    onto `{module}:detections:results`, instead of a raw
    `redis.Redis.from_url(...).rpush(...)` call — same wire format
    (event_type/status/camera_id/timestamp/saved_file_path/region_id),
    just going through the shared bus so heartbeat/backend-contract
    code all share one Redis connection convention.
  * an optional annotated debug video recorder (DEBUG_VIDEO_ENABLED),
    matching the plate/face modules' local-bind-mount debug output —
    see debug_recorder.py.

Detection logic itself (2x2 spatial grid, per-region cascading
threat-verification window, global cooldown/resolution) is UNCHANGED
from the pre-existing video_processor.py — that state machine was
already correct, just needed wiring into the shared platform.
--------------------------------------------------------------------
"""

from __future__ import annotations

import collections
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from debug_recorder import DebugRecorder
from firecore.bus import RedisBus
from firecore.logging_setup import setup_logger
from firecore.minio_store import PRIVATE_BUCKET, ensure_bucket, upload_bytes
from rtsp_reader import RTSPStreamReader


# --------------------------------------------------------------------
# Device resolution — same contract as plate_detector / face_detector's
# engine.py, using the SAFE fallthrough (an unrecognized value warns and
# defaults to cpu, instead of being handed to torch verbatim — that
# verbatim-fallthrough bug is what produced 'Invalid device string:
# "cpu#auto"' elsewhere in this platform when a stray inline comment
# leaked into DETECTION_DEVICE; guarded against here from the start).
# --------------------------------------------------------------------
def resolve_device(preference: str, strict: bool, logger: logging.Logger) -> str:
    """auto | cpu | cuda | cuda:N -> a concrete torch device string.
    Resolved inside the Engine CHILD process only — the parent
    (EngineManager) never touches CUDA."""
    pref = (preference or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return pref
        msg = f"DETECTION_DEVICE={pref!r} requested but no CUDA device is visible"
        if strict:
            raise RuntimeError(msg)
        logger.error("%s — falling back to CPU", msg)
        return "cpu"
    logger.warning("Unrecognized DETECTION_DEVICE=%r, defaulting to cpu", preference)
    return "cpu"


def _apply_roi(frame: np.ndarray, roi: Tuple[float, float, float, float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """roi: (x,y,w,h) normalized in [0,1]."""
    x, y, w, h = roi
    H, W = frame.shape[:2]
    x1 = int(max(0, min(W - 1, x * W)))
    y1 = int(max(0, min(H - 1, y * H)))
    x2 = int(max(0, min(W, (x + w) * W)))
    y2 = int(max(0, min(H, (y + h) * H)))
    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0, W, H)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _new_grid_tracker() -> Dict[str, Any]:
    return {
        "regions": {
            0: {"window": collections.deque(maxlen=config.GRID_WINDOW_SIZE), "last_notified_state": "CLEAR"},
            1: {"window": collections.deque(maxlen=config.GRID_WINDOW_SIZE), "last_notified_state": "CLEAR"},
            2: {"window": collections.deque(maxlen=config.GRID_WINDOW_SIZE), "last_notified_state": "CLEAR"},
            3: {"window": collections.deque(maxlen=config.GRID_WINDOW_SIZE), "last_notified_state": "CLEAR"},
        },
        "cooldown_counter": config.GRID_COOLDOWN_FRAMES,
        "cooldown_active": False,
    }


# --------------------------------------------------------------------
# Engine process (one per detection worker)
# --------------------------------------------------------------------
class Engine:
    def __init__(
            self,
            engine_id: int,
            model_path: str,
            imgsz: int,
            conf: float,
            save_output: bool,
            status_queue: mp.Queue,
            control_queue: mp.Queue,
            stop_event: mp.Event,
            output_dir: str,
            class_labels: Dict[int, str],
    ):
        self.engine_id = int(engine_id)
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.save_output = bool(save_output)

        self.status_queue = status_queue
        self.control_queue = control_queue
        self.stop_event = stop_event

        self.output_dir = output_dir
        self.class_labels = class_labels

        self.logger = setup_logger(f"Engine{self.engine_id}")
        self.device = resolve_device(config.DETECTION_DEVICE, config.STRICT_DEVICE, self.logger)
        if self.device == "cpu":
            n = config.TORCH_NUM_THREADS or max(1, (os.cpu_count() or 4) - 1)
            torch.set_num_threads(n)
            self.logger.info(f"[INIT] CPU mode: torch threads set to {n}")

        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.debug_recorders: Dict[str, DebugRecorder] = {}

        # Each spawned child builds its OWN RedisBus / MinIO client —
        # neither is fork-safe to share (see firecore.bus's own
        # docstring on this), and this Engine always runs under the
        # spawn context (see engine_manager.py).
        self.bus = RedisBus(module=config.REDIS_MODULE)

        # Thread-safe background I/O pipeline — MinIO upload + Redis
        # publish never block the inference loop.
        self.io_queue: "queue.Queue" = queue.Queue()
        self.io_worker_thread = threading.Thread(
            target=self._io_worker_loop, daemon=True, name=f"Engine-{self.engine_id}-WorkerIO"
        )
        self.io_worker_thread.start()

        self.model = YOLO(self.model_path).to(self.device)
        self.logger.info(
            f"[INIT] YOLO model loaded model={self.model_path} device={self.device} "
            f"imgsz={self.imgsz} conf={self.conf}"
        )

        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)

    # ==================================================================
    # Background I/O — MinIO upload + Redis publish
    # ==================================================================
    def _io_worker_loop(self):
        self.logger.info("Background I/O worker thread started")
        try:
            ensure_bucket(PRIVATE_BUCKET)
        except Exception as e:
            self.logger.warning(f"Could not verify/create MinIO bucket {PRIVATE_BUCKET}: {e}")

        while not self.stop_event.is_set():
            try:
                payload = self.io_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._handle_io_payload(payload)
            except Exception as e:
                self.logger.error(f"Error handling reporting packet in background I/O loop: {e}")
            finally:
                self.io_queue.task_done()

    def _handle_io_payload(self, payload: Dict[str, Any]):
        camera_id = payload["camera_id"]
        timestamp_str = payload["timestamp_str"]
        event_type = payload["type"]  # "THREAT" or "RESOLUTION"
        event_label = payload["event"]
        frame = payload["frame"]

        if event_type == "THREAT":
            region_id = payload["region_id"]
            filename = f"Camera{camera_id}_{timestamp_str}_Region{region_id}_{event_label}.jpg"
        else:
            filename = f"Camera{camera_id}_{timestamp_str}_{event_label}.jpg"

        # dynamics/FireSmoke/<camera_id>/<filename> — mirrors the old
        # local bind-dir layout minus the bind-dir prefix.
        object_key = f"dynamics/FireSmoke/{camera_id}/{filename}"

        if not self._upload_frame_to_minio(object_key, frame):
            return

        result_payload = {
            "camera_id": camera_id,
            "event_type": event_type,
            "status": event_label,
            "timestamp": payload["timestamp_epoch"],
            "saved_file_path": object_key,
            "region_id": payload.get("region_id"),
        }
        self.bus.push_result(result_payload)
        self.logger.info(
            "[PUBLISHED %s - %s] camera=%s region_id=%s -> %s",
            event_type, event_label, camera_id, result_payload["region_id"], self.bus.keys.detections_results,
        )

    def _upload_frame_to_minio(self, key: str, frame: np.ndarray) -> bool:
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                self.logger.error(f"MinIO upload skipped: frame invalid/empty -> {key}")
                return False
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self.logger.error(f"MinIO upload skipped: cv2.imencode returned False -> {key}")
                return False
            upload_bytes(buf.tobytes(), key=key, bucket=PRIVATE_BUCKET, content_type="image/jpeg")
            return True
        except Exception as e:
            self.logger.error(f"Error uploading frame to MinIO -> {key}: {e}")
            return False

    # ==================================================================
    # Camera lifecycle
    # ==================================================================
    def add_camera(self, camera_id: str, url: str, roi: Tuple[float, float, float, float], **_ignored):
        camera_id = str(camera_id)
        if camera_id in self.cameras:
            self.cameras[camera_id]["url"] = url
            self.cameras[camera_id]["roi"] = roi
            return

        reader = RTSPStreamReader(url, camera_id).start()

        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "url": url,
            "roi": roi,
            "reader": reader,
            "fid": 0,
            "frames_processed": 0,
            "last_frame_ts": None,
            "grid_tracker": _new_grid_tracker(),
        }
        if self.save_output:
            self.debug_recorders[camera_id] = DebugRecorder(camera_id, self.output_dir)

        self._send_msg(camera_id, {"status": "running"})
        self.logger.info(f"[INIT] Added camera {camera_id} -> {url}")

    def remove_camera(self, camera_id: str):
        camera_id = str(camera_id)
        cam = self.cameras.pop(camera_id, None)
        if not cam:
            return
        try:
            cam["reader"].stop()
        except Exception:
            pass

        rec = self.debug_recorders.pop(camera_id, None)
        if rec is not None:
            try:
                rec.close()
            except Exception:
                self.logger.exception("failed to close debug recorder for camera %s", camera_id)

        self._send_msg(camera_id, {"status": "stopped"})
        self.logger.info(f"Removed camera {camera_id}")

    def _send_msg(self, camera_id: str, payload: Dict[str, Any]):
        msg = {"camera_id": str(camera_id)}
        msg.update(payload)
        try:
            self.status_queue.put_nowait(msg)
        except Exception:
            pass

    def _emit_heartbeat(self, camera_id: str):
        cam = self.cameras.get(camera_id)
        if not cam:
            return
        self._send_msg(camera_id, {
            "frames_processed": cam.get("frames_processed", 0),
            "last_frame_ts": cam.get("last_frame_ts"),
            "status": "running",
        })

    def _drain_control(self):
        while True:
            try:
                cmd = self.control_queue.get_nowait()
            except Exception:
                break
            if not isinstance(cmd, dict):
                continue
            c = (cmd.get("cmd") or "").lower().strip()
            if c == "add":
                self.add_camera(cmd["camera_id"], cmd["url"], cmd.get("roi", (0, 0, 1, 1)))
            elif c == "remove":
                self.remove_camera(cmd["camera_id"])
            elif c == "stop":
                self.stop_event.set()
            else:
                self.logger.warning(f"Unknown cmd: {cmd}")

    # ==================================================================
    # Main loop
    # ==================================================================
    def run(self):
        self.logger.info("Engine inference loop started")

        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(source=[dummy], imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        except Exception:
            pass

        last_stats_log = time.time()

        while not self.stop_event.is_set():
            self._drain_control()

            if not self.cameras:
                time.sleep(0.05)
                continue

            frames, cam_ids = [], []

            for camera_id, cam in list(self.cameras.items()):
                ret, frame = cam["reader"].read()
                if not (ret and frame is not None):
                    continue
                cam["fid"] += 1
                cam["frames_processed"] += 1
                cam["last_frame_ts"] = time.time()

                roi_frame, _ = _apply_roi(frame, cam["roi"])
                frames.append(roi_frame)
                cam_ids.append(camera_id)

            if not frames:
                time.sleep(0.02)
                continue

            _t0 = time.time()
            try:
                results = self.model.predict(
                    source=frames, imgsz=self.imgsz, conf=self.conf,
                    device=self.device, verbose=False, half=False,
                )
            except Exception as e:
                for cid in cam_ids:
                    self._send_msg(cid, {"status": "error", "error": str(e)})
                time.sleep(0.1)
                continue
            batch_ms = (time.time() - _t0) * 1000.0
            if batch_ms > config.SLOW_BATCH_WARN_MS:
                self.logger.warning("[INFER-SLOW] batch of %d camera(s) took %.1fms", len(cam_ids), batch_ms)

            for idx, res in enumerate(results):
                camera_id = cam_ids[idx]
                cam = self.cameras.get(camera_id)
                if cam is None:
                    continue
                self._emit_heartbeat(camera_id)
                self._process_result(camera_id, cam, frames[idx], res)

            if time.time() - last_stats_log > config.PIPELINE_STATS_LOG_INTERVAL_SEC:
                last_stats_log = time.time()
                self.logger.info(
                    "[PIPELINE] cameras=%d total_frames=%s",
                    len(self.cameras), {cid: c["frames_processed"] for cid, c in self.cameras.items()},
                )

        self.logger.info("Engine processing stopping...")
        self.cleanup()

    def _process_result(self, camera_id: str, cam: Dict[str, Any], roi_frame: np.ndarray, res):
        H, W = roi_frame.shape[:2]
        detector_boxes: List[Dict[str, Any]] = []
        region_detections = {0: set(), 1: set(), 2: set(), 3: set()}

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()

            for i, bb in enumerate(boxes):
                cls_id = int(classes[i])
                if cls_id not in (0, 1):  # 0: Smoke, 1: Fire
                    continue
                x1, y1, x2, y2 = map(int, bb)
                detector_boxes.append({"bbox": (x1, y1, x2, y2), "score": float(confs[i]), "class_id": cls_id})

                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                col = 0 if cx < (W / 2.0) else 1
                row = 0 if cy < (H / 2.0) else 1
                region_detections[row * 2 + col].add(cls_id)

        grid_tracker = cam["grid_tracker"]
        current_verdicts: Dict[int, str] = {}
        now_epoch = time.time()
        now_dt_str = datetime.fromtimestamp(now_epoch).strftime("%Y-%m-%d_%H-%M-%S")

        for r_id, r_state in grid_tracker["regions"].items():
            classes_in_region = region_detections[r_id]
            if 0 in classes_in_region and 1 in classes_in_region:
                frame_label = "BOTH"
            elif 1 in classes_in_region:
                frame_label = "FIRE"
            elif 0 in classes_in_region:
                frame_label = "SMOKE"
            else:
                frame_label = "CLEAR"

            r_state["window"].append(frame_label)
            window_list = list(r_state["window"])
            both_score = window_list.count("BOTH")
            fire_score = window_list.count("FIRE") + both_score
            smoke_score = window_list.count("SMOKE") + both_score

            if both_score >= config.GRID_VERIFY_THRESHOLD:
                verified_verdict = "BOTH"
            elif fire_score >= config.GRID_VERIFY_THRESHOLD:
                verified_verdict = "FIRE"
            elif smoke_score >= config.GRID_VERIFY_THRESHOLD:
                verified_verdict = "SMOKE"
            else:
                verified_verdict = "CLEAR"
            current_verdicts[r_id] = verified_verdict

            last_notified = r_state["last_notified_state"]
            if verified_verdict != "CLEAR" and verified_verdict != last_notified:
                r_state["last_notified_state"] = verified_verdict
                self.io_queue.put_nowait({
                    "type": "THREAT", "camera_id": camera_id, "timestamp_epoch": now_epoch,
                    "timestamp_str": now_dt_str, "region_id": r_id, "event": verified_verdict,
                    "frame": roi_frame.copy(),
                })

        any_active_threats = any(v != "CLEAR" for v in current_verdicts.values())
        if any_active_threats:
            grid_tracker["cooldown_counter"] = config.GRID_COOLDOWN_FRAMES
            grid_tracker["cooldown_active"] = False
        else:
            any_region_escalated = any(
                r_state["last_notified_state"] != "CLEAR" for r_state in grid_tracker["regions"].values()
            )
            if any_region_escalated:
                grid_tracker["cooldown_active"] = True
                grid_tracker["cooldown_counter"] -= 1
                if grid_tracker["cooldown_counter"] <= 0:
                    for r_state in grid_tracker["regions"].values():
                        r_state["last_notified_state"] = "CLEAR"
                    self.io_queue.put_nowait({
                        "type": "RESOLUTION", "camera_id": camera_id, "timestamp_epoch": now_epoch,
                        "timestamp_str": now_dt_str, "event": "CLEAR", "frame": roi_frame.copy(),
                    })
                    grid_tracker["cooldown_counter"] = config.GRID_COOLDOWN_FRAMES
                    grid_tracker["cooldown_active"] = False

        rec = self.debug_recorders.get(camera_id)
        if rec is not None:
            rec.write(roi_frame, detector_boxes, self.class_labels, current_verdicts, grid_tracker["cooldown_active"])

    def cleanup(self):
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)
        time.sleep(0.2)


def _engine_process_main(
        engine_id: int,
        model_path: str,
        imgsz: int,
        conf: float,
        save_output: bool,
        status_queue: mp.Queue,
        control_queue: mp.Queue,
        stop_event: mp.Event,
        output_dir: str,
        class_labels: Dict[int, str],
):
    setup_logger(f"Engine{engine_id}")
    eng = Engine(
        engine_id=engine_id, model_path=model_path, imgsz=imgsz, conf=conf,
        save_output=save_output, status_queue=status_queue, control_queue=control_queue,
        stop_event=stop_event, output_dir=output_dir, class_labels=class_labels,
    )
    eng.run()
