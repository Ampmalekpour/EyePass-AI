"""
engine.py
--------------------------------------------------------------------
The Engine: one YOLO model + one BYTETracker per camera it owns,
running in its own subprocess (see _engine_process_main /
EngineManager). Detection, tracking and spatial-trigger logic are
carried over from the reference video_processor.py unchanged; the only
structural change is where face-crop tasks go and where their results
come back from:

    reference pipeline:  self.fr_input_queue.put(task)   (mp.Queue,
                          consumed by AFRWorker subprocesses this same
                          Engine spawned and owned 1:1)

    this pipeline:        self.rec_client.submit(task)    (Redis LIST,
                          consumed by ANY worker in the recognizer
                          service's independently-sized pool)

Everything downstream of that call — the periodic/spatial dispatch
state machine, the finalize fast-lane vs high-fidelity routes, the
best-crop ranking, the final publish to the backend — is the same
logic as the reference pipeline, because the task/result SHAPES are
unchanged; only the transport is.
--------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import logging
import math
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
import debug_extras
from debug_recorder import DebugConfig, DebugRecorder
from facecore.bus import RedisBus
from facecore.codec import DateTimeEncoder
from facecore.logging_setup import setup_logger
from liveness import LivenessAnalyzer, LivenessConfig
from rec_client import RecognitionClient
from rtsp_reader import RTSPStreamReader, encode_image, measure_sharpness
from tracker import BYTETracker
from triggers import process_track_triggers


# --------------------------------------------------------------------
# Compatibility placeholder — kept so anything that historically
# imported FaceState from video_processor keeps working unchanged.
# --------------------------------------------------------------------
class FaceState:
    def __init__(self):
        self.running = True
        self.finished = False

    def stop(self):
        self.running = False

    def set_finished(self):
        self.finished = True


class TrackerConfig:
    def __init__(self, track_thresh=0.5, match_thresh=0.9, track_buffer=90, nms_thresh=0.5, mot20=True):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.nms_thresh = nms_thresh
        self.mot20 = mot20


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
        msg = f"DETECTION_DEVICE={pref} requested but no CUDA device is visible"
        if strict:
            raise RuntimeError(msg)
        logger.error("%s — falling back to CPU", msg)
        return "cpu"
    logger.warning("Unrecognized DETECTION_DEVICE=%r, defaulting to cpu", preference)
    return "cpu"


class Engine:
    def __init__(
            self,
            engine_id: int,
            model_path: str,
            imgsz: int,
            conf: float,
            save_output: bool,
            output_dir: str,
            class_labels: Dict[int, str],
            save_as_video: bool = True,
            status_queue: mp.Queue = None,
            control_queue: mp.Queue = None,
            stop_event: mp.Event = None,
            absent_n: int = config.DEFAULT_ABSENT_FRAMES,
            min_seen_frames: int = config.DEFAULT_MIN_SEEN_FRAMES,
            min_crops_to_finalize: int = config.DEFAULT_MIN_CROPS_TO_FINALIZE,
            n_best: int = config.DEFAULT_N_BEST_CROPS,
            conf_digits: int = config.DEFAULT_CONF_DIGITS,
    ):
        self.engine_id = int(engine_id)
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.save_output = bool(save_output)
        self.save_as_video = bool(save_as_video)

        self.status_queue = status_queue
        self.control_queue = control_queue
        self.stop_event = stop_event

        self.output_dir = output_dir
        self.class_labels = class_labels

        self.logger = setup_logger(f"Engine{self.engine_id}")
        self.device = resolve_device(config.DETECTION_DEVICE, config.STRICT_DEVICE, self.logger)

        self.ABSENT_N = int(absent_n)
        self.MIN_SEEN_FRAMES = int(min_seen_frames)
        self.MIN_CROPS_TO_FINALIZE = int(min_crops_to_finalize)
        self.N_BEST = int(n_best)
        self.conf_digits = int(conf_digits)

        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.writers: Dict[str, cv2.VideoWriter] = {}

        # ---- diagnostics: throttled per-camera pipeline summary --------
        self._last_summary_log: Dict[str, float] = {}
        self._last_loop_log: float = 0.0
        self.SUMMARY_LOG_INTERVAL_SEC = float(os.environ.get("SUMMARY_LOG_INTERVAL_SEC", "5.0"))

        self.model = YOLO(self.model_path).to(self.device)
        self.logger.info(f"YOLO loaded on {self.device} | {self.model_path}")

        # ---- Redis: publish crop tasks / receive recognition results --
        self.bus = RedisBus(module=config.REDIS_MODULE)
        self.rec_client = RecognitionClient(self.bus, self.engine_id)

        # ---- liveness (anti-spoof) + annotated debug video -------------
        # Both are per-camera objects created in add_camera(); these are
        # just the env snapshots, taken once here so every camera on this
        # engine agrees on the settings.
        self.liveness_cfg = LivenessConfig()
        self.debug_cfg = DebugConfig()
        self.logger.info(
            "liveness=%s | debug_video=%s (segments %.0fs -> %s)",
            "ON" if self.liveness_cfg.enabled else "off",
            "ON" if self.debug_cfg.enabled else "off",
            self.debug_cfg.segment_seconds, self.debug_cfg.dir,
        )

        self.PERIODIC_MODE = config.PERIODIC_MODE
        self.PERIODIC_FRAME_INTERVAL = config.PERIODIC_FRAME_INTERVAL
        self.PERIODIC_TIME_INTERVAL = config.PERIODIC_TIME_INTERVAL
        self.PERIODIC_RECOG_CONF_THRESH = config.PERIODIC_RECOG_CONF_THRESH
        self.FINALIZE_MAX_CROPS = config.FINALIZE_MAX_CROPS

        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)

    # ============================================================================
    # HEAD POSE ESTIMATION
    # ============================================================================
    MODEL_3D = np.array([
        [0.0, 0.0, 0.0], [0.0, 55.0, 20.0], [0.0, -65.0, 20.0], [0.0, -25.0, 15.0],
        [-35.0, -20.0, 30.0], [35.0, -20.0, 30.0], [-15.0, -20.0, 20.0], [15.0, -20.0, 20.0],
        [-20.0, 25.0, 20.0], [20.0, 25.0, 20.0], [-15.0, 10.0, 15.0], [15.0, 10.0, 15.0],
        [-75.0, 10.0, 75.0], [75.0, 10.0, 75.0]
    ], dtype=np.float64)

    def _annotate_and_write_detections(self, camera_id: str, frame: np.ndarray, track_data: list, detector_boxes):
        if not self.save_output or frame is None or frame.size == 0:
            return

        vis_frame = frame.copy()

        for det in detector_boxes:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            color = (0, 165, 255)  # ORANGE (BGR)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"D:{score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for track in track_data:
            tx1, ty1, tx2, ty2 = track['bbox']
            track_id = track['id']
            track_class = track['class']
            score = track['score']
            landmarks = track.get('landmarks')
            angles = track.get('yaw_pitch_roll')
            valid_lm = track.get('valid_landmarks', 0)
            sharpness = track.get('sharpness', 0.0)
            resolution = track.get('resolution', 0)

            cv2.rectangle(vis_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

            label = f"ID:{track_id} | C:{track_class} | Conf:{score:.2f}"
            if angles is not None:
                yaw = angles.get("yaw", 0)
                pitch = angles.get("pitch", 0)
                roll = angles.get("roll", 0)
                label += f" | Y:{yaw:+.1f} P:{pitch:+.1f} R:{roll:+.1f}"
            else:
                label += " | No landmarks"

            label += f" | LM:{valid_lm} | Sharp:{sharpness:.0f} | Res:{resolution}"

            cv2.putText(vis_frame, label, (tx1, max(10, ty1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

            if landmarks is not None:
                for pt in landmarks:
                    x, y, conf = pt
                    if conf > config.LANDMARK_VALID_CONF_THRESHOLD and (x > 0.0 or y > 0.0):
                        cv2.circle(vis_frame, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

            if landmarks is not None:
                right_x = tx2 + 5
                y_start = ty1 + 15
                line_height = 18
                for i, pt in enumerate(landmarks):
                    x, y, conf = pt
                    if conf > config.LANDMARK_VALID_CONF_THRESHOLD:
                        conf_text = f"{conf:.2f}"
                        cv2.putText(vis_frame, conf_text, (right_x, y_start + i * line_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        if self.save_as_video:
            if camera_id not in self.writers:
                video_dir = os.path.join(self.output_dir, str(camera_id))
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"camera_{camera_id}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                h, w = frame.shape[:2]
                self.writers[camera_id] = cv2.VideoWriter(video_path, fourcc, config.LEGACY_VIDEO_FPS, (w, h))
            self.writers[camera_id].write(vis_frame)
        else:
            camera_dir = os.path.join(self.output_dir, str(camera_id))
            os.makedirs(camera_dir, exist_ok=True)
            timestamp = time.time()
            filepath = os.path.join(camera_dir, f"frame_{timestamp:.3f}.jpg")
            success = cv2.imwrite(filepath, vis_frame)
            if not success:
                self.logger.error(f"Failed to write frame to {filepath}")

    def compute_head_pose(self, keypoints: np.ndarray, frame_w: int, frame_h: int) -> Optional[Dict[str, float]]:
        if keypoints is None or len(keypoints) < 6:
            return None

        valid_points_2d = []
        valid_points_3d = []
        for i, pt in enumerate(keypoints):
            if i >= len(self.MODEL_3D):
                break
            pt = np.array(pt).flatten()
            if pt.size < 2:
                continue
            x = pt[0]
            y = pt[1]
            conf = pt[2] if pt.size > 2 else 1.0
            if conf > config.LANDMARK_VALID_CONF_THRESHOLD:
                valid_points_2d.append([x, y])
                valid_points_3d.append(self.MODEL_3D[i])

        if len(valid_points_2d) < 6:
            return None

        valid_points_2d = np.asarray(valid_points_2d, dtype=np.float64).reshape(-1, 1, 2)
        valid_points_3d = np.asarray(valid_points_3d, dtype=np.float64).reshape(-1, 1, 3)

        if valid_points_2d.shape[0] < 6 or valid_points_3d.shape[0] < 6:
            return None

        focal_length = float(frame_w if frame_w > frame_h else frame_h)
        cam_matrix = np.array([
            [focal_length, 0, frame_w / 2.0],
            [0, focal_length, frame_h / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            valid_points_3d, valid_points_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0

        return {"pitch": math.degrees(pitch), "yaw": math.degrees(yaw), "roll": math.degrees(roll)}

    def classify_pose(self, angles: Optional[Dict[str, float]]) -> int:
        if angles is None:
            return 4
        yaw = angles["yaw"]
        pitch = angles["pitch"]
        assigned_class = 4
        if -config.FRONTAL_YAW_LIMIT <= yaw <= config.FRONTAL_YAW_LIMIT and \
                -config.FRONTAL_PITCH_LIMIT <= pitch <= config.FRONTAL_PITCH_LIMIT:
            assigned_class = 1
        elif config.QUARTER_YAW_LIMIT_MIN <= abs(yaw) <= config.QUARTER_YAW_LIMIT_MAX and \
                -config.QUARTER_PITCH_LIMIT <= int(pitch) <= config.QUARTER_PITCH_LIMIT:
            assigned_class = 2
        elif config.PROFILE_YAW_LIMIT_MIN <= abs(yaw):
            assigned_class = 3
        return assigned_class

    def _compute_pose_from_track(self, track_landmarks: np.ndarray, frame_w: int, frame_h: int):
        if track_landmarks is None or len(track_landmarks) == 0:
            return None, 0
        valid_count = sum(1 for x, y, conf in track_landmarks
                          if conf > config.LANDMARK_VALID_CONF_THRESHOLD and (abs(x) > 1e-6 or abs(y) > 1e-6))
        if valid_count < 6:
            return None, valid_count
        angles = self.compute_head_pose(track_landmarks, frame_w, frame_h)
        return angles, valid_count

    # ============================================================================

    def _rank(self, det_score: float, resolution: int) -> Tuple[int, int]:
        n = self.conf_digits
        conf_bucket = int(float(det_score) * (10 ** n))
        return (conf_bucket, int(resolution))

    def _quality_check_crop(self, track_class: int, crop_image: np.ndarray, sharpness: float) -> bool:
        if crop_image is None or crop_image.size == 0:
            return False
        h, w = crop_image.shape[:2]
        if h <= 0 or w <= 0:
            return False
        resolution = w * h
        aspect_ratio = w / h if h > 0 else 0.0
        if sharpness < config.SHARPNESS_MIN_THRESHOLD:
            return False
        if int(track_class) == 0:
            if not (config.CLASS_0_ASPECT_RATIO_MIN <= aspect_ratio <= config.CLASS_0_ASPECT_RATIO_MAX):
                return False
        else:
            if not (config.GENERIC_ASPECT_RATIO_MIN <= aspect_ratio <= config.GENERIC_ASPECT_RATIO_MAX):
                return False
        if resolution < config.RESOLUTION_MIN_AREA:
            return False
        return True

    def add_camera(
            self,
            camera_id: str,
            url: str,
            roi: Tuple[float, float, float, float],
            cond_per_trig: bool = False,
            cross_line_trig: bool = False,
            stop_roi_trig: bool = False,
            leave_scene_trig: bool = False,
            line_p1_x: Optional[int] = None,
            line_p1_y: Optional[int] = None,
            line_p2_x: Optional[int] = None,
            line_p2_y: Optional[int] = None,
            stop_roi_p1_x: Optional[int] = None,
            stop_roi_p1_y: Optional[int] = None,
            stop_roi_p2_x: Optional[int] = None,
            stop_roi_p2_y: Optional[int] = None,
            stop_roi_p3_x: Optional[int] = None,
            stop_roi_p3_y: Optional[int] = None,
            stop_roi_p4_x: Optional[int] = None,
            stop_roi_p4_y: Optional[int] = None
    ):
        camera_id = str(camera_id)

        if line_p1_x is not None and line_p1_y is not None and line_p2_x is not None and line_p2_y is not None:
            line_coords = ((int(line_p1_x), int(line_p1_y)), (int(line_p2_x), int(line_p2_y)))
        else:
            line_coords = ((0, 0), (0, 0))

        explicit_roi = [stop_roi_p1_x, stop_roi_p1_y, stop_roi_p2_x, stop_roi_p2_y,
                        stop_roi_p3_x, stop_roi_p3_y, stop_roi_p4_x, stop_roi_p4_y]
        if None not in explicit_roi:
            stop_roi_coords = (
                (int(stop_roi_p1_x), int(stop_roi_p1_y)),
                (int(stop_roi_p2_x), int(stop_roi_p2_y)),
                (int(stop_roi_p3_x), int(stop_roi_p3_y)),
                (int(stop_roi_p4_x), int(stop_roi_p4_y))
            )
        else:
            stop_roi_coords = ((0, 0), (0, 0), (0, 0), (0, 0))

        if camera_id in self.cameras:
            self.cameras[camera_id]["url"] = url
            self.cameras[camera_id]["roi"] = roi
            self.cameras[camera_id]["periodic_enabled"] = bool(cond_per_trig)
            self.cameras[camera_id]["cross_line_enabled"] = bool(cross_line_trig)
            self.cameras[camera_id]["flag_stop_roi_enabled"] = bool(stop_roi_trig)
            self.cameras[camera_id]["leave_scene_enabled"] = bool(leave_scene_trig)
            self.cameras[camera_id]["line_points"] = line_coords
            self.cameras[camera_id]["stop_roi"] = stop_roi_coords
            return

        self.logger.info("add_camera(%s): step 1/5 — starting RTSP reader thread", camera_id)
        reader = RTSPStreamReader(url, camera_id, max_queue_size=1000).start()

        self.logger.info("add_camera(%s): step 2/5 — reader thread started, building tracker", camera_id)
        tracker = BYTETracker(TrackerConfig(
            track_thresh=config.TRACKER_TRACK_THRESH,
            match_thresh=config.TRACKER_MATCH_THRESH,
            track_buffer=config.TRACKER_TRACK_BUFFER,
            nms_thresh=config.TRACKER_NMS_THRESH,
            mot20=config.TRACKER_MOT20,
        ))

        self.logger.info("add_camera(%s): step 3/5 — tracker built, registering camera", camera_id)

        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "url": url,
            "roi": roi,
            "reader": reader,
            "tracker": tracker,
            "fid": 0,
            "frames_processed": 0,
            "last_frame_ts": None,
            "_started_ts": time.time(),
            "track_meta": {},
            "best_crops": {},
            "best_frame": {},
            "periodic_enabled": bool(cond_per_trig),
            "cross_line_enabled": bool(cross_line_trig),
            "flag_stop_roi_enabled": bool(stop_roi_trig),
            "leave_scene_enabled": bool(leave_scene_trig),
            "line_points": line_coords,
            "stop_roi": stop_roi_coords,
            # anti-spoof analyzer + annotated debug recorder, per camera
            "liveness": LivenessAnalyzer(camera_id, self.liveness_cfg, self.logger)
            if self.liveness_cfg.enabled else None,
            "recorder": DebugRecorder(camera_id, self.engine_id, self.debug_cfg, self.logger)
            if self.debug_cfg.enabled else None,
        }

        self.logger.info("add_camera(%s): step 4/5 — registered, sending status to parent", camera_id)
        self._send_msg(camera_id, {"status": "running"})
        self.logger.info("add_camera(%s): step 5/5 — done", camera_id)
        self.logger.info(f"Added camera {camera_id} -> {url}")

    def remove_camera(self, camera_id: str):
        camera_id = str(camera_id)
        cam = self.cameras.pop(camera_id, None)
        if not cam:
            return
        try:
            cam["reader"].stop()
        except Exception:
            pass

        rec = cam.get("recorder")
        if rec is not None:
            try:
                rec.close()
            except Exception:
                pass

        w = self.writers.pop(camera_id, None)
        if w is not None:
            try:
                w.release()
                time.sleep(0.1)
                self.logger.info(f"Video saved and closed for camera {camera_id}")
            except Exception as e:
                self.logger.error(f"Error releasing writer for {camera_id}: {e}")

        self._send_msg(camera_id, {"status": "stopped"})
        self.logger.info(f"Removed camera {camera_id}")

    def camera_ids(self) -> List[str]:
        return list(self.cameras.keys())

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
            "last_frame_ts": cam.get("last_frame_ts", None),
            "status": "running",
        })

    def _apply_roi(self, frame: np.ndarray, roi: Tuple[float, float, float, float]):
        x, y, w, h = roi
        H, W = frame.shape[:2]
        x1 = int(max(0, min(W - 1, x * W)))
        y1 = int(max(0, min(H - 1, y * H)))
        x2 = int(max(0, min(W, (x + w) * W)))
        y2 = int(max(0, min(H, (y + h) * H)))
        if x2 <= x1 or y2 <= y1:
            return frame, (0, 0, W, H)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _set_best_frame(self, cam: Dict[str, Any], track_id: int,
                        full_frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int],
                        det_score: float, resolution: int):
        cam["best_frame"][track_id] = {
            "frame": full_frame.copy(),
            "bbox": tuple(map(int, bbox_xyxy)),
            "det_score": float(det_score),
            "resolution": int(resolution),
            "ts": time.time(),
        }

    def _drain_control(self):
        while True:
            try:
                cmd = self.control_queue.get_nowait()
            except Exception:
                break
            if not isinstance(cmd, dict):
                continue
            c = (cmd.get("cmd") or "").lower().strip()
            self.logger.info("control cmd received: %s | camera_id=%s", c, cmd.get("camera_id"))
            if c == "add":
                try:
                    self.add_camera(
                        camera_id=cmd["camera_id"],
                        url=cmd["url"],
                        roi=cmd.get("roi", (0, 0, 1, 1)),
                        cond_per_trig=cmd.get("cond_per_trig", False),
                        cross_line_trig=cmd.get("cross_line_trig", False),
                        stop_roi_trig=cmd.get("stop_roi_trig", False),
                        leave_scene_trig=cmd.get("leave_scene_trig", False),
                        line_p1_x=cmd.get("line_p1_x"),
                        line_p1_y=cmd.get("line_p1_y"),
                        line_p2_x=cmd.get("line_p2_x"),
                        line_p2_y=cmd.get("line_p2_y"),
                        stop_roi_p1_x=cmd.get("stop_roi_p1_x"),
                        stop_roi_p1_y=cmd.get("stop_roi_p1_y"),
                        stop_roi_p2_x=cmd.get("stop_roi_p2_x"),
                        stop_roi_p2_y=cmd.get("stop_roi_p2_y"),
                        stop_roi_p3_x=cmd.get("stop_roi_p3_x"),
                        stop_roi_p3_y=cmd.get("stop_roi_p3_y"),
                        stop_roi_p4_x=cmd.get("stop_roi_p4_x"),
                        stop_roi_p4_y=cmd.get("stop_roi_p4_y"),
                    )
                except Exception:
                    self.logger.exception("add_camera FAILED for camera %s", cmd.get("camera_id"))
            elif c == "remove":
                self.remove_camera(cmd["camera_id"])
            elif c == "stop":
                self.stop_event.set()
            else:
                self.logger.warning(f"Unknown cmd: {cmd}")

    # ---------------- best-crop bookkeeping ----------------
    def _update_best_crops(self, cam: Dict[str, Any], track_id: int, track_class: int,
                           crop_img: np.ndarray, det_score: float, resolution: int,
                           aspect_ratio: float, sharpness: float,
                           mlc: float, yaw_group: int,
                           landmarks: Optional[np.ndarray] = None,
                           crop_offset: Tuple[int, int] = (0, 0)) -> Tuple[bool, bool]:

        if track_id not in cam["best_crops"]:
            cam["best_crops"][track_id] = {"reg1": [], "reg2": [], "reg3": []}

        store = cam["best_crops"][track_id]

        candidate = {
            "image": crop_img.copy(),
            "class_flag": int(track_class),
            "det_score": float(det_score),
            "resolution": int(resolution),
            "aspect_ratio": float(aspect_ratio),
            "sharpness": float(sharpness),
            "frame_number": int(cam["fid"]),
            "mlc": float(mlc),
            "yaw_group": int(yaw_group)
        }

        if landmarks is not None and len(landmarks) > 0:
            offset_x, offset_y = crop_offset
            crop_landmarks = landmarks.copy().astype(np.float64)
            crop_landmarks[:, 0] -= offset_x
            crop_landmarks[:, 1] -= offset_y
            candidate["landmarks"] = crop_landmarks
        else:
            candidate["landmarks"] = None

        ranking_funcs = {
            "reg1": lambda x: x["mlc"] + (x["yaw_group"] * 1e-7) + (x["resolution"] * 1e-9),
        }

        was_added = False
        became_top = False

        for reg_name, rank_func in ranking_funcs.items():
            reg_store = store[reg_name]
            cand_rank = rank_func(candidate)

            old_best = reg_store[0] if reg_store else None
            old_best_rank = rank_func(old_best) if old_best else None

            if len(reg_store) < self.N_BEST:
                reg_store.append(candidate)
                reg_store.sort(key=rank_func, reverse=True)
                was_added = True
                if old_best_rank is None or cand_rank > old_best_rank:
                    became_top = True
            else:
                worst = reg_store[-1]
                worst_rank = rank_func(worst)
                if cand_rank > worst_rank:
                    reg_store[-1] = candidate
                    reg_store.sort(key=rank_func, reverse=True)
                    was_added = True
                    if cand_rank > old_best_rank:
                        became_top = True

        return was_added, became_top

    def _get_yaw_group(self, yaw: float) -> int:
        abs_yaw = abs(yaw)
        if abs_yaw <= config.YAW_GROUP_FRONTAL_MAX_DEG:
            return 3
        elif abs_yaw <= config.YAW_GROUP_QUARTER_MAX_DEG:
            return 2
        else:
            return 1

    def _should_send_periodic_check(self, meta: dict) -> bool:
        if not meta:
            return False
        seen_frames = meta.get("seen_frames", 0)

        if self.PERIODIC_MODE == "time":
            current_time = time.time()
            last_periodic_ts = meta.get("last_periodic_timestamp", 0.0)
            if last_periodic_ts == 0.0:
                meta["last_periodic_timestamp"] = current_time
                return seen_frames == self.PERIODIC_FRAME_INTERVAL
            elapsed_time = current_time - last_periodic_ts
            if elapsed_time >= self.PERIODIC_TIME_INTERVAL:
                meta["last_periodic_timestamp"] = current_time
                return True
        else:
            if seen_frames > 0 and seen_frames % self.PERIODIC_FRAME_INTERVAL == 0:
                return True
        return False

    # ============================================================================
    # Debug-recorder context
    # ============================================================================
    def _debug_context(self, cam: Dict[str, Any], camera_id: str, fid: int,
                       detector_boxes: List[dict], debug_tracks: List[dict],
                       batch_size: int, roi_offset: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Everything the recorder needs for one frame. The geometry is
        translated out of full-frame coordinates into the ROI frame,
        because the ROI frame is what we actually draw on."""
        rx1, ry1, _rx2, _ry2 = roi_offset

        line = cam.get("line_points", ((0, 0), (0, 0)))
        line_px = (
            (line[0][0] - rx1, line[0][1] - ry1),
            (line[1][0] - rx1, line[1][1] - ry1),
        )
        stop_roi_px = [(p[0] - rx1, p[1] - ry1) for p in
                       cam.get("stop_roi", ((0, 0), (0, 0), (0, 0), (0, 0)))]

        return {
            "fid": fid,
            "url": cam.get("url", ""),
            "frames_processed": cam.get("frames_processed", 0),
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "batch_size": batch_size,
            "liveness_enabled": cam.get("liveness") is not None,
            "detections": detector_boxes,
            "tracks": debug_tracks,
            "rec_pending_count": sum(1 for t in debug_tracks if t.get("rec_pending")),
            "line_px": line_px,
            "stop_roi_px": stop_roi_px,
            "trig_cross": cam.get("cross_line_enabled", False),
            "trig_stop": cam.get("flag_stop_roi_enabled", False),
            "any_stopped": False,
            "triggers": {
                "periodic": cam.get("periodic_enabled", False),
                "cross_line": cam.get("cross_line_enabled", False),
                "stop_roi": cam.get("flag_stop_roi_enabled", False),
                "leave_scene": cam.get("leave_scene_enabled", False),
            },
        }

    def _recorder(self, camera_id: str):
        cam = self.cameras.get(camera_id)
        return cam.get("recorder") if cam else None

    # ============================================================================
    # MAIN LOOP
    # ============================================================================
    def run(self):
        self.logger.info("Engine loop started")

        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(source=[dummy], imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        except Exception:
            pass

        while not self.stop_event.is_set():
            # ---- unconditional loop heartbeat -------------------------
            # Fires even when there are no cameras and no frames, so a
            # frozen process is instantly distinguishable from a loop
            # that is spinning fine but starved of frames.
            now = time.time()
            if now - self._last_loop_log >= self.SUMMARY_LOG_INTERVAL_SEC:
                self._last_loop_log = now
                qdepths = {cid: len(c["reader"].queue) for cid, c in self.cameras.items()}
                self.logger.info("loop alive | cameras=%d reader_queue_depth=%s",
                                  len(self.cameras), qdepths or "{}")

            self._drain_control()
            self._drain_recognition_outputs()

            if not self.cameras:
                time.sleep(0.05)
                continue

            frames = []
            cam_ids = []
            roi_offsets = []

            for camera_id, cam in list(self.cameras.items()):
                ret, frame = cam["reader"].read()
                if not (ret and frame is not None):
                    continue
                cam["fid"] += 1
                cam["frames_processed"] += 1
                cam["last_frame_ts"] = time.time()

                roi_frame, (rx1, ry1, rx2, ry2) = self._apply_roi(frame, cam["roi"])
                frames.append(roi_frame)
                cam_ids.append(camera_id)
                roi_offsets.append((rx1, ry1, rx2, ry2))

            if not frames:
                time.sleep(0.02)
                continue

            try:
                results = self.model.predict(
                    source=frames, imgsz=self.imgsz, conf=self.conf,
                    device=self.device, verbose=False, half=False
                )
            except Exception as e:
                for cid in cam_ids:
                    self._send_msg(cid, {"status": "error", "error": str(e)})
                time.sleep(0.1)
                continue

            for idx, res in enumerate(results):
                camera_id = cam_ids[idx]
                cam = self.cameras.get(camera_id)
                if cam is None:
                    continue

                self._emit_heartbeat(camera_id)

                roi_frame = frames[idx]
                H, W = roi_frame.shape[:2]

                boxes_for_vis = None
                confs = None
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes_for_vis = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()

                detector_boxes = []
                if boxes_for_vis is not None:
                    for i, bb in enumerate(boxes_for_vis):
                        x1, y1, x2, y2 = map(int, bb)
                        detector_boxes.append({"bbox": (x1, y1, x2, y2), "score": float(confs[i])})

                if hasattr(res, 'keypoints') and res.keypoints is not None:
                    keypoints_for_vis = res.keypoints.data.cpu().numpy()
                else:
                    keypoints_for_vis = None

                if keypoints_for_vis is not None and len(keypoints_for_vis) > 0:
                    pose_classes_for_vis = []
                    for kpts in keypoints_for_vis:
                        angles = self.compute_head_pose(kpts, W, H)
                        p_class = self.classify_pose(angles)
                        pose_classes_for_vis.append(p_class)
                    flattened_kpts = keypoints_for_vis.reshape(len(keypoints_for_vis), -1)
                else:
                    pose_classes_for_vis = [0] * len(boxes_for_vis) if boxes_for_vis is not None else []
                    flattened_kpts = np.zeros((len(boxes_for_vis) if boxes_for_vis is not None else 0, 42),
                                              dtype=np.float64)

                pose_clss_array = np.array(pose_classes_for_vis, dtype=np.float64)

                if boxes_for_vis is not None and len(boxes_for_vis) > 0:
                    detections = np.column_stack([boxes_for_vis, confs, pose_clss_array, flattened_kpts]).astype(np.float64)
                else:
                    detections = np.empty((0, 6), dtype=np.float64)

                online_targets = cam["tracker"].update(detections, roi_frame.shape[:2], roi_frame.shape[:2])

                fid = cam["fid"]
                rx1, ry1, rx2, ry2 = roi_offsets[idx]

                # ---- diagnostics: throttled per-camera pipeline summary ----
                now = time.time()
                if now - self._last_summary_log.get(camera_id, 0.0) >= self.SUMMARY_LOG_INTERVAL_SEC:
                    self._last_summary_log[camera_id] = now
                    self.logger.info(
                        "cam=%s frames=%d fps~=%.1f detections=%d tracks=%d tracked_meta=%d has_landmarks=%s",
                        camera_id, cam["frames_processed"],
                        cam["frames_processed"] / max(1.0, now - (cam.get("_started_ts") or now)),
                        len(detector_boxes), len(online_targets), len(cam["track_meta"]),
                        keypoints_for_vis is not None,
                    )

                track_vis_data = []
                debug_tracks = []

                # Grayscale once per camera per frame — the liveness
                # analyzer needs it for every track on this frame, and
                # converting per track would be wasteful.
                gray = None
                if cam.get("liveness") is not None:
                    try:
                        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    except Exception:
                        gray = None

                for track in online_targets:
                    track_id = int(track.track_id)

                    meta = cam["track_meta"].get(track_id)
                    if meta is None:
                        cam["track_meta"][track_id] = {
                            "first_seen_fid": fid,
                            "last_seen_fid": fid,
                            "seen_frames": 1,
                            "recognition_history": {},
                            "deferred_events": {}
                        }
                        meta = cam["track_meta"][track_id]
                        self.logger.info("cam=%s track=%d NEW", camera_id, track_id)
                        _rec = cam.get("recorder")
                        if _rec is not None:
                            _rec.log("TRACK_NEW", f"trk{track_id} appeared", track_id=track_id)
                    else:
                        meta["last_seen_fid"] = fid
                        meta["seen_frames"] += 1
                        meta.setdefault("recognition_history", {})
                        meta.setdefault("deferred_events", {})

                    if meta.get("pending_recognition", False):
                        elapsed = time.time() - meta.get("pending_since", 0.0)
                        if elapsed > 1.0:
                            meta["pending_recognition"] = False

                    track_class = int(track.flag_fdf)

                    if track.detbb is None:
                        continue
                    x1, y1, x2, y2 = map(int, track.detbb)

                    track_landmarks = None
                    yaw_pitch_roll = None
                    valid_landmarks = 0
                    mlc = 0.0
                    yaw_group = 1

                    if hasattr(track, 'landmarks') and track.landmarks is not None:
                        track_landmarks = track.landmarks.reshape(-1, 3)
                        yaw_pitch_roll, valid_landmarks = self._compute_pose_from_track(track_landmarks, W, H)
                        mlc = float(np.mean(track_landmarks[:, 2]))

                    if track_landmarks is None or mlc <= 0.0:
                        existing_crops = cam["best_crops"].get(track_id, {}).get("reg1", [])
                        if not existing_crops:
                            track_landmarks = None
                            yaw_group = 1
                            mlc = 0.01
                        else:
                            continue

                    if yaw_pitch_roll is not None:
                        yaw_group = self._get_yaw_group(yaw_pitch_roll["yaw"])

                    x1 = max(0, min(x1, W - 1))
                    x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H - 1))
                    y2 = max(0, min(y2, H))

                    crop = roi_frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue

                    sharp = measure_sharpness(crop)
                    score = float(getattr(track, "score", 0.0))

                    hh, ww = crop.shape[:2]
                    reso = int(ww * hh)
                    ar = float(ww / hh) if hh > 0 else 0.0

                    was_added, updated_best = self._update_best_crops(
                        cam, track_id, track_class, crop, score, reso, ar, sharp, mlc, yaw_group,
                        landmarks=track_landmarks, crop_offset=(x1, y1)
                    )

                    if updated_best:
                        full_bbox = (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
                        self._set_best_frame(cam, track_id, roi_frame, full_bbox, score, reso)
                        meta["yaw_group"] = yaw_group
                        meta["track_class"] = track_class
                        meta["det_confidence"] = score

                    # ============================================================
                    # LIVENESS / PRESENTATION-ATTACK CHECK
                    # ============================================================
                    # Writes liveness / liveness_score / liveness_reason /
                    # liveness_evals straight onto meta, so it rides along
                    # into every recognition task AND into the ai:results
                    # payload without any extra plumbing — _publish_face_update
                    # deep-copies meta as-is.
                    analyzer = cam.get("liveness")
                    if analyzer is not None and gray is not None:
                        prev_verdict = meta.get("liveness")
                        verdict = analyzer.update(
                            track_id, gray, (x1, y1, x2, y2),
                            track_landmarks, yaw_pitch_roll, fid,
                        )
                        meta.update(verdict)

                        now_verdict = verdict.get("liveness")
                        if now_verdict != prev_verdict and now_verdict in ("real", "fake"):
                            self.logger.info(
                                "cam=%s track=%d LIVENESS=%s (score=%s, %s)",
                                camera_id, track_id, now_verdict.upper(),
                                verdict.get("liveness_score"), verdict.get("liveness_reason"),
                            )
                            rec = cam.get("recorder")
                            if rec is not None:
                                rec.log(
                                    "SPOOF" if now_verdict == "fake" else "LIVENESS",
                                    f"trk{track_id} {now_verdict} "
                                    f"({verdict.get('liveness_reason')})",
                                    track_id=track_id, data=verdict,
                                )
                            if now_verdict == "fake":
                                debug_extras.save_liveness_reject(
                                    camera_id, track_id, roi_frame, (x1, y1, x2, y2), meta
                                )

                    # ============================================================================
                    # STATE-AWARE SPATIAL TRIGGER HANDLERS (CASES A, B, C)
                    # ============================================================================
                    try:
                        trigger_states = cam.setdefault("trigger_states", {})
                        spatial_events = process_track_triggers(
                            trigger_states=trigger_states,
                            line_points=cam.get("line_points", ((0, 0), (0, 0))),
                            stop_roi=cam.get("stop_roi", ((0, 0), (0, 0), (0, 0), (0, 0))),
                            camera_id=camera_id,
                            track_id=track_id,
                            track_class=track_class,
                            score=score,
                            bbox_roi=(x1, y1, x2, y2),
                            roi_offset=roi_offsets[idx],
                        )

                        for event_type, event_data in spatial_events.items():
                            if event_data is None:
                                continue
                            if event_type not in ["line_cross", "stopped_roi"]:
                                continue
                            if event_type == "line_cross" and not cam.get("cross_line_enabled", False):
                                continue
                            if event_type == "stopped_roi" and not cam.get("flag_stop_roi_enabled", False):
                                continue
                            if meta.get(f"{event_type}_sent", False):
                                continue

                            reg1_crops = cam["best_crops"].get(track_id, {}).get("reg1", [])
                            current_best_fid = reg1_crops[0]["frame_number"] if reg1_crops else None
                            has_rec_result = "identified_as" in meta
                            current_conf = float(meta.get("confidence", 0.0))
                            last_sent_fid = meta.get("last_sent_fid", -1)

                            # CASE A: high-confidence identity already known -> dispatch immediately.
                            if has_rec_result and current_conf >= self.PERIODIC_RECOG_CONF_THRESH:
                                self._publish_face_update(camera_id, track_id, meta, event_type)
                                meta[f"{event_type}_sent"] = True
                            else:
                                # CASE B/C: no/low-confidence identity -> buffer + maybe ask for one.
                                if event_type not in meta["deferred_events"]:
                                    meta["deferred_events"][event_type] = event_data

                                if not meta.get("pending_recognition", False):
                                    if current_best_fid != last_sent_fid:
                                        if self._send_spatial_event_crop(camera_id, track_id, meta, event_type):
                                            if current_best_fid is not None:
                                                meta["last_sent_fid"] = current_best_fid
                    except Exception as trigger_fault:
                        self.logger.error(f"Failed spatial validation step on track {track_id}: {trigger_fault}")

                    # ====================== CONDITIONAL PERIODIC SEND ======================
                    if self._should_send_periodic_check(meta):
                        if cam.get("periodic_enabled", False):
                            reg1_crops = cam["best_crops"].get(track_id, {}).get("reg1", [])
                            current_best_fid = reg1_crops[0]["frame_number"] if reg1_crops else None
                            has_rec_result = "identified_as" in meta
                            current_conf = float(meta.get("confidence", 0.0))
                            last_sent_fid = meta.get("last_sent_fid", -1)

                            should_send = False
                            if meta.get("pending_recognition", False):
                                should_send = False
                            elif not has_rec_result:
                                should_send = current_best_fid != last_sent_fid
                            else:
                                if current_conf >= self.PERIODIC_RECOG_CONF_THRESH:
                                    should_send = False
                                else:
                                    should_send = current_best_fid is not None and current_best_fid > last_sent_fid

                            if should_send:
                                if self._send_intermediate_best_crop(camera_id, track_id, meta):
                                    meta["last_sent_fid"] = current_best_fid

                    if self.save_output and track.detbb is not None:
                        tx1, ty1, tx2, ty2 = map(int, track.detbb)
                        track_vis_data.append({
                            'bbox': (tx1, ty1, tx2, ty2), 'id': track_id, 'class': track_class,
                            'score': score, 'landmarks': track_landmarks, 'yaw_pitch_roll': yaw_pitch_roll,
                            'valid_landmarks': valid_landmarks, 'sharpness': sharp, 'resolution': reso
                        })

                    if cam.get("recorder") is not None:
                        pend_since = meta.get("pending_since", 0.0)
                        debug_tracks.append({
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "cls": track_class,
                            "score": score,
                            "seen_frames": meta.get("seen_frames", 0),
                            "landmarks": track_landmarks,
                            "pose": yaw_pitch_roll,
                            "valid_landmarks": valid_landmarks,
                            "yaw_group": yaw_group,
                            "mlc": mlc,
                            "sharpness": sharp,
                            "resolution": reso,
                            "aspect": ar,
                            "quality_ok": self._quality_check_crop(track_class, crop, sharp),
                            "n_crops": len(cam["best_crops"].get(track_id, {}).get("reg1", [])),
                            "liveness": meta.get("liveness"),
                            "liveness_score": meta.get("liveness_score"),
                            "liveness_reason": meta.get("liveness_reason"),
                            "liveness_evals": meta.get("liveness_evals"),
                            "liveness_metrics": meta.get("liveness_metrics"),
                            "identified_as": meta.get("identified_as"),
                            "rec_confidence": meta.get("confidence", 0.0),
                            "rec_pending": meta.get("recognition_event_type")
                            if meta.get("pending_recognition") else None,
                            "rec_pending_age": (time.time() - pend_since) if pend_since else 0.0,
                            "rec_latency_ms": meta.get("rec_latency_ms"),
                        })

                if self.save_output:
                    self._annotate_and_write_detections(camera_id, roi_frame, track_vis_data, detector_boxes)

                recorder = cam.get("recorder")
                if recorder is not None:
                    recorder.write(roi_frame, self._debug_context(
                        cam, camera_id, fid, detector_boxes, debug_tracks,
                        len(frames), roi_offsets[idx]))

                to_finalize = []
                for tid, tmeta in list(cam["track_meta"].items()):
                    if (fid - int(tmeta["last_seen_fid"])) > self.ABSENT_N:
                        to_finalize.append((tid, tmeta))
                finalize_max_crops = getattr(self, "FINALIZE_MAX_CROPS", 1)
                for tid, tmeta in to_finalize:
                    self._finalize_or_drop_track(camera_id, tid, tmeta, finalize_max_crops)

        self.logger.info("Engine stopping...")
        self.cleanup()

    # ============================================================================
    # Recognition dispatch (Redis-backed)
    # ============================================================================
    def _send_intermediate_best_crop(self, camera_id: str, track_id: int, meta: dict) -> bool:
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        crops_dict = cam["best_crops"].get(track_id, {})
        best_reg = crops_dict.get("reg1", []) or crops_dict.get("reg2", []) or crops_dict.get("reg3", [])
        if not best_reg:
            return False

        best_crop_item = best_reg[0]
        landmarks = best_crop_item.get("landmarks")
        if landmarks is None:
            return False

        mean_conf = float(np.mean(landmarks[:, 2]))
        if mean_conf <= 0.6:
            return False

        encoded = encode_image(best_crop_item["image"])
        if encoded is None:
            return False

        landmarks_payload = [
            {"index": int(i), "x": float(x), "y": float(y), "conf": float(c)}
            for i, (x, y, c) in enumerate(landmarks)
        ]

        task = {
            "task_type": "periodic",
            "engine_id": self.engine_id,
            "process_id": self.engine_id,
            "stream_idx": camera_id,
            "camera_id": camera_id,
            "video_source": cam["url"],
            "track_id": track_id,
            "is_intermediate": True,
            "meta": {
                "seen_frames": meta.get("seen_frames", 0),
                "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                "liveness": meta.get("liveness"),
                "liveness_score": meta.get("liveness_score"),
                "liveness_reason": meta.get("liveness_reason"),
            },
            "crops": {
                "reg1": [{
                    "image_bytes": encoded,
                    "class_flag": best_crop_item.get("class_flag"),
                    "mlc": best_crop_item.get("mlc", 0.0),
                    "yaw_group": best_crop_item.get("yaw_group", 0),
                    "resolution": best_crop_item.get("resolution", 0),
                    "frame_number": best_crop_item.get("frame_number", 0),
                    "landmarks": landmarks_payload,
                    "num_landmarks": len(landmarks_payload),
                    "mean_landmark_conf": mean_conf
                }],
                "reg2": [], "reg3": [],
            },
            "best_frame": None,
        }

        meta["pending_recognition"] = True
        meta["pending_since"] = time.time()
        meta["recognition_event_type"] = "periodic"

        self.logger.info("cam=%s track=%d -> recognizer (periodic, mean_lm_conf=%.2f)",
                          camera_id, track_id, mean_conf)
        rec = self._recorder(camera_id)
        if rec is not None:
            rec.log("REC_SUBMIT", f"trk{track_id} periodic mlc{mean_conf:.2f} "
                                  f"live={meta.get('liveness')}", track_id=track_id)
        submitted_reg = "reg1" if crops_dict.get("reg1") else ("reg2" if crops_dict.get("reg2") else "reg3")
        debug_extras.save_best_crop_montage(camera_id, track_id, "periodic", crops_dict, submitted_reg)
        self.rec_client.submit(task)
        return True

    def _send_spatial_event_crop(self, camera_id: str, track_id: int, meta: dict, event_type: str) -> bool:
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        crops_dict = cam["best_crops"].get(track_id, {})
        best_reg = crops_dict.get("reg1", []) or crops_dict.get("reg2", []) or crops_dict.get("reg3", [])
        if not best_reg:
            return False

        best_crop_item = best_reg[0]
        landmarks = best_crop_item.get("landmarks")
        if landmarks is None:
            return False

        mean_conf = float(np.mean(landmarks[:, 2]))
        if mean_conf <= 0.6:
            return False

        encoded = encode_image(best_crop_item["image"])
        if encoded is None:
            return False

        landmarks_payload = [
            {"index": int(i), "x": float(x), "y": float(y), "conf": float(c)}
            for i, (x, y, c) in enumerate(landmarks)
        ]

        task = {
            "task_type": event_type,
            "engine_id": self.engine_id,
            "process_id": self.engine_id,
            "stream_idx": camera_id,
            "camera_id": camera_id,
            "video_source": cam["url"],
            "track_id": track_id,
            "is_intermediate": True,
            "meta": {
                "seen_frames": meta.get("seen_frames", 0),
                "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                "trigger_context": f"spatial_activation_{event_type}",
                "liveness": meta.get("liveness"),
                "liveness_score": meta.get("liveness_score"),
                "liveness_reason": meta.get("liveness_reason"),
            },
            "crops": {
                "reg1": [{
                    "image_bytes": encoded,
                    "class_flag": best_crop_item.get("class_flag"),
                    "mlc": best_crop_item.get("mlc", 0.0),
                    "yaw_group": best_crop_item.get("yaw_group", 0),
                    "resolution": best_crop_item.get("resolution", 0),
                    "frame_number": best_crop_item.get("frame_number", 0),
                    "landmarks": landmarks_payload,
                    "num_landmarks": len(landmarks_payload),
                    "mean_landmark_conf": mean_conf
                }],
                "reg2": [], "reg3": [],
            },
            "best_frame": None,
        }

        meta["pending_recognition"] = True
        meta["pending_since"] = time.time()
        meta["recognition_event_type"] = event_type
        meta[f"{event_type}_sent"] = True

        self.logger.info("cam=%s track=%d -> recognizer (%s, mean_lm_conf=%.2f)",
                          camera_id, track_id, event_type, mean_conf)
        rec = self._recorder(camera_id)
        if rec is not None:
            rec.log("REC_SUBMIT", f"trk{track_id} {event_type} mlc{mean_conf:.2f} "
                                  f"live={meta.get('liveness')}", track_id=track_id)
        submitted_reg = "reg1" if crops_dict.get("reg1") else ("reg2" if crops_dict.get("reg2") else "reg3")
        debug_extras.save_best_crop_montage(camera_id, track_id, event_type, crops_dict, submitted_reg)
        self.rec_client.submit(task)
        return True

    def _finalize_or_drop_track(self, camera_id: str, track_id: int, meta: Dict[str, int],
                                finalize_max_crops: int = None):
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        leave_scene_trig = cam.get("leave_scene_enabled", False)

        # ROUTE A: fast lane — dispatch whatever identity state we hold now.
        if not leave_scene_trig:
            self.logger.info("cam=%s track=%d -> ai:results (fast finalize, leave_scene disabled)",
                              camera_id, track_id)
            self._publish_face_update(camera_id, track_id, meta, "finalize")
            cam["track_meta"].pop(track_id, None)
            cam["best_crops"].pop(track_id, None)
            cam["best_frame"].pop(track_id, None)
            if "trigger_states" in cam and track_id in cam["trigger_states"]:
                del cam["trigger_states"][track_id]
            if cam.get("liveness") is not None:
                cam["liveness"].drop(track_id)
            return

        # ROUTE B: high-fidelity lane — send crops for a final verification pass.
        seen_frames = int(meta.get("seen_frames", 0))
        crops_dict = cam["best_crops"].get(track_id, {})

        if finalize_max_crops is None:
            finalize_max_crops = getattr(self, "FINALIZE_MAX_CROPS", 1)

        num_crops = max([len(lst) for lst in crops_dict.values()] if crops_dict else [0])
        min_required = 1 if finalize_max_crops == 1 else self.MIN_CROPS_TO_FINALIZE
        should_finalize = (num_crops >= min_required) and (seen_frames >= self.MIN_SEEN_FRAMES)

        if not should_finalize:
            self.logger.info(
                "cam=%s track=%d DROPPED at absence timeout — no dispatch "
                "(crops=%d/%d required, seen_frames=%d/%d required)",
                camera_id, track_id, num_crops, min_required, seen_frames, self.MIN_SEEN_FRAMES,
            )
            rec = self._recorder(camera_id)
            if rec is not None:
                rec.log("DROP", f"trk{track_id} crops={num_crops}/{min_required} "
                                f"seen={seen_frames}/{self.MIN_SEEN_FRAMES}", track_id=track_id)

        if should_finalize:
            crops_data = {"reg1": [], "reg2": [], "reg3": []}
            proceed_with_dispatch = True

            crop_dir = None
            if self.save_output:
                crop_dir = os.path.join(self.output_dir, str(camera_id), "crops", f"track_{track_id}")
                os.makedirs(crop_dir, exist_ok=True)

            if finalize_max_crops == 1:
                best_reg = crops_dict.get("reg1", []) or crops_dict.get("reg2", []) or crops_dict.get("reg3", [])
                if best_reg:
                    best_crop_item = best_reg[0]
                    landmarks = best_crop_item.get("landmarks")

                    if landmarks is not None and len(landmarks) > 0:
                        mean_conf = float(np.mean(landmarks[:, 2]))
                    else:
                        mean_conf = float(best_crop_item.get("mlc", 0.0))

                    encoded = encode_image(best_crop_item["image"])
                    if encoded is not None:
                        landmarks_payload = []
                        if landmarks is not None:
                            landmarks_payload = [
                                {"index": int(i), "x": float(x), "y": float(y), "conf": float(c)}
                                for i, (x, y, c) in enumerate(landmarks)
                            ]

                        mlc = best_crop_item.get("mlc", 0.0)
                        yaw_group = best_crop_item.get("yaw_group", 0)
                        resolution = best_crop_item.get("resolution", 0)

                        if self.save_output:
                            filename = f"{track_id}_reg1_mlc_{mlc:.6f}_yaw_{yaw_group}_res_{resolution}_idx0.jpg"
                            cv2.imwrite(os.path.join(crop_dir, filename), best_crop_item["image"])

                        crops_data["reg1"].append({
                            "image_bytes": encoded, "class_flag": best_crop_item["class_flag"],
                            "mlc": mlc, "yaw_group": yaw_group, "resolution": resolution,
                            "frame_number": best_crop_item["frame_number"],
                            "landmarks": landmarks_payload, "num_landmarks": len(landmarks_payload),
                            "mean_landmark_conf": mean_conf
                        })
                    else:
                        self.logger.warning("cam=%s track=%d DROPPED — best crop failed to JPEG-encode",
                                             camera_id, track_id)
                        proceed_with_dispatch = False
                else:
                    self.logger.warning("cam=%s track=%d DROPPED — should_finalize but best_crops is empty",
                                         camera_id, track_id)
                    proceed_with_dispatch = False
            else:
                for reg_key in ["reg1", "reg2", "reg3"]:
                    for idx, item in enumerate(crops_dict.get(reg_key, [])):
                        img = item["image"]
                        encoded = encode_image(img)
                        if encoded is None:
                            continue

                        mlc = item.get("mlc", 0.0)
                        yaw_group = item.get("yaw_group", 0)
                        resolution = item.get("resolution", 0)

                        landmarks_payload = []
                        if item.get("landmarks") is not None:
                            landmarks_payload = [
                                {"index": int(i), "x": float(x), "y": float(y), "conf": float(c)}
                                for i, (x, y, c) in enumerate(item["landmarks"])
                            ]

                        if self.save_output:
                            filename = f"{track_id}_{reg_key}_mlc_{mlc:.6f}_yaw_{yaw_group}_res_{resolution}_idx{idx}.jpg"
                            cv2.imwrite(os.path.join(crop_dir, filename), img)

                        crops_data[reg_key].append({
                            "image_bytes": encoded, "class_flag": item["class_flag"],
                            "mlc": mlc, "yaw_group": yaw_group, "resolution": resolution,
                            "frame_number": item["frame_number"],
                            "landmarks": landmarks_payload, "num_landmarks": len(landmarks_payload),
                            "mean_landmark_conf": float(np.mean([p["conf"] for p in landmarks_payload]))
                            if landmarks_payload else 0.0
                        })

            if proceed_with_dispatch:
                best_frame_data = None
                if track_id in cam["best_frame"]:
                    bf = cam["best_frame"][track_id]
                    if self.save_output:
                        cv2.imwrite(os.path.join(crop_dir, f"{track_id}_best_frame.jpg"), bf["frame"])
                    encoded_frame = encode_image(bf["frame"])
                    if encoded_frame:
                        best_frame_data = {
                            "image_bytes": encoded_frame, "bbox": bf["bbox"], "resolution": bf["resolution"],
                        }

                task = {
                    "task_type": "finalize",
                    "finalize_max_crops": finalize_max_crops,
                    "engine_id": self.engine_id,
                    "process_id": self.engine_id,
                    "stream_idx": camera_id,
                    "camera_id": camera_id,
                    "video_source": cam["url"],
                    "track_id": track_id,
                    "meta": {
                        "seen_frames": seen_frames,
                        "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                        "liveness": meta.get("liveness"),
                        "liveness_score": meta.get("liveness_score"),
                        "liveness_reason": meta.get("liveness_reason"),
                    },
                    "crops": crops_data,
                    "best_frame": best_frame_data,
                }

                self.logger.info("cam=%s track=%d -> recognizer (finalize, crops=%d, seen_frames=%d)",
                                  camera_id, track_id, num_crops, seen_frames)
                rec = self._recorder(camera_id)
                if rec is not None:
                    rec.log("FINALIZE", f"trk{track_id} -> recognizer crops={num_crops} "
                                        f"live={meta.get('liveness')}", track_id=track_id)
                self.rec_client.submit(task)
                cam.setdefault("finalizing_meta", {})[track_id] = cam["track_meta"].pop(track_id, meta)
            else:
                cam["track_meta"].pop(track_id, None)
        else:
            cam["track_meta"].pop(track_id, None)

        cam["best_crops"].pop(track_id, None)
        cam["best_frame"].pop(track_id, None)
        if "trigger_states" in cam and track_id in cam["trigger_states"]:
            del cam["trigger_states"][track_id]
        if cam.get("liveness") is not None:
            cam["liveness"].drop(track_id)

    def _drain_recognition_outputs(self):
        """Non-blocking drain of everything the recognizer has finished
        for this engine since the last tick — same call-site shape as
        the reference fr_output_queue.get_nowait() loop, backed by
        RecognitionClient.drain() instead."""
        for recognition_payload in self.rec_client.drain():
            if not isinstance(recognition_payload, dict):
                continue

            track_id = recognition_payload.get("track_id")
            event_type = recognition_payload.get("event_type")
            personnel_id = recognition_payload.get("personnelid")
            first_name = recognition_payload.get("first_name")
            last_name = recognition_payload.get("last_name")
            score = recognition_payload.get("detection_score")
            face_image_url = recognition_payload.get("face_image")
            camera_image_url = recognition_payload.get("camera_image")

            self.logger.info(
                f"[FR Output] '{event_type.upper()}' for Track {track_id} -> "
                f"ID: {personnel_id} ({first_name} {last_name}) | Conf: {score if score else 0.0:.4f}"
            )
            for _cid, _cam in self.cameras.items():
                _rec = _cam.get("recorder")
                if _rec is not None:
                    _rec.log("REC_RESULT",
                             f"trk{track_id} {event_type} -> {personnel_id} "
                             f"({first_name} {last_name}) {float(score or 0.0):.3f}",
                             track_id=track_id if isinstance(track_id, int) else None)

            if event_type in ["periodic", "line_cross", "stopped_roi"]:
                for cam_id, cam in self.cameras.items():
                    if track_id in cam.get("track_meta", {}):
                        meta = cam["track_meta"][track_id]
                        meta.setdefault("recognition_history", {})
                        meta.setdefault("deferred_events", {})

                        step_snapshot = {
                            "personnel_id": personnel_id, "first_name": first_name, "last_name": last_name,
                            "confidence": float(score if score else 0.0),
                            "face_image_url": face_image_url, "camera_image_url": camera_image_url,
                            "timestamp": time.time()
                        }

                        if event_type == "periodic":
                            meta["recognition_history"].setdefault("periodic", [])
                            meta["recognition_history"]["periodic"].append(step_snapshot)
                        else:
                            meta["recognition_history"][event_type] = step_snapshot

                        if camera_image_url:
                            meta["last_saved_camera_image"] = camera_image_url

                        old_max_conf = float(meta.get("confidence", 0.0))
                        new_conf = float(score if score else 0.0)
                        if "identified_as" not in meta or new_conf > old_max_conf:
                            meta["identified_as"] = personnel_id
                            meta["confidence"] = new_conf
                            meta["last_saved_face_image"] = face_image_url

                        # Round-trip latency: submit -> ai:results, shown
                        # in the debug video HUD and logged below. Purely
                        # observational — never gates anything.
                        pending_since = meta.get("pending_since")
                        if pending_since:
                            meta["rec_latency_ms"] = (time.time() - pending_since) * 1000.0

                        meta["pending_recognition"] = False

                        deferred_map = meta.get("deferred_events", {})
                        if event_type in ["line_cross", "stopped_roi"]:
                            deferred_map.pop(event_type, None)
                            self._publish_face_update(cam_id, track_id, meta, event_type)
                            meta[f"{event_type}_sent"] = True

                        for cached_type in list(deferred_map.keys()):
                            deferred_map.pop(cached_type)
                            self._publish_face_update(cam_id, track_id, meta, cached_type)
                            meta[f"{cached_type}_sent"] = True

            elif event_type == "finalize":
                for cam_id, cam in self.cameras.items():
                    if track_id in cam.get("finalizing_meta", {}):
                        meta = cam["finalizing_meta"][track_id]

                        if camera_image_url:
                            meta["last_saved_camera_image"] = camera_image_url

                        old_max_conf = float(meta.get("confidence", 0.0))
                        new_conf = float(score if score else 0.0)
                        if "identified_as" not in meta or new_conf > old_max_conf:
                            meta["identified_as"] = personnel_id
                            meta["confidence"] = new_conf
                            meta["last_saved_face_image"] = face_image_url
                            meta["first_name"] = first_name
                            meta["last_name"] = last_name

                        pending_since = meta.get("pending_since")
                        if pending_since:
                            meta["rec_latency_ms"] = (time.time() - pending_since) * 1000.0

                        self._publish_face_update(cam_id, track_id, meta, "finalize")
                        cam["finalizing_meta"].pop(track_id, None)
                        break

    def _publish_face_update(self, camera_id, track_id, meta, update_type):
        """Publishes a mid-track / final face-recognition update onto
        `{module}:ai:results` (the backend contract), replacing the
        reference pipeline's HTTP POST to Django."""
        raw_meta_snapshot = copy.deepcopy(meta)

        payload = {
            "camera_id": camera_id,
            "track_id": track_id,
            "event_type": update_type,
            "timestamp": time.time(),
            "is_final": update_type == "finalize",
            "meta": raw_meta_snapshot,
        }

        try:
            self.bus.push_result(payload, json_encoder=DateTimeEncoder)
            rec = self._recorder(camera_id)
            if rec is not None:
                rec.log("PUBLISH",
                        f"trk{track_id} {update_type} id={meta.get('identified_as')} "
                        f"conf={float(meta.get('confidence', 0.0)):.3f} "
                        f"live={meta.get('liveness')}",
                        track_id=track_id if isinstance(track_id, int) else None)
        except Exception:
            self.logger.exception("failed to publish face update for track %s (%s)", track_id, update_type)

    def cleanup(self):
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)

        self.rec_client.stop()

        for w in list(self.writers.values()):
            try:
                w.release()
            except Exception:
                pass
        self.writers.clear()

        time.sleep(0.3)


# --------------------------------------------------------------------
# Engine process entry point
# --------------------------------------------------------------------
def _engine_process_main(
        engine_id: int,
        model_path: str,
        imgsz: int,
        conf: float,
        save_output: bool,
        save_as_video: bool,
        status_queue: mp.Queue,
        control_queue: mp.Queue,
        stop_event: mp.Event,
        output_dir: str,
        class_labels: Dict[int, str],
):
    setup_logger(f"Engine{engine_id}")
    eng = Engine(
        engine_id=engine_id,
        model_path=model_path,
        imgsz=imgsz,
        conf=conf,
        save_output=save_output,
        save_as_video=save_as_video,
        status_queue=status_queue,
        control_queue=control_queue,
        stop_event=stop_event,
        output_dir=output_dir,
        class_labels=class_labels,
    )
    eng.run()
