"""
engine.py
--------------------------------------------------------------------
The plate detection/tracking/trigger engine — one YOLO model per
process, handling however many cameras EngineManager assigns it via
batched Ultralytics inference. Ported from the reference
video_processor.py's `Engine` class with one structural change: OCR is
no longer an in-process pool of `OCRWorker` subprocesses this Engine
spawns and owns (`ocr_input_queue`/`ocr_output_queue`,
`mp.Queue`-based) — it is the separate `ocr_service`, reached over
Redis through `OcrClient` (see ocr_client.py). Detection capacity and
OCR capacity now scale independently.

Everything else — batch inference, BYTETrack update, spatial triggers,
best-crop ranking, finalize/drop logic, the debug recorder hookup — is
the same math and the same log lines as the reference pipeline, just
reorganized: triggers.py/rtsp_reader.py/tracker.py/debug_recorder.py
are now their own files instead of living inside this one 2900-line
module.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import datetime
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from debug_recorder import DebugConfig, DebugRecorder
from platecore.bus import RedisBus
from platecore.codec import DateTimeEncoder
from platecore.logging_setup import setup_logger
from ocr_client import OcrClient
from rtsp_reader import RTSPStreamReader, encode_image, measure_sharpness
from tracker import BYTETracker, PlateTrackerConfig
from triggers import TriggerTrackState, process_track_triggers
import debug_extras


# --------------------------------------------------------------------
# Compatibility: keep the state object the reference alpr_service.py used
# --------------------------------------------------------------------
class PlateLPRState:
    def __init__(self):
        self.running = True
        self.finished = False

    def stop(self):
        self.running = False

    def set_finished(self):
        self.finished = True


def build_tracker_config() -> PlateTrackerConfig:
    """Builds tracker.py's full PlateTrackerConfig from config.TRACKER_*.

    Previously this module defined its OWN small local `TrackerConfig`
    (track_thresh/match_thresh/track_buffer/nms_thresh/mot20 only) and
    passed THAT to BYTETracker — even though tracker.py ships a much
    richer PlateTrackerConfig with ~19 additional fields (the "FIX 1-7"
    enhancements: young-track survival, new-track motion seeding, a
    recovery pass, GMC camera-jolt compensation). BYTETracker.__init__
    reads every field via getattr(args, name, default), so those extra
    fields were always silently falling back to PlateTrackerConfig's own
    hardcoded defaults — current runtime behavior is unchanged, but none
    of it was configurable. Every default in config.py's TRACKER_* block
    matches PlateTrackerConfig's own declared default exactly, so an
    unset .env reproduces prior behavior bit-for-bit."""
    return PlateTrackerConfig(
        track_thresh=config.TRACKER_TRACK_THRESH,
        match_thresh=config.TRACKER_MATCH_THRESH,
        track_buffer=config.TRACKER_TRACK_BUFFER,
        nms_thresh=config.TRACKER_NMS_THRESH,
        mot20=config.TRACKER_MOT20,
        second_thresh=config.TRACKER_SECOND_THRESH,
        duplicate_thresh=config.TRACKER_DUPLICATE_THRESH,
        predict_unconfirmed=config.TRACKER_PREDICT_UNCONFIRMED,
        unconfirmed_thresh=config.TRACKER_UNCONFIRMED_THRESH,
        unconfirmed_max_miss=config.TRACKER_UNCONFIRMED_MAX_MISS,
        new_track_vel_std_scale=config.TRACKER_NEW_TRACK_VEL_STD_SCALE,
        seed_velocity_on_first_update=config.TRACKER_SEED_VELOCITY_ON_FIRST_UPDATE,
        seed_max_ratio=config.TRACKER_SEED_MAX_RATIO,
        reseed_after_gap=config.TRACKER_RESEED_AFTER_GAP,
        recovery_enabled=config.TRACKER_RECOVERY_ENABLED,
        recovery_thresh=config.TRACKER_RECOVERY_THRESH,
        recovery_expansion=config.TRACKER_RECOVERY_EXPANSION,
        recovery_base_radius=config.TRACKER_RECOVERY_BASE_RADIUS,
        recovery_radius_growth=config.TRACKER_RECOVERY_RADIUS_GROWTH,
        recovery_max_radius=config.TRACKER_RECOVERY_MAX_RADIUS,
        recovery_shape_weight=config.TRACKER_RECOVERY_SHAPE_WEIGHT,
        recovery_class_penalty=config.TRACKER_RECOVERY_CLASS_PENALTY,
        gmc_enabled=config.TRACKER_GMC_ENABLED,
        gmc_min_pairs=config.TRACKER_GMC_MIN_PAIRS,
        gmc_max_shift_ratio=config.TRACKER_GMC_MAX_SHIFT_RATIO,
        max_removed_history=config.TRACKER_MAX_REMOVED_HISTORY,
    )


def resolve_device(preference: str, strict: bool, logger: logging.Logger) -> str:
    """auto | cpu | cuda | cuda:N -> a concrete torch device string.
    NEW — the reference Engine hardcoded self.device = "cpu" (a leftover
    from a benchmarking session). This restores the auto-detect the
    original comment said used to be there, gated by DETECTION_DEVICE."""
    pref = (preference or "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        msg = f"DETECTION_DEVICE={preference!r} requested but CUDA is not available"
        if strict:
            raise RuntimeError(msg)
        logger.error("%s — falling back to CPU", msg)
        return "cpu"
    return pref


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
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
            absent_n: int = 30,
            min_seen_frames: int = 8,
            min_crops_to_finalize: int = 1,
            n_best: int = 5,
            conf_digits: int = 2,
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

        # track lifecycle config
        self.ABSENT_N = int(absent_n)
        self.MIN_SEEN_FRAMES = int(min_seen_frames)
        self.MIN_CROPS_TO_FINALIZE = int(min_crops_to_finalize)

        # best crops config
        self.N_BEST = int(n_best)
        self.conf_digits = int(conf_digits)

        # cameras: camera_id -> camera state
        self.cameras: Dict[str, Dict[str, Any]] = {}

        # YOLO model
        self.model = YOLO(self.model_path).to(self.device)
        self.logger.info(
            f"[INIT] YOLO model loaded model={self.model_path} device={self.device} "
            f"imgsz={self.imgsz} conf={self.conf}"
        )

        # OCR — pushed to the separate ocr_service over Redis instead of
        # spawned in-process (see ocr_client.py's docstring).
        self.bus = RedisBus(module=config.REDIS_MODULE)
        self.ocr_client = OcrClient(self.bus, self.engine_id)
        self.logger.info("OCR client ready (engine_id=%s)", self.engine_id)

        # writers optional
        self.writers: Dict[str, cv2.VideoWriter] = {}
        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)

        self._last_status_emit = 0.0

        # DEBUG VIDEO: one annotated recorder per camera.
        self.debug_cfg = DebugConfig()
        self._last_infer_ms = 0.0
        self._batch_count = 0
        self._last_infer_breakdown: Dict[str, float] = {}
        if self.debug_cfg.enabled:
            self.logger.info(f"[DEBUG-REC] enabled -> {self.debug_cfg.as_dict()}")
        else:
            self.logger.info("[DEBUG-REC] disabled (set DEBUG_VIDEO_ENABLED=1 to record)")

    # ---------------- debug helpers ----------------
    def _dbg(self, camera_id: str) -> Optional[DebugRecorder]:
        cam = self.cameras.get(str(camera_id))
        if not cam:
            return None
        return cam.get("debug")

    def _dbg_log(self, camera_id: str, kind: str, text: str,
                 track_id: Optional[int] = None, data: Optional[dict] = None):
        rec = self._dbg(camera_id)
        if rec is not None:
            try:
                rec.log(kind, text, track_id=track_id, data=data)
            except Exception:
                pass

    def _ocr_rows_for_track(self, cam: Dict[str, Any], track_id: int):
        rows = []
        store = cam.get("track_ocr", {}).get(track_id, {}) or {}
        for stage in ("cross_line", "stop_roi", "periodic", "leave_scene"):
            res = store.get(stage)
            if not res:
                continue
            conf = res.get("confidence")
            conf = 0.0 if conf is None else float(conf)
            rows.append((
                stage,
                f"{str(res.get('plate_text'))[:12]:<12} {conf:.2f} "
                f"{'OK' if res.get('is_valid') else 'INVALID'}"
            ))
        return rows

    def _handle_ocr_result(self, result: dict) -> bool:
        if not isinstance(result, dict):
            self.logger.warning(f"[OCR-RESULT] Ignoring non-dict OCR result: {type(result)}")
            return False

        task_id = result.get("task_id")
        camera_id = result.get("camera_id")
        track_id = result.get("track_id")
        trigger_type = result.get("trigger_type")

        missing = [
            name for name, value in {
                "task_id": task_id, "camera_id": camera_id,
                "track_id": track_id, "trigger_type": trigger_type,
            }.items() if value is None
        ]
        if missing:
            self.logger.warning(f"[OCR-RESULT] Dropping result with missing routing fields={missing}; result={result}")
            return False

        valid_trigger_types = {"cross_line", "stop_roi", "periodic", "leave_scene"}
        if trigger_type not in valid_trigger_types:
            self.logger.warning(f"[OCR-RESULT] Dropping result with invalid trigger_type={trigger_type!r}; task_id={task_id}")
            return False

        cam = self.cameras.get(camera_id)
        if not cam:
            self.logger.warning(f"[OCR-RESULT] Dropping result for unknown camera_id={camera_id!r}; task_id={task_id}")
            return False

        if track_id not in cam.get("track_meta", {}):
            self.logger.warning(
                f"[OCR-LATE] result arrived after track cleanup camera={camera_id} "
                f"track={track_id} trigger={trigger_type} task={task_id}"
            )
            self._dbg_log(camera_id, "OCR_LATE", f"#{track_id} {trigger_type} arrived after cleanup",
                          track_id=track_id, data={"task_id": task_id})
            return False

        cam.setdefault("track_ocr", {})
        cam.setdefault("track_paths", {})

        track_bucket = cam["track_ocr"].setdefault(track_id, {})
        paths_bucket = cam["track_paths"].setdefault(track_id, {})

        track_bucket[trigger_type] = result

        payload = result.get("payload", {}) or {}
        plate_path = payload.get("plate_image")
        frame_path = payload.get("frame_image")
        paths_bucket[trigger_type] = {"plate_path": plate_path, "frame_path": frame_path}

        self.logger.info(
            f"[OCR-RESULT] Stored OCR result camera_id={camera_id} track_id={track_id} "
            f"trigger_type={trigger_type} status={result.get('status')} valid={result.get('is_valid')} "
            f"plate={result.get('plate_text')!r} task_id={task_id} paths={paths_bucket[trigger_type]}"
        )

        _conf = result.get("confidence")
        self._dbg_log(
            camera_id, "OCR_RESULT",
            f"#{track_id} {trigger_type} -> '{result.get('plate_text')}' "
            f"conf={0.0 if _conf is None else float(_conf):.2f} valid={result.get('is_valid')}",
            track_id=track_id,
            data={"task_id": task_id, "status": result.get("status"), "plate_text": result.get("plate_text"),
                  "confidence": _conf, "is_valid": result.get("is_valid"),
                  "plate_path": plate_path, "frame_path": frame_path},
        )

        tmeta = cam.get("track_meta", {}).get(track_id)
        if tmeta is not None:
            if self._is_track_waiting_for_ocr(tmeta) and tmeta.get("ocr_pending_trigger_type") == trigger_type:
                # Round-trip latency: detector-submit -> result-arrival,
                # the analog of the face module's pending_since/
                # rec_latency_ms pattern. Complements ocr_service's own
                # _queue_latency_ms(task_id) (detector-submit -> worker
                # picked it up) — together the two numbers say whether a
                # slow OCR round trip is queue wait or actual processing.
                submitted_at = tmeta.get("ocr_pending_submitted_at")
                ocr_latency_ms = (time.time() - float(submitted_at)) * 1000.0 if submitted_at else None
                self._clear_track_ocr_pending(tmeta)
                if ocr_latency_ms is not None:
                    tmeta["last_ocr_latency_ms"] = round(ocr_latency_ms, 1)
                self.logger.info(
                    f"[OCR-RESULT] Cleared OCR pending meta camera_id={camera_id} "
                    f"track_id={track_id} trigger_type={trigger_type} task_id={task_id} "
                    f"ocr_latency_ms={tmeta.get('last_ocr_latency_ms')}"
                )
                self._dbg_log(camera_id, "OCR_LATENCY",
                              f"#{track_id} {trigger_type} round_trip={tmeta.get('last_ocr_latency_ms')}ms",
                              track_id=track_id, data={"ocr_latency_ms": tmeta.get("last_ocr_latency_ms")})

            if trigger_type in ("cross_line", "stop_roi"):
                self._publish_vehicle_update(camera_id=camera_id, track_id=track_id, meta=tmeta, update_type=trigger_type)

        if trigger_type == "leave_scene":
            tmeta = cam.get("track_meta", {}).get(track_id)
            if tmeta is not None and tmeta.get("finalize_waiting_for_ocr", False):
                expected_stage = tmeta.get("finalize_ocr_trigger_type", "leave_scene")
                if expected_stage == "leave_scene":
                    self.logger.info(
                        f"[OCR-RESULT->FINALIZE] camera={camera_id} track_id={track_id} "
                        f"trigger_type={trigger_type} task_id={task_id}"
                    )
                    tmeta.pop("finalize_waiting_for_ocr", None)
                    tmeta.pop("finalize_ocr_trigger_type", None)
                    tmeta.pop("finalize_ocr_submitted_at", None)
                    self._finalize_or_drop_track(camera_id, track_id, tmeta)

        return True

    def _drain_ocr_results(self, max_items: int = 100) -> int:
        drained = 0
        for result in self.ocr_client.drain()[:max_items]:
            try:
                self._handle_ocr_result(result)
            except Exception as e:
                self.logger.exception(f"[OCR-RESULT] Failed handling OCR result: {e}")
            drained += 1
        return drained

    # ---------------- ranking ----------------
    def _rank(self, det_score: float, resolution: int) -> Tuple[int, int]:
        n = self.conf_digits
        conf_bucket = int(float(det_score) * (10 ** n))
        return (conf_bucket, int(resolution))

    # ---------------- quality ----------------
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

    # ---------------- camera add/remove ----------------
    def add_camera(self, camera_id: str, url: str, roi: Tuple[float, float, float, float],
                    cond_per_trig: bool = False, cross_line_trig: bool = False,
                    stop_roi_trig: bool = False, leave_scene_trig: bool = False,
                    line_p1_x=0.0, line_p1_y=0.0, line_p2_x=0.0, line_p2_y=0.0,
                    stop_roi_p1_x=0.0, stop_roi_p1_y=0.0, stop_roi_p2_x=0.0, stop_roi_p2_y=0.0,
                    stop_roi_p3_x=0.0, stop_roi_p3_y=0.0, stop_roi_p4_x=0.0, stop_roi_p4_y=0.0):
        camera_id = str(camera_id)

        line_points = ((line_p1_x, line_p1_y), (line_p2_x, line_p2_y))
        stop_roi = (
            (stop_roi_p1_x, stop_roi_p1_y), (stop_roi_p2_x, stop_roi_p2_y),
            (stop_roi_p3_x, stop_roi_p3_y), (stop_roi_p4_x, stop_roi_p4_y),
        )
        triggers = {
            "cond_per_trig": bool(cond_per_trig),
            "cross_line_trig": bool(cross_line_trig),
            "stop_roi_trig": bool(stop_roi_trig),
            "leave_scene_trig": bool(leave_scene_trig),
        }

        if camera_id in self.cameras:
            self.cameras[camera_id]["url"] = url
            self.cameras[camera_id]["roi"] = roi
            self.cameras[camera_id]["line_points"] = line_points
            self.cameras[camera_id]["stop_roi"] = stop_roi
            self.cameras[camera_id]["triggers"] = triggers
            self.cameras[camera_id]["line_points_px"] = None
            self.cameras[camera_id]["stop_roi_px"] = None
            self.cameras[camera_id]["trigger_states"] = {}
            self._dbg_log(camera_id, "CAMERA", "config updated url/roi/line/stop_roi/triggers",
                          data={"url": url, "roi": roi, "line_points": line_points,
                                "stop_roi": stop_roi, "triggers": triggers})
            self.logger.info(f"Updated camera {camera_id} -> {url}")
            return

        reader = RTSPStreamReader(url, camera_id).start()
        tracker = BYTETracker(build_tracker_config())

        self.cameras[camera_id] = {
            "camera_id": camera_id, "url": url, "roi": roi,
            "line_points": line_points, "stop_roi": stop_roi,
            "line_points_px": None, "stop_roi_px": None, "triggers": triggers,
            "trigger_states": {}, "track_ocr": {}, "track_paths": {},
            "reader": reader, "tracker": tracker, "fid": 0, "frames_processed": 0,
            "last_frame_ts": None, "track_meta": {}, "best_crops": {}, "best_frame": {},
            "active": True,
            "debug": DebugRecorder(camera_id, self.engine_id, self.debug_cfg, self.logger),
            "dbg_read_fail": 0,
        }

        self._dbg_log(camera_id, "CAMERA", f"camera added -> {url}",
                      data={"roi": roi, "line_points": line_points, "stop_roi": stop_roi, "triggers": triggers})
        self._send_msg(camera_id, {"status": "running"})
        self.logger.info(
            f"[CAMERA-ADD] camera={camera_id} url={url} roi={roi} "
            f"line_points={line_points} stop_roi={stop_roi} triggers={triggers}"
        )

    def remove_camera(self, camera_id: str):
        camera_id = str(camera_id)
        cam = self.cameras.pop(camera_id, None)
        if not cam:
            return
        try:
            cam["reader"].stop()
        except Exception:
            pass
        rec = cam.get("debug")
        if rec is not None:
            try:
                rec.log("CAMERA", "camera removed")
                rec.close()
            except Exception:
                pass
        w = self.writers.pop(camera_id, None)
        if w is not None:
            try:
                w.release()
            except Exception:
                pass
        self._send_msg(camera_id, {"status": "stopped"})
        self.logger.info(f"Removed camera {camera_id}")

    def camera_ids(self) -> List[str]:
        return list(self.cameras.keys())

    # ---------------- IPC messages to detector process ----------------
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

    def _set_best_frame(self, cam: Dict[str, Any], track_id: int,
                        full_frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int],
                        det_score: float, resolution: int):
        cam["best_frame"][track_id] = {
            "frame": full_frame.copy(), "bbox": tuple(map(int, bbox_xyxy)),
            "det_score": float(det_score), "resolution": int(resolution), "ts": time.time(),
        }

    # ---------------- control queue drain ----------------
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
                kwargs = {k: v for k, v in cmd.items() if k not in ("cmd", "camera_id", "url", "roi")}
                self.add_camera(cmd["camera_id"], cmd["url"], cmd.get("roi", (0, 0, 1, 1)), **kwargs)
            elif c == "remove":
                self.remove_camera(cmd["camera_id"])
            else:
                self.logger.warning(f"Unknown cmd: {cmd}")

    # ---------------- best crops update ----------------
    def _update_best_crops(self, cam: Dict[str, Any], track_id: int, track_class: int,
                           crop_img: np.ndarray, det_score: float, resolution: int,
                           aspect_ratio: float, sharpness: float) -> Tuple[bool, bool]:
        store = cam["best_crops"].setdefault(track_id, [])
        candidate = {
            "image": crop_img.copy(), "class_flag": int(track_class), "det_score": float(det_score),
            "resolution": int(resolution), "aspect_ratio": float(aspect_ratio), "sharpness": float(sharpness),
            "frame_number": int(cam["fid"]),
        }
        cand_rank = self._rank(det_score, resolution)
        store.sort(key=lambda x: self._rank(x["det_score"], x["resolution"]), reverse=True)
        old_best = store[0] if store else None
        old_best_rank = self._rank(old_best["det_score"], old_best["resolution"]) if old_best else None

        if len(store) < self.N_BEST:
            store.append(candidate)
            store.sort(key=lambda x: self._rank(x["det_score"], x["resolution"]), reverse=True)
            updated_best = (old_best_rank is None) or (cand_rank > old_best_rank)
            return True, updated_best

        worst = store[-1]
        worst_rank = self._rank(worst["det_score"], worst["resolution"])
        if cand_rank > worst_rank:
            store[-1] = candidate
            store.sort(key=lambda x: self._rank(x["det_score"], x["resolution"]), reverse=True)
            updated_best = (old_best_rank is None) or (cand_rank > old_best_rank)
            return True, updated_best
        return False, False

    # ---------------- OCR submission ----------------
    def _is_track_waiting_for_ocr(self, meta):
        return bool(meta.get("ocr_waiting_for_result", False))

    def _mark_track_ocr_pending(self, meta, trigger_type):
        meta["ocr_waiting_for_result"] = True
        meta["ocr_pending_trigger_type"] = trigger_type
        meta["ocr_pending_submitted_at"] = time.time()

    def _clear_track_ocr_pending(self, meta):
        meta["ocr_waiting_for_result"] = False
        meta["ocr_pending_trigger_type"] = None
        meta["ocr_pending_submitted_at"] = None

    def _should_submit_ocr_for_stage(self, cam, camera_id, track_id, submit_stage,
                                      ocr_conf_threshold=None):
        if ocr_conf_threshold is None:
            ocr_conf_threshold = config.OCR_CONF_SKIP_THRESHOLD
        existing_ocr = cam.get("track_ocr", {}).get(track_id, {})
        real_ocr_results = [(s, r) for s, r in existing_ocr.items() if r is not None]

        if not real_ocr_results:
            return True

        best_conf, best_stage, best_result = -1.0, None, None
        for old_stage, ocr_res in real_ocr_results:
            conf_value = float(ocr_res.get("confidence") or 0.0)
            if conf_value > best_conf:
                best_conf, best_stage, best_result = conf_value, old_stage, ocr_res

        if best_conf >= ocr_conf_threshold:
            self._dbg_log(camera_id, "OCR_SKIP",
                          f"#{track_id} {submit_stage} skipped: have "
                          f"'{best_result.get('plate_text')}' {best_conf:.2f} from {best_stage}",
                          track_id=track_id)
            return False
        return True

    def _submit_track_to_ocr(self, camera_id: str, track_id: int, meta: Dict[str, int],
                              trigger_type: str = "periodic") -> bool:
        cam = self.cameras.get(camera_id)
        if not cam:
            return False

        valid_trigger_types = {"cross_line", "stop_roi", "leave_scene", "periodic"}
        if trigger_type not in valid_trigger_types:
            self.logger.error(f"[OCR-TASK] invalid trigger_type={trigger_type!r} camera={camera_id} track={track_id}")
            return False

        seen_frames = int(meta.get("seen_frames", 0))
        crops = cam["best_crops"].get(track_id, [])
        if not crops:
            return False

        crops_data = []
        for item in crops:
            encoded = encode_image(item["image"])
            if encoded is None:
                continue
            crops_data.append({
                "image_bytes": encoded, "class_flag": item["class_flag"], "score": item["det_score"],
                "resolution": item["resolution"], "frame_number": item["frame_number"],
            })
        if not crops_data:
            return False

        best_frame_data = None
        if track_id in cam["best_frame"]:
            bf = cam["best_frame"][track_id]
            encoded_frame = encode_image(bf["frame"])
            if encoded_frame:
                best_frame_data = {"image_bytes": encoded_frame, "bbox": bf["bbox"], "resolution": bf["resolution"]}

        task_id = f"{self.engine_id}:{camera_id}:{track_id}:{trigger_type}:{time.time_ns()}"
        task = {
            "task_id": task_id, "process_id": self.engine_id, "stream_idx": camera_id,
            "camera_id": camera_id, "video_source": cam["url"], "track_id": track_id,
            "trigger_type": trigger_type,
            "meta": {
                "seen_frames": seen_frames,
                "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                "trigger_reason": trigger_type, "trigger_snapshot": True,
            },
            "crops": crops_data, "best_frame": best_frame_data,
        }
        self.ocr_client.submit(task)

        if config.DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED:
            debug_extras.save_ocr_submission_montage(
                camera_id=camera_id, track_id=track_id, trigger_type=trigger_type,
                task_id=task_id, crops=crops, best_frame=cam["best_frame"].get(track_id),
                logger=self.logger,
            )

        det_scores = [round(float(c["score"]), 3) for c in crops_data]
        self.logger.info(
            f"[OCR-SUBMIT] camera={camera_id} track={track_id} stage={trigger_type} "
            f"seen_frames={seen_frames} crops={len(crops_data)} det_scores={det_scores} "
            f"best_frame={'yes' if best_frame_data else 'no'} task_id={task_id}"
        )
        self._dbg_log(camera_id, "OCR_SUBMIT",
                      f"#{track_id} {trigger_type} crops={len(crops_data)} "
                      f"best_frame={'yes' if best_frame_data else 'no'} seen={seen_frames}f",
                      track_id=track_id,
                      data={"task_id": task_id, "trigger_type": trigger_type,
                            "n_crops": len(crops_data), "det_scores": det_scores})
        return True

    # ---------------- publish / finalize ----------------
    def _publish_vehicle_update(self, camera_id, track_id, meta, update_type):
        cam = self.cameras.get(camera_id)
        if not cam:
            return
        track_paths = cam.get("track_paths", {}).get(track_id, {})
        payload = {
            "process_id": self.engine_id, "camera_id": camera_id, "track_id": track_id,
            "update_type": update_type, "is_final": False, "meta": meta,
            "events": meta.get("events", {}),
            "ocr_results": cam.get("track_ocr", {}).get(track_id, {}),
            "track_paths": track_paths,
        }
        try:
            self.bus.push_result(payload, json_encoder=DateTimeEncoder)
            self._dbg_log(camera_id, "PUBLISH", f"#{track_id} {update_type} -> core (not final)",
                          track_id=track_id,
                          data={"update_type": update_type, "is_final": False,
                                "events": list(meta.get("events", {}).keys()), "track_paths": track_paths})
            self.logger.info(
                f"[PUBLISH] camera={camera_id} track={track_id} update_type={update_type} "
                f"events={list(meta.get('events', {}).keys())}"
            )
        except Exception as e:
            self._dbg_log(camera_id, "ERROR", f"#{track_id} redis publish FAILED ({update_type}): {e}", track_id=track_id)
            self.logger.error(f"[PUBLISH-FAILED] camera={camera_id} track={track_id} update_type={update_type}: {e}")

    def _finalize_or_drop_track(self, camera_id: str, track_id: int, meta: dict):
        cam = self.cameras.get(camera_id)
        if not cam:
            return

        seen_frames = int(meta.get("seen_frames", 0))
        crops = cam["best_crops"].get(track_id, [])
        num_crops = len(crops)
        should_finalize = (num_crops >= self.MIN_CROPS_TO_FINALIZE) and (seen_frames >= self.MIN_SEEN_FRAMES)

        if should_finalize:
            ocr_results = cam.get("track_ocr", {}).get(track_id, {})
            track_paths = cam.get("track_paths", {}).get(track_id, {})
            payload = {
                "update_type": "leave_scene", "is_final": True, "process_id": self.engine_id,
                "stream_idx": camera_id, "camera_id": camera_id, "video_source": cam["url"],
                "track_id": track_id,
                "meta": {
                    "seen_frames": seen_frames,
                    "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                },
                "events": meta.get("events", {}), "ocr_results": ocr_results, "track_paths": track_paths,
            }
            try:
                self.bus.push_result(payload, json_encoder=DateTimeEncoder)
                _plates = {s: r.get("plate_text") for s, r in (ocr_results or {}).items() if r}
                self._dbg_log(camera_id, "FINALIZE",
                              f"#{track_id} leave_scene FINAL -> core seen={seen_frames}f crops={num_crops} plates={_plates}",
                              track_id=track_id,
                              data={"is_final": True, "events": list(payload["events"].keys()),
                                    "plates": _plates, "track_paths": track_paths})
                lifetime_s = time.time() - float(meta.get("first_seen_wall_ts", time.time()))
                self.logger.info(
                    f"[FINALIZE] camera={camera_id} track={track_id} -> published leave_scene "
                    f"seen_frames={seen_frames} crops={num_crops} lifetime={lifetime_s:.2f}s "
                    f"events={list(payload['events'].keys())} plates={_plates}"
                )
            except Exception as e:
                self._dbg_log(camera_id, "ERROR", f"#{track_id} finalize publish FAILED: {e}", track_id=track_id)
                self.logger.error(f"[FINALIZE-FAILED] camera={camera_id} track={track_id}: {e}")
        else:
            done = [s for s, r in cam.get("track_ocr", {}).get(track_id, {}).items() if r]
            _reasons = []
            if num_crops < self.MIN_CROPS_TO_FINALIZE:
                _reasons.append(f"crops {num_crops}<{self.MIN_CROPS_TO_FINALIZE}")
            if seen_frames < self.MIN_SEEN_FRAMES:
                _reasons.append(f"seen {seen_frames}<{self.MIN_SEEN_FRAMES}")
            self._dbg_log(camera_id, "DROP", f"#{track_id} DROPPED: {', '.join(_reasons)} (ocr done: {done})",
                          track_id=track_id,
                          data={"seen_frames": seen_frames, "n_crops": num_crops, "ocr_stages_done": done})
            lifetime_s = time.time() - float(meta.get("first_seen_wall_ts", time.time()))
            first_fid = meta.get("first_seen_fid", 0)
            last_fid = meta.get("last_seen_fid", 0)
            self.logger.warning(
                f"[DROP] camera={camera_id} track={track_id} reasons=[{', '.join(_reasons)}] "
                f"seen_frames={seen_frames}/{self.MIN_SEEN_FRAMES}min crops={num_crops}/{self.MIN_CROPS_TO_FINALIZE}min "
                f"lifetime={lifetime_s:.2f}s fid_range=[{first_fid}..{last_fid}] ocr_stages_done={done}"
            )

        cam["track_meta"].pop(track_id, None)
        cam["best_crops"].pop(track_id, None)
        cam["best_frame"].pop(track_id, None)
        cam.get("trigger_states", {}).pop(track_id, None)
        cam.get("track_ocr", {}).pop(track_id, None)
        cam.get("track_paths", {}).pop(track_id, None)

    # ---------------- debug frame assembly ----------------
    def _write_debug_frame(self, cam, camera_id, full_frame, roi_offset, dbg_dets, dbg_tracks, fid):
        rec: DebugRecorder = cam["debug"]
        n_live = len(dbg_tracks)
        for tid, tmeta in list(cam.get("track_meta", {}).items()):
            if tid in dbg_tracks:
                continue
            bbox = tmeta.get("last_bbox_full")
            if bbox is None:
                continue
            absent = int(fid) - int(tmeta.get("last_seen_fid", fid))
            if absent > max(self.ABSENT_N, self.debug_cfg.ghost_frames):
                continue
            tstate = cam.get("trigger_states", {}).get(tid)
            crops_store = cam.get("best_crops", {}).get(tid, [])
            best_item = crops_store[0] if crops_store else None
            dbg_tracks[tid] = {
                "track_id": tid, "ghost": True, "bbox_full": bbox,
                "cls": best_item["class_flag"] if best_item else "?",
                "score": float(best_item["det_score"]) if best_item else 0.0,
                "seen_frames": tmeta.get("seen_frames", 0), "absent": absent, "absent_limit": self.ABSENT_N,
                "n_crops": len(crops_store), "n_best": self.N_BEST,
                "best_score": float(best_item["det_score"]) if best_item else 0.0,
                "best_res": int(best_item["resolution"]) if best_item else 0,
                "sharpness": float(best_item["sharpness"]) if best_item else 0.0,
                "resolution": int(best_item["resolution"]) if best_item else 0,
                "aspect": float(best_item["aspect_ratio"]) if best_item else 0.0,
                "quality_ok": True,
                "anchor": (tstate.position_history[-1] if tstate and len(tstate.position_history) else None),
                "trail": list(tstate.position_history) if tstate else [],
                "side": tstate.side_of_line if tstate else None,
                "inside_frames": tstate.inside_roi_frames if tstate else 0,
                "inside_roi": bool(tstate.roi_confirmed) if tstate else False,
                "velocity": tstate.recent_velocity() if tstate else None,
                "stopped": bool(tstate.stop_reported) if tstate else False,
                "stop_duration": 0.0, "events": dict(tmeta.get("events", {})),
                "ocr_rows": self._ocr_rows_for_track(cam, tid), "ocr_summary": None,
                "ocr_pending": tmeta.get("ocr_pending_trigger_type") if tmeta.get("ocr_waiting_for_result") else None,
                "ocr_pending_age": time.time() - float(tmeta.get("ocr_pending_submitted_at") or time.time()),
                "last_ocr_latency_ms": tmeta.get("last_ocr_latency_ms"),
                "finalize_waiting": bool(tmeta.get("finalize_waiting_for_ocr")),
                "finalize_stage": tmeta.get("finalize_ocr_trigger_type"),
                "finalize_age": time.time() - float(tmeta.get("finalize_ocr_submitted_at") or time.time()),
            }

        triggers = cam.get("triggers") or {}
        H, W = full_frame.shape[:2]
        ctx = {
            "fid": fid, "url": cam.get("url", ""), "cam_status": "running" if cam.get("active", True) else "idle",
            "device": self.device, "imgsz": self.imgsz, "conf": self.conf, "batch_size": len(self.cameras),
            "infer_ms": self._last_infer_ms, "ocr_qin": -1, "ocr_qout": -1,
            "frame_wh": (W, H), "roi_px": roi_offset, "line_px": cam.get("line_points_px"),
            "stop_roi_px": cam.get("stop_roi_px"), "triggers": triggers,
            "trig_cross": bool(triggers.get("cross_line_trig", False)),
            "trig_stop": bool(triggers.get("stop_roi_trig", False)),
            "detections": dbg_dets, "tracks": list(dbg_tracks.values()), "n_live": n_live,
            "n_ghost": len(dbg_tracks) - n_live,
            "any_stopped": any(t.get("stopped") for t in dbg_tracks.values()),
        }
        rec.write(full_frame, ctx)

    # ---------------- engine main loop ----------------
    def run(self):
        self.logger.info("Engine loop started")
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(source=[dummy], imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        except Exception:
            pass

        while not self.stop_event.is_set():
            self._drain_control()
            if not self.cameras:
                time.sleep(0.05)
                continue

            self._drain_ocr_results()

            frames, cam_ids, roi_offsets, full_frames = [], [], [], []

            for camera_id, cam in list(self.cameras.items()):
                if not cam.get("active", True):
                    continue
                ret, frame = cam["reader"].read()
                if not (ret and frame is not None):
                    cam["dbg_read_fail"] = cam.get("dbg_read_fail", 0) + 1
                    if cam["dbg_read_fail"] in (1, 50, 500):
                        self._dbg_log(camera_id, "READ_FAIL", f"no frame from reader x{cam['dbg_read_fail']}")
                    continue

                if cam.get("dbg_read_fail", 0):
                    self._dbg_log(camera_id, "CAMERA", f"stream recovered after {cam['dbg_read_fail']} empty reads")
                    cam["dbg_read_fail"] = 0

                cam["fid"] += 1
                cam["frames_processed"] += 1
                cam["last_frame_ts"] = time.time()

                reader_stats = cam["reader"].get_stats()
                captured_now = reader_stats["frames_captured"]
                prev_captured = cam.get("_last_reader_frames_captured")
                if prev_captured is not None:
                    skipped = max(0, captured_now - prev_captured - 1)
                    cam["_last_iter_skipped"] = skipped
                    if skipped > 0:
                        cam["frames_skipped_total"] = cam.get("frames_skipped_total", 0) + skipped
                        cam["_win_skipped"] = cam.get("_win_skipped", 0) + skipped
                cam["_last_reader_frames_captured"] = captured_now
                cam["_win_processed"] = cam.get("_win_processed", 0) + 1

                _now = time.time()
                if _now - cam.get("_last_stats_log_ts", 0) >= config.PIPELINE_STATS_LOG_INTERVAL_SEC:
                    win_start = cam.get("_win_start_ts", _now)
                    win_elapsed = max(_now - win_start, 1e-6)
                    win_processed = cam.get("_win_processed", 0)
                    win_skipped = cam.get("_win_skipped", 0)
                    win_total = win_processed + win_skipped
                    fps_out = win_processed / win_elapsed
                    fps_in = win_total / win_elapsed
                    win_skip_pct = (100.0 * win_skipped / win_total) if win_total > 0 else 0.0

                    cam["_last_stats_log_ts"] = _now
                    cam["_win_start_ts"] = _now
                    cam["_win_processed"] = 0
                    cam["_win_skipped"] = 0

                    processed_total = cam["frames_processed"]
                    skipped_total = cam.get("frames_skipped_total", 0)
                    lifetime_skip_pct = (
                        100.0 * skipped_total / (processed_total + skipped_total)
                        if (processed_total + skipped_total) > 0 else 0.0
                    )
                    ib = self._last_infer_breakdown or {}
                    infer_str = (
                        f"pre={ib.get('preprocess', 0.0):.1f}ms fwd={ib.get('inference', 0.0):.1f}ms "
                        f"post={ib.get('postprocess', 0.0):.1f}ms total={self._last_infer_ms:.1f}ms"
                    ) if ib else f"total={self._last_infer_ms:.1f}ms"

                    self.logger.info(
                        f"[PIPELINE] camera={camera_id} fps_in={fps_in:.1f} fps_out={fps_out:.1f} "
                        f"skip_rate={win_skip_pct:.1f}% (lifetime={lifetime_skip_pct:.1f}%) infer[{infer_str}] "
                        f"tracks={len(cam.get('track_meta', {}))} reconnects={reader_stats['reconnects']}"
                    )

                # NOTE: line_points / stop_roi are treated as NORMALIZED
                # [0..1] fractions of the frame here, matching the
                # reference video_processor.py exactly. If the values
                # your `cameras:config` sends are raw pixel coordinates
                # (as e.g. redis_tools.py's docstring describes), this
                # ported-as-is math will scale them a second time. This
                # mismatch already existed in the reference pipeline —
                # it is preserved here rather than silently "fixed", so
                # behavior does not change under this rewrite. Worth a
                # deliberate look before going live if line-cross/stop-
                # ROI triggers seem to never fire.
                if cam.get("line_points_px") is None or cam.get("stop_roi_px") is None:
                    H, W = frame.shape[:2]
                    if cam.get("line_points_px") is None:
                        lp = cam.get("line_points", ((0, 0), (0, 0)))
                        try:
                            cam["line_points_px"] = (
                                (float(lp[0][0]) * W, float(lp[0][1]) * H),
                                (float(lp[1][0]) * W, float(lp[1][1]) * H),
                            )
                        except Exception:
                            cam["line_points_px"] = ((0, 0), (0, 0))
                        self.logger.info(f"Camera {camera_id}: line_points normalized={lp} -> pixel={cam['line_points_px']} (frame {W}x{H})")
                    if cam.get("stop_roi_px") is None:
                        sr = cam.get("stop_roi", ((0, 0), (0, 0), (0, 0), (0, 0)))
                        try:
                            cam["stop_roi_px"] = tuple((float(p[0]) * W, float(p[1]) * H) for p in sr)
                        except Exception:
                            cam["stop_roi_px"] = ((0, 0), (0, 0), (0, 0), (0, 0))
                        self.logger.info(f"Camera {camera_id}: stop_roi normalized={sr} -> pixel={cam['stop_roi_px']} (frame {W}x{H})")

                roi_frame, (rx1, ry1, rx2, ry2) = _apply_roi(frame, cam["roi"])
                frames.append(roi_frame)
                cam_ids.append(camera_id)
                roi_offsets.append((rx1, ry1, rx2, ry2))
                full_frames.append(frame)

            if not frames:
                time.sleep(0.02)
                continue

            _infer_t0 = time.time()
            try:
                results = self.model.predict(
                    source=frames, imgsz=self.imgsz, conf=self.conf, device=self.device,
                    verbose=False, half=False,
                )
            except Exception as e:
                for cid in cam_ids:
                    self._send_msg(cid, {"status": "error", "error": str(e)})
                    self._dbg_log(cid, "ERROR", f"batch inference failed: {e}")
                time.sleep(0.1)
                continue

            self._last_infer_ms = (time.time() - _infer_t0) * 1000.0
            self._batch_count += 1
            if results:
                self._last_infer_breakdown = dict(results[0].speed)

            ib = self._last_infer_breakdown or {}
            batch_size = len(cam_ids)
            avg_per_cam_ms = self._last_infer_ms / batch_size if batch_size else 0.0
            frame_skip_snapshot = {
                cid: {
                    "skipped_total": self.cameras[cid].get("frames_skipped_total", 0),
                    "skipped_this_iter": self.cameras[cid].pop("_last_iter_skipped", 0),
                } for cid in cam_ids if cid in self.cameras
            }
            self.logger.info(
                f"[BATCH-INFER] engine={self.engine_id} batch_no={self._batch_count} batch_size={batch_size} "
                f"cameras={cam_ids} infer_total_ms={self._last_infer_ms:.2f} pre={ib.get('preprocess', 0.0):.2f}ms "
                f"fwd={ib.get('inference', 0.0):.2f}ms post={ib.get('postprocess', 0.0):.2f}ms "
                f"avg_per_camera_ms={avg_per_cam_ms:.2f} frame_skip={frame_skip_snapshot}"
            )

            if self._last_infer_ms > config.SLOW_BATCH_WARN_MS:
                self.logger.warning(
                    f"[INFER-SLOW] engine={self.engine_id} cameras_in_batch={batch_size} "
                    f"infer_ms={self._last_infer_ms:.1f} (> {config.SLOW_BATCH_WARN_MS:g}ms threshold)"
                )

            for idx, res in enumerate(results):
                camera_id = cam_ids[idx]
                cam = self.cameras.get(camera_id)
                if cam is None:
                    continue
                try:
                    self._emit_heartbeat(camera_id)

                    detections = np.empty((0, 6), dtype=np.float64)
                    if res.boxes is not None and len(res.boxes) > 0:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy()
                        detections = np.column_stack([boxes, confs, clss]).astype(np.float64)

                    _rx1, _ry1, _rx2, _ry2 = roi_offsets[idx]
                    dbg_dets = []
                    dbg_tracks: Dict[int, Dict[str, Any]] = {}
                    if detections.shape[0] > 0:
                        for _d in detections:
                            dbg_dets.append((
                                float(_d[0]) + _rx1, float(_d[1]) + _ry1,
                                float(_d[2]) + _rx1, float(_d[3]) + _ry1,
                                float(_d[4]), float(_d[5]),
                            ))

                    roi_frame = frames[idx]
                    online_targets = cam["tracker"].update(detections, roi_frame.shape[:2], roi_frame.shape[:2])

                    fid = cam["fid"]
                    rx1, ry1, rx2, ry2 = roi_offsets[idx]

                    for track in online_targets:
                        track_id = int(track.track_id)
                        meta = cam["track_meta"].get(track_id)
                        is_new_track = meta is None
                        if meta is None:
                            meta = {
                                "first_seen_fid": fid, "last_seen_fid": fid,
                                "first_seen_wall_ts": time.time(), "seen_frames": 1, "events": {},
                                "ocr_waiting_for_result": False, "ocr_pending_trigger_type": None,
                                "ocr_pending_submitted_at": None,
                                "finalize_waiting_for_ocr": False, "finalize_ocr_trigger_type": None,
                                "finalize_ocr_submitted_at": None,
                            }
                            cam["track_meta"][track_id] = meta
                        else:
                            meta["last_seen_fid"] = fid
                            meta["seen_frames"] += 1
                            meta.setdefault("ocr_waiting_for_result", False)
                            meta.setdefault("ocr_pending_trigger_type", None)
                            meta.setdefault("ocr_pending_submitted_at", None)
                            meta.setdefault("finalize_waiting_for_ocr", False)
                            meta.setdefault("finalize_ocr_trigger_type", None)
                            meta.setdefault("finalize_ocr_submitted_at", None)

                        cam.setdefault("track_ocr", {}).setdefault(
                            track_id, {"cross_line": None, "stop_roi": None, "leave_scene": None, "periodic": None}
                        )

                        track_class = int(track.flag_fdf)
                        if track.detbb is None:
                            continue
                        x1, y1, x2, y2 = map(int, track.detbb)

                        H, W = roi_frame.shape[:2]
                        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W))
                        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H))
                        crop = roi_frame[y1:y2, x1:x2]
                        if crop is None or crop.size == 0:
                            continue

                        sharp = measure_sharpness(crop)
                        score = float(getattr(track, "score", 0.0))

                        if is_new_track:
                            self.logger.info(
                                f"[TRACK-NEW] camera={camera_id} track={track_id} fid={fid} "
                                f"cls={track_class} det_conf={score:.3f} bbox_roi=({x1},{y1},{x2},{y2})"
                            )

                        triggers = cam.get("triggers") or {}
                        cross_line_trig = bool(triggers.get("cross_line_trig", False))
                        stop_roi_trig = bool(triggers.get("stop_roi_trig", False))

                        trigger_events = process_track_triggers(
                            trigger_states=cam.setdefault("trigger_states", {}),
                            line_points=cam.get("line_points_px") or cam.get("line_points", ((0, 0), (0, 0))),
                            stop_roi=cam.get("stop_roi_px") or cam.get("stop_roi", ((0, 0), (0, 0), (0, 0), (0, 0))),
                            camera_id=camera_id, track_id=track_id, track_class=track_class, score=score,
                            bbox_roi=(x1, y1, x2, y2), roi_offset=(rx1, ry1, rx2, ry2),
                        )

                        if cross_line_trig and trigger_events.get("line_cross"):
                            ev = trigger_events["line_cross"]
                            if "cross_line" not in meta["events"]:
                                meta["events"]["cross_line"] = datetime.datetime.now().isoformat()
                            self.logger.info(
                                f"[LINE-CROSS] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} "
                                f"direction={ev['direction']} point={ev['point']} confidence={ev['confidence']:.2f}"
                            )
                            if not self._is_track_waiting_for_ocr(meta):
                                if self._should_submit_ocr_for_stage(cam, camera_id, track_id, "cross_line"):
                                    submitted = self._submit_track_to_ocr(camera_id, track_id, meta, "cross_line")
                                    if submitted:
                                        self._mark_track_ocr_pending(meta, "cross_line")
                                    else:
                                        self._publish_vehicle_update(camera_id, track_id, meta, "cross_line")

                        if stop_roi_trig and trigger_events.get("roi_entry"):
                            ev = trigger_events["roi_entry"]
                            self.logger.info(
                                f"[ROI-ENTRY] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} "
                                f"point={ev['point']} confidence={ev['confidence']:.2f}"
                            )

                        if stop_roi_trig and trigger_events.get("stopped_roi"):
                            ev = trigger_events["stopped_roi"]
                            if "stop_roi" not in meta["events"]:
                                meta["events"]["stop_roi"] = datetime.datetime.now().isoformat()
                            self.logger.info(
                                f"[STOPPED-ROI] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} "
                                f"duration={ev['duration']:.1f}s velocity={ev['velocity']:.1f}px/s "
                                f"point={ev['point']} confidence={ev['confidence']:.2f}"
                            )
                            if not self._is_track_waiting_for_ocr(meta):
                                if self._should_submit_ocr_for_stage(cam, camera_id, track_id, "stop_roi"):
                                    submitted = self._submit_track_to_ocr(camera_id, track_id, meta, "stop_roi")
                                    if submitted:
                                        self._mark_track_ocr_pending(meta, "stop_roi")
                                else:
                                    self._publish_vehicle_update(camera_id, track_id, meta, "stop_roi")

                        if stop_roi_trig and trigger_events.get("roi_exit"):
                            ev = trigger_events["roi_exit"]
                            self.logger.info(f"[ROI-EXIT] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} point={ev['point']}")

                        quality_ok = self._quality_check_crop(track_class, crop, sharp)
                        hh, ww = crop.shape[:2]
                        reso = int(ww * hh)
                        ar = float(ww / hh) if hh > 0 else 0.0

                        updated_topN, updated_best = self._update_best_crops(
                            cam=cam, track_id=track_id, track_class=track_class, crop_img=crop,
                            det_score=score, resolution=reso, aspect_ratio=ar, sharpness=sharp,
                        )

                        if updated_best:
                            full_bbox = (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
                            self._set_best_frame(cam=cam, track_id=track_id, full_frame=roi_frame,
                                                 bbox_xyxy=full_bbox, det_score=score, resolution=reso)

                        meta["last_bbox_full"] = (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
                        if cam.get("debug") is not None and cam["debug"].enabled:
                            tstate = cam.get("trigger_states", {}).get(track_id)
                            crops_store = cam["best_crops"].get(track_id, [])
                            best_item = crops_store[0] if crops_store else None
                            dbg_tracks[track_id] = {
                                "track_id": track_id, "ghost": False, "bbox_full": meta["last_bbox_full"],
                                "cls": track_class, "score": score, "seen_frames": meta.get("seen_frames", 0),
                                "absent": 0, "absent_limit": self.ABSENT_N, "n_crops": len(crops_store),
                                "n_best": self.N_BEST, "best_score": float(best_item["det_score"]) if best_item else 0.0,
                                "best_res": int(best_item["resolution"]) if best_item else 0, "sharpness": sharp,
                                "resolution": reso, "aspect": ar, "quality_ok": bool(quality_ok),
                                "topn_updated": bool(updated_topN), "best_updated": bool(updated_best),
                                "anchor": tstate.position_history[-1] if tstate and len(tstate.position_history) else None,
                                "trail": list(tstate.position_history) if tstate else [],
                                "side": tstate.side_of_line if tstate else None,
                                "inside_frames": tstate.inside_roi_frames if tstate else 0,
                                "inside_roi": bool(tstate.roi_confirmed) if tstate else False,
                                "velocity": tstate.recent_velocity() if tstate else None,
                                "stopped": bool(tstate.stop_reported) if tstate else False,
                                "stop_duration": (time.time() - tstate.roi_entry_time) if (tstate and tstate.roi_entry_time) else 0.0,
                                "events": dict(meta.get("events", {})),
                                "ocr_rows": self._ocr_rows_for_track(cam, track_id),
                                "ocr_summary": " | ".join(f"{s}:{t.split()[0]}" for s, t in self._ocr_rows_for_track(cam, track_id)) or None,
                                "ocr_pending": meta.get("ocr_pending_trigger_type") if meta.get("ocr_waiting_for_result") else None,
                                "ocr_pending_age": time.time() - float(meta.get("ocr_pending_submitted_at") or time.time()),
                                "last_ocr_latency_ms": meta.get("last_ocr_latency_ms"),
                                "finalize_waiting": bool(meta.get("finalize_waiting_for_ocr")),
                                "finalize_stage": meta.get("finalize_ocr_trigger_type"),
                                "finalize_age": time.time() - float(meta.get("finalize_ocr_submitted_at") or time.time()),
                            }

                    # finalize inactive tracks
                    to_finalize = []
                    for tid, tmeta in list(cam["track_meta"].items()):
                        if (fid - int(tmeta["last_seen_fid"])) > self.ABSENT_N:
                            to_finalize.append((tid, tmeta))

                    for tid, tmeta in to_finalize:
                        if tmeta.get("finalize_waiting_for_ocr", False):
                            continue
                        triggers = cam.get("triggers") or {}
                        leave_scene_trig = bool(triggers.get("leave_scene_trig", False))
                        self._dbg_log(camera_id, "FINALIZE",
                                      f"#{tid} absent > {self.ABSENT_N} frames -> leaving scene (leave_scene_trig={leave_scene_trig})",
                                      track_id=tid)
                        finalize_now = True

                        if leave_scene_trig:
                            submit_stage = "leave_scene"
                            if "leave_scene" not in tmeta["events"]:
                                tmeta["events"]["leave_scene"] = datetime.datetime.now().isoformat()

                            if self._is_track_waiting_for_ocr(tmeta):
                                pending_trigger = tmeta.get("ocr_pending_trigger_type")
                                pending_submitted_at = tmeta.get("ocr_pending_submitted_at")
                                self.logger.info(
                                    f"[LEAVE-SCENE-BLOCKED] camera={camera_id} track={tid} "
                                    f"pending={pending_trigger} -> delaying finalize until it completes"
                                )
                                tmeta["finalize_waiting_for_ocr"] = True
                                tmeta["finalize_ocr_trigger_type"] = pending_trigger or "leave_scene"
                                tmeta["finalize_ocr_submitted_at"] = (
                                    float(pending_submitted_at) if pending_submitted_at is not None else time.time()
                                )
                                finalize_now = False
                            else:
                                if self._should_submit_ocr_for_stage(cam, camera_id, tid, submit_stage):
                                    submitted = self._submit_track_to_ocr(camera_id, tid, tmeta, "leave_scene")
                                    if submitted:
                                        self._mark_track_ocr_pending(tmeta, "leave_scene")
                                        tmeta["finalize_waiting_for_ocr"] = True
                                        tmeta["finalize_ocr_trigger_type"] = "leave_scene"
                                        tmeta["finalize_ocr_submitted_at"] = time.time()
                                        finalize_now = False
                                        self.logger.info(f"[FINALIZE-DELAYED] camera={camera_id} track={tid} waiting for leave_scene OCR result")
                                    else:
                                        self.logger.warning(f"[FINALIZE-FALLBACK] camera={camera_id} track={tid} OCR submit failed; finalizing without it")
                                else:
                                    finalize_now = True

                        if finalize_now:
                            self._finalize_or_drop_track(camera_id, tid, tmeta)

                    waiting_to_finalize = [
                        (tid, tmeta) for tid, tmeta in list(cam["track_meta"].items())
                        if tmeta.get("finalize_waiting_for_ocr", False)
                    ]
                    for tid, tmeta in waiting_to_finalize:
                        expected_stage = tmeta.get("finalize_ocr_trigger_type", "leave_scene")
                        submitted_at = float(tmeta.get("finalize_ocr_submitted_at", 0.0))
                        waited = time.time() - submitted_at
                        existing_ocr = cam.get("track_ocr", {}).get(tid, {})
                        stage_result = existing_ocr.get(expected_stage)

                        if stage_result is not None:
                            self.logger.info(f"[FINALIZE-RESUME] camera={camera_id} track={tid} stage={expected_stage} result arrived; finalizing")
                            tmeta.pop("finalize_waiting_for_ocr", None)
                            tmeta.pop("finalize_ocr_trigger_type", None)
                            tmeta.pop("finalize_ocr_submitted_at", None)
                            self._finalize_or_drop_track(camera_id, tid, tmeta)
                            continue

                        if waited >= config.OCR_FINALIZE_TIMEOUT_SEC:
                            self.logger.warning(
                                f"[FINALIZE-TIMEOUT] camera={camera_id} track={tid} waited={waited:.2f}s "
                                f"stage={expected_stage} (limit {config.OCR_FINALIZE_TIMEOUT_SEC:g}s); finalizing without OCR"
                            )
                            self._dbg_log(camera_id, "TIMEOUT",
                                          f"#{tid} waited {waited:.1f}s for {expected_stage} OCR "
                                          f"(limit {config.OCR_FINALIZE_TIMEOUT_SEC:g}s) - finalizing without it",
                                          track_id=tid)
                            tmeta.pop("finalize_waiting_for_ocr", None)
                            tmeta.pop("finalize_ocr_trigger_type", None)
                            tmeta.pop("finalize_ocr_submitted_at", None)
                            self._finalize_or_drop_track(camera_id, tid, tmeta)

                    rec = cam.get("debug")
                    if rec is not None and rec.enabled:
                        try:
                            _dbgw_t0 = time.time()
                            self._write_debug_frame(cam=cam, camera_id=camera_id, full_frame=full_frames[idx],
                                                    roi_offset=roi_offsets[idx], dbg_dets=dbg_dets,
                                                    dbg_tracks=dbg_tracks, fid=fid)
                            cam["_last_debug_write_ms"] = (time.time() - _dbgw_t0) * 1000.0
                        except Exception as e:
                            self.logger.warning(f"[DEBUG-REC] frame write failed: {e}")
                    else:
                        cam["_last_debug_write_ms"] = 0.0

                except Exception as e:
                    self.logger.exception(f"[CAMERA-PROCESSING-ERROR] camera_id={camera_id}: {e}")
                    self._send_msg(camera_id, {"status": "error", "error": str(e)})
                    self._dbg_log(camera_id, "ERROR", f"camera processing error: {e}")
                    continue

        self.logger.info("Engine stopping...")
        self.cleanup()

    def cleanup(self):
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)
        self.ocr_client.stop()


def _engine_process_main(
        engine_id: int, model_path: str, imgsz: int, conf: float, save_output: bool,
        status_queue: mp.Queue, control_queue: mp.Queue, stop_event: mp.Event,
        output_dir: str, class_labels: Dict[int, str],
):
    setup_logger(f"Engine{engine_id}")
    eng = Engine(
        engine_id=engine_id, model_path=model_path, imgsz=imgsz, conf=conf, save_output=save_output,
        status_queue=status_queue, control_queue=control_queue, stop_event=stop_event,
        output_dir=output_dir, class_labels=class_labels,
        absent_n=config.DEFAULT_ABSENT_FRAMES, min_seen_frames=config.DEFAULT_MIN_SEEN_FRAMES,
        min_crops_to_finalize=config.DEFAULT_MIN_CROPS_TO_FINALIZE, n_best=config.DEFAULT_N_BEST_CROPS,
        conf_digits=config.DEFAULT_CONF_DIGITS,
    )
    eng.run()
