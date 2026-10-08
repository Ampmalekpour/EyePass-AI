"""
engine.py
--------------------------------------------------------------------
The plate detection/tracking/trigger engine — one YOLO model per
process, handling however many cameras EngineManager assigns it. Batch
inference, BYTETrack update, spatial-trigger geometry, best-crop
ranking and the debug recorder run here, per frame.

Inference goes through inference_backends.py:
  GPU  -> the engine's cameras are batched into ONE Ultralytics .pt
          predict() per loop (the original GPU pipeline)
  CPU  -> ONE model instance per camera of the engine (OpenVINO FP32 /
          INT8 through Ultralytics on intel:cpu, or an own ONNX Runtime
          session), all cameras' frames run in parallel
Every backend returns the same [x1, y1, x2, y2, conf, cls] float64
array per frame, so nothing below the inference call knows which one
ran. A camera is only processed when its reader has a NEW frame;
frames replaced before the engine got to them are counted as missed
(perf_stats.py -> ⏱️ [PERF] / 📊 [STATS] log lines).

A track's OCR STATE lives in the control hub (control-hub/), not here:

    detector (this file)                 control hub
    --------------------                 -----------
    track born     -> track_started ---> holds the track's state
    trigger fires  -> trigger ---------> publishes cross_line / stop_roi
                     (+ submits crops)    now (plate known) or when its
                                          OCR result lands
    crops sent     -> submitted -------> tracks in-flight tasks
    OCR result ------------------------> votes across stages, decides
                   <- ctl: result/state   "satisfied" (detector stops
                                          sending)
                   <- ctl: request        periodic re-query (cond_per_trig,
                                          which used to be ignored)
    track gone     -> track_ended ------> waits for the leave_scene OCR,
                     (+ leave_scene crops) publishes the final record

The engine never pushes to plate:vehicle:results itself any more and
forgets a track the moment it emits track_ended.
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

import config
import capacity
import sysinfo
import inference_backends
from perf_stats import EnginePerf
from debug_recorder import DebugConfig, DebugRecorder
from platecore.bus import RedisBus
from platecore.hub import (
    K_SUBMITTED, K_TRACK_ENDED, K_TRACK_STARTED, K_TRACK_UPDATE, K_TRIGGER,
    HubClient, TrackRecState, new_task_id, new_track_uid,
)
from platecore.logging_setup import setup_logger
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


def resolve_runtime(logger: logging.Logger) -> "inference_backends.RuntimePlan":
    """.env DETECTION_DEVICE / DETECTION_GPU_MODEL / DETECTION_CPU_MODEL
    -> gpu + pt (the .pt model), or cpu + openvino_fp32/openvino_int8/onnx."""
    return inference_backends.resolve_runtime(
        config.DETECTION_DEVICE, config.DETECTION_GPU_MODEL, config.DETECTION_CPU_MODEL,
        config.CPU_MODEL_NAME, config.GPU_DEVICE, config.STRICT_DEVICE, config.MODEL_ALIASES, logger,
    )


def backend_settings() -> "inference_backends.BackendSettings":
    return inference_backends.BackendSettings(
        model_root=config.MODEL_ROOT,
        conf=config.CONF_THRESHOLD,
        nms_iou=config.NMS_IOU,
        max_det=config.MAX_DET,
        warmup_runs=config.WARMUP_RUNS,
        gpu_half=config.GPU_HALF,
        ov_device=config.OV_DEVICE,
        onnx_cpu_threads=config.ONNX_CPU_THREADS,
        int8_fix=dict(config.INT8_FIX),
        manifest_keys=config.MODEL_MANIFEST_KEYS,
        file_patterns=config.MODEL_FILE_PATTERNS,
        class_labels=config.CLASS_LABELS,
        fallback_to_pt=config.BACKEND_FALLBACK_TO_PT,
    )


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


def to_frame_pixels(points, W: int, H: int):
    """Line / polygon points -> full-frame pixel coordinates.

    Accepts fractions of the frame (every coordinate within 0..1) or
    pixel coordinates (anything larger). Returns (points_px, mode) with
    mode "fractions" | "pixels" | "invalid". An all-zero shape (trigger
    not configured) stays all zeros."""
    try:
        pts = [(float(p[0]), float(p[1])) for p in points]
    except Exception:
        return tuple((0.0, 0.0) for _ in range(len(points) if hasattr(points, "__len__") else 2)), "invalid"
    if all(0.0 <= v <= 1.0 for pt in pts for v in pt):
        return tuple((x * W, y * H) for x, y in pts), "fractions"
    return tuple(pts), "pixels"


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
        self.model_path = model_path or ""
        self.conf = float(conf)
        self.save_output = bool(save_output)

        self.status_queue = status_queue
        self.control_queue = control_queue
        self.stop_event = stop_event

        self.output_dir = output_dir
        self.class_labels = class_labels

        self.logger = setup_logger(f"Engine{self.engine_id}")

        # Detection interval in use (camera frames between two detections of a
        # camera; may be fractional): config.DETECT_EVERY_N_FRAMES, raised
        # automatically on CPU when the cameras exceed the measured real-time
        # capacity (config.CAPACITY_AUTO_DEGRADE, see _apply_cadence()).
        self.detect_interval = float(max(1, int(config.DETECT_EVERY_N_FRAMES)))
        self.capacity_profile = None

        # ---- detection backend (GPU: .pt/Ultralytics; CPU: OpenVINO/ONNX)
        # Built HERE, in the engine child process — openvino/onnxruntime
        # are imported and the model compiled inside the child, never in
        # the parent (spawn-safe; required on Windows).
        self.runtime = resolve_runtime(self.logger)
        self.logger.info(f"[INIT] detection runtime: {self.runtime.describe()} root={config.MODEL_ROOT}")
        self.backend = inference_backends.build_backend(self.runtime, backend_settings(), self.logger)
        self.device = self.backend.device
        self.imgsz = self.backend.imgsz_label

        if self.backend.label.startswith("cpu"):
            # cv2's own thread pool (ROI crop / resize) is separate from the
            # inference runtime's — kept small so it doesn't compete with it.
            cv2.setNumThreads(max(1, config.CV2_NUM_THREADS))
            if self.backend.kind == "pt":
                # PyTorch only computes on the CPU in the .pt fallback.
                import torch
                n = config.TORCH_NUM_THREADS or max(1, config.usable_cpu_count() - 1)
                torch.set_num_threads(n)
                self.logger.info(f"[INIT] CPU/pt fallback: torch threads={n} cv2 threads={config.CV2_NUM_THREADS}")
            else:
                tn = int(config.CPU_TORCH_NUM_THREADS)
                if tn > 0:
                    try:
                        import torch
                        torch.set_num_threads(tn)
                    except ImportError:
                        tn = 0
                self.logger.info(f"[INIT] CPU/{self.backend.kind}: one model instance per camera, "
                                 f"cv2 threads={config.CV2_NUM_THREADS} "
                                 f"torch threads={tn or 'default'} (pre/post-processing only)")

        # track lifecycle config
        self.ABSENT_N = int(absent_n)
        self.MIN_SEEN_FRAMES = int(min_seen_frames)
        self.MIN_CROPS_TO_FINALIZE = int(min_crops_to_finalize)

        # best crops config
        self.N_BEST = int(n_best)
        self.conf_digits = int(conf_digits)

        # cameras: camera_id -> camera state
        self.cameras: Dict[str, Dict[str, Any]] = {}

        # OCR tasks out to the ocr_service, track events/ctl with the
        # control hub (see platecore/hub.py).
        self.bus = RedisBus(module=config.REDIS_MODULE)
        self.hub = HubClient(self.bus, self.engine_id)
        self._uid_index: Dict[str, Tuple[str, int]] = {}
        # when several requests are pending, the task carries the most
        # important label (one set of crops answers all of them)
        self.STAGE_PRIORITY = {"leave_scene": 3, "cross_line": 2, "stop_roi": 2, "periodic": 1}
        # one emoji per OCR-triggering stage so it's spottable at a glance in
        # a scrolling log — periodic is the routine one, the other three are
        # spatial/lifecycle events worth an eye catching more than plain text.
        self.STAGE_EMOJI = {"periodic": "🔁", "cross_line": "🚧", "stop_roi": "🛑", "leave_scene": "🏁"}
        self.logger.info("control-hub client ready (engine_id=%s boot=%s)", self.engine_id, self.hub.boot_id)

        # ---- performance logs: ⏱️ [PERF] per camera, 📊 [STATS] per engine
        self.perf = EnginePerf(self.engine_id, self.backend.label,
                               config.PERF_LOG_INTERVAL_SEC, config.STATS_EVERY_N_BATCHES)

        # writers optional
        self.writers: Dict[str, cv2.VideoWriter] = {}
        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)

        self._last_status_emit = 0.0

        # DEBUG VIDEO: one annotated recorder per camera.
        self.debug_cfg = DebugConfig()
        self._last_infer_ms = 0.0
        self._batch_count = 0
        if self.debug_cfg.enabled:
            self.logger.info(f"[DEBUG-REC] enabled -> {self.debug_cfg.as_dict()}")
        else:
            self.logger.info("[DEBUG-REC] disabled (config.DEBUG_VIDEO_ENABLED)")

    def _tlog(self, msg: str, *args, **kwargs):
        """Per-track event lines — config.LOG_TRACK_EVENTS turns them off."""
        if config.LOG_TRACK_EVENTS:
            self.logger.info(msg, *args, **kwargs)

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

    @staticmethod
    def _ocr_rows_for_track(cam: Dict[str, Any], track_id: int):
        """Per-stage OCR rows for the debug video, as last reported by
        the control hub (ctl `display.rows`)."""
        rs = cam.get("rec_state", {}).get(track_id)
        rows = (rs.display.get("rows") if rs is not None else None) or []
        return [tuple(r) for r in rows if isinstance(r, (list, tuple)) and len(r) == 2]

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
        # frame_rate scales BYTETracker's own step-counted lost-track
        # buffer (track_buffer -> max_time_lost, see tracker.py) into
        # real time. A tracker "step" only happens on a detect frame
        # (tracker.update() is skipped entirely on coast frames — see
        # the run loop), so with DETECT_EVERY_N_FRAMES=N each step now
        # spans N camera frames of wall-clock time instead of 1; without
        # this, a lost track would be kept around N times longer in real
        # seconds than TRACKER_TRACK_BUFFER was tuned for. Passing the
        # true step rate (camera fps / N) keeps that wall-clock window
        # the same regardless of N.
        step_rate = config.CAMERA_ASSUMED_FPS / self.detect_interval
        tracker = BYTETracker(build_tracker_config(), frame_rate=step_rate, name=camera_id)

        self.cameras[camera_id] = {
            "camera_id": camera_id, "url": url, "roi": roi,
            "line_points": line_points, "stop_roi": stop_roi,
            "line_points_px": None, "stop_roi_px": None, "triggers": triggers,
            "trigger_states": {}, "rec_state": {},
            "reader": reader, "tracker": tracker, "fid": 0, "frames_processed": 0,
            "last_frame_ts": None, "track_meta": {}, "best_crops": {}, "best_frame": {},
            "active": True,
            "debug": DebugRecorder(camera_id, self.engine_id, self.debug_cfg, self.logger),
            "dbg_read_fail": 0,
            # detect-cadence bookkeeping (see config.DETECT_EVERY_N_FRAMES):
            # _next_due paces and staggers the cameras' detect frames
            # (see _apply_cadence); _last_tracker_fid lets us
            # compute the real elapsed-frame dt for the next tracker.update().
            "_next_due": 0.0,           # fid (float) from which this camera is detected again
            "_last_tracker_fid": -1,
            "_last_captured": None,
        }
        # CPU backends: one model instance per camera (no-op on the GPU)
        try:
            self.backend.ensure_streams(len(self.cameras))
        except Exception as e:
            self.logger.exception(f"[CAMERA-ADD] camera={camera_id}: could not create its model instance: {e}")
        self._apply_cadence(f"camera {camera_id} added")

        self._dbg_log(camera_id, "CAMERA", f"camera added -> {url}",
                      data={"roi": roi, "line_points": line_points, "stop_roi": stop_roi, "triggers": triggers})
        self._send_msg(camera_id, {"status": "running"})
        self.logger.info(
            f"[CAMERA-ADD] camera={camera_id} url={url} roi={roi} "
            f"line_points={line_points} stop_roi={stop_roi} triggers={triggers}"
        )

    def remove_camera(self, camera_id: str, reason: str = "camera_removed"):
        camera_id = str(camera_id)
        cam = self.cameras.get(camera_id)
        if not cam:
            return
        # hand every live track to the hub first — deactivation, rebalance
        # migration and engine shutdown still produce a final record
        for tid, tmeta in list(cam.get("track_meta", {}).items()):
            try:
                self._end_track(camera_id, tid, tmeta, reason=reason)
            except Exception:
                self.logger.exception(f"camera={camera_id} track={tid}: ending on removal failed")
        self.cameras.pop(camera_id, None)
        self.perf.forget(camera_id)
        self._apply_cadence(f"camera {camera_id} removed")
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

    # ---------------- control hub: events, crops, track end ----------------
    @staticmethod
    def _hub_triggers(cam: Dict[str, Any]) -> Dict[str, bool]:
        """Camera trigger flags in the hub's (= this module's event) names."""
        t = cam.get("triggers") or {}
        return {
            "periodic": bool(t.get("cond_per_trig", False)),
            "cross_line": bool(t.get("cross_line_trig", False)),
            "stop_roi": bool(t.get("stop_roi_trig", False)),
            "leave_scene": bool(t.get("leave_scene_trig", False)),
        }

    @staticmethod
    def _trigger_detail(ev: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("direction", "confidence", "duration", "velocity", "class"):
            v = ev.get(k)
            if v is None:
                continue
            out[k] = v if isinstance(v, str) else float(v)
        pt = ev.get("point")
        if pt is not None:
            out["point"] = [int(pt[0]), int(pt[1])]
        return out

    def _track_stats(self, cam, track_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "uid": meta["uid"], "camera_id": cam["camera_id"], "track_id": track_id,
            "seen_frames": int(meta.get("seen_frames", 0)),
            "n_crops": len(cam["best_crops"].get(track_id, [])),
            "duration_frames": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
        }

    def _build_task(self, cam, camera_id: str, track_id: int, meta: dict, stage: str) -> Optional[dict]:
        crops = cam["best_crops"].get(track_id, [])
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
            return None
        best_frame_data = None
        bf = cam["best_frame"].get(track_id)
        if bf:
            encoded_frame = encode_image(bf["frame"])
            if encoded_frame:
                best_frame_data = {"image_bytes": encoded_frame, "bbox": bf["bbox"], "resolution": bf["resolution"]}
        return {
            "task_id": new_task_id(meta["uid"], stage), "track_uid": meta["uid"], "stage": stage,
            "engine_id": self.engine_id, "process_id": self.engine_id, "stream_idx": camera_id,
            "camera_id": camera_id, "video_source": cam["url"], "track_id": track_id,
            "trigger_type": stage,
            "meta": {
                "seen_frames": int(meta.get("seen_frames", 0)),
                "duration": int(meta.get("last_seen_fid", 0) - meta.get("first_seen_fid", 0)),
                "trigger_reason": stage, "trigger_snapshot": True,
            },
            "crops": crops_data, "best_frame": best_frame_data,
        }

    def _dispatch(self, cam, camera_id: str, track_id: int, meta: dict, task: dict):
        n = len(task["crops"])
        self.hub.emit(K_SUBMITTED, {"uid": meta["uid"], "camera_id": camera_id, "track_id": track_id,
                                    "task_id": task["task_id"], "stage": task["stage"], "n_crops": n})
        self.hub.submit(task)
        if config.DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED:
            debug_extras.save_ocr_submission_montage(
                camera_id=camera_id, track_id=track_id, trigger_type=task["stage"],
                task_id=task["task_id"], crops=cam["best_crops"].get(track_id, []),
                best_frame=cam["best_frame"].get(track_id), logger=self.logger,
            )
        det_scores = [round(float(c["score"]), 3) for c in task["crops"]]
        emoji = self.STAGE_EMOJI.get(task["stage"], "🔁")
        self._tlog(
            f"{emoji} [OCR-SUBMIT] camera={camera_id} track={track_id} uid={meta['uid']} stage={task['stage']} "
            f"seen_frames={meta.get('seen_frames', 0)} crops={n} det_scores={det_scores} "
            f"best_frame={'yes' if task['best_frame'] else 'no'} task_id={task['task_id']}"
        )
        self._dbg_log(camera_id, "OCR_SUBMIT",
                      f"#{track_id} {task['stage']} crops={n} best_frame={'yes' if task['best_frame'] else 'no'}",
                      track_id=track_id, data={"task_id": task["task_id"], "n_crops": n, "det_scores": det_scores})

    def _try_submit(self, camera_id: str, track_id: int, meta: dict,
                    rec_state: TrackRecState, now: float) -> Optional[str]:
        """Send the current top-N crops if something is requested, the hub
        has not said "satisfied", nothing is in flight, and the crop set
        changed since the last send. Returns the task_id or None."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return None
        stage = rec_state.next_stage(now, config.SUBMIT_TIMEOUT_SEC, self.STAGE_PRIORITY)
        if stage is None:
            return None
        crops = cam["best_crops"].get(track_id, [])
        if not crops:
            return None
        key = f"{crops[0]['frame_number']}:{len(crops)}"
        if key == rec_state.last_sent_key:
            self._dbg_log(camera_id, "OCR_SKIP", f"#{track_id} {stage} waiting for a better crop set",
                          track_id=track_id)
            return None
        task = self._build_task(cam, camera_id, track_id, meta, stage)
        if task is None:
            return None
        rec_state.mark_submitted(task["task_id"], stage, key, now)
        self._dispatch(cam, camera_id, track_id, meta, task)
        return task["task_id"]

    def _end_track(self, camera_id: str, track_id: int, meta: dict, reason: str = "absent"):
        """The track left (or its camera/engine is going away): optionally
        send the leave_scene OCR pass, tell the hub, forget the track."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return
        rec_state: Optional[TrackRecState] = cam["rec_state"].pop(track_id, None)
        uid = meta.get("uid")
        seen_frames = int(meta.get("seen_frames", 0))
        num_crops = len(cam["best_crops"].get(track_id, []))
        leave_scene_trig = bool((cam.get("triggers") or {}).get("leave_scene_trig", False))

        finalize_task_id = None
        if leave_scene_trig and rec_state is not None and not rec_state.satisfied \
                and seen_frames >= self.MIN_SEEN_FRAMES and num_crops >= self.MIN_CROPS_TO_FINALIZE:
            task = self._build_task(cam, camera_id, track_id, meta, "leave_scene")
            if task is not None:
                self._dispatch(cam, camera_id, track_id, meta, task)
                finalize_task_id = task["task_id"]
        elif leave_scene_trig:
            why = ("hub satisfied" if rec_state is not None and rec_state.satisfied
                   else f"seen={seen_frames}/{self.MIN_SEEN_FRAMES} crops={num_crops}/{self.MIN_CROPS_TO_FINALIZE}")
            self._dbg_log(camera_id, "OCR_SKIP", f"#{track_id} leave_scene OCR skipped ({why})", track_id=track_id)

        if uid:
            self.hub.emit(K_TRACK_ENDED, {**self._track_stats(cam, track_id, meta), "reason": reason,
                                          "finalize_task_id": finalize_task_id})
            self._uid_index.pop(uid, None)

        lifetime_s = time.time() - float(meta.get("first_seen_wall_ts", time.time()))
        self._tlog(
            f"[TRACK-END] camera={camera_id} track={track_id} uid={uid} reason={reason} "
            f"seen_frames={seen_frames} crops={num_crops} lifetime={lifetime_s:.2f}s "
            f"leave_scene_task={finalize_task_id}"
        )
        self._dbg_log(camera_id, "FINALIZE",
                      f"#{track_id} {reason} -> hub seen={seen_frames}f crops={num_crops} "
                      f"leave_scene_ocr={'yes' if finalize_task_id else 'no'}", track_id=track_id)

        cam["track_meta"].pop(track_id, None)
        cam["best_crops"].pop(track_id, None)
        cam["best_frame"].pop(track_id, None)
        cam.get("trigger_states", {}).pop(track_id, None)

    def _drain_hub_ctl(self):
        now = time.time()
        for msg in self.hub.drain():
            loc = self._uid_index.get(msg.get("uid"))
            if not loc:
                continue  # track already ended here — the hub owns it now
            camera_id, track_id = loc
            cam = self.cameras.get(camera_id)
            rec_state = cam["rec_state"].get(track_id) if cam else None
            if rec_state is None:
                continue
            was_satisfied = rec_state.satisfied
            rec_state.apply_ctl(msg, now)
            disp = rec_state.display or {}
            action = msg.get("action")
            if action == "result":
                self._tlog(
                    f"🔎 [OCR-RESULT] camera={camera_id} track={track_id} task={msg.get('task_id')} "
                    f"hub_answer={disp.get('label')!r} conf={disp.get('confidence')} "
                    f"satisfied={rec_state.satisfied} rtt_ms={rec_state.last_latency_ms}"
                )
                self._dbg_log(camera_id, "OCR_RESULT",
                              f"#{track_id} -> '{disp.get('label')}' conf={disp.get('confidence')}",
                              track_id=track_id, data={"task_id": msg.get("task_id"), "display": disp})
            elif action == "request":
                self._dbg_log(camera_id, "HUB_REQUEST", f"#{track_id} {msg.get('stage')}", track_id=track_id)
            if rec_state.satisfied and not was_satisfied:
                self._tlog(f"✅ [SATISFIED] camera={camera_id} track={track_id} plate={disp.get('label')!r} "
                                 f"— no more OCR for this track")

    # ---------------- debug frame assembly ----------------
    def _debug_ocr_fields(self, cam, track_id: int, now: float) -> Dict[str, Any]:
        rs = cam.get("rec_state", {}).get(track_id)
        rows = self._ocr_rows_for_track(cam, track_id)
        if rs is None:
            return {"ocr_rows": rows, "ocr_summary": None, "ocr_pending": None, "ocr_pending_age": 0.0,
                    "last_ocr_latency_ms": None, "finalize_waiting": False, "finalize_stage": None,
                    "finalize_age": 0.0}
        label = rs.display.get("label")
        return {
            "ocr_rows": rows,
            "ocr_summary": (f"{label} {float(rs.display.get('confidence') or 0.0):.2f}"
                            f"{' SAT' if rs.satisfied else ''}") if label else None,
            "ocr_pending": rs.pending_label(now, config.SUBMIT_TIMEOUT_SEC),
            "ocr_pending_age": (now - rs.in_flight_since) if rs.in_flight_task else 0.0,
            "last_ocr_latency_ms": rs.last_latency_ms,
            # the finalize wait happens in the control hub now
            "finalize_waiting": False, "finalize_stage": None, "finalize_age": 0.0,
        }

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
                **self._debug_ocr_fields(cam, tid, time.time()),
            }

        triggers = cam.get("triggers") or {}
        H, W = full_frame.shape[:2]
        ctx = {
            "fid": fid, "url": cam.get("url", ""), "cam_status": "running" if cam.get("active", True) else "idle",
            "device": self.backend.label, "imgsz": self.imgsz, "conf": self.conf, "batch_size": len(self.cameras),
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

    # ---------------- real-time capacity (see capacity.py) ----------------
    def _startup_capacity(self):
        """Engine 0 measures how many cameras this machine serves in real
        time (before any camera is attached) and publishes the profile to
        Redis; the bridge checks every attached camera against it. Other
        engines start while engine 0's cameras are running, so measuring
        there would be wrong (and would steal CPU from live cameras)."""
        try:
            if not config.CAPACITY_CALIBRATION_ENABLED or not self.backend.label.startswith("cpu"):
                if self.engine_id == 0:
                    capacity.clear(self.bus)
                    self.logger.info("🧪 [CAPACITY] not calibrated (%s)",
                                     "disabled in config" if not config.CAPACITY_CALIBRATION_ENABLED
                                     else "GPU engine: capacity is a CPU measurement")
                return
            if self.engine_id != 0:
                prof = capacity.load(self.bus)
                self.capacity_profile = prof
                self.logger.info("🧪 [CAPACITY] engine=%s: not re-measured while other engines run; "
                                 "device real-time capacity = %s cameras (measured by engine 0)",
                                 self.engine_id, prof.get("max_cameras") if prof else "unknown")
                return
            capacity.clear(self.bus)
            profile = capacity.calibrate(self.backend, self.logger, self.engine_id)
            try:
                profile["system"] = sysinfo.collect()
                for line in sysinfo.render(profile, profile["system"]):
                    self.logger.info(line)
            except Exception:
                self.logger.exception("hardware report failed (calibration result is unaffected)")
            capacity.publish(self.bus, profile)
            self.capacity_profile = profile
            self.backend.reset_streams()   # instances are re-created as cameras are added
        except Exception as e:
            self.logger.exception(f"🧪 [CAPACITY] calibration failed ({e}) — continuing without a capacity profile")
            try:
                self.backend.reset_streams()
            except Exception:
                pass

    def _apply_cadence(self, why: str):
        """CPU: keep the cameras inside the measured real-time capacity by
        lowering every camera's detection rate to what fits (the tracker
        coasts the frames in between) instead of letting frames be missed.
        Back to the configured rate as soon as the cameras fit again. No-op
        without a profile / on GPU / when CAPACITY_AUTO_DEGRADE is off."""
        n_cams = len(self.cameras)
        base = float(max(1, int(config.DETECT_EVERY_N_FRAMES)))
        old = self.detect_interval
        if config.CAPACITY_AUTO_DEGRADE and self.capacity_profile and n_cams:
            max_iv = max(base, config.CAMERA_ASSUMED_FPS / float(config.DETECT_MIN_FPS))
            iv, loop_ms, fits = capacity.pick_detect_interval(self.capacity_profile, n_cams, base, max_iv)
        else:
            iv, loop_ms, fits = old, 0.0, True
        # spread the cameras' detect frames over one interval
        for k, cam in enumerate(self.cameras.values()):
            cam["_next_due"] = cam.get("fid", 0) + k * iv / max(1, n_cams)
        self.perf.target_fps = config.CAMERA_ASSUMED_FPS / iv
        dcfg = getattr(self, "debug_cfg", None)
        if dcfg is not None and config.DEBUG_VIDEO_FPS_FOLLOWS_DETECTION:
            dcfg.fps = config.CAMERA_ASSUMED_FPS / iv     # the next segment is written at the detection rate
        if abs(iv - old) < 1e-9:
            return
        self.detect_interval = iv
        for cam in self.cameras.values():
            # a tracker "step" now spans `iv` camera frames -> keep the lost-track
            # window the same in wall-clock time (see add_camera)
            tr = cam["tracker"]
            tr.buffer_size = int(config.CAMERA_ASSUMED_FPS / iv / 30.0 * tr.track_buffer)
            tr.max_time_lost = tr.buffer_size
        fps = config.CAMERA_ASSUMED_FPS / iv
        if iv > old:
            self.logger.warning(
                "🐢 [CAPACITY] engine=%s: %d cameras need ~%.0f ms per loop at full rate, real-time allows %.0f ms "
                "(%s) → detection lowered to %.1f fps per camera (of %.0f; every %.2f frames, tracker coasts the rest)%s",
                self.engine_id, n_cams, loop_ms, capacity.budget_for(1.0), why, fps, config.CAMERA_ASSUMED_FPS, iv,
                "" if fits else f" — STILL over capacity at the floor DETECT_MIN_FPS={config.DETECT_MIN_FPS:g}: "
                                f"expect missed frames; use fewer cameras or a lighter model")
        else:
            self.logger.info("🐇 [CAPACITY] engine=%s: %d cameras (%s) → detection %.1f fps per camera%s",
                             self.engine_id, n_cams, why, fps,
                             " (full rate)" if iv <= base else "")

    # ---------------- engine main loop ----------------
    def run(self):
        self.logger.info("Engine loop started")
        self._startup_capacity()
        try:
            self.backend.warmup()
        except Exception as e:
            self.logger.warning(f"[INIT] warm-up failed: {e}")

        while not self.stop_event.is_set():
            self._drain_control()
            if not self.cameras:
                time.sleep(0.05)
                continue

            self._drain_hub_ctl()
            self.perf.maybe_log_cameras(self.logger, len(self.cameras))
            _loop_t0 = time.time()

            frames, cam_ids, roi_offsets, full_frames, frame_ts = [], [], [], [], []
            any_frame_read = False
            _loop_missed = _loop_captured = 0

            for camera_id, cam in list(self.cameras.items()):
                if not cam.get("active", True):
                    continue
                ret, frame, f_ts, captured_now = cam["reader"].read_latest()
                if not (ret and frame is not None):
                    cam["dbg_read_fail"] = cam.get("dbg_read_fail", 0) + 1
                    if cam["dbg_read_fail"] in (1, 50, 500):
                        self._dbg_log(camera_id, "READ_FAIL", f"no frame from reader x{cam['dbg_read_fail']}")
                    continue
                any_frame_read = True

                if cam.get("dbg_read_fail", 0):
                    self._dbg_log(camera_id, "CAMERA", f"stream recovered after {cam['dbg_read_fail']} empty reads")
                    cam["dbg_read_fail"] = 0

                # Only NEW frames: the reader keeps the latest frame, so if
                # nothing arrived since the last loop there is nothing to do
                # for this camera; frames that arrived and were replaced before
                # we got here are the misses (same rule as the benchmark).
                prev = cam["_last_captured"]
                cam["_last_captured"] = captured_now
                new_frames = captured_now - prev if prev is not None else 1
                if new_frames <= 0:
                    continue
                missed = max(0, new_frames - 1)
                self.perf.frame_arrived(camera_id, new_frames, missed)
                _loop_missed += missed
                _loop_captured += new_frames

                # fid counts CAMERA frames (missed ones included), so the
                # tracker's dt and ABSENT_N stay in real camera frames.
                cam["fid"] += new_frames
                cam["frames_processed"] += 1
                cam["last_frame_ts"] = time.time()

                # line_points / stop_roi arrive either as fractions of the
                # frame (0..1, what the reference video_processor.py
                # assumed) or as pixel coordinates (what redis_tools.py
                # and the backend send). They used to be ALWAYS scaled by
                # the frame size, so pixel coordinates landed far outside
                # the frame and line-cross / stop-ROI never fired.
                # to_frame_pixels() decides per shape: all values <= 1 ->
                # fractions, otherwise already pixels.
                if cam.get("line_points_px") is None or cam.get("stop_roi_px") is None:
                    H, W = frame.shape[:2]
                    if cam.get("line_points_px") is None:
                        lp = cam.get("line_points", ((0, 0), (0, 0)))
                        cam["line_points_px"], mode = to_frame_pixels(lp, W, H)
                        self.logger.info(f"Camera {camera_id}: line_points {lp} read as {mode} -> "
                                         f"pixel={cam['line_points_px']} (frame {W}x{H})")
                    if cam.get("stop_roi_px") is None:
                        sr = cam.get("stop_roi", ((0, 0), (0, 0), (0, 0), (0, 0)))
                        cam["stop_roi_px"], mode = to_frame_pixels(sr, W, H)
                        self.logger.info(f"Camera {camera_id}: stop_roi {sr} read as {mode} -> "
                                         f"pixel={cam['stop_roi_px']} (frame {W}x{H})")

                roi_frame, (rx1, ry1, rx2, ry2) = _apply_roi(frame, cam["roi"])

                # ---- detect cadence gate (config.DETECT_EVERY_N_FRAMES) ----
                # N=1 (default) -> every frame goes to the model, unchanged.
                # N>1 -> only 1-in-N frames are sent to YOLO; the rest are
                # "coasted": no inference, no tracker.update() call at all
                # (cheaper than calling update() with empty detections every
                # frame, and mathematically equivalent since the next real
                # update() advances the Kalman filter by the true elapsed
                # dt in one step). The debug video still gets a frame every
                # iteration via the tracker's existing ghost-track fallback
                # in _write_debug_frame(), drawing each track's last known
                # box so playback doesn't drop to 1/N fps.
                # Each camera is due every `detect_interval` frames (float: 1.25
                # = 4 of 5 frames). Debt is never accumulated: a camera that
                # fell behind is simply due now (no catch-up burst).
                do_detect = cam["fid"] >= cam["_next_due"]
                if not do_detect:
                    self.perf.frame_coasted(camera_id)
                    rec = cam.get("debug")
                    if rec is not None and rec.enabled and not config.DEBUG_VIDEO_ONLY_DETECTED_FRAMES:
                        try:
                            self._write_debug_frame(
                                cam=cam, camera_id=camera_id, full_frame=frame,
                                roi_offset=(rx1, ry1, rx2, ry2), dbg_dets=[], dbg_tracks={},
                                fid=cam["fid"],
                            )
                        except Exception as e:
                            self.logger.warning(f"[DEBUG-REC] coast frame write failed: {e}")
                    continue

                frames.append(roi_frame)
                cam_ids.append(camera_id)
                roi_offsets.append((rx1, ry1, rx2, ry2))
                full_frames.append(frame)
                frame_ts.append(f_ts)

            if not frames:
                # no camera has a new frame yet: wait a little instead of spinning
                time.sleep(0.02 if not any_frame_read else 0.002)
                continue

            _infer_t0 = time.time()
            try:
                batch_dets, frame_timings = self.backend.infer(frames)
            except Exception as e:
                for cid in cam_ids:
                    self._send_msg(cid, {"status": "error", "error": str(e)})
                    self._dbg_log(cid, "ERROR", f"batch inference failed: {e}")
                time.sleep(0.1)
                continue

            self._last_infer_ms = (time.time() - _infer_t0) * 1000.0
            self._batch_count += 1
            batch_size = len(cam_ids)

            if self._last_infer_ms > config.SLOW_BATCH_WARN_MS:
                self.logger.warning(
                    f"⚠️ [INFER-SLOW] engine={self.engine_id} cameras_in_batch={batch_size} "
                    f"infer_ms={self._last_infer_ms:.1f} (> {config.SLOW_BATCH_WARN_MS:g}ms threshold)"
                )

            _batch_detections = 0
            _batch_tracks = 0

            for idx, detections in enumerate(batch_dets):
                camera_id = cam_ids[idx]
                cam = self.cameras.get(camera_id)
                if cam is None:
                    continue
                _cam_t0 = time.time()
                try:
                    self._emit_heartbeat(camera_id)

                    # [x1, y1, x2, y2, conf, cls] float64 in ROI pixels,
                    # whichever backend produced it (inference_backends.py)
                    _batch_detections += int(detections.shape[0])

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
                    # real elapsed camera frames since the tracker last saw this
                    # camera -- normally N (config.DETECT_EVERY_N_FRAMES), but can
                    # be larger if the reader also silently dropped frames in
                    # between (missed frames). BYTETracker uses
                    # this to advance its Kalman filters by the true gap in one
                    # step instead of assuming 1 frame passed.
                    _dt = cam["fid"] - cam.get("_last_tracker_fid", cam["fid"] - 1)
                    cam["_last_tracker_fid"] = cam["fid"]
                    cam["_next_due"] = max(cam["_next_due"] + self.detect_interval, float(cam["fid"]))
                    online_targets = cam["tracker"].update(
                        detections, roi_frame.shape[:2], roi_frame.shape[:2], dt=_dt
                    )
                    _batch_tracks += len(online_targets)

                    fid = cam["fid"]
                    rx1, ry1, rx2, ry2 = roi_offsets[idx]

                    for track in online_targets:
                        track_id = int(track.track_id)
                        now_ts = time.time()
                        meta = cam["track_meta"].get(track_id)
                        is_new_track = meta is None
                        if meta is None:
                            uid = new_track_uid(camera_id, self.engine_id)
                            meta = {
                                "uid": uid, "first_seen_fid": fid, "last_seen_fid": fid,
                                "first_seen_wall_ts": now_ts, "seen_frames": 1, "events": {},
                            }
                            cam["track_meta"][track_id] = meta
                            cam["rec_state"][track_id] = TrackRecState(uid)
                            self._uid_index[uid] = (camera_id, track_id)
                            self.hub.emit(K_TRACK_STARTED, {
                                "uid": uid, "camera_id": camera_id, "track_id": track_id,
                                "video_source": cam["url"], "triggers": self._hub_triggers(cam),
                            })
                        else:
                            meta["last_seen_fid"] = fid
                            meta["seen_frames"] += 1
                        rec_state: TrackRecState = cam["rec_state"][track_id]

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
                            self._tlog(
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

                        # Spatial triggers -> control hub. Both behave the
                        # same now (they used to differ: cross_line published
                        # nothing when a confident read already existed, and
                        # stop_roi published nothing when its OCR submit
                        # failed). The hub publishes each exactly once.
                        for ev_key, stage, enabled in (("line_cross", "cross_line", cross_line_trig),
                                                       ("stopped_roi", "stop_roi", stop_roi_trig)):
                            ev = trigger_events.get(ev_key)
                            if not (enabled and ev):
                                continue
                            if stage in meta["events"]:
                                continue  # once per event per track (the hub dedupes too)
                            meta["events"][stage] = datetime.datetime.now().isoformat()
                            rec_state.request(stage, now_ts)
                            task_id = self._try_submit(camera_id, track_id, meta, rec_state, now_ts)
                            self.hub.emit(K_TRIGGER, {
                                "uid": meta["uid"], "camera_id": camera_id, "track_id": track_id,
                                "event": stage, "task_id": task_id, "detail": self._trigger_detail(ev),
                                "seen_frames": meta.get("seen_frames", 0),
                                "n_crops": len(cam["best_crops"].get(track_id, [])),
                            })
                            emoji = self.STAGE_EMOJI.get(stage, "🔔")
                            self.logger.info(
                                f"{emoji} [{'LINE-CROSS' if stage == 'cross_line' else 'STOPPED-ROI'}] camera={camera_id} "
                                f"track={track_id} uid={meta['uid']} class={ev['class']} detail={self._trigger_detail(ev)} "
                                f"-> hub (task={task_id})"
                            )

                        if stop_roi_trig and trigger_events.get("roi_entry"):
                            ev = trigger_events["roi_entry"]
                            self._tlog(
                                f"[ROI-ENTRY] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} "
                                f"point={ev['point']} confidence={ev['confidence']:.2f}"
                            )
                        if stop_roi_trig and trigger_events.get("roi_exit"):
                            ev = trigger_events["roi_exit"]
                            self._tlog(f"[ROI-EXIT] camera={ev['camera_id']} track={ev['track_id']} class={ev['class']} point={ev['point']}")

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

                        # pending requests (hub periodic re-query, or a trigger
                        # whose crops were not new yet) + low-rate heartbeat
                        if rec_state.requested:
                            self._try_submit(camera_id, track_id, meta, rec_state, now_ts)
                        if now_ts - rec_state.last_update_emit >= config.TRACK_UPDATE_INTERVAL_SEC:
                            rec_state.last_update_emit = now_ts
                            self.hub.emit(K_TRACK_UPDATE, self._track_stats(cam, track_id, meta))
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
                                **self._debug_ocr_fields(cam, track_id, now_ts),
                            }

                    # tracks gone for > ABSENT_N frames -> hand them to the hub
                    for tid, tmeta in list(cam["track_meta"].items()):
                        if (fid - int(tmeta["last_seen_fid"])) > self.ABSENT_N:
                            self._dbg_log(camera_id, "FINALIZE", f"#{tid} absent > {self.ABSENT_N} frames -> leaving scene",
                                          track_id=tid)
                            self._end_track(camera_id, tid, tmeta, reason="absent")

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
                finally:
                    _now = time.time()
                    ts = frame_ts[idx]
                    self.perf.frame_done(
                        camera_id, frame_timings[idx] if idx < len(frame_timings) else {},
                        track_ms=(_now - _cam_t0) * 1000.0,
                        latency_ms=(_now - ts) * 1000.0 if ts else None,
                        n_dets=int(detections.shape[0]),
                    )

            self.perf.batch_done(
                n_frames=batch_size, batch_ms=self._last_infer_ms,
                loop_ms=(time.time() - _loop_t0) * 1000.0,
                infer_frame_ms=[t.get("inference", 0.0) for t in frame_timings],
                missed=_loop_missed, captured=_loop_captured,
                dets=_batch_detections, tracks=_batch_tracks,
            )
            self.perf.maybe_log_engine(self.logger, len(self.cameras))

        self.logger.info("Engine stopping...")
        self.cleanup()

    def cleanup(self):
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id, reason="engine_stopped")
        self.hub.stop()
        try:
            self.backend.close()
        except Exception:
            pass


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
