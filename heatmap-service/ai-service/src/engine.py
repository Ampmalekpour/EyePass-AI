# engine.py
# --------------------------------------------------------------------
# Batch-capable video processor - Multiple cameras supported.
# Integrated with 3D Spatial Grid Accumulation and ROI adjustments.
#
# Renamed from the pre-existing standalone build's video_processor.py
# (same name pattern as the plate/face/fire modules' own detector/src/
# engine.py) with three real additions, everything else carried over
# unchanged:
#   * results/ai_status go through common/heatmapcore.bus.RedisBus
#     instead of the module-local redis_client.py
#   * MinIO goes through common/heatmapcore.minio_store.MinioArrayStore
#     (boto3, matching the other modules) instead of minio_client.py
#   * optional DebugRecorder wiring — this module previously had no
#     visual-debug capability beyond a plain, ungridded bbox overlay
#
# Kept as a single file intentionally (easy to hand to an AI or read
# top-to-bottom while debugging). DetectionResult is defined directly
# below rather than imported from a separate models.py, so this file
# has no external dependency that can silently go missing.
# --------------------------------------------------------------------

import os
import sys
import cv2
import time
import torch
import logging
import threading
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import collections

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Please run: pip install ultralytics")
    sys.exit(1)

from config import (
    GridConfig, StorageConfig, DetectionConfig,
    HEATMAP_DATA_DIR, VIDEO_OUTPUT_DIR,
    SAVE_OUTPUT, SAVE_AS_VIDEO,
    CLASS_LABELS, MAX_CAMERAS_PER_ENGINE, DETECTION_DEVICE,
    DETECTOR_DEBUG_VIDEO_ENABLED, DEBUG_VIDEO_DIR,
)
from heatmap_manager import HeatmapCubeManager
from heatmapcore.bus import RedisBus
from heatmapcore.minio_store import MinioArrayStore
from debug_recorder import DebugRecorder

_default_grid = GridConfig()
_default_storage = StorageConfig()
_default_detection = DetectionConfig()


# --------------------------------------------------------------------
# Detection result (formerly models.py)
# --------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Structured result for a single detected object."""
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, float(y2)

    def get_point(self, mode: str) -> Tuple[float, float]:
        if mode == "center":
            return self.center
        elif mode == "bottom_center":
            return self.bottom_center
        else:
            raise ValueError(f"Invalid point mode: {mode}")

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
        }


# --------------------------------------------------------------------
# Compatibility stub for api.py's per-camera process entry
# --------------------------------------------------------------------
class CameraState:
    def __init__(self):
        self.running = True
        self.finished = False

    def stop(self):
        self.running = False

    def set_finished(self):
        self.finished = True


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# --------------------------------------------------------------------
# ResultsDispatcher — fire-and-forget wrapper over RedisBus.push_result,
# same .dispatch()/.shutdown() shape used elsewhere in this file.
# --------------------------------------------------------------------
class ResultsDispatcher:
    def __init__(self, bus: Optional[RedisBus] = None, max_workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor
        self.bus = bus or RedisBus()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def dispatch(self, payload: dict) -> None:
        try:
            self._executor.submit(self.bus.push_result, payload)
        except RuntimeError:
            # Executor already shut down (mid-teardown) — publish inline
            # so the final flush notification is not lost.
            self.bus.push_result(payload)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# --------------------------------------------------------------------
# RTSP reader thread
# --------------------------------------------------------------------
class RTSPStreamReader:
    def __init__(self, url: str, camera_id: str, max_queue_size: int = 10):
        self.url = url
        self.camera_id = str(camera_id)
        self.cap = None
        self.stopped = False
        self.max_queue_size = max_queue_size
        self.queue = collections.deque(maxlen=self.max_queue_size)

        self.thread = threading.Thread(
            target=self._update, daemon=True, name=f"RTSP-{self.camera_id}"
        )

    def _open(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            self.cap.open(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logging.warning(f"Camera {self.camera_id}: opening stream...")
                    self._open()
                    time.sleep(0.2)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning(f"Camera {self.camera_id}: read failed; reconnecting...")
                    self._open()
                    time.sleep(0.5)
                    continue

                self.queue.append(frame)

            except Exception as e:
                logging.warning(f"Camera {self.camera_id}: reader exception: {e}")
                time.sleep(0.2)

    def read(self):
        try:
            frame = self.queue.popleft()
            return True, frame
        except IndexError:
            return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.queue.clear()
        logging.info(f"[Reader - Cam {self.camera_id}] Stopped and queue cleared.")


# --------------------------------------------------------------------
# Engine Worker Process Instance
# --------------------------------------------------------------------
class Engine:
    def __init__(
            self,
            engine_id: int,
            model_path: str = _default_detection.model_path,
            imgsz: int = _default_detection.img_size,
            conf: float = _default_detection.conf_threshold,
            save_output: bool = SAVE_OUTPUT,
            output_dir: str = VIDEO_OUTPUT_DIR,
            class_labels: Optional[Dict[int, str]] = None,
            save_as_video: bool = SAVE_AS_VIDEO,
            status_queue: mp.Queue = None,
            control_queue: mp.Queue = None,
            stop_event: mp.Event = None,
            grid_width: int = _default_grid.grid_width,
            grid_height: int = _default_grid.grid_height,
            time_resolution_minutes: int = _default_storage.time_resolution_minutes,
            save_interval_seconds: int = _default_storage.save_interval_seconds,
            detection_accumulation_interval: int = _default_detection.detection_accumulation_interval,
            point_mode: str = _default_detection.point_mode,
            target_classes: Optional[List[int]] = None,
            results_publisher: Optional[ResultsDispatcher] = None,
            device: str = "auto",
            debug_video_enabled: bool = DETECTOR_DEBUG_VIDEO_ENABLED,
            debug_video_dir: str = DEBUG_VIDEO_DIR,
    ):
        self.engine_id = int(engine_id)
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        self.save_output = save_output
        self.save_as_video = save_as_video
        self.output_dir = output_dir

        self.status_queue = status_queue
        self.control_queue = control_queue
        self.stop_event = stop_event

        self.class_labels = class_labels if class_labels is not None else CLASS_LABELS
        self.target_classes = target_classes if target_classes is not None else list(_default_detection.target_classes)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.time_resolution_minutes = time_resolution_minutes
        self.save_interval_seconds = save_interval_seconds
        self.detection_accumulation_interval = detection_accumulation_interval
        self.point_mode = point_mode

        self.debug_video_enabled = debug_video_enabled
        self.debug_video_dir = debug_video_dir
        self.debug_recorders: Dict[str, DebugRecorder] = {}

        self.logger = setup_logger(f"Engine{self.engine_id}")
        # Resolve the device PREFERENCE ("auto"|"cpu"|"cuda"|"cuda:N") into a
        # concrete torch device here, inside the child process, so the
        # parent (api.py) never has to touch CUDA.
        requested = (device or "auto").strip().lower()
        if requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "device=%s requested but CUDA is not available; falling back to cpu",
                requested,
            )
            self.device = "cpu"
        else:
            self.device = requested

        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.writers: Dict[str, cv2.VideoWriter] = {}
        self.heatmap_managers: Dict[str, HeatmapCubeManager] = {}

        self.model = YOLO(self.model_path).to(self.device)
        self.logger.info(f"YOLO loaded on {self.device} | {self.model_path}")

        if self.save_output:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.debug_video_enabled:
            os.makedirs(self.debug_video_dir, exist_ok=True)

        self.last_flush_time = time.time()

        self.results_publisher = results_publisher or ResultsDispatcher()

        # One shared MinIO store per Engine process, handed to every
        # camera's HeatmapCubeManager, instead of each manager creating
        # (and bucket-checking) its own client independently.
        self.minio_store = MinioArrayStore()
        # One RedisBus per Engine process too, for ai_status writes.
        self.bus = RedisBus()

    def _annotate_and_write_detections(self, camera_id: str, frame: np.ndarray, detections: List[DetectionResult]):
        if not self.save_output or frame is None or frame.size == 0:
            return

        vis_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            score = det.confidence
            cls_id = det.class_id
            class_name = self.class_labels.get(cls_id, f"Class {cls_id}")

            color = (0, 165, 255)  # ORANGE
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} | Conf:{score:.2f}"
            cv2.putText(vis_frame, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if self.save_as_video:
            if camera_id not in self.writers:
                video_dir = os.path.join(self.output_dir, str(camera_id))
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"camera_{camera_id}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                h, w = frame.shape[:2]
                self.writers[camera_id] = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            self.writers[camera_id].write(vis_frame)
        else:
            camera_dir = os.path.join(self.output_dir, str(camera_id))
            os.makedirs(camera_dir, exist_ok=True)
            timestamp = time.time()
            filepath = os.path.join(camera_dir, f"frame_{timestamp:.3f}.jpg")
            cv2.imwrite(filepath, vis_frame)

    def add_camera(self, camera_id: str, url: str, roi: Tuple[float, float, float, float]):
        camera_id = str(camera_id)
        if camera_id in self.cameras:
            self.cameras[camera_id]["url"] = url
            self.cameras[camera_id]["roi"] = roi
            return

        reader = RTSPStreamReader(url, camera_id).start()

        frame_w, frame_h = 1920, 1080  # fallback defaults
        for _ in range(20):
            if reader.cap and reader.cap.isOpened():
                w = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    frame_w, frame_h = w, h
                    break
            time.sleep(0.05)

        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "url": url,
            "roi": roi,
            "reader": reader,
            "fid": 0,
            "frames_processed": 0,
            "last_frame_ts": None,
        }

        grid_cfg = GridConfig(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            frame_width=frame_w,
            frame_height=frame_h
        )
        storage_cfg = StorageConfig(
            storage_dir=HEATMAP_DATA_DIR,
            time_resolution_minutes=self.time_resolution_minutes,
            save_interval_seconds=self.save_interval_seconds,
            camera_id=camera_id
        )

        self.heatmap_managers[camera_id] = HeatmapCubeManager(
            grid_config=grid_cfg, storage_config=storage_cfg, store=self.minio_store
        )

        if self.debug_video_enabled:
            self.debug_recorders[camera_id] = DebugRecorder(camera_id, self.debug_video_dir)

        self._send_msg(camera_id, {"status": "running"})
        self.bus.write_ai_status(camera_id, current="running")
        self.logger.info(f"Added camera {camera_id} ({frame_w}x{frame_h}) with dedicated HeatmapCubeManager.")

    def remove_camera(self, camera_id: str):
        camera_id = str(camera_id)
        cam = self.cameras.pop(camera_id, None)
        if not cam:
            return
        try:
            cam["reader"].stop()
        except Exception:
            pass

        manager = self.heatmap_managers.pop(camera_id, None)
        if manager:
            try:
                manager.save_all()
                self.logger.info(f"Tracking cube flushed safely to MinIO for camera {camera_id}")

                current_date = datetime.now().strftime("%Y-%m-%d")
                object_key = manager._get_cube_key(current_date)

                self.results_publisher.dispatch({
                    "camera_id": camera_id,
                    "event": "matrix_sync",
                    "date": current_date,
                    "bucket": manager.store.bucket,
                    "object_key": object_key,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"Error during shutdown data dump for {camera_id}: {e}")

        rec = self.debug_recorders.pop(camera_id, None)
        if rec is not None:
            rec.close()

        w = self.writers.pop(camera_id, None)
        if w is not None:
            try:
                w.release()
            except Exception:
                pass

        self._send_msg(camera_id, {"status": "stopped"})
        self.bus.write_ai_status(camera_id, current="stopped")
        self.logger.info(f"Removed camera {camera_id}")

        self.results_publisher.dispatch({
            "camera_id": camera_id,
            "event": "processing_finished",
            "timestamp": time.time()
        })

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

            now_time = time.time()
            if now_time - self.last_flush_time > self.save_interval_seconds:
                self.logger.info("Triggering periodic matrix write loops across active streams...")
                for camera_id, mgr in list(self.heatmap_managers.items()):
                    try:
                        mgr.save_all()
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        object_key = mgr._get_cube_key(current_date)
                        self.results_publisher.dispatch({
                            "camera_id": str(camera_id),
                            "event": "matrix_sync",
                            "date": current_date,
                            "bucket": mgr.store.bucket,
                            "object_key": object_key,
                            "timestamp": time.time()
                        })
                    except Exception as e:
                        self.logger.error(f"Periodic flush execution failed for camera {camera_id}: {e}")
                self.last_flush_time = now_time

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
                    self.bus.write_ai_status(cid, current="error", error=str(e))
                time.sleep(0.1)
                continue

            for idx, res in enumerate(results):
                camera_id = cam_ids[idx]
                cam = self.cameras.get(camera_id)
                mgr = self.heatmap_managers.get(camera_id)
                if cam is None or mgr is None:
                    continue

                self._emit_heartbeat(camera_id)
                roi_frame = frames[idx]
                rx1, ry1, _, _ = roi_offsets[idx]
                frame_ts = datetime.now()

                detections: List[DetectionResult] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()

                    for i, bb in enumerate(boxes):
                        cls_id = int(classes[i])
                        if cls_id not in self.target_classes:
                            continue

                        x1, y1, x2, y2 = map(int, bb)
                        det = DetectionResult(bbox=(x1, y1, x2, y2), confidence=float(confs[i]), class_id=cls_id)
                        detections.append(det)

                        if cam["frames_processed"] % self.detection_accumulation_interval == 0:
                            px_cropped, py_cropped = det.get_point(self.point_mode)
                            px_global = px_cropped + rx1
                            py_global = py_cropped + ry1
                            mgr.accumulate(x=px_global, y=py_global, timestamp=frame_ts)

                if self.save_output:
                    self._annotate_and_write_detections(camera_id, roi_frame, detections)

                rec = self.debug_recorders.get(camera_id)
                if rec is not None:
                    rec.write(
                        frame=roi_frame,
                        detections=[d.to_dict() | {"score": d.confidence} for d in detections],
                        class_labels=self.class_labels,
                        grid_slot_counts=mgr.current_slot_counts(frame_ts),
                        frames_processed=cam["frames_processed"],
                    )

        self.logger.info("Engine stopping...")
        self.cleanup()

    def cleanup(self):
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)

        for w in list(self.writers.values()):
            try:
                w.release()
            except Exception:
                pass
        self.writers.clear()
        time.sleep(0.3)

        try:
            self.logger.info("Waiting for remaining background result pushes to clear...")
            self.results_publisher.shutdown(wait=True)
        except Exception:
            pass


def _engine_process_main(
        engine_id: int, model_path: str, imgsz: int, conf: float,
        save_output: bool, save_as_video: bool,
        status_queue: mp.Queue, control_queue: mp.Queue, stop_event: mp.Event,
        output_dir: str, class_labels: Dict[int, str],
        grid_width: int, grid_height: int, time_resolution_minutes: int,
        save_interval_seconds: int, detection_accumulation_interval: int, point_mode: str,
        target_classes: List[int], device: str,
        debug_video_enabled: bool, debug_video_dir: str,
):
    setup_logger(f"Engine{engine_id}")
    eng = Engine(
        engine_id=engine_id, model_path=model_path, imgsz=imgsz, conf=conf,
        save_output=save_output, save_as_video=save_as_video,
        status_queue=status_queue, control_queue=control_queue, stop_event=stop_event,
        output_dir=output_dir, class_labels=class_labels,
        grid_width=grid_width, grid_height=grid_height,
        time_resolution_minutes=time_resolution_minutes,
        save_interval_seconds=save_interval_seconds,
        detection_accumulation_interval=detection_accumulation_interval,
        point_mode=point_mode,
        target_classes=target_classes,
        device=device,
        debug_video_enabled=debug_video_enabled,
        debug_video_dir=debug_video_dir,
    )
    eng.run()


# --------------------------------------------------------------------
# EngineManager
# --------------------------------------------------------------------
class EngineManager:
    def __init__(
            self,
            model_path: str = _default_detection.model_path,
            imgsz: int = _default_detection.img_size,
            conf: float = _default_detection.conf_threshold,
            save_output: bool = SAVE_OUTPUT,
            output_dir: str = VIDEO_OUTPUT_DIR,
            class_labels: Optional[Dict[int, str]] = None,
            save_as_video: bool = SAVE_AS_VIDEO,
            max_cameras_per_engine: int = MAX_CAMERAS_PER_ENGINE,
            device: str = DETECTION_DEVICE,
            grid_width: int = _default_grid.grid_width,
            grid_height: int = _default_grid.grid_height,
            time_resolution_minutes: int = _default_storage.time_resolution_minutes,
            save_interval_seconds: int = _default_storage.save_interval_seconds,
            detection_accumulation_interval: int = _default_detection.detection_accumulation_interval,
            point_mode: str = _default_detection.point_mode,
            target_classes: Optional[List[int]] = None,
            debug_video_enabled: bool = DETECTOR_DEBUG_VIDEO_ENABLED,
            debug_video_dir: str = DEBUG_VIDEO_DIR,
    ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.save_output = bool(save_output)
        self.save_as_video = bool(save_as_video)
        self.output_dir = output_dir
        self.class_labels = class_labels if class_labels is not None else CLASS_LABELS
        self.max_cameras_per_engine = int(max_cameras_per_engine)
        self.device = device
        self.target_classes = target_classes if target_classes is not None else list(_default_detection.target_classes)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.time_resolution_minutes = time_resolution_minutes
        self.save_interval_seconds = save_interval_seconds
        self.detection_accumulation_interval = detection_accumulation_interval
        self.point_mode = point_mode
        self.debug_video_enabled = debug_video_enabled
        self.debug_video_dir = debug_video_dir

        self.status_queue: mp.Queue = mp.Queue()
        self._lock = threading.Lock()
        self.engines: Dict[int, Dict[str, Any]] = {}
        self.camera_to_engine: Dict[str, int] = {}

    def start(self, initial_engines: int = 1):
        with self._lock:
            for _ in range(int(initial_engines)):
                self._start_engine_locked()

    def engine_count(self) -> int:
        with self._lock:
            return len(self.engines)

    def _start_engine_locked(self) -> int:
        engine_id = 0 if not self.engines else (max(self.engines.keys()) + 1)

        control_queue = mp.Queue()
        stop_event = mp.Event()

        proc = mp.Process(
            target=_engine_process_main,
            args=(
                engine_id, self.model_path, self.imgsz, self.conf,
                self.save_output, self.save_as_video,
                self.status_queue, control_queue, stop_event,
                self.output_dir, self.class_labels,
                self.grid_width, self.grid_height, self.time_resolution_minutes,
                self.save_interval_seconds, self.detection_accumulation_interval, self.point_mode,
                self.target_classes, self.device,
                self.debug_video_enabled, self.debug_video_dir,
            ),
            daemon=False
        )
        proc.start()

        self.engines[engine_id] = {
            "proc": proc,
            "control_queue": control_queue,
            "stop_event": stop_event,
            "cameras": set(),
        }
        return engine_id

    def _pick_engine_locked(self) -> int:
        alive = []
        for eid, info in self.engines.items():
            p = info["proc"]
            if p is not None and p.is_alive():
                alive.append((len(info["cameras"]), eid))
        if not alive:
            return self._start_engine_locked()

        alive.sort()
        load, eid = alive[0]
        if load >= self.max_cameras_per_engine:
            return self._start_engine_locked()
        return eid

    def add_camera(self, camera_id: str, url: str, roi: Tuple[float, float, float, float]) -> Dict[str, Any]:
        camera_id = str(camera_id)
        url = str(url)

        with self._lock:
            if camera_id in self.camera_to_engine:
                eid = self.camera_to_engine[camera_id]
                info = self.engines.get(eid)
                if info and info["proc"].is_alive():
                    info["control_queue"].put({"cmd": "add", "camera_id": camera_id, "url": url, "roi": roi})
                    return {"camera_id": camera_id, "engine_id": eid, "status": "already_running"}
                self.camera_to_engine.pop(camera_id, None)

            eid = self._pick_engine_locked()
            info = self.engines[eid]
            info["cameras"].add(camera_id)
            self.camera_to_engine[camera_id] = eid
            info["control_queue"].put({"cmd": "add", "camera_id": camera_id, "url": url, "roi": roi})

            return {"camera_id": camera_id, "engine_id": eid, "status": "started"}

    def remove_camera(self, camera_id: str) -> Dict[str, Any]:
        camera_id = str(camera_id)
        with self._lock:
            eid = self.camera_to_engine.pop(camera_id, None)
            if eid is None:
                return {"camera_id": camera_id, "status": "not_running"}
            info = self.engines.get(eid)
            if info:
                info["cameras"].discard(camera_id)
                try:
                    info["control_queue"].put({"cmd": "remove", "camera_id": camera_id})
                except Exception:
                    pass
            return {"camera_id": camera_id, "engine_id": eid, "status": "stopped"}

    def shutdown(self):
        with self._lock:
            for eid, info in list(self.engines.items()):
                try:
                    info["control_queue"].put({"cmd": "stop"})
                    info["stop_event"].set()
                except Exception:
                    pass

            for eid, info in list(self.engines.items()):
                p = info.get("proc")
                if not p:
                    continue
                try:
                    p.join(timeout=5.0)
                except Exception:
                    pass
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
            self.engines.clear()
            self.camera_to_engine.clear()
