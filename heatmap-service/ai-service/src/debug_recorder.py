"""
debug_recorder.py
--------------------------------------------------------------------
Visual debug output for the heatmap ai_service — the same idea as the
plate/face/fire modules' own debug_recorder.py (a rolling, annotated
MP4 per camera plus a JSONL sidecar), which this module did not
previously have: the pre-existing standalone build's SAVE_OUTPUT only
wrote a plain bounding-box overlay with no view of the accumulation
grid itself.

What's drawn, on top of each processed frame:
  * every raw detection this frame (bounding box + confidence)
  * the accumulation grid (GRID_WIDTH x GRID_HEIGHT cells), each cell
    tinted by how "hot" it currently is in TODAY's in-memory cube —
    the same log-normalised heat used by HeatmapCubeManager.render_heatmap,
    recomputed cheaply from the cube slice already in RAM
  * camera id / timestamp stamp

Written to the local bind-mounted /debug volume — this is debugging
output only, never MinIO (MinIO is reserved for the cube data itself —
see minio_store.py / heatmap_manager.py).

Enabled by DETECTOR_DEBUG_VIDEO_ENABLED (config.py); off by default,
since annotated-video encoding is real CPU/disk cost.

Layout on disk (matches the other modules' convention):
    <DEBUG_VIDEO_DIR>/<camera_id>/seg_<timestamp>.mp4
    <DEBUG_VIDEO_DIR>/<camera_id>/seg_<timestamp>.jsonl   (one JSON per frame)

Segments roll over every DEBUG_VIDEO_SEGMENT_SECONDS, and only the
newest DEBUG_VIDEO_MAX_SEGMENTS are kept per camera (oldest deleted).
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config


class DebugRecorder:
    def __init__(self, camera_id: str, base_dir: str):
        self.camera_id = str(camera_id)
        self.dir = os.path.join(base_dir, self.camera_id)
        os.makedirs(self.dir, exist_ok=True)

        self.fps = max(1.0, float(config.DEBUG_VIDEO_FPS))
        self.segment_seconds = float(config.DEBUG_VIDEO_SEGMENT_SECONDS)
        self.max_segments = int(config.DEBUG_VIDEO_MAX_SEGMENTS)
        self.every_n = max(1, int(config.DEBUG_VIDEO_EVERY_N))
        self.codec = config.DEBUG_VIDEO_CODEC
        self.ext = config.DEBUG_VIDEO_EXT
        self.jsonl_enabled = bool(config.DEBUG_VIDEO_JSONL)

        self._writer: Optional[cv2.VideoWriter] = None
        self._jsonl_fh = None
        self._segment_started_at = 0.0
        self._frame_idx = 0
        self._size: Optional[tuple] = None

    # ------------------------------------------------------------------
    def _segment_base_name(self) -> str:
        return f"seg_{time.strftime('%Y%m%d-%H%M%S')}"

    def _roll_segment(self, frame_shape: tuple):
        self._close_writer()
        h, w = frame_shape[:2]
        self._size = (w, h)
        base = self._segment_base_name()
        video_path = os.path.join(self.dir, base + self.ext)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
        if self.jsonl_enabled:
            self._jsonl_fh = open(os.path.join(self.dir, base + ".jsonl"), "a", encoding="utf-8")
        self._segment_started_at = time.time()
        self._prune_old_segments()

    def _prune_old_segments(self):
        try:
            vids = sorted(
                (f for f in os.listdir(self.dir) if f.startswith("seg_") and f.endswith(self.ext)),
            )
        except FileNotFoundError:
            return
        excess = len(vids) - self.max_segments
        for f in vids[:max(0, excess)]:
            stem = f[:-len(self.ext)]
            for ext in (self.ext, ".jsonl"):
                p = os.path.join(self.dir, stem + ext)
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _close_writer(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_overlay(vis: np.ndarray, grid_slot_counts: Optional[np.ndarray]) -> np.ndarray:
        """Tints each accumulation-grid cell by its current heat (log1p
        of the current 5-min slot's counts, same normalisation
        HeatmapCubeManager.render_heatmap uses), alpha-blended under
        the frame so detections drawn afterward stay legible."""
        if grid_slot_counts is None or grid_slot_counts.size == 0:
            return vis

        H, W = vis.shape[:2]
        heat = np.log1p(grid_slot_counts.astype(np.float32))
        if heat.max() > 0:
            heat = (heat / heat.max() * 255.0).astype(np.uint8)
        else:
            heat = heat.astype(np.uint8)
        heat_resized = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
        return cv2.addWeighted(vis, 0.7, colored, 0.3, 0)

    def write(
            self,
            frame: np.ndarray,
            detections: List[Dict[str, Any]],
            class_labels: Dict[int, str],
            grid_slot_counts: Optional[np.ndarray] = None,
            frames_processed: int = 0,
    ):
        self._frame_idx += 1
        if self._frame_idx % self.every_n != 0:
            return
        if frame is None or frame.size == 0:
            return

        if self._writer is None or (time.time() - self._segment_started_at) > self.segment_seconds \
                or self._size != (frame.shape[1], frame.shape[0]):
            self._roll_segment(frame.shape)

        vis = frame.copy()
        if config.DEBUG_VIDEO_SCALE != 1.0:
            vis = cv2.resize(vis, None, fx=config.DEBUG_VIDEO_SCALE, fy=config.DEBUG_VIDEO_SCALE)

        vis = self._grid_overlay(vis, grid_slot_counts)

        scale = config.DEBUG_VIDEO_SCALE
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if scale != 1.0:
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            score = det["score"]
            cls_id = det["class_id"]
            color = (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{class_labels.get(cls_id, cls_id)} {score:.2f}"
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        H, W = vis.shape[:2]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(vis, f"cam={self.camera_id}  {ts}  frames={frames_processed}  people={len(detections)}",
                    (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self._writer.write(vis)

        if self._jsonl_fh is not None:
            event = {
                "ts": time.time(),
                "camera_id": self.camera_id,
                "frames_processed": frames_processed,
                "detections": [
                    {"bbox": d["bbox"], "score": round(d["score"], 3), "class_id": d["class_id"]}
                    for d in detections
                ],
            }
            self._jsonl_fh.write(json.dumps(event) + "\n")

    def close(self):
        self._close_writer()
