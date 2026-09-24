"""
debug_recorder.py
--------------------------------------------------------------------
Visual debug output for the fire/smoke detector — the same idea as
plate_detector's/face_detector's debug_recorder.py (a rolling,
annotated MP4 per camera plus a JSONL sidecar of events), scaled down
to what this module actually needs: bounding boxes, the 2x2 spatial
grid, each region's current verdict, and the global cooldown state.
Written to the local bind-mounted /debug volume — this is debugging
output, never MinIO (MinIO is reserved for the THREAT/RESOLUTION
alert crops the backend actually consumes — see engine.py).

Enabled by DEBUG_VIDEO_ENABLED (config.py); off by default since
annotated-video encoding is real CPU/disk cost.

Layout on disk (matches the plate/face modules' DEBUGGING.md
convention):
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

_REGION_COLOR = {
    "CLEAR": (80, 200, 80),
    "SMOKE": (0, 200, 255),
    "FIRE": (0, 80, 255),
    "BOTH": (0, 0, 255),
}
_CLASS_COLOR = {0: (0, 200, 255), 1: (0, 80, 255)}  # 0=Smoke, 1=Fire


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
    def write(
            self,
            frame: np.ndarray,
            detections: List[Dict[str, Any]],
            class_labels: Dict[int, str],
            region_verdicts: Dict[int, str],
            cooldown_active: bool,
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

        H, W = vis.shape[:2]
        # 2x2 grid lines
        cv2.line(vis, (W // 2, 0), (W // 2, H), (90, 90, 90), 1)
        cv2.line(vis, (0, H // 2), (W, H // 2), (90, 90, 90), 1)

        # per-region verdict label
        region_origin = {0: (8, 22), 1: (W // 2 + 8, 22), 2: (8, H // 2 + 22), 3: (W // 2 + 8, H // 2 + 22)}
        for r_id, verdict in region_verdicts.items():
            color = _REGION_COLOR.get(verdict, (200, 200, 200))
            ox, oy = region_origin[r_id]
            cv2.putText(vis, f"R{r_id}: {verdict}", (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # detection boxes
        scale = config.DEBUG_VIDEO_SCALE
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if scale != 1.0:
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            score = det["score"]
            cls_id = det["class_id"]
            color = _CLASS_COLOR.get(cls_id, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{class_labels.get(cls_id, cls_id)} {score:.2f}"
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if cooldown_active:
            cv2.putText(vis, "COOLDOWN", (W - 150, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(vis, f"cam={self.camera_id}  {ts}", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        self._writer.write(vis)

        if self._jsonl_fh is not None:
            event = {
                "ts": time.time(),
                "camera_id": self.camera_id,
                "region_verdicts": region_verdicts,
                "cooldown_active": cooldown_active,
                "detections": [
                    {"bbox": d["bbox"], "score": round(d["score"], 3), "class_id": d["class_id"]}
                    for d in detections
                ],
            }
            self._jsonl_fh.write(json.dumps(event) + "\n")

    def close(self):
        self._close_writer()
