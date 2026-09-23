"""
alignment.py (recognizer)
--------------------------------------------------------------------
Verbatim port of the alignment-related methods of the reference
AFRWorker (fr_worker.py): `_align_face`, `_align_with_yolo_landmarks`
and `_save_landmarked_crop`. Extracted into a standalone `FaceAligner`
so `worker.py` stays focused on process/queue plumbing.

Two alignment paths, unchanged from the reference:
  * YOLO-landmark alignment (preferred): the detector already ran a
    landmark-capable YOLO model and attached 14 keypoints to each
    crop; 5 of them (YOLO_5PT_IDX) are warped against
    REFERENCE_FACIAL_POINTS via warp_and_crop_face, from the
    operator-supplied `face_alignment` package.
  * Legacy MTCNN-style alignment (fallback): face_alignment.align's
    own MTCNN-based detector runs on the crop directly.

DEBUG_SAVE_LANDMARKED_CROPS / DEBUG_SAVE_ALIGNED write to the local
bind-mounted debug volume only (never MinIO) per the module's storage
split.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

import config
from facecore.logging_setup import setup_logger

logger = setup_logger("recognizer.alignment")

try:
    from face_alignment import align as _align
    from face_alignment.mtcnn_pytorch.src.align_trans import warp_and_crop_face
except Exception as e:  # pragma: no cover - only hit if the operator
    # hasn't bind-mounted face_alignment/ yet; worker.py's startup
    # check surfaces this clearly instead of failing on first crop.
    _align = None
    warp_and_crop_face = None
    logger.warning("face_alignment package not importable yet: %s", e)


class FaceAligner:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        if config.DEBUG_SAVE_LANDMARKED_CROPS:
            os.makedirs(config.DEBUG_LANDMARKED_DIR, exist_ok=True)
        if config.DEBUG_SAVE_ALIGNED:
            os.makedirs(config.DEBUG_ALIGNED_DIR, exist_ok=True)

    def save_landmarked_crop(self, crop_bgr: np.ndarray, landmarks: Any, track_id: int, crop_idx: int):
        """Draw landmarks on the original crop and save it for debugging."""
        if not config.DEBUG_SAVE_LANDMARKED_CROPS or crop_bgr is None:
            return
        try:
            img = crop_bgr.copy()

            if isinstance(landmarks, np.ndarray) and landmarks.ndim == 2:
                points = landmarks[:, :2].astype(int)
                confs = landmarks[:, 2] if landmarks.shape[1] >= 3 else None
                for i, (x, y) in enumerate(points):
                    color = (0, 255, 0) if (confs is None or confs[i] > 0.5) else (0, 165, 255)
                    cv2.circle(img, (x, y), 3, color, -1)
                    cv2.circle(img, (x, y), 5, (255, 255, 255), 1)
            elif isinstance(landmarks, list) and landmarks:
                for lm in landmarks:
                    if isinstance(lm, dict):
                        x, y = int(lm.get("x", 0)), int(lm.get("y", 0))
                        conf = lm.get("conf", 1.0)
                    else:
                        x, y = int(lm[0]), int(lm[1])
                        conf = lm[2] if len(lm) >= 3 else 1.0
                    color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                    cv2.circle(img, (x, y), 3, color, -1)
                    cv2.circle(img, (x, y), 5, (255, 255, 255), 1)

            cv2.putText(img, f"trk{track_id}_c{crop_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            timestamp = int(time.time() * 1000)
            save_path = os.path.join(config.DEBUG_LANDMARKED_DIR, f"landmarked_trk{track_id}_c{crop_idx}_{timestamp}.jpg")
            cv2.imwrite(save_path, img)
        except Exception as e:
            logger.warning("[AFR-%s] Failed to save landmarked crop: %s", self.worker_id, e)

    def align_face(self, crop_bgr: np.ndarray, landmarks: Any = None, track_id: int = 0) -> Optional[Image.Image]:
        """Main alignment entry point. Prefers YOLO-landmark alignment,
        falls back to legacy MTCNN-style alignment, then to a raw
        (unaligned) crop as a last resort — matching the reference's
        "always return something" contract so callers never see a hard
        failure from a bad crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        method = "YOLO_Landmark" if (config.USE_YOLO_ALIGNMENT and landmarks is not None) else "MTCNN_Style"
        t0 = time.time()

        try:
            aligned_pil = None
            if config.USE_YOLO_ALIGNMENT and landmarks is not None:
                aligned_pil = self._align_with_yolo_landmarks(crop_bgr, landmarks)

            if aligned_pil is None:
                if _align is None:
                    raise RuntimeError("face_alignment package not available")
                face_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_input = Image.fromarray(face_rgb)
                aligned_pil = _align.get_aligned_face(None, rgb_pil_image=pil_input)
                if aligned_pil is None:
                    aligned_pil = pil_input

            if config.DEBUG_SAVE_ALIGNED and aligned_pil is not None:
                timestamp = int(time.time() * 1000)
                avg_conf = 0.0
                if landmarks is not None:
                    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 2 and landmarks.shape[1] >= 3:
                        avg_conf = float(np.mean(landmarks[:, 2]))
                    elif isinstance(landmarks, list) and landmarks and isinstance(landmarks[0], dict):
                        avg_conf = float(np.mean([lm.get("conf", 0) for lm in landmarks if isinstance(lm, dict)]))
                save_path = os.path.join(
                    config.DEBUG_ALIGNED_DIR,
                    f"aligned_{timestamp}_trk{track_id}_conf{avg_conf:.3f}_{method}.jpg",
                )
                aligned_pil.save(save_path, quality=95)

            return aligned_pil
        except Exception as e:
            logger.warning("[AFR-%s] Alignment error (%s): %s", self.worker_id, method, e)
            fallback_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            fallback_pil = Image.fromarray(fallback_rgb)
            if config.DEBUG_SAVE_ALIGNED:
                save_path = os.path.join(config.DEBUG_ALIGNED_DIR, f"fallback_{int(time.time() * 1000)}.jpg")
                fallback_pil.save(save_path, quality=90)
            return fallback_pil

    def _align_with_yolo_landmarks(self, crop_bgr: np.ndarray, landmarks: Any) -> Optional[Image.Image]:
        """Align using YOLO 14 keypoints -> 5 facial points -> warp_and_crop_face.
        Returns an RGB PIL Image of size 112x112."""
        if warp_and_crop_face is None:
            return None

        try:
            if isinstance(landmarks, np.ndarray):
                if landmarks.ndim != 2 or landmarks.shape[1] < 3:
                    return None
                landmarks_list = [{"x": float(pt[0]), "y": float(pt[1]), "conf": float(pt[2])} for pt in landmarks]
            else:
                landmarks_list = landmarks

            if len(landmarks_list) < 10:
                return None

            facial5points = []
            for idx in config.YOLO_5PT_IDX:
                if idx >= len(landmarks_list):
                    return None
                lm = landmarks_list[idx]
                x = lm["x"] if isinstance(lm, dict) else float(lm[0])
                y = lm["y"] if isinstance(lm, dict) else float(lm[1])
                conf = lm["conf"] if isinstance(lm, dict) else float(lm[2])
                if conf < config.YOLO_LANDMARK_CONF_THR:
                    return None
                facial5points.append([x, y])

            facial5points = np.array(facial5points, dtype=np.float32).copy()

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).copy()
            pil_img = Image.fromarray(crop_rgb)
            img_for_warp = np.array(pil_img)

            safe_ref_pts = np.array(config.REFERENCE_FACIAL_POINTS, dtype=np.float32).copy()

            aligned_np = warp_and_crop_face(img_for_warp, facial5points, safe_ref_pts, crop_size=(112, 112))
            if aligned_np is None:
                return None

            return Image.fromarray(aligned_np)
        except Exception as e:
            logger.warning("[AFR-%s] YOLO landmark alignment failed: %s", self.worker_id, e)
            return None
