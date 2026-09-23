"""
debug_extras.py (detector)
--------------------------------------------------------------------
Two small, fully optional visual-debug writers that sit alongside
debug_recorder.py's annotated video, both off by default:

  save_best_crop_montage()   Called from engine.py's two crop-
                              submission functions (_send_intermediate
                              _best_crop / _send_spatial_event_crop).
                              Saves a labelled grid of every candidate
                              currently held in the best-crop ladder
                              (reg1/reg2/reg3) for this track, with the
                              one actually submitted to the recognizer
                              marked — answers "what pixels actually
                              left the detector", which the annotated
                              video's on-screen quality NUMBERS don't
                              show you directly.

  save_liveness_reject()     Called the moment a track's liveness
                              verdict flips to "fake" (engine.py, the
                              same spot that already logs a SPOOF event
                              to the recorder). Keeps a standing,
                              easy-to-scan folder of just the rejects —
                              spoof attempts are short and easy to miss
                              scrubbing a 30s rolling video segment.

Same fail-safe contract as debug_recorder.py: any failure here is
caught, logged once at DEBUG level, and never propagates — a bad debug
write must never be able to take a camera's engine process down.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

import config
from facecore.debugging import ensure_dir, prune_dir, save_montage

logger = logging.getLogger("detector.debug_extras")


def _montage_dir(camera_id: str) -> str:
    return os.path.join(config.DEBUG_VIDEO_DIR, "best_crop_montages", f"camera_{camera_id}")


def _liveness_reject_dir(camera_id: str) -> str:
    return os.path.join(config.DEBUG_VIDEO_DIR, "liveness_rejects", f"camera_{camera_id}")


def save_best_crop_montage(camera_id: str, track_id: int, event_type: str,
                            crops_dict: Dict[str, list], submitted_reg: str) -> None:
    """crops_dict: cam["best_crops"][track_id] — {"reg1": [...], "reg2":
    [...], "reg3": [...]}, each item a dict with at least "image"
    (BGR ndarray), "mlc", "yaw_group", "resolution". submitted_reg is
    whichever of reg1/reg2/reg3 actually got sent this time (see the
    `best_reg = crops_dict.get("reg1", []) or crops_dict.get("reg2", [])
    or ...` fallback chain at the call site)."""
    if not config.DEBUG_BEST_CROP_MONTAGE_ENABLED:
        return
    try:
        cells = []
        for reg_name in ("reg1", "reg2", "reg3"):
            items = crops_dict.get(reg_name) or []
            for rank, item in enumerate(items):
                img = item.get("image")
                sent = " -> SENT" if (reg_name == submitted_reg and rank == 0) else ""
                labels = [
                    f"{reg_name}#{rank}{sent}",
                    f"mlc {item.get('mlc', 0.0):.2f} yg{item.get('yaw_group', 0)}",
                    f"res {item.get('resolution', 0)}",
                ]
                cells.append((img, labels))

        if not cells:
            return

        out_dir = _montage_dir(camera_id)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = os.path.join(out_dir, f"trk{track_id}_{event_type}_{stamp}.jpg")

        ok = save_montage(
            cells, out_path, cell_size=(140, 140), columns=5,
            title=f"cam {camera_id} track {track_id} ({event_type}) — submitted: {submitted_reg}",
            logger=logger,
        )
        if ok:
            prune_dir(out_dir, config.DEBUG_BEST_CROP_MONTAGE_MAX_FILES, (".jpg",), logger)
    except Exception as e:
        logger.debug("save_best_crop_montage failed (cam=%s track=%s): %s", camera_id, track_id, e)


def save_liveness_reject(camera_id: str, track_id: int, roi_frame: Optional[np.ndarray],
                          bbox, liveness_meta: Dict[str, Any]) -> None:
    """One still at the moment of a FAKE verdict: the track's crop plus
    a small label block with the score/reason/metrics that produced it.
    `bbox` is (x1, y1, x2, y2) in roi_frame coordinates."""
    if not config.DEBUG_LIVENESS_REJECTS_ENABLED:
        return
    try:
        if roi_frame is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = roi_frame.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        crop = roi_frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return

        metrics = liveness_meta.get("liveness_metrics") or {}
        labels = [
            f"cam {camera_id} trk {track_id}",
            f"score {liveness_meta.get('liveness_score', 0.0):.3f}",
            f"reason {str(liveness_meta.get('liveness_reason', ''))[:24]}",
            f"pl {metrics.get('planar_residual', '-')} ring {metrics.get('ring_follow', '-')}",
            f"rig {metrics.get('rigid_residual', '-')} dpose {metrics.get('pose_delta_deg', '-')}",
        ]

        out_dir = _liveness_reject_dir(camera_id)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = os.path.join(out_dir, f"trk{track_id}_{stamp}.jpg")

        ok = save_montage([(crop, labels)], out_path, cell_size=(220, 220), columns=1,
                          title="LIVENESS REJECT", logger=logger)
        if ok:
            prune_dir(out_dir, config.DEBUG_LIVENESS_REJECTS_MAX_FILES, (".jpg",), logger)
    except Exception as e:
        logger.debug("save_liveness_reject failed (cam=%s track=%s): %s", camera_id, track_id, e)
