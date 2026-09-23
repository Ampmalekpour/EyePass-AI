"""
debug_extras.py (detector)
--------------------------------------------------------------------
New per-section visual debugging, separate from debug_recorder.py's
always-on annotated video. This is the detector's half of the
"visual debugging per section" work — a small labelled-grid JPEG
saved every time a track is submitted to OCR, showing exactly which
crops (and best-frame) went out in that task. The fastest way to
answer "why did OCR get a bad crop" without scrubbing the full debug
video for the right timestamp.

Gated by config.DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED — off by default,
zero cost when disabled. Every function here follows debug_recorder.py's
own rule: a debug failure must never touch the pipeline, so everything
is wrapped and only ever logs, never raises.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import config
from platecore.debugging import ensure_dir, prune_dir, save_montage


def save_ocr_submission_montage(
    camera_id: str,
    track_id: int,
    trigger_type: str,
    task_id: str,
    crops: List[Dict[str, Any]],
    best_frame: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """crops: cam["best_crops"][track_id] entries — each has "image"
    (BGR ndarray), "class_flag", "det_score", "resolution",
    "frame_number". best_frame: cam["best_frame"][track_id], or None —
    has "frame", "bbox", "resolution"."""
    try:
        out_dir = config.DEBUG_OCR_SUBMISSION_MONTAGE_DIR
        if not ensure_dir(out_dir, logger):
            return

        cells = []
        for item in crops:
            img = item.get("image")
            labels = [
                f"score={float(item.get('det_score', 0.0)):.2f}",
                f"res={int(item.get('resolution', 0))}",
                f"frame#{int(item.get('frame_number', 0))}",
            ]
            cells.append((img, labels))

        if best_frame is not None:
            cells.append((best_frame.get("frame"), ["best_frame", f"bbox={best_frame.get('bbox')}"]))

        if not cells:
            return

        # task_id's own trailing field is a nanosecond timestamp (see
        # engine.py's task_id = f"{engine}:{camera}:{track}:{trigger}:{time.time_ns()}")
        # — reuse it verbatim so filenames sort chronologically and tie
        # back to the exact submission unambiguously.
        ts_suffix = task_id.split(":")[-1] if task_id else "0"
        fname = f"{camera_id}_{track_id}_{trigger_type}_{ts_suffix}.jpg"
        out_path = os.path.join(out_dir, fname)

        title = f"OCR SUBMIT cam={camera_id} track={track_id} stage={trigger_type}"
        ok = save_montage(cells, out_path, cell_size=(180, 180), columns=4, title=title, logger=logger)
        if ok:
            prune_dir(out_dir, config.DEBUG_OCR_SUBMISSION_MONTAGE_MAX_FILES, suffixes=(".jpg",), logger=logger)
    except Exception as e:
        if logger:
            logger.debug(f"save_ocr_submission_montage failed camera={camera_id} track={track_id}: {e}")
