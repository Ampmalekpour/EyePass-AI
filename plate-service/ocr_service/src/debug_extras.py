"""
debug_extras.py (ocr_service)
--------------------------------------------------------------------
New per-section visual debugging — the OCR service's counterpart to
detector/src/debug_extras.py's OCR-submission montage. Saves a small
labelled-grid JPEG per finalized task: the crop(s) that went into
recognition, each crop's raw OCR candidate + confidence, which one won
the vote, and the final validation result. The fastest way to see
*why* a plate was read (or misread, or rejected) without piecing it
together from log lines alone.

Gated by config.OCR_SAVE_DECISION_DEBUG — off by default, zero cost
when disabled. Local disk only (OCR_DECISION_DEBUG_DIR under the
bind-mounted debug tree), never MinIO — same storage-split rule as
everywhere else in this system: pipeline data goes to MinIO, debug
output stays local. Every function here follows debug_recorder.py's
rule: a debug failure must never touch the pipeline, so this only ever
logs, never raises.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

import numpy as np

import config
from platecore.debugging import ensure_dir, prune_dir, save_montage


def save_decision_montage(
    task_id: str,
    camera_id: str,
    track_id: int,
    trigger_type: str,
    voted_class: str,
    crops_bgr: List[np.ndarray],
    candidate_labels: List[str],
    voted_text: str,
    voted_conf: float,
    is_valid: bool,
    reason: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    try:
        out_dir = config.OCR_DECISION_DEBUG_DIR
        if not ensure_dir(out_dir, logger):
            return
        if not crops_bgr:
            return

        cells = []
        for i, img in enumerate(crops_bgr):
            label = candidate_labels[i] if i < len(candidate_labels) else "?"
            cells.append((img, [f"crop#{i}", label]))

        verdict = "VALID" if is_valid else "INVALID"
        title = (
            f"OCR {voted_class.upper()} cam={camera_id} track={track_id} stage={trigger_type} "
            f"-> '{voted_text}' conf={voted_conf:.2f} {verdict} ({reason})"
        )

        # task_id's own trailing field is a nanosecond timestamp (see
        # detector/src/engine.py's task_id format); fall back to
        # wall-clock if it's ever missing/malformed so this never blocks
        # on a parsing assumption about the caller's task_id shape.
        try:
            ts_suffix = task_id.split(":")[-1]
            int(ts_suffix)
        except Exception:
            ts_suffix = str(time.time_ns())
        fname = f"{camera_id}_{track_id}_{trigger_type}_{ts_suffix}.jpg"
        out_path = os.path.join(out_dir, fname)

        ok = save_montage(cells, out_path, cell_size=(180, 180), columns=4, title=title, logger=logger)
        if ok:
            prune_dir(out_dir, config.OCR_DECISION_DEBUG_MAX_FILES, suffixes=(".jpg",), logger=logger)
    except Exception as e:
        if logger:
            logger.debug(f"save_decision_montage failed task={task_id}: {e}")
