"""
debug_extras.py (recognizer, live path)
--------------------------------------------------------------------
Two optional visual-debug writers for the LIVE recognition path (the
add-face/enroll equivalents live in pose.py + worker.py's enroll
handlers, see DEBUGGING.md for the full map). Both off by default,
both fail-safe: any failure here is caught, logged once at DEBUG
level, and never propagates into the recognition hot path.

  save_match_visualization()   crop -> aligned 112x112 -> top-K gallery
                                matches, each thumbnail loaded straight
                                from this worker's already-downloaded
                                KNOWN_FACES_DIR, labelled with its raw
                                cosine similarity and which personnelid
                                block it belongs to. Answers "what did
                                it actually match against, and how
                                close was the runner-up" — the question
                                a bare confidence number in a log line
                                can't answer.

  save_rejected_crop()         Same composite, saved when the decision
                                math lands on Unknown — tells "genuinely
                                unknown person" apart from "known
                                person, bad angle/lighting caused a
                                miss" at a glance.

Both are called from worker.py's `_recognize_face_once` right after
`compare_image()` returns, where the ranked similarity list, the
original crop, and the aligned face are all already in hand — this
module does no recognition work of its own, only rendering.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config
from facecore.debugging import prune_dir, save_montage

logger = logging.getLogger("recognizer.debug_extras")


def _pil_to_bgr(img: Optional[Image.Image]) -> Optional[np.ndarray]:
    if img is None:
        return None
    try:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _load_gallery_thumb(known_faces_dir: str, gallery_path: str) -> Optional[np.ndarray]:
    """`gallery_path` is the .pkl "identity" string (a filesystem path
    recorded at embedding time — normally already inside
    known_faces_dir, but resolved defensively in case the gallery
    folder moved since the .pkl was built)."""
    try:
        candidate = gallery_path
        if not os.path.isfile(candidate):
            candidate = os.path.join(known_faces_dir, os.path.basename(gallery_path))
        if not os.path.isfile(candidate):
            return None
        return cv2.imread(candidate)
    except Exception:
        return None


def _pid_for(path: str, images_per_person: int) -> str:
    match = re.search(r"c(\d+)", os.path.basename(path))
    if not match:
        return "?"
    image_num = int(match.group(1))
    return str((image_num - 1) // images_per_person + 1)


def save_match_visualization(
    face_bgr: Optional[np.ndarray], aligned_pil: Optional[Image.Image],
    ranked_list: List[Tuple[str, float]], person_id: str, confidence: float,
    known_faces_dir: str, track_id: Any,
) -> None:
    if not config.DEBUG_SAVE_MATCHES:
        return
    try:
        cells = [
            (face_bgr, ["input crop"]),
            (_pil_to_bgr(aligned_pil), ["aligned 112x112"]),
        ]
        for path, score in ranked_list[:max(1, config.DEBUG_MATCHES_TOP_N)]:
            pid = _pid_for(path, 3)
            marker = " <-- BEST" if pid == person_id else ""
            cells.append((
                _load_gallery_thumb(known_faces_dir, path),
                [f"pid {pid}{marker}", f"sim {score:.4f}", os.path.basename(path)],
            ))

        out_dir = config.DEBUG_MATCHES_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = os.path.join(out_dir, f"trk{track_id}_{stamp}_pid{person_id}_conf{confidence:.3f}.jpg")

        ok = save_montage(
            cells, out_path, cell_size=(130, 130), columns=len(cells),
            title=f"track {track_id} -> pid {person_id} (conf {confidence:.3f})",
            logger=logger,
        )
        if ok:
            prune_dir(out_dir, config.DEBUG_MATCHES_MAX_FILES, (".jpg",), logger)
    except Exception as e:
        logger.debug("save_match_visualization failed (track=%s): %s", track_id, e)


def save_rejected_crop(
    face_bgr: Optional[np.ndarray], aligned_pil: Optional[Image.Image],
    ranked_list: List[Tuple[str, float]], confidence: float, track_id: Any,
) -> None:
    if not config.DEBUG_SAVE_REJECTED:
        return
    try:
        top = ranked_list[:3] if ranked_list else []
        labels = [f"REJECTED conf {confidence:.3f}"]
        labels += [f"{os.path.basename(p)} sim{s:.3f}" for p, s in top] or ["no candidates"]

        out_dir = config.DEBUG_REJECTED_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = os.path.join(out_dir, f"trk{track_id}_{stamp}.jpg")

        cells = [(face_bgr, labels[:1]), (_pil_to_bgr(aligned_pil), labels[1:])]
        ok = save_montage(cells, out_path, cell_size=(150, 150), columns=2,
                          title="REJECTED (Unknown)", logger=logger)
        if ok:
            prune_dir(out_dir, config.DEBUG_REJECTED_MAX_FILES, (".jpg",), logger)
    except Exception as e:
        logger.debug("save_rejected_crop failed (track=%s): %s", track_id, e)
