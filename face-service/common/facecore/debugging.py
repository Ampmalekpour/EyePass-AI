"""
debugging.py (facecore)
--------------------------------------------------------------------
Shared visual-debug + timing utilities for both services (and the
add-face enrollment path). Everything here follows the rule
debug_recorder.py already established: a debug failure must never
touch the pipeline — every public function swallows its own
exceptions, optionally logs once, and returns None/False rather than
raising.

What lives here:
  * env_bool/env_int/env_float/env_str — identical semantics to each
    service's own config.py _bool/_int/_float, factored out once so
    new debug code in EITHER service (and facecore itself) doesn't
    reimplement env parsing a fourth time. Each service's config.py
    keeps its OWN copies too (nothing here replaces those) — this is
    only for new shared debug code that isn't naturally a config.py
    concern.
  * Stopwatch / timed() — for the "give me times" part of
    professional logging: consistent elapsed_ms everywhere instead of
    ad hoc time.time() deltas.
  * ensure_dir — os.makedirs(..., exist_ok=True) that never raises and
    proves the dir is actually writable (bind mount not attached yet,
    permissions, etc. all degrade to "debug disabled", not a crash).
  * save_montage — arrange N labelled BGR images into one grid JPEG;
    used by the detector's best-crop montage, the recognizer's
    match-visualization, and add-face's pose-check / enrollment cards.
  * prune_dir — bounded retention for any debug folder (same idea as
    DebugRecorder._prune(), factored out so it isn't reimplemented in
    four more places).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------
def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def ensure_dir(path: str, logger: Optional[logging.Logger] = None) -> bool:
    """os.makedirs that never raises — returns whether the dir is
    actually usable (writes and removes a small probe file). Debug
    output must never be able to take the pipeline down just because a
    bind mount isn't attached yet or perms are wrong."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception as e:
        if logger:
            logger.warning("debug dir not writable (%s): %s", path, e)
        return False


# ---------------------------------------------------------------------
# timing — "whatever a debugger needs to see in terms of times"
# ---------------------------------------------------------------------
class Stopwatch:
    """
        with Stopwatch() as sw:
            ...
        sw.elapsed_ms   # float, set on __exit__

    Or inline: sw = Stopwatch(); ...; ms = sw.stop().
    Measures only — never swallows an exception raised inside it.
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return self.elapsed_ms


@contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.DEBUG, **fields):
    """
        with timed(logger, "embedding", track_id=7):
            ...

    Logs one line with elapsed_ms on exit, success or not (the
    exception, if any, is named in the log line and then re-raised
    unchanged — this never hides a real failure). `fields` are cheap
    keyword context (ids, counts) meant to always be computed; for
    genuinely expensive debug payloads (full similarity vectors, image
    bytes), guard the computation itself with
    `logger.isEnabledFor(logging.DEBUG)` before entering the block.
    """
    t0 = time.perf_counter()
    try:
        yield
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if logger.isEnabledFor(level):
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.log(level, "%s FAILED elapsed_ms=%.2f %s (%s: %s)",
                       label, elapsed_ms, extra, type(e).__name__, e)
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if logger.isEnabledFor(level):
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.log(level, "%s elapsed_ms=%.2f %s", label, elapsed_ms, extra)


# ---------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------
def prune_dir(dir_path: str, max_files: int, suffixes: Tuple[str, ...] = (),
              logger: Optional[logging.Logger] = None) -> None:
    """Keeps at most `max_files` most-recently-modified files (matching
    `suffixes` if given, e.g. (".jpg",)) in dir_path, deleting the
    oldest first. Same behaviour DebugRecorder._prune() already has for
    video segments, factored out so per-frame debug dumps (crops,
    montages, enrollment cards) don't grow unbounded either.
    max_files <= 0 means unlimited (never prunes)."""
    if max_files <= 0:
        return
    try:
        files = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if (not suffixes) or f.lower().endswith(suffixes)
        ]
        files.sort(key=lambda p: os.path.getmtime(p))
        for old in files[:-max_files]:
            try:
                os.remove(old)
            except Exception:
                pass
    except Exception as e:
        if logger:
            logger.debug("prune_dir(%s) skipped: %s", dir_path, e)


# ---------------------------------------------------------------------
# montage rendering — shared by the detector's best-crop montage, the
# recognizer's match-visualization, and add-face's pose-check /
# enrollment-card images.
# ---------------------------------------------------------------------
_MONTAGE_BG = (24, 24, 24)
_MONTAGE_TEXT = (235, 235, 235)
_MONTAGE_DIM = (150, 150, 150)


def _label_block(img: np.ndarray, lines: Sequence[str], origin: Tuple[int, int],
                  color=_MONTAGE_TEXT, scale: float = 0.42) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        yy = y + int(i * 16) + 12
        (tw, th), base = cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x - 2, yy - th - 2), (x + tw + 2, yy + base), (10, 10, 10), -1)
        cv2.putText(img, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def save_montage(
    cells: Sequence[Tuple[Optional[np.ndarray], List[str]]],
    out_path: str,
    cell_size: Tuple[int, int] = (160, 160),
    columns: int = 4,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """cells: [(bgr_image_or_None, [label_line, ...]), ...] — one cell
    per crop / match / candidate. A None image renders as a visibly
    distinct "no image" placeholder rather than a blank gap, so a
    missing image is never mistaken for a rendering bug.
    Returns whether the file was written; never raises."""
    try:
        if not cells:
            return False
        cw, ch = cell_size
        columns = max(1, int(columns))
        rows = (len(cells) + columns - 1) // columns
        title_h = 26 if title else 0
        label_h = 16 * max((len(lbls) for _, lbls in cells), default=1) + 6
        canvas_h = title_h + rows * (ch + label_h + 8) + 8
        canvas_w = columns * (cw + 8) + 8
        canvas = np.full((canvas_h, canvas_w, 3), _MONTAGE_BG, dtype=np.uint8)

        if title:
            cv2.putText(canvas, str(title), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _MONTAGE_TEXT, 1, cv2.LINE_AA)

        for idx, (img, labels) in enumerate(cells):
            r, c = divmod(idx, columns)
            x0 = 8 + c * (cw + 8)
            y0 = title_h + 8 + r * (ch + label_h + 8)

            cell = np.full((ch, cw, 3), (40, 40, 40), dtype=np.uint8)
            if img is not None and getattr(img, "size", 0) > 0:
                resized = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
                if resized.ndim == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                cell = resized
            else:
                cv2.putText(cell, "no image", (10, ch // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, _MONTAGE_DIM, 1, cv2.LINE_AA)

            canvas[y0:y0 + ch, x0:x0 + cw] = cell
            _label_block(canvas, labels, origin=(x0, y0 + ch))

        ensure_dir(os.path.dirname(out_path) or ".", logger)
        return bool(cv2.imwrite(out_path, canvas))
    except Exception as e:
        if logger:
            logger.warning("save_montage failed for %s: %s", out_path, e)
        return False
