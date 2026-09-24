"""
rtsp_reader.py
--------------------------------------------------------------------
RTSPStreamReader, unchanged in behavior from the reference
video_processor.py: a background thread that keeps only the LATEST
frame from the camera (not a queue of several — the engine loop
consumes whatever frame is sitting in `self.frame` once per batch
cycle; if the camera delivers frames faster than the engine can
process them, frames are silently overwritten, and `frames_captured`
below is how the engine measures that — see Engine.run()'s
[FRAME-SKIP]/[PIPELINE] logging).

The URL handed to this reader is always the MediaMTX relay URL (see
backend_bridge.rtsp_url), never the camera's own RTSP address.
--------------------------------------------------------------------
"""

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

import config


class RTSPStreamReader:
    """
    Background thread that keeps only the LATEST frame from the camera.

    `frames_captured` counts every frame this thread actually received
    from the camera; the engine compares consecutive snapshots of it
    against its own processed-frame count to report how many were
    silently skipped.
    """

    def __init__(self, url: str, camera_id: str):
        self.url = url
        self.camera_id = str(camera_id)
        self.logger = logging.getLogger(f"rtsp.{self.camera_id}")

        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False

        self._lock = threading.Lock()

        # -------- diagnostics (additive only, doesn't change read()/stop()) --------
        self.frames_captured = 0
        self.reconnects = 0
        self.consecutive_read_failures = 0
        self.last_capture_ts: Optional[float] = None
        self.last_read_latency_ms: float = 0.0

        self.thread = threading.Thread(
            target=self._update,
            daemon=True,
            name=f"RTSP-{self.camera_id}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Non-invasive snapshot for the engine loop to log/diff against."""
        return {
            "frames_captured": self.frames_captured,
            "reconnects": self.reconnects,
            "consecutive_read_failures": self.consecutive_read_failures,
            "last_capture_ts": self.last_capture_ts,
            "last_read_latency_ms": self.last_read_latency_ms,
        }

    def _open(self):
        # stimeout is FFmpeg's own RTSP socket timeout, in MICROSECONDS,
        # read at open time by FFmpeg's RTSP demuxer — without it a
        # stalled handshake can hang the constructor call forever.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;tcp|stimeout;{config.RTSP_FFMPEG_STIMEOUT_US}"
        )

        new_cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.RTSP_OPEN_TIMEOUT_MS)
        except Exception:
            pass
        try:
            new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.RTSP_READ_TIMEOUT_MS)
        except Exception:
            pass
        try:
            # Keep only the latest frame — a design invariant (this is
            # what makes the reader "always give me the freshest frame,
            # never a stale queued one"), not an operational tunable, so
            # this one stays a literal.
            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        with self._lock:
            if self.stopped:
                try:
                    new_cap.release()
                except Exception:
                    pass
                return
            old_cap = self.cap
            self.cap = new_cap

        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass

        self.reconnects += 1
        self.logger.info(
            f"[RTSP] connected url={self.url} (reconnect #{self.reconnects})"
        )

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            try:
                with self._lock:
                    cap = self.cap
                    stopped = self.stopped
                if stopped:
                    break

                if cap is None or not cap.isOpened():
                    if self.stopped:
                        break
                    self.logger.warning("[RTSP] opening stream...")
                    self._open()
                    if self.stopped:
                        break
                    time.sleep(config.RTSP_RECONNECT_BACKOFF_SEC)
                    continue

                _read_t0 = time.time()
                ret, frame = cap.read()
                self.last_read_latency_ms = (time.time() - _read_t0) * 1000.0

                if self.stopped:
                    break

                if not ret or frame is None:
                    self.ret = False
                    self.frame = None
                    self.consecutive_read_failures += 1
                    n = self.consecutive_read_failures
                    if n in (1, 10, 50, 200) or n % 1000 == 0:
                        self.logger.warning(
                            f"[RTSP] read failed x{n} consecutive; reconnecting..."
                        )
                    if not self.stopped:
                        self._open()
                    time.sleep(config.RTSP_READ_FAIL_BACKOFF_SEC)
                    continue

                if self.consecutive_read_failures:
                    self.logger.info(
                        f"[RTSP] stream recovered after {self.consecutive_read_failures} failed reads"
                    )
                    self.consecutive_read_failures = 0

                self.ret = True
                self.frame = frame
                self.frames_captured += 1
                self.last_capture_ts = time.time()

            except Exception as e:
                self.ret = False
                self.frame = None
                if self.stopped:
                    break
                self.logger.warning(f"[RTSP] reader exception: {e}")
                time.sleep(config.RTSP_RECONNECT_BACKOFF_SEC)

    def read(self):
        if not self.ret or self.frame is None:
            return False, None
        return True, self.frame

    def stop(self):
        with self._lock:
            self.stopped = True
            cap = self.cap
            self.cap = None

        if cap is not None:
            try:
                cap.release()
                self.logger.info(
                    f"[RTSP] stopped, released capture "
                    f"(captured {self.frames_captured} frames, {self.reconnects} reconnects)"
                )
            except Exception:
                pass

        if self.thread.is_alive():
            self.thread.join(timeout=config.RTSP_READER_JOIN_TIMEOUT_SEC)

        self.ret = False
        self.frame = None


def encode_image(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return buf.tobytes()


def measure_sharpness(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
