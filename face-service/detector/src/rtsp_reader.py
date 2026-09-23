"""
rtsp_reader.py
--------------------------------------------------------------------
RTSPStreamReader, unchanged from the reference video_processor.py: a
background thread that keeps grabbing frames from OpenCV so the main
detection loop never blocks on a slow/stalled network read, plus the
small encode/sharpness helpers used across the engine.

The URL handed to this reader is always the MediaMTX relay URL
(see backend_bridge._rtsp_url), never the camera's own RTSP address —
that stays true here exactly as in the reference pipeline.
--------------------------------------------------------------------
"""

import collections
import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np


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
        # stimeout is FFmpeg's own RTSP socket timeout, in MICROSECONDS.
        # Unlike CAP_PROP_OPEN_TIMEOUT_MSEC/CAP_PROP_READ_TIMEOUT_MSEC below
        # (which are set AFTER cv2.VideoCapture(...) already connects, so
        # they cannot bound that first connect at all), this is read by
        # FFmpeg's RTSP demuxer at open time and DOES bound it. Without it,
        # a stalled handshake (e.g. caught mid-reconnect on the MediaMTX
        # side) can hang the constructor call forever, with the reader
        # thread stuck and never retrying — no error, no log, nothing.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
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
        """Pops the oldest available frame from the queue (FIFO)."""
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