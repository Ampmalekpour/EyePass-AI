"""
image_store.py (ocr_service)
--------------------------------------------------------------------
How the OCR service stores the images of a recognition in MinIO.

  1. The object KEYS are defined right after the recognition (build_image_keys)
     - valid -> vp/vf, invalid -> ip/if, under dynamics/<Valid|Invalid>...Image/ -
     and go into the result that is sent to the control hub at once.
  2. The pictures themselves are compressed to WebP and uploaded by a
     background thread (ImageUploader), so neither the compression nor the
     upload delays the recognition answer.

What arrives from the detector is JPEG bytes (crops as the detector made
them, the FULL frame at JPEG q80); the uploader decodes them and stores them
as WebP (default q70).

This module does not import config / paddle / boto3, so it can be tested
alone; the worker passes the settings and the upload function in.
--------------------------------------------------------------------
"""
from __future__ import annotations

import io
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

try:                                   # Pillow gives control over the WebP effort (method)
    from PIL import Image
except Exception:                      # pragma: no cover - OpenCV fallback below
    Image = None


# --------------------------------------------------------------------
# keys
# --------------------------------------------------------------------
def build_image_keys(stream_idx, is_valid: bool, ext: str = ".webp",
                     now: Optional[datetime] = None) -> Tuple[str, str]:
    """(plate_key, frame_key). The layout is the one the backend already resolves
    (valid -> vp/vf, invalid -> ip/if); only the extension changed."""
    now = now or datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S-%f")
    cam = str(stream_idx)
    if is_valid:
        plate_folder, frame_folder, pp, fp = "dynamics/ValidPlateImage", "dynamics/ValidVehicleImage", "vp", "vf"
    else:
        plate_folder, frame_folder, pp, fp = "dynamics/InvalidPlateImage", "dynamics/InvalidVehicleImage", "ip", "if"
    return (f"{plate_folder}/{pp}_{cam}_{date_str}_{time_str}{ext}",
            f"{frame_folder}/{fp}_{cam}_{date_str}_{time_str}{ext}")


# --------------------------------------------------------------------
# compression
# --------------------------------------------------------------------
def to_webp(data: bytes, quality: int = 70, method: int = 4) -> bytes:
    """Any image bytes cv2 can decode (the detector's JPEG) -> WebP bytes."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("image bytes could not be decoded")
    if Image is not None:
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "WEBP", quality=int(quality), method=int(method))
        return buf.getvalue()
    ok, enc = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    if not ok:
        raise ValueError("webp encode failed")
    return enc.tobytes()


# --------------------------------------------------------------------
# background uploader
# --------------------------------------------------------------------
class ImageUploader:
    """Bounded queue + worker thread(s): bytes -> WebP -> upload, with retries.

    submit() never blocks: when the queue is full the job is dropped (WARNING +
    counter), because recognition must not wait for images. stop() drains what is
    queued, for at most `timeout` seconds."""

    def __init__(self, upload_fn: Callable[[bytes, str], None], logger, *, quality: int = 70, method: int = 4,
                 queue_max: int = 200, retries: int = 3, backoff_sec: float = 0.5, threads: int = 1,
                 content_type: str = "image/webp", name: str = "img-upload"):
        self._upload = upload_fn
        self.log = logger
        self.quality, self.method = quality, method
        self.retries, self.backoff = max(1, int(retries)), float(backoff_sec)
        self.content_type = content_type
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(queue_max)))
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, daemon=True, name=f"{name}-{i}")
                         for i in range(max(1, int(threads)))]
        self.stats: Dict[str, float] = {"submitted": 0, "uploaded": 0, "failed": 0, "dropped": 0,
                                        "encode_ms": 0.0, "upload_ms": 0.0}
        self._lock = threading.Lock()

    def start(self) -> "ImageUploader":
        for t in self._threads:
            t.start()
        return self

    def submit(self, key: str, data: bytes, tag: str = "") -> bool:
        if not data:
            return False
        try:
            self._q.put_nowait((key, data, tag))
        except queue.Full:
            with self._lock:
                self.stats["dropped"] += 1
            self.log.warning("[UPLOAD] queue full (%d) - dropped %s %s (its path will not exist in MinIO)",
                             self._q.maxsize, tag, key)
            return False
        with self._lock:
            self.stats["submitted"] += 1
        return True

    def pending(self) -> int:
        return self._q.qsize()

    def snapshot_and_reset(self) -> Dict[str, float]:
        with self._lock:
            s = dict(self.stats)
            for k in self.stats:
                self.stats[k] = 0 if k not in ("encode_ms", "upload_ms") else 0.0
        s["pending"] = self.pending()
        return s

    def _process(self, key: str, data: bytes, tag: str):
        t0 = time.perf_counter()
        try:
            webp = to_webp(data, self.quality, self.method)
        except Exception as e:
            with self._lock:
                self.stats["failed"] += 1
            self.log.error("[UPLOAD] %s %s: compression failed: %s", tag, key, e)
            return
        t1 = time.perf_counter()
        err = None
        for attempt in range(1, self.retries + 1):
            try:
                self._upload(webp, key)
                err = None
                break
            except Exception as e:
                err = e
                if attempt < self.retries and not self._stop.is_set():
                    time.sleep(self.backoff * attempt)
        t2 = time.perf_counter()
        with self._lock:
            self.stats["encode_ms"] += (t1 - t0) * 1000.0
            self.stats["upload_ms"] += (t2 - t1) * 1000.0
            self.stats["uploaded" if err is None else "failed"] += 1
        if err is not None:
            self.log.error("[UPLOAD] %s %s FAILED after %d attempt(s): %s", tag, key, self.retries, err)

    def _loop(self):
        while True:
            try:
                key, data, tag = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self._process(key, data, tag)
            finally:
                self._q.task_done()

    def stop(self, timeout: float = 25.0) -> int:
        """Finish the queued uploads (up to `timeout` s), then stop. Returns how many were left."""
        end = time.time() + max(0.0, timeout)
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.05)
        left = self._q.unfinished_tasks
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        if left:
            self.log.error("[UPLOAD] %d image(s) were not uploaded before shutdown", left)
        return left
