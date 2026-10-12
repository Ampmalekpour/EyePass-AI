"""OCR image storage: key layout, WebP compression, the background uploader."""
import logging
import os
import sys
import threading
import time
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ocr_service", "src"))

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

import image_store  # noqa: E402

LOG = logging.getLogger("test_image_store")


def _jpeg(h=60, w=160):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class KeyTests(unittest.TestCase):
    NOW = datetime(2026, 10, 12, 9, 8, 7, 123456)

    def test_valid_layout(self):
        p, f = image_store.build_image_keys(3, True, ".webp", self.NOW)
        self.assertEqual(p, "dynamics/ValidPlateImage/vp_3_2026-10-12_09-08-07-123456.webp")
        self.assertEqual(f, "dynamics/ValidVehicleImage/vf_3_2026-10-12_09-08-07-123456.webp")

    def test_invalid_layout(self):
        p, f = image_store.build_image_keys("cam9", False, ".webp", self.NOW)
        self.assertTrue(p.startswith("dynamics/InvalidPlateImage/ip_cam9_"))
        self.assertTrue(f.startswith("dynamics/InvalidVehicleImage/if_cam9_"))


@unittest.skipIf(cv2 is None, "opencv not installed")
class WebpTests(unittest.TestCase):
    def test_jpeg_becomes_webp(self):
        out = image_store.to_webp(_jpeg(), 70, 4)
        self.assertEqual(out[:4], b"RIFF")
        self.assertEqual(out[8:12], b"WEBP")
        dec = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(dec.shape, (60, 160, 3))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            image_store.to_webp(b"not an image")


@unittest.skipIf(cv2 is None, "opencv not installed")
class UploaderTests(unittest.TestCase):
    def _make(self, upload, **kw):
        kw.setdefault("backoff_sec", 0.01)
        return image_store.ImageUploader(upload, LOG, **kw).start()

    def test_uploads_webp_in_the_background(self):
        got = {}
        up = self._make(lambda data, key: got.__setitem__(key, data))
        self.assertTrue(up.submit("a.webp", _jpeg(), "crop"))
        self.assertEqual(up.stop(5), 0)
        self.assertEqual(got["a.webp"][8:12], b"WEBP")
        self.assertEqual(up.stats["uploaded"], 1)

    def test_submit_does_not_wait_for_a_slow_upload(self):
        gate = threading.Event()
        up = self._make(lambda d, k: gate.wait(5))
        t0 = time.time()
        for i in range(3):
            up.submit(f"k{i}", _jpeg(), "crop")
        self.assertLess(time.time() - t0, 0.5)
        gate.set()
        up.stop(5)

    def test_retry_then_success(self):
        calls = []

        def flaky(data, key):
            calls.append(key)
            if len(calls) < 3:
                raise RuntimeError("minio down")

        up = self._make(flaky, retries=3)
        up.submit("r.webp", _jpeg(), "frame")
        up.stop(5)
        self.assertEqual(len(calls), 3)
        self.assertEqual((up.stats["uploaded"], up.stats["failed"]), (1, 0))

    def test_retries_exhausted_is_counted_failed(self):
        def broken(data, key):
            raise RuntimeError("down")

        up = self._make(broken, retries=2)
        up.submit("x.webp", _jpeg(), "frame")
        up.stop(5)
        self.assertEqual((up.stats["uploaded"], up.stats["failed"]), (0, 1))

    def test_queue_full_drops_and_counts(self):
        gate = threading.Event()
        up = self._make(lambda d, k: gate.wait(5), queue_max=2)
        results = [up.submit(f"k{i}", _jpeg(), "crop") for i in range(8)]
        self.assertIn(False, results)
        self.assertGreater(up.stats["dropped"], 0)
        gate.set()
        up.stop(5)

    def test_undecodable_bytes_count_as_failed_not_crash(self):
        up = self._make(lambda d, k: None)
        up.submit("bad.webp", b"garbage", "crop")
        up.stop(5)
        self.assertEqual(up.stats["failed"], 1)

    def test_stop_drains_what_is_queued(self):
        got = []
        up = self._make(lambda d, k: got.append(k))
        for i in range(5):
            up.submit(f"k{i}", _jpeg(), "crop")
        self.assertEqual(up.stop(10), 0)
        self.assertEqual(len(got), 5)

    def test_empty_data_is_not_queued(self):
        up = self._make(lambda d, k: None)
        self.assertFalse(up.submit("e.webp", b"", "crop"))
        up.stop(1)


if __name__ == "__main__":
    unittest.main()
