import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

try:
    import cv2
    import numpy as np
except ImportError:          # tooling-only environments
    cv2 = None

from platecore import imgcodec


@unittest.skipIf(cv2 is None, "opencv not installed")
class ImgCodecTests(unittest.TestCase):
    def test_jpeg_roundtrip_is_visually_lossless(self):
        rng = np.random.default_rng(0)
        img = cv2.GaussianBlur(rng.integers(0, 255, (120, 300, 3), dtype=np.uint8), (0, 0), 2)
        data = imgcodec.encode_jpeg(img)
        self.assertTrue(data[:2] == b"\xff\xd8")                 # JPEG magic
        dec = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(dec.shape, img.shape)
        mse = float(np.mean((dec.astype(np.float32) - img.astype(np.float32)) ** 2))
        psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
        self.assertGreater(psnr, 40.0)

    def test_empty_or_none_gives_none(self):
        self.assertIsNone(imgcodec.encode_jpeg(None))
        self.assertIsNone(imgcodec.encode_jpeg(np.zeros((0, 0, 3), np.uint8)))

    def test_params_request_444_when_available(self):
        p = imgcodec.jpeg_params(90)
        self.assertEqual(p[:2], [cv2.IMWRITE_JPEG_QUALITY, 90])
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
            self.assertIn(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444, p)


if __name__ == "__main__":
    unittest.main()
