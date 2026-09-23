"""
test_debugging.py
--------------------------------------------------------------------
Exercises common/facecore/debugging.py directly — no Redis, no torch,
no face_alignment; only cv2/numpy, both available in this sandbox.
Mirrors tests/test_pose.py's "import the real module, feed it
synthetic input" approach.

Run with:
    PYTHONPATH=common python3 tests/test_debugging.py
--------------------------------------------------------------------
"""

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

import numpy as np  # noqa: E402

from facecore import debugging  # noqa: E402


class EnvParsingTests(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_env_bool_accepts_common_truthy_spellings(self):
        for val in ("1", "true", "True", "yes", "on", "y"):
            os.environ["X_FLAG"] = val
            self.assertTrue(debugging.env_bool("X_FLAG", False), f"{val!r} should be truthy")

    def test_env_bool_falls_back_to_default_when_unset(self):
        os.environ.pop("X_FLAG_UNSET", None)
        self.assertTrue(debugging.env_bool("X_FLAG_UNSET", True))
        self.assertFalse(debugging.env_bool("X_FLAG_UNSET", False))

    def test_env_int_and_env_float_fall_back_on_garbage(self):
        os.environ["X_INT"] = "not-a-number"
        os.environ["X_FLOAT"] = "also-not-a-number"
        self.assertEqual(debugging.env_int("X_INT", 7), 7)
        self.assertEqual(debugging.env_float("X_FLOAT", 1.5), 1.5)


class StopwatchTests(unittest.TestCase):
    def test_measures_a_positive_elapsed_time(self):
        with debugging.Stopwatch() as sw:
            time.sleep(0.01)
        self.assertGreater(sw.elapsed_ms, 0.0)

    def test_reraises_the_caller_exception(self):
        with self.assertRaises(ValueError):
            with debugging.Stopwatch():
                raise ValueError("boom")


class EnsureDirTests(unittest.TestCase):
    def test_creates_and_confirms_a_writable_directory(self):
        tmp = tempfile.mkdtemp()
        try:
            target = os.path.join(tmp, "a", "b", "c")
            self.assertTrue(debugging.ensure_dir(target))
            self.assertTrue(os.path.isdir(target))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_returns_false_instead_of_raising_when_path_is_unusable(self):
        # A path through a FILE (not a directory) can never be mkdir'd.
        tmp = tempfile.mkdtemp()
        try:
            blocker = os.path.join(tmp, "im_a_file")
            with open(blocker, "w") as f:
                f.write("x")
            target = os.path.join(blocker, "subdir")
            self.assertFalse(debugging.ensure_dir(target))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PruneDirTests(unittest.TestCase):
    def test_keeps_only_the_n_most_recently_modified_files(self):
        tmp = tempfile.mkdtemp()
        try:
            paths = []
            for i in range(5):
                p = os.path.join(tmp, f"f{i}.jpg")
                with open(p, "w") as f:
                    f.write("x")
                os.utime(p, (i, i))  # deterministic mtimes, oldest first
                paths.append(p)

            debugging.prune_dir(tmp, max_files=2, suffixes=(".jpg",))

            remaining = sorted(os.listdir(tmp))
            self.assertEqual(remaining, ["f3.jpg", "f4.jpg"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_zero_or_negative_max_files_never_deletes_anything(self):
        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, "f0.jpg"), "w") as f:
                f.write("x")
            debugging.prune_dir(tmp, max_files=0, suffixes=(".jpg",))
            self.assertEqual(os.listdir(tmp), ["f0.jpg"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SaveMontageTests(unittest.TestCase):
    def test_writes_a_readable_image_file(self):
        import cv2
        tmp = tempfile.mkdtemp()
        try:
            img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            out_path = os.path.join(tmp, "montage.jpg")
            ok = debugging.save_montage(
                [(img, ["cell one"]), (None, ["no image here"])],
                out_path, cell_size=(50, 50), columns=2, title="test montage",
            )
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(out_path))
            readback = cv2.imread(out_path)
            self.assertIsNotNone(readback)
            self.assertGreater(readback.shape[0], 0)
            self.assertGreater(readback.shape[1], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_cells_list_returns_false_without_raising(self):
        tmp = tempfile.mkdtemp()
        try:
            out_path = os.path.join(tmp, "montage.jpg")
            ok = debugging.save_montage([], out_path)
            self.assertFalse(ok)
            self.assertFalse(os.path.isfile(out_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
