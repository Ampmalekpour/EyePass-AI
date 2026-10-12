"""Polygon ROI: bounding-box crop + grey mask, the camera-config plumbing, the perf timer."""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

import config  # noqa: E402
import engine as engine_mod  # noqa: E402
from backend_bridge import build_camera_job  # noqa: E402
from perf_stats import EnginePerf  # noqa: E402

GREY = config.ROI_GREY_VALUE


def _frame(h=200, w=400):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = (10, 20, 30)
    return f


class RoiStateTests(unittest.TestCase):
    def test_polygon_crops_bbox_and_greys_outside(self):
        f = _frame()
        # trapezoid in pixels: top 150..250 at y=20, bottom 50..350 at y=180
        poly = [(150, 20), (250, 20), (350, 180), (50, 180)]
        st = engine_mod._build_roi_state(f.shape, (0, 0, 1, 1), poly, GREY)
        self.assertEqual(st["bbox"], (50, 20, 351, 181))          # the polygon's bounding box
        img = engine_mod._roi_image(f, st)
        self.assertEqual(img.shape, (161, 301, 3))
        self.assertTrue(np.all(img[0, 0] == GREY))                # top-left corner is outside the polygon
        self.assertTrue(np.all(img[0, 300] == GREY))              # top-right too
        self.assertTrue(np.all(img[80, 150] == (10, 20, 30)))     # the middle is the video
        self.assertEqual(st["poly"].shape, (4, 2))

    def test_grey_buffer_is_refreshed_every_frame(self):
        poly = [(150, 20), (250, 20), (350, 180), (50, 180)]
        st = engine_mod._build_roi_state((200, 400, 3), (0, 0, 1, 1), poly, GREY)
        a = _frame(); b = _frame(); b[:] = (99, 98, 97)
        engine_mod._roi_image(a, st)
        img = engine_mod._roi_image(b, st)
        self.assertTrue(np.all(img[80, 150] == (99, 98, 97)))
        self.assertTrue(np.all(img[0, 0] == GREY))

    def test_fraction_polygon_is_read_as_fractions(self):
        st = engine_mod._build_roi_state((200, 400, 3), (0, 0, 1, 1),
                                         [(0.25, 0.1), (0.75, 0.1), (0.75, 0.9), (0.25, 0.9)], GREY)
        x1, y1, x2, y2 = st["bbox"]
        self.assertAlmostEqual(x1, 100, delta=1); self.assertAlmostEqual(x2, 300, delta=2)
        self.assertAlmostEqual(y1, 20, delta=1); self.assertAlmostEqual(y2, 180, delta=2)
        self.assertIsNone(st["mask"])        # a rectangle polygon fills its bbox -> no grey needed

    def test_no_polygon_is_the_old_rectangle_view(self):
        f = _frame()
        st = engine_mod._build_roi_state(f.shape, (0.25, 0.5, 0.5, 0.5), None, GREY)
        img = engine_mod._roi_image(f, st)
        self.assertTrue(np.shares_memory(img, f))                 # still a view, no copy
        self.assertEqual(st["bbox"], (100, 100, 300, 200))
        self.assertIsNone(st["poly"])

    def test_unusable_polygon_falls_back_to_rectangle(self):
        st = engine_mod._build_roi_state((200, 400, 3), (0, 0, 0.5, 1.0), [(1, 1), (2, 2)], GREY)
        self.assertEqual(st["bbox"], (0, 0, 200, 200))

    def test_invalid_rectangle_uses_the_whole_frame(self):
        st = engine_mod._build_roi_state((200, 400, 3), (0.5, 0.5, 0.0, 0.0), None, GREY)
        self.assertEqual(st["bbox"], (0, 0, 400, 200))

    def test_polygon_points_outside_the_frame_are_clipped(self):
        st = engine_mod._build_roi_state((200, 400, 3), (0, 0, 1, 1), [(-50, -10), (500, -10), (500, 300), (-50, 300)], GREY)
        x1, y1, x2, y2 = st["bbox"]
        self.assertTrue(0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 200)


class _FakeHub:
    boot_id = "b"

    def __init__(self, *a, **k):
        self.events, self.tasks = [], []

    def emit(self, kind, data):
        self.events.append((kind, dict(data)))

    def submit(self, task):
        self.tasks.append(task)

    def drain(self):
        return []

    def stop(self, *a, **k):
        pass


class _FakeReader:
    def __init__(self, frame):
        self.frame, self.n = frame, 0

    def start(self):
        return self

    def read_latest(self):
        self.n += 1
        return True, self.frame.copy(), time.time(), self.n

    def get_stats(self):
        return {}

    def stop(self):
        pass


class _FakeBackend:
    label, kind, device, imgsz_label = "gpu/pt-fp32", "pt", "cuda:0", 480

    def __init__(self, stop_event, det, stop_after=6):
        self.stop_event, self.det, self.stop_after = stop_event, det, stop_after
        self.seen = []

    def ensure_streams(self, n):
        pass

    def warmup(self):
        pass

    def reset_streams(self):
        pass

    def close(self):
        pass

    def infer(self, frames):
        self.seen.append([f.copy() for f in frames])
        if len(self.seen) >= self.stop_after:
            self.stop_event.set()
        d = np.array([self.det] * len(frames) if False else self.det, np.float64).reshape(-1, 6)
        return [d.copy() for _ in frames], [{"preprocess": 0.1, "inference": 0.2, "postprocess": 0.1}] * len(frames)


class EngineLoopPolygonTests(unittest.TestCase):
    """The real Engine.run() loop with a fake reader / model: what the model receives, where
    crops and the best frame come from."""

    def _run(self, polygon, det):
        import multiprocessing as mp
        import types
        stop = mp.Event()
        patches = {
            "RedisBus": lambda module=None: object(), "HubClient": _FakeHub,
            "RTSPStreamReader": lambda url, cid: _FakeReader(self.frame),
        }
        saved = {k: getattr(engine_mod, k) for k in patches}
        saved_rt, saved_bb = engine_mod.resolve_runtime, engine_mod.inference_backends.build_backend
        backend = _FakeBackend(stop, det)
        for k, v in patches.items():
            setattr(engine_mod, k, v)
        engine_mod.resolve_runtime = lambda logger: types.SimpleNamespace(describe=lambda: "fake")
        engine_mod.inference_backends.build_backend = lambda *a, **k: backend
        try:
            eng = engine_mod.Engine(0, "", 0, 0.25, False, type("Q", (), {"put_nowait": lambda s, m: None})(),
                                    type("C", (), {"get_nowait": lambda s: (_ for _ in ()).throw(Exception())})(),
                                    stop, "/tmp", {0: "Car"})
            eng.add_camera("c1", "rtsp://x", (0, 0, 1, 1), roi_polygon=polygon)
            eng.cleanup = lambda: None          # keep the camera state for the assertions
            eng.run()
        finally:
            for k, v in saved.items():
                setattr(engine_mod, k, v)
            engine_mod.resolve_runtime = saved_rt
            engine_mod.inference_backends.build_backend = saved_bb
        return eng, backend

    def setUp(self):
        # a 1280x720 frame; a bright plate-like block at full-frame (700..820, 400..440)
        self.frame = np.full((720, 1280, 3), (10, 20, 30), np.uint8)
        self.frame[400:440, 700:820] = (200, 210, 220)

    def test_model_gets_greyed_bbox_and_crop_comes_from_the_full_frame(self):
        poly = [(500, 100), (900, 100), (1100, 600), (300, 600)]       # pixels -> bbox x 300..1100, y 100..600
        # detection in MODEL-IMAGE pixels: plate at full (700..820, 400..440) -> bbox offset (300, 100)
        eng, backend = self._run(poly, [400, 300, 520, 340, 0.95, 0])
        model_img = backend.seen[0][0]
        self.assertEqual(model_img.shape, (501, 801, 3))                 # the polygon's bounding box (corners inclusive)
        self.assertTrue(np.all(model_img[0, 0] == config.ROI_GREY_VALUE))        # outside the polygon -> grey
        self.assertTrue(np.all(model_img[250, 400] == (10, 20, 30)))              # inside -> the video
        cam = eng.cameras["c1"]
        crops = [c for lst in cam["best_crops"].values() for c in lst]
        self.assertTrue(crops, "a track with a crop was expected")
        self.assertEqual(crops[0]["image"].shape, (40, 120, 3))
        self.assertTrue(np.all(crops[0]["image"] == (200, 210, 220)))             # the real pixels, found via the offset
        bf = [b for b in cam["best_frame"].values()][0]
        self.assertEqual(bf["frame"].shape, (720, 1280, 3))                       # the FULL frame, not the ROI
        self.assertEqual(bf["bbox"], (700, 400, 820, 440))                        # box in full-frame pixels

    def test_rectangle_camera_still_gets_a_view_of_the_roi(self):
        eng, backend = self._run(None, [100, 100, 220, 140, 0.95, 0])
        self.assertEqual(backend.seen[0][0].shape, (720, 1280, 3))
        crops = [c for lst in eng.cameras["c1"]["best_crops"].values() for c in lst]
        self.assertTrue(crops)
        bf = [b for b in eng.cameras["c1"]["best_frame"].values()][0]
        self.assertEqual(bf["frame"].shape, (720, 1280, 3))


class CameraJobPolygonTests(unittest.TestCase):
    BASE = {"address": "x", "roi": {"x": 0, "y": 0, "w": 1, "h": 1}}

    def test_dict_points(self):
        job = build_camera_job("1", dict(self.BASE, roi_polygon=[{"x": .1, "y": .1}, {"x": .9, "y": .1}, {"x": .9, "y": .9}]))
        self.assertEqual(job.roi_polygon, ((.1, .1), (.9, .1), (.9, .9)))

    def test_pair_points(self):
        job = build_camera_job("1", dict(self.BASE, roi_polygon=[[1, 2], [3, 4], [5, 6], [7, 8]]))
        self.assertEqual(len(job.roi_polygon), 4)

    def test_absent_or_bad_polygon_is_none(self):
        self.assertIsNone(build_camera_job("1", self.BASE).roi_polygon)
        self.assertIsNone(build_camera_job("1", dict(self.BASE, roi_polygon=[[1, 2], [3, 4]])).roi_polygon)
        self.assertIsNone(build_camera_job("1", dict(self.BASE, roi_polygon="nonsense")).roi_polygon)


class PerfRoiTimerTests(unittest.TestCase):
    def test_roi_time_is_in_the_perf_line(self):
        lines = []

        class L:
            def info(self, m, *a, **k):
                lines.append(m)

        p = EnginePerf(0, "t", 0.0, 50)
        p.frame_arrived("1", 1, 0)
        p.roi_done("1", 0.25)
        p.frame_done("1", {"preprocess": 1, "inference": 2, "postprocess": 3}, 1.0, 5.0, 0)
        p.maybe_log_cameras(L(), 1)
        self.assertTrue(any("roi(crop+mask)=0.25" in m for m in lines), lines)


class FrameQualityTests(unittest.TestCase):
    def test_frame_q80_is_smaller_than_default(self):
        from rtsp_reader import encode_image
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
        self.assertLess(len(encode_image(img, 80)), len(encode_image(img)))
        self.assertEqual(config.FRAME_JPEG_QUALITY, 80)


if __name__ == "__main__":
    unittest.main()
