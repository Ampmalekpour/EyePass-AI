"""
test_inference_backends.py
--------------------------------------------------------------------
The detector's model plumbing that needs no model runtime:
  * model_files.resolve_model — export_info.yaml manifest first, then
    file names; per-model folders; actionable errors
  * inference_backends.resolve_runtime — gpu/cpu/auto, model aliases,
    missing GPU
  * letterbox + onnx_postprocess (the benchmark's own-ONNX engine math)
  * the INT8 fix's overlap helper
  * perf_stats.EnginePerf accounting (missed / processed / latency)
Real-model accuracy and speed are tools/bench_multistream.py's job.
--------------------------------------------------------------------
"""

import logging
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

import numpy as np  # noqa: E402

import config  # noqa: E402
import inference_backends as ib  # noqa: E402
import model_files as mf  # noqa: E402
from perf_stats import EnginePerf  # noqa: E402
import capacity  # noqa: E402

LOG = logging.getLogger("test")


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _resolve(root, name, variant):
    return mf.resolve_model(root, name, variant, config.MODEL_MANIFEST_KEYS, config.MODEL_FILE_PATTERNS,
                            config.CLASS_LABELS)


class ModelFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.root = self.tmp.name
        d = os.path.join(r, "plate_v8n_480")
        _touch(os.path.join(d, "plate_v8n_480.pt"))
        _touch(os.path.join(d, "plate_v8n_480_288x480.onnx"))
        _touch(os.path.join(d, "plate_v8n_480_fp32_openvino_model", "plate_v8n_480.xml"))
        _touch(os.path.join(d, "plate_v8n_480_int8_box_openvino_model", "plate_v8n_480.xml"))
        _touch(os.path.join(d, "export_info.yaml"),
               "model_name: plate_v8n_480\nimgsz: [288, 480]\n"
               "names: {0: car_plate, 1: motorcycle_plate}\n"
               "pt: plate_v8n_480.pt\nonnx: plate_v8n_480_288x480.onnx\n"
               "openvino:\n  ov_fp32: plate_v8n_480_fp32_openvino_model\n"
               "  ov_int8_box: plate_v8n_480_int8_box_openvino_model\n")
        _touch(os.path.join(r, "plate_v8s_640", "plate_v8s_640.pt"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_variants(self):
        s = _resolve(self.root, "plate_v8n_480", "openvino_int8")
        self.assertTrue(s.path.endswith("plate_v8n_480_int8_box_openvino_model"))
        self.assertEqual(s.input_hw, (288, 480))
        self.assertEqual(s.names, {0: "car_plate", 1: "motorcycle_plate"})
        self.assertTrue(_resolve(self.root, "plate_v8n_480", "openvino_fp32").path.endswith("_fp32_openvino_model"))
        self.assertTrue(_resolve(self.root, "plate_v8n_480", "onnx").path.endswith("_288x480.onnx"))
        pt = _resolve(self.root, "plate_v8n_480", "pt")
        self.assertTrue(pt.path.endswith("plate_v8n_480.pt"))
        self.assertEqual(pt.imgsz, 480)

    def test_folder_without_manifest(self):
        s = _resolve(self.root, "plate_v8s_640", "pt")
        self.assertTrue(s.path.endswith(os.path.join("plate_v8s_640", "plate_v8s_640.pt")))
        self.assertEqual(s.imgsz, 640)
        self.assertIsNone(s.manifest)

    def test_missing_variant_is_actionable(self):
        with self.assertRaises(FileNotFoundError) as cm:
            _resolve(self.root, "plate_v8s_640", "openvino_fp32")
        self.assertIn("plate_v8s_640", str(cm.exception))

    def test_parsers(self):
        self.assertEqual(mf.parse_hw([288, 480]), (288, 480))
        self.assertEqual(mf.parse_hw("384x640"), (384, 640))
        self.assertEqual(mf.nominal_imgsz("plate_v8s_640"), 640)
        self.assertEqual(mf.parse_names(["a", "b"]), {0: "a", 1: "b"})


class RuntimePlanTest(unittest.TestCase):
    def setUp(self):
        self._orig = ib._cuda_available

    def tearDown(self):
        ib._cuda_available = self._orig

    def plan(self, device, gpu="plate_v8n_480", cpu="openvino_fp32", strict=False):
        return ib.resolve_runtime(device, gpu, cpu, config.CPU_MODEL_NAME, "cuda:0", strict,
                                  config.MODEL_ALIASES, LOG)

    def test_gpu(self):
        ib._cuda_available = lambda: True
        p = self.plan("gpu", gpu="v8s")
        self.assertEqual((p.device_kind, p.variant, p.model_name, p.torch_device),
                         ("gpu", "pt", "plate_v8s_640", "cuda:0"))
        self.assertEqual(self.plan("auto").device_kind, "gpu")

    def test_cpu_variants(self):
        ib._cuda_available = lambda: False
        for alias, want in (("openvino_fp32", "openvino_fp32"), ("int8", "openvino_int8"), ("onnx", "onnx")):
            p = self.plan("cpu", cpu=alias)
            self.assertEqual((p.device_kind, p.variant, p.model_name), ("cpu", want, config.CPU_MODEL_NAME))
        with self.assertRaises(ValueError):
            self.plan("cpu", cpu="tensorrt")

    def test_missing_gpu(self):
        ib._cuda_available = lambda: False
        self.assertEqual(self.plan("gpu").device_kind, "cpu")
        with self.assertRaises(RuntimeError):
            self.plan("gpu", strict=True)


class OnnxMathTest(unittest.TestCase):
    def test_letterbox(self):
        img, r, left, top = ib.letterbox(np.full((1080, 1920, 3), 7, np.uint8), (288, 480))
        self.assertEqual(img.shape, (288, 480, 3))
        self.assertAlmostEqual(r, 0.25)
        self.assertEqual((left, top), (0, 9))
        self.assertEqual(int(img[0, 0, 0]), 114)

    def test_postprocess(self):
        cands = np.array([
            [100, 109, 40, 20, 0.9, 0.1],   # kept, class 0
            [101, 109, 40, 20, 0.8, 0.1],   # same class, overlaps -> suppressed
            [100, 109, 40, 20, 0.1, 0.7],   # other class -> kept (class-aware)
            [300, 200, 30, 10, 0.1, 0.2],   # below conf
        ], dtype=np.float32)
        dets = ib.onnx_postprocess(cands.T[None], 0.25, 0, 9, (1080, 1920), 0.25, 0.7, 300)
        self.assertEqual(dets.shape, (2, 6))
        self.assertEqual(dets.dtype, np.float64)
        np.testing.assert_allclose(dets[0], [320, 360, 480, 440, 0.9, 0], rtol=1e-5)
        self.assertEqual(int(dets[1, 5]), 1)
        self.assertEqual(ib.onnx_postprocess(np.zeros((1, 6, 5), np.float32), 1, 0, 0, (10, 10),
                                             0.25, 0.7, 300).shape, (0, 6))

    def test_int8_overlaps(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], np.float32)
        iou, iomin = ib._overlaps(a, b)
        np.testing.assert_allclose(iou[0], [1.0, 0.25], rtol=1e-5)
        np.testing.assert_allclose(iomin[0], [1.0, 1.0], rtol=1e-5)


class PerfStatsTest(unittest.TestCase):
    def test_counts(self):
        p = EnginePerf(0, "cpu/onnx", 0.0, 2)
        p.frame_arrived("a", new_frames=3, missed=2)
        p.frame_done("a", {"preprocess": 1, "inference": 5, "postprocess": 1}, 2.0, 40.0, 1)
        p.frame_coasted("a")
        self.assertEqual((p.cams["a"].captured, p.cams["a"].missed, p.cams["a"].processed,
                          p.cams["a"].coasted), (3, 2, 1, 1))
        lines = []

        class L:
            def info(self, m):
                lines.append(m)
        p.maybe_log_cameras(L(), 1)
        self.assertIn("missed=2", lines[0])
        p.batch_done(1, 10, 12, [5], 2, 3, 1, 1)
        p.batch_done(1, 10, 12, [5], 0, 1, 1, 1)
        p.maybe_log_engine(L(), 1)
        self.assertIn("[STATS]", lines[-1])


class _FakeBackend:
    """infer(n frames) sleeps n * per_frame_ms (+ a contention penalty)."""
    label = "cpu/fake"
    imgsz_label = "288x480"

    class spec:
        variant = "openvino_fp32"

    def __init__(self, per_frame_ms):
        self.per_frame_ms, self.streams, self.resets = per_frame_ms, 0, 0

    def ensure_streams(self, n):
        self.streams = max(self.streams, n)

    def infer(self, frames):
        import time
        time.sleep(len(frames) * self.per_frame_ms / 1000.0)
        return [None] * len(frames), []


class CapacityTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(config, k) for k in (
            "CAPACITY_ROUNDS", "CAPACITY_WARMUP_ROUNDS", "CAPACITY_FRAME_SIZE", "CAPACITY_MAX_CAMERAS_TESTED",
            "REALTIME_MIN_FPS", "CAPACITY_SAFETY_MARGIN", "DETECT_EVERY_N_FRAMES", "CAPACITY_STOP_AFTER_FAILS")}
        config.CAPACITY_ROUNDS, config.CAPACITY_WARMUP_ROUNDS = 3, 0
        config.CAPACITY_FRAME_SIZE, config.CAPACITY_MAX_CAMERAS_TESTED = (32, 32), 8
        config.REALTIME_MIN_FPS, config.CAPACITY_SAFETY_MARGIN = 25.0, 0.7
        config.DETECT_EVERY_N_FRAMES, config.CAPACITY_STOP_AFTER_FAILS = 1, 2

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)

    def test_budget(self):
        self.assertAlmostEqual(capacity.budget_ms(), 28.0)
        config.DETECT_EVERY_N_FRAMES = 2
        self.assertAlmostEqual(capacity.budget_ms(), 56.0)

    def test_calibrate_finds_largest_fitting_count(self):
        # 10 ms per camera: 1 -> 10, 2 -> 20 (fit 28), 3 -> 30 (miss), 4 -> 40 (miss) -> stop
        prof = capacity.calibrate(_FakeBackend(10.0), LOG)
        self.assertEqual(prof["max_cameras"], 2)
        self.assertEqual([r["n"] for r in prof["table"]], [1, 2, 3, 4])
        self.assertEqual([r["ok"] for r in prof["table"]], [True, True, False, False])
        self.assertFalse(prof["max_is_lower_bound"])
        self.assertEqual(prof["budget_ms"], 28.0)

    def test_nothing_fits_and_everything_fits(self):
        self.assertEqual(capacity.calibrate(_FakeBackend(60.0), LOG)["max_cameras"], 0)
        prof = capacity.calibrate(_FakeBackend(0.5), LOG)
        self.assertEqual((prof["max_cameras"], prof["max_is_lower_bound"]), (8, True))

    def test_check_camera(self):
        prof = {"max_cameras": 4, "variant": "openvino_int8", "target_fps": 25.0}
        lvl, msg = capacity.check_camera(prof, 3, "c3")
        self.assertEqual(lvl, "info")
        self.assertIn("3/4", msg)
        lvl, msg = capacity.check_camera(prof, 5, "c5")
        self.assertEqual(lvl, "warning")
        self.assertIn("🚨", msg)
        self.assertEqual(capacity.check_camera(None, 5, "x"), (None, ""))
        lvl, _ = capacity.check_camera({**prof, "max_is_lower_bound": True}, 9, "c9")
        self.assertEqual(lvl, "info")


INT8_TABLE = {"table": [{"n": n, "p95_ms": p} for n, p in
                        [(1, 7.3), (2, 12.1), (3, 14.6), (4, 19.9), (5, 29.4), (6, 31.4)]]}


class CadenceTest(unittest.TestCase):
    """Auto-degrade: smallest detection interval N that fits the budget."""

    def setUp(self):
        self._saved = {k: getattr(config, k) for k in ("REALTIME_MIN_FPS", "CAPACITY_SAFETY_MARGIN")}
        config.REALTIME_MIN_FPS, config.CAPACITY_SAFETY_MARGIN = 25.0, 0.7

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)

    def test_loop_ms_measured_and_scaled(self):
        self.assertEqual(capacity.loop_ms(INT8_TABLE, 4), 19.9)
        self.assertAlmostEqual(capacity.loop_ms(INT8_TABLE, 8), 31.4 * 8 / 6)

    def test_pick(self):
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 4, 1, 4)[::2], (1, True))
        # 5 cameras: p95 29.4 > 28 (N=1) but <= 56 (N=2)
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 5, 1, 4)[::2], (2, True))
        # 8 cameras ~ 41.9 ms -> N=2 (56 ms)
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 8, 1, 4)[::2], (2, True))
        # 12 cameras ~ 62.8 ms -> N=3 (84 ms)
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 12, 1, 4)[::2], (3, True))
        # too many for the maximum N: capped, not fitting
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 40, 1, 4)[::2], (4, False))
        # a configured base N is never lowered
        self.assertEqual(capacity.pick_detect_every_n(INT8_TABLE, 2, 3, 4)[::2], (3, True))


if __name__ == "__main__":
    unittest.main()
