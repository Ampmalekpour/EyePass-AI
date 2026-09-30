"""
test_inference_backends.py
--------------------------------------------------------------------
The CPU-pipeline plumbing that needs no model runtime: model file
resolution (model_files.py), GPU/CPU runtime planning, the
Ultralytics-exact letterbox / auto-shape / postprocess math
(inference_backends.py) and the OpenVINO stream/thread plan
(cpu_topology.openvino_plan). openvino / onnxruntime / torch are not
needed — parity against real models is tools/parity_test.py's job.
--------------------------------------------------------------------
"""

import logging
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

import numpy as np  # noqa: E402

import config  # noqa: E402
import cpu_topology  # noqa: E402
import inference_backends as ib  # noqa: E402
import model_files as mf  # noqa: E402

LOG = logging.getLogger("test")


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class ModelFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.root = self.tmp.name
        _touch(os.path.join(r, "plate_v8n_480.pt"))
        _touch(os.path.join(r, "plate_v8n_480_288x480.onnx"))
        _touch(os.path.join(r, "plate_v8n_480_fp32_openvino_model", "plate_v8n_480.xml"))
        _touch(os.path.join(r, "plate_v8n_480_fp32_openvino_model", "metadata.yaml"),
               "imgsz:\n- 288\n- 480\nnames:\n  0: car_plate\n  1: motorcycle_plate\n")
        _touch(os.path.join(r, "plate_v8n_480_int8_openvino_model", "plate_v8n_480.xml"))
        _touch(os.path.join(r, "export_info.yaml"),
               "input_size: [288, 480]\nclass_names: [car_plate, motorcycle_plate]\n")
        _touch(os.path.join(r, "plate_v8s_640.pt"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_nominal_size_and_parsers(self):
        self.assertEqual(mf.nominal_imgsz("plate_v8n_480"), 480)
        self.assertEqual(mf.nominal_imgsz("/m/plate_v8s_640.pt"), 640)
        self.assertIsNone(mf.nominal_imgsz("best"))
        self.assertEqual(mf.parse_hw("288x480"), (288, 480))
        self.assertEqual(mf.parse_hw([1, 3, 384, 640]), (384, 640))
        self.assertEqual(mf.parse_hw({"height": 288, "width": 480}), (288, 480))
        self.assertEqual(mf.parse_hw(640), (640, 640))
        self.assertEqual(mf.parse_names("{0: 'a', 1: 'b'}"), {0: "a", 1: "b"})

    def test_pt_follows_model_name(self):
        s = mf.resolve_model(self.root, "plate_v8n_480", "pt")
        self.assertTrue(s.path.endswith("plate_v8n_480.pt"))
        self.assertEqual(s.imgsz, 480)
        s = mf.resolve_model(self.root, "plate_v8s_640", "pt")
        self.assertEqual(s.imgsz, 640)
        # explicit DETECTION_IMG_SIZE still wins
        self.assertEqual(mf.resolve_model(self.root, "plate_v8s_640", "pt", imgsz_override=512).imgsz, 512)

    def test_openvino_fp32_and_int8(self):
        s = mf.resolve_model(self.root, "plate_v8n_480", "openvino", "fp32")
        self.assertTrue(s.path.endswith(os.path.join("plate_v8n_480_fp32_openvino_model", "plate_v8n_480.xml")))
        self.assertEqual(s.input_hw, (288, 480))
        self.assertEqual(s.names, {0: "car_plate", 1: "motorcycle_plate"})
        s8 = mf.resolve_model(self.root, "plate_v8n_480", "openvino", "int8")
        self.assertIn("int8_openvino_model", s8.path)
        # int8 folder has no metadata.yaml -> shared export_info.yaml (matches 480)
        self.assertEqual(s8.input_hw, (288, 480))
        self.assertEqual(s8.names, {0: "car_plate", 1: "motorcycle_plate"})

    def test_onnx_picks_shaped_file(self):
        s = mf.resolve_model(self.root, "plate_v8n_480", "onnx")
        self.assertTrue(s.path.endswith("plate_v8n_480_288x480.onnx"))
        self.assertEqual(s.input_hw, (288, 480))

    def test_shared_export_info_not_applied_to_other_model(self):
        _touch(os.path.join(self.root, "plate_v8s_640_384x640.onnx"))
        s = mf.resolve_model(self.root, "plate_v8s_640", "onnx")
        # export_info.yaml describes the 480 model -> ignored; file name used
        self.assertEqual(s.input_hw, (384, 640))
        self.assertEqual(s.sources["input_hw"], "file name")

    def test_missing_variant_is_actionable(self):
        with self.assertRaises(FileNotFoundError) as cm:
            mf.resolve_model(self.root, "plate_v8s_640", "openvino")
        self.assertIn("plate_v8s_640_fp32_openvino_model", str(cm.exception))
        with self.assertRaises(ValueError):
            mf.resolve_model(self.root, "plate_v8n_480", "onnx", "int8")


class RuntimePlanTest(unittest.TestCase):
    def setUp(self):
        self._orig = ib._cuda_available

    def tearDown(self):
        ib._cuda_available = self._orig

    def test_gpu_always_pt(self):
        ib._cuda_available = lambda: True
        p = ib.resolve_runtime("gpu", "openvino", "int8", False, LOG)
        self.assertEqual((p.device_kind, p.backend, p.torch_device, p.precision), ("gpu", "pt", "cuda", "fp32"))
        self.assertEqual(ib.resolve_runtime("cuda:1", "", "fp32", False, LOG).torch_device, "cuda:1")
        self.assertEqual(ib.resolve_runtime("auto", "", "fp32", False, LOG).device_kind, "gpu")

    def test_cpu_defaults_to_openvino(self):
        ib._cuda_available = lambda: False
        p = ib.resolve_runtime("cpu", "", "fp32", False, LOG)
        self.assertEqual((p.device_kind, p.backend, p.torch_device), ("cpu", "openvino", "cpu"))
        self.assertEqual(ib.resolve_runtime("auto", "onnx", "int8", False, LOG).precision, "fp32")

    def test_missing_gpu(self):
        ib._cuda_available = lambda: False
        self.assertEqual(ib.resolve_runtime("gpu", "", "fp32", False, LOG).device_kind, "cpu")
        with self.assertRaises(RuntimeError):
            ib.resolve_runtime("gpu", "", "fp32", True, LOG)


class PrePostProcessTest(unittest.TestCase):
    def test_auto_shape_matches_ultralytics(self):
        self.assertEqual(ib.auto_shape((1080, 1920), 480)[0], (288, 480))
        self.assertEqual(ib.auto_shape((1080, 1920), 640)[0], (384, 640))
        self.assertEqual(ib.auto_shape((1080, 960), 480)[0], (480, 448))
        self.assertEqual(ib.auto_shape((540, 1920), 480)[0], (160, 480))

    def test_letterbox_centred_114(self):
        img = np.full((1080, 1920, 3), 7, np.uint8)
        blob, gain, pad = ib.preprocess(img, (288, 480))
        self.assertEqual(blob.shape, (1, 3, 288, 480))
        self.assertEqual(blob.dtype, np.float32)
        self.assertAlmostEqual(gain, 0.25)
        self.assertEqual(pad, (0, 9))                 # 270 rows centred in 288
        self.assertAlmostEqual(float(blob[0, 0, 0, 0]), 114 / 255, places=5)
        self.assertAlmostEqual(float(blob[0, 0, 100, 100]), 7 / 255, places=5)

    def test_postprocess_nms_classes_and_mapping(self):
        # 3 candidates in a 288x480 input for a 1080x1920 frame (gain .25, pad (0, 9)):
        #  a: class 0 conf .9 ; b: overlaps a, class 0 conf .8 -> suppressed
        #  c: same box as a but class 1 conf .7 -> kept (class-aware NMS)
        #  d: below conf
        cands = np.array([
            # cx,  cy,  w,  h,  s0,  s1
            [100, 109, 40, 20, 0.9, 0.1],
            [101, 109, 40, 20, 0.8, 0.1],
            [100, 109, 40, 20, 0.1, 0.7],
            [300, 200, 30, 10, 0.1, 0.2],
        ], dtype=np.float32)
        raw = cands.T[None]                            # (1, 6, 4)
        dets = ib.postprocess(raw, 0.25, (0, 9), (1080, 1920), conf_thres=0.25)
        self.assertEqual(dets.dtype, np.float64)
        self.assertEqual(dets.shape, (2, 6))
        np.testing.assert_allclose(dets[0], [320, 360, 480, 440, 0.9, 0], rtol=1e-5)
        self.assertEqual(int(dets[1, 5]), 1)
        agnostic = ib.postprocess(raw, 0.25, (0, 9), (1080, 1920), conf_thres=0.25, agnostic=True)
        self.assertEqual(agnostic.shape[0], 1)

    def test_postprocess_empty(self):
        raw = np.zeros((1, 6, 10), np.float32)
        self.assertEqual(ib.postprocess(raw, 1.0, (0, 0), (100, 100), 0.25).shape, (0, 6))


class OpenVINOPlanTest(unittest.TestCase):
    def setUp(self):
        self._override = config.CPU_CORES_OVERRIDE
        self._phys = cpu_topology.physical_cores
        cpu_topology.physical_cores = lambda: 12

    def tearDown(self):
        config.CPU_CORES_OVERRIDE = self._override
        cpu_topology.physical_cores = self._phys

    def test_one_engine_lets_openvino_choose_threads(self):
        config.CPU_CORES_OVERRIDE = 0
        self.assertEqual(cpu_topology.openvino_plan(4, 6, 1, "openvino"),
                         {"cpu_streams": 4, "cpu_infer_threads": 0})

    def test_several_engines_share_cores(self):
        config.CPU_CORES_OVERRIDE = 0
        self.assertEqual(cpu_topology.openvino_plan(4, 1, 4, "openvino"),
                         {"cpu_streams": 1, "cpu_infer_threads": 3})
        config.CPU_CORES_OVERRIDE = 8
        self.assertEqual(cpu_topology.openvino_plan(4, 6, 1, "openvino")["cpu_infer_threads"], 8)

    def test_onnx_sessions_split_cores(self):
        config.CPU_CORES_OVERRIDE = 0
        self.assertEqual(cpu_topology.openvino_plan(4, 6, 1, "onnx"),
                         {"cpu_streams": 4, "cpu_infer_threads": 3})


if __name__ == "__main__":
    unittest.main()
