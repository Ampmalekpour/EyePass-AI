"""
stub_modules.py
--------------------------------------------------------------------
Installs lightweight stand-ins for third-party packages this sandbox
cannot install (no network access to PyPI — see the module README's
"Testing" section) into sys.modules, so the modules under test can be
imported without ImportError. Only import-time surface is stubbed;
nothing here is meant to run real inference.

Call install() before importing anything from common/facecore,
detector/src or recognizer/src that transitively imports redis/torch/
ultralytics/onnxruntime/boto3.
--------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import types


def _install_redis():
    if "redis" in sys.modules and getattr(sys.modules["redis"], "_is_fake", False):
        return
    from . import fake_redis
    mod = types.ModuleType("redis")
    mod.Redis = fake_redis.FakeRedis
    mod._is_fake = True
    sys.modules["redis"] = mod


def _install_torch():
    if "torch" in sys.modules:
        return
    torch_mod = types.ModuleType("torch")

    class _FakeDevice:
        def __init__(self, spec="cpu"):
            self.spec = spec

        def __repr__(self):
            return f"device({self.spec!r})"

    torch_mod.device = _FakeDevice

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    torch_mod.cuda = _FakeCuda()
    torch_mod.no_grad = lambda: _NullContext()
    torch_mod.from_numpy = lambda x: x
    torch_mod.load = lambda *a, **k: {"state_dict": {}}

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = types.ModuleType("torch.nn")
    torch_mod.nn = sys.modules["torch.nn"]


def _install_ultralytics():
    if "ultralytics" in sys.modules:
        return
    mod = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *a, **k):
            pass

    mod.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = mod


def _install_tracker_native_deps():
    """cython_bbox and lap are native-compiled acceleration libraries
    BYTETracker (tracker.py) uses. Not installable in this sandbox —
    stub just enough surface for import + the two call sites actually
    used by the rebalance/engine_manager tests (which never run a real
    tracking step, only construct EngineManager)."""
    if "cython_bbox" not in sys.modules:
        mod = types.ModuleType("cython_bbox")

        def bbox_overlaps(boxes, query_boxes):
            import numpy as np
            return np.zeros((len(boxes), len(query_boxes)), dtype=float)

        mod.bbox_overlaps = bbox_overlaps
        sys.modules["cython_bbox"] = mod

    if "lap" not in sys.modules:
        mod = types.ModuleType("lap")

        def lapjv(cost, extend_cost=True, cost_limit=None):
            import numpy as np
            n = cost.shape[0]
            return 0.0, np.arange(n), np.arange(n)

        mod.lapjv = lapjv
        sys.modules["lap"] = mod


def _install_boto3():
    if "boto3" in sys.modules:
        return
    mod = types.ModuleType("boto3")
    mod.client = lambda *a, **k: None
    sys.modules["boto3"] = mod
    botocore = types.ModuleType("botocore")
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **k: None
    sys.modules["botocore"] = botocore
    sys.modules["botocore.config"] = botocore_config


def install():
    _install_redis()
    _install_torch()
    _install_ultralytics()
    _install_tracker_native_deps()
    _install_boto3()
