"""
test_heatmap.py
--------------------------------------------------------------------
The face detector's optional occupancy heatmap (detector/src/heatmap.py)
and its engine hookup, with an in-memory stand-in for MinIO:

  * points land in the right time slot / grid cell (heatmap-service's
    cube format: (slots, grid_h, grid_w) uint32, key <camera>/<date>.npy)
  * a flush MERGES into what is stored — a restarted engine or a camera
    moved to another engine keeps adding instead of overwriting
  * MinIO down -> samples are kept and saved by the next flush
  * a changed grid never corrupts an existing cube
  * the engine samples every N frames per live track, only for cameras
    with the heatmap on, and flushes when a camera is removed
  * per-camera "heatmap" in the camera config overrides the global switch
--------------------------------------------------------------------
"""

import json
import logging
import os
import sys
import time
import unittest
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

for _m in ("config",):  # make sure the DETECTOR's config is the one loaded
    _c = sys.modules.get(_m)
    if _c is not None and "detector" not in (getattr(_c, "__file__", "") or ""):
        del sys.modules[_m]

import config  # noqa: E402
import heatmap as hm_mod  # noqa: E402
import engine as engine_mod  # noqa: E402
from backend_bridge import build_camera_job  # noqa: E402


class MemStore:
    def __init__(self):
        self.bucket = "face-heatmap"
        self.objects = {}
        self.down = False

    def get(self, key):
        if self.down:
            raise ConnectionError("minio down")
        a = self.objects.get(key)
        return None if a is None else a.copy()

    def put(self, key, arr):
        if self.down:
            raise ConnectionError("minio down")
        self.objects[key] = arr.copy()


class FakeRT:
    def __init__(self):
        self.lists = {}

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(json.loads(value))


class FakeBus:
    def __init__(self):
        self.rt = FakeRT()


def make(store=None, bus=None):
    config.HEATMAP_SAVE_INTERVAL_SEC = 3600  # no background flush during tests
    return hm_mod.FaceHeatmap(bus or FakeBus(), 1, store=store or MemStore(), results_key="face:heatmap:results")


TS = datetime(2026, 9, 25, 10, 7)          # slot 10*60+7 = 607 // 5 = 121
KEY = "cam1/2026-09-25.npy"


class HeatmapAccumulatorTests(unittest.TestCase):
    def test_cell_and_slot_match_heatmap_service_math(self):
        h = make()
        h.sample("cam1", 960, 540, 1920, 1080, TS)          # centre of a 1920x1080 frame
        h.sample("cam1", 1919, 1079, 1920, 1080, TS)        # bottom-right corner, clamped
        h.flush()
        cube = h.store.objects[KEY]
        self.assertEqual(cube.shape, (288, 72, 128))
        self.assertEqual(cube.dtype, np.uint32)
        self.assertEqual(cube[121, 36, 64], 1)
        self.assertEqual(cube[121, 71, 127], 1)
        self.assertEqual(int(cube.sum()), 2)

    def test_flush_merges_so_restarts_and_rebalances_keep_counting(self):
        store = MemStore()
        a = make(store)
        a.sample("cam1", 10, 10, 1920, 1080, TS)
        a.flush()
        b = make(store)                          # "restarted engine" / camera moved
        b.sample("cam1", 10, 10, 1920, 1080, TS)
        b.sample("cam1", 10, 10, 1920, 1080, TS)
        b.flush()
        self.assertEqual(int(store.objects[KEY].sum()), 3)

    def test_minio_outage_keeps_samples(self):
        store = MemStore()
        h = make(store)
        h.sample("cam1", 10, 10, 1920, 1080, TS)
        store.down = True
        self.assertFalse(h.flush())
        h.sample("cam1", 10, 10, 1920, 1080, TS)   # gathered while down
        self.assertEqual(h.pending_samples(), 2)
        store.down = False
        self.assertTrue(h.flush())
        self.assertEqual(int(store.objects[KEY].sum()), 2)
        self.assertEqual(h.pending_samples(), 0)

    def test_changed_grid_never_corrupts_existing_cube(self):
        store = MemStore()
        store.objects[KEY] = np.ones((288, 36, 64), dtype=np.uint32)
        h = make(store)
        h.sample("cam1", 10, 10, 1920, 1080, TS)
        h.flush()
        self.assertEqual(int(store.objects[KEY].sum()), 288 * 36 * 64)   # untouched
        self.assertIn("cam1/2026-09-25.288x72x128.npy", store.objects)

    def test_announces_every_flush_and_camera_finish(self):
        bus = FakeBus()
        h = make(bus=bus)
        h.sample("cam1", 10, 10, 1920, 1080, TS)
        h.flush("cam1", finished=True)
        events = bus.rt.lists["face:heatmap:results"]
        self.assertEqual([e["event"] for e in events], ["matrix_sync", "processing_finished"])
        self.assertEqual(events[0]["object_key"], KEY)
        self.assertEqual(events[0]["bucket"], "face-heatmap")
        self.assertEqual(events[0]["samples_added"], 1)

    def test_flush_one_camera_leaves_others_pending(self):
        h = make()
        h.sample("cam1", 10, 10, 1920, 1080, TS)
        h.sample("cam2", 10, 10, 1920, 1080, TS)
        h.flush("cam1")
        self.assertIn(KEY, h.store.objects)
        self.assertEqual(h.pending_samples(), 1)

    def test_point_modes(self):
        config.HEATMAP_POINT_MODE = "center"
        self.assertEqual(hm_mod.FaceHeatmap.point((0, 0, 10, 20)), (5.0, 10.0))
        config.HEATMAP_POINT_MODE = "bottom_center"
        self.assertEqual(hm_mod.FaceHeatmap.point((0, 0, 10, 20)), (5.0, 20.0))
        config.HEATMAP_POINT_MODE = "center"


class _Track:
    def __init__(self, bbox):
        self.detbb = bbox


class EngineHookupTests(unittest.TestCase):
    def _engine(self, heatmap_on=True):
        eng = object.__new__(engine_mod.Engine)
        eng.engine_id = 1
        eng.logger = logging.getLogger("test-heatmap-engine")
        eng.bus = FakeBus()
        eng._heatmap = make(bus=eng.bus)
        eng.cameras = {"cam1": {"frame_wh": (1920, 1080), "heatmap_enabled": heatmap_on}}
        return eng

    def test_samples_every_live_track_in_full_frame_coords(self):
        eng = self._engine()
        eng._sample_heatmap("cam1", eng.cameras["cam1"],
                            [_Track((0, 0, 20, 20)), _Track((100, 100, 140, 140)), _Track(None)], (960, 540))
        self.assertEqual(eng._heatmap.pending_samples(), 2)
        cur = eng._heatmap.current_slot("cam1")
        # first head centre at (970, 550) full-frame -> col 64, row 36
        self.assertEqual(cur[36, 64], 1)

    def test_sampling_never_raises_into_the_frame_loop(self):
        eng = self._engine()
        eng.cameras["cam1"]["frame_wh"] = (0, 0)
        eng._sample_heatmap("cam1", eng.cameras["cam1"], [_Track((0, 0, 1, 1))], (0, 0))
        eng._sample_heatmap("cam1", eng.cameras["cam1"], [object()], (0, 0))   # malformed track
        self.assertEqual(eng._heatmap.pending_samples(), 0)

    def test_camera_removal_flushes_its_cube(self):
        eng = self._engine()
        eng.hub = type("H", (), {"emit": lambda *a, **k: None})()
        eng._uid_index = {}
        eng.writers = {}
        eng.status_queue = type("Q", (), {"put_nowait": lambda self, m: None})()
        eng.cameras["cam1"].update({"track_meta": {}, "rec_state": {}, "recorder": None, "liveness": None,
                                    "reader": type("R", (), {"stop": lambda self: None})()})
        eng._heatmap.sample("cam1", 10, 10, 1920, 1080, datetime.now())
        eng.remove_camera("cam1")
        deadline = time.time() + 3            # flushed on a background thread
        while len(eng.bus.rt.lists.get("face:heatmap:results", [])) < 2 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(eng._heatmap.pending_samples(), 0)
        events = [e["event"] for e in eng.bus.rt.lists.get("face:heatmap:results", [])]
        self.assertEqual(events, ["matrix_sync", "processing_finished"])


class PerCameraSwitchTests(unittest.TestCase):
    BASE = {"address": "rtsp://x", "roi": {"x": 0, "y": 0, "w": 1, "h": 1}}

    def test_camera_config_overrides_global_switch(self):
        self.assertIsNone(build_camera_job("1", dict(self.BASE)).heatmap_trig)
        self.assertTrue(build_camera_job("1", {**self.BASE, "heatmap": True}).heatmap_trig)
        self.assertFalse(build_camera_job("1", {**self.BASE, "heatmap": False}).heatmap_trig)


if __name__ == "__main__":
    unittest.main()
