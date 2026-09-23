"""
test_engine_manager_rebalance.py
--------------------------------------------------------------------
Exercises EngineManager.rebalance() / _migrate_camera_locked() — the
new consolidation behaviour that does not exist in the reference plate
pipeline (video_processor.py's own EngineManager only ever grows the
engine pool, never shrinks it — see engine_manager.py's module
docstring). Ported from face_service's own
test_engine_manager_rebalance.py; rebalance()/_migrate_camera_locked()
were ported into plate's engine_manager.py essentially verbatim, so
this is the same test against the same logic, just plate's kwarg shape
for add_camera (line_p1_x/... /stop_roi_p4_y instead of face's own
line_p1_x/.../stop_roi_p4_y — actually identical field names, since
face's own trigger config shape was itself carried over unchanged from
plate's reference alpr_api.py in the first place).

Engine subprocesses are never actually spawned here — `proc` is a
lightweight fake with `.is_alive()`/`.pid`/`.join()`/`.terminate()`,
and `control_queue`/`stop_event` are simple recorders. This tests the
bin-packing/migration LOGIC in isolation from YOLO/torch/CUDA/lap,
which this sandbox may not have installed anyway (see
tests/fakes/stub_modules.py).

Run with:
    PYTHONPATH=common:detector/src python3 tests/test_engine_manager_rebalance.py
--------------------------------------------------------------------
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

from engine_manager import EngineManager  # noqa: E402


class _FakeProc:
    def __init__(self):
        self.pid = id(self)
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeEvent:
    def __init__(self):
        self.is_set_flag = False

    def set(self):
        self.is_set_flag = True


def _make_manager(max_cameras_per_engine=5):
    topology_changes = []
    mgr = EngineManager(
        model_path="/models/best.pt", imgsz=480, conf=0.25,
        save_output=False, output_dir="/data", class_labels={0: "Car", 1: "Motorcycle"},
        max_cameras_per_engine=max_cameras_per_engine,
        on_topology_changed=lambda: topology_changes.append(True),
    )
    mgr._topology_changes = topology_changes
    return mgr


def _inject_engine(mgr: EngineManager, engine_id: int, camera_ids):
    mgr.engines[engine_id] = {
        "proc": _FakeProc(),
        "control_queue": _FakeQueue(),
        "stop_event": _FakeEvent(),
        "cameras": set(camera_ids),
    }
    for cid in camera_ids:
        mgr.camera_to_engine[cid] = engine_id
        mgr._camera_last_config[cid] = {
            "cmd": "add", "camera_id": cid, "url": f"rtsp://mediamtx:8554/{cid}",
            "roi": (0, 0, 1, 1),
        }


class RebalanceTests(unittest.TestCase):
    def test_no_rebalance_needed_when_already_tight(self):
        mgr = _make_manager(max_cameras_per_engine=5)
        _inject_engine(mgr, 0, ["1", "2", "3", "4", "5"])
        _inject_engine(mgr, 1, ["6", "7"])
        # 7 cameras / cap 5 => needed=2 engines, currently 2 engines — already tight.
        mgr.rebalance()
        self.assertEqual(set(mgr.engines.keys()), {0, 1})
        self.assertEqual(mgr.camera_engine("6"), 1)

    def test_fragmented_10_cameras_across_3_engines_consolidates_to_2(self):
        mgr = _make_manager(max_cameras_per_engine=5)
        _inject_engine(mgr, 0, ["1", "2", "3", "4"])
        _inject_engine(mgr, 1, ["5", "6", "7"])
        _inject_engine(mgr, 2, ["8", "9", "10"])

        self.assertEqual(len(mgr.engines), 3)

        mgr.rebalance()

        # 10 cameras / cap 5 => needed = ceil(10/5) = 2 engines.
        self.assertEqual(len(mgr.engines), 2, "rebalance must consolidate down to 2 engines")

        all_cameras = set()
        for eid, info in mgr.engines.items():
            self.assertLessEqual(len(info["cameras"]), 5)
            all_cameras |= info["cameras"]
        self.assertEqual(all_cameras, {str(i) for i in range(1, 11)})

        for cid in all_cameras:
            self.assertIn(mgr.camera_engine(cid), mgr.engines.keys())

        self.assertTrue(len(mgr._topology_changes) >= 1, "on_topology_changed must fire so lifecycle checkpoints the new count")

    def test_migrated_camera_config_is_replayed_on_new_engine(self):
        mgr2 = _make_manager(max_cameras_per_engine=5)
        _inject_engine(mgr2, 0, ["1"])
        _inject_engine(mgr2, 1, ["2"])
        mgr2.rebalance()  # 2 cameras / cap 5 => needed=1, currently 2 engines -> consolidate

        self.assertEqual(len(mgr2.engines), 1)
        remaining_eid = next(iter(mgr2.engines.keys()))
        queue_items = mgr2.engines[remaining_eid]["control_queue"].items
        added_camera_ids = {item["camera_id"] for item in queue_items if item.get("cmd") == "add"}
        self.assertTrue(added_camera_ids, "the surviving engine must receive an 'add' replay for the migrated camera")
        self.assertTrue(added_camera_ids.issubset({"1", "2"}))
        both_cameras_present = mgr2.engines[remaining_eid]["cameras"]
        self.assertEqual(both_cameras_present, {"1", "2"}, "both cameras must end up tracked on the surviving engine")

    def test_single_engine_never_rebalances(self):
        mgr = _make_manager(max_cameras_per_engine=5)
        _inject_engine(mgr, 0, ["1", "2"])
        mgr.rebalance()
        self.assertEqual(len(mgr.engines), 1)

    def test_camera_without_cached_config_is_left_in_place_not_lost(self):
        mgr = _make_manager(max_cameras_per_engine=5)
        _inject_engine(mgr, 0, ["1"])
        _inject_engine(mgr, 1, ["2"])
        # Simulate a cache miss (shouldn't normally happen — add_camera
        # always populates it — but must never silently drop a camera).
        mgr._camera_last_config.pop("1", None)

        mgr.rebalance()

        self.assertIn("1", mgr.camera_to_engine)
        self.assertEqual(mgr.camera_engine("1"), 0)
        self.assertIn(0, mgr.engines, "engine with an un-migratable camera must not be stopped")


def _fake_start_engine(mgr: EngineManager):
    """Replaces EngineManager._start_engine_locked with one that injects
    a _FakeProc-backed engine instead of actually spawning a subprocess
    (which would try to run _engine_process_main — torch/YOLO/RTSP and
    all — for real). Returns the new engine_id, same contract as the
    original."""

    def _start(_self=mgr):
        engine_id = 0 if not _self.engines else (max(_self.engines.keys()) + 1)
        _self.engines[engine_id] = {
            "proc": _FakeProc(), "control_queue": _FakeQueue(),
            "stop_event": _FakeEvent(), "cameras": set(),
        }
        _self.on_topology_changed()
        return engine_id

    mgr._start_engine_locked = _start
    return mgr


class AddRemoveCameraTests(unittest.TestCase):
    """Basic placement behavior — not covered by the rebalance tests
    above, which inject engines directly rather than going through
    add_camera(). _start_engine_locked is faked out here (see
    _fake_start_engine) so add_camera never actually spawns a real
    subprocess."""

    def test_add_camera_spreads_across_engines_past_the_cap(self):
        mgr = _fake_start_engine(_make_manager(max_cameras_per_engine=2))
        for cid in ["1", "2", "3"]:
            mgr.add_camera(camera_id=cid, url=f"rtsp://mediamtx:8554/{cid}", roi=(0, 0, 1, 1))
        # cap=2: cameras 1,2 fill engine 0; camera 3 spills into engine 1.
        self.assertEqual(len(mgr.engines), 2)
        self.assertEqual(mgr.camera_engine("1"), mgr.camera_engine("2"))
        self.assertNotEqual(mgr.camera_engine("2"), mgr.camera_engine("3"))

    def test_remove_camera_frees_its_slot(self):
        mgr = _fake_start_engine(_make_manager(max_cameras_per_engine=2))
        mgr.add_camera(camera_id="1", url="rtsp://mediamtx:8554/1", roi=(0, 0, 1, 1))
        result = mgr.remove_camera("1")
        self.assertEqual(result["status"], "stopped")
        self.assertIsNone(mgr.camera_engine("1"))

    def test_remove_unknown_camera_is_a_no_op(self):
        mgr = _fake_start_engine(_make_manager(max_cameras_per_engine=2))
        result = mgr.remove_camera("does-not-exist")
        self.assertEqual(result["status"], "not_running")


if __name__ == "__main__":
    unittest.main()
