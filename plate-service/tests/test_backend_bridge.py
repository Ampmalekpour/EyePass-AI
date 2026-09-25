"""
test_backend_bridge.py
--------------------------------------------------------------------
Exercises detector/src/backend_bridge.py's config-field mapping
(build_camera_job) and its ai_status / active_state bookkeeping around
activate/deactivate and the internal-processing-error auto-restart —
the plate-specific logic that has no face_service equivalent (see
backend_bridge.py's module docstring). Runs against the in-memory
FakeRedis and a scripted fake EngineManager, so no real Redis, engine
subprocess, or camera is needed.

Run with:
    PYTHONPATH=common:detector/src python3 tests/test_backend_bridge.py
--------------------------------------------------------------------
"""

import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()
from tests.fakes.fake_redis import reset_all  # noqa: E402

from platecore.bus import RedisBus  # noqa: E402
from backend_bridge import DetectorBridge, build_camera_job  # noqa: E402


class _FakeEngineManager:
    def __init__(self):
        self.status_queue = _FakeStatusQueue()
        self.added = []
        self.removed = []

    def add_camera(self, **kwargs):
        self.added.append(kwargs)
        return {"camera_id": kwargs["camera_id"], "engine_id": 0, "status": "started"}

    def remove_camera(self, camera_id):
        self.removed.append(camera_id)
        return {"camera_id": camera_id, "status": "stopped"}


class _FakeStatusQueue:
    def get(self, timeout=None):
        raise Exception("empty")  # never has anything — not exercised here


class BuildCameraJobTests(unittest.TestCase):
    def test_full_config_maps_every_field(self):
        cfg = {
            "address": "rtsp://cam/1",
            "roi": {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.7},
            "cross_line": {"start": {"x": 100, "y": 400}, "end": {"x": 900, "y": 400}},
            "stop_roi": {"x": 150, "y": 350, "w": 500, "h": 200},
        }
        job = build_camera_job("1", cfg)
        self.assertEqual(job.roi, (0.1, 0.2, 0.8, 0.7))
        self.assertEqual(job.line_p1, (100.0, 400.0))
        self.assertEqual(job.line_p2, (900.0, 400.0))
        self.assertEqual(job.stop_roi[0], (150.0, 350.0))
        self.assertEqual(job.stop_roi[2], (650.0, 550.0))  # x+w, y+h corner
        self.assertTrue(job.cross_line_trig)
        self.assertTrue(job.stop_roi_trig)
        # video_path is always the shared relay, never cfg["address"]
        # directly — on the one relay path per physical camera that the
        # suite's camera_stream registers (platecore/relay.py).
        from platecore.relay import relay_path_for
        self.assertTrue(job.video_path.endswith("/" + relay_path_for("1", "rtsp://cam/1")))
        self.assertTrue(job.video_path.startswith("rtsp://"))
        self.assertNotIn("rtsp://cam/1", job.video_path)

    def test_relay_path_from_details_wins(self):
        from backend_bridge import rtsp_url
        self.assertTrue(rtsp_url("1", {"relay_path": "cam_abc", "address": "rtsp://x"}).endswith("/cam_abc"))
        self.assertTrue(rtsp_url("1", {}).endswith("/1"))   # nothing known: camera id

    def test_missing_optional_fields_default_safely(self):
        job = build_camera_job("2", {"address": "rtsp://cam/2"})
        self.assertEqual(job.roi, (0.0, 0.0, 1.0, 1.0))
        self.assertFalse(job.cross_line_trig)
        self.assertFalse(job.stop_roi_trig)
        self.assertEqual(job.stop_roi[0], (0.0, 0.0))


class DetectorBridgeActivateDeactivateTests(unittest.TestCase):
    def setUp(self):
        reset_all()
        os.environ["REDIS_URL"] = "redis://fake-backend-bridge-test/0"
        self.bus = RedisBus(module="testbridge")
        self.engine_manager = _FakeEngineManager()
        self.demand_events = []
        self.bridge = DetectorBridge(self.bus, self.engine_manager,
                                      on_demand_changed=lambda active: self.demand_events.append(active))

        cfg = {
            "address": "rtsp://cam/9",
            "roi": {"x": 0, "y": 0, "w": 1, "h": 1},
        }
        self.bus.rt.hset(self.bus.keys.cameras_config, "9", json.dumps(cfg))
        self.bus.rt.hset(self.bus.keys.cameras_details, "9", json.dumps({"connected": True}))

    def _ai_status(self, camera_id):
        raw = self.bus.rt.hget(self.bus.keys.ai_status(camera_id), camera_id)
        return json.loads(raw) if raw else None

    def test_activate_attaches_and_writes_running_status(self):
        ok = self.bridge.handle_activated("9", "req-1")
        self.assertTrue(ok)
        self.assertTrue(self.bridge.is_running("9"))
        self.assertEqual(len(self.engine_manager.added), 1)
        self.assertEqual(self._ai_status("9")["current"], "running")
        self.assertTrue(self.bridge.active_state.is_active("9"))
        self.assertEqual(self.demand_events, [True])

    def test_deactivate_detaches_and_writes_stopped_by_user(self):
        self.bridge.handle_activated("9", "req-1")
        ok = self.bridge.handle_deactivated("9")
        self.assertTrue(ok)
        self.assertFalse(self.bridge.is_running("9"))
        self.assertEqual(self.engine_manager.removed, ["9"])
        self.assertEqual(self._ai_status("9")["current"], "stopped_by_user")
        self.assertFalse(self.bridge.active_state.is_active("9"))
        self.assertEqual(self.demand_events, [True, False])

    def test_activate_while_offline_marks_active_but_does_not_attach(self):
        self.bus.rt.hset(self.bus.keys.cameras_details, "9", json.dumps({"connected": False}))
        ok = self.bridge.handle_activated("9", "req-2")
        self.assertTrue(ok, "activating an offline camera is not an error — it waits for the online event")
        self.assertFalse(self.bridge.is_running("9"))
        self.assertTrue(self.bridge.active_state.is_active("9"), "durable desired state must still be set")
        self.assertEqual(len(self.engine_manager.added), 0)

    def test_camera_event_online_attaches_only_if_active(self):
        # Not activated yet — an online event must be ignored.
        self.bridge.on_camera_event({"id": "9", "connected": True})
        self.assertFalse(self.bridge.is_running("9"))

        # Now activate (offline), then the online event should attach it.
        self.bus.rt.hset(self.bus.keys.cameras_details, "9", json.dumps({"connected": False}))
        self.bridge.handle_activated("9", "req-3")
        self.assertFalse(self.bridge.is_running("9"))

        self.bridge.on_camera_event({"id": "9", "connected": True})
        self.assertTrue(self.bridge.is_running("9"))


class ErrorRestartTests(unittest.TestCase):
    def setUp(self):
        reset_all()
        os.environ["REDIS_URL"] = "redis://fake-backend-bridge-errtest/0"
        self.bus = RedisBus(module="testbridge2")
        self.engine_manager = _FakeEngineManager()
        self.bridge = DetectorBridge(self.bus, self.engine_manager)

        cfg = {"address": "rtsp://cam/9", "roi": {"x": 0, "y": 0, "w": 1, "h": 1}}
        self.bus.rt.hset(self.bus.keys.cameras_config, "9", json.dumps(cfg))
        self.bus.rt.hset(self.bus.keys.cameras_details, "9", json.dumps({"connected": True}))
        self.bridge.handle_activated("9", "req-1")
        self.engine_manager.added.clear()  # only care about restart's own add_camera calls

    def _ai_status(self, camera_id):
        raw = self.bus.rt.hget(self.bus.keys.ai_status(camera_id), camera_id)
        return json.loads(raw) if raw else None

    def test_engine_error_detaches_immediately_and_marks_stopped(self):
        self.bridge.handle_engine_error("9", "batch inference exception")
        self.assertFalse(self.bridge.is_running("9"))
        self.assertEqual(self.engine_manager.removed, ["9"])
        self.assertEqual(self._ai_status("9")["current"], "stopped")
        self.assertEqual(self._ai_status("9")["error"], "batch inference exception")

    def test_engine_error_schedules_a_background_reattach(self):
        # Patch the module-level config the restart worker reads its
        # delay from, so the test doesn't have to sleep the real default.
        import config as detector_config
        original_delay = detector_config.ERROR_RESTART_DELAY_SEC
        detector_config.ERROR_RESTART_DELAY_SEC = 0.05
        try:
            self.bridge.handle_engine_error("9", "boom")
            deadline = time.time() + 2.0
            while time.time() < deadline and not self.bridge.is_running("9"):
                time.sleep(0.02)
            self.assertTrue(self.bridge.is_running("9"), "the bounded auto-restart must reattach the camera")
            self.assertEqual(self._ai_status("9")["current"], "running")
        finally:
            detector_config.ERROR_RESTART_DELAY_SEC = original_delay

    def test_error_retries_are_bounded(self):
        import config as detector_config
        original_max = detector_config.ERROR_RESTART_MAX_RETRIES
        detector_config.ERROR_RESTART_MAX_RETRIES = 1
        try:
            # First error: within budget, detaches and schedules a retry.
            self.bridge.handle_engine_error("9", "err1")
            # Second error before the retry counter resets: over budget,
            # must NOT schedule another restart attempt — just stays stopped.
            self.bridge.handle_engine_error("9", "err2")
            self.assertEqual(self._ai_status("9")["current"], "stopped")
            self.assertEqual(self._ai_status("9")["error"], "err2")
        finally:
            detector_config.ERROR_RESTART_MAX_RETRIES = original_max


if __name__ == "__main__":
    unittest.main()
