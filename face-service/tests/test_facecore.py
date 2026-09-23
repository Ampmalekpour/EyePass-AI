"""
test_facecore.py
--------------------------------------------------------------------
Unit tests for facecore.codec, facecore.lifecycle and
facecore.active_state, run against the in-memory FakeRedis (see
tests/fakes/) since this sandbox cannot install the real redis package
or stand up a real Redis server.

Run with:
    PYTHONPATH=common python3 tests/test_facecore.py
--------------------------------------------------------------------
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

from tests.fakes.fake_redis import reset_all  # noqa: E402

from facecore.bus import RedisBus  # noqa: E402
from facecore.codec import decode_task, decode_result, encode_task, encode_result, DateTimeEncoder  # noqa: E402
from facecore.lifecycle import ServiceLifecycle, PHASE_IDLE, PHASE_PROCESSING, PHASE_STOPPED  # noqa: E402
from facecore.active_state import ActiveCameraState  # noqa: E402


class CodecTests(unittest.TestCase):
    def test_task_round_trip(self):
        task = {"track_id": 42, "crops": {"reg1": [{"image_bytes": b"\x00\x01jpegbytes"}]}, "engine_id": 3}
        raw = encode_task(task)
        self.assertIsInstance(raw, bytes)
        back = decode_task(raw)
        self.assertEqual(back, task)

    def test_result_round_trip_is_same_envelope_as_task(self):
        result = {"track_id": 1, "personnelid": "7", "face_image": None}
        self.assertEqual(decode_result(encode_result(result)), result)

    def test_datetime_encoder(self):
        import datetime
        import json
        payload = {"date": datetime.date(2026, 1, 1), "time": datetime.time(9, 30), "n": 1}
        raw = json.dumps(payload, cls=DateTimeEncoder)
        self.assertIn("2026-01-01", raw)
        self.assertIn("09:30:00", raw)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        reset_all()
        os.environ["REDIS_URL"] = "redis://fake-lifecycle-test/0"
        self.bus = RedisBus(module="testface")
        self.loaded = []
        self.stopped_idle = []
        self.started_process = []
        self.stopped_process = []
        self.count = {"n": 0}

        def on_start_idle(n):
            self.count["n"] = n
            self.loaded.append(n)
            return n

        def on_stop_idle():
            self.stopped_idle.append(True)
            self.count["n"] = 0

        def on_start_process():
            self.started_process.append(True)

        def on_stop_process():
            self.stopped_process.append(True)

        self.lifecycle = ServiceLifecycle(
            bus=self.bus,
            state_key="testface:internal:x:state",
            on_start_idle=on_start_idle,
            on_stop_idle=on_stop_idle,
            on_start_process=on_start_process,
            on_stop_process=on_stop_process,
            get_unit_count=lambda: self.count["n"],
            default_count=3,
            unit_name="engine",
        )

    def test_start_idle_then_start_process_persists_checkpoint(self):
        self.lifecycle.start_idle(5)
        self.assertEqual(self.lifecycle.phase, PHASE_IDLE)
        self.lifecycle.start_process()
        self.assertEqual(self.lifecycle.phase, PHASE_PROCESSING)

        checkpoint = self.lifecycle.read_checkpoint()
        self.assertEqual(checkpoint["phase"], PHASE_PROCESSING)
        self.assertEqual(checkpoint["engine_count"], 5)

    def test_stop_process_returns_to_idle_without_unloading(self):
        self.lifecycle.start_idle(2)
        self.lifecycle.start_process()
        self.lifecycle.stop_process()
        self.assertEqual(self.lifecycle.phase, PHASE_IDLE)
        self.assertEqual(len(self.stopped_idle), 0, "stop_process must not unload — only stop_idle does")

    def test_stop_idle_from_processing_implies_stop_process_first(self):
        self.lifecycle.start_idle(2)
        self.lifecycle.start_process()
        self.lifecycle.stop_idle()
        self.assertEqual(self.lifecycle.phase, PHASE_STOPPED)
        self.assertEqual(len(self.stopped_process), 1)
        self.assertEqual(len(self.stopped_idle), 1)

    def test_self_heal_restores_processing_phase_and_count(self):
        # First "boot": go to processing with 4 units, persisted to Redis.
        self.lifecycle.start_idle(4)
        self.lifecycle.start_process()

        # Simulate a crash/restart: a brand new ServiceLifecycle instance
        # against the SAME (fake) Redis, with fresh callback state.
        new_count = {"n": 0}
        loaded_calls = []

        def on_start_idle(n):
            new_count["n"] = n
            loaded_calls.append(n)
            return n

        started_process = []

        lifecycle2 = ServiceLifecycle(
            bus=self.bus,
            state_key="testface:internal:x:state",
            on_start_idle=on_start_idle,
            on_stop_idle=lambda: None,
            on_start_process=lambda: started_process.append(True),
            on_stop_process=lambda: None,
            get_unit_count=lambda: new_count["n"],
            default_count=1,
            unit_name="engine",
        )
        lifecycle2.self_heal()

        self.assertEqual(loaded_calls, [4], "self_heal must restore the persisted unit count, not default_count")
        self.assertEqual(len(started_process), 1, "self_heal must resume PROCESSING because that was the last phase")
        self.assertEqual(lifecycle2.phase, PHASE_PROCESSING)

    def test_self_heal_on_fresh_boot_goes_idle_only(self):
        # No prior checkpoint at all — self_heal should bring up the
        # default idle floor and NOT start processing.
        checkpoint = self.lifecycle.read_checkpoint()
        self.assertEqual(checkpoint["phase"], PHASE_STOPPED)

        self.lifecycle.self_heal()
        self.assertEqual(self.lifecycle.phase, PHASE_IDLE)
        self.assertEqual(self.loaded, [3])  # default_count
        self.assertEqual(len(self.started_process), 0)

    def test_checkpoint_now_updates_count_without_changing_phase(self):
        self.lifecycle.start_idle(2)
        self.lifecycle.start_process()
        self.count["n"] = 5  # e.g. EngineManager grew organically
        self.lifecycle.checkpoint_now()

        checkpoint = self.lifecycle.read_checkpoint()
        self.assertEqual(checkpoint["phase"], PHASE_PROCESSING)
        self.assertEqual(checkpoint["engine_count"], 5)


class ActiveCameraStateTests(unittest.TestCase):
    def setUp(self):
        reset_all()
        os.environ["REDIS_URL"] = "redis://fake-active-state-test/0"
        self.bus = RedisBus(module="testface2")
        self.state = ActiveCameraState(self.bus)

    def test_mark_active_then_is_active(self):
        self.assertFalse(self.state.is_active("7"))
        roi = {"x": 0, "y": 0, "w": 1, "h": 1}
        self.state.mark_active("7", roi, request_id="r1", extra={"config": {"roi": roi}})
        self.assertTrue(self.state.is_active("7"))
        entry = self.state.get("7")
        self.assertEqual(entry["roi"]["w"], 1)
        self.assertEqual(entry["config"]["roi"]["w"], 1)
        self.assertIn("activated_at", entry)

    def test_mark_inactive_removes_entry(self):
        self.state.mark_active("9", {}, request_id="r2")
        self.assertTrue(self.state.is_active("9"))
        self.state.mark_inactive("9")
        self.assertFalse(self.state.is_active("9"))
        self.assertIsNone(self.state.get("9"))

    def test_all_active_returns_every_entry(self):
        self.state.mark_active("1", {}, request_id="a")
        self.state.mark_active("2", {}, request_id="b")
        all_active = self.state.all_active()
        self.assertEqual(set(all_active.keys()), {"1", "2"})


if __name__ == "__main__":
    unittest.main()
