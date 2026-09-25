"""
End-to-end wire test against a REAL redis-server (skipped when the
binary or the redis-py package is missing):

    facecore/platecore HubClient + RedisBus  (the detector side, real code)
        -> {m}:internal:hub:events stream
    a fake worker popping the real task queue
        -> {m}:internal:hub:results stream   (bus.hub_push_result, real code)
    ModuleRunner (real hub service: consumer group, lease, checkpoints)
        -> {m} backend results list  +  {m}:internal:hub:ctl:{engine}

Also proves crash recovery: a runner killed with a deferred trigger in
memory is replaced by a fresh one that restores the checkpoint and
still publishes the trigger.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

try:
    import redis  # noqa: F401
    HAVE_REDIS_PY = True
except Exception:
    HAVE_REDIS_PY = False

REDIS_BIN = shutil.which("redis-server")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait(pred, timeout=8.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


@unittest.skipUnless(REDIS_BIN and HAVE_REDIS_PY, "needs redis-server and redis-py")
class HubEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp()
        cls.proc = subprocess.Popen([REDIS_BIN, "--port", str(cls.port), "--save", "", "--appendonly", "no",
                                     "--dir", cls.tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["REDIS_URL"] = f"redis://127.0.0.1:{cls.port}/0"
        import redis as _r
        cls.r = _r.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        _wait(lambda: cls._ping(cls.r))

    @staticmethod
    def _ping(r):
        try:
            return r.ping()
        except Exception:
            return False

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(5)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.r.flushall()
        self.runners = []

    def tearDown(self):
        for rn in self.runners:
            rn.stop()
        for rn in self.runners:
            rn.join(3)

    # ------------------------------------------------------------------
    def _runner(self, module, **cfg_over):
        import config as hub_config
        from config import module_config
        from service import ModuleRunner
        hub_config.LEASE_TTL_SEC = 2.0
        hub_config.LEASE_RENEW_SEC = 0.5
        hub_config.READ_BLOCK_MS = 50
        hub_config.TICK_INTERVAL_SEC = 0.05
        cfg = module_config(module)
        for k, v in cfg_over.items():
            setattr(cfg, k, v)
        rn = ModuleRunner(cfg)
        rn.start()
        self.runners.append(rn)
        _wait(lambda: rn.is_leader and rn.core is not None)
        time.sleep(0.2)  # consumer group created
        return rn

    def _import_core(self, module):
        """Import facecore / platecore fresh (both are named differently,
        so they can coexist)."""
        common = os.path.join(ROOT, f"{module}-service", "common")
        if common not in sys.path:
            sys.path.insert(0, common)
        pkg = "facecore" if module == "face" else "platecore"
        bus_mod = __import__(f"{pkg}.bus", fromlist=["RedisBus"])
        hub_mod = __import__(f"{pkg}.hub", fromlist=["HubClient"])
        codec = __import__(f"{pkg}.codec", fromlist=["decode_task"])
        return bus_mod.RedisBus(module=module), hub_mod, codec

    def _results(self, key):
        return [json.loads(x) for x in self.r.lrange(key, 0, -1)]

    # ------------------------------------------------------------------
    def test_face_full_flow(self):
        self._runner("face", periodic_first_delay_sec=0.2)
        bus, hub, codec = self._import_core("face")
        client = hub.HubClient(bus, engine_id=1)
        uid = hub.new_track_uid("cam1", 1)
        client.emit(hub.K_TRACK_STARTED, {"uid": uid, "camera_id": "cam1", "track_id": 3,
                                          "video_source": "rtsp://relay/cam1",
                                          "triggers": {"periodic": True, "line_cross": True,
                                                       "stopped_roi": False, "leave_scene": True}})
        # hub asks for a periodic pass
        req = _wait(lambda: [m for m in client.drain() if m.get("action") == "request"])
        self.assertEqual(req[0]["stage"], "periodic")
        self.assertEqual(req[0]["uid"], uid)

        rs = hub.TrackRecState(uid)
        rs.apply_ctl(req[0], time.time())
        stage = rs.next_stage(time.time(), 5.0, {"line_cross": 2, "periodic": 1})
        self.assertEqual(stage, "periodic")

        # detector submits, worker answers (low confidence)
        tid = hub.new_task_id(uid, stage)
        client.emit(hub.K_SUBMITTED, {"uid": uid, "task_id": tid, "stage": stage})
        client.submit({"task_type": stage, "stage": stage, "task_id": tid, "track_uid": uid,
                       "camera_id": "cam1", "track_id": 3})
        rs.mark_submitted(tid, stage, 1, time.time())
        task = codec.decode_task(bus.pop_task(timeout=1))
        self.assertEqual(task["track_uid"], uid)
        bus.hub_push_result({"uid": uid, "task_id": tid, "stage": stage, "status": "ok",
                             "is_valid": True, "confidence": 0.4, "personnelid": "42",
                             "first_name": "Ali", "last_name": "R", "face_image": "kf.png",
                             "camera_image": "kc.png", "detection_score": 0.4})
        ack = _wait(lambda: [m for m in client.drain() if m.get("action") == "result"])
        self.assertFalse(ack[0]["satisfied"])
        rs.apply_ctl(ack[0], time.time())
        self.assertIsNone(rs.in_flight_task)

        # trigger with its own task -> held until that task answers
        tid2 = hub.new_task_id(uid, "line_cross")
        client.emit(hub.K_SUBMITTED, {"uid": uid, "task_id": tid2, "stage": "line_cross"})
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "line_cross", "task_id": tid2,
                                    "detail": {"direction": "positive_to_negative"}})
        time.sleep(0.3)
        self.assertEqual(self._results("face:ai:results"), [])
        bus.hub_push_result({"uid": uid, "task_id": tid2, "stage": "line_cross", "status": "ok",
                             "is_valid": True, "confidence": 0.9, "personnelid": "42",
                             "first_name": "Ali", "last_name": "R", "face_image": "kf2.png",
                             "camera_image": "kc2.png", "detection_score": 0.9})
        pub = _wait(lambda: self._results("face:ai:results"))
        self.assertEqual(pub[0]["event_type"], "line_cross")
        self.assertEqual(pub[0]["meta"]["identified_as"], "42")
        self.assertEqual(pub[0]["meta"]["event"]["direction"], "positive_to_negative")
        self.assertEqual(pub[0]["track_uid"], uid)
        sat = _wait(lambda: [m for m in client.drain() if m.get("satisfied")])
        self.assertTrue(sat)

        # end -> final
        client.emit(hub.K_TRACK_ENDED, {"uid": uid, "reason": "absent", "seen_frames": 50, "n_crops": 5,
                                        "finalize_task_id": None})
        pub = _wait(lambda: len(self._results("face:ai:results")) >= 2 and self._results("face:ai:results"))
        final = pub[-1]
        self.assertTrue(final["is_final"])
        self.assertEqual(final["event_type"], "finalize")
        self.assertEqual(final["meta"]["identified_as"], "42")
        self.assertEqual(final["meta"]["votes"], 2)
        client.stop()

    def test_crash_recovery_restores_deferred_trigger(self):
        rn = self._runner("face", trigger_max_wait_sec=1.0)
        bus, hub, _ = self._import_core("face")
        client = hub.HubClient(bus, engine_id=2)
        uid = hub.new_track_uid("cam2", 2)
        client.emit(hub.K_TRACK_STARTED, {"uid": uid, "camera_id": "cam2", "track_id": 1,
                                          "triggers": {"line_cross": True}})
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "line_cross", "task_id": None, "detail": {}})
        _wait(lambda: rn.core and uid in rn.core.tracks and rn.core.tracks[uid]["events"])
        time.sleep(0.2)  # checkpoint written
        # "kill" the hub: stop the thread without letting it publish,
        # and drop its lease like an expired container would
        rn.stop()
        rn.join(3)
        self.r.delete("face:internal:hub:leader")
        self.assertEqual(self._results("face:ai:results"), [])

        self._runner("face", trigger_max_wait_sec=1.0)
        pub = _wait(lambda: self._results("face:ai:results"), timeout=6)
        self.assertEqual(pub[0]["event_type"], "line_cross")
        self.assertEqual(pub[0]["meta"]["resolution"], "timeout")
        client.stop()

    def test_standby_does_not_process(self):
        a = self._runner("face")
        import config as hub_config
        from config import module_config
        from service import ModuleRunner
        b = ModuleRunner(module_config("face"))
        b.start()
        self.runners.append(b)
        time.sleep(0.5)
        self.assertTrue(a.is_leader)
        self.assertFalse(b.is_leader)

    def test_plate_flow_uses_plate_vocabulary_and_list(self):
        self._runner("plate")
        bus, hub, _ = self._import_core("plate")
        client = hub.HubClient(bus, engine_id=5)
        uid = hub.new_track_uid("gate", 5)
        client.emit(hub.K_TRACK_STARTED, {"uid": uid, "camera_id": "gate", "track_id": 9,
                                          "triggers": {"cross_line": True, "stop_roi": True,
                                                       "leave_scene": True, "periodic": False}})
        tid = hub.new_task_id(uid, "cross_line")
        client.emit(hub.K_SUBMITTED, {"uid": uid, "task_id": tid, "stage": "cross_line"})
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "cross_line", "task_id": tid,
                                    "detail": {"direction": "negative_to_positive"}})
        bus.hub_push_result({"uid": uid, "task_id": tid, "stage": "cross_line", "trigger_type": "cross_line",
                             "camera_id": "gate", "track_id": 9, "status": "ok", "plate_text": "12B34567",
                             "confidence": 0.93, "is_valid": True, "voted_class": 0,
                             "payload": {"plate_image": "vp.png", "frame_image": "vf.png", "plate_type": 1}})
        pub = _wait(lambda: self._results("plate:vehicle:results"))
        self.assertEqual(pub[0]["update_type"], "cross_line")
        self.assertEqual(pub[0]["resolved"]["plate_text"], "12B34567")
        # satisfied -> stop_roi published immediately, no OCR
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "stop_roi", "task_id": None, "detail": {"duration": 3.4}})
        pub = _wait(lambda: len(self._results("plate:vehicle:results")) >= 2 and self._results("plate:vehicle:results"))
        self.assertEqual(pub[1]["update_type"], "stop_roi")
        self.assertEqual(pub[1]["meta"]["resolution"], "immediate")
        client.emit(hub.K_TRACK_ENDED, {"uid": uid, "reason": "absent", "seen_frames": 30, "n_crops": 4})
        pub = _wait(lambda: len(self._results("plate:vehicle:results")) >= 3 and self._results("plate:vehicle:results"))
        self.assertTrue(pub[2]["is_final"])
        self.assertEqual(pub[2]["update_type"], "leave_scene")
        self.assertIn("leave_scene", pub[2]["events"])
        client.stop()


if __name__ == "__main__":
    unittest.main()
