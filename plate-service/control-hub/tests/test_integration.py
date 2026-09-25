"""
End-to-end wire test against a REAL redis-server (skipped when the
binary or the redis-py package is missing):

    platecore HubClient + RedisBus  (the detector side, real code)
        -> {m}:internal:hub:events stream
    a fake worker popping the real task queue
        -> {m}:internal:hub:results stream   (bus.hub_push_result, real code)
    ModuleRunner (real hub service: consumer group, lease, checkpoints)
        -> {m} backend results list  +  {m}:internal:hub:ctl:{engine}

Also proves: the reported empty-payload bug is gone (a vehicle leaving
while its OCR is queued), crash recovery restores a held trigger, and
the detector's outbox survives a Redis outage.
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
MODULE_DIR = os.path.dirname(os.path.dirname(HERE))      # plate-service/
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
        common = os.path.join(MODULE_DIR, "common")
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


    def test_plate_vehicle_leaves_while_ocr_runs(self):
        """The reported bug, end to end: leave_scene off, the vehicle leaves
        while its cross_line OCR is still in the queue. The final must wait
        for it and carry the plate — not go out empty."""
        self._runner("plate")
        bus, hub, codec = self._import_core("plate")
        client = hub.HubClient(bus, engine_id=6)
        uid = hub.new_track_uid("gate", 6)
        client.emit(hub.K_TRACK_STARTED, {"uid": uid, "camera_id": "gate", "track_id": 2,
                                          "triggers": {"cross_line": True, "stop_roi": False,
                                                       "leave_scene": False, "periodic": False}})
        tid = hub.new_task_id(uid, "cross_line")
        client.emit(hub.K_SUBMITTED, {"uid": uid, "task_id": tid, "stage": "cross_line"})
        client.submit({"task_id": tid, "track_uid": uid, "stage": "cross_line", "trigger_type": "cross_line",
                       "camera_id": "gate", "track_id": 2, "crops": []})
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "cross_line", "task_id": tid, "detail": {}})
        client.emit(hub.K_TRACK_ENDED, {"uid": uid, "reason": "absent", "seen_frames": 30, "n_crops": 4,
                                        "finalize_task_id": None})
        time.sleep(1.0)                                   # OCR busy: nothing published yet
        self.assertEqual(self._results("plate:vehicle:results"), [])
        task = codec.decode_task(bus.pop_task(timeout=2))
        bus.hub_push_result({"uid": task["track_uid"], "task_id": task["task_id"], "stage": "cross_line",
                             "trigger_type": "cross_line", "camera_id": "gate", "track_id": 2, "status": "ok",
                             "plate_text": "22B33344", "confidence": 0.91, "is_valid": True, "voted_class": 0,
                             "payload": {"plate_image": "vp.png", "frame_image": "vf.png", "plate_type": 1}})
        pub = _wait(lambda: len(self._results("plate:vehicle:results")) >= 2 and self._results("plate:vehicle:results"))
        final = pub[-1]
        self.assertTrue(final["is_final"])
        self.assertTrue(final["complete"])
        self.assertEqual(final["resolved"]["plate_text"], "22B33344")
        self.assertEqual(final["ocr_results"]["cross_line"]["plate_text"], "22B33344")
        client.stop()

    def test_crash_recovery_restores_deferred_trigger(self):
        rn = self._runner("plate", trigger_max_wait_sec=1.0)
        bus, hub, _ = self._import_core("plate")
        client = hub.HubClient(bus, engine_id=2)
        uid = hub.new_track_uid("gate", 2)
        client.emit(hub.K_TRACK_STARTED, {"uid": uid, "camera_id": "gate", "track_id": 1,
                                          "triggers": {"cross_line": True}})
        client.emit(hub.K_TRIGGER, {"uid": uid, "event": "cross_line", "task_id": None, "detail": {}})
        _wait(lambda: rn.core and uid in rn.core.tracks and rn.core.tracks[uid]["events"])
        time.sleep(0.2)
        rn.stop()
        rn.join(3)
        self.r.delete("plate:internal:hub:leader")
        self.assertEqual(self._results("plate:vehicle:results"), [])
        self._runner("plate", trigger_max_wait_sec=1.0)
        pub = _wait(lambda: self._results("plate:vehicle:results"), timeout=6)
        self.assertEqual(pub[0]["update_type"], "cross_line")
        client.stop()

@unittest.skipUnless(REDIS_BIN and HAVE_REDIS_PY, "needs redis-server and redis-py")
class DetectorOutboxSurvivesRedisOutage(unittest.TestCase):
    """The detector keeps emitting while Redis is DOWN; every event is
    delivered, in order, once Redis is back — nothing blocks, nothing lost."""

    def test_outage(self):
        port = _free_port()
        tmp = tempfile.mkdtemp()
        args = [REDIS_BIN, "--port", str(port), "--save", "", "--appendonly", "no", "--dir", tmp]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            import redis as _r
            url = f"redis://127.0.0.1:{port}/0"
            r = _r.Redis.from_url(url, decode_responses=True)
            _wait(lambda: HubEndToEnd._ping(r))
            common = os.path.join(MODULE_DIR, "common")
            if common not in sys.path:
                sys.path.insert(0, common)
            from platecore.bus import RedisBus
            from platecore import hub
            bus = RedisBus(module="plate", url=url)
            client = hub.HubClient(bus, engine_id=7)
            _wait(lambda: r.xlen("plate:internal:hub:events") >= 1)      # engine_started

            proc.terminate(); proc.wait(5)                                # Redis goes down
            t0 = time.time()
            for i in range(50):
                client.emit(hub.K_TRACK_UPDATE, {"uid": f"u{i}", "seq": i})
            client.submit({"task_id": "t-during-outage", "track_uid": "u0"})
            self.assertLess(time.time() - t0, 0.5, "emit/submit must never block the frame loop")
            time.sleep(1.0)
            self.assertGreater(client.pending(), 0)

            proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # back up
            _wait(lambda: HubEndToEnd._ping(r))
            _wait(lambda: client.pending() == 0, timeout=20)
            entries = r.xrange("plate:internal:hub:events")
            seqs = [json.loads(f["data"]).get("seq") for _, f in entries if f["kind"] == "track_update"]
            self.assertEqual(seqs, list(range(50)))                      # all of them, in order
            self.assertEqual(r.llen(bus.keys.ocr_tasks), 1)
            client.stop()
        finally:
            proc.terminate()
            proc.wait(5)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
