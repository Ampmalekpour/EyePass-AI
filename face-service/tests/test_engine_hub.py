"""
test_engine_hub.py
--------------------------------------------------------------------
The detector engine's side of the control-hub contract, exercised on
the REAL Engine methods (no YOLO/torch — an Engine is assembled with
object.__new__ and only the attributes these methods touch):

  * a trigger submits the best eligible crop at once and tells the hub
    which task belongs to it
  * nothing is sent while a task is in flight, after the hub said
    "satisfied", or when the best crop is unchanged
  * crops without confident landmarks are never sent (the recognizer
    would drop them) — including on the finalize pass
  * hub ctl messages clear the in-flight lock / set satisfied / queue a
    periodic request
  * track end: finalize pass only when leave_scene is on and the hub is
    not satisfied, then track_ended, then the engine forgets the track
  * removing a camera ends its live tracks instead of dropping them
--------------------------------------------------------------------
"""

import logging
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

import engine as engine_mod  # noqa: E402
from facecore.hub import TrackRecState  # noqa: E402


class FakeHub:
    def __init__(self):
        self.events = []
        self.tasks = []
        self.ctl = []
        self.boot_id = "boot"

    def emit(self, kind, data):
        self.events.append((kind, dict(data)))

    def submit(self, task):
        self.tasks.append(task)

    def drain(self):
        out, self.ctl = self.ctl, []
        return out

    def stop(self):
        pass

    def kinds(self):
        return [k for k, _ in self.events]


def crop(fid, lm_conf=0.9):
    img = np.full((60, 50, 3), 128, dtype=np.uint8)
    lms = np.array([[10.0 + i, 20.0 + i, lm_conf] for i in range(14)])
    return {"image": img, "class_flag": 1, "det_score": 0.9, "resolution": 3000, "aspect_ratio": 0.8,
            "sharpness": 200.0, "frame_number": fid, "mlc": lm_conf, "yaw_group": 3, "landmarks": lms}


def make_engine(leave_scene=True, finalize_max_crops=1):
    eng = object.__new__(engine_mod.Engine)
    eng.engine_id = 1
    eng.logger = logging.getLogger("test-engine")
    eng.hub = FakeHub()
    eng._uid_index = {}
    eng.FINALIZE_MAX_CROPS = finalize_max_crops
    eng.STAGE_PRIORITY = {"finalize": 3, "line_cross": 2, "stopped_roi": 2, "periodic": 1}
    eng.MIN_SEEN_FRAMES = 8
    eng.MIN_CROPS_TO_FINALIZE = 1
    eng.save_output = False
    eng.output_dir = "/tmp"
    eng.writers = {}
    eng.cameras = {"cam1": {
        "camera_id": "cam1", "url": "rtsp://relay/cam1", "track_meta": {}, "rec_state": {},
        "best_crops": {}, "best_frame": {}, "trigger_states": {}, "liveness": None, "recorder": None,
        "reader": type("R", (), {"stop": lambda self: None})(),
        "periodic_enabled": True, "cross_line_enabled": True, "flag_stop_roi_enabled": True,
        "leave_scene_enabled": leave_scene,
    }}
    eng.status_queue = type("Q", (), {"put_nowait": lambda self, m: None})()
    return eng


def add_track(eng, tid=5, crops=(), seen=20):
    cam = eng.cameras["cam1"]
    uid = f"cam1-1-{tid}"
    cam["track_meta"][tid] = {"uid": uid, "first_seen_fid": 1, "last_seen_fid": seen, "seen_frames": seen}
    cam["rec_state"][tid] = TrackRecState(uid)
    cam["best_crops"][tid] = {"reg1": list(crops), "reg2": [], "reg3": []}
    eng._uid_index[uid] = ("cam1", tid)
    return uid, cam["track_meta"][tid], cam["rec_state"][tid]


class EngineHubTests(unittest.TestCase):
    def test_trigger_submits_best_eligible_crop_once(self):
        eng = make_engine()
        uid, meta, rs = add_track(eng, crops=[crop(10)])
        rs.request("line_cross", 1.0)
        tid = eng._try_submit("cam1", 5, meta, rs, 1.0)
        self.assertIsNotNone(tid)
        task = eng.hub.tasks[0]
        self.assertEqual(task["task_type"], "line_cross")
        self.assertEqual(task["track_uid"], uid)
        self.assertEqual(task["task_id"], tid)
        self.assertEqual(len(task["crops"]["reg1"]), 1)
        self.assertEqual(eng.hub.kinds(), ["submitted"])
        # in flight -> nothing more
        rs.request("periodic", 1.1)
        cam = eng.cameras["cam1"]
        cam["best_crops"][5]["reg1"].insert(0, crop(20))
        self.assertIsNone(eng._try_submit("cam1", 5, meta, rs, 1.2))

    def test_same_crop_is_not_resent_until_a_better_one_exists(self):
        eng = make_engine()
        _, meta, rs = add_track(eng, crops=[crop(10)])
        rs.request("periodic", 0)
        tid = eng._try_submit("cam1", 5, meta, rs, 0)
        rs.apply_ctl({"action": "result", "uid": rs.uid, "task_id": tid, "satisfied": False}, 0.5)
        rs.request("line_cross", 1)
        self.assertIsNone(eng._try_submit("cam1", 5, meta, rs, 1))
        self.assertIn("line_cross", rs.requested)            # stays pending
        eng.cameras["cam1"]["best_crops"][5]["reg1"].insert(0, crop(30))
        self.assertIsNotNone(eng._try_submit("cam1", 5, meta, rs, 2))
        self.assertEqual(eng.hub.tasks[-1]["stage"], "line_cross")

    def test_low_landmark_crops_never_sent(self):
        eng = make_engine()
        _, meta, rs = add_track(eng, crops=[crop(10, lm_conf=0.3)])
        rs.request("line_cross", 0)
        self.assertIsNone(eng._try_submit("cam1", 5, meta, rs, 0))
        self.assertEqual(eng.hub.tasks, [])

    def test_satisfied_stops_submissions(self):
        eng = make_engine()
        uid, meta, rs = add_track(eng, crops=[crop(10)])
        eng.hub.ctl = [{"action": "state", "uid": uid, "satisfied": True, "display": {"label": "42 A B"}}]
        eng._drain_hub_ctl()
        self.assertTrue(rs.satisfied)
        rs.request("line_cross", 0)
        self.assertIsNone(eng._try_submit("cam1", 5, meta, rs, 0))

    def test_hub_request_and_result_ack(self):
        eng = make_engine()
        uid, meta, rs = add_track(eng, crops=[crop(10)])
        eng.hub.ctl = [{"action": "request", "uid": uid, "stage": "periodic"}]
        eng._drain_hub_ctl()
        tid = eng._try_submit("cam1", 5, meta, rs, 0)
        self.assertEqual(eng.hub.tasks[0]["stage"], "periodic")
        eng.hub.ctl = [{"action": "result", "uid": uid, "task_id": tid, "satisfied": False,
                        "display": {"label": "unknown", "confidence": 0.0}}]
        eng._drain_hub_ctl()
        self.assertIsNone(rs.in_flight_task)
        self.assertEqual(rs.display["label"], "unknown")

    def test_end_track_sends_finalize_then_forgets(self):
        eng = make_engine(leave_scene=True)
        uid, meta, rs = add_track(eng, crops=[crop(10), crop(11)], seen=30)
        eng._end_track("cam1", 5, meta, reason="absent")
        self.assertEqual(eng.hub.kinds(), ["submitted", "track_ended"])
        self.assertEqual(eng.hub.tasks[0]["stage"], "finalize")
        self.assertEqual(len(eng.hub.tasks[0]["crops"]["reg1"]), 1)
        ended = eng.hub.events[-1][1]
        self.assertEqual(ended["finalize_task_id"], eng.hub.tasks[0]["task_id"])
        self.assertEqual(ended["n_crops"], 2)
        cam = eng.cameras["cam1"]
        self.assertNotIn(5, cam["track_meta"])
        self.assertNotIn(5, cam["rec_state"])
        self.assertNotIn(uid, eng._uid_index)

    def test_end_track_no_finalize_when_satisfied_or_disabled_or_short(self):
        for kw, seen, satisfied in ((dict(leave_scene=False), 30, False),
                                    (dict(leave_scene=True), 30, True),
                                    (dict(leave_scene=True), 3, False)):
            eng = make_engine(**kw)
            _, meta, rs = add_track(eng, crops=[crop(10)], seen=seen)
            rs.satisfied = satisfied
            eng._end_track("cam1", 5, meta)
            self.assertEqual(eng.hub.kinds(), ["track_ended"], (kw, seen, satisfied))
            self.assertIsNone(eng.hub.events[-1][1]["finalize_task_id"])

    def test_finalize_multi_crop_only_eligible(self):
        eng = make_engine(finalize_max_crops=5)
        _, meta, _ = add_track(eng, crops=[crop(10), crop(11), crop(12, lm_conf=0.2)], seen=30)
        eng._end_track("cam1", 5, meta)
        self.assertEqual(len(eng.hub.tasks[0]["crops"]["reg1"]), 2)

    def test_remove_camera_ends_live_tracks(self):
        eng = make_engine(leave_scene=False)
        add_track(eng, tid=1, crops=[crop(1)])
        add_track(eng, tid=2, crops=[crop(2)])
        eng.remove_camera("cam1")
        ended = [d for k, d in eng.hub.events if k == "track_ended"]
        self.assertEqual(sorted(d["track_id"] for d in ended), [1, 2])
        self.assertTrue(all(d["reason"] == "camera_removed" for d in ended))
        self.assertNotIn("cam1", eng.cameras)

    def test_trigger_detail_is_plain_json(self):
        import json
        d = engine_mod.Engine._trigger_detail({"direction": "positive_to_negative",
                                               "point": (np.int64(3), np.int64(4)),
                                               "confidence": np.float32(0.7), "track_id": 5})
        json.dumps(d)
        self.assertEqual(d["point"], [3, 4])
        self.assertNotIn("track_id", d)


if __name__ == "__main__":
    unittest.main()
