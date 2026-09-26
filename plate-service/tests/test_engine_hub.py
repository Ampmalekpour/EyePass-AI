"""
test_engine_hub.py
--------------------------------------------------------------------
The plate detector engine's side of the control-hub contract, on the
REAL Engine methods (an Engine assembled with object.__new__; no
YOLO/torch):

  * a trigger submits all top-N crops + best frame, tagged with the
    track uid and a task_id the hub can match
  * no submission while in flight / once the hub says satisfied / when
    the crop set is unchanged
  * the OCR in-flight lock EXPIRES (SUBMIT_TIMEOUT_SEC) — previously a
    lost OCR result blocked every later trigger on that track forever
  * track end: leave_scene OCR only when enabled and not satisfied,
    then track_ended, then the engine forgets the track
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

import config  # noqa: E402
import engine as engine_mod  # noqa: E402
from platecore.hub import TrackRecState  # noqa: E402


class FakeHub:
    def __init__(self):
        self.events, self.tasks, self.ctl = [], [], []
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


def crop(fid, score=0.9):
    return {"image": np.full((40, 120, 3), 100, dtype=np.uint8), "class_flag": 0, "det_score": score,
            "resolution": 4800, "aspect_ratio": 3.0, "sharpness": 300.0, "frame_number": fid}


def make_engine(leave_scene=True):
    eng = object.__new__(engine_mod.Engine)
    eng.engine_id = 3
    eng.logger = logging.getLogger("test-plate-engine")
    eng.hub = FakeHub()
    eng._uid_index = {}
    eng.STAGE_PRIORITY = {"leave_scene": 3, "cross_line": 2, "stop_roi": 2, "periodic": 1}
    eng.MIN_SEEN_FRAMES = 8
    eng.MIN_CROPS_TO_FINALIZE = 1
    eng.writers = {}
    eng.status_queue = type("Q", (), {"put_nowait": lambda self, m: None})()
    eng.cameras = {"gate": {
        "camera_id": "gate", "url": "rtsp://relay/gate", "track_meta": {}, "rec_state": {},
        "best_crops": {}, "best_frame": {}, "trigger_states": {}, "debug": None,
        "reader": type("R", (), {"stop": lambda self: None})(),
        "triggers": {"cond_per_trig": False, "cross_line_trig": True, "stop_roi_trig": True,
                     "leave_scene_trig": leave_scene},
    }}
    return eng


def add_track(eng, tid=9, crops=(), seen=20):
    cam = eng.cameras["gate"]
    uid = f"gate-3-{tid}"
    cam["track_meta"][tid] = {"uid": uid, "first_seen_fid": 1, "last_seen_fid": seen, "seen_frames": seen,
                              "events": {}}
    cam["rec_state"][tid] = TrackRecState(uid)
    cam["best_crops"][tid] = list(crops)
    eng._uid_index[uid] = ("gate", tid)
    return uid, cam["track_meta"][tid], cam["rec_state"][tid]


class PlateEngineHubTests(unittest.TestCase):
    def test_trigger_submits_all_top_crops(self):
        eng = make_engine()
        uid, meta, rs = add_track(eng, crops=[crop(5), crop(4), crop(3)])
        rs.request("cross_line", 0)
        tid = eng._try_submit("gate", 9, meta, rs, 0)
        task = eng.hub.tasks[0]
        self.assertEqual(task["task_id"], tid)
        self.assertEqual(task["trigger_type"], "cross_line")
        self.assertEqual(task["track_uid"], uid)
        self.assertEqual(len(task["crops"]), 3)
        self.assertEqual(eng.hub.kinds(), ["submitted"])
        # OCR worker's queue-latency parsing still works on the new id
        self.assertTrue(str(tid).rsplit(":", 1)[-1].isdigit())

    def test_in_flight_lock_expires(self):
        eng = make_engine()
        _, meta, rs = add_track(eng, crops=[crop(5)])
        rs.request("cross_line", 0)
        eng._try_submit("gate", 9, meta, rs, 0)
        eng.cameras["gate"]["best_crops"][9].insert(0, crop(6))
        rs.request("stop_roi", 1)
        self.assertIsNone(eng._try_submit("gate", 9, meta, rs, 1))
        later = config.SUBMIT_TIMEOUT_SEC + 0.1
        self.assertIsNotNone(eng._try_submit("gate", 9, meta, rs, later))
        self.assertEqual(eng.hub.tasks[-1]["stage"], "stop_roi")

    def test_unchanged_crop_set_not_resent(self):
        eng = make_engine()
        _, meta, rs = add_track(eng, crops=[crop(5)])
        rs.request("cross_line", 0)
        tid = eng._try_submit("gate", 9, meta, rs, 0)
        rs.apply_ctl({"action": "result", "task_id": tid, "satisfied": False}, 0.2)
        rs.request("stop_roi", 1)
        self.assertIsNone(eng._try_submit("gate", 9, meta, rs, 1))

    def test_satisfied_skips_leave_scene_ocr(self):
        eng = make_engine(leave_scene=True)
        uid, meta, rs = add_track(eng, crops=[crop(5)], seen=30)
        eng.hub.ctl = [{"action": "result", "uid": uid, "task_id": "x", "satisfied": True,
                        "display": {"label": "12B34567", "confidence": 0.93,
                                    "rows": [["cross_line", "12B34567 0.93 OK"]]}}]
        eng._drain_hub_ctl()
        self.assertEqual(engine_mod.Engine._ocr_rows_for_track(eng.cameras["gate"], 9),
                         [("cross_line", "12B34567 0.93 OK")])
        eng._end_track("gate", 9, meta)
        self.assertEqual(eng.hub.kinds(), ["track_ended"])
        self.assertIsNone(eng.hub.events[-1][1]["finalize_task_id"])

    def test_leave_scene_ocr_when_not_satisfied(self):
        eng = make_engine(leave_scene=True)
        uid, meta, rs = add_track(eng, crops=[crop(5), crop(4)], seen=30)
        eng._end_track("gate", 9, meta)
        self.assertEqual(eng.hub.kinds(), ["submitted", "track_ended"])
        self.assertEqual(eng.hub.tasks[0]["stage"], "leave_scene")
        self.assertEqual(eng.hub.events[-1][1]["finalize_task_id"], eng.hub.tasks[0]["task_id"])
        self.assertNotIn(9, eng.cameras["gate"]["track_meta"])
        self.assertNotIn(uid, eng._uid_index)

    def test_remove_camera_ends_tracks(self):
        eng = make_engine(leave_scene=False)
        add_track(eng, tid=1, crops=[crop(1)])
        add_track(eng, tid=2, crops=[crop(2)])
        eng.remove_camera("gate")
        ended = [d for k, d in eng.hub.events if k == "track_ended"]
        self.assertEqual(sorted(d["track_id"] for d in ended), [1, 2])
        self.assertNotIn("gate", eng.cameras)

    def test_hub_trigger_flag_names(self):
        flags = engine_mod.Engine._hub_triggers(make_engine().cameras["gate"])
        self.assertEqual(flags, {"periodic": False, "cross_line": True, "stop_roi": True, "leave_scene": True})


class FramePixelsTests(unittest.TestCase):
    """Line / stop-ROI coordinates: fractions and pixels both land on the
    frame (pixels used to be scaled a second time, so triggers never fired)."""

    def test_pixels_kept(self):
        pts, mode = engine_mod.to_frame_pixels(((100, 400), (900, 400)), 1920, 1080)
        self.assertEqual(mode, "pixels")
        self.assertEqual(pts, ((100.0, 400.0), (900.0, 400.0)))

    def test_fractions_scaled(self):
        pts, mode = engine_mod.to_frame_pixels(((0.1, 0.5), (0.9, 0.5)), 1920, 1080)
        self.assertEqual(mode, "fractions")
        self.assertEqual(pts, ((192.0, 540.0), (1728.0, 540.0)))

    def test_unconfigured_stays_zero(self):
        pts, _ = engine_mod.to_frame_pixels(((0, 0), (0, 0), (0, 0), (0, 0)), 1920, 1080)
        self.assertEqual(pts, ((0.0, 0.0),) * 4)

    def test_garbage_is_harmless(self):
        pts, mode = engine_mod.to_frame_pixels((("a", None), (1, 2)), 1920, 1080)
        self.assertEqual(mode, "invalid")


if __name__ == "__main__":
    unittest.main()
