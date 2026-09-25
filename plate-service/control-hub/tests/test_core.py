"""
Unit tests for the PLATE control hub's pure core (core.py + policy.py),
with a fake clock — no Redis, no threads. Same state machine as face,
plate vocabulary, plus the plate-specific fixes (both triggers behave
alike, only VALID reads satisfy, a vehicle leaving mid-OCR waits for the
OCR, a record never goes out empty while OCR read something).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import protocol as P  # noqa: E402
from config import ModuleConfig  # noqa: E402
from core import HubCore  # noqa: E402
from policy import PlatePolicy  # noqa: E402

PLATE_TRIGGERS = {"periodic": False, "cross_line": True, "stop_roi": True, "leave_scene": True}


def plate_core(**kw):
    cfg = ModuleConfig(module="plate", **{"satisfied_conf": 0.85, **kw})
    return HubCore(cfg, PlatePolicy(cfg))


def start(core, uid="u1", t=0.0, triggers=None, engine=1):
    return core.handle(P.K_TRACK_STARTED, {
        "uid": uid, "camera_id": "cam1", "track_id": 7, "engine_id": engine, "boot_id": "b1",
        "video_source": "rtsp://relay/cam1",
        "triggers": triggers if triggers is not None else PLATE_TRIGGERS,
    }, t)


def plate_result(core, uid, task_id, stage, text, conf, valid=True, t=0.0):
    return core.handle(P.K_RESULT, {
        "uid": uid, "task_id": task_id, "stage": stage, "trigger_type": stage, "camera_id": "cam1",
        "track_id": 7, "status": "ok" if valid else "invalid", "plate_text": text, "confidence": conf,
        "is_valid": valid, "voted_class": 0,
        "payload": {"plate_image": f"vp/{task_id}.png", "frame_image": f"vf/{task_id}.png", "plate_type": 1},
    }, t)


def ctl_actions(fx):
    return [(m["action"], m.get("satisfied"), m.get("stage")) for _, m in fx.ctl]


class PlateCases(unittest.TestCase):
    def test_both_triggers_publish_immediately_when_satisfied(self):
        # previously cross_line published NOTHING here while stop_roi did
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.95, t=1)
        for ev in ("cross_line", "stop_roi"):
            fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": ev, "detail": {"direction": "a_to_b"}}, 2)
            self.assertEqual(len(fx.publish), 1, ev)
            p = fx.publish[0]
            self.assertEqual(p["update_type"], ev)
            self.assertFalse(p["is_final"])
            self.assertEqual(p["resolved"]["plate_text"], "12B34567")
            self.assertEqual(p["meta"]["event"]["direction"], "a_to_b")

    def test_invalid_high_confidence_read_does_not_stop_ocr(self):
        # previously a 0.9 INVALID read skipped every later OCR stage
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "1234", 0.93, valid=False, t=1)
        self.assertFalse(c.tracks["u1"]["satisfied"])
        plate_result(c, "u1", "t1", "stop_roi", "12B34567", 0.9, t=2)
        self.assertTrue(c.tracks["u1"]["satisfied"])

    def test_final_payload_keeps_previous_shape_and_adds_resolved(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "t0", "stage": "cross_line"}, 0.9)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "cross_line", "detail": {}, "task_id": "t0"}, 1)
        fx = plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.7, t=1.5)
        self.assertEqual(fx.publish[0]["update_type"], "cross_line")
        c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 20, "n_crops": 5, "finalize_task_id": "tf"}, 3)
        fx = plate_result(c, "u1", "tf", "leave_scene", "12B34567", 0.8, t=4)
        final = fx.publish[-1]
        self.assertTrue(final["is_final"])
        self.assertEqual(final["update_type"], "leave_scene")
        self.assertEqual(set(final["ocr_results"]), {"cross_line", "leave_scene"})
        self.assertEqual(set(final["events"]), {"cross_line", "leave_scene"})
        self.assertEqual(final["track_paths"]["leave_scene"]["plate_path"], "vp/tf.png")
        self.assertEqual(final["resolved"]["plate_text"], "12B34567")
        self.assertEqual(final["resolved"]["votes"], 2)
        self.assertEqual(final["stream_idx"], "cam1")
        self.assertEqual(final["process_id"], 1)

    def test_plate_periodic_now_implemented(self):
        c = plate_core(periodic_first_delay_sec=0.5)
        start(c, triggers={**PLATE_TRIGGERS, "periodic": True})
        self.assertEqual(ctl_actions(c.tick(0.6)), [("request", None, "periodic")])

    def test_lost_task_does_not_block_forever(self):
        c = plate_core(finalize_timeout_sec=10, periodic_first_delay_sec=0.5, periodic_interval_sec=3,
                       track_stale_sec=1000)
        start(c, triggers={**PLATE_TRIGGERS, "periodic": True})
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "lost", "stage": "cross_line"}, 0.1)
        self.assertEqual(ctl_actions(c.tick(1)), [])  # blocked by in-flight
        c.tick(31)                                    # 3 x finalize timeout -> freed
        c.handle(P.K_TRACK_UPDATE, {"uid": "u1"}, 33.9)
        self.assertEqual(ctl_actions(c.tick(34.1)), [("request", None, "periodic")])

    def test_vehicle_leaving_while_ocr_runs_waits_for_it(self):
        # THE old bug: leave_scene off, cross_line OCR still running when the
        # vehicle left -> the final was published with empty ocr_results and
        # the OCR answer was dropped as "late".
        c = plate_core()
        start(c, triggers={**PLATE_TRIGGERS, "leave_scene": False})
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "t0", "stage": "cross_line"}, 1)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "cross_line", "detail": {}, "task_id": "t0"}, 1)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 25, "n_crops": 4}, 1.5)
        self.assertEqual(fx.publish, [])                    # nothing empty goes out
        fx = plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.9, t=4)
        self.assertEqual([p["update_type"] for p in fx.publish], ["cross_line", "leave_scene"])
        final = fx.publish[-1]
        self.assertTrue(final["is_final"])
        self.assertTrue(final["complete"])
        self.assertEqual(final["resolved"]["plate_text"], "12B34567")
        self.assertEqual(final["ocr_results"]["cross_line"]["plate_text"], "12B34567")
        self.assertEqual(final["track_paths"]["cross_line"]["plate_path"], "vp/t0.png")
        self.assertNotIn(None, final["ocr_results"].values())

    def test_no_valid_read_still_reports_what_ocr_saw(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "12B345", 0.8, valid=False, t=1)
        plate_result(c, "u1", "t1", "stop_roi", "1234", 0.4, valid=False, t=2)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 25, "n_crops": 4}, 3)
        r = fx.publish[-1]["resolved"]
        self.assertFalse(r["is_valid"])
        self.assertEqual(r["plate_text"], "0")
        self.assertEqual(r["raw_text"], "12B345")          # best attempt, not empty
        self.assertEqual(r["plate_image"], "vp/t0.png")
        self.assertEqual(r["n_results"], 2)

    def test_failed_upload_image_filled_from_same_plate(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.7, t=1)
        c.handle(P.K_RESULT, {"uid": "u1", "task_id": "t1", "stage": "stop_roi", "status": "ok",
                              "plate_text": "12B34567", "confidence": 0.95, "is_valid": True, "voted_class": 0,
                              "payload": {"plate_image": None, "frame_image": None, "plate_type": 1}}, 2)
        r = c.policy.resolve(c.tracks["u1"])
        self.assertAlmostEqual(r["confidence"], 0.95)
        self.assertEqual(r["plate_image"], "vp/t0.png")
        self.assertEqual(r["frame_image"], "vf/t0.png")

    def test_ocr_error_result_does_not_break_anything(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        c.handle(P.K_RESULT, {"uid": "u1", "task_id": "t0", "stage": "cross_line", "status": "error",
                              "plate_text": "", "confidence": 0.0, "is_valid": False, "payload": None,
                              "error": "boom"}, 1)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 25, "n_crops": 4}, 2)
        r = fx.publish[-1]["resolved"]
        self.assertFalse(r["is_valid"])
        self.assertIsNone(r["plate_image"])

    def test_late_plate_read_republishes_final(self):
        c = plate_core(finalize_timeout_sec=10)
        start(c, triggers=PLATE_TRIGGERS)
        c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 25, "n_crops": 4, "finalize_task_id": "tf"}, 1)
        first = c.tick(11).publish[-1]
        self.assertFalse(first["resolved"]["is_valid"])
        again = plate_result(c, "u1", "tf", "leave_scene", "12B34567", 0.9, t=20).publish[-1]
        self.assertEqual(again["revision"], 2)
        self.assertEqual(again["resolved"]["plate_text"], "12B34567")
        self.assertEqual(again["update_type"], "leave_scene")


class PlateLifecycle(unittest.TestCase):
    """The shared state machine, exercised in plate vocabulary."""

    def test_deferred_trigger_times_out_when_no_crop_was_sent(self):
        c = plate_core(trigger_max_wait_sec=3)
        start(c, triggers=PLATE_TRIGGERS)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "stop_roi", "detail": {"duration": 3.1}, "task_id": None}, 1)
        self.assertEqual(c.tick(3.9).publish, [])
        fx = c.tick(4.0)
        self.assertEqual(fx.publish[0]["update_type"], "stop_roi")
        self.assertEqual(fx.publish[0]["meta"]["resolution"], "timeout")
        self.assertEqual(fx.publish[0]["meta"]["event"]["duration"], 3.1)

    def test_trigger_published_once(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.95, t=1)
        n = sum(len(c.handle(P.K_TRIGGER, {"uid": "u1", "event": "cross_line", "detail": {}}, t).publish)
                for t in (2, 3, 4))
        self.assertEqual(n, 1)

    def test_duplicate_result_ignored(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.7, t=1)
        plate_result(c, "u1", "t0", "cross_line", "12B34567", 0.7, t=1.1)
        self.assertEqual(len(c.tracks["u1"]["results"]), 1)

    def test_noise_track_dropped(self):
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 2, "n_crops": 1}, 1)
        self.assertEqual(fx.publish, [])

    def test_engine_restart_and_stale(self):
        c = plate_core(track_stale_sec=60)
        start(c, uid="a", engine=1)
        start(c, uid="b", engine=2)
        for u in ("a", "b"):
            c.handle(P.K_TRACK_UPDATE, {"uid": u, "seen_frames": 20, "n_crops": 3}, 1)
        c.handle(P.K_ENGINE_STARTED, {"engine_id": 1, "boot_id": "new"}, 2)
        self.assertEqual(c.tracks["a"]["end"]["reason"], "engine_restarted")
        fx = c.tick(62)
        self.assertEqual(fx.publish[-1]["meta"]["event"]["reason"], "stale")

    def test_checkpoint_roundtrip(self):
        import json
        c = plate_core()
        start(c, triggers=PLATE_TRIGGERS)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "cross_line", "detail": {"direction": "a_to_b"}}, 1)
        c2 = plate_core()
        c2.load([json.loads(json.dumps(c.tracks["u1"]))])
        fx = c2.tick(10)
        self.assertEqual(fx.publish[0]["meta"]["event"]["direction"], "a_to_b")


if __name__ == "__main__":
    unittest.main()
