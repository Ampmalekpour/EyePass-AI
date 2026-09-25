"""
Unit tests for the control hub's pure core (core.py + policy.py).

Every case from the recognition-flow analysis is pinned here, for both
modules, with a fake clock — no Redis, no threads:

  face  A  trigger with a satisfied identity      -> published at once
  face  B  trigger without one                    -> held for ITS result
  face  C  trigger whose crop never gets sent     -> published on timeout
  face  D  periodic re-query                      -> only when enabled, unsatisfied, idle
  face  E  track end                              -> waits for finalize result, gated final
  plate    same state machine, plate vocabulary + the three plate fixes
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import protocol as P  # noqa: E402
from config import ModuleConfig  # noqa: E402
from core import HubCore  # noqa: E402
from policy import FacePolicy, PlatePolicy  # noqa: E402

FACE_TRIGGERS = {"periodic": True, "line_cross": True, "stopped_roi": True, "leave_scene": True}
PLATE_TRIGGERS = {"periodic": False, "cross_line": True, "stop_roi": True, "leave_scene": True}


def face_core(**kw):
    cfg = ModuleConfig(module="face", **kw)
    return HubCore(cfg, FacePolicy(cfg))


def plate_core(**kw):
    cfg = ModuleConfig(module="plate", **{"satisfied_conf": 0.85, **kw})
    return HubCore(cfg, PlatePolicy(cfg))


def start(core, uid="u1", t=0.0, triggers=None, engine=1):
    return core.handle(P.K_TRACK_STARTED, {
        "uid": uid, "camera_id": "cam1", "track_id": 7, "engine_id": engine, "boot_id": "b1",
        "video_source": "rtsp://relay/cam1",
        "triggers": triggers if triggers is not None else FACE_TRIGGERS,
    }, t)


def face_result(uid, task_id, stage, pid, conf, valid=True, t=0.0, core=None):
    return core.handle(P.K_RESULT, {
        "uid": uid, "task_id": task_id, "stage": stage, "camera_id": "cam1", "track_id": 7,
        "status": "ok", "is_valid": valid, "confidence": conf, "personnelid": pid if valid else "0",
        "first_name": "Ali" if valid else "Unknown", "last_name": "R" if valid else "Unknown",
        "face_image": f"dynamics/KnownFaceImage/{task_id}.png", "camera_image": f"frame/{task_id}.png",
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


class FaceCases(unittest.TestCase):
    def test_case_a_satisfied_trigger_publishes_immediately_with_direction(self):
        c = face_core()
        start(c)
        fx = face_result("u1", "t0", "periodic", "42", 0.9, core=c, t=1)
        self.assertIn(("result", True, None), ctl_actions(fx))
        fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross",
                                    "detail": {"direction": "positive_to_negative"}, "task_id": None}, 2)
        self.assertEqual(len(fx.publish), 1)
        p = fx.publish[0]
        self.assertEqual(p["event_type"], "line_cross")
        self.assertFalse(p["is_final"])
        self.assertEqual(p["meta"]["identified_as"], "42")
        self.assertEqual(p["meta"]["event"]["direction"], "positive_to_negative")
        self.assertEqual(p["meta"]["resolution"], "immediate")
        self.assertEqual(p["result"]["personnelid"], "42")

    def test_case_b_trigger_waits_for_its_own_task_not_an_older_one(self):
        c = face_core()
        start(c)
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "tp", "stage": "periodic"}, 0.5)
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "tl", "stage": "line_cross"}, 1.0)
        fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {}, "task_id": "tl"}, 1.0)
        self.assertEqual(fx.publish, [])
        fx = face_result("u1", "tp", "periodic", "42", 0.4, core=c, t=1.2)   # older task: not enough
        self.assertEqual(fx.publish, [])
        fx = face_result("u1", "tl", "line_cross", "42", 0.5, core=c, t=1.4)
        self.assertEqual([p["event_type"] for p in fx.publish], ["line_cross"])
        self.assertEqual(fx.publish[0]["meta"]["resolution"], "after_recognition")
        # sum-vote: 0.4 + 0.5 for the same person, max conf reported
        self.assertEqual(fx.publish[0]["meta"]["identified_as"], "42")
        self.assertAlmostEqual(fx.publish[0]["meta"]["confidence"], 0.5)
        self.assertEqual(fx.publish[0]["meta"]["votes"], 2)

    def test_case_b_trigger_without_task_uses_next_result(self):
        c = face_core()
        start(c)
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "tp", "stage": "periodic"}, 0.5)
        fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": "stopped_roi", "detail": {"duration": 3.2},
                                    "task_id": None}, 1.0)
        self.assertEqual(fx.publish, [])
        fx = face_result("u1", "tp", "periodic", "0", 0.0, valid=False, core=c, t=1.3)
        self.assertEqual(fx.publish[0]["event_type"], "stopped_roi")
        self.assertEqual(fx.publish[0]["meta"]["identified_as"], "0")
        self.assertEqual(fx.publish[0]["meta"]["event"]["duration"], 3.2)

    def test_result_overtaking_its_trigger_publishes_at_once(self):
        c = face_core()
        start(c)
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "tl", "stage": "line_cross"}, 1.0)
        face_result("u1", "tl", "line_cross", "42", 0.5, core=c, t=1.1)   # result first
        fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {}, "task_id": "tl"}, 1.2)
        self.assertEqual([p["meta"]["resolution"] for p in fx.publish], ["after_recognition"])

    def test_case_c_deferred_trigger_times_out(self):
        c = face_core(trigger_max_wait_sec=3.0)
        start(c)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {}, "task_id": None}, 1.0)
        self.assertEqual(c.tick(3.9).publish, [])
        fx = c.tick(4.0)
        self.assertEqual(fx.publish[0]["meta"]["resolution"], "timeout")
        # never twice
        self.assertEqual(c.tick(9.0).publish, [])

    def test_trigger_published_once_per_event(self):
        c = face_core()
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.95, core=c, t=1)
        n = 0
        for t in (2, 3, 4):
            n += len(c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {}}, t).publish)
        self.assertEqual(n, 1)

    def test_disabled_trigger_is_ignored(self):
        c = face_core()
        start(c, triggers={**FACE_TRIGGERS, "stopped_roi": False})
        face_result("u1", "t0", "periodic", "42", 0.95, core=c, t=1)
        fx = c.handle(P.K_TRIGGER, {"uid": "u1", "event": "stopped_roi", "detail": {}}, 2)
        self.assertEqual(fx.publish, [])

    def test_case_d_periodic_requests(self):
        c = face_core(periodic_first_delay_sec=1.0, periodic_interval_sec=3.0)
        start(c, t=0)
        self.assertEqual(ctl_actions(c.tick(0.5)), [])
        self.assertEqual(ctl_actions(c.tick(1.0)), [("request", None, "periodic")])
        # in flight -> no request
        c.handle(P.K_SUBMITTED, {"uid": "u1", "task_id": "t1", "stage": "periodic"}, 1.1)
        self.assertEqual(ctl_actions(c.tick(4.1)), [])
        face_result("u1", "t1", "periodic", "42", 0.95, core=c, t=4.2)
        # satisfied -> no more requests ever
        self.assertEqual(ctl_actions(c.tick(20)), [])

    def test_periodic_off_when_camera_flag_off(self):
        c = face_core()
        start(c, triggers={**FACE_TRIGGERS, "periodic": False})
        self.assertEqual(ctl_actions(c.tick(10)), [])

    def test_case_e_finalize_waits_for_finalize_task(self):
        c = face_core(finalize_timeout_sec=10)
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.4, core=c, t=1)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "reason": "absent", "seen_frames": 40, "n_crops": 5,
                                        "finalize_task_id": "tf"}, 5)
        self.assertEqual(fx.publish, [])
        self.assertEqual(c.tick(6).publish, [])
        fx = face_result("u1", "tf", "finalize", "42", 0.8, core=c, t=7)
        finals = [p for p in fx.publish if p["is_final"]]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["event_type"], "finalize")
        self.assertEqual(finals[0]["meta"]["identified_as"], "42")
        self.assertAlmostEqual(finals[0]["meta"]["confidence"], 0.8)
        self.assertIn("finalize", finals[0]["meta"]["recognition_history"])

    def test_finalize_timeout_closes_anyway(self):
        c = face_core(finalize_timeout_sec=10)
        start(c)
        c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 40, "n_crops": 5, "finalize_task_id": "tf"}, 5)
        self.assertEqual(c.tick(14.9).publish, [])
        fx = c.tick(15.0)
        self.assertEqual(len(fx.publish), 1)
        self.assertEqual(fx.publish[0]["meta"]["identified_as"], "0")

    def test_noise_track_dropped_but_published_track_always_closed(self):
        c = face_core(final_min_seen_frames=8)
        start(c, uid="noise")
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "noise", "seen_frames": 2, "n_crops": 1}, 1)
        self.assertEqual(fx.publish, [])
        self.assertEqual(c.stats["dropped_tracks"], 1)

        start(c, uid="short")
        face_result("short", "t0", "periodic", "42", 0.9, core=c, t=1)
        c.handle(P.K_TRIGGER, {"uid": "short", "event": "line_cross", "detail": {}}, 1.1)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "short", "seen_frames": 3, "n_crops": 1}, 2)
        self.assertEqual([p["is_final"] for p in fx.publish], [True])

    def test_deferred_trigger_flushed_before_final(self):
        c = face_core()
        start(c)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {}, "task_id": None}, 1)
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 20, "n_crops": 2}, 1.5)
        self.assertEqual([(p["event_type"], p["is_final"]) for p in fx.publish],
                         [("line_cross", False), ("finalize", True)])
        self.assertEqual(fx.publish[0]["meta"]["resolution"], "track_ended")

    def test_late_and_duplicate_results(self):
        c = face_core(late_result_grace_sec=30)
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.9, core=c, t=1)
        n_before = len(c.tracks["u1"]["results"])
        self.assertEqual(face_result("u1", "t0", "periodic", "42", 0.9, core=c, t=1.1).ctl, [])
        self.assertEqual(len(c.tracks["u1"]["results"]), n_before)
        c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 20, "n_crops": 2}, 2)
        fx = face_result("u1", "t9", "periodic", "43", 0.99, core=c, t=3)
        self.assertEqual(fx.publish, [])
        self.assertEqual(c.stats["late_results"], 1)
        fx = c.tick(40)
        self.assertIn("u1", fx.delete)
        self.assertNotIn("u1", c.tracks)

    def test_vote_prefers_repeated_identity_over_single_spike(self):
        c = face_core(satisfied_conf=0.95, consensus_min=0)
        start(c)
        face_result("u1", "a", "periodic", "7", 0.6, core=c, t=1)
        face_result("u1", "b", "periodic", "9", 0.5, core=c, t=2)
        face_result("u1", "c", "periodic", "9", 0.5, core=c, t=3)
        d = c.policy.resolve(c.tracks["u1"])
        self.assertEqual(d["personnelid"], "9")
        self.assertEqual(d["candidates"], 2)

    def test_consensus_satisfies(self):
        c = face_core(satisfied_conf=0.95, consensus_min=3)
        start(c)
        for i, t in enumerate((1, 2)):
            face_result("u1", f"t{i}", "periodic", "7", 0.5, core=c, t=t)
        self.assertFalse(c.tracks["u1"]["satisfied"])
        fx = face_result("u1", "t2", "periodic", "7", 0.5, core=c, t=3)
        self.assertTrue(c.tracks["u1"]["satisfied"])
        self.assertIn(("result", True, None), ctl_actions(fx))

    def test_invalid_results_never_identify(self):
        c = face_core()
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.99, valid=False, core=c, t=1)
        self.assertFalse(c.tracks["u1"]["satisfied"])
        self.assertEqual(c.policy.resolve(c.tracks["u1"])["personnelid"], "0")

    def test_liveness_reject_policy(self):
        c = face_core(liveness_policy="reject")
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.95, core=c, t=1)
        c.handle(P.K_TRACK_UPDATE, {"uid": "u1", "seen_frames": 30,
                                         "liveness": {"liveness": "fake", "liveness_reason": "printed_photo"}}, 2)
        self.assertTrue(c.tracks["u1"]["satisfied"])
        fx = c.handle(P.K_TRACK_ENDED, {"uid": "u1", "seen_frames": 30, "n_crops": 3}, 3)
        payload = fx.publish[-1]
        self.assertTrue(payload["is_final"])
        self.assertEqual(payload["meta"]["identified_as"], "0")
        self.assertTrue(payload["meta"]["spoof_rejected"])
        self.assertEqual(payload["meta"]["liveness"], "fake")

    def test_liveness_annotate_policy_keeps_identity(self):
        c = face_core(liveness_policy="annotate")
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.95, core=c, t=1)
        c.handle(P.K_TRACK_UPDATE, {"uid": "u1", "liveness": {"liveness": "fake"}}, 2)
        payload = c.policy.final_payload(c.tracks["u1"], 3)
        self.assertEqual(payload["meta"]["identified_as"], "42")
        self.assertEqual(payload["meta"]["liveness"], "fake")

    def test_engine_restart_ends_its_tracks(self):
        c = face_core()
        start(c, uid="a", engine=1)
        start(c, uid="b", engine=2)
        c.handle(P.K_TRACK_UPDATE, {"uid": "a", "seen_frames": 20, "n_crops": 2}, 1)
        c.handle(P.K_ENGINE_STARTED, {"engine_id": 1, "boot_id": "b2"}, 2)
        self.assertTrue(c.tracks["a"]["ended"])
        self.assertEqual(c.tracks["a"]["end"]["reason"], "engine_restarted")
        self.assertFalse(c.tracks["b"]["ended"])

    def test_stale_track_is_ended(self):
        c = face_core(track_stale_sec=120)
        start(c, t=0)
        c.handle(P.K_TRACK_UPDATE, {"uid": "u1", "seen_frames": 20, "n_crops": 2}, 1)
        c.tick(100)
        self.assertFalse(c.tracks["u1"]["ended"])
        fx = c.tick(121)
        self.assertTrue(c.tracks["u1"]["closed"])
        self.assertEqual(fx.publish[0]["meta"]["event"]["reason"], "stale")

    def test_result_before_track_started(self):
        c = face_core()
        face_result("u1", "t0", "line_cross", "42", 0.9, core=c, t=1)
        fx = start(c, t=1.1)
        self.assertEqual(c.tracks["u1"]["camera_id"], "cam1")
        self.assertIn(("state", True, None), ctl_actions(fx))

    def test_checkpoint_roundtrip(self):
        import json
        c = face_core()
        start(c)
        face_result("u1", "t0", "periodic", "42", 0.5, core=c, t=1)
        c.handle(P.K_TRIGGER, {"uid": "u1", "event": "line_cross", "detail": {"direction": "x"}}, 2)
        snap = json.loads(json.dumps(c.tracks["u1"]))
        c2 = face_core()
        c2.load([snap])
        fx = c2.tick(10)
        self.assertEqual(fx.publish[0]["event_type"], "line_cross")
        self.assertEqual(fx.publish[0]["meta"]["event"]["direction"], "x")


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


if __name__ == "__main__":
    unittest.main()
