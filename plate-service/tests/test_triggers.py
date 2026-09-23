"""
test_triggers.py
--------------------------------------------------------------------
Exercises the spatial trigger math in detector/src/triggers.py
(line-crossing, ROI entry/exit, stopped-in-ROI) directly — no Redis,
no torch, no camera needed; only cv2/numpy. Adapted from face_service's
own tests/test_triggers.py; the math under test is verbatim from the
reference video_processor.py (only the anchor point differs from
face's own trigger test — bbox_bottom_center here, a head centroid
there — but process_track_triggers() itself only ever receives already
-computed points in these tests, so that difference doesn't show up
below).

Run with:
    PYTHONPATH=detector/src python3 tests/test_triggers.py
--------------------------------------------------------------------
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

import config  # noqa: E402
from triggers import TriggerTrackState, process_track_triggers, bbox_bottom_center  # noqa: E402


NO_LINE = ((0, 0), (0, 0))
NO_ROI = ()


class BboxAnchorTests(unittest.TestCase):
    def test_bottom_center_is_correct_for_a_vehicle_box(self):
        # x1,y1,x2,y2 = 10,20,110,220 -> bottom-center = (60, 220)
        self.assertEqual(bbox_bottom_center(10, 20, 110, 220), (60, 220))


class LineCrossingTests(unittest.TestCase):
    def setUp(self):
        self.states = {}
        # Vertical line at x=100 from (100,0) to (100,200): points with
        # x<100 are on one side, x>100 on the other.
        self.line = ((100, 0), (100, 200))

    def _feed(self, track_id, x, y, score=0.9):
        return process_track_triggers(
            self.states, self.line, NO_ROI, camera_id="cam1",
            track_id=track_id, track_class=0, score=score,
            bbox_roi=(x - 5, y - 5, x + 5, y + 5), roi_offset=(0, 0, 0, 0),
        )

    def test_no_crossing_while_track_too_young(self):
        events = None
        for _ in range(config.MIN_TRACK_AGE_FOR_CROSSING - 1):
            events = self._feed(1, x=50, y=100)
        self.assertIsNone(events["line_cross"])

    def test_crossing_fires_once_when_track_moves_across_the_line(self):
        for _ in range(config.MIN_TRACK_AGE_FOR_CROSSING + 2):
            events = self._feed(1, x=50, y=100)
        self.assertIsNone(events["line_cross"])

        events = self._feed(1, x=150, y=100)
        self.assertIsNotNone(events["line_cross"])
        self.assertEqual(events["line_cross"]["direction"], "positive_to_negative")

        events = self._feed(1, x=50, y=100)
        self.assertIsNone(events["line_cross"], "CROSSING_COOLDOWN_FRAMES must suppress an immediate re-cross")

    def test_low_confidence_track_never_crosses(self):
        for _ in range(config.MIN_TRACK_AGE_FOR_CROSSING + 2):
            events = self._feed(2, x=50, y=100, score=0.01)
        events = self._feed(2, x=150, y=100, score=0.01)
        self.assertIsNone(events["line_cross"], "avg confidence below MIN_CONFIDENCE_FOR_CROSSING must never fire a crossing")

    def test_independent_tracks_have_independent_state(self):
        for _ in range(config.MIN_TRACK_AGE_FOR_CROSSING + 2):
            self._feed(10, x=50, y=100)
            self._feed(20, x=150, y=100)  # track 20 starts on the OTHER side

        ev10 = self._feed(10, x=150, y=100)  # track 10 crosses right
        ev20 = self._feed(20, x=50, y=100)   # track 20 crosses left, independently

        self.assertIsNotNone(ev10["line_cross"])
        self.assertIsNotNone(ev20["line_cross"])
        self.assertNotEqual(ev10["line_cross"]["direction"], ev20["line_cross"]["direction"])


class RoiTests(unittest.TestCase):
    def setUp(self):
        self.states = {}
        # A simple square ROI from (0,0) to (100,100).
        self.roi = ((0, 0), (100, 0), (100, 100), (0, 100))

    def _feed(self, track_id, x, y, score=0.9):
        return process_track_triggers(
            self.states, NO_LINE, self.roi, camera_id="cam1",
            track_id=track_id, track_class=0, score=score,
            bbox_roi=(x - 5, y - 5, x + 5, y + 5), roi_offset=(0, 0, 0, 0),
        )

    def test_roi_entry_requires_confirmation_frames(self):
        events = None
        for i in range(config.ROI_ENTRY_CONFIRMATION_FRAMES - 1):
            events = self._feed(1, x=50, y=50)
        self.assertIsNone(events["roi_entry"], "roi_entry must not fire before ROI_ENTRY_CONFIRMATION_FRAMES")

        events = self._feed(1, x=50, y=50)
        self.assertIsNotNone(events["roi_entry"])

    def test_roi_entry_fires_once_then_exit_on_leaving(self):
        fired = [self._feed(1, x=50, y=50) for _ in range(config.ROI_ENTRY_CONFIRMATION_FRAMES)]
        self.assertIsNotNone(fired[-1]["roi_entry"])
        self.assertTrue(all(e["roi_entry"] is None for e in fired[:-1]))

        events = self._feed(1, x=51, y=51)
        self.assertIsNone(events["roi_entry"])
        self.assertIsNone(events["roi_exit"])

        events = self._feed(1, x=500, y=500)
        self.assertIsNotNone(events["roi_exit"])

    def test_point_outside_roi_never_enters(self):
        for _ in range(config.ROI_ENTRY_CONFIRMATION_FRAMES + 2):
            events = self._feed(1, x=500, y=500)
        self.assertIsNone(events["roi_entry"])
        self.assertIsNone(events["roi_exit"])

    def test_stopped_roi_fires_when_stationary_long_enough(self):
        state = TriggerTrackState()
        self.states[1] = state

        for _ in range(config.ROI_ENTRY_CONFIRMATION_FRAMES + 1):
            self._feed(1, x=50, y=50)

        import time as time_mod
        now = time_mod.time()
        state.roi_entry_time = now - (config.STOP_TIME_SECONDS + 1)
        state.roi_position_buffer = [(50, 50)] * config.STOP_MIN_SAMPLES
        state.roi_timestamp_buffer = [
            state.roi_entry_time + i for i in range(config.STOP_MIN_SAMPLES)
        ]

        events = self._feed(1, x=50, y=50)
        self.assertIsNotNone(events["stopped_roi"], "a stationary track past STOP_TIME_SECONDS must fire stopped_roi")
        self.assertLessEqual(events["stopped_roi"]["velocity"], config.STOP_VELOCITY_THRESHOLD)

        events = self._feed(1, x=50, y=50)
        self.assertIsNone(events["stopped_roi"])


if __name__ == "__main__":
    unittest.main()
