"""Pipeline events in the debug video: toasts, markers, the lit-up line, hub notices."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector", "src"))

from tests.fakes import stub_modules  # noqa: E402
stub_modules.install()

import config  # noqa: E402
import engine as engine_mod  # noqa: E402
from debug_recorder import DebugConfig, DebugRecorder  # noqa: E402


def _recorder(tmp):
    old = (config.DEBUG_VIDEO_ENABLED, config.DEBUG_VIDEO_DIR)
    config.DEBUG_VIDEO_ENABLED, config.DEBUG_VIDEO_DIR = True, tmp
    try:
        return DebugRecorder("1", 0, DebugConfig())
    finally:
        config.DEBUG_VIDEO_ENABLED, config.DEBUG_VIDEO_DIR = old


CTX = {"fid": 1, "frame_wh": (640, 360), "roi_px": (0, 0, 640, 360), "line_px": ((100, 200), (500, 200)),
       "trig_cross": True, "tracks": [], "detections": [], "n_live": 0, "n_ghost": 0, "triggers": {}}


class DebugEventTests(unittest.TestCase):
    def test_event_with_a_point_leaves_a_marker_and_lights_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _recorder(tmp)
            frame = np.zeros((360, 640, 3), np.uint8)
            quiet = rec._render(frame, dict(CTX)).copy()
            rec.log("LINE_CROSS", "#3 negative_to_positive -> hub", track_id=3, point=(300, 120))
            self.assertEqual(len(rec.markers), 1)
            self.assertGreater(rec.line_flash, 0)
            shown = rec._render(frame, dict(CTX))
            self.assertFalse(np.array_equal(quiet, shown))
            self.assertTrue(shown[120:130, 300:312].any())           # the marker ring / dot
            self.assertTrue(shown[290:360, 0:300].any())             # the toast, bottom-left
            for _ in range(rec.cfg.marker_frames + 1):
                rec._decay_flashes()
            self.assertEqual(rec.markers, [])
            self.assertEqual(rec.line_flash, 0)
            rec.close()

    def test_new_event_kinds_have_their_own_colour(self):
        from debug_recorder import EVENT_COLORS
        for k in ("TRACK_NEW", "LINE_CROSS", "ROI_ENTRY", "ROI_EXIT", "STOPPED", "SATISFIED", "PUBLISH", "FINAL", "DROPPED"):
            self.assertIn(k, EVENT_COLORS)


class LineLabelTests(unittest.TestCase):
    def test_pos_label_is_on_the_side_the_detector_calls_positive(self):
        from debug_recorder import line_side_label_points
        from triggers import point_side_of_line
        for p1, p2 in (((1225, 403), (1823, 409)), ((1823, 409), (1225, 403)), ((100, 50), (120, 400)), ((300, 300), (50, 20))):
            pos_pt, neg_pt = line_side_label_points(p1, p2)
            self.assertGreater(point_side_of_line(pos_pt, p1, p2), 0, (p1, p2))
            self.assertLess(point_side_of_line(neg_pt, p1, p2), 0, (p1, p2))


class HubNoticeTests(unittest.TestCase):
    def _engine(self):
        eng = object.__new__(engine_mod.Engine)
        self.lines, self.recorded = [], []

        class L:
            def info(s, m, *a, **k): self.lines.append(m)
        eng.logger = L()
        eng.cameras = {"1": {"track_meta": {7: {}}}}
        eng._dbg_log = lambda cid, kind, text, **kw: self.recorded.append((cid, kind, text, kw.get("track_id")))
        return eng

    def test_published_final_dropped_late_are_shown(self):
        eng = self._engine()
        eng._show_hub_notice({"action": "published", "camera_id": "1", "track_id": 7, "event": "cross_line",
                              "resolution": "immediate", "label": "12B34567", "confidence": 0.93})
        eng._show_hub_notice({"action": "final", "camera_id": "1", "track_id": 9, "reason": "absent",
                              "revision": 1, "complete": True, "label": "12B34567", "confidence": 0.93})
        eng._show_hub_notice({"action": "dropped", "camera_id": "1", "track_id": 9, "why": "seen 1/8 frames"})
        eng._show_hub_notice({"action": "late_result", "camera_id": "1", "track_id": 9, "revision": 2, "label": "X"})
        kinds = [r[1] for r in self.recorded]
        self.assertEqual(kinds, ["PUBLISH", "FINAL", "DROPPED", "LATE_RESULT"])
        self.assertEqual(self.recorded[0][3], 7)          # track still alive -> its box flashes
        self.assertIsNone(self.recorded[1][3])            # track already gone -> panel/toast only
        self.assertEqual(len(self.lines), 4)


if __name__ == "__main__":
    unittest.main()
