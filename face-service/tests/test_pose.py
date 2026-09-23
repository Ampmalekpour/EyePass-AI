"""
test_pose.py
--------------------------------------------------------------------
Exercises the pure pose-math in recognizer/src/pose.py directly — no
Redis, no torch, no face_alignment package needed; only cv2/numpy/PIL,
all available in this sandbox. Mirrors tests/test_triggers.py's own
"import the real module, feed it synthetic geometry" approach.

Deliberately does NOT test detect_face_5pt()/verify_pose() end-to-end
— those need the operator-supplied face_alignment package (MTCNN
weights), which is bind-mounted at deploy time and not part of this
repo/sandbox. See enroll_tools.py for an end-to-end check against a
running stack instead.

Run with:
    PYTHONPATH=recognizer/src python3 tests/test_pose.py
--------------------------------------------------------------------
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recognizer", "src"))

import config  # noqa: E402
import pose  # noqa: E402


# A frontal, upright synthetic 5-point landmark set (pixel coords in a
# 200x200 frame) — eyes level, nose centered, mouth level.
FRONTAL_LANDMARKS = [
    [70.0, 80.0],   # left eye
    [130.0, 80.0],  # right eye
    [100.0, 110.0], # nose
    [75.0, 140.0],  # left mouth
    [125.0, 140.0], # right mouth
]
FRAME_SHAPE = (200, 200, 3)


def _shift_x(points, dx):
    return [[x + dx, y] for x, y in points]


class ComputeYawTests(unittest.TestCase):
    def test_frontal_landmarks_yield_near_zero_yaw(self):
        yaw = pose.compute_yaw(FRONTAL_LANDMARKS, FRAME_SHAPE)
        self.assertLess(abs(yaw), 15.0, f"expected a near-frontal yaw, got {yaw}")

    def test_malformed_input_returns_zero_not_an_exception(self):
        # Any failure inside compute_yaw (wrong point count, bad shape,
        # solvePnP rejecting the geometry) must degrade to 0.0, never
        # raise — verify_pose()'s caller depends on this being total.
        # Note duplicate/degenerate *coordinates* are NOT guaranteed to
        # make solvePnP itself fail (OpenCV can still return a — bogus
        # but "successful" — solution for them); a malformed *shape*
        # (wrong number of points) is what reliably exercises the
        # except-branch, via the reshape() calls inside compute_yaw.
        too_few_points = [[70.0, 80.0], [130.0, 80.0]]
        yaw = pose.compute_yaw(too_few_points, FRAME_SHAPE)
        self.assertEqual(yaw, 0.0)


class ComputePitchTests(unittest.TestCase):
    def test_frontal_landmarks_yield_a_finite_pitch(self):
        pitch = pose.compute_pitch_ratio(FRONTAL_LANDMARKS)
        self.assertIsNotNone(pitch)
        self.assertLess(abs(pitch), 90.0)

    def test_zero_nose_to_mouth_distance_returns_none(self):
        degenerate = [
            [70.0, 80.0], [130.0, 80.0], [100.0, 110.0],
            [75.0, 110.0], [125.0, 110.0],  # mouth == nose height
        ]
        self.assertIsNone(pose.compute_pitch_ratio(degenerate))


class CheckPoseApprovalTests(unittest.TestCase):
    def test_frontal_pose_approved_for_flag_2(self):
        yaw_status, pitch_status = pose.check_pose_approval(yaw=0.0, pitch=0.0, flag=2)
        self.assertEqual(yaw_status, "Approved")
        self.assertEqual(pitch_status, "Approved")

    def test_frontal_pose_rejected_for_flag_1_right_profile(self):
        # flag=1 expects a strong rightward yaw (see config.ENROLL_YAW_WINDOWS);
        # a frontal 0-degree yaw must NOT satisfy it.
        yaw_status, _ = pose.check_pose_approval(yaw=0.0, pitch=0.0, flag=1)
        self.assertEqual(yaw_status, "Not Approved")

    def test_each_flag_window_is_satisfied_by_its_own_midpoint(self):
        for flag, (lo, hi) in config.ENROLL_YAW_WINDOWS.items():
            midpoint = (lo + hi) / 2.0
            yaw_status, _ = pose.check_pose_approval(yaw=midpoint, pitch=0.0, flag=flag)
            self.assertEqual(yaw_status, "Approved", f"flag {flag} midpoint {midpoint} should be Approved")

    def test_pitch_outside_window_always_rejects_regardless_of_flag(self):
        lo, hi = config.ENROLL_PITCH_WINDOW
        for flag in config.ENROLL_YAW_WINDOWS:
            _, pitch_status = pose.check_pose_approval(yaw=0.0, pitch=hi + 20.0, flag=flag)
            self.assertEqual(pitch_status, "Not Approved")


class RenderPoseDebugImageTests(unittest.TestCase):
    """render_pose_debug_image() is pure (no MTCNN, no I/O) — exactly
    the kind of function this sandbox CAN exercise end-to-end, unlike
    detect_face_5pt()/verify_pose() which need the real face_alignment
    weights (see module docstring)."""

    def _frame(self):
        import numpy as np
        return (np.random.rand(200, 200, 3) * 255).astype("uint8")

    def test_returns_an_image_the_same_size_as_the_input(self):
        frame = self._frame()
        bbox = pose.np.array([40.0, 40.0, 160.0, 160.0, 0.98])
        out = pose.render_pose_debug_image(
            frame, bbox, FRONTAL_LANDMARKS, 2.0, 1.0, "Approved", "Approved", flag=2
        )
        self.assertEqual(out.shape, frame.shape)

    def test_handles_no_face_detected_case_without_raising(self):
        frame = self._frame()
        out = pose.render_pose_debug_image(frame, None, None, None, None, "N/A", "N/A", flag=1)
        self.assertEqual(out.shape, frame.shape)

    def test_never_raises_even_with_a_malformed_bbox(self):
        frame = self._frame()
        bad_bbox = pose.np.array([1.0, 2.0])  # too short to unpack as x1,y1,x2,y2,score
        # render_pose_debug_image itself has no try/except (verify_pose
        # wraps it) — this test documents that a malformed bbox raises
        # a plain, catchable exception rather than corrupting the frame
        # or hanging, which is what verify_pose()'s own try/except relies on.
        with self.assertRaises(Exception):
            pose.render_pose_debug_image(frame, bad_bbox, FRONTAL_LANDMARKS, 0.0, 0.0,
                                         "Approved", "Approved", flag=2)


class CropWithPaddingTests(unittest.TestCase):
    def test_crop_is_padded_and_clipped_to_frame_bounds(self):
        import numpy as np
        frame = (np.random.rand(200, 200, 3) * 255).astype("uint8")
        bbox = np.array([50.0, 50.0, 100.0, 100.0, 0.99])  # 50x50 box
        crop = pose.crop_with_padding(frame, bbox, padding_ratio=0.15)
        # 50 * 0.15 = 7.5 padding per side -> ~65x65, clipped inside [0,200)
        self.assertGreater(crop.shape[0], 50)
        self.assertGreater(crop.shape[1], 50)
        self.assertLessEqual(crop.shape[0], 200)
        self.assertLessEqual(crop.shape[1], 200)

    def test_crop_near_frame_edge_clips_without_erroring(self):
        import numpy as np
        frame = (np.random.rand(100, 100, 3) * 255).astype("uint8")
        bbox = np.array([0.0, 0.0, 20.0, 20.0, 0.99])
        crop = pose.crop_with_padding(frame, bbox, padding_ratio=0.5)
        self.assertGreater(crop.size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
