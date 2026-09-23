"""
triggers.py
--------------------------------------------------------------------
Spatial trigger detection: line-crossing, ROI entry/exit, and
"stopped inside ROI". Extracted from Engine._process_track_triggers in
the reference video_processor.py into a standalone function so it can
be exercised in isolation (see tests/test_triggers.py) and so Engine
itself stays focused on the detect -> track -> dispatch loop.

The math (which side of the line, point-in-polygon, sliding-window
velocity) and the threshold constants are unchanged from the reference
pipeline — only the packaging is different.
--------------------------------------------------------------------
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import config


class TriggerTrackState:
    def __init__(self):
        self.frame_count = 0
        self.side_of_line = None
        self.frames_since_last_cross = 999
        self.inside_roi_frames = 0
        self.roi_confirmed = False
        self.roi_entry_reported = False
        self.stop_reported = False
        self.roi_entry_time = None
        self.roi_position_buffer = []
        self.roi_timestamp_buffer = []
        self.confidences = []

    def add_observation(self, point, ts, score):
        self.frame_count += 1
        self.frames_since_last_cross += 1
        self.confidences.append(score)
        if len(self.confidences) > 30:
            self.confidences.pop(0)

    def get_avg_confidence(self):
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def reset_roi_state(self):
        self.roi_confirmed = False
        self.roi_entry_reported = False
        self.stop_reported = False
        self.roi_entry_time = None
        self.roi_position_buffer.clear()
        self.roi_timestamp_buffer.clear()


def point_side_of_line(point, line_p1, line_p2):
    """Returns positive/negative depending on which side of the line the point is on."""
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def get_side_label(value):
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    return None


def point_in_polygon(point, polygon):
    """polygon: numpy array Nx2 of type np.int32"""
    if polygon is None or len(polygon) < 3:
        return False
    pt = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


def normalize_line_points(line_points):
    try:
        p1 = line_points[0]
        p2 = line_points[1]
        return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    except Exception:
        return (0, 0), (0, 0)


def normalize_stop_roi(stop_roi):
    try:
        pts = []
        for p in stop_roi:
            pts.append((int(p[0]), int(p[1])))
        return np.array(pts, dtype=np.int32)
    except Exception:
        return np.array([], dtype=np.int32)


def process_track_triggers(
    trigger_states: Dict[int, TriggerTrackState],
    line_points: Tuple[Tuple[int, int], Tuple[int, int]],
    stop_roi: Tuple[Tuple[int, int], ...],
    camera_id: str,
    track_id: int,
    track_class: int,
    score: float,
    bbox_roi: Tuple[int, int, int, int],
    roi_offset: Tuple[int, int, int, int],
) -> Dict[str, Optional[dict]]:
    """
    Detects/activates trigger events using the head bounding box
    centroid. Mutates `trigger_states` in place (per-track state), the
    same way the reference Engine mutated cam["trigger_states"].
    """
    events: Dict[str, Optional[dict]] = {
        "line_cross": None,
        "roi_entry": None,
        "roi_exit": None,
        "stopped_roi": None,
    }

    line_p1, line_p2 = normalize_line_points(line_points)
    roi_polygon = normalize_stop_roi(stop_roi)

    x1, y1, x2, y2 = bbox_roi
    rx1, ry1, rx2, ry2 = roi_offset

    # Head-crop optimization: true geometric centroid.
    point_roi = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    point_full = (int(point_roi[0] + rx1), int(point_roi[1] + ry1))

    current_time = time.time()

    if track_id not in trigger_states:
        trigger_states[track_id] = TriggerTrackState()
    state = trigger_states[track_id]
    state.add_observation(point_full, current_time, score)

    avg_conf = state.get_avg_confidence()

    # ============================================================
    # 1. Line crossing
    # ============================================================
    valid_line = not (line_p1 == (0, 0) and line_p2 == (0, 0))

    if valid_line:
        if state.frame_count >= config.MIN_TRACK_AGE_FOR_CROSSING:
            if avg_conf >= config.MIN_CONFIDENCE_FOR_CROSSING:
                side_value = point_side_of_line(point_full, line_p1, line_p2)
                current_side = get_side_label(side_value)

                if current_side is not None:
                    if state.side_of_line is None:
                        state.side_of_line = current_side
                    elif state.side_of_line != current_side:
                        if state.frames_since_last_cross >= config.CROSSING_COOLDOWN_FRAMES:
                            direction = f"{state.side_of_line}_to_{current_side}"
                            events["line_cross"] = {
                                "camera_id": camera_id,
                                "track_id": track_id,
                                "class": track_class,
                                "direction": direction,
                                "point": point_full,
                                "confidence": avg_conf,
                            }
                            state.frames_since_last_cross = 0
                        state.side_of_line = current_side

    # ============================================================
    # 2. ROI entry / exit / stopped inside ROI
    # ============================================================
    valid_roi = roi_polygon is not None and len(roi_polygon) >= 3

    if valid_roi:
        inside_roi = point_in_polygon(point_full, roi_polygon)

        if inside_roi and avg_conf >= config.MIN_CONFIDENCE_FOR_ROI:
            state.inside_roi_frames += 1

            if state.inside_roi_frames >= config.ROI_ENTRY_CONFIRMATION_FRAMES:
                if not state.roi_confirmed:
                    state.roi_confirmed = True
                    state.roi_entry_time = current_time
                    state.roi_position_buffer.clear()
                    state.roi_timestamp_buffer.clear()
                    state.stop_reported = False

                if not state.roi_entry_reported:
                    events["roi_entry"] = {
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "class": track_class,
                        "point": point_full,
                        "confidence": avg_conf,
                    }
                    state.roi_entry_reported = True

                state.roi_position_buffer.append(point_full)
                state.roi_timestamp_buffer.append(current_time)

        else:
            state.inside_roi_frames = 0

            if state.roi_confirmed:
                events["roi_exit"] = {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "class": track_class,
                    "point": point_full,
                }
                state.reset_roi_state()

        if state.roi_confirmed:
            positions = np.array(state.roi_position_buffer)
            timestamps = np.array(state.roi_timestamp_buffer)

            elapsed = current_time - state.roi_entry_time if state.roi_entry_time is not None else 0.0

            if len(positions) >= 2:
                recent_positions = positions[-config.STOP_MIN_SAMPLES:]
                recent_timestamps = timestamps[-config.STOP_MIN_SAMPLES:]

                total_dist = 0.0
                for i in range(1, len(recent_positions)):
                    total_dist += np.linalg.norm(recent_positions[i] - recent_positions[i - 1])

                total_time = recent_timestamps[-1] - recent_timestamps[0]
                avg_velocity = total_dist / total_time if total_time > 0 else 0.0
            else:
                avg_velocity = 0.0

            if (len(positions) >= config.STOP_MIN_SAMPLES and elapsed >= config.STOP_TIME_SECONDS
                    and not state.stop_reported):
                if avg_velocity <= config.STOP_VELOCITY_THRESHOLD:
                    events["stopped_roi"] = {
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "class": track_class,
                        "duration": elapsed,
                        "velocity": avg_velocity,
                        "point": point_full,
                        "confidence": avg_conf,
                    }
                    state.stop_reported = True

    return events
