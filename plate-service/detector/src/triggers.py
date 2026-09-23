"""
triggers.py
--------------------------------------------------------------------
Spatial trigger detection: line-crossing, ROI entry/exit, and
"stopped inside ROI". Extracted from Engine._process_track_triggers in
the reference video_processor.py into a standalone function so it can
be tested in isolation (see tests/test_triggers.py) and so Engine
itself stays focused on the detect -> track -> dispatch loop.

The math (which side of the line, point-in-polygon, sliding-window
velocity) and the threshold constants are UNCHANGED from the reference
pipeline — only the packaging is different. In particular, the anchor
point is the bounding box BOTTOM-CENTER (`bbox_bottom_center`), not a
centroid — correct for a vehicle bounding box (the point that actually
crosses a stop line or enters a parking box), unlike a head/face box
where a centroid is the natural anchor.
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

        self.position_history = []
        self.timestamp_history = []
        self.confidence_history = []

        # line crossing
        self.side_of_line = None
        self.frames_since_last_cross = 999

        # ROI entry
        self.inside_roi_frames = 0
        self.roi_confirmed = False
        self.roi_entry_reported = False
        self.roi_entry_time = None

        # stop detection
        self.roi_position_buffer = []
        self.roi_timestamp_buffer = []
        self.stop_reported = False

    def add_observation(self, point, timestamp, confidence):
        self.position_history.append(point)
        self.timestamp_history.append(timestamp)
        self.confidence_history.append(float(confidence))
        if len(self.position_history) > config.TRIGGER_POSITION_HISTORY_MAX:
            self.position_history.pop(0)
            self.timestamp_history.pop(0)
        if len(self.confidence_history) > config.TRIGGER_CONFIDENCE_HISTORY_MAX:
            self.confidence_history.pop(0)
        self.frame_count += 1
        self.frames_since_last_cross += 1

    def get_avg_confidence(self):
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))

    def recent_velocity(self, n: int = None):
        """Instantaneous-ish speed in px/s over the last n observations.
        Debug-only: the stop detector keeps using its own ROI buffers."""
        if n is None:
            n = config.TRIGGER_VELOCITY_WINDOW
        if len(self.position_history) < 2:
            return None
        pts = self.position_history[-n:]
        tss = self.timestamp_history[-n:]
        if len(pts) < 2 or len(tss) < 2:
            return None
        dist = 0.0
        for i in range(1, len(pts)):
            dist += float(np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1])))
        dt = float(tss[-1] - tss[0])
        if dt <= 0:
            return None
        return dist / dt

    def reset_roi_state(self):
        self.inside_roi_frames = 0
        self.roi_confirmed = False
        self.roi_entry_reported = False
        self.roi_entry_time = None
        self.roi_position_buffer = []
        self.roi_timestamp_buffer = []
        self.stop_reported = False


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
    """point: (x, y). polygon: numpy array Nx2."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def bbox_bottom_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int(y2))


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
    line_points: Tuple[Tuple[float, float], Tuple[float, float]],
    stop_roi: Tuple[Tuple[float, float], ...],
    camera_id: str,
    track_id: int,
    track_class: int,
    score: float,
    bbox_roi: Tuple[int, int, int, int],
    roi_offset: Tuple[int, int, int, int],
) -> Dict[str, Optional[dict]]:
    """
    Detects/activates trigger events. Mutates `trigger_states` in
    place (per-track state), the same way the reference Engine mutated
    cam["trigger_states"].

    This function ONLY detects/activates trigger events. It does NOT
    check the per-camera trigger flags (cross_line_trig/stop_roi_trig)
    and does NOT log — the caller (Engine.run()) decides whether to
    act on/log a given event based on that camera's configured flags.
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

    point_roi = bbox_bottom_center(x1, y1, x2, y2)
    point_full = (int(point_roi[0] + rx1), int(point_roi[1] + ry1))

    current_time = time.time()

    if track_id not in trigger_states:
        trigger_states[track_id] = TriggerTrackState()
    state = trigger_states[track_id]
    state.add_observation(point_full, current_time, score)

    avg_conf = state.get_avg_confidence()

    # ============================================================
    # 1. Detect line crossing
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
    # 2. Detect ROI entry / ROI exit / stopped inside ROI
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
                    state.roi_position_buffer = []
                    state.roi_timestamp_buffer = []
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

        # --------------------------------------------------------
        # STOPPED INSIDE ROI detection
        # --------------------------------------------------------
        if state.roi_confirmed and len(state.roi_position_buffer) >= config.STOP_MIN_SAMPLES:
            if state.roi_entry_time is not None:
                elapsed = current_time - state.roi_entry_time

                if elapsed >= config.STOP_TIME_SECONDS and not state.stop_reported:
                    positions = np.array(state.roi_position_buffer)
                    timestamps = np.array(state.roi_timestamp_buffer)

                    total_dist = 0.0
                    for i in range(1, len(positions)):
                        total_dist += np.linalg.norm(positions[i] - positions[i - 1])

                    total_time = timestamps[-1] - timestamps[0]
                    avg_velocity = total_dist / total_time if total_time > 0 else 0.0

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
