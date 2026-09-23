"""
pose.py (recognizer) — NEW, add-face
--------------------------------------------------------------------
Head-pose verification for enrollment. Ported near-verbatim from the
reference `ApiFace_utility.py` (compute_yaw / compute_pitch_ratio /
check_pose_approval / process_frame_for_face's crop-with-padding step)
— the math itself must not drift, only what detects the face changes.

Detector swap, and why it's not a new model:
    The reference used a standalone RetinaNet instance for this. That
    detector does not exist anywhere in this codebase and would be a
    second face detector next to the one the recognizer already loads.
    Instead this module detects with the SAME MTCNN instance
    `face_alignment.align` already constructs at import time
    (`align.py`'s module-level `mtcnn_model`) — the recognizer imports
    that module today, lazily, inside
    `recognition_engine.FaceRecognition.generate_embedding()`. Calling
    `.detect_faces()` on that existing object costs nothing extra: no
    new weights, no new process, no new device placement decision.

    One consequence worth being explicit about: gallery images (every
    `c<N>.jpg`, including the ones this module produces) have ALWAYS
    been embedded via `align.get_aligned_face()`, i.e. this same MTCNN
    path — see `recognition_engine.py::generate_embedding`. Live probe
    crops during real-time recognition instead prefer YOLO-landmark
    alignment (`alignment.py::FaceAligner`), falling back to this same
    MTCNN path only when YOLO landmarks aren't available. So using
    MTCNN here does not introduce a new mismatch between "how
    enrollment sees a face" and "how the gallery embeds a face" — they
    were already the same path. It only means enrollment's pose-check
    detector and the gallery's embedding-time detector are now
    (correctly) the same object, where the old reference had the
    pose-check on a third, disconnected detector (RetinaNet) that
    never touched the embedding path at all.

Landmark convention (unchanged): 5 points — left eye, right eye, nose,
left mouth corner, right mouth corner — as [[x0,y0], [x1,y1], ...],
exactly what `align.py`'s own `align()` method builds from MTCNN's
raw [n, 10] landmark output (`landmarks[0][j], landmarks[0][j+5]`).
--------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from PIL import Image

import config

# BGR palette, same convention as debug_recorder.py's C_* constants.
_C_OK = (110, 230, 130)
_C_BAD = (80, 90, 250)
_C_TEXT = (235, 235, 235)
_C_LM = (0, 0, 255)


class NoFaceDetected(Exception):
    """Raised when MTCNN finds nothing in the frame."""


def detect_face_5pt(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the shared MTCNN instance on a single BGR frame and returns
    (bbox, facial5points) for the highest-confidence face.

    bbox:            [x1, y1, x2, y2, score] (float64)
    facial5points:   5 x [x, y] — leye, reye, nose, lmouth, rmouth

    Raises NoFaceDetected if nothing was found.
    """
    # Local import, same reason recognition_engine.py does it lazily:
    # `face_alignment` is operator-supplied (bind-mounted), not a pip
    # dependency baked into the image — importing it eagerly at module
    # load time would make pose.py fail to import in any environment
    # that hasn't mounted that folder yet (e.g. a unit test).
    from face_alignment import align as _align

    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    mtcnn_model = _align.mtcnn_model  # the one instance align.py already constructed
    bboxes, landmarks = mtcnn_model.detect_faces(
        pil_img,
        min_face_size=config.ENROLL_MIN_FACE_SIZE,
        thresholds=config.ENROLL_MTCNN_THRESHOLDS,
        nms_thresholds=config.ENROLL_MTCNN_NMS_THRESHOLDS,
        factor=config.ENROLL_MTCNN_FACTOR,
    )

    if bboxes is None or len(bboxes) == 0:
        raise NoFaceDetected("no face detected in enrollment frame")

    # detect_faces already returns faces sorted by pipeline order, not
    # explicitly by score — pick the highest-confidence one explicitly
    # rather than assuming index 0 (index 4 of each row is the score).
    best_idx = int(np.argmax(bboxes[:, 4]))
    bbox = bboxes[best_idx]
    facial5points = np.array(
        [[landmarks[best_idx][j], landmarks[best_idx][j + 5]] for j in range(5)],
        dtype=np.float32,
    )
    return bbox, facial5points


def compute_yaw(landmark_set, frame_shape) -> float:
    """Verbatim from the reference (ApiFace_utility.py::compute_yaw).
    solvePnP against a rough generic 3D face model; returns yaw in
    degrees, or 0 if the solve fails."""
    try:
        model_points = np.array([
            [-30.0, 0.0, -30.0],   # Left eye
            [30.0, 0.0, -30.0],    # Right eye
            [0.0, 0.0, 0.0],       # Nose
            [-20.0, -30.0, -30.0], # Left mouth
            [20.0, -30.0, -30.0],  # Right mouth
        ], dtype=np.float32)

        image_points = np.array(landmark_set, dtype=np.float32)

        mid_eye_2d = np.mean(image_points[0:2], axis=0).reshape(1, 2)
        mid_eye_3d = np.mean(model_points[0:2], axis=0).reshape(1, 3)

        image_points_aug = np.vstack([image_points, mid_eye_2d]).reshape(-1, 1, 2)
        model_points_aug = np.vstack([model_points, mid_eye_3d]).reshape(-1, 1, 3)

        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv.solvePnP(model_points_aug, image_points_aug, camera_matrix, dist_coeffs)
        if not success:
            return 0.0

        rmat, _ = cv.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        yaw = math.atan2(-rmat[2, 0], sy)
        return float(np.degrees(yaw))
    except Exception:
        return 0.0


def compute_pitch_ratio(landmark_set) -> Optional[float]:
    """Verbatim from the reference. Purely geometric — vertical
    eye/nose/mouth distance ratio, normalized by inter-eye distance,
    mapped to a pitch angle via a fixed scaling factor. Returns None
    if the geometry is degenerate or the result is an outlier."""
    try:
        landmarks = np.array(landmark_set, dtype=np.float32)
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

        eye_midpoint = (left_eye + right_eye) / 2
        mouth_midpoint = (left_mouth + right_mouth) / 2

        eye_distance = np.linalg.norm(right_eye - left_eye)
        eye_to_nose = np.abs(eye_midpoint[1] - nose[1])
        nose_to_mouth = np.abs(nose[1] - mouth_midpoint[1])

        if nose_to_mouth == 0:
            return None
        ratio = eye_to_nose / nose_to_mouth
        normalized_ratio = ratio * (100 / eye_distance) if eye_distance > 0 else 0

        pitch = (normalized_ratio - 1.0) * config.ENROLL_PITCH_SCALING_FACTOR

        if abs(pitch) > 90:
            return None
        return float(pitch)
    except Exception:
        return None


def check_pose_approval(yaw: Optional[float], pitch: Optional[float], flag: int) -> Tuple[str, str]:
    """Verbatim decision logic from the reference. `flag` is the
    requested angle: 1/2/3 map to the three ENROLL_YAW_WINDOWS entries
    below (front/left/right — exact mapping is config's job, not this
    function's, so the windows can be retuned without touching code).
    Pitch has one fixed acceptable window regardless of flag."""
    yaw_status = "Not Approved" if yaw is not None else "N/A"
    pitch_status = "Not Approved" if pitch is not None else "N/A"

    pitch_lo, pitch_hi = config.ENROLL_PITCH_WINDOW
    if pitch is not None and pitch_lo <= pitch <= pitch_hi:
        pitch_status = "Approved"

    window = config.ENROLL_YAW_WINDOWS.get(flag)
    if yaw is not None and window is not None:
        lo, hi = window
        if lo <= yaw <= hi:
            yaw_status = "Approved"

    return yaw_status, pitch_status


def crop_with_padding(frame_bgr: np.ndarray, bbox: np.ndarray, padding_ratio: float = None) -> np.ndarray:
    """Verbatim crop-expansion logic from the reference: pad the box by
    `padding_ratio` on every side (default from config), then clip to
    the frame. Crops from the ORIGINAL frame, not a resized copy."""
    padding_ratio = config.ENROLL_CROP_PADDING_RATIO if padding_ratio is None else padding_ratio
    x1, y1, x2, y2 = bbox[:4].astype(float)
    width, height = x2 - x1, y2 - y1
    x1 -= width * padding_ratio
    y1 -= height * padding_ratio
    x2 += width * padding_ratio
    y2 += height * padding_ratio
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2 = min(frame_bgr.shape[1], int(x2))
    y2 = min(frame_bgr.shape[0], int(y2))
    return frame_bgr[y1:y2, x1:x2]


def render_pose_debug_image(
    frame_bgr: np.ndarray,
    bbox: Optional[np.ndarray],
    facial5points: Optional[np.ndarray],
    yaw: Optional[float],
    pitch: Optional[float],
    yaw_status: str,
    pitch_status: str,
    flag: int,
) -> np.ndarray:
    """Pure rendering, no I/O, no MTCNN — takes whatever verify_pose()
    already computed and draws it: the detected box, the 5 landmarks,
    and the yaw/pitch numbers with each axis's pass/fail shown
    SEPARATELY (matching check_pose_approval's two-part return), so a
    rejected enrollment shows exactly which axis failed rather than
    just a single "no" that could mean either. Never raises — worst
    case returns the original frame with just a status line burned in,
    so a debug-image failure never blocks the pose-check response
    (see worker.py's enroll handler, which wraps this in try/except
    regardless as a second line of defense).
    """
    img = frame_bgr.copy()

    if bbox is not None:
        x1, y1, x2, y2, score = [float(v) for v in bbox[:5]]
        box_col = _C_OK if (yaw_status == "Approved" and pitch_status == "Approved") else _C_BAD
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_col, 2)
        cv.putText(img, f"det {score:.2f}", (int(x1), max(12, int(y1) - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv.LINE_AA)

    if facial5points is not None:
        for x, y in facial5points:
            cv.circle(img, (int(x), int(y)), 3, _C_LM, -1)

    def _line(text: str, y: int, color) -> None:
        (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(img, (6, y - th - 4), (10 + tw, y + base), (15, 15, 15), -1)
        cv.putText(img, text, (8, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    yaw_col = _C_OK if yaw_status == "Approved" else _C_BAD
    pitch_col = _C_OK if pitch_status == "Approved" else _C_BAD
    yaw_txt = f"{yaw:+.1f}deg" if isinstance(yaw, (int, float)) else "n/a"
    pitch_txt = f"{pitch:+.1f}deg" if isinstance(pitch, (int, float)) else "n/a"
    window = config.ENROLL_YAW_WINDOWS.get(flag)
    window_txt = f"[{window[0]:+.0f},{window[1]:+.0f}]" if window else "?"
    pitch_lo, pitch_hi = config.ENROLL_PITCH_WINDOW

    _line(f"flag {flag} (expect yaw in {window_txt})", 20, _C_TEXT)
    _line(f"yaw {yaw_txt} -> {yaw_status}", 38, yaw_col)
    _line(f"pitch {pitch_txt} (window [{pitch_lo:+.0f},{pitch_hi:+.0f}]) -> {pitch_status}", 56, pitch_col)

    return img


def verify_pose(frame_bgr: np.ndarray, flag: int) -> dict:
    """End-to-end single-frame pose check — the whole thing worker.py's
    `_process_enroll_pose_check` needs. Returns a plain dict so it can
    go straight into a Redis result payload:

        {"approved": True,  "yaw": .., "pitch": .., "crop_bgr": ndarray}
        {"approved": False, "reason": "no_face_detected"}
        {"approved": False, "reason": "pose_mismatch", "yaw": .., "pitch": ..,
         "yaw_status": .., "pitch_status": ..}
    """
    try:
        bbox, facial5points = detect_face_5pt(frame_bgr)
    except NoFaceDetected:
        debug_image = None
        try:
            debug_image = render_pose_debug_image(frame_bgr, None, None, None, None, "N/A", "N/A", flag)
        except Exception:
            pass
        return {"approved": False, "reason": "no_face_detected", "debug_image": debug_image}

    yaw = compute_yaw(facial5points, frame_bgr.shape)
    pitch = compute_pitch_ratio(facial5points)
    yaw_status, pitch_status = check_pose_approval(yaw, pitch, flag)

    debug_image = None
    try:
        debug_image = render_pose_debug_image(
            frame_bgr, bbox, facial5points, yaw, pitch, yaw_status, pitch_status, flag
        )
    except Exception:
        pass  # rendering must never block the actual pose-check response

    if yaw_status != "Approved" or pitch_status != "Approved":
        return {
            "approved": False,
            "reason": "pose_mismatch",
            "yaw": yaw, "pitch": pitch,
            "yaw_status": yaw_status, "pitch_status": pitch_status,
            "debug_image": debug_image,
        }

    crop_bgr = crop_with_padding(frame_bgr, bbox)
    return {"approved": True, "yaw": yaw, "pitch": pitch, "crop_bgr": crop_bgr, "debug_image": debug_image}
