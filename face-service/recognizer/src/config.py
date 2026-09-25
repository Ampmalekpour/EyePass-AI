"""
config.py (recognizer)
--------------------------------------------------------------------
Every tunable the recognizer service reads. The AdaFace thresholds,
alignment reference points and decision-math hyperparameters are
verbatim from the reference config.py — changing any of them changes
recognition behaviour, so they stay exactly where they were, just
env-overridable like everything else in this system.

ADD-FACE (2026-09): section 11 adds enrollment's own tunables — pose
windows, MTCNN detection params for the enrollment path, crop padding,
and the gallery lock timeout. All ported verbatim from the reference
pose-check code (yaw/pitch windows, 15% crop padding) except where
noted; all env-overridable, same convention as everything above.
--------------------------------------------------------------------
"""

import os

import numpy as np


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ============================================================================
# 1. REDIS / MODULE IDENTITY
# ============================================================================
REDIS_MODULE = os.getenv("REDIS_MODULE", "face")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = os.getenv("REDIS_URL", "")

# ============================================================================
# 2. PATHS
# ============================================================================
# next to the code, bind-mounted by the operator (see README "What you
# need to provide"):
#   pretrained/               adaface_ir50_ms1mv2.ckpt (pytorch fallback)
#                              warmup.jpg (used to warm the model up)
#   face_alignment/            the alignment package (align.get_aligned_face,
#                              mtcnn_pytorch's warp_and_crop_face) — ALSO
#                              where add-face's pose-check detector
#                              (align.mtcnn_model) comes from, see pose.py.
BIND_DIR = os.getenv("BIND_DIR", "/app/bind")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/adaface_ir50_cpu.onnx")
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH", "pretrained/adaface_ir50_ms1mv2.ckpt")
WARMUP_IMAGE_PATH = os.getenv("WARMUP_IMAGE_PATH", "pretrained/warmup.jpg")

# Gallery + person database live in MinIO (pipeline data) and are
# downloaded into the container on worker start.
GALLERY_MINIO_PREFIX = os.getenv("GALLERY_MINIO_PREFIX", "dynamics/CTDBUR/")
GALLERY_DB_MINIO_KEY = os.getenv("GALLERY_DB_MINIO_KEY", "dynamics/CTDBUR/brieface.db")

# ============================================================================
# 3. WORKER POOL / SELF-HEALING
# ============================================================================
# How many worker subprocesses to bring up idle on a completely fresh
# boot (no self-healing checkpoint in Redis yet). After the first run,
# the persisted worker_count from facecore.lifecycle takes over.
DEFAULT_WORKER_COUNT = _int("DEFAULT_WORKER_COUNT", 2)
USE_ONNX = _bool("USE_ONNX", "true")
MODEL_NAME = os.getenv("MODEL_NAME", "ir_50")

# How long to wait for a freshly spawned worker to finish loading its
# models + gallery + warmup before start_idle() gives up waiting on it
# (the worker keeps loading in the background regardless; this only
# bounds how long start_idle()/self_heal() blocks at startup).
WORKER_LOAD_TIMEOUT_SEC = _float("WORKER_LOAD_TIMEOUT_SEC", 180.0)

# How long to wait for a worker subprocess to exit cleanly on shutdown
# before it is terminated forcibly.
WORKER_SHUTDOWN_TIMEOUT_SEC = _float("WORKER_SHUTDOWN_TIMEOUT_SEC", 30.0)

# How often the pool's watchdog checks that every worker subprocess it
# expects to be alive actually is, respawning any that crashed — the
# recognizer's own self-healing at the worker-process level, distinct
# from (and in addition to) the whole-service self-healing that
# ServiceLifecycle.self_heal() does on a full container restart.
WATCHDOG_INTERVAL_SEC = _float("WATCHDOG_INTERVAL_SEC", 5.0)

# ============================================================================
# 4. DEBUG OUTPUT (bind volume — never MinIO; see README storage split,
#    and DEBUGGING.md at the repo root for the full picture across all
#    three services)
# ============================================================================
# NOTE: /data is now a HOST BIND MOUNT (see compose.yaml), not the
# named `recognizer_data` docker volume it used to be — so everything
# under it (this section, plus LOG_PATH below) is directly browsable
# on disk, the same way the detector's DEBUG_VIDEO_DIR always was.
DEBUG_SAVE_LANDMARKED_CROPS = _bool("DEBUG_SAVE_LANDMARKED_CROPS", "true")
DEBUG_LANDMARKED_DIR = os.getenv("DEBUG_LANDMARKED_DIR", "/data/live/debug_landmarked_crops")
DEBUG_SAVE_ALIGNED = _bool("DEBUG_SAVE_ALIGNED", "true")
DEBUG_ALIGNED_DIR = os.getenv("DEBUG_ALIGNED_DIR", "/data/live/debug_aligned_faces")
DEBUG_LANDMARKED_MAX_FILES = _int("DEBUG_LANDMARKED_MAX_FILES", 500)
DEBUG_ALIGNED_MAX_FILES = _int("DEBUG_ALIGNED_MAX_FILES", 500)

# ---- match visualization (recognition_engine.py) -----------------------
# For every recognition decision: crop -> aligned 112x112 -> top-K
# gallery matches with their similarity scores, one composite image —
# answers "what did it actually match against, and how close was the
# runner-up", which raw confidence numbers in a log line don't show.
DEBUG_SAVE_MATCHES = _bool("DEBUG_SAVE_MATCHES", "false")
DEBUG_MATCHES_DIR = os.getenv("DEBUG_MATCHES_DIR", "/data/live/debug_matches")
DEBUG_MATCHES_TOP_N = _int("DEBUG_MATCHES_TOP_N", 5)
DEBUG_MATCHES_MAX_FILES = _int("DEBUG_MATCHES_MAX_FILES", 500)

# ---- rejected (below-threshold) crops -----------------------------------
# Tells "genuinely unknown person" apart from "known person, bad angle
# caused a miss" — saved whenever find_person()/the decision math lands
# on Unknown, separate from the accepted-match folder above.
DEBUG_SAVE_REJECTED = _bool("DEBUG_SAVE_REJECTED", "false")
DEBUG_REJECTED_DIR = os.getenv("DEBUG_REJECTED_DIR", "/data/live/debug_rejected")
DEBUG_REJECTED_MAX_FILES = _int("DEBUG_REJECTED_MAX_FILES", 500)

# ---- add-face / enrollment debug (worker.py, pose.py) -------------------
# On by default (unlike the live-path toggles above): this is the
# newest, least battle-tested code path in the system, and a bad
# enrollment silently consumes a gallery range — see ADD_FACE.md /
# DEBUGGING.md for what each image shows.
DEBUG_ENROLL_ENABLED = _bool("DEBUG_ENROLL_ENABLED", "true")
DEBUG_ENROLL_POSE_DIR = os.getenv("DEBUG_ENROLL_POSE_DIR", "/data/enroll/pose_checks")
DEBUG_ENROLL_COMMIT_DIR = os.getenv("DEBUG_ENROLL_COMMIT_DIR", "/data/enroll/commits")
DEBUG_ENROLL_MAX_FILES = _int("DEBUG_ENROLL_MAX_FILES", 500)

LOG_PATH = os.getenv("LOG_PATH", "/data/recognition_logs.txt")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # text | json — see facecore.logging_setup

# Per-recognition decision trace (top-K raw scores, tsallis confidence,
# fused scores per identity, final decision) — logged via
# logger.debug(..., extra={"fields": {...}}) so it costs nothing unless
# LOG_LEVEL=DEBUG, and reads as structured JSON when LOG_FORMAT=json.
LOG_DECISION_TRACE = _bool("LOG_DECISION_TRACE", "true")

# ============================================================================
# 5. DETECTION & PIPELINE CRITERIA (verbatim from the reference config.py)
# ============================================================================
MIN_TRACK_AGE_FOR_CROSSING = _int("MIN_TRACK_AGE_FOR_CROSSING", 10)
MIN_CONFIDENCE_FOR_CROSSING = _float("MIN_CONFIDENCE_FOR_CROSSING", 0.5)
CROSSING_COOLDOWN_FRAMES = _int("CROSSING_COOLDOWN_FRAMES", 30)
MIN_CONFIDENCE_FOR_ROI = _float("MIN_CONFIDENCE_FOR_ROI", 0.5)
ROI_ENTRY_CONFIRMATION_FRAMES = _int("ROI_ENTRY_CONFIRMATION_FRAMES", 5)
STOP_MIN_SAMPLES = _int("STOP_MIN_SAMPLES", 15)
STOP_TIME_SECONDS = _float("STOP_TIME_SECONDS", 3.0)
STOP_VELOCITY_THRESHOLD = _float("STOP_VELOCITY_THRESHOLD", 1.5)

# (PERIODIC_* / PERIODIC_RECOG_CONF_THRESH / FINALIZE_MAX_CROPS used to
# be duplicated here from the detector's config; nothing in the
# recognizer read them. Periodic cadence and the "satisfied" threshold
# now live in the control hub, FINALIZE_MAX_CROPS in the detector.)

# ============================================================================
# 6. QUALITY GATES & HEAD POSE (verbatim from the reference config.py)
# ============================================================================
SHARPNESS_MIN_THRESHOLD = _float("SHARPNESS_MIN_THRESHOLD", 100.0)
RESOLUTION_MIN_AREA = _int("RESOLUTION_MIN_AREA", 1000)
CLASS_0_ASPECT_RATIO_MIN = _float("CLASS_0_ASPECT_RATIO_MIN", 1.5)
CLASS_0_ASPECT_RATIO_MAX = _float("CLASS_0_ASPECT_RATIO_MAX", 7.5)
GENERIC_ASPECT_RATIO_MIN = _float("GENERIC_ASPECT_RATIO_MIN", 1.0)
GENERIC_ASPECT_RATIO_MAX = _float("GENERIC_ASPECT_RATIO_MAX", 5.5)

FRONTAL_YAW_LIMIT = _float("FRONTAL_YAW_LIMIT", 30.0)
FRONTAL_PITCH_LIMIT = _float("FRONTAL_PITCH_LIMIT", 35.0)
QUARTER_YAW_LIMIT_MIN = _float("QUARTER_YAW_LIMIT_MIN", 30.0)
QUARTER_YAW_LIMIT_MAX = _float("QUARTER_YAW_LIMIT_MAX", 50.0)
QUARTER_PITCH_LIMIT = _float("QUARTER_PITCH_LIMIT", 40.0)
PROFILE_YAW_LIMIT_MIN = _float("PROFILE_YAW_LIMIT_MIN", 60.0)

# ============================================================================
# 7. RECOGNITION QUALITY & MATH HYPERPARAMETERS (verbatim)
# ============================================================================
USE_YOLO_ALIGNMENT = _bool("USE_YOLO_ALIGNMENT", "true")
LANDMARK_CONF_THRESHOLD = _float("LANDMARK_CONF_THRESHOLD", 0.60)
YOLO_LANDMARK_CONF_THR = _float("YOLO_LANDMARK_CONF_THR", 0.25)
YOLO_5PT_IDX = [4, 5, 0, 8, 9]

REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

DECISION_TOP_K = _int("DECISION_TOP_K", 12)
POOLING_ALPHA = _float("POOLING_ALPHA", 0.5)
TSALLIS_TAU = _float("TSALLIS_TAU", 0.05)

REC_UP_THR = _float("REC_UP_THR", 0.75)
REC_MID_THR = _float("REC_MID_THR", 0.50)
REC_DOWN_THR = _float("REC_DOWN_THR", 0.35)
MIN_RAW_SIMILARITY_THR = _float("MIN_RAW_SIMILARITY_THR", 0.20)
WORKER_CONF_THRESHOLD = _float("WORKER_CONF_THRESHOLD", 0.3)

# `_decide_person_and_confidence`'s own confidence-shaping rules — were
# three inline literals (0.6, 2.0/3.0, 0.5) with no env override.
# Defaults below reproduce the exact behaviour that was already
# running; this only ADDS the ability to override, it changes nothing
# by itself. See recognition_engine.py's docstring for why this
# function's math otherwise stays "verbatim" / must-not-drift.
#
#   confidence >= REC_UP_THR:
#     top_raw_score >= HIGH_CONFIDENCE_RAW_SCORE_THR -> unchanged
#     else                                            -> * CONF_PENALTY_HIGH_LOW_RAW
#   REC_MID_THR <= confidence < REC_UP_THR             -> * CONF_PENALTY_MID
HIGH_CONFIDENCE_RAW_SCORE_THR = _float("HIGH_CONFIDENCE_RAW_SCORE_THR", 0.6)
CONF_PENALTY_HIGH_LOW_RAW = _float("CONF_PENALTY_HIGH_LOW_RAW", 2.0 / 3.0)
CONF_PENALTY_MID = _float("CONF_PENALTY_MID", 0.5)

# ONNX InferenceSession thread/warmup tuning — were hardcoded at the
# _load_onnx_model() call site.
ONNX_INTRA_OP_THREADS = _int("ONNX_INTRA_OP_THREADS", 4)
ONNX_INTER_OP_THREADS = _int("ONNX_INTER_OP_THREADS", 2)
ONNX_WARMUP_ITERATIONS = _int("ONNX_WARMUP_ITERATIONS", 3)

# _save_embeddings_to_pkl()'s retry/backoff — were hardcoded local
# variables (max_retries=3, retry_delay=0.5).
PKL_SAVE_MAX_RETRIES = _int("PKL_SAVE_MAX_RETRIES", 3)
PKL_SAVE_RETRY_DELAY_SEC = _float("PKL_SAVE_RETRY_DELAY_SEC", 0.5)

# ============================================================================
# 8. TRACK AGGREGATION & BUFFERS (verbatim)
# ============================================================================
TRACK_N_MIN_HIGH = _int("TRACK_N_MIN_HIGH", 4)
TRACK_N_MIN_MID = _int("TRACK_N_MIN_MID", 2)
MULTIPLIER_FULL = _float("MULTIPLIER_FULL", 1.0)
MULTIPLIER_MID = _float("MULTIPLIER_MID", 0.8)
MULTIPLIER_LOW = _float("MULTIPLIER_LOW", 0.5)

# ============================================================================
# 9. PERFORMANCE
# ============================================================================
USE_BATCHING = _bool("USE_BATCHING", "true")
BATCH_SIZE = _int("BATCH_SIZE", 16)

# ============================================================================
# 10. LOGGING / API
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_PORT = _int("API_PORT", 8011)

# ============================================================================
# 11. ADD-FACE — enrollment (NEW)
# ============================================================================
# ---- camera capture (same relay rule the detector follows) --------------
# Reused, not reintroduced: this is the SAME variable the detector reads
# for the same reason — frames only ever come from the MediaMTX relay,
# never a direct camera connection. One .env value, two consumers.
MTX_RTSP_BASE_URL = os.getenv("MTX_RTSP_BASE_URL", "rtsp://mediamtx:8554")
ENROLL_SNAPSHOT_RETRIES = _int("ENROLL_SNAPSHOT_RETRIES", 3)
ENROLL_SNAPSHOT_TIMEOUT_SEC = _float("ENROLL_SNAPSHOT_TIMEOUT_SEC", 5.0)

# ---- MTCNN params for the pose-check detector (pose.py) -----------------
# Verbatim from face_alignment/mtcnn.py's own FaceDetector defaults
# (mtcnn.py:39-42) — reusing the SAME instance means these should stay
# in lockstep with that file rather than drift independently, but they
# are exposed here, env-overridable, because a real deployment may want
# a lower min_face_size for enrollment (a deliberately close-up,
# cooperative photo) than what's tuned for live camera tracking.
ENROLL_MIN_FACE_SIZE = _float("ENROLL_MIN_FACE_SIZE", 20.0)
ENROLL_MTCNN_THRESHOLDS = [
    _float("ENROLL_MTCNN_THR_P", 0.6),
    _float("ENROLL_MTCNN_THR_R", 0.7),
    _float("ENROLL_MTCNN_THR_O", 0.9),
]
ENROLL_MTCNN_NMS_THRESHOLDS = [
    _float("ENROLL_MTCNN_NMS_P", 0.7),
    _float("ENROLL_MTCNN_NMS_R", 0.7),
    _float("ENROLL_MTCNN_NMS_O", 0.7),
]
ENROLL_MTCNN_FACTOR = _float("ENROLL_MTCNN_FACTOR", 0.85)

# ---- pose windows (verbatim from ApiFace_utility.py::check_pose_approval) -
# Pitch: one fixed window regardless of requested angle.
ENROLL_PITCH_WINDOW = (
    _float("ENROLL_PITCH_MIN", -35.0),
    _float("ENROLL_PITCH_MAX", 80.0),
)
ENROLL_PITCH_SCALING_FACTOR = _float("ENROLL_PITCH_SCALING_FACTOR", 50.0)

# Yaw: flag-specific window. flag=1 -> right profile, flag=2 -> frontal,
# flag=3 -> left profile (same convention as the reference; the operator
# UI decides what it *calls* these three shots, this is just the math).
ENROLL_YAW_WINDOWS = {
    1: (_float("ENROLL_YAW_FLAG1_MIN", 30.0), _float("ENROLL_YAW_FLAG1_MAX", 75.0)),
    2: (_float("ENROLL_YAW_FLAG2_MIN", -30.0), _float("ENROLL_YAW_FLAG2_MAX", 30.0)),
    3: (_float("ENROLL_YAW_FLAG3_MIN", -75.0), _float("ENROLL_YAW_FLAG3_MAX", -30.0)),
}

# ---- crop (verbatim: 15% padding on every side) --------------------------
ENROLL_CROP_PADDING_RATIO = _float("ENROLL_CROP_PADDING_RATIO", 0.15)

# ---- how many pose-approved shots make one enrollment --------------------
# Must match len(ENROLL_YAW_WINDOWS) in spirit — one shot per flag — and
# must match the block size gallery.py/image_filenames() and
# recognition_engine.py's own `(image_num - 1) // 3 + 1` assume.
ENROLL_IMAGES_PER_PERSON = _int("ENROLL_IMAGES_PER_PERSON", 3)

# ---- coordination ----------------------------------------------------------
ENROLL_LOCK_TIMEOUT_SEC = _float("ENROLL_LOCK_TIMEOUT_SEC", 30.0)
ENROLL_LOCK_BLOCKING_TIMEOUT_SEC = _float("ENROLL_LOCK_BLOCKING_TIMEOUT_SEC", 15.0)
# How long the coordinator waits for a dispatched worker task
# (pose-check or commit) to come back on the per-request result queue
# before giving up and replying with an error.
ENROLL_TASK_TIMEOUT_SEC = _float("ENROLL_TASK_TIMEOUT_SEC", 20.0)
ENROLL_COMMIT_TASK_TIMEOUT_SEC = _float("ENROLL_COMMIT_TASK_TIMEOUT_SEC", 60.0)
