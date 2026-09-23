"""
config.py (detector)
--------------------------------------------------------------------
Every tunable the detector service reads, all from the environment so
dev vs prod stays "same image, different .env" like the heatmap module.

Split into the same sections the reference config.py used, minus
everything that is recognition-only (AdaFace thresholds, alignment
reference points, gallery paths) — those now live in
recognizer/src/config.py, next to the code that actually uses them.
--------------------------------------------------------------------
"""

import os


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
REDIS_URL = os.getenv("REDIS_URL", "")  # if set, wins over host/port/db/password

# ============================================================================
# 2. RTSP RELAY
# ============================================================================
# Frames always come from the relay, never from the camera directly —
# camera_stream owns the single real connection to each camera and
# re-serves it to any number of consumers (us included).
MTX_RTSP_BASE_URL = os.getenv("MTX_RTSP_BASE_URL", "rtsp://localhost:8554")

# ============================================================================
# 3. COMPUTE / MODEL
# ============================================================================
# auto | cpu | cuda | cuda:N — resolved to a concrete torch device
# inside each Engine CHILD process, so the parent (this file) never
# initialises CUDA.
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "auto")
STRICT_DEVICE = _bool("STRICT_DEVICE", "false")
TORCH_NUM_THREADS = _int("TORCH_NUM_THREADS", 0)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best1.pt")
IMG_SIZE = _int("DETECTION_IMG_SIZE", 800)
CONF_THRESHOLD = _float("DETECTION_CONF_THRESHOLD", 0.25)
CLASS_LABELS = {0: "Head"}

# ============================================================================
# 4. ENGINE TOPOLOGY / SELF-HEALING
# ============================================================================
# A new engine subprocess is spawned past this many cameras.
MAX_CAMERAS_PER_ENGINE = _int("MAX_CAMERAS_PER_ENGINE", 6)

# How many engines to bring up idle on a completely fresh boot (no
# self-healing checkpoint in Redis yet). After the first run, the
# persisted engine_count from facecore.lifecycle takes over.
DEFAULT_ENGINE_COUNT = _int("DEFAULT_ENGINE_COUNT", 1)

# How many AFR (recognition) workers this engine used to spin up
# in-process. Kept only as a knob for anyone still comparing against
# the reference pipeline; the recognizer's own worker pool now owns
# recognition capacity entirely, independent of engine count.
LEGACY_FR_WORKERS_PER_ENGINE = _int("LEGACY_FR_WORKERS_PER_ENGINE", 1)

# How often EngineManager checks whether the current camera spread
# could be consolidated into fewer engines (see engine_manager.rebalance).
ENGINE_REBALANCE_INTERVAL_SEC = _float("ENGINE_REBALANCE_INTERVAL_SEC", 30.0)

# How long to wait for engine subprocesses to exit cleanly on shutdown.
ENGINE_SHUTDOWN_TIMEOUT_SEC = _float("ENGINE_SHUTDOWN_TIMEOUT_SEC", 30.0)

# ============================================================================
# 5. CAMERA HEALTH
# ============================================================================
HEARTBEAT_TIMEOUT_SEC = _float("HEARTBEAT_TIMEOUT_SEC", 15.0)

# After camera_stream reports a camera offline, wait this long before
# tearing its engine assignment down — absorbs a camera that is merely
# mid-reconnect. Keep >= the RTSP read timeout.
CAMERA_OFFLINE_GRACE_SECONDS = _float("CAMERA_OFFLINE_GRACE_SECONDS", 5.0)

# ============================================================================
# 6. DEBUG OUTPUT (bind volume — never MinIO; see README storage split,
#    and DEBUGGING.md at the repo root for the full picture across all
#    three services)
# ============================================================================
SAVE_OUTPUT = _bool("SAVE_OUTPUT", "false")
SAVE_AS_VIDEO = _bool("SAVE_AS_VIDEO", "true")
VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "/data/debug_output")
# fps written into the legacy raw video (SAVE_OUTPUT/SAVE_AS_VIDEO path)
# — was a bare literal `20.0` at the cv2.VideoWriter call site.
LEGACY_VIDEO_FPS = _float("LEGACY_VIDEO_FPS", 20.0)

# ---- annotated debug video (debug_recorder.py) -----------------------
# A rolling, fully annotated MP4 per camera showing detections, tracks,
# landmarks, pose, crop quality, liveness and the whole recognition
# round-trip. Separate from SAVE_OUTPUT above, which is the older raw
# frame/video dump. Writes to a bind-mounted host dir — see compose.yaml.
#
# IMPORTANT: engine.py constructs the recorder's settings as
# `DebugConfig()` (debug_recorder.py) — that class reads ALL of
# DEBUG_VIDEO_* (and more: DEBUG_VIDEO_EVERY_N, _SCALE, _PANEL_WIDTH,
# _CODEC, _EXT, _JSONL, _GHOST_FRAMES, _EVENT_LINES, _DRAW_LANDMARKS)
# directly from the environment ITSELF — it does not read them from
# this config.py. The five below are kept here too purely as the
# single documented reference for what's tunable in this service (see
# DEBUGGING.md); changing them here has NO effect — change the env var
# itself (compose.yaml / .env), which both this file and DebugConfig
# read independently. This was true before this revision; noted rather
# than silently left for anyone auditing "is everything really from
# env" (answer: yes — just via two independent readers of the same
# var names, not a single shared source).
DEBUG_VIDEO_ENABLED = _bool("DEBUG_VIDEO_ENABLED", "true")
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "/debug")
DEBUG_VIDEO_SEGMENT_SECONDS = _float("DEBUG_VIDEO_SEGMENT_SECONDS", 30.0)
DEBUG_VIDEO_FPS = _float("DEBUG_VIDEO_FPS", 12.0)
DEBUG_VIDEO_MAX_SEGMENTS = _int("DEBUG_VIDEO_MAX_SEGMENTS", 40)

# ---- best-crop montage (debug_extras.py) ------------------------------
# On every recognition submission, save a labelled grid of the crop
# candidates actually being tracked in the best-crop ladder (reg1/reg2/
# reg3) with the one that got sent highlighted — answers "what pixels
# actually left the detector", which the annotated video's on-screen
# quality NUMBERS don't show you. Same bind-mounted debug root as the
# video (DEBUG_VIDEO_DIR), under best_crop_montages/camera_<id>/.
DEBUG_BEST_CROP_MONTAGE_ENABLED = _bool("DEBUG_BEST_CROP_MONTAGE_ENABLED", "false")
DEBUG_BEST_CROP_MONTAGE_MAX_FILES = _int("DEBUG_BEST_CROP_MONTAGE_MAX_FILES", 200)

# ---- liveness-reject stills (debug_extras.py) --------------------------
# One still per track the moment its liveness verdict flips to "fake" —
# spoof attempts are short and easy to miss scrubbing 30s video
# segments; this keeps a standing, easy-to-scan folder of just those.
DEBUG_LIVENESS_REJECTS_ENABLED = _bool("DEBUG_LIVENESS_REJECTS_ENABLED", "false")
DEBUG_LIVENESS_REJECTS_MAX_FILES = _int("DEBUG_LIVENESS_REJECTS_MAX_FILES", 200)

# ---- recognition round-trip latency (shown in the annotated video and
# logged — see engine.py's REC_RESULT logging) --------------------------
# No toggle needed: this is just a timestamp diff already available
# from data the engine tracks (meta["pending_since"]); always computed,
# only ever DISPLAYED when DEBUG_VIDEO_ENABLED already is.

# ============================================================================
# 6b. LIVENESS / PRESENTATION-ATTACK DETECTION (liveness.py)
# ============================================================================
# Non-DL anti-spoofing: is this a live face or a photo/phone/monitor
# held up to the camera? Costs one sparse optical-flow pass per face
# every LIVENESS_EVAL_EVERY_N_FRAMES frames; no extra model.
#
# The verdict rides along on the track and lands in the recognition
# task meta AND in the ai:results payload the backend consumes, as
# `liveness` / `liveness_score` / `liveness_reason`.
#
# Thresholds want calibrating on your own footage — see liveness.py for
# what each one measures. The defaults are deliberately conservative:
# they would rather say `insufficient_evidence` than guess.
LIVENESS_ENABLED = _bool("LIVENESS_ENABLED", "true")
LIVENESS_EVAL_EVERY_N_FRAMES = _int("LIVENESS_EVAL_EVERY_N_FRAMES", 3)
LIVENESS_MIN_POSE_DELTA_DEG = _float("LIVENESS_MIN_POSE_DELTA_DEG", 10.0)
LIVENESS_PLANAR_RESIDUAL_THR = _float("LIVENESS_PLANAR_RESIDUAL_THR", 0.006)
LIVENESS_RING_FOLLOW_THR = _float("LIVENESS_RING_FOLLOW_THR", 0.60)
LIVENESS_RIGID_RESIDUAL_THR = _float("LIVENESS_RIGID_RESIDUAL_THR", 0.004)
LIVENESS_MIN_EVALS = _int("LIVENESS_MIN_EVALS", 3)

# ============================================================================
# 7. DETECTION & PIPELINE CRITERIA (verbatim from the reference config.py)
# ============================================================================
MIN_TRACK_AGE_FOR_CROSSING = _int("MIN_TRACK_AGE_FOR_CROSSING", 10)
MIN_CONFIDENCE_FOR_CROSSING = _float("MIN_CONFIDENCE_FOR_CROSSING", 0.5)
CROSSING_COOLDOWN_FRAMES = _int("CROSSING_COOLDOWN_FRAMES", 30)
MIN_CONFIDENCE_FOR_ROI = _float("MIN_CONFIDENCE_FOR_ROI", 0.5)
ROI_ENTRY_CONFIRMATION_FRAMES = _int("ROI_ENTRY_CONFIRMATION_FRAMES", 5)
STOP_MIN_SAMPLES = _int("STOP_MIN_SAMPLES", 15)
STOP_TIME_SECONDS = _float("STOP_TIME_SECONDS", 3.0)
STOP_VELOCITY_THRESHOLD = _float("STOP_VELOCITY_THRESHOLD", 1.5)

PERIODIC_MODE = os.getenv("PERIODIC_MODE", "frame")
PERIODIC_FRAME_INTERVAL = _int("PERIODIC_FRAME_INTERVAL", 60)
PERIODIC_TIME_INTERVAL = _float("PERIODIC_TIME_INTERVAL", 3.0)
PERIODIC_RECOG_CONF_THRESH = _float("PERIODIC_RECOG_CONF_THRESH", 0.70)
FINALIZE_MAX_CROPS = _int("FINALIZE_MAX_CROPS", 1)

# ============================================================================
# 8. QUALITY GATES & HEAD POSE (verbatim from the reference config.py)
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

# Landmark-validity gate used throughout engine.py wherever a landmark
# point's own confidence decides whether to trust/draw/use it (pose
# calc, best-crop ladder ranking, debug overlays). Was four separate
# hardcoded `conf > 0.2` literals with no single source of truth.
LANDMARK_VALID_CONF_THRESHOLD = _float("LANDMARK_VALID_CONF_THRESHOLD", 0.2)

# Engine._get_yaw_group()'s own 3-tier bucket used to rank best-crop
# candidates (prefer more-frontal faces), DISTINCT from the
# FRONTAL/QUARTER/PROFILE quality-gate windows above (those decide
# accept/reject; this only decides ranking priority among accepted
# crops). Were two inline literals (15.0, 30.0).
YAW_GROUP_FRONTAL_MAX_DEG = _float("YAW_GROUP_FRONTAL_MAX_DEG", 15.0)
YAW_GROUP_QUARTER_MAX_DEG = _float("YAW_GROUP_QUARTER_MAX_DEG", 30.0)

# ============================================================================
# 8b. TRACKER (BYTETracker, tracker.py) — was a single hardcoded
# `TrackerConfig(track_buffer=35)` call site in engine.py, the other
# four fields silently falling back to TrackerConfig's own class
# defaults (which do NOT match these — e.g. its own default
# track_buffer is 90, never actually used because the call site always
# overrode it). Defaults below reproduce exactly what was actually
# running: track_buffer=35 (the real prior behaviour), the rest at
# TrackerConfig's original defaults.
# ============================================================================
TRACKER_TRACK_THRESH = _float("TRACKER_TRACK_THRESH", 0.5)
TRACKER_MATCH_THRESH = _float("TRACKER_MATCH_THRESH", 0.9)
TRACKER_TRACK_BUFFER = _int("TRACKER_TRACK_BUFFER", 35)
TRACKER_NMS_THRESH = _float("TRACKER_NMS_THRESH", 0.5)
TRACKER_MOT20 = _bool("TRACKER_MOT20", "true")

# ============================================================================
# 9. TRACK AGGREGATION & BUFFERS (verbatim from the reference config.py)
# ============================================================================
DEFAULT_ABSENT_FRAMES = _int("DEFAULT_ABSENT_FRAMES", 60)
DEFAULT_MIN_SEEN_FRAMES = _int("DEFAULT_MIN_SEEN_FRAMES", 8)
DEFAULT_MIN_CROPS_TO_FINALIZE = _int("DEFAULT_MIN_CROPS_TO_FINALIZE", 1)
DEFAULT_N_BEST_CROPS = _int("DEFAULT_N_BEST_CROPS", 5)
DEFAULT_CONF_DIGITS = _int("DEFAULT_CONF_DIGITS", 2)

# ============================================================================
# 10. LOGGING / API
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # text | json — see facecore.logging_setup
API_PORT = _int("API_PORT", 8010)
