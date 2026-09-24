"""
config.py (detector)
--------------------------------------------------------------------
Every tunable the fire/smoke detector service reads, all from the
environment so dev vs prod stays "same image, different .env" — the
same pattern plate_detector's and face_detector's config.py use.
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
REDIS_MODULE = os.getenv("REDIS_MODULE", "fire")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = os.getenv("REDIS_URL", "")  # if set, wins over host/port/db/password

# ============================================================================
# 2. RTSP RELAY
# ============================================================================
# Frames always come from the relay, never from the camera directly —
# camera_stream (eyepass-camera-stream) owns the single real connection
# to each camera and re-serves it to any number of consumers.
MTX_RTSP_BASE_URL = os.getenv("MTX_RTSP_BASE_URL", "rtsp://mediamtx:8554")

# ============================================================================
# 3. COMPUTE / MODEL
# ============================================================================
# auto | cpu | cuda | cuda:N — resolved per engine subprocess (see
# engine.py's resolve_device()); each engine subprocess resolves this
# independently so the parent process never touches CUDA.
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "auto")
STRICT_DEVICE = _bool("STRICT_DEVICE", "false")
# Torch CPU threads PER ENGINE PROCESS. 0 = "cpu_count - 1" (leaves one
# core for the RTSP reader / IO worker threads).
TORCH_NUM_THREADS = _int("TORCH_NUM_THREADS", 0)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
IMG_SIZE = _int("DETECTION_IMG_SIZE", 640)
CONF_THRESHOLD = _float("DETECTION_CONF_THRESHOLD", 0.25)
# class 0 = Smoke, 1 = Fire — verbatim from the pre-existing
# video_processor.py's class split.
CLASS_LABELS = {0: "Smoke", 1: "Fire"}

# ============================================================================
# 4. ENGINE TOPOLOGY / SELF-HEALING
# ============================================================================
# A new engine subprocess is spawned past this many cameras.
MAX_CAMERAS_PER_ENGINE = _int("MAX_CAMERAS_PER_ENGINE", 6)

# How many engines to bring up idle on a completely fresh boot (no
# self-healing checkpoint in Redis yet). After the first run, the
# persisted engine_count from firecore.lifecycle takes over.
DEFAULT_ENGINE_COUNT = _int("DEFAULT_ENGINE_COUNT", 1)

# How often EngineManager checks whether the current camera spread
# could be consolidated into fewer engines.
ENGINE_REBALANCE_INTERVAL_SEC = _float("ENGINE_REBALANCE_INTERVAL_SEC", 30.0)

# How long to wait for engine subprocesses to exit cleanly on shutdown.
ENGINE_SHUTDOWN_TIMEOUT_SEC = _float("ENGINE_SHUTDOWN_TIMEOUT_SEC", 30.0)

# ============================================================================
# 5. CAMERA HEALTH
# ============================================================================
HEARTBEAT_TIMEOUT_SEC = _float("HEARTBEAT_TIMEOUT_SEC", 15.0)

# Wait this long after an "offline" transition before detaching a
# camera — absorbs a reconnect blip instead of churning the grid
# tracker's rolling window on every network hiccup. An "online" event
# within the window cancels the pending teardown. Keep >= the RTSP
# read timeout.
CAMERA_OFFLINE_GRACE_SECONDS = _float("CAMERA_OFFLINE_GRACE_SECONDS", 10.0)

# ============================================================================
# 5b. INTERNAL PROCESSING-ERROR AUTO-RESTART
# ============================================================================
# Distinct from the offline-grace path above: this is for an internal
# engine error (a batch-inference exception, a transient resource
# shortage) rather than a physical camera disconnect.
ERROR_RESTART_MAX_RETRIES = _int("ERROR_RESTART_MAX_RETRIES", 1)
ERROR_RESTART_DELAY_SEC = _float("ERROR_RESTART_DELAY_SEC", 3.0)
ERROR_RESTART_COUNTER_RESET_AFTER_SEC = _float("ERROR_RESTART_COUNTER_RESET_AFTER_SEC", 600.0)

# ============================================================================
# 6. DEBUG OUTPUT (bind volume — never MinIO; see README storage split)
# ============================================================================
# Rolling, annotated MP4 per camera showing detections, the 2x2
# spatial grid, each region's verdict and the cooldown state. See
# debug_recorder.py.
DEBUG_VIDEO_ENABLED = _bool("DEBUG_VIDEO_ENABLED", "false")
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "/debug")
DEBUG_VIDEO_SEGMENT_SECONDS = _float("DEBUG_VIDEO_SEGMENT_SECONDS", 240.0)
DEBUG_VIDEO_FPS = _float("DEBUG_VIDEO_FPS", 12.0)
DEBUG_VIDEO_MAX_SEGMENTS = _int("DEBUG_VIDEO_MAX_SEGMENTS", 12)
DEBUG_VIDEO_SCALE = _float("DEBUG_VIDEO_SCALE", 1.0)
DEBUG_VIDEO_EVERY_N = _int("DEBUG_VIDEO_EVERY_N", 1)
DEBUG_VIDEO_CODEC = os.getenv("DEBUG_VIDEO_CODEC", "mp4v")
DEBUG_VIDEO_EXT = os.getenv("DEBUG_VIDEO_EXT", ".mp4")
DEBUG_VIDEO_JSONL = _bool("DEBUG_VIDEO_JSONL", "true")

# ============================================================================
# 7. SPATIAL-GRID THREAT VERIFICATION (verbatim thresholds from the
#    pre-existing video_processor.py's cascading state machine)
# ============================================================================
# Rolling per-region window size (frames) the FIRE/SMOKE/BOTH/CLEAR
# verdict is voted over.
GRID_WINDOW_SIZE = _int("GRID_WINDOW_SIZE", 30)
# A region must accumulate at least this many matching frames within
# its rolling window before its verdict latches from CLEAR to a threat.
GRID_VERIFY_THRESHOLD = _int("GRID_VERIFY_THRESHOLD", 22)
# Frames of continued CLEAR (no detection in ANY region) required
# before an escalated region is allowed to resolve back to CLEAR and
# fire a RESOLUTION event.
GRID_COOLDOWN_FRAMES = _int("GRID_COOLDOWN_FRAMES", 90)

# ============================================================================
# 8. RTSP READER (rtsp_reader.py)
# ============================================================================
RTSP_OPEN_TIMEOUT_MS = _int("RTSP_OPEN_TIMEOUT_MS", 5000)
RTSP_READ_TIMEOUT_MS = _int("RTSP_READ_TIMEOUT_MS", 5000)
RTSP_FFMPEG_STIMEOUT_US = _int("RTSP_FFMPEG_STIMEOUT_US", 5000000)
RTSP_RECONNECT_BACKOFF_SEC = _float("RTSP_RECONNECT_BACKOFF_SEC", 0.2)
RTSP_READ_FAIL_BACKOFF_SEC = _float("RTSP_READ_FAIL_BACKOFF_SEC", 0.5)
RTSP_READER_JOIN_TIMEOUT_SEC = _float("RTSP_READER_JOIN_TIMEOUT_SEC", 2.0)

# ============================================================================
# 9. ENGINE LOOP
# ============================================================================
# A per-camera batch taking longer than this logs a WARNING.
SLOW_BATCH_WARN_MS = _float("SLOW_BATCH_WARN_MS", 150.0)
# How often (wall-clock seconds) the engine logs its rolling
# per-camera frame-count summary line.
PIPELINE_STATS_LOG_INTERVAL_SEC = _float("PIPELINE_STATS_LOG_INTERVAL_SEC", 5.0)

# ============================================================================
# 10. HEARTBEAT
# ============================================================================
HEARTBEAT_INTERVAL_SEC = _float("HEARTBEAT_INTERVAL_SEC", 10.0)
HEARTBEAT_TTL_SEC = _int("HEARTBEAT_TTL_SEC", 30)

# ============================================================================
# 11. LOGGING / API
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# text | json — see common/firecore/logging_setup.py.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
API_PORT = _int("API_PORT", 8012)
