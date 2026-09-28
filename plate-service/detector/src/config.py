"""
config.py (detector)
--------------------------------------------------------------------
Every tunable the plate detector service reads, all from the
environment so dev vs prod stays "same image, different .env" — the
pattern the face module's config.py established, applied here.

Split into sections matching the reference video_processor.py /
alpr_api.py, minus everything that is OCR-only (PaddleOCR model
paths, confidence threshold, plate-format regexes) — those now live
in ocr_service/src/config.py, next to the code that actually uses
them.
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
REDIS_MODULE = os.getenv("REDIS_MODULE", "plate")
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
# Plain torch CPU inference — this is a deliberate rollback from an
# earlier OpenVINO AsyncInferQueue experiment (it hit a thread-count
# regression from a container-wide OMP_NUM_THREADS=1 leftover from the
# old per-process model; see git history / the earlier debugging
# session if you want to revisit OpenVINO later). "auto" tries CUDA
# first and falls back to CPU. Every engine subprocess resolves this
# independently for itself (source of truth for what IT runs on);
# main.py also resolves it once, read-only, in the parent purely to
# plan engine topology (see cpu_topology.py) before any engine is
# spawned — spawned children get a fresh interpreter either way (this
# module uses mp.get_context("spawn"), never fork), so that read-only
# torch.cuda.is_available() probe in the parent carries none of the
# fork-time CUDA-context hazards this comment used to warn about. The
# parent still never builds a model or runs inference itself.
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "auto")
STRICT_DEVICE = _bool("STRICT_DEVICE", "false")
# Torch CPU threads PER ENGINE PROCESS. 0 = "cpu_count - 1" (leaves one
# core for reader/OCR-client threads), matching the reference Engine.
TORCH_NUM_THREADS = _int("TORCH_NUM_THREADS", 0)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.pt")
IMG_SIZE = _int("DETECTION_IMG_SIZE", 480)
CONF_THRESHOLD = _float("DETECTION_CONF_THRESHOLD", 0.25)
# class 0 = Car, class 1 = Motorcycle — verbatim from the reference
# alpr_service.py's class_labels and ocr_worker.py's voted_class split.
CLASS_LABELS = {0: "Car", 1: "Motorcycle"}

# Run YOLO detection on 1-out-of-N camera frames instead of every frame.
# Frames in between are not sent to the model at all -- the tracker coasts
# them via BYTETracker's own Kalman prediction (see tracker.update(dt=...)),
# so a value of 1 (the default) means "every frame, unchanged behavior" and
# a value of 3 means "detect on frame 0, 3, 6, ... coast frames 1-2, 4-5, ...".
# This is the main CPU-throughput lever for running several cameras per
# engine: it divides the inference cost per camera by N with the tracker
# absorbing the gap, instead of the batch simply falling behind and the
# engine silently dropping frames under overload.
DETECT_EVERY_N_FRAMES = _int("DETECT_EVERY_N_FRAMES", 1)

# ============================================================================
# 4. ENGINE TOPOLOGY / SELF-HEALING
# ============================================================================
# A new engine subprocess is spawned past this many cameras.
MAX_CAMERAS_PER_ENGINE = _int("MAX_CAMERAS_PER_ENGINE", 6)

# How many engines to bring up idle on a completely fresh boot (no
# self-healing checkpoint in Redis yet). After the first run, the
# persisted engine_count from platecore.lifecycle takes over.
DEFAULT_ENGINE_COUNT = _int("DEFAULT_ENGINE_COUNT", 1)

# How often EngineManager checks whether the current camera spread
# could be consolidated into fewer engines (see engine_manager.rebalance
# — NEW, the reference EngineManager only ever grew, never shrank).
ENGINE_REBALANCE_INTERVAL_SEC = _float("ENGINE_REBALANCE_INTERVAL_SEC", 30.0)

# How long to wait for engine subprocesses to exit cleanly on shutdown.
ENGINE_SHUTDOWN_TIMEOUT_SEC = _float("ENGINE_SHUTDOWN_TIMEOUT_SEC", 30.0)

# ============================================================================
# 4b. CPU ENGINE TOPOLOGY (device == "cpu" only -- GPU always uses
# MAX_CAMERAS_PER_ENGINE/TORCH_NUM_THREADS above, unmodified, since a GPU
# wants bigger batches, not more processes). See cpu_topology.py for the
# sizing model this feeds; main.py calls it once at startup, before the
# EngineManager is constructed, to pick the actual max_cameras_per_engine
# and TORCH_NUM_THREADS this process will run with.
# ============================================================================
# single -- unchanged prior behavior: one engine batches up to
#           MAX_CAMERAS_PER_ENGINE cameras into one model.predict() call
#           per loop iteration.
# multi  -- one engine PROCESS per camera (CPU_MULTI_CAMERAS_PER_ENGINE,
#           default 1): same Engine code, same DETECT_EVERY_N_FRAMES,
#           same tracker config -- "recognition" is identical, only the
#           OS-level scheduling differs (real parallelism across
#           processes instead of in-process batching). Each engine's
#           torch thread pool is capped low (CPU_MULTI_TORCH_THREADS)
#           on purpose, so N concurrent single-camera processes don't
#           each try to grab (cores-1) threads and oversubscribe.
# auto   -- compute MAX_CAMERAS_PER_ENGINE for "single"-style batching
#           from core count + the estimated per-camera cost below,
#           instead of a hand-set cap.
CPU_ENGINE_MODE = os.getenv("CPU_ENGINE_MODE", "single").strip().lower()

# "multi" mode only:
CPU_MULTI_CAMERAS_PER_ENGINE = _int("CPU_MULTI_CAMERAS_PER_ENGINE", 1)
CPU_MULTI_TORCH_THREADS = _int("CPU_MULTI_TORCH_THREADS", 1)

# "auto" mode only -- capacity-planning inputs:
# cores held back for OS overhead + the per-camera RTSP capture/decode
# threads (rtsp_reader.py), never handed to torch's intra-op pool.
CPU_RESERVE_CORES = _int("CPU_RESERVE_CORES", 1)
# Measured baseline cost of ONE camera's forward pass at the current
# DETECTION_IMG_SIZE/model, with a full thread pool available (see the
# detector's own [STATS] "infer avg" / batch_size in the logs — that's
# where this number should come from, not a guess). Re-measure after
# changing DETECTION_IMG_SIZE, the model file, or the CPU itself.
CPU_MS_PER_CAMERA_FRAME = _float("CPU_MS_PER_CAMERA_FRAME", 28.0)
# What a camera's real input fps is assumed to be, for turning the ms
# estimate above into "how many cameras fit in one frame interval."
CAMERA_ASSUMED_FPS = _float("CAMERA_ASSUMED_FPS", 25.0)
# Fraction of the frame interval budgeted to inference alone, leaving
# the rest for tracking/crop-harvesting/debug-video/OCR-submit work in
# the same loop iteration. Lower = more conservative (fewer cameras/engine).
CPU_AUTO_SAFETY_MARGIN = _float("CPU_AUTO_SAFETY_MARGIN", 0.7)

# ============================================================================
# 5. CAMERA HEALTH
# ============================================================================
HEARTBEAT_TIMEOUT_SEC = _float("HEARTBEAT_TIMEOUT_SEC", 15.0)

# NEW — the reference alpr_api.py tore a camera's engine assignment
# down the instant a "disconnected"/"offline" event arrived, with no
# grace period, so a camera mid-reconnect churned its BYTETracker
# (and every in-flight track) on every network blip. Wait this long
# after an "offline" transition before tearing it down; an "online"
# event within the window cancels the pending teardown. Keep >= the
# RTSP read timeout.
CAMERA_OFFLINE_GRACE_SECONDS = _float("CAMERA_OFFLINE_GRACE_SECONDS", 10.0)

# ============================================================================
# 5b. INTERNAL PROCESSING-ERROR AUTO-RESTART (existing behavior, kept)
# ============================================================================
# Distinct from the offline-grace path above: this is for an internal
# engine error (a batch-inference exception, a transient resource
# shortage) rather than a physical camera disconnect. Ported from the
# reference alpr_api.py's _attempt_processing_error_restart.
ERROR_RESTART_MAX_RETRIES = _int("ERROR_RESTART_MAX_RETRIES", 1)
ERROR_RESTART_DELAY_SEC = _float("ERROR_RESTART_DELAY_SEC", 3.0)
ERROR_RESTART_COUNTER_RESET_AFTER_SEC = _float("ERROR_RESTART_COUNTER_RESET_AFTER_SEC", 600.0)

# ============================================================================
# 6. DEBUG OUTPUT (bind volume — never MinIO; see README storage split)
# ============================================================================
# Rolling, fully annotated MP4 per camera showing detections, tracks,
# triggers, crop quality and the OCR round-trip. See debug_recorder.py.
DEBUG_VIDEO_ENABLED = _bool("DEBUG_VIDEO_ENABLED", "false")
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "/debug")
DEBUG_VIDEO_SEGMENT_SECONDS = _float("DEBUG_VIDEO_SEGMENT_SECONDS", 240.0)
DEBUG_VIDEO_FPS = _float("DEBUG_VIDEO_FPS", 12.0)
DEBUG_VIDEO_MAX_SEGMENTS = _int("DEBUG_VIDEO_MAX_SEGMENTS", 12)
DEBUG_VIDEO_SCALE = _float("DEBUG_VIDEO_SCALE", 1.0)
DEBUG_VIDEO_EVERY_N = _int("DEBUG_VIDEO_EVERY_N", 1)
DEBUG_VIDEO_CODEC = os.getenv("DEBUG_VIDEO_CODEC", "mp4v")
DEBUG_VIDEO_EXT = os.getenv("DEBUG_VIDEO_EXT", ".mp4")
# These four are read directly by debug_recorder.py's own DebugConfig
# (it self-reads env, the same pattern face_service's DebugConfig uses —
# see that module's config.py for precedent). Declared here too, purely
# so every DEBUG_VIDEO_* knob is discoverable in ONE file instead of
# only inside debug_recorder.py; changing the value here has no effect
# by itself, only the env var does.
DEBUG_VIDEO_JSONL = _bool("DEBUG_VIDEO_JSONL", "true")
DEBUG_VIDEO_GHOST_FRAMES = _int("DEBUG_VIDEO_GHOST_FRAMES", 45)
DEBUG_VIDEO_EVENT_LINES = _int("DEBUG_VIDEO_EVENT_LINES", 14)
DEBUG_VIDEO_TRAIL = _int("DEBUG_VIDEO_TRAIL", 30)
DEBUG_VIDEO_PANEL_WIDTH = _int("DEBUG_VIDEO_PANEL_WIDTH", 430)

# New per-section visual-debug output (debug_extras.py), separate from
# the always-annotated-video system above: a small labelled-grid JPEG
# saved every time a track is submitted to OCR, showing exactly which
# crops/best-frame were sent — the fastest way to see "why did OCR get
# a bad crop" without scrubbing the full debug video.
DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED = _bool("DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED", "false")
DEBUG_OCR_SUBMISSION_MONTAGE_DIR = os.getenv(
    "DEBUG_OCR_SUBMISSION_MONTAGE_DIR", os.path.join(DEBUG_VIDEO_DIR, "ocr_submissions")
)
DEBUG_OCR_SUBMISSION_MONTAGE_MAX_FILES = _int("DEBUG_OCR_SUBMISSION_MONTAGE_MAX_FILES", 200)

# ============================================================================
# 7. TRIGGER / TRACK CRITERIA (verbatim thresholds from video_processor.py)
# ============================================================================
MIN_TRACK_AGE_FOR_CROSSING = _int("MIN_TRACK_AGE_FOR_CROSSING", 5)
MIN_CONFIDENCE_FOR_CROSSING = _float("MIN_CONFIDENCE_FOR_CROSSING", 0.40)
CROSSING_COOLDOWN_FRAMES = _int("CROSSING_COOLDOWN_FRAMES", 30)

ROI_ENTRY_CONFIRMATION_FRAMES = _int("ROI_ENTRY_CONFIRMATION_FRAMES", 3)
MIN_CONFIDENCE_FOR_ROI = _float("MIN_CONFIDENCE_FOR_ROI", 0.35)

STOP_TIME_SECONDS = _float("STOP_TIME_SECONDS", 3.0)
STOP_VELOCITY_THRESHOLD = _float("STOP_VELOCITY_THRESHOLD", 8.0)  # px/s
STOP_MIN_SAMPLES = _int("STOP_MIN_SAMPLES", 10)

# ---- OCR dispatch (control hub era) ------------------------------------
# OCR_CONF_SKIP_THRESHOLD and OCR_FINALIZE_TIMEOUT_SEC moved to the
# control hub as PLATE_SATISFIED_CONF (now counts VALID reads only) and
# PLATE_FINALIZE_TIMEOUT_SEC. The detector only decides whether it may
# send crops right now — see platecore/hub.py.
#
# How long a submitted task blocks the next submission for the same
# track if the hub never acks its result (previously: forever — a lost
# OCR result blocked every later trigger on that track).
SUBMIT_TIMEOUT_SEC = _float("SUBMIT_TIMEOUT_SEC", 15.0)
# Low-rate per-track heartbeat to the hub.
TRACK_UPDATE_INTERVAL_SEC = _float("TRACK_UPDATE_INTERVAL_SEC", 5.0)

# ============================================================================
# 7b. TRIGGER HISTORY BUFFERS (triggers.py's TriggerTrackState — verbatim
#     bounds, previously bare literals inside add_observation/recent_velocity)
# ============================================================================
TRIGGER_POSITION_HISTORY_MAX = _int("TRIGGER_POSITION_HISTORY_MAX", 30)
TRIGGER_CONFIDENCE_HISTORY_MAX = _int("TRIGGER_CONFIDENCE_HISTORY_MAX", 10)
TRIGGER_VELOCITY_WINDOW = _int("TRIGGER_VELOCITY_WINDOW", 10)

# ============================================================================
# 8. QUALITY GATES (verbatim from video_processor.py's _quality_check_crop)
# ============================================================================
SHARPNESS_MIN_THRESHOLD = _float("SHARPNESS_MIN_THRESHOLD", 100.0)
RESOLUTION_MIN_AREA = _int("RESOLUTION_MIN_AREA", 1000)
CLASS_0_ASPECT_RATIO_MIN = _float("CLASS_0_ASPECT_RATIO_MIN", 1.5)   # car
CLASS_0_ASPECT_RATIO_MAX = _float("CLASS_0_ASPECT_RATIO_MAX", 7.5)
GENERIC_ASPECT_RATIO_MIN = _float("GENERIC_ASPECT_RATIO_MIN", 1.0)   # motorcycle
GENERIC_ASPECT_RATIO_MAX = _float("GENERIC_ASPECT_RATIO_MAX", 5.5)

# ============================================================================
# 9. TRACK AGGREGATION & BUFFERS (verbatim)
# ============================================================================
DEFAULT_ABSENT_FRAMES = _int("DEFAULT_ABSENT_FRAMES", 30)
DEFAULT_MIN_SEEN_FRAMES = _int("DEFAULT_MIN_SEEN_FRAMES", 8)
DEFAULT_MIN_CROPS_TO_FINALIZE = _int("DEFAULT_MIN_CROPS_TO_FINALIZE", 1)
DEFAULT_N_BEST_CROPS = _int("DEFAULT_N_BEST_CROPS", 5)
DEFAULT_CONF_DIGITS = _int("DEFAULT_CONF_DIGITS", 2)

# ============================================================================
# 9b. TRACKER (BYTETracker / tracker.py's PlateTrackerConfig — full surface)
# ============================================================================
# Previously, engine.py built its OWN minimal local `TrackerConfig`
# (track_thresh/match_thresh/track_buffer/nms_thresh/mot20 only) and
# passed that to BYTETracker, even though tracker.py ships a much
# richer PlateTrackerConfig with ~19 additional tuning fields (the
# "FIX 1-7" enhancements: young-track survival, new-track motion
# seeding, a recovery pass for briefly-lost tracks, and GMC camera-jolt
# compensation). BYTETracker.__init__ reads every field via
# getattr(args, name, default), so those extra fields were always
# silently falling back to PlateTrackerConfig's own hardcoded defaults
# — current behavior didn't change, but none of them were reachable
# from .env. engine.py now imports and uses PlateTrackerConfig directly;
# every default below matches PlateTrackerConfig's own declared default
# exactly, so an unset .env reproduces prior behavior bit-for-bit.
TRACKER_TRACK_THRESH = _float("TRACKER_TRACK_THRESH", 0.5)
TRACKER_MATCH_THRESH = _float("TRACKER_MATCH_THRESH", 0.99)
TRACKER_TRACK_BUFFER = _int("TRACKER_TRACK_BUFFER", 60)
TRACKER_NMS_THRESH = _float("TRACKER_NMS_THRESH", 0.5)
TRACKER_MOT20 = _bool("TRACKER_MOT20", "false")
TRACKER_SECOND_THRESH = _float("TRACKER_SECOND_THRESH", 0.5)
TRACKER_DUPLICATE_THRESH = _float("TRACKER_DUPLICATE_THRESH", 0.15)
# Young-track survival: give a just-created track a few extra frames of
# grace before BYTETrack's normal removal rules kick in.
TRACKER_PREDICT_UNCONFIRMED = _bool("TRACKER_PREDICT_UNCONFIRMED", "true")
TRACKER_UNCONFIRMED_THRESH = _float("TRACKER_UNCONFIRMED_THRESH", 0.9)
TRACKER_UNCONFIRMED_MAX_MISS = _int("TRACKER_UNCONFIRMED_MAX_MISS", 5)
# New-track motion seeding: give a brand-new track an initial velocity
# estimate instead of assuming it starts stationary.
TRACKER_NEW_TRACK_VEL_STD_SCALE = _float("TRACKER_NEW_TRACK_VEL_STD_SCALE", 3.0)
TRACKER_SEED_VELOCITY_ON_FIRST_UPDATE = _bool("TRACKER_SEED_VELOCITY_ON_FIRST_UPDATE", "true")
TRACKER_SEED_MAX_RATIO = _float("TRACKER_SEED_MAX_RATIO", 1.5)
TRACKER_RESEED_AFTER_GAP = _float("TRACKER_RESEED_AFTER_GAP", 3.0)
# Recovery pass: a second, more permissive matching attempt for tracks
# that would otherwise be lost this frame (occlusion, a missed
# detection), using shape/class similarity within a growing radius.
TRACKER_RECOVERY_ENABLED = _bool("TRACKER_RECOVERY_ENABLED", "true")
TRACKER_RECOVERY_THRESH = _float("TRACKER_RECOVERY_THRESH", 0.7)
TRACKER_RECOVERY_EXPANSION = _float("TRACKER_RECOVERY_EXPANSION", 0.5)
TRACKER_RECOVERY_BASE_RADIUS = _float("TRACKER_RECOVERY_BASE_RADIUS", 1.5)
TRACKER_RECOVERY_RADIUS_GROWTH = _float("TRACKER_RECOVERY_RADIUS_GROWTH", 0.4)
TRACKER_RECOVERY_MAX_RADIUS = _float("TRACKER_RECOVERY_MAX_RADIUS", 4.0)
TRACKER_RECOVERY_SHAPE_WEIGHT = _float("TRACKER_RECOVERY_SHAPE_WEIGHT", 0.3)
TRACKER_RECOVERY_CLASS_PENALTY = _float("TRACKER_RECOVERY_CLASS_PENALTY", 0.15)
# GMC (Global Motion Compensation): corrects track positions for a
# sudden camera jolt/pan instead of misreading it as every track moving.
TRACKER_GMC_ENABLED = _bool("TRACKER_GMC_ENABLED", "true")
TRACKER_GMC_MIN_PAIRS = _int("TRACKER_GMC_MIN_PAIRS", 2)
TRACKER_GMC_MAX_SHIFT_RATIO = _float("TRACKER_GMC_MAX_SHIFT_RATIO", 0.25)
TRACKER_MAX_REMOVED_HISTORY = _int("TRACKER_MAX_REMOVED_HISTORY", 512)

# ============================================================================
# 9c. RTSP READER (rtsp_reader.py — previously bare literals)
# ============================================================================
RTSP_OPEN_TIMEOUT_MS = _int("RTSP_OPEN_TIMEOUT_MS", 5000)
RTSP_READ_TIMEOUT_MS = _int("RTSP_READ_TIMEOUT_MS", 5000)
RTSP_FFMPEG_STIMEOUT_US = _int("RTSP_FFMPEG_STIMEOUT_US", 5000000)
RTSP_RECONNECT_BACKOFF_SEC = _float("RTSP_RECONNECT_BACKOFF_SEC", 0.2)
RTSP_READ_FAIL_BACKOFF_SEC = _float("RTSP_READ_FAIL_BACKOFF_SEC", 0.5)
RTSP_READER_JOIN_TIMEOUT_SEC = _float("RTSP_READER_JOIN_TIMEOUT_SEC", 2.0)

# ============================================================================
# 9d. ENGINE LOOP (engine.py — previously bare literals)
# ============================================================================
# A per-camera batch taking longer than this logs a WARNING (helps spot
# a stalled GPU/CPU inference or an oversized batch before it snowballs
# into camera health timeouts).
SLOW_BATCH_WARN_MS = _float("SLOW_BATCH_WARN_MS", 150.0)
# How often (wall-clock seconds) each camera logs its rolling
# fps/latency/queue-depth summary line.
PIPELINE_STATS_LOG_INTERVAL_SEC = _float("PIPELINE_STATS_LOG_INTERVAL_SEC", 5.0)

# ============================================================================
# 9e. HEARTBEAT
# ============================================================================
HEARTBEAT_INTERVAL_SEC = _float("HEARTBEAT_INTERVAL_SEC", 10.0)
HEARTBEAT_TTL_SEC = _int("HEARTBEAT_TTL_SEC", 30)

# ============================================================================
# 10. LOGGING / API
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# text | json — see common/platecore/logging_setup.py.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
API_PORT = _int("API_PORT", 8010)
