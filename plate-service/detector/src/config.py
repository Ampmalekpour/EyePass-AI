"""
config.py (detector)
--------------------------------------------------------------------
Every setting of plate_detector lives HERE, as a plain value. Change it
in this file and rebuild/restart.

.env (through compose.yaml) only decides:
  * which device and model run:
        DETECTION_DEVICE      gpu | cpu | auto
        DETECTION_GPU_MODEL   plate_v8n_480 | plate_v8s_640   (aliases: v8n, v8s)
        DETECTION_CPU_MODEL   openvino_fp32 | openvino_int8 | onnx
  * where things are (deployment): Redis, the RTSP relay.

Everything else (thresholds, tracker, triggers, engine topology, logs,
debug video) is decided below.
--------------------------------------------------------------------
"""

import os

from platecore import logging_setup


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


# ============================================================================
# 0. FROM .env — device / models / deployment
# ============================================================================
DETECTION_DEVICE = _env("DETECTION_DEVICE", "auto").lower()
DETECTION_GPU_MODEL = _env("DETECTION_GPU_MODEL", "plate_v8n_480")
DETECTION_CPU_MODEL = _env("DETECTION_CPU_MODEL", "openvino_fp32").lower()

REDIS_MODULE = _env("REDIS_MODULE", "plate")
REDIS_HOST = _env("REDIS_HOST", "redis")
REDIS_PORT = int(_env("REDIS_PORT", "6379"))
REDIS_DB = int(_env("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = os.getenv("REDIS_URL", "")  # if set, wins over host/port/db/password
# Frames always come from the relay (<base>/<camera_id>), never from the camera.
MTX_RTSP_BASE_URL = _env("MTX_RTSP_BASE_URL", "rtsp://mediamtx:8554")

# In-container paths (compose.yaml mounts the host folders here).
MODEL_ROOT = _env("DETECTION_MODEL_ROOT", "/models")
DEBUG_ROOT = "/debug"

# ============================================================================
# 1. MODELS
# ============================================================================
# Short names accepted in .env.
MODEL_ALIASES = {
    "v8n": "plate_v8n_480", "v8n_480": "plate_v8n_480",
    "v8s": "plate_v8s_640", "v8s_640": "plate_v8s_640",
    "openvino": "openvino_fp32", "ov_fp32": "openvino_fp32", "fp32": "openvino_fp32",
    "ov_int8": "openvino_int8", "ov_int8_box": "openvino_int8", "int8": "openvino_int8",
}
# The model folder the CPU variants are taken from (only plate_v8n_480
# has ONNX / OpenVINO exports). Folder layout: README "Models".
CPU_MODEL_NAME = "plate_v8n_480"
# Where each variant is listed in <model>/export_info.yaml (first match wins) ...
MODEL_MANIFEST_KEYS = {
    "pt": ["pt"],
    "onnx": ["onnx"],
    "openvino_fp32": ["openvino.ov_fp32"],
    "openvino_int8": ["openvino.ov_int8_box", "openvino.ov_int8"],
}
# ... and the file names tried when there is no manifest.
MODEL_FILE_PATTERNS = {
    "pt": ["{name}.pt", "*.pt"],
    "onnx": ["{name}_*x*.onnx", "{name}.onnx", "*.onnx"],
    "openvino_fp32": ["{name}_fp32_openvino_model", "{name}_openvino_model"],
    "openvino_int8": ["{name}_int8_box_openvino_model", "{name}_int8_openvino_model"],
}
# class 0 = car plate, class 1 = motorcycle plate (fallback when the
# manifest has no `names`).
CLASS_LABELS = {0: "Car", 1: "Motorcycle"}

# ============================================================================
# 2. INFERENCE (same values as the multi-stream benchmark)
# ============================================================================
CONF_THRESHOLD = 0.25
NMS_IOU = 0.7
MAX_DET = 300
WARMUP_RUNS = 10

# ---- GPU: Ultralytics .pt, one batched predict() per engine loop -----------
GPU_DEVICE = "cuda:0"
GPU_HALF = False              # FP16 on the GPU (the benchmark's pt_gpu used True)
# A requested-but-missing GPU: True = startup error, False = run the CPU pipeline.
STRICT_DEVICE = False

# ---- CPU: one model instance per camera, run in parallel -------------------
OV_DEVICE = "intel:cpu"       # OpenVINO through Ultralytics, never AUTO
# onnx: CPU threads split between the per-camera sessions; None = all logical CPUs
ONNX_CPU_THREADS = None
# openvino_int8: duplicate fix applied in Ultralytics' postprocess
INT8_FIX = {"nms_iou": 0.5,   # remove a box if IoU with a stronger box >= this
            "iomin": 0.7,     # ... or if the stronger box covers >= 70% of the smaller one
            "agnostic": True,  # across classes (two plates can't occupy the same place)
            "merge_iou": 0.6}  # raw candidates with IoU >= 0.6 are averaged into the kept box
# CPU model failed to load -> log an ERROR and run the .pt with Ultralytics
# on CPU (False = the engine fails instead).
BACKEND_FALLBACK_TO_PT = True
# cv2's own thread pool (ROI crop / resize) — kept small so it doesn't
# compete with inference.
CV2_NUM_THREADS = 2
# Detect on 1 of N frames per camera; the tracker predicts the frames in
# between. 1 = every frame.
DETECT_EVERY_N_FRAMES = 1

# ============================================================================
# 2b. REAL-TIME CAPACITY (CPU) — how many cameras can this machine serve?
# ============================================================================
# Real-time = every camera is served at >= REALTIME_MIN_FPS. For an engine
# with n cameras that means one inference loop must finish within
#     budget = DETECT_EVERY_N_FRAMES * 1000 / REALTIME_MIN_FPS * CAPACITY_SAFETY_MARGIN  ms
# (25 fps, N=1, margin 0.7 -> 28 ms). The margin leaves room for what the
# calibration does not include: RTSP decoding, tracking, OCR hand-off.
# At startup engine 0 measures the real loop time for 1, 2, 3, ... cameras on
# this machine with this model and prints the capacity table (🧪 [CAPACITY]);
# the largest n that fits the budget is the device's real-time capacity,
# stored in Redis (<module>:internal:detector:capacity). Every camera that is
# attached afterwards is checked against it (✅ / 🚨 [CAPACITY]).
REALTIME_MIN_FPS = 25.0
CAPACITY_SAFETY_MARGIN = 0.7
CAPACITY_CALIBRATION_ENABLED = True
CAPACITY_MAX_CAMERAS_TESTED = 12     # upper bound of the sweep
CAPACITY_STOP_AFTER_FAILS = 2        # stop the sweep after this many consecutive misses
CAPACITY_WARMUP_ROUNDS = 3           # untimed loops per camera count
CAPACITY_ROUNDS = 20                 # timed loops per camera count (p95 over these)
CAPACITY_FRAME_SIZE = (1080, 1920)   # (h, w) of the dummy frames (your cameras' resolution)
# When the cameras exceed the measured capacity, lower the detection rate per
# camera smoothly (25 -> 23.4 -> 20 -> ... fps; any value, not just 25/12.5/8.3)
# to exactly what fits, instead of missing frames at random; back up as soon as
# they fit again. The tracker coasts the frames in between. Every change is
# logged (🐢 / 🐇). Switch it in .env: CAPACITY_AUTO_DEGRADE=true|false
# (false = warn only). DETECT_MIN_FPS is the floor.
CAPACITY_AUTO_DEGRADE = _env("CAPACITY_AUTO_DEGRADE", "true").lower() in ("1", "true", "yes", "on")
DETECT_MIN_FPS = 8.0

# ============================================================================
# 3. ENGINES (EngineManager)
# ============================================================================
# Cameras per engine process. A new engine starts past this many.
#   GPU: cameras batched into one predict() per loop.
#   CPU: cameras sharing one process (one model instance each). Keep it at
#        the capacity measured with tools/bench_multistream.py so ALL cameras
#        share one process — two CPU engines would compete for the same cores.
MAX_CAMERAS_PER_ENGINE = 6
CPU_MAX_CAMERAS_PER_ENGINE = 16   # CPU: ONE engine serves all cameras (extra engines would only share the same cores)
DEFAULT_ENGINE_COUNT = 1          # engines started idle on a fresh boot
ENGINE_REBALANCE_INTERVAL_SEC = 30.0
ENGINE_SHUTDOWN_TIMEOUT_SEC = 30.0
# Torch threads for the .pt model on CPU (fallback only). 0 = cpu_count - 1.
# (an env var of the same name overrides it — debugging only, not an .env knob)
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0") or 0)
# Torch threads when the CPU model is OpenVINO/ONNX. Torch computes nothing
# there, but Ultralytics still runs its pre/post-processing (normalise, NMS)
# through torch — with torch's default (all cores) in every camera thread it
# fights the inference runtime for the cores (measured: loop 20-45% slower,
# p95 much worse). 1 = no contention. 0 = leave torch's default.
CPU_TORCH_NUM_THREADS = 1
CAMERA_ASSUMED_FPS = 25.0         # tracker time base (see engine.add_camera)

# ============================================================================
# 4. CAMERA HEALTH / RESTARTS
# ============================================================================
HEARTBEAT_TIMEOUT_SEC = 15.0
CAMERA_OFFLINE_GRACE_SECONDS = 10.0
ERROR_RESTART_MAX_RETRIES = 1
ERROR_RESTART_DELAY_SEC = 3.0
ERROR_RESTART_COUNTER_RESET_AFTER_SEC = 600.0

# ============================================================================
# 5. TRIGGERS / TRACKS / CROPS
# ============================================================================
MIN_TRACK_AGE_FOR_CROSSING = 5
MIN_CONFIDENCE_FOR_CROSSING = 0.40
CROSSING_COOLDOWN_FRAMES = 30
ROI_ENTRY_CONFIRMATION_FRAMES = 3
MIN_CONFIDENCE_FOR_ROI = 0.35
STOP_TIME_SECONDS = 3.0
STOP_VELOCITY_THRESHOLD = 8.0  # px/s
STOP_MIN_SAMPLES = 10
TRIGGER_POSITION_HISTORY_MAX = 30
TRIGGER_CONFIDENCE_HISTORY_MAX = 10
TRIGGER_VELOCITY_WINDOW = 10

# OCR hand-off to the control hub
SUBMIT_TIMEOUT_SEC = 15.0
TRACK_UPDATE_INTERVAL_SEC = 5.0

# crop quality gates
SHARPNESS_MIN_THRESHOLD = 100.0
RESOLUTION_MIN_AREA = 1000
CLASS_0_ASPECT_RATIO_MIN = 1.5   # car
CLASS_0_ASPECT_RATIO_MAX = 7.5
GENERIC_ASPECT_RATIO_MIN = 1.0   # motorcycle
GENERIC_ASPECT_RATIO_MAX = 5.5

# track aggregation
DEFAULT_ABSENT_FRAMES = 30
DEFAULT_MIN_SEEN_FRAMES = 8
DEFAULT_MIN_CROPS_TO_FINALIZE = 1
DEFAULT_N_BEST_CROPS = 5
DEFAULT_CONF_DIGITS = 2

# ============================================================================
# 6. TRACKER (BYTETracker / tracker.py PlateTrackerConfig)
# ============================================================================
TRACKER_TRACK_THRESH = 0.5
TRACKER_MATCH_THRESH = 0.99
TRACKER_TRACK_BUFFER = 60
TRACKER_NMS_THRESH = 0.5
TRACKER_MOT20 = False
TRACKER_SECOND_THRESH = 0.5
TRACKER_DUPLICATE_THRESH = 0.15
TRACKER_PREDICT_UNCONFIRMED = True
TRACKER_UNCONFIRMED_THRESH = 0.9
TRACKER_UNCONFIRMED_MAX_MISS = 5
TRACKER_NEW_TRACK_VEL_STD_SCALE = 3.0
TRACKER_SEED_VELOCITY_ON_FIRST_UPDATE = True
TRACKER_SEED_MAX_RATIO = 1.5
TRACKER_RESEED_AFTER_GAP = 3.0
TRACKER_RECOVERY_ENABLED = True
TRACKER_RECOVERY_THRESH = 0.7
TRACKER_RECOVERY_EXPANSION = 0.5
TRACKER_RECOVERY_BASE_RADIUS = 1.5
TRACKER_RECOVERY_RADIUS_GROWTH = 0.4
TRACKER_RECOVERY_MAX_RADIUS = 4.0
TRACKER_RECOVERY_SHAPE_WEIGHT = 0.3
TRACKER_RECOVERY_CLASS_PENALTY = 0.15
TRACKER_GMC_ENABLED = True
TRACKER_GMC_MIN_PAIRS = 2
TRACKER_GMC_MAX_SHIFT_RATIO = 0.25
TRACKER_MAX_REMOVED_HISTORY = 512

# ============================================================================
# 7. RTSP READER
# ============================================================================
RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 5000
RTSP_FFMPEG_STIMEOUT_US = 5000000
RTSP_RECONNECT_BACKOFF_SEC = 0.2
RTSP_READ_FAIL_BACKOFF_SEC = 0.5
RTSP_FFMPEG_THREADS = 1           # FFmpeg decode threads per stream (0 = all cores)
RTSP_READER_JOIN_TIMEOUT_SEC = 6.0  # > RTSP_READ_TIMEOUT_MS, so stop() never releases mid-read

# ============================================================================
# 8. LOGS
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "text"               # text | json
# 📊 [STATS] engine summary every N inference loops
STATS_EVERY_N_BATCHES = 50
# ⏱️ [PERF] per-camera line every N seconds: fps in / processed, missed %,
# pre / infer / post ms per frame, latency (frame arrival -> result)
PERF_LOG_INTERVAL_SEC = 10.0
# one loop slower than this logs ⚠️ [INFER-SLOW]
SLOW_BATCH_WARN_MS = 150.0
# per-track lines ([TRACK-NEW], [TRACK-END], [OCR-SUBMIT], [OCR-RESULT], ...)
LOG_TRACK_EVENTS = True
HEARTBEAT_INTERVAL_SEC = 10.0
HEARTBEAT_TTL_SEC = 30
API_PORT = 8010

# ============================================================================
# 9. VISUAL DEBUG (local bind mount ./debug_video — never MinIO)
# ============================================================================
# Annotated video per camera: detections, tracks, triggers, crop quality,
# OCR round-trip. Expensive — enable only while looking at it.
DEBUG_VIDEO_ENABLED = False
DEBUG_VIDEO_DIR = DEBUG_ROOT
DEBUG_VIDEO_SEGMENT_SECONDS = 240.0
DEBUG_VIDEO_FPS = 12.0
DEBUG_VIDEO_MAX_SEGMENTS = 12
DEBUG_VIDEO_SCALE = 1.0
DEBUG_VIDEO_EVERY_N = 1
DEBUG_VIDEO_CODEC = "mp4v"
DEBUG_VIDEO_EXT = ".mp4"
DEBUG_VIDEO_JSONL = True
DEBUG_VIDEO_GHOST_FRAMES = 45
DEBUG_VIDEO_EVENT_LINES = 14
DEBUG_VIDEO_TRAIL = 30
DEBUG_VIDEO_PANEL_WIDTH = 430
# A labelled-grid JPEG of the crops sent to OCR, per submission.
DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED = False
DEBUG_OCR_SUBMISSION_MONTAGE_DIR = os.path.join(DEBUG_ROOT, "ocr_submissions")
DEBUG_OCR_SUBMISSION_MONTAGE_MAX_FILES = 200

logging_setup.configure(LOG_LEVEL, LOG_FORMAT)
