"""
config.py (ocr_service)
--------------------------------------------------------------------
Every tunable the OCR service reads. The PaddleOCR model directory
layout, preprocessing sizes, voting logic and plate-format validation
regexes are verbatim from the reference ocr_worker.py — changing any
of the geometry/regex constants changes recognition behavior, so they
stay exactly where they were, just env-overridable like everything
else in this system.

Model paths are now built from ONE base directory (OCR_MODELS_DIR)
instead of being hardcoded relative paths (the reference ocr_worker.py
used bare "./PadOcr/..." — relative to whatever the process's cwd
happened to be, which only worked because alpr_api.py always ran from
the same directory as ocr_worker.py). The five subfolder/file names
themselves (en_PP-OCRv3_det_infer, rec_svrt_fa_final_1,
ch_ppocr_mobile_v2.0_cls_infer, rec_svrt_motor, Final_Dict.txt) are
kept byte-for-byte so the operator's existing PadOcr/ folder can be
bind-mounted as-is at OCR_MODELS_DIR — see the README's "What you need
to provide" table.
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
REDIS_URL = os.getenv("REDIS_URL", "")

# ============================================================================
# 2. PADDLEOCR MODELS — one base dir, five fixed subpaths (verbatim names)
# ============================================================================
OCR_MODELS_DIR = os.getenv("OCR_MODELS_DIR", "/models/ocr")

DET_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "en_PP-OCRv3_det_infer")
CAR_REC_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "rec_svrt_fa_final_1")
CLS_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "ch_ppocr_mobile_v2.0_cls_infer")
MOTOR_REC_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "rec_svrt_motor")
REC_CHAR_DICT_PATH = os.path.join(OCR_MODELS_DIR, "Final_Dict.txt")

# The reference ocr_worker.py hardcoded use_gpu=False for both PaddleOCR
# pipelines (CPU-only OCR, even on a GPU box, was a deliberate choice
# there — OCR crops are tiny and CPU is plenty fast; the GPU stays free
# for detection). Kept as the default here, but now overridable.
OCR_USE_GPU = _bool("OCR_USE_GPU", "false")

# ============================================================================
# 3. VALIDATION / VOTING (verbatim thresholds from ocr_worker.py)
# ============================================================================
CONF_THRESHOLD = _float("OCR_CONF_THRESHOLD", 0.7)
SAVE_ALL_CROPS = _bool("OCR_SAVE_ALL_CROPS", "false")

# CLAHE (contrast-limited adaptive histogram equalization) applied to
# the car-plate crop before OCR — a real operational tunable (lighting
# conditions vary a lot per site), unlike the fixed model-input resize
# dimensions below it in worker.py, which must stay exactly what the
# rec models were trained/calibrated against.
OCR_CLAHE_CLIP_LIMIT = _float("OCR_CLAHE_CLIP_LIMIT", 2.0)
OCR_CLAHE_GRID_SIZE = _int("OCR_CLAHE_GRID_SIZE", 8)

# Motorcycle plate OCR discards detected text boxes smaller than this
# fraction of the largest box's area before voting — filters out noise
# boxes (screws, frame edges) that PaddleOCR's detector occasionally
# fires on.
OCR_MOTOR_MIN_BOX_AREA_RATIO = _float("OCR_MOTOR_MIN_BOX_AREA_RATIO", 0.05)

# ============================================================================
# 4. WORKER POOL / SELF-HEALING
# ============================================================================
# How many worker subprocesses to bring up idle on a completely fresh
# boot (no self-healing checkpoint in Redis yet). After the first run,
# the persisted worker_count from platecore.lifecycle takes over.
DEFAULT_WORKER_COUNT = _int("DEFAULT_WORKER_COUNT", 3)

# How long to wait for a freshly spawned worker to finish loading its
# two PaddleOCR pipelines + warmup before start_idle()/self_heal() gives
# up waiting on it (the worker keeps loading in the background
# regardless; this only bounds how long startup blocks).
WORKER_LOAD_TIMEOUT_SEC = _float("WORKER_LOAD_TIMEOUT_SEC", 180.0)

# How long to wait for a worker subprocess to exit cleanly on shutdown
# before it is terminated forcibly.
WORKER_SHUTDOWN_TIMEOUT_SEC = _float("WORKER_SHUTDOWN_TIMEOUT_SEC", 30.0)

# How often the pool's watchdog checks that every worker subprocess it
# expects to be alive actually is, respawning any that crashed — the
# OCR service's own self-healing at the worker-process level, distinct
# from (and in addition to) the whole-service self-healing that
# ServiceLifecycle.self_heal() does on a full container restart.
WATCHDOG_INTERVAL_SEC = _float("WATCHDOG_INTERVAL_SEC", 5.0)

# How long worker.run()'s main loop sleeps between checks while idle
# (processing_event not set), and how long it backs off after an
# unexpected exception in the loop body — both were bare literals
# before.
OCR_IDLE_POLL_INTERVAL_SEC = _float("OCR_IDLE_POLL_INTERVAL_SEC", 0.2)
OCR_TASK_POP_RETRY_BACKOFF_SEC = _float("OCR_TASK_POP_RETRY_BACKOFF_SEC", 1.0)

HEARTBEAT_INTERVAL_SEC = _float("HEARTBEAT_INTERVAL_SEC", 10.0)
HEARTBEAT_TTL_SEC = _int("HEARTBEAT_TTL_SEC", 30)

# ============================================================================
# 4b. DEBUG OUTPUT (bind volume — never MinIO; see README storage split)
# ============================================================================
# A structured, one-line-per-task decision trace: queue latency,
# voted vehicle class, per-candidate OCR confidences, which candidate
# won the vote, validation pass/fail, and total elapsed_ms — see
# worker.py's _process_task and common/platecore/debugging.timed.
LOG_DECISION_TRACE = _bool("LOG_DECISION_TRACE", "true")

# A small labelled-grid JPEG per finalized task showing the crop(s)
# that went in, each OCR candidate's raw text + confidence, which one
# won, and the final validation result — the OCR-side analog of the
# detector's DEBUG_OCR_SUBMISSION_MONTAGE. Local disk only, same
# storage-split rule as the detector's debug video.
OCR_DEBUG_ROOT_DIR = os.getenv("OCR_DEBUG_ROOT_DIR", "/debug")
OCR_SAVE_DECISION_DEBUG = _bool("OCR_SAVE_DECISION_DEBUG", "false")
OCR_DECISION_DEBUG_DIR = os.getenv(
    "OCR_DECISION_DEBUG_DIR", os.path.join(OCR_DEBUG_ROOT_DIR, "decisions")
)
OCR_DECISION_DEBUG_MAX_FILES = _int("OCR_DECISION_DEBUG_MAX_FILES", 200)

# ============================================================================
# 5. LOGGING / API
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# text | json — see common/platecore/logging_setup.py.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
API_PORT = _int("API_PORT", 8011)
