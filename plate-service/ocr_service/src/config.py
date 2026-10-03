"""
config.py (ocr_service)
--------------------------------------------------------------------
Every setting of the OCR service. The PaddleOCR model directory
layout, preprocessing sizes, voting logic and plate-format validation
regexes are verbatim from the reference ocr_worker.py — changing any
of the geometry/regex constants changes recognition behavior, so they
stay exactly where they were. Every value is set in this file; .env
only picks the device (OCR_DEVICE) and the deployment (Redis, MinIO).

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

from platecore import logging_setup


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


# ============================================================================
# 0. FROM .env — device / deployment (MinIO is read by platecore.minio_store)
# ============================================================================
# cpu (default: crops are tiny, the GPU stays free for detection) | gpu
OCR_DEVICE = _env("OCR_DEVICE", "cpu").lower()
OCR_USE_GPU = OCR_DEVICE in ("gpu", "cuda", "true", "1")

REDIS_MODULE = _env("REDIS_MODULE", "plate")
REDIS_HOST = _env("REDIS_HOST", "redis")
REDIS_PORT = int(_env("REDIS_PORT", "6379"))
REDIS_DB = int(_env("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = os.getenv("REDIS_URL", "")

# ============================================================================
# 1. PADDLEOCR MODELS — one base dir (compose mounts OCR_MODELS_DIR here),
#    five fixed sub-paths (names verbatim from the reference ocr_worker.py)
# ============================================================================
OCR_MODELS_DIR = "/models/ocr"
DET_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "en_PP-OCRv3_det_infer")
CAR_REC_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "rec_svrt_fa_final_1")
CLS_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "ch_ppocr_mobile_v2.0_cls_infer")
MOTOR_REC_MODEL_DIR = os.path.join(OCR_MODELS_DIR, "rec_svrt_motor")
REC_CHAR_DICT_PATH = os.path.join(OCR_MODELS_DIR, "Final_Dict.txt")

# ============================================================================
# 2. RECOGNITION (verbatim thresholds from ocr_worker.py)
# ============================================================================
CONF_THRESHOLD = 0.7
SAVE_ALL_CROPS = False
# CLAHE contrast enhancement on car-plate crops before OCR
OCR_CLAHE_CLIP_LIMIT = 2.0
OCR_CLAHE_GRID_SIZE = 8
# motorcycle OCR drops text boxes smaller than this fraction of the largest
OCR_MOTOR_MIN_BOX_AREA_RATIO = 0.05

# ============================================================================
# 3. WORKER POOL / SELF-HEALING
# ============================================================================
DEFAULT_WORKER_COUNT = 3              # workers on a fresh boot (then the saved count)
WORKER_LOAD_TIMEOUT_SEC = 180.0
WORKER_SHUTDOWN_TIMEOUT_SEC = 30.0
WATCHDOG_INTERVAL_SEC = 5.0           # respawn crashed workers
OCR_IDLE_POLL_INTERVAL_SEC = 0.2
OCR_TASK_POP_RETRY_BACKOFF_SEC = 1.0
HEARTBEAT_INTERVAL_SEC = 10.0
HEARTBEAT_TTL_SEC = 30

# ============================================================================
# 4. LOGS
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "text"                   # text | json
# per-task lines ([TASK-START], [VOTE], [VALIDATE], [TASK-DONE])
LOG_TASK_EVENTS = True
# one-line structured decision trace per task (vote, candidates, validation)
LOG_DECISION_TRACE = True
# 📊 [OCR-STATS] per worker every N seconds: tasks, queue latency, processing ms, valid %
OCR_STATS_LOG_INTERVAL_SEC = 30.0
API_PORT = 8011

# ============================================================================
# 5. VISUAL DEBUG (local bind mount ./debug_ocr — never MinIO)
# ============================================================================
OCR_DEBUG_ROOT_DIR = "/debug"
# a labelled-grid JPEG per finalized task: crops, candidates, decision
OCR_SAVE_DECISION_DEBUG = False
OCR_DECISION_DEBUG_DIR = os.path.join(OCR_DEBUG_ROOT_DIR, "decisions")
OCR_DECISION_DEBUG_MAX_FILES = 200

logging_setup.configure(LOG_LEVEL, LOG_FORMAT)
logging_setup.fail_on_placeholders()
