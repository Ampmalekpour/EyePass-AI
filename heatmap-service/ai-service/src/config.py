"""
config.py
--------------------------------------------------------------------
Single source of truth for every tunable in the AI service. Carried
over from the pre-existing standalone build's own config.py (same two
rules it already followed):

1. EVERYTHING comes from the environment, with a sane default. The
   same image runs in dev and prod; only the env differs.

2. NO heavy imports at module scope. `torch` is imported lazily inside
   resolve_device(), so lightweight tools (redis_tools.py, scripts)
   can `from config import ...` without dragging in CUDA, and the API
   parent process never initialises a CUDA context it does not use —
   the engine child processes are the only ones that touch the GPU.

What's new versus the pre-existing build: REDIS_MODULE-driven key
naming now goes through common/heatmapcore/keys.py instead of ad hoc
f-strings scattered across redis_client.py/ai_state.py; DEBUG_VIDEO_*
settings for the new visual-debug recorder (this module previously had
none — SAVE_OUTPUT/SAVE_AS_VIDEO wrote raw per-camera video/frames with
no grid/heatmap overlay); and DETECTOR_STATE_KEY-style self-healing
checkpoint settings shared with the other modules.
--------------------------------------------------------------------
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid int in %s, using default %s", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid float in %s, using default %s", name, default)
        return default


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Writable state directory. In Docker this is a mounted volume so the
# registry survives a container replacement; outside Docker it defaults
# to sitting next to the code.
DATA_DIR = os.getenv("DATA_DIR", BASE_DIR)

VIDEO_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", os.path.join(DATA_DIR, "runs", "detect"))
CAMERA_REGISTRY_PATH = os.getenv("CAMERA_REGISTRY_PATH", os.path.join(DATA_DIR, "camera_registry.json"))

# Legacy local-disk cube directory. Cubes live in MinIO; StorageConfig
# still carries the field for compatibility with heatmap_manager.py.
HEATMAP_DATA_DIR = os.path.join(DATA_DIR, "heatmap_data")


# =========================================================
# MEDIA SAVE SWITCHES
# Writing plain annotated video (no grid overlay) is the original
# debugging aid; DETECTOR_DEBUG_VIDEO_ENABLED below is the newer,
# richer one (grid + region density + detections). Both are off by
# default and expensive; turn on at most one at a time in dev.
# =========================================================

SAVE_OUTPUT = _env_bool("SAVE_OUTPUT", "false")
SAVE_AS_VIDEO = _env_bool("SAVE_AS_VIDEO", "true")


# =========================================================
# API SETTINGS
# =========================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8002)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# A camera is marked DEGRADED if no heartbeat arrives within this window.
HEARTBEAT_TIMEOUT_SEC = _env_float("HEARTBEAT_TIMEOUT_SEC", 15.0)


# =========================================================
# COMPUTE DEVICE
# ---------------------------------------------------------
# DETECTION_DEVICE accepts:
#   auto      pick cuda:0 if a usable GPU is present, else cpu   (default)
#   cpu       force CPU, never touch CUDA
#   cuda      first visible GPU
#   cuda:N    a specific GPU
#
# Resolution happens inside each engine CHILD process, not here, so the
# API parent never initialises CUDA. What gets passed around is this
# preference string; resolve_device() turns it into a concrete torch
# device at the point of use.
# =========================================================

DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "auto").strip().lower()
STRICT_DEVICE = _env_bool("STRICT_DEVICE", "false")

# Threads per engine process when running on CPU.
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 0)


def resolve_device(preference: Optional[str] = None) -> str:
    """Turn a device preference into a concrete torch device string.
    Call this inside the process that will actually run inference."""
    pref = (preference or DETECTION_DEVICE).strip().lower()

    if pref == "cpu":
        logger.info("Device: cpu (explicitly requested)")
        return "cpu"

    import torch  # local import on purpose — see module docstring

    cuda_available = torch.cuda.is_available()

    if pref in ("auto", ""):
        if cuda_available:
            device = "cuda:0"
            logger.info("Device: %s (auto-selected, %s)", device, torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.info("Device: cpu (auto-selected, no CUDA device visible)")
        return device

    if pref.startswith("cuda"):
        if not cuda_available:
            message = (
                f"DETECTION_DEVICE={pref} was requested but no CUDA device is visible. "
                "Check that the container has a GPU reservation and that "
                "nvidia-container-toolkit is installed on the host."
            )
            if STRICT_DEVICE:
                raise RuntimeError(message)
            logger.error("%s Falling back to CPU — throughput will be far lower.", message)
            return "cpu"

        if ":" in pref:
            try:
                index = int(pref.split(":", 1)[1])
            except ValueError:
                index = 0
            count = torch.cuda.device_count()
            if index >= count:
                message = f"DETECTION_DEVICE={pref} requested but only {count} CUDA device(s) visible."
                if STRICT_DEVICE:
                    raise RuntimeError(message)
                logger.error("%s Falling back to cuda:0.", message)
                return "cuda:0"
            return f"cuda:{index}"

        return "cuda:0"

    message = f"Unrecognised DETECTION_DEVICE={pref!r}; expected auto|cpu|cuda|cuda:N."
    if STRICT_DEVICE:
        raise RuntimeError(message)
    logger.error("%s Falling back to CPU.", message)
    return "cpu"


def apply_cpu_thread_limit() -> None:
    """Call once per engine process, after resolve_device() returns cpu."""
    if TORCH_NUM_THREADS <= 0:
        return
    import torch
    torch.set_num_threads(TORCH_NUM_THREADS)
    logger.info("Torch CPU threads limited to %d", TORCH_NUM_THREADS)


# =========================================================
# ENGINE / SCALING
# =========================================================

MAX_CAMERAS_PER_ENGINE = _env_int("MAX_CAMERAS_PER_ENGINE", 6)
CLASS_LABELS = {0: "Head"}   # maps YOLO class ids to display names

# How long to wait for an engine process to flush its cubes and exit on
# shutdown. Must stay comfortably under the container's stop_grace_period.
ENGINE_SHUTDOWN_TIMEOUT_SEC = _env_float("ENGINE_SHUTDOWN_TIMEOUT_SEC", 30.0)

DEFAULT_ENGINE_COUNT = _env_int("DEFAULT_ENGINE_COUNT", 1)


# =========================================================
# REDIS
# common/heatmapcore/keys.py builds every key from REDIS_MODULE; the
# connection settings themselves stay here since heatmapcore.bus reads
# REDIS_URL / REDIS_HOST / etc. straight from the environment.
# =========================================================

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = _env_int("REDIS_PORT", 6379)
REDIS_DB = _env_int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_MODULE = os.getenv("REDIS_MODULE", "heatmap")

REDIS_RESPONSE_TTL_SECONDS = _env_int("REDIS_RESPONSE_TTL_SECONDS", 60)
REDIS_RETRY_BACKOFF_SECONDS = _env_float("REDIS_RETRY_BACKOFF_SECONDS", 3.0)

# How long a camera may stay reported offline before we tear its engine
# down. Keep at or above the RTSP read timeout (5s) so a camera merely
# mid-reconnect is not killed prematurely.
CAMERA_OFFLINE_GRACE_SECONDS = _env_float("CAMERA_OFFLINE_GRACE_SECONDS", 5.0)

# Where frames are pulled from. Inside docker this is the relay by
# container name; running from an IDE on the host it is localhost.
MTX_RTSP_BASE_URL = os.getenv("MTX_RTSP_BASE_URL", "rtsp://localhost:8554")


# =========================================================
# MINIO
# Connection settings read directly by common/heatmapcore/minio_store.py;
# MINIO_BUCKET is repeated here only for convenience imports elsewhere.
# =========================================================

MINIO_BUCKET = os.getenv("MINIO_BUCKET", "heatmap-data")


# =========================================================
# GRID & FRAME
# The grid divides the frame into cells; each cell counts detections.
# frame_width/height are overridden per camera at runtime with the real
# stream resolution — these are only the fallback.
# =========================================================

@dataclass
class GridConfig:
    grid_width: int = _env_int("GRID_WIDTH", 128)
    grid_height: int = _env_int("GRID_HEIGHT", 72)
    frame_width: int = _env_int("FRAME_WIDTH", 1920)
    frame_height: int = _env_int("FRAME_HEIGHT", 1080)


# =========================================================
# STORAGE
# One cube per camera per day, stored in MinIO as
# "<camera_id>/<YYYY-MM-DD>.npy".
# =========================================================

@dataclass
class StorageConfig:
    storage_dir: str = HEATMAP_DATA_DIR
    # Slot size in the cube. 5 => 288 slots/day. CHANGING THIS CHANGES
    # THE SHAPE OF EVERY NEW CUBE and makes it incompatible with the
    # ones already in the bucket.
    time_resolution_minutes: int = _env_int("TIME_RESOLUTION_MINUTES", 5)
    # Flush cadence. Worst-case data-loss window on an unclean kill; a
    # clean stop always flushes.
    save_interval_seconds: int = _env_int("SAVE_INTERVAL_SECONDS", 60)
    camera_id: Optional[str] = None


# =========================================================
# DETECTION
# =========================================================

@dataclass
class DetectionConfig:
    model_path: str = os.getenv("MODEL_PATH", "/models/yolov8n.pt")
    img_size: int = _env_int("IMG_SIZE", 800)
    device: str = DETECTION_DEVICE
    conf_threshold: float = _env_float("CONF_THRESHOLD", 0.25)
    target_classes: List[int] = field(default_factory=lambda: [0])
    detection_accumulation_interval: int = _env_int("DETECTION_ACCUMULATION_INTERVAL", 10)
    point_mode: Literal["center", "bottom_center"] = os.getenv("POINT_MODE", "center")


# =========================================================
# RENDER
# =========================================================

@dataclass
class RenderConfig:
    colormap: int = _env_int("RENDER_COLORMAP", 2)          # 2 == cv2.COLORMAP_JET
    blur_kernel_size: Tuple[int, int] = (31, 31)
    normalization_mode: Literal["log", "linear"] = os.getenv("NORMALIZATION_MODE", "log")
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    alpha_background: float = _env_float("ALPHA_BACKGROUND", 0.4)
    alpha_heatmap: float = _env_float("ALPHA_HEATMAP", 0.6)


@dataclass
class SourceConfig:
    video_source: str = ""
    is_rtsp: bool = False
    rtsp_reconnect_delay_seconds: int = 2


@dataclass
class HeatmapConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# =========================================================
# VISUAL DEBUG (new — the other modules' detectors all have this;
# the pre-existing standalone build of this module only had the plain
# SAVE_OUTPUT/SAVE_AS_VIDEO switches above, with no grid/density
# overlay). Local bind-mounted volume only — never MinIO, matching the
# storage split every other module uses.
# =========================================================

DETECTOR_DEBUG_VIDEO_ENABLED = _env_bool("DETECTOR_DEBUG_VIDEO_ENABLED", "false")
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "/debug")
DEBUG_VIDEO_SEGMENT_SECONDS = _env_int("DEBUG_VIDEO_SEGMENT_SECONDS", 240)
DEBUG_VIDEO_FPS = _env_int("DEBUG_VIDEO_FPS", 12)
DEBUG_VIDEO_MAX_SEGMENTS = _env_int("DEBUG_VIDEO_MAX_SEGMENTS", 12)
DEBUG_VIDEO_SCALE = _env_float("DEBUG_VIDEO_SCALE", 1.0)
DEBUG_VIDEO_EVERY_N = _env_int("DEBUG_VIDEO_EVERY_N", 1)
DEBUG_VIDEO_CODEC = os.getenv("DEBUG_VIDEO_CODEC", "mp4v")
DEBUG_VIDEO_EXT = os.getenv("DEBUG_VIDEO_EXT", ".mp4")
DEBUG_VIDEO_JSONL = _env_bool("DEBUG_VIDEO_JSONL", "true")
