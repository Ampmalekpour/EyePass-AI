# =====================================================================
# ai-service image
# ---------------------------------------------------------------------
# Built on top of the SAME prepared base image the plate/face/fire
# detectors build from (CUDA runtime, torch, torchvision, ultralytics,
# OpenCV already installed — see .env AI_BASE_IMAGE). We do not install
# or upgrade any of those here — that is exactly how you end up with a
# torch build that does not match the driver on the host.
#
#   docker compose build --build-arg BASE_IMAGE=other/image:tag ai_service
#
# or persistently, via AI_BASE_IMAGE in .env.
# =====================================================================

ARG BASE_IMAGE=base_image_gpu:latest
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Ultralytics writes a settings/config dir on first run; point it
    # somewhere writable and predictable instead of $HOME.
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app \
    # Force RTSP over TCP for every OpenCV capture in this container.
    # This MUST be a real process env var — setting it from Python after
    # cv2 has loaded is not reliably picked up by the bundled FFmpeg.
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|rtsp_flags;prefer_tcp|timeout;5000000|stimeout;5000000|max_delay;500000"

WORKDIR /app

# ---------------------------------------------------------------------
# System libraries OpenCV needs when the base image ships the headless
# wheel's dependencies only. Harmless if already present.
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- service-specific python deps ------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt || \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---- shared internal library, copied into the image -------------------
COPY common/heatmapcore /app/heatmapcore

# ---- this service's own source -----------------------------------------
COPY ai-service/src/ /app/

RUN mkdir -p /app/.ultralytics /app/runs /models /debug

EXPOSE 8002

# Signals: uvicorn installs a SIGTERM handler and the FastAPI lifespan
# hook flushes cubes to MinIO on the way out. `init: true` in compose
# provides the PID-1 reaper for the spawned engine processes.
HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=3).status in (200,503) else 1)"

CMD ["python", "main.py"]
