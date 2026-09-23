# =====================================================================
# plate-detector image
# ---------------------------------------------------------------------
# Built FROM the operator's own prepared base image (CUDA/torch/
# ultralytics/OpenCV already installed — see .env AI_BASE_IMAGE). This
# file only adds: the shared `platecore` library, this service's extra
# python deps (redis, scipy, lap), and this service's own source.
#
#   docker build -f detector/Dockerfile --build-arg BASE_IMAGE=base_image_gpu:latest -t plate/detector:dev .
#
# (run from the repo root — the build context must include both
#  detector/ and common/, hence the COPY sources below; see compose.yaml
#  which sets `context: .`)
# =====================================================================
ARG BASE_IMAGE=base_image_gpu:latest
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    YOLO_CONFIG_DIR=/data/.ultralytics

# ---- service-specific python deps (torch/ultralytics/opencv already
# in the base image; this only adds what the Redis-based split and the
# BYTETracker need) --------------------------------------------------
COPY detector/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt || \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---- shared internal library, copied into both service images -------
COPY common/platecore /app/platecore

# ---- this service's own source ---------------------------------------
COPY detector/src /app

# ---- runtime directories -----------------------------------------------
# /models        bind-mounted, read-only: best.pt (see README)
# /debug         local bind volume: annotated debug video, when enabled
RUN mkdir -p /data /models /debug

EXPOSE 8010

HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3).status==200 else 1)"

CMD ["python", "main.py"]
