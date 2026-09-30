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

# CPU inference runtimes (DETECTION_DEVICE=cpu). PIN THESE to the exact
# versions of the venv the models were exported + benchmarked with
# (`pip freeze` there) — .env OPENVINO_VERSION / ONNXRUNTIME_VERSION.
# Export and run with the same OpenVINO: a newer runtime reads older IR,
# not the other way round. Empty = latest (only for a first try).
# INSTALL_CPU_RUNTIMES=false skips both (GPU-only image).
ARG INSTALL_CPU_RUNTIMES=true
ARG OPENVINO_VERSION=
ARG ONNXRUNTIME_VERSION=

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

# openvino + onnxruntime (CPU). onnxruntime is skipped when the base
# image already ships an onnxruntime build (e.g. onnxruntime-gpu) —
# installing both breaks the import; the detector always names
# CPUExecutionProvider explicitly either way.
RUN set -e; \
    if [ "${INSTALL_CPU_RUNTIMES}" = "true" ]; then \
      PKGS="openvino${OPENVINO_VERSION:+==${OPENVINO_VERSION}}"; \
      python -c "import onnxruntime" 2>/dev/null || PKGS="$PKGS onnxruntime${ONNXRUNTIME_VERSION:+==${ONNXRUNTIME_VERSION}}"; \
      echo "installing CPU runtimes: $PKGS"; \
      pip install --no-cache-dir --break-system-packages $PKGS || pip install --no-cache-dir $PKGS; \
      python -c "import openvino as ov, onnxruntime as ort; print('openvino', ov.__version__, '| onnxruntime', ort.__version__)"; \
    fi

# ---- shared internal library, copied into both service images -------
COPY common/platecore /app/platecore

# ---- this service's own source ---------------------------------------
COPY detector/src /app
# parity test + CPU-model export (README "CPU pipeline")
COPY detector/tools /app/tools

# ---- runtime directories -----------------------------------------------
# /models        bind-mounted, read-only: plate_v8n_480.pt, plate_v8s_640.pt
#                and their ONNX/OpenVINO variants (see README)
# /data          scratch: Ultralytics config, OpenVINO compiled-model cache
# /debug         local bind volume: annotated debug video, when enabled
RUN mkdir -p /data /models /debug

EXPOSE 8010

HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=3).status==200 else 1)"

CMD ["python", "main.py"]
