# =====================================================================
# face-recognizer image
# ---------------------------------------------------------------------
# Built FROM the operator's own prepared base image (torch/OpenCV
# already installed — see .env AI_BASE_IMAGE). Adds: the shared
# `facecore` library, redis/boto3/onnxruntime, and this service's own
# source. The operator supplies (bind-mounted, see README):
#   pretrained/        adaface_ir50_ms1mv2.ckpt (pytorch fallback),
#                       warmup.jpg
#   face_alignment/     the alignment package (align.get_aligned_face,
#                       mtcnn_pytorch's warp_and_crop_face)
#   /models/adaface_ir50_cpu.onnx   (primary ONNX backend)
#
#   docker build -f recognizer/Dockerfile --build-arg BASE_IMAGE=base_image:latest -t face/recognizer:dev .
#
# (run from the repo root, same reasoning as detector/Dockerfile)
# =====================================================================
ARG BASE_IMAGE=base_image:latest
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY recognizer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt || \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---- shared internal library, copied into both service images -------
COPY common/facecore /app/facecore

# ---- this service's own source ---------------------------------------
COPY recognizer/src /app

# ---- runtime directories -----------------------------------------------
# /models             bind-mounted, read-only: adaface_ir50_cpu.onnx
# /app/pretrained     bind-mounted, read-only: .ckpt fallback + warmup.jpg
# /app/face_alignment bind-mounted, read-only: the alignment package
# /data               local bind volume: debug output (landmarked
#                     crops, aligned-face dumps) — never MinIO.
RUN mkdir -p /data /models /app/pretrained

EXPOSE 8011

HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=180s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=3).status==200 else 1)"

CMD ["python", "main.py"]
