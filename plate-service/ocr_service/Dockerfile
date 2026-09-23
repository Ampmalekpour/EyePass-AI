# =====================================================================
# plate-ocr image
# ---------------------------------------------------------------------
# PaddleOCR runs CPU-only here by default (OCR_USE_GPU=false, see
# ocr_service/requirements.txt) and, on its own, needs neither torch,
# ultralytics nor CUDA — so it doesn't strictly need the same heavy
# base image plate_detector builds from.
#
# NOTE — as shipped, compose.yaml's plate_ocr build args still default
# BASE_IMAGE to the SAME value as plate_detector's AI_BASE_IMAGE
# (base_image_gpu:latest), because that is genuinely what this image
# was built and tested against here — switching it sight-unseen risked
# silently breaking the build. What changed: compose.yaml now reads a
# SEPARATE var, OCR_BASE_IMAGE (falling back to base_image_gpu:latest
# — i.e. today's behavior, unchanged), instead of reusing
# ${AI_BASE_IMAGE} directly. Set OCR_BASE_IMAGE in .env to a lighter
# CPU-only base once you've verified `pip install -r
# ocr_service/requirements.txt` succeeds cleanly on it — see
# DEBUGGING.md for the full note on this discrepancy.
#
#   docker build -f ocr_service/Dockerfile -t plate/ocr:dev .
#
# (run from the repo root — the build context must include both
#  ocr_service/ and common/, hence the COPY sources below; see
#  compose.yaml which sets `context: .`)
# =====================================================================
ARG BASE_IMAGE=base_image_gpu:latest
FROM ${BASE_IMAGE}
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app


COPY ocr_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt || \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---- shared internal library, copied into both service images -------
COPY common/platecore /app/platecore

# ---- this service's own source ---------------------------------------
COPY ocr_service/src /app

# ---- runtime directories -----------------------------------------------
# /models/ocr   bind-mounted, read-only: the PaddleOCR model folders
#               (see README "What you need to provide")
# /debug        bind-mounted: decision-trace montages (OCR_SAVE_DECISION_DEBUG)
#               — local disk only, never MinIO (see README storage split)
RUN mkdir -p /models/ocr /debug

EXPOSE 8011

HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=180s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=3).status==200 else 1)"

CMD ["python", "main.py"]
