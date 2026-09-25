# =====================================================================
# control-hub image
# ---------------------------------------------------------------------
# CPU-only, tiny: the hub never touches an image or a model — it only
# reads track events + recognition/OCR results from Redis streams,
# decides, and writes to Redis. No base_image, no GPU.
#
#   docker build -t eyeplate/control-hub:dev control-hub/
# =====================================================================
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app

EXPOSE 8020

HEALTHCHECK --interval=15s --timeout=5s --retries=4 --start-period=20s \
    CMD python -c "import urllib.request,sys,os; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('HUB_HEALTH_PORT','8020')+'/health', timeout=3).status==200 else 1)"

CMD ["python", "main.py"]
