FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Uncomment to enable CAMERA_SNAPSHOT_DEBUG_ENABLED=true (see
# DEBUGGING.md / src/main.py::maybe_capture_snapshot) — grabs one still
# frame per camera off the MediaMTX relay for visual debugging. Kept
# out by default so this image stays minimal for deployments that
# don't need it.
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]