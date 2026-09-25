FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Uncomment to enable CAMERA_SNAPSHOT_DEBUG_ENABLED (periodic per-camera
# JPEG snapshots via ffmpeg for visual debugging — off by default, see
# DEBUGGING.md). Left commented so the default image build stays as
# small/fast as before this feature existed.
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /debug

CMD ["python", "src/main.py"]