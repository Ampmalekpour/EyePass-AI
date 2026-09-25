# test-video/Dockerfile
# --------------------------------------------------------------------
# Tiny ffmpeg image that loops a local video file into mediamtx over
# RTSP, so the rest of the stack sees it exactly like a live camera.
#
# Built on python:3.14-slim-bookworm since that's already cached
# locally (camera_service uses it too) - avoids pulling a fresh base
# image just for this one throwaway container.
# --------------------------------------------------------------------
FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY publish_loop.sh /publish_loop.sh
RUN chmod +x /publish_loop.sh

ENTRYPOINT ["/publish_loop.sh"]