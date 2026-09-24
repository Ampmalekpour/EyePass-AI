"""
main.py
--------------------------------------------------------------------
Entrypoint only. "How to run this" stays separate from "what this app
is" (api.py). Unchanged from the pre-existing standalone build.

The OPENCV_FFMPEG_CAPTURE_OPTIONS block below must run BEFORE anything
imports cv2 — directly or transitively. FFmpeg's RTSP demuxer reads
that variable when the capture is opened, and a value set after the
OpenCV FFmpeg DLL has loaded is not reliably visible to it. Setting it
here, in the parent, also means every engine process spawned later
inherits it as part of its real environment.

In Docker the variable is already set in the image (see Dockerfile);
setdefault leaves that value alone.
--------------------------------------------------------------------
"""

import os

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp"        # never UDP: RTP cannot cross a Docker NAT boundary
    "|rtsp_flags;prefer_tcp"
    "|timeout;5000000"          # microseconds — socket read timeout
    "|stimeout;5000000"         # same, for older ffmpeg builds
    "|max_delay;500000"
)

import logging                      # noqa: E402
import multiprocessing as mp        # noqa: E402

import uvicorn                      # noqa: E402

from api import create_app          # noqa: E402
from config import API_HOST, API_PORT, LOG_LEVEL  # noqa: E402


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_app()

    # workers=1 is deliberate and load-bearing. Cube flushing is
    # read-modify-write against MinIO, so two workers owning the same
    # camera+date would silently overwrite each other. Scale by sharding
    # cameras across separate deployments, never by adding workers.
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        workers=1,
        log_level=LOG_LEVEL.lower(),
        # Give the lifespan shutdown hook room to flush cubes to MinIO.
        timeout_graceful_shutdown=45,
    )
