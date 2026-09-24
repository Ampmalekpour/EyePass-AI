"""
firecore
--------------------------------------------------------------------
Small shared library used by the fire/smoke detector service. Copied
into the Docker image at build time (see detector/Dockerfile) rather
than published to a package index, so the image stays self-contained.

This is the fire/smoke module's equivalent of the plate module's
`platecore` and the face module's `facecore` — same shapes, same
self-healing pattern, adapted for a single-service module (detection
only, no second-stage OCR/recognizer worker):

  * the camera physical-connect/disconnect channel is
    `{module}:camera:events` (singular "camera"), matching the plate
    and face modules' own eyepass-camera-stream convention.
  * results (THREAT / RESOLUTION events) are pushed to
    `{module}:detections:results` — this module's existing name for
    what plate calls `vehicle:results` and face calls `ai:results`
    (kept as-is from the pre-existing partial port so any already
    wired-up backend consumer of `fire:detections:results` keeps
    working unchanged).

Nothing in here imports torch / ultralytics / cv2, so it is cheap to
import from tooling, tests, or redis_tools.py.
--------------------------------------------------------------------
"""

__all__ = [
    "keys",
    "bus",
    "active_state",
    "lifecycle",
    "codec",
    "minio_store",
    "logging_setup",
]
