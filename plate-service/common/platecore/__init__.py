"""
platecore
--------------------------------------------------------------------
Small shared library used by BOTH the plate detector service and the
plate OCR service. Copied into each Docker image at build time (see
detector/Dockerfile and ocr_service/Dockerfile) rather than published
to a package index, so each image stays self-contained.

This is the plate module's equivalent of the face module's `facecore`
— same shapes, same self-healing pattern — with two deliberate
differences to stay compatible with the plate module's ALREADY
DEPLOYED camera_service and Django backend (see keys.py):

  * the camera physical-connect/disconnect channel is
    `{module}:camera:events` (singular "camera"), matching the plate
    module's actual eyepass-camera-stream deployment, not the
    `cameras:events` (plural) spelling the face module's newer
    camera_service copy uses.
  * results are pushed to `{module}:vehicle:results`, the name
    Django already consumes, instead of a generic `ai:results`.

Nothing in here imports torch / ultralytics / paddleocr / cv2, so it
is cheap to import from tooling, tests, or redis_tools.py.
--------------------------------------------------------------------
"""

__all__ = [
    "keys",
    "bus",
    "active_state",
    "lifecycle",
    "codec",
    "hub",
    "minio_store",
    "logging_setup",
]
