"""
facecore
--------------------------------------------------------------------
Small shared library used by BOTH the detector service and the
recognizer service. It is copied into each Docker image at build time
(see detector/Dockerfile and recognizer/Dockerfile) rather than
published to a package index — that keeps each image self-contained
and avoids a private-registry dependency for a two-service module.

Nothing in here imports torch / ultralytics / onnxruntime / cv2, so it
is cheap to import from tooling, tests, or redis_tools.py.
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
