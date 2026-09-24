"""
heatmapcore
--------------------------------------------------------------------
Small shared library used by the heatmap ai_service. Copied into the
Docker image at build time (see ai-service/Dockerfile) rather than
published to a package index, so the image stays self-contained.

This is the heatmap module's equivalent of the plate module's
`platecore`, the face module's `facecore` and the fire/smoke module's
`firecore` — same shapes, same self-healing pattern, adapted for a
single-service module (detection + heatmap accumulation, no
second-stage worker):

  * the camera physical-connect/disconnect channel is
    `{module}:camera:events` (singular "camera"), matching every other
    module's own eyepass-camera-stream convention — the pre-existing
    standalone build of this module used the plural spelling with its
    own bespoke camera-service; both are replaced here with the exact
    camera-service the other modules run.
  * results (where a flushed heatmap cube landed in MinIO) are pushed
    to `{module}:ai:results`, and durable desired camera state lives
    in `{module}:ai:active` — both kept under their pre-existing names
    (matching the face module's own "ai:results" convention) so an
    already-wired backend consumer keeps working unchanged.

Nothing in here imports torch / ultralytics / cv2, so it is cheap to
import from tooling, tests, or redis_tools.py.
--------------------------------------------------------------------
"""

__all__ = [
    "keys",
    "bus",
    "active_state",
    "lifecycle",
    "minio_store",
    "logging_setup",
    "health",
]
