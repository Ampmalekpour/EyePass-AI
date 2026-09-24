"""
keys.py
--------------------------------------------------------------------
Every Redis key/channel name the fire/smoke module touches, in one
place — same pattern as plate's platecore/keys.py and face's
facecore/keys.py.

Two families:

  * BACKEND CONTRACT — names the backend (Django) and camera_stream
    are expected to use. Do not rename these; other processes outside
    this module depend on the exact spelling:

      - `{module}:cameras:config`             camera definitions
      - `{module}:cameras:details`            camera_stream's reported
                                               connect/disconnect state
      - `{module}:camera:events`              SINGULAR "camera" — same
        convention plate/face's own eyepass-camera-stream deployments
        use; do not "fix" this to the plural spelling some other
        camera_service copy may use.
      - `{module}:cmd:ai:request` / `{module}:cmd:ai:response:{id}`
        camera activate/deactivate command channel
      - `{module}:cameras:{camera_id}:ai_status`  per-camera reported
        AI processing status, distinct from the "activated" desired
        state below
      - `{module}:detections:results`         THREAT / RESOLUTION
        events. Kept exactly as the pre-existing partial port of this
        module already named it (MODULE_KEY default "fire" ->
        "fire:detections:results"), so an already-wired backend
        consumer keeps working unchanged.

  * INTERNAL — everything under `{module}:internal:...`. Durable
    desired-state and the self-healing checkpoint. Nothing outside
    this module reads or writes these.
--------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    module: str = "fire"

    # ================================================================
    # BACKEND CONTRACT (fixed spelling)
    # ================================================================
    @property
    def cameras_config(self) -> str:
        """HASH. publisher: backend. listener: camera_stream, detector."""
        return f"{self.module}:cameras:config"

    @property
    def cameras_details(self) -> str:
        """HASH. publisher: camera_stream. listener: backend, detector."""
        return f"{self.module}:cameras:details"

    @property
    def cameras_events(self) -> str:
        """PUB/SUB channel. publisher: camera_stream. listener: detector.

        SINGULAR "camera" — see this file's docstring.
        """
        return f"{self.module}:camera:events"

    @property
    def cmd_request(self) -> str:
        """LIST. publisher: backend. subscriber: detector (BRPOP)."""
        return f"{self.module}:cmd:ai:request"

    def cmd_response(self, request_id: str) -> str:
        """LIST (single entry, 60s TTL). publisher: detector. subscriber: backend."""
        return f"{self.module}:cmd:ai:response:{request_id}"

    def ai_status(self, camera_id: str) -> str:
        """HASH, single field `camera_id` -> json {target, current, error,
        updated_at}. publisher: detector. subscriber: backend (Django).

        Distinct from the desired-state ledger below: this is the
        REPORTED status ("what is actually happening right now"), the
        desired-state ledger is "what should be running".
        """
        return f"{self.module}:cameras:{camera_id}:ai_status"

    @property
    def detections_results(self) -> str:
        """LIST. publisher: detector (per-region THREAT / RESOLUTION
        event). subscriber: backend (Django, BRPOP/LPOP, saves to DB).
        Each payload carries event_type ("THREAT"/"RESOLUTION"),
        status (FIRE/SMOKE/BOTH/CLEAR), camera_id, region_id and the
        MinIO object key of the uploaded frame."""
        return f"{self.module}:detections:results"

    # ================================================================
    # INTERNAL — durable desired state / self-healing checkpoints
    # ================================================================
    @property
    def active_cameras(self) -> str:
        """HASH camera_id -> json(roi, request_id, activated_at).

        The durable "what should be running" ledger. A redeploy
        replays this on startup instead of waiting for the backend to
        resend every activation.
        """
        return f"{self.module}:internal:active_cameras"

    @property
    def detector_state(self) -> str:
        """HASH {phase, engine_count, updated_at} for the detector's
        self-healing checkpoint (idle vs processing, engine count)."""
        return f"{self.module}:internal:detector:state"

    @property
    def detector_heartbeat(self) -> str:
        """STRING with TTL, refreshed periodically. Liveness only."""
        return f"{self.module}:internal:detector:heartbeat"


DEFAULT_KEYS = RedisKeys()
