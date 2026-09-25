"""
keys.py
--------------------------------------------------------------------
Every Redis key/channel name the plate module touches, in one place.

Two families:

  * BACKEND CONTRACT — names dictated by the already-deployed backend
    (Django) and eyepass-camera-stream. Do not rename these; other
    processes outside this module depend on the exact spelling:

      - `{module}:cameras:config`             (existing)
      - `{module}:cameras:details`            (existing)
      - `{module}:camera:events`              (existing — SINGULAR
        "camera", exactly what the plate module's own
        eyepass-camera-stream deployment publishes to; do not "fix"
        this to the plural spelling some other module's copy of
        camera_service may use — that would silently stop working
        against this module's actual camera-stream container)
      - `{module}:cmd:ai:request` / `{module}:cmd:ai:response:{id}`
                                               (existing)
      - `{module}:cameras:{camera_id}:ai_status`  (existing — Django
        reads this per-camera hash to show real AI processing status,
        distinct from the "activated" desired state)
      - `{module}:vehicle:results`            (existing — Django reads
        this list for every mid-track and final plate/vehicle result;
        this is the plate module's name for what the face module
        calls `ai:results`)

  * INTERNAL — everything under `{module}:internal:...`. These are
    NEW in this rewrite: durable desired-state, self-healing
    checkpoints, and the detector<->OCR task/result queues that
    replace the old in-process multiprocessing.Queue pipes. Nothing
    outside the two plate services reads or writes these.
--------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    module: str = "plate"

    # ================================================================
    # BACKEND CONTRACT (fixed spelling — see module README)
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

        SINGULAR "camera", matching this module's actual
        eyepass-camera-stream deployment — see this file's docstring.
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
        REPORTED status ("what is actually happening right now"),
        the desired-state ledger is "what should be running".
        """
        return f"{self.module}:cameras:{camera_id}:ai_status"

    @property
    def vehicle_results(self) -> str:
        """LIST. publisher: detector (mid-track) + OCR service (via
        detector, see codec.py). subscriber: backend (Django, saves to
        DB). Plate's existing name for the face module's `ai:results`."""
        return f"{self.module}:vehicle:results"

    # ================================================================
    # INTERNAL — durable desired state / self-healing checkpoints (NEW)
    # ================================================================
    @property
    def active_cameras(self) -> str:
        """HASH camera_id -> json(roi, triggers, request_id, activated_at).

        The durable "what should be running" ledger. A redeploy replays
        this on startup instead of waiting for the backend to resend
        every activation.
        """
        return f"{self.module}:internal:active_cameras"

    @property
    def detector_state(self) -> str:
        """HASH {phase, engine_count, updated_at} for the detector's
        self-healing checkpoint (idle vs processing, engine count)."""
        return f"{self.module}:internal:detector:state"

    @property
    def ocr_state(self) -> str:
        """HASH {phase, worker_count, updated_at} for the OCR service's
        self-healing checkpoint."""
        return f"{self.module}:internal:ocr:state"

    @property
    def detector_heartbeat(self) -> str:
        """STRING with TTL, refreshed periodically. Liveness only."""
        return f"{self.module}:internal:detector:heartbeat"

    @property
    def ocr_heartbeat(self) -> str:
        return f"{self.module}:internal:ocr:heartbeat"

    # ---- detector -> OCR task queue --------------------------------
    @property
    def ocr_tasks(self) -> str:
        """LIST. Shared work queue. Every detector engine LPUSHes crop
        tasks here; every OCR worker BRPOPs from here. Capacity of
        detection and OCR now scale independently — that is the point
        of splitting them into two services."""
        return f"{self.module}:internal:ocr:tasks"

    def ocr_results(self, engine_id) -> str:
        """LIST, one per detector engine. An OCR worker learns the
        originating engine_id from the task envelope and LPUSHes the
        result back onto that engine's own list, so results route back
        to the same Engine instance that owns the track — mirroring the
        old per-engine ocr_output_queue."""
        return f"{self.module}:internal:ocr:results:{engine_id}"

    @property
    def ocr_tasks_pending_gauge(self) -> str:
        """STRING counter, informational only (for redis insight/commander)."""
        return f"{self.module}:internal:ocr:tasks:pending"

    @property
    def demand_events(self) -> str:
        """PUB/SUB channel. publisher: detector. subscriber: OCR service.

        JSON {"processing": bool} — whether at least one camera is
        currently marked active. This is the ONLY thing the OCR
        service needs to know from the detector; it stays
        camera-agnostic otherwise. Published whenever active_cameras
        transitions between empty and non-empty (backend_bridge's
        handle_activated / handle_deactivated). The OCR service also
        reads active_cameras directly once at boot (see
        ocr_service/src/main.py) to cover the case where it starts up
        fresh while cameras are already active — the pub/sub channel
        alone would miss that, since nothing re-publishes on a steady
        state.
        """
        return f"{self.module}:internal:demand:events"


    # ================================================================
    # INTERNAL — control hub (see control-hub/src/protocol.py)
    # ================================================================
    @property
    def hub_events(self) -> str:
        """STREAM. publisher: detector engines (track_started, trigger,
        submitted, track_update, track_ended, engine_started).
        consumer: control hub (consumer group)."""
        return f"{self.module}:internal:hub:events"

    @property
    def hub_results(self) -> str:
        """STREAM. publisher: recognizer / OCR workers, one entry per
        finished task, keyed by the task's global track uid.
        consumer: control hub (consumer group)."""
        return f"{self.module}:internal:hub:results"

    def hub_ctl(self, engine_id) -> str:
        """LIST, one per detector engine. publisher: control hub
        (result acks, satisfied flag, periodic re-query requests).
        consumer: that engine (BRPOP)."""
        return f"{self.module}:internal:hub:ctl:{engine_id}"


DEFAULT_KEYS = RedisKeys()
