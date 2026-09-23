"""
keys.py
--------------------------------------------------------------------
Every Redis key/channel name the face module touches, in one place.

Two families:

  * BACKEND CONTRACT — names dictated by the backend team (see the
    module README). Do not rename these; the prefix is `{module}`
    (REDIS_MODULE, "face" by default) and every consumer outside this
    module (backend, camera_service) depends on the exact spelling.

  * INTERNAL — everything under `{module}:internal:...`. These belong
    to this module alone: durable desired-state, self-healing
    checkpoints, and the detector<->recognizer task/result queues that
    replaced the old in-process multiprocessing.Queue pipes. Nothing
    outside the two face services reads or writes these.

--------------------------------------------------------------------
ADD-FACE (2026-09) — new in this revision
--------------------------------------------------------------------
Enrollment gets its own backend-contract pair, `cmd:enroll:request` /
`cmd:enroll:response:{request_id}`, deliberately NOT sharing
`cmd:ai:request`. That queue is documented (README) as single-consumer
and scoped to camera activate/deactivate — owned by the detector.
Enrollment is a different consumer (the recognizer), a different
payload shape (carries image bytes, not just a JSON command), and a
different cardinality (occasional admin operations, not a per-camera
control channel). Giving it its own key pair keeps both contracts
simple instead of overloading one with a `type` field to disambiguate.

Because the payload carries raw image bytes, `cmd:enroll:*` is encoded
like the internal task/result queues (pickle, via codec.py) rather
than JSON like `cmd:ai:*` — see bus.py's enroll methods.

Internally, enrollment dispatches its actual image work (pose-check,
align+embed+commit) as ordinary tasks on the EXISTING `rec:tasks`
queue, routed back on a per-request result list via the EXISTING
`rec_results(engine_id)` helper below — using a synthetic engine_id of
the form `enroll:{request_id}` instead of a real detector engine id.
No new internal task/result keys were needed for that reason. What
*is* new: `gallery:updated`, a pub/sub fired after a successful
enrollment commit so every recognizer worker (in this process and any
sibling replica) knows to reload the gallery it downloaded from MinIO
at boot; and `gallery:lock`, a distributed lock key serializing
concurrent enrollments so two admins enrolling different people at the
same moment can't race on `brieface.db`'s range allocation.
--------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    module: str = "face"

    # ================================================================
    # BACKEND CONTRACT (fixed spelling — see README "Redis contract")
    # ================================================================
    @property
    def cameras_config(self) -> str:
        """HASH. publisher: backend. listener: camera_service."""
        return f"{self.module}:cameras:config"

    @property
    def cameras_details(self) -> str:
        """HASH. publisher: camera_service. listener: backend, ai."""
        return f"{self.module}:cameras:details"

    @property
    def cameras_events(self) -> str:
        """PUB/SUB channel. publisher: camera_service. listener: backend, ai."""
        return f"{self.module}:cameras:events"

    @property
    def cmd_request(self) -> str:
        """LIST. publisher: backend. subscriber: ai (BRPOP)."""
        return f"{self.module}:cmd:ai:request"

    def cmd_response(self, request_id: str) -> str:
        """LIST (single entry, 60s TTL). publisher: ai. subscriber: backend."""
        return f"{self.module}:cmd:ai:response:{request_id}"

    @property
    def ai_results(self) -> str:
        """LIST. publisher: ai. subscriber: backend (saves to DB)."""
        return f"{self.module}:ai:results"

    # ---- enrollment (add-face) — its own backend-contract pair -----
    @property
    def enroll_request(self) -> str:
        """LIST (BRPOP). publisher: backend. subscriber: recognizer's
        EnrollCoordinator. Pickled dict payload (see codec.py) — unlike
        cmd:ai:request this carries raw image bytes on `verify_pose`
        requests, so it is NOT JSON like the camera command channel."""
        return f"{self.module}:cmd:enroll:request"

    def enroll_response(self, request_id: str) -> str:
        """LIST (single entry, 60s TTL). publisher: recognizer.
        subscriber: backend. Pickled dict — may carry the cropped,
        pose-approved image back to the caller for preview."""
        return f"{self.module}:cmd:enroll:response:{request_id}"

    # ================================================================
    # INTERNAL — durable desired state / self-healing checkpoints
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
    def recognizer_state(self) -> str:
        """HASH {phase, worker_count, updated_at} for the recognizer's
        self-healing checkpoint."""
        return f"{self.module}:internal:recognizer:state"

    @property
    def detector_heartbeat(self) -> str:
        """STRING with TTL, refreshed periodically. Liveness only."""
        return f"{self.module}:internal:detector:heartbeat"

    @property
    def recognizer_heartbeat(self) -> str:
        return f"{self.module}:internal:recognizer:heartbeat"

    # ---- detector -> recognizer task queue -------------------------
    @property
    def rec_tasks(self) -> str:
        """LIST. Shared work queue. Every detector engine LPUSHes crop
        tasks here; every recognizer worker BRPOPs from here. Capacity
        of detection and recognition now scale independently — that is
        the point of splitting them into two services.

        Add-face reuses this exact queue for its own `enroll_pose_check`
        / `enroll_commit` tasks (see worker.py) — they are handled by
        whichever RecognitionWorker happens to be free, fairly sharing
        capacity with live recognition traffic, rather than needing a
        dedicated queue/pool of their own."""
        return f"{self.module}:internal:rec:tasks"

    def rec_results(self, engine_id) -> str:
        """LIST, one per detector engine (or, for enrollment, one per
        in-flight enroll request — see keys `enroll:{request_id}` used
        as a synthetic engine_id). A recognizer worker learns the
        originating engine_id from the task envelope and LPUSHes the
        result back onto that engine's own list, so results route back
        to whoever is actually waiting on them — mirroring the old
        per-engine fr_output_queue."""
        return f"{self.module}:internal:rec:results:{engine_id}"

    @property
    def rec_tasks_pending_gauge(self) -> str:
        """STRING counter, informational only (for redis insight/commander)."""
        return f"{self.module}:internal:rec:tasks:pending"

    @property
    def demand_events(self) -> str:
        """PUB/SUB channel. publisher: detector. subscriber: recognizer.

        JSON {"processing": bool} — whether at least one camera is
        currently marked active. This is the ONLY thing the recognizer
        needs to know from the detector; it stays camera-agnostic
        otherwise. Published whenever active_cameras transitions
        between empty and non-empty (backend_bridge.handle_activated /
        handle_deactivated). The recognizer also reads active_cameras
        directly once at boot (see recognizer/src/main.py) to cover the
        case where it starts up fresh while cameras are already active
        — the pub/sub channel alone would miss that, since nothing
        re-publishes on a steady state.
        """
        return f"{self.module}:internal:demand:events"

    # ---- add-face — gallery mutation coordination -------------------
    @property
    def gallery_updated(self) -> str:
        """PUB/SUB channel. publisher: recognizer's EnrollCoordinator,
        after a successful enrollment commit. subscriber: every
        recognizer process (this one's own pool, and any sibling
        replica's). JSON {"personnelid": ..., "committed_at": ...}.
        Each subscriber reloads its gallery from MinIO — see
        pool.py::reload_gallery()."""
        return f"{self.module}:internal:gallery:updated"

    @property
    def gallery_lock(self) -> str:
        """STRING, used as a redis-py `Lock` (SET NX PX under the
        hood). Held for the duration of one enrollment commit
        (download brieface.db -> allocate range -> insert row -> embed
        -> upload) so two concurrent enrollments can't allocate the
        same c<N>.jpg range."""
        return f"{self.module}:internal:gallery:lock"


DEFAULT_KEYS = RedisKeys()
