"""
keys.py
--------------------------------------------------------------------
Every Redis key/channel name the heatmap module touches, in one place
— same pattern as plate's platecore/keys.py, face's facecore/keys.py
and fire/smoke's firecore/keys.py.

Two families:

  * BACKEND CONTRACT — names the backend (Django) and camera_stream
    are expected to use. Do not rename these; other processes outside
    this module depend on the exact spelling:

      - `{module}:cameras:config`             camera definitions
      - `{module}:cameras:details`            camera_stream's reported
                                               connect/disconnect state
      - `{module}:camera:events`              SINGULAR "camera" — same
        convention every other module's own eyepass-camera-stream
        deployment uses. The pre-existing standalone build of this
        module used the plural `cameras:events` with its own
        `camera-service` implementation; both have been replaced here
        with the SAME camera-service the plate/face/fire modules run,
        which only ever publishes the singular channel. Do not "fix"
        this back to plural.
      - `{module}:cmd:ai:request` / `{module}:cmd:ai:response:{id}`
        camera activate/deactivate command channel. Aligned to the
        same LIST+BLPOP shape plate/face/fire use (backend LPUSHes,
        detector BRPOPs the request; detector RPUSHes the response,
        60s TTL, backend BRPOPs it) — the pre-existing standalone
        build used BLPOP on the request (LIFO, not FIFO) and a plain
        STRING+GET for the response; its own README flagged the
        response shape as a known spec mismatch. See README "Redis
        contract" for the full note to hand to the backend team.
      - `{module}:cameras:{camera_id}:ai_status`  per-camera reported
        AI processing status. NEW in this rewrite, for parity with the
        other modules' Django-facing status field — additive, nothing
        existing reads or depends on its absence.
      - `{module}:ai:results`         Kept exactly as the pre-existing
        build already named it (MODULE_KEY default "heatmap" ->
        "heatmap:ai:results"), matching the face module's own
        "ai:results" convention, so an already-wired backend consumer
        keeps working unchanged.
      - `{module}:ai:active`          Durable desired-state ledger.
        Kept under its pre-existing name (not moved under
        `internal:...` the way plate/fire do) because the module's own
        README already documents this exact key as part of the
        contract ("us -> us; internal; readable for ops").

  * INTERNAL — everything under `{module}:internal:...`. The
    self-healing phase/engine-count checkpoint. NEW in this rewrite —
    the pre-existing build only ever persisted desired camera state
    (`ai:active`), never a phase/engine-count checkpoint, so a redeploy
    always cold-started its engine pool from scratch (harmless here
    since engines carry no per-engine state — camera reconcile is what
    actually matters — but tracked for consistency with the other
    modules and so `GET /health` can report a real phase).
--------------------------------------------------------------------
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisKeys:
    module: str = "heatmap"

    # ================================================================
    # BACKEND CONTRACT
    # ================================================================
    @property
    def cameras_config(self) -> str:
        """HASH. publisher: backend. listener: camera_stream, ai_service."""
        return f"{self.module}:cameras:config"

    @property
    def cameras_details(self) -> str:
        """HASH. publisher: camera_stream. listener: backend, ai_service."""
        return f"{self.module}:cameras:details"

    @property
    def cameras_events(self) -> str:
        """PUB/SUB channel. publisher: camera_stream. listener: ai_service.

        SINGULAR "camera" — see this file's docstring.
        """
        return f"{self.module}:camera:events"

    @property
    def cmd_request(self) -> str:
        """LIST. publisher: backend (LPUSH). subscriber: ai_service (BRPOP)."""
        return f"{self.module}:cmd:ai:request"

    def cmd_response(self, request_id: str) -> str:
        """LIST (single entry, 60s TTL). publisher: ai_service (RPUSH).
        subscriber: backend (BRPOP)."""
        return f"{self.module}:cmd:ai:response:{request_id}"

    def ai_status(self, camera_id: str) -> str:
        """HASH, single field `camera_id` -> json {target, current, error,
        updated_at}. publisher: ai_service. subscriber: backend (Django).

        Distinct from the desired-state ledger below: this is the
        REPORTED status ("what is actually happening right now"), the
        desired-state ledger is "what should be running".
        """
        return f"{self.module}:cameras:{camera_id}:ai_status"

    @property
    def ai_results(self) -> str:
        """LIST. publisher: ai_service (each time a cube is flushed).
        subscriber: backend (BRPOP for FIFO, since we LPUSH). A pointer
        to where the cube landed in MinIO, not the data itself — see
        README section 6."""
        return f"{self.module}:ai:results"

    @property
    def ai_active(self) -> str:
        """HASH camera_id -> json(roi, request_id, activated_at).

        The durable "what should be running" ledger. A redeploy
        replays this on startup instead of waiting for the backend to
        resend every activation. Kept under its pre-existing name
        (not `internal:active_cameras`) — see this file's docstring.
        """
        return f"{self.module}:ai:active"

    # ================================================================
    # INTERNAL — self-healing checkpoint (new in this rewrite)
    # ================================================================
    @property
    def detector_state(self) -> str:
        """HASH {phase, engine_count, updated_at} for the ai_service's
        self-healing checkpoint (idle vs processing, engine count)."""
        return f"{self.module}:internal:detector:state"

    @property
    def detector_heartbeat(self) -> str:
        """STRING with TTL, refreshed periodically. Liveness only."""
        return f"{self.module}:internal:detector:heartbeat"


DEFAULT_KEYS = RedisKeys()
