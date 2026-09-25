"""
hub.py
--------------------------------------------------------------------
Detector-side half of the control-hub contract (the hub side is
control-hub/src/protocol.py + core.py). This file is byte-for-byte
identical in facecore/ and platecore/ — keep it that way.

What moved OUT of the detector engines into the control hub:
    storing recognition/OCR results, merging them into one identity /
    plate, deciding when a trigger may be published, what to publish,
    when a track is final, waiting for late results.

What stays IN the detector (because it needs frames and crops):
    tracking, trigger geometry, best-crop ranking, liveness, and the
    ONE local decision this file encodes: "may I send a crop for this
    track right now?"

`TrackRecState` answers that question per track:

    request(stage)   something wants a recognition pass: a trigger just
                     fired (local, immediate), or the hub asked for a
                     periodic re-query (ctl "request")
    next_stage(now)  the stage to submit now, or None when
                       - the hub said the track is satisfied, or
                       - nothing is requested, or
                       - a task is still in flight (until the hub acks
                         its result, or SUBMIT_TIMEOUT_SEC passes)
    mark_submitted   one submission answers every pending request
    apply_ctl        hub messages: result ack / satisfied flag / request

The engine additionally refuses to re-send the SAME best crop
(`last_sent_key`): a request stays pending until a better crop exists,
and the hub's trigger_max_wait_sec guarantees the trigger is published
anyway.

`HubClient` is the transport: emit events onto `{m}:internal:hub:events`,
push tasks onto the worker queue, and a background thread that
drains this engine's ctl list into a local queue the engine loop reads
without blocking.
--------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .codec import encode_task
from .logging_setup import setup_logger

logger = setup_logger("hub.client")

# event kinds (see control-hub/src/protocol.py)
K_ENGINE_STARTED = "engine_started"
K_TRACK_STARTED = "track_started"
K_TRACK_UPDATE = "track_update"
K_SUBMITTED = "submitted"
K_TRIGGER = "trigger"
K_TRACK_ENDED = "track_ended"


def new_track_uid(camera_id, engine_id) -> str:
    """Globally unique track id: survives engine restarts and camera
    rebalancing, which both reset BYTETrack's per-engine int ids."""
    return f"{camera_id}-{engine_id}-{uuid.uuid4().hex[:10]}"


def new_task_id(uid: str, stage: str) -> str:
    # trailing time_ns kept so the plate OCR worker's queue-latency
    # measurement (task_id.rsplit(':', 1)[-1]) keeps working
    return f"{uid}:{stage}:{time.time_ns()}"


class TrackRecState:
    __slots__ = ("uid", "satisfied", "requested", "in_flight_task", "in_flight_stage",
                 "in_flight_since", "last_sent_key", "display", "last_update_emit",
                 "submissions", "last_latency_ms")

    def __init__(self, uid: str):
        self.uid = uid
        self.satisfied = False
        self.requested: Dict[str, float] = {}
        self.in_flight_task: Optional[str] = None
        self.in_flight_stage: Optional[str] = None
        self.in_flight_since = 0.0
        self.last_sent_key: Any = None
        self.display: Dict[str, Any] = {}
        self.last_update_emit = 0.0
        self.submissions = 0
        self.last_latency_ms: Optional[float] = None

    # ---- requests ---------------------------------------------------------
    def request(self, stage: str, now: float):
        if not self.satisfied:
            self.requested.setdefault(stage, now)

    def in_flight(self, now: float, submit_timeout: float) -> bool:
        return self.in_flight_task is not None and (now - self.in_flight_since) < submit_timeout

    def next_stage(self, now: float, submit_timeout: float, priority: Dict[str, int]) -> Optional[str]:
        if self.satisfied or not self.requested or self.in_flight(now, submit_timeout):
            return None
        return max(self.requested, key=lambda s: (priority.get(s, 0), -self.requested[s]))

    def mark_submitted(self, task_id: str, stage: str, key: Any, now: float):
        self.in_flight_task = task_id
        self.in_flight_stage = stage
        self.in_flight_since = now
        self.last_sent_key = key
        self.requested.clear()
        self.submissions += 1

    # ---- hub -> detector ----------------------------------------------------
    def apply_ctl(self, msg: Dict[str, Any], now: float):
        action = msg.get("action")
        if msg.get("satisfied") is not None:
            self.satisfied = bool(msg["satisfied"])
            if self.satisfied:
                self.requested.clear()
        if isinstance(msg.get("display"), dict):
            self.display = msg["display"]
        if action == "result" and msg.get("task_id") and msg.get("task_id") == self.in_flight_task:
            self.last_latency_ms = (now - self.in_flight_since) * 1000.0
            self.in_flight_task = None
            self.in_flight_stage = None
        elif action == "request" and msg.get("stage"):
            self.request(msg["stage"], now)

    def pending_label(self, now: float, submit_timeout: float) -> Optional[str]:
        return self.in_flight_stage if self.in_flight(now, submit_timeout) else None


class HubClient:
    """One per detector Engine (built inside the engine subprocess)."""

    def __init__(self, bus, engine_id):
        self.bus = bus
        self.engine_id = engine_id
        self.boot_id = uuid.uuid4().hex[:12]
        self._ctl: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True, name=f"HubCtl-{engine_id}")
        self._thread.start()
        self.emit(K_ENGINE_STARTED, {})

    # ---- detector -> hub ----------------------------------------------------
    def emit(self, kind: str, data: Dict[str, Any]):
        data = dict(data)
        data.setdefault("engine_id", self.engine_id)
        data.setdefault("boot_id", self.boot_id)
        data.setdefault("ts", time.time())
        try:
            self.bus.hub_emit(kind, data)
        except Exception as e:
            # The engine keeps detecting/tracking regardless; the hub
            # recovers a missed track_started/update from later events
            # and ends tracks it stops hearing about.
            logger.warning("hub emit %s failed (uid=%s): %s", kind, data.get("uid"), e)

    def submit(self, task: Dict[str, Any]):
        task.setdefault("engine_id", self.engine_id)
        self.bus.push_task(encode_task(task))

    # ---- hub -> detector ----------------------------------------------------
    def _listen(self):
        while not self._stop.is_set():
            try:
                msg = self.bus.pop_hub_ctl(self.engine_id, timeout=1)
            except Exception:
                time.sleep(0.5)
                continue
            if msg:
                self._ctl.put(msg)

    def drain(self) -> List[Dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self._ctl.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self._stop.set()
