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

import collections
import os
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
    """One per detector Engine (built inside the engine subprocess).

    Nothing here touches Redis on the engine's frame loop: `emit()` and
    `submit()` only append to an ordered in-memory OUTBOX, and a sender
    thread delivers it (events -> hub stream, tasks -> worker queue),
    retrying with backoff while Redis is unreachable. So

      * a Redis hiccup never stalls detection, and
      * no event is lost to one: `trigger` / `track_ended` / tasks are
        delivered, in order, once Redis is back.

    `submitted` is emitted before its task is queued, so the hub always
    knows a task exists before its result can arrive. The outbox is
    bounded (HUB_OUTBOX_MAX); on overflow the oldest `track_update`
    heartbeats go first, real events only as a last resort.
    """

    def __init__(self, bus, engine_id):
        self.bus = bus
        self.engine_id = engine_id
        self.boot_id = uuid.uuid4().hex[:12]
        self._outbox: "collections.deque" = collections.deque()
        self._cv = threading.Condition()
        self._max = int(os.environ.get("HUB_OUTBOX_MAX", "20000"))
        self.dropped = 0
        self._last_err_log = 0.0
        self._ctl: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._sender = threading.Thread(target=self._send_loop, daemon=True, name=f"HubOut-{engine_id}")
        self._sender.start()
        self._thread = threading.Thread(target=self._listen, daemon=True, name=f"HubCtl-{engine_id}")
        self._thread.start()
        self.emit(K_ENGINE_STARTED, {})

    # ---- detector -> hub / workers (non-blocking) ---------------------------
    def emit(self, kind: str, data: Dict[str, Any]):
        data = dict(data)
        data.setdefault("engine_id", self.engine_id)
        data.setdefault("boot_id", self.boot_id)
        data.setdefault("ts", time.time())
        self._enqueue(("event", kind, data))

    def submit(self, task: Dict[str, Any]):
        task.setdefault("engine_id", self.engine_id)
        self._enqueue(("task", None, encode_task(task)))

    def pending(self) -> int:
        return len(self._outbox)

    def _enqueue(self, item):
        with self._cv:
            if len(self._outbox) >= self._max:
                victim = next((i for i, it in enumerate(self._outbox)
                               if it[0] == "event" and it[1] == K_TRACK_UPDATE), 0)
                del self._outbox[victim]
                self.dropped += 1
                if self.dropped % 1000 == 1:
                    logger.error("hub outbox full (%d) — Redis unreachable for long? dropped %d so far",
                                 self._max, self.dropped)
            self._outbox.append(item)
            self._cv.notify()

    def _send_loop(self):
        backoff = 0.2
        while True:
            with self._cv:
                while not self._outbox and not self._stop.is_set():
                    self._cv.wait(0.5)
                if not self._outbox:
                    return  # stopped and drained
                item = self._outbox.popleft()
            try:
                kind, name, payload = item
                if kind == "event":
                    self.bus.hub_emit(name, payload)
                else:
                    self.bus.push_task(payload)
                backoff = 0.2
            except Exception as e:
                with self._cv:
                    self._outbox.appendleft(item)  # keep order
                now = time.time()
                if now - self._last_err_log > 10:
                    self._last_err_log = now
                    logger.warning("hub outbox: Redis write failed (%s) — %d item(s) waiting, retrying",
                                   e, len(self._outbox))
                if self._stop.wait(backoff) and self._give_up():
                    return
                backoff = min(backoff * 2, 5.0)

    def _give_up(self) -> bool:
        # stopping and Redis still down: don't hang shutdown forever
        return time.time() > getattr(self, "_stop_deadline", float("inf"))

    def flush(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while self._outbox and time.time() < end:
            time.sleep(0.05)
        return not self._outbox

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

    def stop(self, flush_timeout: float = 5.0):
        """Deliver what is queued (track_ended of every live track on
        shutdown) for up to flush_timeout, then stop."""
        self.flush(flush_timeout)
        self._stop_deadline = time.time() + 0.5
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._outbox:
            logger.error("hub outbox: %d item(s) undelivered at shutdown (Redis unreachable)", len(self._outbox))
