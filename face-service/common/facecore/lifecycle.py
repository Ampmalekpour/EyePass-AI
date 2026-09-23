"""
lifecycle.py
--------------------------------------------------------------------
The idle/process state machine shared by the detector and the
recognizer, and the self-healing checkpoint that survives a restart.

Four operations, same shape on both services:

    start_idle()     load + warm up whatever this service's "models"
                      are (YOLO engines for the detector, AdaFace
                      workers for the recognizer) but do no real work
                      yet. Persists phase=idle.

    stop_idle()       undo start_idle(): release the loaded/warmed
                      resources. Only valid from IDLE or PROCESSING
                      (it implies stop_process() first). Persists
                      phase=stopped.

    start_process()   begin doing the actual job with whatever is
                      already loaded and warm — for the detector, that
                      means reconciling and attaching cameras; for the
                      recognizer, that means workers start pulling
                      tasks off the queue. Persists phase=processing
                      together with however many engines/workers are
                      running, because that count is itself part of
                      what must survive a restart.

    stop_process()    stop doing the job but keep the models loaded —
                      cameras detached / workers stop pulling new
                      tasks, everything else stays warm. Persists
                      phase=idle.

Self-healing: self_heal() reads the last persisted phase + count and
replays start_idle() (+ start_process() if the last phase was
"processing") using that persisted count as the floor for how many
engines/workers to bring up. That is the whole "read its last state
from Redis and get itself back to that state" requirement — it works
whether the process died and was rescheduled, or the whole container
was redeployed.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional
from .logging_setup import setup_logger
from .bus import RedisBus

logger = setup_logger("facecore.lifecycle")

PHASE_STOPPED = "stopped"
PHASE_IDLE = "idle"
PHASE_PROCESSING = "processing"


class ServiceLifecycle:
    """
    Callbacks:
        on_start_idle(count: int)  -> load/warm `count` units (engines
                                       or workers); return the number
                                       actually brought up.
        on_stop_idle()             -> release everything loaded.
        on_start_process()         -> begin real processing with what
                                       is already loaded.
        on_stop_process()          -> stop real processing, stay warm.
        get_unit_count()           -> current number of loaded
                                       engines/workers, for persistence.

    `default_count` is used the very first time a service boots with no
    persisted checkpoint at all (fresh Redis / fresh deployment).
    """

    def __init__(
        self,
        bus: RedisBus,
        state_key: str,
        on_start_idle: Callable[[int], int],
        on_stop_idle: Callable[[], None],
        on_start_process: Callable[[], None],
        on_stop_process: Callable[[], None],
        get_unit_count: Callable[[], int],
        default_count: int = 1,
        unit_name: str = "unit",
    ):
        self.bus = bus
        self.state_key = state_key
        self._on_start_idle = on_start_idle
        self._on_stop_idle = on_stop_idle
        self._on_start_process = on_start_process
        self._on_stop_process = on_stop_process
        self._get_unit_count = get_unit_count
        self.default_count = int(default_count)
        self.unit_name = unit_name

        self._lock = threading.RLock()
        self.phase = PHASE_STOPPED

    # ================================================================
    # Persistence
    # ================================================================
    def _persist(self, phase: str, count: Optional[int] = None):
        payload = {
            "phase": phase,
            f"{self.unit_name}_count": int(count if count is not None else self._get_unit_count()),
            "updated_at": time.time(),
        }
        try:
            self.bus.rt.hset(self.state_key, mapping={k: str(v) for k, v in payload.items()})
        except Exception:
            logger.exception("failed to persist lifecycle state at %s", self.state_key)

    def read_checkpoint(self) -> dict:
        try:
            raw = self.bus.rt.hgetall(self.state_key)
        except Exception:
            logger.exception("failed to read lifecycle checkpoint at %s", self.state_key)
            return {"phase": PHASE_STOPPED, f"{self.unit_name}_count": self.default_count}
        if not raw:
            return {"phase": PHASE_STOPPED, f"{self.unit_name}_count": self.default_count}
        try:
            count = int(raw.get(f"{self.unit_name}_count", self.default_count))
        except (TypeError, ValueError):
            count = self.default_count
        return {"phase": raw.get("phase", PHASE_STOPPED), f"{self.unit_name}_count": max(1, count)}

    # ================================================================
    # Transitions
    # ================================================================
    def start_idle(self, count: Optional[int] = None) -> int:
        with self._lock:
            if self.phase in (PHASE_IDLE, PHASE_PROCESSING):
                logger.info("start_idle() no-op, already %s", self.phase)
                return self._get_unit_count()
            n = int(count) if count is not None else self.default_count
            logger.info("start_idle(%s=%d): loading and warming up...", self.unit_name, n)
            actual = self._on_start_idle(n)
            self.phase = PHASE_IDLE
            self._persist(PHASE_IDLE, actual)
            logger.info("start_idle done: %d %s(s) loaded and warm", actual, self.unit_name)
            return actual

    def stop_idle(self):
        with self._lock:
            if self.phase == PHASE_PROCESSING:
                self.stop_process()
            if self.phase == PHASE_STOPPED:
                logger.info("stop_idle() no-op, already stopped")
                return
            logger.info("stop_idle(): releasing loaded %s(s)...", self.unit_name)
            self._on_stop_idle()
            self.phase = PHASE_STOPPED
            self._persist(PHASE_STOPPED, 0)
            logger.info("stop_idle done")

    def start_process(self):
        with self._lock:
            if self.phase == PHASE_STOPPED:
                self.start_idle()
            if self.phase == PHASE_PROCESSING:
                logger.info("start_process() no-op, already processing")
                return
            logger.info("start_process(): switching %d %s(s) to processing", self._get_unit_count(), self.unit_name)
            self._on_start_process()
            self.phase = PHASE_PROCESSING
            self._persist(PHASE_PROCESSING)
            logger.info("start_process done")

    def stop_process(self):
        with self._lock:
            if self.phase != PHASE_PROCESSING:
                logger.info("stop_process() no-op, phase is %s", self.phase)
                return
            logger.info("stop_process(): returning to idle")
            self._on_stop_process()
            self.phase = PHASE_IDLE
            self._persist(PHASE_IDLE)
            logger.info("stop_process done")

    def checkpoint_now(self):
        """Re-persist the current phase with the CURRENT unit count.

        Call this after anything that changes the engine/worker count
        outside of an explicit transition — an engine spun up because
        camera count grew past MAX_CAMERAS_PER_ENGINE, or a rebalance
        consolidated cameras into fewer engines. Without this, the
        self-healing checkpoint would only reflect the count at the
        last start_idle()/start_process() call and go stale.
        """
        with self._lock:
            if self.phase == PHASE_STOPPED:
                return
            self._persist(self.phase)

    # ================================================================
    # Self-healing
    # ================================================================
    def self_heal(self):
        checkpoint = self.read_checkpoint()
        phase = checkpoint["phase"]
        count = checkpoint[f"{self.unit_name}_count"]
        logger.info("self_heal(): last known state phase=%s %s_count=%d", phase, self.unit_name, count)

        if phase == PHASE_STOPPED:
            # Nothing was running before the restart; still bring up an
            # idle floor so the service is warm and ready the moment a
            # command arrives, rather than paying cold-start latency on
            # the first request after every redeploy.
            self.start_idle(count)
            return

        self.start_idle(count)
        if phase == PHASE_PROCESSING:
            self.start_process()
