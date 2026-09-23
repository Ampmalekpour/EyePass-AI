"""
pool.py (recognizer)
--------------------------------------------------------------------
`RecognizerPool` is the recognizer's equivalent of the detector's
EngineManager: it owns a pool of RecognitionWorker subprocesses and
implements the four lifecycle hooks ServiceLifecycle drives.

    start_idle(count)   spawn `count` RecognitionWorker processes,
                         each loading its model + gallery + warming up
                         but NOT pulling tasks yet (processing_event
                         starts clear). Waits (bounded by
                         WORKER_LOAD_TIMEOUT_SEC) for them to report
                         ready via their loaded_event.

    stop_idle()          tear every worker in the pool down completely
                         (signals stop_event, joins, terminates
                         stragglers). Fully releases GPU/CPU memory.

    start_process()      set the shared processing_event — every
                         worker in the pool starts pulling `rec:tasks`.
                         Cheap: no process restart, models stay warm.

    stop_process()        clear processing_event — workers stop pulling
                         new tasks but stay loaded and warm.

Self-healing at the worker-process level: a background watchdog thread
respawns any worker whose subprocess died unexpectedly, keeping the
pool at its expected size without waiting for the whole service to
restart. (Whole-service self-healing — recovering worker_count and
phase after a container restart — is ServiceLifecycle's job, driven by
main.py.)

--------------------------------------------------------------------
ADD-FACE (2026-09): `reload_gallery()` is the pool-level response to
the `gallery:updated` pub/sub event (see main.py) fired after a
successful enrollment commit. It is a full stop_idle()+start_idle()
cycle — every worker subprocess is torn down and respawned, which
means every worker re-downloads the gallery from MinIO from scratch
(RecognitionWorker._load_models()) and picks up the new person. This
costs a few seconds of recognition downtime across the whole pool; see
the module design notes for the cheaper, more invasive alternative
(an in-place per-worker reload event) that trades that downtime away
in exchange for more moving parts. Start here; only build that if the
downtime turns out to matter operationally.

Note this does NOT apply to the worker that actually processed the
`enroll_commit` task itself — that worker already swapped its own
`face_recognizer.db_conn` in place (see worker.py) and can recognize
the new person immediately. reload_gallery() is for every OTHER
worker (in this pool, and — once each pool independently subscribes —
any sibling recognizer replica).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import config
from worker import RecognitionWorker
from facecore.logging_setup import setup_logger

logger = setup_logger("recognizer.pool")

@dataclass
class _WorkerHandle:
    worker_id: int
    proc: RecognitionWorker
    loaded_event: "mp.Event"


class RecognizerPool:
    def __init__(self, on_topology_changed: Optional[Callable[[], None]] = None):
        self._on_topology_changed = on_topology_changed
        self._lock = threading.RLock()
        self._workers: List[_WorkerHandle] = []
        self._processing_event: Optional["mp.Event"] = None
        self._stop_event: Optional["mp.Event"] = None
        self._next_worker_id = 0

        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self._processing = False  # mirrors whether start_process() has been called

    # ================================================================
    # Lifecycle hooks (wired to ServiceLifecycle in main.py)
    # ================================================================
    def start_idle(self, count: int) -> int:
        with self._lock:
            if self._workers:
                logger.info("start_idle no-op — %d worker(s) already loaded", len(self._workers))
                return len(self._workers)

            count = max(1, int(count))
            self._processing_event = mp.Event()
            self._stop_event = mp.Event()

            for _ in range(count):
                self._spawn_worker_locked()

            self._await_loaded_locked()
            self._start_watchdog_locked()

            logger.info("start_idle done: %d worker(s) loaded and warm", len(self._workers))
            return len(self._workers)

    def stop_idle(self):
        with self._lock:
            if not self._workers:
                logger.info("stop_idle no-op — pool already empty")
                return

            self._stop_watchdog_locked()

            if self._stop_event is not None:
                self._stop_event.set()

            for handle in self._workers:
                handle.proc.join(timeout=config.WORKER_SHUTDOWN_TIMEOUT_SEC)
                if handle.proc.is_alive():
                    logger.warning("worker %s did not exit in time, terminating", handle.worker_id)
                    handle.proc.terminate()
                    handle.proc.join(timeout=5)

            self._workers = []
            self._processing_event = None
            self._stop_event = None
            self._processing = False

    def start_process(self):
        with self._lock:
            self._processing = True
            if self._processing_event is not None:
                self._processing_event.set()

    def stop_process(self):
        with self._lock:
            self._processing = False
            if self._processing_event is not None:
                self._processing_event.clear()

    def get_worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    # ================================================================
    # ADD-FACE — gallery reload (NEW)
    # ================================================================
    def reload_gallery(self):
        """Tear the pool down and bring it back up at the same size and
        the same processing/idle phase it was in — each respawned
        worker re-downloads the gallery from MinIO on the way back up,
        so this is how a worker that did NOT itself handle the
        enroll_commit task learns about the new person. See module
        docstring for the cost/complexity trade this makes."""
        with self._lock:
            count = len(self._workers) or config.DEFAULT_WORKER_COUNT
            was_processing = self._processing

        logger.info("reload_gallery(): recycling %d worker(s) (processing=%s)", count, was_processing)
        self.stop_idle()
        self.start_idle(count)
        if was_processing:
            self.start_process()
        if self._on_topology_changed is not None:
            try:
                self._on_topology_changed()
            except Exception:
                logger.exception("on_topology_changed callback failed after reload_gallery")
        logger.info("reload_gallery() done")

    # ================================================================
    # Worker spawn / watchdog
    # ================================================================
    def _spawn_worker_locked(self) -> _WorkerHandle:
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        loaded_event = mp.Event()
        proc = RecognitionWorker(
            worker_id=worker_id,
            processing_event=self._processing_event,
            stop_event=self._stop_event,
            loaded_event=loaded_event,
        )
        proc.start()
        handle = _WorkerHandle(worker_id=worker_id, proc=proc, loaded_event=loaded_event)
        self._workers.append(handle)
        logger.info("spawned worker %s (pid=%s)", worker_id, proc.pid)
        return handle

    def _await_loaded_locked(self):
        deadline = time.time() + config.WORKER_LOAD_TIMEOUT_SEC
        for handle in self._workers:
            remaining = max(0.0, deadline - time.time())
            if not handle.loaded_event.wait(timeout=remaining):
                logger.warning(
                    "worker %s did not report ready within %.0fs — continuing anyway, "
                    "it will finish loading in the background",
                    handle.worker_id, config.WORKER_LOAD_TIMEOUT_SEC,
                )

    def _start_watchdog_locked(self):
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="recognizer-watchdog")
        self._watchdog_thread.start()

    def _stop_watchdog_locked(self):
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
        self._watchdog_thread = None

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(config.WATCHDOG_INTERVAL_SEC):
            respawned = False
            with self._lock:
                if not self._workers or self._stop_event is None:
                    continue
                for i, handle in enumerate(list(self._workers)):
                    if handle.proc.is_alive():
                        continue
                    logger.error(
                        "worker %s (pid=%s) died unexpectedly (exitcode=%s) — respawning",
                        handle.worker_id, handle.proc.pid, handle.proc.exitcode,
                    )
                    self._workers.remove(handle)
                    new_handle = self._spawn_worker_locked()
                    new_handle.loaded_event.wait(timeout=config.WORKER_LOAD_TIMEOUT_SEC)
                    respawned = True
            if respawned and self._on_topology_changed is not None:
                try:
                    self._on_topology_changed()
                except Exception:
                    logger.exception("on_topology_changed callback failed after respawn")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "worker_count": len(self._workers),
                "processing": self._processing,
                "workers": [
                    {"worker_id": h.worker_id, "pid": h.proc.pid, "alive": h.proc.is_alive()}
                    for h in self._workers
                ],
            }
