"""
rec_client.py
--------------------------------------------------------------------
Detector-side half of the detector<->recognizer contract. Replaces the
reference pipeline's in-process `fr_input_queue` / `fr_output_queue`
(multiprocessing.Queue, one pair per Engine, feeding a pool of AFRWorker
subprocesses spawned BY that engine) with the Redis queues described in
facecore.keys: a shared `rec:tasks` list every engine publishes onto,
and a per-engine `rec:results:{engine_id}` list that routes results
back to the one Engine instance that owns the track.

Recognition capacity is no longer tied to detection engine count — the
recognizer is a separate service with its own worker pool, scaled
independently. An Engine does not wait for the recognizer to be up:
tasks simply queue in Redis until a worker is available, which is a
direct benefit of decoupling the two over Redis instead of an
in-process queue pair.

Used from inside an Engine subprocess (Engine.__init__ already runs in
the spawned child — see engine.py / _engine_process_main), so it is
safe to open the Redis connection here directly.
--------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List

from facecore.bus import RedisBus
from facecore.codec import decode_result, encode_task


class RecognitionClient:
    def __init__(self, bus: RedisBus, engine_id):
        self.bus = bus
        self.engine_id = engine_id
        self._local_results: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name=f"RecResults-{engine_id}"
        )
        self._thread.start()

    def submit(self, task: Dict[str, Any]):
        task.setdefault("engine_id", self.engine_id)
        self.bus.push_task(encode_task(task))

    def _listen(self):
        while not self._stop.is_set():
            try:
                raw = self.bus.pop_result_bytes(self.engine_id, timeout=1)
            except Exception:
                # Transient Redis hiccup — the loop just retries; the
                # detector keeps detecting/tracking regardless, results
                # simply arrive late.
                continue
            if raw is None:
                continue
            try:
                result = decode_result(raw)
            except Exception:
                continue
            self._local_results.put(result)

    def drain(self) -> List[Dict[str, Any]]:
        """Non-blocking: every result currently buffered locally, in the
        order received. Mirrors the old fr_output_queue.get_nowait()
        drain loop so call sites in engine.py need no other changes."""
        out = []
        while True:
            try:
                out.append(self._local_results.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self):
        self._stop.set()
