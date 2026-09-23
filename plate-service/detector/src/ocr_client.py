"""
ocr_client.py
--------------------------------------------------------------------
Detector-side half of the detector<->OCR contract. Replaces the
reference pipeline's in-process `ocr_input_queue` / `ocr_output_queue`
(multiprocessing.Queue, one pair per Engine, feeding a pool of
OCRWorker subprocesses spawned BY that engine, 3 per engine in the
reference alpr_api.py) with the Redis queues described in
platecore.keys: a shared `ocr:tasks` list every engine publishes onto,
and a per-engine `ocr:results:{engine_id}` list that routes results
back to the one Engine instance that owns the track.

OCR capacity is no longer tied to detection engine count — the OCR
service is a separate process pool, scaled independently. An Engine
does not wait for the OCR service to be up: tasks simply queue in
Redis until a worker is available.

Used from inside an Engine subprocess (Engine.__init__ already runs in
the spawned child — see engine.py / _engine_process_main), so it is
safe to open the Redis connection here directly.
--------------------------------------------------------------------
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List

from platecore.bus import RedisBus
from platecore.codec import decode_result, encode_task


class OcrClient:
    def __init__(self, bus: RedisBus, engine_id):
        self.bus = bus
        self.engine_id = engine_id
        self._local_results: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name=f"OcrResults-{engine_id}"
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
        order received. Mirrors the old ocr_output_queue.get_nowait()
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
