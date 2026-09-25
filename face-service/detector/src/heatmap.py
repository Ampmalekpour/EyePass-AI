"""
heatmap.py (face detector)
--------------------------------------------------------------------
Optional crowd/occupancy heatmap gathered from the face detector's own
head tracks — the same data product the heatmap-service produces from
its own person detector, without running a second detector on the
same camera.

On/off:
  * globally:   HEATMAP_ENABLED=true|false   (.env / compose)
  * per camera: optional `"heatmap": true|false` in the camera's
                face:cameras:config entry; absent = follow the global
                switch. Lets the backend turn it on for a lobby camera
                and off for a door camera without a redeploy.

Data format — identical to heatmap-service (HeatmapCubeManager), so
anything that reads those cubes reads these:
  * one "cube" per camera per day: uint32 array
        shape = (24*60 / HEATMAP_TIME_RESOLUTION_MINUTES,
                 HEATMAP_GRID_HEIGHT, HEATMAP_GRID_WIDTH)
  * each sample adds 1 to cube[time_slot][row][col] for the cell the
    head's point falls in (point = head-box centre, or bottom-centre)
  * stored in MinIO as  <bucket>/<camera_id>/<YYYY-MM-DD>.npy
    (bucket HEATMAP_MINIO_BUCKET, default "face-heatmap")
  * every flush is announced on the Redis list  face:heatmap:results
    as {"event": "matrix_sync", camera_id, date, bucket, object_key,
    timestamp}, and a camera leaving the engine additionally sends
    {"event": "processing_finished"} — same event shapes as
    heatmap-service's ai:results.

Sampling: every HEATMAP_SAMPLE_EVERY_N_FRAMES frames each live track
contributes one point (dwell-weighted occupancy, like heatmap-service's
DETECTION_ACCUMULATION_INTERVAL).

Robust by construction:
  * The frame loop only increments an in-memory DELTA cube (no I/O).
  * A background thread merges each delta into the stored cube
    (download -> add -> upload) every HEATMAP_SAVE_INTERVAL_SEC. If
    MinIO is unreachable the delta is kept and merged into the next
    attempt, so counts are delayed, never lost.
  * Merging (instead of overwriting from an in-memory copy) is what
    makes restarts safe: a restarted engine starts from an empty delta
    and simply keeps adding to what is already stored — nothing to
    reload, nothing double-counted. Same for a camera moving to another
    engine during a rebalance.
  * Camera removal and engine shutdown flush synchronously. The worst
    case is an unclean kill (OOM, power loss): up to one save interval
    of counts, same bound as heatmap-service.
--------------------------------------------------------------------
"""

from __future__ import annotations

import io
import json
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

import config
from facecore.logging_setup import setup_logger

logger = setup_logger("heatmap")


def _slots_per_day() -> int:
    return (24 * 60) // config.HEATMAP_TIME_RESOLUTION_MINUTES


class _NpyStore:
    """Tiny .npy get/put on MinIO using facecore's boto3 client (lazy,
    one per process — safe inside the spawned engine subprocess)."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self._client = None
        self._bucket_ok = False

    def _c(self):
        if self._client is None:
            from facecore.minio_store import get_minio_client
            self._client = get_minio_client()
        if not self._bucket_ok:
            from facecore.minio_store import ensure_bucket
            ensure_bucket(self.bucket)
            self._bucket_ok = True
        return self._client

    def get(self, key: str) -> Optional[np.ndarray]:
        client = self._c()
        try:
            obj = client.get_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return np.load(io.BytesIO(obj["Body"].read()), allow_pickle=False)

    def put(self, key: str, arr: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        self._c().put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue(),
                             ContentType="application/octet-stream")


class FaceHeatmap:
    """One per detector Engine. Thread-safe: `sample()` runs on the
    frame loop, flushing runs on its own thread."""

    def __init__(self, bus, engine_id, store: Optional[_NpyStore] = None, results_key: Optional[str] = None):
        self.bus = bus
        self.engine_id = engine_id
        self.store = store or _NpyStore(config.HEATMAP_MINIO_BUCKET)
        self.results_key = results_key or f"{config.REDIS_MODULE}:heatmap:results"
        self.gw, self.gh = config.HEATMAP_GRID_WIDTH, config.HEATMAP_GRID_HEIGHT
        self.slots = _slots_per_day()
        # (camera_id, date) -> delta cube not yet merged into MinIO
        self._delta: Dict[Tuple[str, str], np.ndarray] = {}
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self.samples = 0
        self.last_error: Optional[str] = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"Heatmap-{engine_id}")
        self._thread.start()
        logger.info("face heatmap ON | grid=%dx%d | %d slots/day | sample every %d frames | flush every %ss "
                    "| minio bucket=%s | announce=%s", self.gw, self.gh, self.slots,
                    config.HEATMAP_SAMPLE_EVERY_N_FRAMES, config.HEATMAP_SAVE_INTERVAL_SEC,
                    self.store.bucket, self.results_key)

    # ---- frame loop ---------------------------------------------------------
    @staticmethod
    def point(bbox_full: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox_full
        if config.HEATMAP_POINT_MODE == "bottom_center":
            return (x1 + x2) / 2.0, float(y2)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def sample(self, camera_id: str, x: float, y: float, frame_w: int, frame_h: int,
               ts: Optional[datetime] = None):
        """Adds one point (full-frame pixel coords). Pure memory — no I/O."""
        if frame_w <= 0 or frame_h <= 0:
            return
        ts = ts or datetime.now()
        date = ts.strftime("%Y-%m-%d")
        slot = (ts.hour * 60 + ts.minute) // config.HEATMAP_TIME_RESOLUTION_MINUTES
        col = min(max(int(x / (frame_w / self.gw)), 0), self.gw - 1)
        row = min(max(int(y / (frame_h / self.gh)), 0), self.gh - 1)
        key = (str(camera_id), date)
        with self._lock:
            cube = self._delta.get(key)
            if cube is None:
                cube = self._delta[key] = np.zeros((self.slots, self.gh, self.gw), dtype=np.uint32)
            cube[slot, row, col] += 1
            self.samples += 1

    def current_slot(self, camera_id: str) -> Optional[np.ndarray]:
        """This engine's not-yet-flushed counts for the current slot (debug)."""
        ts = datetime.now()
        key = (str(camera_id), ts.strftime("%Y-%m-%d"))
        slot = (ts.hour * 60 + ts.minute) // config.HEATMAP_TIME_RESOLUTION_MINUTES
        with self._lock:
            cube = self._delta.get(key)
            return None if cube is None else cube[slot].copy()

    # ---- flushing -----------------------------------------------------------------
    def _loop(self):
        while not self._stop.wait(config.HEATMAP_SAVE_INTERVAL_SEC):
            self.flush()

    def flush(self, camera_id: Optional[str] = None, finished: bool = False) -> bool:
        """Merge pending deltas (all cameras, or one) into MinIO.
        Returns True when nothing is left pending for the selection."""
        with self._flush_lock:
            with self._lock:
                keys = [k for k in self._delta if camera_id is None or k[0] == str(camera_id)]
                batch = {k: self._delta.pop(k) for k in keys}
            ok = True
            for (cam, date), delta in batch.items():
                object_key = f"{cam}/{date}.npy"
                try:
                    stored = self.store.get(object_key)
                    if stored is not None and stored.shape != delta.shape:
                        # grid or time resolution was changed mid-day: never
                        # corrupt the existing cube — continue in a sibling
                        # object named after the new shape
                        alt = f"{cam}/{date}.{self.slots}x{self.gh}x{self.gw}.npy"
                        logger.error("heatmap %s has shape %s, expected %s (grid/time resolution changed?) — "
                                     "writing to %s instead", object_key, stored.shape, delta.shape, alt)
                        object_key = alt
                        stored = self.store.get(object_key)
                    merged = delta if stored is None else stored + delta
                    self.store.put(object_key, merged)
                    self._announce({"camera_id": cam, "event": "matrix_sync", "date": date,
                                    "bucket": self.store.bucket, "object_key": object_key,
                                    "source": "face_detector", "samples_added": int(delta.sum()),
                                    "timestamp": time.time()})
                    self.last_error = None
                except Exception as e:
                    ok = False
                    self.last_error = f"{type(e).__name__}: {e}"
                    logger.warning("heatmap flush %s failed (%s) — keeping %d samples for the next attempt",
                                   object_key, e, int(delta.sum()))
                    with self._lock:  # put it back, merged with anything gathered meanwhile
                        cur = self._delta.get((cam, date))
                        self._delta[(cam, date)] = delta if cur is None else cur + delta
            if finished and camera_id is not None:
                self._announce({"camera_id": str(camera_id), "event": "processing_finished",
                                "source": "face_detector", "timestamp": time.time()})
            return ok

    def _announce(self, payload: dict):
        try:
            self.bus.rt.rpush(self.results_key, json.dumps(payload))
        except Exception as e:
            logger.warning("heatmap announce failed: %s", e)

    def pending_samples(self) -> int:
        with self._lock:
            return int(sum(int(c.sum()) for c in self._delta.values()))

    def stop(self):
        """Final synchronous flush (engine shutdown)."""
        self._stop.set()
        if not self.flush():
            logger.error("heatmap: %d samples could not be saved at shutdown (MinIO unreachable)",
                         self.pending_samples())
