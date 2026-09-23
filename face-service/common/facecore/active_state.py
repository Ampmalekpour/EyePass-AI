"""
active_state.py
--------------------------------------------------------------------
Durable desired-state ledger: which cameras this module believes it is
responsible for right now. This is what makes the "gets itself back to
where it was" requirement possible — it is written BEFORE we act, so a
crash between "backend said activate" and "the engine actually started"
is recovered by the next startup reconcile, not lost.

Mirrors ActiveCameraState from the heatmap reference (api.py), renamed
to make clear it is about cameras specifically — the detector service
also persists its own engine_count and the recognizer its worker_count
via lifecycle.py, which is a different kind of state.
--------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .bus import RedisBus


class ActiveCameraState:
    def __init__(self, bus: RedisBus):
        self.bus = bus
        self.key = bus.keys.active_cameras

    def mark_active(self, camera_id: str, roi: dict, request_id: Optional[str] = None,
                     extra: Optional[Dict[str, Any]] = None):
        entry = {
            "roi": roi,
            "request_id": request_id,
            "activated_at": time.time(),
        }
        if extra:
            entry.update(extra)
        self.bus.hset_json(self.key, str(camera_id), entry)

    def mark_inactive(self, camera_id: str):
        self.bus.hdel(self.key, str(camera_id))

    def is_active(self, camera_id: str) -> bool:
        return self.bus.hget_json(self.key, str(camera_id)) is not None

    def get(self, camera_id: str) -> Optional[dict]:
        return self.bus.hget_json(self.key, str(camera_id))

    def all_active(self) -> Dict[str, dict]:
        return self.bus.hgetall_json(self.key)
