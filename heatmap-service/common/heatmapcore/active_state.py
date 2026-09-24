"""
active_state.py
--------------------------------------------------------------------
Durable desired-state ledger: which cameras this module believes it is
responsible for right now. Written BEFORE we act, so a crash between
"backend said activate" and "the engine actually started" is recovered
by the next startup reconcile, not lost.

This existed in the pre-existing standalone build of this module too
(as `ai_state.ActiveCameraState`, backed by `{module}:ai:active`) — it
is carried over here unchanged in behavior, just moved into the shared
`heatmapcore` library so it shares one Redis connection/key-naming
source of truth with everything else, matching the plate/face/fire
modules' own `active_state.py`.
--------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .bus import RedisBus


class ActiveCameraState:
    def __init__(self, bus: RedisBus):
        self.bus = bus
        self.key = bus.keys.ai_active

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
