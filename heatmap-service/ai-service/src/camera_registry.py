"""
camera_registry.py
--------------------------------------------------------------------
Persistent camera_id -> {video_source, roi, status, engine_id, ...}
mapping, backed by a single JSON file.

Why this exists:
  Previously, `launch_api.py` only kept camera info in an in-memory
  dict (`processes`). That meant nothing survived a server restart,
  and no other tool (e.g. the visualizer) had any way to know which
  RTSP URL a given camera_id pointed to without being told again
  by hand.

This module is intentionally simple (flat JSON file + a lock) rather
than a full database — the write volume is low (one write per
start/stop/reset call), and it's the smallest thing that fixes the
actual problem: durable, shared camera_id -> source lookup.
--------------------------------------------------------------------
"""

import json
import os
import threading
from typing import Any, Dict, Optional

from config import CAMERA_REGISTRY_PATH

_lock = threading.Lock()


def _read_all() -> Dict[str, Any]:
    if not os.path.exists(CAMERA_REGISTRY_PATH):
        return {}
    try:
        with open(CAMERA_REGISTRY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable registry shouldn't crash the service —
        # treat it as empty and let the next write repair it.
        return {}


def _write_all(data: Dict[str, Any]) -> None:
    # Write to a temp file then atomically replace, so a crash mid-write
    # never leaves a half-written/corrupt registry file behind.
    tmp_path = CAMERA_REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, CAMERA_REGISTRY_PATH)


def upsert_camera(camera_id: str, **fields: Any) -> None:
    """Create or update a camera's entry with the given fields (partial update)."""
    camera_id = str(camera_id)
    with _lock:
        data = _read_all()
        entry = data.get(camera_id, {})
        entry.update(fields)
        data[camera_id] = entry
        _write_all(data)


def remove_camera(camera_id: str) -> None:
    """Permanently delete a camera's entry from the registry."""
    camera_id = str(camera_id)
    with _lock:
        data = _read_all()
        data.pop(camera_id, None)
        _write_all(data)


def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    """Look up a single camera's stored fields, or None if it isn't registered."""
    return _read_all().get(str(camera_id))


def all_cameras() -> Dict[str, Any]:
    """Return the full registry (all camera_id -> fields)."""
    return _read_all()
