"""
relay.py — the ONE naming rule every service in the suite uses to find a
camera on the shared MediaMTX relay. The same function is copied into each
module's common library (facecore/platecore/heatmapcore/firecore relay.py);
keep them identical.

RELAY_PATH_MODE (env):
  source     (suite default) path = "cam_" + sha1(camera address)[:12]
             One relay path per PHYSICAL camera: when face, plate, heatmap
             and fire all watch the same camera, the relay pulls it once
             and every module reads the same path. Different modules'
             camera ids can also no longer collide ("1" in face and "1" in
             plate are different cameras with different addresses).
  camera_id  path = camera id (the old per-module behaviour — only safe
             when each module has its own relay)
"""

import hashlib
import os
from typing import Optional

RELAY_PATH_MODE = os.getenv("RELAY_PATH_MODE", "source").strip().lower()


def relay_path_for(camera_id, address: Optional[str]) -> str:
    address = (address or "").strip()
    if RELAY_PATH_MODE == "source" and address:
        return "cam_" + hashlib.sha1(address.encode("utf-8")).hexdigest()[:12]
    return str(camera_id)
