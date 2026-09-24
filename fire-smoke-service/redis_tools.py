"""
redis_tools.py (fire module)
--------------------------------------------------------------------
Manual test harness / CLI. Runs OUTSIDE docker, from PyCharm, a REPL,
or `python redis_tools.py <subcommand>`, and stands in for the
backend: it writes cameras:config, sends activate/deactivate commands,
and watches what comes back. Same shape as plate-service's and
face-service's own redis_tools.py, trimmed for this module: fire-smoke
is a SINGLE service (fire_detector) — there is no paired OCR/recognizer
second stage, so those subcommands/helpers don't exist here, and
results land on detections:results (THREAT/RESOLUTION events) instead
of vehicle:results.

It talks to the SAME Redis the containers use, over the published host
port. If you changed the published port or host, set the HOST
OVERRIDES below (or the matching env vars) before running.

Nothing here imports torch/cv2/ultralytics, so this file starts in
under a second either way.

Requires: pip install redis requests

Field shape for the camera config JSON — fire-smoke cameras only carry
a ROI (no cross-line / stop-ROI triggers; those are plate-specific
vehicle-stop triggers that don't apply to a fire/smoke grid-verification
pipeline):

  roi : normalized [0..1] fractions of the frame -> {"x":0,"y":0,"w":1,"h":1}

CLI usage:

  python redis_tools.py set-camera --id 1 --address "rtsp://admin:admin123@192.168.30.49:554/cam/realmonitor?channel=2&subtype=0" \\
      --title "Warehouse 1" --roi 0 0 1 1

  python redis_tools.py activate --id 1
  python redis_tools.py status --id 1
  python redis_tools.py deactivate --id 1
  python redis_tools.py list
  python redis_tools.py remove --id 1

Library usage (import redis_tools as rt) adds: show_active_cameras(),
show_self_healing_state(), show_heartbeats(), watch_camera_events(),
watch_results(), show_status(), test_service_restart(),
test_connect_disconnect_reconnect(), run_engine_consolidation_test().
--------------------------------------------------------------------
"""

import argparse
import json
import os
import sys
import time
import uuid

import redis

# ---------------------------------------------------------------------
# HOST OVERRIDES — set before anything else touches Redis.
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", os.getenv("REDIS_PUBLISH_PORT", "6379")))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_URL = os.getenv("REDIS_URL", "")
MODULE_KEY = os.getenv("REDIS_MODULE", "fire")

DETECTOR_API = os.getenv("DETECTOR_API", "http://localhost:8012")

# ---- backend contract (fixed spelling — see README "Redis contract") --
CFG_HASH = f"{MODULE_KEY}:cameras:config"
DETAILS_HASH = f"{MODULE_KEY}:cameras:details"
CMD_REQ_LIST = f"{MODULE_KEY}:cmd:ai:request"
RESULTS_KEY = f"{MODULE_KEY}:detections:results"
# Singular "camera" — matches this module's actual eyepass-camera-stream
# deployment. Not a typo; see common/firecore/keys.py's docstring.
CAMERA_EVENTS_CHANNEL = f"{MODULE_KEY}:camera:events"

# ---- internal (firecore.keys — self-healing) --------------------------
ACTIVE_CAMERAS_KEY = f"{MODULE_KEY}:internal:active_cameras"
DETECTOR_STATE_KEY = f"{MODULE_KEY}:internal:detector:state"
DETECTOR_HEARTBEAT_KEY = f"{MODULE_KEY}:internal:detector:heartbeat"


def cmd_resp_key(request_id: str) -> str:
    return f"{MODULE_KEY}:cmd:ai:response:{request_id}"


def ai_status_key(camera_id: str) -> str:
    return f"{MODULE_KEY}:cameras:{camera_id}:ai_status"


def get_client() -> "redis.Redis":
    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                        password=REDIS_PASSWORD, decode_responses=True)


r = get_client()

# Docker restart commands — printed by test_service_restart(). Only one
# service here (no paired OCR/recognizer container to restart).
_RESTART_CMDS = {
    "detector": {"graceful": "docker compose restart fire_detector", "hard": "docker kill fire_detector"},
}


# =====================================================================
# Standing in for the BACKEND — config / activate / deactivate
# =====================================================================

def set_camera(camera_id, address, title="Camera", usage="fire",
                roi=(0.0, 0.0, 1.0, 1.0)):
    """roi: (x, y, w, h) normalized 0..1. No cross-line / stop-ROI
    triggers here — those are plate-specific vehicle-stop triggers;
    the fire/smoke pipeline's own spatial verification (2x2 grid) is
    driven entirely by GRID_* config, not per-camera fields."""
    camera_id = str(camera_id)
    cfg = {
        "id": camera_id, "title": title, "address": address, "usage": usage,
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
    }
    r.hset(CFG_HASH, camera_id, json.dumps(cfg, ensure_ascii=False))
    print(f"[set-camera] wrote {CFG_HASH}[{camera_id}] = {json.dumps(cfg)}")


def remove_camera_config(camera_id):
    camera_id = str(camera_id)
    r.hdel(CFG_HASH, camera_id)
    print(f"[remove] deleted {CFG_HASH}[{camera_id}]")


def list_cameras():
    stored = r.hgetall(CFG_HASH)
    if not stored:
        print("(no cameras configured)")
        return {}
    out = {}
    for cid, raw in stored.items():
        print(f"--- camera {cid} ---")
        print(raw)
        status_raw = r.hget(ai_status_key(cid), cid)
        print(f"ai_status: {status_raw}")
        out[cid] = {"config": raw, "ai_status": status_raw}
    return out


def _send_action(camera_id: str, action: str, timeout: float = 10.0):
    camera_id = str(camera_id)
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "camera_id": camera_id, "action": action}
    # fire_detector's cmd worker does BRPOP (reads from the tail), so we
    # must LPUSH (write to the head) to preserve FIFO order.
    r.lpush(CMD_REQ_LIST, json.dumps(payload))
    print(f"[{action}] sent request_id={request_id} camera_id={camera_id}, waiting for response...")

    resp = r.blpop(cmd_resp_key(request_id), timeout=timeout)
    if resp is None:
        print(f"[{action}] TIMEOUT waiting for response (is fire_detector running?)")
        return False
    _, raw = resp
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = raw
    print(f"[{action}] response: {parsed}")
    return parsed.get("status") == "OK" if isinstance(parsed, dict) else False


def activate(camera_id: str):
    return _send_action(camera_id, "activated")


def deactivate(camera_id: str):
    return _send_action(camera_id, "deactivated")


def status(camera_id: str):
    camera_id = str(camera_id)
    raw = r.hget(ai_status_key(camera_id), camera_id)
    if raw is None:
        print(f"(no ai_status recorded yet for camera {camera_id})")
        return None
    parsed = json.loads(raw)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return parsed


def ensure_camera_active(camera_id, address, roi=None):
    """Idempotent: seeds + activates camera_id only if it isn't already
    in active_cameras."""
    if r.hexists(ACTIVE_CAMERAS_KEY, str(camera_id)):
        print(f"camera {camera_id} already active — skipping seed/activate")
        return
    set_camera(camera_id, address, roi=roi or (0.0, 0.0, 1.0, 1.0))
    time.sleep(1)
    activate(camera_id)
    time.sleep(5)


# =====================================================================
# Inspection — backend contract
# =====================================================================

def show_details(camera_id=None):
    """What camera_stream currently believes about each camera."""
    if camera_id is not None:
        raw = r.hget(DETAILS_HASH, str(camera_id))
        entries = {str(camera_id): raw} if raw else {}
    else:
        entries = r.hgetall(DETAILS_HASH)

    if not entries:
        print("(cameras:details is empty — is camera_stream running?)")
        return {}

    out = {}
    for cid, raw in entries.items():
        data = json.loads(raw)
        out[cid] = data
        print(f"cam={cid} connected={data.get('connected')} error={data.get('error')} "
              f"stream_url={data.get('stream_url')}")
    return out


def read_recent_results(count=10):
    """Peeks at the newest entries on detections:results without
    consuming them (THREAT / RESOLUTION events). The real backend
    should BRPOP (FIFO, since the detector RPUSHes)."""
    items = r.lrange(RESULTS_KEY, -count, -1)
    if not items:
        print("(detections:results is empty — nothing has been processed yet)")
    for raw in items:
        print(json.loads(raw))
    return [json.loads(i) for i in items]


def watch_camera_events(seconds=60.0):
    """Prints every camera:events transition as it arrives."""
    pubsub = r.pubsub()
    pubsub.subscribe(CAMERA_EVENTS_CHANNEL)
    print(f"Watching {CAMERA_EVENTS_CHANNEL} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            message = pubsub.get_message(timeout=1.0)
            if message and message.get("type") == "message":
                data = json.loads(message["data"])
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('id', data.get('camera_id'))} "
                      f"connected={data.get('connected')} error={data.get('error')}")
    except KeyboardInterrupt:
        pass
    finally:
        pubsub.close()


def watch_results(seconds=60.0):
    """Blocks on detections:results (BRPOP) and prints each THREAT/
    RESOLUTION event as it's published — the fastest way to see the
    full detector -> grid-verification -> detections:results round
    trip live."""
    print(f"Watching {RESULTS_KEY} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            item = r.brpop(RESULTS_KEY, timeout=1)
            if item:
                _, raw = item
                data = json.loads(raw)
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('camera_id')} "
                      f"event={data.get('event')} class={data.get('class') or data.get('label')} "
                      f"region={data.get('region')} conf={data.get('confidence')}")
    except KeyboardInterrupt:
        pass


# =====================================================================
# Inspection — internal / self-healing
# =====================================================================

def show_active_cameras():
    """The detector's durable desired state: which cameras it believes
    it is responsible for. This is what survives a restart."""
    entries = r.hgetall(ACTIVE_CAMERAS_KEY)
    if not entries:
        print(f"({ACTIVE_CAMERAS_KEY} is empty — nothing is activated)")
        return {}
    out = {}
    for cid, raw in entries.items():
        data = json.loads(raw)
        out[cid] = data
        since = time.strftime("%H:%M:%S", time.localtime(data.get("activated_at", 0)))
        print(f"cam={cid} activated_at={since}")
    return out


def show_self_healing_state():
    """The detector's self-healing checkpoint: phase (stopped/idle/
    processing) and how many engines were running."""
    det = r.hgetall(DETECTOR_STATE_KEY)
    print(f"detector ({DETECTOR_STATE_KEY}): {det or '(empty — never checkpointed yet)'}")
    return {"detector": det}


def show_heartbeats():
    det = r.get(DETECTOR_HEARTBEAT_KEY)

    def _age(ts):
        return None if ts is None else time.time() - float(ts)

    det_age = _age(det)
    print(f"detector heartbeat: {'no heartbeat (service down or TTL expired)' if det_age is None else f'{det_age:.1f}s ago'}")
    return {"detector_age": det_age}


def show_status():
    """Live state from the detector's HTTP health endpoint."""
    import requests
    out = {}
    try:
        health = requests.get(f"{DETECTOR_API}/health", timeout=3).json()
        print(f"detector: {health}")
        out["detector"] = health
    except Exception as e:
        print(f"detector: could not reach {DETECTOR_API}: {e}")
        out["detector"] = None
    return out


# =====================================================================
# Scenarios — service restart (self-healing)
# =====================================================================

def test_service_restart(service="detector", camera_id="1", address="", roi=None, hard_kill=False):
    """Proves "resume where it left off":

        test_service_restart("detector")   # fire_detector restarts

    Steps, with a manual pause in the middle:
      1. Make sure the camera is active.
      2. YOU restart fire_detector in another terminal (the exact
         command is printed for you — pass hard_kill=True for `docker
         kill` instead of a graceful `restart`).
      3. Press Enter here. The restarted service should come back at
         the SAME phase (processing) with the SAME engine count, and
         the camera should be running again WITHOUT any new activate
         command — self_heal() replaying its checkpoint on boot.
    """
    if service not in _RESTART_CMDS:
        raise ValueError(f"service must be one of {list(_RESTART_CMDS)}, got {service!r}")
    if address:
        print(f"\n=== Ensuring camera {camera_id} is active ===")
        ensure_camera_active(camera_id, address, roi)

    print("\n--- before restart ---")
    show_active_cameras()
    show_self_healing_state()
    show_heartbeats()
    show_status()

    cmd = _RESTART_CMDS[service]["hard" if hard_kill else "graceful"]
    print("\n" + "=" * 62)
    print(f"Now run this in another terminal to restart: {service}")
    print(f"  {cmd}")
    print("=" * 62)
    input("Press Enter once it's back up... ")

    print("\n--- after restart (no new activate command was sent) ---")
    show_active_cameras()
    show_self_healing_state()
    show_heartbeats()
    show_status()

    print("\nRecent results:")
    read_recent_results()


def test_connect_disconnect_reconnect(camera_id="1", address="", roi=None, total_seconds=300.0):
    """End-to-end test with a REAL camera. You do the physical
    unplugging/replugging by hand; this registers + activates it, then
    watches camera:events live so you can see camera_stream notice the
    drop and the reconnect, and fire_detector's offline-grace timer
    (CAMERA_OFFLINE_GRACE_SECONDS) absorb a brief blip without tearing
    the camera down."""
    print(f"\n=== Registering + activating camera {camera_id} ===")
    ensure_camera_active(camera_id, address, roi)

    print("\n--- current state ---")
    show_details(camera_id)
    show_active_cameras()
    show_self_healing_state()

    print(f"\n=== Watching for {total_seconds:.0f}s — unplug and replug the camera now ===\n")
    watch_camera_events(seconds=total_seconds)

    print("\n--- state after the disconnect/reconnect window ---")
    show_details(camera_id)
    show_self_healing_state()

    print("\nRecent results:")
    read_recent_results()


def run_engine_consolidation_test(camera_ids, addresses):
    """Activates more cameras than MAX_CAMERAS_PER_ENGINE allows in one
    engine, then deactivates enough of them that everything remaining
    COULD fit in fewer engines — and watches EngineManager.rebalance()
    migrate them back down on its next tick
    (ENGINE_REBALANCE_INTERVAL_SEC). Addresses don't need to be real —
    this is testing engine topology math, not actual detection."""
    assert len(camera_ids) == len(addresses)

    print(f"\n=== Activating {len(camera_ids)} cameras one at a time ===")
    for cid, addr in zip(camera_ids, addresses):
        set_camera(cid, addr)
        time.sleep(0.5)
        activate(cid)
        time.sleep(0.5)

    print("\n--- topology after spreading across engines ---")
    show_status()

    half = len(camera_ids) // 2
    to_remove = camera_ids[half:]
    print(f"\n=== Deactivating {len(to_remove)} cameras to trigger consolidation ===")
    for cid in to_remove:
        deactivate(cid)

    print("\nWaiting for the next rebalance tick...")
    time.sleep(35)

    print("\n--- topology after rebalance (should be fewer engines) ---")
    show_status()


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("set-camera", help="write/update a camera's config in Redis")
    sc.add_argument("--id", required=True)
    sc.add_argument("--address", required=True, help="RTSP URL or video path")
    sc.add_argument("--title", default="Camera")
    sc.add_argument("--usage", default="fire")
    sc.add_argument("--roi", nargs=4, type=float, default=[0, 0, 1, 1], metavar=("X", "Y", "W", "H"))

    rc = sub.add_parser("remove", help="delete a camera's config")
    rc.add_argument("--id", required=True)

    sub.add_parser("list", help="list configured cameras + their ai_status")

    ac = sub.add_parser("activate", help="tell fire_detector to start processing this camera")
    ac.add_argument("--id", required=True)

    dc = sub.add_parser("deactivate", help="tell fire_detector to stop processing this camera")
    dc.add_argument("--id", required=True)

    st = sub.add_parser("status", help="read the current ai_status for a camera")
    st.add_argument("--id", required=True)

    sub.add_parser("self-heal-state", help="show the detector's self-healing checkpoint")
    sub.add_parser("heartbeats", help="show the detector's heartbeat age")
    sub.add_parser("health", help="hit the detector's /health endpoint")

    args = p.parse_args()

    if args.cmd == "set-camera":
        set_camera(args.id, args.address, title=args.title, usage=args.usage,
                    roi=tuple(args.roi))
    elif args.cmd == "remove":
        remove_camera_config(args.id)
    elif args.cmd == "list":
        list_cameras()
    elif args.cmd == "activate":
        sys.exit(0 if activate(args.id) else 1)
    elif args.cmd == "deactivate":
        sys.exit(0 if deactivate(args.id) else 1)
    elif args.cmd == "status":
        status(args.id)
    elif args.cmd == "self-heal-state":
        show_self_healing_state()
    elif args.cmd == "heartbeats":
        show_heartbeats()
    elif args.cmd == "health":
        show_status()


if __name__ == "__main__":
    main()
