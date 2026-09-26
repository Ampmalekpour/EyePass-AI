"""
redis_tools.py (plate module)
--------------------------------------------------------------------
Manual test harness / CLI. Runs OUTSIDE docker, from PyCharm, a REPL,
or `python redis_tools.py <subcommand>`, and stands in for the
backend: it writes cameras:config, sends activate/deactivate commands,
and watches what comes back. Renamed and expanded from the reference
redis_control.py — the original set-camera/activate/deactivate/status/
list/remove subcommands all still work the same way, plus the
introspection/scenario helpers ported from face_service's own
redis_tools.py (self-healing state, heartbeats, queue depth, restart
resilience test, engine consolidation test).

It talks to the SAME Redis the containers use, over the published host
port (compose.infra.yaml's REDIS_PUBLISH_PORT, or your platform's own
Redis). If you changed the published port or host, set the HOST
OVERRIDES below (or the matching env vars) before running.

Nothing here imports torch/cv2/ultralytics/paddleocr, so this file
starts in under a second either way.

Requires: pip install redis requests

Field shapes for the camera config JSON (verbatim from the reference
redis_control.py / alpr_api.py._build_video_config):

  roi        : normalized [0..1] fractions of the frame -> {"x":0,"y":0,"w":1,"h":1}
  cross_line : PIXEL coordinates on the raw frame, from get_line_roi.py
               -> {"start": {"x":..,"y":..}, "end": {"x":..,"y":..}}
  stop_roi   : a PIXEL bounding box (x,y,w,h), turned into a 4-point
               polygon internally -> {"x":..,"y":..,"w":..,"h":..}

CLI usage (unchanged from redis_control.py):

  python redis_tools.py set-camera --id 1 --address "rtsp://admin:admin123@192.168.30.49:554/cam/realmonitor?channel=2&subtype=0" \\
      --title "Gate 1" --roi 0 0 1 1 \\
      --line 100 400 900 400 \\
      --stop-roi 150 350 700 200

  python redis_tools.py activate --id 1
  python redis_tools.py status --id 1
  python redis_tools.py deactivate --id 1
  python redis_tools.py list
  python redis_tools.py remove --id 1

Library usage (import redis_tools as rt) adds: show_active_cameras(),
show_self_healing_state(), show_heartbeats(), show_ocr_queue_depth(),
watch_camera_events(), watch_results(), show_status(),
test_service_restart(), test_connect_disconnect_reconnect(),
measure_performance(), run_engine_consolidation_test().
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
MODULE_KEY = os.getenv("REDIS_MODULE", "plate")

DETECTOR_API = os.getenv("DETECTOR_API", "http://localhost:8010")
OCR_API = os.getenv("OCR_API", "http://localhost:8011")

# ---- backend contract (fixed spelling — see README "Redis contract") --
CFG_HASH = f"{MODULE_KEY}:cameras:config"
DETAILS_HASH = f"{MODULE_KEY}:cameras:details"
CMD_REQ_LIST = f"{MODULE_KEY}:cmd:ai:request"
RESULTS_KEY = f"{MODULE_KEY}:vehicle:results"
# Singular "camera" — matches this module's actual eyepass-camera-stream
# deployment. Not a typo; see common/platecore/keys.py's docstring.
CAMERA_EVENTS_CHANNEL = f"{MODULE_KEY}:camera:events"

# ---- internal (platecore.keys — self-healing / detector<->ocr) --------
ACTIVE_CAMERAS_KEY = f"{MODULE_KEY}:internal:active_cameras"
DETECTOR_STATE_KEY = f"{MODULE_KEY}:internal:detector:state"
OCR_STATE_KEY = f"{MODULE_KEY}:internal:ocr:state"
DETECTOR_HEARTBEAT_KEY = f"{MODULE_KEY}:internal:detector:heartbeat"
OCR_HEARTBEAT_KEY = f"{MODULE_KEY}:internal:ocr:heartbeat"
OCR_TASKS_KEY = f"{MODULE_KEY}:internal:ocr:tasks"
OCR_TASKS_PENDING_KEY = f"{MODULE_KEY}:internal:ocr:tasks:pending"


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

# Docker restart commands per service — printed by test_service_restart().
_RESTART_CMDS = {
    "detector": {"graceful": "docker compose restart plate_detector", "hard": "docker kill plate_detector"},
    "ocr": {"graceful": "docker compose restart plate_ocr", "hard": "docker kill plate_ocr"},
    "both": {
        "graceful": "docker compose restart plate_detector plate_ocr",
        "hard": "docker kill plate_detector plate_ocr",
    },
}


# =====================================================================
# Standing in for the BACKEND — config / activate / deactivate
# =====================================================================

def set_camera(camera_id, address, title="Camera", usage="plate",
                roi=(0.0, 0.0, 1.0, 1.0), line=None, stop_roi=None):
    """roi: (x, y, w, h) normalized 0..1. line: (x1, y1, x2, y2) and
    stop_roi: (x, y, w, h) either in pixel coords of the camera frame or
    as fractions 0..1 (the detector tells them apart), or None to
    disable that trigger."""
    camera_id = str(camera_id)
    cfg = {
        "id": camera_id, "title": title, "address": address, "usage": usage,
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
    }
    if line is not None:
        x1, y1, x2, y2 = line
        cfg["cross_line"] = {"start": {"x": x1, "y": y1}, "end": {"x": x2, "y": y2}}
    if stop_roi is not None:
        x, y, w, h = stop_roi
        cfg["stop_roi"] = {"x": x, "y": y, "w": w, "h": h}

    r.hset(CFG_HASH, camera_id, json.dumps(cfg, ensure_ascii=False))
    # tell camera_stream, exactly like the backend does, so it picks the
    # camera up now instead of on its next restart
    r.publish(f"{MODULE_KEY}:camera:config:updated", camera_id)
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
    # plate_detector's cmd worker does BRPOP (reads from the tail), so we
    # must LPUSH (write to the head) to preserve FIFO order.
    r.lpush(CMD_REQ_LIST, json.dumps(payload))
    print(f"[{action}] sent request_id={request_id} camera_id={camera_id}, waiting for response...")

    resp = r.blpop(cmd_resp_key(request_id), timeout=timeout)
    if resp is None:
        print(f"[{action}] TIMEOUT waiting for response (is plate_detector running?)")
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


def ensure_camera_active(camera_id, address, roi=None, line=None, stop_roi=None):
    """Idempotent: seeds + activates camera_id only if it isn't already
    in active_cameras."""
    if r.hexists(ACTIVE_CAMERAS_KEY, str(camera_id)):
        print(f"camera {camera_id} already active — skipping seed/activate")
        return
    set_camera(camera_id, address, roi=roi or (0.0, 0.0, 1.0, 1.0), line=line, stop_roi=stop_roi)
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
    """Peeks at the newest entries on vehicle:results without consuming
    them. The real backend should BRPOP (FIFO, since the detector/OCR
    service RPUSHes)."""
    items = r.lrange(RESULTS_KEY, -count, -1)
    if not items:
        print("(vehicle:results is empty — nothing has been processed yet)")
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
    """Blocks on vehicle:results (BRPOP) and prints each result as it's
    published — the fastest way to see the full detector -> ocr ->
    control hub -> vehicle:results round trip live. DESTRUCTIVE (pops
    what Django would read) — for a read-only look use
    `python control-hub/hub_tools.py results`."""
    print(f"Watching {RESULTS_KEY} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            item = r.brpop(RESULTS_KEY, timeout=1)
            if item:
                _, raw = item
                data = json.loads(raw)
                res = data.get("resolved") or {}
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('camera_id')} "
                      f"track={data.get('track_id')} update={data.get('update_type')} "
                      f"final={data.get('is_final')} plate={res.get('plate_text')} "
                      f"conf={res.get('confidence')} valid={res.get('is_valid')} "
                      f"via={(data.get('meta') or {}).get('resolution')}")
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
    """Both services' self-healing checkpoints: phase (stopped/idle/
    processing) and how many engines/workers were running."""
    det = r.hgetall(DETECTOR_STATE_KEY)
    ocr = r.hgetall(OCR_STATE_KEY)
    print(f"detector ({DETECTOR_STATE_KEY}): {det or '(empty — never checkpointed yet)'}")
    print(f"ocr      ({OCR_STATE_KEY}): {ocr or '(empty — never checkpointed yet)'}")
    return {"detector": det, "ocr": ocr}


def show_heartbeats():
    det = r.get(DETECTOR_HEARTBEAT_KEY)
    ocr = r.get(OCR_HEARTBEAT_KEY)

    def _age(ts):
        return None if ts is None else time.time() - float(ts)

    det_age, ocr_age = _age(det), _age(ocr)
    print(f"detector heartbeat: {'no heartbeat (service down or TTL expired)' if det_age is None else f'{det_age:.1f}s ago'}")
    print(f"ocr heartbeat:      {'no heartbeat (service down or TTL expired)' if ocr_age is None else f'{ocr_age:.1f}s ago'}")
    return {"detector_age": det_age, "ocr_age": ocr_age}


def show_ocr_queue_depth():
    """How many OCR tasks are queued waiting for a free worker — the
    gauge worker.py/bus.py maintain around ocr:tasks. Non-zero and
    climbing means the OCR pool is under-provisioned for current
    camera/vehicle volume."""
    depth = r.get(OCR_TASKS_PENDING_KEY)
    llen = r.llen(OCR_TASKS_KEY)
    print(f"ocr:tasks pending gauge={depth or 0} | actual list length={llen}")
    return llen


def show_status():
    """Live state from both services' HTTP health endpoints."""
    import requests
    out = {}
    for name, base in (("detector", DETECTOR_API), ("ocr", OCR_API)):
        try:
            health = requests.get(f"{base}/health", timeout=3).json()
            print(f"{name}: {health}")
            out[name] = health
        except Exception as e:
            print(f"{name}: could not reach {base}: {e}")
            out[name] = None
    return out


# =====================================================================
# Scenarios — service restart (self-healing)
# =====================================================================

def test_service_restart(service="both", camera_id="1", address="", roi=None, hard_kill=False):
    """Proves "resume where it left off" — and, when restarting only ONE
    service, proves the OTHER one keeps running undisturbed the whole
    time.

        test_service_restart("detector")   # only plate_detector restarts
        test_service_restart("ocr")        # only plate_ocr restarts
        test_service_restart("both")       # both restart together

    Steps, with a manual pause in the middle:
      1. Make sure the camera is active.
      2. YOU restart the named service(s) in another terminal (the exact
         command is printed for you — pass hard_kill=True for `docker
         kill` instead of a graceful `restart`).
      3. Press Enter here. The restarted service(s) should come back at
         the SAME phase (processing) with the SAME engine/worker count,
         and the camera should be running again WITHOUT any new activate
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
    before_hb = show_heartbeats()
    show_status()

    cmd = _RESTART_CMDS[service]["hard" if hard_kill else "graceful"]
    print("\n" + "=" * 62)
    print(f"Now run this in another terminal to restart: {service}")
    print(f"  {cmd}")
    print("=" * 62)
    input("Press Enter once it/they are back up... ")

    print("\n--- after restart (no new activate command was sent) ---")
    show_active_cameras()
    show_self_healing_state()
    after_hb = show_heartbeats()
    show_status()

    untouched = {"detector": "ocr", "ocr": "detector"}.get(service)
    if untouched:
        before_age = before_hb.get(f"{untouched}_age")
        after_age = after_hb.get(f"{untouched}_age")
        if before_age is not None and after_age is not None:
            print(f"\n{untouched} heartbeat age went from {before_age:.1f}s to {after_age:.1f}s "
                  f"while you restarted {service} — this should look like ordinary elapsed "
                  f"time, NOT a reset back near 0s. A reset means {untouched} restarted too.")

    print("\nRecent results:")
    read_recent_results()


def test_connect_disconnect_reconnect(camera_id="1", address="", roi=None, total_seconds=300.0):
    """End-to-end test with a REAL camera. You do the physical
    unplugging/replugging by hand; this registers + activates it, then
    watches camera:events live so you can see camera_stream notice the
    drop and the reconnect, and plate_detector's offline-grace timer
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
# CLI (unchanged subcommands from the reference redis_control.py)
# =====================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("set-camera", help="write/update a camera's config in Redis")
    sc.add_argument("--id", required=True)
    sc.add_argument("--address", required=True, help="RTSP URL or video path")
    sc.add_argument("--title", default="Camera")
    sc.add_argument("--usage", default="plate")
    sc.add_argument("--roi", nargs=4, type=float, default=[0, 0, 1, 1], metavar=("X", "Y", "W", "H"))
    sc.add_argument("--line", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    sc.add_argument("--stop-roi", nargs=4, type=float, default=None, metavar=("X", "Y", "W", "H"))

    rc = sub.add_parser("remove", help="delete a camera's config")
    rc.add_argument("--id", required=True)

    sub.add_parser("list", help="list configured cameras + their ai_status")

    ac = sub.add_parser("activate", help="tell plate_detector to start processing this camera")
    ac.add_argument("--id", required=True)

    dc = sub.add_parser("deactivate", help="tell plate_detector to stop processing this camera")
    dc.add_argument("--id", required=True)

    st = sub.add_parser("status", help="read the current ai_status for a camera")
    st.add_argument("--id", required=True)

    sub.add_parser("self-heal-state", help="show both services' self-healing checkpoints")
    sub.add_parser("heartbeats", help="show both services' heartbeat ages")
    sub.add_parser("ocr-queue", help="show the ocr:tasks queue depth")
    sub.add_parser("health", help="hit both services' /health endpoints")

    args = p.parse_args()

    if args.cmd == "set-camera":
        set_camera(args.id, args.address, title=args.title, usage=args.usage,
                    roi=tuple(args.roi), line=tuple(args.line) if args.line else None,
                    stop_roi=tuple(args.stop_roi) if args.stop_roi else None)
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
    elif args.cmd == "ocr-queue":
        show_ocr_queue_depth()
    elif args.cmd == "health":
        show_status()


if __name__ == "__main__":
    main()
