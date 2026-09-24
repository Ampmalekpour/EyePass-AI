"""
redis_tools.py (heatmap module)
--------------------------------------------------------------------
Manual test harness / CLI. Runs OUTSIDE docker, from PyCharm, a REPL,
or `python redis_tools.py <subcommand>`, and stands in for the
backend: it writes cameras:config, sends activate/deactivate commands,
and watches what comes back. Same shape as the plate/face/fire
modules' own redis_tools.py.

It talks to the SAME Redis the containers use, over the published host
port. If you changed the published port or host, set the HOST
OVERRIDES below (or the matching env vars) before running.

Nothing here imports torch, cv2 or ultralytics, so this file starts in
under a second either way.

Requires: pip install redis requests

Two things changed from the pre-existing standalone build's own
redis_tools.py, both to match the plate/face/fire modules exactly:

  * `cameras:events` is now the SINGULAR `camera:events` channel (the
    bundled camera-service is now the same one those modules run).
  * activate/deactivate requests are LPUSHed by us and BRPOPped by the
    service (FIFO); the old build's service used BLPOP against the
    same LPUSH, which is LIFO — harmless at queue depth 0-1 but wrong
    under load. The response is now a LIST (RPUSH + 60s TTL, BRPOP by
    us) instead of a STRING (SET/GET) — see common/heatmapcore/keys.py
    for the full note on why.

set_camera() still publishes {module}:camera:config:updated after the
HSET, same as the pre-existing build did (and unlike the plate/fire
modules' own redis_tools.py, which currently don't — an existing,
separately-known gap over there, not something to copy here). Without
this publish, camera_stream only ever picks up a NEW camera on its own
periodic poll / next restart.

CLI usage:

  python redis_tools.py set-camera --id 1 --address "rtsp://admin:admin123@192.168.30.49:554/cam/realmonitor?channel=2&subtype=0" \\
      --title "Lobby 1" --roi 0 0 1 1

  python redis_tools.py activate --id 1
  python redis_tools.py status --id 1
  python redis_tools.py deactivate --id 1
  python redis_tools.py list
  python redis_tools.py remove --id 1

Library usage (import redis_tools as rt) adds: show_active(),
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
MODULE_KEY = os.getenv("REDIS_MODULE", "heatmap")

AI_API = os.getenv("AI_API", "http://localhost:8002")

# ---- backend contract (fixed spelling — see README "Redis contract") --
CFG_HASH = f"{MODULE_KEY}:cameras:config"
DETAILS_HASH = f"{MODULE_KEY}:cameras:details"
CMD_REQ_LIST = f"{MODULE_KEY}:cmd:ai:request"
RESULTS_KEY = f"{MODULE_KEY}:ai:results"
ACTIVE_KEY = f"{MODULE_KEY}:ai:active"
# Singular "camera" — matches this module's bundled camera-service
# (the same one the plate/face/fire modules run).
CAMERA_EVENTS_CHANNEL = f"{MODULE_KEY}:camera:events"
CONFIG_UPDATED_CHANNEL = f"{MODULE_KEY}:camera:config:updated"

# ---- internal (heatmapcore.keys — self-healing) ------------------------
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

_RESTART_CMDS = {
    "detector": {"graceful": "docker compose restart ai_service", "hard": "docker kill heatmap_ai_service"},
}


# =====================================================================
# Standing in for the BACKEND — config / activate / deactivate
# =====================================================================

def set_camera(camera_id, address, title="Camera", usage="ENTRY_EXIT",
                roi=(0.0, 0.0, 1.0, 1.0), stop_roi=None, cross_line=None):
    """roi: (x, y, w, h) normalized 0..1. stop_roi/cross_line are
    accepted and stored (matching cameras:config's documented shape)
    but unused by this module's own pipeline — they're here only in
    case a shared frontend/config editor expects every module's config
    hash to carry the same fields.

    Remember to URL-encode credentials: a password containing '@' must
    be written as '%40' or RTSP parsing breaks."""
    camera_id = str(camera_id)
    cfg = {
        "id": camera_id, "title": title, "address": address, "usage": usage,
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "stop_roi": stop_roi, "cross_line": cross_line,
    }
    r.hset(CFG_HASH, camera_id, json.dumps(cfg, ensure_ascii=False))
    r.publish(CONFIG_UPDATED_CHANNEL, json.dumps({"camera_id": camera_id}))
    print(f"[set-camera] wrote {CFG_HASH}[{camera_id}] and notified camera_stream")


def remove_camera_config(camera_id):
    camera_id = str(camera_id)
    r.hdel(CFG_HASH, camera_id)
    r.publish(CONFIG_UPDATED_CHANNEL, json.dumps({"camera_id": camera_id}))
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
    # ai_service's cmd worker does BRPOP (reads from the tail), so we
    # must LPUSH (write to the head) to preserve FIFO order.
    r.lpush(CMD_REQ_LIST, json.dumps(payload))
    print(f"[{action}] sent request_id={request_id} camera_id={camera_id}, waiting for response...")

    resp = r.blpop(cmd_resp_key(request_id), timeout=timeout)
    if resp is None:
        print(f"[{action}] TIMEOUT waiting for response (is ai_service running?)")
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
    if r.hexists(ACTIVE_KEY, str(camera_id)):
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
    """Peeks at the newest entries on ai:results without consuming them.
    The real backend should BRPOP (FIFO, since we LPUSH)."""
    items = r.lrange(RESULTS_KEY, -count, -1)
    if not items:
        print("(ai:results is empty — no cube has been flushed yet)")
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
    """Blocks on ai:results (BRPOP) and prints each cube-flush pointer as
    it's published."""
    print(f"Watching {RESULTS_KEY} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            item = r.brpop(RESULTS_KEY, timeout=1)
            if item:
                _, raw = item
                data = json.loads(raw)
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('camera_id')} "
                      f"event={data.get('event')} date={data.get('date')} key={data.get('object_key')}")
    except KeyboardInterrupt:
        pass


# =====================================================================
# Inspection — internal / self-healing
# =====================================================================

def show_active():
    """The ai_service's durable desired state: which cameras it believes
    it is responsible for. This is what survives a restart."""
    entries = r.hgetall(ACTIVE_KEY)
    if not entries:
        print(f"({ACTIVE_KEY} is empty — nothing is activated)")
        return {}
    out = {}
    for cid, raw in entries.items():
        data = json.loads(raw)
        out[cid] = data
        since = time.strftime("%H:%M:%S", time.localtime(data.get("activated_at", 0)))
        print(f"cam={cid} activated_at={since} roi={data.get('roi')}")
    return out


def show_self_healing_state():
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
    """Live state from the AI service's HTTP API."""
    import requests
    out = {}
    try:
        health = requests.get(f"{AI_API}/health", timeout=3).json()
        print(f"health: {health}")
        out["health"] = health
        active = requests.get(f"{AI_API}/active", timeout=3).json()
        print(f"activated but not running: {active['activated_but_not_running']}")
        print(f"running but not activated: {active['running_but_not_activated']}")
        out["active"] = active
        for cid, st in requests.get(f"{AI_API}/status", timeout=3).json().items():
            print(f"cam={cid} state={st['state']} frames={st['frames_processed']} last_error={st['last_error']}")
    except Exception as e:
        print(f"could not reach {AI_API}: {e}")
    return out


# =====================================================================
# Scenarios
# =====================================================================

def test_service_restart(camera_id="1", address="", roi=None, hard_kill=False):
    """Proves "resume where it left off" for ai_service."""
    if address:
        print(f"\n=== Ensuring camera {camera_id} is active ===")
        ensure_camera_active(camera_id, address, roi)

    print("\n--- before restart ---")
    show_active()
    show_self_healing_state()
    show_heartbeats()
    show_status()

    cmd = _RESTART_CMDS["detector"]["hard" if hard_kill else "graceful"]
    print("\n" + "=" * 62)
    print(f"Now run this in another terminal to restart: {cmd}")
    print("=" * 62)
    input("Press Enter once it's back up... ")

    print("\n--- after restart (no new activate command was sent) ---")
    show_active()
    show_self_healing_state()
    show_heartbeats()
    show_status()

    print("\nRecent results:")
    read_recent_results()


def test_connect_disconnect_reconnect(camera_id="1", address="", roi=None, total_seconds=300.0):
    """End-to-end test with a REAL camera. You do the physical
    unplugging/replugging by hand; this registers + activates it, then
    watches camera:events live."""
    print(f"\n=== Registering + activating camera {camera_id} ===")
    ensure_camera_active(camera_id, address, roi)

    print("\n--- current state ---")
    show_details(camera_id)
    show_active()
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
    engine, then deactivates enough of them to see the next batch
    consolidate. Addresses don't need to be real."""
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
    print(f"\n=== Deactivating {len(to_remove)} cameras ===")
    for cid in to_remove:
        deactivate(cid)

    print("\n--- topology after deactivation ---")
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
    sc.add_argument("--usage", default="ENTRY_EXIT")
    sc.add_argument("--roi", nargs=4, type=float, default=[0, 0, 1, 1], metavar=("X", "Y", "W", "H"))

    rc = sub.add_parser("remove", help="delete a camera's config")
    rc.add_argument("--id", required=True)

    sub.add_parser("list", help="list configured cameras + their ai_status")

    ac = sub.add_parser("activate", help="tell ai_service to start processing this camera")
    ac.add_argument("--id", required=True)

    dc = sub.add_parser("deactivate", help="tell ai_service to stop processing this camera")
    dc.add_argument("--id", required=True)

    st = sub.add_parser("status", help="read the current ai_status for a camera")
    st.add_argument("--id", required=True)

    sub.add_parser("self-heal-state", help="show the detector's self-healing checkpoint")
    sub.add_parser("heartbeats", help="show the detector's heartbeat age")
    sub.add_parser("health", help="hit the ai_service's /health endpoint")

    args = p.parse_args()

    if args.cmd == "set-camera":
        set_camera(args.id, args.address, title=args.title, usage=args.usage, roi=tuple(args.roi))
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
