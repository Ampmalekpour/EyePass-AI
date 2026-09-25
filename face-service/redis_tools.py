"""
redis_tools.py (face module)
--------------------------------------------------------------------
Manual test harness. Runs OUTSIDE docker, from PyCharm or a REPL, and
stands in for the backend: it writes cameras:config, sends activate /
deactivate commands, and watches what comes back.

It talks to the SAME Redis the containers use, over the published host
port (compose.infra.yaml's REDIS_PUBLISH_PORT). If you changed the
published port or host, set the HOST OVERRIDES below before running.

Nothing here imports torch/cv2/ultralytics/onnxruntime, so this file
starts in under a second either way.

Requires: pip install redis requests

CAMERA_1_ID / CAMERA_1_ADDRESS below are your real, physical camera —
the one at 192.168.30.206 you can walk over to and unplug/replug. Both
are module-level constants (not tucked inside `if __name__ == "__main__"`)
so they're available the moment you `import redis_tools as rt`, e.g.
`rt.CAMERA_1_ID`.
--------------------------------------------------------------------
"""

import json
import os
import time
import uuid

# ---------------------------------------------------------------------
# HOST OVERRIDES — set before anything else touches Redis.
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_MODULE = os.getenv("REDIS_MODULE", "face")

DETECTOR_API = os.getenv("DETECTOR_API", "http://localhost:8010")
RECOGNIZER_API = os.getenv("RECOGNIZER_API", "http://localhost:8011")

# ---------------------------------------------------------------------
# YOUR TEST CAMERA — the one you can physically unplug/replug.
# ---------------------------------------------------------------------
CAMERA_1_ID = "1"
CAMERA_1_ADDRESS = "rtsp://admin:admin110@192.168.30.206:554/cam/realmonitor?channel=1"

import redis  # noqa: E402

r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
    password=REDIS_PASSWORD, decode_responses=True,
)

# ---- backend contract (fixed spelling — see README "Redis contract") --
REQUEST_KEY = f"{REDIS_MODULE}:cmd:ai:request"
RESPONSE_KEY_PREFIX = f"{REDIS_MODULE}:cmd:ai:response"
RESULTS_KEY = f"{REDIS_MODULE}:ai:results"
CAMERAS_CONFIG_KEY = f"{REDIS_MODULE}:cameras:config"
CAMERAS_DETAILS_KEY = f"{REDIS_MODULE}:cameras:details"
CAMERAS_EVENTS_CHANNEL = f"{REDIS_MODULE}:cameras:events"
# Singular "camera" — matches camera_stream's psubscribe pattern. Not a
# typo, do not "fix" it (see the heatmap module's own redis_tools.py).
CONFIG_UPDATED_CHANNEL = f"{REDIS_MODULE}:camera:config:updated"

# ---- internal (facecore.keys — self-healing / detector<->recognizer) --
ACTIVE_CAMERAS_KEY = f"{REDIS_MODULE}:internal:active_cameras"
DETECTOR_STATE_KEY = f"{REDIS_MODULE}:internal:detector:state"
RECOGNIZER_STATE_KEY = f"{REDIS_MODULE}:internal:recognizer:state"
DETECTOR_HEARTBEAT_KEY = f"{REDIS_MODULE}:internal:detector:heartbeat"
RECOGNIZER_HEARTBEAT_KEY = f"{REDIS_MODULE}:internal:recognizer:heartbeat"
REC_TASKS_KEY = f"{REDIS_MODULE}:internal:rec:tasks"
REC_TASKS_PENDING_KEY = f"{REDIS_MODULE}:internal:rec:tasks:pending"

# Docker restart commands per service — printed by test_service_restart(),
# kept in one place so they can't drift out of sync with each other.
_RESTART_CMDS = {
    "detector": {
        "graceful": "docker compose restart face_detector",
        "hard": "docker kill face_detector",
    },
    "recognizer": {
        "graceful": "docker compose restart face_recognizer",
        "hard": "docker kill face_recognizer",
    },
    "both": {
        "graceful": "docker compose restart face_detector face_recognizer",
        "hard": "docker kill face_detector face_recognizer",
    },
}


# =====================================================================
# Standing in for the BACKEND
# =====================================================================

def seed_camera_config(camera_id, address, roi=None, title="", usage="ENTRY_EXIT",
                        stop_roi=None, cross_line=None):
    """
    Writes a camera into cameras:config and pings camera_stream to pick it
    up immediately. From here on camera_stream owns cameras:details and
    cameras:events, not this script.

    roi / stop_roi:    {"x":.., "y":.., "w":.., "h":..}   (normalized 0..1
                        or pixel — whatever camera_stream/backend agree on)
    cross_line:         {"start": {"x":.., "y":..}, "end": {"x":.., "y":..}}

    Remember to URL-encode credentials: a password containing '@' must be
    written as '%40' or RTSP parsing breaks.
    """
    roi = roi or {"x": 0, "y": 0, "w": 1, "h": 1}
    payload = {
        "id": str(camera_id),
        "title": title or f"Camera {camera_id}",
        "address": address,
        "usage": usage,
        "roi": roi,
        "stop_roi": stop_roi,
        "cross_line": cross_line,
    }
    r.hset(CAMERAS_CONFIG_KEY, str(camera_id), json.dumps(payload, ensure_ascii=False))
    r.publish(CONFIG_UPDATED_CHANNEL, json.dumps({"camera_id": str(camera_id)}))
    print(f"Seeded {CAMERAS_CONFIG_KEY}[{camera_id}] and notified camera_stream")


def remove_camera_config(camera_id):
    """Deregisters a camera entirely. camera_stream drops its MediaMTX path
    and deletes its cameras:details entry on the next sync."""
    r.hdel(CAMERAS_CONFIG_KEY, str(camera_id))
    r.publish(CONFIG_UPDATED_CHANNEL, json.dumps({"camera_id": str(camera_id)}))
    print(f"Removed {CAMERAS_CONFIG_KEY}[{camera_id}]")


def send_activate(camera_id) -> str:
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "camera_id": int(camera_id), "action": "activated"}
    r.lpush(REQUEST_KEY, json.dumps(payload))
    print(f"→ activate camera {camera_id} | request_id={request_id}")
    return request_id


def send_deactivate(camera_id) -> str:
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "camera_id": int(camera_id), "action": "deactivated"}
    r.lpush(REQUEST_KEY, json.dumps(payload))
    print(f"→ deactivate camera {camera_id} | request_id={request_id}")
    return request_id


def read_response(request_id, wait_seconds=10.0):
    """Polls for the detector's response to a request_id (cmd_worker_loop
    in backend_bridge.py)."""
    key = f"{RESPONSE_KEY_PREFIX}:{request_id}"
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        items = r.lrange(key, 0, 0)
        if items:
            print(f"← {key} = {items[0]}")
            return json.loads(items[0])
        time.sleep(0.2)
    print(f"← no response at {key} after {wait_seconds:.0f}s "
          f"(face_detector not running, or the request was dropped)")
    return None


def ensure_camera_active(camera_id=CAMERA_1_ID, address=CAMERA_1_ADDRESS, roi=None):
    """Idempotent: seeds + activates camera_id only if it isn't already in
    active_cameras. Every scenario below calls this first instead of
    blindly re-activating, so you can run them back-to-back without
    duplicate activate commands piling up."""
    if r.hexists(ACTIVE_CAMERAS_KEY, str(camera_id)):
        print(f"camera {camera_id} already active — skipping seed/activate")
        return
    seed_camera_config(camera_id=camera_id, address=address, roi=roi)
    time.sleep(1)
    read_response(send_activate(camera_id))
    time.sleep(5)


# =====================================================================
# Inspection — backend contract
# =====================================================================

def show_details(camera_id=None):
    """What camera_stream currently believes about each camera."""
    if camera_id is not None:
        raw = r.hget(CAMERAS_DETAILS_KEY, str(camera_id))
        entries = {str(camera_id): raw} if raw else {}
    else:
        entries = r.hgetall(CAMERAS_DETAILS_KEY)

    if not entries:
        print("(cameras:details is empty — is camera_stream running?)")
        return {}

    out = {}
    for cid, raw in entries.items():
        data = json.loads(raw)
        out[cid] = data
        print(f"cam={cid} connected={data.get('connected')} "
              f"error={data.get('error')} updated={data.get('updated')} "
              f"stream_url={data.get('stream_url')}")
    return out


def read_recent_results(count=10):
    """Peeks at the newest entries on ai:results without consuming them.
    The real backend should BRPOP (FIFO, since the detector RPUSHes)."""
    items = r.lrange(RESULTS_KEY, -count, -1)
    if not items:
        print("(ai:results is empty — nothing has been recognized yet)")
    for raw in items:
        print(json.loads(raw))
    return [json.loads(i) for i in items]


def watch_camera_events(seconds=60.0):
    """Prints every cameras:events transition as it arrives — this is
    what fires when camera_stream notices the physical camera drop off
    or come back (connect/disconnect/reconnect)."""
    pubsub = r.pubsub()
    pubsub.subscribe(CAMERAS_EVENTS_CHANNEL)
    print(f"Watching {CAMERAS_EVENTS_CHANNEL} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            message = pubsub.get_message(timeout=1.0)
            if message and message.get("type") == "message":
                data = json.loads(message["data"])
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('id')} "
                      f"connected={data.get('connected')} error={data.get('error')}")
    except KeyboardInterrupt:
        pass
    finally:
        pubsub.close()


def watch_results(seconds=60.0):
    """Blocks on ai:results (BRPOP) and prints each recognition/finalize
    event as the control hub publishes it — the fastest way to see the full
    detector -> recognizer -> control hub -> ai:results round trip live."""
    print(f"Watching {RESULTS_KEY} for {seconds:.0f}s (Ctrl+C to stop)...")
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            item = r.brpop(RESULTS_KEY, timeout=1)
            if item:
                _, raw = item
                data = json.loads(raw)
                meta = data.get("meta", {})
                print(f"[{time.strftime('%H:%M:%S')}] cam={data.get('camera_id')} "
                      f"track={data.get('track_id')} event={data.get('event_type')} "
                      f"final={data.get('is_final')} person={meta.get('identified_as')} "
                      f"conf={meta.get('confidence')} via={meta.get('resolution')} "
                      f"live={meta.get('liveness')}")
    except KeyboardInterrupt:
        pass


# =====================================================================
# Inspection — internal / self-healing
# =====================================================================

def show_active_cameras():
    """The detector's durable desired state: which cameras it believes it
    is responsible for. This is what survives a restart."""
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
    processing) and how many engines/workers were running. This is what
    self_heal() reads on boot to restore itself without waiting for any
    command to be resent."""
    det = r.hgetall(DETECTOR_STATE_KEY)
    rec = r.hgetall(RECOGNIZER_STATE_KEY)
    print(f"detector   ({DETECTOR_STATE_KEY}): {det or '(empty — never checkpointed yet)'}")
    print(f"recognizer ({RECOGNIZER_STATE_KEY}): {rec or '(empty — never checkpointed yet)'}")
    return {"detector": det, "recognizer": rec}


def show_heartbeats():
    det = r.get(DETECTOR_HEARTBEAT_KEY)
    rec = r.get(RECOGNIZER_HEARTBEAT_KEY)

    def _age(ts):
        if ts is None:
            return None
        return time.time() - float(ts)

    det_age, rec_age = _age(det), _age(rec)
    print(f"detector heartbeat:   {'no heartbeat (service down or TTL expired)' if det_age is None else f'{det_age:.1f}s ago'}")
    print(f"recognizer heartbeat: {'no heartbeat (service down or TTL expired)' if rec_age is None else f'{rec_age:.1f}s ago'}")
    return {"detector_age": det_age, "recognizer_age": rec_age}


def show_rec_queue_depth():
    """How many recognition tasks are queued waiting for a free
    recognizer worker — the gauge worker.py/bus.py maintain around
    rec:tasks. Non-zero and climbing means the recognizer pool is
    under-provisioned for current camera/face volume."""
    depth = r.get(REC_TASKS_PENDING_KEY)
    llen = r.llen(REC_TASKS_KEY)
    print(f"rec:tasks pending gauge={depth or 0} | actual list length={llen}")
    return llen


def show_status():
    """Live state from both services' HTTP health endpoints."""
    import requests
    out = {}
    for name, base in (("detector", DETECTOR_API), ("recognizer", RECOGNIZER_API)):
        try:
            health = requests.get(f"{base}/health", timeout=3).json()
            print(f"{name}: {health}")
            out[name] = health
        except Exception as e:
            print(f"{name}: could not reach {base}: {e}")
            out[name] = None
    return out


# =====================================================================
# Scenarios — connection / disconnection / reconnection
# =====================================================================

def test_connect_disconnect_reconnect(camera_id=CAMERA_1_ID, address=CAMERA_1_ADDRESS,
                                       roi=None, total_seconds=300.0):
    """
    End-to-end test with the REAL camera. You do the physical
    unplugging/replugging by hand; this registers + activates it, then
    watches cameras:events live so you can see camera_stream notice the
    drop and the reconnect.

    Expect roughly camera_stream's own hold time + its offline-grace
    window between an actual unplug and the event flipping connected to
    false, and a near-immediate flip back on replug.

    Watch alongside, in another terminal:
        docker compose logs -f camera_stream face_detector face_recognizer
    """
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


# =====================================================================
# Scenarios — service restart (self-healing)
# =====================================================================

def test_service_restart(service="both", camera_id=CAMERA_1_ID, address=CAMERA_1_ADDRESS,
                          roi=None, hard_kill=False):
    """
    Proves "resume where it left off" — and, when restarting only ONE
    service, proves the OTHER one keeps running undisturbed the whole
    time. This is the test for "what happens if either of the services
    is restarted", run three ways:

        test_service_restart("detector")     # only face_detector restarts
        test_service_restart("recognizer")   # only face_recognizer restarts
        test_service_restart("both")         # both restart together

    Steps, with a manual pause in the middle:
      1. Make sure the camera is active.
      2. YOU restart the named service(s) in another terminal (the exact
         command is printed for you — pass hard_kill=True to get the
         `docker kill` version instead of a graceful `restart`).
      3. Press Enter here. The restarted service(s) should come back at
         the SAME phase (processing) with the SAME engine/worker count,
         and the camera should be running again WITHOUT any new activate
         command — self_heal() replaying its checkpoint on boot. The
         service you did NOT restart should show an heartbeat age that
         kept climbing smoothly through the whole test (proof it was
         never touched), not one that reset back near 0s.
    """
    if service not in _RESTART_CMDS:
        raise ValueError(f"service must be one of {list(_RESTART_CMDS)}, got {service!r}")

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

    untouched = {"detector": "recognizer", "recognizer": "detector"}.get(service)
    if untouched:
        before_age = before_hb.get(f"{untouched}_age")
        after_age = after_hb.get(f"{untouched}_age")
        if before_age is not None and after_age is not None:
            print(f"\n{untouched} heartbeat age went from {before_age:.1f}s to {after_age:.1f}s "
                  f"while you restarted {service} — this should look like ordinary elapsed "
                  f"time, NOT a reset back near 0s. A reset means {untouched} restarted too, "
                  f"which it should not have.")

    print(f"\nThe camera should be RUNNING above, and the restarted service's checkpoint")
    print(f"should show phase=processing with the same engine/worker count as before.")

    print("\nRecent results:")
    read_recent_results()


# =====================================================================
# Scenarios — performance
# =====================================================================

def measure_performance(camera_id=CAMERA_1_ID, address=CAMERA_1_ADDRESS, roi=None,
                         seconds=60.0, poll_interval=5.0):
    """
    Samples throughput while the camera runs, on a fixed interval:
      - detector-side FPS for this camera (frames_processed delta from
        /health's camera_status, the same counter engine.py increments
        per processed frame)
      - rec:tasks backlog (0 and flat = recognizer keeping up; climbing
        = under-provisioned for this camera/face volume)
      - ai:results growth (results appended per interval — recognition
        throughput actually reaching the output queue)
      - both heartbeats, as a liveness sanity check throughout

    Prints a per-tick line, then a summary. Doesn't consume anything off
    ai:results (uses LLEN, not BRPOP), so it's safe to run alongside a
    real backend or another watch_results() call.
    """
    print(f"\n=== Ensuring camera {camera_id} is active ===")
    ensure_camera_active(camera_id, address, roi)

    import requests

    samples = []
    prev_frames = None
    prev_results_len = r.llen(RESULTS_KEY)

    print(f"\nSampling every {poll_interval:.0f}s for {seconds:.0f}s...")
    print(f"{'t':>5s}  {'fps':>6s}  {'rec_backlog':>11s}  {'results/interval':>17s}  "
          f"{'det_hb':>7s}  {'rec_hb':>7s}")

    ticks = max(1, int(seconds // poll_interval))
    t0 = time.time()
    for i in range(ticks):
        time.sleep(poll_interval)
        elapsed = time.time() - t0

        frames = None
        try:
            health = requests.get(f"{DETECTOR_API}/health", timeout=3).json()
            frames = health.get("camera_status", {}).get(str(camera_id), {}).get("frames_processed")
        except Exception:
            pass

        fps = None
        if frames is not None and prev_frames is not None:
            fps = (frames - prev_frames) / poll_interval
        if frames is not None:
            prev_frames = frames

        backlog = r.llen(REC_TASKS_KEY)

        results_len = r.llen(RESULTS_KEY)
        new_results = results_len - prev_results_len
        prev_results_len = results_len

        det_hb = r.get(DETECTOR_HEARTBEAT_KEY)
        rec_hb = r.get(RECOGNIZER_HEARTBEAT_KEY)
        det_age = f"{time.time() - float(det_hb):.0f}s" if det_hb else "down"
        rec_age = f"{time.time() - float(rec_hb):.0f}s" if rec_hb else "down"

        fps_str = f"{fps:.2f}" if fps is not None else "n/a"
        print(f"{elapsed:5.0f}  {fps_str:>6s}  {backlog:11d}  {new_results:17d}  "
              f"{det_hb and det_age:>7s}  {rec_hb and rec_age:>7s}")

        samples.append({"t": elapsed, "fps": fps, "backlog": backlog, "new_results": new_results})

    fps_values = [s["fps"] for s in samples if s["fps"] is not None]
    backlog_values = [s["backlog"] for s in samples]
    total_results = sum(s["new_results"] for s in samples)

    print("\n--- summary ---")
    if fps_values:
        print(f"detector fps  min={min(fps_values):.2f}  max={max(fps_values):.2f}  "
              f"avg={sum(fps_values)/len(fps_values):.2f}")
    else:
        print("detector fps: no samples (camera not connected, or /health unreachable)")
    print(f"rec:tasks backlog  min={min(backlog_values)}  max={max(backlog_values)}  "
          f"(sustained growth = recognizer under-provisioned for this load)")
    print(f"ai:results growth over the whole window: {total_results} "
          f"({total_results/seconds:.2f}/s average)")

    return samples


# =====================================================================
# Scenarios — engine consolidation (unrelated to a single real camera)
# =====================================================================

def run_engine_consolidation_test(camera_ids, addresses):
    """
    Demonstrates engine consolidation: activates more cameras than
    MAX_CAMERAS_PER_ENGINE allows in one engine (spreading them across
    several engines as they're added one at a time), then deactivates
    enough of them that everything remaining COULD fit in fewer engines
    — and watches EngineManager.rebalance() migrate them back down on
    its next tick (ENGINE_REBALANCE_INTERVAL_SEC).

    camera_ids / addresses must be the same length. Unlike the other
    scenarios here, these don't need to be real/reachable cameras — this
    is testing engine topology math, not actual detection — but expect
    RTSP connection failures in the logs for any address that isn't
    real.
    """
    assert len(camera_ids) == len(addresses)

    print(f"\n=== Activating {len(camera_ids)} cameras one at a time ===")
    for cid, addr in zip(camera_ids, addresses):
        seed_camera_config(camera_id=cid, address=addr)
        time.sleep(0.5)
        read_response(send_activate(cid))
        time.sleep(0.5)

    print("\n--- topology after spreading across engines ---")
    show_status()

    half = len(camera_ids) // 2
    to_remove = camera_ids[half:]
    print(f"\n=== Deactivating {len(to_remove)} cameras to trigger consolidation ===")
    for cid in to_remove:
        read_response(send_deactivate(cid))

    print("\nWaiting for the next rebalance tick...")
    time.sleep(35)

    print("\n--- topology after rebalance (should be fewer engines) ---")
    show_status()


if __name__ == "__main__":
    # Pick one:

    # test_service_restart("detector")
    # test_service_restart("recognizer")
    # test_service_restart("both")

    # test_connect_disconnect_reconnect(total_seconds=300.0)

    # measure_performance(seconds=60.0, poll_interval=5.0)

    test_service_restart("both")