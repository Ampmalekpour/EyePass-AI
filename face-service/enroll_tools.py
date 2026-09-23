"""
enroll_tools.py (face module) — NEW, add-face
--------------------------------------------------------------------
Manual test harness for enrollment. Sibling to redis_tools.py, same
conventions: runs OUTSIDE docker, from PyCharm or a REPL, stands in
for the backend, talks to the SAME Redis the containers use over the
published host port.

Unlike redis_tools.py, this file DOES import pickle (to speak the same
binary envelope facecore.codec.encode_task/decode_task uses — see that
module's docstring for why cmd:enroll:* is bytes, not JSON, unlike
cmd:ai:*) — still no torch/cv2/ultralytics/onnxruntime, so it stays
fast to import and safe to run from a plain laptop venv.

Requires: pip install redis

--------------------------------------------------------------------
Typical session (see run_full_enrollment_demo() for the scripted
version of exactly this):

    import enroll_tools as et

    # one flag at a time — front (2), then right (1), then left (3)
    approved = {}
    for flag in (2, 1, 3):
        result = et.verify_pose_from_file("photos/frontal.jpg", flag=flag)
        print(result["status"], result.get("approved"), result.get("reason"))
        if result.get("approved"):
            approved[flag] = result["crop_bytes"]
            et.save_crop(result["crop_bytes"], f"approved_flag{flag}.jpg")

    # once all three are in hand, commit in a fixed flag order
    person = {"name": "Sara", "lastname": "Ahmadi", "section": "HR",
              "codeid": "0099", "personnelid": "2001"}
    commit_result = et.commit([approved[2], approved[1], approved[3]], person)
    print(commit_result)

    # or, capturing from a live camera instead of local files:
    result = et.verify_pose_from_camera(camera_id="1", flag=2)
--------------------------------------------------------------------
"""

import os
import pickle
import time
import uuid

# ---------------------------------------------------------------------
# HOST OVERRIDES — set before anything else touches Redis. Matches
# redis_tools.py's own variables so both files can be imported together
# against the same stack without repeating yourself.
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_MODULE = os.getenv("REDIS_MODULE", "face")

import redis  # noqa: E402

r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
    password=REDIS_PASSWORD, decode_responses=False,  # bytes client — see codec note above
)

ENROLL_REQUEST_KEY = f"{REDIS_MODULE}:cmd:enroll:request"
ENROLL_RESPONSE_KEY_PREFIX = f"{REDIS_MODULE}:cmd:enroll:response"
GALLERY_UPDATED_CHANNEL = f"{REDIS_MODULE}:internal:gallery:updated"
GALLERY_LOCK_KEY = f"{REDIS_MODULE}:internal:gallery:lock"


# =====================================================================
# Wire format — must match common/facecore/codec.py exactly
# =====================================================================
def _encode(payload: dict) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _decode(data: bytes) -> dict:
    return pickle.loads(data)


# =====================================================================
# Low-level request/response
# =====================================================================
def _send(action: str, wait_seconds: float, **fields) -> dict:
    request_id = str(uuid.uuid4())
    payload = {"request_id": request_id, "action": action, **fields}
    r.lpush(ENROLL_REQUEST_KEY, _encode(payload))
    print(f"→ enroll {action} | request_id={request_id}")

    response_key = f"{ENROLL_RESPONSE_KEY_PREFIX}:{request_id}"
    item = r.brpop(response_key, timeout=wait_seconds)
    if item is None:
        return {"status": "error", "message": f"no response within {wait_seconds}s (request_id={request_id})"}
    _, raw = item
    return _decode(raw)


# =====================================================================
# Public API — verify_pose
# =====================================================================
def verify_pose_from_file(image_path: str, flag: int, wait_seconds: float = 20.0) -> dict:
    """flag: 1=right profile, 2=frontal, 3=left profile (see
    recognizer/src/config.py::ENROLL_YAW_WINDOWS for the exact angle
    windows each one checks against)."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return _send("verify_pose", wait_seconds, flag=flag, image_bytes=image_bytes)


def verify_pose_from_camera(camera_id: str, flag: int, wait_seconds: float = 25.0) -> dict:
    """No image_bytes sent — the recognizer's EnrollCoordinator grabs
    one frame itself, straight off the MediaMTX relay for this
    camera_id. wait_seconds is a bit longer than the file-upload
    variant to leave room for the RTSP connect+read."""
    return _send("verify_pose", wait_seconds, flag=flag, camera_id=str(camera_id))


def save_crop(crop_bytes: bytes, out_path: str):
    with open(out_path, "wb") as f:
        f.write(crop_bytes)
    print(f"saved {out_path} ({len(crop_bytes)} bytes)")


# =====================================================================
# Public API — commit
# =====================================================================
def commit(crops: list, person: dict, wait_seconds: float = 60.0) -> dict:
    """crops: exactly ENROLL_IMAGES_PER_PERSON (default 3) approved crop
    byte-strings, in a fixed, consistent flag order you choose yourself
    — the order doesn't have semantic meaning to the recognizer, it
    just becomes c<range_start>.jpg, c<range_start+1>.jpg, ... in the
    order given. person: {"name","lastname","section","codeid","personnelid"}.
    """
    return _send("commit", wait_seconds, person=person, crops=crops)


# =====================================================================
# Inspection helpers
# =====================================================================
def show_gallery_lock_state():
    ttl = r.pttl(GALLERY_LOCK_KEY)
    if ttl is None or ttl < 0:
        print("gallery lock: free")
    else:
        print(f"gallery lock: HELD, expires in {ttl} ms")


def watch_gallery_updated(seconds: float = 30.0):
    """Blocks, printing every gallery:updated event for `seconds` —
    useful to confirm a commit() actually triggered the broadcast that
    makes sibling workers reload."""
    pubsub = r.pubsub()
    pubsub.subscribe(GALLERY_UPDATED_CHANNEL)
    print(f"watching {GALLERY_UPDATED_CHANNEL} for {seconds}s...")
    deadline = time.time() + seconds
    for message in pubsub.listen():
        if time.time() > deadline:
            break
        if message.get("type") != "message":
            continue
        print("gallery:updated ->", message.get("data"))


# =====================================================================
# Scripted end-to-end demo
# =====================================================================
def run_full_enrollment_demo(photo_frontal: str, photo_right: str, photo_left: str, person: dict):
    """Runs the whole flow against three local test photos already
    posed roughly front/right/left, and prints each step. Returns the
    commit result dict. Does NOT retry a rejected pose — if any of the
    three photos fails its pose check, this stops and tells you which
    one and why, exactly like an operator UI would need to.
    """
    flag_photo = {2: photo_frontal, 1: photo_right, 3: photo_left}
    approved_crops = {}

    for flag, path in flag_photo.items():
        print(f"\n--- flag {flag} ({path}) ---")
        result = verify_pose_from_file(path, flag=flag)
        print(result if "crop_bytes" not in result else {**result, "crop_bytes": f"<{len(result['crop_bytes'])} bytes>"})
        if result.get("status") != "ok" or not result.get("approved"):
            print(f"STOPPING: flag {flag} was not approved.")
            return result
        approved_crops[flag] = result["crop_bytes"]
        save_crop(result["crop_bytes"], f"enroll_demo_flag{flag}_approved.jpg")

    print("\n--- committing ---")
    ordered_crops = [approved_crops[2], approved_crops[1], approved_crops[3]]
    commit_result = commit(ordered_crops, person)
    print(commit_result)
    return commit_result


if __name__ == "__main__":
    print(f"enroll_tools ready | redis={REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} module={REDIS_MODULE}")
    print(f"  ENROLL_REQUEST_KEY = {ENROLL_REQUEST_KEY}")
    show_gallery_lock_state()
