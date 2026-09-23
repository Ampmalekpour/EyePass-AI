"""
test_add_face_demo.py
--------------------------------------------------------------------
Manual, PyCharm-runnable demo of the whole add-face flow, end to end,
against your three test photos. Sibling to redis_tools.py /
enroll_tools.py, same conventions — no torch/cv2 needed on THIS side,
just `pip install redis`.

Prereqs:
    pip install redis
    docker compose -f compose.infra.yaml up -d
    docker compose up -d --build
    (give face_recognizer a few seconds to finish loading — it logs
    "Loading Face Recognition models..." then "ready"/"idle" once warm)

What this does:
    1. Loads left.jpg / middle.jpg / right.jpg from PHOTOS_DIR below.
    2. Sends each through verify_pose with the matching flag
       (middle=frontal=2, right=1, left=3).
    3. Saves each APPROVED crop next to the originals as
       <name>_approved.jpg, so you can open them and see exactly what
       got embedded (not just take my word for it).
    4. If a photo gets rejected, this stops right there and tells you
       which one, why, and what to do about it.
    5. Once all three are approved, commits the new person and prints
       the result: personnelid, allocated c<N>.jpg range, filenames.
    6. Prints exactly what to check afterward and where.

Run it: right-click -> Run 'test_add_face_demo', or just run the file.
--------------------------------------------------------------------
"""

import os
import sys

# ---------------------------------------------------------------------
# HOST OVERRIDES — same convention as redis_tools.py / enroll_tools.py.
# Defaults match `docker compose -f compose.infra.yaml up -d`'s
# published ports. Change these if your .env published different ones
# (REDIS_PUBLISH_PORT / MINIO_API_PUBLISH_PORT).
# ---------------------------------------------------------------------
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_MODULE", "face")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enroll_tools as et  # noqa: E402

# ---------------------------------------------------------------------
# Your three pose photos. Any of these extensions is tried in order —
# it just needs to find ONE of them per name.
# ---------------------------------------------------------------------
PHOTOS_DIR = r"C:\Users\eyerik.com\Desktop\dan"
EXTENSIONS = (".jpg", ".jpeg", ".png")

# file-name -> pose flag. See recognizer/src/config.py::ENROLL_YAW_WINDOWS
# for the exact angle window each flag checks against:
#   flag 2 = frontal        <- your "middle" photo
#   flag 1 = right profile  <- your "right" photo
#   flag 3 = left profile   <- your "left" photo
POSE_FILES = {2: "middle", 1: "right", 3: "left"}

# ---------------------------------------------------------------------
# The person this run enrolls. If you re-run this script against the
# SAME gallery without changing personnelid, you'll get a second
# c<N>.jpg block for the same personnelid — harmless, just confusing
# to look at in brieface.db. Bump it (or the whole dict) between runs.
# ---------------------------------------------------------------------
PERSON = {
    "name": "Dan",
    "lastname": "Test",
    "section": "QA",
    "codeid": "0001",
    "personnelid": "9001",
}


def _resolve(name: str) -> str:
    for ext in EXTENSIONS:
        path = os.path.join(PHOTOS_DIR, name + ext)
        if os.path.isfile(path):
            return path
    tried = [name + e for e in EXTENSIONS]
    raise FileNotFoundError(f"none of {tried} found in {PHOTOS_DIR}")


def main():
    print(f"redis   -> {et.REDIS_HOST}:{et.REDIS_PORT}  module={et.REDIS_MODULE}")
    et.show_gallery_lock_state()

    approved_crops = {}
    for flag, name in POSE_FILES.items():
        path = _resolve(name)
        print(f"\n--- {name} (flag={flag}) -> {path} ---")

        result = et.verify_pose_from_file(path, flag=flag)

        if result.get("status") != "ok":
            print("ERROR:", result.get("message"))
            print("\nMost likely cause: face_recognizer isn't up yet, or Redis/host "
                  "settings above don't match your .env's published ports.")
            sys.exit(1)

        approved = result.get("approved")
        print(f"approved={approved}  yaw={result.get('yaw')}  pitch={result.get('pitch')}", end="")
        print(f"  reason={result.get('reason')}" if not approved else "")

        if not approved:
            print(f"\nSTOPPING: '{name}' was rejected.")
            print(f"  -> retake that shot closer to the flag-{flag} angle, or loosen "
                  f"ENROLL_YAW_FLAG{flag}_MIN/MAX / ENROLL_PITCH_MIN/MAX in .env and "
                  f"restart face_recognizer.")
            sys.exit(1)

        approved_path = os.path.join(PHOTOS_DIR, f"{name}_approved.jpg")
        et.save_crop(result["crop_bytes"], approved_path)
        approved_crops[flag] = result["crop_bytes"]

    print("\n--- all three approved, committing ---")
    ordered_crops = [approved_crops[2], approved_crops[1], approved_crops[3]]  # middle, right, left
    commit_result = et.commit(ordered_crops, PERSON)
    print(commit_result)

    if commit_result.get("status") != "ok":
        print("\nCOMMIT FAILED — see message above (most likely the gallery lock "
              "timed out, or brieface.db couldn't be downloaded from MinIO).")
        sys.exit(1)

    rs, re_ = commit_result["range_start"], commit_result["range_end"]
    print(f"\nDone. personnelid={PERSON['personnelid']} now owns "
          f"c{rs}..c{re_ - 1}.jpg -> {commit_result['filenames']}")

    print("\nGo verify it landed:")
    print("  1. MinIO console: http://localhost:9001  (user/pass = MINIO_ROOT_USER/PASSWORD in .env)")
    print("     bucket 'face-private-bucket' (or your MINIO_PRIVATE_BUCKET) -> "
          "dynamics/CTDBUR/  (or your GALLERY_MINIO_PREFIX)")
    print(f"     you should see {commit_result['filenames']} as new objects there, plus an")
    print("     updated brieface.db and representations_ir_50.pkl (newer mtime than before).")
    print("  2. brieface.db directly: download it from that same MinIO path and open it")
    print(f"     (e.g. DB Browser for SQLite) -> SELECT * FROM brieface WHERE personnelid='{PERSON['personnelid']}';")
    print("  3. Propagation to other workers (if RECOGNITION_WORKER_COUNT > 1):")
    print("     >>> et.watch_gallery_updated(seconds=15)   # in a second REPL, BEFORE re-running this script")
    print("     you should see a gallery:updated event print within a second or two of the commit above.")
    print("  4. Recognition: send a live/probe crop of the same photo through the normal")
    print(f"     pipeline (or another verify_pose call) and confirm it now resolves to "
          f"personnelid={PERSON['personnelid']}.")


if __name__ == "__main__":
    main()
