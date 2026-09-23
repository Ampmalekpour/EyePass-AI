# Add-Face — integration notes

Everything in this bundle is additive to `face_service_/` from the
existing repo. No package (`requirements.txt`), Dockerfile, or
`docker-compose` change is needed — see "Why nothing new to build"
below.

## Where every file goes

```
face_service_/
  common/facecore/
    keys.py          REPLACE — adds enroll_request/response, gallery_updated, gallery_lock
    bus.py            REPLACE — adds enroll request/response + gallery lock/pubsub helpers
  recognizer/src/
    config.py          REPLACE — adds section 11 (ADD-FACE) at the bottom, everything above unchanged
    pose.py             NEW — yaw/pitch/pose-approval math + MTCNN-based detection
    gallery.py          NEW — brieface.db range allocation (persisted counter, delete-safe)
    worker.py           REPLACE — adds enroll_pose_check / enroll_commit task handling
    pool.py             REPLACE — adds reload_gallery()
    main.py             REPLACE — starts EnrollCoordinator, subscribes to gallery:updated
    enroll.py           NEW — EnrollCoordinator (the backend-facing Redis listener)
  tests/
    test_pose.py         NEW — pure pose-math unit tests
    test_gallery.py       NEW — pure range-allocation unit tests
enroll_tools.py           NEW — manual test harness, sibling to redis_tools.py, repo root
```

Nothing in `detector/` changes. Nothing in `camera-service/` changes.
`compose.yaml`, `compose.infra.yaml`, `.env.example` need **zero**
structural changes — only new *optional* env vars, all with working
defaults baked into `config.py` (see below), the same convention every
existing tunable already follows.

## Why nothing new to build

- No new pip dependency: `redis`, `boto3`, `onnxruntime` are already in
  `recognizer/requirements.txt`; `torch`, `opencv-python`, `numpy`,
  `Pillow`, `scikit-learn` already come from the base image.
- No new model weights: enrollment's face detector is the SAME MTCNN
  instance `face_alignment.align` already constructs at import time
  (`align.py`'s module-level `mtcnn_model`) — already bind-mounted via
  `FACE_ALIGNMENT_DIR`. Nothing new to place in `models/`.
- No new container, no new port, no new compose service. Enrollment
  runs as a background thread inside the existing `face_recognizer`
  container (`enroll.py::EnrollCoordinator`, started from `main.py`)
  and as two new task types handled by the existing `RecognitionWorker`
  subprocess pool (`worker.py`).

## New environment variables (all optional — defaults are in `config.py`)

```bash
# ---- add-face (all optional; shown with their defaults) -----------------
ENROLL_SNAPSHOT_RETRIES=3
ENROLL_SNAPSHOT_TIMEOUT_SEC=5.0

ENROLL_MIN_FACE_SIZE=20.0
ENROLL_MTCNN_THR_P=0.6
ENROLL_MTCNN_THR_R=0.7
ENROLL_MTCNN_THR_O=0.9
ENROLL_MTCNN_NMS_P=0.7
ENROLL_MTCNN_NMS_R=0.7
ENROLL_MTCNN_NMS_O=0.7
ENROLL_MTCNN_FACTOR=0.85

ENROLL_PITCH_MIN=-35.0
ENROLL_PITCH_MAX=80.0
ENROLL_PITCH_SCALING_FACTOR=50.0
ENROLL_YAW_FLAG1_MIN=30.0    # flag 1 = right profile
ENROLL_YAW_FLAG1_MAX=75.0
ENROLL_YAW_FLAG2_MIN=-30.0   # flag 2 = frontal
ENROLL_YAW_FLAG2_MAX=30.0
ENROLL_YAW_FLAG3_MIN=-75.0   # flag 3 = left profile
ENROLL_YAW_FLAG3_MAX=-30.0

ENROLL_CROP_PADDING_RATIO=0.15
ENROLL_IMAGES_PER_PERSON=3

ENROLL_LOCK_TIMEOUT_SEC=30.0
ENROLL_LOCK_BLOCKING_TIMEOUT_SEC=15.0
ENROLL_TASK_TIMEOUT_SEC=20.0
ENROLL_COMMIT_TASK_TIMEOUT_SEC=60.0
```

One EXISTING variable gets a second reader: `MTX_RTSP_BASE_URL`
(already read by `detector/src/config.py`) is now also read by
`recognizer/src/config.py`, for the camera-snapshot path in
`enroll.py::_grab_camera_snapshot`. Same value, same relay — nothing
to change in `.env` itself.

## The Redis contract this adds

```
face:cmd:enroll:request                LIST (BRPOP)   backend -> recognizer
face:cmd:enroll:response:{request_id}  LIST, 60s TTL  recognizer -> backend
face:internal:gallery:updated          PUB/SUB        recognizer -> recognizer (all replicas)
face:internal:gallery:lock             STRING (lock)  internal only
```

Two request shapes (pickled dicts — see `common/facecore/codec.py`,
same envelope as the internal task queues, NOT JSON like
`cmd:ai:request`):

```python
# 1) verify one pose-angle shot — either a raw upload or a camera grab
{"request_id": "...", "action": "verify_pose", "flag": 1|2|3,
 "image_bytes": b"..."}                      # OR:
{"request_id": "...", "action": "verify_pose", "flag": 1|2|3,
 "camera_id": "7"}

# response:
{"status": "ok", "approved": True,  "yaw": 42.1, "pitch": 3.4, "crop_bytes": b"..."}
{"status": "ok", "approved": False, "reason": "pose_mismatch", "yaw": ..., "pitch": ...}
{"status": "ok", "approved": False, "reason": "no_face_detected"}
{"status": "error", "message": "..."}

# 2) commit — after all ENROLL_IMAGES_PER_PERSON crops are approved
{"request_id": "...", "action": "commit",
 "person": {"name": "...", "lastname": "...", "section": "...",
            "codeid": "...", "personnelid": "..."},
 "crops": [b"...", b"...", b"..."]}

# response:
{"status": "ok", "personnelid": "2001", "range_start": 4, "range_end": 7,
 "filenames": ["c4.jpg", "c5.jpg", "c6.jpg"]}
{"status": "error", "message": "..."}
```

## Testing and verification

### 1. Pure logic — no Redis, no Docker, no models

```bash
cd face_service_
PYTHONPATH=recognizer/src python3 tests/test_pose.py     # 10 tests — yaw/pitch/pose-window math
PYTHONPATH=recognizer/src python3 tests/test_gallery.py  # 6 tests — range allocation, incl. the
                                                            #  delete-doesn't-roll-back-the-counter case
python3 tests/run_all.py                                  # everything, including the pre-existing suite
```
Both new files were written and run against this exact sandbox while
building this feature (no `redis`/`torch`/`face_alignment` needed for
either — `pose.py`'s detector call and `worker.py`'s task handlers are
intentionally not exercised here; that needs a real stack, next).

### 2. Against a running stack — `enroll_tools.py`

Same idea as `redis_tools.py`: stands in for the backend, run from
your own machine against the published Redis port.

```bash
pip install redis
python3
>>> import enroll_tools as et
>>> et.show_gallery_lock_state()

# one angle at a time
>>> r1 = et.verify_pose_from_file("test_photos/frontal.jpg", flag=2)
>>> r1["approved"], r1.get("reason")
>>> et.save_crop(r1["crop_bytes"], "frontal_ok.jpg")   # open it, look at it

# or straight from a live camera already registered with the module
>>> r2 = et.verify_pose_from_camera(camera_id="1", flag=1)
```

Deliberately-bad inputs worth trying by hand before trusting the
happy path:
- A photo with no face → expect `{"approved": False, "reason": "no_face_detected"}`.
- A frontal photo sent with `flag=1` (expects a right-profile angle) →
  expect `{"approved": False, "reason": "pose_mismatch", ...}` — check
  the returned `yaw`/`pitch` against `ENROLL_YAW_WINDOWS`/`ENROLL_PITCH_WINDOW`
  to confirm the numbers make sense for that photo.
- `camera_id` for a camera that isn't registered/reachable → expect a
  `{"status": "error", ...}` naming the RTSP failure, within
  `ENROLL_SNAPSHOT_TIMEOUT_SEC * ENROLL_SNAPSHOT_RETRIES` seconds, not
  a hang.

Once you have three approved crops:

```python
>>> person = {"name": "Sara", "lastname": "Ahmadi", "section": "HR",
...           "codeid": "0099", "personnelid": "2001"}
>>> result = et.commit([crop_frontal, crop_right, crop_left], person)
>>> result   # {"status": "ok", "range_start": ..., "range_end": ..., ...}
```

Or run the whole thing scripted, against three already-posed test
photos:

```python
>>> et.run_full_enrollment_demo("frontal.jpg", "right.jpg", "left.jpg", person)
```

### 3. Confirming propagation (the part that's easy to get wrong silently)

```python
# in one terminal, before committing:
>>> et.watch_gallery_updated(seconds=30)
# in another terminal, run the commit — the first terminal should
# print the gallery:updated event within a second or two of the
# commit response coming back.
```

Then, functionally:
1. Commit a new person.
2. Immediately send a recognition probe crop of THAT person through
   the normal live pipeline (or `find-top-match`-style ad hoc check)
   against a recognizer worker that did **not** handle the commit
   task. It should already resolve to the new personnelid — confirms
   `pool.reload_gallery()` actually ran and every worker re-downloaded
   the gallery.
3. `docker compose restart face_recognizer` and repeat step 2 after
   the health check goes green — confirms the enrolled person survives
   a full container restart (they're in MinIO now, not just in a
   worker's local temp dir).

### 4. Concurrency check

Fire two `commit()` calls back-to-back for two different people
(e.g. from two REPL tabs, near-simultaneously). Expect:
- One completes normally.
- The other either queues behind the lock and completes shortly after
  (`ENROLL_LOCK_BLOCKING_TIMEOUT_SEC` gives it up to 15s to acquire),
  or returns `{"status": "error", "message": "gallery is busy..."}` if
  it couldn't get the lock in time.
- Both people end up with **non-overlapping** ranges afterward — check
  `range_start`/`range_end` in both results directly; this is exactly
  the scenario `test_gallery.py`'s
  `test_two_sequential_enrollments_never_collide` covers at the unit
  level, this step is the same claim under real concurrency.

### 5. What's intentionally NOT covered here

- `pose.py::detect_face_5pt()` / `verify_pose()` end-to-end (needs the
  real `face_alignment` weights — covered by step 2 above instead, not
  a unit test).
- `worker.py`'s enroll handlers end-to-end (needs Redis + a loaded
  model + MinIO — same reason, covered by step 2/3 above).
- Deleting an enrolled person — no delete flow was requested; `gallery.py`
  is written so a future delete doesn't corrupt allocation (see
  `test_allocation_survives_deleting_the_highest_numbered_row`), but
  nothing here removes a person's images from MinIO or their row from
  `brieface.db`. Worth a follow-up if "remove a face" is coming next.
