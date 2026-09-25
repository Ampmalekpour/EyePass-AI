# Face module — detector + recognizer

Two independently deployable, self-healing services that replace the
original single-process face pipeline (`alpr_api.py` / `face_service.py`
+ `video_processor.py` + `fr_worker.py`). Same detection/tracking/
trigger/recognition **behaviour** as the reference code — the math,
thresholds, alignment, decision logic and BYTETrack implementation are
ported near-verbatim — but reorganized into:

```
face_detector      camera frames -> YOLO detect -> BYTETrack -> spatial
                    triggers -> dispatches recognition tasks over Redis
face_recognizer     pulls recognition tasks -> AdaFace embed + compare
                    against the gallery -> pushes results back over Redis
```

Nothing between them is in-process anymore. The old `fr_input_queue` /
`fr_output_queue` (`multiprocessing.Queue`, one pair per detector
engine, feeding a pool of `AFRWorker` subprocesses spawned *by* that
engine) is now two Redis queues (`facecore/keys.py`,
`facecore/codec.py`): a shared work queue every detector engine pushes
onto, and a per-engine result queue every recognizer worker replies on.
Detection capacity and recognition capacity now scale independently.

Communication with the backend is **entirely** the Redis contract below
— neither service makes an HTTP call to Django or anywhere else.


## Control hub (third service — owns recognition state)

The detector no longer merges recognizer results, holds triggers or builds
the final record. A third service, **`control_hub`** (source in
[`../control-hub`](../control-hub/README.md), started by this folder's
`compose.yaml`), owns every track's state:

```
face_detector  --track events-->  control_hub  --> face:ai:results
       |  ^                              ^
 tasks |  | ctl (ack / satisfied /       | results (per task, keyed by the
       v  |  periodic request)           | track's global uid)
face_recognizer  ------------------------+
```

- The detector reports `track_started`, `trigger` (line_cross / stopped_roi, including
  the crossing **direction**), `submitted`, `track_update` and
  `track_ended`, then forgets the track.
- The recognizer sends each result to the hub, not back to the engine.
- The hub votes across results, decides when the track is
  **satisfied** (and tells the detector to stop sending crops),
  publishes each trigger once, and publishes the `finalize` record
  after the last in-flight result arrives.

The backend key and payload shape are unchanged; new fields are
additive. The inconsistencies this fixed are listed in
[`../control-hub/README.md`](../control-hub/README.md#what-changed-and-why).


## Contents

```
common/facecore/       shared library, copied into both Docker images
detector/               face_detector service (src/, Dockerfile, requirements.txt)
recognizer/             face_recognizer service (src/, Dockerfile, requirements.txt)
compose.yaml            mediamtx + face_detector + face_recognizer
compose.infra.yaml      redis + minio (+ inspection tools, --profile tools)
.env / .env.example     every variable the stack understands
redis_tools.py          manual test harness — stands in for the backend
tests/                  unit tests (facecore, engine_manager rebalance, triggers)
README.md               this file
```


## What you need to provide

Nothing in this delivery trains or ships model weights — you bind-mount
what you already have, matching the `.env.example` variables:

| What | Where it goes | `.env` variable |
|---|---|---|
| `best1.pt` (YOLO detection) | `${DETECTION_MODELS_DIR}/best1.pt` | `DETECTION_MODELS_DIR` |
| `adaface_ir50_cpu.onnx` (primary recognition backend) | `${RECOGNITION_MODELS_DIR}/adaface_ir50_cpu.onnx` | `RECOGNITION_MODELS_DIR` |
| `pretrained/` folder (`adaface_ir50_ms1mv2.ckpt` fallback + `warmup.jpg`) | `${RECOGNITION_PRETRAINED_DIR}/` | `RECOGNITION_PRETRAINED_DIR` |
| `face_alignment/` package (the `align` module + `mtcnn_pytorch`'s `warp_and_crop_face`) | `${FACE_ALIGNMENT_DIR}/` | `FACE_ALIGNMENT_DIR` |
| your `base_image:latest` (32.2GB, already has torch/ultralytics/OpenCV) | Docker build arg | `AI_BASE_IMAGE` |

The recognizer downloads its **gallery** (person images + the
`brieface.db` SQLite DB) from MinIO on startup — see "Storage split"
below — so nothing gallery-related needs to be bind-mounted; seed MinIO
once (`GALLERY_MINIO_PREFIX`, `GALLERY_DB_MINIO_KEY`) and every worker
picks it up.


## Run it

```bash
docker network create eyeplate_net        # once, ever
cp .env.example .env                      # then edit paths/ports as needed

docker compose -f compose.infra.yaml up -d              # redis + minio
docker compose -f compose.infra.yaml --profile tools up -d   # + RedisInsight/Redis Commander

docker compose up -d --build              # mediamtx + face_detector + face_recognizer
```

If the platform already runs a shared Redis/MinIO/MediaMTX, skip
`compose.infra.yaml` (and the `mediamtx` block in `compose.yaml`) and
point `.env` at the existing ones instead — see the comments in both
compose files.

Both services start idle automatically, then self-heal to whatever
phase they were last in (see "Self-healing" below) — you don't need to
send any command after a fresh `docker compose up` for the recognizer;
the detector waits for the backend's first `activated` command per
camera, exactly like the reference pipeline.


## Redis contract

### Backend contract (fixed spelling — `REDIS_MODULE=face`)

| Key | Type | Direction |
|---|---|---|
| `face:cameras:config` | HASH `camera_id -> json` | backend → camera_service |
| `face:cameras:details` | HASH `camera_id -> json` | camera_service → backend, detector |
| `face:cameras:events` | PUB/SUB | camera_service → backend, detector |
| `face:cmd:ai:request` | LIST (BRPOP) | backend → detector |
| `face:cmd:ai:response:{request_id}` | LIST, 60s TTL | detector → backend |
| `face:ai:results` | LIST | detector → backend |

The detector is the **only** consumer of `cmd:ai:request` — that queue
is single-consumer (BRPOP) and entirely about camera
activate/deactivate, which is the detector's concern; the recognizer
has no per-camera concept (see `recognizer/src/main.py`'s docstring for
the full reasoning). `ai:results` carries every mid-track and final
recognition update, in the same shape the reference `AFRWorker` used to
hand to `save_prep_for_mysql` — the backend reads `personnelid`,
`first_name`, `last_name`, `face_image` / `camera_image` (MinIO object
keys), `detection_score`, etc.

### Internal (`face:internal:...` — nothing outside these two services reads these)

| Key | Purpose |
|---|---|
| `internal:active_cameras` | durable desired-state ledger (written *before* acting, so a crash mid-activation is recovered by startup reconcile) |
| `internal:detector:state` | detector's self-healing checkpoint: `{phase, engine_count, updated_at}` |
| `internal:recognizer:state` | recognizer's self-healing checkpoint: `{phase, worker_count, updated_at}` |
| `internal:detector:heartbeat`, `internal:recognizer:heartbeat` | liveness (30s TTL, refreshed every 10s) |
| `internal:rec:tasks` | LIST — shared work queue, every detector engine LPUSHes, every recognizer worker BRPOPs |
| `internal:rec:results:{engine_id}` | LIST, one per detector engine — a result routes back to the exact `Engine` instance that owns the track |
| `internal:rec:tasks:pending` | informational counter (watch it in RedisInsight/Commander) |
| `internal:hub:events` | STREAM — detector engines → control hub (track lifecycle, triggers, submissions) |
| `internal:hub:results` | STREAM — recognizer workers → control hub (one entry per task, keyed by track uid) |
| `internal:hub:ctl:{engine_id}` | LIST — control hub → one detector engine (result ack, satisfied flag, periodic request) |
| `internal:hub:track:{uid}` | control hub's per-track checkpoint (restored on restart) |
| `internal:hub:leader` / `internal:hub:heartbeat` | one active hub per module / liveness |

`redis_tools.py` has helpers for every row in both tables
(`seed_camera_config`, `send_activate`/`send_deactivate`,
`show_active_cameras`, `show_self_healing_state`,
`show_rec_queue_depth`, `watch_results`, ...) plus two end-to-end
scenarios (`run_restart_resilience_test`,
`run_engine_consolidation_test`).


## Self-healing

Both services share the same four-method lifecycle
(`facecore/lifecycle.py`, `ServiceLifecycle`):

```
start_idle(n)     load + warm up n units (engines / workers), do NOT
                  process yet. Persists phase=idle, {unit}_count=n.
stop_idle()       release everything start_idle loaded. Implies
                  stop_process() first if currently processing.
                  Persists phase=stopped.
start_process()   begin real work with whatever is already loaded and
                  warm (attach cameras / start pulling rec:tasks).
                  Persists phase=processing.
stop_process()    stop real work, keep everything warm. Persists
                  phase=idle.
```

Every transition — and every time the unit count changes **outside**
an explicit transition (an engine spun up because camera count grew,
or `rebalance()` consolidated engines, or the recognizer's watchdog
respawned a crashed worker) — is checkpointed to
`internal:{service}:state` via `checkpoint_now()`.

On boot, `self_heal()` reads that checkpoint and replays it:
`start_idle(persisted_count)`, then `start_process()` too if the
persisted phase was `processing`. This is true self-healing, not just
"restart and hope": kill `-9` a container mid-run, bring it back up,
and it returns to the exact same phase with the exact same number of
engines/workers, without the backend resending anything. Prove it with
`redis_tools.py`'s `run_restart_resilience_test`.

The recognizer additionally self-heals at the **process** level: a
background watchdog (`RecognizerPool`, `WATCHDOG_INTERVAL_SEC`) detects
a crashed worker subprocess and respawns it to keep the pool at its
expected size, without waiting for the whole container to restart.


## Camera offline handling

`backend_bridge.py`'s offline-grace timer (ported from the reference
`_schedule_offline_stop`/`_handle_offline_expired`) waits
`CAMERA_OFFLINE_GRACE_SECONDS` after a `cameras:events` "offline"
transition before tearing the camera's engine assignment down —
absorbing a camera that is merely mid-reconnect. An "online" event
within the grace window cancels the pending teardown; when it does
have to run, `remove_camera()` is called on the durable ledger.


## Engine consolidation (new — not in the reference pipeline)

The reference `EngineManager` only ever **grows**: a new engine
subprocess is spawned once the busiest one hits
`MAX_CAMERAS_PER_ENGINE`. It never shrinks, so camera churn (a handful
of deactivations here and there) fragments cameras across more engines
than necessary — the request's own example: a 5-per-engine cap, 10
cameras that end up spread 3 + 2 (or worse) across engines that could
hold them all in 2.

`EngineManager.rebalance()` (`detector/src/engine_manager.py`, ticks
every `ENGINE_REBALANCE_INTERVAL_SEC`) detects when the current camera
count would fit into fewer engines, drains the least-loaded engines by
replaying each camera's cached `add_camera()` config
(`_camera_last_config`, populated on every `add_camera` call) onto a
fuller surviving engine, and stops the emptied engines once they are
actually empty. A migrated camera's BYTETracker restarts (track IDs
reset) — the RTSP relay keeps the camera's actual upstream connection
alive throughout, so this is a bookkeeping reset, not a stream
interruption. See `tests/test_engine_manager_rebalance.py` for the
exact 10-camera/5-per-engine scenario, verified.


## Storage split

| | Where | Why |
|---|---|---|
| Gallery images + `brieface.db`, recognized/unrecognized face crops & camera frames sent to the backend | **MinIO** (`facecore/minio_store.py`) | pipeline data — durable, shared, what the backend resolves into URLs |
| Landmarked-crop debug images, aligned-face dumps, detector debug videos | **local bind-mounted volume** (`detector_data` / `recognizer_data`, `DEBUG_*` env vars) | debugging only — never uploaded, never read by anything outside the container that wrote it |

The recognizer is the only service that touches MinIO at all — the
detector never does (see `detector/requirements.txt`).


## Camera frames — always via the MediaMTX relay

The detector never opens an RTSP connection to a camera directly; it
always goes through the relay, via `MTX_RTSP_BASE_URL` and
`backend_bridge.rtsp_url()`:

```python
def rtsp_url(camera_id: str) -> str:
    return f"{config.MTX_RTSP_BASE_URL.rstrip('/')}/{camera_id}"
```

exactly mirroring the heatmap module's own `_rtsp_url()` pattern.


## Testing

This delivery was verified with `python3 -m py_compile` across every
file (clean) plus a real unit test suite
(`tests/`, `python3 tests/run_all.py`) covering:

- `facecore.codec` — task/result pickle round-trips, the `DateTimeEncoder`.
- `facecore.lifecycle` — every transition, `checkpoint_now()`, and
  (most importantly) a full `self_heal()` round trip: start_idle(4) +
  start_process(), simulate a restart with a fresh `ServiceLifecycle`
  against the same Redis, confirm it comes back at `engine_count=4`,
  `phase=processing`, with no external command.
- `facecore.active_state` — the durable desired-state ledger.
- `detector.engine_manager.rebalance()` — the exact 10-camera,
  5-per-engine consolidation scenario from the request, plus edge
  cases (already-tight, single-engine, a camera with no cached config
  is never silently dropped).
- `detector.triggers` — line-crossing (age/confidence gates, cooldown,
  independent per-track state), ROI entry/exit (confirmation frames,
  fires once), stopped-in-ROI (velocity/duration gates, fires once).

This sandbox has no network access to install `redis`, `fakeredis`,
`torch`, `ultralytics` or `boto3` (`pypi.org` returns HTTP 403 through
the sandbox's proxy), so `tests/fakes/` ships a minimal in-memory
`FakeRedis` (just the subset of the redis-py API `facecore/bus.py`
actually calls) plus import-time stubs for `torch`/`ultralytics`/
`cython_bbox`/`lap`/`boto3` — enough to import and exercise the real
logic in `facecore`, `engine_manager` and `triggers` without a GPU, a
real Redis server, or those packages installed. cv2/numpy/scipy *are*
available here and are used for real (not stubbed) in the trigger
tests. None of this stubbing ships in the Docker images — it is
`tests/`-only, and the Dockerfiles install the real `redis`/`boto3`/
`onnxruntime` packages and build from your base image, which already
has torch/ultralytics/OpenCV.

`tests/test_engine_hub.py` covers the engine's side of the control-hub
contract on the real Engine methods: crop submission and its gates
(in flight / satisfied / unchanged crop / weak landmarks), hub ctl handling, the
track-end finalize pass, and ending live tracks on camera removal. The
hub itself has its own suite (`../control-hub/tests`), including an
end-to-end run against a real `redis-server`.

Run it yourself:
```bash
python3 tests/run_all.py
```


## Reference material this was built from

The detection/tracking pipeline (`Engine`, `EngineManager`,
`RTSPStreamReader`, spatial triggers, best-crop ranking) comes from
`video_processor.py`; BYTETrack from `eyerik_face_tracker.py`; the
recognition engine (ONNX/PyTorch dual backend, gallery `.pkl` caching,
alignment, the Tsallis-entropy decision function, track aggregation)
from `fr_worker.py`; the AdaFace backbone from `net.py`; image loading
helpers from `image_utils.py`; MinIO helpers generalized from
`minio_uploader.py`. The self-healing/durable-desired-state pattern,
`RedisBus`, `_rtsp_url()`, offline-grace timer and the two-compose-file
structure were ported from the heatmap module's own `api.py` /
`compose.yaml` / `compose.infra.yaml` / `redis_tools.py`, which were
supplied as the reference for how this platform's modules are built.
