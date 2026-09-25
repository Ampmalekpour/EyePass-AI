# Plate module — detector + OCR

> **EyePass suite (`develop-suite` branch).** This module runs as part
> of the suite: start it from the repository root (`docker compose up -d`
> with `plate` in `COMPOSE_PROFILES`). Redis, MinIO, the MediaMTX relay and
> the camera manager (`camera-service/`) are shared and defined in the
> root `compose.yaml`. This module no longer ships its own copies or a
> `compose.infra.yaml`. Frames are read from the relay path
> `cam_<sha1(camera address)>`. Parts of the text below describe the
> standalone layout. See the root `README.md` for the suite setup.

Two independently deployable, self-healing services that replace the
original single-process ALPR pipeline (`alpr_api.py` + `video_processor.py`
+ `ocr_worker.py`). Same detection/tracking/trigger/OCR **behaviour**
as the reference code — the math, thresholds, BYTETrack implementation,
quality gates and plate-format validation/voting are ported near-
verbatim — but reorganized into:

```
plate_detector    camera frames -> YOLO detect -> BYTETrack -> spatial
                  triggers (line-cross / stop-ROI / leave-scene) ->
                  dispatches OCR tasks over Redis
plate_ocr         pulls OCR tasks -> car/motorcycle PaddleOCR pipeline
                  -> vote + validate -> uploads crops to MinIO ->
                  pushes results back over Redis
```

Nothing between them is in-process anymore. The old `ocr_input_queue` /
`ocr_output_queue` (`multiprocessing.Queue`, one pair per detector
engine, feeding a pool of 3 `OCRWorker` subprocesses spawned *by* that
engine) is now two Redis queues (`common/platecore/keys.py`,
`common/platecore/codec.py`): a shared work queue every detector engine
pushes onto, and a per-engine result queue every OCR worker replies on.
Detection capacity and OCR capacity now scale independently — you no
longer get exactly 3 OCR workers per detector engine.

Communication with the backend is **entirely** the same Redis contract
your existing deployment already uses — neither service makes an HTTP
call to Django, and the key names/shapes that Django and
`eyepass-camera-stream` already depend on are **unchanged** (see
"Redis contract" below — this matters, it's the whole reason this port
is safe to deploy in place).


## Control hub (third service — owns recognition state)

The detector no longer merges OCR results, holds triggers or builds
the final record. A third service, **`control_hub`** (source in
[`./control-hub`](control-hub/README.md), part of this module and
started by this folder's `compose.yaml`), owns every track's state:

```
plate_detector  --track events-->  control_hub  --> plate:vehicle:results
       |  ^                              ^
 tasks |  | ctl (ack / satisfied /       | results (per task, keyed by the
       v  |  periodic request)           | track's global uid)
plate_ocr  ------------------------------+
```

- The detector reports `track_started`, `trigger` (cross_line / stop_roi, including
  the crossing **direction**), `submitted`, `track_update` and
  `track_ended`, then forgets the track.
- The OCR workers send each result to the hub, not back to the engine.
- The hub votes across results, decides when the track is
  **satisfied** (and tells the detector to stop sending crops),
  publishes each trigger once, and publishes the `leave_scene` record
  after the last in-flight result arrives.

The backend key and payload shape are unchanged; new fields are
additive. The inconsistencies this fixed are listed in
[`control-hub/README.md`](control-hub/README.md).


## Contents

```
common/platecore/      shared library, copied into both Docker images
detector/               plate_detector service (src/, Dockerfile, requirements.txt)
ocr_service/            plate_ocr service (src/, Dockerfile, requirements.txt)
compose.yaml            plate_detector + plate_ocr
compose.infra.yaml      redis + minio (+ inspection tools, --profile tools)
.env.example            every variable the stack understands
redis_tools.py          manual test harness / CLI — stands in for the backend
tests/                  unit tests (platecore, engine_manager rebalance, triggers, backend_bridge)
README.md               this file
```


## What you need to provide

Nothing in this delivery trains or ships model weights — you bind-mount
what you already have, matching the `.env.example` variables. **This is
the exact same `best.pt` and `PadOcr/` folder your current deployment
already uses** — nothing about the model files themselves changes.

| What | Where it goes | `.env` variable |
|---|---|---|
| `best.pt` (YOLO detection) | `${DETECTION_MODELS_DIR}/best.pt` | `DETECTION_MODELS_DIR` (default `./models/detection`) |
| Your whole `PadOcr/` folder, **with these exact subfolder/file names kept** — see below | `${OCR_MODELS_DIR}/` | `OCR_MODELS_DIR` (default `./models/ocr`) |
| your GPU base image (already has torch/ultralytics/OpenCV) | Docker build arg | `AI_BASE_IMAGE` (default `base_image_gpu:latest`, matching the existing `plate/Dockerfile`) |

The `OCR_MODELS_DIR` folder must contain, **exactly as named** (these
are read by `ocr_service/src/config.py`, which builds every PaddleOCR
model path from `OCR_MODELS_DIR` + these fixed names — same names the
reference `ocr_worker.py` hardcoded as `./PadOcr/...`):

```
${OCR_MODELS_DIR}/
├── en_PP-OCRv3_det_infer/       (shared detector model, car + motorcycle)
├── rec_svrt_fa_final_1/         (car plate recognizer)
├── ch_ppocr_mobile_v2.0_cls_infer/   (shared angle classifier)
├── rec_svrt_motor/              (motorcycle plate recognizer)
└── Final_Dict.txt               (shared character dictionary)
```

If your existing deployment has a `PadOcr/` folder next to
`ocr_worker.py` today, **point `OCR_MODELS_DIR` straight at it** — no
renaming, no copying, it's a bind mount.

Both are plain bind mounts (`compose.yaml`'s `volumes:`), never baked
into the image, so swapping either model needs no rebuild — just
overwrite the file/folder and restart the container.


## Run it

```bash
docker network create eyeplate_net        # once, ever — skip if it already exists
cp .env.example .env                      # then edit paths/ports as needed

# Only if the platform doesn't already run a shared Redis/MinIO —
# it almost certainly does (eyepass-redis / eyepass-minio):
docker compose -f compose.infra.yaml up -d
docker compose -f compose.infra.yaml --profile tools up -d   # + RedisInsight/Redis Commander

docker compose up -d --build              # plate_detector + plate_ocr
```

This module does **not** bring its own MediaMTX/camera_stream — your
existing `eyepass-camera-stream` deployment keeps running exactly as it
is today (that's what the singular `plate:camera:events` channel name
is — it matches your real, already-running container; see "Redis
contract" below). Point `MTX_RTSP_BASE_URL` and `REDIS_*` in `.env` at
your existing relay/Redis.

Both services start idle automatically, then self-heal to whatever
phase they were last in (see "Self-healing" below). After a *fresh*
`docker compose up` with an empty Redis, both come up idle and wait —
`plate_detector` for the backend's first `activated` command per
camera (exactly like the reference pipeline), `plate_ocr` camera-
agnostically for the first task. If you're migrating an
**already-running** deployment (cameras already marked active in the
existing `plate:cameras:config`), see "Migrating from the existing
single-process deployment" below.


## Redis contract

### Backend contract (fixed spelling — `REDIS_MODULE=plate`, unchanged from the existing deployment)

| Key | Type | Direction |
|---|---|---|
| `plate:cameras:config` | HASH `camera_id -> json` | backend → camera_stream, detector |
| `plate:cameras:details` | HASH `camera_id -> json` | camera_stream → backend, detector |
| `plate:camera:events` (**singular** "camera") | PUB/SUB | camera_stream → detector |
| `plate:cmd:ai:request` | LIST (BRPOP) | backend → detector |
| `plate:cmd:ai:response:{request_id}` | LIST, 60s TTL | detector → backend |
| `plate:cameras:{camera_id}:ai_status` | HASH, field=`camera_id` -> json | detector → backend (Django) |
| `plate:vehicle:results` | LIST | detector + OCR service → backend |

`plate:camera:events` is deliberately **singular** — your real,
already-deployed `eyepass-camera-stream` publishes to that exact
channel (verified against its own `src/main.py`). Do **not** "fix"
this to the plural `cameras:events` spelling you may see in other
modules' own camera-stream copies — that would silently stop this
detector from ever hearing a camera online/offline event. See
`common/platecore/keys.py`'s docstring.

`ai_status` (`current`: `"running"` / `"stopped"` / `"stopped_by_user"`)
is written at the exact same transition points the reference
`alpr_api.py` used — Django's existing read side needs no changes.
`vehicle:results` carries every mid-track and final OCR update, in the
same shape the reference `OCRWorker` used to hand to
`save_prep_for_mysql` — the backend reads `plate_result`, `plate_type`,
`plate_image` / `frame_image` (MinIO object keys), etc.

The detector is the **only** consumer of `cmd:ai:request` — that queue
is single-consumer (BRPOP) and entirely about camera
activate/deactivate, which is the detector's concern; the OCR service
has no per-camera concept (see `ocr_service/src/main.py`'s docstring).

### Internal (`plate:internal:...` — brand new, nothing outside these two services reads these)

| Key | Purpose |
|---|---|
| `internal:active_cameras` | durable desired-state ledger (written *before* acting, so a crash mid-activation is recovered by startup reconcile) — **new**, the reference pipeline had no equivalent |
| `internal:detector:state` | detector's self-healing checkpoint: `{phase, engine_count, updated_at}` |
| `internal:ocr:state` | OCR service's self-healing checkpoint: `{phase, worker_count, updated_at}` |
| `internal:detector:heartbeat`, `internal:ocr:heartbeat` | liveness (30s TTL, refreshed every 10s) |
| `internal:demand:events` | PUB/SUB `{"processing": bool}` — detector tells the OCR service whether any camera is active |
| `internal:ocr:tasks` | LIST — shared work queue, every detector engine LPUSHes, every OCR worker BRPOPs |
| `internal:ocr:results:{engine_id}` | LIST, one per detector engine — a result routes back to the exact `Engine` instance that owns the track |
| `internal:ocr:tasks:pending` | informational counter (watch it in RedisInsight/Commander, or `redis_tools.py`'s `show_ocr_queue_depth()`) |
| `internal:hub:events` | STREAM — detector engines → control hub (track lifecycle, triggers, submissions) |
| `internal:hub:results` | STREAM — OCR workers → control hub (one entry per task, keyed by track uid) |
| `internal:hub:ctl:{engine_id}` | LIST — control hub → one detector engine (result ack, satisfied flag, periodic request) |
| `internal:hub:track:{uid}` | control hub's per-track checkpoint (restored on restart) |
| `internal:hub:leader` / `internal:hub:heartbeat` | one active hub per module / liveness |

`redis_tools.py` has helpers for every row in both tables
(`set_camera`, `activate`/`deactivate`, `show_active_cameras`,
`show_self_healing_state`, `show_ocr_queue_depth`, `watch_results`,
...) plus scenarios (`test_service_restart`,
`test_connect_disconnect_reconnect`, `run_engine_consolidation_test`)
and the original CLI subcommands (`set-camera`, `activate`,
`deactivate`, `status`, `list`, `remove`).


## Self-healing

Both services share the same four-method lifecycle
(`common/platecore/lifecycle.py`, `ServiceLifecycle`) — entirely new,
the reference `alpr_api.py`/`video_processor.py` had no such state
machine (a container restart depended entirely on `startup_reconcile()`
re-adding every camera found in `cameras:config`, with no persisted
notion of "was this supposed to be idle or processing"):

```
start_idle(n)     load + warm up n units (engines / OCR workers), do
                  NOT process yet. Persists phase=idle, {unit}_count=n.
stop_idle()       release everything start_idle loaded. Implies
                  stop_process() first if currently processing.
                  Persists phase=stopped.
start_process()   begin real work with whatever is already loaded and
                  warm (attach cameras / start pulling ocr:tasks).
                  Persists phase=processing.
stop_process()    stop real work, keep everything warm. Persists
                  phase=idle.
```

Every transition — and every time the unit count changes **outside**
an explicit transition (an engine spun up because camera count grew,
`rebalance()` consolidated engines, or the OCR pool's watchdog
respawned a crashed worker) — is checkpointed to
`internal:{service}:state` via `checkpoint_now()`.

On boot, `self_heal()` reads that checkpoint and replays it:
`start_idle(persisted_count)`, then `start_process()` too if the
persisted phase was `processing`. Kill `-9` a container mid-run, bring
it back up, and it returns to the exact same phase with the exact same
number of engines/workers — and every camera in `active_cameras`
gets re-attached — without the backend resending anything. Prove it
with `redis_tools.py`'s `test_service_restart()`.

The OCR service additionally self-heals at the **process** level: a
background watchdog (`OcrPool`, `WATCHDOG_INTERVAL_SEC`) detects a
crashed worker subprocess and respawns it to keep the pool at its
expected size, without waiting for the whole container to restart.


## Camera offline handling

`backend_bridge.py`'s offline-grace timer waits
`CAMERA_OFFLINE_GRACE_SECONDS` after a `camera:events` offline
transition before detaching the camera — absorbing a camera that is
merely mid-reconnect (the reference `alpr_api.py` tore the camera down
**instantly** on any offline event, churning its BYTETracker and every
in-flight track on the smallest network blip). An online event within
the grace window cancels the pending teardown.

Separately, and unchanged from the reference: an **internal**
processing error (a batch-inference exception, not a physical
disconnect) triggers a bounded auto-restart
(`ERROR_RESTART_MAX_RETRIES`/`ERROR_RESTART_DELAY_SEC`/
`ERROR_RESTART_COUNTER_RESET_AFTER_SEC`, ported from
`alpr_api.py`'s own `_attempt_processing_error_restart`) — the camera
is detached, `ai_status` is written `stopped` with the error, and a
background retry reattaches it after the delay, up to the retry limit.


## Engine consolidation (new — not in the reference pipeline)

The reference `EngineManager` only ever **grows**: a new engine
subprocess is spawned once the busiest one hits
`MAX_CAMERAS_PER_ENGINE`. It never shrinks, so camera churn fragments
cameras across more engines than necessary.

`EngineManager.rebalance()` (`detector/src/engine_manager.py`, ticks
every `ENGINE_REBALANCE_INTERVAL_SEC`) detects when the current camera
count would fit into fewer engines, drains the least-loaded engines by
replaying each camera's cached `add_camera()` config onto a fuller
surviving engine, and stops the emptied engines once they are actually
empty. A migrated camera's BYTETracker restarts (track IDs reset) — the
RTSP relay keeps the camera's actual upstream connection alive
throughout, so this is a bookkeeping reset, not a stream interruption.
See `tests/test_engine_manager_rebalance.py`, verified.


## Storage split

| | Where | Why |
|---|---|---|
| Valid/invalid plate crops + vehicle frames sent to the backend | **MinIO** (`common/platecore/minio_store.py`, defaults `eyepass-private-bucket`/`eyepass-public-bucket` — matching your existing deployment) | pipeline data — durable, shared, what the backend resolves into URLs |
| Annotated detector debug video | **local bind-mounted volume** (`detector_data`, `DEBUG_VIDEO_*` env vars) | debugging only — never uploaded, never read by anything outside the container that wrote it |

The OCR service is the only service that touches MinIO at all — the
detector never does (see `detector/requirements.txt`).


## Camera frames — always via the MediaMTX relay

The detector never opens an RTSP connection to a camera directly; it
always goes through the relay, via `MTX_RTSP_BASE_URL` and
`backend_bridge.rtsp_url()`:

```python
def rtsp_url(camera_id: str) -> str:
    return f"{config.MTX_RTSP_BASE_URL.rstrip('/')}/{camera_id}"
```


## Migrating from the existing single-process deployment

Because every backend-facing Redis key name, `ai_status` value, and the
`plate:camera:events` channel spelling are unchanged, you can point
this module at the **same** Redis your current `eyepass_plate`
container uses and it will pick up exactly where that container left
off:

1. Stop the old `eyepass_plate` (single-process) container.
2. Bring up `plate_detector` + `plate_ocr` (same Redis, same MinIO,
   same `best.pt`, same `PadOcr/` folder as above).
3. On boot, `reconcile_on_startup()` reads `plate:internal:active_cameras`
   — empty on a first run against the old deployment's Redis, since
   that ledger is new. In that case, also run each currently-active
   camera through `redis_tools.py activate --id <id>` once (or let the
   backend re-send `activated`, which it will do on its own on its
   next reconciliation pass, if it has one) to populate the ledger.
   From then on, every restart is a true self-heal — no manual step
   needed again.


## Testing

This delivery was verified with `python3 -m py_compile` across every
file (clean), an import-level check of every module under
`detector/src` and `ocr_service/src` against stubbed third-party deps,
and a real unit test suite (`tests/`, `python3 tests/run_all.py`,
**44/44 passing**) covering:

- `platecore.codec` — task/result pickle round-trips, the
  `DateTimeEncoder` (including its bytes → base64 handling, added
  beyond face's own encoder to match the reference `video_processor.py`).
- `platecore.keys` — the singular `camera:events` spelling and the
  `vehicle:results`/`ai_status` names, pinned so a future "helpful"
  rename gets caught immediately.
- `platecore.lifecycle` — every transition, `checkpoint_now()`, and a
  full `self_heal()` round trip.
- `platecore.active_state` — the durable desired-state ledger.
- `platecore.bus.write_ai_status` — the `ai_status` JSON shape.
- `detector.engine_manager` — `add_camera`/`remove_camera` placement,
  and `rebalance()`'s exact consolidation scenario (10 cameras,
  5-per-engine cap, fragmented 4+3+3 → consolidates to 2) plus edge
  cases (already-tight, single-engine, a camera with no cached config
  is never silently dropped).
- `detector.triggers` — line-crossing (age/confidence gates, cooldown,
  independent per-track state), ROI entry/exit (confirmation frames,
  fires once), stopped-in-ROI (velocity/duration gates, fires once),
  and the bottom-center bbox anchor.
- `detector.backend_bridge` — `build_camera_job`'s config-field mapping
  (roi/cross_line/stop_roi → the engine's kwarg shape), activate/
  deactivate writing the correct `ai_status` and durable ledger state,
  activating an offline camera (waits rather than attaching), a
  `camera:events` online transition only attaching an *activated*
  camera, and the bounded internal-error auto-restart (detach + delayed
  reattach, retry budget enforced).

`tests/fakes/` ships a minimal in-memory `FakeRedis` (just the subset
of the redis-py API `platecore/bus.py` actually calls) plus import-time
stubs for `torch`/`ultralytics`/`lap`/`boto3`/`paddleocr` — enough to
import and exercise the real logic without a GPU, a real Redis server,
or those packages installed. cv2/numpy/scipy *are* used for real (not
stubbed) in the trigger tests. None of this stubbing ships in the
Docker images — it is `tests/`-only; the Dockerfiles install the real
`redis`/`boto3`/`paddleocr`/`paddlepaddle` packages and build from your
base image (detector) or a plain python image (OCR service — see
`ocr_service/Dockerfile`'s comment on why it doesn't need the heavy
torch/CUDA base image at all).

`tests/test_engine_hub.py` covers the engine's side of the control-hub
contract on the real Engine methods: crop submission and its gates
(in flight / satisfied / unchanged crop), hub ctl handling, the
track-end finalize pass, and ending live tracks on camera removal. The
hub itself has its own suite (`control-hub/tests`, `python3 control-hub/tests/run_all.py`), including an
end-to-end run against a real `redis-server`.

Run it yourself:
```bash
python3 tests/run_all.py
```


## Reference material this was built from

The detection/tracking pipeline (`Engine`, `EngineManager`,
`RTSPStreamReader`, spatial triggers, best-crop ranking, quality gates)
comes from `video_processor.py`; BYTETrack from `plate_tracker.py`; the
debug video recorder from `debug_recorder.py`, copied verbatim; the
OCR pipeline (dual PaddleOCR car/motorcycle setup, CLAHE preprocessing,
char-by-char voting, plate-format validation regexes,
`save_prep_for_mysql`'s MinIO key layout) from `ocr_worker.py`; MinIO
helpers generalized from `minio_uploader.py`; the backend
activate/deactivate/`ai_status`/camera-events handling and the bounded
error-restart from `alpr_api.py`. The self-healing/durable-desired-
state pattern, `RedisBus`, two-service Redis-only split, engine
consolidation, offline-grace timer, and the two-compose-file structure
were ported from `face_service`'s own `detector`/`recognizer` split
(`backend_bridge.py`, `engine_manager.py`, `pool.py`, `worker.py`,
`compose.yaml`, `compose.infra.yaml`, `redis_tools.py`), which was
supplied as the reference for how this platform's modules should be
built.
