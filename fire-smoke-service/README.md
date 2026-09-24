# Fire/Smoke module — detector

A single, independently deployable, self-healing service that replaces
the original single-process fire/smoke pipeline (`fire_and_smoke_api.py`
+ `video_processor.py` + `model_manager.py`). Same detection / spatial
threat-verification **behaviour** as the reference code — the YOLO
batch-inference loop and the 2x2 spatial-grid rolling-window
threat-verification state machine are ported near-verbatim — but
reorganized to match this platform's `plate-service`/`face-service`
pattern:

```
fire_detector    camera frames -> YOLO fire/smoke detect -> per-region
                  2x2 grid vote -> THREAT / RESOLUTION verdicts ->
                  uploads crops to MinIO -> pushes results over Redis
```

Unlike the plate and face modules, there is no second-stage worker
service here — fire/smoke has nothing equivalent to OCR or face
recognition sitting downstream of detection, so `fire_detector` both
detects and verifies in one process, then talks to the backend
directly over Redis.

This module **does** bundle its own `mediamtx` relay + `camera_stream`
(unlike `plate-service`, which points at an already-running shared
one) so it can be brought up and tested completely on its own, next to
the other modules, on host ports offset from theirs (see `.env.example`)
so all three can run on one host without a collision. Point
`MTX_RTSP_BASE_URL` at an existing shared relay instead if this
deployment already runs one every module attaches to.


## Contents

```
common/firecore/        shared library, copied into the Docker image
detector/                fire_detector service (src/, Dockerfile, requirements.txt)
camera-service/          bundled mediamtx relay + camera_stream (copied from the other modules, unmodified)
compose.yaml             mediamtx + camera_stream + fire_detector (+ test-video publishers, watchdog)
.env.example             every variable the stack understands
redis_tools.py           manual test harness / CLI — stands in for the backend
Dockerfile.publisher     builds the --profile test-video RTSP loopers
publish_loop.sh          the publisher container's loop script
.gitignore               keeps weights/debug output/.env out of git
README.md                this file
```


## What you need to provide

Nothing in this delivery trains or ships model weights — you bind-mount
what you already have, matching the `.env.example` variables.

| What | Where it goes | `.env` variable |
|---|---|---|
| `best.pt` (YOLO fire/smoke detection) | `${DETECTION_MODELS_DIR}/best.pt` | `DETECTION_MODELS_DIR` (default `./models/detection`) |
| your GPU base image (already has torch/ultralytics/OpenCV) | Docker build arg | `AI_BASE_IMAGE` (default `base_image_gpu:latest`, matching the plate/face detectors) |

`models/detection/` ships with a `.gitkeep` placeholder so the folder
exists before you drop `best.pt` in — the weight file itself is
gitignored.

The model bind mount is never baked into the image, so swapping it
needs no rebuild — just overwrite the file and restart the container.


## Run it

```bash
# 1. Extract this folder next to your other modules (plate-service/, face-service/, ...)

# 2. Create the shared network once, ever — skip if it already exists
docker network create eyeplate_net

# 3. Configure
cp .env.example .env
# put best.pt in models/detection/

# 4. Bring the stack up
docker compose up -d --build

# 5. (optional) test without real cameras — loops a local clip into
#    cameras 1-3 over the bundled mediamtx relay
cp /path/to/a/clip.mp4 ./test_video.mp4
docker compose --profile test-video up -d --build
```

`fire_detector` starts idle automatically, then self-heals to whatever
phase it was last in (see "Self-healing" below). After a *fresh*
`docker compose up` with an empty Redis, it comes up idle and waits for
the backend's first `activated` command per camera.

Register a camera and activate it with the bundled test harness:

```bash
python redis_tools.py set-camera --id 1 --address "rtsp://mediamtx:8554/1" --title "Warehouse 1" --roi 0 0 1 1
python redis_tools.py activate --id 1
python redis_tools.py status --id 1
python redis_tools.py list
```

(`redis_tools.py` connects to Redis over its published host port — set
`REDIS_HOST`/`REDIS_PORT`/`REDIS_URL` env vars if you didn't use the
defaults, exactly like the plate/face modules' own copy.)


## Redis contract

### Backend contract (fixed spelling — `REDIS_MODULE=fire`)

| Key | Type | Direction |
|---|---|---|
| `fire:cameras:config` | HASH `camera_id -> json` | backend → camera_stream, detector |
| `fire:cameras:details` | HASH `camera_id -> json` | camera_stream → backend, detector |
| `fire:camera:events` (**singular** "camera") | PUB/SUB | camera_stream → detector |
| `fire:cmd:ai:request` | LIST (BRPOP) | backend → detector |
| `fire:cmd:ai:response:{request_id}` | LIST, 60s TTL | detector → backend |
| `fire:cameras:{camera_id}:ai_status` | HASH, field=`camera_id` -> json | detector → backend |
| `fire:detections:results` | LIST | detector → backend |

`fire:camera:events` is deliberately **singular**, matching the same
spelling every other module's `eyepass-camera-stream` deployment
actually publishes to — see `common/firecore/keys.py`'s docstring. Do
not "fix" this to a plural spelling.

`detections:results` carries every THREAT / RESOLUTION verdict the 2x2
grid state machine produces — this key name is preserved from the
pre-existing partial port of this module for backend compatibility, so
no backend-side changes are needed to read it.

### Internal (`fire:internal:...` — brand new, nothing outside this service reads these)

| Key | Purpose |
|---|---|
| `internal:active_cameras` | durable desired-state ledger (written *before* acting, so a crash mid-activation is recovered by startup reconcile) |
| `internal:detector:state` | detector's self-healing checkpoint: `{phase, engine_count, updated_at}` |
| `internal:detector:heartbeat` | liveness (30s TTL, refreshed every 10s) |

`redis_tools.py` has helpers for every row in both tables
(`set_camera`, `activate`/`deactivate`, `show_active_cameras`,
`show_self_healing_state`, `watch_results`, ...) plus scenarios
(`test_service_restart`, `test_connect_disconnect_reconnect`,
`run_engine_consolidation_test`) and the CLI subcommands (`set-camera`,
`activate`, `deactivate`, `status`, `list`, `remove`).


## Self-healing

The service uses the same four-method lifecycle
(`common/firecore/lifecycle.py`, `ServiceLifecycle`) as the plate and
face modules:

```
start_idle(n)     load + warm up n engines, do NOT process yet.
                  Persists phase=idle, engine_count=n.
stop_idle()       release everything start_idle loaded. Implies
                  stop_process() first if currently processing.
                  Persists phase=stopped.
start_process()   begin real work with whatever is already loaded and
                  warm (attach cameras). Persists phase=processing.
stop_process()    stop real work, keep everything warm. Persists
                  phase=idle.
```

Every transition — and every time the engine count changes **outside**
an explicit transition (an engine spun up because camera count grew,
or `rebalance()` consolidated engines) — is checkpointed to
`internal:detector:state` via `checkpoint_now()`.

On boot, `self_heal()` reads that checkpoint and replays it:
`start_idle(persisted_count)`, then `start_process()` too if the
persisted phase was `processing`. Kill `-9` the container mid-run,
bring it back up, and it returns to the exact same phase with the
exact same number of engines — and every camera in `active_cameras`
gets re-attached — without the backend resending anything. Prove it
with `redis_tools.py`'s `test_service_restart()`.


## Camera offline handling

`backend_bridge.py`'s offline-grace timer waits
`CAMERA_OFFLINE_GRACE_SECONDS` after a `camera:events` offline
transition before detaching the camera — absorbing a camera that is
merely mid-reconnect instead of tearing it (and its grid-verification
state for that camera) down on the smallest network blip. An online
event within the grace window cancels the pending teardown.

Separately: an **internal** processing error (a batch-inference
exception, not a physical disconnect) triggers a bounded auto-restart
(`ERROR_RESTART_MAX_RETRIES`/`ERROR_RESTART_DELAY_SEC`/
`ERROR_RESTART_COUNTER_RESET_AFTER_SEC`) — the camera is detached,
`ai_status` is written `stopped` with the error, and a background retry
reattaches it after the delay, up to the retry limit.


## Threat verification — 2x2 spatial grid

Each camera's frame is split into a 2x2 grid of regions. Every frame,
each region gets a verdict (`clear` / `smoke` / `fire` / `both`) from
whatever detections fall inside it. A rolling window per region
(`GRID_WINDOW_SIZE` frames) counts matching verdicts; once a region
hits `GRID_VERIFY_THRESHOLD` matching frames within that window, it
**latches** from clear to the threatened state and a `THREAT` event is
pushed to `detections:results`. A latched region only resolves back to
clear after `GRID_COOLDOWN_FRAMES` consecutive fully-clear frames, at
which point a `RESOLUTION` event is pushed. This is the same math as
the pre-existing `video_processor.py`, ported unchanged — only the
config variable names and where the event lands (Redis instead of an
in-process callback) changed.


## Engine consolidation

`EngineManager.rebalance()` (`detector/src/engine_manager.py`, ticks
every `ENGINE_REBALANCE_INTERVAL_SEC`) detects when the current camera
count would fit into fewer engines, drains the least-loaded engines by
replaying each camera's cached `add_camera()` config onto a fuller
surviving engine, and stops the emptied engines once they are actually
empty. The RTSP relay keeps each camera's actual upstream connection
alive throughout, so a migration is a bookkeeping reset, not a stream
interruption. Same logic as the plate/face modules.


## Storage split

| | Where | Why |
|---|---|---|
| THREAT/RESOLUTION crop images | **MinIO** (`common/firecore/minio_store.py`, default `eyepass-private-bucket`, matching the other modules) | pipeline/alert data — durable, shared, what the backend resolves into URLs |
| Annotated detector debug video (grid lines, region verdicts, bboxes) | **local bind-mounted volume** (`DEBUG_VIDEO_*` env vars, off by default) | debugging only — never uploaded, never read by anything outside the container that wrote it |


## Camera frames — always via the MediaMTX relay

The detector never opens an RTSP connection to a camera directly; it
always goes through the relay, via `MTX_RTSP_BASE_URL`.


## Visual debug

Set `DETECTOR_DEBUG_VIDEO_ENABLED=true` to record rolling per-camera
MP4 segments (`DEBUG_VIDEO_SEGMENT_SECONDS` long,
`DEBUG_VIDEO_MAX_SEGMENTS` kept) under `./debug_video/<camera_id>/`,
each with an overlaid 2x2 grid, per-region verdict labels, detection
boxes, a cooldown indicator, and a camera-id/timestamp stamp — plus an
optional JSONL sidecar (`DEBUG_VIDEO_JSONL=true`) with one line per
frame's raw detections and grid state, for offline analysis. Same
pattern as the plate and face modules' own debug recorders. This is
local-disk only, expensive, and off by default — leave it off in
production.


## Testing without real cameras

`docker compose --profile test-video up -d --build` starts
`video_publisher_1..3`, each looping `TEST_VIDEO_FILE` into the bundled
mediamtx relay as camera `1`, `2`, `3` (`Dockerfile.publisher`,
`publish_loop.sh` — identical to the plate/face modules' own). Register
and activate them with `redis_tools.py` as shown above, then
`redis_tools.py watch_results()` or tail `fire_detector`'s logs to see
detections and grid verdicts flow.


## Reference material this was built from

The batch-inference engine loop, `EngineManager`, RTSP reader, and the
2x2 spatial-grid threat-verification state machine come from this
module's own `video_processor.py`. The self-healing/durable-desired-
state pattern, `RedisBus`, the Redis contract shape, the offline-grace
timer, the debug-video recorder layout, and the overall service
structure were ported from `plate-service`'s and `face-service`'s own
`detector` services (`backend_bridge.py`, `engine_manager.py`,
`main.py`, `compose.yaml`, `redis_tools.py`), which were supplied as
the reference for how this platform's modules should be built.
