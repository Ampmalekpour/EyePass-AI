# Heatmap module — ai_service

> **EyePass suite (`develop-suite` branch).** This module runs as part
> of the suite: start it from the repository root (`docker compose up -d`
> with `heatmap` in `COMPOSE_PROFILES`). Redis, MinIO, the MediaMTX relay and
> the camera manager (`camera-service/`) are shared and defined in the
> root `compose.yaml`. This module no longer ships its own copies or a
> `compose.infra.yaml`. Frames are read from the relay path
> `cam_<sha1(camera address)>`. Parts of the text below describe the
> standalone layout. See the root `README.md` for the suite setup.

Crowd-density heatmaps from RTSP cameras. `ai_service` detects people
(or heads, depending on your model) in each camera's frame, accumulates
their positions into a per-camera, per-day 3D grid ("cube"), stores
those cubes in MinIO, and tells the backend where to find them — all
over Redis. Same detection/accumulation **behaviour** as the
pre-existing standalone build of this module — the cube math, grid
indexing and MinIO round-trip are ported unchanged — but reorganized
to match this platform's `plate-service`/`face-service`/`fire-smoke-service`
pattern:

```
ai_service    camera frames -> YOLO person/head detect -> ROI crop ->
              per-cell accumulation into today's cube -> periodic
              flush to MinIO -> announces the flush over Redis
```

Like `fire-smoke-service`, there is no second-stage worker downstream
of detection — heatmap accumulation IS the result, so there's nothing
to scale independently the way plate's OCR or face's recognizer are.

This module **does** bundle its own `mediamtx` relay + `camera_stream`
— the **exact same** camera-service the plate/face/fire modules run
(previously this module shipped its own, subtly different
implementation — see "What changed" below) — so it can be brought up
and tested completely on its own, next to the other modules, on host
ports offset from theirs (see `.env.example`).


## What changed versus the pre-existing standalone build

That build was already solid — self-healing camera state, an offline
grace timer, graceful-shutdown cube flushing, and a genuinely thorough
README (§13 of it was literally written *for* an AI assistant to work
from). This rewrite keeps all of that and aligns the plumbing with the
rest of the platform:

| | Before | Now |
|---|---|---|
| Camera online/offline channel | `{module}:cameras:events` (plural) via this module's own bespoke `camera-service` | `{module}:camera:events` (**singular**) via the exact camera-service plate/face/fire run |
| Config-updated notify | `{module}:camera:config:updated` (already singular — unchanged) | unchanged |
| Redis/MinIO | this module's own `compose.infra.yaml` (private Redis + MinIO) | the platform's shared Redis/MinIO, matching every other module |
| activate/deactivate request | backend `LPUSH`, service `BLPOP` (LIFO) | backend `LPUSH`, service `BRPOP` (**FIFO** — matches plate/face/fire) |
| activate/deactivate response | `STRING` (`SET`/`GET`, 60s TTL) | `LIST` (`RPUSH`/`BRPOP`, 60s TTL) — matches plate/face/fire; flagged as a spec mismatch in the old README, fixed here |
| Per-camera reported status | none | `{module}:cameras:{camera_id}:ai_status`, matching the Django-facing field the other modules expose |
| Self-healing checkpoint | camera-level only (`ai:results`... `ai:active`) | same, **plus** a phase/engine-count checkpoint (`{module}:internal:detector:state`), matching the other modules |
| Visual debug | plain bbox overlay only (`SAVE_OUTPUT`) | rolling annotated MP4 + JSONL sidecar with the live accumulation grid overlaid, matching plate/face/fire's `debug_recorder.py` |
| MinIO client | `minio` SDK | `boto3`, matching the other modules' `minio_store.py` |
| Results/active-state key names | `{module}:ai:results` / `{module}:ai:active` | **unchanged** — these already matched the face module's own convention, so nothing here needed fixing; see `common/heatmapcore/keys.py`'s docstring |

**Tell the backend team about the response-shape change** (`STRING`→`LIST`)
if anything already talks to the old build — see "Redis contract" below.


## Contents

```
common/heatmapcore/     shared library, copied into the Docker image
ai-service/              ai_service (src/, Dockerfile, requirements.txt)
camera-service/          bundled mediamtx relay + camera_stream (copied from the other modules, unmodified)
compose.yaml             mediamtx + camera_stream + ai_service (+ test-video publishers, watchdog)
.env.example             every variable the stack understands
redis_tools.py           manual test harness / CLI — stands in for the backend
Dockerfile.publisher     builds the --profile test-video RTSP loopers
publish_loop.sh          the publisher container's loop script
.gitignore               keeps weights/debug output/.env out of git
README.md                this file
```


## What you need to provide

Nothing in this delivery trains or ships model weights — you bind-mount
what you already have.

| What | Where it goes | `.env` variable |
|---|---|---|
| Detection weights (`yolov8n.pt` works out of the box — COCO class 0 = person; swap for a head-detection model if you have one) | `${DETECTION_MODELS_DIR}/yolov8n.pt` | `DETECTION_MODELS_DIR` (default `./models/detection`), `MODEL_PATH` (default `/models/yolov8n.pt`) |
| your GPU base image (already has torch/ultralytics/OpenCV) | Docker build arg | `AI_BASE_IMAGE` (default `base_image_gpu:latest`, matching the plate/face/fire detectors) |

`models/detection/` ships with a `.gitkeep` placeholder so the folder
exists before you drop your weights in — the weight file itself is
gitignored. If you swap in a different model, keep its class-0 label
meaning "the thing to accumulate" (person/head), or edit
`CLASS_LABELS`/`target_classes` in `ai-service/src/config.py`.


## Run it

```bash
# 1. Extract this folder next to your other modules (plate-service/, face-service/, fire-smoke-service/, ...)

# 2. Create the shared network once, ever — skip if it already exists
docker network create eyeplate_net

# 3. Configure
cp .env.example .env
# put yolov8n.pt (or your own weights) in models/detection/

# 4. Bring the stack up
docker compose up -d --build

# 5. (optional) test without real cameras — loops a local clip into
#    cameras 1-3 over the bundled mediamtx relay
cp /path/to/a/clip.mp4 ./test_video.mp4
docker compose --profile test-video up -d --build
```

Watch it come up:

```bash
docker compose logs -f ai_service
```

You want to see `YOLO loaded on ...` with no import errors, then
`self_heal(): last known state phase=...` before touching anything
else.

`ai_service` starts idle automatically, then self-heals to whatever
phase it was last in (see "Self-healing" below). After a *fresh*
`docker compose up` with an empty Redis, it comes up idle and waits for
the backend's first `activated` command per camera.

Register a camera and activate it with the bundled test harness:

```bash
python redis_tools.py set-camera --id 1 --address "rtsp://mediamtx:8554/1" --title "Lobby 1" --roi 0 0 1 1
python redis_tools.py activate --id 1
python redis_tools.py status --id 1
python redis_tools.py list
```

Dev URLs once everything is up: AI service `:8002/docs` (Swagger — the
HTTP endpoints below are operational only; backend integration is
entirely over Redis), MediaMTX API `:9977`.

To confirm the GPU is actually being used:

```bash
docker compose exec ai_service python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Once that prints `True` and your card's name, set `DETECTION_DEVICE=cuda:0`
and `STRICT_DEVICE=true` in `.env` and `docker compose up -d` again.


## Redis contract

### Backend contract (fixed spelling — `REDIS_MODULE=heatmap`)

| Key | Type | Direction |
|---|---|---|
| `heatmap:cameras:config` | HASH | backend → camera_stream, ai_service |
| `heatmap:camera:config:updated` | PUB/SUB | backend → camera_stream |
| `heatmap:cameras:details` | HASH | camera_stream → backend, ai_service |
| `heatmap:camera:events` (**singular** "camera") | PUB/SUB | camera_stream → ai_service |
| `heatmap:cmd:ai:request` | LIST (BRPOP) | backend → ai_service |
| `heatmap:cmd:ai:response:{request_id}` | LIST, 60s TTL | ai_service → backend |
| `heatmap:cameras:{camera_id}:ai_status` | HASH, field=`camera_id` -> json | ai_service → backend |
| `heatmap:ai:results` | LIST | ai_service → backend |
| `heatmap:ai:active` | HASH | ai_service → ai_service (internal; readable for ops) |

`heatmap:camera:events` is deliberately **singular** — the bundled
camera-service is the exact one the plate/face/fire modules run, and
it only ever publishes the singular channel. See "What changed" above
if you're migrating from the pre-existing standalone build, which used
the plural spelling.

`cameras:config` field shape, `cmd:ai:request`/`response` shapes, and
`ai:results`/`ai:active` payload shapes are all otherwise **unchanged**
from the pre-existing standalone build's own documented contract —
copied here verbatim since none of it needed fixing:

```json
// cameras:config[camera_id]
{
  "id": 2, "title": "دوربین تست",
  "address": "rtsp://admin:admin123@192.168.30.49:554/cam/realmonitor?channel=3&subtype=0",
  "usage": "ENTRY_EXIT",
  "roi": {"x": 0.168, "y": 0.078, "w": 0.779, "h": 0.822},
  "stop_roi": null, "cross_line": null
}
```

- **`roi` is normalised (0..1)** relative to the full frame. `ai_service`
  crops to it before detection. Omit it, or send `{"x":0,"y":0,"w":1,"h":1}`,
  for the whole frame.
- **URL-encode credentials.** A password containing `@` must be written
  `%40`, or both FFmpeg and MediaMTX mis-parse the host.
- `stop_roi`/`cross_line` are accepted and stored (so a shared
  frontend/config editor can write the same shape it writes for plate)
  but unused by this module's own pipeline.

```json
// cmd:ai:request  (LPUSH by backend)
{"request_id": "c237434c-...", "camera_id": 2, "action": "activated"}
{"request_id": "d5b8a6ec-...", "camera_id": 2, "action": "deactivated"}
```

The RTSP URL is not in this message — `ai_service` resolves it from
`cameras:details`, so **the camera must exist in `cameras:config` first**.
Activating an unknown camera returns `ERROR`. Activating a camera that
is currently *offline* returns `OK`, not `ERROR` — we have accepted
responsibility for it and will start it when it comes back.

```json
// cmd:ai:response:{request_id}  (RPUSH by ai_service, 60s TTL)
["{\"status\": \"OK\"}"]
```

```json
// ai:results  (LPUSH by ai_service, each time a cube is flushed)
{
  "camera_id": "2", "event": "matrix_sync", "date": "2026-09-16",
  "bucket": "heatmap-data", "object_key": "2/2026-09-16.npy",
  "timestamp": 1789456123.44
}
```

A **pointer, not data**. Consume with `BRPOP` for FIFO order, since
entries are pushed with `LPUSH`. Entries are **idempotent and
repeated**: the same `(camera_id, date)` is announced on every flush,
because the same object is overwritten with a larger cube each time.
**Upsert on `(camera_id, date)` — do not append rows.**

### Internal (`heatmap:internal:...` — new in this rewrite)

| Key | Purpose |
|---|---|
| `internal:detector:state` | ai_service's self-healing checkpoint: `{phase, engine_count, updated_at}` |
| `internal:detector:heartbeat` | liveness (30s TTL, refreshed every 10s) |

`redis_tools.py` has helpers for every row in both tables
(`set_camera`, `activate`/`deactivate`, `show_active`,
`show_self_healing_state`, `watch_results`, ...) plus scenarios
(`test_service_restart`, `test_connect_disconnect_reconnect`,
`run_engine_consolidation_test`) and the CLI subcommands (`set-camera`,
`activate`, `deactivate`, `status`, `list`, `remove`).


## The data: heatmap cubes in MinIO

**Bucket:** `MINIO_BUCKET` (default `heatmap-data`)
**Key:** `<camera_id>/<YYYY-MM-DD>.npy` — one object per camera per day
**Format:** NumPy `.npy`, `dtype=uint32`, shape `(time_slots, grid_height, grid_width)`,
default `(288, 72, 128)`.

- `time_slots = 1440 / TIME_RESOLUTION_MINUTES` → 288 at the 5-minute
  default. Slot `i` covers `[i*5min, (i+1)*5min)` local time.
- `grid_height × grid_width` divides the camera frame into cells;
  `[row][col]` holds the detection count for that cell.
- Raw counts, never normalised. Summing is always valid.

Reading a window, e.g. 09:00–12:00:

```python
import io, numpy as np
from minio import Minio

client = Minio("minio:9000", access_key="...", secret_key="...", secure=False)
obj = client.get_object("heatmap-data", "2/2026-09-16.npy")
cube = np.load(io.BytesIO(obj.read()))          # (288, 72, 128)

RES = 5
start = (9 * 60) // RES          # 108
end   = (12 * 60) // RES         # 144
heat_2d = cube[start:end].sum(axis=0)           # (72, 128) counts per cell

total = int(heat_2d.sum())
busiest_cell = np.unravel_index(heat_2d.argmax(), heat_2d.shape)
```

Cell → pixel: `x = col * (frame_width / grid_width)`, same for rows. To
render, normalise (`log1p` suits crowd data), resize to frame size,
blur, apply a colormap — `heatmap_manager.HeatmapCubeManager.render_heatmap`
does exactly this.


## Self-healing

`ai_service` uses the same four-method lifecycle
(`common/heatmapcore/lifecycle.py`, `ServiceLifecycle`) as the
plate/face/fire modules:

```
start_idle(n)     ensure at least n engine processes are up, warm and
                  loaded, but not yet accepting cameras. Persists
                  phase=idle, engine_count=n.
stop_idle()       release everything start_idle loaded. Implies
                  stop_process() first if currently processing.
                  Persists phase=stopped.
start_process()   begin real work with whatever is already loaded and
                  warm. Persists phase=processing.
stop_process()    stop real work, keep everything warm. Persists
                  phase=idle.
```

On boot, `self_heal()` reads the last persisted `{phase, engine_count}`
and replays it, and then `_reconcile_on_startup()` reads
`{module}:ai:active` — the durable "which cameras should be
running" ledger — and re-attaches every camera in it (waiting for the
online event if a camera happens to be offline right now). Kill `-9`
the container mid-run, bring it back up, and every previously-active
camera resumes **without the backend resending anything**, and its
cube continues accumulating rather than resetting (cubes are
downloaded from MinIO before a camera starts accumulating into them —
see `HeatmapCubeManager._load_or_create_cube`). Prove it with
`redis_tools.py`'s `test_service_restart()`.

> **Single-writer rule**, unchanged from the pre-existing build and
> just as load-bearing here: exactly one `ai_service` may own a given
> camera+date cube. Flushing is read-modify-write, so two writers
> silently clobber each other. Never run more than one `ai_service`
> replica against the same Redis/MinIO pair, and keep uvicorn at
> `workers=1` (already the case in `main.py`). To scale horizontally,
> shard by camera id — one deployment per disjoint set.


## Camera offline handling

The offline-grace timer waits `CAMERA_OFFLINE_GRACE_SECONDS` after a
`camera:events` offline transition before flushing the camera's cube
and tearing its engine attachment down — absorbing a camera that is
merely mid-reconnect. An online event within the grace window cancels
the pending teardown. The camera **stays** in `ai:active` throughout,
so the next `connected: true` restarts it automatically; the backend
is not involved and does not re-send anything.


## Visual debug

Set `DETECTOR_DEBUG_VIDEO_ENABLED=true` to record rolling per-camera
MP4 segments (`DEBUG_VIDEO_SEGMENT_SECONDS` long,
`DEBUG_VIDEO_MAX_SEGMENTS` kept) under `./debug_video/<camera_id>/`,
each with every raw detection box, the current accumulation grid tinted
by its live heat for the current 5-minute slot, and a camera-id/
timestamp/frame-count stamp — plus an optional JSONL sidecar
(`DEBUG_VIDEO_JSONL=true`) with one line per frame's raw detections.
Same pattern as the plate/face/fire modules' own debug recorders — the
pre-existing standalone build only had a plain, ungridded bbox overlay
(`SAVE_OUTPUT`, still available, off by default, superseded by this).
This is local-disk only, expensive, and off by default — leave it off
in production.


## Engine topology

Cameras are spread across engine (YOLO inference) subprocesses,
`MAX_CAMERAS_PER_ENGINE` per engine — the same `EngineManager` shape
the pre-existing standalone build already had. Unlike the plate/face/
fire modules, there's currently no `rebalance()` consolidation pass
here (engines only ever grow with camera count, matching the
pre-existing build's own behaviour) — camera churn on this module
tends to be low-frequency (crowd cameras are provisioned once, not
activated/deactivated per vehicle), so it wasn't carried over; the
`EngineManager` shape leaves room to add one later if that changes.


## Storage split

| | Where | Why |
|---|---|---|
| Heatmap cubes | **MinIO** (`common/heatmapcore/minio_store.py`, default `heatmap-data`) | pipeline data — durable, shared, what the backend resolves and reads back |
| Annotated debug video (grid overlay, raw detections) | **local bind-mounted volume** (`DEBUG_VIDEO_*` env vars, off by default) | debugging only — never uploaded, never read by anything outside the container that wrote it |


## Camera frames — always via the MediaMTX relay

`ai_service` never opens an RTSP connection to a camera directly; it
always goes through the relay, via `MTX_RTSP_BASE_URL`.


## Testing without real cameras

`docker compose --profile test-video up -d --build` starts
`video_publisher_1..3`, each looping `TEST_VIDEO_FILE` into the bundled
mediamtx relay as camera `1`, `2`, `3` (`Dockerfile.publisher`,
`publish_loop.sh` — identical to the plate/face/fire modules' own).
Register and activate them with `redis_tools.py` as shown above, then
`redis_tools.watch_results()` or tail `ai_service`'s logs to see cubes
accumulate and flush.

`redis_tools.py` also carries three prepared scenarios (edit the
bottom of the file to pick one): `test_service_restart()` proves
"resume where it left off"; `test_connect_disconnect_reconnect()` is
an end-to-end test with a real camera you unplug/replug by hand;
`run_engine_consolidation_test()` spreads more cameras than
`MAX_CAMERAS_PER_ENGINE` across engines and back down.


## Reference material this was built from

The detection/accumulation pipeline (`Engine`, `EngineManager`,
`RTSPStreamReader`, `HeatmapCubeManager`'s cube math), the FastAPI
operational surface (`/health`, `/active`, `/status`, `/cameras`,
manual `/start_video`/`/stop_video`/`/reset_video` overrides), and the
resilience properties in "Self-healing"/"Camera offline handling" above
all come from this module's own pre-existing standalone build. The
Redis key-naming alignment (singular `camera:events`, `heatmapcore`
library shape, LIST-based cmd response, per-camera `ai_status`,
self-healing phase checkpoint), the bundled camera-service, the debug
recorder pattern, and the overall directory/compose structure were
ported from the plate/face/fire modules, which were supplied as the
reference for how this platform's modules should be built.
