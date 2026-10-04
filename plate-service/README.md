# Plate module — detector + OCR

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
detector/               plate_detector service (src/, tools/, Dockerfile, requirements.txt)
                          src/config.py              every detector setting (.env picks device + model)
                          src/inference_backends.py  GPU batch (.pt) / CPU per-camera (OpenVINO, ONNX)
                          src/model_files.py         export_info.yaml manifest -> model files
                          src/perf_stats.py          ⏱️ [PERF] / 📊 [STATS] performance logs
                          tools/bench_multistream.py the multi-stream benchmark (GPU reference vs CPU variants)
                          tools/export_cpu_models.py .pt -> ONNX + OpenVINO FP32 in the model-folder layout
ocr_service/            plate_ocr service (src/, Dockerfile, requirements.txt)
compose.yaml            plate_detector + plate_ocr + control_hub (+ standalone relay, test clips, autoheal as profiles)
compose.gpu.yaml        GPU reservation for plate_detector (COMPOSE_FILE in .env)
compose.infra.yaml      redis + minio (+ inspection tools, --profile tools)
.env.example            dev/test .env (device, models, deployment)
.env.prod.example       production .env with [FILL_IN] placeholders
redis_tools.py          manual test harness / CLI — stands in for the backend
tests/                  unit tests (platecore, engine_manager rebalance, triggers, backend_bridge, inference backends)
README.md               this file
```


## Quick start

```bash
docker network create eyeplate_net      # once, ever — skip if it already exists
cp .env.example .env                    # production: cp .env.prod.example .env (see "Production")
docker compose -f compose.infra.yaml up -d   # dev only: Redis + MinIO (production uses the platform's)
docker compose up -d --build            # reads COMPOSE_FILE / COMPOSE_PROFILES from .env
docker compose logs -f plate_detector   # [INIT] / [MODEL] lines, then ⏱️ [PERF] and 📊 [STATS]
```

`.env` decides only **which device and model** run and **where things
are**. Everything else is set in each service's `config.py`:

| In `.env` | Values |
|---|---|
| `COMPOSE_FILE` | GPU host: `compose.yaml:compose.gpu.yaml` · CPU host: `compose.yaml` |
| `DETECTION_DEVICE` | `gpu` · `cpu` · `auto` |
| `DETECTION_GPU_MODEL` | `plate_v8n_480` · `plate_v8s_640` (or `v8n` / `v8s`) |
| `DETECTION_CPU_MODEL` | `openvino_fp32` · `openvino_int8` · `onnx` |
| `OCR_DEVICE` | `cpu` · `gpu` |
| `COMPOSE_PROFILES` | `standalone-stream` (own relay, dev) · empty (the system's streamer) · `+test-video`, `+watchdog` |
| Redis / MinIO / relay / ports / folders | deployment values (`[FILL_IN]` in `.env.prod.example`) |

| Everything else | File |
|---|---|
| detection thresholds, GPU FP16, OpenVINO device, INT8 fix, engines, tracker, triggers, crops, RTSP, logs, debug video | `detector/src/config.py` |
| OCR thresholds, worker pool, logs, decision montages | `ocr_service/src/config.py` |
| control hub policy (satisfied conf, waits, final gate), logs | `control-hub/src/config.py` |

Changing a `config.py` value: edit it, then `docker compose up -d --build <service>`.
Changing `.env`: `docker compose up -d`.


## Models

### Which files, and where

The detector reads one **folder per model** in `DETECTION_MODELS_DIR`
(default `./models/detection`, mounted read-only at `/models`), the same
layout the multi-stream benchmark uses:

```
models/detection/
├── plate_v8n_480/                               ← copy your whole plate_v8n_480 model folder
│   ├── export_info.yaml                         manifest: model_name, imgsz, names, pt, onnx,
│   │                                            openvino: {ov_fp32, ov_int8_box}
│   ├── plate_v8n_480.pt                         GPU (DETECTION_GPU_MODEL=plate_v8n_480) + CPU fallback
│   ├── plate_v8n_480_288x480.onnx               DETECTION_CPU_MODEL=onnx
│   ├── plate_v8n_480_fp32_openvino_model/       DETECTION_CPU_MODEL=openvino_fp32
│   └── plate_v8n_480_int8_box_openvino_model/   DETECTION_CPU_MODEL=openvino_int8
│                                                (from requantize_int8_head_fp32.py)
└── plate_v8s_640/
    └── plate_v8s_640.pt                         GPU (DETECTION_GPU_MODEL=plate_v8s_640)

models/ocr/                                      ← the CONTENTS of your PadOcr/ folder, names unchanged
├── en_PP-OCRv3_det_infer/
├── rec_svrt_fa_final_1/
├── ch_ppocr_mobile_v2.0_cls_infer/
├── rec_svrt_motor/
└── Final_Dict.txt
```

- **Paths come from `export_info.yaml`** (`pt`, `onnx`,
  `openvino.ov_fp32`, `openvino.ov_int8_box`, `imgsz`, `names`), exactly
  like the benchmark. Without a manifest the files are found by name
  (`MODEL_FILE_PATTERNS` in `detector/src/config.py`), so a folder with
  only `plate_v8s_640.pt` works for the GPU.
- **CPU variants are taken from `plate_v8n_480/`** (`CPU_MODEL_NAME` in
  `config.py`), the model that has ONNX/OpenVINO exports. To run
  `plate_v8s_640` on the CPU later, export it
  (`tools/export_cpu_models.py`) and set `CPU_MODEL_NAME`.
- **Input size follows the model.** GPU: 480 / 640 from the name, as
  before. CPU: the export size from `export_info.yaml` (288x480).
- `bench_final/`, `export_eval.json` and other extra files in the
  folder are ignored.
- The models are bind-mounted: replacing a file needs only
  `docker compose restart plate_detector`.

### Make CPU exports for another model

```bash
docker compose run --rm -v "$PWD/models/detection:/export" plate_detector \
    python tools/export_cpu_models.py --model /export/plate_v8s_640/plate_v8s_640.pt --out /export
# -> plate_v8s_640/{export_info.yaml, plate_v8s_640_384x640.onnx, plate_v8s_640_fp32_openvino_model/}
```

Static ONNX (opset 17, simplified, batch 1, FP32, no NMS) and OpenVINO
FP32 from it, as benchmarked. INT8 is not made here; it comes from
`requantize_int8_head_fp32.py` and is added to `export_info.yaml` as
`openvino.ov_int8_box`.


## How the detector runs the models

The engine layer is the same for GPU and CPU. `EngineManager` starts
engine processes and adds cameras to them; a new engine starts when one
is full; `rebalance()` consolidates cameras every 30 s; self-healing
restores the engine count after a restart. Each engine loops over its
cameras:

```
for each camera with a NEW frame (the RTSP reader keeps only the latest; replaced frames = missed)
    → ROI crop
→ inference for all of them             ← the only part that differs, below
→ per camera: tracker → triggers → best crops → OCR tasks / control hub   (unchanged)
```

| `.env` | Inside one engine | Cameras per engine (`config.py`) |
|---|---|---|
| `DETECTION_DEVICE=gpu` | **one** Ultralytics YOLO `.pt` on CUDA; the engine's cameras go into **one batched `predict()`** per loop (the original GPU pipeline) | `MAX_CAMERAS_PER_ENGINE` (6) |
| `DETECTION_CPU_MODEL=openvino_fp32` | **one Ultralytics YOLO instance per camera** on the OpenVINO FP32 model, `device="intel:cpu"`; all cameras' frames run **in parallel** | `CPU_MAX_CAMERAS_PER_ENGINE` (16) |
| `DETECTION_CPU_MODEL=openvino_int8` | same, on the INT8 model (box branch FP32) + the benchmark's **INT8 duplicate fix** in Ultralytics' postprocess (`INT8_FIX` in `config.py`) | same |
| `DETECTION_CPU_MODEL=onnx` | **one ONNX Runtime session per camera**, CPU threads split between them (`cpu_count ÷ cameras`, rebuilt when a camera is added), own letterbox + OpenCV NMS | same |

These are the deployments that measured best in the multi-stream
benchmark (`tools/bench_multistream.py`), implemented the same way
(`detector/src/inference_backends.py`):

- **CPU instances are created as cameras are added** (about 0.3 s each
  after the first). No camera count to configure.
- **All CPU cameras share one engine process**, as in the benchmark.
  Keep `CPU_MAX_CAMERAS_PER_ENGINE` at the capacity the benchmark
  measured: a second CPU engine would compete for the same cores.
- **OpenVINO never runs on AUTO / the iGPU.** It uses `intel:cpu`; at
  startup the engine logs `EXECUTION_DEVICES=['CPU']` and recompiles on
  CPU if it isn't.
- **ONNX** names `CPUExecutionProvider` explicitly.
- **Fallback:** if a CPU model is missing or fails to load, the engine
  logs an ERROR and runs the `.pt` with Ultralytics on CPU (slower,
  same detections). `BACKEND_FALLBACK_TO_PT = False` makes that a hard
  failure.
- **GPU without CUDA:** `DETECTION_DEVICE=gpu` on a box without a GPU
  logs an ERROR and runs the CPU pipeline (`STRICT_DEVICE = True` in
  `config.py` makes it a startup error).
- **OpenVINO / ONNX Runtime versions:** pin `OPENVINO_VERSION` /
  `ONNXRUNTIME_VERSION` in `.env` to the venv the models were exported
  and benchmarked with, then `docker compose build plate_detector`.
  Export and run with the same OpenVINO.


## Real-time capacity (CPU)

**Rule:** real-time means every camera is served at ≥ `REALTIME_MIN_FPS`
(25). With N = `DETECT_EVERY_N_FRAMES`, one inference loop for n cameras
must finish within

```
budget = N × 1000 / REALTIME_MIN_FPS × CAPACITY_SAFETY_MARGIN   →   1 × 40 ms × 0.7 = 28 ms
```

The margin leaves room for what the measurement leaves out (RTSP decoding,
tracking, OCR hand-off). Loop time is not linear in n (model instances
compete for the cores), so it is measured, not computed.

**At startup, engine 0 calibrates** (CPU engines, before any camera is
attached): it runs the real inference path for 1, 2, 3, … cameras on dummy
frames and prints the table. The largest n that fits the budget, with all
smaller counts also fitting, is the machine's real-time capacity for the
chosen model (`openvino_fp32`, `openvino_int8` or `onnx` each get their own):

```
🧪 [CAPACITY] engine=0 cpu/openvino_fp32 | calibrating real-time capacity on Intel(R) Core(TM) i7-12700K (20 logical cores)
🧪 [CAPACITY] real-time = ≥25 fps per camera → one loop must finish in 28.0 ms (40.0 ms frame period × 0.70 safety margin) ...
🧪 [CAPACITY] engine=0 cpu/openvino_fp32 |  1 camera : loop avg   11.2 ms  p95   13.0 ms  →  76.9 fps/camera  ✅ real-time
🧪 [CAPACITY] engine=0 cpu/openvino_fp32 |  4 cameras: loop avg   22.0 ms  p95   25.1 ms  →  39.8 fps/camera  ✅ real-time
🧪 [CAPACITY] engine=0 cpu/openvino_fp32 |  5 cameras: loop avg   31.5 ms  p95   36.0 ms  →  27.8 fps/camera  ❌ too slow (budget 28.0 ms)
🏁 [CAPACITY] engine=0 cpu/openvino_fp32 | ✅ REAL-TIME CAPACITY: 4 cameras at ≥25 fps (model openvino_fp32, input 288x480; measured in 21 s)
```
(numbers are an example)

**Every camera that is attached afterwards is checked against it**
(`backend_bridge`):

```
✅ [CAPACITY] camera 3 attached: 3/4 real-time cameras (openvino_fp32 @ ≥25 fps) — 1 more fit
🚨 [CAPACITY] camera 5 attached: 5 cameras > real-time capacity 4 (openvino_fp32 @ ≥25 fps) — NOT real-time, expect missed frames on every camera ...
```

The profile (capacity, the full table, CPU model, cores, container CPU
limit, budget) is stored in Redis at `plate:internal:detector:capacity`
(look at it in Redis Commander) and shown under `realtime_capacity` in
`GET :8010/health`. Settings are in `detector/src/config.py`, section
*2b. REAL-TIME CAPACITY* (`REALTIME_MIN_FPS`, `CAPACITY_SAFETY_MARGIN`,
`CAPACITY_MAX_CAMERAS_TESTED`, `CAPACITY_ROUNDS`, `CAPACITY_FRAME_SIZE`,
`CAPACITY_CALIBRATION_ENABLED`).

Things to know:
- **It warns, it does not refuse.** A camera over capacity is still
  attached (and logged with 🚨). Rejecting it is a separate decision.
- **Startup takes longer:** about 10–60 s (the first OpenVINO instance
  compiles first). Cameras added meanwhile are attached right after.
- **Measured on an idle machine,** without decoding or OCR: that is what
  the safety margin is for. Confirm with the live `⏱️ [PERF]` numbers
  (`missed %`); if real runs miss frames at the stated capacity, lower
  `CAPACITY_SAFETY_MARGIN` (e.g. 0.6) and restart.
- **Device-wide, measured by engine 0 only.** Engines started later (when
  one engine is full) do not re-measure, because that would run under load.
  Keep all CPU cameras in one engine (`CPU_MAX_CAMERAS_PER_ENGINE`).
- **Over capacity, the engine degrades by itself** (`CAPACITY_AUTO_DEGRADE`,
  `DETECT_EVERY_N_MAX` in `config.py`). It picks the smallest detection
  interval N for which the attached cameras fit the budget (N × 28 ms) and
  logs it; the tracker coasts the frames in between and the lost-track
  window keeps its length in seconds. Cameras stay covered at a lower
  detection rate instead of losing 20–40% of their frames at random. It
  goes back to the configured N when cameras are removed:

  ```
  🐢 [CAPACITY] engine=0: 8 cameras need ~42 ms per loop but real-time at ≥25 fps allows 56 ms (camera 8 added) → detecting every 2nd frame per camera (12.5 detections/s each, tracker coasts the rest)
  🐇 [CAPACITY] engine=0: 4 cameras fit real-time again (camera 5 removed) → detecting every  frame (25.0 detections/s per camera)
  ```
  `False` = warn only. If even `DETECT_EVERY_N_MAX` is not enough the 🐢 line
  says so (use fewer cameras or a lighter model).
- **GPU engines are not calibrated** (the key is cleared).
- **Set `CAPACITY_FRAME_SIZE`** to your cameras' resolution: the resize
  from the camera frame to the model input is part of the cost.


## Logs

Every component logs to stdout (`docker compose logs -f <service>`).
Levels, formats and switches are in each `config.py` (`LOG_LEVEL`,
`LOG_FORMAT` = `text`/`json`; the `LOG_LEVEL` env var still overrides
for a quick debug session).

**Detector performance** (`detector/src/perf_stats.py`, same columns as
the benchmark):

```
⏱️ [PERF] engine=0 cpu/openvino_fp32 camera=3 | fps in=25.0 proc=24.6 | missed=4 (1.6%) coasted=0 |
         pre=1.9 infer=11.8 (p95 14.2) post=0.9 track=2.1 ms/frame | latency=31 (p95 44) ms | dets/frame=0.85 | lifetime missed=1.2%
📊 [STATS] engine=0 cpu/openvino_fp32 cameras=4 | last 50 loops in 2.1s | frames/loop=3.9 | processed=96.2/s |
         missed=6 (1.5%) | infer/frame avg=11.9 p95=14.5 ms | batch avg=13.8 p95=17.0 ms | loop avg=20.4 p95=26.1 ms | ...
```

| Field | Meaning |
|---|---|
| `fps in` / `proc` | frames the camera delivered / frames that went through the model |
| `missed` | frames replaced by a newer one before the engine reached them (dropped) |
| `coasted` | frames skipped on purpose (`DETECT_EVERY_N_FRAMES > 1`) |
| `pre / infer / post` | ms per frame (Ultralytics `r.speed`, own timers for ONNX; a GPU batch is split over its frames) |
| `track` | tracker + triggers + crops + OCR hand-off, ms per frame |
| `latency` | frame arrival → result handled (avg / p95) |
| `batch` / `loop` | one inference call / one whole engine loop |

Intervals: `PERF_LOG_INTERVAL_SEC` (10 s), `STATS_EVERY_N_BATCHES` (50).
`⚠️ [INFER-SLOW]` when a loop exceeds `SLOW_BATCH_WARN_MS`.
Per-track lines (`[TRACK-NEW]`, `🔁 [OCR-SUBMIT]`, `🔎 [OCR-RESULT]`,
`✅ [SATISFIED]`, `🚧 [LINE-CROSS]`, `[TRACK-END]`, …) switch off with
`LOG_TRACK_EVENTS = False`.

**OCR**: `📊 [OCR-STATS]` per worker every `OCR_STATS_LOG_INTERVAL_SEC`
(tasks/s, ok/invalid/error, valid %, processing and queue-wait avg/p95);
per-task lines (`[TASK-START]`, `[CAR-OCR]`, `[VOTE]`, `[VALIDATE]`,
`[TASK-DONE]`) switch off with `LOG_TASK_EVENTS = False`; one-line
decision trace with `LOG_DECISION_TRACE`.

**Visual debug** (`config.py`, local folders, never MinIO):

| Switch | File | Output |
|---|---|---|
| `DEBUG_VIDEO_ENABLED` | `detector/src/config.py` | annotated MP4 per camera in `./debug_video/camera_<id>/` |
| `DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED` | `detector/src/config.py` | JPEG of the crops sent to OCR, `./debug_video/ocr_submissions/` |
| `OCR_SAVE_DECISION_DEBUG` | `ocr_service/src/config.py` | JPEG per OCR decision, `./debug_ocr/decisions/` |

See [DEBUGGING.md](DEBUGGING.md) for what each shows.


## Testing multiple streams

**1. The benchmark (model level, exactly the script used for the
decision).** `detector/tools/bench_multistream.py` is your benchmark
with command-line options. It needs the GPU (pt_gpu is the reference),
so run it on the GPU machine, inside the detector image (which has
ultralytics, openvino, onnxruntime and CUDA torch):

```bash
docker compose run --rm -v "<folder with video2.mp4>:/clips:ro" plate_detector \
    python tools/bench_multistream.py --video /clips/video2.mp4 --streams 4 8 12
# --model-dir /models/plate_v8n_480 (default)   --frames 3000   --variants pt_gpu ov_fp32 onnx
# results: ./debug_video/bench_final/<date_time>/<N>_streams/  (SPEED / ACCURACY tables also printed)
```

Without Docker, the same file runs on the host with the benchmark venv:
`python detector/tools/bench_multistream.py --model-dir <...>\plate_v8n_480 --video <...>\video2.mp4 --results <...>\bench_final --streams 4`.

**2. The whole service with N simulated cameras.** The `test-video`
profile loops `TEST_VIDEO_FILE` into the relay as cameras `1`..`8`:

```bash
# .env:  COMPOSE_PROFILES=standalone-stream,test-video   TEST_VIDEO_FILE=./video2.mp4
docker compose up -d --build
python redis_tools.py set-camera --id 1 --address publisher --roi 0 0 1 1   # "publisher": the test clip is PUSHED into the relay
python redis_tools.py activate --id 1        # repeat for 2, 3, 4 ...
docker compose logs -f plate_detector | grep -E "PERF|STATS|INFER-SLOW"
```

Read `missed %`, `latency` and `infer p95` per camera, and compare them
with the benchmark's SPEED table for the same number of streams. Add
cameras until `missed` rises: that is the box's capacity. Put it in
`CPU_MAX_CAMERAS_PER_ENGINE` (CPU) or `MAX_CAMERAS_PER_ENGINE` (GPU).


## Production

1. **Start from the production template:**
   ```bash
   cp .env.prod.example .env
   ```
   Every value the backend/DevOps team must provide is `[FILL_IN]`
   (Redis host/port/db/password/URL, MinIO endpoint/keys/buckets/public
   URL, the system relay URL, the shared Docker network, image tag,
   OpenVINO/ONNX Runtime versions). Search for `[FILL_IN]` and replace
   all of them. The device/model lines are ours to set.
2. **Production runs only** `plate_detector`, `plate_ocr` and
   `control_hub`. `COMPOSE_PROFILES` is empty, so the platform's Redis,
   MinIO and shared RTSP relay are used (see the next section).
3. Put the models in place (see "Models"), then:
   ```bash
   docker compose up -d --build
   docker compose ps
   curl localhost:8010/health ; curl localhost:8011/health ; curl localhost:8021/health
   docker compose logs plate_detector | grep -E "MODEL|EXECUTION_DEVICES|FAILED|FALLING BACK"
   ```
   Expect the right `[MODEL]` line, `EXECUTION_DEVICES=['CPU']` on CPU,
   and no `FAILED` / `FALLING BACK`.


## Put this service in its own folder (e.g. `C:\Users\eyerik.com\Desktop\plate-service`)

The service is the `plate-service/` folder of the repo. Two ways:

**A. Git clone, then use only `plate-service/`** (recommended: you can `git pull` updates):

```bat
cd C:\Users\eyerik.com\Desktop
git clone -b armin_claude_plate https://github.com/Ampmalekpour/EyePass-AI.git eyepass-ai
:: the service is now in C:\Users\eyerik.com\Desktop\eyepass-ai\plate-service
```

To have exactly `C:\Users\eyerik.com\Desktop\plate-service` with only
the plate service in it (sparse checkout):

```bat
cd C:\Users\eyerik.com\Desktop
git clone --no-checkout -b armin_claude_plate https://github.com/Ampmalekpour/EyePass-AI.git plate-service-repo
cd plate-service-repo
git sparse-checkout set plate-service
git checkout armin_claude_plate
:: service folder: C:\Users\eyerik.com\Desktop\plate-service-repo\plate-service
:: later updates:  git pull
```

**B. Copy the folder** (no git in the target): download the branch as
ZIP from GitHub (branch `armin_claude_plate` → Code → Download ZIP) and
copy its `plate-service\` contents into
`C:\Users\eyerik.com\Desktop\plate-service`.

Then, in the service folder:

```bat
copy .env.example .env            :: or .env.prod.example for production
:: put the models:  models\detection\plate_v8n_480\...   models\detection\plate_v8s_640\...   models\ocr\...
docker network create eyeplate_net
docker compose up -d --build
```

Everything the service needs is inside that folder (`compose*.yaml`,
`.env*`, `detector/`, `ocr_service/`, `control-hub/`, `camera-service/`,
`common/`, `models/`, `redis_tools.py`). It does not use anything from
the other modules.


## Using the system's streamer instead of the standalone one

**What the standalone streamer is.** With `COMPOSE_PROFILES=standalone-stream`
this module runs its **own** RTSP relay: `plate_mediamtx` (MediaMTX) and
`plate_camera_stream`. `camera_stream` reads `plate:cameras:config`,
registers each camera in MediaMTX as path `<camera_id>`, and reports
online/offline on `plate:camera:events` and `plate:cameras:details`.
That is right for a laptop or a single-module test. On a full EyePass
deployment the system already runs **one shared** MediaMTX +
camera_stream for every module (it scans `*:cameras:config`, so it
already serves the plate cameras). A second relay would open a second
connection to every camera.

The detector needs three things from any streamer:

1. frames at `MTX_RTSP_BASE_URL/<camera_id>`;
2. online/offline events on Redis channel `plate:camera:events`
   (singular `camera`) and details in `plate:cameras:details`, in the
   Redis the detector uses;
3. a Docker network where the relay's container name resolves.

**Switch:**

1. **Find the system's names** (relay container, its network, its Redis):
   ```bash
   docker ps --format '{{.Names}}\t{{.Image}}' | grep -iE 'mediamtx|camera|redis|minio'
   docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' <system-mediamtx-container>
   ```
2. **Edit `.env`:**
   ```ini
   COMPOSE_PROFILES=                                    # drop standalone-stream
   SHARED_NETWORK=<the system's network>
   MTX_RTSP_BASE_URL=rtsp://<system-mediamtx-container>:8554
   REDIS_HOST=<system redis container>                  # the SAME Redis as the system camera_stream + backend
   REDIS_URL=redis://<system redis container>:6379/0
   MINIO_ENDPOINT=http://<system minio container>:9000
   TEST_VIDEO_RTSP_BASE_URL=rtsp://<system-mediamtx-container>:8554   # only with test-video
   ```
   Or keep `eyeplate_net` and attach the system containers to it:
   `docker network connect eyeplate_net <system-mediamtx-container>`.
3. **Remove the standalone containers** (no longer in the profiles, so
   `up` would leave them running): `docker rm -f plate_mediamtx plate_camera_stream`
4. **Start:** `docker compose up -d`.
5. **Verify:**
   ```bash
   docker exec plate_detector python -c "import cv2; c=cv2.VideoCapture('rtsp://<system-mediamtx-container>:8554/<camera_id>'); print(c.read()[0])"
   docker exec <system redis container> redis-cli SUBSCRIBE plate:camera:events
   docker compose logs -f plate_detector | grep -E "CAMERA-ADD|PERF"
   ```
   No frames in `[PERF]`: the relay path must equal the camera id, and
   `MTX_RTSP_BASE_URL` must use the container name, not `localhost`.

**Back to standalone:** put `standalone-stream` back in
`COMPOSE_PROFILES`, set `MTX_RTSP_BASE_URL=rtsp://mediamtx:8554`, then
`docker compose up -d`.


## Upgrading an existing `.env`

`.env` got much shorter: copy `.env.example` (or `.env.prod.example`)
to `.env` and re-enter your Redis/MinIO/relay values. Variables removed
from `.env` are now set in `config.py` and are ignored if left in
`.env` (the containers no longer read `.env` directly). Renamed:
`DETECTION_MODEL` → `DETECTION_GPU_MODEL`, `DETECTION_BACKEND` +
`DETECTION_PRECISION` → `DETECTION_CPU_MODEL`, `OCR_USE_GPU` →
`OCR_DEVICE`. Models move into per-model folders (see "Models").
Rebuild once: `docker compose build`.


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
   the weights in their model folder, see "Models", same `PadOcr/`
   folder as above).
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
**66/66 passing**) covering:

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

- `detector.model_files` / `inference_backends` / `perf_stats` —
  manifest-first model lookup (per-model folders, file-name fallback,
  actionable errors), GPU/CPU runtime planning with model aliases and
  the missing-GPU fallback, the own-ONNX letterbox + OpenCV NMS math,
  the INT8 fix's overlap helper, and the missed/processed/latency
  accounting behind the performance logs.

Model accuracy and real-time capacity are `detector/tools/bench_multistream.py`'s
job (see "Testing multiple streams").

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
