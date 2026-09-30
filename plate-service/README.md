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
                          src/inference_backends.py  GPU (.pt) / CPU (OpenVINO, ONNX) inference
                          src/model_files.py         DETECTION_MODEL -> files, input size, class names
                          tools/parity_test.py       acceptance test: .pt vs CPU backend on a clip
                          tools/export_cpu_models.py .pt -> ONNX + OpenVINO in the expected layout
ocr_service/            plate_ocr service (src/, Dockerfile, requirements.txt)
compose.yaml            plate_detector + plate_ocr + control_hub (+ standalone relay, test clips, autoheal as profiles)
compose.gpu.yaml        GPU reservation for plate_detector (COMPOSE_FILE in .env)
compose.infra.yaml      redis + minio (+ inspection tools, --profile tools)
.env.example            every variable the stack understands
redis_tools.py          manual test harness / CLI — stands in for the backend
tests/                  unit tests (platecore, engine_manager rebalance, triggers, backend_bridge, inference backends)
README.md               this file
```


## Quick start

```bash
docker network create eyeplate_net      # once, ever — skip if it already exists
cp .env.example .env                    # then set the lines below
```

| You want | Set in `.env` |
|---|---|
| **GPU** detection | `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` and `DETECTION_DEVICE=gpu` |
| **CPU** detection | `COMPOSE_FILE=compose.yaml` and `DETECTION_DEVICE=cpu` (OpenVINO FP32 by default) |
| The small, fast model | `DETECTION_MODEL=plate_v8n_480` |
| The larger model | `DETECTION_MODEL=plate_v8s_640` |
| This module's own relay (dev, single module) | `COMPOSE_PROFILES=standalone-stream` |
| The whole system's shared relay | `COMPOSE_PROFILES=` (empty) + see [Using the system's streamer](#using-the-systems-streamer-instead-of-the-standalone-one) |

```bash
docker compose up -d --build            # reads COMPOSE_FILE / COMPOSE_PROFILES from .env
docker compose logs -f plate_detector   # check the [INIT] / [MODEL] / [CPU-TOPOLOGY] lines
```

Everything else in `.env.example` has a working default. The rest of
this section explains each piece.


## What you need to provide

Nothing in this delivery trains or ships model weights: you bind-mount
what you already have.

| What | Where it goes | `.env` variable |
|---|---|---|
| The detection models (layout below) | `${DETECTION_MODELS_DIR}/` (mounted at `/models`) | `DETECTION_MODELS_DIR` (default `./models/detection`) |
| Your whole `PadOcr/` folder, **with these exact subfolder/file names kept** (see below) | `${OCR_MODELS_DIR}/` | `OCR_MODELS_DIR` (default `./models/ocr`) |
| Your base image (already has torch/ultralytics/OpenCV) | Docker build arg | `AI_BASE_IMAGE` (default `base_image_gpu:latest`) |

### The detection models folder

```
${DETECTION_MODELS_DIR}/                       (= /models inside plate_detector)
├── plate_v8n_480.pt                           PyTorch: GPU pipeline, and the CPU fallback
├── plate_v8n_480_288x480.onnx                 ONNX FP32, static 288x480      (DETECTION_BACKEND=onnx)
├── plate_v8n_480_fp32_openvino_model/         OpenVINO FP32                   (DETECTION_BACKEND=openvino, default)
│   ├── plate_v8n_480.xml
│   ├── plate_v8n_480.bin
│   └── metadata.yaml
├── plate_v8n_480_int8_openvino_model/         OpenVINO INT8                   (DETECTION_PRECISION=int8, see below)
│   ├── plate_v8n_480.xml
│   ├── plate_v8n_480.bin
│   └── metadata.yaml
├── export_info.yaml                           input size + class names of the 480 export
├── export_eval.json                           (not read by the detector)
├── plate_v8s_640.pt                           PyTorch: GPU pipeline
└── plate_v8s_640_*                            its CPU exports, once made (tools/export_cpu_models.py)
```

**The names matter.** `DETECTION_MODEL` is the base name, and every
file is found from it (`detector/src/model_files.py`):

| Pipeline | File used for `DETECTION_MODEL=<name>` |
|---|---|
| GPU, or `DETECTION_BACKEND=pt` | `<name>.pt` |
| CPU, `openvino` + `fp32` | `<name>_fp32_openvino_model/<name>.xml` |
| CPU, `openvino` + `int8` | `<name>_int8_openvino_model/<name>.xml` |
| CPU, `onnx` | `<name>_<H>x<W>.onnx` (or `<name>.onnx`) |

**Input size follows the model.** You don't set it.

- GPU: `imgsz` is the trailing number of the name, so `plate_v8n_480`
  runs at 480 and `plate_v8s_640` runs at 640. Ultralytics letterboxes
  each ROI to that size, as before. `DETECTION_IMG_SIZE` stays `0`.
  Setting it forces another size and logs a warning.
- CPU: the model's own input shape (e.g. 288x480) is read from the
  graph, `metadata.yaml`, `<name>_export_info.yaml` / `export_info.yaml`
  or the ONNX file name, in that order, and logged at startup. The
  shared `export_info.yaml` is only applied to the model whose size it
  describes, so the 480 file is never used for `plate_v8s_640`. Class
  names are read the same way.

**`plate_v8s_640` on the CPU** needs its own exports. Make them inside
the detector image, so they come out with the same OpenVINO version
the detector runs:

```bash
docker compose run --rm -v "$PWD/models/detection:/export" plate_detector \
    python tools/export_cpu_models.py --model /export/plate_v8s_640.pt --out /export
# -> plate_v8s_640_384x640.onnx, plate_v8s_640_fp32_openvino_model/, plate_v8s_640_export_info.yaml
```

Ultralytics installs `onnx`/`onnxslim` on the first export if the
image lacks them, which needs internet access; otherwise
`pip install onnx onnxslim` in the image first. The default shape is
16:9 rounded up to a multiple of 32 (`480 → 288x480`,
`640 → 384x640`); use `--input-size HxW` to override it. The script
exports with the benchmarked settings (`opset=17, simplify=True,
dynamic=False, batch=1, half=False`, no NMS in the graph, then
`ov.convert_model` + `ov.save_model(compress_to_fp16=False)`). It
converts OpenVINO from a *dynamic* ONNX export so the model can be
compiled per camera ROI; see [ROIs and input shapes](#rois-and-input-shapes).

The OCR folder must contain, **exactly as named** (read by
`ocr_service/src/config.py`):

```
${OCR_MODELS_DIR}/
├── en_PP-OCRv3_det_infer/       (shared detector model, car + motorcycle)
├── rec_svrt_fa_final_1/         (car plate recognizer)
├── ch_ppocr_mobile_v2.0_cls_infer/   (shared angle classifier)
├── rec_svrt_motor/              (motorcycle plate recognizer)
└── Final_Dict.txt               (shared character dictionary)
```

Both folders are plain bind mounts, never baked into the image, so a
model swap needs no rebuild. Replace the files and run
`docker compose restart plate_detector`. Switching `DETECTION_MODEL`,
`DETECTION_DEVICE` or `DETECTION_BACKEND` also only needs a restart
(`docker compose up -d`), except GPU ↔ CPU, which also changes
`COMPOSE_FILE`.


## GPU or CPU

One variable picks the pipeline, and a matching `COMPOSE_FILE` adds or
drops the NVIDIA reservation:

```ini
# GPU host
COMPOSE_FILE=compose.yaml:compose.gpu.yaml
DETECTION_DEVICE=gpu            # or cuda:1 for a specific GPU

# CPU host
COMPOSE_FILE=compose.yaml
DETECTION_DEVICE=cpu
DETECTION_BACKEND=openvino      # default; onnx or pt also possible
DETECTION_PRECISION=fp32        # default
CPU_EXPECTED_CAMERAS=4          # your real camera count (sizes the parallel requests)
```

`DETECTION_DEVICE=auto` picks the GPU when torch sees CUDA, otherwise
the CPU. If you ask for `gpu` and there is none, the detector logs an
ERROR and runs the CPU pipeline. With `STRICT_DEVICE=true` it refuses
to start instead.

### GPU pipeline: unchanged

On the GPU nothing about the architecture changed. Each engine process
loads `<DETECTION_MODEL>.pt` with Ultralytics (`YOLO(path).to("cuda")`)
and runs one batched `model.predict(source=frames, imgsz, conf,
half=False)` per loop over all its cameras, as before. Only the model
file and `imgsz` now come from `DETECTION_MODEL`. `DETECTION_BACKEND`,
`DETECTION_PRECISION` and the other `DETECTION_CPU_*` settings are
ignored on the GPU (with a warning if set).

### CPU pipeline: how it works

```
RTSP relay ─► RTSPStreamReader (per camera) ─► ROI crop
                                                   │
      one engine process, all cameras              ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ per camera frame: letterbox (Ultralytics-exact) → blob         │
      │   → start_async on that camera's OpenVINO infer request       │
      │ wait for all requests → copy outputs → decode + class-aware    │
      │ NMS (IoU 0.7, max 300) → boxes back to ROI pixels              │
      └──────────────────────────────────────────────────────────────┘
                                                   │  [x1,y1,x2,y2,conf,cls] float64
                                                   ▼
                         BYTETracker → triggers → best crops → control hub / OCR
                         (unchanged: same array the GPU path produces)
```

What the detector does on the CPU, and why (`detector/src/inference_backends.py`):

- **The OpenVINO runtime is used directly, not through Ultralytics.**
  Ultralytics 8.3.x compiles OpenVINO models with device `AUTO`, which
  moves inference onto the integrated GPU (measured 2-3× slower, plus a
  ~200 ms stall at the switch). The detector compiles on the named
  `"CPU"` device, checks `EXECUTION_DEVICES == ['CPU']`, and fails
  loudly otherwise. It also avoids the patched Ultralytics ONNX backend
  that prints `[DEBUG] ... ONNX Inference time` on every frame.
- **ONNX Runtime** always gets `providers=["CPUExecutionProvider"]`
  explicitly, and the detector verifies it.
- **FP32 is the default.** It gives the same detections as PyTorch
  (benchmark: 100% recall, 100% precision, IoU 1.000 over 2000 frames).
  `INFERENCE_PRECISION_HINT=f32` is pinned, so a CPU with bf16/AMX
  never lowers precision silently. FP16 gave no speed-up and is not used.
- **Pre- and post-processing reproduce Ultralytics exactly.** Letterbox
  uses `INTER_LINEAR` resize and centred 114 padding; the blob is
  BGR→RGB, `/255`, NCHW float32. The output `(1, 4+nc, N)` is decoded
  as cx,cy,w,h plus sigmoided class scores, then `conf > DETECTION_CONF_THRESHOLD`,
  class-aware NMS (7680 px class offset, IoU `DETECTION_IOU_THRESHOLD=0.7`,
  max `DETECTION_MAX_DET=300`), and the boxes are mapped back through
  the letterbox and clipped.
- **Cameras run in parallel, not as one serial batch.** The engine
  compiles one model with `PERFORMANCE_HINT=THROUGHPUT`,
  `NUM_STREAMS = cameras on the engine`, and one infer request per
  camera. Every camera's frame is started asynchronously and the engine
  waits for all of them. Outputs are copied before a request is reused,
  because a request's output buffer is overwritten by its next run.
- **Threads.** The detector does not force `INFERENCE_NUM_THREADS` to
  the logical core count, which measured 2× slower (15 ms vs 7.2 ms,
  work spread onto E-cores and hyper-threads); OpenVINO chooses. PyTorch
  does no work on this path, so `torch.set_num_threads` is not called,
  and OpenCV is capped at `cv2.setNumThreads(2)` so it doesn't compete
  with OpenVINO.
- **Spawn-safe.** OpenVINO and ONNX Runtime are imported, and the model
  compiled, inside each engine child process, never in the parent.
- **Warm-up and cache.** Each request is warmed up
  (`DETECTION_WARMUP_RUNS`) at the real input shape, and compiled
  models are cached in `OPENVINO_CACHE_DIR` for faster restarts.
- **Fallback, never silent.** If the ONNX/OpenVINO model is missing or
  fails to compile, the engine logs an ERROR with the reason and runs
  `<DETECTION_MODEL>.pt` with PyTorch on the CPU (slower, still
  correct). Set `DETECTION_BACKEND_FALLBACK=false` to make that a hard
  failure instead.

#### ROIs and input shapes

Each camera's ROI is cropped before detection, so frames reach the
model in different aspect ratios.

- **`DETECTION_CPU_SHAPE_MODE=roi` (default, OpenVINO).** Every ROI is
  letterboxed exactly the way Ultralytics does it for the `.pt` model:
  ratio `min(size/h, size/w)`, then the smallest 32-aligned rectangle.
  A full 1920x1080 frame at 480 gives 288x480, the exported shape; a
  960x1080 ROI gives 480x448. The model is reshaped and compiled
  **once** per distinct ROI shape (logged as `[SHAPE] compiled ...`,
  about 0.2 s). So the CPU sees exactly the pixels the GPU path sees,
  and the detections match for every ROI. This needs an OpenVINO model
  converted from a *dynamic* ONNX export, which
  `tools/export_cpu_models.py` produces. A **static** export (anchors
  baked in for one size) cannot be reshaped: the detector then logs a
  `[SHAPE]` warning and switches to `fixed`.
- **`DETECTION_CPU_SHAPE_MODE=fixed`.** Every ROI is letterboxed into
  the model's one static shape, and the padding absorbs the aspect
  ratio. This is fine for full frames and wide ROIs. A tall ROI gets
  far fewer pixels than on the GPU (a 960x1080 ROI is scaled 0.27×
  instead of 0.44×). The ONNX backend always works this way, because
  the export is static, and logs a `[SHAPE]` warning for such ROIs.

Measured here, with YOLOv8n and the same clip through `.pt` and
OpenVINO FP32 (`tools/parity_test.py`): full frame and every ROI tried
gave 100% / 100% / IoU 1.0000 in `roi` mode. With a static model in
`fixed` mode, full frames still match exactly, but a portrait ROI fell
to about 89% recall.

#### Concurrency: one engine for all cameras

Keep `CPU_ENGINE_MODE=single`: one engine process runs OpenVINO for
all cameras, with one stream and request per camera. Set
`CPU_EXPECTED_CAMERAS` to your camera count. `DETECTION_CPU_STREAMS=0`
follows it, and `[CPU-TOPOLOGY]` logs the plan at startup.

If several engine processes run (`multi`/`auto` mode, or more cameras
than `MAX_CAMERAS_PER_ENGINE`), the detector sets
`INFERENCE_NUM_THREADS` to physical cores ÷ engines, so the engines
share the cores instead of each claiming all of them. Under a Docker
CPU quota, set `CPU_CORES_OVERRIDE` to the same number as
`DETECTOR_CPU_LIMIT`; the threads are then split from that budget. The
ONNX backend uses one session per camera, with
`intra_op_num_threads = cores ÷ sessions` and `inter_op_num_threads = 1`,
because sessions don't coordinate threads with each other.

#### Capacity and load

Benchmark (i7-12700K, 4 streams, frames/s the CPU can process):

| Model / input | PyTorch | ONNX | OpenVINO FP32 | OpenVINO INT8 |
|---|---|---|---|---|
| YOLOv8s 640x384 | 25 | 53 | 49 | 145 |
| YOLOv8n 640x384 | | | ≈168 (est.) | |
| YOLOv8n 480x288 | | | ≈280 (est.) | |

Keep `cameras × camera fps ÷ DETECT_EVERY_N_FRAMES` below about
60-70% of your measured capacity, so RTSP decoding, tracking, JPEG
encoding, Redis and OCR (if it shares the PC) still have headroom. For
example, 4 cameras × 25 fps = 100 inferences/s is about 36% of
`plate_v8n_480`'s ~280/s. `plate_v8s_640` (~49/s at 384x640 FP32) needs
`DETECT_EVERY_N_FRAMES=4` for the same cameras (25/s, about 50%). The
detector logs this planned load at startup.

The `[STATS]` line (every 15 batches) now shows the backend, inference
average and **p95**, and loop average and p95 (numbers are an example):

```
📊 [STATS] engine=0 cpu/openvino-fp32 last 15 batches | sources=4.0 | infer avg=14.2ms p95=16.0ms | loop avg=21.3ms p95=24.9ms | ...
```

A p95 far above the average usually means some requests landed on
E-cores. Try `OPENVINO_SCHEDULING_CORE_TYPE=PCORE_ONLY` or fewer
streams, then measure again.

#### INT8: only through the accuracy gate

INT8 is **not** accurate enough for plates yet. Against PyTorch it
reached 90% precision, average IoU 0.89 and worst 0.67, and its crops
go to OCR. `DETECTION_PRECISION=int8` works, but logs a warning. Adopt
it only after:

1. Quantizing with `nncf.quantize_with_accuracy_control` on the
   labelled validation set, calibrated on varied training images (not
   one video), ideally with the detection head kept in FP32.
2. Validation mAP50-95 dropping by less than 0.01, and not at all for
   `motorcycle_plate`.
3. Re-tuning `TRACKER_TRACK_THRESH` and the other confidence
   thresholds: INT8 shifted confidences by 0.035 on average.

#### Versions

Use one pinned environment for export, benchmark and production.
OpenVINO and ONNX Runtime are installed in the detector image from
`.env`:

```ini
OPENVINO_VERSION=<exact version from `pip freeze` of the benchmark venv>
ONNXRUNTIME_VERSION=<same>
```

Then rebuild with `docker compose build plate_detector`. Empty means
latest, which is only for a first try. Export and run with the same
OpenVINO version: a newer runtime reads older IR files, but not the
other way round. Re-export after upgrading OpenVINO, and re-run the
parity test after upgrading Python, OpenVINO or ONNX Runtime. If the
base image already contains `onnxruntime-gpu`, the plain `onnxruntime`
package is skipped. The startup log prints the OpenVINO version in use.

### Acceptance tests before rollout (CPU)

**1. Parity.** Run the same recorded clip through PyTorch and the CPU
backend inside the detector image, with the same env and ROI crop as
the service:

```bash
docker compose run --rm -v "$PWD/clips:/clips:ro" plate_detector \
    python tools/parity_test.py --video /clips/gate.mp4 --candidate openvino --streams 4 \
    --roi 0 0.3 1 0.7          # optional: a camera's ROI (x y w h, fractions)
# --candidate onnx | --precision int8 | --frames 2000 | --save /clips/out
```

It matches detections by same class and IoU ≥ 0.5. It **passes** at
recall and precision ≥ 99.5% and average IoU ≥ 0.98; FP32 should give
100% / 100% / 1.000. It prints ms/frame for both backends and exits
non-zero on failure.

**2. Capacity.** Run 4 real RTSP streams for 60 minutes with the full
pipeline, OCR included. Watch per-camera dropped frames
(`frames_skipped` in `[STATS]`, `skip_rate` in `[PIPELINE]`), infer
p95, CPU below about 70% (`docker stats`), and the CPU temperature.

**3. Log checks.**

```bash
docker compose logs plate_detector | grep "EXECUTION_DEVICES"   # every engine: ['CPU']
docker compose logs plate_detector | grep -c "\[DEBUG\]"        # 0
docker compose logs plate_detector | grep -E "FAILED|FALLING BACK"  # nothing
```

### Detector environment reference (model and compute)

| Variable | Default | Meaning |
|---|---|---|
| `COMPOSE_FILE` | `compose.yaml:compose.gpu.yaml` | GPU host: include `compose.gpu.yaml`. CPU host: `compose.yaml` only |
| `DETECTION_DEVICE` | `gpu` | `gpu` / `cuda:N` / `cpu` / `auto` |
| `STRICT_DEVICE` | `false` | `true`: a missing GPU is a startup error instead of a CPU fallback |
| `DETECTION_MODELS_DIR` | `./models/detection` | host folder with the models, mounted at `/models` |
| `DETECTION_MODEL` | `plate_v8n_480` | `plate_v8n_480` / `plate_v8s_640`; picks files and input size |
| `DETECTION_MODEL_PATH` | *(empty)* | explicit in-container file for non-standard names (`.pt` for pt, `.onnx`/`.xml` for the CPU backends) |
| `DETECTION_IMG_SIZE` | `0` | `0` = the model's size; otherwise forces the square PyTorch size |
| `DETECTION_CONF_THRESHOLD` | `0.25` | detection confidence, all backends |
| `DETECTION_BACKEND` | `openvino` | CPU only: `openvino` / `onnx` / `pt` |
| `DETECTION_PRECISION` | `fp32` | CPU/openvino only: `fp32` / `int8` (see the gate above) |
| `DETECTION_CPU_SHAPE_MODE` | `roi` | `roi` = per-ROI shapes like the GPU path; `fixed` = one static shape |
| `DETECTION_CPU_INPUT_SIZE` | *(empty)* | `HxW` forces one input shape (implies `fixed`) |
| `DETECTION_CPU_STREAMS` | `0` | parallel requests per engine; `0` = cameras per engine |
| `DETECTION_CPU_THREADS` | `0` | OpenVINO `INFERENCE_NUM_THREADS` / ORT intra-op threads; `0` = auto |
| `OPENVINO_SCHEDULING_CORE_TYPE` | *(empty)* | `PCORE_ONLY` etc. for hybrid CPUs |
| `OPENVINO_CACHE_DIR` | `/data/openvino_cache` | compiled-model cache (empty = off) |
| `DETECTION_IOU_THRESHOLD` / `DETECTION_MAX_DET` | `0.7` / `300` | NMS, Ultralytics' defaults |
| `DETECTION_WARMUP_RUNS` | `3` | warm-up inferences per request |
| `DETECTION_BACKEND_FALLBACK` | `true` | ONNX/OpenVINO failure: ERROR log + PyTorch on CPU (`false` = fail) |
| `INSTALL_CPU_RUNTIMES`, `OPENVINO_VERSION`, `ONNXRUNTIME_VERSION` | `true`, *(latest)*, *(latest)* | build args: CPU runtimes in the image, **pin them** |
| `CPU_ENGINE_MODE`, `CPU_EXPECTED_CAMERAS`, `CPU_CORES_OVERRIDE`, `DETECTOR_CPU_LIMIT` | `single`, `1`, `0`, `0` | engine topology and CPU budget (comments in `.env.example`) |
| `DETECT_EVERY_N_FRAMES` | `1` | detect on 1 in N frames; the tracker coasts the rest |
| `CV2_NUM_THREADS` | `0` | `0` = 2 on openvino/onnx |
| `TORCH_NUM_THREADS` | `0` | only when PyTorch computes on the CPU (`pt` / fallback) |


## Run it

```bash
docker network create eyeplate_net        # once, ever — skip if it already exists
cp .env.example .env                      # then set GPU/CPU, model, streamer (Quick start)

# Only if the platform doesn't already run a shared Redis/MinIO:
docker compose -f compose.infra.yaml up -d
docker compose -f compose.infra.yaml --profile tools up -d   # + RedisInsight/Redis Commander

docker compose up -d --build              # plate_detector + plate_ocr + control_hub
                                          # (+ mediamtx + camera_stream with the standalone-stream profile)
```

`docker compose` reads `COMPOSE_FILE`, `COMPOSE_PROFILES` and
`COMPOSE_PATH_SEPARATOR` from `.env`, so one command covers every
combination. The equivalent by hand is
`docker compose -f compose.yaml -f compose.gpu.yaml --profile standalone-stream up -d`.
Add `test-video` to `COMPOSE_PROFILES` to loop `TEST_VIDEO_FILE` into
the relay as cameras `1`..`6`.

Both services start idle automatically, then self-heal to whatever
phase they were last in (see "Self-healing" below). After a *fresh*
`docker compose up` with an empty Redis, both come up idle and wait:
`plate_detector` for the backend's first `activated` command per
camera (as in the reference pipeline), and `plate_ocr` for the first
task. If you're migrating an **already-running** deployment (cameras
already marked active in the existing `plate:cameras:config`), see
"Migrating from the existing single-process deployment" below.


## Using the system's streamer instead of the standalone one

**What the standalone streamer is.** With `COMPOSE_PROFILES=standalone-stream`,
this module runs its **own** RTSP relay: `plate_mediamtx` (MediaMTX)
and `plate_camera_stream`. `camera_stream` reads
`plate:cameras:config`, registers each camera in MediaMTX as path
`<camera_id>`, and reports online/offline on `plate:camera:events` and
`plate:cameras:details`. That is right for a laptop or a single-module
test. On a full EyePass deployment, the system already runs **one
shared** MediaMTX + camera_stream for every module. That shared
camera_stream scans `*:cameras:config`, so it already serves the plate
cameras. Running a second relay would open a second connection to
every camera.

The detector only needs three things from any streamer:

1. frames at `MTX_RTSP_BASE_URL/<camera_id>`
   (`backend_bridge.rtsp_url()`);
2. online/offline events on the Redis channel `plate:camera:events`
   (singular `camera`) and details in `plate:cameras:details`, in the
   Redis the detector uses;
3. a Docker network where the relay's container name resolves.

**Switch to the system's streamer:**

1. **Find the system's names.** You need the relay's container name,
   the Docker network it is on, and the Redis it uses:
   ```bash
   docker ps --format '{{.Names}}\t{{.Image}}' | grep -iE 'mediamtx|camera|redis|minio'
   docker inspect -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' <system-mediamtx-container>
   ```
2. **Edit `.env`:**
   ```ini
   COMPOSE_PROFILES=                                  # drop standalone-stream (keep test-video/watchdog if used)
   SHARED_NETWORK=<the system's network>              # or keep eyeplate_net and run the `docker network connect` below
   MTX_RTSP_BASE_URL=rtsp://<system-mediamtx-container>:8554
   REDIS_HOST=<system redis container>                # the SAME Redis the system camera_stream + backend use
   REDIS_PORT=6379
   REDIS_URL=redis://<system redis container>:6379/0
   MINIO_ENDPOINT=http://<system minio container>:9000
   REDIS_MODULE=plate                                 # unchanged: the system camera_stream must see plate:cameras:config
   TEST_VIDEO_RTSP_BASE_URL=rtsp://<system-mediamtx-container>:8554   # only if you use test-video
   ```
   To keep `eyeplate_net` instead, attach the system containers to it:
   `docker network connect eyeplate_net <system-mediamtx-container>`
   (the same for its Redis/MinIO if they're not reachable yet).
3. **Remove the standalone containers.** They are not in the active
   profiles any more, so `docker compose up` would leave them running:
   ```bash
   docker rm -f plate_mediamtx plate_camera_stream
   ```
4. **Start:** `docker compose up -d`. Only `plate_detector`,
   `plate_ocr` and `control_hub` (plus any other profiles you kept)
   start. The detector has no `depends_on` on a relay; it reconnects to
   whatever `MTX_RTSP_BASE_URL` points at.
5. **Verify:**
   ```bash
   # the relay is reachable from the detector and serves the camera path
   docker exec plate_detector python -c "import cv2,sys; c=cv2.VideoCapture('rtsp://<system-mediamtx-container>:8554/<camera_id>'); print(c.read()[0])"
   # the system camera_stream publishes plate camera events
   docker exec <system redis container> redis-cli SUBSCRIBE plate:camera:events
   # the detector attached the camera and is reading frames
   docker compose logs -f plate_detector | grep -E "CAMERA-ADD|PIPELINE|STATS"
   ```
   If `[PIPELINE]` shows no frames: the path name must equal the
   camera id, and `MTX_RTSP_BASE_URL` must use the container name, not
   `localhost`.

The ports this module published for its relay (`RTSP_PORT`,
`WEBRTC_PORT`, `HLS_PORT`, `MTX_API_PORT`, `RTMP_PORT`) are no longer
used, so they can't clash with the system's.

**Back to standalone:** put `standalone-stream` back in
`COMPOSE_PROFILES`, set `MTX_RTSP_BASE_URL=rtsp://mediamtx:8554` (and
`TEST_VIDEO_RTSP_BASE_URL`), then `docker compose up -d`.


## Upgrading an existing `.env`

These changes need edits to an existing `.env`. Compare it with
`.env.example`.

- **Add at the top:** `COMPOSE_FILE`, `COMPOSE_PATH_SEPARATOR=:` and
  `COMPOSE_PROFILES=standalone-stream`. Without `COMPOSE_FILE`, the GPU
  reservation is not applied, because it moved from `compose.yaml` to
  `compose.gpu.yaml`. Without `COMPOSE_PROFILES`, the standalone
  mediamtx/camera_stream no longer start, because they are now a
  profile.
- **Models:** rename `best.pt` to `plate_v8n_480.pt` (or whichever it
  is), set `DETECTION_MODEL`, clear `DETECTION_MODEL_PATH`, and set
  `DETECTION_IMG_SIZE=0`. An old `DETECTION_MODEL_PATH=/models/best.pt`
  still works for the GPU/pt path. The CPU backends ignore a `.pt` path.
- **New:** the CPU pipeline block (`DETECTION_BACKEND` …
  `DETECTION_BACKEND_FALLBACK`), `OPENVINO_VERSION` /
  `ONNXRUNTIME_VERSION` / `INSTALL_CPU_RUNTIMES` and
  `TEST_VIDEO_RTSP_BASE_URL`.
- **Rebuild** the detector image once for the CPU runtimes:
  `docker compose build plate_detector`.


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
   the same weights renamed to `plate_v8n_480.pt` or pointed at with
   `DETECTION_MODEL_PATH`, same `PadOcr/` folder as above).
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
**71/71 passing**) covering:

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

- `detector.model_files` / `inference_backends` / `cpu_topology` —
  `DETECTION_MODEL` → file per backend/precision (and actionable errors
  for a missing variant), input size from the name / metadata /
  `export_info.yaml` (never the 480 file for the 640 model), GPU/CPU
  runtime planning (GPU always `.pt`, missing GPU → CPU or strict
  error), the Ultralytics-exact letterbox / auto shape / class-aware
  NMS / box mapping, and the OpenVINO stream/thread split.

Parity of the CPU backends against Ultralytics needs real models and
is `detector/tools/parity_test.py`'s job (see "Acceptance tests").

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
