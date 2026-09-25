# EyePass AI

AI modules for the EyePass platform. Each module is an independently
deployable Docker Compose stack that talks to the backend only through
Redis (commands and results) and MinIO (images).

```
face-service/        face detection + tracking + liveness -> AdaFace recognition
plate-service/       vehicle/plate detection + tracking -> PaddleOCR
control-hub/         owns every face/plate track's recognition state; decides what the backend receives
heatmap-service/     people heatmaps
fire-smoke-service/  fire / smoke detection
```

Face and plate each run as four processes: `camera_stream` + `mediamtx`
(the RTSP relay), `*_detector` (GPU), `face_recognizer` / `plate_ocr`, and
`control_hub`. See [`control-hub/README.md`](control-hub/README.md) for
how a track moves between them.


## Set it up on your laptop

### 0. Prerequisites

- Docker Engine + Compose v2 (`docker compose version`).
- For GPU: an NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
  **No GPU?** Delete the `deploy:` block under `face_detector` /
  `plate_detector` in each `compose.yaml`, and set
  `DETECTION_DEVICE=cpu` in `.env`.
- Your prepared **base images**, which already contain
  torch/ultralytics/OpenCV (and PaddleOCR for plate):
  `base_image:latest` (face) and `base_image_gpu:latest` (plate). If
  you have them as tar files, run `docker load -i base_image.tar`. If
  they're named differently, set `AI_BASE_IMAGE` in each module's `.env`.
- Python 3.10+ on the host, only for the helper scripts (`pip install redis requests`).

### 1. Get the code

```bash
git clone https://github.com/Ampmalekpour/EyePass-AI.git
cd EyePass-AI
git checkout claude/eloquent-volta-6hg9mi     # this branch, until it's merged
```

Already cloned? Update with
`git fetch origin && git checkout claude/eloquent-volta-6hg9mi && git pull`.

### 2. Put the models and data in place

Nothing below is in git. Weights and the face gallery are
bind-mounted, so replacing a file only needs a container restart, not a
rebuild.

**Face** (`face-service/`)

| Put this | Here | Used by |
|---|---|---|
| `best1.pt`, the YOLO head + 14-landmark model | `face-service/models/detection/best1.pt` | face_detector |
| `adaface_ir50_cpu.onnx` | `face-service/models/recognition/adaface_ir50_cpu.onnx` | face_recognizer (primary) |
| `adaface_ir50_ms1mv2.ckpt` (PyTorch fallback, optional when USE_ONNX=true) | `face-service/recognizer/pretrained/adaface_ir50_ms1mv2.ckpt` (next to the `warmup.jpg` already there) | face_recognizer |
| MTCNN weights `pnet.npy`, `rnet.npy`, `onet.npy` | `face-service/recognizer/face_alignment/mtcnn_pytorch/src/weights/` | alignment + enrollment |
| The gallery: `brieface.db` + the reference images `c1.jpg, c2.jpg, …` (+ `representations_ir_50.pkl` if you have one), all in **one flat folder** | `face-service/recognizer/gallery/` | `gallery_seed` uploads it to MinIO; every recognizer worker downloads it from there |

**Plate** (`plate-service/`)

| Put this | Here | Used by |
|---|---|---|
| `best.pt`, the YOLO vehicle/plate model | `plate-service/models/detection/best.pt` | plate_detector |
| Your whole `PadOcr/` folder's **contents**, names unchanged: `en_PP-OCRv3_det_infer/`, `rec_svrt_fa_final_1/`, `ch_ppocr_mobile_v2.0_cls_infer/`, `rec_svrt_motor/`, `Final_Dict.txt` | `plate-service/models/ocr/` | plate_ocr |
| *(optional)* a test clip for `--profile test-video` | `plate-service/test_video.mp4` | video_publisher |

Different location? Point `DETECTION_MODELS_DIR`,
`RECOGNITION_MODELS_DIR`, `RECOGNITION_GALLERY_DIR` or `OCR_MODELS_DIR`
in the module's `.env` at it.

The control hub needs no models or files.

### 3. Configure

```bash
cp face-service/.env.example  face-service/.env
cp plate-service/.env.example plate-service/.env
```

Edit at least these:

- `AI_BASE_IMAGE`: your base image tag.
- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`: **must be identical in
  both `.env` files** if both modules share the one MinIO from step 4.
  The two examples ship different defaults.
- `DETECTION_DEVICE`: `auto`, `cpu` or `cuda:0`.

### 4. Start (the shared network and Redis/MinIO once, then the modules)

```bash
docker network create eyeplate_net                                  # once, ever

cd face-service
docker compose -f compose.infra.yaml up -d                          # redis + minio (shared by all modules)
docker compose up -d --build                                        # mediamtx, camera_stream, face_detector,
                                                                     # gallery_seed, face_recognizer, control_hub
cd ../plate-service
docker compose up -d --build                                        # ..., plate_detector, plate_ocr, control_hub
```

`compose.infra.yaml` is the same stack in both folders, so start it
from only one of them. If the platform already runs Redis/MinIO, skip
it and point `REDIS_*` / `MINIO_*` in `.env` at those instead.

**Running face and plate on the same laptop.** Both stacks publish the
same host ports by default. In `plate-service/.env`, move them:

```
RTSP_PORT=8654
WEBRTC_PORT=8989
WEBRTC_ICE_UDP_PORT=8289
HLS_PORT=8988
MTX_API_PORT=9998
RTMP_PORT=2935
DETECTOR_API_PORT=8110
OCR_API_PORT=8111
# HUB_API_PORT is already 8021 for plate, vs 8020 for face
```

### 5. Check it's alive

```bash
docker compose ps                       # everything "running"/"healthy" (detector/recognizer take ~2 min to warm up)
curl localhost:8010/health              # detector
curl localhost:8011/health              # recognizer / OCR
curl localhost:8020/health              # face control hub   (plate: 8021)
```

Point a camera at it and watch the results:

```bash
cd face-service
python3 -c "import redis_tools as rt; rt.ensure_camera_active('1', 'rtsp://user:pass@CAMERA_IP:554/...')"
python3 -c "import redis_tools as rt; rt.watch_results(120)"      # destructive: pops what the backend would read

cd ../plate-service
python3 redis_tools.py set-camera --id 1 --address "rtsp://..." --roi 0 0 1 1 --line 100 400 900 400
python3 redis_tools.py activate --id 1

cd ../control-hub                         # read-only views of the hub
python3 hub_tools.py tracks --module face
python3 hub_tools.py tail   --module plate
python3 hub_tools.py results --module face -n 3
```

Annotated debug videos land in `face-service/debug/` and
`plate-service/debug_video/` (see each module's `DEBUGGING.md`). The
per-track overlay shows the hub's current answer and whether it's
satisfied.

### 6. Tests (no GPU, no models needed)

```bash
pip install redis numpy opencv-python-headless scipy pillow
python3 control-hub/tests/run_all.py            # hub core + end-to-end against redis-server if installed
(cd face-service  && python3 tests/run_all.py)
(cd plate-service && python3 tests/run_all.py)
```


## Updating a running deployment

- Rebuild and restart **all three** of the module's services together
  (`docker compose up -d --build`). The detector and the worker must
  agree on the new task/result routing. A worker that receives a task
  from an old detector (no `track_uid`) still answers it the old way,
  so a short overlap is harmless.
- The first time the control hub starts, it only processes events
  created from that moment on.
- Removed variables (safe to delete from `.env`): face
  `PERIODIC_MODE`, `PERIODIC_FRAME_INTERVAL`, `PERIODIC_TIME_INTERVAL`,
  `PERIODIC_RECOG_CONF_THRESH`; plate `OCR_CONF_SKIP_THRESHOLD`,
  `OCR_FINALIZE_TIMEOUT_SEC`. The replacements are `FACE_*` / `PLATE_*`
  in the control-hub section of `.env.example`.
