# EyePass AI — suite

All EyePass AI modules run as **one Docker Compose project** with a single
shared infrastructure. Turn each module on or off with a compose profile.
This is the `develop-suite` branch. On `develop-standalone`, each module
runs on its own.

```
                         ┌──────────────── shared, always on ────────────────┐
 cameras ──RTSP──►  mediamtx (ONE relay)  ◄── camera_stream (ONE camera manager)
                         │   one path per physical camera       │  reads <m>:cameras:config,
                         │   (cam_<sha1(address)>)              │  writes <m>:cameras:details,
                         ▼                                      │  publishes online/offline per module
      ┌──────────── face ────────────┐ ┌──────── plate ────────┐ ┌─ heatmap ─┐ ┌── fire ───┐
      face_detector  face_recognizer   plate_detector plate_ocr   ai_service    fire_detector
      face_control_hub  gallery_seed   plate_control_hub
      └──────────────────────────────── redis (ONE) · minio (ONE) ─────────────────────────┘
```

| Folder | What |
|---|---|
| `compose.yaml` | The suite: redis, minio, mediamtx, camera_stream, plus optional extras. It includes each module's `compose.yaml` |
| `.env.example` | Shared settings and **which modules run** (`COMPOSE_PROFILES`) |
| `camera-service/` | The single camera manager used by every module (with tests) |
| `face-service/` | Face detection, tracking and liveness → AdaFace recognition, control hub, optional occupancy heatmap |
| `plate-service/` | Vehicle/plate detection and tracking → PaddleOCR, control hub |
| `heatmap-service/` | People-occupancy heatmaps |
| `fire-smoke-service/` | Fire/smoke detection |
| `test-video/` | Loops a local clip into the relay as a fake camera (`test-video` profile) |

## Choosing what runs

Set this in the root `.env`:

```
COMPOSE_PROFILES=face,plate,heatmap,fire     # any combination
```

| Profile | Starts |
|---|---|
| `face` | face_detector, face_recognizer, face_control_hub, gallery_seed |
| `plate` | plate_detector, plate_ocr, plate_control_hub |
| `heatmap` | ai_service |
| `fire` | fire_detector |
| `watchdog` | autoheal (restarts containers whose healthcheck turns unhealthy) |
| `tools` | RedisInsight (:8001), Redis Commander (:8081) |
| `test-video` | 3 looping fake cameras at `rtsp://mediamtx:8554/test1..3` |

redis, minio, mediamtx and camera_stream have no profile, so they always
run. You can also override for one command, for example
`docker compose --profile face up -d`. A module that is off keeps its
data in Redis and MinIO; turning it back on resumes where it was, via
each service's self-healing checkpoint.

## What is shared, and how the modules stay apart

- **Redis:** every key is prefixed with its module (`face:*`, `plate:*`,
  `heatmap:*`, `fire:*`), so modules never read each other's keys.
- **MinIO:** each module keeps its own buckets. `minio_init` creates all
  of them (`MINIO_BUCKETS`).
- **Camera relay:** camera_stream registers **one MediaMTX path per
  physical camera**, `cam_<sha1(address)>`.
  - When face and heatmap watch the same camera, it is pulled from the
    camera once.
  - Camera IDs can't collide: face camera `1` and plate camera `1` are
    different cameras with different addresses, so they get different paths.
  - Detectors compute the same path from the camera's address, using the
    same rule in `common/*/relay.py`. camera_stream also writes it into
    `<module>:cameras:details` as `relay_path` and `stream_url`.
  - A path is removed from MediaMTX only when no module uses that camera any more.
- **Camera events:** each module gets online/offline events on the
  channel it already listens to: `face:cameras:events`,
  `plate:camera:events`, `heatmap:camera:events`, `fire:camera:events`.
- **Settings:** each module's `.env` holds its own tuning. The root `.env`
  is layered on top, so shared values (Redis, MinIO, relay URL, TZ,
  logging) live in one place and always win.
- **Host ports:** every host port is unique across the suite:

  | Service | Port |
  |---|---|
  | face detector | 8010 |
  | face recognizer | 8011 |
  | face hub | 8020 |
  | plate detector | 8110 |
  | plate OCR | 8111 |
  | plate hub | 8021 |
  | heatmap | 8002 |
  | fire | 8012 |
  | Redis | 6379 |
  | MinIO | 9000 / 9001 |
  | RTSP | 8554 |
  | WebRTC | 8889 |
  | MediaMTX API | 9997 |

## Set it up

**0. Prerequisites**
- Docker with **Compose v2.20 or later**: `docker compose version`. Earlier versions lack `include`.
- For GPU: an NVIDIA driver and nvidia-container-toolkit. Without a GPU,
  delete the `deploy:` blocks in the module compose files and set
  `DETECTION_DEVICE=cpu`.
- Your prepared base images, as each module's `AI_BASE_IMAGE` expects:
  `base_image:latest` for face, `base_image_gpu:latest` for the others.

**1. Get the code**
```bash
git clone https://github.com/Ampmalekpour/EyePass-AI.git
cd EyePass-AI && git checkout develop-suite
```

**2. Create the env files** (all five must exist, even for modules you don't run)
```bash
cp .env.example .env
for m in face plate heatmap fire-smoke; do cp $m-service/.env.example $m-service/.env; done
```
Then edit the root `.env`: `COMPOSE_PROFILES`, `MINIO_ROOT_PASSWORD`, and `STREAM_BASE_URL`.

**3. Put models and data in place** (none of this is in git)

| Module | Put this | Here |
|---|---|---|
| face | `best1.pt` | `face-service/models/detection/` |
| face | `adaface_ir50_cpu.onnx` | `face-service/models/recognition/` |
| face | `adaface_ir50_ms1mv2.ckpt` (optional fallback) | `face-service/recognizer/pretrained/` |
| face | MTCNN `pnet.npy`, `rnet.npy`, `onet.npy` | `face-service/recognizer/face_alignment/mtcnn_pytorch/src/weights/` |
| face | gallery: `brieface.db` + `c*.jpg`, one flat folder | `face-service/recognizer/gallery/` |
| plate | `best.pt` | `plate-service/models/detection/` |
| plate | contents of your `PadOcr/` folder (names unchanged) | `plate-service/models/ocr/` |
| heatmap | person or head model, e.g. `yolov8n.pt` | `heatmap-service/models/detection/` |
| fire | `best.pt` | `fire-smoke-service/models/detection/` |
| test-video | a clip | `test-video/video1.mp4` |

**4. Start**
```bash
docker compose up -d --build          # the modules in COMPOSE_PROFILES + shared services
docker compose ps
```

**5. Check it**
```bash
curl localhost:8010/health     # face detector       (plate: 8110, heatmap: 8002, fire: 8012)
curl localhost:8020/health     # face control hub    (plate: 8021)
curl localhost:9997/v3/paths/list    # relay paths — one per physical camera
docker compose logs -f camera_stream
```
Each module's `redis_tools.py` still stands in for the backend. For
example, `cd plate-service && python3 redis_tools.py set-camera ...` and
then `activate`.

## Tests (no GPU or models needed)
```bash
pip install redis httpx numpy opencv-python-headless scipy pillow
python3 camera-service/tests/test_camera_stream.py      # needs redis-server; else skipped
for m in face-service plate-service; do (cd $m && python3 tests/run_all.py && python3 control-hub/tests/run_all.py); done
```

## For the backend
- **Adding or changing a camera** works as before: write
  `<module>:cameras:config`, then publish `<module>:camera:config:updated`.
  camera_stream also rescans every 30 s, so a missed nudge isn't lost.
- **Live view** uses `stream_url` from `<module>:cameras:details`. It is
  now `STREAM_BASE_URL/<relay_path>/` instead of `/<camera_id>/`.
- **Joining the network:** the suite network is `eyeplate_net`
  (`SHARED_NETWORK`). Attach backend containers to it as an external network.
