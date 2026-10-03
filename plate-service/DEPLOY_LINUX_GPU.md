# Deploying the plate service on a Linux server (Docker + NVIDIA GPU)

For whoever installs this service on the server. Written for **Ubuntu
22.04 / 24.04** with an **NVIDIA GPU** (developed and tested on an RTX
4060). For other distributions, follow the NVIDIA links in each step;
the Docker part is the same.

The service is three containers: `plate_detector` (GPU), `plate_ocr`
and `control_hub`. It uses the platform's Redis, MinIO and RTSP relay
(mediamtx + camera_stream); it does not start its own.

---

## 1. What the host needs

| Component | Why | Check |
|---|---|---|
| NVIDIA driver | the GPU itself | `nvidia-smi` |
| Docker Engine + Compose v2 | runs the containers | `docker compose version` |
| NVIDIA Container Toolkit | lets Docker give the GPU to containers | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |

**You do NOT install the CUDA toolkit or cuDNN on the host.** CUDA,
cuDNN and PyTorch are inside the base image (`AI_BASE_IMAGE`). The host
only needs a driver that is **new enough** for the image's CUDA version.
`nvidia-smi` prints `CUDA Version: X.Y` in its header: that is the
highest CUDA the driver supports, and it must be ≥ the image's CUDA
(step 5 shows how to read it). A recent driver (550 or newer) covers
CUDA 12.x.

---

## 2. NVIDIA driver

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices                  # shows the "recommended" driver
sudo ubuntu-drivers install             # installs the recommended one
# or a specific version:  sudo apt install -y nvidia-driver-550
sudo reboot
```

After the reboot:

```bash
nvidia-smi
```

It must list the GPU and show `Driver Version` and `CUDA Version`.

- **Secure Boot** enabled: the installer asks for a password to enroll a
  key (MOK). On the next boot choose *Enroll MOK* and enter it, or the
  driver won't load (`nvidia-smi` says it can't communicate with the
  driver).
- Servers without a display: the `-server` driver packages
  (`nvidia-driver-550-server`) are fine too.

Reference: https://ubuntu.com/server/docs/nvidia-drivers-installation

---

## 3. Docker Engine and Compose

Skip if `docker compose version` already works (Compose **v2**, the
`docker compose` plugin, not the old `docker-compose`).

```bash
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER          # then log out and back in
docker compose version
```

Reference: https://docs.docker.com/engine/install/ubuntu/

---

## 4. NVIDIA Container Toolkit (makes Docker use the GPU)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker   # writes /etc/docker/daemon.json
sudo systemctl restart docker
```

Test — this must print the same table as `nvidia-smi` on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Reference: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

---

## 5. Load and check the base image

The detector and OCR images are built on top of the AI team's base
image (torch + ultralytics + OpenCV + CUDA). It is delivered as a tar
file:

```bash
docker load -i base_image_gpu.tar
docker images | grep base_image          # name:tag -> AI_BASE_IMAGE / OCR_BASE_IMAGE in .env
```

Check that PyTorch inside it sees the GPU, and which CUDA it was built
for:

```bash
docker run --rm --gpus all base_image_gpu:latest \
  python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU ok:', torch.cuda.is_available())"
```

- `GPU ok: True` → ready.
- `GPU ok: False` → step 4 isn't working (re-run its test).
- `CUDA driver version is insufficient` → the host driver is older
  than the image's CUDA (`torch.version.cuda`): install a newer driver
  (step 2).

---

## 6. Install the service

```bash
# code: only the plate-service folder of the production branch
git clone --no-checkout -b armin_claude_plate_prod https://github.com/Ampmalekpour/EyePass-AI.git eyepass-plate
cd eyepass-plate
git sparse-checkout set plate-service
git checkout armin_claude_plate_prod
cd plate-service
```

Put the models (the AI team provides them):

```
models/detection/plate_v8n_480/   export_info.yaml, plate_v8n_480.pt, plate_v8n_480_288x480.onnx,
                                  plate_v8n_480_fp32_openvino_model/, plate_v8n_480_int8_box_openvino_model/
models/detection/plate_v8s_640/   plate_v8s_640.pt
models/ocr/                       en_PP-OCRv3_det_infer/, rec_svrt_fa_final_1/,
                                  ch_ppocr_mobile_v2.0_cls_infer/, rec_svrt_motor/, Final_Dict.txt
```

Configure:

```bash
cp .env.prod.example .env
grep -n "\[FILL_IN\]" .env          # every line listed here needs a value
nano .env
```

| `[FILL_IN]` | Value |
|---|---|
| `SHARED_NETWORK` | the Docker network Redis / MinIO / mediamtx are on (`docker network ls`, `docker inspect <redis container>`) |
| `REDIS_*` | the same Redis Django and camera_stream use (container name, port, db, password) |
| `MINIO_*` | MinIO endpoint, access/secret key, buckets, public URL |
| `MTX_RTSP_BASE_URL` | `rtsp://<mediamtx container>:8554` |
| `AI_BASE_IMAGE`, `OCR_BASE_IMAGE` | the loaded base image, e.g. `base_image_gpu:latest` |
| `IMAGE_TAG` | a version label for the built images |
| `OPENVINO_VERSION`, `ONNXRUNTIME_VERSION` | from the AI team |

Use container **names** (not `localhost`) for Redis, MinIO and
mediamtx: the services reach them over `SHARED_NETWORK`.

Start:

```bash
docker compose up -d --build
docker compose ps                                   # all "Up", detector/OCR "healthy" after 1-3 min
curl -s localhost:8010/health; echo
curl -s localhost:8011/health; echo
curl -s localhost:8021/health; echo
docker compose logs plate_detector | grep -E "MODEL|INIT|ERROR"
```

The detector log must show the GPU, for example:

```
[INIT] Ultralytics plate_v8n_480.pt on NVIDIA GeForce RTX 4060 | FP32 | batched over cameras
```

and no `ERROR ... CUDA is not available` / `FALLING BACK`.

Cameras are activated by the backend through Redis
(`plate:cmd:ai:request`), the same as before. Live performance:

```bash
docker compose logs -f plate_detector | grep -E "PERF|STATS|INFER-SLOW"
```

Services restart automatically (`restart: unless-stopped`), also after
a server reboot, and self-heal back to their last state from Redis.

---

## 7. CPU-only server (no GPU)

Skip steps 2 and 4. In `.env`:

```ini
COMPOSE_FILE=compose.yaml
DETECTION_DEVICE=cpu
DETECTION_CPU_MODEL=openvino_fp32
```

---

## 8. Updating

```bash
cd eyepass-plate/plate-service
git pull
docker compose up -d --build
```

Model files are bind-mounted: replacing them only needs
`docker compose restart plate_detector`.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Container Toolkit missing or Docker not restarted: step 4 |
| `nvidia-smi`: *couldn't communicate with the NVIDIA driver* | driver not loaded: reboot; Secure Boot MOK not enrolled; after a kernel update reinstall the driver (`sudo ubuntu-drivers install`) |
| detector log: `CUDA is not available — falling back to the CPU pipeline` | GPU not passed in: `COMPOSE_FILE` must include `compose.gpu.yaml`; check step 4/5 |
| `CUDA driver version is insufficient for CUDA runtime version` | host driver older than the image's CUDA: newer driver (step 2) |
| build fails at `FROM base_image_gpu:latest` | base image not loaded or `AI_BASE_IMAGE` wrong (step 5) |
| `network ... declared as external, but could not be found` | `SHARED_NETWORK` wrong: `docker network ls` |
| `Redis not reachable` in the logs | `REDIS_HOST`/`REDIS_URL` wrong, or the Redis container isn't on `SHARED_NETWORK` |
| cameras never get frames (no `[PERF]` lines) | `MTX_RTSP_BASE_URL` wrong, or the mediamtx path name isn't the camera id: `docker exec plate_detector python -c "import cv2; print(cv2.VideoCapture('rtsp://<mediamtx>:8554/<id>').read()[0])"` |
| server has no internet | build on a connected machine, then `docker save eyeplate/plate-detector:<tag> eyeplate/plate-ocr:<tag> eyeplate/control-hub:<tag> -o plate-images.tar`, copy, `docker load -i plate-images.tar`, and start with `docker compose up -d` (no `--build`) |
