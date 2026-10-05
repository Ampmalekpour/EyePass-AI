# plate-service on a Linux CPU server (4 test cameras)

Target: a Linux box with Docker + Compose, no GPU (e.g. Core i3-7100, 2 cores / 4 threads).
Everything runs from the `plate-service` folder of this repository.

## 1. Get the code
```bash
git clone <repo-url> plate_for_sanat        # private repo: use a token or an SSH key
cd plate_for_sanat/plate-service
```

## 2. Put in the files that are not in git
```
plate-service/
├── video2.mp4                                   H.264 test clip (re-encode: ffmpeg -i in.mp4 -c:v libx264 -preset veryfast -an video2.mp4)
└── models/
    ├── detection/plate_v8n_480/                 the model folder (export_info.yaml, *_int8_box_openvino_model/, *_fp32_openvino_model/, *.onnx, *.pt)
    └── ocr/                                     the CONTENTS of the PadOcr folder (en_PP-OCRv3_det_infer, rec_svrt_fa_final_1, ch_ppocr_mobile_v2.0_cls_infer, rec_svrt_motor, Final_Dict.txt)
```
Copy from your PC with scp (PowerShell/cmd/Linux alike):
```
scp -r models video2.mp4 USER@SERVER:~/plate_for_sanat/plate-service/
```

## 3. Build the CPU base image once (about 15-30 min)
```bash
docker build -f docker/Dockerfile.base-cpu -t base_image_cpu:latest .
```

## 4. Configure
```bash
cp .env.cpu-server.example .env
docker network create eyeplate_net
```
Nothing else to edit: CPU, `openvino_int8`, auto-degrade on, both base images = `base_image_cpu:latest`.

## 5. Start (4 cameras = publishers 1..4 only)
```bash
docker compose up -d --build redis mediamtx video_publisher plate_video_publisher_2 \
  plate_video_publisher_3 plate_video_publisher_4 plate_detector plate_ocr control_hub
docker compose ps            # all Up / healthy (the detector and OCR need ~2-3 min to load)
```
Optional Redis UI: add `redis_commander` (http://SERVER:8081).

## 6. Register and activate the cameras
```bash
python3 -m venv .venv && .venv/bin/pip install redis
export REDIS_URL=redis://localhost:6379/0
for i in 1 2 3 4; do
  .venv/bin/python redis_tools.py set-camera --id $i --address publisher --title "Test $i" --roi 0 0 1 1
  .venv/bin/python redis_tools.py activate --id $i
done
.venv/bin/python redis_tools.py list
```

## 7. Look at the results
The startup report and capacity table (engine 0, once):
```bash
docker compose logs plate_detector | grep -E "CAPACITY|╔|║|╚" | head -80
```
Frame misses per camera (every 10 s):
```bash
docker compose logs -f plate_detector | grep "PERF"
```
```
⏱️ [PERF] engine=0 cpu/openvino_int8 camera=1 | fps in=25.0 proc=…  | detect target=… got=… fps (…%) ✅ | missed=… (…%) coasted=… | pre/infer/post … | latency=… ms
```
- `fps in`    frames the camera delivers
- `detect target / got`  the detection rate the engine aims for (it lowers it by itself when 4 cameras exceed this CPU's capacity) and what it achieved; ✅ = at least 90 % of the target
- `missed`   frames replaced by a newer one before the engine looked at them; `coasted` = frames skipped on purpose
- engine summary: `docker compose logs -f plate_detector | grep STATS`
- when the rate changes: `grep -E "🐢|🐇"`

Recognition results: `docker compose logs -f control_hub`, or Redis key `plate:vehicle:results`.

## 8. Stop / reset
```bash
docker compose down            # keep data
docker compose down -v         # also wipe Redis/MinIO volumes
```
