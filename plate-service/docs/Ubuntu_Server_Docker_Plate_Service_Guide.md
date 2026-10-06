# Ubuntu Server + Docker + plate-service (CPU): the full guide

A step-by-step record of everything we did together: installing the server, installing Docker, building and moving the plate-service images, running four test cameras, reading the logs, and shutting everything down. Written so it can be repeated from scratch.

**Tags used below**

- **YOUR PC**: the Windows computer that builds the images, makes the USB and connects over SSH (commands are for `cmd` unless a block says PowerShell)
- **SERVER** / **DESTINATION**: the machine that became the server (Windows was erased)

## Server details

| Item | Value |
| --- | --- |
| Hostname | `sanatmadan` |
| Username | `sanatmadan` |
| Password | not recorded here, write it in your password manager |
| IP address | `<SERVER_IP>`, find it with `hostname -I` (it can change after a reboot). It was `192.168.30.180` |
| OS | Ubuntu Server 24.04.5 LTS |
| ISO | `ubuntu-24.04.5-live-server-amd64.iso` (folder `C:\Users\eyerik.com\Desktop\ubuntu_server`) |
| Disk | WDC WD10EZEX, about 1 TB |
| CPU | Intel Core i3-7100, 3.9 GHz, 2 cores / 4 threads (no AVX-512, no VNNI) |
| Network interface | `enp2s0` (DHCP) |
| Code on the server | `~/plate-service` |
| Code on YOUR PC | `C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service` |
| Git branch | `plate_for_sanat` of the repo `Ampmalekpour/EyePass-AI` |
| Exported Docker images (PC) | `E:\docker_images\sanat_plate_images` |

## Contents

1. Install Ubuntu Server (Parts 1 to 6)
2. SSH and Docker (Parts 7 and 8)
3. What plate-service is (Part 9)
4. Get the code, models and video on YOUR PC (Part 10)
5. Build the images on YOUR PC (Part 11)
6. Move everything to the server (Part 12)
7. Configure and start (Part 13)
8. Add the cameras and watch the logs (Parts 14 and 15)
9. Settings reference (Part 16)
10. Stop, start, reset (Part 17)
11. Change settings or code later (Part 18)
12. Start automatically at boot (Part 19)
13. Daily basics: power off, health, updates, accounts (Part 20)
14. Command reference
15. Troubleshooting
16. Not done yet

---

## Part 1: Preparation

- [ ] **DESTINATION**: back up everything from Windows to an external drive. The install erases the disk.
- [ ] **YOUR PC**: download the Ubuntu Server 24.04 ISO, the `SHA256SUMS` file from [https://releases.ubuntu.com/24.04/SHA256SUMS](https://releases.ubuntu.com/24.04/SHA256SUMS), and Rufus (rufus.ie)
- [ ] **YOUR PC**: one USB stick, 8 GB or more (it gets erased)
- [ ] **DESTINATION**: Ethernet cable to the router, plus a monitor and keyboard
- [ ] Decide the hostname, username and password beforehand

## Part 2: Verify the ISO (YOUR PC)

Open the ISO folder, click the address bar, type `powershell`, press Enter.

```powershell
Get-FileHash .\ubuntu-24.04.5-live-server-amd64.iso -Algorithm SHA256
Select-String -SimpleMatch "ubuntu-24.04.5-live-server-amd64.iso" .\SHA256SUMS
```

Or let PowerShell compare them (`True` means the ISO is good):

```powershell
$expected = (Select-String -SimpleMatch "ubuntu-24.04.5-live-server-amd64.iso" .\SHA256SUMS).Line.Split(" ")[0]
$actual = (Get-FileHash .\ubuntu-24.04.5-live-server-amd64.iso -Algorithm SHA256).Hash
$actual -eq $expected
```

If it is `False`, download the ISO again.

## Part 3: Make the bootable USB (YOUR PC, Rufus)

1. Plug in the USB stick and open Rufus
2. **Device**: the USB stick (double-check it)
3. **Boot selection**: SELECT, then pick the ISO
4. **Partition scheme**: GPT. **Target system**: UEFI (non CSM).
5. Click **START**. Accept extra file downloads. Choose **ISO Image mode**.
6. Confirm the erase warning, wait for **READY**, then eject the stick

## Part 4: Boot the destination from the USB

1. Plug in the USB, shut the machine down fully, then power on and tap the boot menu key (F12, F11, F10, F9, F2 or Esc depending on the brand)
2. Choose the entry starting with **UEFI:** and the USB's name
3. If the USB is not listed: enter BIOS/UEFI (Del or F2), set **Secure Boot** to Disabled, check USB boot is enabled, save, retry
4. At GRUB choose **Try or Install Ubuntu Server**

## Part 5: Installer screens

Keys: arrows move, **Space** ticks or selects, **Enter** confirms, **Tab** jumps to buttons.

| Screen | What to do |
| --- | --- |
| Language, keyboard | Pick yours. Skip any installer update. |
| Choose the base | Keep **(X) Ubuntu Server**. Leave "minimized" and "Search for third-party drivers" empty. Move to Done. |
| Network | `enp2s0` should show DHCPv4 `192.168.x.x`. Leave as is, Done. |
| Proxy | Leave empty, Done |
| Mirror | Leave default, wait for the test to pass, Done |
| Guided storage | **Use an entire disk**, **Set up this disk as an LVM group** ticked, **Encrypt with LUKS** left off (encryption asks for a passphrase at every boot) |
| Storage summary | See the resize note below, then Done, then **Continue** on the destructive warning |
| Profile | Name, server name, username, password |
| Ubuntu Pro | Skip for now |
| SSH | Tick **Install OpenSSH server** and **Allow password authentication over SSH**. Import SSH key: No. |
| Featured snaps | Tick nothing, Done (Docker is installed from Docker's repository later) |
| Finish | Wait for install and updates, **Reboot Now**, remove the USB when told |

**Enlarge the root volume.** The installer makes `/` only 100 GB and leaves about 828 GB unused. The `/` line in FILE SYSTEM SUMMARY is read-only (it only offers Close and Unmount). Instead, scroll down to **USED DEVICES**, select the `100.000G ▸` marker on the `ubuntu-lv` line, choose **Edit**, set Size to the maximum (about 928G), keep ext4 and `/`, and Save. If you skip this, fix it after install (see Part 6).

## Part 6: First boot (DESTINATION)

Press Enter once to redraw the login prompt (the cloud-init text on screen is normal), then log in. The password does not show while typing.

```bash
hostname -I                     # the IP, capital I
ping -c 3 8.8.8.8
ping -c 3 google.com
df -h /                         # Size should be about 900G
sudo apt update && sudo apt upgrade -y
```

If `df -h /` shows about 98G, enlarge the volume:

```bash
sudo lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
df -h /
```

Reboot if asked: `sudo reboot`

## Part 7: Connect from your PC with SSH (YOUR PC)

```powershell
ping <SERVER_IP>
ssh sanatmadan@<SERVER_IP>
```

Type `yes` at the fingerprint question, then enter the password. A prompt like `sanatmadan@sanatmadan:~$` means you are on the server. Both machines must be on the same network.

**How to close SSH.** Type `exit` (or press Ctrl+D). The prompt returns to your Windows `C:\...>`.

**Why one `exit` is sometimes not enough.** The command `newgrp docker` (Part 8) opens a *sub-shell* inside your SSH session. The first `exit` only leaves that sub-shell and you stay on the server. If the prompt after `exit` still says `sanatmadan@sanatmadan:~$`, type `exit` a second time.

**You can open as many SSH windows as you like.** Open another `cmd` and run the same `ssh` command. We use one window to add cameras and one to watch the logs.

**Closing SSH does not stop the containers.** They run on the server. Use `docker compose down` to stop them (Part 17).

## Part 8: Install Docker (on the server, over SSH)

Paste one block at a time.

```bash
# 1. Remove conflicting packages (fine if none are installed)
sudo apt remove -y docker.io docker-compose docker-doc podman-docker containerd runc 2>/dev/null

# 2. Prerequisites and Docker's signing key
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 3. Add the repository (one command)
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
cat /etc/apt/sources.list.d/docker.list      # should end in "noble stable"

# 4. Install
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Check and enable at boot
sudo systemctl status docker --no-pager      # expect: active (running)
sudo systemctl enable docker

# 6. Run docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# 7. Test
docker --version
docker compose version
docker run hello-world                       # expect "Hello from Docker!"
docker rm $(docker ps -aq) 2>/dev/null
docker rmi hello-world

# 8. Limit container log size
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
EOF
sudo systemctl restart docker
docker info | grep -i "logging driver"       # expect json-file
```

**Tip:** if `systemctl status` leaves you in a screen ending with `(END)`, press **q** to leave it. That is a pager, not a freeze.

**Note:** after `newgrp docker`, remember Part 7: you will need `exit` twice to close SSH.

---

## Part 9: What plate-service is

A licence-plate pipeline that runs as Docker containers. Frames come from an RTSP relay, the detector finds plates, OCR reads them, and the control hub decides the final record.

| Container | Image | What it does | Port on the server |
| --- | --- | --- | --- |
| `plate_detector` | `eyeplate/plate-detector:cpu` | reads each camera's frames, runs the plate model (OpenVINO INT8 on CPU), tracks plates, hands crops to OCR | 8010 |
| `plate_ocr` | `eyeplate/plate-ocr:cpu` | PaddleOCR workers read the plate crops | 8011 |
| `plate_control_hub` | `eyeplate/control-hub:cpu` | votes over OCR results, publishes the final result | 8021 |
| `eyeplate_redis` | `redis:7-alpine` | the message bus and the camera configuration | 6379 |
| `plate_mediamtx` | `bluenviron/mediamtx:latest` | the RTSP relay the detector reads from | 8554 |
| `plate_video_publisher_1..4` | `eyeplate-infra-...publisher` | loop one test video into the relay as cameras 1 to 4 | none |

How a camera flows: video file -> publisher -> mediamtx (`rtsp://mediamtx:8554/<id>`) -> detector -> OCR -> control hub -> Redis.

The compose project name is `eyeplate-infra`, which is why the publisher images are called `eyeplate-infra-...`.

### The CPU pipeline in short

- The detector holds one model instance per camera and runs them in parallel threads inside **one engine** (up to 16 cameras).
- Models are in `models/detection/plate_v8n_480/`. `DETECTION_CPU_MODEL` in `.env` picks `openvino_int8`, `openvino_fp32` or `onnx`.
- **Capacity test at startup.** Engine 0 measures how long one detection loop takes for 1, 2, 3 ... cameras on this exact CPU and prints a boxed hardware report. Real-time means at least 25 fps per camera, so one loop must finish in 40 ms x 0.7 = 28 ms.
- **Auto-degrade.** If the cameras you add exceed that capacity, the engine lowers every camera's detection rate smoothly to what fits (the interval is the measured loop time divided by 28 ms). The tracker coasts the frames in between. It goes back up when cameras are removed. Switch: `CAPACITY_AUTO_DEGRADE=true|false` in `.env`.

### Reference measurements we took (i7-12700K)

| Setup | Model | Real-time capacity (25 fps) | Ceiling |
| --- | --- | --- | --- |
| 14 threads | OpenVINO INT8 | 4 cameras | about 200 detections/s |
| 14 threads | OpenVINO FP32 | 2 to 3 cameras | about 125 detections/s |
| 14 threads | ONNX | 4 cameras | about 190 detections/s |
| 4 threads, AVX2 only (simulated i3 shape) | OpenVINO INT8 | 2 cameras | about 100 detections/s |

The i3-7100 has slower cores than the 12700K, so expect less. Write your own results here after the first run:

| Date | Model | Capacity in the report | Cameras tried | Detection rate (got) | Missed % |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

---

## Part 10: Get the code, models and video on YOUR PC

### 10.1 Clear any proxy that breaks git (YOUR PC, cmd)

The error `Unsupported proxy syntax in ' '` means a proxy setting contains just a space.

```
git config --global --unset http.proxy
git config --global --unset https.proxy
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set | findstr /i proxy
```

The last command should print nothing. If it still lists variables, remove them permanently and open a new cmd window:

```
reg delete "HKCU\Environment" /v HTTPS_PROXY /f
reg delete "HKCU\Environment" /v HTTP_PROXY /f
```

### 10.2 Clone the branch (YOUR PC, cmd)

```
cd C:\Users\eyerik.com\Desktop
git clone --branch plate_for_sanat --single-branch https://github.com/Ampmalekpour/EyePass-AI.git plate_for_sanat
```

The folder `plate_for_sanat` must be empty or not exist. The code is then in `plate_for_sanat\plate-service`. To update it later: `cd C:\Users\eyerik.com\Desktop\plate_for_sanat` then `git pull origin plate_for_sanat`.

### 10.3 Put the models and the video in (YOUR PC, cmd)

These are not in git.

```
robocopy "C:\Users\eyerik.com\Desktop\plate-service-repo\plate-service\models" "C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service\models" /E
copy "C:\Users\eyerik.com\Desktop\plate-service-repo\plate-service\video2.mp4" "C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service\video2.mp4"
```

Check that these exist:

```
plate-service\models\detection\plate_v8n_480\    export_info.yaml, the int8, fp32, onnx and pt files
plate-service\models\ocr\                        the PadOcr folders and Final_Dict.txt
plate-service\video2.mp4                         an H.264 clip (a FILE, not a folder)
```

If `video2.mp4` shows up as a folder, Docker created it by mistake. Delete the folder and copy the real file in. To re-encode another format:

```
ffmpeg -i in.mp4 -c:v libx264 -preset veryfast -an video2.mp4
```

---

## Part 11: Build the images on YOUR PC

**Why the images are built on your PC.** The server can reach Docker Hub and apt, but it **cannot download packages from PyPI** (`files.pythonhosted.org` times out). The `pip install` steps inside the Dockerfiles would fail there. So everything that needs PyPI is built on your PC, and the finished images are copied to the server.

### 11.1 Free disk space and clear old images (YOUR PC, cmd)

You need about 40 GB free inside Docker. A full disk is what corrupted our first base image.

```
docker system df
docker image prune -f
docker builder prune -a -f
```

### 11.2 Build the clean CPU base image (YOUR PC, cmd)

The Dockerfile is `plate-service\docker\Dockerfile.base-cpu`. It is the original CPU Dockerfile plus a `python` symlink, longer pip timeouts, `onnxruntime==1.23.2`, `openvino==2026.4.1` and a final import check that fails the build if anything is broken.

```
cd C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service
docker build -f docker/Dockerfile.base-cpu -t base_image_cpu:latest .
```

It takes 30 to 60 minutes or more and about 16 GB. If it fails on a network error, run the same command again (finished layers are kept). Then check it:

```
docker run --rm --entrypoint python base_image_cpu:latest -c "import numpy, onnxruntime, openvino, cv2, torch, ultralytics, paddle; print('all imports OK', numpy.__version__, torch.__version__)"
```

Expected: `all imports OK 1.24.4 1.13.1+cpu`.

**Do not use `base-image-cpu-only.tar`.** That exported image was corrupted (`numpy ... file too short`, a broken `torch_version.py`). If you ever see `ImportError ... file too short`, the image is damaged: delete it and rebuild from the Dockerfile.

### 11.3 Set up the build `.env` (YOUR PC, cmd)

```
cd C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service
copy .env.cpu-server.example .env
notepad .env
```

Check these lines (they keep the CPU images from overwriting your GPU `:dev` images):

```
IMAGE_TAG=cpu
AI_BASE_IMAGE=base_image_cpu:latest
OCR_BASE_IMAGE=base_image_cpu:latest
INSTALL_CPU_RUNTIMES=true
OPENVINO_VERSION=2026.4.1
ONNXRUNTIME_VERSION=1.23.2
```

### 11.4 Build the service images (YOUR PC, cmd)

```
docker compose build plate_detector plate_ocr control_hub video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4
docker images "eyeplate/*:cpu"
docker images "eyeplate-infra-*"
```

You should see `eyeplate/plate-detector:cpu`, `eyeplate/plate-ocr:cpu`, `eyeplate/control-hub:cpu` and four `eyeplate-infra-...publisher` images.

The detector's code and `config.py` are **baked into** the detector image. Check the values before you build:

```
findstr /n "DETECT_EVERY_N_FRAMES DEBUG_VIDEO_ENABLED CPU_TORCH_NUM_THREADS" detector\src\config.py
```

Repo defaults are `DETECT_EVERY_N_FRAMES = 1` and `DEBUG_VIDEO_ENABLED = False`.

### 11.5 Check the packages are inside the detector image (YOUR PC, cmd)

```
docker run --rm --entrypoint python eyeplate/plate-detector:cpu -c "import openvino, onnxruntime, cv2, ultralytics, redis, scipy; print('openvino', openvino.__version__, '| onnxruntime', onnxruntime.__version__)"
```

Expected: `openvino 2026.4.1 | onnxruntime 1.23.2`.

### 11.6 Export the images into one folder (YOUR PC, cmd)

```
mkdir E:\docker_images\sanat_plate_images
docker save eyeplate/plate-detector:cpu eyeplate/plate-ocr:cpu -o E:\docker_images\sanat_plate_images\ai_images.tar
docker save eyeplate/control-hub:cpu redis:7-alpine bluenviron/mediamtx:latest rediscommander/redis-commander:latest eyeplate-infra-video_publisher eyeplate-infra-plate_video_publisher_2 eyeplate-infra-plate_video_publisher_3 eyeplate-infra-plate_video_publisher_4 -o E:\docker_images\sanat_plate_images\small_images.tar
dir E:\docker_images\sanat_plate_images
```

`ai_images.tar` is about 17 to 19 GB (it contains the base). `small_images.tar` is about 1 GB. Keep 40 GB or more free on `E:`.

---

## Part 12: Move everything to the server

### 12.1 Find the server address (SERVER)

```
hostname -I
```

Use the first address. Below it is written `192.168.30.180`.

### 12.2 Copy the code folder (YOUR PC, cmd)

This copies the code, the models and `video2.mp4` in one go.

```
cd C:\Users\eyerik.com\Desktop\plate_for_sanat
scp -r plate-service sanatmadan@192.168.30.180:~/
```

It asks for the server password, and it can take a long time because the models are large. Wait until the prompt comes back.

On the server, check it arrived and fix Windows line endings:

```
cd ~/plate-service
ls
ls models/detection/plate_v8n_480
ls models/ocr
ls -lh video2.mp4
sed -i 's/\r$//' .env.cpu-server.example publish_loop.sh redis_tools.py
```

### 12.3 Copy the images (YOUR PC, cmd)

```
scp -r E:\docker_images\sanat_plate_images sanatmadan@192.168.30.180:~/
```

### 12.4 Load the images on the server (SERVER)

Check the disk first. You need about 40 GB free, because the tar files and the loaded images exist at the same time.

```
df -h ~
cd ~/sanat_plate_images
docker load -i ai_images.tar
docker load -i small_images.tar
docker images
cd ~
rm -r sanat_plate_images
```

---

## Part 13: Configure and start (SERVER)

```
cd ~/plate-service
cp .env.cpu-server.example .env
sed -i 's/^IMAGE_TAG=.*/IMAGE_TAG=cpu/' .env
grep IMAGE_TAG .env
docker network create eyeplate_net
```

`grep` must print `IMAGE_TAG=cpu`. The network is created once. If it says "already exists", ignore it.

Start the services. `--no-build` is important: the server must not try to build (no PyPI there).

```
docker compose up -d --no-build redis mediamtx video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4 plate_detector plate_ocr control_hub
docker compose ps
```

The detector and OCR need 2 to 3 minutes to load. Wait until `docker compose ps` shows `(healthy)`. If it stays on `health: starting` for more than 5 minutes, look at `docker compose logs plate_detector`.

### Shortcuts (optional, SERVER)

Make the camera helper `rt` available in every SSH session, plus two log/stop shortcuts:

```
cat >> ~/.bashrc <<'EOF'
rt() { docker run --rm --network host -e REDIS_URL=redis://localhost:6379/0 -v ~/plate-service:/w -w /w eyeplate/plate-detector:cpu python redis_tools.py "$@"; }
alias plate-perf='cd ~/plate-service && docker compose logs --tail 0 -f plate_detector | grep --line-buffered -E "PERF|CAPACITY"'
alias plate-down='cd ~/plate-service && docker compose down'
EOF
source ~/.bashrc
```

Without this, paste the `rt() { ... }` line again in every new SSH session.

---

## Part 14: Add the cameras one by one and watch (two windows)

### Window B (a new `cmd` on YOUR PC): the log window

```
ssh sanatmadan@192.168.30.180
cd ~/plate-service
docker compose logs --tail 0 -f plate_detector | grep --line-buffered -E "PERF|CAPACITY"
```

`--tail 0` shows only new lines, so nothing appears until you add a camera. Ctrl+C stops it (not the containers).

### Window A (your first SSH window): add the cameras

Define the helper if you did not add it to `.bashrc`:

```
rt() { docker run --rm --network host -e REDIS_URL=redis://localhost:6379/0 -v ~/plate-service:/w -w /w eyeplate/plate-detector:cpu python redis_tools.py "$@"; }
```

Camera 1, then wait about 40 seconds and look at window B:

```
rt set-camera --id 1 --address publisher --title "Test 1" --roi 0 0 1 1
rt activate --id 1
```

Then the same for cameras 2, 3 and 4 (wait 40 to 60 seconds between them):

```
rt set-camera --id 2 --address publisher --title "Test 2" --roi 0 0 1 1
rt activate --id 2
rt set-camera --id 3 --address publisher --title "Test 3" --roi 0 0 1 1
rt activate --id 3
rt set-camera --id 4 --address publisher --title "Test 4" --roi 0 0 1 1
rt activate --id 4
```

The address `publisher` is required for the test publishers. A URL pointing at the relay itself blocks the stream.

All four at once:

```
for i in 1 2 3 4; do
  rt set-camera --id $i --address publisher --title "Test $i" --roi 0 0 1 1
  rt activate --id $i
done
rt list
```

Other camera commands:

```
rt list                       # every configured camera and its ai_status
rt status --id 1              # one camera's status
rt deactivate --id 4          # stop processing it
rt remove --id 4              # delete its configuration
rt self-heal-state          # both services' self-healing checkpoints
rt heartbeats                 # how recent each service's heartbeat is
rt ocr-queue                  # OCR queue depth
```

---

## Part 15: Reading the logs

### The `[PERF]` line (one per camera, every 10 seconds)

```
⏱️ [PERF] engine=0 cpu/openvino_int8 camera=1 | fps in=25.0 proc=14.0 | detect target=15.2 got=14.0 fps (92%) ✅ | missed=68 (27.2%) coasted=41 | pre=2.8 infer=17.2 (p95 45.7) post=2.6 track=1.1 ms/frame | latency=55 (p95 108) ms | dets/frame=0.97 | lifetime missed=15.9%
```

| Field | Meaning |
| --- | --- |
| `fps in` | frames per second the camera delivers (about 25) |
| `proc` | frames per second actually run through the detector |
| `detect target` / `got` | the detection rate the engine aims for (it lowers the target when the cameras exceed the measured capacity) and what it reached. ✅ means 90% or more of the target, ⚠️ below that |
| `missed` | frames the camera delivered that the engine never looked at, because a newer frame replaced them. With auto-degrade ON, many of these are frames that were not needed anyway, so judge by `got` versus `target` |
| `coasted` | frames the engine saw and skipped on purpose (the tracker fills the gap) |
| `pre / infer / post / track` | milliseconds per frame for preprocessing, the model, post-processing and the tracker |
| `latency` | milliseconds from a frame arriving to its result (average and p95) |
| `dets/frame` | average plates found per processed frame |
| `lifetime missed` | the miss rate since the camera started |

### The `[CAPACITY]` lines

| Line | Meaning |
| --- | --- |
| `🧪 [CAPACITY] ... N cameras: loop avg ... p95 ...` | the startup test: how long one loop takes for N cameras |
| `🏁 ... REAL-TIME CAPACITY: N cameras at >=25 fps` | the result of that test |
| `✅ [CAPACITY] camera X attached: 3/4 real-time cameras` | the camera fits within capacity |
| `🚨 [CAPACITY] camera X attached: 5 cameras > real-time capacity` | too many cameras for full rate |
| `🐢 [CAPACITY] ... detection lowered to 16.7 fps per camera` | auto-degrade lowered the rate |
| `🐇 [CAPACITY] ... detection 25.0 fps per camera (full rate)` | the rate went back up |

### The boxed hardware report (printed once at startup)

It shows the CPU, cores, instruction sets, how many cores Docker really gives the container, memory, the model, the measured **CEILING** (detections per second for the whole machine), **SAFE USE** (the ceiling times the 0.7 margin), the capacity at full 25 fps, a **planning table** (what 1 to 10 cameras need and what rate you get), and what to do. Show it with:

```
docker compose logs plate_detector | grep -E "CAPACITY|║"
```

### The `[STATS]` line (engine summary)

```
📊 [STATS] engine=0 cpu/openvino_int8 cameras=4 | last 50 loops in 2.5s | frames/loop=4.00 | processed=161.0/s | missed=... | infer/frame avg=... p95=... ms | batch avg=... | loop avg=... p95=... ms | detections=5 tracks=2
```

### Log commands, with examples

Always run from `~/plate-service` on the server.

| Goal | Command |
| --- | --- |
| follow the detector live | `docker compose logs -f plate_detector` (Ctrl+C stops) |
| speed and misses per camera | `docker compose logs --tail 0 -f plate_detector \| grep --line-buffered -E "PERF\|CAPACITY"` |
| only the PERF lines of camera 2 | `docker compose logs -f plate_detector \| grep --line-buffered "camera=2"` |
| the startup report | `docker compose logs plate_detector \| grep -E "CAPACITY\|║"` |
| engine summaries | `docker compose logs -f plate_detector \| grep STATS` |
| rate lowered or raised | `docker compose logs plate_detector \| grep -E "🐢\|🐇"` |
| plate tracks (new, end, OCR hand-off) | `docker compose logs -f plate_detector \| grep -E "TRACK\|OCR"` |
| errors and warnings | `docker compose logs plate_detector \| grep -E "ERROR\|WARNING"` |
| slow loops | `docker compose logs plate_detector \| grep "INFER-SLOW"` |
| last 100 lines | `docker compose logs --tail 100 plate_detector` |
| last 5 minutes | `docker compose logs --since 5m plate_detector` |
| how many lines mention a word | `docker compose logs plate_detector \| grep -c "missed"` |
| save everything to a file | `docker compose logs plate_detector > detector_log.txt` |
| the OCR log | `docker compose logs -f plate_ocr` |
| the hub log (final results) | `docker compose logs -f control_hub` |
| all services together | `docker compose logs -f` |

`grep` in a pipe with `-f` needs `--line-buffered` to show lines as they arrive. Add `-i` to ignore upper and lower case.

### Status and resources

| Goal | Command |
| --- | --- |
| containers and health | `docker compose ps` |
| live CPU and memory per container | `docker stats` (Ctrl+C stops) |
| detector status | `curl -s http://localhost:8010/health` |
| OCR status | `curl -s http://localhost:8011/health` |
| hub status | `curl -s http://localhost:8021/health` |
| CPU of the whole box | `top` (press `q` to leave) |
| disk space | `df -h ~` |
| Docker disk use | `docker system df` |
| images on the server | `docker images` |

Redis can be inspected with Redis Commander (add `redis_commander` to the `up` command, then open `http://<SERVER_IP>:8081`).

---

## Part 16: Settings reference

### In `.env` (no rebuild; run `docker compose up -d --force-recreate plate_detector` after editing)

| Setting | Value we use | What it does |
| --- | --- | --- |
| `DETECTION_DEVICE` | `cpu` | `cpu`, `gpu` or `auto` |
| `DETECTION_CPU_MODEL` | `openvino_int8` | `openvino_fp32`, `openvino_int8` or `onnx` |
| `CAPACITY_AUTO_DEGRADE` | `true` | lower the detection rate when over capacity; `false` = warn only |
| `OCR_DEVICE` | `cpu` | where OCR runs |
| `IMAGE_TAG` | `cpu` | which image set compose uses |
| `DETECTOR_CPU_LIMIT`, `OCR_CPU_LIMIT` | `0` | CPU cap per container, 0 = none |
| `COMPOSE_FILE` | `compose.yaml:compose.infra.yaml` | which compose files are merged; the `:` list is read in order and the later files add to or override the earlier ones |
| `COMPOSE_PROFILES` | `standalone-stream,test-video` | which optional services exist (own relay, test publishers) |

Edit on the server with `nano .env` (Ctrl+O then Enter saves, Ctrl+X leaves).

### In `detector/src/config.py` (baked into the image; rebuild to change)

| Setting | Default | What it does |
| --- | --- | --- |
| `DETECT_EVERY_N_FRAMES` | `1` | detect every Nth frame per camera; also the minimum interval for auto-degrade |
| `REALTIME_MIN_FPS` | `25` | what counts as real-time |
| `CAPACITY_SAFETY_MARGIN` | `0.7` | share of the frame time the detector may use (the rest is for decode, tracking, OCR) |
| `CAPACITY_CALIBRATION_ENABLED` | `True` | run the capacity test at startup |
| `CAPACITY_MAX_CAMERAS_TESTED` | `12` | the test tries 1 up to this many cameras |
| `CAPACITY_FRAME_SIZE` | `(1080, 1920)` | size of the dummy test frames, set it to your cameras' size |
| `DETECT_MIN_FPS` | `8` | auto-degrade never goes below this rate |
| `CPU_MAX_CAMERAS_PER_ENGINE` | `16` | cameras served by one CPU engine |
| `CPU_TORCH_NUM_THREADS` | `1` | threads for Ultralytics' pre/post-processing (1 avoids fighting OpenVINO) |
| `CV2_NUM_THREADS` | `2` | OpenCV thread pool |
| `RTSP_FFMPEG_THREADS` | `1` | decode threads per camera stream |
| `CONF_THRESHOLD` | `0.25` | minimum detection confidence |
| `NMS_IOU` | `0.7` | duplicate-box threshold |
| `PERF_LOG_INTERVAL_SEC` | `10` | how often `[PERF]` prints |
| `STATS_EVERY_N_BATCHES` | `50` | how often `[STATS]` prints |
| `LOG_TRACK_EVENTS` | `True` | per-track log lines |
| `DEBUG_VIDEO_ENABLED` | `False` | write an annotated video per camera into `debug_video/` |
| `DEBUG_VIDEO_FPS` / `DEBUG_VIDEO_SCALE` | `12` / `1.0` | playback speed and frame size of that video (0.5 is much cheaper) |
| OCR workers (`ocr_service/src/config.py`) | `DEFAULT_WORKER_COUNT = 3` | number of PaddleOCR workers on a fresh boot |

The debug video costs CPU the capacity test does not measure, so leave it off when you judge capacity.

---

## Part 17: Stop, start and reset (SERVER)

### Stop everything cleanly

```
cd ~/plate-service
for i in 1 2 3 4; do rt deactivate --id $i; done
docker compose down
docker compose ps
docker ps
```

Wait about 10 seconds after the `deactivate` loop. `down` removes the containers and the compose network. It keeps the images, your files and the Redis data volume. Both `ps` commands should be empty. Then close SSH (Part 7: `exit`, twice if you used `newgrp docker`).

| Command | What it does |
| --- | --- |
| `docker compose stop` | stops containers, keeps them; `docker compose start` brings them back |
| `docker compose down` | removes containers and the network; keeps images, files and Redis data |
| `docker compose down -v` | also deletes the Redis and MinIO data volumes: a clean slate, camera configs gone |
| `docker compose stop plate_detector` | stops one service only |
| `docker compose restart plate_detector` | restarts one service |

### Start again next time

```
cd ~/plate-service
docker compose up -d --no-build redis mediamtx video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4 plate_detector plate_ocr control_hub
docker compose ps
```

Wait for `(healthy)`, then add the cameras (Part 14). After `down -v`, register them again with `set-camera`.

### Reset the whole test

```
for i in 1 2 3 4; do rt deactivate --id $i; done
docker compose down -v
docker compose up -d --no-build redis mediamtx video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4 plate_detector plate_ocr control_hub
```

If the server reboots, containers with a restart policy start again by themselves. `docker compose down` prevents that.

---

## Part 18: Change settings or code later

**An `.env` setting** (model, auto-degrade, CPU caps): edit `.env` on the server, then recreate the service.

```
nano .env
docker compose up -d --force-recreate plate_detector
```

**Code or `config.py`:** rebuild the image on YOUR PC, send it, load it, recreate the service.

YOUR PC, cmd:

```
cd C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service
docker compose build plate_detector
docker save eyeplate/plate-detector:cpu -o E:\docker_images\sanat_plate_images\detector_only.tar
scp E:\docker_images\sanat_plate_images\detector_only.tar sanatmadan@192.168.30.180:~/
```

SERVER:

```
docker load -i ~/detector_only.tar
rm ~/detector_only.tar
cd ~/plate-service
docker compose up -d --force-recreate --no-build plate_detector
```

The same pattern works for `plate_ocr` and `control_hub`. This has not been tested on the server yet. If the detector does not start afterwards, read `docker compose logs plate_detector`.

**Update the code folder only** (compose files, `redis_tools.py`, docs): edit on YOUR PC, then copy the single file, for example:

```
scp C:\Users\eyerik.com\Desktop\plate_for_sanat\plate-service\compose.yaml sanatmadan@192.168.30.180:~/plate-service/
```

---

## Part 19: Start everything automatically when the server turns on (SERVER)

Goal: switch the server on, wait a few minutes, and the cameras are being processed, with nobody logged in.

### Why it can already mostly work

- Docker itself starts at boot (`sudo systemctl enable docker` in Part 8).
- Every plate-service container has `restart: unless-stopped`, so Docker brings back the containers that were running when the server went down.
- Redis keeps its data on disk (`appendonly yes`) in the `eyeplate_redis_data` volume, and the detector replays the list of active cameras from Redis when it starts. So the cameras you activated come back on their own.

This breaks in one case: after `docker compose down` the containers no longer exist, so there is nothing for Docker to restart. A manually stopped container also stays stopped. That is why we add a systemd service that runs `docker compose up -d` at every boot.

### Install the boot service (SERVER, once)

Check Docker starts at boot (expect `enabled`):

```bash
systemctl is-enabled docker
```

Create the service. The folder is `/home/sanatmadan/plate-service`; change the path if yours differs.

```bash
sudo tee /etc/systemd/system/plate-service.service > /dev/null <<'EOF'
[Unit]
Description=plate-service (docker compose stack)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/sanatmadan/plate-service
Environment=HOME=/root
ExecStart=/usr/bin/docker compose up -d --no-build redis mediamtx video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4 plate_detector plate_ocr control_hub
ExecStop=/usr/bin/docker compose stop
TimeoutStartSec=300
TimeoutStopSec=180

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now plate-service
systemctl status plate-service --no-pager
```

The same file is in the repo as `deploy/plate-service.service`. In `systemctl status` expect `active (exited)`; that is normal for this kind of service, because it only starts the containers and then ends. Press **q** if the output stops at `(END)`.

### Test it (SERVER, then YOUR PC)

Make sure the cameras are activated first (`rt list`, Part 14). Then:

```bash
sudo reboot
```

Wait 3 to 5 minutes (the detector and OCR load their models), then from YOUR PC:

```
ssh sanatmadan@192.168.30.180
cd ~/plate-service
docker compose ps
docker compose logs --tail 20 plate_detector
rt list
```

You should see every container `Up`/`healthy` and `[PERF]` lines coming for each camera. If `rt` is not defined, paste the helper from Part 13.

### Everyday control

| Goal | Command |
| --- | --- |
| stack status | `systemctl status plate-service --no-pager` and `docker compose ps` |
| stop the stack (until you start it again or the server reboots) | `sudo systemctl stop plate-service` |
| start it again | `sudo systemctl start plate-service` |
| restart everything | `sudo systemctl restart plate-service` |
| turn the auto-start off | `sudo systemctl disable plate-service` |
| turn it on again | `sudo systemctl enable plate-service` |
| the boot service's own log | `journalctl -u plate-service --no-pager -n 50` |

Notes:

- `sudo reboot` and `sudo shutdown -h now` stop the containers gracefully first, and they come back at the next boot.
- With the boot service enabled, `docker compose down` is no longer a problem: the next boot recreates the containers. Cameras that were registered with `set-camera` and activated stay in Redis, unless you used `down -v`, which wipes Redis. After `down -v`, register and activate the cameras again.
- If you change the service list (for example you add a fifth publisher), edit `ExecStart` in `/etc/systemd/system/plate-service.service` and run `sudo systemctl daemon-reload`.
- If the server should also switch **itself** on after a power cut, enable "Restore on AC Power Loss / Power On" in the BIOS (the exact name depends on the board).
- This service was written for this guide and has **not** been tested on your server yet. If it fails, run `journalctl -u plate-service --no-pager -n 50` and send me the output.

---

## Part 20: Daily basics on the server (SERVER)

The small things you need every day: switching the server off and on, checking it is healthy, updates, accounts and the clock.

### Turn the server off or restart it safely

| Goal | Command |
| --- | --- |
| power off now | `sudo shutdown -h now` (same as `sudo poweroff`) |
| restart now | `sudo reboot` |
| power off in 30 minutes | `sudo shutdown -h +30` |
| power off at a clock time | `sudo shutdown -h 23:00` |
| cancel a scheduled shutdown or reboot | `sudo shutdown -c` |

Safe order:

1. (Optional, cleaner) stop the stack first: `sudo systemctl stop plate-service`, wait until it returns (up to about 30 seconds). If you skip this, the shutdown stops the containers anyway.
2. Run `sudo shutdown -h now`. Your SSH window ends with `Connection closed`.
3. Wait until the fans and the power light have stopped before you unplug anything or press the power button again.

Do not just pull the power cable. Redis writes to disk about once a second, so an unclean power cut can lose the last second of data, and the file system may need a check at the next boot.

To switch it on: press the power button. Wait **3 to 5 minutes** (Ubuntu boots, Docker starts, then the detector and OCR load their models). With the boot service from Part 19 the cameras start by themselves.

After switching on, check:

```bash
uptime
systemctl status plate-service --no-pager
cd ~/plate-service
docker compose ps
```

If you cannot connect over SSH after a restart, the IP may have changed. Read it on the server's own screen with `hostname -I`, or in the router's list of connected devices. Reserve the IP in the router so it stays the same (see "Not done yet").

### Is the server healthy? A quick check

| Goal | Command |
| --- | --- |
| how long it has been running | `uptime` |
| memory | `free -h` |
| disk | `df -h /` |
| live CPU (q to leave) | `top` |
| CPU and memory per container (one snapshot) | `docker stats --no-stream` |
| failed services | `systemctl --failed` |
| errors since the last boot | `journalctl -p err -b --no-pager \| tail -n 30` |
| who is logged in now | `who` |
| recent logins | `last -n 10` |
| CPU temperature (install once: `sudo apt install -y lm-sensors`) | `sensors` |
| does a reboot wait for you? | `cat /var/run/reboot-required` (the file exists only if a reboot is needed) |

### Updates

```bash
sudo apt update
sudo apt upgrade -y
```

Do it when the cameras are idle, because upgrading Docker restarts the Docker service and the containers restart with it. If `/var/run/reboot-required` exists, run `sudo reboot`. Afterwards check `docker compose ps`.

### Accounts, sudo and the docker group

| Goal | Command |
| --- | --- |
| change your password | `passwd` |
| which groups am I in? | `id` |
| add me to the docker group | `sudo usermod -aG docker $USER` and then log out and back in (`exit`, then `ssh` again) |

`sudo` asks for your own password. The password does not show while you type.

The error `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock` means the current login is not in the `docker` group yet. A new login fixes it (or `newgrp docker` for one window; then `exit` twice to close SSH). The boot service is not affected: it runs as root.

### Time and name

| Goal | Command |
| --- | --- |
| clock, time zone, time sync | `timedatectl` |
| set the time zone | `sudo timedatectl set-timezone Asia/Tehran` |
| hostname and OS | `hostnamectl` |

Container logs are stamped in UTC.

### systemd services in short

| Goal | Command |
| --- | --- |
| status | `systemctl status NAME --no-pager` |
| start / stop / restart | `sudo systemctl start NAME`, `stop NAME`, `restart NAME` |
| start at boot on / off | `sudo systemctl enable NAME`, `disable NAME` |
| its log | `journalctl -u NAME --no-pager -n 50` |
| all services that run | `systemctl list-units --type=service --state=running` |

Names we use: `plate-service`, `docker`, `ssh`.

### Long jobs and dropped connections

Containers keep running when SSH closes. But a long command in the SSH window (a big `docker load`, an apt upgrade) is stopped when the connection drops. Use `tmux` for those:

```bash
sudo apt install -y tmux
tmux new -s work          # start a session and run the long command inside it
# detach with Ctrl+B, then D. Reconnect later:
tmux attach -t work
tmux ls                   # list sessions
exit                      # inside tmux: ends the session
```

If an `scp` from your PC is interrupted, run it again.

### Where things are on the server

| What | Where |
| --- | --- |
| the code, `.env`, compose files | `~/plate-service` |
| models and the test video | `~/plate-service/models`, `~/plate-service/video2.mp4` |
| debug videos (when enabled) | `~/plate-service/debug_video` |
| the Redis data (camera configs, queues) | Docker volume `eyeplate_redis_data` (`docker volume ls`) |
| the boot service | `/etc/systemd/system/plate-service.service` |
| Docker's log size limit | `/etc/docker/daemon.json` |
| the `rt` helper and the shortcuts | `~/.bashrc` |
| all Docker data (images, volumes) | `/var/lib/docker` |

What to keep a copy of on your PC: the `plate_for_sanat` folder (code, models, video, `.env`) and `E:\docker_images\sanat_plate_images`. With those you can rebuild the server's stack from nothing (Parts 12 and 13).

---

## Command reference

### Network, on the server

| Goal | Command |
| --- | --- |
| Get the IP (use capital I) | `hostname -I` |
| All interfaces, short | `ip -br a` |
| One interface | `ip -4 a show enp2s0` |
| Test internet and DNS | `ping -c 3 8.8.8.8` then `ping -c 3 google.com` |
| Can the server reach PyPI? | `curl -4 -sS -m 30 -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" https://files.pythonhosted.org/` (it could not: HTTP 000) |
| Bring an interface up | `sudo ip link set enp2s0 up` |
| Apply network config | `sudo netplan apply` |
| Check DHCP is on | `sudo cat /etc/netplan/*.yaml` (expect `dhcp4: true`) |

`hostname -i` (lowercase) returns `127.0.1.1`, which is only a local placeholder and never works for connecting.

### Network, on a Windows PC (the "Media disconnected" troubleshooting)

| Goal | Command |
| --- | --- |
| Show adapter state, IP, gateway, DNS | `ipconfig /all` |
| Test the stack, gateway, internet, DNS | `ping 127.0.0.1`, `ping <gateway>`, `ping 8.8.8.8`, `ping google.com` |
| Renew DHCP (admin Command Prompt) | `ipconfig /release` then `ipconfig /renew` |
| Clear DNS cache | `ipconfig /flushdns` |
| Reset the network stack, then reboot | `netsh winsock reset` and `netsh int ip reset` |
| Remove and reinstall all adapters, then reboot | `netcfg -d` |
| Network adapters window | `Win + R`, then `ncpa.cpl` |

Reading `ipconfig /all`: **Media disconnected** means no physical link (cable, port, NIC, driver, adapter disabled); an address starting **169.254** means DHCP failed; empty Default Gateway means no valid lease. Other fixes that were suggested: reseat or swap the cable, try another router port, set Speed & Duplex to Auto Negotiation, disable Energy Efficient Ethernet and power saving, reinstall the driver in Device Manager, check that Onboard LAN is enabled in BIOS.

### SSH

| Goal | Command |
| --- | --- |
| Connect from your PC | `ssh sanatmadan@<SERVER_IP>` |
| Disconnect | `exit` (twice if you used `newgrp docker`) or Ctrl+D |
| Server: is SSH running? | `sudo systemctl status ssh --no-pager` |
| Server: start and enable SSH | `sudo systemctl enable --now ssh` |
| Server: install SSH if missing | `sudo apt install -y openssh-server` |
| PC: remove an old saved fingerprint (after a reinstall) | `ssh-keygen -R <SERVER_IP>` |

### Copy files between YOUR PC and the server (run on YOUR PC)

| Goal | Command |
| --- | --- |
| a folder to the server | `scp -r C:\path\folder sanatmadan@<SERVER_IP>:~/` |
| one file to the server | `scp C:\path\file.txt sanatmadan@<SERVER_IP>:~/plate-service/` |
| a file from the server to the PC | `scp sanatmadan@<SERVER_IP>:~/plate-service/detector_log.txt C:\Users\eyerik.com\Desktop\` |
| a folder from the server to the PC | `scp -r sanatmadan@<SERVER_IP>:~/plate-service/debug_video C:\Users\eyerik.com\Desktop\` |

### Docker on YOUR PC (build and export)

| Goal | Command |
| --- | --- |
| build the CPU base image | `docker build -f docker/Dockerfile.base-cpu -t base_image_cpu:latest .` |
| build the service images | `docker compose build plate_detector plate_ocr control_hub video_publisher plate_video_publisher_2 plate_video_publisher_3 plate_video_publisher_4` |
| list images (with a filter) | `docker images "eyeplate/*:cpu"` |
| run a Python check inside an image | `docker run --rm --entrypoint python IMAGE -c "import numpy; print(numpy.__version__)"` |
| export images | `docker save IMAGE1 IMAGE2 -o E:\path\file.tar` |
| import images | `docker load -i file.tar` |
| delete an image | `docker rmi IMAGE` |
| docker disk use | `docker system df` |

### Docker on the server

| Goal | Command |
| --- | --- |
| start the stack | see Part 17 |
| stop the stack | `docker compose down` |
| one service's shell | `docker compose exec plate_detector bash` (type `exit` to leave) |
| rebuild nothing, recreate one service | `docker compose up -d --force-recreate --no-build plate_detector` |
| show the compose configuration actually used | `docker compose config` |
| list the service names | `docker compose config --services` |

### Docker cleanup and removal (careful: these delete data)

| Goal | Command |
| --- | --- |
| Remove all stopped containers | `docker rm $(docker ps -aq)` |
| Remove an image | `docker rmi hello-world` |
| Remove unused containers, networks, dangling images | `docker system prune` |
| Remove the build cache | `docker builder prune -a -f` |
| Remove everything unused including volumes | `docker system prune -a --volumes` |
| Uninstall Docker packages | `sudo apt purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin` |
| Delete all Docker data (images, containers, volumes) | `sudo rm -rf /var/lib/docker /var/lib/containerd` |
| Remove the Docker repository and key | `sudo rm /etc/apt/sources.list.d/docker.list /etc/apt/keyrings/docker.asc` |

Run `docker compose down` before `docker image prune -a`, because it removes images no container uses.

### Disk

| Goal | Command |
| --- | --- |
| Free space on `/` | `df -h /` |
| Grow `/` to use the whole disk | `sudo lvextend -r -l +100%FREE /dev/ubuntu-vg/ubuntu-lv` |
| Size of a folder | `du -sh ~/plate-service` |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| USB not in the boot menu | Disable Secure Boot, try a USB 2.0 port, or re-flash in Rufus with **DD Image mode** |
| Black screen or stuck installer | At GRUB press `e`, add `nomodeset` to the end of the `linux` line, press F10 |
| Windows boots after the install | In BIOS, put the **ubuntu** entry first in the boot order |
| `apt update` fails | Check the cable, then `ping 8.8.8.8` and `ping google.com` |
| `permission denied ... docker.sock` | `newgrp docker`, or `exit` and SSH in again |
| `rt list` or `docker ...`: `permission denied while trying to connect to the docker API` | This login is not in the `docker` group yet. Run `id`; if `docker` is missing, `sudo usermod -aG docker $USER`, then `exit` and SSH in again (Part 20) |
| `rt: command not found` | The helper was only defined in an older SSH session. Add it to `~/.bashrc` (Part 13, Shortcuts) and run `source ~/.bashrc` |
| the boot service shows `active (exited)` | That is normal. It only starts the containers and ends. Look at `docker compose ps` for the real state |
| `Unable to locate package docker-ce` | The repository step failed, redo step 3 of Part 8 |
| SSH `Connection timed out` | Wrong IP, or the PC is on a different network |
| SSH `Connection refused` | SSH service is not running (see the SSH table) |
| SSH `Permission denied` | Wrong username or password |
| SSH `REMOTE HOST IDENTIFICATION HAS CHANGED` | `ssh-keygen -R <SERVER_IP>` on your PC |
| After `exit` the prompt is still `sanatmadan@sanatmadan:~$` | You were in the `newgrp docker` sub-shell. Type `exit` again |
| `git`: `Unsupported proxy syntax in ' '` | A proxy variable holds a space. See Part 10.1 |
| `pip` on the server: `Read timed out` / `files.pythonhosted.org` unreachable | The server cannot reach PyPI. Build on YOUR PC and copy the images (Parts 11 and 12) |
| `docker compose` on Windows: `compose file "...yaml#compose.yaml" ... invalid` | `COMPOSE_FILE` contains a `#`. A `#` in `.env` starts a comment. Use `COMPOSE_FILE=compose.yaml:compose.infra.yaml` with `COMPOSE_PATH_SEPARATOR=:` |
| Build: `/bin/sh: python: not found` (exit 127) | The base image has only `python3`. Use `docker/Dockerfile.base-cpu` (it adds the `python` symlink) |
| `ImportError: ... numpy ... file too short`, or `cannot import name '__version__' from 'torch.torch_version'` | A corrupted image (an exported tar damaged by a full disk). Free space, delete the image, rebuild from the Dockerfile |
| `exec /publish_loop.sh: no such file or directory` | Windows line endings. `sed -i 's/\r$//' publish_loop.sh` (the Dockerfile and `.gitattributes` also strip them) |
| `/media/video.mp4: Is a directory` | `video2.mp4` was missing, so Docker made a folder. Delete the folder and copy the real file |
| Cameras show `Network Down` / offline | The camera address must be `publisher` for the test publishers (`--address publisher`) |
| `up --no-build` says an image is missing | The image names did not load. Run `docker images` and compare with Part 11.4. `IMAGE_TAG` in `.env` must be `cpu` |
| Detector stays `health: starting` | Wait 3 minutes, then `docker compose logs plate_detector`. Check the model folder exists on the server |
| `[PERF]` shows `got` far below `target` | The CPU is full (decode, OCR and detection compete). Check `docker stats`, lower the number of cameras, try `openvino_int8`, or reduce OCR workers |
| `no space left on device` | `docker system df`, then `docker image prune -f`, `docker builder prune -a -f`, remove old images |
| `docker info` on Windows shows fewer CPUs than the PC has | `C:\Users\<you>\.wslconfig` limits Docker (`processors=`). Raise it, run `wsl --shutdown`, restart Docker Desktop |

## Not done yet (recommended next)

Security was left out on purpose. Allow SSH **before** enabling the firewall, or you will lock yourself out.

```bash
sudo apt install -y ufw fail2ban unattended-upgrades
sudo ufw allow OpenSSH
sudo ufw enable
sudo systemctl enable --now fail2ban
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

Then set up SSH key login (`ssh-keygen -t ed25519` on your PC, copy the public key to the server), and only after key login works, set `PasswordAuthentication no` in `/etc/ssh/sshd_config` and run `sudo systemctl restart ssh`.

Docker note: ports published with `-p 8080:80` bypass `ufw`. The plate-service compose file publishes Redis (6379) and the service ports to all interfaces, so anyone on your network can reach them. Reserve the server's IP in the router (or set a static IP), because a DHCP address can change.

Also still open for the plate-service itself: the i3-7100 results (fill in the table in Part 9), the choice between `openvino_int8` and `openvino_fp32` after checking accuracy on your own clips, and the number of OCR workers for a 2-core machine.
