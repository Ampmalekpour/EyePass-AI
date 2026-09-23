import os
import sys
import asyncio
import json
import subprocess
import time
import httpx
import logging
import redis.asyncio as redis
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
env_path = base_path / "config" / ".env"
load_dotenv(dotenv_path=env_path)


# ---------------------------------------------------------------------
# ENV PARSING — this service is deliberately kept dependency-light
# (redis + httpx + python-dotenv only, see requirements.txt) and is
# shared infrastructure used by more than just this face module (see
# discover_modules() below, which scans for *ANY* "<module>:cameras:
# config" prefix) — so it does not import common/facecore the way the
# detector/recognizer do. These are the same _bool/_int/_float
# semantics as every other config.py in this repo, just kept local.
# ---------------------------------------------------------------------
def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEBUG = _bool("CAMERA_STREAM_DEBUG", "false")

MTX_HOST = os.getenv("MTX_HOST")
MTX_PORT = os.getenv("MTX_PORT")
MTX_USER = os.getenv("MTX_USER")
MTX_PASS = os.getenv("MTX_PASS")

MTX_API_BASE = f"http://{MTX_HOST}:{MTX_PORT}/v3"
MTX_API_ADD = f"{MTX_API_BASE}/config/paths/add/"
MTX_API_LIST = f"{MTX_API_BASE}/paths/list"
MTX_API_DELETE = f"{MTX_API_BASE}/config/paths/delete/"

AUTH = None

REDIS_URL = os.getenv("REDIS_URL")

STREAM_BASE_URL = os.getenv("STREAM_BASE_URL")

# ---- operational tunables (ALL env-overridable; defaults preserve the
# exact behaviour this file had before) ------------------------------
# NOTE: this was previously a bare module-level literal (`= 10`) that
# never actually read the OFFLINE_HOLD_SECONDS env var — compose.yaml
# passed it into the container, but nothing in this file consumed it.
# Fixed here: this now genuinely is the env-configurable value.
OFFLINE_HOLD_SECONDS = _float("OFFLINE_HOLD_SECONDS", 10.0)

# When True, a camera counts as "online" purely from MediaMTX's own
# `ready` flag; when False, it additionally requires a successful raw
# TCP probe to the camera's RTSP port (see is_reachable()). Was a
# hardcoded `True` with no way to turn it off short of editing code.
TRUST_MTX_ONLY = _bool("TRUST_MTX_ONLY", "true")

# TCP reachability probe (is_reachable()) — was hardcoded (port=554,
# timeout=0.2) as function-default arguments.
CAMERA_RTSP_PORT = _int("CAMERA_RTSP_PORT", 554)
CAMERA_PING_TIMEOUT_SEC = _float("CAMERA_PING_TIMEOUT_SEC", 0.2)

# mediamtx_poller() / monitor_loop() poll cadence — were hardcoded
# `await asyncio.sleep(0.5)` / `await asyncio.sleep(0.3)` inline.
MTX_POLL_INTERVAL_SEC = _float("MTX_POLL_INTERVAL_SEC", 0.5)
MONITOR_LOOP_INTERVAL_SEC = _float("MONITOR_LOOP_INTERVAL_SEC", 0.3)
MTX_API_TIMEOUT_SEC = _float("MTX_API_TIMEOUT_SEC", 1.5)
REGISTER_CAMERA_TIMEOUT_SEC = _float("REGISTER_CAMERA_TIMEOUT_SEC", 1.0)
REMOVE_FROM_MTX_TIMEOUT_SEC = _float("REMOVE_FROM_MTX_TIMEOUT_SEC", 1.0)

# ---- logging --------------------------------------------------------
# DEBUG_ROOT_DIR is the same shared, bind-mounted parent every section
# of this system now writes its visual/status debug output under (see
# DEBUGGING.md at the repo root) — one folder per service, so an
# operator only ever has to look in one place on the host disk.
DEBUG_ROOT_DIR = os.getenv("DEBUG_ROOT_DIR", "/debug")
CAMERA_STATUS_LOG_PATH = os.getenv(
    "CAMERA_STATUS_LOG_PATH",
    os.path.join(DEBUG_ROOT_DIR, "camera-service", "camera_status.log"),
)

LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("camera_stream")
logger.setLevel(LOG_LEVEL)

status_logger = logging.getLogger("camera_status")
status_logger.setLevel(logging.INFO)
status_logger.propagate = False
try:
    os.makedirs(os.path.dirname(CAMERA_STATUS_LOG_PATH), exist_ok=True)
    status_fh = logging.FileHandler(CAMERA_STATUS_LOG_PATH)
except Exception as e:
    # Debug output must never be able to take this service down: fall
    # back to the old cwd-relative path (still useful inside the
    # container even if the shared bind mount isn't attached).
    logger.warning("could not open %s (%s); falling back to ./camera_status.log", CAMERA_STATUS_LOG_PATH, e)
    status_fh = logging.FileHandler('camera_status.log')
status_fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
status_logger.addHandler(status_fh)


# ---------------------------------------------------------------------
# OPTIONAL visual debug: a single still frame per camera, pulled off
# the SAME MediaMTX relay every other consumer (the detector included)
# reads from — this is the "is this camera actually producing a usable
# image" check described in DEBUGGING.md, done without adding a heavy
# video dependency (opencv/torch) to what is otherwise a deliberately
# minimal redis+httpx container.
#
# Uses `ffmpeg` as a subprocess (grab exactly one frame, then exit) —
# NOT bundled in this image by default (see camera-service/Dockerfile),
# so this stays OFF by default and fails soft + logs once per camera
# if the binary isn't present, rather than crashing the poll loop.
# To turn it on: set CAMERA_SNAPSHOT_DEBUG_ENABLED=true AND add
# `apt-get install -y --no-install-recommends ffmpeg` to the
# Dockerfile (one line — see the comment there).
# ---------------------------------------------------------------------
CAMERA_SNAPSHOT_DEBUG_ENABLED = _bool("CAMERA_SNAPSHOT_DEBUG_ENABLED", "false")
CAMERA_SNAPSHOT_INTERVAL_SEC = _float("CAMERA_SNAPSHOT_INTERVAL_SEC", 60.0)
CAMERA_SNAPSHOT_TIMEOUT_SEC = _float("CAMERA_SNAPSHOT_TIMEOUT_SEC", 5.0)
CAMERA_SNAPSHOT_MAX_PER_CAMERA = _int("CAMERA_SNAPSHOT_MAX_PER_CAMERA", 20)

_snapshot_last_attempt: dict = {}
_ffmpeg_warned = False


def _snapshot_dir(cam_id: str) -> str:
    return os.path.join(DEBUG_ROOT_DIR, "camera-service", f"camera_{cam_id}")


def _prune_snapshots(dir_path: str, max_files: int) -> None:
    if max_files <= 0:
        return
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]
        files.sort(key=lambda p: os.path.getmtime(p))
        for old in files[:-max_files]:
            try:
                os.remove(old)
            except Exception:
                pass
    except Exception:
        pass


def maybe_capture_snapshot(cam) -> None:
    """Best-effort, fail-safe, rate-limited: grabs one JPEG frame from
    this camera's relay path and drops it under
    DEBUG_ROOT_DIR/camera-service/camera_<id>/. Never raises — any
    failure here must never take mediamtx_poller()/monitor_loop() down."""
    global _ffmpeg_warned
    if not CAMERA_SNAPSHOT_DEBUG_ENABLED:
        return
    try:
        now = time.monotonic()
        last = _snapshot_last_attempt.get(cam.id, 0.0)
        if (now - last) < CAMERA_SNAPSHOT_INTERVAL_SEC:
            return
        _snapshot_last_attempt[cam.id] = now

        relay_url = f"{STREAM_BASE_URL}/{cam.id}/" if STREAM_BASE_URL else None
        if not relay_url:
            return

        out_dir = _snapshot_dir(cam.id)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"snapshot_{stamp}.jpg")

        proc = subprocess.run(
            ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", relay_url,
             "-frames:v", "1", "-q:v", "3", out_path],
            capture_output=True, timeout=CAMERA_SNAPSHOT_TIMEOUT_SEC,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            logger.debug("[SNAPSHOT] cam=%s ffmpeg exit=%s: %s",
                         cam.id, proc.returncode, proc.stderr[-300:] if proc.stderr else "")
            return

        _prune_snapshots(out_dir, CAMERA_SNAPSHOT_MAX_PER_CAMERA)
    except FileNotFoundError:
        if not _ffmpeg_warned:
            _ffmpeg_warned = True
            logger.warning(
                "[SNAPSHOT] CAMERA_SNAPSHOT_DEBUG_ENABLED=true but `ffmpeg` is not installed "
                "in this container — add it to camera-service/Dockerfile to use this feature. "
                "Disabling snapshot capture for the rest of this run."
            )
    except Exception as e:
        logger.debug("[SNAPSHOT] cam=%s failed: %s", cam.id, e)

mtx_cache = {}
last_status = {}
pending_status = {}  # فقط برای candidate آفلاین: cam_id -> monotonic start time
active_cameras = []
registered_cameras = set()
r = None
current_module = ""

class Camera:
    def __init__(self, data_json):
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        self.id = str(data.get('id', '')).strip()
        self.title = data.get('title', 'Unknown')
        self.address = data.get('address', '')
        self.usage = data.get('usage', '')
        self.roi = data.get('roi', {})
        self.stop_roi = data.get('stop_roi', {})
        self.cross_line = data.get('cross_line', {})
        self.raw_data = data
        self.ip = self._extract_ip(self.address)

    @staticmethod
    def _extract_ip(address: str):
        if not address:
            return None
        try:
            parsed = urlparse(address)
            return parsed.hostname
        except Exception:
            return None

def merge_camera_fields(details_data, cam):
    details_data['id'] = cam.id
    details_data['title'] = cam.title
    details_data['address'] = cam.address
    details_data['usage'] = cam.usage
    details_data['roi'] = cam.roi
    details_data['stop_roi'] = cam.stop_roi
    details_data['cross_line'] = cam.cross_line
    details_data['stream_url'] = f'{STREAM_BASE_URL}/{cam.id}/'
    return details_data

async def is_reachable(ip, port=None, timeout=None):
    port = CAMERA_RTSP_PORT if port is None else port
    timeout = CAMERA_PING_TIMEOUT_SEC if timeout is None else timeout
    if not ip: return False
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except: return False

async def remove_from_mtx(client, cam_id):
    try:
        res = await client.delete(MTX_API_DELETE + cam_id, timeout=REMOVE_FROM_MTX_TIMEOUT_SEC)
        if res.status_code in [200, 404]:
            registered_cameras.discard(cam_id)
            return True
    except: return False

async def sync_cameras(client, module_name):
    global active_cameras, registered_cameras, current_module
    current_module = module_name
    
    config_key = f"{module_name}:cameras:config"
    details_key = f"{module_name}:cameras:details"
    
    stored = await r.hgetall(config_key)
    current_ids = set(stored.keys())
    
    to_remove = registered_cameras - current_ids
    if to_remove:
        await asyncio.gather(*(remove_from_mtx(client, rid) for rid in to_remove))

    pipe = r.pipeline()
    for rid in to_remove:
        pipe.hdel(details_key, rid)
        if rid in last_status: del last_status[rid]
        if rid in pending_status: del pending_status[rid]
        if rid in mtx_cache: del mtx_cache[rid]

    all_detail_keys = await r.hkeys(details_key)
    for dk in all_detail_keys:
        if dk not in current_ids:
            pipe.hdel(details_key, dk)

    await pipe.execute()

    active_cameras = []
    for cid, data_json in stored.items():
        try:
            cam = Camera(data_json)
            active_cameras.append(cam)
            
            existing_details = await r.hget(details_key, cid)
            details_data = json.loads(existing_details) if existing_details else {}
            
            config_data = json.loads(data_json) if isinstance(data_json, str) else data_json
            for key, value in config_data.items():
                details_data[key] = value
            
            details_data = merge_camera_fields(details_data, cam)
            
            if 'connected' not in details_data:
                details_data['connected'] = False
            if 'error' not in details_data:
                details_data['error'] = "Waiting for connection"
            
            await r.hset(details_key, cid, json.dumps(details_data))
                
        except Exception as e:
            logger.error(f"Error parsing camera data for {cid}: {e}")

async def discover_modules():
    modules = set()
    async for key in r.scan_iter(match="*:cameras:config"):
        modules.add(key.split(':')[0])
    return modules

async def initial_sync(client):
    modules = await discover_modules()
    for module_name in modules:
        await sync_cameras(client, module_name)

async def config_listener():
    pubsub = r.pubsub()
    await pubsub.psubscribe("*:camera:config:updated")
    async with httpx.AsyncClient(auth=AUTH if AUTH else None) as client:
        async for message in pubsub.listen():
            if message['type'] == 'pmessage':
                channel = message['channel']
                module_name = channel.split(':')[0]
                await sync_cameras(client, module_name)

async def mediamtx_poller():
    global mtx_cache
    async with httpx.AsyncClient(auth=AUTH if AUTH else None) as client:
        while True:
            try:
                res = await client.get(MTX_API_LIST, timeout=MTX_API_TIMEOUT_SEC)
                logger.debug(f'MEDIA MTX RES: {res.json()}')
                if res.status_code == 200:
                    items = res.json().get("items", [])
                    mtx_cache = {str(i["name"]).strip(): i.get("ready", False) for i in items}
            except httpx.NetworkError as e:
                logger.critical(f'Network Error: {e}')
            except Exception as e:
                logger.critical(f"❌ Error in mediamtx_poller: {repr(e)}")
                logger.critical(f"❌ MTX_API_LIST: {MTX_API_LIST}")
                logger.critical(f"❌ AUTH: {AUTH}")
            await asyncio.sleep(MTX_POLL_INTERVAL_SEC)

async def register_camera(client, cam):
    if cam.id in registered_cameras: return
    payload = {"source": cam.address, "sourceOnDemand": False, "rtspTransport": "tcp"}
    try:
        res = await client.post(MTX_API_ADD + cam.id, json=payload, timeout=REGISTER_CAMERA_TIMEOUT_SEC)
        logger.critical(f"✅ success in register_camera")
        if res.status_code in [200, 201, 400]:
            registered_cameras.add(cam.id)
    except Exception as e: 
        logger.critical(f"❌ Error in register_camera: {str(e)}")

async def check_camera(client, cam, pipe):
    if not cam.address or not cam.id: return
    
    ping_ok = await is_reachable(cam.ip)
    mtx_ready = mtx_cache.get(cam.id, False)
    
    connected = mtx_ready if TRUST_MTX_ONLY else (ping_ok and mtx_ready)
    status = "online" if connected else "offline"
    
    if cam.id not in last_status:
        last_status[cam.id] = "unknown"
    
    old_status = last_status[cam.id]
    now = time.monotonic()
    promoted = False
    held_for = 0.0

    if status == "online":
        pending_status.pop(cam.id, None)
        if old_status != "online":
            last_status[cam.id] = "online"
            promoted = True
    else:
        if old_status == "offline":
            pending_status.pop(cam.id, None)
        else:
            started_at = pending_status.get(cam.id)
            if started_at is None:
                pending_status[cam.id] = now
                logger.debug(f"[HOLD] cam={cam.id} candidate=offline started")
            else:
                held_for = now - started_at
                if held_for >= OFFLINE_HOLD_SECONDS:
                    last_status[cam.id] = "offline"
                    pending_status.pop(cam.id, None)
                    promoted = True
                else:
                    logger.debug(
                        f"[HOLD] cam={cam.id} candidate=offline "
                        f"elapsed={held_for:.1f}s / {OFFLINE_HOLD_SECONDS}s"
                    )

    if promoted:
        status_logger.debug(
            f"CAMERA: {cam.id} | TITLE: {cam.title} | "
            f"CHANGE: {old_status} -> {status} (held {held_for:.1f}s)"
        )

        details_key = f"{current_module}:cameras:details"
        existing_details = await r.hget(details_key, cam.id)
        details_data = json.loads(existing_details) if existing_details else {}

        details_data = merge_camera_fields(details_data, cam)
        details_data['connected'] = connected
        details_data['error'] = None if connected else ("Stream Not Ready" if ping_ok else "Network Down")
        details_data['updated'] = datetime.now().strftime('%H:%M:%S')

        pipe.hset(details_key, cam.id, json.dumps(details_data))

        events_channel = f"{current_module}:cameras:events"
        pipe.publish(events_channel, json.dumps(details_data))

    if ping_ok and cam.id not in registered_cameras:
        asyncio.create_task(register_camera(client, cam))

    if connected:
        maybe_capture_snapshot(cam)

async def monitor_loop():
    async with httpx.AsyncClient(auth=AUTH if AUTH else None) as client:
        while True:
            if not active_cameras:
                await asyncio.sleep(1)
                continue
            pipe = r.pipeline()
            await asyncio.gather(*(check_camera(client, cam, pipe) for cam in active_cameras))
            await pipe.execute()
            await asyncio.sleep(MONITOR_LOOP_INTERVAL_SEC)

async def main():
    global r, current_module

    logger.info("🚀 camera_stream main() started")
    logger.info(f"DEBUG={DEBUG}")
    logger.info(f"MTX_HOST={MTX_HOST}")
    logger.info(f"MTX_PORT={MTX_PORT}")
    logger.info(f"MTX_API_BASE={MTX_API_BASE}")
    logger.info(f"STREAM_BASE_URL={STREAM_BASE_URL}")
    logger.info(f"REDIS_URL is set: {bool(REDIS_URL)}")
    logger.info(f"OFFLINE_HOLD_SECONDS={OFFLINE_HOLD_SECONDS}")
    logger.info(f"TRUST_MTX_ONLY={TRUST_MTX_ONLY}  CAMERA_RTSP_PORT={CAMERA_RTSP_PORT}  "
                f"CAMERA_PING_TIMEOUT_SEC={CAMERA_PING_TIMEOUT_SEC}")
    logger.info(f"CAMERA_SNAPSHOT_DEBUG_ENABLED={CAMERA_SNAPSHOT_DEBUG_ENABLED} "
                f"(dir={DEBUG_ROOT_DIR}/camera-service/camera_<id>/)")

    r = redis.from_url(REDIS_URL, decode_responses=True)
    await r.ping()

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=20)
    async with httpx.AsyncClient(auth=AUTH if AUTH else None, limits=limits) as client:
        await initial_sync(client)
        await asyncio.gather(
            config_listener(),
            mediamtx_poller(),
            monitor_loop()
        )

if __name__ == "__main__":
    logger.info("🚀 camera_stream container started")
    try:
        asyncio.run(main())
    except KeyboardInterrupt: pass