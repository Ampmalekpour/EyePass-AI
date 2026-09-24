import os
import sys
import asyncio
import json
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


# --------------------------------------------------------------------
# camera-service is deliberately dependency-light (no torch/opencv/etc
# in its own requirements.txt), so it can't import common/platecore —
# it keeps its own tiny env-parsing helpers instead, same idea as
# platecore.debugging.env_bool/env_int/env_float, just duplicated here
# on purpose.
# --------------------------------------------------------------------
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


DEBUG = os.getenv("CAMERA_STREAM_DEBUG", "false").lower() == "true"

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

# How long an "offline" candidate must stay offline before we actually
# promote it and publish the transition — was a bare literal before,
# never configurable, same bug the face module's camera-service shipped
# with (and was fixed for there).
OFFLINE_HOLD_SECONDS = _float("OFFLINE_HOLD_SECONDS", 10.0)

# When True, "online" is decided purely from mediamtx's own /paths/list
# ("ready": true) and the raw TCP reachability ping is only used for the
# offline error message (Stream Not Ready vs Network Down). Set False to
# require BOTH mtx-ready AND a reachable ping before calling a camera
# online — stricter, but more sensitive to transient network blips.
TRUST_MTX_ONLY = _bool("TRUST_MTX_ONLY", "true")

CAMERA_RTSP_PORT = _int("CAMERA_RTSP_PORT", 554)
CAMERA_PING_TIMEOUT_SEC = _float("CAMERA_PING_TIMEOUT_SEC", 0.2)
MTX_POLL_INTERVAL_SEC = _float("MTX_POLL_INTERVAL_SEC", 0.5)
MTX_API_TIMEOUT_SEC = _float("MTX_API_TIMEOUT_SEC", 1.5)
MONITOR_LOOP_INTERVAL_SEC = _float("MONITOR_LOOP_INTERVAL_SEC", 0.3)
MONITOR_IDLE_SLEEP_SEC = _float("MONITOR_IDLE_SLEEP_SEC", 1.0)
REGISTER_CAMERA_TIMEOUT_SEC = _float("REGISTER_CAMERA_TIMEOUT_SEC", 1.0)
REMOVE_FROM_MTX_TIMEOUT_SEC = _float("REMOVE_FROM_MTX_TIMEOUT_SEC", 1.0)

# Debug/status output — moved under the bind-mounted debug tree so it
# survives container recreation and is inspectable from the host,
# instead of the previous bare relative path (which landed wherever the
# container's cwd happened to be, i.e. nowhere useful on the host).
DEBUG_ROOT_DIR = os.getenv("DEBUG_ROOT_DIR", "/debug")
CAMERA_STATUS_LOG_PATH = os.getenv(
    "CAMERA_STATUS_LOG_PATH", os.path.join(DEBUG_ROOT_DIR, "camera_stream", "camera_status.log")
)

# Optional: periodically save a JPEG snapshot per camera (via ffmpeg) to
# the debug tree, purely for "is this camera actually pointed at what I
# think it is" debugging — mirrors the face module's camera-service
# snapshot-debug feature. Off by default; needs ffmpeg in the image
# (see Dockerfile) and is a plain best-effort side feature — a failure
# here must never affect camera registration/health tracking.
CAMERA_SNAPSHOT_DEBUG_ENABLED = _bool("CAMERA_SNAPSHOT_DEBUG_ENABLED", "false")
CAMERA_SNAPSHOT_DIR = os.getenv("CAMERA_SNAPSHOT_DIR", os.path.join(DEBUG_ROOT_DIR, "camera_stream", "snapshots"))
CAMERA_SNAPSHOT_INTERVAL_SEC = _float("CAMERA_SNAPSHOT_INTERVAL_SEC", 60.0)
CAMERA_SNAPSHOT_TIMEOUT_SEC = _float("CAMERA_SNAPSHOT_TIMEOUT_SEC", 5.0)

LOG_FORMAT = os.getenv("LOG_FORMAT", "text").strip().lower()
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO


class _JsonFormatter(logging.Formatter):
    """Same shape as common/platecore/logging_setup.py's _JsonFormatter —
    duplicated here since this service can't import platecore (see the
    env-helper note above)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                if k not in payload:
                    payload[k] = v
        try:
            return json.dumps(payload, default=str)
        except Exception:
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "logger": payload["logger"], "message": payload["message"]})


_stream_handler = logging.StreamHandler(sys.stdout)
if LOG_FORMAT == "json":
    _stream_handler.setFormatter(_JsonFormatter())
else:
    _stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

logging.basicConfig(level=LOG_LEVEL, handlers=[_stream_handler], force=True)

logger = logging.getLogger("camera_stream")
logger.setLevel(LOG_LEVEL)

status_logger = logging.getLogger("camera_status")
status_logger.setLevel(logging.INFO)
status_logger.propagate = False
try:
    os.makedirs(os.path.dirname(CAMERA_STATUS_LOG_PATH), exist_ok=True)
    status_fh = logging.FileHandler(CAMERA_STATUS_LOG_PATH)
except Exception as e:
    # Bind mount not attached yet, perms issue, etc — status logging is
    # a debug convenience, never a reason to take the service down.
    logger.warning(f"camera_status log file unavailable ({CAMERA_STATUS_LOG_PATH}): {e}; falling back to stdout only")
    status_fh = logging.StreamHandler(sys.stdout)
status_fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
status_logger.addHandler(status_fh)

mtx_cache = {}
last_status = {}
pending_status = {}  # فقط برای candidate آفلاین: cam_id -> monotonic start time
active_cameras = []
registered_cameras = set()
r = None
current_module = ""
_last_snapshot_at = {}  # cam_id -> monotonic time of last saved snapshot

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

async def is_reachable(ip, port=CAMERA_RTSP_PORT, timeout=CAMERA_PING_TIMEOUT_SEC):
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


async def save_camera_snapshot(cam):
    """Best-effort periodic JPEG snapshot per camera, purely for visual
    debugging ("is this camera really pointed where I think it is").
    Never raises — a failure here must never affect camera
    registration/health tracking, which is why this is called via
    asyncio.create_task and wrapped end-to-end in try/except."""
    if not CAMERA_SNAPSHOT_DEBUG_ENABLED or not cam.address:
        return
    now = time.monotonic()
    last = _last_snapshot_at.get(cam.id, 0.0)
    if now - last < CAMERA_SNAPSHOT_INTERVAL_SEC:
        return
    _last_snapshot_at[cam.id] = now
    try:
        os.makedirs(CAMERA_SNAPSHOT_DIR, exist_ok=True)
        out_path = os.path.join(CAMERA_SNAPSHOT_DIR, f"{cam.id}.jpg")
        tmp_path = out_path + ".tmp"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", cam.address,
            "-frames:v", "1", "-q:v", "3", tmp_path,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=CAMERA_SNAPSHOT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            return
        if proc.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, out_path)
    except FileNotFoundError:
        logger.debug("CAMERA_SNAPSHOT_DEBUG_ENABLED is set but ffmpeg is not installed in this image")
    except Exception as e:
        logger.debug(f"snapshot failed for cam={cam.id}: {e}")

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

        # NOTE: this MUST be the singular "camera:events", matching
        # common/platecore/keys.py's RedisKeys.cameras_events exactly
        # (that property's own docstring warns not to "fix" this to the
        # plural spelling — plural is what some OTHER module's copy of
        # camera_service uses, and publishing there would silently stop
        # working against this module's real camera-stream container,
        # matching the deployed eyepass-camera-stream's own channel
        # naming). This shipped copy previously used the plural
        # "cameras:events", which meant plate_detector's subscriber
        # (backend_bridge.py's on_camera_event, via bus.keys.cameras_events)
        # never received these events at all.
        events_channel = f"{current_module}:camera:events"
        pipe.publish(events_channel, json.dumps(details_data))

    if ping_ok and cam.id not in registered_cameras:
        asyncio.create_task(register_camera(client, cam))

    if CAMERA_SNAPSHOT_DEBUG_ENABLED:
        asyncio.create_task(save_camera_snapshot(cam))

async def monitor_loop():
    async with httpx.AsyncClient(auth=AUTH if AUTH else None) as client:
        while True:
            if not active_cameras:
                await asyncio.sleep(MONITOR_IDLE_SLEEP_SEC)
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