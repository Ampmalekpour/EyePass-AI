"""
camera_stream — the suite's single camera manager
--------------------------------------------------------------------
One instance serves every enabled AI module (face, plate, heatmap,
fire). It replaces the four per-module copies, which each assumed they
were the only module on their relay (one global camera list, one
"current module" — syncing module B removed module A's cameras).

What it does, per module m in {face, plate, heatmap, fire, ...}:
  reads     m:cameras:config            (HASH camera_id -> json, written by the backend)
  listens   *:camera:config:updated     (pub/sub nudge from the backend) + a periodic resync
  registers each camera's RTSP address on MediaMTX as ONE relay path per
            physical camera (relay.py: "cam_" + sha1(address)) — shared by
            every module that uses that camera, removed when none does
  watches   MediaMTX /paths/list + a TCP reachability probe, with an
            OFFLINE_HOLD_SECONDS debounce, per relay path
  writes    m:cameras:details           (connected, error, relay_path, stream_url, ...)
  publishes the module's camera-events channel on every online/offline
            transition — face listens on "face:cameras:events" (plural),
            the others on "<m>:camera:events"; see EVENTS_CHANNELS.

Self-healing: it holds no state of its own. On (re)start it rebuilds
everything from the modules' cameras:config in Redis; if MediaMTX
restarts and forgets its runtime paths, the next poll notices they are
gone and re-registers them.
--------------------------------------------------------------------
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
import redis.asyncio as redis

from relay import RELAY_PATH_MODE, relay_path_for


def _bool(name, default):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEBUG = _bool("CAMERA_STREAM_DEBUG", "false")
MTX_HOST = os.getenv("MTX_HOST", "mediamtx")
MTX_PORT = os.getenv("MTX_PORT", "9997")
MTX_USER = os.getenv("MTX_USER")
MTX_PASS = os.getenv("MTX_PASS")
MTX_API_BASE = f"http://{MTX_HOST}:{MTX_PORT}/v3"
MTX_API_ADD = f"{MTX_API_BASE}/config/paths/add/"
MTX_API_LIST = f"{MTX_API_BASE}/paths/list"
MTX_API_DELETE = f"{MTX_API_BASE}/config/paths/delete/"
AUTH = (MTX_USER, MTX_PASS) if MTX_USER and MTX_PASS else None

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
STREAM_BASE_URL = (os.getenv("STREAM_BASE_URL") or "").rstrip("/")

# Modules to serve. Explicit list (the suite sets it from the enabled
# profiles) plus anything discovered as "<m>:cameras:config" in Redis.
CAMERA_MODULES = [m.strip() for m in os.getenv("CAMERA_MODULES", "").split(",") if m.strip()]

# Per-module camera-events channel. Face's detector listens on the plural
# spelling, plate/heatmap/fire on the singular one (their keys.py files).
EVENTS_CHANNEL_TEMPLATE = os.getenv("CAMERA_EVENTS_CHANNEL_TEMPLATE", "{module}:camera:events")
EVENTS_CHANNELS = {"face": "face:cameras:events"}
for _k, _v in os.environ.items():
    if _k.startswith("CAMERA_EVENTS_CHANNEL_") and _k != "CAMERA_EVENTS_CHANNEL_TEMPLATE" and _v:
        EVENTS_CHANNELS[_k[len("CAMERA_EVENTS_CHANNEL_"):].lower()] = _v

OFFLINE_HOLD_SECONDS = _float("OFFLINE_HOLD_SECONDS", 10.0)
TRUST_MTX_ONLY = _bool("TRUST_MTX_ONLY", "true")
CAMERA_RTSP_PORT = _int("CAMERA_RTSP_PORT", 554)
CAMERA_PING_TIMEOUT_SEC = _float("CAMERA_PING_TIMEOUT_SEC", 0.2)
MTX_POLL_INTERVAL_SEC = _float("MTX_POLL_INTERVAL_SEC", 0.5)
MTX_API_TIMEOUT_SEC = _float("MTX_API_TIMEOUT_SEC", 1.5)
MONITOR_LOOP_INTERVAL_SEC = _float("MONITOR_LOOP_INTERVAL_SEC", 0.3)
REGISTER_CAMERA_TIMEOUT_SEC = _float("REGISTER_CAMERA_TIMEOUT_SEC", 1.0)
REMOVE_FROM_MTX_TIMEOUT_SEC = _float("REMOVE_FROM_MTX_TIMEOUT_SEC", 1.0)
CONFIG_RESYNC_SEC = _float("CONFIG_RESYNC_SEC", 30.0)

DEBUG_ROOT_DIR = os.getenv("DEBUG_ROOT_DIR", "/debug")
CAMERA_STATUS_LOG_PATH = os.getenv("CAMERA_STATUS_LOG_PATH",
                                   os.path.join(DEBUG_ROOT_DIR, "camera_stream", "camera_status.log"))
CAMERA_SNAPSHOT_DEBUG_ENABLED = _bool("CAMERA_SNAPSHOT_DEBUG_ENABLED", "false")
CAMERA_SNAPSHOT_DIR = os.getenv("CAMERA_SNAPSHOT_DIR", os.path.join(DEBUG_ROOT_DIR, "camera_stream", "snapshots"))
CAMERA_SNAPSHOT_INTERVAL_SEC = _float("CAMERA_SNAPSHOT_INTERVAL_SEC", 60.0)
CAMERA_SNAPSHOT_TIMEOUT_SEC = _float("CAMERA_SNAPSHOT_TIMEOUT_SEC", 5.0)

# ---------------------------------------------------------------- logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger("camera_stream")
# httpx logs every request at INFO — the MediaMTX poll alone would be
# two lines per second
logging.getLogger("httpx").setLevel(logging.DEBUG if DEBUG else logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
status_logger = logging.getLogger("camera_status")
status_logger.setLevel(logging.INFO)
status_logger.propagate = False
try:
    os.makedirs(os.path.dirname(CAMERA_STATUS_LOG_PATH), exist_ok=True)
    _fh = logging.FileHandler(CAMERA_STATUS_LOG_PATH)
except Exception as e:  # noqa: BLE001
    logger.warning("camera_status log unavailable (%s): %s — stdout only", CAMERA_STATUS_LOG_PATH, e)
    _fh = logging.StreamHandler(sys.stdout)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
status_logger.addHandler(_fh)


def events_channel(module: str) -> str:
    return EVENTS_CHANNELS.get(module, EVENTS_CHANNEL_TEMPLATE.format(module=module))


# ======================================================================
# State (all rebuilt from Redis on start — nothing to persist)
# ======================================================================
class Camera:
    def __init__(self, module: str, cam_id: str, data: dict):
        self.module = module
        self.id = str(cam_id)
        self.data = data
        self.title = data.get("title", "Unknown")
        self.address = (data.get("address") or "").strip()
        self.relay_path = relay_path_for(self.id, self.address)


class RelayPath:
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        parsed = urlparse(address) if address else None
        self.host = parsed.hostname if parsed else None
        self.port = (parsed.port if parsed and parsed.port else CAMERA_RTSP_PORT)
        self.refs: Set[Tuple[str, str]] = set()      # (module, camera_id)
        self.registered = False
        self.status = "unknown"                      # online | offline | unknown
        self.pending_offline_since: Optional[float] = None
        self.last_ping_ok = False
        self.last_snapshot = 0.0


cameras: Dict[Tuple[str, str], Camera] = {}
paths: Dict[str, RelayPath] = {}
mtx_ready: Dict[str, bool] = {}
mtx_poll_ok = False
r: Optional[redis.Redis] = None
_sync_lock = asyncio.Lock()


def details_key(module: str) -> str:
    return f"{module}:cameras:details"


def _details_for(cam: Camera, existing: dict, connected: bool, error: Optional[str]) -> dict:
    d = dict(existing or {})
    d.update(cam.data)                     # everything the backend configured
    d["id"] = cam.id
    d["relay_path"] = cam.relay_path
    d["stream_url"] = f"{STREAM_BASE_URL}/{cam.relay_path}/" if STREAM_BASE_URL else f"/{cam.relay_path}/"
    d["connected"] = connected
    d["error"] = error
    return d


# ======================================================================
# MediaMTX
# ======================================================================
async def mtx_register(client: httpx.AsyncClient, p: RelayPath):
    payload = {"source": p.address, "sourceOnDemand": False, "rtspTransport": "tcp"}
    try:
        res = await client.post(MTX_API_ADD + p.name, json=payload, timeout=REGISTER_CAMERA_TIMEOUT_SEC)
        if res.status_code in (200, 201, 400):   # 400 = already exists
            if not p.registered:
                logger.info("relay path %s registered -> %s (%d user(s))", p.name, p.address, len(p.refs))
            p.registered = True
        else:
            logger.warning("registering %s failed: HTTP %s %s", p.name, res.status_code, res.text[:200])
    except Exception as e:  # noqa: BLE001
        logger.warning("registering %s failed: %s", p.name, e)


async def mtx_remove(client: httpx.AsyncClient, name: str):
    try:
        res = await client.delete(MTX_API_DELETE + name, timeout=REMOVE_FROM_MTX_TIMEOUT_SEC)
        if res.status_code in (200, 404):
            logger.info("relay path %s removed (no module uses it any more)", name)
    except Exception as e:  # noqa: BLE001
        logger.warning("removing relay path %s failed: %s (will be retried on next sync)", name, e)


async def mediamtx_poller():
    global mtx_ready, mtx_poll_ok
    async with httpx.AsyncClient(auth=AUTH) as client:
        while True:
            try:
                res = await client.get(MTX_API_LIST, timeout=MTX_API_TIMEOUT_SEC)
                if res.status_code == 200:
                    items = res.json().get("items", [])
                    mtx_ready = {str(i["name"]).strip(): bool(i.get("ready", False)) for i in items}
                    if not mtx_poll_ok:
                        logger.info("MediaMTX API reachable (%d path(s))", len(mtx_ready))
                    mtx_poll_ok = True
                    # MediaMTX restarted and lost its runtime paths -> re-register
                    for p in paths.values():
                        if p.registered and p.name not in mtx_ready:
                            logger.warning("relay path %s vanished from MediaMTX (restart?) — re-registering", p.name)
                            p.registered = False
            except Exception as e:  # noqa: BLE001
                if mtx_poll_ok:
                    logger.error("MediaMTX API unreachable: %s", e)
                mtx_poll_ok = False
            await asyncio.sleep(MTX_POLL_INTERVAL_SEC)


# ======================================================================
# Config sync
# ======================================================================
async def discover_modules() -> List[str]:
    found = set(CAMERA_MODULES)
    async for key in r.scan_iter(match="*:cameras:config"):
        found.add(key.split(":")[0])
    return sorted(found)


async def sync_module(client: httpx.AsyncClient, module: str):
    stored = await r.hgetall(f"{module}:cameras:config")
    wanted: Dict[str, Camera] = {}
    for cid, raw in stored.items():
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            cam = Camera(module, cid, data or {})
            if not cam.address:
                logger.warning("[%s] camera %s has no address — skipped", module, cid)
                continue
            wanted[cam.id] = cam
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] bad config for camera %s: %s", module, cid, e)

    removed_paths: List[str] = []
    # drop cameras this module no longer has (or whose address changed)
    for key in [k for k in cameras if k[0] == module]:
        old = cameras[key]
        new = wanted.get(key[1])
        if new is None or new.relay_path != old.relay_path:
            p = paths.get(old.relay_path)
            if p:
                p.refs.discard(key)
                if not p.refs:
                    paths.pop(old.relay_path, None)
                    removed_paths.append(old.relay_path)
            del cameras[key]
    for name in removed_paths:
        await mtx_remove(client, name)

    dkey = details_key(module)
    pipe = r.pipeline()
    for dk in await r.hkeys(dkey):
        if dk not in wanted:
            pipe.hdel(dkey, dk)
    for cid, cam in wanted.items():
        cameras[(module, cid)] = cam
        p = paths.get(cam.relay_path)
        if p is None:
            p = paths[cam.relay_path] = RelayPath(cam.relay_path, cam.address)
        p.refs.add((module, cid))
        existing = await r.hget(dkey, cid)
        existing = json.loads(existing) if existing else {}
        connected = p.status == "online"
        error = None if connected else (existing.get("error") or "Waiting for connection")
        pipe.hset(dkey, cid, json.dumps(_details_for(cam, existing, connected, error)))
    await pipe.execute()


async def full_sync(client: httpx.AsyncClient):
    async with _sync_lock:
        for m in await discover_modules():
            try:
                await sync_module(client, m)
            except Exception as e:  # noqa: BLE001
                logger.error("sync of module %s failed: %s", m, e)
        logger.debug("sync: %d camera(s) across %d relay path(s)", len(cameras), len(paths))


async def config_listener(client: httpx.AsyncClient):
    while True:
        try:
            pubsub = r.pubsub()
            await pubsub.psubscribe("*:camera:config:updated", "*:cameras:config:updated")
            logger.info("listening for camera config updates")
            async for message in pubsub.listen():
                if message.get("type") == "pmessage":
                    module = str(message["channel"]).split(":")[0]
                    async with _sync_lock:
                        await sync_module(client, module)
        except Exception as e:  # noqa: BLE001
            logger.error("config listener dropped (%s) — reconnecting in 2s", e)
            await asyncio.sleep(2)


async def periodic_resync(client: httpx.AsyncClient):
    """Covers a missed pub/sub nudge and modules enabled after start."""
    while True:
        await asyncio.sleep(CONFIG_RESYNC_SEC)
        await full_sync(client)


# ======================================================================
# Health monitoring (per relay path, fanned out to every module using it)
# ======================================================================
async def is_reachable(host, port) -> bool:
    if not host:
        return False
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=CAMERA_PING_TIMEOUT_SEC)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:  # noqa: BLE001
        return False


async def publish_transition(p: RelayPath, connected: bool, ping_ok: bool, held: float):
    error = None if connected else ("Stream Not Ready" if ping_ok else "Network Down")
    pipe = r.pipeline()
    for module, cid in sorted(p.refs):
        cam = cameras.get((module, cid))
        if cam is None:
            continue
        dkey = details_key(module)
        existing = await r.hget(dkey, cid)
        d = _details_for(cam, json.loads(existing) if existing else {}, connected, error)
        d["updated"] = datetime.now().strftime("%H:%M:%S")
        pipe.hset(dkey, cid, json.dumps(d))
        pipe.publish(events_channel(module), json.dumps(d))
        status_logger.info("MODULE: %s | CAMERA: %s | TITLE: %s | PATH: %s | %s (held %.1fs)",
                           module, cid, cam.title, p.name, "ONLINE" if connected else f"OFFLINE ({error})", held)
    await pipe.execute()


async def check_path(client: httpx.AsyncClient, p: RelayPath):
    ping_ok = await is_reachable(p.host, p.port)
    p.last_ping_ok = ping_ok
    if not p.registered and (ping_ok or TRUST_MTX_ONLY):
        await mtx_register(client, p)
    ready = mtx_ready.get(p.name, False)
    connected = ready if TRUST_MTX_ONLY else (ready and ping_ok)
    now = time.monotonic()
    if connected:
        p.pending_offline_since = None
        if p.status != "online":
            p.status = "online"
            await publish_transition(p, True, ping_ok, 0.0)
    elif p.status != "offline":
        if p.pending_offline_since is None:
            p.pending_offline_since = now
        held = now - p.pending_offline_since
        # never seen online yet: report offline immediately, no debounce
        if p.status == "unknown" or held >= OFFLINE_HOLD_SECONDS:
            p.status = "offline"
            p.pending_offline_since = None
            await publish_transition(p, False, ping_ok, held)
    if CAMERA_SNAPSHOT_DEBUG_ENABLED and now - p.last_snapshot >= CAMERA_SNAPSHOT_INTERVAL_SEC:
        p.last_snapshot = now
        asyncio.create_task(save_snapshot(p))


async def monitor_loop():
    async with httpx.AsyncClient(auth=AUTH) as client:
        while True:
            current = list(paths.values())
            if current:
                results = await asyncio.gather(*(check_path(client, p) for p in current), return_exceptions=True)
                for p, res in zip(current, results):
                    if isinstance(res, Exception):
                        logger.error("health check of %s failed: %s", p.name, res)
            await asyncio.sleep(MONITOR_LOOP_INTERVAL_SEC)


async def save_snapshot(p: RelayPath):
    """Optional debug still per relay path (needs ffmpeg in the image)."""
    try:
        os.makedirs(CAMERA_SNAPSHOT_DIR, exist_ok=True)
        out = os.path.join(CAMERA_SNAPSHOT_DIR, f"{p.name}.jpg")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", p.address, "-frames:v", "1", "-q:v", "3", out + ".tmp.jpg",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=CAMERA_SNAPSHOT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            return
        if proc.returncode == 0:
            os.replace(out + ".tmp.jpg", out)
    except Exception as e:  # noqa: BLE001
        logger.debug("snapshot of %s failed: %s", p.name, e)


# ======================================================================
async def main():
    global r
    logger.info("camera_stream starting | relay=%s | path mode=%s | modules=%s (+ discovered) | stream base=%s",
                MTX_API_BASE, RELAY_PATH_MODE, CAMERA_MODULES or "-", STREAM_BASE_URL or "-")
    r = redis.from_url(REDIS_URL, decode_responses=True)
    while True:
        try:
            await r.ping()
            break
        except Exception as e:  # noqa: BLE001
            logger.warning("Redis not reachable (%s) — retrying in 2s", e)
            await asyncio.sleep(2)
    async with httpx.AsyncClient(auth=AUTH, limits=httpx.Limits(max_connections=20)) as client:
        await full_sync(client)
        logger.info("serving %d camera(s) on %d relay path(s)", len(cameras), len(paths))
        await asyncio.gather(config_listener(client), periodic_resync(client), mediamtx_poller(), monitor_loop())


if __name__ == "__main__":
    asyncio.run(main())
