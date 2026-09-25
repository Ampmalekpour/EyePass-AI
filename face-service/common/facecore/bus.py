"""
bus.py
--------------------------------------------------------------------
Thin Redis wrapper shared by the detector and the recognizer. It knows
about the two payload shapes the module uses:

  * JSON  — everything that crosses the backend contract
            (cameras:*, cmd:ai:*, ai:results) must stay human-readable
            because the backend and camera_service are not our code.
  * bytes — the internal detector<->recognizer task/result queues,
            where a "message" is a pickled dict that may contain raw
            JPEG bytes for a face crop. See codec.py.

  ADD-FACE: `cmd:enroll:*` is a THIRD, backend-facing pair that still
  uses the pickled/bytes envelope, not JSON — see keys.py's docstring
  for why (it carries image bytes on `verify_pose`). Its request/
  response helpers below deliberately mirror pop_request/send_response
  in shape (poll a list, reply on a per-request TTL'd list) so callers
  already familiar with the camera-command contract recognize the
  pattern immediately, even though the wire format differs.

Nothing here blocks process startup on Redis being reachable — every
method that talks to Redis is expected to be called from a thread/loop
that already retries, the same discipline the heatmap module uses.
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import redis

from .keys import RedisKeys
from .logging_setup import setup_logger

_HUB_STREAM_MAXLEN = int(os.environ.get("HUB_STREAM_MAXLEN", "200000"))

logger = setup_logger("facecore.bus")


def _redis_url_from_env() -> str:
    """REDIS_URL wins if set (matches camera_service's convention);
    otherwise assemble from host/port/db/password."""
    url = os.environ.get("REDIS_URL")
    if url:
        return url
    host = os.environ.get("REDIS_HOST", "redis")
    port = os.environ.get("REDIS_PORT", "6379")
    db = os.environ.get("REDIS_DB", "0")
    password = os.environ.get("REDIS_PASSWORD", "")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


class RedisBus:
    """One instance per process. Safe to share across threads (redis-py
    connection pools are thread-safe); NOT safe to share across a
    process fork — each multiprocessing.Process must build its own."""

    def __init__(self, module: Optional[str] = None, url: Optional[str] = None,
                 client: Optional["redis.Redis"] = None):
        self.keys = RedisKeys(module=module or os.environ.get("REDIS_MODULE", "face"))
        self.url = url or _redis_url_from_env()
        self.r = client or redis.Redis.from_url(self.url, decode_responses=False)
        # A second, decode_responses=True client is convenient for the
        # JSON-only backend-contract calls so we are not sprinkling
        # .decode() everywhere in that code path.
        self.rt = redis.Redis.from_url(self.url, decode_responses=True)

    # ================================================================
    # Liveness
    # ================================================================
    def ping(self) -> bool:
        try:
            return bool(self.rt.ping())
        except Exception:
            return False

    def wait_until_available(self, interval: float = 2.0):
        while not self.ping():
            logger.warning("Redis not reachable yet at %s — retrying in %.0fs", self.url, interval)
            time.sleep(interval)
        logger.info("Redis reachable at %s", self.url)

    def heartbeat(self, key: str, ttl_seconds: int = 30):
        try:
            self.rt.set(key, str(time.time()), ex=ttl_seconds)
        except Exception as e:
            logger.debug("heartbeat write failed for %s: %s", key, e)

    # ================================================================
    # Backend contract — cmd request/response (camera activate/deactivate)
    # ================================================================
    def pop_request(self, timeout: int = 2) -> Optional[Dict[str, Any]]:
        try:
            item = self.rt.brpop(self.keys.cmd_request, timeout=timeout)
        except Exception as e:
            logger.error("cmd request BRPOP failed: %s", e)
            time.sleep(1)
            return None
        if not item:
            return None
        _, raw = item
        try:
            return json.loads(raw)
        except Exception:
            logger.error("bad json on %s: %r", self.keys.cmd_request, raw)
            return None

    def send_response(self, request_id: str, status: str, ttl_seconds: int = 60):
        if not request_id:
            return
        key = self.keys.cmd_response(request_id)
        try:
            self.rt.rpush(key, json.dumps({"status": status}))
            self.rt.expire(key, ttl_seconds)
        except Exception as e:
            logger.error("failed to push response for %s: %s", request_id, e)

    # ================================================================
    # Backend contract — camera config/details/events
    # ================================================================
    def get_camera_config(self, camera_id: str) -> Optional[dict]:
        raw = self.rt.hget(self.keys.cameras_config, str(camera_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            logger.error("bad json in cameras:config[%s]", camera_id)
            return None

    def get_camera_details(self, camera_id: str) -> Optional[dict]:
        raw = self.rt.hget(self.keys.cameras_details, str(camera_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            logger.error("bad json in cameras:details[%s]", camera_id)
            return None

    def all_camera_configs(self) -> Dict[str, dict]:
        out = {}
        for cid in self.rt.hkeys(self.keys.cameras_config):
            cfg = self.get_camera_config(cid)
            if cfg:
                out[cid] = cfg
        return out

    def subscribe(self, channel: str, handler: Callable[[dict], None], thread_name: str):
        """Fire-and-forget: a daemon thread that (re)subscribes forever,
        calling handler(parsed_json) for every message. Mirrors the
        camera_events_listener reconnect loop from the reference code —
        camera_stream restarting, or a network blip, must not kill this
        thread."""

        def _run():
            while True:
                try:
                    pubsub = self.rt.pubsub()
                    pubsub.subscribe(channel)
                    logger.info("Subscribed to %s", channel)
                    for message in pubsub.listen():
                        if message.get("type") != "message":
                            continue
                        raw = message.get("data")
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            logger.warning("bad json on %s: %r", channel, raw)
                            continue
                        try:
                            handler(payload)
                        except Exception:
                            logger.exception("handler failed for message on %s", channel)
                except Exception as e:
                    logger.error("subscription to %s dropped: %s — retrying in 2s", channel, e)
                    time.sleep(2)

        t = threading.Thread(target=_run, daemon=True, name=thread_name)
        t.start()
        return t

    def publish_camera_event(self, payload: dict):
        """Used only by test harnesses standing in for camera_service."""
        self.rt.publish(self.keys.cameras_events, json.dumps(payload))

    # ================================================================
    # Backend contract — ai:results
    # ================================================================
    def push_result(self, payload: dict, json_encoder=None):
        try:
            self.rt.rpush(self.keys.ai_results, json.dumps(payload, cls=json_encoder, ensure_ascii=False))
        except Exception:
            logger.exception("failed to push ai:results payload")

    # ================================================================
    # Backend contract — enrollment (add-face). Bytes, not JSON — see
    # module docstring. Mirrors pop_request/send_response in shape.
    # ================================================================
    def pop_enroll_request(self, timeout: int = 2) -> Optional[bytes]:
        """BRPOP one pickled enroll command. Returns the raw bytes —
        callers decode with facecore.codec.decode_task (same envelope
        as the internal task queues; see codec.py)."""
        try:
            item = self.r.brpop(self.keys.enroll_request, timeout=timeout)
        except Exception as e:
            logger.error("enroll request BRPOP failed: %s", e)
            time.sleep(1)
            return None
        if not item:
            return None
        _, raw = item
        return raw

    def push_enroll_request(self, data: bytes):
        """publisher-side helper — used by the backend (or a test
        harness standing in for it), not by the recognizer itself."""
        self.r.lpush(self.keys.enroll_request, data)

    def send_enroll_response(self, request_id: str, payload: dict, ttl_seconds: int = 60):
        """Pushes a pickled reply (may include image bytes, e.g. the
        pose-approved crop) onto the per-request response list and
        sets its TTL, exactly like send_response()'s JSON counterpart."""
        if not request_id:
            return
        from .codec import encode_task  # local import: keeps bus.py decoupled from codec's pickle choice
        key = self.keys.enroll_response(request_id)
        try:
            self.r.rpush(key, encode_task(payload))
            self.r.expire(key, ttl_seconds)
        except Exception:
            logger.exception("failed to push enroll response for %s", request_id)

    def read_enroll_response(self, request_id: str, timeout: int = 30) -> Optional[dict]:
        """Blocking read of the reply — used by callers (backend or
        test harness) waiting on a `verify_pose`/`commit` result."""
        from .codec import decode_task
        key = self.keys.enroll_response(request_id)
        try:
            item = self.r.brpop(key, timeout=timeout)
        except Exception as e:
            logger.error("enroll response BRPOP failed for %s: %s", request_id, e)
            return None
        if not item:
            return None
        _, raw = item
        try:
            return decode_task(raw)
        except Exception:
            logger.error("failed to decode enroll response for %s", request_id)
            return None

    # ================================================================
    # INTERNAL — durable hash state (active cameras, phase/count)
    # ================================================================
    def hset_json(self, key: str, field: str, value: dict):
        self.rt.hset(key, field, json.dumps(value, ensure_ascii=False))

    def hget_json(self, key: str, field: str) -> Optional[dict]:
        raw = self.rt.hget(key, field)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def hgetall_json(self, key: str) -> Dict[str, dict]:
        out = {}
        for field, raw in self.rt.hgetall(key).items():
            try:
                out[field] = json.loads(raw)
            except Exception:
                continue
        return out

    def hdel(self, key: str, field: str):
        self.rt.hdel(key, field)

    # ================================================================
    # INTERNAL — rec task/result queues (binary payloads, see codec.py)
    #
    # ADD-FACE reuses push_task/pop_task as-is for enroll_pose_check /
    # enroll_commit tasks (task_type distinguishes them — see
    # worker.py), and reuses push_result_bytes/pop_result_bytes as-is
    # with a synthetic engine_id of "enroll:{request_id}" instead of a
    # real detector engine id. No changes needed in this section.
    # ================================================================
    def push_task(self, data: bytes):
        self.r.lpush(self.keys.rec_tasks, data)
        try:
            self.rt.incr(self.keys.rec_tasks_pending_gauge)
        except Exception:
            pass

    def pop_task(self, timeout: int = 1) -> Optional[bytes]:
        item = self.r.brpop(self.keys.rec_tasks, timeout=timeout)
        if not item:
            return None
        try:
            self.rt.decr(self.keys.rec_tasks_pending_gauge)
        except Exception:
            pass
        _, raw = item
        return raw

    def push_result_bytes(self, engine_id, data: bytes):
        self.r.lpush(self.keys.rec_results(engine_id), data)

    def pop_result_bytes(self, engine_id, timeout: int = 1) -> Optional[bytes]:
        item = self.r.brpop(self.keys.rec_results(engine_id), timeout=timeout)
        if not item:
            return None
        _, raw = item
        return raw

    # ================================================================
    # INTERNAL — add-face gallery coordination
    # ================================================================
    def gallery_lock(self, timeout: float = 30.0, blocking_timeout: float = 15.0):
        """Returns a redis-py Lock (context-manager) guarding a gallery
        mutation. `timeout` is how long the lock is held before it
        auto-expires (protects against a dead holder wedging every
        future enrollment forever); `blocking_timeout` is how long a
        second caller waits to acquire it before giving up. Usage:

            with bus.gallery_lock():
                ... download db, allocate range, insert, embed, upload ...
        """
        return self.r.lock(self.keys.gallery_lock, timeout=timeout, blocking_timeout=blocking_timeout)

    def publish_gallery_updated(self, payload: dict):
        try:
            self.rt.publish(self.keys.gallery_updated, json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.exception("failed to publish gallery:updated")

    # ================================================================
    # INTERNAL — control hub (streams in, per-engine ctl list out).
    # JSON, not pickle: the hub is a separate service and these entries
    # are meant to be readable in RedisInsight while debugging.
    # ================================================================
    def hub_emit(self, kind: str, data: dict):
        """detector engine -> hub:events"""
        self.rt.xadd(self.keys.hub_events,
                     {"kind": kind, "data": json.dumps(data, cls=_HubEncoder, ensure_ascii=False)},
                     maxlen=_HUB_STREAM_MAXLEN, approximate=True)

    def hub_push_result(self, data: dict):
        """recognizer / OCR worker -> hub:results"""
        self.rt.xadd(self.keys.hub_results,
                     {"kind": "result", "data": json.dumps(data, cls=_HubEncoder, ensure_ascii=False)},
                     maxlen=_HUB_STREAM_MAXLEN, approximate=True)

    def pop_hub_ctl(self, engine_id, timeout: int = 1) -> Optional[dict]:
        """hub -> this engine. BRPOP (the hub LPUSHes, so this is FIFO)."""
        item = self.rt.brpop(self.keys.hub_ctl(engine_id), timeout=timeout)
        if not item:
            return None
        _, raw = item
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("bad json on %s: %r", self.keys.hub_ctl(engine_id), raw)
            return None


class _HubEncoder(json.JSONEncoder):
    """JSON for hub stream entries: datetimes as ISO strings, numpy
    scalars/arrays as plain numbers/lists (a numpy int in a liveness
    field must never crash the detector loop), raw bytes dropped."""

    def default(self, obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, (bytes, bytearray)):
            return None
        if hasattr(obj, "tolist"):      # numpy array / scalar
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)
