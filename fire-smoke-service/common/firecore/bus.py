"""
bus.py
--------------------------------------------------------------------
Thin Redis wrapper for the fire/smoke detector service, ported from
the plate module's platecore.bus (same shape, same self-healing
pattern — see the plate/face modules for the two-service version of
this split). This module is single-service (detection only, no
second-stage worker), so the OCR/recognizer-style internal task/result
queues were dropped; everything here is the JSON backend contract
(cameras:*, cmd:ai:*, detections:results) that the backend and
camera_stream (not our code) depend on staying human-readable.

Nothing here blocks process startup on Redis being reachable — every
method that talks to Redis is expected to be called from a thread/loop
that already retries.
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import redis

from .keys import RedisKeys
from .logging_setup import setup_logger

logger = setup_logger("firecore.bus")


def _redis_url_from_env() -> str:
    """REDIS_URL wins if set (matches the reference alpr_api.py's own
    convention); otherwise assemble from host/port/db/password."""
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
        self.keys = RedisKeys(module=module or os.environ.get("REDIS_MODULE", "fire"))
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
    # Backend contract — cmd request/response
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
        calling handler(parsed_json) for every message. camera_stream
        restarting, or a network blip, must not kill this thread."""

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
        """Used only by test harnesses standing in for camera_stream."""
        self.rt.publish(self.keys.cameras_events, json.dumps(payload))

    # ================================================================
    # Backend contract — ai_status (per-camera reported status)
    # ================================================================
    def write_ai_status(self, camera_id: str, current: str, error: Optional[str] = None):
        payload = {
            "target": "on",
            "current": current,
            "error": error,
            "updated_at": time.time(),
        }
        try:
            self.rt.hset(self.keys.ai_status(camera_id), str(camera_id),
                         json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning("failed to write ai_status for camera %s: %s", camera_id, e)

    # ================================================================
    # Backend contract — detections:results (THREAT / RESOLUTION events)
    # ================================================================
    def push_result(self, payload: dict, json_encoder=None):
        try:
            self.rt.rpush(self.keys.detections_results,
                           json.dumps(payload, cls=json_encoder, ensure_ascii=False, default=str))
        except Exception:
            logger.exception("failed to push detections:results payload")

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

