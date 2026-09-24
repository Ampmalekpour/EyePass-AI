"""
bus.py
--------------------------------------------------------------------
Thin Redis wrapper for the heatmap ai_service, ported from the
fire/smoke module's firecore.bus (same shape, same self-healing
pattern — see the plate/face/fire modules for the same split). Single
service (detection + accumulation, no second-stage worker), so there
is only the JSON backend contract here (cameras:*, cmd:ai:*,
ai:results, ai:active) — the pieces the backend and camera_stream
(not our code) depend on staying exactly this shape.

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

logger = setup_logger("heatmapcore.bus")


def _redis_url_from_env() -> str:
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
        self.keys = RedisKeys(module=module or os.environ.get("REDIS_MODULE", "heatmap"))
        self.url = url or _redis_url_from_env()
        self.rt = client or redis.Redis.from_url(
            self.url,
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
        )

    # ================================================================
    # Liveness
    # ================================================================
    def ping(self) -> bool:
        try:
            return bool(self.rt.ping())
        except Exception:
            return False

    def wait_until_available(self, interval: float = 3.0):
        attempt = 0
        while not self.ping():
            attempt += 1
            if attempt == 1 or attempt % 10 == 0:
                logger.warning("Redis not reachable yet at %s (attempt %d) — retrying every %.0fs",
                                self.url, attempt, interval)
            time.sleep(interval)
        logger.info("Redis reachable at %s", self.url)

    def heartbeat(self, key: str, ttl_seconds: int = 30):
        try:
            self.rt.set(key, str(time.time()), ex=ttl_seconds)
        except Exception as e:
            logger.debug("heartbeat write failed for %s: %s", key, e)

    # ================================================================
    # Backend contract — cmd request/response
    #
    # Aligned to the platform-wide LIST+BLPOP convention (LPUSH by the
    # backend, BRPOP by us, for FIFO order; response is RPUSH+EXPIRE,
    # BRPOP by the backend) — see keys.py's docstring for what this
    # replaces.
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
            logger.warning("send_response called without a request_id; skipping")
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

    def subscribe_camera_events(self, on_event: Callable[[dict], None]) -> threading.Thread:
        """Daemon thread calling on_event(payload) for every online/
        offline update camera_stream publishes. Survives Redis restarts
        by tearing the subscription down and rebuilding it."""

        def _run():
            while True:
                pubsub = None
                try:
                    pubsub = self.rt.pubsub()
                    pubsub.subscribe(self.keys.cameras_events)
                    logger.info("Subscribed to %s", self.keys.cameras_events)
                    for message in pubsub.listen():
                        if message.get("type") != "message":
                            continue
                        raw = message.get("data")
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw)
                        except Exception:
                            logger.warning("bad json on %s: %r", self.keys.cameras_events, raw)
                            continue
                        try:
                            on_event(payload)
                        except Exception:
                            logger.exception("camera_events handler raised")
                except Exception as e:
                    logger.warning("camera_events listener lost connection (%s), resubscribing in 2s", e)
                    time.sleep(2)
                finally:
                    if pubsub is not None:
                        try:
                            pubsub.close()
                        except Exception:
                            pass

        t = threading.Thread(target=_run, daemon=True, name="camera-events-listener")
        t.start()
        return t

    def publish_camera_event(self, payload: dict):
        """Used only by test harnesses standing in for camera_stream."""
        self.rt.publish(self.keys.cameras_events, json.dumps(payload))

    # ================================================================
    # Backend contract — ai_status (per-camera reported status; new)
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
    # Backend contract — ai:results (where a flushed cube landed)
    # ================================================================
    def push_result(self, payload: dict) -> None:
        try:
            self.rt.lpush(self.keys.ai_results, json.dumps(payload, ensure_ascii=False, default=str))
            logger.info("Published result to %s: %s", self.keys.ai_results, payload)
        except Exception:
            logger.exception("failed to push ai:results payload")

    # ================================================================
    # INTERNAL — durable hash state (active cameras, phase/count)
    # ================================================================
    def hset_json(self, key: str, field: str, value: dict):
        self.rt.hset(key, field, json.dumps(value, ensure_ascii=False, default=str))

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
