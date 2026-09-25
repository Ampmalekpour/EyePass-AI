"""
redis_io.py (control hub)
--------------------------------------------------------------------
Everything the hub does against Redis, for ONE module:

  * inbound: XREADGROUP over `{m}:internal:hub:events` and
    `{m}:internal:hub:results` (consumer group, XACK after the effects
    of an entry are applied -> at-least-once, survives a hub crash)
  * outbound to the backend: RPUSH onto the module's existing results
    list (`face:ai:results` / `plate:vehicle:results`), same call the
    detectors used to make
  * outbound to detectors: LPUSH onto `{m}:internal:hub:ctl:{engine}`
    (the engine BRPOPs it -> FIFO), with a TTL so a dead engine's list
    evaporates
  * per-track checkpoints: one JSON string per live track, TTL'd
  * leader lease: SET NX PX + compare-and-renew (Lua), so a second hub
    container is a hot standby, never a split brain
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis

import config
import protocol as P

# Where each module's backend already reads its results from. Override
# with HUB_RESULTS_KEY_<MODULE> if a deployment renamed it.
DEFAULT_RESULTS_KEYS = {"face": "face:ai:results"}

_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
else
  return 0
end
"""


def redis_url_from_env() -> str:
    url = os.environ.get("REDIS_URL")
    if url:
        return url
    host = os.environ.get("REDIS_HOST", "redis")
    port = os.environ.get("REDIS_PORT", "6379")
    db = os.environ.get("REDIS_DB", "0")
    password = os.environ.get("REDIS_PASSWORD", "")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, (bytes, bytearray)):
            return None  # never ship raw bytes to the backend
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def dumps(obj) -> str:
    return json.dumps(obj, cls=_JsonEncoder, ensure_ascii=False)


class HubRedis:
    def __init__(self, module: str, client: Optional["redis.Redis"] = None, url: Optional[str] = None):
        self.module = module
        self.k = P.keys(module)
        self.r = client or redis.Redis.from_url(url or redis_url_from_env(), decode_responses=True)
        self.results_key = os.environ.get(f"HUB_RESULTS_KEY_{module.upper()}",
                                          DEFAULT_RESULTS_KEYS.get(module, f"{module}:ai:results"))
        self.lease_token = f"{os.environ.get('HOSTNAME', 'hub')}:{uuid.uuid4().hex[:8]}"
        self._renew = self.r.register_script(_RENEW_LUA)

    # ---- liveness -------------------------------------------------------
    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def heartbeat(self, info: Dict[str, Any]):
        self.r.set(self.k["heartbeat"], dumps({"ts": time.time(), **info}), ex=30)

    # ---- leader lease -------------------------------------------------------
    def acquire_lease(self) -> bool:
        return bool(self.r.set(self.k["leader"], self.lease_token, nx=True,
                               px=int(config.LEASE_TTL_SEC * 1000)))

    def renew_lease(self) -> bool:
        return bool(self._renew(keys=[self.k["leader"]],
                                args=[self.lease_token, int(config.LEASE_TTL_SEC * 1000)]))

    def release_lease(self):
        try:
            if self.r.get(self.k["leader"]) == self.lease_token:
                self.r.delete(self.k["leader"])
        except Exception:
            pass

    # ---- inbound streams ------------------------------------------------------
    def ensure_groups(self):
        for stream in (self.k["events"], self.k["results"]):
            try:
                # "$": a brand-new group starts at the tail, so a first-ever
                # hub start never replays ancient tracks from a stream that
                # detectors filled before the hub existed. Once the group
                # exists it persists — restarts resume exactly where they
                # stopped.
                self.r.xgroup_create(stream, config.CONSUMER_GROUP, id="$", mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    def read(self, pending: bool, block_ms: int, count: int) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Returns [(stream, entry_id, kind, data)], events stream first."""
        start = "0" if pending else ">"
        resp = self.r.xreadgroup(
            config.CONSUMER_GROUP, config.CONSUMER_NAME,
            {self.k["events"]: start, self.k["results"]: start},
            count=count, block=None if pending else block_ms,
        ) or []
        by_stream: Dict[str, List] = {s: entries for s, entries in resp}
        out = []
        for stream in (self.k["events"], self.k["results"]):
            for entry_id, fields in by_stream.get(stream, []) or []:
                if not fields:  # already-deleted entry still in the PEL
                    out.append((stream, entry_id, None, None))
                    continue
                kind = fields.get("kind")
                try:
                    data = json.loads(fields.get("data") or "{}")
                except Exception:
                    data = None
                out.append((stream, entry_id, kind, data))
        return out

    def ack(self, acks: Dict[str, List[str]]):
        pipe = self.r.pipeline()
        for stream, ids in acks.items():
            if ids:
                pipe.xack(stream, config.CONSUMER_GROUP, *ids)
        pipe.execute()

    # ---- outbound ----------------------------------------------------------------
    def publish(self, payloads: Iterable[Dict[str, Any]]):
        pipe = self.r.pipeline()
        n = 0
        for p in payloads:
            pipe.rpush(self.results_key, dumps(p))
            n += 1
        if n:
            pipe.execute()

    def send_ctl(self, messages: Iterable[Tuple[Any, Dict[str, Any]]]):
        pipe = self.r.pipeline()
        keys = set()
        for engine_id, msg in messages:
            key = f"{self.k['ctl_prefix']}{engine_id}"
            pipe.lpush(key, dumps(msg))
            keys.add(key)
        for key in keys:
            pipe.expire(key, config.CTL_TTL_SEC)
        if keys:
            pipe.execute()

    # ---- checkpoints ----------------------------------------------------------------
    def save_states(self, states: Iterable[Dict[str, Any]]):
        pipe = self.r.pipeline()
        n = 0
        for st in states:
            pipe.set(f"{self.k['track_prefix']}{st['uid']}", dumps(st), ex=config.STATE_TTL_SEC)
            n += 1
        if n:
            pipe.execute()

    def delete_states(self, uids: Iterable[str]):
        keys = [f"{self.k['track_prefix']}{u}" for u in uids]
        if keys:
            self.r.delete(*keys)

    def load_states(self) -> List[Dict[str, Any]]:
        out = []
        keys = list(self.r.scan_iter(match=f"{self.k['track_prefix']}*", count=500))
        for i in range(0, len(keys), 500):
            for raw in self.r.mget(keys[i:i + 500]):
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except Exception:
                    continue
        return out

    # ---- introspection ----------------------------------------------------------------
    def lag(self) -> Dict[str, Any]:
        out = {}
        for name in ("events", "results"):
            try:
                groups = self.r.xinfo_groups(self.k[name])
                g = next((g for g in groups if g.get("name") == config.CONSUMER_GROUP), None)
                out[name] = {"pending": g.get("pending") if g else None,
                             "lag": g.get("lag") if g else None,
                             "length": self.r.xlen(self.k[name])}
            except Exception:
                out[name] = None
        return out
