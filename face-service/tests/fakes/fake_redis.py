"""
fake_redis.py
--------------------------------------------------------------------
Minimal in-memory stand-in for the `redis` package, covering only the
subset of the redis-py API that facecore/bus.py actually calls
(ping, get/set/incr/decr/expire/exists, hset/hget/hgetall/hdel/hkeys,
rpush/lpush/brpop/lrange/llen, pubsub/publish).

This sandbox has no network access to install the real `redis` or
`fakeredis` packages (see the module README's "Testing" section for
why), so this file exists purely to let facecore's Redis-dependent
modules be imported and exercised in a unit test without a real Redis
server or the redis-py client library installed.

Not a general-purpose Redis emulator — only what this module's own
code paths touch. If a new bus.py method starts using a command not
implemented here, add it here rather than reaching for a shortcut in
the test.
--------------------------------------------------------------------
"""

from __future__ import annotations

import fnmatch
import time
from collections import deque
from typing import Any, Dict, List, Optional


class FakeRedis:
    def __init__(self, decode_responses: bool = False):
        self.decode_responses = decode_responses
        # Shared backing store across every FakeRedis instance created
        # from the same FakeRedisServer, so a decode_responses=True
        # client and a decode_responses=False client (bus.py opens one
        # of each) see the same data — exactly like two real redis-py
        # clients pointed at the same server.
        self._store: Dict[str, Any] = {}
        self._expires: Dict[str, float] = {}

    # ---- construction, mirroring redis.Redis.from_url -----------------
    @classmethod
    def from_url(cls, url: str, decode_responses: bool = False):
        server = _SERVERS.setdefault(url, _Backing())
        inst = cls(decode_responses=decode_responses)
        inst._store = server.store
        inst._expires = server.expires
        inst._pubsub_channels = server.pubsub_channels
        return inst

    # ---- internal helpers ----------------------------------------------
    def _alive(self, key: str) -> bool:
        exp = self._expires.get(key)
        if exp is not None and time.time() > exp:
            self._store.pop(key, None)
            self._expires.pop(key, None)
            return False
        return True

    def _enc(self, v):
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.decode() if self.decode_responses else v
        s = str(v) if not isinstance(v, (str, bytes)) else v
        if self.decode_responses:
            return s if isinstance(s, str) else s.decode()
        return s.encode() if isinstance(s, str) else s

    # ---- liveness -------------------------------------------------------
    def ping(self) -> bool:
        return True

    # ---- string ----------------------------------------------------------
    def set(self, key, value, ex: Optional[int] = None):
        self._store[key] = value if isinstance(value, (bytes, str)) else str(value)
        if ex is not None:
            self._expires[key] = time.time() + ex
        else:
            self._expires.pop(key, None)
        return True

    def get(self, key):
        if key not in self._store or not self._alive(key):
            return None
        return self._enc(self._store[key])

    def exists(self, key) -> int:
        return 1 if (key in self._store and self._alive(key)) else 0

    def expire(self, key, seconds):
        if key in self._store:
            self._expires[key] = time.time() + seconds
        return True

    def incr(self, key):
        cur = int(self._store.get(key, 0) or 0)
        cur += 1
        self._store[key] = str(cur)
        return cur

    def decr(self, key):
        cur = int(self._store.get(key, 0) or 0)
        cur -= 1
        self._store[key] = str(cur)
        return cur

    # ---- hash --------------------------------------------------------------
    def hset(self, key, field=None, value=None, mapping=None):
        h = self._store.setdefault(key, {})
        if mapping:
            h.update({k: v for k, v in mapping.items()})
        if field is not None:
            h[field] = value
        return 1

    def hget(self, key, field):
        h = self._store.get(key, {})
        if field not in h:
            return None
        return self._enc(h[field])

    def hgetall(self, key):
        h = self._store.get(key, {})
        return {self._enc(k): self._enc(v) for k, v in h.items()}

    def hdel(self, key, field):
        h = self._store.get(key, {})
        return 1 if h.pop(field, None) is not None else 0

    def hkeys(self, key):
        h = self._store.get(key, {})
        return [self._enc(k) for k in h.keys()]

    # ---- list --------------------------------------------------------------
    def _list(self, key) -> deque:
        return self._store.setdefault(key, deque())

    def rpush(self, key, value):
        self._list(key).append(value)
        return len(self._store[key])

    def lpush(self, key, value):
        self._list(key).appendleft(value)
        return len(self._store[key])

    def brpop(self, key, timeout: int = 0):
        lst = self._list(key)
        if lst:
            return key, self._enc(lst.pop())
        return None

    def lrange(self, key, start, end):
        lst = list(self._list(key))
        if end == -1:
            end = len(lst) - 1
        return [self._enc(v) for v in lst[start:end + 1]]

    def llen(self, key):
        return len(self._list(key))

    # ---- pubsub (unused by the unit tests below, stubbed for import safety) --
    def publish(self, channel, message):
        return 0

    def pubsub(self):
        raise NotImplementedError("FakeRedis pubsub is not implemented — not exercised by these tests")


class _Backing:
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.expires: Dict[str, float] = {}
        self.pubsub_channels: List[str] = []


_SERVERS: Dict[str, _Backing] = {}


def reset_all():
    """Call between tests that must not see each other's data."""
    _SERVERS.clear()


# Module shape mimics the real `redis` package well enough for
# `import redis; redis.Redis.from_url(...)` call sites to work
# unmodified.
class _RedisModule:
    Redis = FakeRedis


Redis = FakeRedis
