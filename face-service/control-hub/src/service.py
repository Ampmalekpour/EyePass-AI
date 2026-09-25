"""
service.py (control hub)
--------------------------------------------------------------------
`ModuleRunner` — one thread per module (face, plate). It glues the
pure HubCore (core.py) to Redis (redis_io.py):

    standby ──acquire lease──► restore checkpoints ──► drain own pending
        ▲                                                 entries (crash
        │                                                 recovery)
        └──lease lost / Redis error──  loop: XREADGROUP ─► core.handle
                                             tick every TICK_INTERVAL
                                             apply effects:
                                               publish -> backend list
                                               ctl     -> detector engines
                                               save/delete checkpoints
                                             XACK

Effects are applied BEFORE the entry is acknowledged, so a crash can
duplicate a publish but never lose one (at-least-once). Every result
carries task_id, and every event is idempotent in the core (trigger
once per name, ended once, duplicate results ignored), so replaying a
partially-applied batch after a restart is harmless.
--------------------------------------------------------------------
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import config
from config import ModuleConfig
from core import Effects, HubCore
from logging_setup import setup_logger
from policy import make_policy
from redis_io import HubRedis


class ModuleRunner(threading.Thread):
    def __init__(self, cfg: ModuleConfig, io: Optional[HubRedis] = None):
        super().__init__(daemon=True, name=f"hub-{cfg.module}")
        self.cfg = cfg
        self.module = cfg.module
        self.log = setup_logger(f"hub.{cfg.module}")
        self.io = io or HubRedis(cfg.module)
        self.core: Optional[HubCore] = None
        self.is_leader = False
        self.stop_event = threading.Event()
        self.last_loop_ts = 0.0
        self.last_error: Optional[str] = None

    # ---------------------------------------------------------------
    def run(self):
        while not self.stop_event.is_set():
            try:
                self._wait_for_redis()
                self._become_leader()
                if self.stop_event.is_set():
                    break
                self._serve()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self.log.exception("runner error — dropping leadership, retrying in 2s")
                self.is_leader = False
                time.sleep(2)
        self.io.release_lease()

    def stop(self):
        self.stop_event.set()

    # ---------------------------------------------------------------
    def _wait_for_redis(self):
        while not self.stop_event.is_set() and not self.io.ping():
            self.log.warning("Redis not reachable — retrying in 2s")
            time.sleep(2)

    def _become_leader(self):
        announced = False
        while not self.stop_event.is_set():
            if self.io.acquire_lease():
                self.is_leader = True
                self.log.info("acquired leadership for module %s (token %s)", self.module, self.io.lease_token)
                return
            if not announced:
                self.log.info("another hub owns module %s — standing by", self.module)
                announced = True
            self.last_loop_ts = time.time()
            time.sleep(min(config.LEASE_RENEW_SEC, 2.0))

    def _serve(self):
        self.core = HubCore(self.cfg, make_policy(self.cfg), self.log)
        states = self.io.load_states()
        self.core.load(states)
        self.log.info("restored %d track checkpoint(s)", len(states))
        self.io.ensure_groups()

        # 1) whatever this consumer read but never acked before a crash
        while not self.stop_event.is_set():
            batch = self.io.read(pending=True, block_ms=0, count=config.READ_COUNT)
            if not batch:
                break
            self._process(batch)
            if len(batch) < config.READ_COUNT:
                break

        # 2) live
        last_tick = last_renew = last_hb = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_renew >= config.LEASE_RENEW_SEC:
                if not self.io.renew_lease():
                    self.log.error("lost leadership for module %s — back to standby", self.module)
                    self.is_leader = False
                    return
                last_renew = now
            batch = self.io.read(pending=False, block_ms=config.READ_BLOCK_MS, count=config.READ_COUNT)
            if batch:
                self._process(batch)
            now = time.time()
            if now - last_tick >= config.TICK_INTERVAL_SEC:
                self._apply(self.core.tick(now))
                last_tick = now
            if now - last_hb >= 10.0:
                try:
                    self.io.heartbeat({"module": self.module, **self.core.summary()})
                except Exception:
                    pass
                last_hb = now
            self.last_loop_ts = now

    def _process(self, batch):
        fx = Effects()
        acks: Dict[str, List[str]] = {}
        for stream, entry_id, kind, data in batch:
            acks.setdefault(stream, []).append(entry_id)
            if kind is None or not isinstance(data, dict):
                if kind is not None:
                    self.log.warning("undecodable entry %s on %s — acked and dropped", entry_id, stream)
                continue
            try:
                fx.merge(self.core.handle(kind, data, time.time()))
            except Exception:
                self.log.exception("handler failed for %s %s (uid=%s) — acked and dropped",
                                   kind, entry_id, data.get("uid"))
        self._apply(fx)
        self.io.ack(acks)

    def _apply(self, fx: Effects):
        if fx.empty():
            return
        if fx.publish:
            self.io.publish(fx.publish)
        if fx.ctl:
            self.io.send_ctl(fx.ctl)
        if fx.save:
            self.io.save_states(self.core.tracks[u] for u in fx.save if u in self.core.tracks)
        if fx.delete:
            self.io.delete_states(fx.delete)

    # ---------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "module": self.module, "alive": self.is_alive(), "leader": self.is_leader,
            "last_loop_age_sec": round(time.time() - self.last_loop_ts, 1) if self.last_loop_ts else None,
            "last_error": self.last_error,
        }
        if self.core is not None and self.is_leader:
            info.update(self.core.summary())
            try:
                info["streams"] = self.io.lag()
            except Exception:
                pass
        return info

    def tracks(self) -> List[Dict[str, Any]]:
        if self.core is None:
            return []
        out = []
        for st in self.core.tracks.values():
            decision = self.core.policy.resolve(st)
            out.append({
                "uid": st["uid"], "camera_id": st.get("camera_id"), "track_id": st.get("track_id"),
                "engine_id": st.get("engine_id"), "ended": st["ended"], "closed": st["closed"],
                "satisfied": st["satisfied"], "in_flight": list(st["in_flight"]),
                "events": {k: v.get("status") for k, v in st["events"].items()},
                "results": len(st["results"]), "display": self.core.policy.display(st, decision),
                "seen_frames": st.get("seen_frames"), "liveness": (st.get("liveness") or {}).get("liveness"),
            })
        return out
