"""
core.py (control hub)
--------------------------------------------------------------------
The per-track recognition state machine — the part that used to be
scattered through face Engine._drain_recognition_outputs /
_finalize_or_drop_track and plate Engine._handle_ocr_result /
_should_submit_ocr_for_stage / the finalize-wait loop, now in ONE
place, shared by both modules.

Pure by design: no Redis, no threads, no wall clock. Every entry point
takes `now` and returns an `Effects` object describing what to do —
which payloads to publish to the backend, which ctl messages to send
back to which detector engine, which track checkpoints to save or
delete. service.py applies those effects; tests/ drive this class
directly with a fake clock.

Lifecycle of one track (uid):

    track_started ──► LIVE ──track_ended──► ENDED ──(results or timeout)──► CLOSED ──grace──► deleted
                       │                      │                              (final published or dropped)
                       │ trigger ─► deferred ─┤ published when: satisfied │ its task's result │
                       │                      │   trigger_max_wait_sec    │ track end
                       │ result ─► vote ─► satisfied? ─► ctl "state" to the detector
                       │ tick ─► periodic request (camera flag) / stale (no events for track_stale_sec)

Rules, identical for face and plate (the module differences live in
policy.py):

  * A trigger is published ONCE per event name per track.
    satisfied  -> immediately ("immediate")
    otherwise  -> held until the result of the task the detector
                  submitted for it ("after_recognition"), or until the
                  identity becomes satisfied, or trigger_max_wait_sec
                  ("timeout"), or the track ends ("track_ended").
                  The event's own detail (line direction, stop
                  duration, ...) always travels with it.
  * satisfied = the resolved decision is good enough to stop spending
    recognizer/OCR time on this track. Sent to the detector as soon as
    it changes; the detector then stops submitting and skips its
    finalize pass.
  * Final: once the track ended and every in-flight task answered (or
    finalize_timeout_sec passed), publish the final payload — gated by
    final_min_seen_frames / final_min_crops, UNLESS something was
    already published for the track (the backend always gets closure
    for a track it has heard about).
  * Results that arrive after the final are logged and dropped (kept
    for late_result_grace_sec so duplicates are recognized).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import protocol as P
from config import ModuleConfig
from policy import BasePolicy


@dataclass
class Effects:
    publish: List[Dict[str, Any]] = field(default_factory=list)
    ctl: List[Tuple[Any, Dict[str, Any]]] = field(default_factory=list)
    save: Set[str] = field(default_factory=set)
    delete: Set[str] = field(default_factory=set)

    def merge(self, other: "Effects") -> "Effects":
        self.publish.extend(other.publish)
        self.ctl.extend(other.ctl)
        self.save |= other.save
        self.delete |= other.delete
        self.save -= self.delete
        return self

    def empty(self) -> bool:
        return not (self.publish or self.ctl or self.save or self.delete)


def new_state(uid: str, now: float) -> Dict[str, Any]:
    return {
        "uid": uid,
        "camera_id": None, "track_id": None, "engine_id": None, "boot_id": None,
        "video_source": None, "triggers": {},
        "created_ts": now, "started_ts": None, "last_event_ts": now,
        "seen_frames": 0, "n_crops": 0, "duration_frames": 0, "liveness": {},
        "results": [], "completed_tasks": [], "in_flight": {},
        "events": {},
        "satisfied": False, "published_any": False,
        "next_periodic_ts": None,
        "ended": False, "ended_ts": None, "end": {},
        "closed": False, "closed_ts": None, "final_published": False,
    }


class HubCore:
    def __init__(self, cfg: ModuleConfig, policy: BasePolicy, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.policy = policy
        self.log = logger or logging.getLogger(f"hub.{cfg.module}")
        self.tracks: Dict[str, Dict[str, Any]] = {}
        self.engine_boots: Dict[str, str] = {}
        self.stats = {"published_events": 0, "published_finals": 0, "dropped_tracks": 0,
                      "late_results": 0, "results": 0, "tracks_started": 0}

    # ================================================================
    # Restore
    # ================================================================
    def load(self, states: Iterable[Dict[str, Any]]):
        for st in states:
            if isinstance(st, dict) and st.get("uid"):
                base = new_state(st["uid"], st.get("created_ts") or 0.0)
                base.update(st)
                self.tracks[st["uid"]] = base

    # ================================================================
    # Inbound
    # ================================================================
    def handle(self, kind: str, data: Dict[str, Any], now: float) -> Effects:
        fx = Effects()
        if kind == P.K_ENGINE_STARTED:
            return self._on_engine_started(data, now)

        uid = data.get("uid")
        if not uid:
            self.log.warning("dropping %s without uid: %s", kind, data)
            return fx

        st = self.tracks.get(uid)
        if st is None:
            if kind == P.K_RESULT:
                # result for a track this hub has no record of: its
                # track_started was lost/trimmed, or it closed more than
                # late_result_grace_sec ago. Kept as an orphan so it is
                # still published (the stale timer closes it).
                self.log.warning("result %s for unknown uid=%s — keeping as orphan track",
                                 data.get("task_id"), uid)
            st = self.tracks[uid] = new_state(uid, now)
        st["last_event_ts"] = now
        self._absorb_identity(st, data)

        handler = {
            P.K_TRACK_STARTED: self._on_started,
            P.K_TRACK_UPDATE: self._on_update,
            P.K_SUBMITTED: self._on_submitted,
            P.K_TRIGGER: self._on_trigger,
            P.K_TRACK_ENDED: self._on_ended,
            P.K_RESULT: self._on_result,
        }.get(kind)
        if handler is None:
            self.log.warning("unknown event kind %r for uid=%s", kind, uid)
            return fx
        fx.merge(handler(st, data, now))
        if uid in self.tracks:
            fx.save.add(uid)
        return fx

    def _absorb_identity(self, st, data):
        for k in ("camera_id", "track_id", "engine_id", "boot_id", "video_source"):
            if data.get(k) is not None and st.get(k) is None:
                st[k] = data[k]
        if data.get("engine_id") is not None:
            st["engine_id"] = data["engine_id"]

    def _absorb_stats(self, st, data):
        for k in ("seen_frames", "n_crops", "duration_frames"):
            if data.get(k) is not None:
                st[k] = max(int(st.get(k) or 0), int(data[k]))
        if isinstance(data.get("liveness"), dict) and data["liveness"]:
            st["liveness"] = data["liveness"]

    # ---- engine_started ---------------------------------------------------
    def _on_engine_started(self, data, now) -> Effects:
        fx = Effects()
        eid, boot = str(data.get("engine_id")), data.get("boot_id")
        self.engine_boots[eid] = boot
        for st in list(self.tracks.values()):
            if st["ended"] or str(st.get("engine_id")) != eid or st.get("boot_id") in (None, boot):
                continue
            self.log.info("uid=%s: engine %s restarted — ending track", st["uid"], eid)
            fx.merge(self._end(st, {"reason": "engine_restarted"}, now))
            fx.save.add(st["uid"])
        return fx

    # ---- track_started ------------------------------------------------------
    def _on_started(self, st, data, now) -> Effects:
        fx = Effects()
        if st.get("started_ts") is None:
            st["started_ts"] = data.get("ts") or now
            self.stats["tracks_started"] += 1
        if isinstance(data.get("triggers"), dict):
            st["triggers"] = data["triggers"]
        if self.policy.periodic_enabled(st) and st.get("next_periodic_ts") is None:
            st["next_periodic_ts"] = now + self.cfg.periodic_first_delay_sec
        self._absorb_stats(st, data)
        # A result may have beaten track_started here (different streams);
        # tell the detector what we already know.
        if st["results"]:
            fx.merge(self._send_state(st))
        return fx

    def _on_update(self, st, data, now) -> Effects:
        self._absorb_stats(st, data)
        fx = Effects()
        # liveness may have just turned `fake` under the reject policy
        fx.merge(self._refresh_satisfied(st, now))
        return fx

    # ---- submitted ------------------------------------------------------------
    def _on_submitted(self, st, data, now) -> Effects:
        tid = data.get("task_id")
        if tid and tid not in st["completed_tasks"]:
            st["in_flight"][tid] = {"stage": data.get("stage"), "ts": now}
            # a trigger that fired before any crop could be sent now
            # waits for THIS task (the detector sends it as soon as a
            # usable crop exists) instead of the first unrelated answer
            for ev in st["events"].values():
                if ev.get("status") == "deferred" and not ev.get("awaiting"):
                    ev["awaiting"] = tid
                    ev["awaiting_done"] = False
        self._absorb_stats(st, data)
        return Effects()

    # ---- trigger ----------------------------------------------------------------
    def _on_trigger(self, st, data, now) -> Effects:
        fx = Effects()
        name = data.get("event")
        if name not in self.policy.trigger_events:
            self.log.warning("uid=%s: ignoring unknown trigger %r", st["uid"], name)
            return fx
        if not self.policy.trigger_enabled(st, name):
            return fx
        if name in st["events"]:
            return fx  # once per event name per track
        tid = data.get("task_id")
        if tid and tid not in st["completed_tasks"]:
            st["in_flight"].setdefault(tid, {"stage": name, "ts": now})
        st["events"][name] = {
            "ts": data.get("ts") or now, "received_ts": now, "detail": data.get("detail") or {},
            "status": "deferred", "awaiting": tid,
            # a task that already answered cannot "arrive" later: treat
            # the trigger as waiting for any next result instead
            "awaiting_done": bool(tid and tid in st["completed_tasks"]),
        }
        self._absorb_stats(st, data)
        if st["closed"]:
            # track already finalized (late trigger, e.g. replayed stream) — record only
            st["events"][name]["status"] = "late"
            return fx
        if st["satisfied"]:
            fx.merge(self._publish_event(st, name, "immediate", now))
        elif tid and tid in st["completed_tasks"]:
            # its result overtook the trigger event (different streams) —
            # the answer is already in, publish now rather than time out
            fx.merge(self._publish_event(st, name, "after_recognition", now))
        return fx

    # ---- track_ended ---------------------------------------------------------------
    def _on_ended(self, st, data, now) -> Effects:
        self._absorb_stats(st, data)
        if st["ended"]:
            return Effects()
        end = {"reason": data.get("reason", "absent")}
        ftid = data.get("finalize_task_id")
        if ftid:
            end["finalize_task_id"] = ftid
            if ftid not in st["completed_tasks"]:
                st["in_flight"].setdefault(ftid, {"stage": self.policy.final_event, "ts": now})
        return self._end(st, end, now)

    def _end(self, st, end, now) -> Effects:
        st["ended"] = True
        st["ended_ts"] = now
        st["end"] = end
        if self.policy.leave_scene_enabled(st) and self.policy.final_event not in st["events"]:
            st["events"][self.policy.final_event] = {
                "ts": now, "received_ts": now, "detail": {"reason": end.get("reason")},
                "status": "final", "awaiting": None,
            }
        return self._maybe_close(st, now)

    # ---- result -----------------------------------------------------------------------
    def _on_result(self, st, data, now) -> Effects:
        fx = Effects()
        tid = data.get("task_id")
        if tid and tid in st["completed_tasks"]:
            return fx  # duplicate delivery (stream replay)
        self.stats["results"] += 1
        if tid:
            st["completed_tasks"].append(tid)
            st["completed_tasks"] = st["completed_tasks"][-200:]
            st["in_flight"].pop(tid, None)

        if st["closed"]:
            return self._on_late_result(st, data, now)

        st["results"].append(self.policy.summarize(data, now))
        if len(st["results"]) > self.cfg.max_results_per_track:
            st["results"] = st["results"][-self.cfg.max_results_per_track:]

        fx.merge(self._refresh_satisfied(st, now, send=False))
        # ack (carries the new satisfied flag too) so the detector clears its in-flight lock right away
        if st.get("engine_id") is not None:
            decision = self.policy.resolve(st)
            fx.ctl.append((st["engine_id"], {
                "action": P.A_RESULT, "uid": st["uid"], "task_id": tid,
                "satisfied": st["satisfied"], "display": self.policy.display(st, decision),
            }))

        # deferred triggers waiting for this result (or for any result)
        for name, ev in st["events"].items():
            if ev.get("status") != "deferred":
                continue
            awaiting = ev.get("awaiting")
            if awaiting is None or awaiting == tid or ev.get("awaiting_done") \
                    or awaiting not in st["in_flight"]:
                fx.merge(self._publish_event(st, name, "after_recognition", now))

        if st["ended"]:
            fx.merge(self._maybe_close(st, now))
        return fx

    def _on_late_result(self, st, data, now) -> Effects:
        """A result for a track whose final was already sent (its task sat
        in a backlogged queue longer than finalize_timeout_sec).

        republish_if_changed (default): if it changes the answer — e.g.
        the final went out with no plate / unknown face and this is the
        plate / identity — publish the final AGAIN with revision+1, so the
        backend (upserting on track_uid) ends up with the right answer
        instead of an empty record. drop: log only.
        """
        fx = Effects()
        self.stats["late_results"] += 1
        late_by = now - (st.get("closed_ts") or now)
        if self.cfg.late_result_policy != "republish_if_changed" or not st.get("final_published"):
            self.log.warning("uid=%s: result %s (%s) arrived %.1fs after the track was closed — dropped",
                             st["uid"], data.get("task_id"), data.get("stage"), late_by)
            return fx
        before = self.policy.answer_key(self.policy.resolve(st))
        st["results"].append(self.policy.summarize(data, now))
        st["results"] = st["results"][-self.cfg.max_results_per_track:]
        after = self.policy.answer_key(self.policy.resolve(st))
        if after == before:
            self.log.info("uid=%s: late result %s (+%.1fs) does not change the answer %s",
                          st["uid"], data.get("task_id"), late_by, after)
            return fx
        st["revision"] = int(st.get("revision") or 1) + 1
        st["missing_tasks"] = [t for t in st.get("missing_tasks") or [] if t != data.get("task_id")]
        fx.publish.append(self._final(st, now, "late_result"))
        self.log.warning(
            "🧟📤 uid=%s cam=%s track=%s LATE RESULT after track closed (%.1fs ago, end_reason=%s) "
            "— re-dispatching to plate:vehicle:results | task=%s answer %s -> %s | revision now %d",
            st["uid"], st.get("camera_id"), st.get("track_id"), late_by, (st.get("end") or {}).get("reason"),
            data.get("task_id"), before, after, st["revision"],
        )
        return fx

    def _final(self, st, now, resolution: str) -> Dict[str, Any]:
        payload = self.policy.final_payload(st, now)
        payload["revision"] = int(st.get("revision") or 1)
        payload["complete"] = not st.get("missing_tasks")
        payload["missing_tasks"] = list(st.get("missing_tasks") or [])
        if resolution != "track_ended":
            payload.setdefault("meta", {})["resolution"] = resolution
        return payload

    # ================================================================
    # Timers
    # ================================================================
    def tick(self, now: float) -> Effects:
        fx = Effects()
        for uid, st in list(self.tracks.items()):
            if st["closed"]:
                if now - (st.get("closed_ts") or now) >= self.cfg.late_result_grace_sec:
                    del self.tracks[uid]
                    fx.delete.add(uid)
                continue
            if st["ended"]:
                sub = self._maybe_close(st, now)
                if not sub.empty():
                    fx.merge(sub)
                    fx.save.add(uid)
                continue
            if now - st["last_event_ts"] >= self.cfg.track_stale_sec:
                self.log.warning(
                    "⏱️💀 uid=%s cam=%s track=%s STALE — no events for %.0fs (limit=%.0fs), "
                    "forcing track end | seen=%s crops=%s in_flight=%s",
                    uid, st.get("camera_id"), st.get("track_id"),
                    now - st["last_event_ts"], self.cfg.track_stale_sec,
                    st.get("seen_frames"), st.get("n_crops"), list(st.get("in_flight") or {}),
                )
                fx.merge(self._end(st, {"reason": "stale"}, now))
                fx.save.add(uid)
                continue
            # deferred triggers past their deadline
            # (its own task is queued -> wait longer, so a backlogged
            # recognizer/OCR delays the event instead of emptying it)
            for name, ev in st["events"].items():
                if ev.get("status") != "deferred":
                    continue
                queued = bool(ev.get("awaiting")) and ev["awaiting"] in st["in_flight"]
                limit = self.cfg.trigger_task_wait_sec if queued else self.cfg.trigger_max_wait_sec
                if now - ev.get("received_ts", now) >= limit:
                    fx.merge(self._publish_event(st, name, "timeout", now))
                    fx.save.add(uid)
            # periodic re-query
            npt = st.get("next_periodic_ts")
            if npt is not None and now >= npt:
                st["next_periodic_ts"] = now + self.cfg.periodic_interval_sec
                fx.save.add(uid)
                if not st["satisfied"] and not st["in_flight"] and st.get("engine_id") is not None \
                        and self.policy.periodic_enabled(st):
                    fx.ctl.append((st["engine_id"], {"action": P.A_REQUEST, "uid": uid,
                                                     "stage": self.policy.periodic_stage}))
            # in-flight tasks that will never answer (lost task, worker
            # crash) must not block periodic forever
            for tid, info in list(st["in_flight"].items()):
                if now - info.get("ts", now) >= self.cfg.finalize_timeout_sec * 3:
                    st["in_flight"].pop(tid, None)
                    fx.save.add(uid)
        return fx

    # ================================================================
    # Internals
    # ================================================================
    def _refresh_satisfied(self, st, now, send: bool = True) -> Effects:
        fx = Effects()
        decision = self.policy.resolve(st)
        sat = bool(self.policy.satisfied(st, decision))
        changed = sat != st["satisfied"]
        st["satisfied"] = sat
        if changed and send and st.get("engine_id") is not None:
            fx.ctl.append((st["engine_id"], {"action": P.A_STATE, "uid": st["uid"],
                                             "satisfied": sat,
                                             "display": self.policy.display(st, decision)}))
        if sat and not st["closed"]:
            for name, ev in st["events"].items():
                if ev.get("status") == "deferred":
                    fx.merge(self._publish_event(st, name, "after_recognition", now))
        return fx

    def _send_state(self, st) -> Effects:
        fx = Effects()
        if st.get("engine_id") is None:
            return fx
        decision = self.policy.resolve(st)
        fx.ctl.append((st["engine_id"], {"action": P.A_STATE, "uid": st["uid"],
                                         "satisfied": st["satisfied"],
                                         "display": self.policy.display(st, decision)}))
        return fx

    def _publish_event(self, st, name, resolution, now) -> Effects:
        fx = Effects()
        ev = st["events"][name]
        if ev.get("status") == "published":
            return fx
        ev["status"] = "published"
        ev["published_ts"] = now
        ev["resolution"] = resolution
        fx.publish.append(self.policy.event_payload(st, name, resolution, now))
        st["published_any"] = True
        self.stats["published_events"] += 1
        self.log.info("uid=%s cam=%s track=%s -> %s published (%s)",
                      st["uid"], st.get("camera_id"), st.get("track_id"), name, resolution)
        return fx

    def _maybe_close(self, st, now) -> Effects:
        fx = Effects()
        if st["closed"]:
            return fx
        waited = now - (st.get("ended_ts") or now)
        if st["in_flight"] and waited < self.cfg.finalize_timeout_sec:
            return fx
        if st["in_flight"]:
            self.log.warning("uid=%s: finalize timeout after %.1fs, still waiting on %s — closing without them "
                             "(a late answer re-publishes the final if it changes it)",
                             st["uid"], waited, list(st["in_flight"]))
            st["missing_tasks"] = list(st["in_flight"])
            st["in_flight"] = {}

        for name, ev in st["events"].items():
            if ev.get("status") == "deferred":
                fx.merge(self._publish_event(st, name, "track_ended", now))

        gate_ok = int(st.get("seen_frames") or 0) >= self.cfg.final_min_seen_frames and \
            int(st.get("n_crops") or 0) >= self.cfg.final_min_crops
        if gate_ok or st["published_any"]:
            fx.publish.append(self._final(st, now, "track_ended"))
            st["final_published"] = True
            self.stats["published_finals"] += 1
            decision = self.policy.resolve(st)
            reason = (st.get("end") or {}).get("reason")
            emoji = "☠️📤" if reason == "stale" else "✅📤"
            self.log.info(
                "%s uid=%s cam=%s track=%s -> FINAL DISPATCHED to plate:vehicle:results | reason=%s "
                "seen=%s crops=%s results=%d display=%s gate_ok=%s published_any=%s",
                emoji, st["uid"], st.get("camera_id"), st.get("track_id"), reason,
                st.get("seen_frames"), st.get("n_crops"), len(st["results"]),
                self.policy.display(st, decision), gate_ok, st.get("published_any"),
            )
        else:
            self.stats["dropped_tracks"] += 1
            self.log.info("uid=%s cam=%s track=%s DROPPED (seen=%s/%s crops=%s/%s, nothing published)",
                          st["uid"], st.get("camera_id"), st.get("track_id"),
                          st.get("seen_frames"), self.cfg.final_min_seen_frames,
                          st.get("n_crops"), self.cfg.final_min_crops)
        st["closed"] = True
        st["closed_ts"] = now
        return fx

    # ================================================================
    # Introspection (health endpoint / hub_tools)
    # ================================================================
    def summary(self) -> Dict[str, Any]:
        live = [s for s in self.tracks.values() if not s["ended"]]
        ending = [s for s in self.tracks.values() if s["ended"] and not s["closed"]]
        return {
            "live_tracks": len(live), "ending_tracks": len(ending),
            "closed_tracks_in_grace": len(self.tracks) - len(live) - len(ending),
            "stats": dict(self.stats),
        }
