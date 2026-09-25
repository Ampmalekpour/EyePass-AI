"""
policy.py (face control hub)
--------------------------------------------------------------------
Module-specific knowledge the generic hub core (core.py) delegates to:

  * vocabulary  — which event names are spatial triggers, what the
                  final event is called, in the module's OWN spelling
                  (face: line_cross / stopped_roi / finalize;
                  plate: cross_line / stop_roi / leave_scene), so the
                  backend keeps receiving exactly the names it already
                  consumes.
  * resolve()   — turn every result a track has received into ONE
                  decision (who / which plate, how sure, which images).
  * satisfied() — is that decision good enough to stop asking?
  * payloads    — the exact JSON the backend reads from
                  face:ai:results / plate:vehicle:results. Every key
                  the previous pipeline published is still there;
                  new keys are additive.

Resolution rule, both modules: a confidence-weighted vote over VALID
results only. Each candidate (personnel id / plate text) scores the
sum of its confidences; the highest sum wins, its best single result
supplies the images. This replaces two different, inconsistent rules:
face kept a running MAX over every result (valid or not), plate kept
per-stage results and never picked a winner, leaving that to Django.

Everything here is pure: no Redis, no clocks except the `now`
argument, so tests/test_policy.py can pin every case.
--------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import ModuleConfig


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(float(ts)).isoformat()


def vote(results: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    """Confidence-weighted vote over valid result summaries.

    Returns (winner, n_candidates). winner = {key, sum, count, max,
    best (the summary with the highest confidence for that key)}.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if not r.get("valid"):
            continue
        key = r.get("key")
        if key in (None, "", "0", "Unknown"):
            continue
        g = groups.setdefault(key, {"key": key, "sum": 0.0, "count": 0, "max": -1.0, "best": None})
        c = float(r.get("confidence") or 0.0)
        g["sum"] += c
        g["count"] += 1
        if c > g["max"]:
            g["max"], g["best"] = c, r
    if not groups:
        return None, 0
    winner = max(groups.values(), key=lambda g: (g["sum"], g["max"]))
    return winner, len(groups)


class BasePolicy:
    module = "base"
    trigger_events: Tuple[str, ...] = ()
    final_event = "finalize"
    periodic_stage = "periodic"
    # triggers dict key (from track_started) that enables each event
    trigger_flags: Dict[str, str] = {}
    leave_scene_flag = "leave_scene"
    periodic_flag = "periodic"

    def __init__(self, cfg: ModuleConfig):
        self.cfg = cfg

    # ---- results --------------------------------------------------------
    def summarize(self, raw: Dict[str, Any], now: float) -> Dict[str, Any]:
        raise NotImplementedError

    def resolve(self, st: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def satisfied(self, st: Dict[str, Any], decision: Dict[str, Any]) -> bool:
        w = decision.get("_winner")
        if not w:
            return False
        if w["max"] >= self.cfg.satisfied_conf:
            return True
        if self.cfg.consensus_min > 0 and w["count"] >= self.cfg.consensus_min \
                and decision.get("candidates", 0) == 1:
            return True
        return False

    def display(self, st: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def answer_key(self, decision: Dict[str, Any]) -> Tuple:
        """What counts as 'the answer changed' for a late result."""
        raise NotImplementedError

    # ---- payloads -------------------------------------------------------
    def event_payload(self, st, event: str, resolution: str, now: float) -> Dict[str, Any]:
        raise NotImplementedError

    def final_payload(self, st, now: float) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- helpers --------------------------------------------------------
    def trigger_enabled(self, st: Dict[str, Any], event: str) -> bool:
        flags = st.get("triggers") or {}
        if not flags:
            # track_started not seen (orphan) — trust the detector, which
            # only emits triggers for cameras that have them enabled.
            return True
        flag = self.trigger_flags.get(event, event)
        return bool(flags.get(flag, False))

    def periodic_enabled(self, st: Dict[str, Any]) -> bool:
        return bool((st.get("triggers") or {}).get(self.periodic_flag, False))

    def leave_scene_enabled(self, st: Dict[str, Any]) -> bool:
        return bool((st.get("triggers") or {}).get(self.leave_scene_flag, False))

    @staticmethod
    def _events_detail(st: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: {"ts": ev.get("ts"), "at": _iso(ev.get("ts")), "detail": ev.get("detail") or {},
                   "published_at": _iso(ev.get("published_ts")), "resolution": ev.get("resolution")}
            for name, ev in (st.get("events") or {}).items()
        }


# ====================================================================
# FACE
# ====================================================================
class FacePolicy(BasePolicy):
    module = "face"
    trigger_events = ("line_cross", "stopped_roi")
    final_event = "finalize"
    trigger_flags = {"line_cross": "line_cross", "stopped_roi": "stopped_roi"}

    def summarize(self, raw, now):
        pid = str(raw.get("personnelid") or "0")
        valid = bool(raw.get("is_valid")) and pid not in ("0", "Unknown", "")
        return {
            "task_id": raw.get("task_id"),
            "stage": raw.get("stage"),
            "ts": now,
            "status": raw.get("status", "ok"),
            "valid": valid,
            "key": pid if valid else "0",
            "confidence": float(raw.get("confidence", raw.get("detection_score")) or 0.0),
            "raw": raw,
        }

    def _spoof(self, st) -> bool:
        return self.cfg.liveness_policy == "reject" and \
            (st.get("liveness") or {}).get("liveness") == "fake"

    def resolve(self, st):
        results = st.get("results") or []
        winner, n = vote(results)
        spoof = self._spoof(st)
        latest = results[-1] if results else None

        if winner and not spoof:
            best = winner["best"]
            raw = best["raw"]
            pid, conf = winner["key"], float(winner["max"])
            # images: the winner's own; if its upload failed, the newest
            # ones that belong to the same person — never an empty field
            # while another result for the same answer has one.
            face_image = raw.get("face_image") or next(
                (r["raw"].get("face_image") for r in reversed(results)
                 if r.get("key") == pid and r["raw"].get("face_image")), None)
            camera_image = raw.get("camera_image") or next(
                (r["raw"].get("camera_image") for r in reversed(results)
                 if r.get("key") == pid and r["raw"].get("camera_image")), None)
        else:
            raw = latest["raw"] if latest else {}
            pid, conf = "0", 0.0
            face_image = next((r["raw"].get("face_image") for r in reversed(results)
                               if r["raw"].get("face_image")), None)
            camera_image = next((r["raw"].get("camera_image") for r in reversed(results)
                                 if r["raw"].get("camera_image")), None)

        return {
            "_winner": None if spoof else winner,
            "personnelid": pid,
            "confidence": conf,
            "first_name": raw.get("first_name", "Unknown") if pid != "0" else "Unknown",
            "last_name": raw.get("last_name", "Unknown") if pid != "0" else "Unknown",
            "national_code": raw.get("national_code", "0") if pid != "0" else "0",
            "department": raw.get("department", "0") if pid != "0" else "0",
            "face_image": face_image,
            "camera_image": camera_image,
            "votes": winner["count"] if winner else 0,
            "candidates": n,
            "spoof_rejected": spoof,
            "suppressed_identity": winner["key"] if (winner and spoof) else None,
            "chosen_raw": raw or None,
        }

    def satisfied(self, st, decision):
        if decision.get("spoof_rejected"):
            return True  # nothing more to learn from a spoof
        return super().satisfied(st, decision)

    def answer_key(self, decision):
        return (decision["personnelid"], bool(decision.get("spoof_rejected")))

    def display(self, st, decision):
        pid = decision["personnelid"]
        label = f"{pid} {decision['first_name']} {decision['last_name']}" if pid != "0" else "unknown"
        if decision.get("spoof_rejected"):
            label = "SPOOF"
        return {"label": label, "confidence": round(decision["confidence"], 3),
                "votes": decision["votes"], "n_results": len(st.get("results") or [])}

    def _history(self, st):
        hist: Dict[str, Any] = {}
        for r in st.get("results") or []:
            raw = r["raw"]
            snap = {
                "personnel_id": raw.get("personnelid", "0"),
                "first_name": raw.get("first_name"), "last_name": raw.get("last_name"),
                "confidence": r["confidence"], "valid": r["valid"],
                "face_image_url": raw.get("face_image"), "camera_image_url": raw.get("camera_image"),
                "timestamp": r["ts"], "task_id": r["task_id"],
            }
            stage = r.get("stage") or "unknown"
            if stage == "periodic":
                hist.setdefault("periodic", []).append(snap)
            else:
                hist[stage] = snap
        return hist

    def _meta(self, st, decision, resolution, event_detail=None):
        lv = st.get("liveness") or {}
        meta = {
            "track_uid": st["uid"],
            "identified_as": decision["personnelid"],
            "confidence": decision["confidence"],
            "first_name": decision["first_name"],
            "last_name": decision["last_name"],
            "national_code": decision["national_code"],
            "department": decision["department"],
            "last_saved_face_image": decision["face_image"],
            "last_saved_camera_image": decision["camera_image"],
            "recognition_history": self._history(st),
            "votes": decision["votes"],
            "conflicting_identities": decision["candidates"] > 1,
            "spoof_rejected": decision["spoof_rejected"],
            "seen_frames": st.get("seen_frames", 0),
            "duration": st.get("duration_frames", 0),
            "first_seen_ts": st.get("started_ts"),
            "events": self._events_detail(st),
            "resolution": resolution,
        }
        for k in ("liveness", "liveness_score", "liveness_reason", "liveness_evals"):
            meta[k] = lv.get(k)
        if event_detail is not None:
            meta["event"] = event_detail
        return meta

    def _payload(self, st, event, is_final, resolution, now, event_detail=None):
        decision = self.resolve(st)
        chosen = decision.get("chosen_raw") or None
        return {
            "camera_id": st.get("camera_id"),
            "track_id": st.get("track_id"),
            "track_uid": st["uid"],
            "event_type": event,
            "timestamp": now,
            "is_final": is_final,
            "meta": self._meta(st, decision, resolution, event_detail),
            # the worker-shaped record (personnelid, first_name, face_image,
            # camera_image, detection_score, date, time, ...) for the
            # chosen result, identity withheld when it was rejected
            "result": self._result_record(chosen, decision) if chosen else None,
        }

    @staticmethod
    def _result_record(raw, decision):
        rec = {k: v for k, v in raw.items() if k not in ("uid",)}
        rec["personnelid"] = decision["personnelid"]
        rec["first_name"] = decision["first_name"]
        rec["last_name"] = decision["last_name"]
        rec["national_code"] = decision["national_code"]
        rec["department"] = decision["department"]
        rec["detection_score"] = decision["confidence"]
        rec["face_image"] = decision["face_image"]
        rec["camera_image"] = decision["camera_image"]
        return rec

    def event_payload(self, st, event, resolution, now):
        ev = (st.get("events") or {}).get(event) or {}
        detail = dict(ev.get("detail") or {})
        detail["ts"] = ev.get("ts")
        return self._payload(st, event, False, resolution, now, event_detail=detail)

    def final_payload(self, st, now):
        end = st.get("end") or {}
        detail = {"reason": end.get("reason"), "ts": st.get("ended_ts")}
        return self._payload(st, self.final_event, True, "track_ended", now, event_detail=detail)


POLICIES = {"face": FacePolicy}


def make_policy(cfg: ModuleConfig) -> BasePolicy:
    try:
        return POLICIES[cfg.module](cfg)
    except KeyError:
        raise ValueError(f"no policy for module {cfg.module!r} (known: {sorted(POLICIES)})")
