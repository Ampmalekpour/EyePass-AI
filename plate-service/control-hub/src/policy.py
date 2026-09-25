"""
policy.py (plate control hub)
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
# PLATE
# ====================================================================
class PlatePolicy(BasePolicy):
    module = "plate"
    trigger_events = ("cross_line", "stop_roi")
    final_event = "leave_scene"
    trigger_flags = {"cross_line": "cross_line", "stop_roi": "stop_roi"}

    def summarize(self, raw, now):
        text = str(raw.get("plate_text") or "")
        valid = bool(raw.get("is_valid")) and text not in ("", "0")
        return {
            "task_id": raw.get("task_id"),
            "stage": raw.get("stage") or raw.get("trigger_type"),
            "ts": now,
            "status": raw.get("status", "ok"),
            "valid": valid,
            "key": text if valid else "0",
            "confidence": float(raw.get("confidence") or 0.0),
            "raw": raw,
        }

    def resolve(self, st):
        results = st.get("results") or []
        winner, n = vote(results)
        if winner:
            chosen = winner["best"]
            same = [r for r in results if r.get("key") == winner["key"]]
        else:
            # no valid read: surface the most confident attempt (text,
            # images, why it failed) instead of an empty record
            attempts = [r for r in results if r["raw"].get("status") in ("ok", "invalid")]
            chosen = max(attempts, key=lambda r: r["confidence"], default=results[-1] if results else None)
            same = [chosen] if chosen else []
        raw = chosen["raw"] if chosen else {}
        payload = raw.get("payload") or {}

        def _img(field):
            # the chosen result's own image; if its MinIO upload failed,
            # the newest one from a result with the same answer
            return payload.get(field) or next(
                ((r["raw"].get("payload") or {}).get(field) for r in reversed(same)
                 if (r["raw"].get("payload") or {}).get(field)), None)

        return {
            "_winner": winner,
            "plate_text": winner["key"] if winner else "0",
            "confidence": float(winner["max"]) if winner else 0.0,
            "is_valid": bool(winner),
            # best raw OCR text even when nothing validated (was invisible
            # before — the record looked empty although OCR had read something)
            "raw_text": raw.get("plate_text") or None,
            "raw_confidence": float(raw.get("confidence") or 0.0) if raw else 0.0,
            "voted_class": raw.get("voted_class"),
            "plate_type": payload.get("plate_type"),
            "stage": chosen.get("stage") if chosen else None,
            "plate_image": _img("plate_image"),
            "frame_image": _img("frame_image"),
            "votes": winner["count"] if winner else 0,
            "candidates": n,
            "n_results": len(results),
            "description": raw.get("description"),
        }

    def answer_key(self, decision):
        return (decision["plate_text"], decision["is_valid"])

    def display(self, st, decision):
        rows = []
        for stage, raw in self._latest_by_stage(st).items():
            conf = float(raw.get("confidence") or 0.0)
            rows.append([stage, f"{str(raw.get('plate_text'))[:12]:<12} {conf:.2f} "
                                f"{'OK' if raw.get('is_valid') else 'INVALID'}"])
        return {"label": decision["plate_text"] if decision["is_valid"] else "no valid plate",
                "confidence": round(decision["confidence"], 3), "votes": decision["votes"],
                "rows": rows}

    @staticmethod
    def _latest_by_stage(st) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in st.get("results") or []:
            out[r.get("stage") or "unknown"] = r["raw"]
        return out

    def _common(self, st, update_type, is_final, resolution, now, event_detail=None):
        decision = self.resolve(st)
        by_stage = self._latest_by_stage(st)
        events_detail = self._events_detail(st)
        public_decision = {k: v for k, v in decision.items() if not k.startswith("_")}
        meta = {
            "track_uid": st["uid"],
            "seen_frames": st.get("seen_frames", 0),
            "duration": st.get("duration_frames", 0),
            "first_seen_ts": st.get("started_ts"),
            "resolution": resolution,
        }
        if event_detail is not None:
            meta["event"] = event_detail
        return {
            "process_id": st.get("engine_id"),
            "stream_idx": st.get("camera_id"),
            "camera_id": st.get("camera_id"),
            "video_source": st.get("video_source"),
            "track_id": st.get("track_id"),
            "track_uid": st["uid"],
            "update_type": update_type,
            "is_final": is_final,
            "timestamp": now,
            "meta": meta,
            # previous shape: {event_name: iso timestamp}
            "events": {name: ev["at"] for name, ev in events_detail.items()},
            "events_detail": events_detail,
            # previous shape: {stage: full OCR result dict}
            "ocr_results": by_stage,
            "track_paths": {
                stage: {"plate_path": (raw.get("payload") or {}).get("plate_image"),
                        "frame_path": (raw.get("payload") or {}).get("frame_image")}
                for stage, raw in by_stage.items()
            },
            # NEW — the hub's single answer for this track
            "resolved": public_decision,
        }

    def event_payload(self, st, event, resolution, now):
        ev = (st.get("events") or {}).get(event) or {}
        detail = dict(ev.get("detail") or {})
        detail["ts"] = ev.get("ts")
        return self._common(st, event, False, resolution, now, event_detail=detail)

    def final_payload(self, st, now):
        end = st.get("end") or {}
        return self._common(st, self.final_event, True, "track_ended", now,
                            event_detail={"reason": end.get("reason"), "ts": st.get("ended_ts")})


POLICIES = {"plate": PlatePolicy}


def make_policy(cfg: ModuleConfig) -> BasePolicy:
    try:
        return POLICIES[cfg.module](cfg)
    except KeyError:
        raise ValueError(f"no policy for module {cfg.module!r} (known: {sorted(POLICIES)})")
