"""
perf_stats.py
--------------------------------------------------------------------
Real-time performance accounting for one engine, with the same
columns as the multi-stream benchmark (tools/bench_multistream.py), so
the live service and the benchmark can be compared directly.

Per camera (⏱️ [PERF], every config.PERF_LOG_INTERVAL_SEC):
    fps in        frames the camera delivered (RTSP reader)
    fps proc      frames that went through the model
    missed        frames replaced by a newer one before the engine got to
                  them (the reader keeps only the latest frame) = dropped
    coasted       frames skipped on purpose (DETECT_EVERY_N_FRAMES > 1)
    pre/inf/post  ms per frame (Ultralytics r.speed, own timers for onnx;
                  for a GPU batch the batch time is split over its frames)
    track         tracker + triggers + crops, ms per frame
    latency       frame arrival -> result handled, ms (avg / p95)

Per engine (📊 [STATS], every config.STATS_EVERY_N_BATCHES loops):
    cameras per batch, batch wall time, loop time, totals and miss %.
--------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np


def _avg(v: List[float]) -> float:
    return float(np.mean(v)) if v else 0.0


def _p95(v: List[float]) -> float:
    return float(np.percentile(v, 95)) if v else 0.0


class _CamWin:
    __slots__ = ("captured", "processed", "missed", "coasted", "pre", "inf", "post", "track",
                 "lat", "dets", "t0")

    def __init__(self):
        self.captured = self.processed = self.missed = self.coasted = self.dets = 0
        self.pre: List[float] = []
        self.inf: List[float] = []
        self.post: List[float] = []
        self.track: List[float] = []
        self.lat: List[float] = []
        self.t0 = time.time()


class EnginePerf:
    def __init__(self, engine_id: int, label: str, cam_interval_sec: float, engine_every_n: int):
        self.engine_id = engine_id
        self.label = label
        self.cam_interval = float(cam_interval_sec)
        self.engine_every_n = max(1, int(engine_every_n))
        self.cams: Dict[str, _CamWin] = {}
        self.totals: Dict[str, Dict[str, int]] = {}
        self._last_cam_flush = time.time()
        self._new_engine_window()

    # ---------------- recording ----------------
    def _cam(self, cid: str) -> _CamWin:
        w = self.cams.get(cid)
        if w is None:
            w = self.cams[cid] = _CamWin()
            self.totals.setdefault(cid, {"processed": 0, "missed": 0, "captured": 0})
        return w

    def frame_arrived(self, cid: str, new_frames: int, missed: int):
        w = self._cam(cid)
        w.captured += new_frames
        w.missed += missed
        t = self.totals[cid]
        t["captured"] += new_frames
        t["missed"] += missed

    def frame_coasted(self, cid: str):
        self._cam(cid).coasted += 1

    def frame_done(self, cid: str, timing: Dict[str, float], track_ms: float, latency_ms: float, n_dets: int):
        w = self._cam(cid)
        w.processed += 1
        w.pre.append(timing.get("preprocess", 0.0))
        w.inf.append(timing.get("inference", 0.0))
        w.post.append(timing.get("postprocess", 0.0))
        w.track.append(track_ms)
        if latency_ms is not None:
            w.lat.append(latency_ms)
        w.dets += n_dets
        self.totals[cid]["processed"] += 1

    def forget(self, cid: str):
        self.cams.pop(cid, None)
        self.totals.pop(cid, None)

    def _new_engine_window(self):
        self.ew = {"batches": 0, "frames": 0, "batch_ms": [], "loop_ms": [], "infer_frame_ms": [],
                   "missed": 0, "captured": 0, "dets": 0, "tracks": 0, "t0": time.time()}

    def batch_done(self, n_frames: int, batch_ms: float, loop_ms: float, infer_frame_ms: List[float],
                   missed: int, captured: int, dets: int, tracks: int):
        e = self.ew
        e["batches"] += 1
        e["frames"] += n_frames
        e["batch_ms"].append(batch_ms)
        e["loop_ms"].append(loop_ms)
        e["infer_frame_ms"].extend(infer_frame_ms)
        e["missed"] += missed
        e["captured"] += captured
        e["dets"] += dets
        e["tracks"] += tracks

    # ---------------- reporting ----------------
    def maybe_log_cameras(self, logger, n_cameras: int):
        now = time.time()
        if now - self._last_cam_flush < self.cam_interval:
            return
        self._last_cam_flush = now
        for cid, w in sorted(self.cams.items()):
            el = max(now - w.t0, 1e-6)
            seen = w.processed + w.missed + w.coasted
            miss_pct = 100.0 * w.missed / seen if seen else 0.0
            t = self.totals.get(cid, {})
            tot_seen = t.get("processed", 0) + t.get("missed", 0)
            logger.info(
                f"⏱️ [PERF] engine={self.engine_id} {self.label} camera={cid} | "
                f"fps in={w.captured / el:.1f} proc={w.processed / el:.1f} | "
                f"missed={w.missed} ({miss_pct:.1f}%) coasted={w.coasted} | "
                f"pre={_avg(w.pre):.1f} infer={_avg(w.inf):.1f} (p95 {_p95(w.inf):.1f}) "
                f"post={_avg(w.post):.1f} track={_avg(w.track):.1f} ms/frame | "
                f"latency={_avg(w.lat):.0f} (p95 {_p95(w.lat):.0f}) ms | dets/frame="
                f"{w.dets / max(w.processed, 1):.2f} | lifetime missed="
                f"{100.0 * t.get('missed', 0) / tot_seen if tot_seen else 0.0:.1f}%"
            )
            self.cams[cid] = _CamWin()

    def maybe_log_engine(self, logger, n_cameras: int):
        e = self.ew
        if e["batches"] < self.engine_every_n:
            return
        el = max(time.time() - e["t0"], 1e-6)
        seen = e["frames"] + e["missed"]
        logger.info(
            f"📊 [STATS] engine={self.engine_id} {self.label} cameras={n_cameras} | "
            f"last {e['batches']} loops in {el:.1f}s | frames/loop={e['frames'] / e['batches']:.2f} | "
            f"processed={e['frames'] / el:.1f}/s | missed={e['missed']} "
            f"({100.0 * e['missed'] / seen if seen else 0.0:.1f}%) | infer/frame avg="
            f"{_avg(e['infer_frame_ms']):.1f} p95={_p95(e['infer_frame_ms']):.1f} ms | batch avg="
            f"{_avg(e['batch_ms']):.1f} p95={_p95(e['batch_ms']):.1f} ms | loop avg="
            f"{_avg(e['loop_ms']):.1f} p95={_p95(e['loop_ms']):.1f} ms | detections={e['dets']} "
            f"tracks={e['tracks']}"
        )
        self._new_engine_window()
