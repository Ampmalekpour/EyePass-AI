# debug_recorder.py
# --------------------------------------------------------------------
# Per-camera debug video recorder for the ALPR pipeline.
#
# Writes an annotated MP4 per camera onto the bind-mounted debug drive,
# showing everything a debugger's eye needs:
#
#   * raw YOLO detections (pre-tracker)            -> thin grey boxes
#   * confirmed tracks (post-BYTETracker)          -> colored boxes + id
#   * the anchor point actually used for geometry  -> dot + trail
#   * ROI rectangle, trigger line, stop-ROI polygon
#   * per-track stability: age, absent counter, velocity, side-of-line,
#     frames inside ROI, crop quality (sharpness/res/AR/pass-fail),
#     best-crop ladder state
#   * OCR lifecycle: submitted / pending / result + plate + confidence,
#     per trigger stage
#   * every hand-off to the core: redis publish (cross_line, stop_roi,
#     leave_scene final), drop decisions with the reason
#   * a scrolling event log + HUD side panel
#
# Everything is off a single env switch and never raises into the
# pipeline: any failure disables the recorder for that camera only.
# --------------------------------------------------------------------

import os
import json
import time
import math
import datetime
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------
# Config (all env driven so you can tune it without rebuilding)
# --------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


class DebugConfig:
    """Snapshot of the debug-recorder env config (picklable, spawn-safe)."""

    def __init__(self):
        # master switch
        self.enabled = _env_bool("DEBUG_VIDEO_ENABLED", False)

        # where the videos go: this MUST be your bind-mounted host dir
        self.dir = os.environ.get("DEBUG_VIDEO_DIR", "/debug")

        # nominal fps written into the container header. The engine loop is
        # not isochronous, so we also burn the real wall clock on every frame.
        self.fps = _env_float("DEBUG_VIDEO_FPS", 12.0)

        # roll a new file every N seconds (0 = never roll)
        self.segment_seconds = _env_float("DEBUG_VIDEO_SEGMENT_SECONDS", 300.0)

        # keep at most N segments per camera on disk (0 = keep everything)
        self.max_segments = _env_int("DEBUG_VIDEO_MAX_SEGMENTS", 12)

        # write every Nth processed frame (1 = all). Bump to 2/3 on busy rigs.
        self.every_n = max(1, _env_int("DEBUG_VIDEO_EVERY_N", 1))

        # downscale the video part before annotating (1.0 = native)
        self.scale = _env_float("DEBUG_VIDEO_SCALE", 1.0)

        # width of the right-hand telemetry panel in px (0 = no panel)
        self.panel_width = _env_int("DEBUG_VIDEO_PANEL_WIDTH", 430)

        # fourcc; mp4v is the safe default inside the nvcr/pytorch image
        self.codec = os.environ.get("DEBUG_VIDEO_CODEC", "mp4v")
        self.ext = os.environ.get("DEBUG_VIDEO_EXT", ".mp4")

        # also emit a machine-greppable event log next to each segment
        self.jsonl = _env_bool("DEBUG_VIDEO_JSONL", True)

        # keep drawing tracks for N frames after they vanish (ghosts)
        self.ghost_frames = _env_int("DEBUG_VIDEO_GHOST_FRAMES", 45)

        # how many events the side panel shows
        self.event_lines = _env_int("DEBUG_VIDEO_EVENT_LINES", 14)

        # trail length in points
        self.trail_len = _env_int("DEBUG_VIDEO_TRAIL", 30)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------
# Palette (BGR)
# --------------------------------------------------------------------
C_BG = (18, 18, 18)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 150)
C_ROI = (200, 200, 60)
C_LINE = (60, 220, 255)
C_STOPROI = (220, 120, 255)
C_DET = (130, 130, 130)
C_TRACK = (90, 230, 120)
C_TRACK_OCR = (255, 190, 70)
C_TRACK_GHOST = (90, 90, 200)
C_TRACK_STOP = (70, 90, 255)
C_TRAIL = (255, 220, 120)
C_OK = (110, 230, 130)
C_WARN = (70, 190, 255)
C_BAD = (80, 90, 250)

# event kind -> color
EVENT_COLORS = {
    "CAMERA": (200, 200, 200),
    "READ_FAIL": C_BAD,
    "LINE_CROSS": (60, 220, 255),
    "ROI_ENTRY": (140, 255, 140),
    "ROI_EXIT": (170, 170, 170),
    "STOPPED": (70, 90, 255),
    "OCR_SUBMIT": C_TRACK_OCR,
    "OCR_SKIP": C_DIM,
    "OCR_RESULT": (140, 255, 200),
    "OCR_LATE": C_BAD,
    "PUBLISH": (255, 140, 220),
    "FINALIZE": (255, 140, 220),
    "DROP": C_BAD,
    "TIMEOUT": C_BAD,
    "ERROR": C_BAD,
}


def _color_for_track(tid: int) -> Tuple[int, int, int]:
    """Stable, readable per-track hue."""
    h = (int(tid) * 47) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# --------------------------------------------------------------------
# small draw helpers
# --------------------------------------------------------------------
def _text(img, s, org, color=C_TEXT, scale=0.42, thick=1, bg=None):
    if bg is not None:
        (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _alpha_poly(img, pts, color, alpha=0.18):
    if pts is None or len(pts) < 3:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts.astype(np.int32)], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _dashed_rect(img, p1, p2, color, thick=1, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thick)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thick)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thick)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thick)


# --------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------
class DebugRecorder:
    """
    One instance per camera. Lives inside the engine process, is driven
    from the engine loop, and is fully fail-safe: if anything goes wrong
    it disables itself and the pipeline keeps running.
    """

    def __init__(self, camera_id: str, engine_id: int, cfg: DebugConfig, logger=None):
        self.camera_id = str(camera_id)
        self.engine_id = int(engine_id)
        self.cfg = cfg
        self.logger = logger

        self.enabled = bool(cfg.enabled)
        self.broken = False

        self.writer: Optional[cv2.VideoWriter] = None
        self.jsonl_fh = None
        self.segment_path: Optional[str] = None
        self.segment_started = 0.0
        self.segment_index = 0
        self.out_size: Optional[Tuple[int, int]] = None

        self.frames_written = 0
        self.frames_seen = 0

        self.events = deque(maxlen=400)          # (ts, kind, text)
        self.flashes: Dict[int, Tuple[str, int, Tuple[int, int, int]]] = {}
        self._last_wall = time.time()
        self._fps_ema = 0.0
        self._draw_ms_ema = 0.0

        self.cam_dir = os.path.join(self.cfg.dir, f"camera_{self.camera_id}")

        if self.enabled:
            try:
                os.makedirs(self.cam_dir, exist_ok=True)
                # fail fast & loud if the bind mount is read-only
                probe = os.path.join(self.cam_dir, ".write_probe")
                with open(probe, "w") as f:
                    f.write("ok")
                os.remove(probe)
            except Exception as e:
                self._fail(f"debug dir not writable ({self.cam_dir}): {e}")

    # ---------------- lifecycle ----------------
    def _fail(self, msg: str):
        self.broken = True
        self.enabled = False
        if self.logger:
            self.logger.error(f"[DEBUG-REC][{self.camera_id}] disabled: {msg}")
        else:
            print(f"[DEBUG-REC][{self.camera_id}] disabled: {msg}", flush=True)

    def _open_segment(self, w: int, h: int):
        self._close_segment()

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"cam{self.camera_id}_eng{self.engine_id}_{stamp}_{self.segment_index:03d}"
        path = os.path.join(self.cam_dir, base + self.cfg.ext)

        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        writer = cv2.VideoWriter(path, fourcc, float(self.cfg.fps), (w, h))

        if not writer.isOpened():
            # fall back to a codec that exists in every opencv build
            path = os.path.join(self.cam_dir, base + ".avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                     float(self.cfg.fps), (w, h))
            if not writer.isOpened():
                self._fail("cv2.VideoWriter could not be opened (codec/size)")
                return

        self.writer = writer
        self.segment_path = path
        self.segment_started = time.time()
        self.out_size = (w, h)
        self.segment_index += 1

        if self.cfg.jsonl:
            try:
                self.jsonl_fh = open(os.path.splitext(path)[0] + ".events.jsonl",
                                     "a", buffering=1)
            except Exception:
                self.jsonl_fh = None

        self.log("CAMERA", f"segment open {os.path.basename(path)} {w}x{h}@{self.cfg.fps:g}")
        if self.logger:
            self.logger.info(f"[DEBUG-REC][{self.camera_id}] writing {path} ({w}x{h})")

        self._prune()

    def _close_segment(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None
        if self.jsonl_fh is not None:
            try:
                self.jsonl_fh.close()
            except Exception:
                pass
        self.jsonl_fh = None

    def _prune(self):
        if self.cfg.max_segments <= 0:
            return
        try:
            files = [
                os.path.join(self.cam_dir, f)
                for f in os.listdir(self.cam_dir)
                if f.lower().endswith((".mp4", ".avi"))
            ]
            files.sort(key=lambda p: os.path.getmtime(p))
            for old in files[:-self.cfg.max_segments]:
                try:
                    os.remove(old)
                    side = os.path.splitext(old)[0] + ".events.jsonl"
                    if os.path.exists(side):
                        os.remove(side)
                except Exception:
                    pass
        except Exception:
            pass

    def close(self):
        if self.enabled:
            self.log("CAMERA", "recorder closed")
        self._close_segment()

    # ---------------- events ----------------
    def log(self, kind: str, text: str, track_id: Optional[int] = None,
            flash_frames: int = 25, data: Optional[dict] = None):
        """Record a pipeline event: it lands in the panel, on the track box, and in the jsonl."""
        if not self.enabled or self.broken:
            return
        ts = time.time()
        self.events.append((ts, kind, text))

        if track_id is not None and flash_frames > 0:
            self.flashes[int(track_id)] = (
                kind, int(flash_frames), EVENT_COLORS.get(kind, C_TEXT)
            )

        if self.jsonl_fh is not None:
            try:
                rec = {
                    "ts": ts,
                    "iso": datetime.datetime.fromtimestamp(ts).isoformat(),
                    "camera_id": self.camera_id,
                    "engine_id": self.engine_id,
                    "kind": kind,
                    "text": text,
                }
                if track_id is not None:
                    rec["track_id"] = int(track_id)
                if data:
                    rec["data"] = data
                self.jsonl_fh.write(json.dumps(rec, default=str) + "\n")
            except Exception:
                pass

    # ---------------- main entry ----------------
    def write(self, frame_full: np.ndarray, ctx: Dict[str, Any]):
        """
        frame_full : the untouched full camera frame (BGR)
        ctx        : everything the engine knows about this iteration,
                     see Engine._debug_context() for the shape.
        """
        if not self.enabled or self.broken or frame_full is None:
            return

        self.frames_seen += 1
        if (self.frames_seen % self.cfg.every_n) != 0:
            self._decay_flashes()
            return

        t0 = time.time()
        try:
            canvas = self._render(frame_full, ctx)

            h, w = canvas.shape[:2]
            if self.writer is None or self.out_size != (w, h):
                self._open_segment(w, h)
                if self.writer is None:
                    return
            elif self.cfg.segment_seconds > 0 and \
                    (time.time() - self.segment_started) >= self.cfg.segment_seconds:
                self._open_segment(w, h)
                if self.writer is None:
                    return

            self.writer.write(canvas)
            self.frames_written += 1
        except Exception as e:
            self._fail(f"render/write error: {e}")
            return
        finally:
            self._decay_flashes()

        dt = (time.time() - t0) * 1000.0
        self._draw_ms_ema = dt if self._draw_ms_ema == 0 else (0.9 * self._draw_ms_ema + 0.1 * dt)

    def _decay_flashes(self):
        for tid in list(self.flashes.keys()):
            kind, n, col = self.flashes[tid]
            if n <= 1:
                self.flashes.pop(tid, None)
            else:
                self.flashes[tid] = (kind, n - 1, col)

    # ---------------- rendering ----------------
    def _render(self, frame_full: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        now = time.time()
        dt = now - self._last_wall
        self._last_wall = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps_ema = inst if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * inst)

        img = frame_full
        if self.cfg.scale and abs(self.cfg.scale - 1.0) > 1e-3:
            img = cv2.resize(img, None, fx=self.cfg.scale, fy=self.cfg.scale,
                             interpolation=cv2.INTER_AREA)
            s = self.cfg.scale
        else:
            img = img.copy()
            s = 1.0

        self._draw_geometry(img, ctx, s)
        self._draw_detections(img, ctx, s)
        self._draw_tracks(img, ctx, s)
        self._draw_banner(img, ctx)

        if self.cfg.panel_width > 0:
            panel = self._draw_panel(img.shape[0], ctx)
            img = np.hstack([img, panel])

        # encoders want even dimensions
        h, w = img.shape[:2]
        if w % 2 or h % 2:
            img = img[: h - (h % 2), : w - (w % 2)]
        return img

    # ---- static geometry: ROI box, trigger line, stop polygon ----
    def _draw_geometry(self, img, ctx, s):
        roi_px = ctx.get("roi_px")
        if roi_px:
            x1, y1, x2, y2 = [int(v * s) for v in roi_px]
            _dashed_rect(img, (x1, y1), (x2, y2), C_ROI, 1)
            _text(img, "ROI (inference area)", (x1 + 4, max(12, y1 - 6)), C_ROI, 0.42)

        lp = ctx.get("line_px")
        if lp and not (tuple(map(int, lp[0])) == (0, 0) and tuple(map(int, lp[1])) == (0, 0)):
            p1 = (int(lp[0][0] * s), int(lp[0][1] * s))
            p2 = (int(lp[1][0] * s), int(lp[1][1] * s))
            on = ctx.get("trig_cross", False)
            col = C_LINE if on else C_DIM
            cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
            # side labels: which half is "positive"
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            n = math.hypot(dx, dy) or 1.0
            nx, ny = int(-dy / n * 34), int(dx / n * 34)
            _text(img, "neg", (mx + nx - 10, my + ny), col, 0.45, 1, (20, 20, 20))
            _text(img, "pos", (mx - nx - 10, my - ny), col, 0.45, 1, (20, 20, 20))
            _text(img, f"CROSS LINE [{'ON' if on else 'OFF'}]",
                  (p1[0] + 4, p1[1] - 8), col, 0.45, 1, (20, 20, 20))

        sr = ctx.get("stop_roi_px")
        if sr is not None and len(sr) >= 3:
            pts = np.array([(int(p[0] * s), int(p[1] * s)) for p in sr], dtype=np.int32)
            if not np.all(pts == 0):
                on = ctx.get("trig_stop", False)
                col = C_STOPROI if on else C_DIM
                _alpha_poly(img, pts, col, 0.14 if not ctx.get("any_stopped") else 0.32)
                cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
                _text(img, f"STOP ROI [{'ON' if on else 'OFF'}]",
                      (int(pts[:, 0].min()) + 4, int(pts[:, 1].min()) - 8),
                      col, 0.45, 1, (20, 20, 20))

    # ---- raw YOLO output, before the tracker had a say ----
    def _draw_detections(self, img, ctx, s):
        for d in ctx.get("detections", []):
            x1, y1, x2, y2, conf, cls = d
            p1 = (int(x1 * s), int(y1 * s))
            p2 = (int(x2 * s), int(y2 * s))
            cv2.rectangle(img, p1, p2, C_DET, 1)
            _text(img, f"det {conf:.2f} c{int(cls)}", (p1[0], max(10, p1[1] - 3)),
                  C_DET, 0.36)

    # ---- tracks: the meat ----
    def _draw_tracks(self, img, ctx, s):
        for t in ctx.get("tracks", []):
            tid = int(t["track_id"])
            ghost = bool(t.get("ghost"))
            x1, y1, x2, y2 = [int(v * s) for v in t["bbox_full"]]

            base = _color_for_track(tid)
            col = base
            if ghost:
                col = C_TRACK_GHOST
            elif t.get("stopped"):
                col = C_TRACK_STOP
            elif t.get("ocr_pending"):
                col = C_TRACK_OCR

            flash = self.flashes.get(tid)
            thick = 3 if flash else 2
            if flash:
                col = flash[2]

            if ghost:
                _dashed_rect(img, (x1, y1), (x2, y2), col, 2, 6)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)

            # trail of the anchor point the geometry actually uses
            trail = t.get("trail") or []
            if len(trail) > 1:
                pts = np.array([(int(p[0] * s), int(p[1] * s)) for p in trail[-self.cfg.trail_len:]],
                               dtype=np.int32)
                cv2.polylines(img, [pts], False, C_TRAIL, 1, cv2.LINE_AA)
            anchor = t.get("anchor")
            if anchor:
                ax, ay = int(anchor[0] * s), int(anchor[1] * s)
                cv2.circle(img, (ax, ay), 4, C_TRAIL, -1)
                if t.get("inside_roi"):
                    cv2.circle(img, (ax, ay), 9, C_STOPROI, 2)

            # ---- label stack above the box ----
            lines = []
            head = f"#{tid} c{t.get('cls')} s{t.get('score', 0.0):.2f}"
            if ghost:
                head += f"  ABSENT {t.get('absent')}/{t.get('absent_limit')}"
            lines.append((head, col))

            lines.append((
                f"age {t.get('seen_frames')}f  crops {t.get('n_crops')}/{t.get('n_best')}"
                f"  best {t.get('best_score', 0.0):.2f}/{t.get('best_res', 0)}px",
                C_TEXT))

            q = t.get("quality_ok")
            qcol = C_OK if q else C_BAD
            lines.append((
                f"sharp {t.get('sharpness', 0.0):.0f} res {t.get('resolution', 0)} "
                f"ar {t.get('aspect', 0.0):.2f} q:{'PASS' if q else 'FAIL'}",
                qcol))

            side = t.get("side") or "-"
            vel = t.get("velocity")
            stab = f"side {side}  inroi {t.get('inside_frames', 0)}"
            if vel is not None:
                stab += f"  v {vel:.1f}px/s"
            if t.get("stopped"):
                stab += f"  STOPPED {t.get('stop_duration', 0.0):.1f}s"
            lines.append((stab, C_WARN if t.get("stopped") else C_DIM))

            ocr = t.get("ocr_summary")
            if ocr:
                lines.append((ocr, C_TRACK_OCR))
            if t.get("ocr_pending"):
                lines.append((
                    f"OCR PENDING {t.get('ocr_pending')} {t.get('ocr_pending_age', 0.0):.1f}s",
                    C_TRACK_OCR))
            elif t.get("last_ocr_latency_ms") is not None:
                # Resolved round-trip latency (detector-submit -> result
                # arrival) for the most recent OCR round trip on this
                # track — only shown once nothing is pending, so it
                # never fights with the PENDING line above for space.
                lines.append((f"rtt {t.get('last_ocr_latency_ms'):.0f}ms", C_DIM))
            if t.get("finalize_waiting"):
                lines.append((
                    f"FINALIZE HELD for {t.get('finalize_stage')} "
                    f"{t.get('finalize_age', 0.0):.1f}s", C_BAD))
            if flash:
                lines.append((f">>> {flash[0]}", flash[2]))

            y = y1 - 6 - (len(lines) - 1) * 13
            if y < 14:
                y = y2 + 16
            for txt, tcol in lines:
                _text(img, txt, (x1, y), tcol, 0.40, 1, (15, 15, 15))
                y += 13

    def _draw_banner(self, img, ctx):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 22), (12, 12, 12), -1)
        wall = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _text(img,
              f"cam {self.camera_id} | eng {self.engine_id} | fid {ctx.get('fid')} | "
              f"{wall} | {self._fps_ema:.1f} fps | dets {len(ctx.get('detections', []))} | "
              f"tracks {ctx.get('n_live')}(+{ctx.get('n_ghost')} ghost)",
              (6, 15), C_TEXT, 0.45)

    # ---- right-hand telemetry panel ----
    def _draw_panel(self, height: int, ctx: Dict[str, Any]) -> np.ndarray:
        W = int(self.cfg.panel_width)
        panel = np.full((height, W, 3), C_BG, dtype=np.uint8)
        y = 18

        def row(txt, col=C_TEXT, scale=0.42, step=15):
            nonlocal y
            if y < height - 4:
                _text(panel, txt[:64], (8, y), col, scale)
            y += step

        row(f"CAMERA {self.camera_id}   ENGINE {self.engine_id}", C_LINE, 0.52, 20)
        row(ctx.get("url", "")[:60], C_DIM, 0.36)
        row(f"status {ctx.get('cam_status')}   fid {ctx.get('fid')}   "
            f"written {self.frames_written}", C_DIM)
        row(f"loop {self._fps_ema:5.1f} fps   overlay {self._draw_ms_ema:4.1f} ms", C_DIM)
        row(f"device {ctx.get('device')}  imgsz {ctx.get('imgsz')}  conf {ctx.get('conf')}", C_DIM)
        row(f"batch {ctx.get('batch_size')}  infer {ctx.get('infer_ms', 0.0):.0f} ms", C_DIM)
        row(f"ocr queue in {ctx.get('ocr_qin')} / out {ctx.get('ocr_qout')}", C_DIM)

        tg = ctx.get("triggers", {})
        row("triggers: " + "  ".join(
            f"{k.replace('_trig','')}:{'ON' if v else 'off'}" for k, v in tg.items()
        ), C_WARN, 0.38)

        frame_wh = ctx.get("frame_wh")
        row(f"frame {frame_wh}  roi_px {tuple(int(v) for v in (ctx.get('roi_px') or (0,0,0,0)))}",
            C_DIM, 0.36)

        y += 4
        cv2.line(panel, (6, y), (W - 6, y), (60, 60, 60), 1)
        y += 16
        row("TRACKS", C_LINE, 0.48, 18)

        tracks = sorted(ctx.get("tracks", []), key=lambda t: (t.get("ghost", False), t["track_id"]))
        for t in tracks[:10]:
            tid = int(t["track_id"])
            col = C_TRACK_GHOST if t.get("ghost") else _color_for_track(tid)
            row(f"#{tid} c{t.get('cls')} s{t.get('score',0.0):.2f} age{t.get('seen_frames')} "
                f"{'GHOST ' + str(t.get('absent')) if t.get('ghost') else ''}", col, 0.42, 14)
            row(f"   crops {t.get('n_crops')}/{t.get('n_best')} best {t.get('best_score',0.0):.2f} "
                f"sharp {t.get('sharpness',0.0):.0f} q:{'P' if t.get('quality_ok') else 'F'}",
                C_DIM, 0.36, 13)
            row(f"   side {t.get('side') or '-'} inroi {t.get('inside_frames',0)} "
                f"v {(t.get('velocity') if t.get('velocity') is not None else float('nan')):.1f} "
                f"{'STOP' if t.get('stopped') else ''}", C_DIM, 0.36, 13)
            for stage, txt in (t.get("ocr_rows") or []):
                row(f"   {stage:<11}{txt}", C_TRACK_OCR, 0.36, 13)
            if t.get("ocr_pending"):
                row(f"   PENDING {t.get('ocr_pending')} {t.get('ocr_pending_age',0.0):.1f}s",
                    C_WARN, 0.36, 13)
            elif t.get("last_ocr_latency_ms") is not None:
                row(f"   rtt {t.get('last_ocr_latency_ms'):.0f}ms", C_DIM, 0.36, 13)
            if t.get("events"):
                row("   fired: " + ",".join(t["events"].keys()), C_OK, 0.36, 14)

        y += 4
        cv2.line(panel, (6, y), (W - 6, y), (60, 60, 60), 1)
        y += 16
        row("PIPELINE EVENTS", C_LINE, 0.48, 18)

        for ts, kind, text in list(self.events)[-self.cfg.event_lines:]:
            hh = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            row(f"{hh} {kind} {text}", EVENT_COLORS.get(kind, C_TEXT), 0.36, 14)

        return panel
