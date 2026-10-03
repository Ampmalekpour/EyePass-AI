"""
bench_multistream.py — FINAL real-time multi-stream benchmark of the plate detector.
Every model runs with its best deployment, and everything is compared against PyTorch on the GPU.
(The detector service runs these same deployments: see detector/src/inference_backends.py.)

    variant       device      deployment (the best one measured)
    pt_gpu        CUDA GPU    Ultralytics predict, FP16, ONE batcher: the pending frames of all streams go
                              into one predict() call (waits until every stream has a frame, or the oldest
                              frame has waited MAX_WAIT_MS)                    <- the REFERENCE / baseline
    ov_int8_box   intel:cpu   Ultralytics predict, INT8 with the box branch kept in FP32
                              (requantize_int8_head_fp32.py), + INT8 duplicate fix in Ultralytics' postprocess
    ov_fp32       intel:cpu   Ultralytics predict, OpenVINO FP32, no extra post-processing needed
    onnx          CPU         own ONNX Runtime engine: one session per worker, CPU threads split between them,
                              own letterbox + OpenCV NMS (through Ultralytics every session grabs all cores)
CPU variants: one worker thread per stream, each with its own model instance / session.

Streams: N simulated cameras play the same clip at the video fps, phase-shifted. Each camera keeps only its
LATEST frame: a frame not picked up before the next one arrives is DROPPED (missed). --streams can hold
several counts, e.g. 4 8 12, to find how many cameras each variant can serve.

Accuracy reference = pt on the GPU (same FP16 / device settings as pt_gpu), predicted once per frame on every
frame, cached. Each variant is compared on the frames it processed: recall, precision, F1, box IoU, confidence
difference, and every unmatched box is classified (loose / duplicate / low-conf / isolated).
"eff rec%" counts the plates in dropped frames as missed = what you really get in real time.

Times (ms): pre / infer / post per frame (Ultralytics r.speed, own timers for onnx; for the GPU batch the
batch time is split over the frames in it), busy = compute time per frame, latency = arrival -> result.

Usage (inside the detector container — it has ultralytics, openvino, onnxruntime and CUDA torch):

    docker compose run --rm -v "<clips folder>:/clips:ro" plate_detector \
        python tools/bench_multistream.py --video /clips/video2.mp4 --streams 4 8

    --model-dir   model folder with export_info.yaml       (default /models/plate_v8n_480)
    --results     where results go (must be writable)      (default /debug/bench_final -> ./debug_video/bench_final)
    --variants    subset, e.g. --variants pt_gpu ov_fp32
    --frames N    frames per stream (default 3000)  --compare-only <N>_streams folder

Needs a CUDA GPU (pt_gpu and the reference). Results: <results>/<date_time>/<N>_streams/
"""
import argparse
import csv
import gc
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# ===================== SETTINGS (defaults; command line overrides) =====================
MODEL_DIR = Path("/models/plate_v8n_480")
VIDEO_PATH = Path("/clips/video2.mp4")
RESULTS_ROOT = Path("/debug/bench_final")

GPU_DEVICE = 0             # CUDA device for pt_gpu and the reference
GPU_HALF = True            # FP16 on the GPU (pt_gpu and the reference use the same setting)
MAX_WAIT_MS = None         # GPU batcher: max wait of the oldest frame for the batch to fill; None = 1 frame period
OV_DEVICE = "intel:cpu"    # OpenVINO device

INT8_FIX = {"nms_iou": 0.5,     # remove a box if IoU with a stronger box >= this
            "iomin": 0.7,       # ... or if the stronger box covers >= 70% of the smaller one
            "agnostic": True,   # across classes (two plates can't occupy the same place)
            "merge_iou": 0.6}   # raw candidates with IoU >= 0.6 are averaged into the kept box

# variant: (engine, deployment, INT8 fix) - order = run order; pt_gpu must stay (it is the baseline)
VARIANTS = {
    "pt_gpu":      ("pt",          "gpu_batch",   None),
    "ov_int8_box": ("ov_int8_box", "ultralytics", INT8_FIX),
    "ov_fp32":     ("ov_fp32",     "ultralytics", None),
    "onnx":        ("onnx",        "own",         None),
}

STREAM_COUNTS = [4]        # e.g. [4, 8, 12] for a capacity test
START_FRAME = 0
NUM_FRAMES = 3000
STREAM_FPS = None          # None = video fps
ONNX_CPU_THREADS = None    # onnx: threads split between the sessions; None = all logical CPUs

CONF_THRES = 0.25
NMS_IOU = 0.7
MAX_DET = 300
IOU_MATCH = 0.5
WARMUP_RUNS = 10
COOLDOWN_SEC = 5
REPORT_EVERY_SEC = 10
COMPARE_ONLY = None        # path of an existing "<N>_streams" folder -> only recompute the comparison
# =======================================================================================

NAN = float("nan")
COLS = ["stream", "frame", "status", "worker", "batch_size", "t_arrival_s", "wait_ms", "pre_ms", "infer_ms",
        "post_ms", "busy_ms", "latency_ms", "n_dets"]
C = {c: k for k, c in enumerate(COLS)}


# ---------------------------------------------------------------- frames
def load_frames(size):
    """Cache frames resized exactly like Ultralytics' letterbox (no padding: it is added at run time)."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = min(NUM_FRAMES, total - START_FRAME) if total > 0 else NUM_FRAMES
    if n < NUM_FRAMES:
        print(f"WARNING: video has only {total - START_FRAME} frames after START_FRAME - using {n}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    h, w = size
    r = min(h / fh, w / fw)
    nh, nw = round(fh * r), round(fw * r)
    frames = np.empty((n, nh, nw, 3), dtype=np.uint8)
    print(f"Decoding {n} frames ({fw}x{fh} -> {nw}x{nh}, padded to {w}x{h} at run time) ...")
    i = 0
    while i < n:
        ok, f = cap.read()
        if not ok:
            break
        frames[i] = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
        i += 1
    cap.release()
    print(f"  {i} frames cached ({frames[:i].nbytes / 1e9:.2f} GB) | video {fps:.2f} fps")
    return frames[:i], {"sx": fw / nw, "sy": fh / nh, "r": r, "fw": fw, "fh": fh, "nw": nw, "nh": nh,
                        "top": (h - nh) // 2, "left": (w - nw) // 2}, fps


def ultra_dets(r, scale):
    """Ultralytics Results (boxes in cached-frame pixels) -> list of (cls, conf, [x1,y1,x2,y2]) in video pixels."""
    b = r.boxes
    sx, sy = scale["sx"], scale["sy"]
    return [(c, s, [x[0] * sx, x[1] * sy, x[2] * sx, x[3] * sy])
            for c, s, x in zip(b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist())]


# ---------------------------------------------------------------- INT8 fix inside Ultralytics
def overlaps(a, b):
    """Pairwise IoU and IoMin (intersection / smaller area) between box arrays a (N,4) and b (M,4)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    ab = np.prod(b[:, 2:] - b[:, :2], axis=1)
    iou = inter / (aa[:, None] + ab[None, :] - inter + 1e-9)
    iomin = inter / (np.minimum(aa[:, None], ab[None, :]) + 1e-9)
    return iou, iomin


def apply_fix(data, raw, in_shape, orig_shape, fix):
    """data: Ultralytics' kept boxes (n,6) = x1 y1 x2 y2 conf cls (frame pixels).
    raw: this image's raw model output (4+nc, N) in model-input pixels (cx cy w h + class scores)."""
    import torch
    from ultralytics.utils import ops
    data = data[np.argsort(-data[:, 4])]
    agn = fix.get("agnostic", True)
    iu, im = overlaps(data[:, :4], data[:, :4])                 # second suppression
    alive = np.ones(len(data), bool)
    for a in range(len(data)):
        if not alive[a]:
            continue
        for c in range(a + 1, len(data)):
            if alive[c] and (agn or data[a, 5] == data[c, 5]) and \
                    (iu[a, c] >= fix["nms_iou"] or im[a, c] >= fix["iomin"]):
                alive[c] = False
    data = data[alive]
    if fix.get("merge_iou") and len(data):                       # merge
        p = raw.T
        scores = p[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        m = conf > CONF_THRES
        if m.any():
            b = p[m, :4]
            xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                    b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]).astype(np.float32)
            xyxy = ops.scale_boxes(in_shape, torch.from_numpy(xyxy), orig_shape).numpy()
            iu2, _ = overlaps(data[:, :4], xyxy)
            w = (iu2 >= fix["merge_iou"]) * conf[m][None, :]
            if not agn:
                w = w * (cls[m][None, :] == data[:, 5][:, None])
            s = w.sum(1)
            ok = s > 0
            data[ok, :4] = (w @ xyxy)[ok] / s[ok, None]
    return data


def make_predictor(fix):
    """Ultralytics DetectionPredictor whose postprocess() also applies the INT8 fix."""
    from ultralytics.models.yolo.detect import DetectionPredictor

    class FixPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            # copy BEFORE Ultralytics' NMS: it converts the boxes xywh -> xyxy in place
            raw = raw.detach().float().cpu().numpy().copy() if hasattr(raw, "detach") else np.array(raw, np.float32)
            results = super().postprocess(preds, img, orig_imgs, **kwargs)
            import torch
            for i, r in enumerate(results):
                if len(r.boxes):
                    fixed = apply_fix(r.boxes.data.float().cpu().numpy(), raw[i], img.shape[2:], r.orig_shape, fix)
                    r.update(boxes=torch.from_numpy(np.ascontiguousarray(fixed)))
            return results

    return FixPredictor


# ---------------------------------------------------------------- runners
class GpuBatchRunner:
    """pt_gpu: one YOLO instance, one predict() per batch of frames (all streams), FP16 on CUDA."""

    def __init__(self, info, size, scale):
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available - pt_gpu (and the reference) need the GPU.")
        torch.backends.cudnn.benchmark = True
        self.model = YOLO(str(MODEL_DIR / info["pt"]))
        self.kw = dict(imgsz=list(size), conf=CONF_THRES, iou=NMS_IOU, max_det=MAX_DET,
                       half=GPU_HALF, device=GPU_DEVICE, verbose=False)
        self.scale = scale
        self.desc = (f"Ultralytics {info['pt']} on {torch.cuda.get_device_name(GPU_DEVICE)} | "
                     f"{'FP16' if GPU_HALF else 'FP32'} | batched over streams")

    def warmup(self, frames, n_streams):
        for j in range(WARMUP_RUNS):                       # every batch size that can occur
            for bs in range(1, n_streams + 1):
                self.model.predict([frames[(j + k) % len(frames)] for k in range(bs)], **self.kw)

    def __call__(self, imgs):
        res = self.model.predict(list(imgs), **self.kw)
        sp = res[0].speed                                  # per image (batch time / batch size)
        return [ultra_dets(r, self.scale) for r in res], sp["preprocess"], sp["inference"], sp["postprocess"]


class UltraCpuRunner:
    """ov_fp32 / ov_int8_box: one YOLO instance per worker on OV_DEVICE (+ INT8 fix)."""

    def __init__(self, engine, fix, info, size, scale):
        from ultralytics import YOLO
        path = MODEL_DIR / info["openvino"][engine]
        self.model = YOLO(str(path), task="detect")
        self.predictor_cls = make_predictor(fix) if fix else None
        self.kw = dict(imgsz=list(size), conf=CONF_THRES, iou=NMS_IOU, max_det=MAX_DET,
                       device=OV_DEVICE, half=False, verbose=False)
        self.scale = scale
        self.desc = f"Ultralytics {path.name} on {OV_DEVICE}" + (" + INT8 fix" if fix else "")

    def warmup(self, frames, n_streams):
        for j in range(WARMUP_RUNS):
            self(frames[j])

    def __call__(self, img):
        r = self.model.predict(img, predictor=self.predictor_cls, **self.kw)[0]
        sp = r.speed
        return ultra_dets(r, self.scale), sp["preprocess"], sp["inference"], sp["postprocess"]


def onnx_postprocess(out, meta):
    """Class-aware NMS (OpenCV) + mapping back to original video pixels."""
    p = out[0].T
    scores = p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    m = conf > CONF_THRES
    if not m.any():
        return []
    b, cls, conf = p[m, :4], cls[m], conf[m]
    xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])
    nb = xyxy + cls[:, None].astype(np.float32) * 7680.0
    keep = cv2.dnn.NMSBoxes(np.column_stack([nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0],
                                             nb[:, 3] - nb[:, 1]]).tolist(), conf.tolist(), 0.0, NMS_IOU)
    keep = np.array(keep).reshape(-1)
    keep = keep[np.argsort(-conf[keep])][:MAX_DET]
    xyxy = xyxy[keep]
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - meta["left"]) / meta["r"]).clip(0, meta["fw"])
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - meta["top"]) / meta["r"]).clip(0, meta["fh"])
    return [(int(c), float(s), bx.tolist()) for c, s, bx in zip(cls[keep], conf[keep], xyxy)]


class OnnxOwnRunner:
    """onnx: own ONNX Runtime session per worker with a fixed share of the CPU threads."""

    def __init__(self, info, size, scale, threads):
        import onnxruntime as ort
        path = MODEL_DIR / info["onnx"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.n = self.s.get_inputs()[0].name
        h, w = size
        self.meta = scale
        self.buf = np.full((h, w, 3), 114, dtype=np.uint8)      # padding stays 114, only the centre changes
        self.sl = (slice(scale["top"], scale["top"] + scale["nh"]), slice(scale["left"], scale["left"] + scale["nw"]))
        self.desc = f"own ONNX Runtime {path.name} on CPU, {threads} threads per session"

    def warmup(self, frames, n_streams):
        for j in range(WARMUP_RUNS):
            self(frames[j])

    def __call__(self, img):
        t0 = time.perf_counter()
        self.buf[self.sl] = img
        x = cv2.dnn.blobFromImage(self.buf, scalefactor=1 / 255.0, swapRB=True)
        t1 = time.perf_counter()
        out = self.s.run(None, {self.n: x})[0]
        t2 = time.perf_counter()
        dets = onnx_postprocess(out, self.meta)
        t3 = time.perf_counter()
        return dets, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def make_runners(engine, deploy, fix, info, size, scale, n_streams):
    if deploy == "gpu_batch":
        r = GpuBatchRunner(info, size, scale)
        return "batch", [r], r.desc
    if deploy == "own":
        per = max(1, (ONNX_CPU_THREADS or os.cpu_count() or 4) // n_streams)
        rs = [OnnxOwnRunner(info, size, scale, per) for _ in range(n_streams)]
        return "per_frame", rs, f"{rs[0].desc} | {n_streams} sessions"
    rs = [UltraCpuRunner(engine, fix, info, size, scale) for _ in range(n_streams)]
    return "per_frame", rs, f"{rs[0].desc} | {n_streams} YOLO instances"


# ---------------------------------------------------------------- real-time pass
def realtime_pass(name, mode, runners, frames, fps, n_streams):
    n = len(frames)
    period = 1.0 / fps
    max_wait = (MAX_WAIT_MS / 1000) if MAX_WAIT_MS else period
    lock = threading.Condition()
    pending = [None] * n_streams
    missed = [0] * n_streams
    frame_rows, det_rows = [], []
    state = {"done": False, "inflight": 0, "max_inflight": 0}

    t_start = time.perf_counter() + 0.3
    schedule = sorted((t_start + i * period + s * period / n_streams, s, i)
                      for i in range(n) for s in range(n_streams))

    def camera():
        for t_arr, s, i in schedule:
            while True:
                now = time.perf_counter()
                if now >= t_arr:
                    break
                time.sleep(max(0.0, t_arr - now - 0.0005))
            with lock:
                old = pending[s]
                if old is not None:
                    missed[s] += 1
                pending[s] = (i, t_arr)
                lock.notify_all()
            if old is not None:
                frame_rows.append((s, old[0], "dropped", -1, 0, round(old[1] - t_start, 4), *([NAN] * 6), 0))
        with lock:
            state["done"] = True
            lock.notify_all()

    def record(items, results, wid, t0, t1, pre, inf, post):
        bs = len(items)
        for (s, i, t_arr), d in zip(items, results):
            frame_rows.append((s, i, "processed", wid, bs, round(t_arr - t_start, 4), round((t0 - t_arr) * 1000, 3),
                               round(pre, 3), round(inf, 3), round(post, 3), round((t1 - t0) * 1000 / bs, 3),
                               round((t1 - t_arr) * 1000, 3), len(d)))
            for c, conf, b in d:
                det_rows.append((s, i, c, round(conf, 5), *(round(v, 2) for v in b)))

    def take(s):
        i, t_arr = pending[s]
        pending[s] = None
        return s, i, t_arr

    def worker(wid, run):                  # CPU: one frame at a time, oldest waiting first
        while True:
            with lock:
                while all(p is None for p in pending) and not state["done"]:
                    lock.wait(0.05)
                waiting = [(p[1], s) for s, p in enumerate(pending) if p is not None]
                if not waiting:
                    return
                item = take(min(waiting)[1])
                state["inflight"] += 1
                state["max_inflight"] = max(state["max_inflight"], state["inflight"])
            t0 = time.perf_counter()
            d, pre, inf, post = run(frames[item[1]])
            t1 = time.perf_counter()
            with lock:
                state["inflight"] -= 1
            record([item], [d], wid, t0, t1, pre, inf, post)

    def batcher(run):                      # GPU: all pending frames in one predict()
        bid = 0
        while True:
            with lock:
                while True:
                    waiting = [s for s, p in enumerate(pending) if p is not None]
                    if not waiting:
                        if state["done"]:
                            return
                        lock.wait(0.05)
                        continue
                    if len(waiting) == n_streams or state["done"]:
                        break
                    left = min(pending[s][1] for s in waiting) + max_wait - time.perf_counter()
                    if left <= 0:
                        break
                    lock.wait(left)
                items = [take(s) for s in waiting]
                state["max_inflight"] = max(state["max_inflight"], len(items))
            t0 = time.perf_counter()
            dets, pre, inf, post = run([frames[i] for _, i, _ in items])
            t1 = time.perf_counter()
            record(items, dets, bid, t0, t1, pre, inf, post)
            bid += 1

    stop = threading.Event()

    def reporter():
        last, t_last = 0, time.perf_counter()
        while not stop.wait(REPORT_EVERY_SEC):
            rows = frame_rows[last:]
            last += len(rows)
            now = time.perf_counter()
            proc = [r for r in rows if r[2] == "processed"]
            if proc:
                a = np.array([[r[C["batch_size"]], r[C["infer_ms"]], r[C["latency_ms"]], r[C["n_dets"]]] for r in proc])
                print(f"  [{name}] t={now - t_start:5.0f}s | frames {len(proc):5d} ({len(proc) / (now - t_last):6.1f}/s)"
                      f" | dropped {len(rows) - len(proc):4d} (total {sum(missed):5d}) | batch {a[:, 0].mean():4.2f}"
                      f" | infer/frame {a[:, 1].mean():6.2f} (p95 {np.percentile(a[:, 1], 95):6.2f}) ms"
                      f" | lat {a[:, 2].mean():6.1f} ms | dets/frame {a[:, 3].mean():.2f}", flush=True)
            t_last = now

    if mode == "batch":
        threads = [threading.Thread(target=batcher, args=(runners[0],), daemon=True)]
    else:
        threads = [threading.Thread(target=worker, args=(k, r), daemon=True) for k, r in enumerate(runners)]
    cam = threading.Thread(target=camera, daemon=True)
    rep = threading.Thread(target=reporter, daemon=True)
    for t in threads:
        t.start()
    cam.start()
    rep.start()
    cam.join()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start
    stop.set()
    rep.join()
    frame_rows.sort(key=lambda r: (r[0], r[1]))
    det_rows.sort(key=lambda r: (r[0], r[1]))
    return {"frame_rows": frame_rows, "det_rows": det_rows, "elapsed": elapsed,
            "max_inflight": state["max_inflight"], "missed": missed}


# ---------------------------------------------------------------- reference: pt on the GPU, every frame
def reference(info, size, frames, scale, out_dir):
    cache = RESULTS_ROOT / "reference_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pt = MODEL_DIR / info["pt"]
    key = {"pt": str(pt), "pt_mtime": pt.stat().st_mtime, "video": str(VIDEO_PATH), "start": START_FRAME,
           "n": len(frames), "imgsz": list(size), "conf": CONF_THRES, "iou": NMS_IOU, "max_det": MAX_DET,
           "device": str(GPU_DEVICE), "half": GPU_HALF}
    ref_csv, key_json = cache / "reference_pt_gpu.csv", cache / "reference_pt_gpu_key.json"
    if ref_csv.exists() and key_json.exists() and json.load(open(key_json)) == key:
        print(f"Reference (pt GPU): loaded from cache ({ref_csv})")
    else:
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available - the reference is pt on the GPU.")
        model = YOLO(str(pt))
        kw = dict(imgsz=list(size), conf=CONF_THRES, iou=NMS_IOU, max_det=MAX_DET, half=GPU_HALF,
                  device=GPU_DEVICE, verbose=False)
        print(f"Reference: pt on {torch.cuda.get_device_name(GPU_DEVICE)} "
              f"({'FP16' if GPU_HALF else 'FP32'}), every frame - runs once ...")
        t0 = time.perf_counter()
        with open(ref_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "cls", "conf", "x1", "y1", "x2", "y2"])
            for i, img in enumerate(frames):
                for c, s, b in ultra_dets(model.predict(img, **kw)[0], scale):
                    w.writerow([i, c, round(s, 5), *(round(v, 2) for v in b)])
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(frames)} ({time.perf_counter() - t0:.0f} s)")
        json.dump(key, open(key_json, "w"), indent=2)
    shutil.copy2(ref_csv, out_dir / "reference_detections.csv")


# ---------------------------------------------------------------- comparison
def box_ovl(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    u = aa + ab - inter
    return (inter / u if u > 0 else 0.0), (inter / min(aa, ab) if min(aa, ab) > 0 else 0.0)


def match(ref, dets):
    """Greedy, same class, IoU >= IOU_MATCH, one-to-one. Returns (ref_idx, det_idx, iou, |dconf|, cls)."""
    used, pairs = set(), []
    for ri in sorted(range(len(ref)), key=lambda k: -ref[k][1]):
        rc, rconf, rbox = ref[ri]
        best, bj = IOU_MATCH, -1
        for j, (c, conf, box) in enumerate(dets):
            if j not in used and c == rc:
                v = box_ovl(rbox, box)[0]
                if v >= best:
                    best, bj = v, j
        if bj >= 0:
            used.add(bj)
            pairs.append((ri, bj, best, abs(rconf - dets[bj][1]), rc))
    return pairs


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def read_boxes(p, key_cols):
    d = defaultdict(list)
    for r in read_csv(p):
        k = tuple(int(r[c]) for c in key_cols)
        d[k if len(k) > 1 else k[0]].append(
            (int(r["cls"]), float(r["conf"]), [float(r[c]) for c in ("x1", "y1", "x2", "y2")]))
    return d


def compare(rdir):
    rdir = Path(rdir)
    st = json.load(open(rdir / "settings.json"))
    names = {int(k): v for k, v in st["names"].items()}
    nc = len(names)
    ns, n_frames, fps = st["n_streams"], st["n_frames"], st["fps"]
    conf_thr = st.get("conf", CONF_THRES)
    ref = read_boxes(rdir / "reference_detections.csv", ["frame"])
    ref_per_stream = sum(len(ref[i]) for i in range(n_frames))

    speed, acc = [], []
    for v in st["variants"]:
        if not (rdir / f"frames_{v}.csv").exists():
            continue
        run = json.load(open(rdir / f"run_{v}.json"))
        fr = read_csv(rdir / f"frames_{v}.csv")
        proc = [r for r in fr if r["status"] == "processed"]
        drop = [r for r in fr if r["status"] == "dropped"]
        col = lambda k: np.array([float(r[k]) for r in proc]) if proc else np.array([NAN])  # noqa: E731
        inf, lat, busy = col("infer_ms"), col("latency_ms"), col("busy_ms")
        el = run["elapsed_s"]
        speed.append({
            "variant": v, "engine": run["engine"], "batch": float(np.mean(col("batch_size"))),
            "pre_ms": np.mean(col("pre_ms")), "infer_ms": np.mean(inf), "infer_p50": np.median(inf),
            "infer_p95": np.percentile(inf, 95), "infer_max": np.max(inf), "post_ms": np.mean(col("post_ms")),
            "busy_ms": np.mean(busy), "wait_ms": np.mean(col("wait_ms")), "lat_ms": np.mean(lat),
            "lat_p95": np.percentile(lat, 95), "lat_max": np.max(lat), "total_fps": len(proc) / el,
            "needed_fps": ns * fps, "processed": len(proc), "missed": len(drop),
            "missed_pct": 100 * len(drop) / max(len(fr), 1),
            "missed_per_stream": [sum(int(r["stream"]) == s for r in drop) for s in range(ns)],
            "load_pct": 100 * np.nansum(busy) / 1000 / el / (1 if run["mode"] == "batch" else ns),
            "max_inflight": run["max_inflight"]})

        dets = read_boxes(rdir / f"detections_{v}.csv", ["stream", "frame"])
        m_c, r_c, d_c = np.zeros(nc), np.zeros(nc), np.zeros(nc)
        ious, dconf, same_count = [], [], 0
        loose = dup = low = iso = 0
        for r in proc:
            k = (int(r["stream"]), int(r["frame"]))
            rd, dd = ref[k[1]], dets.get(k, [])
            pairs = match(rd, dd)
            for c, _, _ in rd:
                r_c[c] += 1
            for c, _, _ in dd:
                d_c[c] += 1
            for _, _, u, dc, c in pairs:
                m_c[c] += 1
                ious.append(u)
                dconf.append(dc)
            same_count += len(rd) == len(dd)
            mr, md = {p[0] for p in pairs}, {p[1] for p in pairs}
            for j, (c, cf, bx) in enumerate(dd):
                if j in md:
                    continue
                if any(box_ovl(rd[ri][2], bx)[0] > 0.1 for ri in range(len(rd)) if ri not in mr):
                    loose += 1
                elif any(max(box_ovl(dd[q][2], bx)[0] >= 0.3, box_ovl(dd[q][2], bx)[1] >= 0.5)
                         for q in range(len(dd)) if q != j):
                    dup += 1
                elif cf < conf_thr + 0.10:
                    low += 1
                else:
                    iso += 1
        mt, rt, dt = m_c.sum(), r_c.sum(), d_c.sum()
        ious = np.array(ious)
        rec = 100 * mt / rt if rt else 100.0
        prec = 100 * mt / dt if dt else 100.0
        miou = float(ious.mean()) if len(ious) else 0.0
        row = {"variant": v, "frames_checked": len(proc), "ref_boxes": int(rt), "boxes": int(dt),
               "recall": rec, "precision": prec, "f1": 2 * rec * prec / (rec + prec) if rec + prec else 0.0,
               "miou": miou, "min_iou": float(ious.min()) if len(ious) else 0.0,
               "iou_ge_090_pct": 100 * float((ious >= 0.9).mean()) if len(ious) else 0.0,
               "iou_lt_075_pct": 100 * float((ious < 0.75).mean()) if len(ious) else 0.0,
               "dconf": float(np.mean(dconf)) if dconf else 0.0,
               "same_count_pct": 100 * same_count / max(len(proc), 1),
               "eff_recall": 100 * mt / max(ns * ref_per_stream, 1),
               "missed_ref_boxes": int(rt - mt), "extra_boxes": int(dt - mt),
               "extra_loose": loose, "extra_duplicate": dup, "extra_lowconf": low, "extra_isolated": iso,
               "verdict": ("match" if rec >= 99.5 and prec >= 99.5 and miou >= 0.98 else
                           "close" if rec >= 98 and prec >= 98 and miou >= 0.95 else "DIFFERS")}
        for c in range(nc):
            row[f"recall_{names[c]}"] = 100 * m_c[c] / r_c[c] if r_c[c] else NAN
            row[f"prec_{names[c]}"] = 100 * m_c[c] / d_c[c] if d_c[c] else NAN
        acc.append(row)

    base = next((x for x in speed if x["variant"] == "pt_gpu"), None)
    line = "=" * 158
    print(f"\n{line}\nSPEED  {ns} streams x {fps:.1f} fps = {ns * fps:.0f} frames/s needed, {n_frames} frames per stream"
          f" | ms per frame | busy = compute per frame | load = share of the available workers' time used"
          f" | x GPU = infer per frame / pt_gpu's")
    print(f"{'variant':<13}{'batch':>6}{'pre':>6}{'infer':>8}{'p50':>7}{'p95':>7}{'max':>7}{'post':>6}{'busy':>7}"
          f"{'wait':>7}{'lat':>8}{'lat p95':>9}{'lat max':>9}{'fps':>8}{'missed':>8}{'miss%':>7}{'load%':>7}"
          f"{'x GPU':>7}   missed per stream")
    for x in speed:
        ratio = x["infer_ms"] / base["infer_ms"] if base and base["infer_ms"] > 0 else NAN
        x["infer_x_gpu"] = ratio
        print(f"{x['variant']:<13}{x['batch']:6.2f}{x['pre_ms']:6.2f}{x['infer_ms']:8.2f}{x['infer_p50']:7.2f}"
              f"{x['infer_p95']:7.2f}{x['infer_max']:7.1f}{x['post_ms']:6.2f}{x['busy_ms']:7.2f}{x['wait_ms']:7.2f}"
              f"{x['lat_ms']:8.2f}{x['lat_p95']:9.2f}{x['lat_max']:9.1f}{x['total_fps']:8.1f}{x['missed']:8d}"
              f"{x['missed_pct']:7.2f}{x['load_pct']:7.1f}{ratio:7.2f}   {x['missed_per_stream']}")
    for x in speed:
        print(f"  {x['variant']:<13}{x['engine']}")

    cls_cols = [f"recall_{names[c]}" for c in names] + [f"prec_{names[c]}" for c in names]
    print(f"\nACCURACY vs reference = pt on the GPU ({'FP16' if st.get('gpu_half') else 'FP32'}), every frame "
          f"(same class, IoU >= {IOU_MATCH})")
    print(f"{'variant':<13}{'frames':>8}{'ref box':>9}{'boxes':>8}{'recall%':>9}{'prec%':>8}{'F1':>7}{'mIoU':>8}"
          f"{'minIoU':>8}{'|dconf|':>9}{'same#%':>8}{'eff rec%':>10}{'verdict':>9}   "
          + "  ".join(c.replace("recall_", "R ").replace("prec_", "P ") for c in cls_cols))
    for x in acc:
        print(f"{x['variant']:<13}{x['frames_checked']:8d}{x['ref_boxes']:9d}{x['boxes']:8d}{x['recall']:9.2f}"
              f"{x['precision']:8.2f}{x['f1']:7.2f}{x['miou']:8.4f}{x['min_iou']:8.3f}{x['dconf']:9.4f}"
              f"{x['same_count_pct']:8.2f}{x['eff_recall']:10.2f}{x['verdict']:>9}   "
              + "  ".join(f"{x[c]:6.2f}" for c in cls_cols))

    print("\nWHAT THE DIFFERENCES ARE (processed frames only)")
    print(f"{'variant':<13}{'missed ref':>11}{'extra':>7}{'loose':>7}{'duplicate':>11}{'low-conf':>10}"
          f"{'isolated':>10}{'IoU>=0.9 %':>12}{'IoU<0.75 %':>12}")
    for x in acc:
        print(f"{x['variant']:<13}{x['missed_ref_boxes']:11d}{x['extra_boxes']:7d}{x['extra_loose']:7d}"
              f"{x['extra_duplicate']:11d}{x['extra_lowconf']:10d}{x['extra_isolated']:10d}"
              f"{x['iou_ge_090_pct']:12.2f}{x['iou_lt_075_pct']:12.2f}")
    print("loose = overlaps an unmatched reference plate (box too far off) | duplicate = overlaps another box of the "
          "same frame\nlow-conf = isolated, conf < threshold + 0.10 | isolated = isolated and confident | "
          "eff rec% = recall when dropped frames count as missed")
    print(line)

    def dump(rows, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in r.items()})
    if speed:
        dump(speed, rdir / "comparison_speed.csv")
        dump(acc, rdir / "comparison_accuracy.csv")
        json.dump({"speed": speed, "accuracy": acc}, open(rdir / "comparison.json", "w"), indent=2, default=float)
        print(f"Saved comparison in {rdir}")
    return speed, acc


# ---------------------------------------------------------------- command line
def parse_args():
    global MODEL_DIR, VIDEO_PATH, RESULTS_ROOT, STREAM_COUNTS, NUM_FRAMES, START_FRAME, STREAM_FPS
    global GPU_DEVICE, GPU_HALF, MAX_WAIT_MS, ONNX_CPU_THREADS, COMPARE_ONLY, VARIANTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--video", default=str(VIDEO_PATH))
    ap.add_argument("--results", default=str(RESULTS_ROOT))
    ap.add_argument("--streams", type=int, nargs="+", default=STREAM_COUNTS)
    ap.add_argument("--frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--start-frame", type=int, default=START_FRAME)
    ap.add_argument("--fps", type=float, default=STREAM_FPS)
    ap.add_argument("--gpu", type=int, default=GPU_DEVICE)
    ap.add_argument("--fp32-gpu", action="store_true", help="GPU in FP32 instead of FP16")
    ap.add_argument("--max-wait-ms", type=float, default=MAX_WAIT_MS)
    ap.add_argument("--onnx-threads", type=int, default=ONNX_CPU_THREADS)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--compare-only", default=COMPARE_ONLY)
    a = ap.parse_args()
    MODEL_DIR, VIDEO_PATH, RESULTS_ROOT = Path(a.model_dir), Path(a.video), Path(a.results)
    STREAM_COUNTS, NUM_FRAMES, START_FRAME, STREAM_FPS = a.streams, a.frames, a.start_frame, a.fps
    GPU_DEVICE, GPU_HALF, MAX_WAIT_MS = a.gpu, not a.fp32_gpu, a.max_wait_ms
    ONNX_CPU_THREADS, COMPARE_ONLY = a.onnx_threads, a.compare_only
    if "pt_gpu" not in a.variants:
        a.variants = ["pt_gpu"] + a.variants          # the baseline must stay
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in a.variants}


# ---------------------------------------------------------------- main
def main():
    parse_args()
    if COMPARE_ONLY:
        compare(COMPARE_ONLY)
        return
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)

    info = yaml.safe_load(open(MODEL_DIR / "export_info.yaml", encoding="utf-8"))
    size = tuple(info["imgsz"])
    frames, scale, video_fps = load_frames(size)
    fps = STREAM_FPS or video_fps
    variants = {k: v for k, v in VARIANTS.items() if v[0] in ("pt", "onnx") or v[0] in info.get("openvino", {})}
    for k in VARIANTS:
        if k not in variants:
            print(f"{k}: model not found in export_info.yaml - skipped"
                  + (" (run requantize_int8_head_fp32.py first)" if "int8_box" in k else ""))
    root = RESULTS_ROOT / time.strftime("%Y%m%d_%H%M%S")
    print(f"Model {info.get('model_name', MODEL_DIR.name)} | input {size[1]}x{size[0]} | streams {STREAM_COUNTS} "
          f"@ {fps:.1f} fps | logical CPUs {os.cpu_count()}\nResults -> {root}")

    capacity = []
    for ns in STREAM_COUNTS:
        out_dir = root / f"{ns}_streams"
        out_dir.mkdir(parents=True)
        json.dump({"model_dir": str(MODEL_DIR), "video": str(VIDEO_PATH), "start_frame": START_FRAME,
                   "n_frames": len(frames), "n_streams": ns, "fps": fps, "logical_cpus": os.cpu_count(),
                   "gpu_device": str(GPU_DEVICE), "gpu_half": GPU_HALF, "max_wait_ms": MAX_WAIT_MS,
                   "ov_device": OV_DEVICE, "imgsz": list(size),
                   "names": {int(k): v for k, v in info["names"].items()},
                   "conf": CONF_THRES, "nms_iou": NMS_IOU, "iou_match": IOU_MATCH, "variants": list(variants),
                   "variant_config": {k: {"engine": e, "deployment": d, "fix": f} for k, (e, d, f) in variants.items()}},
                  open(out_dir / "settings.json", "w"), indent=2)
        reference(info, size, frames, scale, out_dir)

        for k, (name, (engine, deploy, fix)) in enumerate(variants.items()):
            if k:
                time.sleep(COOLDOWN_SEC)
            mode, runners, engine_info = make_runners(engine, deploy, fix, info, size, scale, ns)
            print(f"\n=== {ns} streams | {name} | {engine_info} ===")
            for r in runners:
                r.warmup(frames, ns)
            res = realtime_pass(name, mode, runners, frames, fps, ns)

            with open(out_dir / f"frames_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(COLS)
                w.writerows(res["frame_rows"])
            with open(out_dir / f"detections_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stream", "frame", "cls", "conf", "x1", "y1", "x2", "y2"])
                w.writerows(res["det_rows"])
            json.dump({"variant": name, "engine": engine_info, "deployment": deploy, "mode": mode, "fix": fix,
                       "elapsed_s": res["elapsed"], "max_inflight": res["max_inflight"],
                       "missed_per_stream": res["missed"]}, open(out_dir / f"run_{name}.json", "w"), indent=2)
            proc = [r for r in res["frame_rows"] if r[2] == "processed"]
            tot = len(res["frame_rows"])
            print(f"--- {name}: processed {len(proc)} / {tot} | missed {sum(res['missed'])} "
                  f"({100 * sum(res['missed']) / max(tot, 1):.2f}%) | infer/frame avg "
                  f"{np.mean([r[C['infer_ms']] for r in proc]):.2f} ms | latency avg "
                  f"{np.mean([r[C['latency_ms']] for r in proc]):.2f} ms | {len(proc) / res['elapsed']:.1f} frames/s")
            del runners
            gc.collect()

        speed, acc = compare(out_dir)
        capacity += [(ns, s["variant"], s["missed_pct"], s["lat_ms"], s["load_pct"], a["eff_recall"])
                     for s, a in zip(speed, acc)]

    if len(STREAM_COUNTS) > 1:
        print("\nCAPACITY (missed % / latency ms / load % / effective recall %)")
        for name in variants:
            cells = [f"{ns:>3} st: {m:6.2f}% {lat:6.1f} ms {ld:5.1f}% {e:6.2f}%"
                     for ns, v, m, lat, ld, e in capacity if v == name]
            print(f"  {name:<13}" + " | ".join(cells))


if __name__ == "__main__":
    main()
