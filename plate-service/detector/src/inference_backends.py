"""
inference_backends.py
--------------------------------------------------------------------
The one place the detector turns camera ROI frames into boxes. Each
backend is the deployment that measured best in the multi-stream
benchmark (tools/bench_multistream.py), implemented the same way:

  variant         device      deployment
  pt (GPU)        CUDA        Ultralytics predict, FP16, ONE batched predict()
                              over the frames of all cameras of the engine
  openvino_fp32   intel:cpu   Ultralytics predict on the OpenVINO FP32 model,
                              ONE YOLO instance per camera, run in parallel
  openvino_int8   intel:cpu   same, on the INT8 model whose box branch is kept
                              in FP32 (requantize_int8_head_fp32.py), plus the
                              INT8 duplicate fix in Ultralytics' postprocess
  onnx            CPU         own ONNX Runtime engine: ONE session per camera,
                              CPU threads split between the sessions, own
                              letterbox + OpenCV NMS

Engine (engine.py) only calls:

    backend.ensure_streams(n_cameras)      # CPU: one instance per camera
    backend.warmup()
    detections, timings = backend.infer(frames)

and gets back, per frame, the same float64 [[x1, y1, x2, y2, conf, cls]]
array (ROI-frame pixels) it always built from Ultralytics results, plus
per-frame {"preprocess", "inference", "postprocess"} ms. The tracker,
triggers, crops and control hub are unchanged.

torch / ultralytics / openvino / onnxruntime are imported inside the
backend that needs them, in the engine child process — never in the
parent (spawn-safe, Windows-safe).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from model_files import ModelSpec, resolve_model

EMPTY_DETS = np.empty((0, 6), dtype=np.float64)
GPU, CPU = "gpu", "cpu"
CPU_VARIANTS = ("openvino_fp32", "openvino_int8", "onnx")


# ====================================================================
# Runtime plan
# ====================================================================
@dataclass
class RuntimePlan:
    device_kind: str     # gpu | cpu
    torch_device: str    # what the pt backend hands to Ultralytics: "cuda:0", "cpu"
    variant: str         # pt | openvino_fp32 | openvino_int8 | onnx
    model_name: str      # plate_v8n_480 / plate_v8s_640

    def describe(self) -> str:
        return f"device={self.device_kind} variant={self.variant} model={self.model_name}"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_runtime(device_pref: str, gpu_model: str, cpu_model: str, cpu_model_name: str,
                    gpu_device: str, strict: bool, aliases: Dict[str, str],
                    log: logging.Logger) -> RuntimePlan:
    """DETECTION_DEVICE (gpu|cpu|auto) + DETECTION_GPU_MODEL +
    DETECTION_CPU_MODEL -> a plan. torch is only probed for gpu/auto."""
    pref = (device_pref or "auto").strip().lower()
    pref = {"cuda": GPU}.get(pref, pref)
    if pref.startswith("cuda:"):
        gpu_device, pref = pref, GPU
    gpu_name = aliases.get((gpu_model or "").strip().lower(), (gpu_model or "").strip())
    cpu_variant = aliases.get((cpu_model or "").strip().lower(), (cpu_model or "").strip().lower())

    if pref == "auto":
        pref = GPU if _cuda_available() else CPU
        log.info("DETECTION_DEVICE=auto -> %s", pref)
    if pref == GPU:
        if _cuda_available():
            return RuntimePlan(GPU, gpu_device, "pt", gpu_name)
        msg = f"DETECTION_DEVICE={device_pref!r} but CUDA is not available"
        if strict:
            raise RuntimeError(msg)
        log.error("%s — falling back to the CPU pipeline (DETECTION_CPU_MODEL=%s)", msg, cpu_variant)
        pref = CPU
    if pref != CPU:
        raise ValueError(f"DETECTION_DEVICE={device_pref!r} — use gpu, cpu or auto")
    if cpu_variant not in CPU_VARIANTS:
        raise ValueError(f"DETECTION_CPU_MODEL={cpu_model!r} — use one of {CPU_VARIANTS}")
    if cpu_variant == "openvino_int8":
        log.warning("DETECTION_CPU_MODEL=openvino_int8: INT8 (box branch FP32) + duplicate fix. Check "
                    "its accuracy against the GPU with tools/bench_multistream.py before production; "
                    "INT8 shifts confidences, re-tune TRACKER_TRACK_THRESH if needed.")
    return RuntimePlan(CPU, "cpu", cpu_variant, cpu_model_name)


# ====================================================================
# Shared pieces
# ====================================================================
def _ultra_dets(r) -> np.ndarray:
    b = r.boxes
    if b is None or len(b) == 0:
        return np.empty((0, 6), dtype=np.float64)
    return np.column_stack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                            b.cls.cpu().numpy()]).astype(np.float64)


def _timing(pre: float, inf: float, post: float) -> Dict[str, float]:
    return {"preprocess": float(pre), "inference": float(inf), "postprocess": float(post)}


def letterbox(img: np.ndarray, new_hw: Tuple[int, int], color: int = 114
              ) -> Tuple[np.ndarray, float, int, int]:
    """Ultralytics LetterBox(auto=False): keep aspect ratio, INTER_LINEAR,
    centred 114 padding. Returns (image, r, left, top)."""
    h0, w0 = img.shape[:2]
    h, w = int(new_hw[0]), int(new_hw[1])
    r = min(h / h0, w / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (w - nw) / 2.0, (h - nh) / 2.0
    if (w0, h0) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, left, top


def onnx_postprocess(out: np.ndarray, r: float, left: int, top: int, orig_hw: Tuple[int, int],
                     conf_thres: float, nms_iou: float, max_det: int) -> np.ndarray:
    """The benchmark's onnx_postprocess: class-aware OpenCV NMS (7680 px
    class offset) + mapping back to the original frame. Returns float64
    [[x1, y1, x2, y2, conf, cls]]."""
    p = out[0].T
    scores = p[:, 4:]
    if scores.shape[1] == 0:
        return np.empty((0, 6), dtype=np.float64)
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    m = conf > conf_thres
    if not m.any():
        return np.empty((0, 6), dtype=np.float64)
    b, cls, conf = p[m, :4], cls[m], conf[m]
    xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])
    nb = xyxy + cls[:, None].astype(np.float32) * 7680.0
    keep = cv2.dnn.NMSBoxes(np.column_stack([nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0],
                                             nb[:, 3] - nb[:, 1]]).tolist(), conf.tolist(), 0.0, nms_iou)
    keep = np.array(keep).reshape(-1)
    keep = keep[np.argsort(-conf[keep])][:max_det]
    xyxy = xyxy[keep].astype(np.float64)
    h0, w0 = orig_hw
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - left) / r).clip(0, w0)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - top) / r).clip(0, h0)
    return np.column_stack([xyxy, conf[keep].astype(np.float64), cls[keep].astype(np.float64)])


# ---- INT8 duplicate fix (benchmark's apply_fix / FixPredictor) --------
def _overlaps(a: np.ndarray, b: np.ndarray):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    ab = np.prod(b[:, 2:] - b[:, :2], axis=1)
    iou = inter / (aa[:, None] + ab[None, :] - inter + 1e-9)
    iomin = inter / (np.minimum(aa[:, None], ab[None, :]) + 1e-9)
    return iou, iomin


def apply_int8_fix(data: np.ndarray, raw: np.ndarray, in_shape, orig_shape, fix: Dict[str, Any],
                   conf_thres: float) -> np.ndarray:
    """data: kept boxes (n, 6) x1 y1 x2 y2 conf cls (frame pixels).
    raw: this image's raw output (4+nc, N) in model-input pixels.
    1) second suppression (IoU >= nms_iou or IoMin >= iomin, class-agnostic)
    2) merge: each kept box becomes the conf-weighted mean of the raw
       candidates with IoU >= merge_iou."""
    import torch
    from ultralytics.utils import ops
    data = data[np.argsort(-data[:, 4])]
    agn = fix.get("agnostic", True)
    iu, im = _overlaps(data[:, :4], data[:, :4])
    alive = np.ones(len(data), bool)
    for a in range(len(data)):
        if not alive[a]:
            continue
        for c in range(a + 1, len(data)):
            if alive[c] and (agn or data[a, 5] == data[c, 5]) and \
                    (iu[a, c] >= fix["nms_iou"] or im[a, c] >= fix["iomin"]):
                alive[c] = False
    data = data[alive]
    if fix.get("merge_iou") and len(data):
        p = raw.T
        scores = p[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        m = conf > conf_thres
        if m.any():
            b = p[m, :4]
            xyxy = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                    b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]).astype(np.float32)
            xyxy = ops.scale_boxes(in_shape, torch.from_numpy(xyxy), orig_shape).numpy()
            iu2, _ = _overlaps(data[:, :4], xyxy)
            w = (iu2 >= fix["merge_iou"]) * conf[m][None, :]
            if not agn:
                w = w * (cls[m][None, :] == data[:, 5][:, None])
            s = w.sum(1)
            ok = s > 0
            data[ok, :4] = (w @ xyxy)[ok] / s[ok, None]
    return data


def make_int8_fix_predictor(fix: Dict[str, Any], conf_thres: float):
    """Ultralytics DetectionPredictor whose postprocess() also applies
    the INT8 fix (verbatim from the benchmark)."""
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
                    fixed = apply_int8_fix(r.boxes.data.float().cpu().numpy(), raw[i], img.shape[2:],
                                           r.orig_shape, fix, conf_thres)
                    r.update(boxes=torch.from_numpy(np.ascontiguousarray(fixed)))
            return results

    return FixPredictor


# ====================================================================
# Backends
# ====================================================================
@dataclass
class BackendSettings:
    model_root: str
    conf: float
    nms_iou: float
    max_det: int
    warmup_runs: int
    gpu_half: bool
    ov_device: str
    onnx_cpu_threads: Optional[int]
    int8_fix: Dict[str, Any]
    manifest_keys: Dict[str, List[str]]
    file_patterns: Dict[str, List[str]]
    class_labels: Dict[int, str]
    fallback_to_pt: bool = True


class DetectorBackend:
    kind = "base"

    def __init__(self, spec: ModelSpec, s: BackendSettings, log: logging.Logger):
        self.spec, self.s, self.log = spec, s, log
        self.device = "cpu"

    @property
    def label(self) -> str:
        return f"{self.device}/{self.kind}"

    @property
    def imgsz_label(self):
        return self.spec.imgsz

    def ensure_streams(self, n: int):
        pass

    def warmup(self):
        pass

    def infer(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        raise NotImplementedError

    def close(self):
        pass


class UltralyticsBatchBackend(DetectorBackend):
    """pt: ONE YOLO instance, ONE predict() per loop over every camera's
    frame (the GPU pipeline; FP16 on CUDA like the benchmark's pt_gpu).
    Also the CPU fallback (FP32) when an ONNX/OpenVINO model fails."""
    kind = "pt"

    def __init__(self, spec: ModelSpec, s: BackendSettings, device: str, log: logging.Logger):
        super().__init__(spec, s, log)
        import torch
        from ultralytics import YOLO
        self.device = device
        self.on_gpu = str(device).startswith("cuda")
        self.half = bool(s.gpu_half) and self.on_gpu
        if self.on_gpu:
            torch.backends.cudnn.benchmark = True
        self.model = YOLO(spec.path)
        self.kw = dict(imgsz=int(spec.imgsz), conf=s.conf, iou=s.nms_iou, max_det=s.max_det,
                       half=self.half, device=device, verbose=False)
        gpu_name = torch.cuda.get_device_name(torch.device(device)) if self.on_gpu else "CPU"
        log.info(f"[INIT] Ultralytics {os.path.basename(spec.path)} on {gpu_name} | "
                 f"{'FP16' if self.half else 'FP32'} | batched over cameras | imgsz={spec.imgsz} "
                 f"conf={s.conf} iou={s.nms_iou}")

    @property
    def label(self) -> str:
        return f"{'gpu' if self.on_gpu else 'cpu'}/pt-{'fp16' if self.half else 'fp32'}"

    def warmup(self):
        dummy = np.zeros((max(32, self.spec.imgsz * 9 // 16), self.spec.imgsz, 3), np.uint8)
        t0 = time.perf_counter()
        for _ in range(max(1, self.s.warmup_runs)):
            self.model.predict([dummy], **self.kw)
        self.log.info("[INIT] warm-up %d runs in %.0f ms", self.s.warmup_runs, (time.perf_counter() - t0) * 1000)

    def infer(self, frames):
        results = self.model.predict(list(frames), **self.kw)
        dets = [_ultra_dets(r) for r in results]
        # r.speed is per image (batch time / batch size), as in the benchmark
        tim = [_timing(r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]) for r in results]
        return dets, tim


class _PerStreamBackend(DetectorBackend):
    """One model instance per camera, run in parallel (the benchmark's
    CPU deployment: one worker per stream, each with its own instance).
    Instances are created as cameras are added (ensure_streams)."""

    def __init__(self, spec: ModelSpec, s: BackendSettings, log: logging.Logger):
        super().__init__(spec, s, log)
        if not spec.input_hw:
            raise RuntimeError(f"{spec.path}: export input size unknown — export_info.yaml needs `imgsz`")
        self.input_hw = tuple(spec.input_hw)
        self.instances: List[Any] = []
        self.pool: Optional[ThreadPoolExecutor] = None

    @property
    def imgsz_label(self):
        return f"{self.input_hw[0]}x{self.input_hw[1]}"

    def _new_instance(self, n_total: int):
        raise NotImplementedError

    def _run(self, inst, frame) -> Tuple[np.ndarray, Dict[str, float]]:
        raise NotImplementedError

    def _warm(self, inst):
        dummy = np.full((self.input_hw[0], self.input_hw[1], 3), 114, np.uint8)
        for _ in range(max(1, self.s.warmup_runs)):
            self._run(inst, dummy)

    def ensure_streams(self, n: int):
        n = max(1, int(n))
        if n <= len(self.instances):
            return
        t0 = time.perf_counter()
        while len(self.instances) < n:
            inst = self._new_instance(n)
            self._warm(inst)
            self.instances.append(inst)
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self.pool = ThreadPoolExecutor(max_workers=len(self.instances), thread_name_prefix=self.kind)
        self.log.info("[INIT] %s: %d instance(s) ready (one per camera) in %.0f ms",
                      self.label, len(self.instances), (time.perf_counter() - t0) * 1000)

    def warmup(self):
        self.ensure_streams(1)

    def _bucket(self, inst, items):
        return [(i, *self._run(inst, f)) for i, f in items]

    def infer(self, frames):
        self.ensure_streams(len(frames))
        n = len(self.instances)
        buckets: List[List] = [[] for _ in range(n)]
        for i, f in enumerate(frames):
            buckets[i % n].append((i, f))
        dets: List[np.ndarray] = [EMPTY_DETS] * len(frames)
        tim: List[Dict[str, float]] = [_timing(0, 0, 0)] * len(frames)
        if len(frames) == 1:
            res = [self._bucket(self.instances[0], buckets[0])]
        else:
            futs = [self.pool.submit(self._bucket, self.instances[k], b) for k, b in enumerate(buckets) if b]
            res = [f.result() for f in futs]
        for part in res:
            for i, d, t in part:
                dets[i], tim[i] = d, t
        return dets, tim

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False)


class OpenVINOUltralyticsBackend(_PerStreamBackend):
    """openvino_fp32 / openvino_int8: Ultralytics predict on intel:cpu,
    one YOLO instance per camera (+ the INT8 fix predictor for int8)."""

    def __init__(self, spec: ModelSpec, s: BackendSettings, log: logging.Logger):
        super().__init__(spec, s, log)
        self.kind = spec.variant
        self.predictor_cls = make_int8_fix_predictor(s.int8_fix, s.conf) if spec.variant == "openvino_int8" else None
        self.kw = dict(imgsz=list(self.input_hw), conf=s.conf, iou=s.nms_iou, max_det=s.max_det,
                       device=s.ov_device, half=False, verbose=False)
        log.info(f"[INIT] Ultralytics {os.path.basename(spec.path)} on {s.ov_device}"
                 f"{' + INT8 fix ' + str(s.int8_fix) if self.predictor_cls else ''} | input "
                 f"{self.input_hw[0]}x{self.input_hw[1]} | one YOLO instance per camera")

    def _new_instance(self, n_total: int):
        from ultralytics import YOLO
        return YOLO(self.spec.path, task="detect")

    def _warm(self, inst):
        super()._warm(inst)
        self._check_device(inst)

    def _check_device(self, inst):
        """EXECUTION_DEVICES must be ['CPU'] (OpenVINO AUTO moves to the
        iGPU: 2-3x slower + a stall at the switch). If not, recompile the
        backend's model on the named CPU device."""
        be = getattr(getattr(inst, "predictor", None), "model", None)
        holder = next((h for h in (be, getattr(be, "backend", None))
                       if h is not None and getattr(h, "ov_compiled_model", None) is not None), None)
        if holder is None:
            self.log.warning("[INIT] could not read OpenVINO EXECUTION_DEVICES from Ultralytics")
            return
        devs = list(holder.ov_compiled_model.get_property("EXECUTION_DEVICES"))
        if devs != ["CPU"]:
            self.log.error("[INIT] OpenVINO EXECUTION_DEVICES=%s (expected ['CPU']) — recompiling on CPU", devs)
            import glob
            from functools import partial
            import openvino as ov
            core = ov.Core()
            xml = self.spec.path if self.spec.path.endswith(".xml") else \
                sorted(glob.glob(os.path.join(self.spec.path, "*.xml")))[0]
            holder.compile_model = partial(core.compile_model, device_name="CPU",
                                           config={"PERFORMANCE_HINT": "LATENCY"})
            holder.ov_compiled_model = holder.compile_model(core.read_model(xml))
            devs = list(holder.ov_compiled_model.get_property("EXECUTION_DEVICES"))
            if devs != ["CPU"]:
                raise RuntimeError(f"OpenVINO still runs on {devs} after recompiling on CPU")
        if not self.instances:
            self.log.info("[INIT] OpenVINO EXECUTION_DEVICES=%s", devs)

    def _run(self, inst, frame):
        r = inst.predict(frame, predictor=self.predictor_cls, **self.kw)[0]
        sp = r.speed
        return _ultra_dets(r), _timing(sp["preprocess"], sp["inference"], sp["postprocess"])


class OnnxOwnBackend(_PerStreamBackend):
    """onnx: own ONNX Runtime session per camera, CPU threads split
    between the sessions, own letterbox + OpenCV NMS (through
    Ultralytics every session would grab all cores)."""
    kind = "onnx"

    def __init__(self, spec: ModelSpec, s: BackendSettings, log: logging.Logger):
        super().__init__(spec, s, log)
        import onnxruntime as ort
        self.ort = ort
        log.info(f"[INIT] own ONNX Runtime {ort.__version__} engine {os.path.basename(spec.path)} | input "
                 f"{self.input_hw[0]}x{self.input_hw[1]} | one session per camera, "
                 f"{s.onnx_cpu_threads or os.cpu_count()} CPU threads split between them")

    def _threads(self, n_total: int) -> int:
        return max(1, (self.s.onnx_cpu_threads or os.cpu_count() or 4) // max(1, n_total))

    def _new_instance(self, n_total: int):
        so = self.ort.SessionOptions()
        so.intra_op_num_threads = self._threads(n_total)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        sess = self.ort.InferenceSession(self.spec.path, so, providers=["CPUExecutionProvider"])
        if sess.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"ONNX Runtime providers={sess.get_providers()}, expected CPUExecutionProvider")
        return sess

    def ensure_streams(self, n: int):
        n = max(1, int(n))
        if n > len(self.instances) and self.instances:
            self.instances = []          # the thread split changes with the session count -> rebuild all
        super().ensure_streams(n)

    def _run(self, sess, frame):
        t0 = time.perf_counter()
        img, r, left, top = letterbox(frame, self.input_hw)
        x = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        t1 = time.perf_counter()
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        t2 = time.perf_counter()
        dets = onnx_postprocess(out, r, left, top, frame.shape[:2], self.s.conf, self.s.nms_iou, self.s.max_det)
        t3 = time.perf_counter()
        return dets, _timing((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)


# ====================================================================
# Factory
# ====================================================================
def resolve_spec(plan: RuntimePlan, s: BackendSettings) -> ModelSpec:
    return resolve_model(s.model_root, plan.model_name, plan.variant, s.manifest_keys,
                         s.file_patterns, s.class_labels)


def build_backend(plan: RuntimePlan, s: BackendSettings, log: logging.Logger) -> DetectorBackend:
    if plan.variant == "pt":
        spec = resolve_spec(plan, s)
        log.info("[INIT] model: %s", spec.describe())
        return UltralyticsBatchBackend(spec, s, plan.torch_device, log)
    try:
        spec = resolve_spec(plan, s)
        log.info("[INIT] model: %s", spec.describe())
        if plan.variant == "onnx":
            return OnnxOwnBackend(spec, s, log)
        return OpenVINOUltralyticsBackend(spec, s, log)
    except Exception as e:
        if not s.fallback_to_pt:
            log.error("[INIT] %s failed and BACKEND_FALLBACK_TO_PT is off: %s", plan.variant, e)
            raise
        log.error("[INIT] %s FAILED (%s: %s) — FALLING BACK to the .pt model with Ultralytics on CPU "
                  "(FP32, slower). Fix the model files / runtime and restart.",
                  plan.variant, type(e).__name__, e, exc_info=True)
        fb = RuntimePlan(CPU, "cpu", "pt", plan.model_name)
        spec = resolve_spec(fb, s)
        return UltralyticsBatchBackend(spec, s, "cpu", log)
