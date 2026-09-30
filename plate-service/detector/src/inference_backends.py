"""
inference_backends.py
--------------------------------------------------------------------
The one place the detector turns a batch of ROI frames into boxes.
Engine (engine.py) calls exactly two things on a backend:

    backend.warmup()
    detections, timing = backend.infer(frames)

and gets back, per frame, the same `np.float64` array it always built
from Ultralytics results:

    [[x1, y1, x2, y2, conf, cls], ...]     (ROI-frame pixel coordinates)

so the tracker, crops, triggers and control hub need no change,
whichever backend produced the boxes.

Backends
--------
  pt        Ultralytics YOLO(.pt).predict — THE GPU PIPELINE, unchanged
            (same YOLO(...).to(device), same batched predict(source=
            frames, imgsz, conf, device, half=False)). Also the CPU
            fallback when an ONNX/OpenVINO model fails to load.
  openvino  OpenVINO runtime used directly (never through Ultralytics),
            compiled on the named "CPU" device, THROUGHPUT hint, one
            infer request per camera started asynchronously; compiled
            once per camera-ROI input shape (shape_mode="roi").
  onnx      ONNX Runtime used directly, CPUExecutionProvider named
            explicitly, one session per concurrent worker.

Why ONNX/OpenVINO bypass Ultralytics (see README "CPU pipeline"):
  - Ultralytics 8.3.x compiles an OpenVINO model with device "AUTO"
    even for device="cpu"; AUTO migrates to the integrated GPU, which
    measured 2-3x slower plus a ~200 ms stall at the switch.
  - A patched Ultralytics install prints a [DEBUG] line per ONNX frame.
  - Batched synchronous predict() serialises cameras; parallel async
    requests let OpenVINO run all cameras' frames at once.

preprocess() / postprocess() below reproduce Ultralytics exactly
(letterbox with centred 114-padding + INTER_LINEAR, BGR->RGB, /255,
NCHW; class-aware NMS at IoU 0.7 with a 7680 px class offset, max 300
boxes, boxes mapped back through the letterbox and clipped). FP32
ONNX/OpenVINO then give the same detections as PyTorch.

Import discipline: torch/ultralytics/openvino/onnxruntime are imported
lazily inside the backend that needs them, so importing this module
from the parent process (main.py) loads none of them — each engine
child process compiles its own model (spawn-safe, Windows-safe).
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cpu_topology import physical_cores
from model_files import ModelSpec, parse_hw, parse_names, resolve_model

EMPTY_DETS = np.empty((0, 6), dtype=np.float64)

# Ultralytics non_max_suppression constants (ultralytics/utils/ops.py)
NMS_MAX_WH = 7680      # class offset for class-aware NMS
NMS_MAX_CANDIDATES = 30000


# ====================================================================
# Runtime plan: GPU or CPU, and which backend
# ====================================================================
@dataclass
class RuntimePlan:
    device_kind: str     # "gpu" | "cpu"
    torch_device: str    # what the pt backend passes to Ultralytics: "cuda", "cuda:0", "cpu"
    backend: str         # "pt" | "onnx" | "openvino"
    precision: str       # "fp32" | "int8"

    def describe(self) -> str:
        return (f"device={self.device_kind} backend={self.backend} precision={self.precision} "
                f"torch_device={self.torch_device if self.backend == 'pt' else '-'}")


def _cuda_available() -> bool:
    try:
        import torch  # only probed for DETECTION_DEVICE=auto/gpu
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def normalize_device(pref: str) -> str:
    p = (pref or "auto").strip().lower()
    if p in ("gpu", "cuda"):
        return "cuda"
    return p


def resolve_runtime(device_pref: str, backend_pref: str, precision_pref: str,
                    strict: bool, log: logging.Logger) -> RuntimePlan:
    """DETECTION_DEVICE (+ DETECTION_BACKEND / DETECTION_PRECISION) ->
    a concrete plan.

      gpu | cuda | cuda:N  GPU pipeline: PyTorch/Ultralytics, unchanged.
      cpu                  CPU pipeline: DETECTION_BACKEND (default openvino).
      auto                 GPU if torch sees CUDA, else CPU.

    torch is only probed for auto/gpu, so a CPU-only box never imports
    it here (the old resolve_device() returned "cuda" for auto on any
    CUDA-enabled torch; it must not feed the ONNX/OpenVINO path)."""
    pref = normalize_device(device_pref)
    backend = (backend_pref or "").strip().lower()
    precision = (precision_pref or "fp32").strip().lower()

    if pref == "auto":
        pref = "cuda" if _cuda_available() else "cpu"
        log.info("DETECTION_DEVICE=auto -> %s", "gpu" if pref == "cuda" else "cpu")

    if pref.startswith("cuda"):
        if not _cuda_available():
            msg = f"DETECTION_DEVICE={device_pref!r} requested but CUDA is not available"
            if strict:
                raise RuntimeError(msg)
            log.error("%s — falling back to the CPU pipeline", msg)
            pref = "cpu"
        else:
            if backend not in ("", "pt", "auto"):
                log.warning("DETECTION_BACKEND=%s only applies to DETECTION_DEVICE=cpu — the GPU "
                            "pipeline always runs the PyTorch (.pt) model", backend)
            if precision != "fp32":
                log.warning("DETECTION_PRECISION=%s ignored on GPU (PyTorch runs FP32)", precision)
            return RuntimePlan("gpu", pref, "pt", "fp32")

    if pref != "cpu":
        raise ValueError(f"DETECTION_DEVICE={device_pref!r} — use gpu, cpu, auto or cuda:N")

    if backend in ("", "auto"):
        backend = "openvino"
    if backend not in ("pt", "onnx", "openvino"):
        raise ValueError(f"DETECTION_BACKEND={backend_pref!r} — use openvino, onnx or pt")
    if backend != "openvino" and precision != "fp32":
        log.warning("DETECTION_PRECISION=%s only exists for the openvino backend — %s runs fp32",
                    precision, backend)
        precision = "fp32"
    if precision == "int8":
        log.warning("DETECTION_PRECISION=int8: benchmarked at ~90% precision / avg IoU 0.89 "
                    "against PyTorch, and it shifts confidences (~0.035). Its crops feed OCR. "
                    "Only use an INT8 model that passed the accuracy gate (README), and re-tune "
                    "TRACKER_TRACK_THRESH and the other confidence thresholds.")
    return RuntimePlan("cpu", "cpu", backend, precision)


# ====================================================================
# Pre/post-processing — must match Ultralytics exactly
# ====================================================================
def letterbox(img: np.ndarray, new_hw: Tuple[int, int], color: int = 114,
              r: Optional[float] = None) -> np.ndarray:
    """Ultralytics LetterBox(scaleup=True, center=True): keep aspect
    ratio, INTER_LINEAR resize, centred constant padding up to new_hw.
    `r` is the resize ratio; None = min(new_h/h, new_w/w) (auto=False)."""
    h0, w0 = img.shape[:2]
    new_h, new_w = int(new_hw[0]), int(new_hw[1])
    if r is None:
        r = min(new_h / h0, new_w / w0)
    new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (new_w - new_unpad_w) / 2.0, (new_h - new_unpad_h) / 2.0
    if (w0, h0) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                              value=(color, color, color))


def auto_shape(orig_hw: Tuple[int, int], imgsz: int, stride: int = 32) -> Tuple[Tuple[int, int], float]:
    """What Ultralytics predict() feeds a .pt model for a frame of
    orig_hw at `imgsz` (LetterBox auto=True): the ratio
    r = min(imgsz/h, imgsz/w) and the smallest stride-aligned rectangle
    around the resized frame. 1920x1080 @ 480 -> (288, 480);
    a 960x1080 ROI @ 480 -> (480, 448). Returns ((H, W), r)."""
    h0, w0 = orig_hw
    r = min(imgsz / h0, imgsz / w0)
    uw, uh = int(round(w0 * r)), int(round(h0 * r))
    return (uh + (imgsz - uh) % stride, uw + (imgsz - uw) % stride), r


def scale_params(input_hw: Tuple[int, int], orig_hw: Tuple[int, int]) -> Tuple[float, Tuple[int, int]]:
    """Ultralytics scale_boxes(): gain and (left, top) padding recomputed
    from the two shapes, used to map boxes back. Same formula as current
    Ultralytics (the resized size is rounded first, so the padding is
    exactly what letterbox() added; some 8.3.x releases skipped that
    rounding and were 1 px off on odd-sized ROIs — full 16:9 frames are
    identical either way)."""
    gain = min(input_hw[0] / orig_hw[0], input_hw[1] / orig_hw[1])
    pad_x = round((input_hw[1] - round(orig_hw[1] * gain)) / 2 - 0.1)
    pad_y = round((input_hw[0] - round(orig_hw[0] * gain)) / 2 - 0.1)
    return gain, (int(pad_x), int(pad_y))


def preprocess(img_bgr: np.ndarray, input_hw: Tuple[int, int], r: Optional[float] = None
               ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """BGR uint8 frame -> ((1, 3, H, W) float32 RGB/255 contiguous blob,
    gain, (left, top)) — gain/pad are what postprocess() needs."""
    lb = letterbox(img_bgr, input_hw, r=r)
    blob = cv2.dnn.blobFromImage(lb, scalefactor=1.0 / 255.0, swapRB=True)
    gain, pad = scale_params(input_hw, img_bgr.shape[:2])
    return np.ascontiguousarray(blob, dtype=np.float32), gain, pad


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS with torchvision.ops.nms semantics (areas without +1,
    suppress IoU > thres, highest score first). Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def postprocess(pred: np.ndarray, gain: float, pad: Tuple[int, int], orig_hw: Tuple[int, int],
                conf_thres: float, iou_thres: float = 0.7, max_det: int = 300,
                agnostic: bool = False) -> np.ndarray:
    """Raw YOLOv8 output (1, 4+nc, N) -> [[x1,y1,x2,y2,conf,cls]] float64
    in the ORIGINAL frame's pixels (= Ultralytics non_max_suppression +
    scale_boxes). Rows 0-3 are cx,cy,w,h in input pixels, the rest are
    already-sigmoided class scores."""
    p = pred[0] if pred.ndim == 3 else pred
    p = p.T                                            # (N, 4+nc)
    if p.shape[0] == 0 or p.shape[1] <= 4:
        return EMPTY_DETS.copy()
    scores = p[:, 4:]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(scores.shape[0]), cls]
    m = conf > conf_thres
    if not np.any(m):
        return EMPTY_DETS.copy()
    xywh, conf, cls = p[m, :4], conf[m], cls[m]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    if boxes.shape[0] > NMS_MAX_CANDIDATES:
        top = np.argsort(-conf, kind="stable")[:NMS_MAX_CANDIDATES]
        boxes, conf, cls = boxes[top], conf[top], cls[top]
    offset = (0.0 if agnostic else float(NMS_MAX_WH)) * cls.astype(boxes.dtype)
    keep = _nms(boxes + offset[:, None], conf, iou_thres)[:max_det]
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    left, top = pad
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= gain
    h0, w0 = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    return np.column_stack([boxes, conf, cls.astype(boxes.dtype)]).astype(np.float64)


# ====================================================================
# Backends
# ====================================================================
class DetectorBackend:
    kind = "base"

    def __init__(self, spec: ModelSpec, conf: float, log: logging.Logger):
        self.spec = spec
        self.conf = float(conf)
        self.log = log
        self.device = "cpu"
        self.precision = spec.precision

    @property
    def label(self) -> str:
        """Short description for logs / the debug video overlay."""
        return f"{self.device}/{self.kind}-{self.precision}"

    @property
    def imgsz_label(self):
        return self.spec.imgsz

    def warmup(self):
        pass

    def infer(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, float]]:
        raise NotImplementedError

    def close(self):
        pass


class UltralyticsBackend(DetectorBackend):
    """The original engine.py inference, moved here verbatim. This is
    the GPU pipeline — nothing about it changed."""
    kind = "pt"

    def __init__(self, spec: ModelSpec, conf: float, device: str, log: logging.Logger):
        super().__init__(spec, conf, log)
        from ultralytics import YOLO
        self.device = device
        self.imgsz = int(spec.imgsz)
        self.model = YOLO(spec.path).to(self.device)
        log.info(f"[INIT] YOLO model loaded model={spec.path} device={self.device} "
                 f"imgsz={self.imgsz} conf={self.conf}")

    @property
    def label(self) -> str:
        return f"{self.device}/pt"

    def warmup(self):
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(source=[dummy], imgsz=self.imgsz, conf=self.conf,
                                   device=self.device, verbose=False)
        except Exception:
            pass

    def infer(self, frames):
        results = self.model.predict(
            source=frames, imgsz=self.imgsz, conf=self.conf, device=self.device,
            verbose=False, half=False,
        )
        out = []
        for res in results:
            detections = np.empty((0, 6), dtype=np.float64)
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                detections = np.column_stack([boxes, confs, clss]).astype(np.float64)
            out.append(detections)
        timing = dict(results[0].speed) if results else {}
        return out, timing


class _StaticShapeBackend(DetectorBackend):
    """Shared by the OpenVINO and ONNX Runtime backends."""

    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int,
                 warmup_runs: int, log: logging.Logger):
        super().__init__(spec, conf, log)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.warmup_runs = max(0, int(warmup_runs))
        self.input_hw: Tuple[int, int] = (spec.imgsz, spec.imgsz)   # the model's native input
        self.names: Dict[int, str] = dict(spec.names)
        self.concurrency = 1

    @property
    def imgsz_label(self):
        return f"{self.input_hw[0]}x{self.input_hw[1]}"

    def _check_output(self, nc: Optional[int]):
        if nc is None:
            return
        if nc <= 0:
            raise RuntimeError(f"model output has {nc} classes — is this a YOLOv8 detect model "
                               f"exported WITHOUT NMS (output (1, 4+nc, N))?")
        if self.names and len(self.names) != nc:
            self.log.warning("model has %d classes but names=%s (%d) — check metadata.yaml / "
                             "export_info.yaml", nc, self.names, len(self.names))

    @staticmethod
    def _hw_from_graph(dims) -> Optional[Tuple[int, int]]:
        """(N, C, H, W) with ints (static) or None/str/-1 (dynamic)."""
        try:
            h, w = dims[2], dims[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(h), int(w)
        except Exception:
            pass
        return None

    def _pick_hw(self, graph_hw: Optional[Tuple[int, int]], can_reshape: bool) -> Tuple[int, int]:
        """The model's native (h, w): the static graph shape, unless
        DETECTION_CPU_INPUT_SIZE asks for another one (reshape)."""
        hint = self.spec.input_hw
        from_env = self.spec.sources.get("input_hw") == "DETECTION_CPU_INPUT_SIZE"
        if graph_hw:
            if hint and tuple(hint) != tuple(graph_hw):
                if from_env and can_reshape:
                    self.log.warning("DETECTION_CPU_INPUT_SIZE=%dx%d differs from the model's static "
                                     "%dx%d — reshaping (YOLO is fully convolutional; both must be "
                                     "multiples of 32)", hint[0], hint[1], graph_hw[0], graph_hw[1])
                    return int(hint[0]), int(hint[1])
                if from_env:
                    raise RuntimeError(f"DETECTION_CPU_INPUT_SIZE={hint[0]}x{hint[1]} but the {self.kind} "
                                       f"model is static {graph_hw[0]}x{graph_hw[1]} and cannot be "
                                       f"reshaped — re-export it or clear DETECTION_CPU_INPUT_SIZE")
                self.log.warning("%s says input %s but the model graph is static %s — using the graph",
                                 self.spec.sources.get("input_hw"), hint, graph_hw)
            return graph_hw
        hw = tuple(hint) if hint else (self.spec.imgsz, self.spec.imgsz)
        self.log.info("model input is dynamic — native shape %dx%d (%s)", hw[0], hw[1],
                      self.spec.sources.get("input_hw", "square imgsz"))
        return int(hw[0]), int(hw[1])

    def warmup(self):
        """A few inferences per request/worker at the REAL input shape
        (the old warm-up used a square dummy image, which did not match)."""
        if self.warmup_runs <= 0:
            return
        h, w = self.input_hw
        dummy = np.full((h, w, 3), 114, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(self.warmup_runs):
            self.infer([dummy] * self.concurrency)
        self.log.info("[INIT] warm-up: %d x %d parallel inferences at %dx%d in %.0f ms",
                      self.warmup_runs, self.concurrency, h, w, (time.perf_counter() - t0) * 1000)


class OpenVINOBackend(_StaticShapeBackend):
    """OpenVINO runtime on the named CPU device.

    shape_mode="roi" (default): every frame is letterboxed exactly like
    Ultralytics does for the .pt model — ratio min(imgsz/h, imgsz/w)
    into the smallest 32-aligned rectangle (LetterBox auto=True). A
    full 16:9 frame at 480 is 288x480 (= the exported shape); a camera
    whose ROI has another aspect ratio gets its own shape, and the
    model is reshaped and compiled ONCE for it (YOLO is fully
    convolutional, one file serves every multiple-of-32 size). Same
    input as the GPU/.pt path for every ROI, so detections match.
    shape_mode="fixed": every frame is letterboxed into the model's one
    static shape (padding absorbs the aspect ratio; cheaper for odd
    ROIs, but a portrait ROI gets far fewer pixels than on the GPU)."""
    kind = "openvino"

    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int, streams: int,
                 threads: int, cache_dir: str, core_type: str, warmup_runs: int,
                 log: logging.Logger, shape_mode: str = "roi"):
        super().__init__(spec, conf, iou, max_det, warmup_runs, log)
        import openvino as ov
        self._ov_version = getattr(ov, "__version__", "?")

        self._core = ov.Core()
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                self._core.set_property({"CACHE_DIR": cache_dir})
            except Exception as e:
                log.warning("OpenVINO CACHE_DIR=%s not usable (%s) — compiling without cache", cache_dir, e)

        model = self._core.read_model(spec.path)
        if len(model.inputs) != 1:
            raise RuntimeError(f"expected 1 model input, found {len(model.inputs)}")
        ps = model.inputs[0].get_partial_shape()
        dims = [d.get_length() if d.is_static else None for d in ps]
        self.input_hw = self._pick_hw(self._hw_from_graph(dims), can_reshape=True)
        if not self.names:
            self.names = self._names_from_rt_info(model)

        mode = (shape_mode or "roi").strip().lower()
        if self.spec.sources.get("input_hw") == "DETECTION_CPU_INPUT_SIZE":
            mode = "fixed"          # an explicit size means "always this shape"
        if mode not in ("roi", "fixed"):
            log.warning("DETECTION_CPU_SHAPE_MODE=%r unknown — using roi", shape_mode)
            mode = "roi"
        self.shape_mode = mode
        # the square size the .pt path would letterbox to (480 / 640)
        self.letterbox_size = int(spec.imgsz) if spec.imgsz else max(self.input_hw)

        self.streams = max(1, int(streams))
        self._cfg = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": str(self.streams),
            # Pin FP32 math: on CPUs with bf16/AMX support OpenVINO would
            # otherwise silently lower an FP32 model to bf16. FP16 gave no
            # speed-up on the benchmark CPU and is never used.
            "INFERENCE_PRECISION_HINT": "f32",
        }
        if threads and int(threads) > 0:
            self._cfg["INFERENCE_NUM_THREADS"] = str(int(threads))
        if core_type:
            self._cfg["SCHEDULING_CORE_TYPE"] = core_type.strip().upper()

        # input shape -> (compiled model, [infer requests])
        self._pools: Dict[Tuple[int, int], Tuple[object, list]] = {}
        self._pool_for(self.input_hw, model=model, first=True)
        self.concurrency = self.streams
        log.info("[INIT] OpenVINO backend ready model=%s precision=%s native_input=%dx%d "
                 "shape_mode=%s letterbox_size=%d names=%s requests_per_shape=%d conf=%.2f iou=%.2f",
                 spec.path, spec.precision, self.input_hw[0], self.input_hw[1], self.shape_mode,
                 self.letterbox_size, self.names, self.streams, self.conf, self.iou)

    @staticmethod
    def _names_from_rt_info(model) -> Dict[int, str]:
        try:
            return parse_names(model.get_rt_info(["model_info", "names"]).astype(str))
        except Exception:
            return {}

    def _pool_for(self, hw: Tuple[int, int], model=None, first: bool = False):
        pool = self._pools.get(hw)
        if pool is not None:
            return pool
        t0 = time.perf_counter()
        if model is None:
            model = self._core.read_model(self.spec.path)
        cur = [d.get_length() if d.is_static else None for d in model.inputs[0].get_partial_shape()]
        if cur != [1, 3, hw[0], hw[1]]:
            try:
                model.reshape([1, 3, hw[0], hw[1]])
            except Exception as e:
                if first:
                    raise RuntimeError(f"cannot reshape {self.spec.path} to {hw[0]}x{hw[1]}: {e}") from e
                # A STATIC Ultralytics export bakes its anchors/reshape
                # constants for one size — it only runs at that size.
                self.shape_mode = "fixed"
                self.log.warning("[SHAPE] %s is a static export and cannot be reshaped to %dx%d — "
                                 "switching to DETECTION_CPU_SHAPE_MODE=fixed (every ROI letterboxed "
                                 "into %dx%d). For per-ROI shapes, convert the OpenVINO model from a "
                                 "dynamic ONNX export (tools/export_cpu_models.py does this).",
                                 self.spec.path, hw[0], hw[1], self.input_hw[0], self.input_hw[1])
                return self._pools[self.input_hw]
        # Named "CPU" device — never AUTO (AUTO hops to the iGPU).
        compiled = self._core.compile_model(model, "CPU", self._cfg)
        exec_devices = list(compiled.get_property("EXECUTION_DEVICES"))
        if exec_devices != ["CPU"]:
            raise RuntimeError(f"OpenVINO EXECUTION_DEVICES={exec_devices}, expected ['CPU']")

        def _prop(name):
            try:
                return compiled.get_property(name)
            except Exception:
                return "?"

        if first:
            self.log.info("[INIT] OpenVINO %s compiled: EXECUTION_DEVICES=%s NUM_STREAMS=%s "
                          "INFERENCE_NUM_THREADS=%s INFERENCE_PRECISION_HINT=%s "
                          "OPTIMAL_NUMBER_OF_INFER_REQUESTS=%s SCHEDULING_CORE_TYPE=%s",
                          self._ov_version, exec_devices, _prop("NUM_STREAMS"),
                          _prop("INFERENCE_NUM_THREADS"), _prop("INFERENCE_PRECISION_HINT"),
                          _prop("OPTIMAL_NUMBER_OF_INFER_REQUESTS"), _prop("SCHEDULING_CORE_TYPE"))
            out_shape = compiled.outputs[0].get_partial_shape()
            nc = None
            if out_shape.rank.get_length() == 3 and out_shape[1].is_static:
                nc = out_shape[1].get_length() - 4
            self._check_output(nc)
        # One request per concurrent inference: a request is not
        # thread-safe and its output buffer is overwritten by the next
        # call, so outputs are copied before the request is reused.
        requests = [compiled.create_infer_request() for _ in range(self.streams)]
        self._pools[hw] = (compiled, requests)
        if not first:
            self.log.info("[SHAPE] compiled OpenVINO model for input %dx%d in %.0f ms "
                          "(EXECUTION_DEVICES=%s) — %d input shape(s) now",
                          hw[0], hw[1], (time.perf_counter() - t0) * 1000.0, exec_devices, len(self._pools))
        return self._pools[hw]

    def _plan(self, frame: np.ndarray) -> Tuple[Tuple[int, int], Optional[float]]:
        if self.shape_mode == "fixed":
            return self.input_hw, None
        hw, r = auto_shape(frame.shape[:2], self.letterbox_size)
        return hw, r

    def infer(self, frames):
        n = len(frames)
        out: List[np.ndarray] = [EMPTY_DETS] * n
        pre_ms = fwd_ms = post_ms = 0.0
        for start in range(0, n, self.streams):
            chunk = frames[start:start + self.streams]
            used: Dict[Tuple[int, int], int] = {}
            in_flight = []
            t_chunk = time.perf_counter()
            for i, frame in enumerate(chunk):
                hw, r = self._plan(frame)
                _, requests = self._pool_for(hw)
                if self.shape_mode == "fixed" and hw != self.input_hw:
                    # the model could not be reshaped -> native shape/pool
                    hw, r = self.input_hw, None
                    _, requests = self._pools[hw]
                k = used.get(hw, 0)                   # never the same request twice per chunk
                used[hw] = k + 1
                req = requests[k]
                t0 = time.perf_counter()
                blob, gain, pad = preprocess(frame, hw, r)
                pre_ms += (time.perf_counter() - t0) * 1000.0
                # start right away so frame i infers while frame i+1 is preprocessed
                req.start_async({0: blob})
                in_flight.append((i, req, gain, pad, frame.shape[:2]))
            raws = []
            for i, req, gain, pad, hw0 in in_flight:
                req.wait()
                raws.append((i, req.get_output_tensor(0).data.copy(), gain, pad, hw0))
            fwd_ms += (time.perf_counter() - t_chunk) * 1000.0
            t0 = time.perf_counter()
            for i, raw, gain, pad, hw0 in raws:
                out[start + i] = postprocess(raw, gain, pad, hw0, self.conf, self.iou, self.max_det)
            post_ms += (time.perf_counter() - t0) * 1000.0
        fwd_ms = max(0.0, fwd_ms - pre_ms)
        return out, {"preprocess": pre_ms, "inference": fwd_ms, "postprocess": post_ms}


class OnnxRuntimeBackend(_StaticShapeBackend):
    """ONNX Runtime on CPUExecutionProvider. The exported ONNX model is
    static, so every frame is letterboxed into its one shape (a ROI
    with another aspect ratio gets fewer pixels than on the .pt path —
    logged once per ROI shape; the openvino backend avoids this)."""
    kind = "onnx"

    def __init__(self, spec: ModelSpec, conf: float, iou: float, max_det: int, workers: int,
                 threads: int, warmup_runs: int, log: logging.Logger):
        super().__init__(spec, conf, iou, max_det, warmup_runs, log)
        import onnxruntime as ort

        self.workers = max(1, int(workers))
        # sessions don't coordinate threads with each other — size them by hand
        intra = int(threads) if threads and int(threads) > 0 else max(1, physical_cores() // self.workers)
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]  # always explicit: installed EPs can change the default
        self._sessions = [ort.InferenceSession(spec.path, sess_options=so, providers=providers)
                          for _ in range(self.workers)]
        used = self._sessions[0].get_providers()
        if used != providers:
            raise RuntimeError(f"ONNX Runtime providers={used}, expected {providers}")

        inp = self._sessions[0].get_inputs()[0]
        self._input_name = inp.name
        meta = {}
        try:
            meta = dict(self._sessions[0].get_modelmeta().custom_metadata_map or {})
        except Exception:
            pass
        if self.spec.input_hw is None and meta.get("imgsz"):
            self.spec.input_hw = parse_hw(meta["imgsz"])
            self.spec.sources["input_hw"] = "onnx metadata"
        if not self.names and meta.get("names"):
            self.names = parse_names(meta["names"])
        graph_hw = self._hw_from_graph(list(inp.shape))
        self.input_hw = self._pick_hw(graph_hw, can_reshape=False)
        self._dynamic = graph_hw is None
        self.letterbox_size = int(spec.imgsz) if spec.imgsz else max(self.input_hw)
        self._warned_shapes = set()

        out = self._sessions[0].get_outputs()[0]
        nc = (out.shape[1] - 4) if len(out.shape) == 3 and isinstance(out.shape[1], int) else None
        self._check_output(nc)

        self.concurrency = self.workers
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="ort")
        log.info("[INIT] ONNX Runtime %s backend ready model=%s providers=%s sessions=%d "
                 "intra_op_threads=%d inter_op_threads=1 input=%dx%d%s names=%s conf=%.2f iou=%.2f",
                 ort.__version__, spec.path, used, self.workers, intra, self.input_hw[0],
                 self.input_hw[1], " (dynamic: per-ROI shapes)" if self._dynamic else " (static)",
                 self.names, self.conf, self.iou)

    def _plan(self, frame: np.ndarray) -> Tuple[Tuple[int, int], Optional[float]]:
        if self._dynamic:
            return auto_shape(frame.shape[:2], self.letterbox_size)
        shp = frame.shape[:2]
        if shp not in self._warned_shapes:
            self._warned_shapes.add(shp)
            _, r_pt = auto_shape(shp, self.letterbox_size)
            r_fixed = min(self.input_hw[0] / shp[0], self.input_hw[1] / shp[1])
            if r_fixed < 0.9 * r_pt:
                self.log.warning("[SHAPE] frames of %dx%d are letterboxed into the static %dx%d ONNX "
                                 "input at scale %.3f (the .pt/GPU path uses %.3f) — expect fewer/"
                                 "different small detections. Use DETECTION_BACKEND=openvino "
                                 "(per-ROI shapes) or export the ONNX at this ROI's aspect ratio.",
                                 shp[1], shp[0], self.input_hw[0], self.input_hw[1], r_fixed, r_pt)
        return self.input_hw, None

    def _run_worker(self, wi: int, items):
        sess = self._sessions[wi]
        res, pre, fwd, post = [], 0.0, 0.0, 0.0
        for idx, frame in items:
            t0 = time.perf_counter()
            hw, r = self._plan(frame)
            blob, gain, pad = preprocess(frame, hw, r)
            t1 = time.perf_counter()
            raw = sess.run(None, {self._input_name: blob})[0]
            t2 = time.perf_counter()
            dets = postprocess(raw, gain, pad, frame.shape[:2], self.conf, self.iou, self.max_det)
            t3 = time.perf_counter()
            res.append((idx, dets))
            pre += (t1 - t0) * 1000.0
            fwd += (t2 - t1) * 1000.0
            post += (t3 - t2) * 1000.0
        return res, pre, fwd, post

    def infer(self, frames):
        n = len(frames)
        out: List[np.ndarray] = [EMPTY_DETS] * n
        buckets = [[] for _ in range(self.workers)]
        for i, f in enumerate(frames):
            buckets[i % self.workers].append((i, f))
        futures = [self._pool.submit(self._run_worker, wi, b) for wi, b in enumerate(buckets) if b]
        pre = fwd = post = 0.0
        for fut in futures:
            res, p, f_, q = fut.result()
            pre, fwd, post = pre + p, fwd + f_, post + q
            for idx, dets in res:
                out[idx] = dets
        return out, {"preprocess": pre, "inference": fwd, "postprocess": post}

    def close(self):
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass


# ====================================================================
# Factory
# ====================================================================
@dataclass
class BackendSettings:
    model_root: str
    model_name: str
    model_path_override: str = ""
    imgsz_override: int = 0
    cpu_input_hw: Optional[Tuple[int, int]] = None
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    cpu_streams: int = 1
    cpu_threads: int = 0
    openvino_cache_dir: str = ""
    openvino_core_type: str = ""
    warmup_runs: int = 3
    shape_mode: str = "roi"
    fallback_to_pt: bool = True
    class_labels: Optional[Dict[int, str]] = None


def _build_pt(plan: RuntimePlan, s: BackendSettings, log: logging.Logger) -> UltralyticsBackend:
    spec = resolve_model(s.model_root, s.model_name, "pt", "fp32", s.model_path_override,
                         s.imgsz_override, None, s.class_labels)
    return UltralyticsBackend(spec, s.conf, plan.torch_device, log)


def model_path_for(backend: str, override: str) -> str:
    """DETECTION_MODEL_PATH only applies when it points at this
    backend's format — older .env files carry a .pt path there, which
    must not be handed to OpenVINO / ONNX Runtime."""
    want_ext = {"onnx": ".onnx", "openvino": ".xml"}.get(backend)
    if override and want_ext and not override.lower().endswith(want_ext):
        return ""
    return override or ""


def build_backend(plan: RuntimePlan, s: BackendSettings, log: logging.Logger) -> DetectorBackend:
    """Builds the planned backend. An ONNX/OpenVINO failure falls back
    to the PyTorch model on CPU with an ERROR log (never silently), or
    re-raises when DETECTION_BACKEND_FALLBACK=false."""
    if plan.backend == "pt":
        return _build_pt(plan, s, log)
    try:
        override = model_path_for(plan.backend, s.model_path_override)
        if s.model_path_override and not override:
            log.info("DETECTION_MODEL_PATH=%s is not a %s model — ignored for backend=%s",
                     s.model_path_override, plan.backend, plan.backend)
        spec = resolve_model(s.model_root, s.model_name, plan.backend, plan.precision, override,
                             s.imgsz_override, s.cpu_input_hw, s.class_labels)
        log.info("[INIT] model files: %s", spec.describe())
        if plan.backend == "openvino":
            return OpenVINOBackend(spec, s.conf, s.iou, s.max_det, s.cpu_streams, s.cpu_threads,
                                   s.openvino_cache_dir, s.openvino_core_type, s.warmup_runs, log,
                                   shape_mode=s.shape_mode)
        return OnnxRuntimeBackend(spec, s.conf, s.iou, s.max_det, s.cpu_streams, s.cpu_threads,
                                  s.warmup_runs, log)
    except Exception as e:
        if not s.fallback_to_pt:
            log.error("[INIT] %s backend failed and DETECTION_BACKEND_FALLBACK=false: %s", plan.backend, e)
            raise
        log.error("[INIT] %s backend FAILED (%s: %s) — FALLING BACK to PyTorch/Ultralytics on CPU. "
                  "Detection still works but is ~2x slower; fix the model files / runtime and "
                  "restart.", plan.backend, type(e).__name__, e, exc_info=True)
        fb = RuntimePlan("cpu", "cpu", "pt", "fp32")
        return _build_pt(fb, s, log)
