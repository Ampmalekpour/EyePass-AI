"""
model_files.py
--------------------------------------------------------------------
Turns the .env model choice into concrete files on disk.

    DETECTION_MODEL=plate_v8n_480      (or plate_v8s_640, ...)
    DETECTION_BACKEND=pt | onnx | openvino
    DETECTION_PRECISION=fp32 | int8

The models folder (bind-mounted at /models, see compose.yaml) is
expected to look like this — one base name per model, every exported
variant named after it:

    /models/
    ├── plate_v8n_480.pt                         PyTorch   (GPU, and CPU fallback)
    ├── plate_v8n_480_288x480.onnx               ONNX FP32 (static 288x480 input)
    ├── plate_v8n_480_fp32_openvino_model/       OpenVINO FP32
    │   ├── plate_v8n_480.xml
    │   ├── plate_v8n_480.bin
    │   └── metadata.yaml
    ├── plate_v8n_480_int8_openvino_model/       OpenVINO INT8
    │   └── ...
    ├── export_info.yaml                         input size + class names
    ├── plate_v8s_640.pt
    └── ...

Nothing here imports torch, ultralytics, openvino or onnxruntime: it
is plain filesystem + YAML work, safe to call from the parent process
(main.py validates the files at startup, before any engine exists).

Input size and class names are READ, never hard-coded (roadmap §3):
  - PyTorch (Ultralytics) path: `imgsz` is one square number. It comes
    from DETECTION_IMG_SIZE when that is set (> 0), otherwise from the
    trailing number of the model name (plate_v8n_480 -> 480,
    plate_v8s_640 -> 640). Ultralytics letterboxes to that size itself.
  - ONNX / OpenVINO path: a static rectangular (h, w), e.g. 288x480.
    The graph's own static input shape is authoritative (the inference
    backend reads it after loading); what this module finds in
    metadata.yaml / export_info.yaml / the file name is used for
    dynamic-shape models and cross-checked against the graph.
--------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import glob
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("model_files")

BACKENDS = ("pt", "onnx", "openvino")
PRECISIONS = ("fp32", "int8")

# keys an export_info.yaml / metadata.yaml may use for the input size
_HW_KEYS = ("input_hw", "input_size", "input_shape", "imgsz", "img_size", "image_size", "shape")
_NAMES_KEYS = ("names", "class_names", "classes")


@dataclass
class ModelSpec:
    name: str                      # base name, e.g. "plate_v8n_480"
    backend: str                   # pt | onnx | openvino
    precision: str                 # fp32 | int8
    path: str                      # .pt / .onnx / .xml
    imgsz: int                     # square size for the Ultralytics path
    input_hw: Optional[Tuple[int, int]] = None   # (h, w) hint for onnx/openvino
    names: Dict[int, str] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)  # where each value came from

    def describe(self) -> str:
        hw = f"{self.input_hw[0]}x{self.input_hw[1]}" if self.input_hw else "-"
        return (f"model={self.name} backend={self.backend} precision={self.precision} "
                f"path={self.path} imgsz={self.imgsz} input_hw={hw} names={self.names} "
                f"sources={self.sources}")


# --------------------------------------------------------------------
# small parsers
# --------------------------------------------------------------------
def nominal_imgsz(name: str) -> Optional[int]:
    """plate_v8n_480 -> 480, plate_v8s_640.pt -> 640, best -> None."""
    base = os.path.basename(str(name or ""))
    base = re.sub(r"\.(pt|onnx|xml)$", "", base)
    m = re.search(r"_(\d{3,4})$", base)
    return int(m.group(1)) if m else None


def parse_hw(value: Any) -> Optional[Tuple[int, int]]:
    """Anything that plausibly describes an input size -> (h, w).

    480 -> (480, 480); "288x480" -> (288, 480); [288, 480] -> (288, 480);
    [1, 3, 288, 480] -> (288, 480); {"h": 288, "w": 480} -> (288, 480)."""
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            v = int(value)
            return (v, v) if v > 0 else None
        if isinstance(value, str):
            s = value.strip().lower()
            m = re.fullmatch(r"(\d+)\s*[x,*× ]\s*(\d+)", s)
            if m:
                return int(m.group(1)), int(m.group(2))
            if s.isdigit():
                v = int(s)
                return (v, v) if v > 0 else None
            if s.startswith("[") or s.startswith("("):
                return parse_hw(ast.literal_eval(s))
            return None
        if isinstance(value, dict):
            lower = {str(k).lower(): v for k, v in value.items()}
            for hk, wk in (("h", "w"), ("height", "width")):
                if hk in lower and wk in lower:
                    return int(lower[hk]), int(lower[wk])
            return None
        if isinstance(value, (list, tuple)):
            vals = [int(v) for v in value]
            if len(vals) == 1:
                return (vals[0], vals[0]) if vals[0] > 0 else None
            if len(vals) == 2:
                return vals[0], vals[1]
            if len(vals) == 4:          # NCHW
                return vals[2], vals[3]
            return None
    except (TypeError, ValueError, SyntaxError):
        return None
    return None


def parse_names(value: Any) -> Dict[int, str]:
    """{0: 'car_plate', 1: 'motorcycle_plate'} / ['car_plate', ...] /
    the string form Ultralytics writes into ONNX metadata -> dict."""
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, dict):
            return {int(k): str(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return {i: str(v) for i, v in enumerate(value)}
    except (TypeError, ValueError, SyntaxError):
        pass
    return {}


def _load_yaml(path: str) -> Optional[Any]:
    if not path or not os.path.isfile(path):
        return None
    try:
        import yaml  # PyYAML ships with Ultralytics; also in detector/requirements.txt
    except ImportError:
        logger.warning("PyYAML is not installed — cannot read %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning("could not parse %s: %s", path, e)
        return None


def _find_key(obj: Any, keys, depth: int = 3):
    """First value under any of `keys` in a nested dict (breadth-first)."""
    level = [obj]
    for _ in range(depth + 1):
        nxt = []
        for node in level:
            if isinstance(node, dict):
                lower = {str(k).lower(): v for k, v in node.items()}
                for k in keys:
                    if k in lower and lower[k] is not None:
                        return lower[k]
                nxt.extend(v for v in node.values() if isinstance(v, dict))
        level = nxt
        if not level:
            break
    return None


def read_metadata_file(path: str) -> Tuple[Optional[Tuple[int, int]], Dict[int, str]]:
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return None, {}
    return parse_hw(_find_key(data, _HW_KEYS)), parse_names(_find_key(data, _NAMES_KEYS))


def _export_info_candidates(root: str, name: str) -> List[str]:
    return [os.path.join(root, f"{name}_export_info.yaml"),
            os.path.join(root, name, "export_info.yaml"),
            os.path.join(root, "export_info.yaml")]


def _export_info_for(root: str, name: str) -> Tuple[Optional[Tuple[int, int]], Dict[int, str], Optional[str]]:
    """export_info.yaml for THIS model. A shared, unprefixed
    export_info.yaml is only trusted when its input size matches the
    model's nominal size (plate_v8n_480 -> longest side 480), so the
    480 model's file is never applied to plate_v8s_640."""
    nominal = nominal_imgsz(name)
    for p in _export_info_candidates(root, name):
        if not os.path.isfile(p):
            continue
        hw, names = read_metadata_file(p)
        shared = os.path.basename(p) == "export_info.yaml" and os.path.dirname(p) == root
        if shared and nominal and hw and max(hw) != nominal:
            logger.info("ignoring %s for %s (its input %s does not match nominal size %d)",
                        p, name, hw, nominal)
            continue
        return hw, names, p
    return None, {}, None


# --------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------
def _pick_onnx(root: str, name: str) -> Optional[str]:
    exact = os.path.join(root, f"{name}.onnx")
    if os.path.isfile(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(root, f"{name}_*.onnx")))
    # plate_v8n_480_288x480.onnx is the expected pattern; prefer it over
    # anything else that happens to share the prefix
    shaped = [h for h in hits if re.search(r"_\d+x\d+\.onnx$", h)]
    return (shaped or hits or [None])[0]


def _pick_openvino_xml(root: str, name: str, precision: str) -> Optional[str]:
    folder = os.path.join(root, f"{name}_{precision}_openvino_model")
    if not os.path.isdir(folder):
        return None
    exact = os.path.join(folder, f"{name}.xml")
    if os.path.isfile(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(folder, "*.xml")))
    return hits[0] if hits else None


def resolve_model(
        model_root: str,
        model_name: str,
        backend: str,
        precision: str = "fp32",
        model_path_override: str = "",
        imgsz_override: int = 0,
        input_hw_override: Optional[Tuple[int, int]] = None,
        default_names: Optional[Dict[int, str]] = None,
) -> ModelSpec:
    """Raises FileNotFoundError / ValueError with an actionable message."""
    backend = (backend or "pt").strip().lower()
    precision = (precision or "fp32").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"DETECTION_BACKEND={backend!r} — must be one of {BACKENDS}")
    if precision not in PRECISIONS:
        raise ValueError(f"DETECTION_PRECISION={precision!r} — must be one of {PRECISIONS}")
    if backend != "openvino" and precision != "fp32":
        raise ValueError(f"DETECTION_PRECISION={precision} is only available for the openvino "
                         f"backend (backend={backend} runs FP32)")

    name = (model_name or "").strip()
    override = (model_path_override or "").strip()
    if not name and override:
        name = re.sub(r"\.(pt|onnx|xml)$", "", os.path.basename(override))
    if not name:
        raise ValueError("DETECTION_MODEL is empty — set it to e.g. plate_v8n_480 or plate_v8s_640")

    sources: Dict[str, str] = {}

    # ---- file ----------------------------------------------------------
    if override:
        path = override
        sources["path"] = "DETECTION_MODEL_PATH"
    elif backend == "pt":
        path = os.path.join(model_root, f"{name}.pt")
        sources["path"] = "DETECTION_MODEL"
    elif backend == "onnx":
        path = _pick_onnx(model_root, name) or os.path.join(model_root, f"{name}_<H>x<W>.onnx")
        sources["path"] = "DETECTION_MODEL"
    else:
        path = (_pick_openvino_xml(model_root, name, precision)
                or os.path.join(model_root, f"{name}_{precision}_openvino_model", f"{name}.xml"))
        sources["path"] = "DETECTION_MODEL"
    if not os.path.isfile(path):
        present = sorted(os.listdir(model_root)) if os.path.isdir(model_root) else []
        raise FileNotFoundError(
            f"model file for backend={backend} precision={precision} not found: {path} "
            f"(DETECTION_MODEL={name!r}; contents of {model_root}: {present})"
        )

    # ---- square imgsz (Ultralytics path) -------------------------------
    nominal = nominal_imgsz(name) or nominal_imgsz(path)
    if imgsz_override and imgsz_override > 0:
        imgsz = int(imgsz_override)
        sources["imgsz"] = "DETECTION_IMG_SIZE"
        if nominal and nominal != imgsz:
            logger.warning("DETECTION_IMG_SIZE=%d overrides %s's native size %d — clear "
                           "DETECTION_IMG_SIZE (or set it to 0) to follow the model", imgsz, name, nominal)
    elif nominal:
        imgsz = nominal
        sources["imgsz"] = "model name"
    else:
        imgsz = 480
        sources["imgsz"] = "fallback (no size in model name)"
        logger.warning("could not derive the input size from model name %r — using 480; "
                       "set DETECTION_IMG_SIZE explicitly", name)

    # ---- (h, w) + names for the static-shape backends -------------------
    input_hw: Optional[Tuple[int, int]] = None
    names: Dict[int, str] = {}
    if input_hw_override:
        input_hw = tuple(int(v) for v in input_hw_override)  # type: ignore[assignment]
        sources["input_hw"] = "DETECTION_CPU_INPUT_SIZE"
    if backend == "openvino":
        meta = os.path.join(os.path.dirname(path), "metadata.yaml")
        hw, nm = read_metadata_file(meta)
        if hw and input_hw is None:
            input_hw, sources["input_hw"] = hw, meta
        if nm:
            names, sources["names"] = nm, meta
    if backend in ("onnx", "openvino") and (input_hw is None or not names):
        hw, nm, p = _export_info_for(model_root, name)
        if hw and input_hw is None:
            input_hw, sources["input_hw"] = hw, p
        if nm and not names:
            names, sources["names"] = nm, p
    if backend in ("onnx", "openvino") and input_hw is None:
        m = re.search(r"_(\d+)x(\d+)\.onnx$", path)
        if m:
            input_hw, sources["input_hw"] = (int(m.group(1)), int(m.group(2))), "file name"
    if not names and default_names:
        names, sources["names"] = dict(default_names), "CLASS_LABELS (config.py)"

    return ModelSpec(name=name, backend=backend, precision=precision, path=path, imgsz=imgsz,
                     input_hw=input_hw, names=names, sources=sources)
