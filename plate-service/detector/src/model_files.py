"""
model_files.py
--------------------------------------------------------------------
Turns the .env model choice into concrete files on disk, the same way
the multi-stream benchmark (tools/bench_multistream.py) finds them:
one folder per model, with an export_info.yaml manifest inside.

    /models/                                   (= DETECTION_MODELS_DIR on the host)
    ├── plate_v8n_480/
    │   ├── export_info.yaml                    manifest: model_name, imgsz, names,
    │   │                                       pt, onnx, openvino: {ov_fp32, ov_int8_box}
    │   ├── plate_v8n_480.pt
    │   ├── plate_v8n_480_288x480.onnx
    │   ├── plate_v8n_480_fp32_openvino_model/
    │   └── plate_v8n_480_int8_box_openvino_model/
    └── plate_v8s_640/
        └── plate_v8s_640.pt                    (+ export_info.yaml / exports if made)

Every path comes from export_info.yaml when it is there. Without a
manifest the files are looked up by name (config.MODEL_FILE_PATTERNS),
so a folder holding only plate_v8s_640.pt still works for the GPU.

Nothing here imports torch, ultralytics, openvino or onnxruntime — it
is plain filesystem + YAML work, safe in the parent process (main.py
validates the files before any engine starts).
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


@dataclass
class ModelSpec:
    name: str                      # base model, e.g. "plate_v8n_480"
    variant: str                   # pt | openvino_fp32 | openvino_int8 | onnx
    path: str                      # .pt file / *_openvino_model folder / .onnx file
    folder: str                    # the model's folder
    imgsz: int                     # square size for the GPU (.pt) path: 480 / 640
    input_hw: Optional[Tuple[int, int]] = None   # export (h, w), e.g. (288, 480)
    names: Dict[int, str] = field(default_factory=dict)
    manifest: Optional[str] = None               # export_info.yaml used, if any

    def describe(self) -> str:
        hw = f"{self.input_hw[0]}x{self.input_hw[1]}" if self.input_hw else "-"
        return (f"model={self.name} variant={self.variant} path={self.path} imgsz={self.imgsz} "
                f"export_input={hw} names={self.names} manifest={self.manifest or 'none (file names)'}")


# --------------------------------------------------------------------
# small parsers
# --------------------------------------------------------------------
def nominal_imgsz(name: str) -> Optional[int]:
    """plate_v8n_480 -> 480, plate_v8s_640.pt -> 640, best -> None."""
    base = re.sub(r"\.(pt|onnx|xml)$", "", os.path.basename(str(name or "")))
    m = re.search(r"_(\d{3,4})$", base)
    return int(m.group(1)) if m else None


def parse_hw(value: Any) -> Optional[Tuple[int, int]]:
    """480 -> (480, 480); "288x480" / [288, 480] / [1, 3, 288, 480] -> (288, 480)."""
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return (int(value), int(value)) if int(value) > 0 else None
        if isinstance(value, str):
            m = re.fullmatch(r"\s*(\d+)\s*[x,*× ]\s*(\d+)\s*", value.lower())
            if m:
                return int(m.group(1)), int(m.group(2))
            if value.strip().isdigit():
                return parse_hw(int(value))
            return parse_hw(ast.literal_eval(value)) if value.strip()[:1] in "[(" else None
        if isinstance(value, dict):
            low = {str(k).lower(): v for k, v in value.items()}
            for hk, wk in (("h", "w"), ("height", "width")):
                if hk in low and wk in low:
                    return int(low[hk]), int(low[wk])
            return None
        if isinstance(value, (list, tuple)):
            vals = [int(v) for v in value]
            return {1: lambda: (vals[0], vals[0]), 2: lambda: (vals[0], vals[1]),
                    4: lambda: (vals[2], vals[3])}.get(len(vals), lambda: None)()
    except (TypeError, ValueError, SyntaxError):
        return None
    return None


def parse_names(value: Any) -> Dict[int, str]:
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


def load_export_info(folder: str) -> Tuple[Dict[str, Any], Optional[str]]:
    path = os.path.join(folder, "export_info.yaml")
    if not os.path.isfile(path):
        return {}, None
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data if isinstance(data, dict) else {}), path
    except Exception as e:
        logger.warning("could not read %s: %s", path, e)
        return {}, None


# --------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------
def model_folder(root: str, name: str) -> str:
    """<root>/<name>/ — or <root> itself for a flat folder (older layout)."""
    sub = os.path.join(root, name)
    return sub if os.path.isdir(sub) else root


def _first(folder: str, patterns: List[str], name: str) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(folder, pat.format(name=name))))
        if hits:
            return hits[0]
    return None


def resolve_model(root: str, name: str, variant: str, manifest_keys: Dict[str, List[str]],
                  file_patterns: Dict[str, List[str]],
                  default_names: Optional[Dict[int, str]] = None) -> ModelSpec:
    """name: plate_v8n_480 / plate_v8s_640. variant: pt / openvino_fp32 /
    openvino_int8 / onnx. manifest_keys / file_patterns come from config.
    Raises FileNotFoundError / ValueError with an actionable message."""
    if variant not in file_patterns:
        raise ValueError(f"unknown model variant {variant!r} — one of {sorted(file_patterns)}")
    folder = model_folder(root, name)
    info, manifest = load_export_info(folder)

    path = None
    if info:
        for key in manifest_keys.get(variant, []):     # e.g. ["openvino", "ov_int8_box"]
            node: Any = info
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, str) and node:
                cand = node if os.path.isabs(node) else os.path.join(folder, node)
                if os.path.exists(cand):
                    path = cand
                    break
                logger.warning("%s lists %s=%s but it does not exist", manifest, key, node)
    if path is None:
        path = _first(folder, file_patterns[variant], name)
    if path is None or not os.path.exists(path):
        present = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        raise FileNotFoundError(
            f"no {variant} model for {name!r} in {folder} (manifest: {manifest or 'none'}; "
            f"looked for {file_patterns[variant]}; folder contains: {present})"
        )

    input_hw = parse_hw(info.get("imgsz")) if info else None
    if variant == "onnx" and input_hw is None:
        m = re.search(r"_(\d+)x(\d+)\.onnx$", path)
        input_hw = (int(m.group(1)), int(m.group(2))) if m else None
    if variant.startswith("openvino") and input_hw is None:
        mpath = os.path.join(path, "metadata.yaml") if os.path.isdir(path) else ""
        if os.path.isfile(mpath):
            try:
                import yaml
                with open(mpath, "r", encoding="utf-8") as f:
                    input_hw = parse_hw((yaml.safe_load(f) or {}).get("imgsz"))
            except Exception:
                pass

    imgsz = nominal_imgsz(name) or (max(input_hw) if input_hw else None)
    if imgsz is None:
        imgsz = 480
        logger.warning("no input size in model name %r or its manifest — using 480", name)

    names = parse_names(info.get("names")) if info else {}
    if not names and default_names:
        names = dict(default_names)
    return ModelSpec(name=name, variant=variant, path=path, folder=folder, imgsz=imgsz,
                     input_hw=input_hw, names=names, manifest=manifest)
