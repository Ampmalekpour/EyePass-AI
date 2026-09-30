"""
export_cpu_models.py
--------------------------------------------------------------------
Exports a plate .pt model into the CPU variants the detector's
onnx / openvino backends load, with the exact names model_files.py
looks for:

    <out>/<name>_<H>x<W>.onnx                  ONNX FP32, static, no NMS
    <out>/<name>_fp32_openvino_model/<name>.xml/.bin + metadata.yaml
                                               (converted from a dynamic-shape
                                               ONNX, so it can be reshaped per ROI)
    <out>/<name>_export_info.yaml              input size + class names

Example (plate_v8s_640.pt -> 384x640, the 16:9 rectangle for 640):

    python tools/export_cpu_models.py --model /export/plate_v8s_640.pt --out /export

Settings follow the benchmarked recipe: Ultralytics ONNX export with
opset=17, simplify=True, dynamic=False, batch=1, half=False and no NMS
in the graph; then ONNX -> OpenVINO with ov.convert_model and
ov.save_model(compress_to_fp16=False). The OpenVINO model is converted
from a second, dynamic=True ONNX export (use --static-openvino for the
old behaviour) so the detector can compile it once per camera-ROI
shape; at the exported shape it gives the same results as the static
one. Run it with the SAME OpenVINO
version the detector runs (a newer runtime reads older IR, not the
other way round) — easiest is inside the detector image, see README.

INT8 is deliberately not produced here: it has to go through
nncf.quantize_with_accuracy_control on the labelled validation set and
pass the accuracy gate in the README before it is used.
--------------------------------------------------------------------
"""

import argparse
import os
import re
import shutil
import sys


def _default_hw(size: int):
    """16:9 input rounded up to a multiple of 32: 480 -> 288x480, 640 -> 384x640."""
    h = int(-(-round(size * 9 / 16) // 32) * 32)
    return h, size


def _parse_hw(s: str):
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", s or "")
    if not m:
        raise argparse.ArgumentTypeError("use HxW, e.g. 288x480")
    h, w = int(m.group(1)), int(m.group(2))
    if h % 32 or w % 32:
        raise argparse.ArgumentTypeError("both sides must be multiples of 32")
    return h, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to <name>.pt, e.g. plate_v8s_640.pt")
    ap.add_argument("--out", default=None, help="output folder (default: next to the .pt)")
    ap.add_argument("--input-size", type=_parse_hw, default=None,
                    help="static HxW (default: 16:9 from the name, 480 -> 288x480, 640 -> 384x640)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--static-openvino", action="store_true",
                    help="convert OpenVINO from the static ONNX (fixed shape only, no per-ROI reshape)")
    args = ap.parse_args()

    import yaml
    import openvino as ov
    from ultralytics import YOLO

    pt = os.path.abspath(args.model)
    name = re.sub(r"\.pt$", "", os.path.basename(pt))
    out = os.path.abspath(args.out or os.path.dirname(pt))
    os.makedirs(out, exist_ok=True)
    if args.input_size:
        h, w = args.input_size
    else:
        m = re.search(r"_(\d{3,4})$", name)
        if not m:
            sys.exit(f"cannot derive a size from {name!r}; pass --input-size HxW")
        h, w = _default_hw(int(m.group(1)))

    model = YOLO(pt)
    names = {int(k): str(v) for k, v in model.names.items()}
    print(f"exporting {pt} -> {out}  input={h}x{w}  names={names}")

    # the export writes <name>.onnx next to the .pt — move it away before
    # the dynamic export below overwrites it
    tmp = model.export(format="onnx", imgsz=(h, w), opset=args.opset, simplify=True,
                       dynamic=False, batch=1, half=False, nms=False, device="cpu")
    onnx_path = os.path.join(out, f"{name}_{h}x{w}.onnx")
    if os.path.abspath(tmp) != onnx_path:
        shutil.move(tmp, onnx_path)
    print("ONNX FP32      ->", onnx_path)

    ov_dir = os.path.join(out, f"{name}_fp32_openvino_model")
    os.makedirs(ov_dir, exist_ok=True)
    if args.static_openvino:
        ov_model = ov.convert_model(onnx_path)
    else:
        # Converted from a DYNAMIC-shape ONNX so the detector can reshape
        # it per camera ROI (DETECTION_CPU_SHAPE_MODE=roi): a static export
        # bakes anchors for one size and cannot be reshaped. The detector
        # always compiles it at a fixed shape, so it runs as a static model.
        dyn = model.export(format="onnx", imgsz=(h, w), opset=args.opset, simplify=True,
                           dynamic=True, batch=1, half=False, nms=False, device="cpu")
        ov_model = ov.convert_model(dyn)
        os.remove(dyn)
    xml = os.path.join(ov_dir, f"{name}.xml")
    ov.save_model(ov_model, xml, compress_to_fp16=False)
    meta = {
        "description": f"{name} FP32 OpenVINO (converted from {os.path.basename(onnx_path)})",
        "task": "detect", "stride": 32, "batch": 1, "imgsz": [h, w], "names": names,
        "dynamic": not args.static_openvino,
        "openvino_version": ov.__version__,
    }
    with open(os.path.join(ov_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print("OpenVINO FP32  ->", xml)

    info = {"model": name, "imgsz": [h, w], "input_hw": [h, w], "names": names,
            "onnx": os.path.basename(onnx_path), "openvino_fp32": os.path.basename(ov_dir),
            "opset": args.opset, "openvino_version": ov.__version__}
    info_path = os.path.join(out, f"{name}_export_info.yaml")
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print("export info    ->", info_path)
    print("done — now run tools/parity_test.py against a recorded clip before using it in production")


if __name__ == "__main__":
    main()
