"""
export_cpu_models.py
--------------------------------------------------------------------
Makes the CPU variants of a plate .pt model in the folder layout the
detector (and tools/bench_multistream.py) read:

    <out>/<name>/
    ├── export_info.yaml                     manifest: model_name, imgsz, names, pt, onnx,
    │                                        openvino: {ov_fp32: ...}
    ├── <name>.pt                            (copied)
    ├── <name>_<H>x<W>.onnx                  ONNX FP32, static, no NMS
    └── <name>_fp32_openvino_model/          OpenVINO FP32

    python tools/export_cpu_models.py --model /export/plate_v8s_640.pt --out /export
        -> /export/plate_v8s_640/...   (input 384x640, 16:9 rounded up to /32)

Export settings follow the benchmarked recipe: opset 17, simplify,
static, batch 1, FP32, no NMS; OpenVINO converted from that ONNX with
compress_to_fp16=False. INT8 (ov_int8_box) is NOT made here — it comes
from requantize_int8_head_fp32.py and must be added to export_info.yaml
as openvino.ov_int8_box. Run this with the SAME OpenVINO version the
detector image runs (inside the image is easiest).
--------------------------------------------------------------------
"""

import argparse
import os
import re
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to <name>.pt")
    ap.add_argument("--out", required=True, help="models root; the model folder <out>/<name>/ is created")
    ap.add_argument("--input-size", default=None, help="static HxW (default: 16:9 of the name's size)")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    import yaml
    import openvino as ov
    from ultralytics import YOLO

    pt = os.path.abspath(a.model)
    name = re.sub(r"\.pt$", "", os.path.basename(pt))
    folder = os.path.join(os.path.abspath(a.out), name)
    os.makedirs(folder, exist_ok=True)
    if a.input_size:
        h, w = (int(v) for v in a.input_size.lower().split("x"))
    else:
        m = re.search(r"_(\d{3,4})$", name)
        if not m:
            raise SystemExit(f"cannot read a size from {name!r} (expected e.g. plate_v8s_640) — pass --input-size HxW")
        size = int(m.group(1))
        h, w = -(-round(size * 9 / 16) // 32) * 32, size
    assert h % 32 == 0 and w % 32 == 0, "input sides must be multiples of 32"

    pt_dst = os.path.join(folder, f"{name}.pt")
    if os.path.abspath(pt) != pt_dst:
        shutil.copy2(pt, pt_dst)
    model = YOLO(pt_dst)
    names = {int(k): str(v) for k, v in model.names.items()}

    tmp = model.export(format="onnx", imgsz=(h, w), opset=a.opset, simplify=True, dynamic=False,
                       batch=1, half=False, nms=False, device="cpu")
    onnx_name = f"{name}_{h}x{w}.onnx"
    shutil.move(tmp, os.path.join(folder, onnx_name))

    ov_name = f"{name}_fp32_openvino_model"
    os.makedirs(os.path.join(folder, ov_name), exist_ok=True)
    ov.save_model(ov.convert_model(os.path.join(folder, onnx_name)),
                  os.path.join(folder, ov_name, f"{name}.xml"), compress_to_fp16=False)
    with open(os.path.join(folder, ov_name, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"task": "detect", "stride": 32, "batch": 1, "imgsz": [h, w], "names": names},
                       f, sort_keys=False)

    info_path = os.path.join(folder, "export_info.yaml")
    info = {}
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
    info.update({"model_name": name, "imgsz": [h, w], "names": names, "pt": f"{name}.pt",
                 "onnx": onnx_name, "openvino_version": ov.__version__})
    info.setdefault("openvino", {})["ov_fp32"] = ov_name
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"done -> {folder}\n{yaml.safe_dump(info, sort_keys=False)}")


if __name__ == "__main__":
    main()
