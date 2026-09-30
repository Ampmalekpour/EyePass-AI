"""
parity_test.py
--------------------------------------------------------------------
Acceptance test 1 (README "CPU pipeline -> acceptance tests"): run the
SAME recorded clip through two detector backends and compare their
detections frame by frame, with the benchmark's matching rule
(same class, IoU >= 0.5, greedy by confidence).

    python tools/parity_test.py --video /debug/clip.mp4                       # pt vs openvino fp32
    python tools/parity_test.py --video clip.mp4 --candidate onnx
    python tools/parity_test.py --video clip.mp4 --candidate openvino --precision int8
    python tools/parity_test.py --video clip.mp4 --roi 0 0.3 1 0.7            # engine ROI (x y w h, 0..1)

PASS: recall >= 99.5 %, precision >= 99.5 %, average IoU >= 0.98.
FP32 ONNX/OpenVINO should hit 100 % / 100 % / 1.000.

It uses the detector's real code (model_files.resolve_model +
inference_backends.build_backend with fallback DISABLED), the same
DETECTION_MODEL / DETECTION_CONF_THRESHOLD / ... env as the service,
and the same ROI crop the engine feeds the model, so a pass here is a
pass for the pipeline. Optional --save writes every detection to a
JSON file per backend for offline inspection.
--------------------------------------------------------------------
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # detector/src when run from the repo
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config  # noqa: E402
from inference_backends import BackendSettings, RuntimePlan, build_backend  # noqa: E402
from model_files import parse_hw  # noqa: E402


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def agreement(ref: np.ndarray, cand: np.ndarray, iou_thr: float = 0.5):
    """Greedy one-to-one matching by reference confidence: same class,
    IoU >= iou_thr. Returns (n_matched, [ious], [conf deltas])."""
    if len(ref) == 0 or len(cand) == 0:
        return 0, [], []
    ious = iou_matrix(ref[:, :4], cand[:, :4])
    same = ref[:, None, 5] == cand[None, :, 5]
    ious = np.where(same, ious, 0.0)
    used = set()
    matched, iou_list, dconf = 0, [], []
    for i in np.argsort(-ref[:, 4]):
        order = np.argsort(-ious[i])
        for j in order:
            if j in used or ious[i, j] < iou_thr:
                continue
            used.add(j)
            matched += 1
            iou_list.append(float(ious[i, j]))
            dconf.append(abs(float(ref[i, 4] - cand[j, 4])))
            break
    return matched, iou_list, dconf


def apply_roi(frame, roi):
    x, y, w, h = roi
    H, W = frame.shape[:2]
    x1, y1 = int(max(0, min(W - 1, x * W))), int(max(0, min(H - 1, y * H)))
    x2, y2 = int(max(0, min(W, (x + w) * W))), int(max(0, min(H, (y + h) * H)))
    return frame if (x2 <= x1 or y2 <= y1) else frame[y1:y2, x1:x2]


def make_backend(name: str, precision: str, streams: int, log):
    plan = RuntimePlan("cpu", "cpu", name, precision if name == "openvino" else "fp32")
    s = BackendSettings(
        model_root=config.MODEL_ROOT, model_name=config.DETECTION_MODEL,
        model_path_override=config.MODEL_PATH if name == "pt" else "",
        imgsz_override=config.IMG_SIZE, cpu_input_hw=parse_hw(config.CPU_INPUT_SIZE) if config.CPU_INPUT_SIZE else None,
        conf=config.CONF_THRESHOLD, iou=config.IOU_THRESHOLD, max_det=config.MAX_DET,
        cpu_streams=streams, cpu_threads=config.CPU_INFER_THREADS,
        openvino_cache_dir="", openvino_core_type=config.OPENVINO_SCHEDULING_CORE_TYPE,
        warmup_runs=2, shape_mode=config.CPU_SHAPE_MODE, fallback_to_pt=False,
        class_labels=config.CLASS_LABELS,
    )
    return build_backend(plan, s, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", default="pt", choices=["pt", "onnx", "openvino"])
    ap.add_argument("--candidate", default="openvino", choices=["pt", "onnx", "openvino"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"], help="candidate precision (openvino)")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--streams", type=int, default=1, help="frames per batch (= cameras), 1 or 4")
    ap.add_argument("--roi", type=float, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--save", default=None, help="folder to write per-frame detections JSON")
    ap.add_argument("--min-recall", type=float, default=0.995)
    ap.add_argument("--min-precision", type=float, default=0.995)
    ap.add_argument("--min-iou", type=float, default=0.98)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("parity")
    cv2.setNumThreads(2)

    ref = make_backend(args.reference, "fp32", args.streams, log)
    cand = make_backend(args.candidate, args.precision, args.streams, log)
    ref.warmup()
    cand.warmup()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")

    n_ref = n_cand = n_match = 0
    ious, dconfs = [], []
    t_ref = t_cand = 0.0
    frames_done = 0
    saved = {"reference": [], "candidate": []}
    while frames_done < args.frames:
        batch = []
        for _ in range(args.streams):
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(apply_roi(frame, args.roi) if args.roi else frame)
        if not batch:
            break
        t0 = time.perf_counter()
        r_dets, _ = ref.infer(batch)
        t1 = time.perf_counter()
        c_dets, _ = cand.infer(batch)
        t2 = time.perf_counter()
        t_ref += t1 - t0
        t_cand += t2 - t1
        for rd, cd in zip(r_dets, c_dets):
            m, il, dc = agreement(rd, cd)
            n_ref += len(rd)
            n_cand += len(cd)
            n_match += m
            ious += il
            dconfs += dc
            if args.save:
                saved["reference"].append(rd.tolist())
                saved["candidate"].append(cd.tolist())
        frames_done += len(batch)

    recall = n_match / n_ref if n_ref else 1.0
    precision = n_match / n_cand if n_cand else 1.0
    avg_iou = float(np.mean(ious)) if ious else 1.0
    worst_iou = float(np.min(ious)) if ious else 1.0
    avg_dconf = float(np.mean(dconfs)) if dconfs else 0.0
    ok = recall >= args.min_recall and precision >= args.min_precision and avg_iou >= args.min_iou

    print()
    print(f"reference : {ref.label} input={ref.imgsz_label}  {1000 * t_ref / max(frames_done, 1):.1f} ms/frame")
    print(f"candidate : {cand.label} input={cand.imgsz_label}  {1000 * t_cand / max(frames_done, 1):.1f} ms/frame")
    print(f"frames={frames_done} streams={args.streams} roi={args.roi} conf={config.CONF_THRESHOLD}")
    print(f"detections: reference={n_ref} candidate={n_cand} matched={n_match}")
    print(f"recall={100 * recall:.2f}%  precision={100 * precision:.2f}%  "
          f"avg IoU={avg_iou:.4f}  worst IoU={worst_iou:.4f}  avg |dconf|={avg_dconf:.4f}")
    print("RESULT:", "PASS" if ok else "FAIL",
          f"(needs recall/precision >= {100 * args.min_recall:.1f}% and avg IoU >= {args.min_iou})")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for k, v in saved.items():
            with open(os.path.join(args.save, f"{k}.json"), "w") as f:
                json.dump(v, f)
    ref.close()
    cand.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
