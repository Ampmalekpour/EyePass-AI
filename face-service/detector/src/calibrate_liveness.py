"""
calibrate_liveness.py
--------------------------------------------------------------------
Threshold calibration for liveness.py, run against YOUR footage.

The defaults in config.py are conservative starting points, not
tuned values — the right numbers depend on your camera, lens,
resolution, lighting and how far people stand from it. This script
measures the three signals on recorded clips and prints the actual
distributions, so the thresholds get set from evidence instead of
guesswork.

Record two short clips through the SAME camera at the SAME distance:

    live.mp4    a real person, turning their head left/right ~20-30
                degrees and talking. 15-30 seconds.
    spoof.mp4   a photo on a phone (and/or a printed photo) held up
                and tilted/turned through a similar range. Same
                framing and distance as live.mp4.

Then, from inside the detector container:

    python3 /app/calibrate_liveness.py /debug/live.mp4 /debug/spoof.mp4

It prints, per class, the distribution of:
    planar_residual   (check 1 — the authoritative one)
    ring_follow       (check 2)
    rigid_residual    (check 3)

and suggests thresholds placed between the two populations.

A good separation looks like the live planar_residual sitting well
ABOVE the spoof one with no overlap. If they overlap heavily, the
subject probably did not turn enough — the check needs real rotation
to have anything to measure — so re-record with more head movement
before touching any threshold.

Needs no face detector: it runs the analyzer on the WHOLE frame as a
single "face" box, which is fine for calibration clips where the face
fills most of the frame. Pass --box x1,y1,x2,y2 to pin it down.
--------------------------------------------------------------------
"""

import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from liveness import LivenessAnalyzer, LivenessConfig  # noqa: E402


def collect(path: str, box: Optional[str], every_n: int) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  !! cannot open {path}")
        return {}

    cfg = LivenessConfig()
    cfg.enabled = True
    # calibration wants every sample it can get
    cfg.eval_every_n = max(1, every_n)
    cfg.min_evals = 1
    analyzer = LivenessAnalyzer("calib", cfg)

    out = {"planar_residual": [], "ring_follow": [], "rigid_residual": [],
           "pose_delta_deg": [], "planar_inliers": []}

    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        if box:
            x1, y1, x2, y2 = [int(v) for v in box.split(",")]
        else:
            # middle 70% — good enough when the face fills the clip
            x1, y1 = int(W * 0.15), int(H * 0.15)
            x2, y2 = int(W * 0.85), int(H * 0.85)

        # No landmarks/pose available here, so drive the evidence gate
        # off measured frame-to-frame motion instead: calibration clips
        # are expected to contain deliberate rotation throughout.
        analyzer.update(1, gray, (x1, y1, x2, y2), None,
                        {"yaw": fid * 2.0, "pitch": 0.0, "roll": 0.0}, fid)
        st = analyzer.tracks.get(1)
        if st is not None and st.last_metrics:
            for k in out:
                v = st.last_metrics.get(k)
                if isinstance(v, (int, float)):
                    out[k].append(float(v))
        fid += 1

    cap.release()
    return out


def describe(name: str, vals: List[float]) -> Optional[dict]:
    if not vals:
        print(f"    {name:18s} (no samples)")
        return None
    a = np.array(vals, dtype=float)
    q = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"    {name:18s} n={len(a):4d}  p5={q[0]:.5f} p25={q[1]:.5f} "
          f"med={q[2]:.5f} p75={q[3]:.5f} p95={q[4]:.5f}")
    return {"p5": q[0], "p25": q[1], "med": q[2], "p75": q[3], "p95": q[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live", help="clip of a REAL person turning their head")
    ap.add_argument("spoof", help="clip of a photo/phone held up and tilted")
    ap.add_argument("--box", default=None, help="x1,y1,x2,y2 face box (default: middle 70%%)")
    ap.add_argument("--every-n", type=int, default=1)
    args = ap.parse_args()

    print(f"\n=== LIVE  : {args.live}")
    live = collect(args.live, args.box, args.every_n)
    stats_live = {k: describe(k, v) for k, v in live.items()}

    print(f"\n=== SPOOF : {args.spoof}")
    spoof = collect(args.spoof, args.box, args.every_n)
    stats_spoof = {k: describe(k, v) for k, v in spoof.items()}

    print("\n=== SUGGESTED THRESHOLDS ===")

    sl, ss = stats_live.get("planar_residual"), stats_spoof.get("planar_residual")
    if sl and ss:
        # put the threshold between spoof's upper tail and live's lower tail
        lo, hi = ss["p95"], sl["p5"]
        if lo < hi:
            thr = (lo + hi) / 2.0
            print(f"  LIVENESS_PLANAR_RESIDUAL_THR={thr:.5f}")
            print(f"    (spoof p95={lo:.5f} < live p5={hi:.5f} — clean separation)")
        else:
            print(f"  !! planar_residual OVERLAPS: spoof p95={lo:.5f} >= live p5={hi:.5f}")
            print("     The subject probably did not rotate enough, or the spoof")
            print("     was moved so erratically that tracking noise dominates.")
            print("     Re-record with deliberate 20-30 degree head turns before")
            print("     changing this threshold.")

    rl, rs = stats_live.get("rigid_residual"), stats_spoof.get("rigid_residual")
    if rl and rs and rs["p95"] < rl["p5"]:
        print(f"  LIVENESS_RIGID_RESIDUAL_THR={((rs['p95'] + rl['p5']) / 2.0):.5f}")

    fl, fs = stats_live.get("ring_follow"), stats_spoof.get("ring_follow")
    if fl and fs and fl["p95"] < fs["p5"]:
        print(f"  LIVENESS_RING_FOLLOW_THR={((fl['p95'] + fs['p5']) / 2.0):.3f}")

    print("\nPut whichever of these you trust into .env, then restart face_detector.\n")


if __name__ == "__main__":
    main()
