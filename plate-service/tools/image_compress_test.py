"""
image_compress_test.py
--------------------------------------------------------------------
Measures how the OCR service stores images in MinIO today and what a
better (still lossless) setting would give, on YOUR images.

Today (ocr_service/src/worker.py `_upload_image_to_minio`):
    ok, buf = cv2.imencode(".png", img)        # no parameters
OpenCV's defaults for PNG are: compression level 1 (fastest, biggest),
strategy RLE. PNG is lossless, so nothing is lost - but the files are
bigger than they need to be.

What this script does for every image in the input folder:
  1. CURRENT method  -> saves  <out>/current/<name>.png  and prints size + time
  2. tries other encoders (PNG levels / strategies, WebP lossless, JPEG)
     and prints size, encode time, decode time and whether the result is
     bit-identical (lossless) or its PSNR (lossy)
  3. OPTIMIZED       -> saves the smallest LOSSLESS PNG that encodes within
     MAX_ENCODE_MS (40 ms; the slowest PNG settings can take ~1 s) as
     <out>/optimized/<name>.png and prints size + time
At the end it prints the average of every method over all images.

Usage (cmd):
    pip install opencv-python numpy
    python image_compress_test.py
    python image_compress_test.py "C:\\path\\to\\images"
--------------------------------------------------------------------
"""
import os
import statistics
import sys
import time

import cv2
import numpy as np

DEFAULT_DIR = r"C:\Users\eyerik.com\Desktop\Snowa\faulty_ocr\small"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
REPEAT = 5   # encodes per method; the median time is reported
MAX_ENCODE_MS = 40.0   # "optimized" = smallest lossless PNG that encodes within this time


def kb(n):
    return f"{n / 1024:8.1f} KB"


def method_list():
    """(name, extension, params). Skips options this OpenCV build lacks."""
    g = lambda n: getattr(cv2, n, None)
    m = [("CURRENT png default (level 1, RLE)", ".png", None)]
    strategies = [("RLE", g("IMWRITE_PNG_STRATEGY_RLE")),
                  ("FILTERED", g("IMWRITE_PNG_STRATEGY_FILTERED")),
                  ("DEFAULT", g("IMWRITE_PNG_STRATEGY_DEFAULT"))]
    for level in (3, 6, 9):
        for sname, sval in strategies:
            if sval is None or g("IMWRITE_PNG_COMPRESSION") is None:
                continue
            m.append((f"png level {level} {sname}", ".png",
                      [cv2.IMWRITE_PNG_COMPRESSION, level, cv2.IMWRITE_PNG_STRATEGY, sval]))
    if g("IMWRITE_WEBP_QUALITY") is not None:
        m.append(("webp LOSSLESS", ".webp", [cv2.IMWRITE_WEBP_QUALITY, 101]))
    for q in (95, 98, 100):
        p = [cv2.IMWRITE_JPEG_QUALITY, q]
        if g("IMWRITE_JPEG_SAMPLING_FACTOR_444") is not None:
            p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        if g("IMWRITE_JPEG_OPTIMIZE") is not None:
            p += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        m.append((f"jpeg q{q} 4:4:4 optimized (lossy)", ".jpg", p))
    return m


def encode(img, ext, params):
    t = []
    buf = None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(ext, img, params) if params else cv2.imencode(ext, img)
        t.append((time.perf_counter() - t0) * 1000)
        if not ok:
            return None, None
    return buf, statistics.median(t)


def check(img, buf):
    t0 = time.perf_counter()
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    dt = (time.perf_counter() - t0) * 1000
    if dec is None or dec.shape != img.shape:
        return dt, "decode failed", None
    if np.array_equal(dec, img):
        return dt, "lossless", None
    mse = float(np.mean((dec.astype(np.float32) - img.astype(np.float32)) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return dt, "lossy", psnr


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(src):
        print(f"Folder not found: {src}")
        return
    out = os.path.join(src, "compressed_test")
    cur_dir, opt_dir = os.path.join(out, "current"), os.path.join(out, "optimized")
    os.makedirs(cur_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)]
    if not files:
        print(f"No images in {src}")
        return
    print(f"OpenCV {cv2.__version__} | {len(files)} image(s) from {src}\nOutput: {out}\n")

    methods = method_list()
    totals = {name: {"size": [], "enc": [], "dec": [], "psnr": [], "lossless": True} for name, _, _ in methods}
    raw_total = 0
    for fname in files:
        path = os.path.join(src, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (cannot read): {fname}")
            continue
        h, w = img.shape[:2]
        raw = img.nbytes
        raw_total += raw
        print("=" * 100)
        print(f"{fname}   {w}x{h}   raw pixels {kb(raw)}   original file {kb(os.path.getsize(path))}")
        print("-" * 100)
        best_png = None
        rows = []
        for name, ext, params in methods:
            buf, enc_ms = encode(img, ext, params)
            if buf is None:
                continue
            dec_ms, kind, psnr = check(img, buf)
            size = len(buf)
            tt = totals[name]
            tt["size"].append(size); tt["enc"].append(enc_ms); tt["dec"].append(dec_ms)
            if kind == "lossy":
                tt["lossless"] = False
                tt["psnr"].append(psnr)
            rows.append((name, size, enc_ms, dec_ms, kind, psnr))
            if ext == ".png" and kind == "lossless" and enc_ms <= MAX_ENCODE_MS and (best_png is None or size < best_png[1]):
                best_png = (name, size, enc_ms, buf)
            if name.startswith("CURRENT"):
                with open(os.path.join(cur_dir, os.path.splitext(fname)[0] + ".png"), "wb") as f:
                    f.write(buf)
                cur_size, cur_ms = size, enc_ms
                print(f"CURRENT METHOD saved -> current\\{os.path.splitext(fname)[0]}.png   size {kb(size)}   compress time {enc_ms:7.2f} ms")
        if best_png:
            with open(os.path.join(opt_dir, os.path.splitext(fname)[0] + ".png"), "wb") as f:
                f.write(best_png[3])
            print(f"OPTIMIZED      saved -> optimized\\{os.path.splitext(fname)[0]}.png  size {kb(best_png[1])}   compress time {best_png[2]:7.2f} ms"
                  f"   ({best_png[0]})   {100 * (1 - best_png[1] / cur_size):.1f}% smaller than current")
        print(f"\n{'method':42s} {'size':>12s} {'vs current':>10s} {'encode ms':>10s} {'decode ms':>10s}  quality")
        for name, size, enc_ms, dec_ms, kind, psnr in rows:
            q = "bit-identical" if kind == "lossless" else (f"PSNR {psnr:.1f} dB" if psnr else kind)
            print(f"{name:42s} {kb(size):>12s} {100 * size / cur_size:9.1f}% {enc_ms:10.2f} {dec_ms:10.2f}  {q}")
        print()

    print("=" * 100)
    print("AVERAGE OVER ALL IMAGES")
    print("-" * 100)
    base = statistics.mean(totals[methods[0][0]]["size"]) if totals[methods[0][0]]["size"] else 1
    print(f"{'method':42s} {'avg size':>12s} {'vs current':>10s} {'encode ms':>10s} {'decode ms':>10s}  quality")
    for name, _, _ in methods:
        tt = totals[name]
        if not tt["size"]:
            continue
        s = statistics.mean(tt["size"])
        q = "bit-identical" if tt["lossless"] else f"PSNR {statistics.mean(tt['psnr']):.1f} dB"
        print(f"{name:42s} {kb(s):>12s} {100 * s / base:9.1f}% {statistics.mean(tt['enc']):10.2f} "
              f"{statistics.mean(tt['dec']):10.2f}  {q}")
    print(f"\nFiles written to: {out}")


if __name__ == "__main__":
    main()
