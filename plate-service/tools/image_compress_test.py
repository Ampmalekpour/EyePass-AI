"""
image_compress_test.py  (OpenCV + Pillow)
--------------------------------------------------------------------
Today the OCR service stores images with   cv2.imencode(".png", img)
(no parameters = PNG level 1, the fastest and biggest setting).

This script compares that with other ways to store the same image, on
YOUR images:  OpenCV PNG levels, Pillow PNG (compress_level, optimize),
WebP lossless (several effort levels) and JPEG 4:4:4 (lossy).

For every image in the folder it
  1. saves the CURRENT method to   <out>\\current\\       and prints size + time
  2. saves the best lossless result that encodes within MAX_ENCODE_MS to
     <out>\\optimized\\  (any format) and the best PNG to <out>\\optimized_png\\
  3. prints a table: size, encode ms, decode ms, bit-identical / PSNR
and at the end the average per method.

It never "freezes": every method prints its name BEFORE it runs, slow
settings are measured once instead of 5 times, and settings that would
take many seconds on a big image are skipped.

    pip install opencv-python numpy pillow
    python image_compress_test.py
    python image_compress_test.py "C:\\path\\to\\images"
--------------------------------------------------------------------
"""
import io
import os
import statistics
import sys
import time

import cv2
import numpy as np

try:
    from PIL import Image, features
except ImportError:
    Image = None

DEFAULT_DIR = r"C:\Users\eyerik.com\Desktop\Snowa\faulty_ocr\small"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
MAX_ENCODE_MS = 40.0      # "optimized" = smallest lossless result within this encode time
SKIP_ABOVE_MS = 4000.0    # a method whose first run exceeds this is not repeated / not run on later images
BIG_PIXELS = 1_500_000    # above this, the very slow settings are skipped up front

SKIPPED = set()           # methods that proved too slow on this machine


def kb(n):
    return f"{n / 1024:8.1f} KB"


# ---------------------------------------------------------------- encoders
def cv_encode(ext, params=None):
    def f(ctx):
        ok, buf = cv2.imencode(ext, ctx["bgr"], params) if params else cv2.imencode(ext, ctx["bgr"])
        return buf.tobytes() if ok else None
    return f


def pil_encode(fmt, **kw):
    def f(ctx):
        b = io.BytesIO()
        ctx["pil"].save(b, fmt, **kw)
        return b.getvalue()
    return f


def method_list():
    """(name, extension, encoder, 'slow' flag). Skips what this install lacks."""
    g = lambda n: getattr(cv2, n, None)
    m = [("CURRENT  cv2 png default (level 1)", ".png", cv_encode(".png"), False)]
    if g("IMWRITE_PNG_COMPRESSION") is not None and g("IMWRITE_PNG_STRATEGY_DEFAULT") is not None:
        for level in (3, 6, 9):
            m.append((f"cv2 png level {level} (zlib default)", ".png",
                      cv_encode(".png", [cv2.IMWRITE_PNG_COMPRESSION, level,
                                         cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT]), level == 9))
    if Image is not None:
        m.append(("pil png compress_level 6", ".png", pil_encode("PNG", compress_level=6), False))
        m.append(("pil png compress_level 9", ".png", pil_encode("PNG", compress_level=9), True))
        m.append(("pil png level 9 + optimize", ".png", pil_encode("PNG", compress_level=9, optimize=True), True))
        if features.check("webp"):
            for meth in (0, 2, 4, 6):
                m.append((f"pil webp LOSSLESS method {meth}", ".webp",
                          pil_encode("WEBP", lossless=True, quality=100, method=meth), meth == 6))
    elif g("IMWRITE_WEBP_QUALITY") is not None:
        m.append(("cv2 webp LOSSLESS", ".webp", cv_encode(".webp", [cv2.IMWRITE_WEBP_QUALITY, 101]), True))
    for q in (95, 98):
        p = [cv2.IMWRITE_JPEG_QUALITY, q]
        if g("IMWRITE_JPEG_SAMPLING_FACTOR_444") is not None:
            p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        if g("IMWRITE_JPEG_OPTIMIZE") is not None:
            p += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        m.append((f"cv2 jpeg q{q} 4:4:4 (lossy)", ".jpg", cv_encode(".jpg", p), False))
    if Image is not None:
        m.append(("pil jpeg q95 4:4:4 optimize (lossy)", ".jpg",
                  pil_encode("JPEG", quality=95, subsampling=0, optimize=True), False))
    return m


def timed_encode(enc, ctx):
    t0 = time.perf_counter()
    data = enc(ctx)
    first = (time.perf_counter() - t0) * 1000
    if data is None:
        return None, None
    reps = 1 if first > 300 else (3 if first > 50 else 7)
    times = [first]
    for _ in range(reps - 1):
        t0 = time.perf_counter()
        enc(ctx)
        times.append((time.perf_counter() - t0) * 1000)
    return data, statistics.median(times)


def check(bgr, data):
    t0 = time.perf_counter()
    try:
        arr = np.frombuffer(data, np.uint8)
        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if dec is None and Image is not None:
            dec = cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        dec = None
    dt = (time.perf_counter() - t0) * 1000
    if dec is None or dec.shape != bgr.shape:
        return dt, "decode failed", None
    if np.array_equal(dec, bgr):
        return dt, "lossless", None
    mse = float(np.mean((dec.astype(np.float32) - bgr.astype(np.float32)) ** 2))
    return dt, "lossy", (99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(src):
        print(f"Folder not found: {src}")
        return
    out = os.path.join(src, "compressed_test")
    d_cur, d_opt, d_png = (os.path.join(out, n) for n in ("current", "optimized", "optimized_png"))
    for d in (d_cur, d_opt, d_png):
        os.makedirs(d, exist_ok=True)
    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)]
    if not files:
        print(f"No images in {src}")
        return
    print(f"OpenCV {cv2.__version__} | Pillow {'%s' % __import__('PIL').__version__ if Image else 'NOT installed (pip install pillow)'}"
          f" | {len(files)} image(s) | output: {out}\n", flush=True)

    methods = method_list()
    totals = {n: {"size": [], "ratio": [], "enc": [], "dec": [], "psnr": [], "lossless": True} for n, *_ in methods}
    for fname in files:
        bgr = cv2.imread(os.path.join(src, fname), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip (cannot read): {fname}")
            continue
        h, w = bgr.shape[:2]
        ctx = {"bgr": bgr}
        if Image is not None:
            ctx["pil"] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        base = os.path.splitext(fname)[0]
        print("=" * 104)
        print(f"{fname}   {w}x{h}   raw {kb(bgr.nbytes)}   original file {kb(os.path.getsize(os.path.join(src, fname)))}", flush=True)
        print("-" * 104)
        rows, cur_size = [], None
        best_any = best_png = None
        for name, ext, enc, slow in methods:
            if name in SKIPPED or (slow and w * h > BIG_PIXELS):
                print(f"  skipped (too slow for this size): {name}", flush=True)
                continue
            print(f"  testing {name} ...", end="", flush=True)
            data, ms = timed_encode(enc, ctx)
            if data is None:
                print(" failed")
                continue
            print(f" {ms:8.1f} ms", flush=True)
            if ms > SKIP_ABOVE_MS:
                SKIPPED.add(name)
            dec_ms, kind, psnr = check(bgr, data)
            size = len(data)
            t = totals[name]
            t["size"].append(size); t["enc"].append(ms); t["dec"].append(dec_ms)
            t["ratio"].append(size / cur_size if cur_size else 1.0)
            if kind == "lossy":
                t["lossless"] = False
                t["psnr"].append(psnr)
            rows.append((name, size, ms, dec_ms, kind, psnr))
            if name.startswith("CURRENT"):
                cur_size = size
                with open(os.path.join(d_cur, base + ".png"), "wb") as f:
                    f.write(data)
                print(f"\n  >>> CURRENT METHOD saved -> current\\{base}.png   size {kb(size)}   compress time {ms:8.2f} ms\n", flush=True)
            if kind == "lossless" and ms <= MAX_ENCODE_MS:
                if best_any is None or size < best_any[1]:
                    best_any = (name, size, ms, data, ext)
                if ext == ".png" and (best_png is None or size < best_png[1]):
                    best_png = (name, size, ms, data, ext)
        cur_size = cur_size or 1
        for label, best, folder in (("OPTIMIZED (any lossless format)", best_any, d_opt),
                                    ("OPTIMIZED (PNG only, same format as today)", best_png, d_png)):
            if best:
                with open(os.path.join(folder, base + best[4]), "wb") as f:
                    f.write(best[3])
                print(f"  >>> {label} saved -> {os.path.basename(folder)}\\{base}{best[4]}   size {kb(best[1])}   "
                      f"compress time {best[2]:8.2f} ms   [{best[0]}]   {100 * (1 - best[1] / cur_size):.1f}% smaller")
        print(f"\n{'method':44s} {'size':>12s} {'vs current':>10s} {'encode ms':>10s} {'decode ms':>10s}  quality")
        for name, size, ms, dec_ms, kind, psnr in rows:
            q = "bit-identical" if kind == "lossless" else (f"PSNR {psnr:.1f} dB" if psnr else kind)
            print(f"{name:44s} {kb(size):>12s} {100 * size / cur_size:9.1f}% {ms:10.2f} {dec_ms:10.2f}  {q}")
        print(flush=True)

    print("=" * 104)
    n_img = len(totals[methods[0][0]]["size"])
    print(f"AVERAGE OVER {n_img} IMAGE(S)  (size = average of each image's size relative to the current method;")
    print("n = images the method ran on - very slow settings are skipped on big images)")
    print("-" * 104)
    print(f"{'method':44s} {'size vs current':>16s} {'avg size':>12s} {'encode ms':>10s} {'decode ms':>10s} {'n':>3s}  quality")
    for name, *_ in methods:
        t = totals[name]
        if not t["size"]:
            continue
        q = "bit-identical" if t["lossless"] else f"PSNR {statistics.mean(t['psnr']):.1f} dB"
        print(f"{name:44s} {100 * statistics.mean(t['ratio']):15.1f}% {kb(statistics.mean(t['size'])):>12s} "
              f"{statistics.mean(t['enc']):10.2f} {statistics.mean(t['dec']):10.2f} {len(t['size']):3d}  {q}")
    print(f"\nFiles written to: {out}")


if __name__ == "__main__":
    main()
