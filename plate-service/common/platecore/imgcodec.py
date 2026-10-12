"""
imgcodec.py
--------------------------------------------------------------------
How the pipeline encodes images - ONE place, used by the detector (the
single encode) and by the OCR service (fallback only).

Why JPEG, once:
  * the camera frames are already lossy (H.264 -> decoded), so storing
    them as PNG only preserves the video's own artifacts at 5-7x the size
    (measured: 2688x2160 frame 433 KB as the original JPEG, 3.3 MB as the
    default PNG, 556 KB as JPEG q95 4:4:4);
  * JPEG q95 with full-resolution chroma (4:4:4) and optimized Huffman
    tables is visually lossless (PSNR 56-61 dB measured) and the fastest
    encoder available in OpenCV / Pillow (~35 ms for 2688x2160);
  * the detector encodes ONCE; the OCR service decodes the pixels for
    recognition but uploads the very same bytes to MinIO - no second
    (generation) encode, no extra CPU, no extra quality loss.

cv2 is imported lazily so platecore stays cheap to import.
--------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Optional

DEFAULT_JPEG_QUALITY = 95


def jpeg_params(quality: int = DEFAULT_JPEG_QUALITY) -> list:
    import cv2
    p = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):      # OpenCV >= 4.7
        p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if hasattr(cv2, "IMWRITE_JPEG_OPTIMIZE"):
        p += [cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    return p


def encode_jpeg(img, quality: int = DEFAULT_JPEG_QUALITY) -> Optional[bytes]:
    """BGR array -> JPEG bytes (q95, 4:4:4, optimized by default)."""
    import cv2
    if img is None or getattr(img, "size", 0) == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img, jpeg_params(quality))
    return buf.tobytes() if ok else None
