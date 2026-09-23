"""
test_rtsp2.py — isolates WHY cv2.VideoCapture hangs inside the engine
but not in a plain script.

Four cases, each run in a watchdog thread with a hard 20s limit so the
script always reports instead of hanging:

  1. no env options,      main-thread-style call   (what worked before)
  2. rtsp_transport;tcp,  same call                (engine's env var)
  3. no env options,      called from a THREAD     (engine's threading)
  4. rtsp_transport;tcp,  called from a THREAD     (exact engine conditions)
  5. timeout;5000000 (the modern FFmpeg spelling of stimeout) + tcp, thread

Run:  docker exec -it face_detector python3 /tmp/test_rtsp2.py
"""

import os
import threading
import time

import cv2

URL = "rtsp://mediamtx:8554/1"
LIMIT = 20.0


def attempt(label, env_value):
    result = {}

    def work():
        t = time.time()
        try:
            if env_value is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = env_value
            cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
            opened = cap.isOpened()
            ret, frame = cap.read() if opened else (False, None)
            cap.release()
            result["msg"] = (
                f"opened={opened} read={ret} "
                f"shape={None if frame is None else frame.shape} "
                f"in {time.time() - t:.2f}s"
            )
        except Exception as e:
            result["msg"] = f"EXCEPTION after {time.time() - t:.2f}s: {e!r}"

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(LIMIT)

    if th.is_alive():
        print(f"[{label}] *** HUNG (still blocked after {LIMIT:.0f}s) ***", flush=True)
        return False
    print(f"[{label}] {result.get('msg')}", flush=True)
    return True


print(f"cv2 version: {cv2.__version__}", flush=True)
print(f"URL: {URL}\n", flush=True)

ok1 = attempt("1: no env      ", None)
ok2 = attempt("2: tcp         ", "rtsp_transport;tcp")
ok3 = attempt("3: no env      ", None)
ok4 = attempt("4: tcp         ", "rtsp_transport;tcp")
ok5 = attempt("5: tcp+timeout ", "rtsp_transport;tcp|timeout;5000000")

print("\n--- summary ---", flush=True)
print(f"no-env: {ok1 and ok3} | forced-tcp: {ok2 and ok4} | tcp+timeout: {ok5}", flush=True)