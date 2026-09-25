"""
End-to-end test of the suite's single camera_stream against a REAL
redis-server and a fake MediaMTX HTTP API (skipped if redis-server,
redis-py or httpx is missing):

  * three modules configured at once — face, plate, heatmap
  * the same physical camera used by face AND heatmap -> ONE relay path
    (and plate's camera "1" does not collide with face's camera "1")
  * each module's details carry relay_path/stream_url, and detectors
    compute the same path from the address (facecore/platecore relay.py)
  * online/offline reaches every module using the path, each on its OWN
    events channel (face plural, the others singular)
  * a path is removed from MediaMTX only when no module uses it any more
  * MediaMTX restart (paths forgotten) -> paths are re-registered
    Run:  python3 camera-service/tests/test_camera_stream.py
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

try:
    import httpx  # noqa: F401
    import redis  # noqa: F401
    DEPS = True
except Exception:
    DEPS = False
REDIS_BIN = shutil.which("redis-server")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait(pred, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    return pred()


class FakeMtx:
    """Just enough of the MediaMTX v3 API: add / delete / list."""

    def __init__(self):
        self.paths = {}          # name -> source
        self.ready = set()
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None):
                body = json.dumps(obj or {}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                name = self.path.rsplit("/", 1)[-1]
                n = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(n) or b"{}")
                with fake.lock:
                    if name in fake.paths:
                        return self._send(400, {"error": "path already exists"})
                    fake.paths[name] = data.get("source")
                self._send(200)

            def do_DELETE(self):
                name = self.path.rsplit("/", 1)[-1]
                with fake.lock:
                    existed = fake.paths.pop(name, None) is not None
                self._send(200 if existed else 404)

            def do_GET(self):
                with fake.lock:
                    items = [{"name": n, "ready": n in fake.ready} for n in fake.paths]
                self._send(200, {"items": items})

        self.port = _free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@unittest.skipUnless(REDIS_BIN and DEPS, "needs redis-server, redis-py and httpx")
class CameraStreamSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rport = _free_port()
        cls.tmp = tempfile.mkdtemp()
        cls.redis_proc = subprocess.Popen([REDIS_BIN, "--port", str(cls.rport), "--save", "", "--appendonly", "no",
                                           "--dir", cls.tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import redis as _r
        cls.r = _r.Redis(port=cls.rport, decode_responses=True)
        _wait(lambda: cls._ping())
        cls.mtx = FakeMtx()
        os.environ.update({
            "REDIS_URL": f"redis://127.0.0.1:{cls.rport}/0", "MTX_HOST": "127.0.0.1", "MTX_PORT": str(cls.mtx.port),
            "CAMERA_MODULES": "face,plate,heatmap", "RELAY_PATH_MODE": "source", "TRUST_MTX_ONLY": "true",
            "OFFLINE_HOLD_SECONDS": "0.5", "MTX_POLL_INTERVAL_SEC": "0.1", "MONITOR_LOOP_INTERVAL_SEC": "0.1",
            "CONFIG_RESYNC_SEC": "1", "STREAM_BASE_URL": "http://relay.local:8889",
            "CAMERA_STATUS_LOG_PATH": os.path.join(cls.tmp, "status.log"),
        })
        sys.path.insert(0, os.path.join(ROOT, "camera-service", "src"))
        import main as cs
        cls.cs = cs
        # events subscriber (before anything happens)
        cls.events = []
        cls.ps = cls.r.pubsub()
        cls.ps.psubscribe("*:camera:events", "*:cameras:events")

        def _collect():
            for m in cls.ps.listen():
                if m.get("type") == "pmessage":
                    cls.events.append((m["channel"], json.loads(m["data"])))
        threading.Thread(target=_collect, daemon=True).start()

        cls.r.hset("face:cameras:config", "1", json.dumps({"id": "1", "title": "Lobby", "address": "rtsp://10.255.0.5:554/a"}))
        cls.r.hset("heatmap:cameras:config", "7", json.dumps({"id": "7", "title": "Lobby", "address": "rtsp://10.255.0.5:554/a"}))
        cls.r.hset("plate:cameras:config", "1", json.dumps({"id": "1", "title": "Gate", "address": "rtsp://10.255.0.6:554/b"}))
        cls.loop = asyncio.new_event_loop()
        threading.Thread(target=lambda: cls.loop.run_until_complete(cs.main()), daemon=True).start()

    @classmethod
    def _ping(cls):
        try:
            return cls.r.ping()
        except Exception:
            return False

    @classmethod
    def tearDownClass(cls):
        cls.redis_proc.terminate()
        cls.redis_proc.wait(5)
        cls.mtx.server.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _details(self, module, cid):
        raw = self.r.hget(f"{module}:cameras:details", cid)
        return json.loads(raw) if raw else None

    def test_all(self):
        # ---- one relay path per physical camera, no id collision
        _wait(lambda: len(self.mtx.paths) == 2)
        self.assertEqual(len(self.mtx.paths), 2)
        face = _wait(lambda: self._details("face", "1"))
        heat = self._details("heatmap", "7")
        plate = self._details("plate", "1")
        self.assertEqual(face["relay_path"], heat["relay_path"])
        self.assertNotEqual(face["relay_path"], plate["relay_path"])
        self.assertEqual(face["stream_url"], f"http://relay.local:8889/{face['relay_path']}/")
        self.assertEqual(self.mtx.paths[face["relay_path"]], "rtsp://10.255.0.5:554/a")
        self.assertFalse(face["connected"])

        # detectors derive the same path from the address
        sys.path.insert(0, os.path.join(ROOT, "face-service", "common"))
        sys.path.insert(0, os.path.join(ROOT, "plate-service", "common"))
        from facecore.relay import relay_path_for as face_rule
        from platecore.relay import relay_path_for as plate_rule
        self.assertEqual(face_rule("1", "rtsp://10.255.0.5:554/a"), face["relay_path"])
        self.assertEqual(plate_rule("1", "rtsp://10.255.0.6:554/b"), plate["relay_path"])

        # ---- goes online -> both users of the path told, each on its own channel
        self.events.clear()
        with self.mtx.lock:
            self.mtx.ready.add(face["relay_path"])
        _wait(lambda: len([e for e in self.events if e[1].get("connected")]) >= 2)
        online = {(ch, d["id"]) for ch, d in self.events if d.get("connected")}
        self.assertEqual(online, {("face:cameras:events", "1"), ("heatmap:camera:events", "7")})
        self.assertTrue(self._details("face", "1")["connected"])
        self.assertTrue(self._details("heatmap", "7")["connected"])
        self.assertFalse(self._details("plate", "1")["connected"])

        # ---- goes offline -> reported after the hold
        self.events.clear()
        with self.mtx.lock:
            self.mtx.ready.discard(face["relay_path"])
        _wait(lambda: len([e for e in self.events if e[1].get("connected") is False]) >= 2)
        self.assertEqual(self._details("face", "1")["error"], "Network Down")

        # ---- shared path survives one user leaving, goes when the last leaves
        self.r.hdel("face:cameras:config", "1")
        self.r.publish("face:camera:config:updated", "1")
        _wait(lambda: self._details("face", "1") is None)
        self.assertIn(face["relay_path"], self.mtx.paths)
        self.r.hdel("heatmap:cameras:config", "7")
        self.r.publish("heatmap:camera:config:updated", "1")
        _wait(lambda: face["relay_path"] not in self.mtx.paths)
        self.assertNotIn(face["relay_path"], self.mtx.paths)
        self.assertIn(plate["relay_path"], self.mtx.paths)

        # ---- MediaMTX restart forgets every path -> re-registered
        with self.mtx.lock:
            self.mtx.paths.clear()
        self.assertTrue(_wait(lambda: plate["relay_path"] in self.mtx.paths))

        # ---- a module added later, without a pub/sub nudge (periodic resync)
        self.r.hset("fire:cameras:config", "3", json.dumps({"id": "3", "address": "rtsp://10.255.0.6:554/b"}))
        fire = _wait(lambda: self._details("fire", "3"), timeout=5)
        self.assertEqual(fire["relay_path"], plate["relay_path"])
        self.assertEqual(len(self.mtx.paths), 1)


if __name__ == "__main__":
    unittest.main()
