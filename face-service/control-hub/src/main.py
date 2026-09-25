"""
main.py (control hub)
--------------------------------------------------------------------
Entry point of the face control hub. Starts the ModuleRunner for this
module (REDIS_MODULE, default "face") and a tiny HTTP server:

    GET /health            200 if every runner thread is alive and
                           looping (leader or healthy standby), 503
                           otherwise — per-module summary as JSON
    GET /tracks            live/ending/closed tracks the hub holds,
                           with their current resolved answer — the
                           first place to look when "why was X not
                           published / published as unknown?"

One hub per module per Redis. A second container for the same module
is safe (it stands by on the leader lease) — useful only as a hot
standby.
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import config
from logging_setup import setup_logger
from service import ModuleRunner

log = setup_logger("hub.main")
RUNNERS: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep access logs out of stdout
        return

    def _json(self, code, obj):
        body = json.dumps(obj, default=str, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            modules = {m: r.health() for m, r in RUNNERS.items()}
            ok = all(h["alive"] and (h["last_loop_age_sec"] is None or h["last_loop_age_sec"] < 30)
                     for h in modules.values())
            return self._json(200 if ok else 503, {"ok": ok, "modules": modules})
        if url.path == "/tracks":
            q = parse_qs(url.query)
            mods = q.get("module") or list(RUNNERS)
            return self._json(200, {m: RUNNERS[m].tracks() for m in mods if m in RUNNERS})
        return self._json(404, {"error": "not found", "paths": ["/health", "/tracks?module=face"]})


def main():
    from config import module_config

    for m in config.HUB_MODULES:
        cfg = module_config(m)
        log.info("module %s policy: %s", m, cfg)
        RUNNERS[m] = ModuleRunner(cfg)
        RUNNERS[m].start()

    server = ThreadingHTTPServer(("0.0.0.0", config.HEALTH_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="health").start()
    log.info("control hub up — modules=%s health=:%d", config.HUB_MODULES, config.HEALTH_PORT)

    stop = threading.Event()

    def _sig(*_):
        stop.set()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    while not stop.is_set():
        time.sleep(0.5)

    log.info("shutting down")
    for r in RUNNERS.values():
        r.stop()
    for r in RUNNERS.values():
        r.join(timeout=5)
    server.shutdown()


if __name__ == "__main__":
    main()
