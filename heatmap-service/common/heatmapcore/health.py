"""
health.py
--------------------------------------------------------------------
A minimal, dependency-free HTTP health/status endpoint shared by both
services. Deliberately stdlib-only (http.server) rather than
FastAPI/uvicorn — the detector and OCR images are already large
(torch/ultralytics + paddleocr + OpenCV, the base image); this avoids
one more thing that has to be present for a container healthcheck to
work.

GET /health  -> {"status": "ok"|"starting", ...whatever snapshot_fn()
                 returns...}
GET anything else -> 404

Used by the Docker HEALTHCHECK in each service's Dockerfile.
--------------------------------------------------------------------
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from .logging_setup import setup_logger

logger = setup_logger("heatmapcore.health")


def _make_handler(snapshot_fn: Callable[[], dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass  # keep container logs to what the services print themselves

        def do_GET(self):
            if self.path.rstrip("/") not in ("", "/health"):
                self.send_response(404)
                self.end_headers()
                return
            try:
                payload = snapshot_fn()
                status_code = 200 if payload.get("status") == "ok" else 503
            except Exception as e:
                payload = {"status": "error", "error": str(e)}
                status_code = 500
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve_health(port: int, snapshot_fn: Callable[[], dict]) -> ThreadingHTTPServer:
    """Starts the health server on a daemon thread and returns the
    server object (call .shutdown() to stop it, mostly useful in
    tests)."""
    server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(snapshot_fn))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-http")
    thread.start()
    logger.info("Health endpoint listening on :%d/health", port)
    return server
