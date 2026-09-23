"""logging_setup.py — one setup_logger() shared by both services so log
lines look the same whether they come from the detector, an Engine
subprocess, the recognizer pool, or a worker process.

LOG_FORMAT (env, default "text"):
  text  - human-readable, unchanged from before this revision:
          "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  json  - one JSON object per line: timestamp, level, logger, message,
          plus any structured fields attached via
          `logger.debug("...", extra={"fields": {...}})`. Meant for
          feeding a log aggregator without writing a second parser for
          this system's own log lines — the detector's per-track
          telemetry and the recognizer's per-match decision trace
          (see recognition_engine.py) both use `extra={"fields": ...}`
          for exactly this reason.

Both formats go to stdout only; this module never opens a file — each
service manages its own log file paths via its own config.py where it
wants one (e.g. LOG_PATH in the recognizer, camera_status.log in
camera-service).

LOG_LEVEL (env, default "INFO") is unchanged from before: DEBUG/INFO/
WARNING/ERROR/CRITICAL, same as Python's own logging module.
"""

import json
import logging
import os
import time


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            # Never let a field silently clobber the core keys above.
            for k, v in fields.items():
                if k not in payload:
                    payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except Exception:
            # A field that truly can't serialize must never take logging
            # itself down — fall back to a coarse, always-safe line.
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "logger": payload["logger"], "message": payload["message"]})


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    if os.environ.get("LOG_FORMAT", "text").strip().lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
