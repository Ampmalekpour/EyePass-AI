"""logging_setup.py — the same setup_logger() the face and plate
services use (facecore/platecore.logging_setup), copied here so the
control hub's log lines look identical and LOG_FORMAT=json feeds the
same aggregator without a second parser.

LOG_FORMAT (env, default "text"): text | json
LOG_LEVEL  (env, default "INFO")
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
