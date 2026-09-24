"""
codec.py
--------------------------------------------------------------------
Wire format for the detector<->OCR task/result envelopes.

These travel over `{module}:internal:ocr:tasks` and
`{module}:internal:ocr:results:{engine_id}` — internal, trusted,
same-deployment queues. They are never seen by the backend or by
camera_stream, so there is no reason to pay JSON's cost for a payload
that is mostly raw JPEG/PNG bytes (plate crops, sometimes a full ROI
frame).

pickle (protocol HIGHEST) is used deliberately: it is binary-safe,
round-trips nested dict/list/bytes/float structures without a custom
schema, and both ends are our own trusted Python processes. This is
the same trust boundary multiprocessing.Queue used before the switch
to Redis — only the transport changed, not who can read the payload.

DateTimeEncoder is separate: it is for the JSON payloads pushed onto
`{module}:vehicle:results`, the one place date/time/datetime objects
need to cross a JSON boundary to the backend.
--------------------------------------------------------------------
"""

from __future__ import annotations

import datetime
import json
import pickle
from typing import Any


def encode_task(payload: dict) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def decode_task(data: bytes) -> dict:
    return pickle.loads(data)


# Results use the same envelope format as tasks — kept as distinct
# names so call sites read clearly about which direction they are on.
encode_result = encode_task
decode_result = decode_task


class DateTimeEncoder(json.JSONEncoder):
    """json.dumps(..., cls=DateTimeEncoder) for payloads that may carry
    datetime.date / datetime.time / datetime.datetime, e.g. the
    payload pushed onto {module}:vehicle:results, and raw bytes (e.g.
    an accidentally-included image) base64-encoded rather than
    blowing up json.dumps — mirrors the reference video_processor.py's
    own DateTimeEncoder."""

    def default(self, obj: Any):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode("utf-8")
        return super().default(obj)
