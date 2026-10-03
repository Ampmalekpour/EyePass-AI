"""
config.py (plate control hub)
--------------------------------------------------------------------
Every setting of the control hub, set in this file. .env only gives
the deployment (Redis + REDIS_MODULE); the policy below is decided
here (edit and restart).

Two layers:

  * SERVICE-wide settings (which module, health port, stream/consumer
    names, lease, logs).

  * PER-MODULE policy (ModuleConfig + _MODULE_DEFAULTS["plate"]).

Every default reproduces the pipeline's previous behaviour wherever
that behaviour was intentional (0.70 face "stop asking" gate, 0.85
plate skip gate, 10s finalize wait, 8-frame/1-crop final gate); the
places where the defaults deliberately differ are the inconsistency
fixes listed in the README ("What changed and why").
--------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


# ====================================================================
# Service-wide
# ====================================================================
# This hub serves ONE module — the one it ships with (plate-service/). It
# reads the same REDIS_MODULE the module's other services use, so its
# keys always match theirs.
_left = sorted(k for k, v in os.environ.items() if "[FILL_IN]" in (v or ""))
if _left:
    raise SystemExit(f"FATAL: .env still has [FILL_IN] for: {', '.join(_left)} — set real values and restart.")

HUB_MODULE = _env("REDIS_MODULE", "plate")
HUB_MODULES: List[str] = [HUB_MODULE]

HEALTH_PORT = 8020

# Consumer group on both inbound streams. One group per module stream;
# a fixed consumer name so a restarted hub re-reads its own pending
# (read-but-not-acked) entries instead of orphaning them.
CONSUMER_GROUP = "control-hub"
CONSUMER_NAME = "hub"

READ_BLOCK_MS = 200
READ_COUNT = 200
TICK_INTERVAL_SEC = 0.2

# Leader lease: only one hub instance may own a module at a time (its
# track state is in memory). A second instance of the same container
# simply waits as a hot standby and takes over when the lease expires.
LEASE_TTL_SEC = 15.0
LEASE_RENEW_SEC = 5.0

# Track-state checkpoints (one Redis string per live track) expire on
# their own even if the hub never deletes them.
STATE_TTL_SEC = 3600

# ctl lists (hub -> detector engine) expire when an engine goes away.
CTL_TTL_SEC = 120

# Logs
LOG_LEVEL = "INFO"
LOG_FORMAT = "text"          # text | json


# ====================================================================
# Per-module policy
# ====================================================================
@dataclass
class ModuleConfig:
    module: str

    # --- identity/plate is "good enough": stop asking for more ---------
    satisfied_conf: float = 0.70
    # Also satisfied when this many results agree on the same identity
    # (and none disagree), even if no single one reached satisfied_conf.
    # 0 disables the consensus rule.
    consensus_min: int = 3

    # --- trigger publishing ---------------------------------------------
    # A trigger that fires before the identity is known is held until the
    # result of the task sent for it. While that task is still queued it
    # waits up to trigger_task_wait_sec (a busy recognizer/OCR delays the
    # event rather than publishing it empty); if no crop could be sent at
    # all, it waits trigger_max_wait_sec for one.
    trigger_max_wait_sec: float = 3.0
    trigger_task_wait_sec: float = 15.0

    # --- periodic re-query (only for cameras with the periodic flag) ----
    periodic_interval_sec: float = 3.0
    periodic_first_delay_sec: float = 1.0

    # --- track end -----------------------------------------------------
    # Upper bound only: the final goes out the moment the last in-flight
    # result lands (normally < 1s). Sized for a backlogged queue.
    finalize_timeout_sec: float = 30.0
    final_min_seen_frames: int = 8
    final_min_crops: int = 1

    # --- robustness ----------------------------------------------------
    track_stale_sec: float = 120.0
    late_result_grace_sec: float = 300.0
    # What to do with a result that lands after the final was sent:
    #   republish_if_changed  re-send the final (revision+1) when it changes
    #                         the answer — the backend upserts on track_uid
    #   drop                  log only
    late_result_policy: str = "republish_if_changed"
    max_results_per_track: int = 50

    # --- face only -------------------------------------------------------
    # annotate: attach the liveness verdict, never act on it (previous
    #           behaviour).
    # reject:   a track judged `fake` publishes identity "0" with
    #           spoof_rejected=true and stops being re-queried.
    liveness_policy: str = "annotate"

    extra: Dict[str, str] = field(default_factory=dict)


# The plate policy — edit here.
_MODULE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "plate": dict(
        satisfied_conf=0.85,             # a VALID plate this confident ends re-querying
        consensus_min=3,                 # ... or this many agreeing valid results (0 = off)
        trigger_max_wait_sec=3.0,        # trigger held this long when no crop could be sent
        trigger_task_wait_sec=15.0,      # ... or until its task's result while it is queued
        periodic_interval_sec=3.0,       # periodic re-query cadence (cond_per_trig cameras)
        periodic_first_delay_sec=1.0,
        finalize_timeout_sec=30.0,       # upper bound for the final record after track end
        final_min_seen_frames=8,         # final gate (dropped unless something was published)
        final_min_crops=1,
        track_stale_sec=120.0,           # a track the hub stops hearing about is ended after
        late_result_grace_sec=300.0,
        late_result_policy="republish_if_changed",   # | drop
    ),
}


def module_config(module: str) -> ModuleConfig:
    return ModuleConfig(module=module, **_MODULE_DEFAULTS.get(module, {}))
