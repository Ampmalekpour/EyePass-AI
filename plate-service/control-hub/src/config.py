"""
config.py (plate control hub)
--------------------------------------------------------------------
Every knob the control hub understands, read once from the
environment.

Two layers:

  * SERVICE-wide settings (Redis, which modules to run, health port,
    stream/consumer names). Plain env names.

  * PER-MODULE policy settings. Each is read as `{MODULE}_{NAME}`
    (here PLATE_SATISFIED_CONF, PLATE_FINALIZE_TIMEOUT_SEC, ...) and
    falls back to the default below when unset.

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


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


# ====================================================================
# Service-wide
# ====================================================================
# This hub serves ONE module — the one it ships with (plate-service/). It
# reads the same REDIS_MODULE the module's other services use, so its
# keys always match theirs.
HUB_MODULE = _env("REDIS_MODULE", "plate")
HUB_MODULES: List[str] = [HUB_MODULE]

HEALTH_PORT = _int("HUB_HEALTH_PORT", 8020)

# Consumer group on both inbound streams. One group per module stream;
# a fixed consumer name so a restarted hub re-reads its own pending
# (read-but-not-acked) entries instead of orphaning them.
CONSUMER_GROUP = _env("HUB_CONSUMER_GROUP", "control-hub")
CONSUMER_NAME = _env("HUB_CONSUMER_NAME", "hub")

READ_BLOCK_MS = _int("HUB_READ_BLOCK_MS", 200)
READ_COUNT = _int("HUB_READ_COUNT", 200)
TICK_INTERVAL_SEC = _float("HUB_TICK_INTERVAL_SEC", 0.2)

# Leader lease: only one hub instance may own a module at a time (its
# track state is in memory). A second instance of the same container
# simply waits as a hot standby and takes over when the lease expires.
LEASE_TTL_SEC = _float("HUB_LEASE_TTL_SEC", 15.0)
LEASE_RENEW_SEC = _float("HUB_LEASE_RENEW_SEC", 5.0)

# Track-state checkpoints (one Redis string per live track) expire on
# their own even if the hub never deletes them.
STATE_TTL_SEC = _int("HUB_STATE_TTL_SEC", 3600)

# ctl lists (hub -> detector engine) expire when an engine goes away.
CTL_TTL_SEC = _int("HUB_CTL_TTL_SEC", 120)


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


_MODULE_DEFAULTS: Dict[str, Dict[str, object]] = {
    "plate": dict(satisfied_conf=0.85, consensus_min=3),
}


def module_config(module: str) -> ModuleConfig:
    base = ModuleConfig(module=module, **_MODULE_DEFAULTS.get(module, {}))
    p = module.upper() + "_"
    return ModuleConfig(
        module=module,
        satisfied_conf=_float(p + "SATISFIED_CONF", base.satisfied_conf),
        consensus_min=_int(p + "CONSENSUS_MIN", base.consensus_min),
        trigger_max_wait_sec=_float(p + "TRIGGER_MAX_WAIT_SEC", base.trigger_max_wait_sec),
        trigger_task_wait_sec=_float(p + "TRIGGER_TASK_WAIT_SEC", base.trigger_task_wait_sec),
        periodic_interval_sec=_float(p + "PERIODIC_INTERVAL_SEC", base.periodic_interval_sec),
        periodic_first_delay_sec=_float(p + "PERIODIC_FIRST_DELAY_SEC", base.periodic_first_delay_sec),
        finalize_timeout_sec=_float(p + "FINALIZE_TIMEOUT_SEC", base.finalize_timeout_sec),
        final_min_seen_frames=_int(p + "FINAL_MIN_SEEN_FRAMES", base.final_min_seen_frames),
        final_min_crops=_int(p + "FINAL_MIN_CROPS", base.final_min_crops),
        track_stale_sec=_float(p + "TRACK_STALE_SEC", base.track_stale_sec),
        late_result_grace_sec=_float(p + "LATE_RESULT_GRACE_SEC", base.late_result_grace_sec),
        late_result_policy=_env(p + "LATE_RESULT_POLICY", base.late_result_policy).lower(),
        max_results_per_track=_int(p + "MAX_RESULTS_PER_TRACK", base.max_results_per_track),
        liveness_policy=_env(p + "LIVENESS_POLICY", base.liveness_policy).lower(),
    )
