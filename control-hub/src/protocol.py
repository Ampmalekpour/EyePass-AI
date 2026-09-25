"""
protocol.py (control hub)
--------------------------------------------------------------------
The wire contract between the detectors, the recognizer / OCR
workers, and the control hub. The detector side of the same contract
lives in facecore/hub.py and platecore/hub.py (kept in sync by hand —
same pattern as the rest of each module's shared library).

Redis keys, per module (`{m}` = face | plate):

  {m}:internal:hub:events        STREAM   detector engines -> hub
  {m}:internal:hub:results       STREAM   recognizer / OCR workers -> hub
  {m}:internal:hub:ctl:{engine}  LIST     hub -> one detector engine
  {m}:internal:hub:track:{uid}   STRING   hub's per-track checkpoint
  {m}:internal:hub:leader        STRING   leader lease (one active hub)
  {m}:internal:hub:heartbeat     STRING   liveness, 30s TTL

Every stream entry is two fields: `kind` and `data` (a JSON object).
Every event carries `uid` — a GLOBAL track id minted by the detector
when the track is born (`{camera}-{engine}-{random}`), unique across
engine restarts and rebalances, unlike BYTETrack's per-engine int id.

--------------------------------------------------------------------
detector -> hub (hub:events)
--------------------------------------------------------------------
engine_started  {engine_id, boot_id}
                A detector engine (re)started. Every live track the hub
                still holds for that engine under a different boot_id is
                ended ("engine_restarted") instead of waiting to go stale.

track_started   {uid, camera_id, track_id, engine_id, boot_id,
                 video_source, triggers: {periodic, <trigger>..., leave_scene}}

track_update    {uid, seen_frames, n_crops, duration_frames, liveness?}
                Low-rate heartbeat (TRACK_UPDATE_INTERVAL_SEC).

submitted       {uid, task_id, stage, n_crops}
                A task was pushed to the worker queue for this track.

trigger         {uid, event, detail: {direction, point, confidence, ...},
                 task_id | null}
                A spatial trigger fired on a camera that has it enabled.
                task_id is the task the detector submitted for it in the
                same frame (null if it could not / did not need to).

track_ended     {uid, reason, seen_frames, n_crops, duration_frames,
                 liveness?, finalize_task_id | null}
                reason: absent | camera_removed | engine_stopped
                The detector forgets the track right after emitting
                this; everything after is the hub's job.

--------------------------------------------------------------------
workers -> hub (hub:results)
--------------------------------------------------------------------
result          {uid, task_id, stage, camera_id, track_id, status,
                 is_valid, confidence, ...module fields}
                Always emitted for a hub task, including "skipped" /
                "error", so the hub never waits on a task that will
                not answer.

--------------------------------------------------------------------
hub -> detector (hub:ctl:{engine_id}, JSON per LIST item)
--------------------------------------------------------------------
{"action": "result",  "uid", "task_id", "satisfied", "display"}
{"action": "request", "uid", "stage"}          # e.g. periodic re-query
{"action": "state",   "uid", "satisfied", "display"}
--------------------------------------------------------------------
"""

EVENTS_STREAM = "hub:events"
RESULTS_STREAM = "hub:results"

K_ENGINE_STARTED = "engine_started"
K_TRACK_STARTED = "track_started"
K_TRACK_UPDATE = "track_update"
K_SUBMITTED = "submitted"
K_TRIGGER = "trigger"
K_TRACK_ENDED = "track_ended"
K_RESULT = "result"

A_RESULT = "result"
A_REQUEST = "request"
A_STATE = "state"


def keys(module: str) -> dict:
    p = f"{module}:internal:hub"
    return {
        "events": f"{p}:events",
        "results": f"{p}:results",
        "ctl_prefix": f"{p}:ctl:",
        "track_prefix": f"{p}:track:",
        "leader": f"{p}:leader",
        "heartbeat": f"{p}:heartbeat",
    }
