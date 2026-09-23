# DEBUGGING — visual debug output, structured logs, and every tunable

This is the plate module's counterpart to face_service_'s own
DEBUGGING.md — same structure, same idea, applied to ALPR instead of
face recognition: `plate_detector` (YOLO detect + BYTETrack + spatial
triggers, dispatches OCR tasks) and `plate_ocr` (car/motorcycle
PaddleOCR pipeline, char-by-char voting, plate-format validation).

Everything here is off by default except the structured decision trace
and the debug-video JSONL sidecar — turning on the visual extras (debug
video, the two new montage systems, camera snapshots) has a real cost
(disk, CPU for JPEG/video encoding), so they're opt-in per environment.

---

## 1. The debug folder tree

Three services now write to their own bind-mounted `/debug` — nothing
in this module writes debug output to MinIO; MinIO is pipeline data
only (plate/vehicle crops, best-frame images — see the README's
storage split). Default host paths (all overridable in `.env`):

```
./debug_camera/                 CAMERA_DEBUG_HOST_DIR   (camera_stream:/debug)
  camera_stream/
    camera_status.log           online/offline transitions
    snapshots/<camera_id>.jpg   optional, CAMERA_SNAPSHOT_DEBUG_ENABLED

./debug_video/                  DEBUG_VIDEO_HOST_DIR    (plate_detector:/debug)
  <camera_id>/
    seg_*.mp4                   annotated rolling video (pre-existing)
    seg_*.jsonl                 one JSON event per detection/trigger/OCR event
  ocr_submissions/               DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED (new)
    <camera>_<track>_<stage>_<ts>.jpg

./debug_ocr/                    OCR_DEBUG_HOST_DIR      (plate_ocr:/debug)
  decisions/                    OCR_SAVE_DECISION_DEBUG (new)
    <camera>_<track>_<stage>_<ts>.jpg
```

All three are plain bind mounts (`./debug*` on the host by default) —
open them directly, no container exec needed. Every folder is bounded
(oldest files pruned once a `_MAX_FILES` cap is hit); the annotated
video keeps `DEBUG_VIDEO_MAX_SEGMENTS` segments per camera the same way
it always did.

---

## 2. camera-service

`camera-service/src/main.py` had two real bugs, both fixed:

* **Wrong Redis channel spelling.** It published camera online/offline
  transitions to `{module}:cameras:events` (plural) — but
  `common/platecore/keys.py`'s `RedisKeys.cameras_events` property
  (and its own docstring, which explicitly warns against "fixing" it
  to the plural) returns the **singular** `{module}:camera:events`,
  matching the real, already-deployed `eyepass-camera-stream`. Because
  of the mismatch, `plate_detector`'s subscriber
  (`backend_bridge.on_camera_event`) never received these events at
  all — a camera's online/offline state changes were silently dropped.
  Now fixed to the singular spelling, with a comment at the call site
  so it doesn't drift back.
* **`OFFLINE_HOLD_SECONDS` / `TRUST_MTX_ONLY` were bare literals**,
  never reading their env vars despite looking like config. Both are
  now real env vars (see `.env`'s CAMERA-STREAM SERVICE section).

New: `CAMERA_STATUS_LOG_PATH` moves `camera_status.log` under the
bind-mounted debug tree (it used to land at a bare relative path,
wherever the container's cwd happened to be — effectively lost on
container recreation). New: `CAMERA_SNAPSHOT_DEBUG_ENABLED` — off by
default — periodically saves one JPEG per camera via `ffmpeg`, purely
to answer "is this camera actually pointed at what I think it is."
Needs `ffmpeg` in the image; uncomment the `RUN apt-get install
ffmpeg` line in `camera-service/Dockerfile` to use it. A failure here
(ffmpeg missing, camera unreachable) only logs at DEBUG and never
affects camera registration/health tracking.

---

## 3. Detector — annotated video (pre-existing, `debug_recorder.py`)

Unchanged mechanism, same as before: `DEBUG_VIDEO_ENABLED=true` writes
a rolling, fully annotated MP4 per camera — boxes, tracks, trigger
zones, per-track quality gates, and now also the **resolved OCR
round-trip latency** once a result comes back (see §6). A JSONL
sidecar (`DEBUG_VIDEO_JSONL`, on by default) logs one line per
detection/trigger/OCR event alongside the video, for grepping without
scrubbing footage.

Five of `DebugConfig`'s own env vars (`DEBUG_VIDEO_JSONL`,
`_GHOST_FRAMES`, `_EVENT_LINES`, `_TRAIL`, `_PANEL_WIDTH`) are read
directly by `debug_recorder.py`'s `DebugConfig` — **not** through
`config.py`. They're declared in `config.py`'s Section 6 too (with a
comment saying so) purely so every `DEBUG_VIDEO_*` knob is
discoverable in one file; setting the value in `config.py` itself does
nothing, only the env var does. Same pattern as the face module's own
`DebugConfig`.

---

## 4. Detector — OCR-submission montage (new, off by default)

`DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED=true` saves one labelled-grid
JPEG every time a track is submitted to OCR (`engine.py`'s
`_submit_track_to_ocr`, via the new `detector/src/debug_extras.py`) —
every crop that went into the task, plus the best-frame if one was
attached, each labelled with its detection score/resolution/frame
number. This is the fastest way to answer "why did OCR get a bad
crop" without scrubbing the annotated video for the right timestamp.
Filenames embed the task_id's own nanosecond timestamp, so they sort
chronologically and tie back to a specific `[OCR-SUBMIT]` log line
unambiguously.

---

## 5. OCR service — decision montage (new, off by default)

`OCR_SAVE_DECISION_DEBUG=true` saves one labelled-grid JPEG per
finalized OCR task (`worker.py`'s `_process_task`, via the new
`ocr_service/src/debug_extras.py`) — every crop, each crop's raw OCR
candidate text + confidence, which candidate won the vote, and the
final validation verdict, all in the title bar. Previously the OCR
service had **no local debug output at all** — this and the decision
trace below (§6) are the only way to see *why* a plate was read a
particular way without re-deriving it from log lines.

---

## 6. Structured logging (`LOG_FORMAT`)

`LOG_FORMAT=json` (default `text`) switches every service's logger
(`common/platecore/logging_setup.py`, and camera-service's own
duplicate of the same formatter, since camera-service can't import
platecore) to one JSON object per line — `ts`, `level`, `logger`,
`message`, plus whatever structured fields a call attached via
`logger.debug(msg, extra={"fields": {...}})`. Meant for feeding a log
aggregator without writing a second parser for this system's own log
format.

### The decision trace (`LOG_DECISION_TRACE`, ocr_service)

On by default. Every finalized OCR task logs one DEBUG-level line via
`extra={"fields": {...}}` with: `task_id`, `camera_id`, `track_id`,
`trigger_type`, `queue_latency_ms` (see below), `voted_class`,
`n_crops`, every OCR candidate + confidence, the winning vote, the
validation verdict + reason, and the normalized/compact plate text.
In `LOG_FORMAT=json` this is one grep-able/query-able record per
decision; in `text` mode it's a plain DEBUG line (set `LOG_LEVEL=DEBUG`
to see it).

### OCR latency — two numbers, two different questions

* **Queue latency** (`_queue_latency_ms(task_id)`, already existed in
  `worker.py`) — time from task submission to a worker actually
  picking it up. Derived from the nanosecond timestamp embedded in the
  task_id itself (`f"{engine}:{camera}:{track}:{trigger}:{time.time_ns()}"`),
  logged on every `[OCR-%s] [TASK-START]` line and in the decision
  trace above. Answers "is the OCR queue backed up."
* **Round-trip latency** (`last_ocr_latency_ms`, new) — time from
  `engine.py` submitting a track to OCR
  (`ocr_pending_submitted_at`, already existed) to the detector
  actually receiving and clearing the result
  (`_handle_ocr_result`). Logged in `[OCR-RESULT]`, shown in the debug
  video HUD as `rtt Nms` right where `OCR PENDING ...` was before the
  result came back (both the per-track label stack and the side
  panel), and surfaced in `_dbg_log`'s `OCR_LATENCY` JSONL event.
  Answers "is the full round trip (queue wait + actual OCR processing
  + network) slow," which queue latency alone can't distinguish from
  "OCR processing itself is slow."

---

## 7. Every parameter is now in `.env`

Everything that was a bare literal anywhere in this codebase — a
timeout, a retry count, a buffer cap, a debug toggle — is now an env
var with the exact same default, so an unset `.env` reproduces prior
behavior bit-for-bit. The highest-value fix in this pass:

**`tracker.py`'s full `PlateTrackerConfig` surface was completely
unreachable.** `engine.py` used to define its own tiny local
`TrackerConfig` (`track_thresh`/`match_thresh`/`track_buffer`/
`nms_thresh`/`mot20` only) and pass *that* to `BYTETracker` — even
though `tracker.py` ships a much richer `PlateTrackerConfig` with ~19
additional fields (the "FIX 1-7" enhancements: young-track survival,
new-track motion seeding, a recovery pass for briefly-lost tracks, GMC
camera-jolt compensation). `BYTETracker.__init__` reads every field via
`getattr(args, name, default)`, so the missing fields were always
silently falling back to `PlateTrackerConfig`'s own hardcoded
defaults — **current behavior never changed**, but none of those ~19
knobs were configurable. `engine.py` now imports `PlateTrackerConfig`
directly (`build_tracker_config()`) and every `TRACKER_*` default in
`.env`/`config.py` was verified, field-by-field, against
`PlateTrackerConfig`'s own declared defaults (see the test run in the
delivery notes) — they match exactly.

Other sections newly surfaced to `.env` that previously had no env
var at all (see `.env`'s comments for what each one does):
`TRIGGER_POSITION_HISTORY_MAX`/`_CONFIDENCE_HISTORY_MAX`/
`_VELOCITY_WINDOW` (triggers.py's history buffers), the full
`RTSP_*` block (rtsp_reader.py's timeouts/backoffs), `SLOW_BATCH_WARN_MS`
/ `PIPELINE_STATS_LOG_INTERVAL_SEC` (engine.py's run loop),
`HEARTBEAT_INTERVAL_SEC`/`_TTL_SEC` (detector + ocr_service + already
existing in camera-service), `OCR_CLAHE_CLIP_LIMIT`/`_GRID_SIZE`
(car-plate contrast preprocessing — a genuine per-site tunable, unlike
the model-input resize sizes right next to it, which must stay
literal), `OCR_MOTOR_MIN_BOX_AREA_RATIO` (motorcycle box-noise
filter), `OCR_IDLE_POLL_INTERVAL_SEC`/`_TASK_POP_RETRY_BACKOFF_SEC`
(worker.py's run loop), and the `QUALITY GATES` / `TRACK AGGREGATION`
sections of `detector/src/config.py`, which existed in code already
but were never wired into `.env`/`compose.yaml` at all.

**Deliberately kept as literals** (model/format invariants, not
operational tunables — changing them silently changes recognition
behavior): PaddleOCR model input resize dimensions (256×64 car,
240×200 motorcycle), the Persian-letter plate-format regex and
8-char/3+5-digit validation constants, the Persian/Arabic digit
translation table, the MinIO object-key layout in
`save_prep_for_mysql` (Django's `PrivateMediaStorage` resolves these
keys directly), and `cv2.CAP_PROP_BUFFERSIZE=1` in `rtsp_reader.py`
(a "keep only the latest frame" design invariant, not a tunable).

---

## 8. Other things worth watching for (beyond what's built here)

* **OCR service dependency gap (fixed, verify on your base image).**
  `ocr_service/requirements.txt` previously listed only
  `redis`/`boto3`/`shapely`/`pyclipper` — none of `paddlepaddle`,
  `paddleocr`, `opencv`, `numpy` or `Pillow`, despite `worker.py`
  importing `cv2`, `numpy` and `paddleocr` directly. This only worked
  by accident of whichever `BASE_IMAGE` you happened to build from
  already providing them. Now listed explicitly, but the pinned
  versions are a best-effort placeholder — verify `paddlepaddle`/
  `paddleocr` version compatibility with your actual exported
  `PadOcr/` model files before deploying; PaddleOCR inference-model
  loading can be strict about this. Prefer copying the exact pins from
  wherever your existing `ocr_worker.py` deployment already runs
  successfully.
* **`ocr_service/Dockerfile` vs its own comment.** The comment (and
  the README) said this service builds from a small, CPU-only base
  image; the actual `FROM` line used the same heavy
  torch/ultralytics/CUDA base as `plate_detector`. New `OCR_BASE_IMAGE`
  env var decouples the two — unset, it defaults to exactly the prior
  (contradictory-but-working) value, so this is zero-risk as shipped.
  Switch it once you've verified the (now-explicit) requirements
  install cleanly on a lighter base.
* **README says this module has no mediamtx/camera_stream; compose.yaml
  ships both.** Left as-is (out of scope for this pass) — just be aware
  `docker compose up -d --build` with no profile flags DOES start a
  local `mediamtx` + `camera_stream`, which is the copy the channel-name
  bug above was in.
* **Named docker volume removed.** `compose.yaml` used to mount a
  `detector_data:/data` named volume "for durable local state" — a
  full grep of `detector/src` and `ocr_service/src` found nothing that
  ever writes to `/data`; the only use is Ultralytics' own
  `YOLO_CONFIG_DIR` cache. Self-healing reads its checkpoint from
  Redis, never from this volume. Removed; `/data` is now just
  ephemeral, rebuildable cache inside the container.
* **Queue-depth / OCR backlog.** `platecore.keys.RedisKeys.ocr_tasks_pending_gauge`
  already exists as a key — nothing currently publishes to it. If OCR
  ever becomes the bottleneck (watch `queue_latency_ms` in the decision
  trace trend upward), wiring a periodic gauge update there would be
  the natural next debugging aid, mirroring the face module's own
  "queue-depth/staleness metric" suggestion.
* **Per-engine / per-worker resource snapshots.** Like the face
  module, nothing here currently logs per-subprocess memory/CPU. The
  health endpoint's `topology`/`pool` snapshots give counts, not
  resource usage — a `resource.getrusage` line on each
  `PIPELINE_STATS_LOG_INTERVAL_SEC` tick (detector) and each
  `WATCHDOG_INTERVAL_SEC` tick (OCR pool) would be a natural, low-cost
  addition if a memory leak is ever suspected.
* **End-to-end correlation id.** `task_id` already threads a
  detector-generated id through OCR submission, the decision trace,
  and the debug montages — but there's no single id tracing a plate
  all the way from `[TRIGGER]` through `[OCR-SUBMIT]` to the final
  `[PUBLISH]` to the backend across multiple OCR attempts (a track can
  submit more than once — cross_line, stop_roi, leave_scene). If you
  need that, `track_id` + `camera_id` together are already a stable
  enough compound key to `grep` a track's full lifecycle out of a
  `LOG_FORMAT=json` log stream today.

---

## 9. Bugs fixed along the way

1. **camera-service published to the wrong Redis channel** (plural
   `cameras:events` vs the required singular `camera:events`) — the
   detector's camera online/offline subscriber never received events
   from this shipped camera-service copy at all. Fixed.
2. **`OFFLINE_HOLD_SECONDS`/`TRUST_MTX_ONLY` were bare literals**,
   never configurable despite looking like config. Fixed.
3. **`camera_status.log` wrote to a bare relative path** — landed
   wherever the container's cwd happened to be, not on any bind mount,
   so it was effectively lost on every container recreation. Fixed —
   now under the bind-mounted debug tree.
4. **~19 of `tracker.py`'s `PlateTrackerConfig` fields were dead code**
   from `engine.py`'s perspective — see §7 for the full explanation.
   Current behavior was provably unchanged (verified field-by-field
   against `BYTETracker`'s own fallback defaults); now all configurable.
5. **`ocr_service/requirements.txt` was missing its own core
   dependencies** (paddlepaddle, paddleocr, opencv, numpy, Pillow) —
   only worked by accident of the chosen base image. Fixed, with a
   caveat about verifying the pinned versions (see §8).
6. **`ocr_service/Dockerfile`'s base image contradicted its own
   comment** (and the README) about not needing the heavy GPU base
   image. Decoupled via `OCR_BASE_IMAGE`, default-preserving. See §8.
7. **A dead `detector_data:/data` named docker volume** — nothing in
   this module's code ever wrote to it. Removed.

---

## 10. Quick recipes

**"I want to see everything, cheaply, on one camera while I test":**
```env
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_DECISION_TRACE=true
DETECTOR_DEBUG_VIDEO_ENABLED=true
DEBUG_VIDEO_EVERY_N=1
DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED=true
OCR_SAVE_DECISION_DEBUG=true
```

**"Production, log aggregator, minimal disk use":**
```env
LOG_FORMAT=json
LOG_DECISION_TRACE=true
DETECTOR_DEBUG_VIDEO_ENABLED=false
DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED=false
OCR_SAVE_DECISION_DEBUG=false
CAMERA_SNAPSHOT_DEBUG_ENABLED=false
```

**"A specific plate read looks wrong — how do I find out why":**
1. Find its `track_id`/`camera_id` from the backend result or the
   annotated video's on-screen label.
2. `grep` the OCR service's log (or query, in `LOG_FORMAT=json`) for
   that `track_id` in a `[DECISION-TRACE]` line — shows every OCR
   candidate, the vote, and the validation verdict for that task.
3. If `OCR_SAVE_DECISION_DEBUG=true`, open
   `./debug_ocr/decisions/<camera>_<track>_<stage>_*.jpg` for the
   actual crop(s) and candidate text side by side.
4. If `DEBUG_OCR_SUBMISSION_MONTAGE_ENABLED=true`, cross-check against
   `./debug_video/ocr_submissions/<camera>_<track>_<stage>_*.jpg` to
   see exactly what the detector sent — a bad read is often a bad crop
   (motion blur, wrong angle), not a bad OCR vote.
5. Check `rtt Nms` in the debug video HUD or the `OCR_LATENCY` JSONL
   event — a very slow round trip alongside a low-confidence read can
   point at a backlogged OCR pool rather than a genuinely hard plate.
