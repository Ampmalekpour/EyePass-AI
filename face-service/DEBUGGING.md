# DEBUGGING — visual debug output, structured logs, and every tunable

This is the reference for everything this revision added to help you
watch and understand what the system is actually doing: the on-disk
debug tree (one folder per section, all bind-mounted to your host),
what each image/video shows, the structured decision/timing logs, and
where every runtime parameter now lives in `.env`.

Nothing here changes pipeline behaviour by default except two things
that were bugs before this revision (see "Bugs fixed along the way" at
the bottom) — every debug writer is either off by default or, where
it's genuinely cheap (enrollment images, the annotated video), was
already on by default before this revision.

## 1. The debug folder tree

Everything lives on the HOST under `DEBUG_ROOT_HOST_DIR` (`.env`,
default `./debug`), one subfolder per section — no docker-managed
named volumes anywhere in this module any more, so every file below is
just... a file on your disk, next to the code.

```
debug/                                    ($DEBUG_ROOT_HOST_DIR)
  camera-service/
    camera_status.log                     one line per camera health check (see §2)
    snapshots/
      camera_<id>/
        <timestamp>.jpg                   optional — CAMERA_SNAPSHOT_DEBUG_ENABLED=true

  detector/                               ($DEBUG_VIDEO_HOST_DIR — nests here by default)
    camera_<id>/
      seg_<timestamp>.mp4                 rolling annotated video (see §3)
      seg_<timestamp>.jsonl               one JSON line per frame written to that segment
    legacy_raw_output/
      camera_<id>/...                     SAVE_OUTPUT's raw (non-annotated) dump — off by default
    best_crop_montages/
      camera_<id>/
        trk<id>_<event>_<timestamp>.jpg   candidate ladder for one submitted crop (see §4)
    liveness_rejects/
      camera_<id>/
        trk<id>_<timestamp>.jpg           one still per FAKE liveness verdict (see §5)

  recognizer/                             (mounted at /data in the container)
    live/
      debug_landmarked_crops/             crop with 5-pt landmarks burned in, pre-alignment
      debug_aligned_faces/                the exact 112x112 tensor fed to AdaFace
      debug_matches/
        trk<id>_<timestamp>_pid<p>_conf<c>.jpg   crop -> aligned -> top-K gallery matches (see §6)
      debug_rejected/
        trk<id>_<timestamp>.jpg           same composite, saved on Unknown (see §6)
    enroll/
      pose_checks/
        flag<f>_<verdict>_<timestamp>.jpg pose-check overlay per enroll snapshot (see §7)
      commits/
        personnelid<id>_<timestamp>.jpg   3-photo enrollment card per completed enrollment (see §7)
    recognition_logs.txt                  pre-existing plain-text recognition log (unchanged)
```

Every one of these directories is created on first write and
self-prunes to a `*_MAX_FILES` cap (newest-kept, see `prune_dir()` in
`common/facecore/debugging.py`) — nothing here grows unbounded. The
annotated video prunes by segment count instead (`DEBUG_VIDEO_MAX_SEGMENTS`).

Pipeline data (camera frames, the gallery db, submitted crops) still
goes to MinIO exclusively, same as before this revision — nothing
under `debug/` is ever uploaded anywhere, and nothing from MinIO is
ever written under `debug/`. If you're looking for a specific person's
enrollment photos as they exist in the gallery, that's MinIO's
`GALLERY_MINIO_PREFIX`, not this tree — this tree is only "what did
the system see/decide, and when."

## 2. camera-service

Pre-existing (unaffected by this revision): each health-check tick logs
one line to `CAMERA_STATUS_LOG_PATH` — connect/disconnect transitions,
mediamtx registration outcome, RTSP reachability. Every interval/timeout
that drives this loop (`MTX_POLL_INTERVAL_SEC`, `MONITOR_LOOP_INTERVAL_SEC`,
`CAMERA_PING_TIMEOUT_SEC`, `OFFLINE_HOLD_SECONDS`, ...) is now in `.env`
— see `camera-service/src/main.py`'s own comments for exactly which
hardcoded literal each one used to be.

New, off by default: `CAMERA_SNAPSHOT_DEBUG_ENABLED=true` grabs one
JPEG per camera every `CAMERA_SNAPSHOT_INTERVAL_SEC` via `ffmpeg -rtsp_transport
tcp -i <url> -frames:v 1 ...` (subprocess, fails soft if `ffmpeg` isn't
installed — see the commented-out `apt-get install ffmpeg` line in
`camera-service/Dockerfile`). Answers "is this camera's actual RTSP
feed healthy and pointed where I think it is" without touching the
detector at all — useful when a camera looks "connected" in
`camera_status.log` but nothing downstream is producing tracks.

## 3. Detector — annotated video (pre-existing, `debug_recorder.py`)

This was already the most thorough visual debug in the system before
this revision — rolling per-camera MP4s with detections, tracks,
landmarks, pose, crop quality scores, liveness verdict, and a
scrolling event log burned into a side panel. One addition this
revision makes to it:

**Recognition round-trip latency.** Every track's `pending_since`
(already recorded the moment a crop is submitted for recognition) is
now used to compute `rec_latency_ms` the moment a result comes back,
and it's appended to that track's HUD line as `rtt <N>ms`. This
answers a question the video alone couldn't before: is a track sitting
in "pending" because recognition is slow, or because no crop good
enough to submit has shown up yet. If `rtt` is consistently high across
many cameras, that's the recognizer pool (queue depth, worker count,
GPU contention) — if it's low but tracks still sit pending, that's the
detector's own quality gate.

Every one of `DEBUG_VIDEO_SEGMENT_SECONDS/FPS/MAX_SEGMENTS/SCALE/EVERY_N/
PANEL_WIDTH/CODEC/GHOST_FRAMES/EVENT_LINES/DRAW_LANDMARKS/JSONL` is in
`.env`. **Important:** `DebugConfig` (in `debug_recorder.py`) reads
these env vars directly, at its own construction — the copies of these
names in `detector/src/config.py` are documentation only, kept there so
you can find the full list in one place; changing `config.py`'s copy
has no effect, only `.env` does.

## 4. Detector — best-crop montage (new, off by default)

`DEBUG_BEST_CROP_MONTAGE_ENABLED=true`. Every time the detector submits
a crop for recognition (periodic re-check, a line-cross/stopped-ROI
event, or track finalization), this saves a labelled grid of **every**
candidate currently held in that track's reg1/reg2/reg3 quality ladder
— not just the one that got sent. Each cell shows its multi-landmark-
confidence score, yaw group, and resolution; the cell that was actually
submitted is marked `-> SENT`.

This is the answer to "why did it send THAT crop" — the annotated
video shows you the current best-known score, but not the runners-up
it was chosen over. If recognition keeps missing on a camera with
otherwise-good video, check this folder first: often the ladder is
full of quarter-profile or low-resolution candidates and the "best" one
still isn't good enough.

## 5. Detector — liveness rejects (new, off by default)

`DEBUG_LIVENESS_REJECTS_ENABLED=true`. One still, saved the instant a
track's liveness verdict flips to `fake` — the crop plus the exact
metrics that produced the verdict (`planar_residual`, `ring_follow`,
`rigid_residual`, `pose_delta_deg`, the reason string). Spoof attempts
are usually a handful of frames inside a 30-second rolling video
segment and easy to scrub past; this keeps a standing, chronological
folder of just the rejects so you can eyeball whether your
`LIVENESS_*` thresholds (also now in `.env`) are too strict, too loose,
or correct, without re-watching video.

## 6. Recognizer — live-path match/reject visualization (new, off by default)

`DEBUG_SAVE_MATCHES=true` saves, for every **non**-Unknown decision, a
composite: input crop → aligned 112×112 tensor (literally what AdaFace
saw) → the top `DEBUG_MATCHES_TOP_N` gallery thumbnails, each labelled
with its raw cosine similarity and person id, the winning one marked
`<-- BEST`. `DEBUG_SAVE_REJECTED=true` does the same for Unknown
verdicts (top-3 near-misses instead). Filenames embed the decision
outcome (`pid<id>_conf<c>`) so a folder listing alone tells you
roughly what happened without opening anything.

Both are off by default because they cost one extra image write per
recognition decision — turn one on while chasing a specific camera or
person, then back off. This is the fastest way to answer "it matched
the wrong person" or "it should have matched but didn't": you see the
exact aligned tensor and the exact runner-up score, not just a
confidence number in a log line.

## 7. Recognizer — add-face / enrollment debug (new, ON by default)

Unlike the live-path writers above, `DEBUG_ENROLL_ENABLED=true` by
default — enrollment is low-volume (a handful of images per new
person, not a continuous stream) and these are the images support
most often needs when someone reports "enrollment keeps rejecting my
photo" or "the wrong person got enrolled."

- **Pose-check overlay** (`enroll/pose_checks/`): every snapshot sent
  through `verify_pose()` renders a debug image — detection box in
  green/red for pass/fail, the 5 landmarks, and three separate text
  lines (flag + expected yaw/pitch window, the measured yaw + its
  status, the measured pitch + its status — kept on separate lines
  deliberately, since a rejection is almost always one axis, not both,
  and conflating them into one line hides which axis to fix). Saved for
  every attempt, pass or fail, so a support conversation can point at
  the exact frame instead of asking the person to "try again and see."
- **Enrollment card** (`enroll/commits/`): one composite per completed
  enrollment — the exact three crops that were written into the
  gallery, titled with the person's name, id, section/codeid, and the
  `c<N>.jpg` range they were assigned. A standing audit trail you can
  scroll without opening `brieface.db` or the MinIO console.

`verify_pose()`'s debug image is popped out of the result dict
**before** it's pickled and pushed onto `cmd:enroll:response:*` — it
never reaches the documented backend wire contract in `ADD_FACE.md`,
it only ever touches disk on the recognizer container.

## 8. Structured logging (`LOG_FORMAT`)

`LOG_FORMAT=text` (default) is unchanged: `%(asctime)s | %(levelname)s
| %(name)s | %(message)s`. `LOG_FORMAT=json` switches every service to
one JSON object per line — `ts`, `level`, `logger`, `message`, plus
whatever structured fields a call site attached via `logger.debug(msg,
extra={"fields": {...}})`. Same call sites, same messages either way —
flip it on once you have somewhere to ship logs to (Loki/ELK/CloudWatch/
whatever), nothing else changes. See `common/facecore/logging_setup.py`.

### The decision trace (`LOG_DECISION_TRACE`, recognizer)

The thing this section exists for. With `LOG_DECISION_TRACE=true` (the
default) and `LOG_LEVEL=DEBUG`, every recognition decision in
`recognition_engine.py::_decide_person_and_confidence` logs one
structured event:

```json
{
  "event": "recognition_decision",
  "top_k": 12, "alpha": 0.55, "tau": 0.05,
  "raw_similarities_considered": [{"pid": "3", "score": 0.7421, "path": "3_c1.jpg"}, ...],
  "fused_scores_by_identity": [{"pid": "3", "fused_score": 0.6812}, ...],
  "top_raw_score": 0.7421,
  "tsallis_confidence_raw": 0.812,
  "confidence_final": 0.541,
  "penalty_applied": "high_low_raw",
  "best_pid": "3",
  "thresholds": {"up": 0.62, "mid": 0.45, "down": 0.3}
}
```

Every field on the tsallis-entropy confidence path is here: which raw
similarities were considered, the fused per-identity score used to
pick a winner, the tsallis confidence **before** any penalty, which
penalty (if any) fired and why (`"none" | "high_low_raw" | "mid" |
"below_down_thr->unknown"`), and the final confidence next to the
up/mid/down thresholds it was compared against. This is the single
place to look when a confidence number doesn't make sense — you no
longer have to reconstruct the tsallis math by hand from a bare
`confidence=0.54` log line.

It's gated by `logger.isEnabledFor(logging.DEBUG)` so building this
dict costs nothing at `LOG_LEVEL=INFO` — turn `LOG_LEVEL=DEBUG` on for
the recognizer (and `LOG_FORMAT=json` if you want to grep/jq it) when
you actually need it, and back off in normal operation since it's one
event per recognition.

### Embedding timing (`generate_embedding`)

At `LOG_LEVEL=DEBUG`, every embedding generation logs its own elapsed
time, active backend (`onnx`/`pytorch`), and the output embedding's
dimension — cheap, always-on-when-DEBUG timing that answers "is
inference itself slow, or is the queue backed up in front of it"
without needing the decision trace.

### Recognition round-trip latency (detector HUD + track dict)

Covered in §3 — `rec_latency_ms` is computed from `pending_since` and
now rides in `debug_tracks` alongside every other per-track field, so
it's also available to anything else reading the detector's debug
JSONL (`DEBUG_VIDEO_JSONL=true`), not just the video HUD.

## 9. Every parameter is now in `.env`

Every operational literal that used to be hardcoded anywhere in this
module — timeouts, retry counts, thresholds, tracker parameters, ONNX
thread counts, decision-math penalty multipliers — now has an env var
in `.env` with the exact value that was previously baked in, so
turning this revision on changes nothing until you actually edit a
value. `.env`'s comments say, section by section, what each one does
and (where it isn't obvious) which file/line it used to be a literal
in. A few things were deliberately left as literals, not env vars,
because they are model/math invariants rather than operational knobs:
the 3D anatomical reference points `compute_yaw()` solves `solvePnP`
against, AdaFace's fixed `(x/255.0 - 0.5)/0.5` input normalization, and
`REFERENCE_FACIAL_POINTS` (the alignment template AdaFace was trained
against) — changing any of these doesn't tune behaviour, it silently
breaks correctness, so they stay literals with a comment saying why.

Two pre-existing gaps this revision fixes as part of the same audit
(see "Bugs fixed along the way" below): `camera-service`'s
`OFFLINE_HOLD_SECONDS` was defined as a bare literal that silently
ignored the env var compose was already passing it, and `TrackerConfig`
was constructed with only `track_buffer` overridden — the other four
BYTETracker parameters were always the class's own hardcoded defaults,
invisible from `.env` or `compose.yaml` even though they clearly look
configurable there. Both now actually read from `.env`.

## 10. Other things worth watching for (beyond what's built here)

A few more ideas, roughly in order of "cheap to add if you find you
need it":

- **Redis queue depth / staleness as a metric, not just a log line.**
  Right now the only way to see whether the recognizer's task queue is
  backing up is to watch `rec_latency_ms` climb in the video HUD after
  the fact. A periodic line (or a tiny `/health`-adjacent endpoint)
  reporting current queue length per Redis list, plus the age of the
  oldest pending task, would catch a backlog forming before it shows up
  as visibly slow recognition.
- **A liveness score histogram/log, not just the reject stills.** §5
  only captures the moment a verdict flips to `fake` — the
  near-misses (scores just above the reject threshold) never get
  recorded anywhere. Logging the raw metrics on every evaluation
  (`LIVENESS_EVAL_EVERY_N_FRAMES`) at DEBUG, even for passes, would let
  you plot the score distribution and sanity-check the thresholds
  against real traffic instead of tuning blind with
  `calibrate_liveness.py` alone.
- **Gallery drift detector.** Nothing currently flags when
  `brieface.db`'s row count and the actual image files under a
  person's `c<N>.jpg` range disagree (a partial write, a manual file
  deletion, an interrupted enrollment). A startup or periodic
  consistency check logging any mismatch would catch a corrupted
  gallery before it manifests as confusing recognition results.
- **Per-engine/per-worker resource snapshot on a timer.** CPU/GPU
  memory and wall-clock FPS per engine subprocess, and queue-wait vs.
  compute time per recognizer worker, logged periodically at DEBUG —
  useful once you're tuning `MAX_CAMERAS_PER_ENGINE` /
  `DEFAULT_WORKER_COUNT` against real hardware instead of guessing.
- **A `request_id` / correlation id threaded end to end.** Right now a
  single face's journey — detector track → submitted crop → recognizer
  decision → (if applicable) enroll flow — has to be reassembled by
  timestamp and track/camera id across three separate log streams and
  folders. A single id attached at crop-submission time and echoed into
  every subsequent log line and filename (montage, match viz, HUD)
  would make "trace this one face's entire path" a single grep instead
  of manual correlation.
- **Video segment index / manifest.** `debug/detector/camera_<id>/`
  fills with `seg_<timestamp>.mp4` + matching `.jsonl` pairs but
  nothing indexes them — for a camera with many segments, a small
  `manifest.jsonl` (one line per segment: start/end time, track ids
  seen, any liveness rejects) would let you jump straight to the right
  segment instead of opening several to find one event.

None of these are implemented in this revision — they're the next
layer if/when you find yourself needing them.

## 11. Bugs fixed along the way

Two things were fixed as part of the "make every parameter
configurable" audit — not new behaviour, just making existing,
already-intended behaviour actually take effect:

1. **`camera-service/src/main.py`**: `OFFLINE_HOLD_SECONDS` was a bare
   `= 10` literal that never read `os.getenv("OFFLINE_HOLD_SECONDS")`,
   despite `compose.yaml` passing it in all along — changing it in
   `.env` silently did nothing. Now reads the env var like every other
   camera-service tunable.
2. **`detector/src/engine.py`**: `TrackerConfig` was constructed with
   only `track_buffer=35` overridden; `track_thresh`, `match_thresh`,
   `nms_thresh`, and `mot20` were always that class's own hardcoded
   defaults (`0.5`/`0.9`/`0.5`/`True` — which happen to match the new
   env defaults, so behaviour is unchanged), with no way to see or
   change them from outside the code. All five are now
   `TRACKER_*` env vars in `.env`.

## 12. Quick recipes

**"Recognition on camera 4 keeps saying Unknown for someone who should
be known"**
```
DEBUG_BEST_CROP_MONTAGE_ENABLED=true   # detector: is a good-enough crop even being sent?
DEBUG_SAVE_REJECTED=true               # recognizer: what were the top-3 near-misses?
LOG_LEVEL=DEBUG
LOG_DECISION_TRACE=true                # recognizer: exact tsallis math for the decision
```
Restart both services, reproduce, then check
`debug/detector/best_crop_montages/camera_4/` and
`debug/recognizer/live/debug_rejected/` in timestamp order together
with the recognizer's decision-trace log lines.

**"Enrollment keeps rejecting this person's photos"**
`DEBUG_ENROLL_ENABLED` is already on by default — just look in
`debug/recognizer/enroll/pose_checks/` for the `flag<f>_rejected_*.jpg`
files from their attempts; the burned-in yaw/pitch status tells you
which axis failed and by how much.

**"Is a camera actually producing a healthy RTSP feed?"**
```
CAMERA_SNAPSHOT_DEBUG_ENABLED=true
```
(needs `ffmpeg` — see the commented line in `camera-service/Dockerfile`)
and watch `debug/camera-service/snapshots/camera_<id>/`.

**"Recognition feels slow but I don't know where"**
Watch the annotated video's `rtt <N>ms` HUD line (§3) for a quick
signal, then flip `LOG_LEVEL=DEBUG` on the recognizer to see per-embedding
`elapsed_ms` (§8) and confirm whether it's inference itself or queueing
in front of it.
