# Control hub — one owner for every track's recognition state

The face and plate pipelines both used to keep a track's recognition
state inside the **detector engine** that happened to be tracking it:
results were merged there, triggers were held there, the final record
was built there. That caused lost results, two different
implementations of the same policy that had drifted apart, and no place
to put anything smarter.

The control hub is a small CPU-only service that **owns that state for
both modules**. Detectors report what they see, workers report what
they recognized, and the hub alone decides what the backend receives
and when.

```
                 frames, crops                          (unchanged)
 camera ──► DETECTOR ENGINE ───────────── task queue ──────────► RECOGNIZER / OCR WORKERS
            YOLO · BYTETrack · triggers    {m}:internal:rec|ocr:tasks      │
            best crops · liveness                                          │ result (per task)
                 │  ▲                                                      ▼
   track events  │  │ ctl: result ack / satisfied / request      {m}:internal:hub:results  (STREAM)
 {m}:internal:   │  │ {m}:internal:hub:ctl:{engine} (LIST)                 │
 hub:events      ▼  │                                                      │
 (STREAM)      ┌────┴──────────────────── CONTROL HUB ◄───────────────────┘
               │ per-track state · vote · satisfied? · hold/publish triggers · final record
               │ checkpoints {m}:internal:hub:track:{uid} · leader lease · consumer group
               └──────────────────────────────► face:ai:results / plate:vehicle:results  (backend, unchanged key)
```

**Rule of thumb:** work that runs on every frame (detection, tracking,
trigger geometry, crop ranking, liveness) stays in the detector because
it needs the frames. Work that runs a few times per track (deciding
what the answer is, when to publish it, when to stop asking) is in the
hub.


## How one track flows

| Step | Detector | Hub |
|---|---|---|
| Track appears | mints a **global uid** (`{camera}-{engine}-{random}`, stays unique across restarts and rebalances), emits `track_started` with the camera's trigger flags | creates state; if the camera has the periodic flag, schedules the first re-query |
| Periodic tick | — | sends ctl `request periodic` if the track isn't satisfied and has nothing in flight |
| Trigger fires (`line_cross`/`stopped_roi`, plate `cross_line`/`stop_roi`) | if not satisfied, submits a crop **in the same frame**, then emits `trigger` with the event detail (**direction**, stop duration, …) and that crop's task_id | **satisfied** → publishes immediately (`resolution: immediate`)<br>otherwise → holds it until **that task's** result (`after_recognition`), the track becomes satisfied, `TRIGGER_MAX_WAIT_SEC` passes (`timeout`), or the track ends (`track_ended`) |
| Crop sent | emits `submitted`, pushes the task (tagged `track_uid`, `task_id`, `stage`) | tracks it as in-flight |
| Worker answers | — | adds it to the vote, recomputes **satisfied**, acks to the detector (clears its in-flight lock, sends the new satisfied flag, and a label for the debug video) |
| Track leaves | if `leave_scene` is on and not satisfied, sends the finalize / leave_scene pass, emits `track_ended`, **forgets the track** | waits for every in-flight result (≤ `FINALIZE_TIMEOUT_SEC`), flushes held triggers, publishes the **final** record (gated), keeps the state `LATE_RESULT_GRACE_SEC` to recognize stragglers |

**The detector's one remaining decision** (`facecore/hub.py` / `platecore/hub.py`):
it may send a crop for a track when something requested it, the hub has
not said satisfied, nothing is in flight (or `SUBMIT_TIMEOUT_SEC`
passed), and the best crop has changed since the last send.

### Resolution — one rule for both modules

A **confidence-weighted vote over valid results only**: each candidate
(face: personnel id, plate: plate text) scores the sum of its
confidences, the highest sum wins, and that candidate's best single
result provides the images.

**satisfied** means the winner has confidence ≥ `{M}_SATISFIED_CONF`,
or `{M}_CONSENSUS_MIN` valid results agree on it with no disagreement.
Face with `FACE_LIVENESS_POLICY=reject`: a track judged `fake`
publishes identity `"0"` with `spoof_rejected: true` and is not
re-queried.


## What changed, and why

### Inconsistencies fixed

| # | Before | Now |
|---|---|---|
| 1 | **Plate:** `cross_line` published **nothing** when a confident read already existed; `stop_roi` published **nothing** when its OCR submit failed. The two handlers had swapped `else` branches. | Both triggers go through the same hub path and are each published exactly once. |
| 2 | **Plate:** the "skip further OCR" check ignored `is_valid`, so an **invalid** read at confidence ≥ 0.85 (wrong length, bad format) stopped OCR for the rest of the track. | Only **valid** results count toward satisfied (`PLATE_SATISFIED_CONF`). |
| 3 | **Plate:** no timeout on a mid-track OCR request; a lost result blocked every later trigger on that track. **Face:** the lock expired after 1.0 s, which caused duplicate submissions whenever the recognizer was busy. | One lock, cleared by the hub's ack or after `SUBMIT_TIMEOUT_SEC` (5 s). The hub also stops waiting on a task that never answers. |
| 4 | **Both:** the line-cross **direction** (and stop duration/velocity) was dropped. Face removed it from `deferred_events` before publishing; plate kept only a timestamp. | Every trigger carries its `detail`: `meta.event` on the payload for that event, `meta.events` / `events_detail` on every payload. |
| 5 | **Face:** with `leave_scene` off, **every** track was published as final, including 2-frame noise. | One final gate for both modules (`FINAL_MIN_SEEN_FRAMES` / `FINAL_MIN_CROPS`), **except** that a track already published for (e.g. its line-cross) always gets a final, so the backend always sees the track closed. |
| 6 | **Both:** results arriving after the track finished, after an engine restart, or after a camera rebalance were silently dropped. A face trigger whose crop failed the landmark gate stayed deferred and only appeared inside the finalize meta. | The hub keeps the track until its results are in. Engines end live tracks when a camera is removed or they shut down, and `engine_started` ends tracks left behind by a crash. Held triggers are always published (timeout or track end). |
| 7 | **Face:** identity was a running **max** over every result, valid or not. **Plate:** no winner was picked across stages (left to Django). | One confidence-weighted vote over valid results, in both modules. Plate payloads gain `resolved`. |
| 8 | **Face:** confidence scales didn't line up. The worker's validity threshold was 0.3 while the engine used a separate hard-coded 0.70 gate, and 0.5–0.75 matches were halved, so they could never reach it. | A single documented knob, `FACE_SATISFIED_CONF` (0.70), plus a consensus rule, so three agreeing mid-confidence matches count. |
| 9 | **Plate:** `cond_per_trig` (periodic) was accepted and stored but **never acted on**. | Implemented: the hub requests periodic passes for cameras that enable it. |
| 10 | **Face:** a frame with no landmarks on a track that already had crops `continue`d **before** the trigger geometry, so a crossing on that frame was detected late or missed. | Only the crop update is skipped; triggers always run. |
| 11 | **Face:** the finalize pass sent crops **without** confident landmarks, and the recognizer then rejected all of them. | Only crops with landmark confidence > `MIN_LANDMARK_CONF_FOR_SUBMIT` are sent, mid-track and at finalize. |
| 12 | **Face:** mid-track results had no camera frame (`best_frame: None`). | `SEND_BEST_FRAME_MIDTRACK=true` (default) attaches it, so every published result can carry `camera_image`. |
| 13 | Liveness was attached but could never be enforced. | `FACE_LIVENESS_POLICY=reject` enforces it. The default `annotate` keeps the old behaviour. |

### Backend payloads — additive, same keys

Everything the backend read before is still in the same place. New
fields are additive.

**face:ai:results**: `camera_id`, `track_id`, `event_type`
(`line_cross` / `stopped_roi` / `finalize`), `is_final`, `timestamp`,
and `meta.{identified_as, confidence, first_name, last_name,
last_saved_face_image, last_saved_camera_image, recognition_history,
liveness*}` as before. New: `track_uid`, `meta.event` (the trigger's
direction/duration/point), `meta.events`, `meta.resolution`,
`meta.votes`, `meta.conflicting_identities`, `meta.spoof_rejected`,
`meta.national_code`, `meta.department`, and `result` (the worker-shaped
record of the chosen result: `personnelid`, `face_image`,
`camera_image`, `detection_score`, `date`, `time`, …).

**plate:vehicle:results**: `process_id`, `camera_id`, `track_id`,
`update_type` (`cross_line` / `stop_roi` / `leave_scene`), `is_final`,
`meta`, `events` (`{name: iso time}`), `ocr_results` (`{stage: OCR result}`),
`track_paths`, and on the final `stream_idx` / `video_source`, all as before.
New: `track_uid`, `events_detail` (with direction etc.), `meta.event`,
`meta.resolution`, and **`resolved`**: `{plate_text, confidence,
is_valid, voted_class, plate_type, stage, plate_image, frame_image,
votes, candidates}`. The answer is decided here instead of in Django.


## Running

The hub ships as a `control_hub` service in **each** module's
`compose.yaml` (`HUB_MODULES=face` / `plate`, build context
`../control-hub`), so `docker compose up -d --build` in a module folder
starts it with everything else. Run **exactly one hub per module per
Redis**. A second copy is harmless: it waits on the leader lease as a
hot standby.

To run both modules from one process instead, run this image once with
`HUB_MODULES=face,plate` and remove the `control_hub` block from both
module compose files.

```bash
curl localhost:8020/health               # face hub (plate: 8021)
curl 'localhost:8020/tracks?module=face' # live tracks + their current answer

pip install redis && export REDIS_URL=redis://localhost:6379/0
python hub_tools.py status  --module face   # leader, stream lag/pending, ctl queues
python hub_tools.py tracks  --module plate  # every track checkpoint
python hub_tools.py tail    --module face   # live feed of events + results entering the hub
python hub_tools.py results --module face   # last records sent to the backend (read-only)
```

### Self-healing

- **Hub crash or restart.** Track state is checkpointed to Redis after
  every change and restored on boot. Stream entries it read but never
  acknowledged are replayed. Every handler is idempotent (trigger once
  per name, results deduplicated by `task_id`), so a replay is safe.
  Delivery to the backend is at-least-once.
- **Hub down for a while.** Detectors keep tracking and submitting, and
  workers keep answering. Everything queues in the two streams, and the
  hub catches up when it returns.
- **Detector engine crash.** On restart it emits `engine_started` with a
  new boot id, and the hub ends that engine's old tracks. A track the
  hub stops hearing about for `TRACK_STALE_SEC` is ended as stale.
- **Camera rebalance or deactivation.** The engine ends the camera's
  live tracks (`camera_removed`) before letting go, so each one still
  gets a final.


## Configuration

Service-wide: `REDIS_URL` or `REDIS_HOST`/`PORT`/`DB`/`PASSWORD`,
`HUB_MODULES`, `HUB_HEALTH_PORT` (8020), `HUB_LEASE_TTL_SEC` (15),
`HUB_STATE_TTL_SEC` (3600), `HUB_STREAM_MAXLEN` (200000, set on the
detector and worker side).

Per module (`FACE_…` / `PLATE_…`):

| Variable | Default (face / plate) | Meaning |
|---|---|---|
| `SATISFIED_CONF` | 0.70 / 0.85 | A valid answer at this confidence stops re-querying and releases held triggers |
| `CONSENSUS_MIN` | 3 | …or this many agreeing valid results with no disagreement (0 = off) |
| `TRIGGER_MAX_WAIT_SEC` | 3.0 | Longest a trigger waits for its result |
| `PERIODIC_INTERVAL_SEC` / `PERIODIC_FIRST_DELAY_SEC` | 3.0 / 1.0 | Re-query cadence for cameras with the periodic flag |
| `FINALIZE_TIMEOUT_SEC` | 10 | Longest wait for in-flight results after the track ends |
| `FINAL_MIN_SEEN_FRAMES` / `FINAL_MIN_CROPS` | 8 / 1 | Final-record gate (skipped when something was already published) |
| `TRACK_STALE_SEC` | 120 | End a track the hub stops hearing about |
| `LATE_RESULT_GRACE_SEC` | 30 | Keep closed tracks this long to recognize late or duplicate results |
| `FACE_LIVENESS_POLICY` | annotate | `reject` = spoofed tracks publish identity "0" |

Detector side (module `.env`): `SUBMIT_TIMEOUT_SEC` (5),
`TRACK_UPDATE_INTERVAL_SEC` (5). Face only:
`MIN_LANDMARK_CONF_FOR_SUBMIT` (0.6), `SEND_BEST_FRAME_MIDTRACK` (true),
`FINALIZE_MAX_CROPS` (1).

Removed: `PERIODIC_MODE`, `PERIODIC_FRAME_INTERVAL`,
`PERIODIC_TIME_INTERVAL` and `PERIODIC_RECOG_CONF_THRESH` (face), and
`OCR_CONF_SKIP_THRESHOLD` and `OCR_FINALIZE_TIMEOUT_SEC` (plate). The
hub variables above replace them.


## Where the AI layer plugs in

`policy.py` is the single place that turns results into decisions.
Anything smarter goes there or next to it: a learned "ask again?"
policy, plate + face fusion (a vehicle and its driver), cross-camera
linking, anomaly flags. It sees every result for every track and emits
the one answer the backend stores. `core.py` (lifecycle and timing)
doesn't need to change for any of these.


## Files

```
src/protocol.py    wire contract (keys, event kinds, ctl actions) — read this first
src/core.py        per-track state machine (pure; returns effects)
src/policy.py      face / plate vocabulary, vote, satisfied, backend payloads
src/service.py     ModuleRunner: lease, consumer group, checkpoints, apply effects
src/redis_io.py    every Redis call the hub makes
src/main.py        entry point + /health, /tracks
hub_tools.py       read-only inspection CLI
tests/             test_core.py (every case, fake clock) · test_integration.py (real redis-server)
```

```bash
pip install redis && python3 tests/run_all.py   # integration tests auto-skip without redis-server
```
