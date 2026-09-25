# Face control hub — one owner for every track's recognition state

Part of `face-service/` and started by its `compose.yaml`
(`control_hub`). It has no dependency outside this folder: the face and
plate modules each ship their own copy and run independently.

The detector used to keep each track's recognition state in the engine
subprocess that was tracking it. Results were merged there, triggers
were held there, and the final record was built there. When that
engine let go of the track (the track ended, the camera was re-added
or rebalanced, the engine restarted), any result still on its way was
dropped. Face and plate also implemented the same policy separately, and the two had drifted apart. The hub is a small CPU-only service that owns
that state instead. The detector reports what it sees, the recognizer
reports what it read, and the hub alone decides what reaches
`face:ai:results` and when.

```
 camera ─► mediamtx ─► face_detector ── tasks ─────────────► face_recognizer
                          │  ▲   (crops, tagged uid/task_id)   │
            track events  │  │  ctl: ack / satisfied /         │ one result per task
      face:internal:hub:events │  periodic request              │ (always — also on error)
                          ▼  │                                 ▼
                     ┌───────┴────── control_hub ◄──── face:internal:hub:results
                     │ per-track state · vote · satisfied? · hold/publish triggers · final
                     └──────────────► face:ai:results   (backend key unchanged)
```

## One track, step by step

| # | Detector | Hub |
|---|---|---|
| 1 | Track appears: mints a global `uid` (`{camera}-{engine}-{random}`, unique across restarts, rebalances and camera re-adds), emits `track_started` with the camera's trigger flags | Creates the track state. If periodic is on, schedules a re-query |
| 2 | — | Periodic tick: `request periodic` if the track isn't satisfied and nothing is in flight |
| 3 | Trigger (`line_cross` / `stopped_roi`): if not satisfied, sends crops **in the same frame**, emits `submitted`, then `trigger` with its detail (**direction**, stop duration, point) and that task id | **Satisfied:** publishes now (`immediate`).<br>**Otherwise:** holds the trigger for **its own task's** result (`after_recognition`). The wait is up to `TRIGGER_TASK_WAIT_SEC` while that task is queued, or `TRIGGER_MAX_WAIT_SEC` if no crop could be sent; if a crop is sent later, the trigger waits for that task. It also releases early if another result makes the track satisfied. Past the deadline it publishes as `timeout` |
| 4 | recognizer answers | Adds the result to the vote and re-evaluates **satisfied**. Acks to the detector: clears its in-flight lock, sends the satisfied flag and a debug-video label |
| 5 | Track gone (or camera removed / engine stopping): if `leave_scene` is on and the track isn't satisfied, sends the `finalize` pass. Emits `track_ended` and **forgets the track** | Waits until **every** in-flight result has landed. That's normally under 1 s; the cap is `FINALIZE_TIMEOUT_SEC`. Then it flushes held triggers and publishes the **final** (`is_final`, `complete`, `missing_tasks`, `revision`) |
| 6 | — | A result that lands after the final and **changes the answer** re-publishes the final as `revision` 2, 3, … The backend should upsert on `track_uid` |

**Resolution:** a confidence-weighted vote over **valid** results. Each
candidate scores the sum of its confidences, and the highest sum wins.
**Satisfied** means the winner has confidence ≥ `SATISFIED_CONF`, or
`CONSENSUS_MIN` valid results agree on it with no disagreement.
With `FACE_LIVENESS_POLICY=reject`, a track judged `fake` publishes identity `"0"` with
`spoof_rejected: true` and is not re-queried.

## Robustness: what can go wrong and what happens

| Situation | What happens |
|---|---|
| A crop has weak landmarks | It is never sent (the recognizer would drop it), either mid-track or at finalize. The request stays pending until a usable crop exists |
| A frame has no landmarks | Only the crop update is skipped; trigger geometry still runs, so a crossing isn't missed |
| The recognizer is backlogged | Triggers wait for their own task (≤ `TRIGGER_TASK_WAIT_SEC`). The final waits for in-flight results (≤ `FINALIZE_TIMEOUT_SEC`). A result later still re-publishes the final if it changes the answer |
| Upload to MinIO fails for the best result | Images are taken from another result with the same answer; a field stays empty only if no such result has one |
| A recognizer task fails or raises | The worker still answers (`status: error` / `skipped`), so the hub never waits for nothing |
| A recognizer process is killed mid-task | That answer never comes. The hub stops waiting at its timeouts, and the detector's in-flight lock expires after `SUBMIT_TIMEOUT_SEC`, so the next request can send again |
| Redis unreachable (detector side) | `emit` and `submit` only append to an in-memory **outbox**. A sender thread retries in order, so the frame loop never blocks and nothing is lost. On shutdown the outbox is flushed for 5 s |
| Redis unreachable (worker side) | The result push retries with backoff for about 25 s |
| Hub crashes or restarts | Track state is checkpointed to Redis after every change and restored. Read-but-unacked stream entries are replayed. Handlers are idempotent (triggers once, results deduplicated by `task_id`), so delivery to the backend is at-least-once |
| Hub down for a while | Detectors and workers keep going and everything queues in the streams. The first-ever start begins at the stream tail |
| Detector engine crashes | Its restart emits `engine_started` with a new boot id, and the hub ends that engine's old tracks. A track the hub stops hearing about is ended after `TRACK_STALE_SEC` |
| Camera deactivated, rebalanced or re-added | The engine ends the camera's live tracks first, each with a final. New tracks get new uids, so a late result can never land on another person that reused a track number |
| A result arrives before its track's events (different streams) | Kept as an orphan and joined when `track_started` arrives. A trigger whose result is already in publishes at once |
| Two hub containers | Leader lease: one works, the other stands by |

## Backend payload (`face:ai:results`): additive, same keys

`camera_id`, `track_id`, `event_type` (`line_cross` / `stopped_roi` / `finalize`), `is_final`,
`timestamp`, and `meta.{identified_as, confidence, first_name, last_name, last_saved_face_image,
last_saved_camera_image, recognition_history, liveness*}` as before.

New: `track_uid`; `meta.event` (this trigger's direction / duration / point); `meta.events`;
`meta.resolution` (`immediate` / `after_recognition` / `timeout` / `track_ended` / `late_result`);
`meta.votes`; `meta.conflicting_identities`; `meta.spoof_rejected`; `meta.national_code`;
`meta.department`; `result` (the worker-shaped record of the chosen result); and on finals,
`revision`, `complete` and `missing_tasks`.

## Configuration (`.env`, `FACE_…`)

| Variable | Default | Meaning |
|---|---|---|
| `FACE_SATISFIED_CONF` | 0.70 | A valid answer at this confidence stops re-querying and releases held triggers |
| `FACE_CONSENSUS_MIN` | 3 | …or this many agreeing valid results with no disagreement (0 = off) |
| `FACE_TRIGGER_TASK_WAIT_SEC` | 15 | Longest a trigger waits while its own task is queued |
| `FACE_TRIGGER_MAX_WAIT_SEC` | 3 | Longest a trigger waits when no crop could be sent |
| `FACE_PERIODIC_INTERVAL_SEC` / `_FIRST_DELAY_SEC` | 3 / 1 | Re-query cadence for cameras with the periodic flag (`cond_per_trig`) |
| `FACE_FINALIZE_TIMEOUT_SEC` | 30 | Upper bound on waiting for in-flight results after the track ends |
| `FACE_FINAL_MIN_SEEN_FRAMES` / `_MIN_CROPS` | 8 / 1 | Final gate (skipped when something was already published for the track) |
| `FACE_LATE_RESULT_POLICY` | republish_if_changed | Or `drop` |
| `FACE_LATE_RESULT_GRACE_SEC` | 300 | How long closed tracks are remembered |
| `FACE_TRACK_STALE_SEC` | 120 | End a track the hub stops hearing about |
| `FACE_LIVENESS_POLICY` | annotate | `reject` = spoofed tracks publish identity "0" |

Detector (face only): `MIN_LANDMARK_CONF_FOR_SUBMIT` (0.6), `SEND_BEST_FRAME_MIDTRACK` (true), `FINALIZE_MAX_CROPS` (1).

Detector side: `SUBMIT_TIMEOUT_SEC` (15), `TRACK_UPDATE_INTERVAL_SEC`
(5), `HUB_OUTBOX_MAX` (20000). Hub service: `HUB_HEALTH_PORT` (8020
inside the container, published as `HUB_API_PORT`), `HUB_LEASE_TTL_SEC`,
`HUB_STATE_TTL_SEC`.

## Operating it

```bash
curl localhost:8020/health                  # leader, live tracks, stream lag
curl 'localhost:8020/tracks'                # every track + its current answer

pip install redis && export REDIS_URL=redis://localhost:6379/0
python3 control-hub/hub_tools.py status       # leader, stream length/pending/lag, ctl queues
python3 control-hub/hub_tools.py tracks       # every checkpoint
python3 control-hub/hub_tools.py tail         # live feed of events + results entering the hub
python3 control-hub/hub_tools.py results -n 3 # last records sent to the backend (read-only)
```

## Files

```
src/protocol.py   wire contract: keys, event kinds, ctl actions (read first)
src/core.py       per-track state machine — pure, returns effects
src/policy.py     Face vocabulary, vote, satisfied, backend payloads  ← where AI logic plugs in
src/service.py    lease, consumer group, checkpoints, apply effects
src/redis_io.py   every Redis call the hub makes
src/main.py       entry point + /health, /tracks
hub_tools.py      read-only inspection CLI
tests/            test_core.py (every case, fake clock) · test_integration.py (real redis-server)
```

```bash
python3 control-hub/tests/run_all.py   # integration tests auto-skip without redis-server
```
