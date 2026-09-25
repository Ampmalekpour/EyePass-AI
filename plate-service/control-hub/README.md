# Plate control hub — one owner for every track's OCR state

Part of `plate-service/` and started by its `compose.yaml`
(`control_hub`). It has no dependency outside this folder: the face and
plate modules each ship their own copy and run independently.

The detector used to keep each track's OCR state in the engine
subprocess that was tracking it. Results were merged there, triggers
were held there, and the final record was built there. When that
engine let go of the track (the track ended, the camera was re-added
or rebalanced, the engine restarted), any result still on its way was
dropped. You saw this as a final whose `ocr_results` were `{"cross_line": null, …}`, while the OCR log showed the plate. The hub is a small CPU-only service that owns
that state instead. The detector reports what it sees, the OCR
reports what it read, and the hub alone decides what reaches
`plate:vehicle:results` and when.

```
 camera ─► mediamtx ─► plate_detector ── tasks ─────────────► plate_ocr
                          │  ▲   (crops, tagged uid/task_id)   │
            track events  │  │  ctl: ack / satisfied /         │ one result per task
      plate:internal:hub:events │  periodic request              │ (always — also on error)
                          ▼  │                                 ▼
                     ┌───────┴────── control_hub ◄──── plate:internal:hub:results
                     │ per-track state · vote · satisfied? · hold/publish triggers · final
                     └──────────────► plate:vehicle:results   (backend key unchanged)
```

## One track, step by step

| # | Detector | Hub |
|---|---|---|
| 1 | Track appears: mints a global `uid` (`{camera}-{engine}-{random}`, unique across restarts, rebalances and camera re-adds), emits `track_started` with the camera's trigger flags | Creates the track state. If periodic is on, schedules a re-query |
| 2 | — | Periodic tick: `request periodic` if the track isn't satisfied and nothing is in flight |
| 3 | Trigger (`cross_line` / `stop_roi`): if not satisfied, sends crops **in the same frame**, emits `submitted`, then `trigger` with its detail (**direction**, stop duration, point) and that task id | **Satisfied:** publishes now (`immediate`).<br>**Otherwise:** holds the trigger for **its own task's** result (`after_recognition`). The wait is up to `TRIGGER_TASK_WAIT_SEC` while that task is queued, or `TRIGGER_MAX_WAIT_SEC` if no crop could be sent; if a crop is sent later, the trigger waits for that task. It also releases early if another result makes the track satisfied. Past the deadline it publishes as `timeout` |
| 4 | OCR answers | Adds the result to the vote and re-evaluates **satisfied**. Acks to the detector: clears its in-flight lock, sends the satisfied flag and a debug-video label |
| 5 | Track gone (or camera removed / engine stopping): if `leave_scene` is on and the track isn't satisfied, sends the `leave_scene` pass. Emits `track_ended` and **forgets the track** | Waits until **every** in-flight result has landed. That's normally under 1 s; the cap is `FINALIZE_TIMEOUT_SEC`. Then it flushes held triggers and publishes the **final** (`is_final`, `complete`, `missing_tasks`, `revision`) |
| 6 | — | A result that lands after the final and **changes the answer** re-publishes the final as `revision` 2, 3, … The backend should upsert on `track_uid` |

**Resolution:** a confidence-weighted vote over **valid** results. Each
candidate scores the sum of its confidences, and the highest sum wins.
**Satisfied** means the winner has confidence ≥ `SATISFIED_CONF`, or
`CONSENSUS_MIN` valid results agree on it with no disagreement.

## Robustness: what can go wrong and what happens

| Situation | What happens |
|---|---|
| **Vehicle leaves while its OCR is running** (the reported empty payload) | The final **waits** for that OCR result and carries it. Before, with `leave_scene` off, it was published at once with `null` stages and the result was dropped |
| No OCR read validates | `resolved.is_valid=false`, `plate_text="0"`, **but** `raw_text`, `raw_confidence`, the images and `description` of the best attempt are filled in. A stage without a result is simply absent, never `null` |
| An invalid read has high confidence | It doesn't count toward *satisfied*, so OCR keeps trying (before, it stopped OCR for the rest of the track) |
| The OCR is backlogged | Triggers wait for their own task (≤ `TRIGGER_TASK_WAIT_SEC`). The final waits for in-flight results (≤ `FINALIZE_TIMEOUT_SEC`). A result later still re-publishes the final if it changes the answer |
| Upload to MinIO fails for the best result | Images are taken from another result with the same answer; a field stays empty only if no such result has one |
| A OCR task fails or raises | The worker still answers (`status: error` / `skipped`), so the hub never waits for nothing |
| A OCR process is killed mid-task | That answer never comes. The hub stops waiting at its timeouts, and the detector's in-flight lock expires after `SUBMIT_TIMEOUT_SEC`, so the next request can send again |
| Redis unreachable (detector side) | `emit` and `submit` only append to an in-memory **outbox**. A sender thread retries in order, so the frame loop never blocks and nothing is lost. On shutdown the outbox is flushed for 5 s |
| Redis unreachable (worker side) | The result push retries with backoff for about 25 s |
| Hub crashes or restarts | Track state is checkpointed to Redis after every change and restored. Read-but-unacked stream entries are replayed. Handlers are idempotent (triggers once, results deduplicated by `task_id`), so delivery to the backend is at-least-once |
| Hub down for a while | Detectors and workers keep going and everything queues in the streams. The first-ever start begins at the stream tail |
| Detector engine crashes | Its restart emits `engine_started` with a new boot id, and the hub ends that engine's old tracks. A track the hub stops hearing about is ended after `TRACK_STALE_SEC` |
| Camera deactivated, rebalanced or re-added | The engine ends the camera's live tracks first, each with a final. New tracks get new uids, so a late result can never land on another vehicle that reused a track number |
| A result arrives before its track's events (different streams) | Kept as an orphan and joined when `track_started` arrives. A trigger whose result is already in publishes at once |
| Two hub containers | Leader lease: one works, the other stands by |

## Backend payload (`plate:vehicle:results`): additive, same keys

`process_id`, `camera_id`, `track_id`, `update_type` (`cross_line` / `stop_roi` / `leave_scene`),
`is_final`, `meta`, `events` (`{name: iso time}`), `ocr_results` (`{stage: OCR result}`, only stages
that actually answered), `track_paths`, and on the final `stream_idx` / `video_source`, as before.

New: `track_uid`; `events_detail` (with direction etc.); `meta.event`; `meta.resolution`; and
**`resolved`**: `{plate_text, confidence, is_valid, raw_text, raw_confidence, voted_class, plate_type,
stage, plate_image, frame_image, votes, candidates, n_results, description}`. The single answer is
decided here instead of in Django. Finals also carry `revision`, `complete` and `missing_tasks`.

## Configuration (`.env`, `PLATE_…`)

| Variable | Default | Meaning |
|---|---|---|
| `PLATE_SATISFIED_CONF` | 0.85 | A valid answer at this confidence stops re-querying and releases held triggers |
| `PLATE_CONSENSUS_MIN` | 3 | …or this many agreeing valid results with no disagreement (0 = off) |
| `PLATE_TRIGGER_TASK_WAIT_SEC` | 15 | Longest a trigger waits while its own task is queued |
| `PLATE_TRIGGER_MAX_WAIT_SEC` | 3 | Longest a trigger waits when no crop could be sent |
| `PLATE_PERIODIC_INTERVAL_SEC` / `_FIRST_DELAY_SEC` | 3 / 1 | Re-query cadence for cameras with the periodic flag (`cond_per_trig`) |
| `PLATE_FINALIZE_TIMEOUT_SEC` | 30 | Upper bound on waiting for in-flight results after the track ends |
| `PLATE_FINAL_MIN_SEEN_FRAMES` / `_MIN_CROPS` | 8 / 1 | Final gate (skipped when something was already published for the track) |
| `PLATE_LATE_RESULT_POLICY` | republish_if_changed | Or `drop` |
| `PLATE_LATE_RESULT_GRACE_SEC` | 300 | How long closed tracks are remembered |
| `PLATE_TRACK_STALE_SEC` | 120 | End a track the hub stops hearing about |

Detector side: `SUBMIT_TIMEOUT_SEC` (15), `TRACK_UPDATE_INTERVAL_SEC`
(5), `HUB_OUTBOX_MAX` (20000). Hub service: `HUB_HEALTH_PORT` (8020
inside the container, published as `HUB_API_PORT`), `HUB_LEASE_TTL_SEC`,
`HUB_STATE_TTL_SEC`.

## Operating it

```bash
curl localhost:8021/health                  # leader, live tracks, stream lag
curl 'localhost:8021/tracks'                # every track + its current answer

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
src/policy.py     Plate vocabulary, vote, satisfied, backend payloads  ← where AI logic plugs in
src/service.py    lease, consumer group, checkpoints, apply effects
src/redis_io.py   every Redis call the hub makes
src/main.py       entry point + /health, /tracks
hub_tools.py      read-only inspection CLI
tests/            test_core.py (every case, fake clock) · test_integration.py (real redis-server)
```

```bash
python3 control-hub/tests/run_all.py   # integration tests auto-skip without redis-server
```
