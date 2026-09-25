#!/usr/bin/env python3
"""
hub_tools.py — look inside the plate control hub from your own machine.

    pip install redis
    export REDIS_URL=redis://localhost:6379/0

    python hub_tools.py status          # heartbeat, leader, stream lag, ctl queues
    python hub_tools.py tracks          # every track checkpoint the hub holds
    python hub_tools.py track --uid cam1-0-ab12cd34ef
    python hub_tools.py tail            # live: every event/result entering the hub
    python hub_tools.py results -n 5    # last N records sent to the backend (non-destructive)

--module defaults to REDIS_MODULE, else "plate".

Read-only: nothing here consumes, acks or deletes anything.
"""

import argparse
import json
import os
import time

import redis

BACKEND_KEYS = {"plate": "plate:vehicle:results"}


def r():
    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)


def k(module):
    p = f"{module}:internal:hub"
    return {"events": f"{p}:events", "results": f"{p}:results", "ctl": f"{p}:ctl:",
            "track": f"{p}:track:", "leader": f"{p}:leader", "hb": f"{p}:heartbeat"}


def status(m):
    c, kk = r(), k(m)
    hb = c.get(kk["hb"])
    print(f"leader      : {c.get(kk['leader'])}")
    print(f"heartbeat   : {json.loads(hb) if hb else 'NONE (hub down?)'}")
    for name in ("events", "results"):
        try:
            groups = c.xinfo_groups(kk[name])
        except redis.ResponseError:
            groups = []
        print(f"{name:<12}: length={c.xlen(kk[name])} groups={[(g['name'], g.get('pending'), g.get('lag')) for g in groups]}")
    for key in c.scan_iter(match=kk["ctl"] + "*"):
        print(f"ctl         : {key} queued={c.llen(key)}")
    print(f"backend list: {BACKEND_KEYS.get(m)} length={c.llen(BACKEND_KEYS.get(m, ''))}")


def tracks(m):
    c, kk = r(), k(m)
    rows = []
    for key in c.scan_iter(match=kk["track"] + "*"):
        raw = c.get(key)
        if raw:
            rows.append(json.loads(raw))
    rows.sort(key=lambda s: s.get("created_ts") or 0)
    for st in rows:
        state = "CLOSED" if st.get("closed") else "ENDING" if st.get("ended") else "LIVE"
        ev = {n: e.get("status") for n, e in (st.get("events") or {}).items()}
        best = max((x for x in st.get("results") or [] if x.get("valid")),
                   key=lambda x: x.get("confidence", 0), default=None)
        print(f"{st['uid']:<32} {state:<6} cam={st.get('camera_id')} trk={st.get('track_id')} "
              f"sat={st.get('satisfied')} results={len(st.get('results') or [])} "
              f"best={(best or {}).get('key')}@{(best or {}).get('confidence')} "
              f"inflight={list((st.get('in_flight') or {}).keys())} events={ev}")
    print(f"({len(rows)} tracks)")


def track(m, uid):
    raw = r().get(k(m)["track"] + uid)
    print(json.dumps(json.loads(raw), indent=2) if raw else "not found")


def tail(m):
    c, kk = r(), k(m)
    ids = {kk["events"]: "$", kk["results"]: "$"}
    print("tailing (Ctrl+C to stop)...")
    try:
        while True:
            for stream, entries in c.xread(ids, block=1000) or []:
                for eid, f in entries:
                    ids[stream] = eid
                    d = json.loads(f.get("data") or "{}")
                    brief = {x: d.get(x) for x in ("uid", "event", "task_id", "stage", "reason", "is_valid",
                                                   "confidence", "personnelid", "plate_text") if d.get(x) is not None}
                    print(f"{time.strftime('%H:%M:%S')} {stream.rsplit(':', 1)[-1]:<7} {f.get('kind'):<14} {brief}")
    except KeyboardInterrupt:
        pass


def results(m, n):
    for raw in r().lrange(BACKEND_KEYS[m], -n, -1):
        print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False)[:4000])
        print("-" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "tracks", "track", "tail", "results"])
    ap.add_argument("--module", default=os.environ.get("REDIS_MODULE", "plate"))
    ap.add_argument("--uid")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()
    {"status": lambda: status(a.module), "tracks": lambda: tracks(a.module),
     "track": lambda: track(a.module, a.uid), "tail": lambda: tail(a.module),
     "results": lambda: results(a.module, a.n)}[a.cmd]()
