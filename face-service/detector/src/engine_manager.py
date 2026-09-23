"""
engine_manager.py
--------------------------------------------------------------------
Supervises the pool of Engine subprocesses and which camera is
assigned to which engine. Two behaviours beyond the reference
EngineManager:

  1. It plugs directly into facecore.lifecycle.ServiceLifecycle:
     start_idle(n) brings up n engines with no cameras attached
     (loaded + warm, matching Engine.run()'s own warmup predict call);
     stop_process() detaches every camera but leaves engines running;
     stop_idle() tears every engine down. Camera attachment itself
     (start_process side) is driven by backend_bridge, which owns the
     durable desired-state and therefore knows WHICH cameras to
     reconcile — EngineManager only knows how to place a camera once
     told to.

  2. rebalance(): the reference EngineManager only ever grows (a new
     engine when the busiest one is full) and never shrinks. Over the
     life of a deployment, stops/disconnects fragment cameras across
     more engines than necessary (5-per-engine cap, 10 cameras ->
     3 + 2 after churn). rebalance() detects when the current camera
     count would fit into fewer engines and consolidates: it replays
     each drained camera's last known add_camera() config onto a
     fuller engine, then stops the emptied engine(s). A migrated
     camera's BYTETracker restarts (track IDs reset) — the RTSP relay
     keeps the camera's actual upstream connection alive throughout,
     so this is a bookkeeping reset, not a stream interruption.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine import _engine_process_main

logger = logging.getLogger("engine_manager")


class EngineManager:
    def __init__(
            self,
            model_path: str,
            imgsz: int,
            conf: float,
            save_output: bool,
            output_dir: str,
            class_labels: Dict[int, str],
            save_as_video: bool = True,
            max_cameras_per_engine: int = 6,
            on_topology_changed: Optional[Callable[[], None]] = None,
            engine_shutdown_timeout: float = 30.0,
    ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.save_output = bool(save_output)
        self.save_as_video = bool(save_as_video)
        self.output_dir = output_dir
        self.class_labels = class_labels

        self.max_cameras_per_engine = int(max_cameras_per_engine)
        self.on_topology_changed = on_topology_changed or (lambda: None)
        self.engine_shutdown_timeout = float(engine_shutdown_timeout)

        # ONE explicit spawn context for every IPC object and process.
        #
        # This must not go back to bare mp.Queue()/mp.Process(): the
        # default context on Linux is FORK, and a Queue built under the
        # fork context carries lock/semaphore handles that only survive
        # inheritance across fork(). Hand such a queue to a SPAWN-started
        # child and the child rebuilds it from a pickle with handles that
        # refer to nothing — the first put_nowait() in the child then
        # blocks forever on a semaphore that does not exist, with no
        # error and no traceback, freezing that engine subprocess for
        # good. Creating the context up front and taking Queue/Event/
        # Process from it keeps every object on the same side of that
        # line.
        self._ctx = mp.get_context("spawn")

        self.status_queue = self._ctx.Queue()
        self._lock = threading.RLock()
        self.engines: Dict[int, Dict[str, Any]] = {}
        self.camera_to_engine: Dict[str, int] = {}
        # Last full add_camera() kwargs per camera, so a rebalance can
        # replay a migrated camera's config verbatim on its new engine.
        self._camera_last_config: Dict[str, dict] = {}

    # ================================================================
    # facecore.lifecycle.ServiceLifecycle hooks
    # ================================================================
    def start_idle(self, count: int) -> int:
        """Bring the engine pool up to at least `count` idle (no-camera)
        engines. Never shrinks here — shrinking only happens through
        rebalance(), which has cameras to redistribute first."""
        with self._lock:
            while len(self.engines) < max(1, int(count)):
                self._start_engine_locked()
            return len(self.engines)

    def stop_idle(self):
        self.shutdown()

    def start_process(self):
        """No-op: attaching cameras is driven by backend_bridge's
        startup reconcile (it owns the durable active-camera ledger),
        not by EngineManager itself."""
        return

    def stop_process(self):
        """Detach every camera from every engine; engines stay loaded
        and idle, ready for the next start_process()."""
        with self._lock:
            camera_ids = list(self.camera_to_engine.keys())
        for cid in camera_ids:
            self.remove_camera(cid)

    def get_engine_count(self) -> int:
        with self._lock:
            return len(self.engines)

    # ================================================================
    # Engine subprocess lifecycle
    # ================================================================
    def _start_engine_locked(self) -> int:
        engine_id = 0 if not self.engines else (max(self.engines.keys()) + 1)

        # Same context as status_queue (see __init__) — never
        # mp.set_start_method() + bare mp.Queue() here, which is what
        # left status_queue on the fork context while the child was
        # spawned.
        control_queue = self._ctx.Queue()
        stop_event = self._ctx.Event()

        proc = self._ctx.Process(
            target=_engine_process_main,
            args=(
                engine_id, self.model_path, self.imgsz, self.conf,
                self.save_output, self.save_as_video,
                self.status_queue, control_queue, stop_event,
                self.output_dir, self.class_labels,
            ),
            daemon=False,
            name=f"face-engine-{engine_id}",
        )
        proc.start()

        self.engines[engine_id] = {
            "proc": proc,
            "control_queue": control_queue,
            "stop_event": stop_event,
            "cameras": set(),
        }
        logger.info("Started engine %d (pid=%s)", engine_id, proc.pid)
        self.on_topology_changed()
        return engine_id

    def _stop_engine_locked(self, engine_id: int, timeout: Optional[float] = None):
        info = self.engines.pop(engine_id, None)
        if not info:
            return
        timeout = self.engine_shutdown_timeout if timeout is None else timeout
        try:
            info["control_queue"].put({"cmd": "stop"})
            info["stop_event"].set()
        except Exception:
            pass

        proc = info.get("proc")
        if proc is not None:
            proc.join(timeout=timeout)
            if proc.is_alive():
                logger.warning("Engine %d did not exit in %.0fs — terminating", engine_id, timeout)
                try:
                    proc.terminate()
                except Exception:
                    pass
        logger.info("Stopped engine %d", engine_id)
        self.on_topology_changed()

    def _pick_engine_locked(self) -> int:
        alive = []
        for eid, info in self.engines.items():
            p = info["proc"]
            if p is not None and p.is_alive():
                alive.append((len(info["cameras"]), eid))
        if not alive:
            return self._start_engine_locked()

        alive.sort()
        load, eid = alive[0]
        if load >= self.max_cameras_per_engine:
            return self._start_engine_locked()
        return eid

    # ================================================================
    # Camera placement
    # ================================================================
    def add_camera(self, camera_id: str, url: str, roi: Tuple[float, float, float, float],
                    cond_per_trig: bool = False, cross_line_trig: bool = False,
                    stop_roi_trig: bool = False, leave_scene_trig: bool = False,
                    line_p1_x=None, line_p1_y=None, line_p2_x=None, line_p2_y=None,
                    stop_roi_p1_x=None, stop_roi_p1_y=None, stop_roi_p2_x=None, stop_roi_p2_y=None,
                    stop_roi_p3_x=None, stop_roi_p3_y=None, stop_roi_p4_x=None, stop_roi_p4_y=None,
                    ) -> Dict[str, Any]:
        camera_id = str(camera_id)
        url = str(url)

        cmd_payload = {
            "cmd": "add", "camera_id": camera_id, "url": url, "roi": roi,
            "cond_per_trig": bool(cond_per_trig), "cross_line_trig": bool(cross_line_trig),
            "stop_roi_trig": bool(stop_roi_trig), "leave_scene_trig": bool(leave_scene_trig),
            "line_p1_x": line_p1_x, "line_p1_y": line_p1_y, "line_p2_x": line_p2_x, "line_p2_y": line_p2_y,
            "stop_roi_p1_x": stop_roi_p1_x, "stop_roi_p1_y": stop_roi_p1_y,
            "stop_roi_p2_x": stop_roi_p2_x, "stop_roi_p2_y": stop_roi_p2_y,
            "stop_roi_p3_x": stop_roi_p3_x, "stop_roi_p3_y": stop_roi_p3_y,
            "stop_roi_p4_x": stop_roi_p4_x, "stop_roi_p4_y": stop_roi_p4_y,
        }

        with self._lock:
            self._camera_last_config[camera_id] = cmd_payload

            if camera_id in self.camera_to_engine:
                eid = self.camera_to_engine[camera_id]
                info = self.engines.get(eid)
                if info and info["proc"].is_alive():
                    info["control_queue"].put(cmd_payload)
                    return {"camera_id": camera_id, "engine_id": eid, "status": "already_running"}
                self.camera_to_engine.pop(camera_id, None)

            eid = self._pick_engine_locked()
            info = self.engines[eid]
            info["cameras"].add(camera_id)
            self.camera_to_engine[camera_id] = eid
            info["control_queue"].put(cmd_payload)

            return {"camera_id": camera_id, "engine_id": eid, "status": "started"}

    def remove_camera(self, camera_id: str) -> Dict[str, Any]:
        camera_id = str(camera_id)
        with self._lock:
            self._camera_last_config.pop(camera_id, None)
            eid = self.camera_to_engine.pop(camera_id, None)
            if eid is None:
                return {"camera_id": camera_id, "status": "not_running"}
            info = self.engines.get(eid)
            if info:
                info["cameras"].discard(camera_id)
                try:
                    info["control_queue"].put({"cmd": "remove", "camera_id": camera_id})
                except Exception:
                    pass
            return {"camera_id": camera_id, "engine_id": eid, "status": "stopped"}

    def camera_engine(self, camera_id: str) -> Optional[int]:
        with self._lock:
            return self.camera_to_engine.get(str(camera_id))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "engines": {
                    eid: {"cameras": sorted(info["cameras"]), "alive": bool(info["proc"] and info["proc"].is_alive())}
                    for eid, info in self.engines.items()
                },
                "camera_to_engine": dict(self.camera_to_engine),
            }

    # ================================================================
    # Consolidation
    # ================================================================
    def rebalance(self):
        """If the current cameras would fit into fewer engines than are
        currently running, migrate them onto the fullest engines and
        stop whichever engines end up empty."""
        with self._lock:
            alive_ids = [eid for eid, info in self.engines.items()
                        if info["proc"] is not None and info["proc"].is_alive()]
            if len(alive_ids) <= 1:
                return

            total_cameras = sum(len(self.engines[eid]["cameras"]) for eid in alive_ids)
            if total_cameras == 0:
                return

            needed = max(1, math.ceil(total_cameras / self.max_cameras_per_engine))
            if needed >= len(alive_ids):
                return  # already as tight as it can be

            # Keep the `needed` most-loaded engines; drain the rest.
            by_load = sorted(alive_ids, key=lambda eid: len(self.engines[eid]["cameras"]))
            drain_ids = by_load[:len(alive_ids) - needed]
            keep_ids = by_load[len(alive_ids) - needed:]

            logger.info(
                "rebalance: %d cameras across %d engines (cap=%d) only need %d — draining engines %s into %s",
                total_cameras, len(alive_ids), self.max_cameras_per_engine, needed, drain_ids, keep_ids,
            )

            for eid in drain_ids:
                cam_ids = list(self.engines[eid]["cameras"])
                for cid in cam_ids:
                    self._migrate_camera_locked(cid, keep_ids)
                # Anything left (shouldn't be, defensively re-checked) blocks
                # the stop; only tear down engines that are actually empty.
                if not self.engines[eid]["cameras"]:
                    self._stop_engine_locked(eid)
                else:
                    logger.warning("rebalance: engine %d still has cameras after drain, leaving it up", eid)

            self.on_topology_changed()

    def _migrate_camera_locked(self, camera_id: str, target_ids: List[int]):
        cfg = self._camera_last_config.get(camera_id)
        if cfg is None:
            logger.warning("rebalance: no cached config for camera %s, cannot migrate — leaving in place", camera_id)
            return

        target_eid = None
        for eid in sorted(target_ids, key=lambda e: len(self.engines[e]["cameras"])):
            if len(self.engines[eid]["cameras"]) < self.max_cameras_per_engine:
                target_eid = eid
                break
        if target_eid is None:
            # Shouldn't happen given the ceil() math above, but never lose
            # a camera over a bin-packing edge case — spin a fresh engine.
            target_eid = self._start_engine_locked()

        old_eid = self.camera_to_engine.get(camera_id)
        if old_eid is not None and old_eid in self.engines:
            self.engines[old_eid]["cameras"].discard(camera_id)
            try:
                self.engines[old_eid]["control_queue"].put({"cmd": "remove", "camera_id": camera_id})
            except Exception:
                pass

        add_cmd = dict(cfg)
        add_cmd["cmd"] = "add"
        self.engines[target_eid]["cameras"].add(camera_id)
        self.camera_to_engine[camera_id] = target_eid
        self.engines[target_eid]["control_queue"].put(add_cmd)
        logger.info("rebalance: migrated camera %s: engine %s -> engine %s", camera_id, old_eid, target_eid)

    # ================================================================
    # Full teardown
    # ================================================================
    def shutdown(self):
        with self._lock:
            for eid in list(self.engines.keys()):
                self._stop_engine_locked(eid)
            self.camera_to_engine.clear()
            self._camera_last_config.clear()