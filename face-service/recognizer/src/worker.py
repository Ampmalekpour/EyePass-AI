"""
worker.py (recognizer)
--------------------------------------------------------------------
`RecognitionWorker` replaces the reference `AFRWorker(mp.Process)`.

CONTROL HUB (2026-09): results for live-camera tasks no longer go back
to the detector engine that asked; they go to the control hub's
results stream (`face:internal:hub:results`, keyed by the task's
`track_uid`), which owns every track's recognition state and decides
what the backend sees. See `_emit_track_result`. Enrollment tasks are
unaffected and still reply on their own `rec:results:enroll:{id}` list.

Same job — pull a task, filter crops by landmark confidence, align,
embed, compare against the gallery, aggregate, and hand back a result
— but everything that used to be two in-process `multiprocessing.Queue`
objects (`fr_input_queue` / `fr_output_queue`, both owned by a detector
Engine) is now the shared Redis task queue (`rec:tasks`) and a
per-engine Redis result queue (`rec:results:{engine_id}`), described in
facecore.keys / facecore.codec.

IMPORTANT subtlety (see rec_client.py's docstring for the detector-side
mirror of this): `RecognitionWorker.__init__` runs in the PARENT
process (RecognizerPool constructs these objects before calling
.start()), so it must not touch Redis or load any model — only cheap,
picklable attributes belong here. All of that — the Redis connection,
FaceRecognition/FaceAligner construction, the MinIO gallery download,
warmup — happens inside `run()`, which is the entry point that actually
executes in the spawned child.

Idle vs processing, without tearing the process down:
    `processing_event` (mp.Event, shared with the pool) gates whether
    the run-loop pulls tasks off Redis. `stop_idle()` at the pool level
    terminates the process entirely (unloading the model); pausing
    processing while staying warm just clears the event — the worker
    keeps its models loaded and simply stops popping `rec:tasks`.

--------------------------------------------------------------------
ADD-FACE (2026-09): two new task_types share this same run loop and
this same rec:tasks queue, fairly time-sliced against live recognition
traffic by the same BRPOP:

    "enroll_pose_check"  -> _process_enroll_pose_check()
        One frame, one requested angle (`flag`). Detects+validates
        head pose (pose.py) and returns a pass/fail plus the cropped,
        padded face on approval. No gallery/db/MinIO touched.

    "enroll_commit"       -> _process_enroll_commit()
        The 3 pose-approved crops for one person + their metadata.
        Allocates a brieface.db range (gallery.py), saves the images
        into this worker's already-downloaded gallery folder, embeds
        them through the SAME `FaceRecognition.generate_embeddings()`
        every other gallery image has always gone through (no new
        alignment/embedding code path), uploads the result to MinIO,
        and reopens this worker's own db connection so it can
        recognize the new person on its very next task without waiting
        for the gallery:updated broadcast to every OTHER worker.

Both are dispatched and awaited by `enroll.py`'s EnrollCoordinator,
which runs in the recognizer's main process, not in a worker — see
that file for the request/response side of this.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config
import debug_extras
import gallery
import pose
from alignment import FaceAligner
from facecore.bus import RedisBus
from facecore.codec import decode_task, encode_result
from facecore.debugging import prune_dir, save_montage
from facecore.logging_setup import setup_logger
from facecore.minio_store import (
    PRIVATE_BUCKET,
    download_file_from_minio,
    download_folder_from_minio,
    upload_bytes,
    upload_file,
)
from recognition_engine import FaceRecognition


class RecognitionWorker(mp.Process):
    def __init__(self, worker_id: int, processing_event: "mp.Event", stop_event: "mp.Event",
                 loaded_event: Optional["mp.Event"] = None):
        super().__init__(daemon=True, name=f"RecognitionWorker-{worker_id}")
        self.worker_id = worker_id
        self.processing_event = processing_event
        self.stop_event = stop_event
        self.loaded_event = loaded_event

        # Cheap, picklable-only attributes — everything below is set up
        # inside run(), in the child process.
        self.CONF_THRESHOLD = config.WORKER_CONF_THRESHOLD
        self.use_batching = config.USE_BATCHING
        self.batch_size = config.BATCH_SIZE

        self.bus: Optional[RedisBus] = None
        self.face_recognizer: Optional[FaceRecognition] = None
        self.aligner: Optional[FaceAligner] = None
        self.KNOWN_FACES_DIR: Optional[str] = None
        self.current_track_id = 0
        self.logger: Optional[logging.Logger] = None

    # ------------------------------------------------------------------
    # Startup (child process only)
    # ------------------------------------------------------------------
    def _load_models(self):
        self.logger.info("[AFR-%s] Loading Face Recognition models...", self.worker_id)
        self.face_recognizer = FaceRecognition(model_name=config.MODEL_NAME, use_onnx=config.USE_ONNX)
        self.aligner = FaceAligner(self.worker_id)
        self.logger.info("[AFR-%s] Models loaded. Active backend: %s", self.worker_id, self.face_recognizer.active_backend)

        self.logger.info("[AFR-%s] Downloading gallery from MinIO (%s)...", self.worker_id, config.GALLERY_MINIO_PREFIX)
        self.KNOWN_FACES_DIR = download_folder_from_minio(config.GALLERY_MINIO_PREFIX)

        # Pose-check's face detector (pose.py::detect_face_5pt) reuses
        # face_alignment.align's own module-level MTCNN instance rather
        # than loading a second one — touch the import here, once, at
        # worker startup, so the first enroll_pose_check task a worker
        # handles doesn't pay a cold-import cost mid-request.
        from face_alignment import align as _align  # noqa: F401 (import-for-side-effect)

    def _warmup(self):
        self.logger.info("[AFR-%s] Warming up...", self.worker_id)
        try:
            if os.path.exists(config.WARMUP_IMAGE_PATH):
                self.face_recognizer.generate_embedding(config.WARMUP_IMAGE_PATH, silent=True)
            else:
                self.logger.warning("[AFR-%s] warmup image not found at %s, using dummy array", self.worker_id, config.WARMUP_IMAGE_PATH)
                dummy_pil = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
                self.face_recognizer.generate_embedding(dummy_pil, silent=True)
        except Exception as e:
            self.logger.warning("[AFR-%s] Warmup skipped/failed: %s", self.worker_id, e)
        self.logger.info("[AFR-%s] Warmup done.", self.worker_id)

    # ------------------------------------------------------------------
    # Recognition (verbatim logic from the reference AFRWorker)
    # ------------------------------------------------------------------
    def _decode_image(self, image_bytes: bytes):
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _recognize_face_once(self, crop_info: Dict) -> Tuple[str, float, Dict]:
        img_bytes = crop_info.get("image_bytes")
        landmarks = crop_info.get("landmarks", [])
        if not img_bytes:
            return "Unknown", 0.0, {}

        face_bgr = self._decode_image(img_bytes)
        if face_bgr is None:
            return "Unknown", 0.0, {}

        aligned_pil = self.aligner.align_face(face_bgr, landmarks, track_id=self.current_track_id)
        if aligned_pil is None:
            return "Unknown", 0.0, {}

        try:
            person_id, confidence, ranked_list = self.face_recognizer.compare_image(
                image_input=aligned_pil, folder_path=self.KNOWN_FACES_DIR, silent=True,
            )
            _, _, person_info, rec_status = self.face_recognizer.find_person(
                f"c{person_id}.jpg" if person_id != "Unknown" else "", confidence,
            )
            unknown_result = rec_status == 1 or person_id == "Unknown"

            if config.DEBUG_SAVE_MATCHES and not unknown_result:
                debug_extras.save_match_visualization(
                    face_bgr, aligned_pil, ranked_list, person_id, confidence,
                    self.KNOWN_FACES_DIR, self.current_track_id,
                )
            elif config.DEBUG_SAVE_REJECTED and unknown_result:
                debug_extras.save_rejected_crop(
                    face_bgr, aligned_pil, ranked_list, confidence, self.current_track_id,
                )

            if unknown_result:
                return "Unknown", confidence, person_info
            return str(person_info.get("personnelid", "Unknown")), confidence, person_info
        except Exception as e:
            self.logger.warning("[AFR-%s] Recognition error in single mode: %s", self.worker_id, e)
            return "Unknown", 0.0, {}

    def _recognize_faces_batch(self, crop_infos: List[Dict]) -> List[Tuple[str, float, Dict]]:
        pil_imgs = []
        valid_indices = []

        for i, crop_info in enumerate(crop_infos):
            img_bytes = crop_info.get("image_bytes")
            landmarks = crop_info.get("landmarks", [])
            face_bgr = self._decode_image(img_bytes) if img_bytes else None
            if face_bgr is None:
                pil_imgs.append(None)
                continue

            self.aligner.save_landmarked_crop(face_bgr, landmarks, self.current_track_id, i)
            aligned = self.aligner.align_face(face_bgr, landmarks, track_id=self.current_track_id)
            if aligned is not None:
                pil_imgs.append(aligned)
                valid_indices.append(i)
            else:
                pil_imgs.append(None)

        if not valid_indices:
            return [("Unknown", 0.0, {})] * len(crop_infos)

        aligned_valid = [pil_imgs[i] for i in valid_indices]
        batch_results = self.face_recognizer.compare_images_batched(
            images_input=aligned_valid, folder_path=self.KNOWN_FACES_DIR, silent=True,
        )

        final_results = [("Unknown", 0.0, {})] * len(crop_infos)
        for idx, (person_id, confidence, _ranked) in enumerate(batch_results):
            orig_idx = valid_indices[idx]
            if person_id != "Unknown":
                valid_image_num = (int(person_id) - 1) * 3 + 1
                query_path = f"c{valid_image_num}.jpg"
            else:
                query_path = ""
            _, _, person_info, rec_status = self.face_recognizer.find_person(query_path, confidence)
            pid = "Unknown" if rec_status == 1 or person_id == "Unknown" else str(person_info.get("personnelid", "Unknown"))
            final_results[orig_idx] = (pid, confidence, person_info)

        return final_results

    def _filter_crops_by_landmark_confidence(self, crop_infos: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        threshold = config.LANDMARK_CONF_THRESHOLD
        passed, rejected = [], []
        for crop_info in crop_infos:
            landmarks = crop_info.get("landmarks", [])
            if isinstance(landmarks, np.ndarray) and landmarks.ndim == 2 and landmarks.shape[1] >= 3:
                mean_conf = float(np.mean(landmarks[:, 2]))
            elif isinstance(landmarks, list) and landmarks:
                if isinstance(landmarks[0], dict):
                    confs = [lm.get("conf", 0.0) for lm in landmarks]
                else:
                    confs = [lm[2] if len(lm) >= 3 else 0.0 for lm in landmarks]
                mean_conf = float(np.mean(confs)) if confs else 0.0
            else:
                mean_conf = 0.0
            crop_info["mean_landmark_conf"] = mean_conf
            (passed if mean_conf >= threshold else rejected).append(crop_info)
        return passed, rejected

    def _aggregate_track_results(self, raw_results: List[Tuple[str, float, Dict]]) -> Tuple[str, float, Dict]:
        if not raw_results:
            return "Unknown", 0.0, {}

        person_scores: Dict[str, List[float]] = defaultdict(list)
        person_info_map: Dict[str, Dict] = {}
        for pid, conf, info in raw_results:
            if pid and pid.lower() != "unknown" and pid != "0":
                person_scores[pid].append(conf)
                person_info_map[pid] = info

        if not person_scores:
            return "Unknown", 0.0, {}

        best_pid, best_total = None, -1.0
        for pid, scores in person_scores.items():
            total = sum(scores)
            if total > best_total:
                best_total, best_pid = total, pid

        scores = person_scores[best_pid]
        n = len(scores)
        avg_conf = float(np.mean(scores)) if scores else 0.0

        if n >= config.TRACK_N_MIN_HIGH:
            multiplier = config.MULTIPLIER_FULL
        elif n >= config.TRACK_N_MIN_MID:
            multiplier = config.MULTIPLIER_MID
        else:
            multiplier = config.MULTIPLIER_LOW

        final_confidence = min(1.0, avg_conf * multiplier)
        person_info = person_info_map.get(best_pid, {
            "name": "Unknown", "lastname": "Unknown", "section": "0", "codeid": "0", "personnelid": "0",
        })
        return best_pid, final_confidence, person_info

    def _validate_face(self, person_id: str, confidence: float, conf_threshold: float, info: Dict) -> Tuple[bool, str, str]:
        if not person_id or person_id.lower() == "unknown" or person_id == "0":
            return False, "Face not recognized", "Unknown"
        if confidence < conf_threshold:
            return False, f"Confidence {confidence:.3f} < threshold {conf_threshold:.3f}", person_id
        desc = f"{info.get('name', '')} {info.get('lastname', '')} (Code: {info.get('codeid', '')})"
        return True, desc, person_id

    def _upload_image_to_minio(self, key: str, img: np.ndarray, tag: str) -> bool:
        try:
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                self.logger.warning("[AFR-%s] %s image invalid/empty -> %s", self.worker_id, tag, key)
                return False
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                return False
            upload_bytes(buf.tobytes(), key=key, bucket=PRIVATE_BUCKET, content_type="image/png")
            return True
        except Exception as e:
            self.logger.warning("[AFR-%s] Error uploading %s -> %s: %s", self.worker_id, tag, key, e)
            return False

    def save_prep_for_mysql(
        self, track_id: int, task_type: str, crops_bgr: list, best_frame_bgr: Optional[np.ndarray],
        is_valid: bool, description: str, person_id: str, process_id: int, stream_idx: int,
        video_source: str, voted_info: Dict = None, detection_score: Optional[float] = None,
        missing_count: int = 0,
    ) -> Dict[str, Any]:
        """Uploads the face crop + best frame to MinIO (pipeline data —
        this is the only place in the recognizer that writes to MinIO
        rather than the local debug volume) and returns the result
        payload that gets pushed onto `rec:results:{engine_id}`, in the
        same shape the reference AFRWorker pushed onto `fr_output_queue`."""
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        filename_time_str = current_time.strftime("%H-%M-%S-%f")
        mysql_time_str = current_time.strftime("%H:%M:%S.%f")
        camera_name = str(stream_idx)

        if is_valid and person_id not in ("Unknown", "0"):
            face_folder, frame_folder = "dynamics/KnownFaceImage", "dynamics/KnownCameraImage"
            face_prefix, frame_prefix = "kf", "kc"
            filename_id = person_id
        else:
            face_folder, frame_folder = "dynamics/UnknownFaceImage", "dynamics/UnknownCameraImage"
            face_prefix, frame_prefix = "ukf", "ukc"
            filename_id = "0"

        face_key = f"{face_folder}/{face_prefix}_{camera_name}_{date_str}_{filename_time_str}_t{track_id}_{task_type}_{filename_id}.png"
        frame_key = f"{frame_folder}/{frame_prefix}_{camera_name}_{date_str}_{filename_time_str}_t{track_id}_{task_type}_{filename_id}.png"

        face_url = None
        if crops_bgr and len(crops_bgr) > 0:
            face_bgr = crops_bgr[0]
            if isinstance(face_bgr, np.ndarray) and face_bgr.size > 0:
                if self._upload_image_to_minio(face_key, face_bgr, tag="face_crop"):
                    face_url = face_key

        frame_url = None
        if best_frame_bgr is not None and isinstance(best_frame_bgr, np.ndarray) and best_frame_bgr.size > 0:
            if self._upload_image_to_minio(frame_key, best_frame_bgr, tag="best_frame"):
                frame_url = frame_key

        person_info = voted_info or {}

        return {
            "track_id": track_id,
            "event_type": task_type,
            "personnelid": person_id if is_valid and person_id not in ("Unknown", "0") else "0",
            "date": current_time.date(),
            "time": datetime.strptime(mysql_time_str, "%H:%M:%S.%f").time(),
            "gate_type": video_source,
            "face_image": face_url,
            "camera_image": frame_url,
            "first_name": person_info.get("name", "Unknown"),
            "last_name": person_info.get("lastname", "Unknown"),
            "national_code": person_info.get("codeid", "0"),
            "department": person_info.get("section", "0"),
            "description": description,
            "current_time": current_time,
            "detection_score": detection_score,
            "missing_count": missing_count,
        }

    # ------------------------------------------------------------------
    # Per-task processing (extracted from the reference AFRWorker.run()
    # loop body so run() itself stays a thin dispatch loop)
    # ------------------------------------------------------------------
    def _process_task(self, task: Dict[str, Any]):
        self.current_track_id = task.get("track_id")

        track_id = task.get("track_id")
        crops_dict = task.get("crops", {})
        process_id = task.get("process_id", -1)
        stream_idx = task.get("stream_idx", -1)
        video_source = task.get("video_source", "")

        task_type = task.get("task_type", "finalize")
        finalize_max_crops = task.get("finalize_max_crops", 1)

        best_frame_bgr = None
        bf = task.get("best_frame")
        if bf and bf.get("image_bytes"):
            best_frame_bgr = self._decode_image(bf["image_bytes"])

        crop_infos = []
        for reg_key in ["reg1", "reg2", "reg3"]:
            for crop_info in crops_dict.get(reg_key, []):
                if crop_info.get("image_bytes"):
                    crop_infos.append(crop_info)

        if not crop_infos:
            self.logger.info("[AFR-%s] No valid crops received for track %s", self.worker_id, track_id)
            self._emit_track_result(task, {"personnelid": "0", "description": "no crops in task",
                                           "detection_score": 0.0}, status="skipped", is_valid=False)
            return

        passed_crops, _rejected = self._filter_crops_by_landmark_confidence(crop_infos)

        if not passed_crops:
            voted_pid, voted_conf = "Unknown", 0.0
            voted_info = {"name": "Unknown", "lastname": "Unknown", "section": "0", "codeid": "0", "personnelid": "0"}
            is_valid = False
            desc = "Unknown (All crops rejected by quality filter)"
        else:
            raw_results = []
            if not self.use_batching:
                for crop_info in passed_crops:
                    raw_results.append(self._recognize_face_once(crop_info))
            else:
                for i in range(0, len(passed_crops), self.batch_size):
                    raw_results.extend(self._recognize_faces_batch(passed_crops[i:i + self.batch_size]))

            if task_type in ("periodic", "line_cross", "stopped_roi") or (task_type == "finalize" and finalize_max_crops == 1):
                voted_pid, voted_conf, voted_info = raw_results[0]
            else:
                voted_pid, voted_conf, voted_info = self._aggregate_track_results(raw_results)

            is_valid, desc, _final_pid = self._validate_face(
                person_id=voted_pid, confidence=voted_conf, conf_threshold=self.CONF_THRESHOLD, info=voted_info,
            )

        crops_bgr_for_save = []
        first_crop_bytes = crop_infos[0].get("image_bytes")
        if first_crop_bytes:
            decoded = self._decode_image(first_crop_bytes)
            if decoded is not None:
                crops_bgr_for_save = [decoded]

        display_desc = f"[{task_type.capitalize()}] {desc}"

        payload = self.save_prep_for_mysql(
            track_id=track_id, task_type=task_type, crops_bgr=crops_bgr_for_save,
            best_frame_bgr=best_frame_bgr, is_valid=is_valid, description=display_desc,
            person_id=voted_pid, process_id=process_id, stream_idx=stream_idx,
            video_source=video_source, voted_info=voted_info, detection_score=voted_conf,
            missing_count=0,
        )

        self._emit_track_result(task, payload, status="ok" if is_valid else "unknown", is_valid=is_valid)

    def _emit_track_result(self, task: Dict[str, Any], payload: Dict[str, Any], status: str, is_valid: bool):
        """Every recognition task gets exactly one answer.

        Tasks from the current detector carry `track_uid` and are answered
        on the control hub's results stream (the hub owns the track's
        recognition state). A task without one comes from a pre-hub
        detector (e.g. mid rolling upgrade) and is answered the old way,
        on that engine's own result list."""
        track_uid = task.get("track_uid")
        if track_uid:
            data = dict(payload)
            data.update({
                "uid": track_uid,
                "task_id": task.get("task_id"),
                "stage": task.get("stage") or task.get("task_type"),
                "camera_id": task.get("camera_id"),
                "track_id": task.get("track_id"),
                "status": status,
                "is_valid": bool(is_valid),
                "confidence": float(payload.get("detection_score") or 0.0),
                "worker_id": self.worker_id,
            })
            try:
                self.bus.hub_push_result(data)
            except Exception:
                self.logger.exception("[AFR-%s] failed to push hub result for %s", self.worker_id, track_uid)
            return

        engine_id = task.get("engine_id")
        if engine_id is None:
            self.logger.warning("[AFR-%s] task for track %s carried neither track_uid nor engine_id — "
                                "cannot route result", self.worker_id, task.get("track_id"))
            return
        try:
            self.bus.push_result_bytes(engine_id, encode_result(payload))
        except Exception:
            self.logger.exception("[AFR-%s] failed to push result for track %s", self.worker_id, task.get("track_id"))

    # ------------------------------------------------------------------
    # ADD-FACE — enrollment task handlers (NEW)
    # ------------------------------------------------------------------
    def _process_enroll_pose_check(self, task: Dict[str, Any]):
        """task = {task_type: "enroll_pose_check", engine_id, image_bytes, flag}
        Reply shape — see pose.verify_pose():
            {"approved": True,  "yaw", "pitch", "crop_bytes": <png bytes>}
            {"approved": False, "reason": "no_face_detected" | "pose_mismatch", ...}
        """
        engine_id = task.get("engine_id")
        image_bytes = task.get("image_bytes")
        flag = task.get("flag")

        result: Dict[str, Any]
        try:
            frame_bgr = self._decode_image(image_bytes) if image_bytes else None
            if frame_bgr is None:
                result = {"approved": False, "reason": "bad_image"}
            else:
                verdict = pose.verify_pose(frame_bgr, flag)
                # debug_image (a full BGR ndarray, from pose.py's pure
                # renderer) is for THIS process's own disk write only —
                # never part of the wire contract documented in
                # ADD_FACE.md, so it's popped off before `result` is
                # pickled and sent back to the backend over Redis.
                debug_image = verdict.pop("debug_image", None)
                self._save_enroll_pose_debug(debug_image, flag, verdict.get("approved"),
                                             verdict.get("reason"))
                if verdict.get("approved"):
                    ok, buf = cv2.imencode(".png", verdict.pop("crop_bgr"))
                    verdict["crop_bytes"] = buf.tobytes() if ok else None
                    if not ok:
                        verdict["approved"] = False
                        verdict["reason"] = "crop_encode_failed"
                result = verdict
        except Exception as e:
            self.logger.exception("[AFR-%s] enroll_pose_check failed", self.worker_id)
            result = {"approved": False, "reason": "internal_error", "detail": str(e)}

        if engine_id is None:
            self.logger.warning("[AFR-%s] enroll_pose_check task carried no engine_id", self.worker_id)
            return
        self.bus.push_result_bytes(engine_id, encode_result(result))

    def _save_enroll_pose_debug(self, debug_image, flag, approved: Optional[bool], reason: Optional[str]) -> None:
        """Writes pose.py's rendered debug image (bbox + landmarks + per-
        axis pass/fail burned in) to DEBUG_ENROLL_POSE_DIR. Fail-safe and
        a complete no-op when DEBUG_ENROLL_ENABLED=false or nothing was
        rendered (e.g. a truly malformed image)."""
        if not config.DEBUG_ENROLL_ENABLED or debug_image is None:
            return
        try:
            out_dir = config.DEBUG_ENROLL_POSE_DIR
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            verdict_tag = "approved" if approved else f"rejected_{reason or 'unknown'}"
            out_path = os.path.join(out_dir, f"flag{flag}_{verdict_tag}_{stamp}.jpg")
            cv2.imwrite(out_path, debug_image)
            prune_dir(out_dir, config.DEBUG_ENROLL_MAX_FILES, (".jpg",), self.logger)
        except Exception as e:
            self.logger.debug("[AFR-%s] enroll pose debug-image save failed: %s", self.worker_id, e)

    def _save_enrollment_card(self, person, local_image_paths: List[str], range_start: int,
                               range_end: int, filenames: List[str]) -> None:
        """One composite image per successful enrollment: the exact
        crops that were written into the gallery, plus who they belong
        to and which c<N>.jpg range they got — a standing, at-a-glance
        audit trail of every person ever enrolled through this flow,
        independent of opening brieface.db or the MinIO console (see
        DEBUGGING.md). Purely additive: never touches the actual
        gallery/db/MinIO state, only reads back the files this method
        itself just wrote to disk a moment ago."""
        if not config.DEBUG_ENROLL_ENABLED:
            return
        try:
            cells = []
            for filename, local_path in zip(filenames, local_image_paths):
                img = cv2.imread(local_path)
                cells.append((img, [filename]))

            out_dir = config.DEBUG_ENROLL_COMMIT_DIR
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                out_dir, f"personnelid{person.personnelid}_{stamp}.jpg"
            )
            title = (f"{person.name} {person.lastname} (id {person.personnelid}) "
                     f"| {person.section}/{person.codeid} | range {range_start}-{range_end - 1}")
            ok = save_montage(cells, out_path, cell_size=(160, 160), columns=max(1, len(cells)),
                              title=title, logger=self.logger)
            if ok:
                prune_dir(out_dir, config.DEBUG_ENROLL_MAX_FILES, (".jpg",), self.logger)
        except Exception as e:
            self.logger.debug("[AFR-%s] enrollment card save failed: %s", self.worker_id, e)

    def _process_enroll_commit(self, task: Dict[str, Any]):
        """task = {task_type: "enroll_commit", engine_id, person: {...}, crops: [bytes, ...]}

        person: {"name", "lastname", "section", "codeid", "personnelid"}
        crops:  the ENROLL_IMAGES_PER_PERSON pose-approved crop images
                (raw, unaligned — alignment happens inside
                generate_embeddings(), exactly like every other gallery
                image; see pose.py's module docstring for why that's
                the correct, not merely convenient, choice).

        Caller (enroll.py) is responsible for holding bus.gallery_lock()
        for the full round trip of this task — this method assumes it
        already has exclusive access to the gallery.
        """
        engine_id = task.get("engine_id")
        person_dict = task.get("person") or {}
        crops: List[bytes] = task.get("crops") or []

        result: Dict[str, Any]
        db_local_path = None
        try:
            if not crops:
                raise ValueError("no crops supplied to enroll_commit")

            person = gallery.PersonRecord(
                name=str(person_dict.get("name", "")),
                lastname=str(person_dict.get("lastname", "")),
                section=str(person_dict.get("section", "0")),
                codeid=str(person_dict.get("codeid", "0")),
                personnelid=str(person_dict.get("personnelid", "0")),
            )

            # 1) Fresh copy of brieface.db — not self.face_recognizer's
            #    long-lived read connection, which may be stale relative
            #    to what other workers/commits have already written.
            db_local_path = download_file_from_minio(key=config.GALLERY_DB_MINIO_KEY)
            db_conn = sqlite3.connect(db_local_path)
            try:
                range_start, range_end = gallery.allocate_and_insert(db_conn, person, len(crops))
            finally:
                db_conn.close()

            filenames = gallery.image_filenames(range_start, range_end)

            # 2) Drop the new images into THIS worker's already-downloaded
            #    gallery folder, under the allocated c<N>.jpg names.
            local_image_paths = []
            for filename, crop_bytes in zip(filenames, crops):
                local_path = os.path.join(self.KNOWN_FACES_DIR, filename)
                with open(local_path, "wb") as f:
                    f.write(crop_bytes)
                local_image_paths.append(local_path)

            # 3) Embed — reuses generate_embeddings()'s own incremental
            #    diff (_check_pkl): only these new files get aligned
            #    (align.get_aligned_face — same call every gallery image
            #    has always gone through) and embedded; the rest of the
            #    gallery is untouched. This also rewrites the LOCAL
            #    representations_<model>.pkl in KNOWN_FACES_DIR.
            self.face_recognizer.generate_embeddings(self.KNOWN_FACES_DIR, silent=True)

            # 4) Push everything that changed back to MinIO: the new
            #    images, the updated db, the updated pkl.
            prefix = config.GALLERY_MINIO_PREFIX.rstrip("/")
            for filename, local_path in zip(filenames, local_image_paths):
                upload_file(local_path, key=f"{prefix}/{filename}", content_type="image/jpeg")

            upload_file(db_local_path, key=config.GALLERY_DB_MINIO_KEY, content_type="application/x-sqlite3")

            pkl_filename = f"representations_{config.MODEL_NAME}.pkl"
            pkl_local_path = os.path.join(self.KNOWN_FACES_DIR, pkl_filename)
            if os.path.exists(pkl_local_path):
                upload_file(pkl_local_path, key=f"{prefix}/{pkl_filename}", content_type="application/octet-stream")

            # 5) This worker can recognize the new person immediately —
            #    reopen its own db connection against the file we just
            #    updated, rather than waiting for the gallery:updated
            #    broadcast (that broadcast is still what gets every
            #    OTHER worker/replica to reload; see enroll.py).
            try:
                self.face_recognizer.db_conn.close()
            except Exception:
                pass
            self.face_recognizer.db_conn = sqlite3.connect(db_local_path)
            self.face_recognizer.db_conn.row_factory = sqlite3.Row

            self._save_enrollment_card(person, local_image_paths, range_start, range_end, filenames)

            result = {
                "status": "ok",
                "personnelid": person.personnelid,
                "range_start": range_start,
                "range_end": range_end,
                "filenames": filenames,
            }
            self.logger.info(
                "[AFR-%s] enrolled personnelid=%s as %s..%s",
                self.worker_id, person.personnelid, filenames[0], filenames[-1],
            )
        except Exception as e:
            self.logger.exception("[AFR-%s] enroll_commit failed", self.worker_id)
            result = {"status": "error", "message": str(e)}

        if engine_id is None:
            self.logger.warning("[AFR-%s] enroll_commit task carried no engine_id", self.worker_id)
            return
        self.bus.push_result_bytes(engine_id, encode_result(result))

    # ------------------------------------------------------------------
    # Entry point — runs entirely in the spawned child process
    # ------------------------------------------------------------------
    def run(self):
        self.logger = setup_logger(f"recognizer.worker.{self.worker_id}")
        self.bus = RedisBus(module=config.REDIS_MODULE)
        self.bus.wait_until_available()

        self._load_models()
        self._warmup()

        if self.loaded_event is not None:
            self.loaded_event.set()

        self.logger.info("[AFR-%s] Ready.", self.worker_id)

        while not self.stop_event.is_set():
            if not self.processing_event.is_set():
                time.sleep(0.2)
                continue
            try:
                raw = self.bus.pop_task(timeout=1)
            except Exception:
                self.logger.exception("[AFR-%s] pop_task failed, retrying", self.worker_id)
                time.sleep(1)
                continue
            if raw is None:
                continue
            try:
                task = decode_task(raw)
            except Exception:
                self.logger.exception("[AFR-%s] failed to decode task, dropping", self.worker_id)
                continue
            try:
                task_type = task.get("task_type")
                if task_type == "enroll_pose_check":
                    self._process_enroll_pose_check(task)
                elif task_type == "enroll_commit":
                    self._process_enroll_commit(task)
                else:
                    self._process_task(task)
            except Exception as e:
                self.logger.exception("[AFR-%s] unhandled error processing task", self.worker_id)
                if isinstance(task, dict) and task.get("track_uid"):
                    # the hub must not wait for an answer that will never come
                    self._emit_track_result(task, {"personnelid": "0", "detection_score": 0.0,
                                                   "description": f"error: {e}"},
                                            status="error", is_valid=False)

        self.logger.info("[AFR-%s] Stopped.", self.worker_id)
