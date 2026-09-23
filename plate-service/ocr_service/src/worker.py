"""
worker.py (ocr_service)
--------------------------------------------------------------------
`OcrWorker` replaces the reference `OCRWorker(mp.Process)`
(ocr_worker.py). Same job — pull a task, decode crops, run the
car/motorcycle PaddleOCR pipeline, majority-vote across crops, validate
against the plate-format regex, upload the winning crop + best frame to
MinIO, and hand back a result — but everything that used to be two
in-process `multiprocessing.Queue` objects (`ocr_input_queue` /
`ocr_output_queue`, both owned by a detector Engine, 3 workers spawned
BY that engine) is now the shared Redis task queue
(`plate:internal:ocr:tasks`) and a per-engine Redis result queue
(`plate:internal:ocr:results:{engine_id}`), described in
platecore.keys / platecore.codec.

IMPORTANT subtlety (see detector/src/ocr_client.py's docstring for the
detector-side mirror of this): `OcrWorker.__init__` runs in the PARENT
process (OcrPool constructs these objects before calling .start()), so
it must not touch Redis or load any model — only cheap, picklable
attributes belong here. All of that — the Redis connection, the two
PaddleOCR pipelines, warmup — happens inside `run()`, which is the
entry point that actually executes in the spawned child.

Idle vs processing, without tearing the process down:
    `processing_event` (mp.Event, shared with the pool) gates whether
    the run-loop pulls tasks off Redis. `stop_idle()` at the pool level
    terminates the process entirely (unloading the models); pausing
    processing while staying warm just clears the event — the worker
    keeps its models loaded and simply stops popping the task queue.

Every helper below the OCR-pipeline setup (preprocessing, voting,
validation regexes, save_prep_for_mysql's MinIO key layout) is
UNCHANGED from ocr_worker.py — this is a straight port, not a rewrite
of the recognition logic itself.
--------------------------------------------------------------------
"""

from __future__ import annotations

import multiprocessing as mp
import re
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

import config
import debug_extras
from platecore.bus import RedisBus
from platecore.codec import decode_task, encode_result
from platecore.logging_setup import setup_logger
from platecore.minio_store import PRIVATE_BUCKET, ensure_bucket, upload_bytes


class OcrWorker(mp.Process):
    def __init__(self, worker_id: int, processing_event: "mp.Event", stop_event: "mp.Event",
                 loaded_event: Optional["mp.Event"] = None):
        super().__init__(daemon=True, name=f"OcrWorker-{worker_id}")
        self.worker_id = worker_id
        self.processing_event = processing_event
        self.stop_event = stop_event
        self.loaded_event = loaded_event

        # Cheap, picklable-only attributes — everything below is set up
        # inside run(), in the child process.
        self.CONF_THRESHOLD = config.CONF_THRESHOLD
        self.SAVE_ALL_CROPS = config.SAVE_ALL_CROPS

        self.bus: Optional[RedisBus] = None
        self.car_ocr: Optional[PaddleOCR] = None
        self.motor_ocr: Optional[PaddleOCR] = None
        self.logger = None

    # ------------------------------------------------------------------
    # Startup (child process only)
    # ------------------------------------------------------------------
    def _load_models(self):
        self.logger.info("[OCR-%s] Loading PaddleOCR models...", self.worker_id)
        t0 = time.time()

        self.car_ocr = PaddleOCR(
            use_angle_cls=False,
            det=False,
            rec=True,
            det_model_dir=config.DET_MODEL_DIR,
            rec_model_dir=config.CAR_REC_MODEL_DIR,
            cls_model_dir=config.CLS_MODEL_DIR,
            rec_char_dict_path=config.REC_CHAR_DICT_PATH,
            use_mp=False,
            total_process_num=1,
            use_gpu=config.OCR_USE_GPU,
        )

        self.motor_ocr = PaddleOCR(
            det_model_dir=config.DET_MODEL_DIR,
            rec_model_dir=config.MOTOR_REC_MODEL_DIR,
            cls_model_dir=config.CLS_MODEL_DIR,
            use_angle_cls=False,
            rec_char_dict_path=config.REC_CHAR_DICT_PATH,
            use_mp=False,
            total_process_num=1,
            use_gpu=config.OCR_USE_GPU,
        )

        self.logger.info("[OCR-%s] Models loaded in %.0fms", self.worker_id, (time.time() - t0) * 1000)

        try:
            ensure_bucket(PRIVATE_BUCKET)
        except Exception as e:
            self.logger.warning("[OCR-%s] Could not verify/create MinIO bucket %s: %s", self.worker_id, PRIVATE_BUCKET, e)

    def _warmup(self):
        t0 = time.time()
        dummy_car = np.zeros((64, 256, 3), dtype=np.uint8)
        dummy_motor = np.zeros((200, 240, 3), dtype=np.uint8)
        self.car_ocr.ocr(dummy_car, det=False, rec=True, cls=False)
        self.motor_ocr.ocr(dummy_motor, det=True, rec=True, cls=False)
        self.logger.info("[OCR-%s] Warmup done in %.0fms", self.worker_id, (time.time() - t0) * 1000)

    # ------------------------------------------------------------------
    # Recognition pipeline (verbatim logic from the reference OCRWorker)
    # ------------------------------------------------------------------
    def _decode_image(self, image_bytes: bytes):
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _queue_latency_ms(task_id: Optional[str]) -> Optional[float]:
        """task_id is built as 'engine:camera:track:trigger:<time_ns of
        submission>' (see engine.py's _submit_track_to_ocr). Reusing
        that timestamp gives us queue wait time for free."""
        if not task_id:
            return None
        try:
            submitted_ns = int(str(task_id).rsplit(":", 1)[-1])
            return (time.time_ns() - submitted_ns) / 1e6
        except Exception:
            return None

    def _clahe_hsv(self, bgr, clip=None, grid=None):
        if clip is None:
            clip = config.OCR_CLAHE_CLIP_LIMIT
        if grid is None:
            grid = (config.OCR_CLAHE_GRID_SIZE, config.OCR_CLAHE_GRID_SIZE)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        v2 = clahe.apply(v)
        hsv2 = cv2.merge([h, s, v2])
        return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    def _preprocess_car(self, bgr: np.ndarray) -> np.ndarray:
        img = self._clahe_hsv(bgr)
        # 256x64 is the car recognition model's fixed input size — a
        # model-input invariant, must not drift (unlike the CLAHE
        # params above, which are a genuine operational tunable).
        img = cv2.resize(img, (256, 64), interpolation=cv2.INTER_LANCZOS4)
        return img

    def _ocr_car_once(self, car_img_256x64: np.ndarray):
        rgb = cv2.cvtColor(car_img_256x64, cv2.COLOR_BGR2RGB)
        ocr_result = self.car_ocr.ocr(rgb, det=False, rec=True, cls=False)
        if ocr_result and ocr_result[0]:
            text, conf = ocr_result[0][0]
            return str(text).strip(), float(conf)
        return "", 0.0

    def _vote_car_texts(self, texts, confs):
        """Char-by-char vote for 8-char plates. Only uses candidates
        with len(text)==8. Returns (voted_text, avg_conf)."""
        cand = [(t, c) for t, c in zip(texts, confs) if isinstance(t, str) and len(t) == 8]
        if not cand:
            return "", 0.0

        best_len = 8
        voted = []
        total_conf = 0.0
        for pos in range(best_len):
            votes = Counter()
            best_conf_for_char = {}
            for t, c in cand:
                ch = t[pos]
                votes[ch] += 1
                best_conf_for_char[ch] = max(best_conf_for_char.get(ch, 0.0), c)
            best_char = votes.most_common(1)[0][0]
            voted.append(best_char)
            total_conf += best_conf_for_char.get(best_char, 0.0)

        voted_text = "".join(voted)
        avg_conf = total_conf / best_len
        return voted_text, avg_conf

    def _classify_by_edge_distance(self, box: np.ndarray, img_h: int) -> str:
        min_y = float(box[:, 1].min())
        max_y = float(box[:, 1].max())
        dist_top = min_y
        dist_bottom = img_h - max_y
        return "upper" if dist_top < dist_bottom else "lower"

    def _preprocess_motor(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.resize(bgr, (240, 200), interpolation=cv2.INTER_LANCZOS4)

    def _ocr_motor_once(self, motor_img_240x200: np.ndarray):
        """Returns parts=[upper_dict_or_None, lower_dict_or_None]."""
        W, H = 240, 200
        rgb = cv2.cvtColor(motor_img_240x200, cv2.COLOR_BGR2RGB)
        ocr_result = self.motor_ocr.ocr(rgb, det=True, rec=True, cls=False)

        parts = [None, None]
        if not (ocr_result and ocr_result[0]):
            return parts

        detections = ocr_result[0]
        img_area = float(W * H)

        box_infos = []
        for det in detections:
            box = np.array(det[0], dtype=np.int32)
            text, conf = det[1]
            xs, ys = box[:, 0], box[:, 1]
            w = float(xs.max() - xs.min())
            h = float(ys.max() - ys.min())
            area = w * h
            ratio = area / img_area if img_area else 0.0
            box_infos.append({
                "box": box, "text": str(text).strip(), "conf": float(conf),
                "area": float(area), "ratio": float(ratio), "y": float(ys.mean()),
            })

        big = [b for b in box_infos if b["ratio"] >= config.OCR_MOTOR_MIN_BOX_AREA_RATIO]
        if not big:
            return parts

        top2 = sorted(big, key=lambda b: b["area"], reverse=True)[:2]

        def to_dict(b):
            return {"text": b["text"], "confidence": b["conf"]}

        if len(top2) == 2:
            upper, lower = sorted(top2, key=lambda b: b["y"])
            parts[0] = to_dict(upper)
            parts[1] = to_dict(lower)
        else:
            b = top2[0]
            pos = self._classify_by_edge_distance(b["box"], H)
            if pos == "upper":
                parts[0] = to_dict(b)
            else:
                parts[1] = to_dict(b)

        return parts

    def _vote_line(self, lines, expected_len):
        if not lines:
            return "?" * expected_len
        voted = []
        for pos in range(expected_len):
            votes = Counter(line[pos] for line in lines if len(line) == expected_len)
            voted.append(votes.most_common(1)[0][0] if votes else "?")
        return "".join(voted).strip("?")

    def _vote_motor_parts(self, parts_list):
        """parts_list: list of parts=[upper, lower]. upper expected: 3
        digits, lower expected: 5 digits. Returns (voted_parts, conf)."""
        uppers, lowers = [], []
        upper_confs, lower_confs = [], []

        for parts in parts_list:
            if not parts or len(parts) != 2:
                continue
            u = parts[0] or {"text": "", "confidence": 0.0}
            l = parts[1] or {"text": "", "confidence": 0.0}
            ut = str(u.get("text", "")).strip()
            lt = str(l.get("text", "")).strip()
            uc = float(u.get("confidence", 0.0) or 0.0)
            lc = float(l.get("confidence", 0.0) or 0.0)

            if len(ut) == 3 and ut.isdigit():
                uppers.append(ut)
                upper_confs.append(uc)
            if len(lt) == 5 and lt.isdigit():
                lowers.append(lt)
                lower_confs.append(lc)

        voted_upper = self._vote_line(uppers, expected_len=3) if uppers else ""
        voted_lower = self._vote_line(lowers, expected_len=5) if lowers else ""

        c1 = max(upper_confs) if upper_confs else 0.0
        c2 = max(lower_confs) if lower_confs else 0.0
        final_conf = (c1 + c2) / 2.0

        voted_parts = [
            {"text": voted_upper, "confidence": c1} if voted_upper else None,
            {"text": voted_lower, "confidence": c2} if voted_lower else None,
        ]
        return voted_parts, final_conf

    _DIGIT_MAP = str.maketrans({
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    })

    def _normalize_text(self, text: Optional[str]) -> Tuple[str, str]:
        """Returns (s_for_match, s_compact) — s_compact is the fully
        stripped version used for matching/storage."""
        if text is None:
            return "", ""
        s = str(text).strip()
        s = s.translate(self._DIGIT_MAP)
        s_for_match = re.sub(r"[^\w\n]", "", s)
        s_for_match = s_for_match.replace("_", "")
        s_compact = re.sub(r"[\s\-\_\/\\]+", "", s_for_match)
        return s_for_match, s_compact

    def _digits_only(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        s = str(text).strip().translate(self._DIGIT_MAP)
        return re.sub(r"\D+", "", s)

    def _validate_car(self, text: Optional[str], confidence: Optional[float], conf_threshold: float):
        """Car rules (class=0): confidence gate, normalize + compact,
        must match 2 digits + Persian letter + 3 digits + 2 digits.
        Returns (is_valid, description, normalized_compact)."""
        if text is None:
            return False, "text is None", ""
        if confidence is None:
            return False, "confidence is None", ""
        if confidence < conf_threshold:
            return False, f"confidence {confidence:.3f} < threshold {conf_threshold:.3f}", ""

        _, s_compact = self._normalize_text(text)
        if len(s_compact) != 8:
            return False, f"invalid length ({len(s_compact)}) instead of 8", ""

        letters = "بجلمنیسصقدطعهالف"
        if not re.fullmatch(rf"\d{{2}}[{letters}]\d{{3}}\d{{2}}", s_compact):
            return False, "pattern mismatch (expected ##<PersianLetter>### ##)", ""

        return True, "0", s_compact

    def _validate_motor(self, voted_parts, conf_threshold: float):
        """Motor rules (class=1): need upper (3 digits) + lower (5
        digits), each confidence >= threshold. Returns (is_valid,
        description, display_text, compact_text)."""
        if voted_parts is None or len(voted_parts) != 2:
            return False, "invalid parts length (expected 2)", "", ""

        upper, lower = voted_parts[0], voted_parts[1]
        if upper is None or lower is None:
            return False, "one of parts is None", "", ""

        ut = self._digits_only(upper.get("text"))
        lt = self._digits_only(lower.get("text"))
        uc = upper.get("confidence", None)
        lc = lower.get("confidence", None)

        if not ut or not lt:
            return False, "empty text in one of parts", "", ""
        if len(ut) != 3 or not ut.isdigit():
            return False, f"upper invalid: {ut!r} (expected 3 digits)", "", ""
        if len(lt) != 5 or not lt.isdigit():
            return False, f"lower invalid: {lt!r} (expected 5 digits)", "", ""
        if uc is None or lc is None:
            return False, "missing confidence in one of parts", "", ""
        if uc < conf_threshold or lc < conf_threshold:
            return False, f"confidence below threshold (upper={uc}, lower={lc})", "", ""

        display = f"{ut}\n{lt}"
        compact = f"{ut}{lt}"
        return True, "0", display, compact

    def _upload_image_to_minio(self, key: str, img: np.ndarray, tag: str) -> bool:
        try:
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                self.logger.error("[OCR-%s] [UPLOAD] %s image invalid/empty -> %s", self.worker_id, tag, key)
                return False
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                self.logger.error("[OCR-%s] [UPLOAD] cv2.imencode failed -> %s", self.worker_id, key)
                return False
            upload_bytes(buf.tobytes(), key=key, bucket=PRIVATE_BUCKET, content_type="image/png")
            return True
        except Exception as e:
            self.logger.error("[OCR-%s] [UPLOAD] %s -> %s failed: %s", self.worker_id, tag, key, e)
            return False

    def save_prep_for_mysql(
            self, track_id: int, frame_number: int, crops_bgr: list,
            best_frame_bgr: Optional[np.ndarray], is_valid: bool, description: str,
            plate_number: str, voted_class: int, process_id: int, stream_idx: int,
            video_source: str, save_all_crops: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Uploads crops + best frame to MinIO. Object key layout is
        verbatim from ocr_worker.py: valid -> vp/vf, invalid -> ip/if,
        under dynamics/<Valid|Invalid><Plate|Vehicle>Image/. Django
        resolves these keys through its own PrivateMediaStorage, so the
        layout must not change independently of a backend change."""
        if save_all_crops is None:
            save_all_crops = self.SAVE_ALL_CROPS

        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        filename_time_str = current_time.strftime("%H-%M-%S-%f")
        mysql_time_str = current_time.strftime("%H:%M:%S.%f")
        date_obj = current_time.date()
        time_obj = datetime.strptime(mysql_time_str, "%H:%M:%S.%f").time()

        camera_name = str(stream_idx)

        if is_valid:
            plate_folder, frame_folder = "dynamics/ValidPlateImage", "dynamics/ValidVehicleImage"
            plate_prefix, frame_prefix = "vp", "vf"
        else:
            plate_folder, frame_folder = "dynamics/InvalidPlateImage", "dynamics/InvalidVehicleImage"
            plate_prefix, frame_prefix = "ip", "if"

        plate_key = f"{plate_folder}/{plate_prefix}_{camera_name}_{date_str}_{filename_time_str}.png"
        frame_key = f"{frame_folder}/{frame_prefix}_{camera_name}_{date_str}_{filename_time_str}.png"

        plate_url = None
        if crops_bgr:
            if self._upload_image_to_minio(plate_key, crops_bgr[0], tag="crop"):
                plate_url = plate_key

        frame_url = None
        if best_frame_bgr is not None:
            if self._upload_image_to_minio(frame_key, best_frame_bgr, tag="best_frame"):
                frame_url = frame_key

        vehic_type_hint = 1 if voted_class == 0 else 2 if voted_class == 1 else 0

        return {
            "date": date_obj,
            "time": time_obj,
            "gate_type": video_source,
            "plate_image": plate_url,
            "frame_image": frame_url,
            "description": description,
            "current_time": current_time,
            "plate_result": plate_number if is_valid else "0",
            "plate_type": vehic_type_hint,
        }

    # ------------------------------------------------------------------
    # Result emission
    # ------------------------------------------------------------------
    def _build_result(
            self, task: Dict[str, Any], status: str, plate_text: str = "",
            confidence: float = 0.0, is_valid: bool = False, description: str = "",
            voted_class: int = -1, payload: Optional[Dict[str, Any]] = None,
            error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        result = {
            "task_id": task.get("task_id"),
            "camera_id": task.get("camera_id"),
            "track_id": task.get("track_id"),
            "trigger_type": task.get("trigger_type"),
            "status": status,  # "ok", "invalid", "skipped", "error"
            "worker_id": self.worker_id,
            "plate_text": plate_text,
            "confidence": float(confidence or 0.0),
            "is_valid": bool(is_valid),
            "description": description,
            "voted_class": voted_class,
            "payload": payload,
            "error": error,
            "created_at": time.time(),
        }
        missing = [k for k in ("task_id", "camera_id", "track_id", "trigger_type") if result.get(k) is None]
        if missing:
            self.logger.error("[OCR-%s] [EMIT-FAILED] missing routing fields=%s task_id=%s",
                               self.worker_id, missing, task.get("task_id"))
            return None
        return result

    def _emit_result(self, task: Dict[str, Any], **kwargs):
        result = self._build_result(task, **kwargs)
        if result is None:
            return False
        engine_id = task.get("engine_id")
        if engine_id is None:
            self.logger.warning("[OCR-%s] task %s carried no engine_id — cannot route result",
                                 self.worker_id, task.get("task_id"))
            return False
        try:
            self.bus.push_result_bytes(engine_id, encode_result(result))
        except Exception:
            self.logger.exception("[OCR-%s] failed to push result for task %s", self.worker_id, task.get("task_id"))
            return False
        return True

    # ------------------------------------------------------------------
    # Per-task processing (extracted from the reference OCRWorker.run()
    # loop body so run() itself stays a thin dispatch loop)
    # ------------------------------------------------------------------
    def _process_task(self, task: Dict[str, Any]):
        task_id = task.get("task_id")
        camera_id = task.get("camera_id")
        track_id = task.get("track_id")
        trigger_type = task.get("trigger_type")
        queue_latency_ms = self._queue_latency_ms(task_id)

        missing_contract = [
            name for name, value in {
                "task_id": task_id, "camera_id": camera_id,
                "track_id": track_id, "trigger_type": trigger_type,
            }.items() if value is None
        ]
        if missing_contract:
            self.logger.error("[OCR-%s] [TASK-INVALID] missing=%s task_keys=%s",
                               self.worker_id, missing_contract, list(task.keys()))
            self._emit_result(task, status="error", description="invalid OCR task contract",
                               error=f"missing fields: {missing_contract}")
            return

        crops = task.get("crops", [])
        process_id = task.get("process_id", -1)
        stream_idx = task.get("stream_idx", -1)
        video_source = task.get("video_source", "")

        frame_number = 0
        if crops:
            frame_number = int(crops[0].get("frame_number", 0) or 0)

        self.logger.info(
            "[OCR-%s] [TASK-START] task=%s camera=%s track=%s stage=%s crops=%d queue_latency=%s",
            self.worker_id, task_id, camera_id, track_id, trigger_type, len(crops),
            f"{queue_latency_ms:.0f}ms" if queue_latency_ms is not None else "n/a",
        )

        best_frame_bgr = None
        bf = task.get("best_frame")
        if bf and bf.get("image_bytes"):
            best_frame_bgr = self._decode_image(bf["image_bytes"])

        if not crops:
            self.logger.warning("[OCR-%s] [TASK-SKIP] task=%s reason=no_crops", self.worker_id, task_id)
            self._emit_result(task, status="skipped", description="no crops", error="no crops in OCR task")
            return

        decoded_crops, class_votes, crop_det_scores = [], [], []
        for crop in crops:
            image_bytes = crop.get("image_bytes")
            class_flag = crop.get("class_flag")
            if image_bytes is None:
                continue
            image = self._decode_image(image_bytes)
            if image is None:
                continue
            decoded_crops.append(image)
            class_votes.append(class_flag)
            crop_det_scores.append(round(float(crop.get("score", 0.0) or 0.0), 3))

        if not class_votes:
            self.logger.warning("[OCR-%s] [TASK-SKIP] task=%s reason=all_crop_decodes_failed", self.worker_id, task_id)
            self._emit_result(task, status="skipped", description="no valid crops after decode",
                               error="all crop decodes failed")
            return

        voted_class = max(set(class_votes), key=class_votes.count)
        self.logger.info("[OCR-%s] [VOTE] task=%s voted_class=%s class_votes=%s detection_confidences=%s",
                          self.worker_id, task_id, voted_class, class_votes, crop_det_scores)

        _task_t0 = time.time()

        if voted_class == 0:
            raw_texts, raw_confs, per_crop_ms = [], [], []
            for img in decoded_crops:
                c0 = time.time()
                try:
                    car_in = self._preprocess_car(img)
                    text, conf = self._ocr_car_once(car_in)
                    raw_texts.append(text)
                    raw_confs.append(conf)
                except Exception as e:
                    raw_texts.append("")
                    raw_confs.append(0.0)
                    self.logger.error("[OCR-%s] [CAR-OCR-ERROR] task=%s: %s", self.worker_id, task_id, e)
                per_crop_ms.append((time.time() - c0) * 1000.0)

            voted_text, voted_conf = self._vote_car_texts(raw_texts, raw_confs)
            self.logger.info(
                "[OCR-%s] [CAR-OCR] task=%s candidates=%s per_crop_ms=%s voted=%r voted_conf=%.3f",
                self.worker_id, task_id, list(zip(raw_texts, [round(c, 3) for c in raw_confs])),
                [round(m, 1) for m in per_crop_ms], voted_text, voted_conf,
            )

            is_valid, desc, norm_plate = self._validate_car(voted_text, voted_conf, self.CONF_THRESHOLD)
            self.logger.info("[OCR-%s] [VALIDATE] task=%s valid=%s reason=%r normalized=%r",
                              self.worker_id, task_id, is_valid, desc, norm_plate)

            if config.LOG_DECISION_TRACE:
                self.logger.debug(
                    "[OCR-%s] [DECISION-TRACE] task=%s", self.worker_id, task_id,
                    extra={"fields": {
                        "task_id": task_id, "camera_id": camera_id, "track_id": track_id,
                        "trigger_type": trigger_type, "queue_latency_ms": queue_latency_ms,
                        "voted_class": "car", "n_crops": len(decoded_crops),
                        "candidates": list(zip(raw_texts, [round(c, 3) for c in raw_confs])),
                        "voted_text": voted_text, "voted_conf": round(voted_conf, 3),
                        "is_valid": is_valid, "validation_reason": desc, "normalized_plate": norm_plate,
                    }},
                )
            if config.OCR_SAVE_DECISION_DEBUG:
                debug_extras.save_decision_montage(
                    task_id=task_id, camera_id=camera_id, track_id=track_id, trigger_type=trigger_type,
                    voted_class="car", crops_bgr=decoded_crops,
                    candidate_labels=[f"'{t}' {c:.2f}" for t, c in zip(raw_texts, raw_confs)],
                    voted_text=voted_text, voted_conf=voted_conf, is_valid=is_valid, reason=desc,
                    logger=self.logger,
                )

            payload = self.save_prep_for_mysql(
                track_id=track_id, frame_number=frame_number, crops_bgr=decoded_crops,
                best_frame_bgr=best_frame_bgr, is_valid=is_valid, description=desc,
                plate_number=norm_plate, voted_class=voted_class, process_id=process_id,
                stream_idx=stream_idx, video_source=video_source, save_all_crops=None,
            )
            self._emit_result(task, status="ok" if is_valid else "invalid", plate_text=norm_plate,
                               confidence=voted_conf, is_valid=is_valid, description=desc,
                               voted_class=voted_class, payload=payload)

        elif voted_class == 1:
            raw_parts, per_crop_ms = [], []
            for img in decoded_crops:
                c0 = time.time()
                try:
                    motor_in = self._preprocess_motor(img)
                    parts = self._ocr_motor_once(motor_in)
                    raw_parts.append(parts)
                except Exception as e:
                    raw_parts.append([None, None])
                    self.logger.error("[OCR-%s] [MOTOR-OCR-ERROR] task=%s: %s", self.worker_id, task_id, e)
                per_crop_ms.append((time.time() - c0) * 1000.0)

            voted_parts, voted_conf = self._vote_motor_parts(raw_parts)
            self.logger.info(
                "[OCR-%s] [MOTOR-OCR] task=%s candidates=%s per_crop_ms=%s voted=%s voted_conf=%.3f",
                self.worker_id, task_id, raw_parts, [round(m, 1) for m in per_crop_ms], voted_parts, voted_conf,
            )

            is_valid, desc, display, compact = self._validate_motor(voted_parts, self.CONF_THRESHOLD)
            self.logger.info("[OCR-%s] [VALIDATE] task=%s valid=%s reason=%r display=%r compact=%r",
                              self.worker_id, task_id, is_valid, desc, display, compact)

            if config.LOG_DECISION_TRACE:
                self.logger.debug(
                    "[OCR-%s] [DECISION-TRACE] task=%s", self.worker_id, task_id,
                    extra={"fields": {
                        "task_id": task_id, "camera_id": camera_id, "track_id": track_id,
                        "trigger_type": trigger_type, "queue_latency_ms": queue_latency_ms,
                        "voted_class": "motorcycle", "n_crops": len(decoded_crops),
                        "candidates": raw_parts, "voted_parts": voted_parts,
                        "voted_conf": round(voted_conf, 3), "is_valid": is_valid,
                        "validation_reason": desc, "display_plate": display, "compact_plate": compact,
                    }},
                )
            if config.OCR_SAVE_DECISION_DEBUG:
                debug_extras.save_decision_montage(
                    task_id=task_id, camera_id=camera_id, track_id=track_id, trigger_type=trigger_type,
                    voted_class="motorcycle", crops_bgr=decoded_crops,
                    candidate_labels=[str(p) for p in raw_parts],
                    voted_text=display, voted_conf=voted_conf, is_valid=is_valid, reason=desc,
                    logger=self.logger,
                )

            payload = self.save_prep_for_mysql(
                track_id=track_id, frame_number=frame_number, crops_bgr=decoded_crops,
                best_frame_bgr=best_frame_bgr, is_valid=is_valid, description=desc,
                plate_number=compact, voted_class=voted_class, process_id=process_id,
                stream_idx=stream_idx, video_source=video_source, save_all_crops=None,
            )
            self._emit_result(task, status="ok" if is_valid else "invalid", plate_text=compact,
                               confidence=voted_conf, is_valid=is_valid, description=desc,
                               voted_class=voted_class, payload=payload)

        else:
            self.logger.error("[OCR-%s] [VOTE-ERROR] task=%s unknown voted_class=%s", self.worker_id, task_id, voted_class)
            self._emit_result(task, status="error", description=f"unknown voted_class={voted_class}",
                               voted_class=voted_class, error=f"unknown voted_class={voted_class}")

        total_ms = (time.time() - _task_t0) * 1000.0
        self.logger.info(
            "[OCR-%s] [TASK-DONE] task=%s camera=%s track=%s stage=%s total_processing_ms=%.1f",
            self.worker_id, task_id, camera_id, track_id, trigger_type, total_ms,
            extra={"fields": {
                "task_id": task_id, "camera_id": camera_id, "track_id": track_id,
                "trigger_type": trigger_type, "queue_latency_ms": queue_latency_ms,
                "total_processing_ms": round(total_ms, 1),
            }},
        )

    # ------------------------------------------------------------------
    # Entry point — runs entirely in the spawned child process
    # ------------------------------------------------------------------
    def run(self):
        self.logger = setup_logger(f"ocr_service.worker.{self.worker_id}")
        self.bus = RedisBus(module=config.REDIS_MODULE)
        self.bus.wait_until_available()

        self._load_models()
        self._warmup()

        if self.loaded_event is not None:
            self.loaded_event.set()

        self.logger.info("[OCR-%s] Ready.", self.worker_id)

        while not self.stop_event.is_set():
            if not self.processing_event.is_set():
                time.sleep(config.OCR_IDLE_POLL_INTERVAL_SEC)
                continue
            try:
                raw = self.bus.pop_task(timeout=1)
            except Exception:
                self.logger.exception("[OCR-%s] pop_task failed, retrying", self.worker_id)
                time.sleep(config.OCR_TASK_POP_RETRY_BACKOFF_SEC)
                continue
            if raw is None:
                continue
            try:
                task = decode_task(raw)
            except Exception:
                self.logger.exception("[OCR-%s] failed to decode task, dropping", self.worker_id)
                continue
            try:
                self._process_task(task)
            except Exception:
                self.logger.exception("[OCR-%s] unhandled error processing task", self.worker_id)

        self.logger.info("[OCR-%s] Stopped.", self.worker_id)
