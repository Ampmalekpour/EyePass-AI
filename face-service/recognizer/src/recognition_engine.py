"""
recognition_engine.py (recognizer)
--------------------------------------------------------------------
Verbatim port of fr_worker.py's `FaceRecognition` class (ONNX-primary /
PyTorch-fallback AdaFace embedding + gallery comparison + Tsallis-
entropy decision math) plus the `to_media_url` helper. Behaviour is
unchanged from the reference; only two things were adapted:

  * paths and thresholds now come from recognizer/src/config.py
    (env-overridable) instead of being hardcoded / imported from a
    local config.py,
  * the gallery database and gallery images are pulled from MinIO via
    facecore.minio_store instead of a local minio_uploader.py.

This class is intentionally free of any Redis/multiprocessing
knowledge — it is loaded and used by RecognitionWorker (worker.py),
which owns the process lifecycle, queue plumbing and MinIO uploads for
pipeline artifacts.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import config
from image_utils import find_image_hash, list_images
from facecore.debugging import Stopwatch
from facecore.minio_store import open_sqlite_from_minio
import net
from facecore.logging_setup import setup_logger

logger = setup_logger("recognizer.engine")


def to_media_url(full_path: str, marker: str = "dynamics") -> Optional[str]:
    if not full_path:
        return None
    norm = full_path.replace("\\", "/")
    low = norm.lower()
    key = f"/{marker.lower()}/"
    i = low.find(key)
    if i == -1:
        return None
    return norm[i:]


class FaceRecognition:
    """ONNX-primary / PyTorch-fallback AdaFace embedding + gallery
    comparison engine. One instance per RecognitionWorker process,
    created once in `RecognitionWorker._load_models()`."""

    def __init__(self, model_name: str = None, use_onnx: bool = None):
        self.model_name = model_name or config.MODEL_NAME
        self.use_onnx = config.USE_ONNX if use_onnx is None else use_onnx
        self.active_backend = "pytorch"  # default state, may flip to "onnx" below

        self.adaface_models = {"ir_50": config.PYTORCH_MODEL_PATH}
        self.onnx_models = {"ir_50": config.ONNX_MODEL_PATH}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FaceRecognition device: %s", self.device)

        # --- SMART BACKEND LOADING ---
        if self.use_onnx:
            try:
                self.model = self._load_onnx_model()
                self.active_backend = "onnx"
                logger.info("Successfully loaded ONNX backend for %s", self.model_name)
            except Exception as e:
                logger.warning("ONNX initialization failed: %s. Falling back to PyTorch.", e)
                self.model = self._load_pretrained_model()
        else:
            self.model = self._load_pretrained_model()

        self.transform = None

        self.db_conn = self._load_db_from_minio()

        self.rec_up_thr = config.REC_UP_THR
        self.rec_down_thr = config.REC_DOWN_THR
        self.rec_mid_thr = config.REC_MID_THR

    # ------------------------------------------------------------------
    # Model / gallery-db loading
    # ------------------------------------------------------------------
    def _load_db_from_minio(self) -> sqlite3.Connection:
        return open_sqlite_from_minio(config.GALLERY_DB_MINIO_KEY)

    def _load_onnx_model(self):
        start_time = time.time()
        model_path = self.onnx_models.get(self.model_name)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.ONNX_INTRA_OP_THREADS
        opts.inter_op_num_threads = config.ONNX_INTER_OP_THREADS
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # --- WARMUP ---
        dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
        for _ in range(config.ONNX_WARMUP_ITERATIONS):
            session.run(None, {"input_tensor": dummy_input})

        logger.info("ONNX loading and warmup time: %.3f seconds", time.time() - start_time)
        return session

    def __del__(self):
        if hasattr(self, "db_conn"):
            try:
                self.db_conn.close()
            except Exception:
                pass

    def _load_pretrained_model(self):
        start_time = time.time()
        if self.model_name not in self.adaface_models:
            raise ValueError(f"Invalid architecture: {self.model_name}")
        model = net.build_model(self.model_name)
        model_path = self.adaface_models[self.model_name]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        statedict = torch.load(model_path, map_location=self.device)["state_dict"]
        model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
        model.load_state_dict(model_statedict)
        model.eval()
        model = model.to(self.device)
        logger.info("Model loading time: %.3f seconds", time.time() - start_time)
        return model

    # ------------------------------------------------------------------
    # Gallery .pkl management (verbatim)
    # ------------------------------------------------------------------
    def _check_pkl(self, folder_path: str, silent: bool = False) -> Tuple[
        List[Dict[str, Any]], List[str], List[str], bool
    ]:
        pkl_filename = f"representations_{self.model_name}.pkl"
        pkl_path = os.path.join(folder_path, pkl_filename)
        unchanged_representations = []
        images_to_process = []
        deleted_images = []
        valid_pkl = False
        if not os.path.exists(pkl_path):
            images_to_process = list_images(folder_path)
            return unchanged_representations, images_to_process, deleted_images, valid_pkl
        try:
            with open(pkl_path, "rb") as f:
                representations = pickle.load(f)
            if not isinstance(representations, list):
                raise ValueError(f"Expected list in .pkl, got {type(representations)}")
            valid_pkl = True
        except Exception as e:
            logger.error("Failed to load .pkl file %s: %s", pkl_path, e)
            images_to_process = list_images(folder_path)
            return unchanged_representations, images_to_process, deleted_images, valid_pkl
        for rep in representations:
            if not all(key in rep for key in ["identity", "hash", "embedding"]):
                valid_pkl = False
                break
            if not isinstance(rep["embedding"], list) or len(rep["embedding"]) != 512:
                valid_pkl = False
                break
        if not valid_pkl:
            images_to_process = list_images(folder_path)
            return unchanged_representations, images_to_process, deleted_images, valid_pkl
        current_images = list_images(folder_path)
        current_hashes = {}
        new_images = []
        modified_images = []
        for img_path in current_images:
            try:
                current_hashes[img_path] = find_image_hash(img_path)
            except Exception:
                continue
        pkl_identities = {rep["identity"] for rep in representations}
        for img_path in current_images:
            if img_path not in pkl_identities:
                new_images.append(img_path)
            else:
                for rep in representations:
                    if rep["identity"] == img_path:
                        if rep["hash"] != current_hashes.get(img_path):
                            modified_images.append(img_path)
                        else:
                            unchanged_representations.append(rep)
                        break
        deleted_images = [rep["identity"] for rep in representations if rep["identity"] not in current_images]
        unchanged_representations = [rep for rep in unchanged_representations if rep["identity"] in current_images]
        images_to_process = new_images + modified_images
        return unchanged_representations, images_to_process, deleted_images, valid_pkl

    def _save_embeddings_to_pkl(self, representations: List[Dict[str, Any]], folder_path: str,
                                 silent: bool = False, temp_file: str = None) -> None:
        os.makedirs(folder_path, exist_ok=True)
        pkl_filename = f"representations_{self.model_name}.pkl"
        pkl_path = os.path.join(folder_path, pkl_filename)
        final_path = temp_file if temp_file else pkl_path
        max_retries = config.PKL_SAVE_MAX_RETRIES
        retry_delay = config.PKL_SAVE_RETRY_DELAY_SEC
        for attempt in range(max_retries + 1):
            try:
                with open(final_path, "wb") as f:
                    pickle.dump(representations, f)
                    f.flush()
                    os.fsync(f.fileno())
                if temp_file and final_path != pkl_path:
                    for _ in range(3):
                        try:
                            shutil.move(temp_file, pkl_path)
                            break
                        except Exception:
                            time.sleep(retry_delay)
                    else:
                        raise IOError(f"Failed to move {temp_file} to {pkl_path} after retries")
                break
            except Exception as e:
                logger.error("Attempt %d/%d: failed to save .pkl file %s: %s", attempt + 1, max_retries + 1, pkl_path, e)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                if temp_file:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(representations, f)
                        f.flush()
                        os.fsync(f.fileno())
                    break
                raise IOError(f"Failed to save .pkl file: {e}")

    def _update_pkl(self, unchanged_representations: List[Dict[str, Any]], images_to_process: List[str],
                     deleted_images: List[str], folder_path: str, silent: bool = False) -> List[Dict[str, Any]]:
        new_representations = []
        for image_path in tqdm(images_to_process, desc="Updating embeddings", disable=silent):
            try:
                embedding, file_hash = self.generate_embedding(image_path, silent)
                new_representations.append({"identity": image_path, "hash": file_hash, "embedding": embedding})
            except Exception as e:
                logger.warning("Skipping %s due to error: %s", image_path, e)
                continue
        representations = unchanged_representations + new_representations
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=folder_path, suffix=".pkl") as temp:
                temp_file = temp.name
            self._save_embeddings_to_pkl(representations, folder_path, silent, temp_file)
        except Exception as e:
            if "temp_file" in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            raise IOError(f"Failed to update .pkl: {e}")
        return representations

    def generate_embeddings(self, folder_path: str, normalization: str = "base", silent: bool = False) -> List[
        Dict[str, Any]
    ]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        unchanged_representations, images_to_process, deleted_images, valid_pkl = self._check_pkl(folder_path, silent)
        if valid_pkl and not images_to_process and not deleted_images and unchanged_representations:
            return unchanged_representations
        if not images_to_process and not deleted_images:
            image_paths = list_images(folder_path)
            if not image_paths:
                raise ValueError(f"No valid images (.jpg, .jpeg, .png) found in {folder_path}")
            images_to_process = image_paths
        return self._update_pkl(unchanged_representations, images_to_process, deleted_images, folder_path, silent)

    # ------------------------------------------------------------------
    # Comparison / embedding generation (verbatim)
    # ------------------------------------------------------------------
    def compare_image(self, image_input: Union[str, Image.Image], folder_path: str,
                       similarity_metric: str = "cosine_similarity", silent: bool = False) -> Tuple[
        str, float, List[Tuple[str, float]]
    ]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        if similarity_metric != "cosine_similarity":
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        pkl_path = os.path.join(folder_path, f"representations_{self.model_name}.pkl")
        with open(pkl_path, "rb") as f:
            representations = pickle.load(f)

        if len(representations) == 0:
            return "Unknown", 0.0, []

        try:
            if isinstance(image_input, Image.Image) and image_input.size == (112, 112):
                np_img = np.array(image_input)
                bgr = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
                transposed = bgr.transpose(2, 0, 1).astype(np.float32)

                if self.active_backend == "onnx":
                    batched_np = np.expand_dims(transposed, axis=0)
                    model_output = self.model.run(None, {"input_tensor": batched_np})[0]
                    embedding = model_output[0].flatten().tolist()
                else:
                    img_tensor = torch.from_numpy(transposed).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        model_output = self.model(img_tensor)
                        embedding = model_output[0].cpu().numpy().flatten().tolist()
            else:
                embedding, _ = self.generate_embedding(image_input, silent)
        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            raise

        input_embedding = np.array([embedding], dtype=np.float32)

        pkl_embeddings = []
        pkl_paths = []
        for rep in representations:
            try:
                pkl_emb = np.array(rep["embedding"], dtype=np.float32)
                if pkl_emb.shape != (512,):
                    continue
                pkl_embeddings.append(pkl_emb)
                pkl_paths.append(rep["identity"])
            except Exception:
                continue

        if not pkl_embeddings:
            return "Unknown", 0.0, []

        pkl_embeddings = np.stack(pkl_embeddings)
        cos_sim_matrix = cosine_similarity(input_embedding, pkl_embeddings)
        cos_sim_scores = cos_sim_matrix[0]

        similarities = [(path, float(score)) for path, score in zip(pkl_paths, cos_sim_scores)]
        similarities.sort(key=lambda x: x[1], reverse=True)

        person_id, confidence, ranked_list = self._decide_person_and_confidence(similarities, top_k=config.DECISION_TOP_K)
        return person_id, confidence, ranked_list

    def compare_images_batched(self, images_input: List[Image.Image], folder_path: str,
                                similarity_metric: str = "cosine_similarity", silent: bool = False) -> List[
        Tuple[str, float, List[Tuple[str, float]]]
    ]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        pkl_path = os.path.join(folder_path, f"representations_{self.model_name}.pkl")
        with open(pkl_path, "rb") as f:
            representations = pickle.load(f)

        pkl_embeddings_np = np.array([rep["embedding"] for rep in representations], dtype=np.float32)
        pkl_paths = [rep["identity"] for rep in representations]

        if len(pkl_embeddings_np) == 0:
            return [("Unknown", 0.0, []) for _ in images_input]

        valid_arrays = []
        valid_indices = []
        for i, pil_img in enumerate(images_input):
            if pil_img is None:
                continue
            np_img = np.array(pil_img)
            bgr = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
            transposed = bgr.transpose(2, 0, 1).astype(np.float32)
            valid_arrays.append(transposed)
            valid_indices.append(i)

        if not valid_arrays:
            return [("Unknown", 0.0, []) for _ in images_input]

        batched_np = np.stack(valid_arrays, axis=0)

        if self.active_backend == "onnx":
            batch_embeddings = self.model.run(None, {"input_tensor": batched_np})[0]
        else:
            batch_tensor = torch.from_numpy(batched_np).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor).cpu().numpy()

        cos_sim_matrix = cosine_similarity(batch_embeddings, pkl_embeddings_np)

        results = []
        for batch_idx, original_idx in enumerate(valid_indices):
            scores = cos_sim_matrix[batch_idx]
            raw_sims = [(pkl_paths[j], float(scores[j])) for j in range(len(scores))]
            raw_sims.sort(key=lambda x: x[1], reverse=True)
            person_id, confidence, ranked_list = self._decide_person_and_confidence(raw_sims, top_k=config.DECISION_TOP_K)
            results.append((person_id, confidence, ranked_list))

        return results

    def generate_embedding(self, image_input: Union[str, Image.Image], silent: bool = False) -> Tuple[List[float], str]:
        image_path_for_log = "PIL image" if isinstance(image_input, Image.Image) else image_input
        sw = Stopwatch()
        try:
            if isinstance(image_input, str):
                if not os.path.isfile(image_input):
                    raise ValueError(f"Image file does not exist: {image_input}")
                file_hash = find_image_hash(image_input)
            else:
                file_hash = ""

            # Local import: `align` comes from the operator-supplied
            # `face_alignment` package (see alignment.py / README).
            from face_alignment import align as _align

            if isinstance(image_input, str):
                aligned_rgb_img = _align.get_aligned_face(image_path=image_input)
            else:
                aligned_rgb_img = _align.get_aligned_face(None, rgb_pil_image=image_input)

            if aligned_rgb_img is None:
                raise ValueError(f"No face detected in {image_path_for_log}")

            np_img = np.array(aligned_rgb_img)
            bgr_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
            transposed = bgr_img.transpose(2, 0, 1).astype(np.float32)

            if self.active_backend == "onnx":
                batched_np = np.expand_dims(transposed, axis=0)
                model_output = self.model.run(None, {"input_tensor": batched_np})[0]
                embedding = model_output[0].flatten().tolist()
            else:
                img_tensor = torch.from_numpy(transposed).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    model_output = self.model(img_tensor)
                    embedding = model_output[0].cpu().numpy().flatten().tolist()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "embedding generated elapsed_ms=%.2f backend=%s dim=%d",
                    sw.stop(), self.active_backend, len(embedding),
                    extra={"fields": {
                        "event": "embedding_generated",
                        "image": os.path.basename(str(image_path_for_log)),
                        "backend": self.active_backend,
                        "elapsed_ms": round(sw.elapsed_ms, 2),
                        "embedding_dim": len(embedding),
                        "embedding_norm": round(float(np.linalg.norm(embedding)), 4),
                    }},
                )

            return embedding, file_hash
        except Exception as e:
            logger.error("Failed to generate embedding for %s: %s (elapsed_ms=%.2f)",
                         image_path_for_log, e, sw.stop())
            raise

    # ------------------------------------------------------------------
    # Decision math (verbatim — this is the behaviour that must not
    # drift; see the module README for the rationale)
    # ------------------------------------------------------------------
    def _calc_tsallis_confidence(self, fused_scores: np.ndarray, top_k: int = None, tau: float = None) -> float:
        top_k = config.DECISION_TOP_K if top_k is None else top_k
        tau = config.TSALLIS_TAU if tau is None else tau
        if len(fused_scores) < 2:
            return float(fused_scores[0]) if len(fused_scores) == 1 else 0.0
        k = min(len(fused_scores), top_k)
        sliced_scores = fused_scores[:k]
        exp_scores = np.exp(sliced_scores / tau)
        probs = exp_scores / (np.sum(exp_scores) + 1e-8)
        return float(np.sum(probs ** 2))

    def _decide_person_and_confidence(self, similarities, top_k=None, alpha=None, tau=None) -> Tuple[
        str, float, List[Tuple[str, float]]
    ]:
        top_k = config.DECISION_TOP_K if top_k is None else top_k
        alpha = config.POOLING_ALPHA if alpha is None else alpha
        tau = config.TSALLIS_TAU if tau is None else tau

        if not similarities:
            return "Unknown", 0.0, []

        identity_map: Dict[str, List[float]] = {}
        ranked_references = []

        for path, score in similarities[:top_k]:
            match = re.search(r"c(\d+)", os.path.basename(path))
            if match:
                image_num = int(match.group(1))
                pid = str((image_num - 1) // 3 + 1)
            else:
                pid = "Unknown"

            ranked_references.append((pid, score, path))
            identity_map.setdefault(pid, []).append(score)

        pooled_identities = []
        for pid, scores in identity_map.items():
            if pid == "Unknown":
                fused_score = max(scores) if scores else 0.0
            else:
                sorted_scores = sorted(scores, reverse=True)
                weights = np.exp(-alpha * np.arange(len(sorted_scores)))
                fused_score = np.sum(weights * sorted_scores) / np.sum(weights)
            pooled_identities.append({"pid": pid, "fused_score": float(fused_score)})

        pooled_identities.sort(key=lambda x: x["fused_score"], reverse=True)
        best_pid = pooled_identities[0]["pid"]

        fused_scores_np = np.array([item["fused_score"] for item in pooled_identities])
        raw_tsallis_confidence = self._calc_tsallis_confidence(fused_scores_np, top_k=top_k, tau=tau)
        confidence = min(1.0, max(0.0, raw_tsallis_confidence))

        penalty_applied = "none"
        top_raw_score = ranked_references[0][1] if ranked_references else 0.0

        if best_pid != "Unknown":
            if confidence >= self.rec_up_thr:
                if top_raw_score >= config.HIGH_CONFIDENCE_RAW_SCORE_THR:
                    pass
                else:
                    confidence = confidence * config.CONF_PENALTY_HIGH_LOW_RAW
                    penalty_applied = "high_low_raw"
            elif self.rec_mid_thr <= confidence < self.rec_up_thr:
                confidence = confidence * config.CONF_PENALTY_MID
                penalty_applied = "mid"
            else:
                best_pid = "Unknown"
                penalty_applied = "below_down_thr->unknown"

        if best_pid == "Unknown":
            confidence = 0.0

        # ------------------------------------------------------------------
        # Decision trace — "whatever a debugger needs to see in terms of
        # times and embeddings and similarities and tsallis": every raw
        # similarity considered, the per-identity fused score, the tsallis
        # confidence BEFORE any penalty, which (if any) penalty fired, and
        # the final decision. Costs nothing unless LOG_LEVEL=DEBUG (the
        # isEnabledFor guard skips building this dict entirely otherwise);
        # reads as one JSON object per decision when LOG_FORMAT=json.
        # ------------------------------------------------------------------
        if config.LOG_DECISION_TRACE and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "decision pid=%s confidence=%.4f", best_pid, confidence,
                extra={"fields": {
                    "event": "recognition_decision",
                    "top_k": top_k, "alpha": alpha, "tau": tau,
                    "raw_similarities_considered": [
                        {"pid": pid, "score": round(score, 4), "path": os.path.basename(path)}
                        for pid, score, path in ranked_references
                    ],
                    "fused_scores_by_identity": [
                        {"pid": item["pid"], "fused_score": round(item["fused_score"], 4)}
                        for item in pooled_identities
                    ],
                    "top_raw_score": round(top_raw_score, 4),
                    "tsallis_confidence_raw": round(raw_tsallis_confidence, 4),
                    "confidence_final": round(confidence, 4),
                    "penalty_applied": penalty_applied,
                    "best_pid": best_pid,
                    "thresholds": {"up": self.rec_up_thr, "mid": self.rec_mid_thr, "down": self.rec_down_thr},
                }},
            )

        return best_pid, confidence, similarities

    def postrecprocess(self, similarities: List[Tuple[str, float]]) -> Optional[str]:
        if not isinstance(similarities, list):
            raise ValueError(f"Expected list of tuples, got {type(similarities)}")
        for item in similarities:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], float):
                raise ValueError(f"Invalid similarity list entry: {item}")
        if not similarities:
            return None
        return sorted(similarities, key=lambda x: x[1], reverse=True)[0][0]

    def _find_gallery_folder_with_pkl(self, folder_path: str, silent: bool = False) -> Optional[str]:
        """Search folder_path and its subfolders for representations_<model>.pkl."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        pkl_filename = f"representations_{self.model_name}.pkl"
        direct_pkl = os.path.join(folder_path, pkl_filename)
        if os.path.isfile(direct_pkl):
            return folder_path
        for root, dirs, files in os.walk(folder_path):
            if pkl_filename in files:
                return root
        return None

    def search_folder(self, image_input: Union[str, Image.Image], folder_path: str,
                       similarity_metric: str = "cosine_similarity", silent: bool = False) -> Dict[str, Any]:
        """Complete folder-search pipeline: locate/build the gallery pkl
        under folder_path, embed image_input, and return the recognition
        result plus DB person info. Not on the hot recognition path
        (that uses compare_image/compare_images_batched against the
        already-loaded gallery folder) — kept for parity with the
        reference tool, e.g. ad-hoc gallery inspection."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        if similarity_metric != "cosine_similarity":
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        gallery_folder = self._find_gallery_folder_with_pkl(folder_path, silent=silent)
        if gallery_folder is None:
            self.generate_embeddings(folder_path, silent=silent)
            gallery_folder = folder_path
        else:
            self.generate_embeddings(gallery_folder, silent=silent)

        embedding, _ = self.generate_embedding(image_input, silent=silent)

        pkl_path = os.path.join(gallery_folder, f"representations_{self.model_name}.pkl")
        with open(pkl_path, "rb") as f:
            representations = pickle.load(f)

        unknown_person = {"name": "Unknown", "lastname": "Unknown", "section": "0", "codeid": "0", "personnelid": "0"}

        if not representations:
            return {"gallery_folder": gallery_folder, "person_id": "Unknown", "confidence": 0.0,
                     "ranked_list": [], "person_info": unknown_person, "recognition_status": 1}

        input_embedding = np.array([embedding], dtype=np.float32)
        pkl_embeddings, pkl_paths = [], []
        for rep in representations:
            try:
                pkl_emb = np.array(rep["embedding"], dtype=np.float32)
                if pkl_emb.shape != (512,):
                    continue
                pkl_embeddings.append(pkl_emb)
                pkl_paths.append(rep["identity"])
            except Exception:
                continue

        if not pkl_embeddings:
            return {"gallery_folder": gallery_folder, "person_id": "Unknown", "confidence": 0.0,
                     "ranked_list": [], "person_info": unknown_person, "recognition_status": 1}

        pkl_embeddings = np.stack(pkl_embeddings)
        cos_sim_scores = cosine_similarity(input_embedding, pkl_embeddings)[0]
        similarities = sorted(
            [(path, float(score)) for path, score in zip(pkl_paths, cos_sim_scores)],
            key=lambda x: x[1], reverse=True,
        )

        person_id, confidence, ranked_list = self._decide_person_and_confidence(similarities, top_k=config.DECISION_TOP_K)

        person_info = unknown_person
        recognition_status = 1
        if ranked_list:
            best_path, _ = ranked_list[0]
            _, _, person_info, recognition_status = self.find_person(best_path, confidence)

        return {
            "gallery_folder": gallery_folder,
            "person_id": person_id,
            "confidence": confidence,
            "ranked_list": ranked_list,
            "person_info": person_info,
            "recognition_status": recognition_status,
        }

    def find_person(self, path: str, similarity: float) -> Tuple[str, float, Dict, int]:
        unknown = {"name": "Unknown", "lastname": "Unknown", "section": "0", "codeid": "0", "personnelid": "0"}
        try:
            if similarity < config.MIN_RAW_SIMILARITY_THR:
                return (path, similarity, unknown, 1)
            filename = os.path.basename(path)
            match = re.match(r"^c(\d+)\.jpg$", filename)
            if not match:
                return (path, similarity, unknown, 1)
            num = int(match.group(1))
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT name, lastname, section, codeid, personnelid
                FROM brieface
                WHERE range_start <= ? AND range_end > ?
                """,
                (num, num),
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                person_info = {
                    "name": row["name"],
                    "lastname": row["lastname"],
                    "section": row["section"],
                    "codeid": row["codeid"],
                    "personnelid": row["personnelid"],
                }
                recognition_status = 0 if person_info.get("name") != "Unknown" else 1
                return (path, similarity, person_info, recognition_status)
            return (path, similarity, unknown, 1)
        except Exception as e:
            logger.error("Failed to query database: %s", e)
            return (path, similarity, unknown, 1)
