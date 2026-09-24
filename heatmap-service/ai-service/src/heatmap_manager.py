"""
heatmap_manager.py
--------------------------------------------------------------------
Core data manager for the heatmap system. Carried over unchanged in
behavior from the pre-existing standalone build — this file's math
(cube shape, slot/cell indexing, accumulate/aggregate/render/save) is
the actual reusable pipeline logic this whole module exists to run.
What changed: it now stores through common/heatmapcore/minio_store.py
(boto3, matching the plate/face/fire modules' own client) instead of
the `minio` SDK-based minio_client.py, and gained current_slot_counts()
for the new visual-debug overlay (debug_recorder.py).

Stores detection data as a 3D numpy array called a "cube":
    shape = (time_slots, grid_height, grid_width)
One cube object is stored per day per camera, in MinIO
(key: "<camera_id>/<date>.npy" inside the configured bucket).
--------------------------------------------------------------------
"""

import cv2
import logging
import threading
import numpy as np

from datetime import datetime
from typing import Optional, Any

from config import GridConfig, StorageConfig, RenderConfig
from heatmapcore.minio_store import MinioArrayStore

logger = logging.getLogger(__name__)


class HeatmapCubeManager:
    def __init__(
        self,
        grid_config: GridConfig,
        storage_config: StorageConfig,
        render_config: Optional[RenderConfig] = None,
        store: Optional[MinioArrayStore] = None,
    ):
        self.grid = grid_config
        self.storage = storage_config
        self.render_cfg = render_config or RenderConfig()

        # total number of time slots in a day based on resolution (e.g. 5-min -> 288 slots)
        self.num_time_slots = (24 * 60) // self.storage.time_resolution_minutes

        # Object key prefix — "<camera_id>/" inside a single MinIO bucket.
        self.camera_prefix = self.storage.camera_id or "default"

        self.store = store or MinioArrayStore()

        # pixel size of each grid cell — used to map pixel (x,y) -> grid (col, row)
        self.cell_width = self.grid.frame_width / self.grid.grid_width
        self.cell_height = self.grid.frame_height / self.grid.grid_height

        # in-memory cache: { "YYYY-MM-DD": np.ndarray }
        self.loaded_cubes: dict = {}
        self._lock = threading.Lock()

        logger.info(
            "HeatmapCubeManager initialized | grid=%dx%d | frame=%dx%d | "
            "bucket=%s | prefix=%s",
            self.grid.grid_width, self.grid.grid_height,
            self.grid.frame_width, self.grid.frame_height,
            self.store.bucket, self.camera_prefix,
        )

    # =========================================================
    # INTERNAL HELPERS — always hold self._lock before calling
    # _load_or_create_cube.
    # =========================================================

    def _create_empty_cube(self) -> np.ndarray:
        return np.zeros(
            (self.num_time_slots, self.grid.grid_height, self.grid.grid_width),
            dtype=np.uint32,
        )

    def _get_cube_key(self, date_str: str) -> str:
        return f"{self.camera_prefix}/{date_str}.npy"

    def _load_or_create_cube(self, date_str: str) -> np.ndarray:
        # NOTE: must be called while holding self._lock
        if date_str in self.loaded_cubes:
            return self.loaded_cubes[date_str]

        object_key = self._get_cube_key(date_str)
        cube = self.store.download_array(object_key)
        is_new_allocation = cube is None

        if cube is None:
            cube = self._create_empty_cube()
            logger.debug("Created new cube for date: %s", date_str)
        else:
            logger.debug("Loaded cube from MinIO: %s", object_key)

        size_mb = cube.nbytes / (1024 * 1024)
        allocation_type = "Created Fresh" if is_new_allocation else "Loaded from MinIO"
        logger.debug("[RAM Allocation] Camera: %s | Date: %s | Action: %s | Size: %.2f MB",
                      self.camera_prefix, date_str, allocation_type, size_mb)

        self.loaded_cubes[date_str] = cube
        return cube

    # =========================================================
    # PUBLIC API
    # =========================================================

    def accumulate(self, x: float, y: float, timestamp: Optional[datetime] = None) -> None:
        """Records one detection point into the correct cell of the cube."""
        if timestamp is None:
            timestamp = datetime.now()

        date_str = timestamp.strftime("%Y-%m-%d")
        total_minutes = timestamp.hour * 60 + timestamp.minute
        time_index = total_minutes // self.storage.time_resolution_minutes

        grid_x = int(x / self.cell_width)
        grid_y = int(y / self.cell_height)
        grid_x = max(0, min(grid_x, self.grid.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid.grid_height - 1))

        with self._lock:
            cube = self._load_or_create_cube(date_str)
            cube[time_index][grid_y][grid_x] += 1

    def current_slot_counts(self, timestamp: Optional[datetime] = None) -> Optional[np.ndarray]:
        """Returns the (grid_height, grid_width) slice for the current
        (or given) time slot of TODAY's cube, or None if nothing has
        been accumulated yet — used only by debug_recorder.py's live
        overlay, never by the actual storage path."""
        timestamp = timestamp or datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        total_minutes = timestamp.hour * 60 + timestamp.minute
        time_index = total_minutes // self.storage.time_resolution_minutes

        with self._lock:
            if date_str not in self.loaded_cubes:
                return None
            return self.loaded_cubes[date_str][time_index].copy()

    def aggregate_time_range(
        self, date_str: str, start_hour: int, start_minute: int, end_hour: int, end_minute: int,
    ) -> np.ndarray:
        """Sums all time slots in the given range into a single 2D array."""
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        start_index = start_total // self.storage.time_resolution_minutes
        end_index = end_total // self.storage.time_resolution_minutes

        with self._lock:
            cube = self._load_or_create_cube(date_str)
            aggregated = cube[start_index:end_index].sum(axis=0)

        logger.debug("Aggregated [%s] %02d:%02d -> %02d:%02d | slots=%d",
                      date_str, start_hour, start_minute, end_hour, end_minute, end_index - start_index)
        return aggregated

    def render_heatmap(self, heatmap: np.ndarray, render_config: Optional[RenderConfig] = None) -> np.ndarray:
        """Converts a 2D detection count array into a BGR color image."""
        cfg = render_config or self.render_cfg
        output_width = cfg.output_width or self.grid.frame_width
        output_height = cfg.output_height or self.grid.frame_height

        heatmap = heatmap.astype(np.float32)

        if cfg.normalization_mode == "log":
            heatmap = np.log1p(heatmap)

        normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        resized = cv2.resize(normalized, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
        resized = cv2.GaussianBlur(resized, cfg.blur_kernel_size, 0)
        colored = cv2.applyColorMap(resized, cfg.colormap)
        return colored

    def save_cube(self, date_str: str, executor: Optional[Any] = None) -> None:
        """Writes a single day's cube from RAM to MinIO."""
        with self._lock:
            if date_str not in self.loaded_cubes:
                return
            cube = self.loaded_cubes[date_str]
            object_key = self._get_cube_key(date_str)
            cube_snapshot = cube.copy()

        def _execute_upload():
            try:
                self.store.upload_array(object_key, cube_snapshot)
                logger.info("Uploaded cube snapshot to MinIO: %s/%s", self.store.bucket, object_key)
            except Exception as e:
                logger.error("Failed to upload cube to MinIO in background thread: %s", e)

        if executor is not None:
            executor.submit(_execute_upload)
        else:
            _execute_upload()

    def save_all(self, executor: Optional[Any] = None) -> None:
        """Saves all currently loaded cubes to MinIO."""
        with self._lock:
            date_keys = list(self.loaded_cubes.keys())
        for date_str in date_keys:
            self.save_cube(date_str, executor=executor)
        logger.info("Scheduled save operations for all active cubes (%d dates)", len(date_keys))
