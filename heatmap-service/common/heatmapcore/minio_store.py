"""
minio_store.py
--------------------------------------------------------------------
boto3-based MinIO client, matching the plate/face/fire modules' own
`minio_store.py` implementation (same client construction, same lazy
per-process client so a spawned engine subprocess gets its own),
adapted for this module's actual payload: numpy heatmap cubes stored
as `.npy` objects, one per camera per day, rather than JPEG crops.

Object key layout (unchanged from the pre-existing standalone build's
`minio_client.py`):
    <camera_id>/<date_str>.npy
--------------------------------------------------------------------
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import boto3
from botocore.config import Config

from .logging_setup import setup_logger

logger = setup_logger("heatmapcore.minio")

MINIO_ENDPOINT = os.getenv("AWS_S3_ENDPOINT_URL", os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID_FULL", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_FULL", os.getenv("MINIO_SECRET_KEY", "minioadminpassword"))
MINIO_REGION = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
MINIO_USE_SSL = os.getenv("AWS_S3_USE_SSL", os.getenv("MINIO_SECURE", "False")).lower() == "true"
MINIO_VERIFY = os.getenv("AWS_S3_VERIFY", "False").lower() == "true"

# Its own dedicated bucket — cubes are not image crops and don't belong
# in the plate/face modules' shared eyepass-private-bucket.
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "heatmap-data")

_client = None


def get_minio_client():
    """Lazily creates a single boto3 S3 client per process, pointed at
    MinIO. Safe to call from inside a spawned mp.Process — each process
    gets its own lazily-created client (module-level globals are not
    inherited meaningfully across `spawn`)."""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name=MINIO_REGION,
            use_ssl=MINIO_USE_SSL,
            verify=MINIO_VERIFY,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
    return _client


def ensure_bucket(bucket: str = MINIO_BUCKET):
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=bucket)
            logger.info("Created MinIO bucket: %s", bucket)
        except Exception as e:
            logger.warning("Could not create/verify bucket %s: %s", bucket, e)


class MinioArrayStore:
    """Stores/retrieves numpy arrays as .npy objects in a single MinIO
    bucket. Same shape as the pre-existing standalone build's
    `minio_client.MinioArrayStore`, reimplemented on boto3 (matching
    the other modules' client) instead of the `minio` SDK."""

    def __init__(self, bucket: str = MINIO_BUCKET):
        self.bucket = bucket
        self.client = get_minio_client()
        ensure_bucket(self.bucket)

    def exists(self, object_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:
            return False

    def download_array(self, object_key: str) -> Optional[np.ndarray]:
        """Returns the array at object_key, or None if it doesn't exist."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
            buffer = io.BytesIO(response["Body"].read())
            return np.load(buffer)
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            # botocore raises a generic ClientError for a 404 through
            # some endpoints rather than the typed NoSuchKey above.
            if "NoSuchKey" in str(e) or "404" in str(e):
                return None
            raise

    def upload_array(self, object_key: str, array: np.ndarray) -> None:
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        self.client.put_object(
            Bucket=self.bucket, Key=object_key, Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

    def list_objects(self, prefix: str = ""):
        paginator = self.client.get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            out.extend(page.get("Contents", []))
        return out
