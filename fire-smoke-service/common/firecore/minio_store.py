"""
minio_store.py
--------------------------------------------------------------------
Generalized version of the reference minio_uploader.py (unchanged
behavior, moved into the shared library so BOTH the detector and the
OCR service can import it, instead of only the OCR side).

Per the module's storage split:
    pipeline data (plate/vehicle crops, best-frame images delivered to
    the backend)
        -> MinIO (this module)
    debugging output (annotated overlay video)
        -> local bind-mounted volume, plain cv2.VideoWriter, never
           touches MinIO. See detector's DEBUG_VIDEO_* config.
--------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from .logging_setup import setup_logger
import boto3
from botocore.config import Config

logger = setup_logger("firecore.minio")

MINIO_ENDPOINT = os.getenv("AWS_S3_ENDPOINT_URL", os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000")

MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID_FULL", os.getenv("MINIO_ACCESS_KEY", "admin"))
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_FULL", os.getenv("MINIO_SECRET_KEY", "admin123456"))

MINIO_REGION = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
MINIO_USE_SSL = os.getenv("AWS_S3_USE_SSL", os.getenv("MINIO_SECURE", "False")).lower() == "true"
MINIO_VERIFY = os.getenv("AWS_S3_VERIFY", "False").lower() == "true"

# Defaults match the reference deployment's existing bucket names
# (docker-compose.standalone.yml) so an in-place migration needs no
# MinIO-side changes.
PUBLIC_BUCKET = os.getenv("AWS_PUBLIC_BUCKET_NAME", "eyepass-public-bucket")
PRIVATE_BUCKET = os.getenv("AWS_PRIVATE_BUCKET_NAME", "eyepass-private-bucket")

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
            # NOTE: request_checksum_calculation / response_checksum_validation
            # were deliberately left out — only added to botocore around
            # 1.36 (AWS's newer default-integrity-protections feature).
            # If your base image ships an older botocore, passing them to
            # Config() raises `TypeError: Got unexpected keyword argument
            # 'request_checksum_calculation'`. Neither is needed for MinIO.
            # If your botocore IS new enough and you want them, add:
            #   request_checksum_calculation="when_required",
            #   response_checksum_validation="when_required",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
    return _client


def ensure_bucket(bucket: str):
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=bucket)
            logger.info("Created MinIO bucket: %s", bucket)
        except Exception as e:
            logger.warning("Could not create/verify bucket %s: %s", bucket, e)


def upload_bytes(data: bytes, key: str, bucket: str = PRIVATE_BUCKET,
                  content_type: str = "image/jpeg") -> str:
    client = get_minio_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return f"{MINIO_PUBLIC_URL}/{bucket}/{key}"


def upload_file(local_path: str, key: str, bucket: str = PRIVATE_BUCKET,
                 content_type: str = "image/jpeg", delete_local: bool = False) -> str:
    client = get_minio_client()
    with open(local_path, "rb") as f:
        client.put_object(Bucket=bucket, Key=key, Body=f.read(), ContentType=content_type)
    url = f"{MINIO_PUBLIC_URL}/{bucket}/{key}"
    if delete_local:
        try:
            os.remove(local_path)
        except OSError:
            pass
    return url


def download_file_from_minio(key: str, local_filename: Optional[str] = None,
                              bucket: str = PRIVATE_BUCKET) -> str:
    client = get_minio_client()
    dest = os.path.join(tempfile.gettempdir(), local_filename or Path(key).name)
    response = client.get_object(Bucket=bucket, Key=key)
    with open(dest, "wb") as f:
        f.write(response["Body"].read())
    logger.info("[MinIO] Downloaded %s -> %s", key, dest)
    return dest


def download_folder_from_minio(prefix: str, local_base_dir: Optional[str] = None,
                                bucket: str = PRIVATE_BUCKET) -> str:
    client = get_minio_client()
    folder_name = prefix.rstrip("/").split("/")[-1]
    local_dir = os.path.join(local_base_dir or tempfile.gettempdir(), folder_name)

    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):]
            if not relative:
                continue
            local_path = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            response = client.get_object(Bucket=bucket, Key=key)
            with open(local_path, "wb") as f:
                f.write(response["Body"].read())
            downloaded += 1

    if downloaded == 0:
        raise RuntimeError(f"[MinIO] No objects found under prefix: {prefix}")

    logger.info("[MinIO] Downloaded %d files from '%s' -> %s", downloaded, prefix, local_dir)
    return local_dir


def open_sqlite_from_minio(key: str, bucket: str = PRIVATE_BUCKET) -> sqlite3.Connection:
    local_path = download_file_from_minio(key=key, bucket=bucket)
    conn = sqlite3.connect(local_path)
    conn.row_factory = sqlite3.Row
    return conn
