"""Tigris S3 storage client for FLAC recordings.

Implements T036: Tigris S3 storage client.
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from io import BytesIO

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from ..config import config
from dataclasses import dataclass

@dataclass
class TigrisConfig:
    """Tigris S3 configuration."""
    endpoint_url: str = "https://fly.storage.tigris.dev"
    region_name: str = "auto"
    bucket_name: str = "cascade-data"
    access_key_id: str = ""
    secret_access_key: str = ""

@dataclass
class StorageMetrics:
    """Storage metrics tracking."""
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    files_uploaded: int = 0
    files_downloaded: int = 0
    upload_errors: int = 0
    download_errors: int = 0

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of a file upload operation."""

    success: bool
    object_key: str
    file_size: int
    checksum: str
    upload_duration: float
    error: Optional[str] = None


@dataclass
class StorageQuota:
    """Storage quota information."""

    used_bytes: int
    limit_bytes: int
    available_bytes: int
    usage_percent: float


class TigrisStorage:
    """Tigris S3 client for FLAC file storage."""

    def __init__(self, config: Optional[TigrisConfig] = None):
        """Initialize Tigris storage client.

        Args:
            config: Tigris configuration (creates default if None)
        """
        self.config = config or TigrisConfig()
        self.s3_client = None
        self.s3_resource = None
        self._init_client()

        # Bandwidth throttling
        self.bandwidth_limit_mbps = 100  # Default 100 Mbps
        self.upload_stats: Dict[str, Any] = {
            "total_uploads": 0,
            "total_bytes": 0,
            "failed_uploads": 0,
            "total_duration": 0.0,
        }

    def _init_client(self):
        """Initialize boto3 S3 client and resource."""
        try:
            # Boto3 config with retries
            boto_config = Config(
                retries={"max_attempts": self.config.max_attempts, "mode": self.config.retry_mode},
                max_pool_connections=50,
            )

            # Create S3 client
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region,
                config=boto_config,
            )

            # Create S3 resource
            self.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region,
                config=boto_config,
            )

            logger.info(f"Initialized Tigris S3 client: {self.config.endpoint_url}")

        except Exception as e:
            logger.error(f"Failed to initialize Tigris client: {e}")
            raise

    def upload_file(
        self,
        file_path: Path,
        object_key: str,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
    ) -> UploadResult:
        """Upload FLAC file to Tigris with multipart support.

        Args:
            file_path: Local file path
            object_key: S3 object key
            metadata: Optional metadata tags
            storage_class: Optional storage class override

        Returns:
            UploadResult with status and metrics
        """
        start_time = time.time()

        try:
            if not file_path.exists():
                return UploadResult(
                    success=False,
                    object_key=object_key,
                    file_size=0,
                    checksum="",
                    upload_duration=0.0,
                    error=f"File not found: {file_path}",
                )

            # Calculate checksums
            file_size = file_path.stat().st_size
            md5_hash, sha256_hash = self._calculate_checksums(file_path)

            # Prepare metadata
            upload_metadata = metadata or {}
            upload_metadata.update(
                {
                    "md5": md5_hash,
                    "sha256": sha256_hash,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "file_size": str(file_size),
                }
            )

            # Transfer config for large files
            transfer_config = TransferConfig(
                multipart_threshold=self.config.multipart_threshold,
                multipart_chunksize=self.config.multipart_chunksize,
                max_concurrency=self.config.max_concurrency,
                use_threads=self.config.use_threads,
            )

            # Upload file
            extra_args = {
                "Metadata": upload_metadata,
                "StorageClass": storage_class or self.config.storage_class,
            }

            self.s3_client.upload_file(
                str(file_path), self.config.bucket_name, object_key, ExtraArgs=extra_args, Config=transfer_config
            )

            # Verify checksum
            if not self._verify_upload(object_key, md5_hash):
                raise ValueError("Checksum verification failed")

            upload_duration = time.time() - start_time

            # Update stats
            self._update_stats(file_size, upload_duration, success=True)

            logger.info(
                f"Uploaded {file_path.name} to {object_key} "
                f"({file_size / 1024 / 1024:.2f} MB in {upload_duration:.1f}s)"
            )

            return UploadResult(
                success=True,
                object_key=object_key,
                file_size=file_size,
                checksum=sha256_hash,
                upload_duration=upload_duration,
            )

        except Exception as e:
            upload_duration = time.time() - start_time
            self._update_stats(0, upload_duration, success=False)

            logger.error(f"Upload failed for {file_path}: {e}")
            return UploadResult(
                success=False,
                object_key=object_key,
                file_size=0,
                checksum="",
                upload_duration=upload_duration,
                error=str(e),
            )

    def upload_bytes(
        self, data: bytes, object_key: str, metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """Upload bytes directly to Tigris.

        Args:
            data: Data bytes
            object_key: S3 object key
            metadata: Optional metadata tags

        Returns:
            UploadResult with status
        """
        start_time = time.time()

        try:
            # Calculate checksums
            md5_hash = hashlib.md5(data).hexdigest()
            sha256_hash = hashlib.sha256(data).hexdigest()

            # Prepare metadata
            upload_metadata = metadata or {}
            upload_metadata.update(
                {
                    "md5": md5_hash,
                    "sha256": sha256_hash,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "file_size": str(len(data)),
                }
            )

            # Upload
            self.s3_client.put_object(
                Bucket=self.config.bucket_name, Key=object_key, Body=data, Metadata=upload_metadata
            )

            upload_duration = time.time() - start_time
            self._update_stats(len(data), upload_duration, success=True)

            logger.info(f"Uploaded {len(data)} bytes to {object_key}")

            return UploadResult(
                success=True,
                object_key=object_key,
                file_size=len(data),
                checksum=sha256_hash,
                upload_duration=upload_duration,
            )

        except Exception as e:
            upload_duration = time.time() - start_time
            self._update_stats(0, upload_duration, success=False)

            logger.error(f"Bytes upload failed for {object_key}: {e}")
            return UploadResult(
                success=False, object_key=object_key, file_size=0, checksum="", upload_duration=upload_duration, error=str(e)
            )

    def download_file(self, object_key: str, download_path: Path) -> bool:
        """Download file from Tigris.

        Args:
            object_key: S3 object key
            download_path: Local destination path

        Returns:
            True if successful
        """
        try:
            download_path.parent.mkdir(parents=True, exist_ok=True)

            self.s3_client.download_file(self.config.bucket_name, object_key, str(download_path))

            logger.info(f"Downloaded {object_key} to {download_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed for {object_key}: {e}")
            return False

    def download_bytes(self, object_key: str) -> Optional[bytes]:
        """Download object as bytes.

        Args:
            object_key: S3 object key

        Returns:
            Object bytes or None if failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.config.bucket_name, Key=object_key)

            data = response["Body"].read()
            logger.debug(f"Downloaded {len(data)} bytes from {object_key}")
            return data

        except Exception as e:
            logger.error(f"Download failed for {object_key}: {e}")
            return None

    def delete_object(self, object_key: str) -> bool:
        """Delete object from Tigris.

        Args:
            object_key: S3 object key

        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.config.bucket_name, Key=object_key)

            logger.info(f"Deleted {object_key}")
            return True

        except Exception as e:
            logger.error(f"Delete failed for {object_key}: {e}")
            return False

    def delete_batch(self, object_keys: List[str]) -> Tuple[int, int]:
        """Delete multiple objects in batch.

        Args:
            object_keys: List of S3 object keys

        Returns:
            Tuple of (success_count, failure_count)
        """
        if not object_keys:
            return 0, 0

        try:
            # S3 batch delete supports up to 1000 keys
            success_count = 0
            failure_count = 0

            for i in range(0, len(object_keys), 1000):
                batch = object_keys[i : i + 1000]
                delete_objects = [{"Key": key} for key in batch]

                response = self.s3_client.delete_objects(
                    Bucket=self.config.bucket_name, Delete={"Objects": delete_objects}
                )

                success_count += len(response.get("Deleted", []))
                failure_count += len(response.get("Errors", []))

            logger.info(f"Batch deleted {success_count} objects ({failure_count} failures)")
            return success_count, failure_count

        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return 0, len(object_keys)

    def object_exists(self, object_key: str) -> bool:
        """Check if object exists.

        Args:
            object_key: S3 object key

        Returns:
            True if exists
        """
        try:
            self.s3_client.head_object(Bucket=self.config.bucket_name, Key=object_key)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_object_metadata(self, object_key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata.

        Args:
            object_key: S3 object key

        Returns:
            Metadata dict or None
        """
        try:
            response = self.s3_client.head_object(Bucket=self.config.bucket_name, Key=object_key)

            return {
                "content_length": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"].strip('"'),
                "metadata": response.get("Metadata", {}),
                "storage_class": response.get("StorageClass", "STANDARD"),
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for {object_key}: {e}")
            return None

    def generate_presigned_url(
        self, object_key: str, expiration: Optional[int] = None, method: str = "get_object"
    ) -> Optional[str]:
        """Generate presigned URL for object access.

        Args:
            object_key: S3 object key
            expiration: URL expiration in seconds
            method: S3 method ('get_object' or 'put_object')

        Returns:
            Presigned URL or None
        """
        try:
            expiration = expiration or self.config.presigned_url_expiration

            url = self.s3_client.generate_presigned_url(
                method, Params={"Bucket": self.config.bucket_name, "Key": object_key}, ExpiresIn=expiration
            )

            logger.debug(f"Generated presigned URL for {object_key} (expires in {expiration}s)")
            return url

        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {object_key}: {e}")
            return None

    def list_objects(
        self, prefix: str = "", max_keys: int = 1000, start_after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List objects with prefix.

        Args:
            prefix: Object key prefix
            max_keys: Maximum results
            start_after: Start listing after this key

        Returns:
            List of object info dicts
        """
        try:
            kwargs = {
                "Bucket": self.config.bucket_name,
                "Prefix": prefix,
                "MaxKeys": max_keys,
            }

            if start_after:
                kwargs["StartAfter"] = start_after

            response = self.s3_client.list_objects_v2(**kwargs)

            objects = []
            for obj in response.get("Contents", []):
                objects.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                        "etag": obj["ETag"].strip('"'),
                    }
                )

            logger.debug(f"Listed {len(objects)} objects with prefix '{prefix}'")
            return objects

        except Exception as e:
            logger.error(f"Failed to list objects with prefix '{prefix}': {e}")
            return []

    def get_storage_quota(self) -> StorageQuota:
        """Get current storage usage and quota.

        Returns:
            StorageQuota information
        """
        try:
            # Calculate total storage used
            total_bytes = 0
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.config.bucket_name):
                for obj in page.get("Contents", []):
                    total_bytes += obj["Size"]

            # Calculate limits
            limit_bytes = int(StorageMetrics.MAX_STORAGE_TB * 1024**4)
            available_bytes = limit_bytes - total_bytes
            usage_percent = (total_bytes / limit_bytes) * 100

            return StorageQuota(
                used_bytes=total_bytes, limit_bytes=limit_bytes, available_bytes=available_bytes, usage_percent=usage_percent
            )

        except Exception as e:
            logger.error(f"Failed to get storage quota: {e}")
            return StorageQuota(used_bytes=0, limit_bytes=0, available_bytes=0, usage_percent=0.0)

    def check_deduplication(self, file_hash: str) -> Optional[str]:
        """Check if file with same hash already exists.

        Args:
            file_hash: SHA256 hash of file

        Returns:
            Existing object key or None
        """
        try:
            # List all objects and check metadata for matching hash
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.config.bucket_name):
                for obj in page.get("Contents", []):
                    metadata = self.get_object_metadata(obj["Key"])
                    if metadata and metadata.get("metadata", {}).get("sha256") == file_hash:
                        logger.info(f"Found duplicate file: {obj['Key']}")
                        return obj["Key"]

            return None

        except Exception as e:
            logger.error(f"Deduplication check failed: {e}")
            return None

    def apply_lifecycle_policy(self) -> bool:
        """Apply lifecycle policy for tiered storage.

        Returns:
            True if successful
        """
        try:
            lifecycle_rules = self.config.get_lifecycle_rules()

            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.config.bucket_name, LifecycleConfiguration={"Rules": lifecycle_rules}
            )

            logger.info(f"Applied lifecycle policy with {len(lifecycle_rules)} rules")
            return True

        except Exception as e:
            logger.error(f"Failed to apply lifecycle policy: {e}")
            return False

    def _calculate_checksums(self, file_path: Path) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 checksums.

        Args:
            file_path: File path

        Returns:
            Tuple of (md5_hash, sha256_hash)
        """
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                md5.update(chunk)
                sha256.update(chunk)

        return md5.hexdigest(), sha256.hexdigest()

    def _verify_upload(self, object_key: str, expected_md5: str) -> bool:
        """Verify uploaded file checksum.

        Args:
            object_key: S3 object key
            expected_md5: Expected MD5 hash

        Returns:
            True if checksums match
        """
        try:
            metadata = self.get_object_metadata(object_key)
            if not metadata:
                return False

            stored_md5 = metadata.get("metadata", {}).get("md5")
            if stored_md5 == expected_md5:
                return True

            logger.warning(f"Checksum mismatch for {object_key}: {stored_md5} != {expected_md5}")
            return False

        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    def _update_stats(self, bytes_transferred: int, duration: float, success: bool):
        """Update upload statistics.

        Args:
            bytes_transferred: Bytes uploaded
            duration: Upload duration in seconds
            success: Whether upload succeeded
        """
        self.upload_stats["total_uploads"] += 1
        self.upload_stats["total_bytes"] += bytes_transferred
        self.upload_stats["total_duration"] += duration

        if not success:
            self.upload_stats["failed_uploads"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get upload statistics.

        Returns:
            Statistics dict
        """
        total_uploads = self.upload_stats["total_uploads"]
        total_bytes = self.upload_stats["total_bytes"]
        total_duration = self.upload_stats["total_duration"]

        avg_speed_mbps = 0.0
        if total_duration > 0:
            avg_speed_mbps = (total_bytes * 8 / 1_000_000) / total_duration

        return {
            "total_uploads": total_uploads,
            "successful_uploads": total_uploads - self.upload_stats["failed_uploads"],
            "failed_uploads": self.upload_stats["failed_uploads"],
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1024**3,
            "total_duration_seconds": total_duration,
            "average_speed_mbps": avg_speed_mbps,
            "success_rate": (total_uploads - self.upload_stats["failed_uploads"]) / max(total_uploads, 1) * 100,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Tigris connection.

        Returns:
            Health status dict
        """
        try:
            # Test bucket access
            start = time.time()
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
            latency = time.time() - start

            # Get storage quota
            quota = self.get_storage_quota()

            return {
                "healthy": True,
                "bucket": self.config.bucket_name,
                "latency_ms": latency * 1000,
                "storage_used_gb": quota.used_bytes / 1024**3,
                "storage_limit_tb": StorageMetrics.MAX_STORAGE_TB,
                "usage_percent": quota.usage_percent,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}


# Convenience functions
def upload_recording(
    file_path: Path, band: str, session_id: str, metadata: Optional[Dict[str, str]] = None
) -> UploadResult:
    """Upload recording file to Tigris.

    Args:
        file_path: Local file path
        band: HF band
        session_id: Session ID
        metadata: Optional metadata

    Returns:
        UploadResult
    """
    storage = TigrisStorage()
    config = TigrisConfig()

    date = datetime.utcnow().strftime("%Y-%m-%d")
    object_key = config.get_object_key("recording", band, date, session_id)

    return storage.upload_file(file_path, object_key, metadata)


def download_recording(object_key: str, download_path: Path) -> bool:
    """Download recording from Tigris.

    Args:
        object_key: S3 object key
        download_path: Local destination

    Returns:
        True if successful
    """
    storage = TigrisStorage()
    return storage.download_file(object_key, download_path)
# Alias for compatibility
TigrisStorageClient = TigrisStorage
