"""Tigris S3 storage client for CASCADE collector."""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class TigrisConfig:
    """Tigris S3 configuration."""
    endpoint_url: str = "https://fly.storage.tigris.dev"
    region_name: str = "auto"
    bucket_name: str = "cascade-data"
    access_key_id: str = ""
    secret_access_key: str = ""


class TigrisStorage:
    """Tigris S3 storage client for FLAC recordings."""

    def __init__(self, config: Optional[TigrisConfig] = None):
        """Initialize Tigris storage client.

        Args:
            config: Tigris configuration
        """
        self.config = config or TigrisConfig()
        self.client = None
        self.connected = False
        self.bytes_uploaded = 0
        self.files_uploaded = 0

    async def connect(self):
        """Connect to Tigris S3."""
        if self.connected:
            return

        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region_name,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key
            )

            # Test connection
            self.client.head_bucket(Bucket=self.config.bucket_name)
            self.connected = True
            logger.info(f"Connected to Tigris bucket: {self.config.bucket_name}")

        except ClientError as e:
            logger.error(f"Failed to connect to Tigris: {e}")
            raise

    async def upload_file(self, local_path: str, remote_key: str,
                         metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload file to Tigris.

        Args:
            local_path: Local file path
            remote_key: S3 object key
            metadata: Optional metadata

        Returns:
            True if successful
        """
        if not self.connected:
            await self.connect()

        try:
            path = Path(local_path)
            file_size = path.stat().st_size

            # Upload with metadata
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata

            self.client.upload_file(
                str(path),
                self.config.bucket_name,
                remote_key,
                ExtraArgs=extra_args
            )

            self.bytes_uploaded += file_size
            self.files_uploaded += 1

            logger.info(f"Uploaded {path.name} to s3://{self.config.bucket_name}/{remote_key}")
            return True

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    async def download_file(self, remote_key: str, local_path: str) -> bool:
        """Download file from Tigris.

        Args:
            remote_key: S3 object key
            local_path: Local file path

        Returns:
            True if successful
        """
        if not self.connected:
            await self.connect()

        try:
            self.client.download_file(
                self.config.bucket_name,
                remote_key,
                local_path
            )

            logger.info(f"Downloaded s3://{self.config.bucket_name}/{remote_key}")
            return True

        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False

    async def list_files(self, prefix: str = "") -> list:
        """List files in bucket.

        Args:
            prefix: Key prefix filter

        Returns:
            List of object keys
        """
        if not self.connected:
            await self.connect()

        try:
            response = self.client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )

            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []

        except ClientError as e:
            logger.error(f"List failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "connected": self.connected,
            "files_uploaded": self.files_uploaded,
            "bytes_uploaded": self.bytes_uploaded,
            "mb_uploaded": self.bytes_uploaded / 1024 / 1024
        }