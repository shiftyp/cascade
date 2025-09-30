"""
Tigris S3 configuration for FLAC storage
"""
import os
from typing import Optional, Dict, Any

class TigrisConfig:
    """Tigris S3 configuration for storing compressed IQ data"""

    def __init__(self):
        # Tigris credentials
        self.access_key_id = os.getenv('TIGRIS_ACCESS_KEY_ID')
        self.secret_access_key = os.getenv('TIGRIS_SECRET_ACCESS_KEY')
        self.endpoint_url = os.getenv('TIGRIS_ENDPOINT_URL', 'https://fly.storage.tigris.dev')
        self.region = os.getenv('TIGRIS_REGION', 'auto')

        # Bucket configuration
        self.bucket_name = os.getenv('TIGRIS_BUCKET', 'cascade-iq-data')
        self.recordings_prefix = 'recordings/'
        self.archives_prefix = 'archives/'
        self.qa_samples_prefix = 'qa-samples/'

        # Storage classes (Tigris uses intelligent tiering)
        self.storage_class = 'STANDARD'  # Tigris handles tiering automatically

        # Upload configuration
        self.multipart_threshold = 64 * 1024 * 1024  # 64MB
        self.multipart_chunksize = 16 * 1024 * 1024  # 16MB
        self.max_concurrency = 10
        self.use_threads = True

        # Retry configuration
        self.max_attempts = 3
        self.retry_mode = 'adaptive'

        # Lifecycle configuration (days)
        self.hot_tier_days = 1  # Recent recordings for immediate processing
        self.warm_tier_days = 30  # Last month for reprocessing
        self.cold_tier_days = 365  # Archive after a year

        # URL expiration (seconds)
        self.presigned_url_expiration = 3600  # 1 hour for downloads
        self.upload_url_expiration = 1800  # 30 minutes for uploads

    def get_boto3_config(self) -> Dict[str, Any]:
        """Get configuration dict for boto3 S3 client"""
        return {
            'aws_access_key_id': self.access_key_id,
            'aws_secret_access_key': self.secret_access_key,
            'endpoint_url': self.endpoint_url,
            'region_name': self.region,
        }

    def get_transfer_config(self) -> Dict[str, Any]:
        """Get configuration for S3 transfer manager"""
        from boto3.s3.transfer import TransferConfig
        return TransferConfig(
            multipart_threshold=self.multipart_threshold,
            multipart_chunksize=self.multipart_chunksize,
            max_concurrency=self.max_concurrency,
            use_threads=self.use_threads,
        )

    def get_object_key(self,
                      recording_type: str,
                      band: str,
                      date: str,
                      session_id: str,
                      extension: str = 'flac') -> str:
        """
        Generate S3 object key for a recording

        Args:
            recording_type: 'recording', 'archive', or 'qa-sample'
            band: HF band (e.g., '20m', '40m')
            date: Date string (YYYY-MM-DD)
            session_id: Unique session identifier
            extension: File extension

        Returns:
            S3 object key path
        """
        prefix_map = {
            'recording': self.recordings_prefix,
            'archive': self.archives_prefix,
            'qa-sample': self.qa_samples_prefix,
        }

        prefix = prefix_map.get(recording_type, self.recordings_prefix)
        return f"{prefix}{band}/{date}/{session_id}.{extension}"

    def get_lifecycle_rules(self) -> list:
        """
        Get S3 lifecycle rules for automatic tiering and expiration

        Returns:
            List of lifecycle rule configurations
        """
        return [
            {
                'ID': 'TransitionToWarmStorage',
                'Status': 'Enabled',
                'Transitions': [
                    {
                        'Days': self.hot_tier_days,
                        'StorageClass': 'STANDARD_IA'  # Tigris intelligent tiering
                    }
                ],
                'Filter': {'Prefix': self.recordings_prefix}
            },
            {
                'ID': 'TransitionToColdStorage',
                'Status': 'Enabled',
                'Transitions': [
                    {
                        'Days': self.warm_tier_days,
                        'StorageClass': 'GLACIER'  # Tigris cold storage
                    }
                ],
                'Filter': {'Prefix': self.archives_prefix}
            },
            {
                'ID': 'DeleteOldQASamples',
                'Status': 'Enabled',
                'Expiration': {
                    'Days': 90  # Delete QA samples after 90 days
                },
                'Filter': {'Prefix': self.qa_samples_prefix}
            }
        ]

    def get_bucket_policy(self) -> dict:
        """
        Get S3 bucket policy for access control

        Returns:
            Bucket policy configuration
        """
        return {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Sid': 'AllowWorkerAccess',
                    'Effect': 'Allow',
                    'Principal': {'AWS': '*'},  # Restricted by IAM in production
                    'Action': [
                        's3:GetObject',
                        's3:PutObject',
                        's3:DeleteObject',
                    ],
                    'Resource': f'arn:aws:s3:::{self.bucket_name}/*',
                    'Condition': {
                        'StringLike': {
                            'aws:userid': 'cascade-worker-*'
                        }
                    }
                },
                {
                    'Sid': 'AllowDashboardReadOnly',
                    'Effect': 'Allow',
                    'Principal': {'AWS': '*'},  # Restricted by IAM
                    'Action': [
                        's3:GetObject',
                        's3:ListBucket',
                    ],
                    'Resource': [
                        f'arn:aws:s3:::{self.bucket_name}',
                        f'arn:aws:s3:::{self.bucket_name}/{self.qa_samples_prefix}*'
                    ]
                }
            ]
        }

class StorageMetrics:
    """Storage metrics and limits"""

    # Target storage sizes (TB)
    MIN_STORAGE_TB = 35
    TARGET_STORAGE_TB = 50
    MAX_STORAGE_TB = 75

    # Compression ratios
    FLAC_COMPRESSION_RATIO = 0.5  # 45-55% size reduction

    # Data rates
    HOURLY_RATE_GB = 0.173  # ~173 MB/hour per stream
    DAILY_RATE_TB = 0.1  # ~100 GB/day with 6 streams

    @staticmethod
    def estimate_storage_needed(hours: int, streams: int = 6) -> float:
        """
        Estimate storage needed in TB

        Args:
            hours: Total collection hours
            streams: Number of concurrent streams

        Returns:
            Estimated storage in TB
        """
        gb_per_hour = StorageMetrics.HOURLY_RATE_GB
        total_gb = hours * gb_per_hour * StorageMetrics.FLAC_COMPRESSION_RATIO
        return total_gb / 1024  # Convert GB to TB

    @staticmethod
    def estimate_collection_time(target_tb: float, streams: int = 6) -> int:
        """
        Estimate collection time in days

        Args:
            target_tb: Target storage in TB
            streams: Number of concurrent streams

        Returns:
            Estimated days to collect
        """
        daily_rate = StorageMetrics.DAILY_RATE_TB * streams / 6  # Scale by stream count
        return int(target_tb / daily_rate)