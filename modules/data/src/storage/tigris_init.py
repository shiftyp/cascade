"""Initialize Tigris S3 buckets on startup.

This module ensures the required S3 buckets exist in Tigris.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def initialize_tigris_buckets():
    """Create Tigris S3 buckets if they don't exist.

    Returns:
        bool: True if buckets are ready, False otherwise
    """
    # Get Tigris credentials from environment
    access_key = os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('TIGRIS_ACCESS_KEY')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY') or os.getenv('TIGRIS_SECRET_KEY')
    endpoint = os.getenv('AWS_ENDPOINT_URL_S3') or os.getenv('AWS_ENDPOINT_URL', 'https://fly.storage.tigris.dev')

    if not access_key or not secret_key:
        logger.warning("Tigris credentials not found, skipping bucket initialization")
        return False

    try:
        # Create S3 client for Tigris
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto',
        )

        # Define buckets to create
        buckets = [
            {
                'name': os.getenv('TIGRIS_BUCKET', 'cascade-iq-data'),
                'description': 'Main IQ data archive'
            },
            {
                'name': os.getenv('TIGRIS_BUCKET_QA', 'cascade-qa-samples'),
                'description': 'QA sample storage'
            },
        ]

        # Create each bucket if it doesn't exist
        for bucket_info in buckets:
            bucket_name = bucket_info['name']

            try:
                # Check if bucket exists
                s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' already exists")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    # Bucket doesn't exist, create it
                    try:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': 'auto'
                            }
                        )
                        logger.info(f"Created bucket '{bucket_name}': {bucket_info['description']}")

                        # Set bucket policy for private access
                        bucket_policy = {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Sid": "PrivateAccess",
                                    "Effect": "Deny",
                                    "Principal": "*",
                                    "Action": "s3:GetObject",
                                    "Resource": f"arn:aws:s3:::{bucket_name}/*",
                                    "Condition": {
                                        "StringNotEquals": {
                                            "aws:SourceAccount": os.getenv('FLY_APP_NAME', 'cascade-collector')
                                        }
                                    }
                                }
                            ]
                        }

                        # Note: Bucket policies might not be fully supported by Tigris yet
                        # This is here for future compatibility
                        try:
                            s3_client.put_bucket_policy(
                                Bucket=bucket_name,
                                Policy=str(bucket_policy)
                            )
                        except Exception as policy_error:
                            logger.debug(f"Could not set bucket policy (expected with Tigris): {policy_error}")

                    except ClientError as create_error:
                        logger.error(f"Failed to create bucket '{bucket_name}': {create_error}")
                        return False
                else:
                    logger.error(f"Error checking bucket '{bucket_name}': {e}")
                    return False

        logger.info("All Tigris buckets are ready")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Tigris buckets: {e}")
        return False


def cleanup_old_recordings(s3_client, bucket_name, days_to_keep=7):
    """Clean up old recordings from Tigris to manage storage costs.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: Bucket to clean
        days_to_keep: Number of days to keep recordings
    """
    from datetime import datetime, timedelta

    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        # List objects in bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='recordings/')

        objects_to_delete = []

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                # Check object age
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    objects_to_delete.append({'Key': obj['Key']})

                    # Delete in batches of 1000 (S3 limit)
                    if len(objects_to_delete) >= 1000:
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
                        logger.info(f"Deleted {len(objects_to_delete)} old recordings from {bucket_name}")
                        objects_to_delete = []

        # Delete remaining objects
        if objects_to_delete:
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            logger.info(f"Deleted {len(objects_to_delete)} old recordings from {bucket_name}")

    except Exception as e:
        logger.error(f"Failed to cleanup old recordings: {e}")


if __name__ == "__main__":
    # Initialize buckets when module is run directly
    logging.basicConfig(level=logging.INFO)
    success = initialize_tigris_buckets()
    if success:
        print("✅ Tigris buckets initialized successfully")
    else:
        print("❌ Failed to initialize Tigris buckets")
        exit(1)