#!/usr/bin/env python3
"""Fly.io Pattern Generation Worker

Generates a single pattern trial and uploads results to Tigris.
"""

import os
import sys
import json
from pathlib import Path

# Add parent modules to path
sys.path.insert(0, '/app')

from modules.training.patterns import generate_pattern_set, save_pattern_file
from modules.training.patterns.validator import validate_orthogonality
import numpy as np


def main():
    # Get trial configuration from environment
    trial_id = int(os.environ.get('TRIAL_ID', '0'))
    count = int(os.environ.get('PATTERN_COUNT', '128'))
    seed_base = int(os.environ.get('SEED_BASE', '42'))
    bucket = os.environ.get('TIGRIS_BUCKET', 'cascade-patterns')

    seed = seed_base + trial_id

    print(f"=" * 60)
    print(f"CASCADE Pattern Worker - Trial {trial_id}")
    print(f"=" * 60)
    print(f"Pattern count: {count}")
    print(f"Seed: {seed}")
    print(f"Tigris bucket: {bucket}")
    print()

    try:
        import time
        start_time = time.time()

        # Generate patterns
        print("Generating patterns...")
        patterns = generate_pattern_set(count=count, seed=seed)

        # Validate
        print("\nValidating patterns...")
        passes, stats = validate_orthogonality(patterns, target_db=-37.5)

        elapsed = time.time() - start_time

        if not passes:
            print(f"✗ Validation FAILED - {len(stats['failed_pairs'])} pairs failed")
            sys.exit(1)

        # Compute metrics
        lambdas = [p.iq_complexity_lambda for p in patterns]
        results = {
            'trial_id': trial_id,
            'seed': seed,
            'count': count,
            'passes': passes,
            'min_separation_db': stats['min_correlation_db'],
            'max_separation_db': stats['max_correlation_db'],
            'mean_separation_db': stats['mean_correlation_db'],
            'avg_lambda': float(np.mean(lambdas)),
            'median_lambda': float(np.median(lambdas)),
            'min_lambda': float(np.min(lambdas)),
            'max_lambda': float(np.max(lambdas)),
            'elapsed_hours': elapsed / 3600,
            'score': stats['min_correlation_db'] - 0.1 * np.mean(lambdas)
        }

        print(f"\n✓ Generation complete in {results['elapsed_hours']:.2f} hours")
        print(f"  Min separation: {results['min_separation_db']:.1f} dB")
        print(f"  Avg λ: {results['avg_lambda']:.3f}")
        print(f"  Score: {results['score']:.1f}")

        # Save pattern file locally
        pattern_file = f"/tmp/trial_{trial_id}.bin"
        save_pattern_file(patterns, pattern_file)
        print(f"\n✓ Saved patterns to {pattern_file}")

        # Upload to Tigris
        print("\nUploading to Tigris...")
        upload_to_tigris(pattern_file, f"trials/trial_{trial_id}.bin", bucket)

        # Upload metadata
        metadata_file = f"/tmp/trial_{trial_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(results, f, indent=2)

        upload_to_tigris(metadata_file, f"trials/trial_{trial_id}_metadata.json", bucket)

        print(f"\n✓ Trial {trial_id} complete!")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Trial {trial_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def upload_to_tigris(local_file: str, s3_key: str, bucket: str):
    """Upload file to Tigris S3 storage

    Args:
        local_file: Local file path
        s3_key: S3 object key
        bucket: S3 bucket name
    """
    try:
        import boto3

        # Tigris credentials from environment
        s3 = boto3.client(
            's3',
            endpoint_url='https://fly.storage.tigris.dev',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name='auto'
        )

        s3.upload_file(local_file, bucket, s3_key)
        print(f"  ✓ Uploaded {s3_key}")

    except ImportError:
        print("  ⚠ boto3 not available, skipping upload")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        raise


if __name__ == '__main__':
    main()
