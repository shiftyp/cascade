#!/usr/bin/env python3
"""Coordinator for Distributed Pattern Generation on Fly.io

Spawns worker machines, monitors progress, collects results.
"""

import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.training.patterns import save_pattern_file, load_pattern_file
from modules.training.patterns.visualization import generate_batch_report


def main():
    parser = argparse.ArgumentParser(description='Distributed Pattern Generation Coordinator')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker machines')
    parser.add_argument('--count', type=int, choices=[64, 128], default=128, help='Pattern count')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--region', type=str, default='iad', help='Fly.io region')
    parser.add_argument('--app', type=str, default='cascade-pattern-worker', help='Fly.io app name')
    parser.add_argument('--bucket', type=str, default='cascade-patterns', help='Tigris bucket')
    parser.add_argument('--output', type=str, default=None, help='Output file')

    args = parser.parse_args()

    if args.output is None:
        args.output = f'cascade_patterns_{args.count}.bin'

    print("=" * 70)
    print("CASCADE Distributed Pattern Generation - Coordinator")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    print(f"Pattern count: {args.count}")
    print(f"Base seed: {args.seed}")
    print(f"Region: {args.region}")
    print(f"Tigris bucket: {args.bucket}")
    print()

    # Step 1: Spawn workers
    print("Spawning workers...")
    spawn_workers(args)

    # Step 2: Monitor progress
    print("\nMonitoring progress...")
    monitor_workers(args)

    # Step 3: Collect results
    print("\nCollecting results...")
    results = collect_results(args)

    # Step 4: Select best
    print("\nSelecting best pattern set...")
    best = select_best(results)

    print(f"\nBest trial: {best['trial_id']} (seed={best['seed']})")
    print(f"  Score: {best['score']:.1f}")
    print(f"  Min separation: {best['min_separation_db']:.1f} dB")
    print(f"  Avg λ: {best['avg_lambda']:.3f}")

    # Step 5: Download and save best patterns
    print(f"\nDownloading best pattern set...")
    download_patterns(best['trial_id'], args.bucket, args.output)

    print(f"\n✓ Pattern generation complete!")
    print(f"  Output: {args.output}")

    # Cleanup
    print("\nCleaning up worker machines...")
    cleanup_workers(args)


def spawn_workers(args):
    """Spawn Fly.io worker machines"""
    for trial_id in range(args.workers):
        cmd = [
            'fly', 'machine', 'run',
            f'--app={args.app}',
            f'--region={args.region}',
            '--vm-size=performance-1x',
            f'--env=TRIAL_ID={trial_id}',
            f'--env=PATTERN_COUNT={args.count}',
            f'--env=SEED_BASE={args.seed}',
            f'--env=TIGRIS_BUCKET={args.bucket}',
            'worker'  # Dockerfile CMD
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ Spawned worker {trial_id}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to spawn worker {trial_id}: {e}")


def monitor_workers(args):
    """Monitor worker progress by checking Tigris"""
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url='https://fly.storage.tigris.dev',
        region_name='auto'
    )

    completed = set()
    expected = set(range(args.workers))

    while completed != expected:
        # Check for completed trials
        try:
            response = s3.list_objects_v2(
                Bucket=args.bucket,
                Prefix='trials/trial_',
                Delimiter='/'
            )

            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('_metadata.json'):
                    trial_id = int(key.split('_')[1])
                    if trial_id not in completed:
                        completed.add(trial_id)
                        print(f"  ✓ Trial {trial_id} completed ({len(completed)}/{args.workers})")

        except Exception as e:
            print(f"  ⚠ Error checking progress: {e}")

        if completed != expected:
            time.sleep(60)  # Check every minute

    print(f"\n✓ All {args.workers} workers completed!")


def collect_results(args) -> List[Dict]:
    """Collect all trial results from Tigris"""
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url='https://fly.storage.tigris.dev',
        region_name='auto'
    )

    results = []

    for trial_id in range(args.workers):
        key = f'trials/trial_{trial_id}_metadata.json'
        try:
            obj = s3.get_object(Bucket=args.bucket, Key=key)
            metadata = json.loads(obj['Body'].read())
            results.append(metadata)
            print(f"  ✓ Collected trial {trial_id}: score={metadata['score']:.1f}")
        except Exception as e:
            print(f"  ✗ Failed to collect trial {trial_id}: {e}")

    return results


def select_best(results: List[Dict]) -> Dict:
    """Select best trial based on score"""
    if not results:
        raise ValueError("No results to select from!")

    best = max(results, key=lambda r: r.get('score', -1000))
    return best


def download_patterns(trial_id: int, bucket: str, output_file: str):
    """Download best pattern file from Tigris"""
    import boto3

    s3 = boto3.client(
        's3',
        endpoint_url='https://fly.storage.tigris.dev',
        region_name='auto'
    )

    key = f'trials/trial_{trial_id}.bin'
    s3.download_file(bucket, key, output_file)
    print(f"  ✓ Downloaded {output_file}")


def cleanup_workers(args):
    """Stop and remove worker machines"""
    # List machines
    try:
        result = subprocess.run(
            ['fly', 'machine', 'list', f'--app={args.app}', '--json'],
            capture_output=True,
            text=True,
            check=True
        )

        machines = json.loads(result.stdout)

        for machine in machines:
            machine_id = machine['id']
            subprocess.run(
                ['fly', 'machine', 'destroy', machine_id, f'--app={args.app}', '--force'],
                check=True,
                capture_output=True
            )
            print(f"  ✓ Destroyed machine {machine_id}")

    except Exception as e:
        print(f"  ⚠ Cleanup failed: {e}")


if __name__ == '__main__':
    main()
