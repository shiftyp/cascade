#!/usr/bin/env python3
"""
Parallel CASCADE Dataset Generation

Launches multiple independent processes to generate dataset chunks in parallel.
Each process uses the full GPU and writes to its own chunk files.

Usage:
    python generate_dataset_parallel.py --num-streams 34250000 --num-workers 8

This will launch 8 processes, each generating ~4.28M streams in parallel.
Expected speedup: 8× (linear with number of workers)
"""

import subprocess
import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Parallel CASCADE dataset generation')
    parser.add_argument('--num-streams', type=int, required=True,
                       help='Total number of streams to generate')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel worker processes (default: 8)')
    parser.add_argument('--cache-dir', type=str, default='/tmp/cascade_dataset_cache',
                       help='Cache directory for output')
    parser.add_argument('--validation', action='store_true',
                       help='Generate validation set instead of training')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='GPU batch size per worker (default: 512)')

    args = parser.parse_args()

    print("=" * 80)
    print("CASCADE PARALLEL DATASET GENERATION")
    print("=" * 80)
    print(f"Total streams: {args.num_streams:,}")
    print(f"Workers: {args.num_workers}")
    print(f"Streams per worker: {args.num_streams // args.num_workers:,}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size per worker: {args.batch_size}")
    print("=" * 80)

    # Calculate chunk assignment for each worker
    streams_per_worker = args.num_streams // args.num_workers
    remainder = args.num_streams % args.num_workers

    processes = []
    start_time = time.time()

    print(f"\nLaunching {args.num_workers} worker processes...")

    for worker_id in range(args.num_workers):
        # Calculate this worker's stream range
        start_stream = worker_id * streams_per_worker
        end_stream = start_stream + streams_per_worker

        # Give remainder to last worker
        if worker_id == args.num_workers - 1:
            end_stream += remainder

        num_worker_streams = end_stream - start_stream

        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'generate_dataset_worker.py'),
            '--worker-id', str(worker_id),
            '--start-stream', str(start_stream),
            '--num-streams', str(num_worker_streams),
            '--cache-dir', args.cache_dir,
            '--batch-size', str(args.batch_size),
        ]

        if args.validation:
            cmd.append('--validation')

        print(f"  Worker {worker_id}: GPU {worker_id}, streams {start_stream:,} to {end_stream:,} ({num_worker_streams:,} streams)")

        # Set environment to use specific GPU (one worker per GPU!)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        # Launch process with dedicated GPU
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        processes.append((worker_id, proc, num_worker_streams))

    print(f"\n✓ All {args.num_workers} workers launched!")
    print(f"  Monitoring progress (workers write to logs)...")
    print(f"  Check individual logs: {args.cache_dir}/worker_*.log")

    # Monitor processes
    completed_workers = set()
    while len(completed_workers) < args.num_workers:
        time.sleep(5)

        # Check status
        for worker_id, proc, num_streams in processes:
            if worker_id in completed_workers:
                continue

            if proc.poll() is not None:
                returncode = proc.returncode
                completed_workers.add(worker_id)

                if returncode == 0:
                    print(f"  ✓ Worker {worker_id} completed ({num_streams:,} streams)")
                else:
                    print(f"  ❌ Worker {worker_id} failed (exit code {returncode})")
                    print(f"     Check log: {args.cache_dir}/worker_{worker_id:02d}.log")

        # Show progress
        elapsed = time.time() - start_time
        if len(completed_workers) < args.num_workers:
            print(f"  [{len(completed_workers)}/{args.num_workers}] workers complete ({elapsed:.0f}s elapsed)")

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("PARALLEL GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total streams: {args.num_streams:,}")
    print(f"Effective rate: {args.num_streams/total_time:.1f} streams/sec")
    print(f"Speedup vs single: {args.num_workers:.1f}×")
    print()
    print("Next: Merge worker outputs and create unified dataset")
    print(f"  python merge_dataset_chunks.py --cache-dir {args.cache_dir}")

if __name__ == '__main__':
    main()
