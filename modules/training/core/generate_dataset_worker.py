#!/usr/bin/env python3
"""
CASCADE Dataset Generation Worker

Single worker process that generates a portion of the dataset.
Designed to run independently with other workers in parallel.
"""

import sys
import os
import argparse
from pathlib import Path

# Add CASCADE root to path
cascade_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(cascade_root))

import torch
import numpy as np
from streaming_cascade_dataset import StreamingCascadeDataset

def main():
    parser = argparse.ArgumentParser(description='CASCADE dataset worker')
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--start-stream', type=int, required=True)
    parser.add_argument('--num-streams', type=int, required=True)
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--validation', action='store_true')

    args = parser.parse_args()

    # Enable PyTorch memory optimization for multi-process
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Setup logging to file
    log_file = Path(args.cache_dir) / f'worker_{args.worker_id:02d}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Redirect stdout/stderr to log file
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout

    print(f"Worker {args.worker_id} starting...")
    print(f"  Streams: {args.start_stream:,} to {args.start_stream + args.num_streams:,}")
    print(f"  Total: {args.num_streams:,} streams")
    print(f"  Batch size: {args.batch_size}")

    # Create worker-specific cache directory
    worker_cache = Path(args.cache_dir) / f'worker_{args.worker_id:02d}'

    # Generate dataset for this worker's portion
    # Use worker_id as seed offset for reproducibility
    seed = 42 + args.worker_id if not args.validation else 1042 + args.worker_id

    try:
        dataset = StreamingCascadeDataset(
            num_streams=args.num_streams,
            stream_duration_sec=10.0,
            window_duration_sec=2048 / 48000,
            message_arrival_rate=0.8,
            sample_rate=48000,
            seed=seed,
            cache_dir=str(worker_cache),
            use_cache=True,
            regenerate_cache=True,  # Always regenerate for workers
            batch_size=args.batch_size,
            device='cuda',
            num_workers=4,  # Each worker uses 4 CPU threads
            load_into_memory=False,
            enable_tx_observations=False
        )

        print(f"\n✓ Worker {args.worker_id} completed!")
        print(f"  Generated: {args.num_streams:,} streams")
        print(f"  Cache: {worker_cache}")
        print(f"  Chunks: {len(list(worker_cache.glob('streams_chunk_*.npy')))}")

    except Exception as e:
        print(f"\n❌ Worker {args.worker_id} FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
