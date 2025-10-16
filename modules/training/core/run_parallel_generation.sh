#!/bin/bash
#
# Launch parallel CASCADE dataset generation
#
# Usage: ./run_parallel_generation.sh [num_streams] [num_workers]
# Example: ./run_parallel_generation.sh 34250000 8
#

NUM_STREAMS=${1:-34250000}  # Default: 34.25M
NUM_WORKERS=${2:-8}         # Default: 8 workers
CACHE_DIR=${3:-/tmp/cascade_parallel}

echo "================================================================================"
echo "CASCADE PARALLEL DATASET GENERATION"
echo "================================================================================"
echo "Total streams: $NUM_STREAMS"
echo "Workers: $NUM_WORKERS"
echo "Cache directory: $CACHE_DIR"
echo "================================================================================"
echo

# No MPS needed - each worker gets dedicated GPU via CUDA_VISIBLE_DEVICES

# Make scripts executable
chmod +x generate_dataset_parallel.py
chmod +x generate_dataset_worker.py

# Run parallel generation
python3 generate_dataset_parallel.py \
    --num-streams $NUM_STREAMS \
    --num-workers $NUM_WORKERS \
    --cache-dir $CACHE_DIR \
    --batch-size 4096  # B200 cluster: large batches per GPU!

echo
echo "================================================================================"
echo "GENERATION COMPLETE"
echo "================================================================================"

echo "Check worker logs:"
echo "  ls -lh $CACHE_DIR/worker_*.log"
echo
echo "Worker outputs:"
echo "  ls -lh $CACHE_DIR/worker_*/"
echo
echo "Next: Merge chunks and create unified dataset"
echo "================================================================================"
