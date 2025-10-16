#!/bin/bash
#
# Safe GPU dataset generation script with memory management
#
# Fixes for 80GB OOM error:
# 1. Reduced batch size from 4096 → 2048
# 2. PyTorch expandable segments (reduces fragmentation)
# 3. GPU cache clearing between batches
#

set -e  # Exit on error

echo "================================================================================"
echo "CASCADE DATASET GENERATION (GPU - Memory Optimized)"
echo "================================================================================"
echo ""
echo "Optimizations:"
echo "  - Batch size: 1024 (balanced for memory safety and performance)"
echo "  - Time-varying updates: 10ms (100 updates/sec for realistic fading)"
echo "  - Pre-calculated physics: eliminates CPU bottleneck"
echo "  - PyTorch expandable segments: enabled"
echo "  - GPU cache clearing: enabled between batches"
echo "  - Intermediate tensor cleanup: enabled"
echo ""

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
export CASCADE_TRAIN_SAMPLES=${CASCADE_TRAIN_SAMPLES:-5000000}
export CASCADE_VAL_SAMPLES=${CASCADE_VAL_SAMPLES:-500000}

echo "Dataset configuration:"
echo "  Train samples: $CASCADE_TRAIN_SAMPLES"
echo "  Val samples: $CASCADE_VAL_SAMPLES"
echo ""

# Run with reduced batch size
echo "Starting dataset generation..."
python ./phase2_dataset_creation.py

echo ""
echo "✓ Dataset generation complete!"
