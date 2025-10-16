#!/bin/bash
#
# CASCADE Full Model Training - Cache-Based Streaming Approach
#
# This script runs the complete CASCADE training pipeline:
# - Stage 1: IQ Encoder bootstrap (autoencoder)
# - Stage 2-3: Complete CASCADE model (5 experts + integration decoder)
# - Stage 4: Joint RX/TX training (embedding autoencoder)
#
# Usage:
#   ./run_full_training.sh [START_STAGE]
#
# Arguments:
#   START_STAGE  - Optional: Stage to start from (1-4). Default: 1
#                  Requires checkpoints for previous stages.
#
# Examples:
#   ./run_full_training.sh        # Start from Stage 1 (default)
#   ./run_full_training.sh 2      # Start from Stage 2 (requires Stage 1 checkpoint)
#   ./run_full_training.sh 4      # Start from Stage 4 (requires Stage 1 and 2-3 checkpoints)
#
# CACHE-BASED STREAMING APPROACH:
# ================================
# Instead of generating a massive static dataset (1.7 TB for 100M samples), we use:
#
# 1. Small cache (100K samples = ~17 GB)
#    - Generates in ~15 minutes
#    - Fits easily on NFS/local storage
#    - Fast enough to regenerate periodically
#
# 2. Fresh data each run (or manually via CASCADE_REGENERATE_CACHE=true)
#    - Prevents overfitting to static dataset
#    - Different physics/noise conditions each time
#    - Better model generalization
#
# 3. Numpy memmap format (thread-safe for 32+ DataLoader workers)
#    - Zero decompression overhead
#    - Parallel random access
#    - Optimal for PyTorch DataLoader
#
# SCALING TO LARGER DATASETS:
# ============================
# For production training with more data, you have options:
#
# Option A: Increase cache size (recommended up to 1M samples)
#   export CASCADE_TRAIN_SAMPLES=1000000  # ~170 GB cache, 2.5 hour generation
#
# Option B: Regenerate cache between epochs (requires automation)
#   Train for N epochs → regenerate cache → train N more epochs
#
# Option C: Pure streaming (no cache, experimental)
#   Generate batches on-the-fly during training (requires async generation pipeline)
#
# WHY NOT NPZ COMPRESSION?
# =========================
# - Only 1.5-2× compression (17 GB → 8-11 GB) - not worth the complexity
# - Cannot use memmap (must load entire file into RAM)
# - Incompatible with parallel DataLoader workers (decompression bottleneck)
# - CPU overhead on every epoch
#
# For archival/sharing only, use: tar -czf dataset.tar.gz *.npy
#

set -e  # Exit on error

# Parse command-line arguments
START_STAGE=${1:-1}  # Default to stage 1 if not specified

# Validate START_STAGE
if [[ ! "$START_STAGE" =~ ^[1-4]$ ]]; then
    echo "ERROR: Invalid START_STAGE: $START_STAGE"
    echo "Must be 1, 2, 3, or 4"
    echo ""
    echo "Usage: ./run_full_training.sh [START_STAGE]"
    echo "  START_STAGE=1 (default) - Start from IQ Encoder training"
    echo "  START_STAGE=2 - Start from CASCADE model training (requires Stage 1 checkpoint)"
    echo "  START_STAGE=4 - Start from Joint RX/TX training (requires Stage 1 and 2-3 checkpoints)"
    exit 1
fi

echo "================================================================================"
echo "CASCADE FULL MODEL TRAINING (Joint RX/TX, 100M Samples)"
echo "================================================================================"
if [ "$START_STAGE" -gt 1 ]; then
    echo "📍 Starting from Stage $START_STAGE (skipping earlier stages)"
    echo ""
fi
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Dataset sizes - CACHE-BASED STREAMING APPROACH
# Generate small cache (100K-200K samples) that regenerates periodically
# This provides:
#   - Zero long-term storage cost (cache is temporary, ~17 GB vs 1.7 TB for full dataset)
#   - Fresh data every run (prevents overfitting to static dataset)
#   - Fast startup (cache generates in ~15 minutes at 110 streams/sec)
#   - Different physics/noise each epoch (better generalization)
export CASCADE_TRAIN_SAMPLES=${CASCADE_TRAIN_SAMPLES:-3425}   # 1M training samples (~170 GB cache)
export CASCADE_VAL_SAMPLES=${CASCADE_VAL_SAMPLES:-514}         # 250K validation samples (~17 GB cache)
export CASCADE_DATASET_NUM_WORKERS=51
export CASCADE_DATASET_BATCH_SIZE=64  # Larger batches for better GPU utilization (was 128)

# Cache regeneration strategy
export CASCADE_REGENERATE_CACHE=${CASCADE_REGENERATE_CACHE:-true}  # Set to 'true' to force fresh data
export CASCADE_CACHE_REGEN_INTERVAL=${CASCADE_CACHE_REGEN_INTERVAL:-5}  # Regenerate cache every N epochs (0 = disabled)

# Training configuration
export CASCADE_BATCH_SIZE=8192            # Batch size for model training (not dataset generation)
export CASCADE_EPOCHS=50                # Maximum epochs per stage

# Dataset type (STREAMING is 11× faster!)
export CASCADE_USE_STREAMING=true       # Use streaming dataset (10s streams, 2-5 messages)
                                        # Set to 'false' for old snippet-based dataset

# Joint RX/TX Training (works with streaming!)
export CASCADE_TX_ENCODER_TRAINING=false # Enable TX embedding encoder training
                                        # Streaming dataset will auto-generate TX observations

# Early stopping configuration
export CASCADE_ENABLE_EARLY_STOP=true
export CASCADE_EARLY_STOP_PATIENCE=10   # Stop if no improvement for 10 epochs
export CASCADE_EARLY_STOP_DELTA=0.0001  # Minimum improvement threshold
export CASCADE_EARLY_STOP_WARMUP=10     # Don't trigger before epoch 10

# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CASCADE_VIS_INTERVAL=20480



# ============================================================================
# Display Configuration
# ============================================================================

echo "Configuration:"
echo "  Training samples: $CASCADE_TRAIN_SAMPLES"
echo "  Validation samples: $CASCADE_VAL_SAMPLES"
echo "  Batch size: $CASCADE_BATCH_SIZE"
echo "  Max epochs per stage: $CASCADE_EPOCHS"
if [ "$CASCADE_CACHE_REGEN_INTERVAL" -gt 0 ]; then
    echo "  Cache regeneration: Every $CASCADE_CACHE_REGEN_INTERVAL epochs (periodic)"
else
    echo "  Cache regeneration: Manual only (set CASCADE_CACHE_REGEN_INTERVAL > 0 to enable)"
fi
echo ""
echo "Cache-based streaming benefits:"
echo "  ✓ Small cache (~17 GB vs 1.7 TB for full 100M dataset)"
echo "  ✓ Fast generation (~15 minutes vs 35 days)"
echo "  ✓ Fresh data option (prevents overfitting)"
echo "  ✓ Numpy memmap (thread-safe, 32+ workers)"
echo "  ✓ Zero decompression overhead"
echo ""
echo "Features enabled:"
if [ "$CASCADE_USE_STREAMING" = "true" ]; then
echo "  ✓ STREAMING DATASET (10s I/Q streams)"
echo "    - 10-second streams with 2-5 complete messages"
echo "    - Natural temporal collisions from Poisson arrivals"
echo "    - 5 training windows extracted per stream"
echo "    - GPU polar encoding (89× faster than CPU)"
else
echo "  ✓ Snippet-based dataset (legacy, not recommended)"
fi
echo "  ✓ RX kernel prediction (simplified CASCADE)"
echo "  ✓ Context signals (8 max)"
echo "  ✓ GPU-accelerated generation"
echo "  ✓ Pre-calculated physics (parallel CPU)"
echo "  ✓ Early stopping (patience=$CASCADE_EARLY_STOP_PATIENCE)"
echo ""

# ============================================================================
# GPU Check
# ============================================================================

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "ERROR: CUDA/GPU not available! This training requires GPU."
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEMORY=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')")

echo "GPU Configuration:"
echo "  Device: $GPU_NAME"
echo "  Memory: ${GPU_MEMORY} GB"
echo ""

# ============================================================================
# Estimated Time
# ============================================================================

echo "Estimated time (with cache-based streaming):"
if [ "$CASCADE_USE_STREAMING" = "true" ]; then
    # Calculate cache generation time based on actual samples
    train_streams=$((CASCADE_TRAIN_SAMPLES / 5))  # 5 windows per stream
    val_streams=$((CASCADE_VAL_SAMPLES / 5))
    total_streams=$((train_streams + val_streams))

    # At ~110 streams/sec, calculate time in minutes
    generation_minutes=$((total_streams / 110 / 60))

    echo "  Cache generation: ~${generation_minutes} minutes (${total_streams} streams at 110/sec)"
    echo "    Training cache: $CASCADE_TRAIN_SAMPLES samples (~$((CASCADE_TRAIN_SAMPLES * 17 / 1000)) MB)"
    echo "    Validation cache: $CASCADE_VAL_SAMPLES samples (~$((CASCADE_VAL_SAMPLES * 17 / 1000)) MB)"
    echo "  Cache regeneration: Manual (set CASCADE_REGENERATE_CACHE=true for fresh data)"
else
    echo "  Dataset generation: Not using streaming (very slow)"
fi
echo ""
echo "  Stage 1 (IQ Encoder): ~30-60 min"
echo "  Stage 2-3 (CASCADE Model): ~1-2 hours"
echo "  Stage 4 (Joint RX/TX): ~1-2 hours (if enabled)"
echo ""
echo "Total training time: ~3-5 hours (with 100K sample cache)"
echo ""

# ============================================================================
# Run Training
# ============================================================================

echo "Starting training..."
echo "================================================================================"
echo ""

# Pass START_STAGE to Python script
if [ "$START_STAGE" -gt 1 ]; then
    echo "▶️  Running: python3 ./phase3_model_training.py --start-stage $START_STAGE"
    python3 ./phase3_model_training.py --start-stage "$START_STAGE"
else
    python3 ./phase3_model_training.py
fi

echo ""
echo "================================================================================"
echo "✓ TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "Artifacts generated in:"
echo "  artifacts/phase3/checkpoints/*.pth - Model checkpoints"
echo "  artifacts/phase3/training_log.json - Training history"
echo "  artifacts/phase3/*.png - Training curves"
echo ""
echo "Models trained:"
echo "  ✓ Stage 1: IQ Encoder (2048 → 512 compression)"
echo "  ✓ Stage 2-3: Complete CASCADE (5 experts + decoder with RX kernel prediction)"
echo ""
echo "Cache information:"
echo "  Dataset cache: ./dataset_cache/"
echo "  Size: ~17 GB (100K samples)"
echo "  Regenerate with: CASCADE_REGENERATE_CACHE=true ./run_full_training.sh"
echo "  Scale to 1M samples: CASCADE_TRAIN_SAMPLES=1000000 ./run_full_training.sh"
echo ""
