# CASCADE Training Optimizations for GH200

## Hardware Specifications
- GPU: NVIDIA GH200 Grace Hopper Superchip
- VRAM: 96 GB
- CPU: 64 ARM cores  
- RAM: Large (unified memory architecture)

## Training Optimizations Applied

### 1. Batch Size: 32 → 2048 (64× increase)
```bash
export CASCADE_BATCH_SIZE=2048
```
- Fully utilizes 96GB VRAM
- Much better GPU compute saturation
- Reduces overhead from small batches

### 2. RAM Caching (Enabled by default)
```bash
export CASCADE_LOAD_TO_RAM=true  # Default
```
- **Bypasses HDF5 completely** during training
- Loads entire dataset into RAM at startup
- Eliminates HDF5 thread-safety serialization
- ~40 GB RAM for 100M samples

### 3. DataLoader Workers (Adaptive)
- **RAM cached**: 0 workers (no I/O needed)
- **HDF5**: 32 workers (parallel file loading)
```bash
export CASCADE_DATALOADER_WORKERS=32  # Only used if not RAM cached
```

### 4. Prefetch Factor: 2 → 4
- Each worker prefetches 4 batches
- With 32 workers: 128 batches always ready
- GPU never waits for data

### 5. Mixed Precision FP16
- Enabled by default (torch.cuda.amp.autocast)
- 2-3× faster computation
- Reduces memory bandwidth by 50%
- GradScaler for stable training

### 6. TF32 Math (GH200-specific)
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
- Faster matmul operations
- No accuracy loss
- Free speedup on GH200/A100/H100

### 7. torch.compile (PyTorch 2.0+)
```python
model = torch.compile(model, mode='max-autotune')
```
- Graph optimization
- Kernel fusion
- 30-50% additional speedup

### 8. cuDNN Auto-tuning
```python
torch.backends.cudnn.benchmark = True
```
- Auto-selects fastest convolution algorithms
- First epoch slower (benchmarking), rest faster

### 9. Persistent Workers
```python
persistent_workers=True
```
- Workers stay alive between epochs
- No startup overhead per epoch

### 10. Non-blocking Transfers
```python
.to(device, non_blocking=True)
```
- Overlaps CPU→GPU copies with computation
- Async data movement

## Expected Performance

### Before Optimizations:
- Batch size: 32
- Workers: 0 (single-threaded HDF5)
- No mixed precision
- **Speed: 1 it/sec**
- **GPU Utilization: 10-20%**

### After Optimizations:
- Batch size: 2048
- Workers: 0 (RAM cached, no I/O)
- FP16 + TF32 + torch.compile
- **Speed: 50-200+ it/sec**
- **GPU Utilization: 80-95%**

**Expected speedup: 50-200× faster!**

## Training Statistics Now Shown

Every 100 batches:
- Loss
- Pattern Accuracy (%)
- Frequency Accuracy (%)
- Modulation Accuracy (%)
- Throughput (samples/sec)
- GPU Memory (GB)

Every epoch:
- All above metrics averaged
- Epoch time
- Peak GPU memory

## Environment Variables Summary

```bash
# Training
export CASCADE_BATCH_SIZE=2048           # Large batch for GH200
export CASCADE_EPOCHS=30                  # Number of epochs

# Dataset loading
export CASCADE_LOAD_TO_RAM=true          # Load to RAM (default)
export CASCADE_DATALOADER_WORKERS=32     # Only if not RAM cached

# Dataset generation (from run_full_training.sh)
export CASCADE_TRAIN_SAMPLES=100000000   # 100M samples
export CASCADE_VAL_SAMPLES=1000000        # 1M validation
```

## To Apply:

1. **Stop current training** (if running)
2. **Restart**: `./run_full_training.sh`
3. **You should see**:
   - "🚀 GH200 Grace Hopper Superchip Detected!"
   - "Loading entire dataset into RAM..."
   - "Using 0 DataLoader workers (dataset in RAM)"
   - "Mixed Precision (FP16): ✅ ENABLED"
   - Much higher it/sec (50-200+)
   - GPU utilization 80-95%

## Memory Usage

For 100M training samples (20M streams):
- Dataset in RAM: ~40 GB
- Model: ~6 GB (FP32) or ~3 GB (FP16)
- Activations (batch 2048): ~15 GB
- **Total: ~60 GB** (fits comfortably in 96 GB VRAM + system RAM)

## Troubleshooting

**If RAM loading fails (OOM):**
```bash
export CASCADE_LOAD_TO_RAM=false
export CASCADE_DATALOADER_WORKERS=32
export CASCADE_BATCH_SIZE=1024  # Reduce if needed
```

**If GPU OOM:**
```bash
export CASCADE_BATCH_SIZE=1024  # Or 512
```

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

Look for:
- GPU Utilization: Should be 80-95%
- Memory Usage: Should be 50-70 GB
- Power: Should be near TDP (700W for GH200)
