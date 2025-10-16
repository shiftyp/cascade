# CASCADE Training Throughput Optimizations

**Date:** 2025-10-14
**Issue:** GPU utilization sporadic (0% → 98% → 0%), indicating data loading bottleneck
**Root Cause:** CPU preprocessing in DataLoader workers (NOT GPU transfer!)
**Expected Speedup:** 2-4× overall training throughput

---

## Optimizations Implemented

### Phase 1: High-Impact Quick Wins

#### 1. Pre-computed Normalization Statistics ✅
**File:** `streaming_cascade_dataset.py`

**Problem:**
- Computing `mean()` and `std()` on 96,000 samples × 2 channels per training sample
- With batch size 4096: **786M float operations per batch!**
- Executed in each of 32 DataLoader worker processes

**Solution:**
- Pre-compute mean/std for each stream during dataset generation (lines 1092-1101)
- Store in `normalization_stats.npy` memmap (line 568-573)
- Use pre-computed values in `__getitem__` (lines 1672-1679)

**Expected Gain:** 40-60% faster `__getitem__`

---

#### 2. RAM-cached Labels ✅
**File:** `streaming_cascade_dataset.py`

**Problem:**
- 32 DataLoader workers accessing 21MB HDF5 label file concurrently
- HDF5 file locking creates I/O contention

**Solution:**
- Load all labels into RAM at dataset init (lines 1455-1470)
- Use RAM-cached labels in `__getitem__` (lines 1739-1751)
- ~21MB total, negligible memory overhead

**Expected Gain:** 15-25% faster label access with 32 workers

---

#### 3. Increased Batch Size ✅
**File:** `phase3_model_training.py`

**Change:**
- Batch size: **512 → 8192** (line 2658)

**Rationale:**
- GPU has 97GB memory, only using 25GB
- 72GB free memory available
- Larger batches = better GPU utilization, fewer data loading cycles

**Expected Gain:** 20-30% overall throughput

---

### Phase 2: Medium-Impact Optimizations

#### 4. Mixed Precision Training (AMP) ✅
**Files:** `phase3_model_training.py`

**Changes:**
- Added AMP support to `IQEncoderTrainer` (lines 2122-2152, 2167-2202)
- CASCADE trainer already had AMP enabled (confirmed)

**Benefits:**
- 2× faster matrix operations on NVIDIA GPUs
- Reduced memory usage (FP16 vs FP32)
- No accuracy loss with gradient scaling

**Expected Gain:** 40-50% faster GPU compute time

---

#### 5. Increased Prefetch Factor ✅
**File:** `phase3_model_training.py`

**Change:**
- Prefetch factor: **4 → 8** (lines 2733, 2739, 2768, 2774)

**Benefits:**
- More aggressive prefetching masks data loading latency
- With 32 workers: **256 batches prefetched** (32 workers × 8 batches)

**Expected Gain:** 10-15% better GPU utilization

---

## Dataset Cache Regeneration Required

**IMPORTANT:** The optimizations require regenerating the dataset cache to include:
1. Pre-computed normalization statistics (`normalization_stats.npy`)

### How to Regenerate:

```bash
# Delete old cache
rm -rf modules/training/core/dataset_cache/streaming_cascade_*

# Regenerate (will create new cache with optimizations)
cd modules/training/core
python3 ./phase3_model_training.py
```

The new cache will include the normalization statistics. Subsequent training runs will automatically use the optimizations.

---

## Performance Monitoring

### Before Optimization:
- GPU utilization: 0% → 98% → 0% → 98% (sporadic, starving)
- GPU memory: 25GB / 97GB (only 26% utilization)
- Batch size: 512
- Prefetch: 4 batches/worker (128 total with 32 workers)

### After Optimization:
Expected improvements:
- GPU utilization: More consistent (reduced idle time)
- Batch size: 8192 (16× larger)
- Prefetch: 8 batches/worker (256 total with 32 workers)
- Overall speedup: **2-4× faster training**

### Monitoring Commands:

```bash
# Monitor GPU utilization in real-time
nvidia-smi dmon -s puc

# Check training throughput
tail -f modules/training/core/training.log | grep "samples/sec"
```

---

## Technical Details

### Normalization Statistics Computation

**During dataset generation:**
```python
# Pre-compute stats for each stream
for i in range(actual_batch_size):
    stream_iq = streams_cpu[i]
    iq_i = np.real(stream_iq).astype(np.float32)
    iq_q = np.imag(stream_iq).astype(np.float32)
    iq_stack = np.stack([iq_i, iq_q], axis=0)
    norm_stats_cpu[i, 0] = np.mean(iq_stack)
    norm_stats_cpu[i, 1] = np.std(iq_stack) + 1e-8
```

**During training:**
```python
# FAST PATH: Use pre-computed stats
if hasattr(self, 'numpy_norm_stats') and self.numpy_norm_stats is not None:
    mean = self.numpy_norm_stats[stream_idx, 0]
    std = self.numpy_norm_stats[stream_idx, 1]
else:
    # SLOW PATH: Compute on-the-fly (fallback)
    mean = np.mean(iq_stack)
    std = np.std(iq_stack) + 1e-8

iq_normalized = (iq_stack - mean) / std
```

---

## Files Modified

1. **`modules/training/core/streaming_cascade_dataset.py`**
   - Added normalization stats memmap creation (line 568-573)
   - Updated writer_worker to save norm stats (line 597-604)
   - Compute norm stats during generation (line 1092-1101)
   - Load norm stats in _load_numpy_cache (line 1381-1388)
   - Use pre-computed stats in __getitem__ (line 1672-1679)
   - Cache all labels in RAM at init (line 1455-1470)
   - Use RAM-cached labels in __getitem__ (line 1739-1751)

2. **`modules/training/core/phase3_model_training.py`**
   - Increased batch size to 8192 (line 2658)
   - Increased prefetch_factor to 8 (lines 2733, 2739, 2768, 2774)
   - Added AMP to IQEncoderTrainer (lines 2122-2152, 2167-2202)

---

## Troubleshooting

### NaN Loss in Encoder Training

**Symptom:**
```
Training IQ Encoder: loss = nan
⚠️ NaN/Inf loss detected! Skipping batch.
```

**Root Cause:** Mixed precision (FP16) numerical instability in loss computation

**Fixes Applied:**
1. **FP16-safe epsilon:** Changed from `1e-8` to `1e-3` (FP16 precision is ~1e-4)
2. **FP32 loss computation:** Loss computed in FP32 for numerical stability
3. **Clamping:** Added `torch.clamp()` to prevent extreme values
4. **NaN detection:** Automatically skips batches with NaN/Inf loss

**Workaround (if issues persist):**
Disable mixed precision for encoder training:
```bash
export CASCADE_ENCODER_USE_AMP=false
python3 ./phase3_model_training.py
```

This will make encoder training ~2× slower but eliminates FP16 issues.

---

### Division by Zero Warning in Dataset

**Symptom:**
```
RuntimeWarning: invalid value encountered in divide
  iq_normalized = (iq_stack - mean) / std
```

**Root Cause:** Some streams have constant signals (std=0), causing division by zero

**Fix Applied:**
- Clamp std to minimum `1e-6` in both pre-computation and runtime
- Location: `streaming_cascade_dataset.py:1103, 1700`

This is now automatically handled - no action needed.

---

### Old Cache Detected

**Error:**
```
⚠️  No pre-computed normalization stats found - will compute on-the-fly (slower)
```

**Solution:** Regenerate cache (see above)

---

### Memory Issues with Larger Batch Size

**Symptom:** CUDA out of memory error

**Solution:** Reduce batch size via environment variable:
```bash
export CASCADE_BATCH_SIZE=4096  # Or lower if needed
```

---

### Training Slower Despite Optimizations

**Check:**
1. Verify normalization stats exist: `ls dataset_cache/*/normalization_stats.npy`
2. Monitor GPU utilization: `nvidia-smi dmon -s puc`
3. Check DataLoader workers: Should see "32 workers" in training output

---

## Summary

These optimizations address the **root cause** of slow training: CPU preprocessing bottleneck in DataLoader workers. The bottleneck was NOT GPU transfer, but rather:
1. Computing mean/std on 192K float values per sample
2. HDF5 I/O contention from 32 concurrent workers
3. Underutilized GPU (small batches, low prefetch)

With these changes, the GPU should spend more time computing and less time idle, resulting in **2-4× overall speedup**.
