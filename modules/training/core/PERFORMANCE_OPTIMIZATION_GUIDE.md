# CASCADE Performance Optimization Guide

**Last Updated**: 2025-01-14
**Target Hardware**: NVIDIA GH200 Grace Hopper (97GB GPU + 480GB CPU RAM)

## Quick Wins Summary

🚀 **Expected Overall Speedup**: 4-10× end-to-end
⏱️ **Implementation Time**: < 2 hours
💰 **Code Changes Required**: Minimal (mostly configuration)

---

## Part 1: Dataset Generation Optimizations

### ✅ Already Optimized

The codebase already has excellent optimizations in place:

1. **GPU Signal Generation** (`gpu_signal_generator.py`)
   - ✅ Batch generation (128-512 signals parallel)
   - ✅ GPU Polar encoding (90× faster than CPU)
   - ✅ Pre-computed RC filters cached on GPU
   - ✅ FFT-based convolution (100× faster for long filters)

2. **GPU Channel Simulation** (`gpu_channel_simulator.py`)
   - ✅ Time-varying multipath with adaptive updates
   - ✅ Continuous frequency-selective fading
   - ✅ Batch processing for all channel effects

3. **Streaming Dataset** (`streaming_cascade_dataset.py`)
   - ✅ 10-second continuous streams (80+ samples/sec)
   - ✅ Multi-message collisions (realistic scenarios)
   - ✅ NumPy memmap format (thread-safe for DataLoader)

### 🔧 Recommended Tuning

#### 1. Increase GPU Batch Sizes (Easy - 20% gain)

**File**: `streaming_cascade_dataset.py:96`

```python
# BEFORE
batch_size: int = 128

# AFTER (GH200 has 97GB - use it!)
batch_size: int = 512  # 4× larger batches
```

**Memory check**:
- 512 streams × 480K samples × 8 bytes (complex64) = **1.8GB** ✓
- GH200 can handle 40GB+ easily

#### 2. Optimize Channel Update Intervals (Easy - 3× gain)

**File**: `gpu_channel_simulator.py:274-280`

```python
# BEFORE (adaptive but can be optimized)
if duration_s > 5.0:
    adaptive_update_ms = 100  # 100ms updates
elif duration_s > 2.0:
    adaptive_update_ms = 50   # 50ms updates
else:
    adaptive_update_ms = self.update_interval_ms  # 10ms

# AFTER (realistic for HF + faster)
adaptive_update_ms = 50  # Always use 50ms (still realistic, 5× faster than 10ms)
```

**Justification**:
- HF ionosphere coherence time: 100-500ms typical
- 50ms updates = 20 snapshots/second (still captures fading)
- 10ms was overkill (100 snapshots/sec)

#### 3. Reduce Ramp Duration for Training (Optional - 15% speedup)

**File**: `generator.py:137`

```python
# BEFORE
RAMP_DURATION_MS = 150  # Conservative for real transmissions

# AFTER (for training data only - faster)
RAMP_DURATION_MS = 50   # 3× fewer samples to process downstream
```

**Note**: Keep 150ms for production/real signals. Use 50ms only for training dataset generation.

---

## Part 2: Training Pipeline Optimizations

### ⚠️ CRITICAL: DataLoader Configuration

**File**: `phase3_model_training.py` (needs updates)

Currently missing optimal DataLoader settings!

#### Add This Configuration:

```python
from torch.utils.data import DataLoader

# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=512,              # ← Increase from default (64)
    num_workers=32,              # ← Enable parallel loading (CRITICAL!)
    pin_memory=True,             # ← Faster CPU→GPU transfer
    prefetch_factor=4,           # ← Pre-load 4 batches per worker
    persistent_workers=True,     # ← Reuse workers (avoid startup cost)
    shuffle=True
)

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=512,
    num_workers=16,              # ← Fewer workers for validation
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    shuffle=False
)
```

**Expected Impact**:
- **Without**: Single-threaded I/O bottleneck (GPU idle 80% of time)
- **With**: 10-20× faster data loading, GPU utilization >95%

**Requirements**:
- ✅ NumPy memmap format (already used in `streaming_cascade_dataset.py`)
- ✅ Thread-safe dataset (`__getitem__` must be pure function)

---

### 🚀 Mixed Precision Training (AMP)

**Impact**: 2× faster forward/backward, 50% less memory

Add this to training loop:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (rx_iq, labels) in enumerate(train_loader):
        rx_iq = rx_iq.to(device)

        optimizer.zero_grad()

        # *** AUTOMATIC MIXED PRECISION ***
        with autocast():  # ← FP16 for matmul/conv, FP32 for reductions
            outputs = model(rx_iq, context_signals=None)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**What it does**:
- Matmul/Conv operations: **FP16** (2× faster on Ampere/Hopper)
- Loss/normalization: **FP32** (numerical stability)
- Automatic loss scaling (prevents underflow)

**Compatibility**: Works with all PyTorch models (no code changes needed)

---

### ⚡ torch.compile() - PyTorch 2.0+ JIT

**Impact**: 1.3-1.8× faster training loop

```python
# After model definition
model = CascadeModel()
model = model.to(device)

# *** COMPILE MODEL (PyTorch 2.0+) ***
model = torch.compile(
    model,
    mode='reduce-overhead',  # Optimize for training
    # mode='max-autotune',   # Alternative: more aggressive (slower compile)
)
```

**What it does**:
- Fuses operations (reduce kernel launches)
- Optimizes memory access patterns
- Compiles on first forward pass (~30-60 seconds)
- Subsequent iterations are 1.3-1.8× faster

**Trade-off**:
- First epoch: +30-60s compile time
- All subsequent epochs: 1.3-1.8× faster
- Worth it for >10 epoch training

---

### 📊 Gradient Accumulation (for larger effective batch size)

**Impact**: Better convergence + handle OOM scenarios

```python
accumulation_steps = 4  # Effective batch = 512 × 4 = 2048

for epoch in range(num_epochs):
    optimizer.zero_grad()

    for batch_idx, (rx_iq, labels) in enumerate(train_loader):
        rx_iq = rx_iq.to(device)

        with autocast():
            outputs = model(rx_iq)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # ← Scale loss

        scaler.scale(loss).backward()

        # Step optimizer every N batches
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

**When to use**:
- Want larger batch size but hit memory limits
- Improve training stability
- Better gradient estimates

---

## Part 3: Memory Optimizations

### 1. Enable Gradient Checkpointing (2× more capacity)

**For large models that exceed GPU memory**:

```python
import torch.utils.checkpoint as checkpoint

class CascadeModel(nn.Module):
    def forward(self, x):
        # Checkpoint expensive blocks
        x = checkpoint.checkpoint(self.encoder_block, x)
        x = checkpoint.checkpoint(self.expert_block, x)
        x = self.decoder(x)
        return x
```

**Trade-off**:
- 50% less memory (stores only activations at checkpoints)
- 20% slower (recomputes activations during backward)

### 2. Use channels_last Memory Format

**Better cache locality for Conv layers**:

```python
# After model creation
model = model.to(memory_format=torch.channels_last)

# Also convert inputs
rx_iq = rx_iq.to(memory_format=torch.channels_last)
```

**Expected gain**: 1.1-1.2× faster convolutions

---

## Part 4: Monitoring & Profiling

### GPU Utilization Check

```bash
# During training, run in separate terminal:
watch -n 1 nvidia-smi

# Look for:
# - GPU Util: Should be >90%
# - Memory: Should be 50-80% (leave headroom)
# - Power: Should be near max (400W for GH200)
```

**If GPU Util < 70%**:
- ✅ Increase `num_workers` in DataLoader
- ✅ Increase batch size
- ✅ Check CPU bottleneck (should use <50% per core)

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx >= 10:  # Profile first 10 batches
            break
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## Part 5: Quick Reference Checklist

### Before Training:

- [ ] Dataset in NumPy memmap format (not HDF5)
- [ ] DataLoader: `num_workers=32, pin_memory=True`
- [ ] Mixed precision: `autocast()` + `GradScaler()`
- [ ] Model compiled: `torch.compile(model)`
- [ ] Batch size: 512+ (use GPU memory!)

### During Training:

- [ ] GPU utilization >90% (`nvidia-smi`)
- [ ] No CPU bottleneck (check htop)
- [ ] Loss decreasing (not NaN)
- [ ] Memory usage 50-80% (safe headroom)

### Dataset Generation:

- [ ] GPU batch size: 512 streams
- [ ] Channel update interval: 50ms
- [ ] Ramp duration: 50ms (training) or 150ms (production)

---

## Performance Benchmarks

### Expected Training Speed (GH200):

| Configuration | Samples/sec | GPU Util | Speedup |
|--------------|-------------|----------|---------|
| **Baseline** (single-threaded, FP32) | 50 | 30% | 1× |
| **+ DataLoader workers** | 400 | 85% | 8× |
| **+ Mixed Precision (AMP)** | 800 | 90% | 16× |
| **+ torch.compile()** | 1200 | 95% | 24× |

### Expected Dataset Generation:

| Method | Speed | Notes |
|--------|-------|-------|
| CPU SignalGenerator | 2 samples/sec | Legacy |
| GPU (batch=128) | 80 samples/sec | Current |
| GPU (batch=512) | **120 samples/sec** | Recommended |

---

## Troubleshooting

### OOM (Out of Memory) Errors

1. **Reduce batch size**: 512 → 256 → 128
2. **Enable gradient checkpointing** (see above)
3. **Clear cache**: `torch.cuda.empty_cache()` between epochs

### Slow Data Loading

1. **Check format**: Must use NumPy memmap (not HDF5)
2. **Increase workers**: Try 16, 32, 48
3. **Reduce prefetch**: 4 → 2 (if high CPU usage)

### NaN Loss

1. **Check SNR floor**: Ensure SNR > -14 dB in dataset
2. **Reduce learning rate**: 1e-3 → 1e-4
3. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

---

## Implementation Order

**For maximum impact with minimal effort**:

1. **Day 1 (2 hours)**:
   - ✅ Add DataLoader optimizations
   - ✅ Enable mixed precision (AMP)
   - **Result**: 10-15× speedup immediately

2. **Day 2 (1 hour)**:
   - ✅ Add torch.compile()
   - ✅ Increase GPU batch sizes
   - **Result**: Additional 2× speedup

3. **Day 3 (optional)**:
   - ✅ Optimize channel update intervals
   - ✅ Profile and fine-tune
   - **Result**: Final 1.2-1.5× gain

---

## References

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [torch.compile() Guide](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [DataLoader Best Practices](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)

---

**Questions?** Check existing optimizations in:
- `gpu_signal_generator.py` (signal generation)
- `gpu_channel_simulator.py` (channel effects)
- `streaming_cascade_dataset.py` (dataset structure)
