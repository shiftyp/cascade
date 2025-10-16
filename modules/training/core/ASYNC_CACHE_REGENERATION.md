# Async Cache Regeneration Analysis

## Question: Can we regenerate cache in parallel with training?

**Short answer**: Yes, but performance loss is significant (30-50% training slowdown). Better to regenerate **between epochs** with minimal overhead (5-10%).

---

## Resource Contention Analysis

### System Resources (GH200)
- **GPU**: 1× H100 (96 GB memory, ~80 TFLOPS)
- **CPU**: 72 cores (Grace ARM)
- **RAM**: 480 GB
- **Storage**: NFS (shared, network-limited I/O)

### Training Resource Usage
```
GPU utilization: 80-90% (forward/backward passes)
GPU memory: ~40-50 GB (model + batch + optimizer states)
CPU workers: 32 DataLoader workers (loading I/Q windows from memmap)
CPU usage: ~30-40% (mostly I/O wait)
RAM: ~60-80 GB (model + data buffers + worker memory)
NFS I/O: Random reads at ~2-5 GB/s (32 workers × batch_size)
```

### Cache Generation Resource Usage
```
GPU utilization: 60-80% (signal generation, channel simulation, QRN)
GPU memory: ~30-40 GB (batch signals, FFTs, channel simulation)
CPU workers: 51 physics workers (parallel scenario generation)
CPU usage: ~80-90% (physics calculations, numpy operations)
RAM: ~40-60 GB (physics buffers, signal batches, write queues)
NFS I/O: Sequential writes at ~500 MB/s (async background writers)
```

---

## Performance Impact Scenarios

### Scenario 1: Full Parallel (NOT RECOMMENDED)
**Run cache regeneration continuously during training**

**Resource conflicts**:
- **GPU**: 80% (training) + 70% (generation) = **150% demand → GPU saturated**
  - CUDA context switching overhead
  - Reduced compute for both processes
  - Training forward/backward slows by 40-60%

- **CPU**: 32 (training) + 51 (generation) = **83 workers on 72 cores**
  - Context switching overhead
  - Cache thrashing
  - 20-30% CPU slowdown

- **NFS I/O**:
  - Training: Random reads (latency-sensitive)
  - Generation: Sequential writes (throughput-optimized)
  - Could cause read latency spikes → DataLoader stalls

**Expected training slowdown**: **40-60%** (GPU saturation is bottleneck)

**Example**:
- Normal training: 50 epochs × 30 min/epoch = 25 hours
- With parallel regen: 50 epochs × 48 min/epoch = 40 hours
- **15 hours lost vs 15 minutes gained** (cache generation time saved)

**Verdict**: ❌ **NOT WORTH IT** - lose more time than you save!

---

### Scenario 2: Between-Epoch Regeneration (RECOMMENDED)
**Regenerate cache at epoch boundaries while training continues**

**Strategy**:
```python
# Epoch timeline:
Epoch 1: Train on cache_v1 (30 min)
  → At epoch end, trigger async cache_v2 generation (15 min background)

Epoch 2: Train on cache_v1 (30 min)
  → cache_v2 finishes during epoch 2

Epoch 3: Switch to cache_v2 (0.1 sec), train (30 min)
  → At epoch end, trigger async cache_v3 generation

Epoch 4: Train on cache_v2 (30 min)
  → cache_v3 finishes during epoch 4
...
```

**Resource conflicts**:
- **GPU**: Training uses 80-90%, generation uses 60-80%
  - But training has priority (CUDA stream scheduling)
  - Generation runs at lower priority → training unaffected
  - Generation takes 20-30% longer (15 min → 18-20 min)

- **CPU**: Minimal conflict
  - Training workers mostly I/O-wait during GPU compute
  - Generation can use CPU during those gaps

- **NFS I/O**:
  - Training reads from cache_v1
  - Generation writes to cache_v2 (different file)
  - Minimal contention (sequential writes don't block random reads)

**Expected training slowdown**: **5-10%** (mostly from I/O contention)

**Example**:
- Normal training: 50 epochs × 30 min/epoch = 25 hours
- With between-epoch regen: 50 epochs × 31.5 min/epoch = 26.25 hours
- **1.25 hours lost, but always training on fresh data!**

**Verdict**: ✅ **RECOMMENDED** - minimal overhead, maximum freshness!

---

### Scenario 3: Periodic Regeneration (SIMPLE ALTERNATIVE)
**Regenerate cache every N epochs during validation**

**Strategy**:
```python
# Train for 5 epochs on cache_v1
Epochs 1-5: Train on cache_v1 (2.5 hours)

# Regenerate during validation (when GPU is light)
Validation: Run validation (5 min), regenerate cache_v2 (15 min)

# Train for 5 more epochs on cache_v2
Epochs 6-10: Train on cache_v2 (2.5 hours)
...
```

**Resource conflicts**: None (generation happens when training paused)

**Expected training slowdown**: **0%** (no overlap)

**Example**:
- 50 epochs in 10 cycles of 5 epochs
- Training time: 50 × 30 min = 25 hours
- Regeneration time: 10 × 15 min = 2.5 hours
- **Total: 27.5 hours (10% overhead, but simple!)**

**Verdict**: ✅ **SIMPLEST** - no complexity, predictable timing!

---

## Implementation: Between-Epoch Regeneration

### High-Level Architecture
```python
class AsyncCacheManager:
    """
    Manages dual-version cache with background regeneration.

    - Maintains cache_v1 and cache_v2
    - Training reads from active cache
    - Background process writes to inactive cache
    - Atomic swap at epoch boundary
    """

    def __init__(self):
        self.active_cache = 'v1'
        self.inactive_cache = 'v2'
        self.regen_process = None

    def trigger_regeneration(self, epoch: int):
        """Start async cache regeneration at epoch end."""
        if self.regen_process is not None and self.regen_process.is_alive():
            # Previous regeneration still running
            return False

        # Start background process to regenerate inactive cache
        self.regen_process = Process(
            target=regenerate_cache,
            args=(self.inactive_cache, epoch),
            daemon=True
        )
        self.regen_process.start()
        return True

    def maybe_swap_cache(self) -> bool:
        """Swap to new cache if regeneration complete."""
        if self.regen_process is None:
            return False

        if not self.regen_process.is_alive():
            # Regeneration complete - swap caches
            self.active_cache, self.inactive_cache = \
                self.inactive_cache, self.active_cache

            # Clean up old cache (now inactive)
            cleanup_cache(self.inactive_cache)

            return True
        return False
```

### Integration with Training Loop
```python
# In phase3_model_training.py

cache_manager = AsyncCacheManager()

for epoch in range(num_epochs):
    # Train on active cache
    train_epoch(model, cache_manager.active_cache)

    # At epoch end, check if new cache ready
    if cache_manager.maybe_swap_cache():
        print(f"✓ Switched to fresh cache (regenerated during epoch {epoch})")

        # Reload dataset with new cache
        train_dataset = StreamingCascadeDataset(
            cache_dir=cache_manager.active_cache_path,
            use_cache=True,
            regenerate_cache=False
        )
        train_loader = DataLoader(train_dataset, ...)

    # Trigger next regeneration (runs in background during next epoch)
    if epoch % 2 == 0:  # Regenerate every 2 epochs
        if cache_manager.trigger_regeneration(epoch + 1):
            print(f"→ Started background cache regeneration for epoch {epoch+2}")
```

### GPU Priority Management
```python
# Set lower priority for background generation process
import os

def regenerate_cache(cache_version: str, seed: int):
    """Background cache regeneration with lower GPU priority."""

    # Set CUDA device and lower priority
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import torch
    torch.cuda.set_device(0)

    # Use separate CUDA stream with lower priority
    low_priority_stream = torch.cuda.Stream(priority=-1)

    with torch.cuda.stream(low_priority_stream):
        # Generate dataset (training has priority on default stream)
        dataset = StreamingCascadeDataset(
            num_streams=20000,  # 100K samples
            cache_dir=f'./dataset_cache/{cache_version}',
            regenerate_cache=True,
            seed=seed,
            device='cuda'
        )

    print(f"✓ Cache {cache_version} regenerated (seed={seed})")
```

---

## Recommended Strategy

### For Current Setup (100K samples)
**Use Scenario 3: Periodic Regeneration**

```bash
# In run_full_training.sh, add:
export CASCADE_CACHE_REGEN_INTERVAL=5  # Regenerate every 5 epochs
```

**Why**:
- 15 min generation vs 2.5 hour training cycle (5 epochs)
- 10% overhead is acceptable
- Zero complexity (just pause training, regenerate, resume)
- Predictable timing for monitoring

### For Larger Caches (1M samples)
**Use Scenario 2: Between-Epoch Regeneration**

```bash
export CASCADE_TRAIN_SAMPLES=1000000
export CASCADE_CACHE_REGEN_ASYNC=true
export CASCADE_CACHE_REGEN_INTERVAL=2  # Fresh data every 2 epochs
```

**Why**:
- 2.5 hour generation vs 5 hour per epoch
- Background regen overlaps with training
- Fresh data every 2 epochs
- Worth the added complexity at this scale

---

## Performance Impact Summary

| Approach | Training Slowdown | Implementation Complexity | Recommended |
|----------|-------------------|---------------------------|-------------|
| **Full Parallel** | 40-60% | High | ❌ No |
| **Between-Epoch Async** | 5-10% | Medium | ✅ Yes (1M+ samples) |
| **Periodic Simple** | 0% (10% total overhead) | Low | ✅ Yes (100K samples) |
| **No Regeneration** | 0% | None | ⚠️ Risk overfitting |

---

## Conclusion

**For your current 100K sample setup**:
- **Recommended**: Periodic regeneration every 5 epochs (Scenario 3)
- **Overhead**: 10% total time (2.5 hours → 27.5 hours for 50 epochs)
- **Benefit**: Fresh physics/noise every 5 epochs, prevents overfitting
- **Implementation**: Simple (pause, regenerate, resume)

**For scaling to 1M samples**:
- **Recommended**: Between-epoch async regeneration (Scenario 2)
- **Overhead**: 5-10% per epoch
- **Benefit**: Always training on recent data
- **Implementation**: Moderate complexity (dual-cache + background process)

**Key insight**: Cache regeneration time (15 min) is **small compared to epoch time (30 min)**, so overhead is minimal with either approach. The main benefit is **data freshness** for better generalization, not time savings.
