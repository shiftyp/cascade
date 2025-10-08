# CASCADE Training Strategy: Synthetic to Real

## Executive Summary

**Goal**: Train CASCADE model that generalizes from synthetic data to real HF conditions.

**Approach**: 3-phase hybrid training combining synthetic data, realistic channel simulation, and real-world recordings.

**Expected Performance**:
- Synthetic-only: ~85% accuracy on real data
- With realistic channel: ~92% accuracy on real data
- With real noise augmentation: ~95% accuracy on real data
- Fine-tuned on real data: ~97% accuracy on real data

---

## Phase 1: Synthetic Foundation (Weeks 1-4)

### Objective
Build strong base model using pure synthetic data with realistic channel effects.

### Dataset
```python
# 200,000 training samples (8 GB)
# 50,000 validation samples (2 GB)
# 50,000 test samples (2 GB)

train_dataset = HybridCascadeDataset(
    num_samples=200000,
    use_realistic_channel=True,  # ✓ Hardware + ionospheric effects
    use_real_noise=False,        # Pure synthetic noise
    snr_range=(-15, 20),
    train=True
)
```

### Realistic Channel Features
1. **Hardware Impairments**
   - I/Q imbalance (0.5 dB typical)
   - Phase error (2° typical)
   - DC offset drift
   - Phase noise from LO
   - ADC quantization (14-bit)
   - Frequency drift

2. **Ionospheric Effects**
   - Solar flux variation (SFI 70-250)
   - Geomagnetic activity (K 0-8)
   - Time-of-day propagation
   - D-layer absorption
   - Sporadic E enhancement (5% probability)
   - Auroral flutter (high K-index)

3. **Realistic Interference**
   - Broadcast stations (AM modulated)
   - Pulsed radar
   - Power line harmonics (real structure)

### Training Time
- **Generation**: 4-8 hours (parallelized)
- **Stage 1 (IQ Encoder)**: 2-3 hours
- **Stage 2 (Experts)**: 5-6 hours
- **Stage 3 (Decoder)**: 3-4 hours
- **Total**: ~15-20 hours

### Expected Results
- Synthetic test set: 95%+ accuracy
- Real test set (if available): 85-92% accuracy

---

## Phase 2: Real Noise Integration (Weeks 5-6)

### Objective
Improve robustness using real HF noise mixed with synthetic signals.

### Data Collection

#### Option A: WebSDR (Easiest)
```bash
# 1. Visit http://websdr.org
# 2. Tune to quiet frequency (14.100 MHz)
# 3. Record 10 hours of background noise
# 4. Save as WAV files

# Convert to I/Q
python scripts/convert_wav_to_iq.py websdr_recording.wav data/real_noise/noise1.npy
```

**Time**: 2-3 hours of manual recording + processing

#### Option B: KiwiSDR (Automated)
```bash
# Install client
pip install kiwiclient

# Collect from multiple locations
python scripts/collect_real_noise.py \
    --source multi \
    --freq 14.100 \
    --duration 600 \
    --output data/real_noise/
```

**Time**: 10-20 hours automated recording (run overnight)

#### Option C: Local RTL-SDR
```bash
# Record with RTL-SDR dongle
python scripts/collect_real_noise.py \
    --source rtlsdr \
    --freq 14.100 \
    --duration 3600 \
    --output data/real_noise/
```

**Time**: As long as you want, fully automated

### Target Real Noise Collection
- **Minimum**: 1 hour (sufficient for augmentation)
- **Recommended**: 10 hours (diverse conditions)
- **Optimal**: 50 hours (day/night, different bands, locations)

### Augmented Training
```python
train_dataset = HybridCascadeDataset(
    num_samples=200000,
    use_realistic_channel=True,
    use_real_noise=True,  # ✓ Mix real noise
    real_noise_dir='data/real_noise/',
    train=True
)

# 50% samples use real noise, 50% use synthetic
# Provides domain diversity
```

### Training Time
- **Fine-tuning**: 3-5 hours (same architecture)
- Or train from scratch: 15-20 hours

### Expected Results
- Synthetic test: 95%+ accuracy (maintained)
- Real test: 92-95% accuracy (improved!)

---

## Phase 3: Real Signal Fine-tuning (Weeks 7-8, Optional)

### Objective
Achieve maximum performance on real HF signals through fine-tuning.

### Real Signal Collection

#### Collect Real CASCADE Transmissions
If you have:
1. Two SDRs (TX + RX)
2. HF transceivers
3. Remote stations willing to transmit

```python
# Transmit known CASCADE messages
# Record at receiver
# Create labeled real dataset

real_signals = {
    'iq_samples': recorded_iq,
    'true_pattern': pattern_id,
    'true_frequency': frequency_pair,
    'true_modulation': modulation,
    # etc...
}
```

**Challenge**: Expensive, requires coordination

#### Alternative: Transfer from Similar Modes

Collect recordings of existing HF digital modes:
- FT8 (similar GFSK + data layers)
- PSK31/63
- RTTY

Use for domain adaptation:

```python
class DomainAdaptationTrainer:
    """Learn from unlabeled real HF data."""

    def train_step(self, synthetic_labeled, real_unlabeled):
        # 1. Train on synthetic (supervised)
        loss_supervised = train_on_labels(synthetic_labeled)

        # 2. Encoder must confuse synthetic vs real
        loss_adversarial = domain_confusion(synthetic, real_unlabeled)

        # Combined loss
        loss = loss_supervised + 0.1 * loss_adversarial
```

### Training Time
- **Fine-tuning**: 2-3 hours
- **Full retraining**: Not recommended (lose synthetic coverage)

### Expected Results
- Real CASCADE signals: 97-99% accuracy
- Synthetic test: 93-95% (slight degradation acceptable)

---

## Recommended Training Path

### For Most Users (Good Performance, Low Effort)
```
Phase 1 Only:
└─ 200K synthetic + realistic channel
   └─ Train time: 15-20 hours
      └─ Real performance: 90-92%
```

### For Best Results (Recommended)
```
Phase 1 + Phase 2:
├─ 200K synthetic + realistic channel
└─ 10 hours real noise augmentation
   └─ Train time: 20-25 hours total
      └─ Real performance: 94-96%
```

### For Production Deployment
```
Phase 1 + Phase 2 + Phase 3:
├─ 200K synthetic + realistic channel
├─ 20 hours real noise
└─ Fine-tune on real CASCADE transmissions
   └─ Train time: 25-30 hours total
      └─ Real performance: 97-99%
```

---

## Data Requirements Summary

| Phase | Synthetic Samples | Real Noise Hours | Real Signals | Total Storage | Training Time |
|-------|------------------|------------------|--------------|---------------|---------------|
| **Phase 1** | 250K | 0 | 0 | 10 GB | 15-20 hrs |
| **Phase 2** | 250K | 10 | 0 | 12 GB | 20-25 hrs |
| **Phase 3** | 250K | 20 | 100 hrs | 30 GB | 25-30 hrs |

---

## Quick Start Commands

### Step 1: Generate Synthetic Data (Phase 1)
```python
# In cascade.ipynb:
train_dataset = HybridCascadeDataset(
    num_samples=200000,
    use_realistic_channel=True,
    use_real_noise=False
)

# Train all stages
TRAIN_STAGE1 = True
TRAIN_EXPERTS = True
TRAIN_DECODER = True
# Run cells...
```

### Step 2: Collect Real Noise (Phase 2)
```bash
# Automated collection from KiwiSDR
cd modules/training
python scripts/collect_real_noise.py \
    --source kiwisdr \
    --host websdr.ewi.utwente.nl \
    --freq 14.100 \
    --duration 3600 \
    --output data/real_noise/

# Repeat for 10 hours total
```

### Step 3: Train with Real Noise
```python
# Re-run training with real noise enabled
train_dataset = HybridCascadeDataset(
    num_samples=200000,
    use_realistic_channel=True,
    use_real_noise=True,
    real_noise_dir='data/real_noise/'
)

# Fine-tune (load previous weights and continue)
```

---

## Validation Strategy

### During Training
```python
# Test on multiple conditions
evaluator = CascadeEvaluator(model)

# 1. Synthetic test set (should be 95%+)
synthetic_acc = evaluator.evaluate(synthetic_test_loader)

# 2. Real noise test set (target 92-95%)
real_noise_acc = evaluator.evaluate(real_noise_test_loader)

# 3. SNR sweep (ensure graceful degradation)
snr_results = evaluator.evaluate_snr_sweep(test_loader)
evaluator.plot_snr_performance(snr_results)

# 4. Channel robustness (test all propagation modes)
channel_results = evaluator.evaluate_channel_robustness(test_loader)
evaluator.plot_channel_robustness(channel_results)
```

### Post-Training
```python
# Over-the-air validation (if possible)
# 1. Set up TX/RX stations
# 2. Transmit known messages
# 3. Measure:
#    - Detection rate
#    - Decode accuracy
#    - BER/PER
#    - Minimum SNR
```

---

## Performance Expectations

### Accuracy vs Data Source

| Training Data | Synthetic Test | Real Noise Test | Real Signals |
|--------------|----------------|-----------------|--------------|
| Synthetic only | 98% | 75-80% | 60-70% |
| + Realistic channel | 96% | 85-90% | 75-85% |
| + Real noise | 95% | 92-95% | 85-90% |
| + Real fine-tune | 93% | 95-97% | 95-99% |

### SNR Operating Points (with Phase 2)

| SNR (dB) | Pattern | Frequency | Modulation | Overall |
|----------|---------|-----------|------------|---------|
| -15 | 75% | 40% | 30% | 50% |
| -10 | 90% | 70% | 60% | 75% |
| -5 | 95% | 85% | 80% | 87% |
| 0 | 98% | 92% | 90% | 93% |
| +5 | 99% | 96% | 95% | 97% |
| +10 | 99.5% | 98% | 98% | 98% |

---

## Troubleshooting

### "Model works on synthetic but fails on real data"
- Check if realistic_channel is enabled
- Collect more diverse real noise
- Verify real noise has similar spectral characteristics
- Check I/Q imbalance parameters match your hardware

### "Training takes too long"
- Reduce dataset size to 50K for quick iteration
- Use smaller batch size if GPU memory limited
- Use mixed precision (automatic in trainers)
- Train on cloud GPU (RTL 4090 = 1.5 hrs total)

### "Real noise collection fails"
- Try different KiwiSDR (some are offline)
- Use WebSDR manual recording (always available)
- Start with 1 hour minimum, expand later
- Check network connectivity

---

## Conclusion

**The hybrid approach gives best results**:
1. Start with realistic synthetic data (Phase 1)
2. Augment with real noise (Phase 2)
3. Optionally fine-tune on real signals (Phase 3)

**Minimum viable approach**: Phase 1 only (90%+ real accuracy)
**Recommended**: Phase 1 + 2 (95%+ real accuracy)
**Production**: All phases (97%+ real accuracy)

Total time investment: **20-30 hours** for production-ready model.
