# CASCADE Training Strategy

CASCADE's training emphasizes learning from real-world HF propagation conditions while maintaining privacy and enabling continuous improvement. This overview provides navigation to detailed training documentation.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Training Pipeline Phases](#training-pipeline-phases)
3. [Two-Pass Kernel Training](#two-pass-kernel-training)
4. [Three-Stage Expert Training](#three-stage-expert-training)
5. [Detailed Documentation](#detailed-documentation)

---

## Executive Summary

**Core Challenge:** Train an adaptive HF radio system without existing CASCADE deployments.

**Solution:** Three-stage knowledge transfer:
1. **Data Collection** (Months 1-18): 200K-300K hours real HF propagation from 800-1100 SDRs worldwide
2. **Knowledge Compression** (Month 19): Train embedding VAEs to compress 35-75TB → 15-25GB
3. **CASCADE Training** (Months 20-21): Train on synthetic signals + real embeddings

**Key Innovation:** Embeddings used only during training. At inference, CASCADE processes raw IQ end-to-end with no embedding computation.

**Training Scale:**
- **Data:** 200K-300K hours raw → 3-5TB curated → 15-25GB embeddings
- **Duration:** 18 months collection + 3 months training
- **Compute:** 4× RTX 4090 GPUs for 1-2 weeks
- **Storage:** 35-75TB cold storage, 3-5TB NVMe, 15-25GB embeddings

**Training Principles:**
- Real data only (no synthetic propagation models)
- Diversity-biased sampling (rare events get 100× weight)
- Natural correlation preservation (QRN + propagation stay paired)
- Privacy-first (callsigns hashed, no message content)

---

## Training Pipeline Phases

### Month 1-18: Data Collection

**Objective:** Collect diverse HF propagation data from global SDR network

**Details:** See [Data Pipeline](data_pipeline.md) for comprehensive specification

**Key points:**
- 800-1100 cooperating KiwiSDR/WebSDR owners
- 200K-300K hours total (target: 4K-6K hours per band)
- Geographic diversity: All continents, 40/40/20 hemisphere distribution
- Solar cycle coverage: Mix of solar minimum and approaching maximum
- Storage: 35-75TB with FLAC compression

**Outputs:**
- Raw IQ recordings (35-75TB)
- FT8/WSPR decodes (propagation ground truth)
- Space weather metadata (K-index, SFI, A-index)

### Month 19: Embedding Model Training

**Objective:** Compress propagation data into learned representations

**Details:** See [Embedding Models](embedding_models.md) for VAE architectures

**Key points:**
- Train two VAEs: QRN encoder (atmospheric noise), Propagation encoder (channel characteristics)
- Compress 35-75TB → 15-25GB embeddings (2000-3000× compression)
- Preserve diversity: Rare events maintained in embedding space
- Validation: Embeddings reconstruct propagation statistics accurately

**Outputs:**
- Trained QRN VAE (~50MB model)
- Trained Propagation VAE (~60MB model)
- 15-25GB embedding dataset
- Embedding analytics (clustering, anomaly detection)

### Months 20-21: CASCADE Model Training

**Objective:** Train CASCADE neural networks using real embeddings

**Details:** See sections below and [Smoothness Objectives](smoothness.md)

**Key points:**
- Two-pass kernel training (Pass 1: robustness, Pass 2: optimization)
- Three-stage expert training (independent → conductor → joint)
- Synthetic signals + real embedding augmentation
- Multi-user scenarios (1-80 simultaneous stations)

**Outputs:**
- Trained CASCADE model (~10MB INT8)
- Validation on held-out data
- Ready for deployment

---

## Two-Pass Kernel Training

CASCADE trains in two passes to balance robustness and optimization:

### Pass 1: Random Kernel Training (Robustness)

**Goal:** Model learns to decode WITHOUT kernel hints

```python
def train_pass1_no_kernels(model, data):
    """
    Train with random kernels (noise)
    Forces model to develop robust baseline decode
    """

    for batch in data:
        # Generate random kernel (meaningless)
        random_kernel = np.random.randint(0, 2**64)

        # Model must decode without kernel help
        decoded = model.decode(batch.signal, kernel=random_kernel)

        # Loss: How well can model decode WITHOUT hints?
        loss = cross_entropy(decoded, batch.ground_truth)
        optimize(loss)

# Result: Model learns robust baseline (doesn't rely on kernels)
```

### Pass 2: Generated Kernel Optimization

**Goal:** Model learns to USE kernel hints for fine-tuning

```python
def train_pass2_with_kernels(model, data):
    """
    Use Pass 1 model to generate realistic kernels
    Train Pass 2 model to optimize using these hints
    """

    pass1_model = load_trained_pass1()

    for batch in data:
        # Generate realistic kernel using Pass 1 model
        kernel = pass1_model.generate_kernel(
            batch.signal,
            receiver_hardware=batch.rx_hardware
        )

        # Train Pass 2 model to use kernel
        decoded = model.decode(batch.signal, kernel=kernel)

        loss = cross_entropy(decoded, batch.ground_truth)
        optimize(loss)

# Result: Model learns to optimize decode using kernel hints
#         But doesn't REQUIRE kernels (Pass 1 baseline preserved)
```

**Why two passes?**
- Pass 1: Robustness (works without kernels)
- Pass 2: Optimization (uses kernels when available)
- Deployment: Graceful degradation (no kernel → Pass 1 performance)

---

## Three-Stage Expert Training

CASCADE's expert networks train in three stages:

### Stage 1: Independent Expert Training

Each expert trains independently on its specialty:

```python
# Noise Expert: Train on QRN suppression
noise_expert.train(noisy_signals, clean_signals)

# Propagation Expert: Train on channel equalization
propagation_expert.train(faded_signals, original_signals)

# Signal Expert: Train on multi-user separation
signal_expert.train(mixed_users, individual_users)

# Pattern Complexity Expert: Train on SNR adaptation
pattern_expert.train(varying_snr, optimal_complexity)

# Spectrum Allocation Expert: Train on frequency packing
spectrum_expert.train(interference, optimal_allocation)

# Duration: 1 week per expert (parallel training)
```

### Stage 2: Conductor Training

Conductor learns optimal expert weighting:

```python
# Freeze expert weights
for expert in experts:
    expert.requires_grad = False

# Train conductor only
for batch in data:
    # Conductor predicts weights
    weights = conductor(batch.features)

    # Weighted expert combination
    output = sum(w * expert(batch.signal) for w, expert in zip(weights, experts))

    loss = decode_loss(output, batch.target)
    optimize(conductor.parameters(), loss)

# Duration: 3 days
```

### Stage 3: Joint Fine-Tuning

All networks train together:

```python
# Unfreeze all parameters
for expert in experts:
    expert.requires_grad = True
conductor.requires_grad = True

# Joint optimization
for batch in data:
    output = full_model(batch.signal)
    loss = decode_loss(output, batch.target)
    optimize(all_parameters(), loss)

# Duration: 1 week
```

**Total training time:** ~6 weeks (experts parallel) + 3 days (conductor) + 1 week (joint) ≈ 7-8 weeks

---

## Detailed Documentation

### Comprehensive Guides

- **[Data Pipeline](data_pipeline.md)** - Complete data collection and curation process
  - SDR network coordination (800-1100 sources)
  - Geographic diversity strategy (40/40/20 hemispheres)
  - Solar minimum boost (3× weight for rare conditions)
  - Storage architecture (35-75TB → 3-5TB curated)
  - Correlation preservation (QRN + propagation paired)

- **[Embedding Models](embedding_models.md)** - VAE architectures and compression
  - QRN VAE architecture (~15M parameters)
  - Propagation VAE architecture (~18M parameters)
  - Embedding analytics (clustering, anomalies)
  - Compression validation (2000-3000× reduction)

- **[Smoothness Objectives](smoothness.md)** - Continuous adaptation training
  - SNR transition smoothness
  - Frequency shift gradual changes
  - IQ trajectory continuity
  - Pattern complexity smooth collapse

- **[Continuous Improvement](continuous_improvement.md)** - Post-deployment learning
  - Privacy-preserving telemetry
  - Federated learning (differential privacy ε=1.0)
  - Byzantine-robust aggregation
  - Model updates via secure multi-party computation

- **[Long-Term Roadmap](long_term_roadmap.md)** - Multi-decadal vision
  - Solar cycle 26-28 coverage (2025-2040)
  - Climate change propagation adaptation
  - Scientific data legacy
  - Community-driven evolution

### Quick References

**Data sources:**
- KiwiSDR network: 800-1100 cooperating owners
- WebSDR network: 50-100 high-quality receivers
- NOAA space weather: Real-time K-index, SFI, X-ray flux
- FT8/WSPR decodes: Propagation ground truth

**Training augmentation:**
- Real atmospheric noise (QRN VAE embeddings)
- Real propagation (channel VAE embeddings)
- Synthetic multi-user (1-80 simultaneous stations)
- Interference simulation (QRM, powerline, radar)

**Validation metrics:**
- Decode accuracy across SNR range (-28 to +15 dB)
- Multi-user separation quality (1-140 users)
- Emergency beacon detection (100% at -28 dB)
- Shannon efficiency (target: 50-60%)

---

## Training Infrastructure

**Hardware requirements:**
- GPU cluster: 4× RTX 4090 (or equivalent)
- RAM: 256GB system memory
- Storage: 100TB raw + 10TB NVMe SSD
- Network: 10Gbps for SDR data ingestion

**Software stack:**
- PyTorch 2.0+ (neural network training)
- NumPy, SciPy (signal processing)
- Librosa, torchaudio (audio processing)
- PostgreSQL (metadata)
- Tigris S3 (cold storage)

**Training duration:**
- Data collection: 18 months (ongoing, distributed)
- Embedding training: 2 weeks (GPU)
- CASCADE training: 1-2 weeks (GPU cluster)
- Validation: 1 week
- **Total:** 18 months + 1 month processing

---

## See Also

### Core Training Documentation
- **[Data Pipeline](data_pipeline.md)** - Comprehensive data collection and curation
- **[Embedding Models](embedding_models.md)** - VAE architectures and compression
- **[Smoothness Objectives](smoothness.md)** - Continuous adaptation training
- **[Continuous Improvement](continuous_improvement.md)** - Post-deployment federated learning

### Related Documentation
- **[Pattern Architecture](../model/pattern_architecture.md)** - 4D patterns model is trained on
- **[Privacy](../privacy.md)** - Privacy-preserving data collection and telemetry
- **[Model README](../model/README.md)** - Overall model architecture
- **[TFIQ Dimensions](../model/tfiq_dimensions.md)** - Multi-user separation in 4D space

### Data Collection Module
- **[Data Module](../../modules/data/README.md)** - Implementation of data collection system
- **[KiwiSDR Integration](../../modules/data/cascade_collector/collectors/kiwi_client.py)** - SDR client implementation
