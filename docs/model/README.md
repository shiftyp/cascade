# Model Layer - Continuous Optimization

The model layer handles all continuous optimization in CASCADE. These are parameters best determined through gradient descent and machine learning.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Responsibilities](#responsibilities)
3. [Expert Networks](#expert-networks)
4. [Conductor Network](#conductor-network)
5. [Fixed Pattern Constellation](#fixed-pattern-constellation)
6. [Frame Processing](#frame-processing)
7. [Pairwise Link Adaptation](#pairwise-link-adaptation)
8. [Training Strategy](#training-strategy)
9. [Performance Targets](#performance-targets)

## Executive Summary

CASCADE's model layer implements a **mixture-of-experts architecture** where five specialized neural networks handle different aspects of adaptive radio communication. A conductor network dynamically weights these experts based on current conditions, enabling optimal performance across diverse scenarios.

**Architecture**:
- **[Shared Encoder](shared_encoder.md)** (1024D features): Processes raw IQ samples, extracts universal features
- **[5 Expert Networks](experts.md)** (512D outputs each): Specialize in noise suppression, propagation compensation, multi-user separation, constellation adaptation, and spectrum allocation
- **[Conductor Network](conductor_details.md)**: Learns optimal expert weighting based on channel conditions
- **Decoder**: Combines weighted expert outputs to recover transmitted data

**Key Innovation**: The model decides **HOW, WHEN, and HOW MUCH** continuously:
- **HOW**: Error correction strength, pattern selection, constellation complexity
- **WHEN**: Fragment duration, transmission timing, ACK windows
- **HOW MUCH**: Redundancy factor, bandwidth allocation, power distribution

**Expert Specialization**:
- **Noise Expert** (~1M params, ~2ms): Suppresses QRN/QRM while preserving signal (15-20 dB improvement)
- **Propagation Expert** (~900K params, ~2.5ms): Compensates multipath, fading, Doppler (10-15 dB improvement)
- **Signal Expert** (~1.2M params, ~3ms): Separates 1-50 simultaneous users (>20 dB isolation)
- **Pattern Complexity Expert** (~500K params, ~1ms): Selects pattern pool based on multipath (128 patterns in hierarchical pools)
- **Spectrum Allocation Expert** (~800K params, ~2ms): Packs users efficiently in 2.5 kHz (85-95% utilization)

**Training Strategy**:
1. **Pass 1** - Random kernel training: Builds robustness to unknown conditions
2. **Pass 2** - Generated kernel optimization: Uses Pass 1 model for realistic hints
3. **Three-stage expert training**: Independent → conductor → joint fine-tuning

**Operational Performance**:
- Total inference: <8.5ms on [Raspberry Pi 4](#performance-targets)
- SNR range: -25 to +15 dB (40 dB dynamic range)
- [Multi-user capacity](experts.md#signal-expert-network): 1-50 simultaneous users
- Shannon efficiency: 78-85% (kernel-coordinated chaos: 78% initial, 85% steady state)

## Responsibilities

### HOW - Encoding Optimization
- Error correction strength
- Pattern selection within assigned pool
- Constellation collapse level (64→16→4→2)
- Kernel hint generation
- Pairwise link adaptation

### WHEN - Timing Optimization
- Fragment duration adaptation (0.5-5 seconds)
- Transmission scheduling
- ACK window detection
- Kernel hint timing
- Between-frame ACK opportunities

### HOW MUCH - Resource Optimization
- Redundancy factor (1.0-3.0)
- Bandwidth allocation per link
- Power distribution
- Processing allocation
- Pattern complexity selection

## Continuous Constellation Adaptation

A key CASCADE innovation is **hierarchical pattern organization**. The protocol defines 128 patterns (48 beacon + 80 message) organized by IQ complexity, with the model selecting from appropriate pools based on HF propagation.

### Fixed vs Adaptive Components

**Fixed (Protocol Layer)**:
- 128 orthogonal patterns (48 beacon + 80 message, each uses 4 from 78-tone grid)
- Hierarchical organization: Pattern ID indicates IQ complexity
- Pattern structure: 32 symbols, RS(32,20) aligned
- Orthogonality: <-37.5 dB cross-correlation achieved
- Ensures interoperability across all CASCADE implementations

**Adaptive (Model Layer)**:
- Constellation point positions in IQ space (64-QAM → BPSK collapse)
- Symbol timing (50ms ±20%)
- 4-tone selection from 78-tone grid (adaptive ±3)
- IQ complexity parameter λ (0.0-0.9)
- Power allocation per symbol

**Built-in FEC (Protocol Layer)**:
- RS(32,20) pattern structure provides error correction
- No separate FEC layer needed
- 37.5% erasure tolerance (12 of 32 symbols)
- Aligned protection for pattern ID and data

### Constellation Collapse Continuum

The model adapts constellation geometry continuously based on channel conditions:

```python
# High SNR station (Pattern 5):
constellation = spread_points_8qam()  # Full 3 bits/symbol
iq_points = [complex(1.0, 1.0), complex(1.0, -1.0), ...]  # Widely spaced

# Low SNR station (Pattern 12) - same frequency, same time:
constellation = collapse_to_bpsk()  # ~1 bit/symbol effective
iq_points = [complex(0.95, 0.05), complex(-0.95, -0.05), ...]  # Narrow spacing

# Both use their fixed assigned patterns (orthogonal separation)
# But different constellation geometries (model optimization)
# Model separates them via pattern correlation + constellation analysis
```

**Continuous adaptation** (not discrete modes):
- 8-QAM gradually morphs into QPSK, then BPSK
- Point positions optimize for current interference
- Model learns optimal geometry for each condition

### Multi-User Coexistence

All users transmit simultaneously in the same 2.5 kHz bandwidth:

**User separation mechanism:**
1. **Pattern orthogonality** (primary): <-30 dB isolation from fixed sequences
2. **Constellation diversity** (secondary): Model adapts to avoid interference
3. **Temporal adaptation** (tertiary): Symbol timing micro-adjustments

**Model coordination** (implicit, no explicit signaling):
- Each station's model observes the channel
- Adapts constellation to minimize interference with detected signals
- Emergent cooperation through shared training objective

See [signal_specification.md](../protocol/signal_specification.md) for detailed protocol parameters.

## Expert Networks

CASCADE employs five specialized expert networks - see [experts.md](experts.md) for detailed specifications:
- **Noise Expert** - QRN/QRM suppression
- **Signal Expert** - Multi-user separation
- **Propagation Expert** - Channel compensation
- **Pattern Complexity Expert** - Constellation adaptation
- **Spectrum Allocation Expert** - Frequency optimization

### Expert Summary

| Expert | Purpose | Parameters | Latency |
|--------|---------|------------|---------|
| Noise | Suppress interference | ~1M | ~2ms |
| Signal | Separate users | ~1.2M | ~3ms |
| Propagation | Channel equalization | ~900K | ~2.5ms |
| Pattern Complexity | SNR adaptation | ~500K | ~1ms |
| Spectrum Allocation | Frequency packing | ~800K | ~2ms |

## Conductor Network

Advanced coordination strategies - see [conductor_details.md](conductor_details.md):
- Attention-based weighting
- Hierarchical conductor
- Conditional networks
- Learned gating
- Temporal adaptation

### Weight Patterns by Condition
- **High SNR**: Complexity expert dominates
- **Low SNR**: Noise expert dominates
- **Multipath**: Propagation expert dominates
- **Multi-user**: Signal expert dominates

## 4D Pattern Architecture

CASCADE uses [128 orthogonal 4D patterns](pattern_architecture.md) (48 beacon + 80 message) combining discrete frequency-hopping with continuous IQ modulation:
- **Dimensions**: Time × Tone Selection (4 from 78) × Continuous I × Continuous Q
- **IQ organization**: Hierarchical pools (pattern ID indicates baked-in complexity)
- **Frequency**: Discrete hops (FHSS, not CSS - patent safe)
- **IQ**: Continuous trajectories (HF-realistic λ=0.3-0.6 typical, smooth adaptation)
- **Orthogonality**: <-37.5 dB in 4D space achieved (128-pattern chaos)

See [Pattern Architecture](pattern_architecture.md) for comprehensive specification.

## Frame Processing

### Adaptive Fragmentation
- Model receives constant frame size
- Decides stretch/compression factor (0.5x-10x)
- Creates natural fragments via sliding window
- Streams to [protocol layer](../protocol/README.md)

### Kernel Generation
- **Bidirectional optimization**: Receiver generates, transmitter uses
- **64-bit hints**: Serve as decoder config AND frame ID
- **Sparse inclusion**: Only 1% of symbols
- **Pairwise storage**: Each link has unique hints

## Pairwise Link Adaptation

Key insight: SNR is pairwise, not per-station:
- Each transmission adapts to specific destination
- Bandwidth allocation varies by link quality
- Pattern assignment based on measured SNR
- Continuous learning from ACKs

Example:
```
Station A → Station B: +10 dB (uses 50 Hz)
Station A → Station C: -5 dB (uses 150 Hz)
Station B → Station A: +8 dB (different from A→B!)
```

## Training Strategy

### Data Generation Approach

**Synthetic CASCADE signals + Real HF propagation:**

CASCADE training uses **synthetic signal generation** combined with **real-world HF propagation characteristics** collected from the KiwiSDR network (150,000-300,000 hours of recordings).

```python
def generate_training_batch():
    """Generate training scenario with synthetic CASCADE + real propagation"""

    # 1. Generate synthetic CASCADE traffic
    num_users = random.randint(5, 50)
    cascade_signals = []

    for user_id in range(num_users):
        signal = generate_cascade_transmission(
            pattern=random.choice(range(16, 64)),  # Normal beacon patterns
            modulation=random.choice(['8-QAM', 'QPSK', 'BPSK']),
            snr_db=random.uniform(-25, 15),
            clock_drift_hz=random.uniform(-50, 50),  # Per-user drift
            start_time_us=random.randint(0, 5000000)  # Asynchronous
        )
        cascade_signals.append(signal)

    # 2. Mix all users (overlap in frequency and time)
    mixed_cascade = sum_signals_with_alignment(cascade_signals)

    # 3. Apply REAL HF channel effects from KiwiSDR recordings
    hf_channel = load_random_hf_channel_model(
        # Extracted from real recordings:
        multipath_impulse_response,  # 1-10ms delay spread
        doppler_spread,               # ±2 Hz typical
        fading_coefficients,          # Rayleigh/Rician
        ionospheric_flutter           # Real measured
    )

    propagated = apply_hf_channel(mixed_cascade, hf_channel)

    # 4. Add REAL noise samples from KiwiSDR
    noise_sample = load_random_noise_segment(
        duration=len(propagated),
        source='kiwisdr_recordings'  # Real atmospheric/man-made noise
    )

    final_signal = propagated + noise_sample

    return final_signal, ground_truth_users

# Training uses 100% synthetic CASCADE, 100% real propagation/noise
```

**Why this approach:**
- ✅ CASCADE protocol doesn't exist yet (can't record real CASCADE traffic)
- ✅ HF propagation physics is universal (applies to any signal)
- ✅ Real noise characteristics crucial (QRN, QRM, atmospheric)
- ✅ Drift/timing/multipath effects are identical for real vs synthetic

### Clock Drift as Separation Feature

**Per-user drift tracking:**

The model learns to use **clock drift as a station fingerprint** for multi-user separation, inspired by FT8's drift correction but extended to 50+ simultaneous users.

```python
def train_with_drift_augmentation():
    """Train model to track and exploit per-user drift"""

    # Each user gets independent clock error
    user_drifts = [random.uniform(-50, 50) for _ in range(num_users)]

    for user_id, drift_hz in enumerate(user_drifts):
        # Drift remains constant per user across entire transmission
        user_signal = generate_user_with_drift(
            base_freq=300 + pattern_offset,
            drift_hz=drift_hz,           # -50 to +50 Hz
            duration=5.0                  # Seconds
        )

        # Over 5 seconds at 14 MHz:
        # ±50 Hz drift accumulates significant phase rotation
        # Model learns this as unique per-user signature

    # Ground truth includes drift per user
    ground_truth = {
        'user_0': {'pattern': 5, 'data': bytes, 'drift_hz': +30},
        'user_1': {'pattern': 12, 'data': bytes, 'drift_hz': -15},
        # ...
    }
```

**Model output includes drift estimates:**
```python
class SignalExpert:
    def forward(self, iq_samples):
        # Returns drift alongside decoded data
        return {
            'users': [
                {
                    'pattern': 5,
                    'data': bytes,
                    'drift_hz': +29.7,  # Estimated drift (vs +30 ground truth)
                    'snr_db': -12
                },
                # ...
            ]
        }
```

**Training targets:**
- Pattern separation: <-30 dB cross-correlation (primary)
- Drift as secondary feature: ±50 Hz range, 0.5 Hz accuracy
- Combined: Enables 50+ user separation even with partial pattern overlap

**Drift tolerance:**
- <20 users: ±50 Hz per user (GPS not required)
- ≥20 users: ±25 Hz recommended (GPS-locked preferred)
- Model trained on full ±50 Hz to handle worst case

### Two-Pass Kernel Training
1. **Pass 1**: Random kernels for robustness
2. **Pass 2**: Use Pass 1 model to generate realistic hints

### Three-Stage Expert Training
1. **Stage 1**: Parallel independent expert training
2. **Stage 2**: Conductor training with frozen experts
3. **Stage 3**: Joint fine-tuning of entire system

Adjust based on results - if conductor struggles, allocate more Stage 2.

### Training Data Pipeline

**Data sources:**
- **Synthetic CASCADE signals**: 100% generated (protocol-compliant patterns)
- **Real HF propagation**: Extracted from 150k-300k hours KiwiSDR recordings
- **Real noise**: QRN (atmospheric), QRM (interference), solar conditions
- **Channel models**: Watterson, ITU-R P.533 with measured parameters

**Augmentation:**
- Multi-user count: 5-50 simultaneous (uniform distribution)
- SNR per user: -25 to +15 dB (weighted toward weak signals)
- Clock drift per user: ±50 Hz independent (enables fingerprinting)
- Start time offsets: 0-5s microsecond-resolution (asynchronous)
- Propagation conditions: All solar cycle phases, all bands, all paths

## Performance Targets

- **Inference**: <8.5ms on Raspberry Pi 4
- **Shannon Efficiency**: 78-85% (kernel-driven emergent coordination)
- **Multi-User**: 1-50 simultaneous users
- **SNR Range**: -25 to +15 dB (40 dB dynamic range)
- **Pattern Orthogonality**: <-30 dB cross-correlation
- **Multipath Tolerance**: 10ms delay spread
- **Doppler Tolerance**: ±10 Hz

## Kernel Generation as Learned Behavior

### RX-Optimized Kernels

Kernels are **learned outputs** of the pattern expert network, not separate from the model. Each station generates a 64-bit kernel for their own receiver, which others use when transmitting to that station.

**Kernel generation process**:
- Pattern expert (512-D output) encodes receiver's preferences and capabilities
- 64-bit kernel is lossy compression of this 512-D state
- Kernel includes: hardware tier, FEC preference, constellation limits, capacity
- Model learns optimal kernel generation from telemetry feedback

**Training on kernel effectiveness**:

Telemetry from deployed radios reveals which kernels actually improve decode success. TX telemetry includes the RX kernel generated (512-D neural state), while RX telemetry shows whether that kernel helped encoding. This TX/RX correlation provides ground truth for optimizing kernel generation.

The model learns:
- Which 64-bit kernel configurations work for given hardware/conditions
- How to predict kernel lifetime (estimated_valid_seconds field)
- When receivers need kernel updates (via ACK feedback)
- How to adapt kernels based on antikernel feedback

**Kernel as compressed representation**:
- 64-bit kernel: Discrete hint for decoder configuration
- 512-D pattern expert: Continuous representation of optimal pattern selection
- 3581-D full state: Complete context for kernel generation decisions
- Training uses 512-D (continuous, gradient-friendly) not 64-bit (discrete)

See [telemetry_research.md](../../telemetry_research.md#telemetry-integration-with-kernels-and-antikernels) for kernel/antikernel telemetry integration and training strategies.

## Continuous Improvement via Telemetry

CASCADE improves continuously through three complementary mechanisms:

**Monthly fine-tuning**:
- Updates model with recent telemetry (10K-50K hours)
- Preserves existing knowledge
- 1-5% improvement per update
- Cost: $150-200 per month

**Annual retraining**:
- Full retrain from scratch with combined dataset (100K+ hours)
- Eliminates inherited geographic bias
- 5-15% improvement
- Cost: $2,500-3,500 per year

**Real-time adaptation**:
- Meta-learning (MAML) during active QSOs
- 5-15% improvement during specific conversations
- Reverts after QSO ends
- Hardware-dependent (RPi4: kernel only, Coral: +MAML, x86: full online)

These three mechanisms work together: base model improves monthly/annually (global performance), real-time adaptation provides QSO-specific boost (immediate benefit).

See [Continuous Improvement](../training/continuous_improvement.md) for complete federated learning and model update strategies.

---

## See Also

### Core Model Documentation
- **[Pattern Architecture](pattern_architecture.md)** - 128 patterns with chaos mode and frequency reuse
- **[Expert Networks](experts.md)** - 5 specialized networks with detailed architectures
- **[TFIQ Dimensions](tfiq_dimensions.md)** - 4D user separation in Time × Frequency × I × Q
- **[Shared Encoder](shared_encoder.md)** - Universal feature extraction (1024D)
- **[Conductor Details](conductor_details.md)** - Attention-based expert coordination

### Protocol & Training
- **[Protocol Layer](../protocol/README.md)** - Discrete decisions (WHO/WHETHER/WHAT)
- **[Training Strategy](../training/README.md)** - Real-world data pipeline and training phases
- **[Telemetry Research](../../telemetry_research.md)** - Kernel effectiveness and continuous improvement

### Implementation
- **[Hardware Requirements](../deployment/hardware_requirements.md)** - RPi4/Coral/Desktop/GPU tiers
- **[Architecture Overview](../../architecture.md)** - Executive summary and performance targets
- **[Signal Specification](../protocol/signal_specification.md)** - Physical layer interoperability