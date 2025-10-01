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
- **Pattern Complexity Expert** (~500K params, ~1ms): Adapts 64→16→4→2 pattern collapse (83-93% Shannon efficiency)
- **Spectrum Allocation Expert** (~800K params, ~2ms): Packs users efficiently in 2.5 kHz (85-95% utilization)

**Training Strategy**:
1. **Pass 1** - Random kernel training: Builds robustness to unknown conditions
2. **Pass 2** - Generated kernel optimization: Uses Pass 1 model for realistic hints
3. **Three-stage expert training**: Independent → conductor → joint fine-tuning

**Operational Performance**:
- Total inference: <10ms on [Raspberry Pi 4](#performance-targets)
- SNR range: -25 to +15 dB (40 dB dynamic range)
- [Multi-user capacity](experts.md#signal-expert-network): 1-50 simultaneous users
- [Shannon efficiency](experts.md#shannon-efficiency-targets): 83-93% across all conditions

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

A key CASCADE innovation is **continuous modulation within fixed patterns**. While the protocol defines 64 orthogonal patterns (fixed time-frequency sequences), the model continuously adapts constellation geometry within these patterns.

### Fixed vs Adaptive Components

**Fixed (Protocol Layer)**:
- 64 orthogonal patterns (tone sequences in time)
- Pattern structure: 32 symbols × 8 tones
- Orthogonality: <-30 dB cross-correlation
- Ensures interoperability across all CASCADE implementations

**Adaptive (Model Layer)**:
- Constellation point positions in IQ space
- Symbol timing (50ms ±20%)
- FEC rate (0.3-0.95)
- Power allocation per symbol

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

## Fixed Pattern Constellation

[64 orthogonal patterns](patterns.md) with [hierarchical clustering](patterns.md#hierarchical-clustering):
- **Level 0**: [64 patterns](patterns.md#pattern-design-principles) (6 bits/symbol)
- **Level 1**: 16 clusters (4 bits/symbol)
- **Level 2**: 4 clusters (2 bits/symbol)
- **Level 3**: 2 clusters (1 bit/symbol)

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

### Two-Pass Kernel Training
1. **Pass 1**: Random kernels for robustness
2. **Pass 2**: Use Pass 1 model to generate realistic hints

### Three-Stage Expert Training
1. **Stage 1**: Parallel independent expert training
2. **Stage 2**: Conductor training with frozen experts
3. **Stage 3**: Joint fine-tuning of entire system

Adjust based on results - if conductor struggles, allocate more Stage 2.

## Performance Targets

- **Inference**: <10ms on Raspberry Pi 4
- **Shannon Efficiency**: 83-93% across SNR range
- **Multi-User**: 1-50 simultaneous users
- **SNR Range**: -25 to +15 dB (40 dB dynamic range)
- **Pattern Orthogonality**: <-30 dB cross-correlation
- **Multipath Tolerance**: 10ms delay spread
- **Doppler Tolerance**: ±10 Hz