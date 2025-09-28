# Model Layer - Continuous Optimization

The model layer handles all continuous optimization in CASCADE. These are parameters best determined through gradient descent and machine learning.

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

## Expert Networks

Detailed specifications for each expert:
- [Noise Expert](experts/noise_expert.md) - QRN/QRM suppression
- [Signal Expert](experts/signal_expert.md) - Multi-user separation
- [Propagation Expert](experts/propagation_expert.md) - Channel compensation
- [Pattern Complexity Expert](experts/pattern_complexity_expert.md) - Constellation adaptation
- [Spectrum Allocation Expert](experts/spectrum_allocation_expert.md) - Frequency optimization

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

64 orthogonal patterns with hierarchical clustering - see [patterns.md](patterns.md):
- **Level 0**: 64 patterns (6 bits/symbol)
- **Level 1**: 16 clusters (4 bits/symbol)
- **Level 2**: 4 clusters (2 bits/symbol)
- **Level 3**: 2 clusters (1 bit/symbol)

## Frame Processing

### Adaptive Fragmentation
- Model receives constant frame size
- Decides stretch/compression factor (0.5x-10x)
- Creates natural fragments via sliding window
- Streams to protocol layer

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