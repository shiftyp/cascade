# CASCADE Version Compatibility System

CASCADE v1 ensures interoperability through carefully designed frozen protocol elements while allowing model evolution.

## Design Philosophy

**Key principle**: Frozen protocol elements + evolvable model behavior

**Frozen forever** (ensures interoperability):
- 128 patterns (48 beacon + 80 message) with 2-FSK structure
- Pattern duration: 32 symbols @ 200 sym/s = 0.16s
- 135-tone reference grid (300-3000 Hz, 20 Hz spacing)
- 28-byte kernel structure (3 bytes discrete + 24 bytes embedding)
- Dual-layer encoding (pattern ID + adaptive data payload)

**Can evolve** (negotiated via kernel version):
- Model weights and architectures
- Encoder mutation strategies
- Decoder improvements
- Kernel embedding semantics
- Pro/anti-kernel weighting algorithms

## Version Number in Kernel

The 28-byte kernel includes version information in the discrete portion:

```python
# 28-byte kernel structure
class KernelStructure:
    # Discrete component (3 bytes = 24 bits)
    pattern_id: int       # 7 bits  - Which of 128 patterns
    frequency_pair: int   # 7 bits  - Which tone pair (0-74)
    modulation: int       # 3 bits  - BPSK/QPSK/8-PSK/16-APSK
    protocol_version: int # 4 bits  - Protocol compatibility (0-15)
    model_version: int    # 3 bits  - NN model generation (0-7)

    # Continuous embedding (24 bytes)
    embedding: np.array   # 48 dims × 4-bit quantization
```

## CASCADE v1.0 Specification

```python
v1_0 = {
    'patterns': '128 patterns (48 beacon + 80 message)',  # 2-FSK architecture
    'modulation': 'BPSK/QPSK/8-PSK/16-APSK',             # Adaptive based on SNR
    'kernel_size': 28,                                    # 3 bytes discrete + 24 bytes embedding
    'error_tolerance': 'Pattern recognition (37.5%)',     # QR code-like, no separate FEC
    'encoding': 'Dual-layer (pattern ID + data)',        # 15-39 bits total
    'constellation': 'differential_encoding',             # Drift-immune
    'sync': 'pattern_correlation',                        # No explicit sync needed
    'max_simultaneous': 45,                              # Active users at once
    'total_capacity': 1024                               # Via TDMA/FDMA reuse
}
```

## Model Evolution Within v1

While the protocol is frozen, models can evolve:

```python
def model_evolution_compatibility():
    """
    Newer models within v1 protocol can have improved:
    - Pattern recognition accuracy
    - Kernel generation quality
    - Noise resistance
    - Decoder performance

    But must maintain:
    - Same 128 patterns
    - Same 28-byte kernel format
    - Same dual-layer encoding
    - Same modulation options
    """

    # Model v1.0.0: Initial release
    model_v1_0_0 = {
        'decoder_accuracy': 0.85,
        'kernel_optimization': 'basic',
        'training_hours': 24000
    }

    # Model v1.0.1: Improved training
    model_v1_0_1 = {
        'decoder_accuracy': 0.90,  # Better
        'kernel_optimization': 'advanced',  # Better
        'training_hours': 36000,
        # But same protocol, same patterns, same kernel format
    }
```

## Kernel Version Negotiation

When stations exchange kernels:

```python
def negotiate_compatibility(my_kernel, their_kernel):
    """Ensure protocol compatibility"""

    my_protocol = my_kernel.protocol_version      # e.g., 0 (v1.0)
    their_protocol = their_kernel.protocol_version  # e.g., 0 (v1.0)

    if my_protocol != their_protocol:
        # Future: Handle protocol mismatch
        # For v1.0, this shouldn't happen
        log_warning(f"Protocol mismatch: {my_protocol} vs {their_protocol}")
        return None

    # Model versions can differ - that's fine
    my_model = my_kernel.model_version       # e.g., 0
    their_model = their_kernel.model_version  # e.g., 1

    # Both use same protocol, can communicate
    return {
        'compatible': True,
        'protocol': my_protocol,
        'my_model': my_model,
        'their_model': their_model
    }
```

## Hardware Tier Compatibility

Different hardware capabilities work within same protocol:

```python
def hardware_compatibility():
    """
    All hardware tiers use same v1 protocol
    Differences are in capability, not compatibility
    """

    # QRP station (Raspberry Pi + QMX)
    qrp_station = {
        'transmit_patterns': 1,      # 1 pattern at a time
        'modulation': 'BPSK',       # Low SNR
        'throughput': '94 bps'
    }

    # Modern station (PC + IC-7300)
    modern_station = {
        'transmit_patterns': 4,      # 4 patterns simultaneously
        'modulation': 'QPSK',       # Medium SNR
        'throughput': '575 bps'
    }

    # Premium station (Server + Flex 6600)
    premium_station = {
        'transmit_patterns': 8,      # 8 patterns simultaneously
        'modulation': '16-APSK',    # High SNR
        'throughput': '1950 bps'
    }

    # All three can communicate using v1 protocol
    # Throughput adapts to weakest link
```

## Future Protocol Evolution

When CASCADE eventually needs v2 (years from now):

```python
def future_v2_considerations():
    """
    IF a v2 is ever needed, it would:
    - Be trained to also operate in v1 mode
    - Detect v1 stations via protocol_version field
    - Fall back to v1 behavior when needed
    - Allow v1 and v2 stations to coexist

    But v1 is designed to last many years without changes.
    """
    pass
```

## Summary

CASCADE v1 achieves compatibility through:
1. **Frozen protocol**: 128 patterns, 28-byte kernels, dual-layer encoding
2. **Version fields**: 4-bit protocol + 3-bit model versions
3. **Adaptive operation**: Modulation and throughput adapt to conditions
4. **No traditional FEC**: Pattern recognition provides error tolerance
5. **Hardware agnostic**: All tiers use same protocol, different capabilities

The system is designed to operate for years without protocol changes, with improvements coming from better trained models rather than protocol modifications.