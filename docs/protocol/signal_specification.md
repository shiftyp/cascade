# CASCADE Signal Specification

This document defines the physical layer signal characteristics that all CASCADE implementations must follow for interoperability.

## Overview

CASCADE uses fixed orthogonal patterns with adaptive modulation to achieve near-Shannon efficiency across heterogeneous hardware deployments. The protocol layer defines 64 mathematically orthogonal time-frequency patterns, while the model layer adaptively modulates within these patterns based on channel conditions.

## Base Signal Parameters

### Symbol Timing

**Base symbol duration**: 50ms (20 symbols/second)

**Rationale**:
- Exceeds HF multipath delay spread (1-5ms) by 10× margin
- Allows filter settling time (5-10ms)
- Provides margin for Doppler spread (±2 Hz = 10% of 20 Hz symbol rate)
- Compatible with standard amateur radio sound cards (48 kHz sample rate)
- Sufficient samples per symbol: 48000 × 0.05 = 2,400 samples

**Model adaptation**: The model may stretch or compress symbols within ±20% (40-60ms) based on channel conditions, but 50ms is the nominal baseline.

### Pattern Structure

**Symbols per pattern**: 32 symbols
**Pattern duration**: 32 × 50ms = 1.6 seconds

**Pattern definition**:
Each of CASCADE's 256 patterns (64 beacon + 192 message) is defined as a sequence of tone indices over time:

```python
pattern_structure = {
    'pattern_id': int,           # 0-63
    'sequence': [int] * 32,      # 32 symbols, each 0-7 (tone index)
    'orthogonality': float       # <-30 dB vs all other patterns
}

# Example:
pattern_0 = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, ...]  # 32 tone indices
pattern_1 = [7, 6, 5, 4, 3, 2, 1, 0, 7, 6, ...]  # Different sequence
```

**Orthogonality requirement**: <-30 dB cross-correlation between all pattern pairs ensures multi-user separation.

**Pattern generation**: Zadoff-Chu sequences with computer optimization to achieve <-30 dB cross-correlation.

**Generation algorithm**:
1. Generate Zadoff-Chu base patterns (LTE-proven sequences from 1970s mathematics)
2. Computer-optimize via simulated annealing to exactly -30 dB
3. Store resulting patterns (256 total: 64 beacon + 192 message, deterministic)

See [Pattern Generation](pattern_generation.md) for detailed implementation.

### Modulation

**Base modulation**: 8-QAM constellation (8 points in IQ plane)

**Constellation points (ideal, high SNR)**:
```python
# 8-QAM constellation (3 bits per symbol)
constellation_8qam = [
    complex( 1.0,  1.0),  # 000
    complex( 1.0, -1.0),  # 001
    complex(-1.0,  1.0),  # 010
    complex(-1.0, -1.0),  # 011
    complex( 0.0,  1.5),  # 100
    complex( 0.0, -1.5),  # 101
    complex( 1.5,  0.0),  # 110
    complex(-1.5,  0.0),  # 111
]
```

**Model-driven adaptation**: The model continuously adapts constellation point positions based on:
- Channel SNR
- Interference from other users
- Multipath and fading conditions
- Hardware capabilities

**Constellation collapse hierarchy**:
- **High SNR (>10 dB)**: Full 8-QAM (3 bits/symbol)
- **Medium SNR (0-10 dB)**: Collapse toward QPSK (2 bits/symbol)
- **Low SNR (-10-0 dB)**: Collapse toward BPSK (1 bit/symbol)
- **Very low SNR (<-10 dB)**: Collapse to single tone, heavy repetition

The collapse is **continuous** (gradual point movement in IQ space), not discrete mode switching.

### Frequency Allocation

**Channel bandwidth**: 2.5 kHz per CASCADE channel (300-2800 Hz)
**Reference tone structure**: 70 discrete tones (frequency-hopping grid)
**Tone spacing**: 32 Hz (optimized for HF multipath and drift)

**Discrete tone grid:**
- Lower band: 35 tones (300-1388 Hz, indices 0-34)
- Beacon reservation: 150 Hz (1475-1625 Hz, centered at 1550 Hz)
- Upper band: 35 tones (1700-2788 Hz, indices 35-69)

**Frequency-hopping spread spectrum**: Patterns hop between discrete reference tones (FHSS), NOT continuous frequency modulation (not CSS). Each symbol transmits at exactly one reference tone frequency—no interpolation between tones. This is fundamentally different from LoRa's chirp spread spectrum (patent safe).

**Frequency shift rationale**: All frequencies shifted +300 Hz from baseband to avoid DC blocking in amateur radio transceivers. Most SSB rigs have AC-coupled audio (high-pass at 100-300 Hz) which would severely attenuate or eliminate a 0 Hz tone. The 300-2800 Hz allocation sits optimally within the SSB passband (typically 300-3000 Hz).

**Multi-user access**: All users transmit simultaneously within the same 2.5 kHz, separated by 4D orthogonal patterns (Time × Discrete Freq × I × Q).

```python
# 140+ users share the same 2.5 kHz (300-2800 Hz):
User 1: Pattern 5 hops among discrete tones [12, 34, 5, 18, ...]
User 2: Pattern 12 hops among discrete tones [45, 7, 29, 3, ...]
...
User 140: Pattern 58 hops among discrete tones [22, 68, 11, 50, ...]

# Separated by:
# - Different discrete tone sequences (frequency dimension)
# - Different IQ trajectories (I and Q dimensions)
# - Pattern orthogonality <-30 dB in 4D space
```


## Beacon Protocol (Center-Band Reservation)

CASCADE reserves 150 Hz in the center of the passband (1475-1625 Hz) for beacons, kernel exchange, and emergency coordination.

### Beacon Band Summary

**Frequency allocation:**
- **Reservation:** 1475-1625 Hz (150 Hz, centered at 1550 Hz)
- **Emergency alert:** 1550 Hz (single tone, BPSK, exact center of spectrum)
- **4-FSK beacons:** [1490, 1520, 1580, 1610] Hz (symmetric around emergency)
- **Patterns:** 64 beacon patterns (IDs 0-63) optimized for 4-FSK
- **Separation from messages:** 87 Hz guard (lower), 75 Hz guard (upper)

**Key features:**
- Emergency at exact center (1550 Hz = (300+2800)/2) for optimal HF propagation
- 30 Hz padding around emergency tone (isolation from 4-FSK)
- Beacon patterns use 4-tone alphabet (not 70-tone message patterns)
- Simple IQ complexity (λ max 0.3) for maximum robustness
- Supports 4 simultaneous emergencies via pattern separation
- 48 normal beacon patterns (anti-kernel resilient)

**See [Beacon Reservation](beacon_reservation.md) for complete specification.**

**See [Emergency Relay Network](emergency_relay_network.md) for emergency protocol details.**

## Pattern Storage

```python
# Two-tier pattern storage
beacon_pattern = {
    'id': int,                    # 0-63
    'freq_sequence': np.uint8[32],  # 32 bytes (tone indices 0-3 for 4-FSK)
    'iq_trajectory': complex64[32], # 256 bytes (single complexity level)
}

message_pattern = {
    'id': int,                    # 64-255
    'freq_sequence': np.uint8[32],  # 32 bytes (tone indices 0-69 for messages)
    'iq_trajectory': complex64[32], # 256 bytes (hierarchical complexity)
}

# Per pattern: 292 bytes (freq 32 + iq 256 + metadata 4)
# Total 256 patterns: ~75 KB (64 beacon + 192 message)

# See pattern_architecture.md for complete specification
```

## Interoperability Requirements

**All CASCADE implementations must:**

1. **Use identical patterns**: The 256 patterns (64 beacon + 192 message) are fixed and identical across all deployments
2. **Support variable modulation**: Decode 8-QAM, QPSK, BPSK constellations (model adapts continuously)
3. **Handle mixed modulations**: User A's 8-QAM must coexist with User B's BPSK (different SNR conditions)
4. **Maintain timing**: 50ms ±10ms symbol timing tolerance
5. **Preserve pattern orthogonality**: Cross-correlation must remain <-30 dB even as constellations adapt

**Critical**: Patterns are **protocol-defined** (fixed), constellations are **model-optimized** (variable). The fixed patterns ensure interoperability while adaptive constellations approach Shannon efficiency.

## Signal Processing Requirements

**Minimum receiver capabilities**:
- Sample rate: 48 kHz (standard amateur radio sound card)
- Bit depth: 16-bit (standard)
- Frequency stability: ±50 Hz maximum (GPS-locked preferred for >20 users)
- Processing: Pattern correlation + 8-QAM demodulation per 50ms
- Drift tracking: Per-user frequency offset estimation (FT8-style), used as separation feature

**Drift handling**:
- Model tracks ±50 Hz drift per user independently
- Clock drift used as "station fingerprint" for separation
- Training includes aggressive drift augmentation (each user ±50 Hz random offset)
- Tighter ±25 Hz tolerance recommended for >20 simultaneous users

**Sound card compatibility**:
- Buffer size: 128-2048 samples (supports 50ms symbols)
- Latency: 10-20ms typical (acceptable for 50ms symbols)
- Interface: USB, built-in, or SignaLink/Digirig

## Throughput Calculations

**Shannon capacity (physical limits):**
```
Bandwidth: 2.5 kHz
SNR @ +15 dB: Shannon = 2500 × log₂(1 + 31.6) = 12,570 bps coded maximum

IMPORTANT: Total throughput cannot exceed Shannon capacity regardless of number of users.
Pattern orthogonality enables efficient SHARING of this capacity, not multiplication.
```

**Single user, high SNR (+15 dB):**
```
Shannon limit: 12,570 bps coded (absolute maximum)
With rate 0.5 FEC: ~6,000 bps information throughput
With rate 0.8 FEC: ~10,000 bps information (less robust)

CASCADE achieves 90-95% Shannon efficiency via:
- Adaptive constellation (8-QAM → QPSK → BPSK)
- Pattern-based spreading across 8 tones
- Model-optimized encoding
```

**Multi-user (50 users, +15 dB):**
```
Total capacity: 12,570 bps coded (Shannon limit, shared among all users)
Average per user: ~250 bps coded, ~125 bps information (with rate 0.5 FEC)

Pattern orthogonality (<-30 dB):
- Enables 50+ users to coexist with minimal interference
- Model dynamically allocates capacity based on each user's needs
- Active users share the Shannon-limited capacity efficiently

Example allocation:
- 10 active users: ~1,250 bps coded each (~625 bps info)
- 25 active users: ~500 bps coded each (~250 bps info)
- 50 active users: ~250 bps coded each (~125 bps info)
```

**Hardware-limited scenarios** (e.g., Raspberry Pi 4 decoding 15 of 50 users):
```
Channel capacity: 12,570 bps total (all 50 users transmitting)
RPi decodes: 15 users (30% of traffic)
Throughput to this station: ~3,800 bps (15 users share proportionally)
Other 35 users: Not decoded (insufficient hardware capacity)

Note: Hardware limits # users decoded, not total channel capacity
Shannon efficiency: 90%+ at protocol level, 30% at this receiver (hardware-limited)
```

**Shannon capacity at various SNRs (2.5 kHz bandwidth):**
```
SNR    Shannon (coded)   90% Efficient   With Rate 0.5 FEC
+15 dB   12,570 bps        11,300 bps        5,650 bps info
+10 dB    9,150 bps         8,200 bps        4,100 bps info
 +5 dB    6,500 bps         5,800 bps        2,900 bps info
  0 dB    4,320 bps         3,900 bps        1,950 bps info
 -5 dB    2,740 bps         2,500 bps        1,250 bps info
-10 dB    1,625 bps         1,460 bps          730 bps info
-15 dB      885 bps           800 bps          400 bps info
-22 dB      315 bps           280 bps          140 bps info (FT8 level)
```

## See Also

- **[Hardware Requirements](../deployment/hardware_requirements.md)** - Deployment tiers and capabilities
- **[Protocol Overview](README.md)** - Protocol layer responsibilities
- **[Model Adaptation](../model/README.md)** - How model optimizes within protocol constraints
- **[Pattern Architecture](../model/pattern_architecture.md)** - 256 patterns (64 beacon + 192 message) specification
