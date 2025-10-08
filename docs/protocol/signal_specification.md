# CASCADE Signal Specification (V2)

This document defines the physical layer signal characteristics for CASCADE V2 that all implementations must follow for interoperability.

---

## Base Signal Parameters

### Symbol Timing

**Symbol duration:** 5ms (200 symbols/second)

**Rationale:**
- Matches HF multipath delay spread (1-5ms typical)
- Sufficient samples: 48000 Hz × 0.005s = 240 samples/symbol
- Compatible with standard sound cards
- Good balance of throughput vs. robustness

### Pattern Structure

**8 patterns total:**
- Universal (no pools)
- Nested lengths: 128, 256, 512, 1024, 2048 symbols
- Orthogonality: -21.19 dB @ 2048 symbols (proven)
- Kernel provides pattern ID (eliminates blind detection)

**Pattern definition:**
```python
# Each pattern is ternary sequence (3-FSK)
pattern = [0, 1, 2, 0, 2, 1, 0, 1, ...]  # 2048 symbols max

# Modulated as GMSK 3-FSK:
for symbol in pattern:
    if symbol == 0:
        transmit(tone_A)  # First tone of triple
    elif symbol == 1:
        transmit(tone_B)  # Second tone of triple
    else:  # symbol == 2
        transmit(tone_C)  # Third tone of triple
```

**Durations @ 200 sym/s:**
- 128 symbols: 0.64s
- 256 symbols: 1.28s
- 512 symbols: 2.56s
- 1024 symbols: 5.12s
- 2048 symbols: 10.24s

---

## Frequency Allocation

### 135-Tone Reference Grid

**Specification:**
```python
# 135 discrete tones
tones = [300 + i*20 for i in range(135)]
# [300, 320, 340, ..., 2960, 2980, 3000] Hz

Spacing: 20 Hz
Bandwidth: 2.7 kHz (300-3000 Hz, standard SSB)
```

### 45 Frequency Triples (3-FSK)

**Non-overlapping triples:**
```python
triples = [(i*3, i*3+1, i*3+2) for i in range(45)]

# Examples:
Triple 0:  Tones 0-2   (300-320-340 Hz)
Triple 1:  Tones 3-5   (360-380-400 Hz)
Triple 22: Tones 66-68 (1640-1660-1680 Hz)
...
Triple 44: Tones 132-134 (2940-2960-2980 Hz)
```

**Logical channels:**
```
8 patterns × 45 frequency triples = 360 total logical channels
```

**Multi-user access:**
- All users share 2.7 kHz bandwidth
- Separated by: pattern orthogonality + frequency triples + time offset
- Supports **45 concurrent users** (vs 67 with 2-FSK)
- **+18% network throughput** due to better FEC efficiency (rate-7/8 vs rate-1/2)

**Why 3-FSK wins:**
- **Frequency diversity:** If 1 tone fades → still 67% energy
- **SNR gain:** ~3-4 dB in frequency-selective fading
- **Higher per-user throughput:** Diversity combining enables rate-7/8 FEC
- **Net result:** Fewer users × faster transfers = more total network capacity

---

## Modulation Architecture

### Layer 1: Pattern (GMSK 3-FSK)

**GMSK (Gaussian Minimum Shift Keying):**
- Ternary pattern symbol selects tone A, B, or C from assigned triple
- BT = 0.3 (bandwidth-time product)
- Smooth Gaussian pulse shaping
- Constant envelope (linear amplifier friendly)
- Excellent spectral containment

**3-FSK structure:**
```
Symbol 0: Pattern symbol = 0 → Transmit tone A (e.g., 1300 Hz)
Symbol 1: Pattern symbol = 1 → Transmit tone B (e.g., 1320 Hz)
Symbol 2: Pattern symbol = 2 → Transmit tone C (e.g., 1340 Hz)
Symbol 3: Pattern symbol = 0 → Transmit tone A (1300 Hz)
...
```

**Frequency diversity benefit:**
- If tone B hits fading notch → tones A and C still provide 67% energy
- Pattern detection succeeds with only 2 of 3 tones
- ~3-4 dB SNR gain vs 2-FSK

### Layer 2: Data (Adaptive IQ on Pattern Tones)

**After pattern selects tone, modulate user data on that tone:**

**BPSK (SNR < 0 dB):**
- 1 bit/symbol
- Phase: 0° or 180°
- Most robust

**QPSK (SNR 0-10 dB):**
- 2 bits/symbol
- Phases: 0°, 90°, 180°, 270°
- Good balance

**8-PSK (SNR 10-20 dB):**
- 3 bits/symbol
- 8 phase positions
- Higher throughput

**16-APSK (SNR > 20 dB):**
- 4 bits/symbol
- 4+12 constellation (inner + outer rings)
- Maximum throughput

**Differential encoding:**
- Phase changes carry data (not absolute phase)
- Immune to frequency drift (±10 Hz typical)
- No pilot symbols needed

### Layer 3: Error Correction (Polar Codes)

**Applied at protocol layer:**

**Adaptive rates:**
```
SNR < 0 dB:    Polar 1/2  (2× overhead, strongest)
SNR 0-5 dB:    Polar 2/3  (1.5× overhead)
SNR 5-10 dB:   Polar 3/4  (1.33× overhead)
SNR 10-15 dB:  Polar 4/5  (1.25× overhead)
SNR 15-20 dB:  Polar 5/6  (1.2× overhead)
SNR > 20 dB:   Polar 7/8  (1.14× overhead, lightest)
```

**Rate negotiation:**
- Included in kernel (polar_rate field, 3 bits)
- Sender uses rate from receiver's RX kernel
- Adapts to measured link quality

---

## Throughput Calculations

### Single Pattern Examples

**512 symbols @ 200 sym/s = 2.56s:**

| Modulation | Polar Rate | Raw Bits | After FEC | Throughput |
|------------|-----------|----------|-----------|------------|
| BPSK | 1/2 | 512 | 256 | 100 bps |
| QPSK | 2/3 | 1024 | 683 | 267 bps |
| 8-PSK | 3/4 | 1536 | 1152 | 450 bps |
| 16-APSK | 5/6 | 2048 | 1707 | 667 bps |

**1024 symbols @ 200 sym/s = 5.12s:**

| Modulation | Polar Rate | Raw Bits | After FEC | Throughput |
|------------|-----------|----------|-----------|------------|
| BPSK | 1/2 | 1024 | 512 | 100 bps |
| QPSK | 2/3 | 2048 | 1365 | 267 bps |
| 8-PSK | 3/4 | 3072 | 2304 | 450 bps |
| 16-APSK | 5/6 | 4096 | 3413 | 667 bps |

### Multi-User Network

**40-45 active users @ +15 dB SNR:**
- Per user (1 pattern): ~200-300 bps typical
- Strong receiver (4 patterns): ~800-1200 bps
- Network capacity: Distributed across 536 logical channels

---

## Beacon Structure

**Pattern-based beacons (not frequency-reserved):**

**Content:**
- RX kernel: 28 bytes (pattern ID, frequency pair, modulation, polar rate, embedding)

**Transmission:**
```
Pattern length: 512 symbols
Modulation: BPSK (layer 2)
Polar rate: 1/2 (robust)
Duration: 2.56s @ 200 sym/s
Frequency: Any available pair (station selects)
Pattern: Any of 8 (station selects)
```

**Timing:**
- Transmitted only when active (calling CQ, in QSO, in net)
- Randomized interval: 30-60s
- Not transmitted when idle/listening

**Collision avoidance:**
- 67 frequency pairs
- Random timing
- <5% collision rate

---

## Kernel-Driven Coordination

### 28-Byte Kernel Structure

**RX Kernel (in beacons):**
```python
{
    # Discrete (4 bytes)
    'pattern_id': 3 bits,      # 0-7
    'frequency_pair': 7 bits,  # 0-66
    'modulation': 3 bits,      # BPSK/QPSK/8-PSK/16-APSK
    'polar_rate': 3 bits,      # 1/2 to 7/8
    'protocol_version': 2 bits,
    'model_version': 2 bits,
    'reserved': 12 bits,

    # Embedding (24 bytes)
    'embedding': 192 bits      # NN optimization hints
}
```

**TX Kernel (in RTS/messages):**
- Same structure as RX kernel
- Indicates current transmission state
- Used for anti-collision (other stations avoid this channel)

**See:** [Kernel Encoding Spec](../implementation/kernel_encoding_spec.md)

---

## Interoperability Requirements

**All CASCADE implementations must:**

1. **Support 8 patterns:** Use identical genetically-optimized pattern set
2. **200 symbols/second:** Fixed symbol rate (5ms per symbol)
3. **135-tone grid:** 300-3000 Hz, 20 Hz spacing, 67 pairs
4. **GMSK 2-FSK:** BT=0.3 for pattern modulation
5. **Adaptive data modulation:** BPSK, QPSK, 8-PSK, 16-APSK
6. **Polar codes:** Rates 1/2 to 7/8 at protocol layer
7. **Kernel format:** 28-byte structure (4B discrete + 24B embedding)
8. **Nested patterns:** Support 64-2048 symbol lengths

**Critical:** Patterns are protocol-defined (universal), modulation/polar rates are model/kernel-selected (adaptive).

---

## Signal Processing Requirements

### Receiver Capabilities

**Minimum:**
- Sample rate: 48 kHz (standard sound card)
- Bit depth: 16-bit minimum
- Frequency stability: ±50 Hz maximum drift
- Processing: Pattern correlation + adaptive demodulation

**Recommended:**
- Frequency stability: ±10 Hz (GPS-disciplined for >20 users)
- Processing: RPi4 or better for full 40-45 user capacity

### Drift Handling

**Frequency tracking:**
- Per-user offset estimation (FT8-style)
- ±50 Hz maximum supported
- Tighter ±10 Hz recommended for dense networks

**Differential encoding:**
- Immune to slow drift
- No absolute frequency reference needed
- Works with non-GPS receivers

---

## Capacity Analysis

### Shannon Limit (2.7 kHz bandwidth)

**@ +15 dB SNR:**
```
Shannon = 2700 × log₂(1 + 31.6) = ~13,500 bps (coded maximum)
```

**CASCADE achieves:**
- 40-45 users × ~250 bps avg = ~10,000 bps
- **~74% of Shannon limit**
- Losses from: protocol overhead, polar code overhead, imperfect orthogonality

### Throughput Scaling

**Per user @ QPSK + Polar 2/3:**
- 1 pattern: ~267 bps
- 2 patterns: ~533 bps
- 4 patterns: ~1067 bps (strong receiver)

**Network total @ 40 users:**
- ~10,000 bps aggregate
- Distributed via kernel coordination
- Graceful degradation with hardware limits

---

## Physical Layer Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Patterns** | 8 | Universal, kernel-selected |
| **Pattern lengths** | 64-2048 symbols | Nested extraction |
| **Symbol rate** | 200 sym/s | Fixed (5ms/symbol) |
| **Tone grid** | 135 tones | 300-3000 Hz, 20 Hz spacing |
| **Frequency pairs** | 67 pairs | 2-FSK, non-overlapping |
| **Pattern modulation** | GMSK 2-FSK | BT=0.3 |
| **Data modulation** | BPSK/QPSK/8-PSK/16-APSK | Adaptive |
| **Error correction** | Polar codes | Rates 1/2 to 7/8 |
| **Logical channels** | 536 | 8 patterns × 67 pairs |
| **Active users** | 40-45 | Simultaneous |
| **Kernel size** | 28 bytes | 4B discrete + 24B embedding |

---

## See Also

- **[Pattern Architecture](../model/pattern_architecture.md)** - 8-pattern system details
- **[Protocol Layer](README.md)** - RTS/CTS and coordination
- **[Kernel Encoding](../implementation/kernel_encoding_spec.md)** - 28-byte kernel structure
- **[Hardware Requirements](../deployment/hardware_requirements.md)** - RPi4 and deployment tiers
- **[Architecture Summary](../../architecture.md)** - Executive overview
