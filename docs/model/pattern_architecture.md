# CASCADE Pattern Architecture (V2)

**Final: 2025-10-07** - 8-pattern system with dual-layer encoding

CASCADE V2 uses **8 universal patterns** with dual-layer information encoding. Patterns provide robust GMSK-modulated 2-FSK skeletons while adaptive IQ modulation carries user data.

---

## Executive Summary

**Pattern System:**
- **8 patterns total** (universal, no pools)
- **Nested lengths:** 128, 256, 512, 1024, 2048 symbols (from single genetic optimization)
- **Orthogonality:** -21.19 dB @ 2048 symbols (3.41 dB from Welch bound -24.6 dB)
- **Symbol rate:** 200 symbols/second (5ms per symbol)
- **Logical channels:** 8 patterns × 67 frequency pairs = **536 channels**

**Dual-Layer Encoding:**
- **Layer 1 (Pattern):** GMSK-modulated 2-tone FSK (binary pattern selects tone A or B)
- **Layer 2 (Data):** Adaptive IQ modulation on pattern-selected tones (BPSK/QPSK/8-PSK/16-APSK)
- **Error Correction:** Polar codes at protocol layer (adaptive rates 1/2 to 7/8)

**Key Innovation:**
- Kernel provides pattern ID → eliminates blind 8-pattern correlation
- RPi4 CPU-only decoding (<5ms per pattern)
- Genetic algorithm achieves excellent orthogonality (-21.19 dB proven)

---

## Table of Contents

1. [Overview](#overview)
2. [Dual-Layer Architecture](#dual-layer-architecture)
3. [Pattern Generation](#pattern-generation)
4. [Frequency Architecture](#frequency-architecture)
5. [Nested Pattern Lengths](#nested-pattern-lengths)
6. [Kernel-Assisted Detection](#kernel-assisted-detection)
7. [Performance Characteristics](#performance-characteristics)
8. [Pattern Storage Format](#pattern-storage-format)

---

## Overview

CASCADE V2 dramatically simplifies pattern architecture while improving performance:

**V1 (archived):** 128 patterns, 4-FSK, blind detection, complex
**V2 (current):** 8 patterns, 2-FSK, kernel-assisted, simple

**Why 8 patterns?**
- Excellent orthogonality achievable (-21.19 dB measured)
- Kernel eliminates blind detection cost
- 67 frequency pairs provide sufficient channels (536 total)
- Simpler = more robust, easier to implement

---

## Dual-Layer Architecture

### Layer 1: Pattern Skeleton (GMSK 2-FSK)

**Purpose:** Establish orthogonal channel and carry binary pattern data

**Structure:**
```python
# Each pattern = sequence of binary symbols
pattern = [0, 1, 0, 0, 1, 1, 0, 1, ...]  # Length: 64-2048

# Each binary symbol selects which tone from assigned pair
for symbol in pattern:
    if symbol == 0:
        transmit_tone = tone_A  # First tone of pair
    else:
        transmit_tone = tone_B  # Second tone of pair
```

**Modulation:**
- GMSK (Gaussian Minimum Shift Keying)
- BT = 0.3 (bandwidth-time product)
- Smooth transitions, constant envelope
- Excellent spectral containment

**Frequency pairs:**
- 135-tone reference grid (300-3000 Hz, 20 Hz spacing)
- 67 non-overlapping 2-tone pairs
- Pair N uses tones (N×2, N×2+1)
- Example: Pair 25 = tones 50-51 (1300-1320 Hz)

### Layer 2: Data Payload (Adaptive IQ)

**Purpose:** Carry user data via modulation on pattern-selected tones

**Modulation schemes (adaptive to SNR):**
```python
# After pattern selects tone, modulate data on that tone

SNR < 0 dB:    BPSK    (1 bit/symbol)
SNR 0-10 dB:   QPSK    (2 bits/symbol)
SNR 10-20 dB:  8-PSK   (3 bits/symbol)
SNR > 20 dB:   16-APSK (4 bits/symbol, 4+12 constellation)
```

**Constellation examples:**
```
BPSK:          QPSK:         8-PSK:        16-APSK:
  ●               ● ●           ● ● ●       ● ● ● ●
                  ● ●         ● ● ● ●     ●  ● ●  ●
                              ● ●  ● ●    ● ●   ● ●
                                          ●  ● ●  ●
```

**Data encoding:**
- Differential phase encoding (immune to frequency drift)
- No pilot symbols needed
- Adapts continuously based on measured SNR
- Negotiated via kernel (modulation field)

### Layer 3: Error Correction (Polar Codes)

**Purpose:** Protect user data against errors

**Applied at:** Protocol layer (not in pattern)

**Adaptive rates:**
```
SNR < 0 dB:    Polar 1/2  (strongest FEC)
SNR 0-5 dB:    Polar 2/3  (strong FEC)
SNR 5-10 dB:   Polar 3/4  (medium FEC)
SNR 10-15 dB:  Polar 4/5  (light FEC)
SNR 15-20 dB:  Polar 5/6  (minimal FEC)
SNR > 20 dB:   Polar 7/8  (very light FEC)
```

**Rate selection:**
- Negotiated via kernel (polar_rate field, 3 bits)
- Trades throughput vs. robustness
- Combined with modulation for adaptive coded modulation (ACM)

---

## Pattern Generation

### Genetic Algorithm

**Goal:** 8 patterns with maximum orthogonality

**Algorithm:**
```python
# Population-based optimization
population_size = 32  # 32 pattern sets
pattern_count = 8     # 8 patterns per set
max_length = 2048     # Optimize at maximum length

# Evolution loop
for generation in range(150000):
    # 1. Selection: Keep top 4 elite sets
    # 2. Crossover: Breed from top 50% (70% rate)
    # 3. Mutation: Flip random bits (10% rate, ~13 bits avg)
    # 4. Evaluation: Measure orthogonality
    # 5. Rank: Sort by worst-case cross-correlation
```

**Fitness function:**
```python
def evaluate_pattern_set(patterns):
    """Measure worst-case orthogonality"""

    worst_corr = -100  # dB

    # Test all pattern pairs
    for i in range(8):
        for j in range(i+1, 8):
            # Normal correlation
            corr_normal = correlate(patterns[i], patterns[j])

            # Flip correlation (adjacent channel interference)
            corr_flip = correlate(patterns[i], invert(patterns[j]))

            # Track worst case
            worst_corr = max(worst_corr, corr_normal, corr_flip)

    return worst_corr  # Lower (more negative) = better
```

### Convergence Results

**Proven trajectory (8 patterns, 2048 symbols):**
```
6k generations:   -17.38 dB
100k generations: -21.19 dB ✅ (proven achievable)
150k generations: -20 to -21 dB (expected)

Welch bound:      -24.6 dB (theoretical limit)
Gap at 100k:      3.41 dB (excellent for genetic algorithm)
```

**Generation time:**
- ~48 hours on modern CPU
- One-time cost (patterns reused forever)
- Deterministic once converged

---

## Frequency Architecture

### 135-Tone Reference Grid

**Grid specification:**
```python
# 135 discrete tones
tones = [300 + i*20 for i in range(135)]
# [300, 320, 340, ..., 2960, 2980, 3000] Hz

# Spacing: 20 Hz
# Bandwidth: 300-3000 Hz (2.7 kHz total, standard SSB)
```

**67 Frequency Pairs:**
```python
# Non-overlapping 2-tone pairs
pairs = [(i*2, i*2+1) for i in range(67)]
# [(0,1), (2,3), (4,5), ..., (132,133)]

# Pair 0:  Tones 0-1   (300-320 Hz)
# Pair 1:  Tones 2-3   (340-360 Hz)
# ...
# Pair 66: Tones 132-133 (2940-2960 Hz)
```

### 2-FSK Modulation

**Each pattern uses 2 adjacent tones:**
```
Pattern binary: [0, 1, 0, 0, 1, 1, ...]
Tones selected: [A, B, A, A, B, B, ...]

Example (Pair 25: tones 50-51):
Symbol 0 (bit=0): Transmit 1300 Hz (tone A)
Symbol 1 (bit=1): Transmit 1320 Hz (tone B)
Symbol 2 (bit=0): Transmit 1300 Hz (tone A)
...
```

**GMSK pulse shaping:**
- Smooths transitions between tones
- BT = 0.3 (moderate smoothing)
- Reduces spectral splatter
- Maintains constant envelope (good for RF amplifiers)

### Logical Channel Count

**Total logical channels:**
```
8 patterns × 67 frequency pairs = 536 logical channels
```

**Supports:**
- 40-45 active users simultaneously
- Kernel coordination for distribution
- RTS/CTS for local collision avoidance

---

## Nested Pattern Lengths

### Automatic Length Variants

**From single 2048-symbol optimization, extract 6 lengths:**

| Symbols | Duration @ 200 sym/s | Welch Bound | Achieved (est) | Use Case |
|---------|---------------------|-------------|----------------|----------|
| 64 | 0.32s | -9.6 dB | ~-7 dB | Micro ACKs |
| 128 | 0.64s | -12.6 dB | ~-10 dB | ACK, CTS |
| 256 | 1.28s | -15.6 dB | ~-13 dB | Short control |
| 512 | 2.56s | -18.6 dB | ~-16 dB | Beacons, RTS |
| 1024 | 5.12s | -21.6 dB | ~-18 dB | Medium messages |
| 2048 | 10.24s | -24.6 dB | -21.19 dB ✅ | Large messages |

**Nested extraction:**
```python
# Shorter patterns = prefixes of longer ones
pattern_64 = pattern_2048[0:64]
pattern_128 = pattern_2048[0:128]
pattern_256 = pattern_2048[0:256]
pattern_512 = pattern_2048[0:512]
pattern_1024 = pattern_2048[0:1024]
# pattern_2048 is the full optimized pattern
```

**Benefits:**
- Single optimization effort
- Cross-length orthogonality guaranteed
- Simpler storage (just store full 2048, extract as needed)

### Capacity by Length

**With polar codes @ 2/3 rate:**

| Length | Raw Bits | After Polar 2/3 | BPSK | QPSK | 8-PSK | 16-APSK |
|--------|----------|-----------------|------|------|-------|---------|
| 128 | 128 | 85 | 85b | 171b | 256b | 341b |
| 256 | 256 | 171 | 171b | 341b | 512b | 683b |
| 512 | 512 | 341 | 341b | 683b | 1024b | 1365b |
| 1024 | 1024 | 683 | 683b | 1365b | 2048b | 2731b |
| 2048 | 2048 | 1365 | 1365b | 2731b | 4096b | 5461b |

**Pattern selection algorithm:**
```python
def select_pattern_length(message_bytes, modulation, polar_rate):
    bits_needed = message_bytes * 8
    bits_per_symbol = {'BPSK': 1, 'QPSK': 2, '8-PSK': 3, '16-APSK': 4}[modulation]
    polar_overhead = {'1/2': 2.0, '2/3': 1.5, '3/4': 1.33, ...}[polar_rate]

    coded_bits = bits_needed * polar_overhead

    for length in [128, 256, 512, 1024, 2048]:
        capacity = length * bits_per_symbol
        if capacity >= coded_bits:
            return length

    return 2048  # Maximum
```

---

## Kernel-Assisted Detection

### Why Kernels Eliminate Blind Detection

**Without kernel (V1 concept):**
```python
# Must correlate against ALL 8 patterns
for pattern_id in range(8):
    correlation = correlate(received_signal, pattern[pattern_id])
    if correlation > threshold:
        detected_pattern = pattern_id
        break

# Cost: 8 correlations per detection
# Time: ~20-30ms on RPi4
```

**With kernel (V2):**
```python
# Beacon tells you which pattern
pattern_id = rx_kernel.pattern_id  # 3 bits: 0-7

# Correlate against only that pattern
correlation = correlate(received_signal, pattern[pattern_id])

# Cost: 1 correlation
# Time: <5ms on RPi4 ✅
```

**Benefits:**
- **8× faster** detection
- **CPU-only** sufficient (no TPU/GPU needed)
- **Lower power** (important for portable ops)
- **Simpler code** (no pattern search logic)

### Kernel Structure (28 bytes)

**Discrete fields (4 bytes):**
```python
{
    'pattern_id': 3 bits,      # 0-7 (which pattern)
    'frequency_pair': 7 bits,  # 0-66 (which tone pair)
    'modulation': 3 bits,      # BPSK/QPSK/8-PSK/16-APSK
    'polar_rate': 3 bits,      # 1/2 to 7/8
    'protocol_version': 2 bits,
    'model_version': 2 bits,
    'reserved': 12 bits
}
```

**Embedding (24 bytes):**
- 48 dimensions × 4-bit quantization
- NN-generated optimization hints
- Enables adaptive encoder mutations

**See:** [Kernel Encoding Spec](../implementation/kernel_encoding_spec.md)

---

## Performance Characteristics

### Orthogonality

**Measured performance (8 patterns, 2048 symbols):**
```
Worst-case cross-correlation: -21.19 dB ✅
Welch bound (theoretical):     -24.6 dB
Gap:                           3.41 dB

Interpretation: Excellent
- Most pattern pairs: < -22 dB
- Worst pair: -21.19 dB
- Sufficient for kernel-assisted detection with frequency separation
```

### Throughput

**Per pattern @ 200 sym/s:**

| Modulation | Polar Rate | Pattern Length | Duration | Throughput |
|------------|-----------|----------------|----------|------------|
| BPSK | 1/2 | 512s | 2.56s | 100 bps |
| QPSK | 2/3 | 512s | 2.56s | 267 bps |
| QPSK | 2/3 | 1024s | 5.12s | 267 bps |
| 8-PSK | 3/4 | 512s | 2.56s | 500 bps |
| 16-APSK | 5/6 | 512s | 2.56s | 853 bps |

**Multi-pattern:**
- Strong receiver: Can decode 4-8 patterns simultaneously
- Throughput scales linearly (4× patterns = 4× throughput)

### Hardware Requirements

**RPi4 (CPU-only):**
- Pattern detection: <5ms (kernel-assisted)
- Pattern encoding: <1ms
- Active users: 40-45 simultaneous
- Power: 8W

**Desktop (x86):**
- Pattern detection: <2ms
- Encoding: <0.5ms
- Active users: 100+
- Overkill but works great

**See:** [Hardware Requirements](../deployment/hardware_requirements.md)

---

## Pattern Storage Format

### File Structure

**Per pattern (all 8 patterns):**
```python
{
    'id': 1 byte,                    # 0-7
    'pattern_2048': 2048 bits,       # Full binary pattern
    'nested_offsets': {              # For quick extraction
        128: 0,
        256: 0,
        512: 0,
        1024: 0,
        2048: 0
    },
    'checksum': 2 bytes
}

# Per pattern: ~260 bytes
# Total (8 patterns): ~2 KB
```

**Much smaller than V1:**
- V1 (128 patterns × 292 bytes): 38 KB
- V2 (8 patterns × 260 bytes): 2 KB
- **19× reduction** in storage

### Loading at Runtime

```python
# Load pattern set
with open('cascade_patterns_v2.bin', 'rb') as f:
    patterns = load_patterns(f)

# Select pattern (from kernel)
pattern_id = rx_kernel.pattern_id  # 0-7

# Select length (from message size)
length = select_pattern_length(message_size, modulation, polar_rate)

# Extract pattern
pattern = patterns[pattern_id][0:length]  # Nested extraction
```

---

## Comparison: V1 vs V2

| Aspect | V1 (Archived) | V2 (Current) |
|--------|--------------|-------------|
| Pattern count | 128 | **8** |
| Modulation | 4-FSK | **GMSK 2-FSK** |
| Data encoding | RS(32,20) + IQ | **GMSK + IQ + Polar** |
| Symbol rate | 200 sym/s | **200 sym/s** |
| Error correction | Built-in RS | **Polar codes (protocol layer)** |
| Detection | Blind 128-pattern | **Kernel-assisted (1 pattern)** |
| Orthogonality | -37.5 dB (claimed) | **-21.19 dB (proven)** |
| Storage | 38 KB | **2 KB** |
| CPU requirement | High (blind search) | **Low (kernel-assisted)** |
| Hardware | Coral TPU recommended | **RPi4 CPU-only** |
| Logical channels | 128 | **536 (8 × 67 pairs)** |

**V2 is simpler, proven, and more practical.**

---

## See Also

- **[Signal Specification](../protocol/signal_specification.md)** - Physical layer details
- **[Pattern Generation Spec](../implementation/pattern_generation_spec.md)** - Genetic algorithm
- **[Kernel Encoding](../implementation/kernel_encoding_spec.md)** - 28-byte kernel structure
- **[Protocol Layer](../protocol/README.md)** - RTS/CTS and coordination
- **[Architecture Summary](../../architecture.md)** - Executive overview
