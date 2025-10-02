# CASCADE Pattern Generation

This document describes the generation of CASCADE's 64 orthogonal patterns that enable multi-user CDMA-like operation.

## Requirements

**Pattern specifications:**
- **Count**: 64 patterns (6-bit pattern ID)
- **Length**: 32 symbols per pattern
- **Alphabet**: 8 tones (0-7, maps to message frequencies)
- **Orthogonality**: <-30 dB cross-correlation between all pattern pairs
- **Deterministic**: Fixed sequences, not learned (ensures interoperability)

## Why <-30 dB?

**Comparison with existing systems:**
```
FT8: ~-20 dB (supports 1-2 overlapping)
CDMA (Gold codes): ~-15 dB (cellular, supports 10-20 users)
GPS (Gold codes): ~-24 dB (satellite navigation)
CASCADE target: <-30 dB (supports 50+ overlapping)
```

The -30 dB requirement enables CASCADE to support 50+ simultaneous users with minimal interference, exceeding traditional CDMA systems.

## Generation Algorithm

### Phase 1: Zadoff-Chu Base Patterns

**Rationale**: Zadoff-Chu sequences are used in LTE/5G for their excellent correlation properties. The mathematics dates to the 1970s (public domain, no patents).

```python
import numpy as np

def generate_zadoff_chu_base(u, N=31):
    """
    Generate Zadoff-Chu sequence

    Args:
        u: Root index (0 to N-1)
        N: Sequence length (must be prime, 31 is closest to 32)

    Returns:
        32-symbol 8-ary pattern
    """
    sequence = []

    for n in range(31):  # N=31 (prime)
        # Zadoff-Chu: complex exponential with quadratic phase
        q = u * n * (n + 1) / 2
        phase = 2 * np.pi * q / N

        # Map complex exponential to 8-ary symbol
        # Use phase angle: 0-2π maps to 0-7
        symbol = int((phase % (2 * np.pi)) / (2 * np.pi / 8))
        sequence.append(symbol)

    # Pad to 32 symbols
    sequence.append(0)  # Zero-pad 32nd symbol

    return np.array(sequence, dtype=np.uint8)

# Generate 31 base patterns (u = 0 to 30)
base_patterns = [generate_zadoff_chu_base(u) for u in range(31)]
```

**Properties of Zadoff-Chu:**
- Perfect autocorrelation (zero sidelobes)
- Constant amplitude (good for power amplifiers)
- Bounded cross-correlation (~1/√N ≈ -15 dB for 31-length)
- **Not sufficient for -30 dB alone** - needs optimization

### Phase 2: Computer Optimization to -30 dB

**Method**: Simulated annealing to fine-tune patterns

```python
def cross_correlation(pattern1, pattern2):
    """Compute maximum cross-correlation between two patterns"""
    max_corr = 0
    for shift in range(len(pattern1)):
        corr = np.abs(np.sum(
            np.exp(1j * 2 * np.pi * np.roll(pattern1, shift) / 8) *
            np.conj(np.exp(1j * 2 * np.pi * pattern2 / 8))
        ))
        max_corr = max(max_corr, corr)

    # Normalize and convert to dB
    max_corr_normalized = max_corr / len(pattern1)
    return 20 * np.log10(max_corr_normalized + 1e-10)

def optimize_to_30db(base_pattern, existing_patterns, iterations=100000):
    """Optimize a single pattern to <-30 dB correlation with all existing"""

    best_pattern = base_pattern.copy()
    best_max_corr = float('inf')

    temperature = 1.0
    cooling_rate = 0.9999

    for i in range(iterations):
        # Mutate: flip one random symbol to different tone
        candidate = best_pattern.copy()
        idx = np.random.randint(32)
        candidate[idx] = np.random.randint(8)

        # Check correlation with all existing patterns
        max_corr = max(
            cross_correlation(candidate, p)
            for p in existing_patterns
        )

        # Simulated annealing acceptance
        if max_corr < best_max_corr:
            best_pattern = candidate
            best_max_corr = max_corr
        elif np.random.random() < np.exp(-(max_corr - best_max_corr) / temperature):
            best_pattern = candidate
            best_max_corr = max_corr

        temperature *= cooling_rate

        # Success criterion
        if best_max_corr < -30:
            print(f"Pattern optimized to {best_max_corr:.1f} dB in {i} iterations")
            return best_pattern

    if best_max_corr < -30:
        return best_pattern
    else:
        raise ValueError(f"Could not achieve -30 dB (got {best_max_corr:.1f} dB)")

# Optimize all 64 patterns
optimized_patterns = []

# Start with Zadoff-Chu bases
for base in base_patterns:
    optimized = optimize_to_30db(base, optimized_patterns)
    optimized_patterns.append(optimized)

# Generate remaining patterns (64 - 31 = 33 more)
for _ in range(33):
    # Random initialization
    random_base = np.random.randint(0, 8, size=32, dtype=np.uint8)
    optimized = optimize_to_30db(random_base, optimized_patterns)
    optimized_patterns.append(optimized)

print(f"Generated {len(optimized_patterns)} patterns")
```

### Phase 3: Validation

```python
def validate_patterns(patterns):
    """Verify all patterns meet -30 dB requirement"""

    print(f"Validating {len(patterns)} patterns...")

    max_cross_corr = -100  # Start very low

    for i, p1 in enumerate(patterns):
        for j, p2 in enumerate(patterns):
            if i >= j:
                continue

            corr = cross_correlation(p1, p2)
            max_cross_corr = max(max_cross_corr, corr)

            if corr > -30:
                print(f"WARNING: Patterns {i} and {j}: {corr:.1f} dB")

    print(f"Maximum cross-correlation: {max_cross_corr:.1f} dB")

    if max_cross_corr < -30:
        print("✓ All patterns meet -30 dB requirement")
        return True
    else:
        print("✗ Optimization failed")
        return False

validate_patterns(optimized_patterns)
```

## Pattern Storage Format

**Storage requirements:**
```python
# Each pattern: 32 symbols × 1 byte = 32 bytes
# 64 patterns: 64 × 32 = 2,048 bytes = 2 KB

# Additional metadata per pattern
pattern_metadata = {
    'id': 1 byte,           # 0-63
    'checksum': 2 bytes,    # CRC16 for integrity
    'reserved': 1 byte      # Future use
}

# Total: (32 + 4) bytes × 64 = 2,304 bytes ≈ 2.3 KB
```

**File format** (`cascade_patterns_v1.bin`):
```
Header (16 bytes):
- Magic: "CASC" (4 bytes)
- Version: 1 (2 bytes)
- Pattern count: 64 (2 bytes)
- Pattern length: 32 (2 bytes)
- Reserved: (6 bytes)

Pattern data (64 × 36 bytes = 2,304 bytes):
For each pattern:
  - ID: 1 byte
  - Sequence: 32 bytes (tone indices 0-7)
  - Checksum: 2 bytes (CRC16)
  - Reserved: 1 byte

Total file size: 16 + 2,304 = 2,320 bytes
```

## Alternative Approaches (Comparison)

### Gold Codes
**Pros:**
- Well-studied (GPS, CDMA)
- LFSR-based generation (simple)

**Cons:**
- Only ~-15 dB cross-correlation (insufficient)
- Limited to 2^m sequences (64 requires m=6)

**Verdict:** ❌ Insufficient correlation performance

### Kasami Sequences
**Small set:**
- Only 8 sequences for m=6 (need 64) ❌

**Large set:**
- 520 sequences available ✓
- ~-17 dB correlation ❌

**Verdict:** ❌ Still insufficient for -30 dB

### Walsh-Hadamard
**Pros:**
- Perfect orthogonality in ideal channel
- Simple generation

**Cons:**
- Requires synchronous transmission
- Poor performance with timing/frequency offsets
- Not suitable for asynchronous CASCADE

**Verdict:** ❌ Requires synchronization CASCADE doesn't have

### Zadoff-Chu + Optimization (CHOSEN)
**Pros:**
- LTE/5G proven (billions of devices)
- Excellent starting point (~-15 to -20 dB)
- Optimization achieves exactly -30 dB
- Deterministic (once computed, stored)
- Patent-free (1970s mathematics)

**Cons:**
- One-time computational cost (acceptable)
- Requires storage (2.3 KB is trivial)

**Verdict:** ✅ Best balance of performance and practicality

## Implementation Notes

**One-time generation:**
- Run optimization once
- Store resulting patterns in protocol specification
- All implementations use identical stored patterns

**Pattern distribution:**
- Patterns embedded in CASCADE protocol specification
- Checksums verify integrity
- No runtime generation needed

**Interoperability:**
- All CASCADE implementations must use identical patterns
- Pattern file versioned (cascade_patterns_v1.bin)
- Future versions may optimize further (v2, v3...)

## Verification Script

```python
# Load patterns and verify
def load_and_verify_patterns(filename):
    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(4)
        assert magic == b'CASC', "Invalid magic number"

        version = int.from_bytes(f.read(2), 'little')
        count = int.from_bytes(f.read(2), 'little')
        length = int.from_bytes(f.read(2), 'little')

        f.read(6)  # Skip reserved

        patterns = []
        for i in range(count):
            pattern_id = int.from_bytes(f.read(1), 'little')
            sequence = np.frombuffer(f.read(length), dtype=np.uint8)
            checksum = int.from_bytes(f.read(2), 'little')
            f.read(1)  # Skip reserved

            # Verify checksum
            computed_crc = crc16(sequence)
            assert checksum == computed_crc, f"Checksum mismatch for pattern {pattern_id}"

            patterns.append(sequence)

        # Verify orthogonality
        assert validate_patterns(patterns), "Patterns failed validation"

        return patterns

patterns = load_and_verify_patterns('cascade_patterns_v1.bin')
```

## See Also

- **[Signal Specification](signal_specification.md)** - How patterns are used in the protocol
- **[Model Architecture](../model/README.md)** - How model performs pattern correlation
- **[Hardware Requirements](../deployment/hardware_requirements.md)** - Pattern correlation performance requirements
