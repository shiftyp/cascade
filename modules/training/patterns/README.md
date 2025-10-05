# CASCADE Pattern Generation Module

**Version:** 3.0 (with flip-orthogonality support)
**Date:** October 2025

## Overview

This module generates 128 orthogonal patterns for the CASCADE HF radio system. Each pattern represents a unique signal shape in Time × Frequency × IQ space, designed to maintain -37.5 dB orthogonality under normal conditions and -30 dB orthogonality when FSK-inverted (flip-orthogonality).

## Key Features

### 1. **Flip-Orthogonality Support** (New in v3.0)
- Patterns maintain orthogonality even when FSK-inverted (0↔1)
- Critical for adjacent-channel operation when patterns share tones
- Enables robust operation with overlapping frequency allocations

### 2. **2-FSK Architecture**
- Each pattern uses 2 adjacent tones (20 Hz spacing)
- Tone indices: 0-1 for all patterns
- 135-tone reference grid: 300-3000 Hz (standard SSB)

### 3. **Optimized IQ Complexity**
- All patterns start with λ=0.0 (BPSK, maximum robustness)
- Optimizer increases λ only if needed for orthogonality
- Average λ: 0.08-0.10 for exceptional low-SNR performance

## Pattern Structure

```python
Pattern:
  - pattern_id: 0-127 (7-bit encoding)
  - freq_sequence: 32 × uint8 (tone indices 0-1)
  - iq_trajectory: 32 × complex64 (IQ constellation points)
  - iq_complexity_lambda: float (0.0 to 0.9)
  - flip_orthogonality_stats: dict
    - max_flip_correlation_db: float
    - avg_flip_correlation_db: float
    - adjacent_channel_safe: bool
```

## Flip-Orthogonality

### What is Flip-Orthogonality?

Flip-orthogonality ensures patterns remain separable when FSK-inverted due to:
- Phase inversions in multipath propagation
- Adjacent-channel interference when patterns share tones
- Receiver phase ambiguity

### How It Works

```python
# Normal pattern uses tones [0, 1, 0, 1, ...]
normal_pattern = [tone_0, tone_1, tone_0, tone_1, ...]

# FSK-inverted pattern swaps 0↔1
flipped_pattern = [tone_1, tone_0, tone_1, tone_0, ...]

# Flip-orthogonal patterns maintain <-30 dB correlation
# even when one is FSK-inverted
```

### Adjacent Channel Safety

When patterns use adjacent tone pairs that share a tone:
```
Pattern A: Uses tones [34, 35]
Pattern B: Uses tones [35, 36]
Shared tone: 35

Without flip-orthogonality: High interference risk
With flip-orthogonality: Safe operation maintained
```

## Generation Process

### 1. Optimization with Flip Constraints

```python
from modules.training.patterns.generator import generate_pattern_set

# Generate 128 patterns with flip-orthogonality
patterns = generate_pattern_set(
    count=128,
    seed=42,
    flip_weight=0.5  # Balance normal and flip orthogonality
)
```

### 2. Cost Function

The optimizer balances three objectives:
1. **Primary:** Normal orthogonality (<-37.5 dB)
2. **Secondary:** Flip-orthogonality (<-30 dB)
3. **Tertiary:** Minimize IQ complexity (lower λ)

```python
cost = orthogonality_violation + flip_weight * flip_violation + 0.1 * lambda_penalty
```

### 3. Validation

```python
from modules.training.patterns.validator import validate_flip_orthogonality

# Validate flip-orthogonality
passes, stats = validate_flip_orthogonality(patterns, target_db=-30.0)

print(f"Min flip correlation: {stats['min_flip_corr_db']:.2f} dB")
print(f"Max flip correlation: {stats['max_flip_corr_db']:.2f} dB")
print(f"Adjacent-safe patterns: {stats['adjacent_safe_count']}/{len(patterns)}")
```

## Binary File Format (v3)

The binary format now includes flip-orthogonality statistics:

```
Header (32 bytes):
  - Magic: b'CASC' (4 bytes)
  - Version: 3 (1 byte)
  - Pattern count: uint16 (2 bytes)
  - Reserved: 25 bytes

Per Pattern (304 bytes):
  - Pattern ID: uint8 (1 byte)
  - Freq sequence: 32 × uint8 (32 bytes)
  - IQ trajectory: 32 × complex64 (256 bytes)
  - IQ complexity λ: float32 (4 bytes)
  - Flip stats: (9 bytes)
    - Max flip corr: float32 (4 bytes)
    - Avg flip corr: float32 (4 bytes)
    - Adjacent safe: uint8 (1 byte)
  - Checksum: uint16 (2 bytes)
```

Total file size for 128 patterns: ~39 KB

## Usage Examples

### Generate Patterns with Flip-Orthogonality

```python
from modules.training.patterns.generator import generate_pattern_set

# Generate with strong flip-orthogonality emphasis
patterns = generate_pattern_set(
    count=128,
    seed=42,
    flip_weight=0.8  # Prioritize flip-orthogonality
)
```

### Check Adjacent Channel Safety

```python
from modules.training.patterns.correlation import check_adjacent_channel_safety

# Check if patterns can safely use adjacent tone pairs
tone_pair_a = (34, 35)
tone_pair_b = (35, 36)

is_safe = check_adjacent_channel_safety(
    pattern_a, pattern_b,
    tone_pair_a, tone_pair_b
)

if is_safe:
    print("Patterns can safely operate on adjacent channels")
```

### Generate Validation Report

```python
from modules.training.patterns.validator import generate_flip_validation_report

# Generate comprehensive report
report = generate_flip_validation_report(patterns)
print(report)
```

## Performance Metrics

### Orthogonality Targets
- **Normal:** <-37.5 dB (all pattern pairs)
- **Flip:** <-30.0 dB (FSK-inverted patterns)
- **Phase-robust:** <-35.0 dB (under random phase distortion)

### Typical Results (128 patterns)
- **BPSK patterns (λ<0.05):** 75 patterns (59%)
- **Low complexity (λ<0.25):** 128 patterns (100%)
- **Adjacent-safe:** 120+ patterns (94%+)
- **Generation time:** 72-96 hours (one-time cost)

## Testing

Run the flip-orthogonality test suite:

```bash
pytest modules/training/tests/patterns/test_flip_orthogonality.py -v
```

## Technical Details

### Correlation Functions

```python
# Normal 4D correlation
corr_normal = compute_4d_correlation(pattern_i, pattern_j)

# Flip correlation (pattern_j FSK-inverted)
corr_flip = compute_flip_correlation(pattern_i, pattern_j)

# All correlation types
all_corrs = compute_all_correlations(pattern_i, pattern_j)
# Returns: normal, j_flipped, i_flipped, both_flipped, max_correlation
```

### Optimization Parameters

```python
# Default optimization settings
MAX_ITERATIONS = 150000  # Increased for flip constraints
FLIP_WEIGHT = 0.5       # Balance normal vs flip orthogonality
TARGET_DB = -37.5       # Normal orthogonality target
FLIP_TARGET_DB = -30.0  # Flip orthogonality target (relaxed by 7.5 dB)
```

## Migration from v2

For existing code using v2 patterns (without flip-orthogonality):

1. **Binary files:** v3 loader supports both v2 and v3 formats
2. **Pattern generation:** Add `flip_weight` parameter (default 0.5)
3. **Validation:** Use new flip validation functions for comprehensive testing

## References

- [CASCADE Architecture](../../../docs/architecture.md)
- [Pattern Architecture](../../../docs/model/pattern_architecture.md)
- [Beacon Protocol](../../../docs/protocol/beacon_reservation.md)

## Authors

CASCADE Development Team
October 2025

---

*Note: Flip-orthogonality is a critical advancement for CASCADE's adjacent-channel operation, enabling efficient spectrum utilization while maintaining robust pattern separation under real-world HF propagation conditions.*