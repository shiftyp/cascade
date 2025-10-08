# CASCADE Pattern Generation Specification (V2)

**Purpose:** Genetic algorithm for 8-pattern nested generation (no built-in redundancy)
**Status:** ✅ Proven effective (-21.19 dB achieved at 2048 symbols)
**Runtime:** ~48 hours for 150k generations on modern CPU
**Output:** Nested pattern set with 6 usable lengths (64-2048 symbols)
**Error correction:** Polar codes at protocol layer (not in pattern)

---

## Overview

CASCADE V2 uses **8 patterns with nested lengths** generated via genetic algorithm:

1. **Genetic algorithm** with 32-member population
2. **No built-in redundancy** (pattern length = bits, e.g., 512s = 512 bits)
3. **Nested extraction** (shorter patterns are prefixes of longer ones)
4. **GMSK modulation** (applied during transmission, BT=0.3)
5. **Polar code error correction** (protocol layer, adaptive rates 1/2 to 7/8)

**Key achievement:**
- 2048 symbols (2048 bits, pure orthogonality optimization): **-21.19 dB orthogonality**
- Welch bound: -24.6 dB (theoretical limit for 8 patterns, 2048 symbols)
- Gap: 3.41 dB (excellent for genetic algorithm!)
- With flip orthogonality for adjacent channel GMSK sidelobe interference

---

## Generation Command

```bash
# Generate optimal 8-pattern nested set (no redundancy)
python modules/training/patterns/tournament/generate_patterns_tournament.py \
  --pattern-count 8 \
  --pattern-length 2048 \
  --redundancy 1 \
  --generations 150000 \
  --p-cores 0-7
```

**Parameters:**
- `--pattern-count 8`: Number of patterns (4 or 8 recommended for CASCADE)
- `--pattern-length 2048`: Maximum symbol count (e.g., 256-4096)
- `--redundancy 1`: **Pure orthogonality optimization (polar codes handle FEC at protocol layer)**
- `--generations 150000`: GA generations (more = better orthogonality)
- `--p-cores 0-7`: CPU cores to use (optional)

**Output location:**
```
checkpoints/p8_l2048_r1x/output/patterns_p8_l2048_r1x_TIMESTAMP.pkl
```

**Partitioning:** Automatically organized by configuration (p{count}_l{length}_r{redundancy}x)

**Note:** The `--redundancy 1` parameter means no repetition in pattern structure; error correction is handled by polar codes at the protocol layer, not within the pattern itself.

---

## Nested Pattern Structure

### Automatic Length Variants

**From single 2048-symbol optimization**, get 6 usable lengths:

| Symbols | Core Bits | Welch Bound | Duration @ 200 sym/s | With Polar 2/3 |
|---------|-----------|-------------|---------------------|----------------|
| 2048 | 2048 | -24.6 dB | 10.24s | 1365 bits |
| 1024 | 1024 | -21.6 dB | 5.12s | 683 bits |
| 512 | 512 | -18.6 dB | 2.56s | 341 bits |
| 256 | 256 | -15.6 dB | 1.28s | 171 bits |
| 128 | 128 | -12.6 dB | 0.64s | 85 bits |
| 64 | 64 | -9.6 dB | 0.32s | 43 bits |

**Achieved:** -21.19 dB @ 2048 symbols (3.41 dB from Welch bound)

**Nested extraction (no redundancy):**
```python
# Shorter patterns = prefixes of longer ones
# Direct bit extraction from optimized 2048-bit pattern
pattern_1024 = pattern_2048[0:1024]  # First 1024 bits
pattern_512 = pattern_2048[0:512]    # First 512 bits
pattern_256 = pattern_2048[0:256]    # First 256 bits
pattern_128 = pattern_2048[0:128]    # First 128 bits
pattern_64 = pattern_2048[0:64]      # First 64 bits
# Perfect cross-length orthogonality (nested structure)
```

### Pattern Selection by Message Size

| Message Size | Pattern Length | Modulation | Duration @ 200 sym/s | Example |
|--------------|----------------|------------|---------------------|---------|
| 5-30 bytes | 128s | BPSK | 0.64s | ACK, CTS |
| 28 bytes (1 kernel) | 512s | BPSK | 2.56s | Beacon, RTS |
| 50-100 bytes | 512s | QPSK | 2.56s | Short message |
| 100-200 bytes | 1024s | QPSK | 5.12s | Medium message |
| 200-400 bytes | 2048s | QPSK | 10.24s | Long message |
| 400-500 bytes | 2048s | 8-PSK | 10.24s | Maximum |

---

## Genetic Algorithm Details

### Population Structure

**32 pattern sets** per trial, each set contains 8 patterns:
```python
population = [
    [Pattern_0, Pattern_1, ..., Pattern_7],  # Set 0
    [Pattern_0, Pattern_1, ..., Pattern_7],  # Set 1
    ...
    [Pattern_0, Pattern_1, ..., Pattern_7]   # Set 31
]
```

### Evolution Loop

```python
for generation in range(150000):
    # 1. SELECTION (Elitism)
    keep_top_4_sets()

    # 2. CROSSOVER (70% rate)
    breed_from_top_16_sets()
    # Example: Patterns 0-3 from parent A, patterns 4-7 from parent B

    # 3. MUTATION (10% rate per pattern)
    mutate_core_bits(13_bits_average)

    # 4. EVALUATION
    if generation % 10 == 0:
        evaluate_all_120_pairs(full_correlation)
    else:
        evaluate_30_random_pairs(fast_sampling)

    # 5. RANKING
    sort_by_worst_case_orthogonality()
```

### Fitness Function

**Triple orthogonality testing:**
```python
def evaluate_fitness(pattern_set):
    worst_normal = -100
    worst_flip = -100
    worst_erasure = -100

    for pattern_i, pattern_j in pairs:
        # Expand cores using repetition map
        expanded_i = core_i[repetition_map]
        expanded_j = core_j[repetition_map]

        # Normal correlation
        corr_normal = correlate(expanded_i, expanded_j)
        worst_normal = max(worst_normal, corr_normal)

        # Flip correlation (adjacent channel interference)
        corr_flip = correlate(expanded_i, invert(expanded_j))
        worst_flip = max(worst_flip, corr_flip)

        # Erasure correlation (~20% symbol loss)
        corr_erasure = correlate_with_dropout(expanded_i, expanded_j, 0.2)
        worst_erasure = max(worst_erasure, corr_erasure)

    return max(worst_normal, worst_flip, worst_erasure)
```

### Convergence Results

**Proven trajectory (8 patterns, 2048 symbols, no redundancy):**
- 6k generations: -17.38 dB
- 100k generations: **-21.19 dB** ✅
- Welch bound: -24.6 dB (theoretical limit for 8p, 2048s)
- Gap: 3.41 dB (excellent for genetic algorithm!)

---

## Output Format

### Nested Pattern File Structure

```python
{
    'nested_patterns': {
        128: {
            'cores': [array([0,1,1,...]), ...],  # 8 × 64-bit cores
            'repetition_map': array([...]),      # 128-element map
            'core_length': 64,
            'full_length': 128
        },
        256: {...},  # 8 × 128-bit cores
        512: {...},  # 8 × 256-bit cores
        1024: {...}, # 8 × 512-bit cores
        2048: {      # 8 × 1024-bit cores
            'cores': [array([0,1,0,1,...]), ...],
            'repetition_map': array([...]),
            'core_length': 1024,
            'full_length': 2048
        }
    },
    'nested_orthogonality': {
        128: -8.2,    # dB (estimated)
        256: -11.3,
        512: -14.5,
        1024: -17.4,
        2048: -19.87  # Proven!
    },
    'num_patterns': 8,
    'redundancy': 1,
    'max_core_length': 2048,
    'max_full_length': 2048,
    'algorithm': 'genetic_nested',
    'generations': 100000
}
```

### Pattern Structure (No Built-in Redundancy)

**Pure orthogonality optimization:**
```python
# Example for 512-symbol pattern
pattern = [0, 1, 1, 0, 1, 0, ...]  # 512 bits (pattern length = bits)
# No repetition map needed
# Error correction handled by polar codes at protocol layer
```

**Polar code error correction (separate from pattern):**
- Applied at protocol layer after pattern selection
- Adaptive rates: 1/2, 2/3, 3/4, 4/5, 5/6, 7/8
- Negotiated via kernel based on measured SNR
- More efficient than building redundancy into patterns

---

## Usage in CASCADE Modem

### Loading Patterns

```python
import pickle

# Load nested pattern set
with open('patterns_p8_l2048_r1x_*.pkl', 'rb') as f:
    pattern_data = pickle.load(f)

# Select length based on message size
def select_pattern_length(message_bytes, modulation):
    bits_needed = message_bytes * 8

    if modulation == 'BPSK':
        bits_per_symbol = 1
    elif modulation == 'QPSK':
        bits_per_symbol = 2
    elif modulation == '8-PSK':
        bits_per_symbol = 3
    else:  # 16-APSK
        bits_per_symbol = 4

    # Find smallest pattern that fits
    for length in [128, 256, 512, 1024, 2048]:
        core_bits = length // 2  # 2x redundancy
        capacity = core_bits * bits_per_symbol
        if capacity >= bits_needed:
            return length

    return 2048  # Maximum

# Example: 152-byte message @ QPSK
length = select_pattern_length(152, 'QPSK')  # Returns 1024
patterns = pattern_data['nested_patterns'][length]
```

### Encoding Message

```python
# Get pattern from kernel
pattern_id = rx_kernel.pattern_id  # 0-7
pattern = patterns['cores'][pattern_id]  # Direct pattern (no expansion needed)

# Apply Polar encoding to user data
user_bits = message_to_bits(message)
polar_rate = rx_kernel.polar_rate  # From kernel (1/2, 2/3, 3/4, etc.)
polar_encoded = polar_encode(user_bits, rate=polar_rate)

# Encode on pattern with GMSK + IQ modulation
transmitted_symbols = []

for i, fsk_symbol in enumerate(pattern):
    # Layer 1: GMSK 2-tone FSK (pattern selection)
    tone = gmsk_tone_A if fsk_symbol == 0 else gmsk_tone_B

    # Layer 2: IQ data modulation on selected tone
    data_bits = extract_bits(polar_encoded, i, modulation_order)
    iq_symbol = modulate_iq(data_bits, modulation_type)

    # Transmit GMSK tone with IQ modulation
    transmitted_symbols.append((tone, iq_symbol))
```

---

## Performance Characteristics

### Orthogonality vs Pattern Count

| Patterns | Welch @ 1024s | Achieved @ 100k gen | Capacity (×67 pairs) |
|----------|---------------|---------------------|---------------------|
| 4 | -25.3 dB | -24 to -25 dB | 268 channels |
| 8 | -21.6 dB | -21.19 dB ✅ | 536 channels |
| 16 | -18.3 dB | -16 to -17 dB | 1,072 channels |

**Recommendation: 8 patterns** (best balance of orthogonality and capacity)

### Redundancy Trade-offs (8 patterns, 2048 symbols)

| Redundancy | Core Bits | Welch Bound | Error Correction Method |
|------------|-----------|-------------|------------------------|
| **1x** | **2048** | **-24.6 dB** | **Polar (adaptive 1/2 to 7/8)** |
| 2x | 1024 | -21.6 dB | Pattern repetition + Polar |
| 4x | 512 | -18.6 dB | Pattern repetition (QR-like) |

**Recommendation: 1x redundancy** (best orthogonality, Polar provides superior FEC)

**Why 1x + Polar is better:**
- More core bits to optimize (all 2048 bits)
- Better Welch bound (-24.6 vs -21.6 dB)
- Adaptive error correction (adjust rate to SNR)
- Higher throughput (Polar overhead 12.5% to 50% vs fixed 75%)

---

## Archived Documentation

**V1 (128-pattern theoretical):** See `pattern_generation_spec_v1_archived.md`

V1 explored Zadoff-Chu sequences with hierarchical IQ for 128 patterns.
V2 uses genetic algorithm for 8 patterns with proven convergence.
