# CASCADE Pattern Generation Status

## Current Implementation

Optimized pattern generator for CASCADE HF modem with QR-like erasure coding:

### Pattern Structure
- **Total**: 16 patterns (reused across 67 frequency pairs = 1,072 logical channels)
- **Core**: 128 bits per pattern (the actual pattern signature)
- **Expanded**: 512 symbols (2.56 seconds at 200 symbols/second) via repetition map
- **Modulation**: 2-FSK (binary frequency sequences)
- **Repetition Map**: Each of 128 core bits repeated 4x across 512 symbols
- **Erasure Tolerance**: 37.5% (need only 320 of 512 symbols to recover via majority vote)

### Orthogonality Targets
- **Normal**: -30.0 dB (Welch bound: -30.4 dB)
- **Flip**: -28.0 dB (patterns inverted 0↔1)
- **Erasure**: -27.0 dB (with 37.5% random symbol loss)

### Data Capacity (per pattern after erasure coding)
- **BPSK**: 200 bits (125 bps)
- **QPSK**: 400 bits (250 bps)
- **8-PSK**: 600 bits (375 bps)
- **16-QAM**: 800 bits (500 bps)
- **Multi-pattern**: Up to 3,000 bps with 8 patterns

### Optimization Process
1. **Initial patterns**: 16 random 128-bit cores, expanded to 512 via repetition map
2. **Mutation**: Mutate 128 core bits (19 → 1 bit over time)
3. **Expansion**: Apply repetition map to get 512-symbol pattern
4. **Triple evaluation** on expanded patterns:
   - Normal correlation: Full cross-correlation (1,023 shifts)
   - Flip correlation: Inverted pattern correlation
   - Erasure correlation: 5 trials with 37.5% random dropout
5. **Accept**: Only if improved (greedy hill-climbing)
6. **Global update**: Every 10 iterations across all 120 pattern pairs

### Execution Schedule
- **First run**: 200,000 iterations minimum per trial
- **Subsequent**: 50,000 iteration evaluation intervals
- **Total budget**: 4.8M iterations across 8 trials (600k avg per trial)
- **68x faster** than 128-pattern approach (120 pairs vs 8,128 pairs)
- **Smaller search space**: 128 core bits vs 512 independent bits

### Computational Complexity
- **Per iteration**: ~0.003 seconds (300-400 iter/sec)
  - Mutate 128 core bits
  - Expand to 512 using repetition map
  - Correlate 15 other expanded patterns
  - Every 10th: full pairwise eval (120 pairs)
- **200k iterations**: ~10-13 minutes per trial
- **4.8M total**: ~24-32 hours on 8 P-cores

### Output Format
```python
{
    'patterns': [array([0,1,1,0,...]), ...],  # 16 × 128 core patterns
    'repetition_maps': [array([45,12,78,...]), ...],  # 16 × 512 expansion maps (same for all)
    'num_patterns': 16,
    'pattern_core_length': 128,  # Core bits
    'pattern_full_length': 512,  # After expansion
    'unique_data_positions': 128,
    'redundancy_factor': 4,
    'best_score': -28.5  # dB
}
```

### Files Generated
- `checkpoints/trial_N/final_patterns_*.pkl`: Patterns with repetition maps
- `logs/debug_trial_N.txt`: Optimization progress
- `checkpoints/output/patterns_*.pkl`: Final patterns for CASCADE modem

## Usage

### Nested Pattern Usage (Adaptive Length)

**Patterns generated at maximum length automatically include all shorter variants as prefixes:**

```python
# Load patterns
import pickle
with open('patterns_p8_l2048_r2x_20251006.pkl', 'rb') as f:
    data = pickle.load(f)

# Check available lengths
print(f"Available pattern lengths: {sorted(data['nested_patterns'].keys())}")
# Output: [128, 256, 512, 1024, 2048]

# See orthogonality at each length
for length, orth_db in data['nested_orthogonality'].items():
    print(f"{length} symbols: {orth_db:.2f} dB")
# Output:
#   128 symbols: -12.4 dB (0.64s)
#   256 symbols: -15.6 dB (1.28s)
#   512 symbols: -18.6 dB (2.56s)
#   1024 symbols: -21.6 dB (5.12s)
#   2048 symbols: -24.6 dB (10.24s)

# Use pattern length based on message size
if message_size < 100:
    length = 256  # Fast (1.28s)
elif message_size < 300:
    length = 512  # Balanced (2.56s)
else:
    length = 1024  # High reliability (5.12s)

# Extract pattern and repetition map for chosen length
pattern_data = data['nested_patterns'][length]
pattern_cores = pattern_data['cores']  # 8 patterns
rep_map = pattern_data['repetition_map']

# Kernel tells you which pattern to use
pattern_id = kernel.pattern_id  # From beacon (0-7)
pattern_core = pattern_cores[pattern_id]

# Expand to full length
pattern_full = pattern_core[rep_map]

# Encode IQ data
user_data = [1, 0, 1, 1, ...]  # Core_length bits of user data
iq_symbols = []
for i, fsk_symbol in enumerate(pattern_full):
    data_index = rep_map[i]
    iq_value = user_data[data_index]
    iq_symbols.append(modulate(fsk_symbol, iq_value))
```

### Benefits of Nested Patterns:

1. **Adaptive transmission**: Pick length based on message size
2. **Perfect cross-length orthogonality**: 256s pattern won't interfere with 1024s on adjacent channel
3. **Single optimization**: Generate once, use at any length
4. **Efficient**: Short messages don't pay full pattern overhead

## Next Steps
1. Run tournament with large pattern length (e.g., `--pattern-length 2048`)
2. Get nested variants automatically (128, 256, 512, 1024, 2048)
3. Test in CASCADE modem with adaptive length selection