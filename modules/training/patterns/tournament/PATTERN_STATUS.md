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

### Data Encoding Example
```python
# Load patterns
import pickle
with open('patterns_20251005.pkl', 'rb') as f:
    data = pickle.load(f)

pattern_cores = data['patterns']  # 16 × 128 core patterns
repetition_maps = data['repetition_maps']  # 16 × 512 expansion maps

# Encode data on pattern 0
pattern_core = pattern_cores[0]  # 128 core bits (pattern signature)
rep_map = repetition_maps[0]  # Expansion map (0-127 indices)

# Expand to full 512-symbol pattern
pattern_full = pattern_core[rep_map]  # 512 symbols for transmission

user_data = [1, 0, 1, 1, ...]  # 128 bits of user data
iq_symbols = []

for i, fsk_symbol in enumerate(pattern_full):
    data_index = rep_map[i]  # Which data position (0-127)
    iq_value = user_data[data_index]  # Get the data bit
    # Modulate: fsk_symbol determines 2-FSK tone, iq_value determines IQ phase/amplitude
    iq_symbols.append(modulate(fsk_symbol, iq_value))

# Note: Symbols with same rep_map[i] carry same data bit (redundancy)
# This enables majority vote decoding with 37.5% erasures
```

## Next Steps
1. Run tournament to generate optimal 16 patterns
2. Validate all three orthogonality criteria meet targets
3. Test erasure recovery with simulated fading
4. Integrate with CASCADE modem encoder/decoder