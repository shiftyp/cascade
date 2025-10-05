# CASCADE Pattern Generation Status

## Current Implementation

Optimized pattern generator for CASCADE HF modem with QR-like erasure coding:

### Pattern Structure
- **Total**: 16 patterns (reused across 67 frequency pairs = 1,072 logical channels)
- **Length**: 512 symbols (2.56 seconds at 200 symbols/second)
- **Modulation**: 2-FSK (binary frequency sequences)
- **Repetition Map**: 128 unique data positions, each repeated 4x
- **Erasure Tolerance**: 37.5% (need only 320 of 512 symbols to decode)

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
1. **Initial patterns**: Diverse initialization (sparse, dense, alternating, random)
2. **Mutation**: 2-100 bit flips based on temperature
3. **Triple evaluation**:
   - Normal correlation: Full cross-correlation (1,023 shifts)
   - Flip correlation: Inverted pattern correlation
   - Erasure correlation: 5 trials with 37.5% random dropout
4. **Global update**: Every 10 iterations across all 120 pattern pairs

### Execution Schedule
- **First run**: 50,000 iterations minimum per trial
- **Subsequent**: 10,000 iteration evaluation intervals
- **Total budget**: 3.2M iterations across 8 trials
- **68x faster** than 128-pattern approach (120 pairs vs 8,128 pairs)

### Computational Complexity
- **Per iteration**: ~0.02 seconds (50 iter/sec)
- **50k iterations**: ~17 minutes per trial
- **3.2M total**: ~18 hours on 8 P-cores

### Output Format
```python
{
    'patterns': [array([0,1,1,0,...]), ...],  # 16 binary patterns
    'repetition_maps': [array([45,12,78,...]), ...],  # Data position indices (0-127)
    'num_patterns': 16,
    'pattern_length': 512,
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

patterns = data['patterns']
repetition_maps = data['repetition_maps']

# Encode data on pattern 0
pattern = patterns[0]  # 512 binary symbols (2-FSK tones)
rep_map = repetition_maps[0]  # Which symbols carry same data

user_data = [1, 0, 1, 1, ...]  # 128 bits of user data
iq_symbols = []

for i, fsk_symbol in enumerate(pattern):
    data_index = rep_map[i]  # Which data bit (0-127)
    iq_value = user_data[data_index]  # Get the data bit
    # Modulate: fsk_symbol determines tone, iq_value determines phase/amplitude
    iq_symbols.append(modulate(fsk_symbol, iq_value))
```

## Next Steps
1. Run tournament to generate optimal 16 patterns
2. Validate all three orthogonality criteria meet targets
3. Test erasure recovery with simulated fading
4. Integrate with CASCADE modem encoder/decoder