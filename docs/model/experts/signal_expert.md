# Signal Expert Network

Specializes in detecting, counting, and separating multiple simultaneous users.

## Architecture

```
Input: 1024D shared features
↓
User Detection Branch:
  Dense: 1024 → 256 → 64 → 1
  Activation: Sigmoid
  Output: P(user_exists)
↓
User Counting Branch:
  Dense: 1024 → 512 → 256 → 51
  Activation: Softmax
  Output: P(N_users = 0...50)
↓
Separation Module (per user slot):
  Attention: Query(512) × Key(512) → Attention_weights
  Weighted Features: Attention_weights × Value(512)
  Dense: 512 → 512
  LayerNorm: 512
↓
Output: 512D × N user-specific features
```

## Learned Behaviors

### User Detection
Distinguishes signal from noise:
- Energy detection threshold
- Pattern correlation presence
- Temporal coherence check

### User Counting
Estimates active transmissions:
- 0-50 simultaneous users
- Confidence per count
- Adapts to pattern diversity

### Signal Separation
Isolates individual users through:
- **Frequency diversity**: Different spectrum allocations
- **Pattern diversity**: Orthogonal constellation patterns
- **Time diversity**: Slot-based separation
- **Power diversity**: Near-far differences

## Multi-User Strategies

### Separation Dimensions
```python
def separate_users(mixed_signal):
    # Exploit multiple dimensions
    freq_separated = frequency_demux(mixed_signal)
    pattern_separated = pattern_correlate(freq_separated)
    time_separated = time_slot_extract(pattern_separated)

    # Attention combines all dimensions
    return attention_combine([freq_separated,
                            pattern_separated,
                            time_separated])
```

### Collision Resolution
When patterns collide:
1. **Partial decode**: Extract strongest signal
2. **Successive cancellation**: Remove and repeat
3. **Joint decoding**: Decode multiple together

## Pattern-Based Separation

### Orthogonality Exploitation
```python
def pattern_separate(signal, active_patterns):
    separated = []
    for pattern_id in active_patterns:
        # Correlate with known pattern
        correlation = correlate(signal, PATTERN_TABLE[pattern_id])

        # Extract pattern-specific signal
        user_signal = correlation * PATTERN_TABLE[pattern_id].conj()
        separated.append(user_signal)

    return separated
```

### Cross-Pattern Interference
Even when patterns share clusters:
- Residual differences enable separation
- Phase diversity helps
- Frequency differences assist

## Capacity Limits

### Theoretical Maximum
- **64 patterns available**: 64 users if no collision
- **With clustering**: 16 users at medium SNR
- **Practical limit**: 50 users with time slots

### Degradation Profile
| Users | Throughput Impact |
|-------|------------------|
| 1-10  | No degradation |
| 11-25 | <5% reduction |
| 26-50 | <15% reduction |
| 50+   | Graceful fallback to time slots |

## Training Approach

### Synthetic Multi-User Scenarios
```python
def generate_training_scenario():
    num_users = random.randint(1, 50)
    users = []

    for i in range(num_users):
        user = {
            'pattern': random.choice(range(64)),
            'frequency': random.uniform(0, 2500),
            'power': random.uniform(-20, 10),  # dB
            'timing_offset': random.uniform(0, 1)
        }
        users.append(user)

    return mix_users(users)
```

### Loss Function
```python
def signal_expert_loss(predicted_count, predicted_signals,
                       true_count, true_signals):
    count_loss = cross_entropy(predicted_count, true_count)

    separation_loss = 0
    for pred, true in zip(predicted_signals, true_signals):
        separation_loss += mse_loss(pred, true)

    return count_loss + separation_loss / true_count
```

## Integration with Conductor

The conductor weights this expert based on:
- **High weight**: Multiple users detected, collisions
- **Low weight**: Single user, clean channel
- **Typical range**: 0.1-0.4

## Performance Metrics

- **Detection Accuracy**: >99% for SNR > -10 dB
- **Counting Accuracy**: ±1 user for <20 users
- **Separation Quality**: >20 dB isolation
- **Computation**: ~3ms on Raspberry Pi 4
- **Parameters**: ~1.2M