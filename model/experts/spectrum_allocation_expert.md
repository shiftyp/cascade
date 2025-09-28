# Spectrum Allocation Expert Network

Analyzes spectrum usage, identifies optimal frequency allocations, and enables efficient multi-user packing.

## Architecture

```
Input: 1024D shared features
↓
Frequency Analysis:
  FFT: 1024 → 1024 (frequency domain)
  ↓
  Spectrum Attention:
    Multi-head attention (8 heads)
    Query/Key/Value: 1024 → 128 per head
    Concatenate: 8 × 128 → 1024
↓
Allocation Detection:
  Conv1D: 1024 → 512 (kernel=3)
  ReLU
  Conv1D: 512 → 256 (kernel=3)
  ReLU
  ↓
  User-Frequency Mapping:
    Dense: 256 → (N_users × N_freq_slots)
    Reshape: User × Frequency allocation matrix
↓
Spectrum Efficiency Module:
  Compute utilization metrics
  Identify spectrum holes
  Suggest reallocation
↓
Output: 512D spectrum-aware features + allocation map
```

## Learned Behaviors

### Spectrum Sensing
Identifies occupied vs available spectrum:
- Energy detection per frequency bin
- Pattern signature recognition
- Interference classification

### User Localization
Maps users to frequency regions:
- Associates patterns with frequencies
- Tracks user movement in spectrum
- Predicts future allocations

### Packing Optimization
Maximizes spectrum efficiency:
- Fills spectrum holes
- Minimizes guard bands
- Adapts to user requirements

## Spectrum Allocation Strategy

### Protocol Constraints (Input)
```python
protocol_constraints = {
    'assigned_patterns': [4, 12, 20, 28],  # From protocol
    'bandwidth_limit': 2500,  # Hz total
    'priority': 'NORMAL'
}
```

### Model Optimization (Output)
```python
allocation = {
    'frequency_start': 1250,  # Hz
    'frequency_width': 150,   # Hz
    'guard_bands': [10, 10],  # Hz on each side
    'patterns_to_use': [4, 20]  # Subset of assigned
}
```

## Multi-User Packing

### Packing by Link Quality
```python
def pack_users_by_snr(users):
    # Sort by link quality
    users.sort(key=lambda u: u.link_snr, reverse=True)

    allocations = []
    spectrum_map = np.zeros(2500)  # Hz

    for user in users:
        if user.link_snr > 10:
            # Strong: pack densely, narrow bandwidth
            bandwidth = 50
            guard = 5
        elif user.link_snr > 0:
            # Medium: moderate spacing
            bandwidth = 150
            guard = 25
        else:
            # Weak: wide spacing, more bandwidth
            bandwidth = 400
            guard = 50

        # Find available slot
        slot = find_spectrum_hole(spectrum_map,
                                bandwidth + 2*guard)

        allocations.append({
            'user': user.id,
            'freq': slot,
            'bandwidth': bandwidth
        })

        # Mark as occupied
        spectrum_map[slot:slot+bandwidth+2*guard] = 1

    return allocations
```

### Dynamic Reallocation
```python
def reallocate_for_emergency():
    # Emergency takes priority
    emergency_bw = 500  # Hz needed

    # Find least important users
    victims = find_lowest_priority_users(emergency_bw)

    # Move them to time slots
    for user in victims:
        user.allocation = 'time_slot_1'

    # Allocate spectrum to emergency
    return allocate_spectrum(0, emergency_bw)
```

## Interference Avoidance

### QRM Detection and Avoidance
```python
def avoid_interference(spectrum):
    qrm_bands = detect_qrm(spectrum)

    available = []
    for freq in range(0, 2500, 10):
        if not any(is_in_band(freq, qrm) for qrm in qrm_bands):
            available.append(freq)

    return available
```

### Pattern-Frequency Optimization
```python
def optimize_pattern_frequency(pattern_id, spectrum):
    # Some patterns work better at certain frequencies
    pattern_freq_score = np.zeros(2500)

    for freq in range(2500):
        # Evaluate pattern performance at this frequency
        score = evaluate_pattern_at_freq(pattern_id, freq, spectrum)
        pattern_freq_score[freq] = score

    return np.argmax(pattern_freq_score)
```

## Capacity Analysis

### Theoretical Capacity
```python
def calculate_capacity(allocations):
    total_capacity = 0

    for alloc in allocations:
        snr = get_link_snr(alloc.user)
        bandwidth = alloc.bandwidth

        # Shannon capacity
        capacity = bandwidth * np.log2(1 + snr)
        total_capacity += capacity

    return total_capacity
```

### Practical Limits
| Scenario | Users | Spectrum Efficiency |
|----------|-------|-------------------|
| All strong (>10 dB) | 50 | 95% utilization |
| Mixed SNR | 30 | 85% utilization |
| All weak (<-10 dB) | 10 | 60% utilization |

## Learning from Patterns

### Pattern Success Tracking
```python
def update_pattern_frequency_success(pattern, freq, success):
    # Learn which patterns work at which frequencies
    pattern_freq_matrix[pattern][freq] *= 0.9  # Decay
    pattern_freq_matrix[pattern][freq] += 0.1 * success
```

### Collision Prediction
```python
def predict_collision_probability(user1, user2):
    # Based on historical data
    if overlapping_frequency(user1, user2):
        if same_pattern_cluster(user1, user2):
            return 0.8  # High collision risk
        else:
            return 0.2  # Low risk (orthogonal)
    return 0.0  # No overlap
```

## Integration with Protocol

### Protocol Assigns Pools
```python
# Protocol decision (discrete)
pattern_pool = [0, 8, 16, 24, 32, 40, 48, 56]
```

### Model Selects Within Pool
```python
# Model optimization (continuous)
selected = model.select_best_patterns(
    pool=pattern_pool,
    spectrum=current_spectrum,
    target_snr=link_snr
)
# Returns: [8, 24, 40]  # Best 3 from pool
```

## Integration with Conductor

The conductor weights this expert based on:
- **High weight**: Dense spectrum, many users
- **Low weight**: Sparse spectrum, few users
- **Typical range**: 0.05-0.3

## Performance Metrics

- **Spectrum Utilization**: 85-95% typical
- **Packing Efficiency**: 50 users maximum
- **Allocation Speed**: <5ms for 20 users
- **Computation**: ~2ms on Raspberry Pi 4
- **Parameters**: ~800K