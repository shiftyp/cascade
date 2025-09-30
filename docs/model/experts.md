# CASCADE Expert Networks

CASCADE's mixture-of-experts architecture employs five specialized neural networks, each focused on a specific aspect of adaptive radio communication. The [conductor network](conductor_details.md) dynamically weights these experts based on current channel conditions, enabling optimal performance across diverse scenarios.

For training strategy and data sources, see [Training Strategy](../training/README.md#three-stage-expert-training).

## Overview

The expert networks receive [shared 1024-dimensional features](shared_encoder.md) from the encoder and produce specialized 512-dimensional outputs. Each expert is trained on data tailored to its domain, developing distinct capabilities that complement the others.

| Expert | Focus | Output Dimension | Parameters | Computation |
|--------|-------|-----------------|------------|-------------|
| Noise Expert | QRN/QRM suppression | 512D | ~1M | ~2ms |
| Propagation Expert | Channel compensation | 512D | ~900K | ~2.5ms |
| Signal Expert | Multi-user separation | 512D | ~1.2M | ~3ms |
| Pattern Complexity Expert | Constellation adaptation | 512D | ~500K | ~1ms |
| Spectrum Allocation Expert | Frequency optimization | 512D | ~800K | ~2ms |

Total computation on Raspberry Pi 4: ~10.5ms (well within real-time constraints)

---

## Noise Expert Network

The Noise Expert specializes in identifying and suppressing radio noise (QRN/QRM) while preserving signal content. It employs spectral gating in the frequency domain to selectively attenuate interference while maintaining signal integrity.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
↓
Dense Layer: 1024 → 512
Activation: ReLU
Dropout: 0.1
↓
Spectral Gating Module:
  - FFT Transform: 512 → 512 (frequency domain)
  - Learned Gates: 512D (per-frequency suppression)
  - Gate Application: element-wise multiply
  - IFFT Transform: 512 → 512 (back to time)
↓
Residual Connection: Add input
↓
Dense Layer: 512 → 512
Activation: ReLU
BatchNorm: 512
↓
Output: 512D noise-suppressed features
```

The residual connection ensures that useful signal information is preserved even when aggressive noise suppression is applied. The learned spectral gates adapt to different interference patterns encountered during training.

### Learned Behaviors

**Frequency Selectivity**: The network learns which frequencies typically contain noise:
- Power line harmonics (50/60 Hz multiples)
- Switching noise bands
- Atmospheric noise emphasis at low frequencies

**Temporal Patterns**: The expert identifies different noise types through their temporal characteristics:
- **Burst noise**: Lightning strikes, switching transients (1-10ms duration)
- **Continuous interference**: Power line hum, carrier signals
- **Periodic noise**: Radar sweeps, beacon transmissions

**Adaptive Thresholding**: Suppression strength adapts to channel conditions:
- Aggressive suppression at low SNR to maximize signal recovery
- Gentle cleaning at high SNR to preserve signal fidelity
- Preserves weak signals near the noise floor through intelligent gating

### QRN/QRM Pattern Library

The expert is trained to recognize and suppress common interference patterns:

```python
# Natural Noise (QRN)
thunderstorm_pattern = {
    'type': 'impulsive',
    'frequency': 'broadband with LF emphasis',
    'duration': '1-10ms bursts',
    'suppression': 'blanking + prediction'
}

solar_noise_pattern = {
    'type': 'continuous',
    'frequency': 'HF enhancement',
    'duration': 'hours',
    'suppression': 'adaptive threshold'
}

# Man-Made Noise (QRM)
powerline_pattern = {
    'type': 'harmonic',
    'frequency': [50, 100, 150, ...] or [60, 120, 180, ...],
    'duration': 'continuous',
    'suppression': 'notch filters at harmonics'
}

switching_pattern = {
    'type': 'broadband hash',
    'frequency': 'all bands',
    'duration': 'continuous with switching rate',
    'suppression': 'median filtering'
}
```

### Training Strategy

**Data Sources** (see [Data Pipeline](../training/data_pipeline.md#phase-1-data-collection-months-1-18)):
- 10,000+ hours of WebSDR recordings containing real-world noise
- Synthetic QRN/QRM patterns generated from physics models
- Augmentation by mixing clean signals with diverse noise types

**Loss Function**:
```python
def noise_expert_loss(output, clean_signal, noisy_signal):
    # Preserve signal while removing noise
    signal_preservation = mse_loss(output, clean_signal)
    noise_suppression = -mse_loss(output, noisy_signal)
    return signal_preservation + 0.1 * noise_suppression
```

### Conductor Integration

The conductor assigns high weight to the Noise Expert when:
- Low SNR conditions detected (SNR < 0 dB)
- QRM interference identified in spectrum
- Atmospheric disturbances (thunderstorms, solar events)

**Typical weight range**: 0.05-0.5

### Performance Metrics

- **Noise Suppression**: 15-20 dB improvement in noise-limited scenarios
- **Signal Preservation**: >95% of signal power retained
- **Computation**: ~2ms on Raspberry Pi 4
- **Parameters**: ~1M

---

## Propagation Expert Network

The Propagation Expert specializes in estimating and compensating for channel propagation effects including fading, multipath, and Doppler shifts. It acts as an adaptive equalizer that learns to invert channel distortions.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
↓
Channel Estimation Module:
  Dense: 1024 → 512
  ReLU + Dropout(0.1)
  ↓
  Channel Parameters:
    - Fading coefficients: 512 → 100
    - Delay profile: 512 → 50
    - Doppler estimate: 512 → 1
↓
Channel Compensation Module:
  Build Inverse Filter:
    - Fading inverse: 1/H(f)
    - Multipath equalizer: Zero-forcing or MMSE
  Apply Compensation:
    - Features × Inverse_filter
↓
Adaptive Equalizer:
  Dense: 512 → 512
  ReLU
  Residual: Add input
↓
Output: 512D channel-compensated features
```

### Learned Behaviors

**Channel Estimation**: The network infers propagation characteristics from the received signal:
- **Fading depth**: Amplitude variation range
- **Coherence time**: Rate of channel change
- **Delay spread**: Extent of multipath dispersion

**Inverse Filtering**: Learns to undo various channel distortions:
- **Flat fading**: Simple gain and phase correction
- **Frequency-selective fading**: Per-frequency equalization
- **Time-varying channels**: Adaptive tracking filters

**Multipath Compensation**: Combines delayed signal copies constructively:
- **Constructive combining**: Similar to Rake receiver operation
- **Destructive cancellation**: Avoids frequency nulls
- **Delay estimation**: Handles delays up to 10ms

### Propagation Mode Models

The expert adapts its equalization strategy to different propagation modes:

```python
# Ionospheric Propagation
ionospheric_model = {
    'fading_rate': 0.1-1 Hz,
    'delay_spread': 1-5 ms,
    'doppler_shift': ±10 Hz,
    'compensation': 'slow_tracking_equalizer'
}

# Tropospheric Ducting
ducting_model = {
    'fading_rate': 0.01-0.1 Hz,
    'delay_spread': 10-50 ms,
    'doppler_shift': ±1 Hz,
    'compensation': 'long_equalizer'
}

# Ground Wave
ground_wave_model = {
    'fading_rate': ~0 Hz (stable),
    'delay_spread': <1 ms,
    'doppler_shift': 0 Hz,
    'compensation': 'static_correction'
}
```

### Equalizer Strategies

The expert implements multiple equalization approaches and selects the optimal one:

```python
def zero_forcing(channel_estimate):
    """Perfect inversion (can amplify noise at nulls)"""
    return 1.0 / channel_estimate

def mmse_equalizer(channel_estimate, snr):
    """Optimal trade-off between distortion and noise"""
    H = channel_estimate
    return H.conj() / (|H|^2 + 1/snr)

def dfe_equalizer(signal, decisions):
    """Decision-feedback using past symbols"""
    feedforward = fir_filter(signal)
    feedback = iir_filter(decisions)
    return feedforward - feedback
```

### Training with Real Propagation

The expert is trained on authentic propagation measurements from [FT8/WSPR data](../training/data_pipeline.md#collection-implementation-strategy):

```python
def train_on_ft8_wspr():
    """Use real propagation measurements for training"""
    ft8_reports = load_pskreporter_data()

    for report in ft8_reports:
        # Simulate CASCADE through same path
        simulated = propagate(cascade_signal,
                            report.tx_location,
                            report.rx_location,
                            report.timestamp)

        # Train to match observed characteristics
        loss = match_propagation(simulated, report.snr,
                                report.drift, report.spread)
```

### Fading Mitigation

Different fading types require different mitigation strategies:

**Slow Fading (Ionospheric)**:
- Track with Kalman filter
- Update every 100ms
- Predict next channel state

**Fast Fading (Mobile/Flutter)**:
- Wideband frequency averaging
- Diversity combining
- Time interleaving

**Deep Fades**:
- Cannot fully compensate when signal drops below noise
- Signals conductor to reduce expert weight
- Waits for conditions to improve

### Conductor Integration

The conductor assigns high weight to the Propagation Expert when:
- Multipath detected in delay profile
- Fading observed in signal amplitude
- Doppler shifts indicate channel motion

**Typical weight range**: 0.05-0.6

### Performance Metrics

- **Channel Estimation Error**: <3 dB typically
- **Multipath Compensation**: 10-15 dB improvement
- **Fading Mitigation**: 5-10 dB improvement
- **Computation**: ~2.5ms on Raspberry Pi 4
- **Parameters**: ~900K

---

## Signal Expert Network

The Signal Expert specializes in detecting, counting, and separating multiple simultaneous users. It employs attention mechanisms to isolate individual transmissions from overlapping signals in frequency, time, and pattern domains.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
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

### Learned Behaviors

**User Detection**: Distinguishes signal from noise through multiple indicators:
- Energy detection threshold
- Pattern correlation presence
- Temporal coherence check

**User Counting**: Estimates number of active transmissions (0-50 users):
- Confidence distribution over all possible counts
- Adapts to pattern diversity
- Handles both orthogonal and colliding patterns

**Signal Separation**: Isolates individual users through multiple dimensions:
- **Frequency diversity**: Different spectrum allocations
- **Pattern diversity**: Orthogonal constellation patterns
- **Time diversity**: Slot-based separation
- **Power diversity**: Near-far signal differences

### Multi-User Separation Strategy

The expert exploits all available separation dimensions:

```python
def separate_users(mixed_signal):
    """Exploit multiple dimensions for separation"""
    # Frequency domain separation
    freq_separated = frequency_demux(mixed_signal)

    # Pattern correlation separation
    pattern_separated = pattern_correlate(freq_separated)

    # Time slot extraction
    time_separated = time_slot_extract(pattern_separated)

    # Attention combines all dimensions
    return attention_combine([freq_separated,
                            pattern_separated,
                            time_separated])
```

### Collision Resolution

When patterns collide, the expert applies successive interference cancellation:

1. **Partial decode**: Extract strongest signal first
2. **Successive cancellation**: Remove decoded signal and repeat
3. **Joint decoding**: Decode multiple users simultaneously when beneficial

```python
def pattern_separate(signal, active_patterns):
    """Separate users by pattern correlation"""
    separated = []
    for pattern_id in active_patterns:
        # Correlate with known pattern
        correlation = correlate(signal, PATTERN_TABLE[pattern_id])

        # Extract pattern-specific signal
        user_signal = correlation * PATTERN_TABLE[pattern_id].conj()
        separated.append(user_signal)

    return separated
```

### Capacity Analysis

**Theoretical Maximum**:
- 64 patterns available → 64 users with perfect orthogonality
- With clustering: 16 users at medium SNR
- Practical limit: 50 users with time slot assistance

**Degradation Profile**:
| Users | Throughput Impact |
|-------|------------------|
| 1-10  | No degradation |
| 11-25 | <5% reduction |
| 26-50 | <15% reduction |
| 50+   | Graceful fallback to time slots |

### Training Approach

Synthetic multi-user scenarios with varying numbers of users, power levels, and overlap. **Critically, each user receives independent propagation and station embeddings** to simulate realistic conditions (see [Multi-User Training](../training/embedding_models.md#multi-user-training-critical)):

```python
def generate_training_scenario():
    """Generate realistic multi-user scenario with per-user propagation"""
    num_users = random.randint(1, 50)
    mixed_signal = np.zeros(signal_length)

    for i in range(num_users):
        # Each user gets INDEPENDENT propagation conditions
        user = {
            'pattern': random.choice(range(64)),
            'frequency': random.uniform(0, 2500),
            'power': random.uniform(-20, 10),  # dB
            'timing_offset': random.uniform(0, 1),
            'propagation_emb': sample_embedding('propagation'),  # Independent!
            'station_emb': sample_embedding('station')           # Independent!
        }

        # Generate and augment signal with THIS user's conditions
        user_signal = generate_cascade_transmission(user['data'])
        user_augmented = apply_propagation(user_signal, user['propagation_emb'], user['station_emb'])

        # Add to mix
        mixed_signal += user_augmented * (10 ** (user['power'] / 20))

    # Shared noise applied once to entire mix
    return add_noise(mixed_signal, sample_embedding('noise'))
```

**Loss Function**:
```python
def signal_expert_loss(predicted_count, predicted_signals,
                       true_count, true_signals):
    count_loss = cross_entropy(predicted_count, true_count)

    separation_loss = 0
    for pred, true in zip(predicted_signals, true_signals):
        separation_loss += mse_loss(pred, true)

    return count_loss + separation_loss / true_count
```

### Conductor Integration

The conductor assigns high weight to the Signal Expert when:
- Multiple users detected in spectrum
- Pattern collisions observed
- Multi-user interference present

**Typical weight range**: 0.1-0.4

### Performance Metrics

- **Detection Accuracy**: >99% for SNR > -10 dB
- **Counting Accuracy**: ±1 user for <20 users
- **Separation Quality**: >20 dB isolation between users
- **Computation**: ~3ms on Raspberry Pi 4
- **Parameters**: ~1.2M

---

## Pattern Complexity Expert Network

The Pattern Complexity Expert determines optimal constellation complexity (64/16/4/2 patterns) based on channel conditions. It implements CASCADE's adaptive modulation strategy, collapsing the constellation gracefully as SNR degrades.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
↓
SNR Estimation Branch:
  Dense: 1024 → 256 → 64 → 1
  Output: Estimated SNR (dB)
↓
Complexity Decision Branch:
  Dense: 1024 → 512 → 256 → 4
  Softmax
  Output: P(complexity = [64, 16, 4, 2])
↓
Pattern Adaptation Module:
  If complexity == 64: Full constellation
  If complexity == 16: Merge 4 patterns → 1 cluster
  If complexity == 4: Merge 16 patterns → 1 cluster
  If complexity == 2: Binary mode
↓
Feature Adaptation:
  Dense: 256 + complexity_params → 512
  ReLU + BatchNorm
↓
Output: Complexity decision + 512D adapted features
```

### Learned Behaviors

**SNR Assessment**: Accurate channel quality estimation combining multiple indicators:
- Signal power estimation
- Noise floor measurement
- Interference type and level
- Future SNR trend prediction

**Complexity Selection**: The network discovers optimal thresholds through training rather than using hard-coded values:

```python
def select_complexity(snr_estimate):
    """Learned thresholds (not hard-coded)"""
    if snr_estimate > 12:    # Learned threshold
        return 64  # Full constellation
    elif snr_estimate > 2:    # Learned threshold
        return 16  # Medium complexity
    elif snr_estimate > -8:   # Learned threshold
        return 4   # Low complexity
    else:
        return 2   # Binary mode
```

**Graceful Degradation**: Smooth transitions between complexity levels:
- Hysteresis prevents oscillation between modes
- Soft boundaries rather than hard switches
- Predictive adaptation based on SNR trends

### Constellation Collapse Mechanism

The pattern clustering strategy at each complexity level:

```python
def get_cluster_center(pattern_id, complexity):
    """Determine which pattern represents a cluster"""
    if complexity == 64:
        return pattern_id  # No clustering

    elif complexity == 16:
        # 4 patterns per cluster
        cluster_id = pattern_id // 4
        return cluster_id * 4 + 2  # Center pattern

    elif complexity == 4:
        # 16 patterns per cluster
        cluster_id = pattern_id // 16
        return cluster_id * 16 + 8  # Center pattern

    else:  # complexity == 2
        # Binary: northern vs southern hemisphere
        return 16 if pattern_id < 32 else 48
```

### Shannon Efficiency Targets

The expert learns to achieve high efficiency at each complexity level:

- **64-pattern mode**: 93% of Shannon limit (SNR > 12 dB)
- **16-pattern mode**: 88% of Shannon limit (SNR 2-12 dB)
- **4-pattern mode**: 85% of Shannon limit (SNR -8 to 2 dB)
- **2-pattern mode**: 83% of Shannon limit (SNR < -8 dB)

This enables communication across a 40 dB dynamic range from +15 dB to -25 dB.

### Adaptation Strategy

**Predictive Complexity**:
```python
def predict_future_complexity(snr_history):
    """Anticipate future SNR changes"""
    # Analyze SNR trend
    trend = linear_fit(snr_history[-10:])

    if trend.slope < -1:  # Rapidly worsening
        # Switch down early
        return current_complexity // 2

    elif trend.slope > 1:  # Rapidly improving
        # Consider switching up
        return current_complexity * 2

    else:
        # Maintain current
        return current_complexity
```

**Hysteresis Prevention**:
```python
def apply_hysteresis(new_complexity, current_complexity):
    """Require margin to switch modes"""
    # Require 3 dB margin to prevent oscillation
    if new_complexity > current_complexity:
        if snr < threshold + 3:
            return current_complexity  # Don't switch up yet

    elif new_complexity < current_complexity:
        if snr > threshold - 3:
            return current_complexity  # Don't switch down yet

    return new_complexity
```

### Training Approach

**Objective Function**:
```python
def complexity_loss(predicted_complexity, snr, achieved_rate):
    """Maximize throughput while maintaining reliability"""
    shannon_limit = bandwidth * log2(1 + snr)
    efficiency = achieved_rate / shannon_limit

    # Penalize over-optimistic complexity
    if not decoded_successfully:
        penalty = 10.0
    else:
        penalty = 0.0

    return -efficiency + penalty
```

**Curriculum Learning**: Train on progressively harder scenarios:
1. **Stage 1**: Static SNR conditions
2. **Stage 2**: Slowly varying SNR
3. **Stage 3**: Rapid fading
4. **Stage 4**: Mixed multi-user scenarios

### Conductor Integration

The conductor assigns high weight to the Pattern Complexity Expert when:
- Variable SNR conditions
- Mode transitions needed
- Channel quality rapidly changing

**Typical weight range**: 0.1-0.3

### Performance Metrics

- **SNR Estimation Error**: <2 dB typically
- **Mode Selection Accuracy**: >95%
- **Throughput Efficiency**: 83-93% of Shannon limit
- **Computation**: ~1ms on Raspberry Pi 4
- **Parameters**: ~500K

---

## Spectrum Allocation Expert Network

The Spectrum Allocation Expert analyzes spectrum usage, identifies optimal frequency allocations, and enables efficient multi-user packing within the 2.5 kHz bandwidth. It works within protocol-assigned pattern pools to optimize frequency placement.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
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

### Learned Behaviors

**Spectrum Sensing**: Identifies occupied vs available spectrum:
- Energy detection per frequency bin
- Pattern signature recognition
- Interference type classification

**User Localization**: Maps users to frequency regions:
- Associates specific patterns with frequencies
- Tracks user movement in spectrum over time
- Predicts future allocation needs

**Packing Optimization**: Maximizes spectrum efficiency:
- Fills spectrum holes to minimize wasted bandwidth
- Minimizes guard bands between users
- Adapts spacing to individual user requirements

### Spectrum Allocation Strategy

The expert optimizes frequency placement within protocol-assigned constraints:

**Protocol Constraints (Input)**:
```python
protocol_constraints = {
    'assigned_patterns': [4, 12, 20, 28],  # From protocol layer
    'bandwidth_limit': 2500,  # Hz total
    'priority': 'NORMAL'
}
```

**Model Optimization (Output)**:
```python
allocation = {
    'frequency_start': 1250,  # Hz
    'frequency_width': 150,   # Hz
    'guard_bands': [10, 10],  # Hz on each side
    'patterns_to_use': [4, 20]  # Subset of assigned
}
```

### Multi-User Packing

The expert packs users efficiently based on their link quality:

```python
def pack_users_by_snr(users):
    """Allocate spectrum based on user SNR"""
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

The expert can reallocate spectrum for priority traffic:

```python
def reallocate_for_emergency():
    """Make room for emergency transmission"""
    emergency_bw = 500  # Hz needed

    # Find least important users
    victims = find_lowest_priority_users(emergency_bw)

    # Move them to time slots
    for user in victims:
        user.allocation = 'time_slot_1'

    # Allocate spectrum to emergency
    return allocate_spectrum(0, emergency_bw)
```

### Interference Avoidance

**QRM Detection and Avoidance**:
```python
def avoid_interference(spectrum):
    """Find clean spectrum regions"""
    qrm_bands = detect_qrm(spectrum)

    available = []
    for freq in range(0, 2500, 10):
        if not any(is_in_band(freq, qrm) for qrm in qrm_bands):
            available.append(freq)

    return available
```

**Pattern-Frequency Optimization**:
```python
def optimize_pattern_frequency(pattern_id, spectrum):
    """Find optimal frequency for specific pattern"""
    # Some patterns work better at certain frequencies
    pattern_freq_score = np.zeros(2500)

    for freq in range(2500):
        # Evaluate pattern performance at this frequency
        score = evaluate_pattern_at_freq(pattern_id, freq, spectrum)
        pattern_freq_score[freq] = score

    return np.argmax(pattern_freq_score)
```

### Capacity Analysis

**Theoretical Capacity**:
```python
def calculate_capacity(allocations):
    """Compute total system capacity"""
    total_capacity = 0

    for alloc in allocations:
        snr = get_link_snr(alloc.user)
        bandwidth = alloc.bandwidth

        # Shannon capacity
        capacity = bandwidth * np.log2(1 + snr)
        total_capacity += capacity

    return total_capacity
```

**Practical Limits**:
| Scenario | Users | Spectrum Efficiency |
|----------|-------|-------------------|
| All strong (>10 dB) | 50 | 95% utilization |
| Mixed SNR | 30 | 85% utilization |
| All weak (<-10 dB) | 10 | 60% utilization |

### Learning from Experience

**Pattern Success Tracking**:
```python
def update_pattern_frequency_success(pattern, freq, success):
    """Learn which patterns work at which frequencies"""
    pattern_freq_matrix[pattern][freq] *= 0.9  # Decay
    pattern_freq_matrix[pattern][freq] += 0.1 * success
```

**Collision Prediction**:
```python
def predict_collision_probability(user1, user2):
    """Predict likelihood of pattern collision"""
    if overlapping_frequency(user1, user2):
        if same_pattern_cluster(user1, user2):
            return 0.8  # High collision risk
        else:
            return 0.2  # Low risk (orthogonal)
    return 0.0  # No overlap
```

### Protocol Integration

The expert works within protocol-defined constraints:

**Protocol Assigns Pools**:
```python
# Protocol decision (discrete pattern assignment)
pattern_pool = [0, 8, 16, 24, 32, 40, 48, 56]
```

**Model Selects Within Pool**:
```python
# Model optimization (continuous frequency allocation)
selected = model.select_best_patterns(
    pool=pattern_pool,
    spectrum=current_spectrum,
    target_snr=link_snr
)
# Returns: [8, 24, 40]  # Best 3 from pool for current conditions
```

### Conductor Integration

The conductor assigns high weight to the Spectrum Allocation Expert when:
- Dense spectrum usage with many active users
- Interference detected requiring frequency adjustment
- Multi-user coordination needed

**Typical weight range**: 0.05-0.3

### Performance Metrics

- **Spectrum Utilization**: 85-95% typical
- **Packing Efficiency**: Up to 50 users
- **Allocation Speed**: <5ms for 20 users
- **Computation**: ~2ms on Raspberry Pi 4
- **Parameters**: ~800K

---

## Cross-Expert Coordination

The five experts work together under conductor control, with their combined outputs weighted and merged:

```python
def combine_expert_outputs(experts, weights):
    """Weighted combination of expert features"""
    combined = torch.zeros(512)

    for expert, weight in zip(experts, weights):
        combined += weight * expert.output

    return combined
```

The conductor learns optimal weighting strategies through training on diverse scenarios, adapting the mixture dynamically to current conditions. This enables CASCADE to leverage the right expertise at the right time, achieving robust performance across all channel conditions.

## See Also

- **[Conductor Details](conductor_details.md)** - Deep dive into conductor network architecture and weighting strategies
- **[Shared Encoder](shared_encoder.md)** - How the 1024D input features are generated
- **[Training Strategy](../training/README.md#three-stage-expert-training)** - Three-stage training approach for experts and conductor
- **[Model README](README.md)** - Overall model architecture and responsibilities