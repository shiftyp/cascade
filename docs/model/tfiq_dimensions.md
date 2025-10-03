# Time-Frequency-IQ Dimensions (4D)

CASCADE achieves multi-user support through four dimensions of separation: Time, discrete Frequency hopping, and continuous IQ modulation. This 4D orthogonal space enables 280+ simultaneous users in 2.5 kHz bandwidth.

## Overview

Traditional radio systems use rigid time and frequency slots to separate users (TDMA, FDMA). CASCADE uses a 4-dimensional approach: 128 orthogonal patterns provide structure through discrete frequency-hopping sequences (Time × Frequency dimensions) while the model continuously optimizes IQ trajectories (I × Q dimensions) for maximum efficiency.

This hybrid approach combines:
- **Discrete frequency hopping** (FHSS - patent safe, like Bluetooth)
- **Continuous IQ modulation** (standard QAM/PSK)
- **4D orthogonality** (Time × Discrete Freq × Continuous I × Continuous Q)
- **Neural network optimization** (within discrete frequency grid)

## Four Dimensions of Separation

### 1. Time Dimension (32 symbols)

Each pattern consists of 32 time slots (symbols):
- Symbol duration: 50ms nominal (adaptive 40-60ms)
- Pattern duration: 1.6 seconds
- Asynchronous transmission (no time-slot coordination needed)
- Natural collision avoidance through pattern diversity

### 2. Frequency Dimension (70 discrete tones)

**Discrete frequency-hopping** - NOT continuous frequency modulation:

```python
# 70 reference tones (discrete grid)
REFERENCE_TONES = [
    # Lower: 300, 332, 364, ..., 1324, 1356, 1388 Hz (35 tones, indices 0-34)
    # BEACON: 1475-1625 Hz (150 Hz reserved, emergency at center 1550 Hz)
    # Upper: 1700, 1732, 1764, ..., 2660, 2692, 2788 Hz (35 tones, indices 35-69)
]

# Pattern hops among discrete tones
# Each symbol at EXACT reference frequency
# No interpolation or continuous sweeps

def pattern_frequency_hop(pattern_id, t):
    """
    Pattern defines discrete tone sequence
    Hops between exact frequencies (FHSS)
    """
    base_tone_idx = pattern.freq_sequence[t]  # Integer: 0-69
    frequency_hz = REFERENCE_TONES[base_tone_idx]  # Exact discrete value

    # Model can shift ±3 tones to avoid interference
    # But always stays on discrete grid
    shifted_tone_idx = model.select_discrete_tone(
        base_tone_idx,
        candidates=range(base_tone_idx-3, base_tone_idx+4),
        interference_state=measured_interference
    )

    return REFERENCE_TONES[shifted_tone_idx]  # Still discrete!

# This is frequency-hopping (FHSS), not chirping (CSS)
# Patent safe ✓
```

### 3. I Dimension (Continuous in-phase)

**Continuous trajectory** in I-component of IQ plane:

```python
def i_trajectory(pattern_id, t, complexity_lambda):
    """
    Continuous trajectory for I-component
    Smoothly adapts with complexity parameter
    """
    # Full complexity: Lissajous curve
    freq_a = (pattern_id % 7) + 1
    angle = 2 * np.pi * freq_a * t / 32
    offset = 2 * np.pi * pattern_id / 64
    i_full = np.cos(angle + offset)

    # Collapsed: Simple cosine
    i_collapsed = np.cos(2 * np.pi * pattern_id * t / 32)

    # Interpolate (continuous)
    i = (1 - complexity_lambda) * i_collapsed + complexity_lambda * i_full

    return i  # Continuous value in [-1, 1]
```

### 4. Q Dimension (Continuous quadrature)

**Continuous trajectory** in Q-component of IQ plane:

```python
def q_trajectory(pattern_id, t, complexity_lambda):
    """
    Continuous trajectory for Q-component
    Orthogonal to other patterns in IQ space
    """
    # Full complexity: Lissajous curve
    freq_b = (pattern_id % 5) + 1
    angle = 2 * np.pi * freq_b * t / 32
    offset = 2 * np.pi * pattern_id / 64
    q_full = np.sin(angle + offset)

    # Collapsed: Simple sine
    q_collapsed = np.sin(2 * np.pi * pattern_id * t / 32)

    # Interpolate (continuous)
    q = (1 - complexity_lambda) * q_collapsed + complexity_lambda * q_full

    return q  # Continuous value in [-1, 1]
```

## 4D Orthogonality

Patterns are orthogonal across all four dimensions simultaneously:

```python
def compute_4d_orthogonality(pattern_i, pattern_j):
    """
    Patterns must be <-30 dB orthogonal in 4D space
    """
    correlation = 0

    for t in range(32):  # Dimension 1: TIME
        # Dimension 2: FREQUENCY (discrete)
        tone_i = pattern_i.freq_sequence[t]  # Discrete index
        tone_j = pattern_j.freq_sequence[t]

        # If different discrete tones, no correlation
        if tone_i != tone_j:
            continue

        # Same discrete tone - check IQ orthogonality

        # Dimensions 3 & 4: I and Q (continuous)
        iq_i = complex(pattern_i.i_traj[t], pattern_i.q_traj[t])
        iq_j = complex(pattern_j.i_traj[t], pattern_j.q_traj[t])

        # Inner product in IQ space
        correlation += abs(iq_i * iq_j.conjugate())

    # Normalize
    normalized = correlation / 32
    correlation_db = 20 * np.log10(normalized + 1e-10)

    # Requirement
    assert correlation_db < -30, "Patterns not orthogonal"

    return correlation_db
```

## Discrete Frequency Hopping vs Continuous IQ

### Hybrid Discrete-Continuous Architecture

CASCADE combines discrete and continuous dimensions:

| Dimension | Type | Values | Adaptation | Neural Network |
|-----------|------|--------|------------|----------------|
| Time | Discrete | 32 symbols | Fixed structure | N/A |
| Frequency | Discrete | 70 tones | Model selects ±3 shift | Classification (Gumbel-softmax) |
| I (in-phase) | Continuous | [-1.5, +1.5] | Smooth trajectory | Regression |
| Q (quadrature) | Continuous | [-1.5, +1.5] | Smooth trajectory | Regression |

**Advantages:**
- Discrete frequency: Patent safe (FHSS not CSS), hardware-friendly, FFT-aligned
- Continuous IQ: Optimal modulation, smooth adaptation, high throughput

### Model Learning Strategy

```python
class FourDimensionalModel:
    """
    Model learns both discrete and continuous parameters
    """

    def forward(self, features, target_kernel):
        # DISCRETE: Tone selection (classification)
        tone_logits = self.tone_head(features)  # [68 scores]
        tone_probs = gumbel_softmax(tone_logits, temperature=1.0)
        selected_tone = sample(tone_probs)  # Discrete: 0-67

        # CONTINUOUS: IQ trajectory (regression)
        iq_basis = self.iq_head(features)  # Complex continuous

        # CONTINUOUS: Complexity parameter (regression)
        complexity_lambda = self.complexity_head(features)  # [0, 1]

        # DISCRETE: Pattern count (classification)
        pattern_count_logits = self.pattern_count_head(features)  # [1,2,3,4]
        num_patterns = argmax(pattern_count_logits)  # Discrete

        return {
            'tone_idx': selected_tone,  # Discrete
            'iq_basis': iq_basis,  # Continuous
            'complexity': complexity_lambda,  # Continuous
            'num_patterns': num_patterns,  # Discrete
        }

# Both discrete and continuous learning in single model
# Proven architecture (multimodal NNs)
```

## Multi-User Separation Strategy

### High SNR (>10 dB) - Maximum Capacity Mode

Operating characteristics:
- **Pattern Usage**: All 192 message patterns active
- **Tone Grid**: All 70 discrete tones available
- **IQ Complexity**: λ=0.4-0.6 typical (limited by HF multipath, NOT by SNR)
- **IQ Directions**: 4-6 directions (QPSK to 8-QAM level)
- **User Capacity**: 280+ simultaneous users
- **Throughput per User**: 80-320 bps (1-4 patterns)
- **Spectral Efficiency**: 84.5% utilization (70 tones of 2500 Hz)
- **Shannon Efficiency**: 56-60%
- **Note**: IQ limited by propagation (5ms multipath), not SNR. NVIS (λ=0.7-0.9) is exception.

### Medium SNR (0-10 dB) - Balanced Operation

As SNR degrades, IQ complexity further reduces (multipath already limiting at high SNR):

Operating characteristics:
- **Pattern Usage**: All 192 message patterns active
- **Tone Grid**: ~50-60 tones available (selective fading)
- **IQ Complexity**: λ=0.2-0.4 (limited by SNR now, propagation still factor)
- **IQ Directions**: 2-4 directions (BPSK to QPSK)
- **User Capacity**: 136-200 simultaneous users
- **Throughput per User**: 40-160 bps (1-4 patterns)
- **Shannon Efficiency**: 50-55%

### Low SNR (-10-0 dB) - Survival Mode

Operating characteristics:
- **Pattern Usage**: 128 patterns (with very simple IQ)
- **Tone Grid**: ~30-40 tones available
- **IQ Complexity**: λ=0.05-0.2 (nearly collapsed)
- **IQ Directions**: 2 directions (BPSK worth)
- **User Capacity**: 60-120 simultaneous users
- **Throughput per User**: 20-40 bps (1-2 patterns)
- **Shannon Efficiency**: 45-50%

### Very Low SNR (<-10 dB) - Emergency Fallback

Operating characteristics:
- **Pattern Usage**: Limited patterns (priority traffic only)
- **Tone Grid**: ~10-20 tones available (severe fading)
- **IQ Complexity**: λ=0.0 (fully collapsed, BPSK line only)
- **IQ Directions**: 1 direction (I-axis, phase-insensitive)
- **User Capacity**: 20-40 simultaneous users
- **Throughput per User**: 10-20 bps (single pattern, heavy FEC)
- **Shannon Efficiency**: 40-45%
- **Emergency priority**: Emergency traffic gets all resources
- **Note**: At this SNR, multipath irrelevant (noise dominates). IQ collapse due to SNR, not propagation.
- **Frequency Reuse**: Some overlap tolerated, compensated by pattern diversity
- **Error Correction**: Increased FEC overhead automatically applied

### Low SNR (-10-0 dB) - Survival Mode

In poor conditions, CASCADE prioritizes reliability over capacity. The model aggressively reduces complexity to maintain communication for essential users.

Operating characteristics:
- **Severe Clustering**: Only 4 pattern groups remain distinguishable
- **Minimal Mutation**: ±2% maximum to preserve orthogonality
- **User Capacity**: 3-5 simultaneous users maximum
- **Dense Spectrum Sharing**: Users overlap significantly in frequency
- **Time Separation**: Becomes primary separation mechanism
- **Heavy FEC**: Up to 3× redundancy for critical messages

### Very Low SNR (<-10 dB) - Binary Fallback

At extreme distances or in severe interference, CASCADE falls back to its most robust mode. This is similar to Morse code's ability to get through when voice fails.

Operating characteristics:
- **Binary Patterns**: Only 2 maximally separated patterns used
- **No Frequency Mutation**: Patterns used exactly as designed
- **Single User**: Generally supports only 1-2 users
- **Time Division**: Pure TDMA-like operation
- **Maximum Redundancy**: Each bit may be repeated many times
- **Coherent Integration**: Long symbol times for processing gain

## Model Learning Objectives

During training, the model must learn to balance multiple competing objectives. These objectives are carefully designed to produce a system that is both efficient and robust. The training process uses multi-objective optimization with carefully tuned weights to achieve the desired behavior.

The model learns to:

1. **Preserve Orthogonality**: Mutations shouldn't break pattern separation
   - Pattern modifications must maintain <-30 dB cross-correlation
   - Model learns safe mutation boundaries through gradient descent
   - Violations cause immediate loss penalty during training

2. **Maximize Capacity**: Pack users efficiently in spectrum
   - Learn optimal frequency spacing for given interference levels
   - Balance between tight packing (efficiency) and separation (reliability)
   - Discover natural frequency reuse patterns based on geographic separation

3. **Adapt to Interference**: Shift frequencies to avoid QRM
   - Detect narrowband interference through spectrum sensing
   - Learn to predict interference patterns from partial observations
   - Develop strategies for dynamic frequency hopping when needed

4. **Maintain Smoothness**: Gradual changes as SNR varies
   - Prevent abrupt mode switches that could lose synchronization
   - Learn natural hysteresis bands for stability
   - Ensure receiver can track changes without losing lock

```python
def tfp_training_loss(predicted_allocation, ground_truth):
    # Orthogonality preservation
    ortho_loss = measure_pattern_correlation(predicted_allocation)

    # Capacity maximization
    capacity_loss = -calculate_total_throughput(predicted_allocation)

    # Interference avoidance
    interference_loss = calculate_collision_rate(predicted_allocation)

    # Smoothness across SNR
    smooth_loss = measure_allocation_discontinuity(predicted_allocation)

    return ortho_loss + capacity_loss + interference_loss + smooth_loss
```

## Pattern Mutation Bounds

A critical aspect of CASCADE's design is learning how much patterns can be modified without breaking their orthogonality properties. Too much mutation and patterns interfere; too little and the system loses flexibility. The model discovers these bounds through training, learning different limits for different SNR conditions.

The mutation bounds aren't fixed constants but learned parameters that adapt based on experience. This allows the system to be conservative when first deployed, then gradually expand its operating envelope as it gains confidence. The bounds are also SNR-dependent - at low SNR, even small mutations can break communication, while at high SNR, significant modifications are tolerable.

The model learns appropriate mutation limits:

```python
class LearnedMutationBounds:
    """Model discovers safe mutation ranges"""

    def __init__(self):
        # Learned during training
        self.frequency_shift_limit = nn.Parameter(torch.tensor(0.1))
        self.phase_rotation_limit = nn.Parameter(torch.tensor(0.2))
        self.amplitude_scale_limit = nn.Parameter(torch.tensor(0.15))

    def apply_bounds(self, mutation, snr):
        """Ensure mutations stay within learned safe ranges"""

        # Tighter bounds at low SNR
        snr_factor = torch.sigmoid((snr + 10) / 10)  # 0 to 1

        # Scale bounds by SNR
        freq_bound = self.frequency_shift_limit * snr_factor
        phase_bound = self.phase_rotation_limit * snr_factor
        amp_bound = self.amplitude_scale_limit * snr_factor

        # Clip mutations
        mutation['frequency'] = torch.clamp(mutation['frequency'], -freq_bound, freq_bound)
        mutation['phase'] = torch.clamp(mutation['phase'], -phase_bound, phase_bound)
        mutation['amplitude'] = torch.clamp(mutation['amplitude'], -amp_bound, amp_bound)

        return mutation
```

## Integration with Protocol

The separation of concerns between protocol and model is crucial to CASCADE's architecture. The protocol makes discrete decisions about resource allocation (WHO gets WHICH patterns), while the model continuously optimizes how those resources are used (HOW patterns are transmitted). This clean separation ensures the system is both principled and flexible.

The protocol acts like a conductor assigning instruments to musicians, while the model is like the musicians themselves, deciding exactly how to play their parts. The protocol ensures fairness and prevents conflicts through discrete assignments, while the model ensures efficiency through continuous optimization.

The protocol provides pattern assignments, the model optimizes within them:

```python
# Protocol assigns patterns (discrete decision)
assigned_patterns = protocol.assign_pattern_pool(user_id)  # e.g., [4, 12, 20, 28]

# Model mutates and places optimally (continuous optimization)
for pattern_id in assigned_patterns:
    mutated_pattern = model.mutate_pattern(PATTERN_TABLE[pattern_id], channel_state)
    optimal_frequency = model.find_optimal_frequency(mutated_pattern, spectrum)

    # Use mutated pattern at optimal frequency
    transmit(mutated_pattern, optimal_frequency)
```

## Benefits of Pattern-Based "Slots"

The pattern-based approach offers significant advantages over traditional fixed-slot systems:

1. **Natural Structure**: Patterns provide organization without rigidity
   - Users naturally separate in pattern space without explicit coordination
   - System self-organizes based on interference patterns
   - No need for central scheduling or slot assignment protocols

2. **Flexibility**: Model can mutate within learned bounds
   - Adapt to real-time interference by shifting pattern characteristics
   - Optimize for specific channel conditions between station pairs
   - Gradually expand operating envelope as confidence grows

3. **Robustness**: Orthogonality preserved even with mutations
   - Mathematical guarantee of separation even with 30% pattern changes
   - Graceful degradation as patterns cluster in poor conditions
   - Natural fallback modes without protocol negotiation

4. **Efficiency**: No overhead for slot coordination
   - No beacon slots or control channels needed
   - No synchronization preambles or guard times
   - Every transmission carries useful data

5. **Adaptability**: Continuous optimization within discrete framework
   - Best of both worlds: structure from patterns, flexibility from learning
   - Can discover novel strategies not anticipated by designers
   - Improves over time through experience

## Real-World Analogy

CASCADE's TFIQ (4D) dimensions work like a city's transportation system:

- **Patterns** are like different types of vehicles (cars, buses, bikes) that naturally separate
- **Frequency** is like lanes on a highway that vehicles can smoothly change between
- **Time** is like the natural flow of traffic without rigid traffic light schedules

Just as experienced drivers learn to navigate efficiently without central control, CASCADE's model learns to optimize communication without rigid slot assignments. The patterns provide enough structure to prevent chaos (like having roads), while the continuous optimization allows efficient use of resources (like choosing the best route).

## See Also

- **[Pattern Architecture](pattern_architecture.md)** - The 128 orthogonal 4D patterns that enable TFIQ separation
- **[Signal Expert](experts.md#signal-expert-network)** - Multi-user separation implementation
- **[Spectrum Allocation Expert](experts.md#spectrum-allocation-expert-network)** - Frequency dimension optimization
- **[Model README](README.md)** - Overall architecture overview