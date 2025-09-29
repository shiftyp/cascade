# Time-Frequency-Pattern Dimensions

CASCADE achieves multi-user support through three dimensions of separation. The pattern dimension provides natural "slots" that the model can mutate and optimize based on conditions.

## Overview

Traditional radio systems use rigid time and frequency slots to separate users - think of TDMA (Time Division Multiple Access) or FDMA (Frequency Division Multiple Access) in cellular networks. CASCADE takes a fundamentally different approach: it uses 64 orthogonal patterns as a foundation for separation, but allows the model to continuously optimize how these patterns are used in time and frequency.

This approach combines the robustness of fixed orthogonal patterns with the flexibility of learned optimization. The patterns act like "soft boundaries" - they provide structure and guarantee separation, but the model can adjust them based on real-time conditions. This is similar to how a jazz ensemble has standard chord progressions but musicians can improvise within them.

## Three Dimensions of Separation

### 1. Pattern Dimension (Primary)

The pattern dimension is CASCADE's foundational innovation. Unlike traditional systems that might use different modulation schemes or coding rates, CASCADE uses 64 mathematically orthogonal patterns that naturally don't interfere with each other. Think of these patterns like 64 different "languages" - even if multiple people speak simultaneously, you can still understand the language you're listening for.

Key characteristics:
- **Fixed Foundation**: The 64 patterns are pre-computed and never change, providing a stable foundation
- **Natural Separation**: Each pattern has minimal correlation with others (below -30 dB cross-correlation)
- **Implicit Slots**: Patterns naturally cluster in frequency space, creating soft boundaries
- **Learned Mutations**: The model can adjust patterns by up to 30% while maintaining orthogonality
- **Hierarchical Structure**: Patterns cluster into 16 groups, then 4, then 2, enabling graceful degradation

### 2. Frequency Dimension (Continuous)

Traditional systems divide spectrum into fixed channels (like TV channels or WiFi channels). CASCADE instead treats frequency as a continuous resource that the model optimizes in real-time. This is like the difference between parking in marked spaces versus finding the optimal spot in an open field.
```python
# Patterns suggest natural frequency centers
pattern_freq_centers = [p.dominant_frequency() for p in PATTERN_TABLE]

# Model mutates these based on conditions
def adapt_frequency(pattern_id, channel_state):
    base_freq = pattern_freq_centers[pattern_id]
    # Model learns optimal shift/spread
    mutation = model.frequency_mutation(pattern_id, channel_state)
    return base_freq + mutation  # Continuous, not slotted
```

### 3. Time Dimension (Fragment-based)

Time separation in CASCADE emerges naturally from the model's decisions rather than being pre-scheduled. Traditional TDMA systems might give each user specific millisecond slots; CASCADE instead learns when to transmit based on channel conditions and traffic patterns. The model creates fragments between 0.5 and 5 seconds, optimizing duration based on channel coherence time and interference patterns.

This approach means:
- **No Synchronization Overhead**: Stations don't need GPS or network time
- **Natural Collision Avoidance**: Pattern diversity handles simultaneous transmissions
- **Adaptive Duration**: Short fragments in volatile conditions, long in stable
- **Opportunistic Access**: Model identifies and uses quiet periods

## Pattern-Defined "Slots"

While the model operates continuously, patterns create natural clustering. This section explains how patterns provide structure without rigidity - a key innovation that enables both robustness and flexibility.

```python
class PatternSlots:
    """Patterns define soft boundaries that model can mutate"""

    def __init__(self):
        # 64 patterns suggest 64 soft "slots"
        self.pattern_centers = self.compute_pattern_centers()

    def compute_pattern_centers(self):
        """Each pattern has natural frequency emphasis"""
        centers = []
        for pattern in PATTERN_TABLE:
            # Patterns use different frequency bins
            center = np.average(pattern.active_frequencies)
            centers.append(center)
        return centers

    def model_mutation(self, pattern_id, snr, interference):
        """Model mutates pattern within learned bounds"""
        base = PATTERN_TABLE[pattern_id]

        # Model learns how much to mutate
        if snr > 10:
            # High SNR: More mutation allowed
            mutation_range = 0.3
        elif snr > 0:
            # Medium SNR: Moderate mutation
            mutation_range = 0.1
        else:
            # Low SNR: Minimal mutation (preserve orthogonality)
            mutation_range = 0.02

        # Apply learned mutation
        mutated = model.mutate_pattern(base, mutation_range)
        return mutated
```

## Continuous Frequency Distribution

One of CASCADE's most important departures from traditional radio systems is its continuous frequency distribution. Rather than assigning users to fixed channels or subcarriers, the model learns to place each user at the optimal frequency based on current interference, propagation, and traffic patterns.

This continuous approach has several advantages:
- **Interference Avoidance**: The model can shift frequencies by just a few Hz to avoid narrowband interference
- **Optimal Packing**: Users can be placed at any frequency to maximize total capacity
- **Smooth Adaptation**: Frequencies can shift gradually as conditions change
- **No Guard Bands**: Pattern orthogonality eliminates the need for frequency guard bands

The model learns to distribute users across spectrum:

```python
def learned_frequency_distribution(users, spectrum_state):
    """Model decides frequency allocation continuously"""

    allocations = []
    for user in users:
        # Start with pattern's natural frequency
        pattern_id = user.assigned_patterns[0]
        base_freq = pattern_freq_centers[pattern_id]

        # Model optimizes placement
        optimal_freq = model.optimize_frequency(
            base_freq,
            spectrum_state,
            user.snr,
            user.priority
        )

        allocations.append({
            'user': user,
            'frequency': optimal_freq,  # Continuous value
            'bandwidth': model.optimal_bandwidth(user.snr)
        })

    return allocations
```

## Multi-User Separation Strategy

CASCADE's approach to multi-user support adapts dramatically based on signal conditions. At high SNR, the system can support up to 50 simultaneous users through pattern diversity alone. As conditions degrade, the model intelligently reduces capacity while maintaining reliability for active users. This section details how the separation strategy evolves across different SNR regimes.

### High SNR (>10 dB) - Maximum Capacity Mode

In excellent conditions, CASCADE operates at peak efficiency. The channel can support the full complexity of all 64 patterns, allowing maximum user capacity and throughput. Each user can be assigned unique patterns with significant frequency flexibility.

Operating characteristics:
- **Pattern Usage**: All 64 patterns remain distinct and usable
- **Frequency Flexibility**: Model can shift patterns ±30% in frequency
- **User Capacity**: 50+ simultaneous users possible
- **Spectral Efficiency**: Users spread across full 2500 Hz bandwidth
- **Throughput**: Each user achieves near-Shannon capacity
- **Natural Separation**: Pattern orthogonality provides complete isolation

### Medium SNR (0-10 dB) - Balanced Operation

As conditions degrade, the model begins clustering patterns to maintain reliability. This is like switching from individual conversations to small group discussions - some capacity is sacrificed for robustness.

Operating characteristics:
- **Pattern Clustering**: 64 patterns coalesce into 16 distinguishable groups
- **Frequency Mutation**: Reduced to ±10% to preserve separation
- **User Capacity**: 10-15 simultaneous users
- **Spectrum Concentration**: Users migrate to cleaner spectrum regions
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

CASCADE's TFP dimensions work like a city's transportation system:

- **Patterns** are like different types of vehicles (cars, buses, bikes) that naturally separate
- **Frequency** is like lanes on a highway that vehicles can smoothly change between
- **Time** is like the natural flow of traffic without rigid traffic light schedules

Just as experienced drivers learn to navigate efficiently without central control, CASCADE's model learns to optimize communication without rigid slot assignments. The patterns provide enough structure to prevent chaos (like having roads), while the continuous optimization allows efficient use of resources (like choosing the best route).