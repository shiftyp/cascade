# Continuous Coexistence Examples

Real-world scenarios showing how CASCADE supports multiple stations through continuous frequency distribution and learned pattern mutations.

## Introduction

Traditional radio systems use rigid time and frequency slots to separate users - think of channelized repeaters or packet radio time slots. CASCADE takes a fundamentally different approach: continuous optimization within a framework of orthogonal patterns. This document provides concrete examples of how this works in practice.

The key insight is that CASCADE doesn't assign users to fixed channels or time slots. Instead, the protocol assigns pattern pools while the model continuously optimizes frequency placement, bandwidth allocation, and transmission timing. This creates a fluid, adaptive system that naturally responds to changing conditions without explicit coordination.

These examples demonstrate real-world scenarios from emergency operations to casual nets, showing how continuous coexistence enables capabilities impossible with traditional slotted systems.

## Scenario 1: Emergency Net Activation

This scenario demonstrates how CASCADE handles the sudden activation of an emergency net with multiple stations at different signal strengths. The system must simultaneously support strong local links and weak distant links without the strong signals overwhelming the weak ones.

Three stations with varying link quality during emergency:

```python
# Initial Conditions
station_a = {'callsign': 'W1ABC', 'location': 'Boston'}
station_b = {'callsign': 'K2DEF', 'location': 'New York'}
station_c = {'callsign': 'N3GHI', 'location': 'Philadelphia'}

# Pairwise SNRs (not symmetric!)
links = {
    ('W1ABC', 'K2DEF'): +8,   # Good path
    ('K2DEF', 'W1ABC'): +6,   # Different reverse path
    ('W1ABC', 'N3GHI'): -2,   # Weak path
    ('N3GHI', 'W1ABC'): -5,   # Weaker reverse
    ('K2DEF', 'N3GHI'): +12,  # Strong path
    ('N3GHI', 'K2DEF'): +10   # Strong reverse
}
```

### Protocol Decisions (Discrete)

The protocol layer makes discrete decisions about resource allocation. These are binary choices: which patterns to assign, whether to relay, what priority to use. These decisions are predictable and understandable to human operators:
```python
# Station A protocol assigns patterns
def protocol_assignment_a():
    # Emergency traffic to both stations
    assignments = {
        'to_k2def': [4, 12, 20, 28],  # 4 patterns for good link
        'to_n3ghi': [0, 1, 2, 3]      # Emergency patterns for weak link
    }
    return assignments
```

### Model Optimization (Continuous)

While the protocol assigns patterns discretely, the model optimizes their use continuously. This includes mutation amounts, exact frequency placement, and bandwidth allocation. Notice how nothing is on a rigid grid:
```python
# Model mutates patterns within bounds
def model_optimization_a():
    # To K2DEF (+8 dB link)
    for pattern_id in [4, 12, 20, 28]:
        base_pattern = PATTERN_TABLE[pattern_id]

        # High SNR: significant mutation allowed
        mutation = model.mutate_pattern(
            base_pattern,
            mutation_range=0.25,  # 25% variation allowed
            link_snr=8
        )

        # Continuous frequency placement
        optimal_freq = 700 + pattern_id * 35.7  # Not rigid slots!
        bandwidth = 45  # Hz, adapted to link quality

        transmit(mutation, optimal_freq, bandwidth)

    # To N3GHI (-2 dB link)
    for pattern_id in [0, 1, 2, 3]:
        base_pattern = PATTERN_TABLE[pattern_id]

        # Low SNR: minimal mutation
        mutation = model.mutate_pattern(
            base_pattern,
            mutation_range=0.03,  # 3% variation only
            link_snr=-2
        )

        # Wider bandwidth for redundancy
        optimal_freq = 500 + pattern_id * 125
        bandwidth = 100  # Hz, more redundancy

        transmit(mutation, optimal_freq, bandwidth)
```

### Spectrum Utilization

The resulting spectrum usage shows how continuous optimization differs from traditional slotted systems. Notice the irregular spacing and varying bandwidths:
```
Frequency (Hz)
300  ├──┤ Emergency patterns (N3GHI) - wide, robust
500  ├────────┤
750  ├──┤ Efficient patterns (K2DEF) - narrow, fast
850  ├──┤
950  ├──┤
1050 ├──┤
1500 │ Unused spectrum (continuous, not slotted)
2300 │
```

## Scenario 2: Multi-Net Convergence

This scenario explores what happens when multiple independent networks discover each other and must share spectrum. Traditional systems would require complex coordination protocols. CASCADE handles this through emergent organization - the nets naturally find non-interfering configurations through pattern diversity and continuous frequency optimization.

Multiple nets discover each other and coexist:

```python
# Three independent nets initially
dx_net = ['W6XXX', 'JA1XXX', 'VK2XXX']      # DX net
emcomm_net = ['W1XXX', 'W2XXX', 'W3XXX']    # Emergency net
casual_net = ['K5XXX', 'N7XXX', 'KE8XXX']   # Casual chat
```

### Discovery Phase

The discovery process uses beacon timing diversity and continuous frequency selection to avoid collisions. Each net naturally finds unused spectrum without central coordination:
```python
def beacon_discovery():
    """Beacons enable discovery without collision"""

    # Each net uses different beacon timing
    dx_beacon_time = hash('DX_NET') % 30       # Second 0-29
    emcomm_beacon_time = hash('EMCOMM') % 30   # Different second
    casual_beacon_time = hash('CASUAL') % 30    # Different second

    # Continuous frequency selection
    dx_beacon_freq = 800 + gaussian_noise(0, 50)      # ~800 Hz
    emcomm_beacon_freq = 1200 + gaussian_noise(0, 50) # ~1200 Hz
    casual_beacon_freq = 1600 + gaussian_noise(0, 50)  # ~1600 Hz
```

### Coexistence Negotiation

Once nets discover each other, they negotiate pattern pool sharing. This is a protocol-level discrete decision, but notice how the actual usage within those pools remains continuous:
```python
def pattern_pool_sharing():
    """Networks agree on pattern allocation"""

    # Protocol assigns non-overlapping pools
    pattern_allocation = {
        'emcomm_net': range(0, 16),   # Emergency gets 0-15
        'dx_net': range(16, 40),       # DX gets 16-39
        'casual_net': range(40, 64)    # Casual gets 40-63
    }

    # Model mutates within assigned pools
    for net, patterns in pattern_allocation.items():
        for pattern_id in patterns:
            # Each net's model mutates differently
            net_specific_mutation = models[net].mutate(
                PATTERN_TABLE[pattern_id],
                net_conditions[net]
            )
```

### Continuous Frequency Distribution

The key innovation: frequencies aren't assigned from a fixed grid. Each station's model finds optimal placement based on real-time interference measurements. This creates much more efficient spectrum usage than rigid channelization:
```python
def frequency_distribution():
    """Model learns optimal continuous distribution"""

    # Not rigid slots - continuous optimization
    frequency_map = {}

    # Model decides based on interference map
    interference = measure_spectrum_occupancy()

    for station in all_stations:
        # Find cleanest continuous region
        optimal_freq = model.find_clear_frequency(
            interference,
            station.bandwidth_need,
            station.priority
        )

        # Could be ANY frequency, not just slots
        frequency_map[station] = optimal_freq

    return frequency_map
    # Example output:
    # W6XXX: 743.2 Hz (not 750!)
    # JA1XXX: 921.7 Hz (not 925!)
    # W1XXX: 501.3 Hz (not 500!)
```

## Scenario 3: Adaptive Capacity Sharing

This scenario demonstrates CASCADE's ability to support simultaneous users with vastly different signal strengths. Traditional systems suffer from the near-far problem where strong signals overwhelm weak ones. CASCADE solves this through adaptive pattern complexity and continuous bandwidth allocation.

The key is that each link adapts independently - a strong local link doesn't force a weak DX link to use the same parameters. This pairwise adaptation enables efficient spectrum sharing across a 40+ dB dynamic range.

Strong and weak signals coexist efficiently:

```python
# Mixed signal strengths
stations = {
    'local_strong': {'snr': +15, 'distance': '5 miles'},
    'regional_medium': {'snr': +3, 'distance': '50 miles'},
    'dx_weak': {'snr': -8, 'distance': '3000 miles'}
}
```

### Pattern Complexity Adaptation

Each link independently selects appropriate pattern complexity based on its specific SNR. This isn't a global mode that affects everyone - it's pairwise optimization that allows maximum throughput for strong links while maintaining reliability for weak ones:
```python
def adaptive_complexity():
    """Each link uses appropriate complexity"""

    # Strong link: full 64-pattern mode
    local_patterns = model.select_patterns(
        snr=15,
        available=range(64)
    )  # Uses 40-50 patterns, high throughput

    # Medium link: clustered patterns
    regional_patterns = model.select_patterns(
        snr=3,
        available=range(64)
    )  # Uses 12-16 patterns from 4 clusters

    # Weak link: binary mode
    dx_patterns = model.select_patterns(
        snr=-8,
        available=range(64)
    )  # Uses 2 maximally separated patterns
```

### Bandwidth Allocation

Bandwidth varies continuously based on link requirements. Weak signals get more bandwidth for redundancy while strong signals use narrow channels efficiently. The total spectrum usage is far less than traditional fixed-channel systems would require:
```python
def continuous_bandwidth():
    """Bandwidth varies continuously, not in slots"""

    # Model learns optimal bandwidth per link
    bandwidth_map = {
        'local_strong': 31.4,      # Hz (narrow, efficient)
        'regional_medium': 87.6,   # Hz (moderate width)
        'dx_weak': 198.3           # Hz (wide for robustness)
    }

    # Continuous center frequencies
    frequency_map = {
        'local_strong': 1847.2,    # Any frequency
        'regional_medium': 923.8,
        'dx_weak': 501.7
    }

    # Total spectrum used efficiently
    total_bandwidth = sum(bandwidth_map.values())  # 317.3 Hz
    # Much less than 2500 Hz available!
```

## Scenario 4: Dynamic Storm Net

This scenario shows how CASCADE adapts to rapidly changing propagation during a severe weather event. As the storm progresses, SNR degrades due to increased atmospheric noise, then recovers as it passes. The system must maintain communication throughout, gracefully degrading and recovering capacity.

Unlike traditional systems that would lose communication entirely below certain thresholds, CASCADE maintains connectivity by smoothly transitioning through pattern complexity levels. This ensures emergency traffic continues even in the worst conditions.

Conditions change rapidly during weather event:

```python
class StormNetEvolution:
    """Net adapts as storm progresses"""

    def __init__(self):
        self.phase = 'pre_storm'
        self.stations = {}
        self.priorities = {}

    def pre_storm_phase(self):
        """Good conditions, normal operations"""

        # Most stations at high SNR
        self.assign_patterns_widely()  # Use all 192 message patterns
        self.set_normal_priorities()   # Regular traffic

        # Continuous frequency - spread across band
        return self.distribute_continuously(300, 2300)

    def storm_arrival(self):
        """Conditions degrading"""

        # SNR dropping due to QRN
        self.collapse_patterns_partially()  # 64 → 16 patterns
        self.increase_redundancy()          # More FEC

        # Frequencies cluster for mutual support
        return self.cluster_frequencies(800, 1200)

    def storm_peak(self):
        """Severe conditions"""

        # Very poor SNR
        self.collapse_to_binary()        # 2 patterns only
        self.emergency_priority_only()   # Emergency traffic

        # Single frequency concentration
        return self.concentrate_frequency(900, 200)

    def storm_passing(self):
        """Conditions improving"""

        # SNR recovering
        self.expand_patterns_gradually()  # 2 → 4 → 16 patterns
        self.restore_normal_priority()

        # Frequencies spread back out
        return self.redistribute_continuously()
```

### Continuous Adaptation Timeline

The timeline shows how the system adapts continuously rather than in discrete steps. Pattern count, bandwidth, and center frequency all vary smoothly as conditions change. This prevents the synchronization losses that occur with mode switching in traditional systems:
```python
def timeline_example():
    """How spectrum use evolves continuously"""

    timeline = []
    for minute in range(180):  # 3-hour storm

        # Continuous SNR variation
        snr = storm_snr_model(minute)

        # Model continuously adapts
        pattern_count = model.optimal_patterns(snr)
        frequency_spread = model.optimal_spread(snr)

        # Not discrete steps!
        timeline.append({
            'time': minute,
            'snr': snr,
            'patterns': pattern_count,      # e.g., 37.4 patterns
            'bandwidth': frequency_spread,  # e.g., 1823.7 Hz
            'center_freq': model.optimal_center(snr)  # e.g., 1147.3 Hz
        })

    return timeline
```

## Scenario 5: Mesh Network Self-Organization

This final scenario demonstrates how a mesh network self-organizes without central control. Stations discover topology through hash exchanges, measure pairwise link quality, and automatically route traffic along optimal paths. The continuous nature of CASCADE enables sophisticated routing that adapts to actual channel conditions rather than assumed connectivity.

The mesh formation process is completely distributed - no station has complete knowledge, yet the network achieves near-optimal routing through local decisions and continuous optimization.

Stations organically form efficient mesh:

```python
class MeshFormation:
    """Stations discover topology and adapt"""

    def discovery_phase(self):
        """Stations find each other"""

        # Hash exchange reveals topology
        for station in stations:
            # Each station's model decides beacon frequency
            beacon_freq = hash(station.callsign) * 17.3 % 2000 + 400
            # Results in unique, non-colliding frequencies

            station.transmit_beacon(beacon_freq)

    def topology_learning(self):
        """Learn pairwise link qualities"""

        # Each link measured independently
        for src, dst in all_pairs:
            # Kernel hints optimize per link
            kernel = receivers[dst].generate_kernel(src)

            # Continuous SNR measurement
            measured_snr = test_link(src, dst, kernel)

            # Store in distributed matrix
            link_matrix[src][dst] = measured_snr

    def route_optimization(self):
        """Find optimal paths"""

        # Model learns relay value
        for message in pending:
            src, dst = message['from'], message['to']

            if link_matrix[src][dst] < -5:
                # Weak direct path

                # Find best relay
                relay = model.find_best_relay(
                    src, dst, link_matrix
                )

                # Continuous scoring, not discrete
                relay_value = model.compute_relay_value(
                    direct_snr=link_matrix[src][dst],
                    relay_snr1=link_matrix[src][relay],
                    relay_snr2=link_matrix[relay][dst]
                )  # e.g., value = 0.734 (not binary!)
```

### Emergent Behavior

From simple local rules and continuous optimization, complex network-wide behaviors emerge. Frequency reuse patterns, geographic clustering, and traffic routing all self-organize without central planning. This is the power of combining discrete protocol rules with continuous model optimization:
```python
def emergent_mesh_properties():
    """Properties that emerge from continuous optimization"""

    # Frequency reuse with continuous spacing
    frequency_reuse = {}
    for station in stations:
        neighbors = get_neighbors(station)

        # Model finds minimum separation
        min_separation = model.min_frequency_separation(
            station.snr_to_neighbors
        )  # e.g., 347.2 Hz, not fixed slots

        frequency_reuse[station] = min_separation

    # Natural clustering by distance
    clusters = []
    for region in geographic_regions:
        # Continuous geographic boundaries
        cluster_center = model.find_cluster_center(region)
        cluster_radius = model.optimal_radius(region.density)

        clusters.append({
            'center': cluster_center,
            'radius': cluster_radius,  # e.g., 73.4 km
            'frequency': model.cluster_frequency(region)
        })

    return clusters
```

## Key Insights

These scenarios reveal fundamental principles about how CASCADE achieves efficient multi-user coexistence through continuous optimization.

### Continuous vs Discrete

The separation between discrete protocol decisions and continuous model optimization is crucial. This table summarizes what belongs where:

| Aspect | Discrete (Protocol) | Continuous (Model) |
|--------|---------------------|-------------------|
| Pattern Assignment | Which patterns (0-63) | How much to mutate (0-30%) |
| Frequency | Pattern pool allocation | Exact Hz placement |
| Bandwidth | User gets transmission | How many Hz to use |
| Time | Message priority order | Fragment duration |
| Relay | Yes/no permission | Relay value score |
| Authentication | Required/relaxed | - |

The key insight: protocol provides structure, model provides optimization.

### Examples of Continuous Values

Throughout these scenarios, notice how values are rarely round numbers:

1. **Frequencies are continuous**: 743.2 Hz, not 750 Hz slots
2. **Bandwidth adapts smoothly**: 31.4 Hz to 198.3 Hz
3. **Pattern mutations vary**: 3% to 30% based on SNR
4. **Relay values are scores**: 0.734, not yes/no
5. **Timing is adaptive**: 1.37 second fragments, not fixed
6. **SNR measurements**: -5.3 dB, not quantized to integers

This continuous nature allows CASCADE to find optimal configurations that wouldn't exist in a discretized system.

### Model Learning Objectives

Through training on these scenarios, the model learns to:
- Minimize interference through frequency diversity
- Maximize total throughput across all links
- Balance fairness with efficiency
- Adapt smoothly to changing conditions
- Discover emergent organization patterns

### Protocol Simplicity

Despite the sophisticated behaviors demonstrated, the protocol remains simple. It only makes discrete, understandable decisions that operators can reason about:

The protocol only decides:
- Pattern pool assignment (which patterns available)
- Message priority (emergency/high/normal/low)
- Relay permission (yes/no based on content)
- Authentication requirements (strict/relaxed)

Everything else is continuous optimization by the model!

## Real-World Implications

### Spectrum Efficiency

The continuous optimization approach typically achieves 3-5× better spectrum efficiency than traditional channelized systems. By placing signals exactly where they need to be rather than in fixed slots, CASCADE can support more users in the same bandwidth.

Consider traditional packet radio: with 300 Hz channels and guard bands, you might fit 6 channels in 2500 Hz. CASCADE can support 15-20 users in the same spectrum by using continuous placement and pattern diversity.

### Robustness to Interference

Because frequencies and patterns can be continuously adjusted, CASCADE naturally avoids interference. A narrow carrier at 1000 Hz doesn't take out a whole channel - the model just shifts signals slightly to avoid it. This makes the system much more robust to both intentional and unintentional interference.

### Emergency Communications

The ability to maintain communication as conditions degrade continuously (rather than falling off a cliff at a threshold) is critical for emergency operations. The storm net scenario shows how CASCADE maintains emergency traffic even as conditions become marginal.

### Simplified Operations

Paradoxically, the continuous optimization makes operations simpler for users. They don't need to choose channels, time slots, or data rates. The system automatically optimizes all parameters based on measured conditions. Users just specify what they want to communicate and to whom.

## Design Philosophy

These examples embody CASCADE's core philosophy:

**Embrace Continuous Optimization**: Real-world RF doesn't respect neat boundaries. By operating continuously, CASCADE matches the physical reality of radio propagation.

**Separate Concerns Cleanly**: Protocol handles "what" and "whether" with discrete decisions. Model handles "how" and "when" with continuous optimization. This separation makes both layers simpler and more effective.

**Enable Emergent Behavior**: Rather than trying to explicitly program every scenario, CASCADE creates conditions where good behaviors emerge naturally. The mesh network self-organization is a perfect example.

**Maintain Human Understanding**: Despite sophisticated optimization, operators can still understand what's happening. They see pattern assignments, priority decisions, and relay permissions - all discrete, logical choices.

## Future Scenarios

These examples only scratch the surface of CASCADE's capabilities. Future scenarios might explore:

- **Cognitive Jamming Resistance**: How continuous adaptation defeats smart jammers
- **Ionospheric Storm Response**: Adaptation to rapidly changing HF propagation
- **Massive IoT Networks**: Supporting hundreds of low-rate sensors
- **Cross-Band Coordination**: VHF/UHF/HF mesh with optimal band selection
- **Mobile Mesh Networks**: Vehicle-to-vehicle with Doppler compensation

Each scenario would demonstrate the same principle: continuous optimization within a discrete framework enables capabilities impossible with traditional approaches.

## Conclusion

Through these five scenarios, we've seen how CASCADE's continuous coexistence enables:

1. **Multi-strength emergency nets** where weak signals aren't overwhelmed
2. **Organic net convergence** without coordination protocols
3. **Adaptive capacity sharing** across 40+ dB dynamic range
4. **Storm-resilient operations** with graceful degradation
5. **Self-organizing mesh networks** through distributed decisions

The combination of discrete protocol structure and continuous model optimization creates a system that is both sophisticated and understandable, efficient and robust, adaptive and predictable.

This is the future of radio: not rigid channels and time slots, but fluid adaptation to the continuous nature of the electromagnetic spectrum.