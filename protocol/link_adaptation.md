# Pairwise Link Adaptation

## Key Insight: SNR is Pairwise, Not Per-Station

Traditional radio systems assume each station has "an SNR" that determines its needs. CASCADE recognizes that SNR is fundamentally pairwise - the link quality between any two stations is unique and asymmetric.

## Reality of Pairwise Links

```
Station A → Station B: +10 dB (strong link, uses 50 Hz)
Station A → Station C: -5 dB  (medium link, uses 150 Hz)
Station B → Station C: -20 dB (weak link, uses 400 Hz)
Station B → Station A: +8 dB  (different from A→B!)
```

Each transmission adapts to its specific destination, not a generic "channel condition".

## Link Quality Matrix

### Asymmetric Nature
```python
# Link quality is NOT symmetric
link_quality['W1ABC']['W2DEF'] = +10  # Strong path
link_quality['W2DEF']['W1ABC'] = +5   # Weaker return path

# Reasons for asymmetry:
# - Different antenna gains
# - Local noise differences
# - Terrain shielding
# - Power output differences
```

### Dynamic Tracking
```python
class LinkQualityMatrix:
    def __init__(self):
        self.links = {}  # (src, dst) -> quality
        self.history = defaultdict(deque)  # Historical measurements

    def update_from_ack(self, src, dst, snr_bucket):
        """Update link quality from ACK"""
        # 4-bit coarse SNR: -20, -10, 0, +10 dB
        snr_map = {0: -20, 1: -10, 2: 0, 3: 10}
        measured_snr = snr_map[snr_bucket]

        # Exponential moving average
        if (src, dst) in self.links:
            old_snr = self.links[(src, dst)]
            new_snr = 0.7 * old_snr + 0.3 * measured_snr
        else:
            new_snr = measured_snr

        self.links[(src, dst)] = new_snr
        self.history[(src, dst)].append({
            'time': time.time(),
            'snr': measured_snr
        })

    def get_link_quality(self, src, dst):
        """Get current link quality estimate"""
        if (src, dst) in self.links:
            return self.links[(src, dst)]
        else:
            # Unknown link: conservative estimate
            return -5  # dB
```

## Bandwidth Allocation Per Link

### Strong Links Use Less Spectrum
```python
def allocate_bandwidth(src, dst, priority):
    link_snr = link_quality_matrix.get_link_quality(src, dst)

    if dst in ['CQ', 'ALL', 'BROADCAST']:
        # Broadcast: assume weak receivers
        return 400  # Hz

    elif link_snr > 10:
        # Excellent link: minimal bandwidth
        base_bw = 50
    elif link_snr > 0:
        # Good link: moderate bandwidth
        base_bw = 150
    elif link_snr > -10:
        # Fair link: substantial bandwidth
        base_bw = 300
    else:
        # Poor link: maximum bandwidth
        base_bw = 400

    # Priority adjustment
    if priority == 'EMERGENCY':
        return base_bw * 1.5  # Extra margin
    elif priority == 'LOW':
        return base_bw * 0.8  # Accept some risk

    return base_bw
```

### Pattern Assignment Per Link
```python
def assign_patterns(src, dst):
    link_snr = link_quality_matrix.get_link_quality(src, dst)

    if link_snr > 10:
        # Strong: many patterns available
        num_patterns = 32
        complexity = 64  # Full constellation
    elif link_snr > 0:
        # Medium: moderate patterns
        num_patterns = 16
        complexity = 16  # Clustered
    elif link_snr > -10:
        # Weak: few robust patterns
        num_patterns = 8
        complexity = 4   # Heavy clustering
    else:
        # Very weak: minimal patterns
        num_patterns = 4
        complexity = 2   # Binary

    return {
        'count': num_patterns,
        'complexity': complexity,
        'confidence': calculate_confidence(link_snr)
    }
```

## Mixed SNR Scenarios

### Scenario: Emergency Net
```python
def emergency_net_allocation():
    """28 simultaneous users with vastly different link qualities"""

    allocations = []

    # Emergency coordinator (weak, needs reliability)
    allocations.append({
        'callsign': 'W1EMG',
        'role': 'coordinator',
        'typical_snr': -20,
        'bandwidth': 400,
        'frequency': (0, 400),
        'patterns': [0, 1],  # Most robust
        'time_slot': 'continuous'
    })

    # Emergency stations (medium weak)
    for i in range(2):
        allocations.append({
            'callsign': f'W{i}EMS',
            'role': 'emergency_station',
            'typical_snr': -15,
            'bandwidth': 150,
            'frequency': (400 + i*150, 550 + i*150),
            'patterns': range(i*4, (i+1)*4),
            'time_slot': 'continuous'
        })

    # Relay stations (medium strong)
    for i in range(5):
        allocations.append({
            'callsign': f'W{i}RLY',
            'role': 'relay',
            'typical_snr': 0,
            'bandwidth': 100,
            'frequency': (700 + i*100, 800 + i*100),
            'patterns': range(8 + i*8, 16 + i*8),
            'time_slot': 'continuous'
        })

    # Local check-ins (strong, can pack densely)
    for i in range(20):
        allocations.append({
            'callsign': f'W{i}LOC',
            'role': 'local',
            'typical_snr': 10,
            'bandwidth': 65,
            'frequency': (1200 + i*65, 1265 + i*65),
            'patterns': range(i*3, (i+1)*3),
            'time_slot': 'as_needed'
        })

    return allocations
```

### Scenario: DX and Local Mix
```python
def dx_local_mix():
    """43 users: 3 DX + 10 regional + 30 local"""

    allocations = []

    # DX stations (very weak, need maximum resources)
    for i in range(3):
        allocations.append({
            'callsign': f'{i}DX',
            'typical_snr': -15,
            'bandwidth': 200,
            'frequency': (i*200, (i+1)*200),
            'patterns': range(i*4, (i+1)*4),
            'priority': 'high'
        })

    # Regional stations (medium)
    for i in range(10):
        allocations.append({
            'callsign': f'W{i}REG',
            'typical_snr': 5,
            'bandwidth': 70,
            'frequency': (600 + i*70, 670 + i*70),
            'patterns': range(12 + i*4, 16 + i*4)
        })

    # Local stations (strong, efficient packing)
    for i in range(30):
        allocations.append({
            'callsign': f'W{i}LOC',
            'typical_snr': 15,
            'bandwidth': 40,
            'frequency': (1300 + i*40, 1340 + i*40),
            'patterns': [i % 64]  # Reuse patterns
        })

    return allocations
```

## Learning from ACKs

### ACK-Based Link Learning
```python
def process_ack_for_learning(ack):
    """Extract link quality information from ACK"""

    # Update link quality matrix
    link_quality_matrix.update_from_ack(
        src=ack['destination'],  # ACK reverses direction
        dst=ack['source'],
        snr_bucket=ack['snr']
    )

    # Track pattern success
    for pattern in ack.get('patterns_decoded', []):
        pattern_success[(ack['source'], pattern)] += 1

    # Store kernel hint if provided
    if 'kernel_generated' in ack:
        kernel_cache[ack['source']] = ack['kernel_generated']

    # Update bandwidth allocation
    suggested_bw = calculate_optimal_bandwidth(ack['snr'])
    bandwidth_history[ack['source']].append(suggested_bw)
```

### Predictive Link Adaptation
```python
def predict_link_quality(src, dst, time_offset=0):
    """Predict future link quality"""

    history = link_quality_matrix.history[(src, dst)]

    if len(history) < 3:
        # Not enough data
        return link_quality_matrix.get_link_quality(src, dst)

    # Time series prediction
    times = [h['time'] for h in history]
    snrs = [h['snr'] for h in history]

    # Simple linear regression
    slope, intercept = np.polyfit(times, snrs, 1)

    # Predict future
    future_time = time.time() + time_offset
    predicted_snr = slope * future_time + intercept

    # Bound prediction
    return np.clip(predicted_snr, -25, 15)
```

## Optimizing for Asymmetric Links

### Different Parameters Each Direction
```python
def optimize_bidirectional_link(station_a, station_b):
    """Optimize for asymmetric link"""

    # A → B link
    a_to_b = {
        'snr': link_quality_matrix.get_link_quality(station_a, station_b),
        'bandwidth': allocate_bandwidth(station_a, station_b),
        'patterns': assign_patterns(station_a, station_b),
        'kernel_hint': kernel_cache.get((station_a, station_b))
    }

    # B → A link (can be very different!)
    b_to_a = {
        'snr': link_quality_matrix.get_link_quality(station_b, station_a),
        'bandwidth': allocate_bandwidth(station_b, station_a),
        'patterns': assign_patterns(station_b, station_a),
        'kernel_hint': kernel_cache.get((station_b, station_a))
    }

    return a_to_b, b_to_a
```

### Handling Unknown Links
```python
def handle_unknown_link(src, dst):
    """Conservative approach for first contact"""

    # Start conservative
    initial_params = {
        'bandwidth': 300,  # Hz
        'patterns': [0, 1, 2, 3],  # Most robust
        'redundancy': 2.5,
        'complexity': 4  # 4-cluster mode
    }

    # Mark for learning
    unknown_links.add((src, dst))

    # Will adapt quickly after first ACK
    return initial_params
```

## Benefits of Pairwise Adaptation

1. **Spectrum Efficiency**: Strong links don't waste spectrum
2. **Reliability**: Weak links get resources they need
3. **Capacity**: More total users by optimizing each link
4. **Fairness**: Resources match actual needs
5. **Learning**: System improves with every ACK