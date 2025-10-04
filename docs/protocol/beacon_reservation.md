# Pattern-Based Beacons

CASCADE uses pattern-based beacons that share the 78-tone grid with message traffic. There is NO frequency reservation—all patterns (beacons, messages, emergency) select their 4 tones from the same 78-tone grid: 300-2764 Hz, 32 Hz spacing.

## Architecture Change

**Previous (Reserved Band):**
- Beacon band: 1475-1625 Hz (175 Hz reservation)
- Emergency: 1550 Hz (single tone)
- 4-FSK beacons: [1490, 1520, 1580, 1610] Hz
- Spectrum efficiency: 84.5%

**Current (Pattern-Based):**
- 78-tone grid: 300-2764 Hz (32 Hz spacing)
- Each pattern selects 4 tones from 78 (adaptive)
- Emergency: Patterns 0-15 (detected via correlation)
- Normal beacons: Patterns 16-63
- Spectrum efficiency: 96.7%

## Frequency Allocation

### 4-Tone Shared Grid

```
Total CASCADE channel: 2500 Hz (300-2800 Hz)

78-tone reference grid (all patterns share):
├─ 78 tones: 300-2764 Hz (32 Hz spacing)
└─ Each pattern selects 4 tones from this grid (adaptive)

Tone selection method: Adaptive based on channel conditions
- Each pattern picks best 4 from 78 (can shift ±3 from base)
- Multiple patterns can select overlapping tones
- Separation when overlapping: Time × IQ orthogonality

Traffic types:
- Beacons: Patterns 0-47 (4 tones each from 78-tone grid, simple IQ λ ≤ 0.3)
- Messages: Patterns 48-127 (4 tones each from 78-tone grid, hierarchical IQ pools)
- Emergency: Patterns 0-15 (beacon) + 48-63 (message) - correlation detection
```

## Beacon Pattern Allocation

### 64 Beacon Patterns (IDs 0-63)

```python
BEACON_PATTERNS = {
    # Emergency patterns (detected via correlation)
    'emergency': {
        'pattern_ids': range(0, 16),  # Patterns 0-15
        'iq_complexity': 'minimal (BPSK line)',
        'purpose': 'Emergency alerts, maximum robustness',
        'detection': 'Pattern correlation (zero overhead)',
        'tone_selection': '4 from 78 (adaptive)',
        'example_tones': '[0, 19, 38, 57] → [300, 908, 1516, 2124] Hz',
    },

    # Normal beacons (kernel exchange, coordination)
    'normal': {
        'pattern_ids': range(16, 64),  # Patterns 16-63
        'iq_complexity': 'simple (circles/ellipses)',
        'purpose': 'Kernel exchange, network coordination',
        'anti_kernel': 48,  # Resilient patterns
        'tone_selection': '4 from 78 (adaptive)',
        'example_tones': '[12, 34, 51, 65] → [684, 1388, 1932, 2380] Hz',
    },
}

# Each beacon pattern selects 4 tones from 78-tone grid
# Adaptive selection avoids interference
# Multiple beacons can select overlapping tones (Time × IQ separation)
```

## Emergency Detection

### Pattern 0-15 Correlation (Not Frequency Monitoring)

```python
def detect_emergency(received_signal):
    """
    Emergency detected via pattern correlation
    Zero additional overhead (already correlating all patterns)
    """

    # Model correlates all 128 patterns during receive
    pattern_correlations = model.correlate_all_patterns(received_signal)

    # Check emergency patterns (0-15)
    for pattern_id in range(16):
        if pattern_correlations[pattern_id] > EMERGENCY_THRESHOLD:
            # Emergency detected!
            emergency_data = model.decode_pattern(pattern_id, received_signal)
            return {
                'emergency': True,
                'pattern_id': pattern_id,
                'data': emergency_data,
                'priority': 'CRITICAL'
            }

    return {'emergency': False}

# Benefits:
# - Zero overhead (already doing pattern correlation)
# - Works on any of 4 tones (frequency-agnostic)
# - Adaptive to local interference
# - 4 simultaneous emergencies (patterns 0,1,2,3...)
```

### Emergency vs Normal Beacon Separation

Emergency patterns (0-15) are distinguished by:
1. **Simple IQ**: BPSK line (λ = 0.0) for maximum robustness
2. **Pattern priority**: Emergency patterns checked first
3. **Threshold**: Lower correlation threshold for detection
4. **Payload**: Includes emergency flag in decoded data

Normal beacons (16-63) use slightly more complex IQ (circles/ellipses, λ ≤ 0.3) for better capacity while maintaining robustness.

## Beacon Capacity

### Simultaneous Beacons

```python
BEACON_CAPACITY = {
    'simultaneous_beacons': 64,  # All 64 patterns orthogonal
    'emergency_capacity': 16,  # Patterns 0-15
    'normal_capacity': 48,  # Patterns 16-63

    'separation_method': 'Time × IQ orthogonality',
    'frequency_reuse': 'Full (all patterns select from 78-tone grid)',

    # All beacons can transmit on same tones simultaneously
    # Separated by <-30 dB pattern orthogonality
}
```

## Benefits of Pattern-Based Beacons

### Spectrum Efficiency

```
Reserved band approach:
- Beacon reservation: 175 Hz
- Message tones: 78 tones × 31 Hz = 2418 Hz
- Total occupied: 2345 Hz / 2500 Hz = 84.5% efficiency

Pattern-based approach (78-tone grid):
- All patterns share 78-tone grid: 300-2764 Hz
- Each pattern selects 4 from 78 (adaptive)
- Total occupied: 78 × 31 Hz = 2418 Hz / 2500 Hz = 96.7% efficiency

Improvement: +12.2% spectrum efficiency
```

### Operational Benefits

1. **Simplified frequency coordination**: All patterns select from same 78-tone grid
2. **Adaptive to interference**: Beacons use IQ dimension for separation
3. **Higher capacity**: 64 simultaneous beacons vs limited 4-FSK slots
4. **Simpler architecture**: One unified tone grid
5. **Zero emergency overhead**: Detection via existing pattern correlation
6. **Frequency diversity**: C(78,4) = 1.4M tone combinations for adaptive selection

### Emergency Advantages

**Pattern-based detection:**
- Works regardless of which tone used (adaptive)
- No fixed frequency to monitor
- Zero additional CPU overhead
- 4+ simultaneous emergencies supported
- Robust to local QRM (selects clearest tone)

**Reserved tone (old approach):**
- Fixed 1550 Hz (vulnerable to interference)
- Requires dedicated monitoring
- Single emergency at a time
- No frequency diversity

## IQ Complexity Limits

Beacon patterns use simplified IQ (λ ≤ 0.3) for maximum robustness:

```python
BEACON_IQ_LIMITS = {
    'emergency (0-15)': {
        'max_lambda': 0.0,
        'iq_type': 'BPSK line',
        'iq_directions': 1-2,
        'rationale': 'Maximum robustness for emergencies'
    },

    'normal (16-63)': {
        'max_lambda': 0.3,
        'iq_type': 'Simple circles/ellipses',
        'iq_directions': 2-4,
        'rationale': 'Balance robustness with capacity'
    },
}

# Message patterns (48-127) use higher IQ complexity
# up to λ = 0.9 for exceptional NVIS conditions (patterns 112-127)
```

## Pattern Storage

```python
# Beacon pattern storage (64 patterns)
beacon_pattern = {
    'id': int,  # 0-63
    'freq_sequence': np.uint8[32],  # Tone indices 0-3 (into pattern's selected set)
    'base_tones': np.uint8[4],  # Base 4-tone selection from 78-tone grid
    'iq_trajectory': complex64[32],  # Simple IQ (λ ≤ 0.3)
    'complexity_level': int,  # 0 (emergency) or 1 (normal)
}

# Per pattern: 296 bytes (4 extra for base_tones)
# Total 64 patterns: ~19 KB

# Generation time: 6-8 hours
```

## Usage Examples

### Normal Beacon Transmission

```python
def transmit_beacon(my_kernel, beacon_pattern_id, channel_state):
    """
    Transmit beacon using 4 tones selected from 78-tone grid
    """

    # Select from normal beacon patterns
    pattern_id = beacon_pattern_id  # 16-63

    # Adaptively select 4 best tones from 78-tone grid
    selected_tones = select_4_tones_for_pattern(
        pattern_id,
        available_tones=channel_state.available_tones,
        interference_map=channel_state.interference
    )
    # Example: [12, 35, 51, 65] (indices into 78-tone grid)
    # Maps to: [684, 1420, 1932, 2380] Hz

    # Encode kernel into beacon
    beacon_data = encode_kernel(my_kernel)

    # Transmit using pattern's time-frequency sequence
    for t in range(32):  # 32 symbols
        tone_idx = PATTERNS[pattern_id].freq_sequence[t]  # 0-3 (which of 4 selected)
        grid_tone = selected_tones[tone_idx]  # 0-77 (which of 78 grid tones)
        freq_hz = REFERENCE_TONES[grid_tone]  # Actual frequency

        iq_value = PATTERNS[pattern_id].iq_trajectory[t]
        transmit_symbol(freq_hz, iq_value, duration_ms=50)
```

### Emergency Alert

```python
def transmit_emergency(emergency_message):
    """
    Transmit emergency using patterns 0-3
    """

    # Use emergency pattern (0-15)
    pattern_id = 0  # Simplest, most robust

    # Same 4 tones as everything else
    tones = [600, 1200, 1800, 2400]

    # Emergency flag
    emergency_payload = {
        'emergency': True,
        'message': emergency_message,
        'timestamp': now(),
        'priority': 'CRITICAL'
    }

    # Transmit with BPSK (λ = 0.0)
    for t in range(32):
        tone_idx = PATTERNS[0].freq_sequence[t]
        iq_value = PATTERNS[0].iq_trajectory[t]  # Simple BPSK

        freq_hz = tones[tone_idx]
        transmit_symbol(freq_hz, iq_value, duration_ms=50)

    # Receivers detect via pattern 0 correlation
    # No special frequency monitoring needed
```

## Migration from Reserved Band

For implementations transitioning from reserved-band beacons:

**Changes required:**
1. Remove 1475-1625 Hz reservation logic
2. Update beacon patterns to select 4 tones from 78-tone grid
3. Replace frequency-based emergency detection with pattern correlation
4. Update spectrum allocation (no gaps in message tones)

**Backward compatibility:**
- None—this is a protocol-level change
- All stations must update simultaneously
- Pattern-based approach is not compatible with reserved-band

## See Also

- **[Adaptive Tone Grid](adaptive_tone_grid.md)** - 4-tone specification
- **[Pattern Architecture](../model/pattern_architecture.md)** - Pattern generation and storage
- **[Emergency Relay Network](emergency_relay_network.md)** - Emergency protocol details
- **[Signal Specification](signal_specification.md)** - Physical layer parameters
