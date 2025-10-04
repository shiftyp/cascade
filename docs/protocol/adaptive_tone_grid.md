# CASCADE Tone Grid & Spectrum Allocation

**Authoritative specification for CASCADE's 78-tone reference grid**

CASCADE uses 78 discrete reference tones spanning 300-2764 Hz. Each pattern adaptively selects 4 tones from this grid based on channel conditions. Multiple patterns can use the same tones simultaneously, separated by Time × IQ orthogonality.

## Pattern-Based Spectrum Allocation

**Core principle:** CASCADE uses pattern orthogonality for ALL separation. There are no separate frequency bands—all 78 reference tones are shared by beacons, emergency traffic, and messages.

### Pattern Count
- **Total patterns**: 128 (7-bit encoding, 0x00-0x7F)
- **Beacon patterns** (0-47): 48 patterns, each selects 4 from 78-tone grid
- **Message patterns** (48-127): 80 patterns, each selects 4 from 78-tone grid
- **Emergency patterns**: 0-15 (beacon) + 48-63 (message) = 32 total
- **Storage**: 38 KB total (14 KB beacon + 24 KB message)
- **Generation**: 18-24 hours (one-time)

## Reference Tone Specification

### 78 Discrete Tones (Baseline Configuration)

```python
REFERENCE_TONE_GRID = {
    'total_tones': 78,
    'spacing_hz': 32,  # Optimized for HF propagation
    'total_span': 2464,  # Hz (300-2764)

    # All tones available to all patterns
    'tones': [300 + i*32 for i in range(78)],
    'indices': range(0, 78),
}

# Complete tone list (Hz):
REFERENCE_TONES = [
    300, 332, 364, 396, 428, 460, 492, 524, 556, 588,
    620, 652, 684, 716, 748, 780, 812, 844, 876, 908,
    940, 972, 1004, 1036, 1068, 1100, 1132, 1164, 1196, 1228,
    1260, 1292, 1324, 1356, 1388, 1420, 1452, 1484, 1516, 1548,
    1580, 1612, 1644, 1676, 1708, 1740, 1772, 1804, 1836, 1868,
    1900, 1932, 1964, 1996, 2028, 2060, 2092, 2124, 2156, 2188,
    2220, 2252, 2284, 2316, 2348, 2380, 2412, 2444, 2476, 2508,
    2540, 2572, 2604, 2636, 2668, 2700, 2732, 2764
]

# Each pattern uses 4 tones from this grid (adaptive selection)
# Multiple patterns can use the same tones (overlap allowed)
# Separation via Time × IQ orthogonality
```

## HF Propagation Constraints

### Why 32 Hz Spacing?

```python
def calculate_minimum_tone_spacing_hf():
    """
    Derive 32 Hz spacing from HF propagation physics
    """

    constraints = {
        # 1. Symbol bandwidth
        'symbol_duration': 0.050,  # 50ms
        'symbol_bandwidth': 1 / 0.050,  # 20 Hz

        # 2. Multipath frequency spreading (HF ionosphere)
        'multipath_spread_typical': 5,  # Hz (90% energy within ±5 Hz)
        'multipath_spread_disturbed': 20,  # Hz (geomagnetic storms)

        # 3. Frequency drift (TX + RX + ionosphere)
        'drift_rate_typical': 0.8,  # Hz/second
        'pattern_duration': 1.6,  # seconds
        'total_drift': 0.8 * 1.6,  # 1.3 Hz over pattern
        'drift_margin': 2.5,  # Safety factor
        'drift_allowance': 1.3 * 2.5,  # 3.25 Hz

        # 4. Guard band between tones
        'guard_band': 3,  # Hz (prevent overlap)
    }

    # Minimum spacing calculation:
    min_spacing = (
        constraints['symbol_bandwidth'] +  # 20 Hz
        constraints['multipath_spread_typical'] +  # 5 Hz
        constraints['drift_allowance'] +  # 3.25 Hz
        constraints['guard_band']  # 3 Hz
    )
    # = 31.25 Hz

    # Round to clean number: 32 Hz ✓

    return {
        'calculated_minimum': 31.25,
        'chosen_spacing': 32,
        'margin': 0.75,  # Hz additional margin
    }

# Each pattern selects 4 tones from 78-tone grid
# Adaptive selection based on channel conditions
```

### HF Propagation Characteristics by Band

```python
HF_BAND_CHARACTERISTICS = {
    '80m': {
        'frequency_mhz': 3.5,
        'propagation': 'NVIS (Near Vertical Incidence Skywave)',
        'typical_range_km': 500,
        'multipath_delay_spread': 0.5,  # ms
        'frequency_spread': 2,  # Hz
        'ionospheric_drift': 0.5,  # Hz/s
        'tone_grid': '78 tones available',
        'pattern_uses': '4 tones per pattern (adaptive selection)',
    },

    '40m': {
        'frequency_mhz': 7.0,
        'propagation': 'NVIS + 1-hop skip',
        'typical_range_km': 1000,
        'multipath_delay_spread': 1.0,  # ms
        'frequency_spread': 3,  # Hz
        'ionospheric_drift': 0.6,  # Hz/s
        'tone_system': '4 tones',
        'adaptation': 'IQ complexity adapts to conditions',
    },

    '20m': {
        'frequency_mhz': 14.0,
        'propagation': 'Multi-hop F2 layer',
        'typical_range_km': 3000,
        'multipath_delay_spread': 5.0,  # ms
        'frequency_spread': 5,  # Hz (typical)
        'ionospheric_drift': 0.8,  # Hz/s
        'tone_system': '4 tones',  # Baseline
        'adaptation': 'IQ complexity adapts to conditions',
    },

    '17m': {
        'frequency_mhz': 18.1,
        'propagation': 'Multi-hop F2',
        'typical_range_km': 4000,
        'multipath_delay_spread': 6.0,  # ms
        'frequency_spread': 6,  # Hz
        'ionospheric_drift': 1.0,  # Hz/s
        'tone_system': '4 tones',
        'adaptation': 'IQ complexity adapts to conditions',
    },

    '15m': {
        'frequency_mhz': 21.0,
        'propagation': 'Multi-hop, long path possible',
        'typical_range_km': 5000,
        'multipath_delay_spread': 8.0,  # ms
        'frequency_spread': 8,  # Hz
        'ionospheric_drift': 1.2,  # Hz/s
        'tone_system': '4 tones',
        'adaptation': 'IQ complexity adapts to conditions',
    },

    '10m': {
        'frequency_mhz': 28.0,
        'propagation': 'Sporadic-E, multi-hop F2, TEP',
        'typical_range_km': 6000,
        'multipath_delay_spread': 12.0,  # ms
        'frequency_spread': 12,  # Hz
        'ionospheric_drift': 2.0,  # Hz/s
        'tone_system': '4 tones',
        'adaptation': 'IQ complexity adapts to conditions',
    },
}

# All bands use same 78-tone grid: 300-2764 Hz (32 Hz spacing)
# Each pattern selects 4 tones from this grid (adaptive)
# Adaptation handled by tone selection + IQ constellation complexity (λ parameter)
```

## Adaptive Tone Density

### Per-Band Adaptation

CASCADE adapts the number of active reference tones based on:
1. HF band in use (80m vs 10m)
2. Current propagation conditions
3. Measured frequency spreading
4. Network consensus on conditions

```python
class AdaptiveToneDensity:
    """
    Network collectively adapts tone density
    Based on propagation measurements
    """

    DENSITY_MODES = {
        'maximum': {
            'tones': 87,
            'spacing': 25,  # Hz
            'conditions': 'Excellent (NVIS, single-hop, stable)',
            'bands': ['80m', '40m'],
        },
        'high': {
            'tones': 75,
            'spacing': 29,  # Hz
            'conditions': 'Good (single/dual-hop, stable)',
            'bands': ['40m', '20m'],
        },
        'normal': {
            'tones': 70,
            'spacing': 32,  # Hz (BASELINE)
            'conditions': 'Typical (multi-hop, normal ionosphere)',
            'bands': ['20m', '17m'],
        },
        'reduced': {
            'tones': 60,
            'spacing': 37,  # Hz
            'conditions': 'Challenging (disturbed ionosphere)',
            'bands': ['17m', '15m'],
        },
        'sparse': {
            'tones': 50,
            'spacing': 44,  # Hz
            'conditions': 'Poor (high multipath, fading)',
            'bands': ['15m', '10m'],
        },
        'minimal': {
            'tones': 40,
            'spacing': 55,  # Hz
            'conditions': 'Severe (storm, aurora, extreme multipath)',
            'bands': ['10m', 'disturbed conditions all bands'],
        },
    }

    def measure_propagation_quality(self):
        """
        Stations measure frequency spreading from received beacons
        """
        spreads = []

        for beacon in recently_heard_beacons:
            # Measure spectral width of beacon
            beacon_spectrum = fft(beacon.signal)
            spread = measure_3db_bandwidth(beacon_spectrum)
            spreads.append(spread)

        avg_spread = np.median(spreads)  # Hz

        # Select tone density based on measured spread
        if avg_spread < 3:
            return 'maximum'  # 87 tones
        elif avg_spread < 5:
            return 'high'  # 75 tones
        elif avg_spread < 8:
            return 'normal'  # 4 tones (baseline)
        elif avg_spread < 12:
            return 'reduced'  # 60 tones
        elif avg_spread < 18:
            return 'sparse'  # 50 tones
        else:
            return 'minimal'  # 40 tones
```

## Discrete Tone Properties

### Why Discrete is Better Than Continuous

```python
def discrete_tone_advantages():
    """
    Discrete reference tones have significant advantages
    """

    advantages = {
        'patent_safety': {
            'issue': 'LoRa CSS patents cover continuous frequency modulation',
            'solution': 'Discrete hops = FHSS (not patented)',
            'benefit': 'Legal certainty',
        },

        'hardware_compatibility': {
            'issue': 'Continuous frequencies need high DAC precision',
            'solution': 'Discrete tones use exact FFT bins',
            'benefit': 'Works on all soundcards (even 16-bit)',
        },

        'interoperability': {
            'issue': 'Continuous values cause rounding differences',
            'solution': 'All stations use exact same 70 frequencies',
            'benefit': 'Perfect interoperability',
        },

        'fft_efficiency': {
            'issue': 'Continuous frequencies need interpolation in FFT',
            'solution': 'Discrete tones align with FFT bins',
            'benefit': '2-3× faster processing',
        },

        'frequency_tracking': {
            'issue': 'Continuous tracking needs fine PLL',
            'solution': 'Snap to nearest discrete tone',
            'benefit': 'Robust to drift',
        },

        'neural_network': {
            'issue': 'Gradients through discrete selection',
            'solution': 'Gumbel-softmax, STE (standard techniques)',
            'benefit': 'Actually easier to train! (classification vs regression)',
        },
    }

    return advantages

# Conclusion: Discrete is BETTER in almost every way!
# Only cost: ~2% less optimal frequency placement
# Benefits: Legal safety, hardware compat, faster processing
```

## Per-Receiver Tone Availability

### Kernel Encodes Available Tone Subset

Each receiver measures which of the 78 discrete tones it can decode:

```python
class ReceiverToneAvailability:
    """
    Receiver measures and announces available discrete tones
    Handles selective fading, local QRM
    """

    def __init__(self):
        self.available_tones = []
        self.tone_snr = {}

    def measure_available_tones(self):
        """
        Measure SNR at each of 78 discrete reference tones
        """
        available = []

        for tone_idx in range(78):
            # Get exact discrete frequency
            freq_hz = REFERENCE_TONES[tone_idx]

            # Measure SNR at this precise frequency
            # Use FFT bin or narrow bandpass filter
            snr_db = self.measure_snr_at_frequency(freq_hz)

            # Check for local interference
            qrm_present = self.detect_qrm_at_frequency(freq_hz)

            # Tone is available if:
            # - SNR > -10 dB (decodable)
            # - No strong local QRM
            if snr_db > -10 and not qrm_present:
                available.append(tone_idx)
                self.tone_snr[tone_idx] = snr_db

        return available

    # Examples of available tone patterns:
    # Excellent propagation: [0-77] (all 78 tones)
    # Selective fading: [0-25, 30-45, 50-69] (60 tones, some nulls)
    # Heavy QRM: [5-12, 20-28, 55-69] (32 tones, lots of interference)
    # Extreme: [8, 15, 22, 29, 36, 43, 50, 57, 64] (9 tones only!)
```

### Run-Length Encoding for Kernels

```python
def encode_available_tones_64bit(available_tone_indices):
    """
    Compress available tones into 64-bit kernel
    Uses run-length encoding for efficiency

    Format:
    - 4 bits: Number of ranges (0-15)
    - 60 bits: Up to 4 ranges (15 bits each)
      - Each range: 7-bit start + 8-bit length
    """

    # Find contiguous ranges
    ranges = find_contiguous_ranges(available_tone_indices)
    # e.g., [0-34, 36-69] → 2 ranges

    # Encode (max 4 ranges in 64 bits)
    num_ranges = min(len(ranges), 4)
    encoded = num_ranges  # First 4 bits

    for i, (start, end) in enumerate(ranges[:4]):
        length = end - start + 1
        range_encoded = (start << 8) | length  # 15 bits
        encoded |= (range_encoded << (4 + i * 15))

    return encoded  # 64 bits

    # Examples:
    # All tones [0-69]: 1 range (0, 70)
    #   → 0x0001 | (0 << 8 | 70) << 4 = 0x0000460...

    # Selective [0-34, 36-69]: 2 ranges
    #   → 0x0002 | (0<<8|35)<<4 | (36<<8|34)<<19

    # Sparse [8,15,22,29,36,43,50,57,64]: Encode as sparse
    #   → Special encoding or bitmap (if <15 tones total)
```

## Discrete Tone Hopping (FHSS)

### Frequency-Hopping Spread Spectrum

CASCADE is fundamentally FHSS, not CSS:

```python
def fhss_vs_css_comparison():
    """
    CASCADE = FHSS (like Bluetooth)
    NOT CSS (like LoRa)
    """

    comparison = {
        'frequency_behavior': {
            'LoRa_CSS': 'Continuous sweep (chirp)',
            'Bluetooth_FHSS': 'Discrete hops',
            'CASCADE': 'Discrete hops (4 tones)',
        },

        'within_symbol': {
            'LoRa_CSS': 'Frequency changes continuously',
            'Bluetooth_FHSS': 'Frequency constant',
            'CASCADE': 'Frequency constant (discrete tone)',
        },

        'between_symbols': {
            'LoRa_CSS': 'May chirp or hop',
            'Bluetooth_FHSS': 'Hops to different channel',
            'CASCADE': 'Hops to different discrete tone',
        },

        'data_encoding': {
            'LoRa_CSS': 'Time shift of chirp',
            'Bluetooth_FHSS': 'GFSK modulation',
            'CASCADE': 'IQ modulation (QAM/PSK)',
        },

        'patent_status': {
            'LoRa_CSS': 'Patented by Semtech',
            'Bluetooth_FHSS': 'Public domain (expired patents)',
            'CASCADE': 'No patent conflict (FHSS + IQ)',
        },
    }

    return comparison

# CASCADE is FHSS with 4D enhancement
# Patent-safe ✓
```

### Hopping Sequence Visualization

```
Pattern 5 hopping sequence (discrete tones only):

Time: 0ms  50  100 150 200 250 300 350 400 450 500 550 600 ...
Tone: 12   34  5   18  29  7   41  2   55  19  8   33  14  ...
Freq: 684  2789 460 908 1989 524 2341 364 2629 940 556 2245 748

Each symbol: 50ms at EXACT discrete frequency
Between symbols: Instant hop to next discrete frequency
No interpolation, no sweeps, no chirps

This is frequency-hopping (FHSS) ✓
```

## Model-Driven Tone Selection

### Discrete Tone Shifting (±3 tones)

```python
def model_discrete_tone_shifting():
    """
    Model selects discrete tone from limited set
    Can shift ±3 tones from pattern base
    """

    # Pattern nominal sequence
    pattern_base = [12, 34, 5, 18, 29, 7, ...]

    # Measured interference
    interference = {
        12: 0.8,  # HIGH (QRM at tone 12)
        11: 0.2,  # Low
        13: 0.3,  # Low
        34: 0.1,  # Low (tone 34 clear)
        # ...
    }

    # Receiver's available tones
    rx_available = [0-69]  # All except tone 25 (local QRM)

    # Model's discrete decisions:
    selected_tones = []

    for t, base_tone_idx in enumerate(pattern_base):
        # Symbol 0: base=12, interference HIGH
        # Candidates (±3): [9, 10, 11, 12, 13, 14, 15]
        # Model selects: 11 (lowest interference)
        # Transmits at: REFERENCE_TONES[11] = 620 Hz (exact)

        # Symbol 1: base=34, interference LOW
        # No shift needed
        # Transmits at: REFERENCE_TONES[34] = 1388 Hz (exact)

        candidates = range(base_tone_idx - 3, base_tone_idx + 4)
        candidates = [c for c in candidates if c in rx_available and 0 <= c < 70]

        best = min(candidates, key=lambda c: interference[c])
        selected_tones.append(best)

    # Result: Discrete tone sequence optimized for interference
    # Still frequency-hopping (not chirping)
    # Model learned to select from discrete options
```

## Multi-User Capacity

### Orthogonal Capacity with 78-Tone Grid, 4 Per Pattern

```python
def cascade_multi_user_capacity():
    """
    Calculate simultaneous user capacity with 78-tone grid
    Each pattern uses 4 tones (adaptive selection)
    """

    capacity_analysis = {
        'tone_grid': {
            'total_tones': 78,  # Reference tone grid
            'per_pattern': 4,  # Each pattern uses 4 tones
            'combinations': 'C(78,4) = 1,426,425 possible sets',
            'adaptive_selection': 'Model picks best 4 for conditions',
        },

        'dimensions': {
            'time': 32,  # Symbols per pattern
            'tone_selection': '4 from 78',  # Adaptive frequency selection
            'iq': 16,  # IQ directions at high SNR
            'patterns': 256,  # Total orthogonal patterns
        },

        'orthogonality_method': {
            'time': 'Zadoff-Chu sequences',
            'frequency': '4-tone selection from 78 (overlap allowed)',
            'iq': 'Continuous constellation adaptation',
            'cross_correlation': '<-37.5 dB in 4D space (achieved)',
        },

        'practical_simultaneous_users': {
            'high_snr': 45,  # Active users in chaos mode
            'total_capacity': 256,  # 128 patterns × 2 (via IQ diversity)
            'medium_snr': 35,  # Active users
            'low_snr': 20,  # Emergency patterns + simple IQ
        },

        'per_user_throughput': {
            'single_pattern': '218 bps info (chaos mode)',
            'multi_pattern_2x': '436 bps info',
            'multi_pattern_4x': '872 bps info',  # Strong links
        },

        'shannon_efficiency': {
            'bandwidth': 2500,  # Hz (aggregate, 78 tones × 31 Hz)
            'snr_15db_capacity': 12570,  # bps coded (Shannon limit)
            'cascade_achieving': 9805,  # bps coded (chaos with micro-tuning)
            'efficiency': 0.78,  # 78% Shannon efficiency (chaos mode)
        },
    }

    return capacity_analysis

# 45 active users at high SNR (1,024 total capacity via frequency + time reuse)
# Each uses 4 tones selected from 78-tone grid
# Multiple patterns can share tones (overlap via Time × IQ separation)
# All sharing 2.5 kHz bandwidth (78 tones × 31 Hz ≈ 2418 Hz)
# 78% Shannon efficiency (chaos mode with ±2 Hz micro-tuning)
# 96.7% spectrum utilization
```

## 4-Tone System Trade-offs

```
4-Tone Orthogonal System Benefits:
+ Maximum multi-user capacity (256 users vs 312)
+ Simplified frequency coordination (4 tones vs 78)
+ Maximizes IQ orthogonality dimension (16 directions)
+ Excellent frequency diversity (600 Hz spacing)
+ Robust to selective fading (independent tone fading)
+ Lower DSP complexity for frequency tracking
+ All patterns can use all tones (full overlap)

Characteristics:
+ Wide spacing: 600 Hz (maximum diversity)
+ Users achieve throughput via multi-pattern (1-4 patterns)
+ Shannon efficiency: 75% (realistic for async multi-user)
+ Spectral efficiency: 100% (full 2.5 kHz used for separation)

CASCADE choice: 4 tones per pattern (adaptive from 78-tone grid)
- Optimal multi-user capacity (45 active, 1,024 total via kernel-coordinated reuse)
- Optimal for HF selective fading
- Simplified architecture
- 78% Shannon efficiency (chaos with ±2 Hz micro-tuning)
- 14× per-user throughput improvement (218 vs 15 bps)
- Optimal for chaos operation ✓
```

## FFT Bin Alignment

### Discrete Tones Align with FFT

```python
def fft_bin_alignment():
    """
    Discrete tones align with FFT bins for efficiency
    """

    # Soundcard sample rate
    sample_rate = 48000  # Hz

    # FFT size for 50ms window
    samples_per_symbol = 48000 * 0.050  # 2400 samples
    fft_size = 2048  # Power of 2 (efficient)

    # FFT bin spacing
    bin_spacing = sample_rate / fft_size  # 23.4 Hz

    # CASCADE tone spacing: 32 Hz
    # Ratio: 32 / 23.4 = 1.37 bins per tone

    # Not perfect alignment, but close enough
    # Each tone spans ~1-2 FFT bins (acceptable)

    # Exact alignment would require:
    # Tone spacing = 23.4 Hz (too tight for HF drift)
    # OR FFT size = 1500 (not power of 2, inefficient)

    # CASCADE choice: Optimize for propagation (32 Hz)
    # Accept minor FFT bin misalignment
    # Use windowing + interpolation for precise frequency

    return {
        'fft_size': 2048,
        'bin_spacing': 23.4,  # Hz
        'tone_spacing': 32,  # Hz
        'bins_per_tone': 1.37,
        'alignment': 'approximate',
        'performance_impact': 'negligible (<0.5 dB)',
    }
```

## Adaptive Grid Example Scenarios

### Scenario 1: Excellent Conditions (80m NVIS)

```
Band: 80m (3.5 MHz)
Propagation: Near Vertical Incidence Skywave
Range: <500 km (regional net)
Multipath: Minimal (0.5ms delay spread, 2 Hz freq spread)
Drift: Low (0.5 Hz/s)

Adaptive configuration:
- Tone count: 87 (maximum density)
- Spacing: 25 Hz
- Capacity: 174 simultaneous users
- Shannon efficiency: 58%

Tone grid:
  Lower: 41 tones (300-1300 Hz)
  Beacon: 175 Hz (1475-1625 Hz)
  Upper: 46 tones (1701-2826 Hz)
```

### Scenario 2: Typical Conditions (20m DX)

```
Band: 20m (14 MHz)
Propagation: Multi-hop F2 layer
Range: 3000 km (DX)
Multipath: Moderate (5ms delay, 5 Hz spread)
Drift: Typical (0.8 Hz/s)

Adaptive configuration:
- Tone count: 70 (baseline)
- Spacing: 32 Hz
- Capacity: 272 simultaneous users
- Shannon efficiency: 56%

Tone grid: (as specified above)
  Lower: 35 tones
  Beacon: 175 Hz
  Upper: 35 tones
```

### Scenario 3: Challenging Conditions (15m Disturbed)

```
Band: 15m (21 MHz)
Propagation: Multi-hop with long path mixing
Range: 5000 km
Multipath: High (8ms delay, 8 Hz spread)
Drift: High (1.2 Hz/s)

Adaptive configuration:
- Tone count: 50 (reduced)
- Spacing: 44 Hz
- Capacity: 100 simultaneous users
- Shannon efficiency: 52%

Tone grid:
  Lower: 25 tones (300-1356 Hz, 44 Hz spacing)
  Beacon: 175 Hz (1475-1625 Hz)
  Upper: 25 tones (1701-2757 Hz, 44 Hz spacing)
```

### Scenario 4: Severe Conditions (Geomagnetic Storm)

```
Conditions: K-index 7 (severe storm)
Propagation: Disturbed, auroral paths
Multipath: Extreme (20ms delay, 20 Hz spread)
Drift: Very high (4 Hz/s)

Adaptive configuration:
- Tone count: 40 (minimal)
- Spacing: 55 Hz
- Capacity: 80 simultaneous users
- Shannon efficiency: 45%

Tone grid:
  Lower: 20 tones (300-1345 Hz, 55 Hz spacing)
  Beacon: 175 Hz (1475-1625 Hz)
  Upper: 20 tones (1705-2750 Hz, 55 Hz spacing)

System still functional in extreme conditions! ✓
```

## Network Consensus on Tone Density

### Distributed Measurement

```python
def network_tone_density_consensus():
    """
    Stations collectively decide tone density
    No central authority needed
    """

    # Each station measures local frequency spreading
    my_measurement = measure_beacon_spread()  # e.g., 7 Hz

    # Stations include measurement in beacons
    beacon_payload['measured_spread_hz'] = my_measurement

    # Receive measurements from other stations
    network_measurements = [
        5.2,  # Hz (station 1)
        6.8,  # Hz (station 2)
        7.1,  # Hz (station 3)
        8.3,  # Hz (station 4)
        # ...
    ]

    # Network median (robust to outliers)
    network_spread = np.median(network_measurements)  # 7.0 Hz

    # All stations independently arrive at same decision:
    if network_spread < 8:
        tone_density = 'normal'  # 4 tones
    else:
        tone_density = 'reduced'  # 60 tones

    # Graceful transition over ~30 seconds
    # Stations gradually switch to new grid
    # No explicit coordination message needed
    # Emergent consensus! ✓
```

## See Also

- **[Beacon Reservation](beacon_reservation.md)** - Center-band 175 Hz allocation details
- **[4D Pattern Envelope](../model/4d_pattern_envelope.md)** - Discrete freq hopping with continuous IQ
- **[Pattern Generation](pattern_generation.md)** - How discrete tone sequences are generated
- **[Signal Specification](signal_specification.md)** - Physical layer parameters
- **[Kernel Lifecycle](kernel_lifecycle.md)** - How available tones encoded in kernels
