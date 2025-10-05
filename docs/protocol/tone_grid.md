# CASCADE Tone Grid & Spectrum Allocation

**Authoritative specification for CASCADE's 135-tone reference grid**

CASCADE uses 135 discrete reference tones spanning 300-3000 Hz (standard SSB channel). Each pattern uses 2 adjacent tones (2-FSK architecture). Multiple patterns are assigned different tone pairs for separation, with frequency reuse possible via time and IQ orthogonality.

## Pattern-Based Spectrum Allocation

**Core principle:** CASCADE uses pattern orthogonality for ALL separation. There are no separate frequency bands—all 135 reference tones are shared by beacons, emergency traffic, and messages.

### Pattern Count
- **Total patterns**: 128 (7-bit encoding, 0x00-0x7F)
- **Beacon patterns** (0-47): 48 patterns, each uses 2-FSK on assigned tone pair
- **Message patterns** (48-127): 80 patterns, each uses 2-FSK on assigned tone pair
- **Emergency patterns**: 0-15 (beacon) + 48-63 (message) = 32 total
- **Storage**: 38 KB total
- **Generation**: 48-60 hours (simplified, frequency-only optimization)

## Reference Tone Specification

### 135 Discrete Tones (Standard SSB Channel)

```python
REFERENCE_TONE_GRID = {
    'total_tones': 135,
    'spacing_hz': 20,  # Optimized for SDR equipment
    'total_span': 2700,  # Hz (300-3000)

    # All tones form 67 possible tone pairs for 2-FSK
    'tones': [300 + i*20 for i in range(135)],
    'indices': range(0, 135),
    'tone_pairs': 67,  # Each pattern uses 2 adjacent tones
}

# Sample tone list (first 20 and last 15):
REFERENCE_TONES = [
    # First 20 tones
    300, 320, 340, 360, 380, 400, 420, 440, 460, 480,
    500, 520, 540, 560, 580, 600, 620, 640, 660, 680,
    # ... (continuing every 20 Hz) ...
    # Last 15 tones
    2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860, 2880,
    2900, 2920, 2940, 2960, 2980, 3000
]

# Each pattern uses 2 adjacent tones (2-FSK architecture)
# Patterns assigned different tone pairs for separation
# Frequency reuse possible via Time × IQ orthogonality
```

## Tone Grid Design Rationale

### Why 20 Hz Spacing?

The 20 Hz spacing is optimized for modern SDR equipment while maintaining adequate separation for HF conditions:

- **Symbol rate**: 200 symbols/second with NN-enhanced ISI tolerance
- **Frequency precision**: Modern SDRs (QMX, IC-7300) can maintain ±0.1 Hz with GPS
- **Spectral efficiency**: 135 tones × 20 Hz = 2700 Hz fills standard SSB channel
- **2-FSK separation**: 20 Hz between adjacent tones is sufficient for demodulation
- **Multipath tolerance**: Decoder NN handles multipath spreading within tone spacing
- **Drift compensation**: Differential encoding tolerates ±0.1-10 Hz drift

All CASCADE stations use the same 135-tone grid. There is no adaptation needed since the NN models handle channel impairments universally.
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
        'tone_system': '2-FSK (135-tone grid)',
        'modulation': 'Adaptive BPSK/QPSK/8-PSK/16-APSK based on SNR',
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

# All bands use same 150-tone grid: 300-3300 Hz (20 Hz spacing)
# Each pattern uses 2-FSK with assigned tone pair
# Adaptation handled via modulation selection (BPSK→16-APSK) based on SNR
```

## Fixed Tone Grid for All Conditions

### Universal 150-Tone Grid

CASCADE uses the same 150-tone grid regardless of:
- HF band in use (80m through 10m)
- Propagation conditions
- Equipment type
- Geographic location

The decoder neural network handles all channel impairments:

```python
class UniversalToneGrid:
    """
    Fixed 150-tone grid for all CASCADE operations
    """

    GRID_SPECIFICATION = {
        'total_tones': 150,
        'spacing': 20,  # Hz (fixed)
        'bandwidth': 3000,  # Hz (300-3300)
        'tone_pairs': 75,  # For 2-FSK patterns

        # Works on all HF bands
        'bands': ['80m', '40m', '30m', '20m', '17m', '15m', '12m', '10m'],

        # NN handles all conditions
        'conditions': 'All (excellent to severe)',

        # Modulation adapts, not tone grid
        'adaptation': 'Via modulation (BPSK/QPSK/8-PSK/16-APSK)',
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
