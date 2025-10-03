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
- 256 patterns available (192 message patterns for user traffic)
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

### Multi-User Decode with Hardware Constraints

The Signal Expert must handle variable user counts based on deployment hardware:

**Hardware-adaptive decode**:
```python
def signal_expert_decode(mixed_signal, hardware_capacity):
    """
    Decode users in priority order up to hardware limit

    Args:
        mixed_signal: Combined signal from all users
        hardware_capacity: Max users this hardware can process

    Returns:
        Variable-length list of decoded users (hardware-dependent)
    """
    # Detect all potential users via pattern correlation
    detected_users = correlate_all_patterns(mixed_signal)  # Up to 192 message patterns

    # Sort by signal strength (Shannon-optimal)
    detected_users.sort(key=lambda u: u.snr, reverse=True)

    # Decode up to hardware capacity
    decoded = []
    for user in detected_users[:hardware_capacity]:
        user_signal = separate_user(mixed_signal, user.pattern)
        decoded_data = decode_constellation(user_signal)
        decoded.append(decoded_data)

        if len(decoded) >= hardware_capacity:
            break  # Hardware limit reached

    return decoded  # Length: 10 on RPi4, 50 on Coral, 100+ on GPU
```

**Training across capacities**:
- Trained with random capacity limits (10-100 users)
- Learns to prioritize strong signals
- Graceful degradation when capacity exceeded
- Single model works across all hardware tiers

**Output variability**: Signal Expert returns variable-length results:
- Raspberry Pi 4: Decodes 10-20 users (strongest)
- RPi + Coral: Decodes 50-80 users (nearly everyone)
- GPU: Decodes 100+ users (everyone, plus weak signals)

This is Shannon-optimal: limited hardware naturally prioritizes strong signals.

### Performance Metrics

| Metric | RPi 4 Only | RPi + Coral | Desktop | GPU |
|--------|------------|-------------|---------|-----|
| **Max users decoded** | 10-20 | 50-80 | 25-40 | 100+ |
| **Detection accuracy** | >99% (strong sigs) | >99% (all sigs) | >99% | >99% |
| **Separation quality** | >20 dB | >25 dB | >22 dB | >30 dB |
| **Computation time** | ~25-30ms | ~3-5ms | ~12-15ms | ~2-3ms |
| **Parameters** | ~1.2M (INT8) | ~1.2M (INT8) | ~1.2M (INT8/FP16) | ~1.2M (FP32) |

### Beacon Detection and Separation

The Signal Expert also handles **beacon decoding** alongside message separation. Beacons are transmitted on interstitial frequencies (between message tones) with different symbol timing, allowing the model to distinguish and extract them.

**Beacon characteristics learned by model:**
```python
# Training includes beacons as additional signal type
training_scenario = {
    'messages': [
        {'pattern': 5, 'tones': [0, 312, 625, ...], 'symbols': 50ms, 'mod': '8-QAM'},
        {'pattern': 12, 'tones': [0, 312, 625, ...], 'symbols': 50ms, 'mod': 'QPSK'},
    ],
    'beacons': [
        {'tones': [78, 234, 1718, 1953], 'symbols': 160ms, 'mod': '4-FSK'},  # Normal beacons
        {'tones': [78, 234, 1718, 1953], 'symbols': 160ms, 'mod': '4-FSK'},
    ],
    'interference': qrm_sample
}

# Model learns to separate by time-frequency signature:
# - Messages: 50ms symbols, primary tones (0, 312, 625, ...)
# - Normal beacons: 160ms symbols, interstitial tones (78, 234, 1718, 1953)
# - Emergency beacons: 800ms symbols, RESERVED tones (156, 468, 781, 1093)
# - Different symbol rates → different correlation properties
```

**Signal Expert output includes beacons:**
```python
decoded = signal_expert.separate(mixed_signal)

# Returns labeled items (variable length):
[
    {'type': 'message', 'pattern': 5, 'data': bytes},
    {'type': 'beacon', 'callsign_hash': 0xA3F2, 'snr': -12},  # ← Beacon decoded!
    {'type': 'message', 'pattern': 12, 'data': bytes},
    {'type': 'beacon', 'callsign_hash': 0x7F3D, 'snr': -18},  # Another beacon
]

# Protocol routes beacons to cache, messages to handlers
```

**No protocol-layer decoding** - Signal Expert does all demodulation, protocol just routes based on type field.

**Training loss includes beacon accuracy with emergency weighting:**
```python
def signal_expert_loss(predicted, ground_truth):
    message_loss = separation_loss(predicted.messages, ground_truth.messages)

    # Separate beacon loss by emergency status
    beacon_loss_normal = 0
    beacon_loss_emergency = 0

    for pred_beacon, true_beacon in zip(predicted.beacons, ground_truth.beacons):
        loss = detection_loss(pred_beacon, true_beacon)

        if true_beacon.emergency_flag:
            beacon_loss_emergency += loss
        else:
            beacon_loss_normal += loss

    # Weight emergency beacons 5× higher than normal beacons
    total_beacon_loss = beacon_loss_normal + 5.0 * beacon_loss_emergency

    # Balance messages (80%) and beacons (20%, but emergency beacons effectively 100%)
    return 0.8 * message_loss + 0.2 * total_beacon_loss
```

**Emergency beacon prioritization:**
- Normal beacons: 20% weight in loss (less critical than messages)
- Emergency beacons: 100% weight (5× multiplier = critical as messages)
- Model learns to reliably detect emergency beacons even in high-interference scenarios
- Emergency beacons transmitted 6× per minute (every 10s, redundancy ensures detection)

### Spectrum Allocation Expert - Emergency Frequency Protection

The Spectrum Allocation Expert learns to **avoid emergency frequencies during encoding:**

```python
def spectrum_expert_with_emergency_protection(features):
    """
    Allocate frequency resources while protecting emergency spectrum
    """
    EMERGENCY_FREQS = [156, 468, 781, 1093]  # Hz - RESERVED, never use for messages
    AVAILABLE_SPECTRUM = [
        (0, 146),      # Below first emergency freq
        (166, 458),    # Between emergency freqs 1-2
        (478, 771),    # Between emergency freqs 2-3
        (791, 1083),   # Between emergency freqs 3-4
        (1103, 2500)   # Above last emergency freq
    ]

    # Allocate frequency only within available ranges
    frequency_allocation = self.allocate_within_ranges(features, AVAILABLE_SPECTRUM)

    return frequency_allocation  # Guaranteed no overlap with [156, 468, 781, 1093]

# Training loss for encoder:
def encoder_loss_with_emergency_avoidance(encoded_signal, ground_truth):
    # Standard encode-decode loss
    decoded = model.decode(encoded_signal)
    base_loss = reconstruction_loss(decoded, ground_truth)

    # Check if encoder used emergency frequencies
    spectrum = fft(encoded_signal)
    emergency_interference = 0

    for freq in [156, 468, 781, 1093]:
        energy_at_emergency = measure_energy_at_freq(spectrum, freq, bandwidth=10)
        if energy_at_emergency > -40:  # Any significant energy = violation
            # Massive penalty (100× base loss)
            emergency_interference += 100.0 * energy_at_emergency

    # Total loss
    return base_loss + emergency_interference
```

**Model learns hard constraint**: Emergency frequencies [156 ± 10 Hz, 468 ± 10 Hz, 781 ± 10 Hz, 1093 ± 10 Hz] are completely forbidden for message transmission.

### Multi-Scale Signal Separation

The Signal Expert handles multiple signal types with different symbol rates and frequencies **using a single model**:

**Signal types processed simultaneously:**
```python
# Model receives full-spectrum IQ (48 kHz sampling)
full_spectrum = receive_from_soundcard()  # 0-24 kHz

# Signal Expert separates all types in single forward pass:
decoded_items = signal_expert.separate_all(full_spectrum)

# Returns mixed list:
[
    {'type': 'message', 'pattern': 5, 'tones': [0,312,...], 'symbols': 50ms},
    {'type': 'beacon', 'pattern': 12, 'tones': [78,234,1718,1953], 'symbols': 160ms},
    {'type': 'message', 'pattern': 18, 'tones': [0,312,...], 'symbols': 50ms},
    {'type': 'emergency', 'tones': [156,468,781,1093], 'symbols': 800ms, 'mod': '4-FSK'},
]

# Different frequencies, different symbol rates, all decoded together
```

**Shared Encoder multi-scale architecture enables this:**
```python
# Shared encoder has multiple temporal scales (from shared_encoder.md)
self.conv_short = Conv1d(kernel_size=64)    # 1.3ms (for 50ms symbols)
self.conv_medium = Conv1d(kernel_size=256)  # 5.3ms (for 160ms symbols)
self.conv_long = Conv1d(kernel_size=1024)   # 21ms (for 500ms symbols)

# Features from all scales concatenated
# Signal Expert uses multi-scale features to handle any symbol rate
```

### Envelope-Based Separation for Microsecond Offsets

**Model separates overlapping transmissions with microsecond start-time differences:**

```python
class MicrosecondEnvelopeSeparator:
    """Separate signals with <1ms timing offsets"""

    def __init__(self):
        # High-resolution onset detector
        self.onset_conv = nn.Conv1d(2, 128, kernel_size=96)  # 2ms window @ 48kHz

        # Envelope analyzer
        self.envelope_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64)  # Envelope features
        )

    def separate_by_envelope(self, iq_48khz):
        """
        Separate overlapping signals using envelope analysis

        Input: IQ samples at 48 kHz
        Features used:
        1. Onset timing (amplitude steps when new signal starts)
        2. Beating patterns (amplitude modulation from phase offsets)
        3. Clock drift signatures (unique frequency error per radio)
        4. Phase trajectories (how phase evolves differently per radio)
        """

        # Extract amplitude envelope
        envelope = torch.abs(iq_48khz)  # Complex → magnitude

        # Detect onset times (when new transmissions start)
        onset_features = self.onset_conv(iq_48khz)
        onset_times = detect_amplitude_steps(onset_features)  # Microsecond resolution

        # Analyze envelope beating (from overlapping tones)
        envelope_features = self.envelope_net(envelope)

        # Cluster by:
        # - Onset time (when signal started)
        # - Drift signature (frequency error pattern)
        # - Beating frequency (phase offset from others)

        clusters = cluster_by_timing_and_envelope(
            onset_times,
            envelope_features,
            num_expected_users=estimate_user_count()
        )

        # Separate signals based on clusters
        separated_signals = []
        for cluster in clusters:
            signal = extract_signal_by_cluster(iq_48khz, cluster)
            separated_signals.append(signal)

        return separated_signals  # 10-15 signals from overlapped input
```

**Training on asynchronous overlaps:**
```python
def train_envelope_separation():
    """Train on overlapping signals with random microsecond offsets"""

    for batch in training:
        # Generate 5-15 overlapping transmissions
        num_overlapping = random.randint(5, 15)

        signals = []
        for i in range(num_overlapping):
            signal = generate_cascade_signal(
                pattern=random_pattern(),
                start_offset_us=random.randint(0, 2000000),  # 0-2s in μs
                clock_drift_ppm=random.uniform(-50, 50),     # Unique drift
                symbol_rate='50ms' or '160ms' or '500ms'     # Mixed rates!
            )
            signals.append(signal)

        # Mix with sample-accurate alignment (48 kHz)
        mixed = mix_with_microsecond_precision(signals)

        # Add noise
        mixed_noisy = mixed + noise(snr=random.uniform(-5, 10))

        # Train to separate
        separated = model.separate_all(mixed_noisy)

        # Loss: Correctly identify all signals
        # Accept 50-70% accuracy for 15 overlapping (challenging!)
        # Expect 90%+ for 5 overlapping
        loss = separation_loss(separated, ground_truth)
        optimizer.step(loss)
```

**Model learns to use:**
- Amplitude envelope steps (onset timing)
- Phase beating patterns (from microsecond offsets)
- Clock drift signatures (±50 ppm creates unique trajectories)
- Multi-scale correlation (different symbol rates)

**Result**: Single model handles 50ms/160ms/500ms symbols with 10-15 simultaneous overlaps at microsecond timing resolution.

---

## Pattern Complexity Expert Network

The Pattern Complexity Expert selects the optimal pattern pool based on measured HF propagation conditions (multipath delay spread). CASCADE uses 256 hierarchical patterns organized into complexity pools—this expert picks which pool to use based on channel characteristics.

### Architecture

```
Input: [1024D shared features](shared_encoder.md#architecture)
↓
Propagation Estimation Branch:
  Dense: 1024 → 256 → 64 → 1
  Output: Estimated multipath delay spread (ms)
↓
SNR Estimation Branch:
  Dense: 1024 → 256 → 64 → 1
  Output: Estimated SNR (dB)
↓
Pool Selection Branch:
  Dense: 1024 → 512 → 256 → 6
  Softmax
  Output: P(pool = [emergency, typical_dx, good_prop, nvis, beacon_simple, beacon_emergency])
↓
Pattern Pool Mapping:
  emergency → Patterns 64-79 (minimal IQ)
  typical_dx → Patterns 80-207 (simple-moderate IQ, LARGEST POOL)
  good_prop → Patterns 208-239 (moderate-complex IQ)
  nvis → Patterns 240-255 (complex Lissajous)
  beacon_simple → Patterns 16-63 (beacon normal)
  beacon_emergency → Patterns 0-15 (beacon emergency)
↓
Feature Adaptation:
  Dense: 256 + pool_params → 512
  ReLU + BatchNorm
↓
Output: Pool selection + 512D adapted features
```

### Learned Behaviors

**Propagation Assessment**: Measures HF multipath characteristics:
- Multipath delay spread (0.5-20 ms)
- Frequency-selective fading depth
- Phase coherence time
- Ionospheric mode (NVIS, single-hop, multi-hop)

**Pool Selection**: Model learns to select pattern pool based on propagation:

```python
def select_pattern_pool(multipath_delay_ms, snr_db):
    """
    Learned pool selection based on propagation and SNR
    Returns pattern ID range to use
    """
    # Measure propagation
    if multipath_delay_ms < 1:
        # NVIS or excellent propagation
        if snr_db > 10:
            return range(240, 256)  # NVIS exceptional (complex Lissajous)
        else:
            return range(208, 240)  # Good prop (moderate-complex)

    elif multipath_delay_ms < 8:
        # Typical DX (MOST COMMON on HF)
        return range(80, 208)  # 128 patterns, λ=0.3-0.5

    else:
        # Severe multipath or emergency
        if snr_db < -10:
            return range(64, 80)  # Emergency pool (minimal IQ)
        else:
            return range(80, 144)  # Lower typical DX pool

    # For beacons:
    if using_beacon_channel:
        if emergency:
            return range(0, 16)  # Beacon emergency
        else:
            return range(16, 64)  # Beacon normal
```

**Graceful Adaptation**: Model handles pool transitions:
- Hysteresis prevents oscillation (2ms multipath window)
- Smooth transitions between pools
- Predictive adaptation based on propagation trends

### Pattern Pool Characteristics

Each pool optimized for HF propagation conditions:

- **Emergency pools (0-15, 64-79)**: BPSK line, maximum robustness, -28 dB capable
- **Typical DX pool (80-207)**: 128 patterns, λ=0.3-0.5, **MOST HF OPERATION**
- **Good propagation (208-239)**: 32 patterns, λ=0.5-0.7, single-hop F2
- **NVIS exceptional (240-255)**: 16 patterns, λ=0.7-0.9, rarely used on HF

Enables communication across diverse HF propagation from clean NVIS to severe multipath DX.

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