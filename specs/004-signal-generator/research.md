# Research: Signal Generator Technical Decisions

**Feature**: 004-signal-generator
**Date**: 2025-10-07
**Phase**: Phase 0 - Research & Technical Design
**Status**: Complete

---

## Executive Summary

This research phase establishes the technical foundation for CASCADE's dual-part signal generator:
1. **Core Signal Generator**: Produces clean V2-compliant IQ signals using dual-layer modulation (2-FSK pattern layer + BPSK/QPSK/8-PSK/16-APSK data layer)
2. **Synthetic Data Orchestrator**: Applies realistic HF channel impairments (AWGN, QRN, multipath, QRM)

Key findings:
- Dual-layer architecture requires independent GMSK pulse shaping (pattern) and IQ constellation mapping (data)
- 28-byte kernel provides all discrete parameters (pattern_id, channel, modulation, polar_rate)
- NumPy/SciPy sufficient for all DSP operations (no specialized SDR libraries needed)
- Channel simulation requires separate models for each impairment type

---

## 1. Dual-Layer Modulation Architecture

### Decision: Independent Layer Processing
**Rationale**: CASCADE V2 uses two simultaneous modulation layers that must be combined in the IQ domain.

**Layer 1: Pattern Skeleton (2-FSK)**
- Binary pattern [0,1,1,0...] selects tone A or tone B from frequency pair
- GMSK pulse shaping (BT=0.3) applied to FSK transitions
- Provides ~21 dB processing gain through pattern orthogonality
- Generated independently, then serves as carrier for Layer 2

**Layer 2: Data Payload (BPSK/QPSK/8-PSK/16-APSK)**
- IQ modulation applied ON TOP of GMSK-shaped FSK tones
- Constellation points encode user data bits
- Modulation scheme selected via kernel parameter
- Applied to whichever tone (A or B) the pattern selects at each symbol

**Combination Method**:
```python
# Pseudocode for dual-layer generation
gmsk_signal = generate_gmsk_fsk(pattern_bits, freq_pair)  # Layer 1
iq_symbols = map_constellation(data_bits, modulation)      # Layer 2
combined_signal = gmsk_signal * iq_symbols                 # Multiply in IQ domain
```

**Alternatives Considered**:
- ❌ Single-layer GMSK with embedded data: Cannot achieve 2-4 bits/symbol
- ❌ Time-multiplexed layers: Reduces throughput by 50%
- ✅ **Selected**: IQ domain multiplication preserves both layers simultaneously

**Implementation Approach**:
- Generate GMSK 2-FSK as complex baseband signal (I+jQ)
- Generate constellation symbols as complex scalars
- Element-wise multiplication combines layers
- Maintains constant envelope property of GMSK

---

## 2. GMSK Pulse Shaping (BT=0.3)

### Decision: SciPy gaussian filter + phase integration
**Rationale**: GMSK (Gaussian Minimum Shift Keying) smooths FSK transitions to reduce spectral sidelobes.

**Technical Specifications**:
- **BT Product**: 0.3 (Bandwidth × symbol Time)
- **Symbol Rate**: 200 symbols/second (5ms per symbol)
- **Bandwidth**: 0.3 / 0.005s = 60 Hz (-3dB bandwidth)
- **Occupied Bandwidth**: ~30 Hz (99% power) per channel
- **Filter Length**: 4-5 symbol periods (20-25ms)

**GMSK Generation Algorithm**:
1. Start with binary pattern: [0,1,1,0...] → convert to NRZ: [-1,+1,+1,-1...]
2. Upsample to sample rate (e.g., 48 kHz): Insert zeros between symbols
3. Apply Gaussian filter: `scipy.signal.gaussian(M, std)` where std = sqrt(ln(2)) / (2π × BT)
4. Integrate filtered signal to get phase: `φ(t) = π × ∫ filtered(τ) dτ`
5. Convert to IQ: `I(t) = cos(2πf_c·t + φ(t))`, `Q(t) = sin(2πf_c·t + φ(t))`

**Key Parameters**:
```python
BT = 0.3
symbol_rate = 200  # symbols/second
symbol_duration = 1 / symbol_rate  # 5ms
filter_span = 4  # symbols
sample_rate = 48000  # Hz (standard sound card rate)
samples_per_symbol = sample_rate / symbol_rate  # 240 samples

# Gaussian filter std deviation
std = np.sqrt(np.log(2)) / (2 * np.pi * BT * symbol_duration)
```

**Validation**:
- Measure occupied bandwidth: Should be ~30 Hz at -40 dB
- Check phase continuity: No discontinuities at symbol boundaries
- Verify constant envelope: |I² + Q²| should be constant

**Alternatives Considered**:
- ❌ Simple rectangular pulse: Excessive spectral sidelobes (interferes with adjacent channels)
- ❌ Raised cosine: Not constant envelope (peaks cause distortion)
- ❌ MSK (BT=∞): Wider bandwidth than needed
- ✅ **Selected**: GMSK BT=0.3 balances bandwidth efficiency and constant envelope

**Libraries**:
- `scipy.signal.gaussian()`: Generate Gaussian filter
- `numpy.convolve()`: Apply filter to upsampled NRZ
- `numpy.cumsum()`: Integrate to get phase trajectory

---

## 3. Frequency Plan (135 Channels, 20 Hz Spacing)

### Decision: 135 discrete channels at 20 Hz spacing (300-3000 Hz)
**Rationale**: Updated CASCADE V2 protocol uses individual channels, not 2-tone pairs.

**Channel Grid**:
```
Channel 0:   300 Hz
Channel 1:   320 Hz
Channel 2:   340 Hz
...
Channel 134: 2980 Hz

Total span: 2700 Hz (300-3000 Hz)
Spacing: 2700 / 135 = 20 Hz
```

**Frequency Pair Formation**:
- Core Generator accepts `frequency_pair` parameter (0-66) from kernel
- Each pair uses two adjacent channels: `[2*pair_id, 2*pair_id + 1]`
- Example: `frequency_pair=25` → channels 50 and 51 → tones at 1300 Hz and 1320 Hz

**Tone Selection Logic**:
```python
def get_tone_frequencies(frequency_pair: int) -> tuple[float, float]:
    """Get tone A and tone B frequencies from pair ID."""
    channel_a = 2 * frequency_pair
    channel_b = 2 * frequency_pair + 1

    base_freq = 300  # Hz
    spacing = 20     # Hz

    tone_a = base_freq + channel_a * spacing
    tone_b = base_freq + channel_b * spacing

    return (tone_a, tone_b)
```

**Guard Bands**:
- No guard bands between pairs (flip-orthogonal patterns handle interference)
- GMSK BT=0.3 produces ~30 Hz occupied bandwidth
- 40 Hz spacing (per pair) provides 10 Hz margin

**Validation**:
- Verify all 67 pairs fit within 300-3000 Hz
- Check pair 66: channels 132, 133 → 2940 Hz, 2960 Hz (fits!)
- Measure crosstalk between adjacent pairs with GMSK shaping

**Alternatives Considered**:
- ❌ 10 Hz spacing: Too narrow for GMSK BT=0.3 (would cause inter-channel interference)
- ❌ 40 Hz spacing: Wastes spectrum, reduces capacity
- ✅ **Selected**: 20 Hz spacing matches GMSK bandwidth while maximizing channels

---

## 4. Constellation Mapping (Layer 2 Data)

### Decision: Standard PSK/APSK constellations
**Rationale**: Well-established modulation schemes with known performance characteristics.

**BPSK (SNR < 0 dB)**:
- 2 constellation points: [+1, -1] (on real axis)
- 1 bit per symbol
- Polar encoding required: Polar(n, k) where k/n = rate

**QPSK (SNR 0-10 dB)**:
- 4 constellation points: [1+j, 1-j, -1+j, -1-j] / √2
- 2 bits per symbol
- Gray coding: adjacent points differ by 1 bit

**8-PSK (SNR 10-20 dB)**:
- 8 constellation points: e^(j·2π·k/8) for k=0..7
- 3 bits per symbol
- Gray coding for adjacent points

**16-APSK (SNR > 20 dB)**:
- 16 amplitude-phase shift keying points
- Inner ring: 4 points at radius r1
- Outer ring: 12 points at radius r2
- 4 bits per symbol
- Optimized for non-linear amplifiers (better than 16-QAM)

**IQ Symbol Generation**:
```python
def map_to_constellation(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map bit groups to complex IQ symbols."""
    if modulation == 'BPSK':
        # 1 bit → ±1
        return 2 * bits - 1

    elif modulation == 'QPSK':
        # 2 bits → Gray-coded QPSK
        I = 2 * bits[::2] - 1    # Even bits → I
        Q = 2 * bits[1::2] - 1   # Odd bits → Q
        return (I + 1j * Q) / np.sqrt(2)

    elif modulation == '8-PSK':
        # 3 bits → 8 phases
        symbols = bits[::3] * 4 + bits[1::3] * 2 + bits[2::3]
        return np.exp(1j * 2 * np.pi * symbols / 8)

    elif modulation == '16-APSK':
        # 4 bits → 16-APSK (4+12 ring)
        # Implementation details from DVB-S2 standard
        ...
```

**Power Normalization**:
- All constellations normalized to unit average power
- BPSK: power = 1
- QPSK: power = 1 (factor of 1/√2 already applied)
- 8-PSK: power = 1 (unit circle)
- 16-APSK: Adjust ring radii for unit power

**Alternatives Considered**:
- ❌ QAM: Better spectral efficiency but sensitive to non-linear amplifiers
- ❌ Custom constellations: No performance benefit for HF
- ✅ **Selected**: Standard PSK/APSK (well-tested, good non-linear performance)

---

## 5. Polar Error Correction

### Decision: Use existing Polar codec library
**Rationale**: Polar codes are complex; use proven implementation rather than writing from scratch.

**Library Options**:
1. **sionna** (TensorFlow-based, GPU-accelerated)
   - ✅ Supports Polar codes (rate-matching, systematic encoding)
   - ✅ Battle-tested (used in 5G research)
   - ❌ Heavy dependency (TensorFlow)
   - ❌ Overkill for signal generation

2. **commpy** (Pure Python/NumPy)
   - ✅ Lightweight (NumPy only)
   - ✅ Simple API: `encode_polar(bits, N, K)`
   - ❌ Limited documentation
   - ✅ Sufficient for signal generation

3. **Custom implementation**:
   - ❌ Complex (channel construction, rate matching)
   - ❌ Error-prone
   - ❌ Reinventing the wheel

**Selected: commpy** for V1, migrate to sionna if needed
- Rationale: Lightweight, sufficient for generation task
- Installation: `pip install scikit-commpy`

**Polar Encoding Process**:
```python
from commpy.channelcoding import polar_encode

# Given: data bits and desired code rate
data_bits = np.array([1,0,1,1,0,...])  # k bits
rate = 2/3  # From kernel parameter

# Determine N (codeword length) and K (data bits)
# N must be power of 2, N ≥ pattern_length
N = next_power_of_2(pattern_length)
K = int(N * rate)

# Encode
codeword = polar_encode(data_bits[:K], N, K)
# Returns N bits (includes parity)
```

**Rate Options** (from CASCADE spec):
- 1/2: Maximum robustness (50% overhead)
- 2/3: Balanced (33% overhead)
- 3/4: Light protection (25% overhead)
- 4/5, 5/6, 7/8: High throughput (12-20% overhead)

**Validation**:
- Encode known message, verify codeword length
- Check systematic property (data bits appear in codeword)
- Unit test: Encode → Add noise → Decode (if decoder available)

---

## 6. Pattern Loading

### Decision: Pickle format from genetic algorithm output
**Rationale**: Patterns already generated and saved as .pkl files by pattern generator.

**File Locations**:
```
modules/training/patterns/tournament/
├── pattern_0_len_64.pkl
├── pattern_0_len_128.pkl
├── pattern_0_len_256.pkl
├── pattern_0_len_512.pkl
├── pattern_0_len_1024.pkl
├── pattern_0_len_2048.pkl
├── pattern_1_len_64.pkl
...
└── pattern_7_len_2048.pkl

Total: 8 patterns × 6 lengths = 48 files
```

**File Format** (expected):
```python
# Each .pkl file contains:
{
    'pattern_id': 3,
    'length': 512,
    'bits': np.array([0,1,1,0,1,0,...], dtype=np.uint8),  # 512 bits
    'orthogonality': -21.19,  # dB
    'generation_date': '2025-10-06',
    'generation_params': {...}
}
```

**Loading Strategy**:
```python
import pickle
from pathlib import Path

def load_pattern(pattern_id: int, length: int) -> np.ndarray:
    """Load pattern bits from .pkl file."""
    pattern_dir = Path('modules/training/patterns/tournament')
    pattern_file = pattern_dir / f'pattern_{pattern_id}_len_{length}.pkl'

    with open(pattern_file, 'rb') as f:
        pattern_data = pickle.load(f)

    return pattern_data['bits']
```

**Caching**:
- Load all 48 patterns at startup (total ~1 MB)
- Store in dictionary: `{(pattern_id, length): bits}`
- Avoid repeated file I/O during batch generation

**Validation**:
- Check file exists before loading
- Verify bit length matches expected length
- Confirm pattern_id matches filename

**Alternatives Considered**:
- ❌ Generate patterns on-the-fly: Too slow (genetic algorithm takes hours)
- ❌ Store as JSON: Less efficient for binary data
- ✅ **Selected**: Pickle (native Python, efficient for NumPy arrays)

---

## 7. IQ Sample Rate and Baseband Generation

### Decision: 48 kHz sample rate, complex baseband
**Rationale**: Standard sound card sample rate, simplifies hardware integration.

**Sample Rate Selection**:
- **48 kHz**: Standard sound card rate (Linux ALSA, PortAudio)
- **Nyquist**: Covers 0-24 kHz (exceeds 300-3000 Hz audio bandwidth)
- **Samples per symbol**: 48000 / 200 = 240 samples/symbol
- Alternatives: 12 kHz (less common), 96 kHz (unnecessary)

**Baseband vs Passband**:
- **Baseband**: Complex IQ samples (I+jQ), centered at 0 Hz
  - ✅ DSP-friendly (no carrier oscillator)
  - ✅ Smaller file size (real-valued after conversion)
  - ✅ Matches protocol spec
- **Passband**: Real-valued audio, centered at carrier frequency
  - Used only for final audio output (if WAV needed)

**IQ Format**:
```python
# Complex NumPy array
iq_samples = np.array([0.5+0.3j, 0.2-0.1j, ...], dtype=np.complex64)

# Properties:
# - iq_samples.real: In-phase component (I)
# - iq_samples.imag: Quadrature component (Q)
# - len(iq_samples): Number of IQ samples
```

**Frequency Translation** (baseband → audio):
```python
def baseband_to_audio(iq_baseband, center_freq, sample_rate):
    """Shift baseband IQ to audio frequency."""
    t = np.arange(len(iq_baseband)) / sample_rate
    carrier = np.exp(2j * np.pi * center_freq * t)
    iq_shifted = iq_baseband * carrier

    # Real part is the audio signal
    audio = iq_shifted.real
    return audio
```

**Storage**:
- **Core Generator output**: Complex IQ arrays (NumPy .npy or .npz)
- **Orchestrator output**: Complex IQ arrays with channel effects
- **Optional WAV export**: Convert to real audio via baseband_to_audio()

**Validation**:
- Verify sample rate: `len(iq_samples) / duration ≈ 48000`
- Check dynamic range: IQ samples should be in [-1, +1]
- Spectrum analysis: FFT should show signal at expected frequencies

---

## 8. Channel Simulator Components

### Decision: Expert-based training data generation
**Rationale**: CASCADE uses **5 specialized expert networks**, each requiring **separate training datasets**:

1. **QRN Expert**: Trained on pure atmospheric noise (NO signal)
2. **Signal Expert**: Trained on clean CASCADE signals (NO interference)
3. **Timing Expert**: Trained on collision scenarios (1-3 overlapping signals)
4. **Channel Expert**: Trained on known channel models (multipath, Doppler)
5. **QRM Expert**: Trained on pure interference patterns (NO CASCADE signal)

**CRITICAL REQUIREMENT**: The Synthetic Data Orchestrator must generate **5 separate datasets**, not just general noisy signals!

### Expert Training Data Requirements

**QRN Expert (1M examples, Weeks 7-8)**:
- **Input**: Pure QRN recordings (NO CASCADE signal)
- **Types**: Crackling (Poisson bursts), static (1/f noise), lightning (impulses), power line (harmonics)
- **Labels**: Burst times, noise type, intensity
- **Purpose**: Learn to classify and estimate atmospheric noise floor

**Signal Expert (1M examples, Weeks 9-10)**:
- **Input**: Clean CASCADE signals (NO noise or interference)
- **Labels**: Pattern ID, frequency pair, modulation, Polar codeword
- **Purpose**: Learn dual-layer pattern detection and data demodulation

**Timing Expert (1M examples, Weeks 11-13, most complex)**:
- **Input**: Mixed scenarios - 50% clean signals, 45% collisions (1-3 overlapping), 5% edge cases
- **Collision types**: 5-20ms offset (hard), 20-50ms offset (moderate), >50ms (easy)
- **Labels**: Number of signals, time offsets, signal boundaries, separated signals
- **Purpose**: Learn to detect and separate temporally overlapping transmissions

**Channel Expert (1M examples, Weeks 14-15)**:
- **Input**: Clean signals with KNOWN channel distortion (NO QRN/QRM)
- **Models**: Rayleigh fading, Rician fading, multipath (tapped delay), Doppler shifts
- **Labels**: Impulse response, channel parameters, delay spread, Doppler shift
- **Purpose**: Learn to estimate and compensate for ionospheric propagation

**QRM Expert (1M examples, Weeks 15-16)**:
- **Input**: Pure interference (NO CASCADE signal)
- **Types**: CW (single tone), SSB (voice), FT8, digital modes, radar, power line
- **Labels**: Interference type, frequency offset, modulation parameters
- **Purpose**: Learn to classify and mitigate man-made interference

### Separation Strategy (Critical!)

**DO NOT mix impairments during expert pre-training:**
- ❌ WRONG: QRN Expert sees signal + noise (would learn to decode, not just classify noise)
- ✅ CORRECT: QRN Expert sees ONLY noise (learns pure noise characteristics)

**Integration phase (Weeks 17-22):**
- After expert pre-training, integration decoder sees mixed scenarios
- Experts are frozen, integration layer learns to combine expert outputs
- This is when signals + noise + interference are combined

### Implementation Approach

### 8.1 AWGN (Additive White Gaussian Noise)

**Model**: Classic AWGN with specified SNR
```python
def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at specified SNR."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Complex noise (both I and Q)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) +
        1j * np.random.randn(len(signal))
    )

    return signal + noise
```

**Parameters**:
- SNR range: -35 dB to +20 dB
- Distribution: Gaussian (mean=0, variance=noise_power)

**Validation**:
- Measure output SNR: Should match input parameter within 0.5 dB
- Check noise statistics: Mean ≈ 0, variance matches calculation

### 8.2 QRN (Atmospheric Noise)

**Model**: Poisson-distributed impulse noise (crackling, static bursts)
```python
def add_qrn(signal: np.ndarray, burst_rate: float, intensity: float) -> np.ndarray:
    """Add atmospheric noise bursts."""
    # Poisson process for burst timing
    num_bursts = np.random.poisson(burst_rate * len(signal) / sample_rate)
    burst_times = np.random.randint(0, len(signal), size=num_bursts)

    # Exponential decay bursts
    burst_duration = int(0.01 * sample_rate)  # 10ms bursts
    noise = np.zeros_like(signal)

    for t0 in burst_times:
        t = np.arange(burst_duration)
        burst = intensity * np.exp(-t / (0.003 * sample_rate)) * (
            np.random.randn(burst_duration) +
            1j * np.random.randn(burst_duration)
        )
        noise[t0:t0+burst_duration] += burst

    return signal + noise
```

**Parameters**:
- `burst_rate`: Bursts per second (0.1 - 10)
- `intensity`: Peak amplitude relative to signal (0.1 - 5.0)
- `burst_duration`: 5-20ms typical

**Reference**: ITU-R P.372-14 (Radio Noise)

### 8.3 Multipath Fading

**Model**: Tapped delay line with Rayleigh-distributed taps
```python
def add_multipath(signal: np.ndarray, delay_spread_ms: float, num_taps: int = 3) -> np.ndarray:
    """Simulate frequency-selective fading."""
    sample_rate = 48000
    max_delay_samples = int(delay_spread_ms * sample_rate / 1000)

    # Random tap delays (uniform spacing)
    delays = np.linspace(0, max_delay_samples, num_taps).astype(int)

    # Rayleigh-distributed tap gains (normalized)
    gains = np.random.rayleigh(scale=1.0, size=num_taps)
    gains /= np.sqrt(np.sum(gains ** 2))  # Normalize power

    # Complex phase shifts (random)
    phases = np.exp(2j * np.pi * np.random.rand(num_taps))

    # Apply taps
    output = np.zeros_like(signal)
    for delay, gain, phase in zip(delays, gains, phases):
        tap_signal = np.roll(signal, delay) * gain * phase
        output += tap_signal

    return output
```

**Parameters**:
- `delay_spread_ms`: 1-5ms typical for HF (ionospheric reflection)
- `num_taps`: 2-5 (simple model)
- Doppler spread: Not modeled in V1 (static channels assumed)

**Reference**: Watterson model (ITU-R F.1487)

### 8.4 QRM (Interference from Other Stations)

**Model**: Synthetic interfering signals at nearby frequencies
```python
def add_qrm(signal: np.ndarray, interferer_count: int, strength_db: float) -> np.ndarray:
    """Add interference from other CASCADE stations."""
    qrm = np.zeros_like(signal)

    for _ in range(interferer_count):
        # Random frequency offset (-50 to +50 Hz from signal)
        freq_offset = np.random.uniform(-50, 50)

        # Generate interfering signal (random pattern)
        t = np.arange(len(signal)) / sample_rate
        interferer = np.exp(2j * np.pi * freq_offset * t)

        # Random modulation (simulates other users)
        modulation = np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        interferer *= modulation

        # Scale to specified strength relative to signal
        interferer *= 10 ** (strength_db / 20)

        qrm += interferer

    return signal + qrm
```

**Parameters**:
- `interferer_count`: 0-10 (typical HF congestion)
- `strength_db`: -20 dB to +10 dB relative to signal
- `freq_offset`: ±50 Hz (adjacent channels)

**Validation**:
- Measure SIR (Signal-to-Interference Ratio)
- Spectrum analysis: Should see peaks at offset frequencies

---

## 9. Metadata and Ground Truth

### Decision: JSON sidecar files for signal metadata
**Rationale**: Enables decoder validation and training dataset management.

**Metadata Structure**:
```json
{
  "signal_id": "uuid-v4",
  "generation_timestamp": "2025-10-07T12:34:56Z",
  "generator_version": "1.0.0",

  "kernel_params": {
    "pattern_id": 3,
    "frequency_pair": 25,
    "modulation": "QPSK",
    "polar_rate": "2/3"
  },

  "signal_params": {
    "pattern_length": 512,
    "sample_rate": 48000,
    "duration_seconds": 2.56,
    "tone_a_hz": 1300,
    "tone_b_hz": 1320
  },

  "message": {
    "plaintext": "TEST MESSAGE",
    "data_bits": 341,
    "polar_encoded_bits": 512,
    "checksum": "sha256:abc123..."
  },

  "channel_conditions": {
    "snr_db": -15,
    "awgn_enabled": true,
    "qrn_burst_rate": 2.5,
    "qrn_intensity": 0.8,
    "multipath_delay_spread_ms": 3.0,
    "multipath_taps": 3,
    "qrm_interferer_count": 2,
    "qrm_strength_db": -10
  },

  "ground_truth": {
    "expected_pattern_id": 3,
    "expected_frequency_pair": 25,
    "expected_modulation": "QPSK",
    "expected_message": "TEST MESSAGE",
    "polar_codeword": [1,0,1,1,0,...]
  },

  "files": {
    "clean_iq": "signal_uuid_clean.npy",
    "noisy_iq": "signal_uuid_noisy.npy"
  }
}
```

**Storage Layout**:
```
output/
├── signal_abc123_clean.npy       # Clean IQ samples (complex64)
├── signal_abc123_noisy.npy       # IQ with channel effects
└── signal_abc123_metadata.json   # Metadata + ground truth
```

**Validation Use Case**:
```python
# Load test signal
metadata = json.load(open('signal_abc123_metadata.json'))
iq_signal = np.load('signal_abc123_noisy.npy')

# Feed to decoder
decoded = cascade_decoder.decode(iq_signal)

# Validate
assert decoded.pattern_id == metadata['ground_truth']['expected_pattern_id']
assert decoded.message == metadata['ground_truth']['expected_message']
```

---

## 10. Performance Requirements

### Decision: Optimize for batch generation throughput
**Rationale**: Signal generator will be used for large-scale dataset creation.

**Target Performance**:
- **Single signal**: <100ms for 512-symbol signal (clean generation)
- **Batch processing**: 100 signals in <30s (with channel simulation)
- **Throughput**: ~3-4 signals/second sustained

**Bottleneck Analysis**:
1. **GMSK pulse shaping**: ~20ms (convolution on 240 samples/symbol × 512 symbols)
2. **Polar encoding**: ~10ms (commpy implementation)
3. **Channel simulation**: ~30ms (multipath most expensive)
4. **File I/O**: ~5ms (NumPy save)
**Total**: ~65ms per signal (margin for 100ms target)

**Optimization Strategies**:
- **Pre-compute patterns**: Load all 48 patterns at startup (1ms → 0ms per signal)
- **Vectorize**: Use NumPy operations instead of loops
- **Parallel batch**: Process multiple signals with `multiprocessing.Pool`
- **Lazy channel sim**: Skip if clean signals only

**Memory Footprint**:
- Pattern cache: 48 × 2048 bits = 12 KB
- Single IQ signal: 512 symbols × 240 samples/symbol × 8 bytes (complex64) = 983 KB
- Batch of 100: ~100 MB (fits in RAM)

**Validation**:
- Benchmark: `time python -m signal_generator generate --count 100`
- Profile: `python -m cProfile` to identify bottlenecks
- Target: 100 signals in <30s on Raspberry Pi 4 (ARM Cortex-A72)

---

## 11. CLI and Library API Design

### Decision: Separate CLI (argparse) and library API (Python classes)
**Rationale**: Enables both command-line usage and programmatic integration.

### CLI Interface (User-Facing)

**Core Generator**:
```bash
# Generate clean signal
cascade-signal generate \
  --pattern-id 3 \
  --freq-pair 25 \
  --modulation QPSK \
  --polar-rate 2/3 \
  --message "TEST MESSAGE" \
  --output signal_clean.npy

# Batch generation
cascade-signal generate-batch \
  --config batch_config.yaml \
  --output-dir ./signals/
```

**Orchestrator**:
```bash
# Add channel effects
cascade-orchestrator simulate \
  --input signal_clean.npy \
  --snr -15 \
  --qrn-rate 2.5 \
  --multipath-delay 3.0 \
  --output signal_noisy.npy

# SNR sweep
cascade-orchestrator sweep \
  --input signal_clean.npy \
  --snr-start -30 \
  --snr-stop 10 \
  --snr-step 3 \
  --output-dir ./sweep/
```

### Library API (Programmatic)

**Core Generator**:
```python
from cascade.signal_generator import SignalGenerator

gen = SignalGenerator()

# Generate signal
iq_signal, metadata = gen.generate(
    pattern_id=3,
    frequency_pair=25,
    modulation='QPSK',
    polar_rate=(2, 3),
    message="TEST MESSAGE"
)

# Access components
assert iq_signal.shape == (122880,)  # 512 symbols × 240 samples
assert metadata['pattern_length'] == 512
```

**Orchestrator**:
```python
from cascade.channel_simulator import ChannelOrchestrator

orch = ChannelOrchestrator()

# Apply channel effects
noisy_signal = orch.add_channel_effects(
    clean_iq=iq_signal,
    snr_db=-15,
    qrn_burst_rate=2.5,
    qrn_intensity=0.8,
    multipath_delay_spread_ms=3.0,
    qrm_interferer_count=2,
    qrm_strength_db=-10
)

# Batch processing
batch = orch.generate_batch(
    clean_iq=iq_signal,
    snr_range=(-30, 10, 3),  # start, stop, step
    num_samples=10
)
```

**Design Principles**:
- CLI wraps library API (no duplicate logic)
- Library API returns NumPy arrays (no file I/O in core functions)
- Metadata always included in library returns
- CLI handles file I/O and formatting

---

## 12. Testing Strategy

### Decision: Multi-level testing (unit, contract, integration, property-based)
**Rationale**: Signal generation is correctness-critical; comprehensive testing required.

### Test Levels

**Unit Tests** (individual components):
```python
def test_gmsk_pulse_shaping():
    """Verify GMSK produces constant envelope."""
    pattern = np.array([0, 1, 1, 0, 1])
    iq = gmsk.generate_gmsk_fsk(pattern, freq_pair=25)

    # Constant envelope check
    envelope = np.abs(iq)
    assert np.allclose(envelope, 1.0, atol=0.01)

def test_constellation_mapping():
    """Verify QPSK constellation points."""
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    symbols = modulation.map_qpsk(bits)

    expected = np.array([
        (1+1j)/np.sqrt(2),
        (1-1j)/np.sqrt(2),
        (-1+1j)/np.sqrt(2),
        (-1-1j)/np.sqrt(2)
    ])
    assert np.allclose(symbols, expected)
```

**Contract Tests** (V2 spec compliance):
```python
def test_v2_frequency_grid():
    """Verify 135-channel grid matches spec."""
    for channel_id in range(135):
        freq = get_channel_frequency(channel_id)
        expected = 300 + channel_id * 20
        assert freq == expected

def test_v2_symbol_rate():
    """Verify 200 symbols/second rate."""
    iq = generate_signal(pattern_id=0, length=512)
    duration = len(iq) / 48000  # sample rate
    expected_duration = 512 / 200  # 2.56s
    assert abs(duration - expected_duration) < 0.01
```

**Integration Tests** (end-to-end):
```python
def test_full_generation_pipeline():
    """Generate signal and verify all components."""
    gen = SignalGenerator()
    iq, metadata = gen.generate(
        pattern_id=3,
        frequency_pair=25,
        modulation='QPSK',
        polar_rate=(2, 3),
        message="TEST"
    )

    # Verify output structure
    assert iq.dtype == np.complex64
    assert len(iq) > 0
    assert metadata['kernel_params']['pattern_id'] == 3

    # Apply channel effects
    orch = ChannelOrchestrator()
    noisy = orch.add_awgn(iq, snr_db=-15)

    # Measure SNR
    measured_snr = measure_snr(iq, noisy)
    assert abs(measured_snr - (-15)) < 1.0  # Within 1 dB
```

**Property-Based Tests** (hypothesis):
```python
from hypothesis import given, strategies as st

@given(
    pattern_id=st.integers(min_value=0, max_value=7),
    freq_pair=st.integers(min_value=0, max_value=66),
    snr_db=st.floats(min_value=-35, max_value=20)
)
def test_signal_generation_never_crashes(pattern_id, freq_pair, snr_db):
    """Verify generator handles all valid inputs."""
    gen = SignalGenerator()
    iq, metadata = gen.generate(
        pattern_id=pattern_id,
        frequency_pair=freq_pair,
        modulation='BPSK',
        polar_rate=(1, 2),
        message="X"
    )

    orch = ChannelOrchestrator()
    noisy = orch.add_awgn(iq, snr_db=snr_db)

    # Should never raise exception
    assert len(noisy) == len(iq)
```

### Validation Tests (Signal Quality)

```python
def test_gmsk_bandwidth():
    """Verify GMSK occupies <30 Hz at -40 dB."""
    iq = generate_gmsk_tone(frequency=1500, duration=1.0)
    spectrum = np.fft.fft(iq)
    power_db = 10 * np.log10(np.abs(spectrum) ** 2)

    # Find -40 dB bandwidth
    threshold = np.max(power_db) - 40
    occupied_bins = np.sum(power_db > threshold)
    occupied_bandwidth = occupied_bins * (48000 / len(iq))

    assert occupied_bandwidth < 30

def test_pattern_orthogonality():
    """Verify patterns maintain -21 dB orthogonality."""
    pattern_0 = load_pattern(0, 2048)
    pattern_1 = load_pattern(1, 2048)

    # Generate signals
    iq_0 = generate_gmsk_fsk(pattern_0, freq_pair=25)
    iq_1 = generate_gmsk_fsk(pattern_1, freq_pair=25)

    # Cross-correlation
    xcorr = np.correlate(iq_0, iq_1, mode='full')
    peak = np.max(np.abs(xcorr))
    autocorr_0 = np.max(np.abs(np.correlate(iq_0, iq_0, mode='full')))

    orthogonality_db = 10 * np.log10(peak / autocorr_0)
    assert orthogonality_db < -20  # Better than -20 dB
```

---

## 13. Dependencies and Installation

### Decision: Minimal dependencies (NumPy, SciPy, scikit-commpy)
**Rationale**: Reduce installation complexity and ensure compatibility.

**Required Dependencies**:
```toml
[project]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-commpy>=0.7.0",  # Polar codes
]
```

**Optional Dependencies**:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "hypothesis>=6.0.0",      # Property-based testing
    "matplotlib>=3.5.0",       # Visualization
]

audio = [
    "soundfile>=0.12.0",       # WAV I/O (if needed)
]
```

**Installation**:
```bash
# Core functionality
pip install cascade-signal-generator

# Development
pip install cascade-signal-generator[dev]

# Audio export (optional)
pip install cascade-signal-generator[audio]
```

**Platform Support**:
- ✅ Linux (primary development platform)
- ✅ macOS (tested on ARM and x86)
- ✅ Windows (via WSL or native Python)
- ✅ Raspberry Pi 4 (ARM, Raspbian)

**Python Version**:
- Minimum: Python 3.11 (matches CASCADE project standard)
- Tested: 3.11, 3.12

---

## 14. Embedding Autoencoder Architecture (Updated from CLAUDE.md)

### Decision: Encoder takes kernel parameters as input (not just channel observations)
**Rationale**: Updated CASCADE architecture shows embedding encoder needs kernel context.

**Architecture Update**:
```python
# OLD (incorrect):
embedding = encoder(channel_observations)

# NEW (correct per CLAUDE.md):
embedding = encoder(channel_observations, kernel_parameters)
```

**Encoder Inputs** (line 682, 742 of CLAUDE.md):
1. **Channel features** (128 floats): Output from Channel Expert
2. **Kernel parameters** (discrete values): pattern_id, frequency_pair, modulation, polar_rate, SNR estimate
   - Embedded as learned vectors (pattern embedding, frequency embedding, etc.)
   - Concatenated with channel features
   - Total input: ~128 channel features + 32 kernel embeddings = 160 floats

**Why kernel parameters are needed**:
- Different patterns may require different embedding strategies
- Frequency pair affects what channel distortions are important
- Modulation (BPSK vs 16-APSK) changes SNR sensitivity
- Encoder can learn pattern-specific compression
- Improves quantization efficiency (context-aware)

**Training Pipeline** (line 505 of CLAUDE.md):
```
Channel observations (I/Q samples) + Kernel parameters
    ↓
Encoder Network
    ↓
Continuous embedding (192-256 floats)
    ↓
Learned Quantizer (differentiable, 112 bits)
    ↓
Quantized representation (14 bytes)
    ↓
Decoder Network
    ↓
Reconstructed embedding (192-256 floats)
    ↓
Apply to demodulation task
```

**Implications for Signal Generator**:
- **Core Generator**: Must include kernel parameters in metadata (already done)
- **Orchestrator**: When generating Channel Expert training data, must include kernel parameters in labels
- **Integration scenarios**: Embedding encoder training examples need both channel observations AND kernel parameters

**Updated Channel Expert labels**:
```python
{
    'channel_type': str,
    'impulse_response': np.ndarray,
    'tap_delays': np.ndarray,
    'tap_gains': np.ndarray,
    'delay_spread_ms': float,
    'doppler_shift_hz': float,
    'k_factor_db': float,
    # NEW: Kernel parameters for embedding encoder
    'kernel_parameters': {
        'pattern_id': int,
        'frequency_pair': int,
        'modulation': str,
        'polar_rate': tuple,
        'snr_estimate': float
    }
}
```

**Integration Decoder Outputs** (lines 605, 674-678 of CLAUDE.md):
- **NEW**: Decoder outputs **soft decisions** (log-likelihood ratios), not hard bits
- Soft decisions feed into Polar decoder
- More realistic than hard bits (NN provides confidence scores)
- Enables iterative decoding and better error correction

**TX vs RX Kernel Clarification**:
- **TX Kernel** (28 bytes): Transmitted kernel in beacons/messages (lines 623, 708)
- **RX Kernel**: Receiver-generated parameters for receiving (not transmitted)
- Signal generator produces signals that would correspond to TX kernels

---

## 15. Open Questions and Future Work

### Resolved in This Research
- ✅ Dual-layer modulation architecture
- ✅ GMSK pulse shaping parameters
- ✅ Frequency grid and channel mapping
- ✅ Polar encoding approach
- ✅ Channel simulation models
- ✅ Performance targets
- ✅ API design (CLI + library)

### Deferred to Implementation
- **Polar codec integration**: Verify commpy API matches assumptions
- **Pattern file format**: Confirm actual .pkl structure from genetic algorithm output
- **16-APSK parameters**: Determine optimal ring radii for HF
- **Multipath Doppler**: Add frequency shifts if needed for realism
- **GPU acceleration**: Evaluate if batch generation needs CUDA/OpenCL

### Future Enhancements (Post-V1)
- **Diversity mode support**: 2×/4×/8× frequency diversity (deferred per spec clarification)
- **Real-time generation**: Streaming IQ output for hardware SDR
- **Hardware-in-the-loop**: Interface with LimeSDR/PlutoSDR
- **Adaptive orchestration**: Automatically select channel conditions based on real KiwiSDR statistics
- **WAV export**: Sound card audio output for radio interface
- **Decoder integration**: Close-loop validation (generate → decode → compare)

---

## Summary of Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Modulation Architecture | Dual-layer (GMSK 2-FSK + IQ data) | Matches CASCADE V2 spec, achieves 2-4 bits/symbol |
| GMSK Pulse Shaping | BT=0.3, SciPy Gaussian filter | Balances bandwidth and envelope, standard HF practice |
| Frequency Plan | 135 channels, 20 Hz spacing | Updated CASCADE V2 protocol (was 67 pairs) |
| Constellation Mapping | Standard PSK/APSK | Well-tested, good non-linear performance |
| Polar Encoding | scikit-commpy library | Lightweight, sufficient for generation |
| Pattern Loading | Pickle format, cached at startup | Efficient, matches genetic algorithm output |
| Sample Rate | 48 kHz, complex baseband | Standard sound card rate, DSP-friendly |
| Channel Simulation | Separate models (AWGN, QRN, multipath, QRM) | Realistic HF impairments, additive combination |
| Metadata | JSON sidecar files | Enables decoder validation, human-readable |
| Performance | <100ms single, 100 signals in <30s | Optimized for batch generation |
| API Design | CLI (argparse) + Library (classes) | Supports both automation and scripting |
| Testing | Multi-level (unit, contract, integration, property) | Comprehensive validation of correctness |
| Dependencies | NumPy, SciPy, scikit-commpy | Minimal, cross-platform |

---

## Next Steps (Phase 1)

With research complete, proceed to Phase 1:
1. **Create data-model.md**: Define entities (KernelParameters, CleanIQSignal, ChannelConditions, etc.)
2. **Generate API contracts**: Define function signatures for Core Generator and Orchestrator
3. **Write contract tests**: TDD tests that will fail until implementation
4. **Create quickstart.md**: Step-by-step validation scenario
5. **Update CLAUDE.md**: Add signal generator context incrementally

---

**Research Phase Status**: ✅ COMPLETE
**Ready for Phase 1**: YES
**Blocking Issues**: NONE
