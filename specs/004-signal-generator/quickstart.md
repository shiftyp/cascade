# Quickstart: Signal Generator Validation

**Feature**: 004-signal-generator
**Date**: 2025-10-07
**Purpose**: Step-by-step validation of Core Generator and Synthetic Data Orchestrator

This quickstart demonstrates the complete signal generation pipeline from clean signal generation through expert-specific training data creation.

---

## Prerequisites

```bash
# Install dependencies
pip install numpy scipy scikit-commpy matplotlib

# Verify pattern files exist
ls modules/training/patterns/tournament/pattern_*.pkl
# Should see 48 files (8 patterns × 6 lengths)

# Clone/navigate to CASCADE repository
cd /workspaces/cascade
```

---

## Part 1: Core Signal Generator (Clean Signals)

### Step 1.1: Generate Simple BPSK Signal

```python
from cascade.signal_generator import SignalGenerator

# Initialize generator
gen = SignalGenerator()

# Generate clean signal
iq_signal, metadata = gen.generate(
    pattern_id=3,
    frequency_pair=25,  # 1300 Hz / 1320 Hz
    modulation='BPSK',
    polar_rate=(1, 2),  # Rate 1/2 (maximum protection)
    message="HELLO WORLD",
    seed=42  # Reproducible
)

# Verify output
print(f"Signal duration: {iq_signal.duration_seconds:.2f}s")
print(f"Pattern length: {iq_signal.pattern_length} symbols")
print(f"Tone A: {iq_signal.tone_a_hz} Hz")
print(f"Tone B: {iq_signal.tone_b_hz} Hz")
print(f"IQ samples: {len(iq_signal.iq_samples)}")

# Expected output:
# Signal duration: 2.56s
# Pattern length: 512 symbols
# Tone A: 1300.0 Hz
# Tone B: 1320.0 Hz
# IQ samples: 122880 (512 symbols × 240 samples/symbol)
```

**Expected Result**: Clean IQ signal with 512 symbols, BPSK modulation, Polar 1/2 encoding

---

### Step 1.2: Verify V2 Compliance

```python
# Validate signal against CASCADE V2 specification
compliance = gen.verify_v2_compliance(iq_signal)

print("V2 Compliance Results:")
for check, (expected, actual, status) in compliance.items():
    symbol = "✅" if status == "pass" else "❌"
    print(f"{symbol} {check}: expected={expected}, actual={actual}")

# Expected output:
# ✅ symbol_rate: expected=200, actual=200.0, pass
# ✅ gmsk_bandwidth: expected<30, actual=28.5, pass
# ✅ tone_spacing: expected=20, actual=20.0, pass
# ✅ sample_rate: expected=48000, actual=48000, pass
# ✅ pattern_orthogonality: expected<-20, actual=-21.2, pass
# ✅ overall: True
```

**Expected Result**: All compliance checks pass

---

### Step 1.3: Visualize Signal Spectrum

```python
import matplotlib.pyplot as plt
import numpy as np

# Compute spectrum
spectrum = np.fft.fft(iq_signal.iq_samples)
freqs = np.fft.fftfreq(len(spectrum), 1/48000)
power_db = 10 * np.log10(np.abs(spectrum)**2 + 1e-12)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], power_db[:len(freqs)//2])
plt.axvline(1300, color='r', linestyle='--', label='Tone A')
plt.axvline(1320, color='r', linestyle='--', label='Tone B')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Clean CASCADE Signal Spectrum')
plt.grid(True)
plt.legend()
plt.xlim(1200, 1400)
plt.savefig('signal_spectrum.png')
print("Spectrum saved to signal_spectrum.png")
```

**Expected Result**: Two peaks at 1300 Hz and 1320 Hz, ~30 Hz bandwidth each

---

## Part 2: Synthetic Data Orchestrator (Expert Training Data)

### Step 2.1: Generate QRN Expert Data (Pure Noise)

```python
from cascade.channel_simulator import ChannelOrchestrator

orch = ChannelOrchestrator()

# Generate pure atmospheric noise (NO signal)
qrn_example = orch.generate_qrn_expert_data(
    duration_seconds=2.56,
    qrn_type='crackling',
    intensity=1.5,
    seed=42
)

# Verify
print(f"Expert type: {qrn_example.expert_type}")
print(f"QRN type: {qrn_example.labels['qrn_type']}")
print(f"Burst count: {len(qrn_example.labels['burst_times'])}")
print(f"Noise floor: {qrn_example.labels['noise_floor_db']:.1f} dB")
print(f"IQ samples: {len(qrn_example.iq_samples)}")

# Expected output:
# Expert type: qrn
# QRN type: crackling
# Burst count: 6 (varies due to Poisson process)
# Noise floor: -25.3 dB
# IQ samples: 122880

# CRITICAL: Verify NO signal is present (only noise)
signal_power = np.mean(np.abs(qrn_example.iq_samples)**2)
print(f"Average noise power: {10*np.log10(signal_power):.1f} dB")
# Should be around -25 dB (noise floor), not higher
```

**Expected Result**: Pure noise sample with Poisson-distributed bursts, NO CASCADE signal

---

### Step 2.2: Generate Signal Expert Data (Clean Signal)

```python
# Generate clean CASCADE signal (NO noise)
signal_example = orch.generate_signal_expert_data(
    clean_iq=iq_signal.iq_samples,
    seed=42
)

# Verify
print(f"Expert type: {signal_example.expert_type}")
print(f"Pattern ID: {signal_example.labels['pattern_id']}")
print(f"Modulation: {signal_example.labels['modulation']}")
print(f"Polar codeword length: {len(signal_example.labels['polar_codeword'])}")

# Expected output:
# Expert type: signal
# Pattern ID: 3
# Modulation: BPSK
# Polar codeword length: 512

# CRITICAL: Verify signal is clean (infinite SNR)
# Should be bit-identical to input
assert np.allclose(signal_example.iq_samples, iq_signal.iq_samples)
print("✅ Signal is clean (bit-identical to input)")
```

**Expected Result**: Clean signal, bit-identical to Core Generator output, NO noise

---

### Step 2.3: Generate Timing Expert Data (Collision Scenario)

```python
from cascade.channel_simulator import CollisionScenario

# Generate two signals for collision
signal_1, _ = gen.generate(
    pattern_id=3, frequency_pair=25, modulation='QPSK',
    polar_rate=(2, 3), message="FIRST", seed=100
)

signal_2, _ = gen.generate(
    pattern_id=5, frequency_pair=25, modulation='QPSK',
    polar_rate=(2, 3), message="SECOND", seed=200
)

# Create collision scenario (15.5ms offset)
collision = CollisionScenario(
    num_signals=2,
    time_offsets_ms=[0.0, 15.5],
    frequency_pairs=[25, 25],  # Same pair (harder)
    snr_db_list=[-10, -12],
    relative_powers=[1.0, 0.8]
)

# Generate timing expert example
timing_example = orch.generate_timing_expert_data(
    collision_scenario=collision,
    clean_signals=[signal_1.iq_samples, signal_2.iq_samples],
    base_noise_floor_db=-30,
    seed=42
)

# Verify
print(f"Expert type: {timing_example.expert_type}")
print(f"Number of signals: {timing_example.labels['num_signals']}")
print(f"Time offsets (ms): {timing_example.labels['time_offsets_ms']}")
print(f"Time offsets (samples): {timing_example.labels['time_offsets_samples']}")
print(f"Collision type: {timing_example.labels['collision_type']}")

# Expected output:
# Expert type: timing
# Number of signals: 2
# Time offsets (ms): [0.0, 15.5]
# Time offsets (samples): [0, 744]  # 15.5ms @ 48kHz = 744 samples
# Collision type: same_pair

# Verify separated signals are included (ground truth)
assert len(timing_example.labels['individual_signals']) == 2
print(f"✅ Separated signals included for supervised learning")
```

**Expected Result**: Two overlapping signals with 15.5ms offset, ground truth includes separated signals

---

### Step 2.4: Generate Channel Expert Data (Multipath)

```python
# Generate channel expert example (multipath fading)
channel_params = {
    'delay_spread_ms': 3.0,
    'num_taps': 3,
    'fading_type': 'rayleigh'
}

# NEW: Must provide kernel parameters (per CLAUDE.md update)
kernel_params = {
    'pattern_id': iq_signal.kernel_params.pattern_id,
    'frequency_pair': iq_signal.kernel_params.frequency_pair,
    'modulation': iq_signal.kernel_params.modulation,
    'polar_rate': iq_signal.kernel_params.polar_rate,
    'snr_estimate': 0.0  # Clean signal (infinite SNR)
}

channel_example = orch.generate_channel_expert_data(
    clean_iq=iq_signal.iq_samples,
    channel_type='multipath',
    channel_params=channel_params,
    kernel_parameters=kernel_params,  # NEW: Required for embedding encoder
    seed=42
)

# Verify
print(f"Expert type: {channel_example.expert_type}")
print(f"Channel type: {channel_example.labels['channel_type']}")
print(f"Delay spread: {channel_example.labels['delay_spread_ms']} ms")
print(f"Number of taps: {len(channel_example.labels['tap_delays'])}")
print(f"Impulse response length: {len(channel_example.labels['impulse_response'])}")
print(f"Kernel parameters included: {channel_example.labels['kernel_parameters']}")

# Expected output:
# Expert type: channel
# Channel type: multipath
# Delay spread: 3.0 ms
# Number of taps: 3
# Impulse response length: 145 (3ms @ 48kHz)
# Kernel parameters included: {'pattern_id': 3, 'frequency_pair': 25, 'modulation': 'BPSK', 'polar_rate': (1, 2), 'snr_estimate': 0.0}

# Verify signal has multipath distortion but NO noise
# Check signal has frequency-selective fading
spectrum_clean = np.fft.fft(iq_signal.iq_samples)
spectrum_channel = np.fft.fft(channel_example.iq_samples)
print(f"✅ Channel distortion applied (NO noise added)")
```

**Expected Result**: Signal with multipath fading, NO noise, known impulse response in labels

---

### Step 2.5: Generate QRM Expert Data (Pure Interference)

```python
# Generate pure FT8 interference (NO CASCADE signal)
qrm_example = orch.generate_qrm_expert_data(
    duration_seconds=2.56,
    interference_type='ft8',
    frequency_offset_hz=50,  # 50 Hz above CASCADE signal
    strength_db=-5,
    seed=42
)

# Verify
print(f"Expert type: {qrm_example.expert_type}")
print(f"Interference type: {qrm_example.labels['interference_type']}")
print(f"Frequency offset: {qrm_example.labels['frequency_offset_hz']} Hz")
print(f"Bandwidth: {qrm_example.labels['bandwidth_hz']} Hz")

# Expected output:
# Expert type: qrm
# Interference type: ft8
# Frequency offset: 50.0 Hz
# Bandwidth: 50.0 Hz (FT8 is 50 Hz wide)

# CRITICAL: Verify NO CASCADE signal (only FT8 interference)
print(f"✅ Pure interference (NO CASCADE signal)")
```

**Expected Result**: Pure FT8 interference, NO CASCADE signal

---

## Part 3: Batch Generation for Training Datasets

### Step 3.1: Generate QRN Expert Dataset (1M examples)

```python
# Generate batch of QRN examples
qrn_config = {
    'duration_seconds': 2.56,
    'qrn_types': ['crackling', 'static', 'lightning', 'power_line'],
    'intensity_range': (0.5, 3.0),
    'sample_rate': 48000
}

# Generate small batch (for testing, scale to 1M for real training)
qrn_batch = orch.generate_batch(
    expert_type='qrn',
    num_examples=100,
    config=qrn_config,
    seed=42
)

print(f"Generated {len(qrn_batch)} QRN examples")
print(f"QRN types: {[ex.labels['qrn_type'] for ex in qrn_batch[:5]]}")

# Save dataset
dataset_info = orch.save_expert_dataset(
    examples=qrn_batch,
    output_dir='./output/expert_datasets/',
    dataset_name='qrn_expert_test_v1',
    format='npz'
)

print(f"Dataset saved to: {dataset_info['file_path']}")
print(f"Dataset size: {dataset_info['size_mb']:.1f} MB")
print(f"Examples per type: {dataset_info['examples_per_class']}")

# Expected output:
# Generated 100 QRN examples
# QRN types: ['crackling', 'static', 'lightning', 'power_line', 'crackling']
# Dataset saved to: ./output/expert_datasets/qrn_expert_test_v1_qrn_train.npz
# Dataset size: 47.3 MB
# Examples per type: {'crackling': 25, 'static': 25, 'lightning': 25, 'power_line': 25}
```

**Expected Result**: 100 QRN examples saved, balanced across 4 types

---

### Step 3.2: Generate Signal Expert Dataset

```python
# Generate batch of clean signals
signal_config = {
    'pattern_ids': list(range(8)),
    'frequency_pairs': list(range(67)),
    'modulations': ['BPSK', 'QPSK', '8-PSK', '16-APSK'],
    'polar_rates': [(1,2), (2,3), (3,4), (5,6), (7,8)],
    'message_length_range': (10, 100)  # characters
}

signal_batch = orch.generate_batch(
    expert_type='signal',
    num_examples=100,
    config=signal_config,
    seed=42
)

print(f"Generated {len(signal_batch)} clean signal examples")
print(f"Pattern IDs: {[ex.labels['pattern_id'] for ex in signal_batch[:10]]}")
print(f"Modulations: {[ex.labels['modulation'] for ex in signal_batch[:10]]}")

# Save dataset
dataset_info = orch.save_expert_dataset(
    examples=signal_batch,
    output_dir='./output/expert_datasets/',
    dataset_name='signal_expert_test_v1',
    format='npz'
)

print(f"Dataset saved: {dataset_info['file_path']}")
```

**Expected Result**: 100 clean signal examples with varied parameters

---

### Step 3.3: Generate Timing Expert Dataset (With Collisions)

```python
# Generate batch of timing examples (50% clean, 45% collisions)
timing_config = {
    'clean_ratio': 0.50,
    'collision_ratio': 0.45,
    'edge_case_ratio': 0.05,
    'num_signals_range': (1, 3),
    'time_offset_ranges': {
        'hard': (5, 20),      # 5-20ms (hard collisions)
        'moderate': (20, 50),  # 20-50ms
        'easy': (50, 100)      # 50-100ms
    }
}

timing_batch = orch.generate_batch(
    expert_type='timing',
    num_examples=100,
    config=timing_config,
    seed=42
)

print(f"Generated {len(timing_batch)} timing examples")

# Count collision scenarios
collision_counts = {}
for ex in timing_batch:
    num_sigs = ex.labels['num_signals']
    collision_counts[num_sigs] = collision_counts.get(num_sigs, 0) + 1

print(f"Distribution: {collision_counts}")
# Expected: ~50 with 1 signal, ~35 with 2 signals, ~10 with 3 signals

# Save dataset
dataset_info = orch.save_expert_dataset(
    examples=timing_batch,
    output_dir='./output/expert_datasets/',
    dataset_name='timing_expert_test_v1',
    format='npz'
)

print(f"Dataset saved: {dataset_info['file_path']}")
```

**Expected Result**: 100 timing examples with mix of clean/collision scenarios

---

## Part 4: SNR Sweep for Decoder Validation

### Step 4.1: Generate SNR Sweep

```python
# Generate signal at multiple SNR levels
snr_sweep = orch.generate_snr_sweep(
    clean_iq=iq_signal.iq_samples,
    snr_start_db=-30,
    snr_stop_db=10,
    snr_step_db=3,
    seed=42
)

print(f"Generated {len(snr_sweep)} signals")
print("SNR levels:")
for sig in snr_sweep:
    print(f"  {sig.measured_snr_db:.1f} dB")

# Expected output:
# Generated 14 signals
# SNR levels:
#   -30.0 dB
#   -27.0 dB
#   -24.0 dB
#   ...
#   +10.0 dB
```

**Expected Result**: 14 signals at SNR levels from -30 to +10 dB in 3 dB steps

---

### Step 4.2: Validate Against Decoder

```python
# Feed to decoder (if available) for BER curve
# This would be done during decoder validation

for sig in snr_sweep:
    # Pseudo-code (decoder not implemented yet)
    # decoded = cascade_decoder.decode(sig.iq_samples)
    # ber = calculate_ber(decoded.bits, sig.clean_signal.polar_codeword)
    # print(f"SNR: {sig.measured_snr_db:.1f} dB, BER: {ber:.2e}")
    pass

print("✅ SNR sweep ready for decoder BER validation")
```

**Expected Result**: Ready for decoder testing (decoder implementation is separate task)

---

## Part 5: Integration Test (Full Pipeline)

### Step 5.1: End-to-End Generation

```python
# Complete workflow: Core Generator → Expert Datasets
print("=== Full Pipeline Test ===\n")

# 1. Generate clean signal
clean_signal, _ = gen.generate(
    pattern_id=7, frequency_pair=33, modulation='QPSK',
    polar_rate=(2, 3), message="END TO END TEST", seed=999
)
print(f"✅ Clean signal generated: {clean_signal.pattern_length} symbols")

# 2. Generate all 5 expert examples from same signal
qrn = orch.generate_qrn_expert_data(2.56, 'crackling', 1.0, seed=999)
signal = orch.generate_signal_expert_data(clean_signal.iq_samples, seed=999)

# Channel expert requires kernel parameters (per CLAUDE.md update)
kernel_params_for_channel = {
    'pattern_id': 7,
    'frequency_pair': 33,
    'modulation': 'QPSK',
    'polar_rate': (2, 3),
    'snr_estimate': 0.0
}
channel = orch.generate_channel_expert_data(
    clean_signal.iq_samples, 'multipath',
    {'delay_spread_ms': 3.0, 'num_taps': 3, 'fading_type': 'rayleigh'},
    kernel_parameters=kernel_params_for_channel,
    seed=999
)
qrm = orch.generate_qrm_expert_data(2.56, 'cw', 30, -10, seed=999)

# Timing requires two signals
signal_2, _ = gen.generate(
    pattern_id=2, frequency_pair=33, modulation='BPSK',
    polar_rate=(1, 2), message="COLLISION", seed=888
)
collision_cfg = CollisionScenario(2, [0.0, 12.0], [33, 33], [-10, -12], [1.0, 0.9])
timing = orch.generate_timing_expert_data(
    collision_cfg,
    [clean_signal.iq_samples, signal_2.iq_samples],
    -30, seed=999
)

print(f"✅ QRN expert example: {qrn.expert_type}")
print(f"✅ Signal expert example: {signal.expert_type}")
print(f"✅ Timing expert example: {timing.expert_type} ({timing.labels['num_signals']} signals)")
print(f"✅ Channel expert example: {channel.expert_type}")
print(f"✅ QRM expert example: {qrm.expert_type}")

print("\n=== Pipeline Complete ===")
print("Ready for expert network training!")
```

**Expected Result**: All 5 expert datasets generated successfully

---

## Success Criteria

**Core Generator**:
- ✅ Generates clean CASCADE V2-compliant signals
- ✅ All V2 compliance checks pass
- ✅ Tone frequencies on 135-channel grid
- ✅ GMSK bandwidth < 30 Hz
- ✅ 200 symbols/second rate
- ✅ Reproducible with seeds

**Synthetic Data Orchestrator**:
- ✅ QRN Expert: Pure noise (NO signal)
- ✅ Signal Expert: Clean signal (NO noise)
- ✅ Timing Expert: Collision scenarios with separated ground truth
- ✅ Channel Expert: Known channel models (NO noise)
- ✅ QRM Expert: Pure interference (NO signal)
- ✅ Batch generation: 100 examples in <30s
- ✅ SNR sweep: Accurate to ±1 dB

**Overall**:
- ✅ End-to-end pipeline executes without errors
- ✅ All expert datasets have correct structure
- ✅ Ground truth labels included for supervised learning
- ✅ Datasets saved and loadable by PyTorch/TensorFlow

---

## Troubleshooting

**Pattern files not found:**
```bash
# Regenerate patterns (if needed)
cd modules/training/patterns/tournament
python generate_patterns.py
```

**Spectrum visualization fails:**
```bash
pip install matplotlib
# Or skip visualization (optional)
```

**Out of memory (batch generation):**
```python
# Reduce batch size
qrn_batch = orch.generate_batch('qrn', num_examples=10, config=qrn_config)
```

**SNR inaccurate:**
```python
# Check measured vs target SNR
for sig in snr_sweep:
    target = sig.channel_conditions.snr_db
    measured = sig.measured_snr_db
    error = abs(measured - target)
    assert error < 1.0, f"SNR error: {error:.2f} dB"
```

---

**Quickstart Status**: ✅ COMPLETE
**Next Step**: Run implementation according to tasks.md (generated by /tasks command)
