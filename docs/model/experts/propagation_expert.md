# Propagation Expert Network

Specializes in estimating and compensating for channel propagation effects including fading, multipath, and Doppler.

## Architecture

```
Input: 1024D shared features
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

## Learned Behaviors

### Channel Estimation
Infers channel state from corrupted signal:
- **Fading depth**: How much signal varies
- **Coherence time**: How fast it changes
- **Delay spread**: Multipath extent

### Inverse Filtering
Learns to undo distortions:
- **Flat fading**: Simple gain/phase correction
- **Frequency-selective**: Per-frequency equalization
- **Time-varying**: Tracking filter

### Multipath Compensation
Combines delayed signal copies:
- **Constructive combining**: Like a Rake receiver
- **Destructive cancellation**: Null avoidance
- **Delay estimation**: Up to 10ms spreads

## Propagation Modes

### Ionospheric Propagation
```python
ionospheric_model = {
    'fading_rate': 0.1-1 Hz,
    'delay_spread': 1-5 ms,
    'doppler_shift': ±10 Hz,
    'compensation': 'slow_tracking_equalizer'
}
```

### Tropospheric Ducting
```python
ducting_model = {
    'fading_rate': 0.01-0.1 Hz,
    'delay_spread': 10-50 ms,
    'doppler_shift': ±1 Hz,
    'compensation': 'long_equalizer'
}
```

### Ground Wave
```python
ground_wave_model = {
    'fading_rate': ~0 Hz (stable),
    'delay_spread': <1 ms,
    'doppler_shift': 0 Hz,
    'compensation': 'static_correction'
}
```

## Equalizer Strategies

### Zero-Forcing Equalizer
```python
def zero_forcing(channel_estimate):
    # Perfect inversion (can amplify noise)
    return 1.0 / channel_estimate
```

### MMSE Equalizer
```python
def mmse_equalizer(channel_estimate, snr):
    # Optimal trade-off
    H = channel_estimate
    return H.conj() / (|H|^2 + 1/snr)
```

### Decision-Feedback Equalizer
```python
def dfe_equalizer(signal, decisions):
    # Use past decisions to improve future ones
    feedforward = fir_filter(signal)
    feedback = iir_filter(decisions)
    return feedforward - feedback
```

## Multipath Handling

### Delay Profile Estimation
```python
def estimate_delays(signal):
    # Correlate with known patterns
    delays = []
    for tap in range(50):  # 50 delay taps
        correlation = correlate(signal[tap:], reference)
        if correlation > threshold:
            delays.append({
                'delay': tap,
                'amplitude': correlation,
                'phase': angle(correlation)
            })
    return delays
```

### Rake-like Combining
```python
def rake_combine(signal, delay_profile):
    combined = 0
    for tap in delay_profile:
        # Align and combine coherently
        aligned = shift(signal, -tap.delay)
        weighted = aligned * tap.amplitude.conj()
        combined += weighted
    return combined
```

## Training with Real Propagation

### FT8/WSPR Data Integration
```python
def train_on_ft8_wspr():
    # Use real propagation measurements
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

## Fading Mitigation

### Slow Fading (Ionospheric)
- Track with Kalman filter
- Update every 100ms
- Predict next state

### Fast Fading (Mobile/Flutter)
- Wideband averaging
- Diversity combining
- Interleaving helps

### Deep Fades
- Cannot fully compensate
- Signal conductor to reduce weight
- Wait for conditions to improve

## Integration with Conductor

The conductor weights this expert based on:
- **High weight**: Multipath detected, fading observed
- **Low weight**: Stable channel, line-of-sight
- **Typical range**: 0.05-0.6

## Performance Metrics

- **Channel Estimation Error**: <3 dB typically
- **Multipath Compensation**: 10-15 dB improvement
- **Fading Mitigation**: 5-10 dB improvement
- **Computation**: ~2.5ms on Raspberry Pi 4
- **Parameters**: ~900K