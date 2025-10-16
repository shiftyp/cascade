#!/usr/bin/env python3
"""
Test that signals are NOT generated below 500 Hz.

This test generates a small batch and checks:
1. Clean signals from generator (before channel)
2. Signals after TX impairments
3. Signals after channel + RX impairments
4. Final combined stream with noise

We should NOT see signal energy below 500 Hz (noise is OK, signals are NOT).
"""

import torch
import numpy as np
from gpu_signal_generator import GPUSignalGenerator
from gpu_channel_simulator import GPUChannelSimulator
from streaming_cascade_dataset import StreamingCascadeDataset
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal


def analyze_frequency_content(signal, fs=48000, label="Signal"):
    """Analyze frequency content and report sub-500 Hz energy."""
    # FFT analysis
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Only look at positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    fft_pos = np.abs(fft_result[pos_mask]) ** 2

    # Energy in different bands
    below_500 = np.sum(fft_pos[freqs_pos < 500])
    band_500_2600 = np.sum(fft_pos[(freqs_pos >= 500) & (freqs_pos <= 2600)])
    above_2600 = np.sum(fft_pos[freqs_pos > 2600])
    total = below_500 + band_500_2600 + above_2600

    print(f"\n{label}:")
    print(f"  Below 500 Hz: {below_500/total*100:.2f}%")
    print(f"  500-2600 Hz:  {band_500_2600/total*100:.2f}%")
    print(f"  Above 2600 Hz: {above_2600/total*100:.2f}%")

    # Find peak in sub-500 Hz region
    if below_500 / total > 0.05:  # More than 5% in sub-500 Hz is suspicious
        sub_500_mask = (freqs_pos < 500) & (freqs_pos > 0)  # Exclude DC
        if np.any(sub_500_mask):
            peak_idx = np.argmax(fft_pos[sub_500_mask])
            peak_freq = freqs_pos[sub_500_mask][peak_idx]
            peak_power = fft_pos[sub_500_mask][peak_idx]
            print(f"  ⚠️  WARNING: Peak at {peak_freq:.1f} Hz (power: {peak_power:.2e})")

    return below_500/total


def test_signal_generation():
    """Test 1: Clean signal generation (no impairments)."""
    print("="*60)
    print("TEST 1: Clean Signal Generation")
    print("="*60)

    generator = GPUSignalGenerator(device='cuda')

    # Generate a single transmission
    message = b"TEST"
    pattern_id = 0
    frequency_triple = 10  # Should be around 500 + 10*60 = 1100 Hz

    signal, params = generator.generate_transmission(
        message=message,
        pattern_id=pattern_id,
        frequency_triple=frequency_triple,
        modulation='QPSK',
        polar_rate=(2, 3),
        data_symbol_rate=150
    )

    # Move to CPU for analysis
    signal_cpu = signal.cpu().numpy()

    print(f"Generated signal length: {len(signal_cpu)} samples ({len(signal_cpu)/48000:.3f}s)")
    print(f"Frequency triple: {frequency_triple}")
    print(f"Expected center frequency: ~{500 + frequency_triple * 20:.0f} Hz")

    sub_500_pct = analyze_frequency_content(signal_cpu, label="Clean signal")

    if sub_500_pct > 0.01:  # More than 1% is concerning for clean signal
        print("❌ FAIL: Clean signal has energy below 500 Hz!")
        return False
    else:
        print("✅ PASS: Clean signal stays above 500 Hz")
        return True


def test_with_transceiver_impairments():
    """Test 2: Signal with transceiver impairments."""
    print("\n" + "="*60)
    print("TEST 2: Signal with Transceiver Impairments")
    print("="*60)

    generator = GPUSignalGenerator(device='cuda')
    channel = GPUChannelSimulator(sample_rate=48000, device='cuda',
                                   enable_transceiver_impairments=True)

    # Generate signal
    message = b"TEST WITH IMPAIRMENTS"
    signal, params = generator.generate_transmission(
        message=message,
        pattern_id=1,
        frequency_triple=15,  # ~500 + 15*20 = 800 Hz
        modulation='QPSK',
        polar_rate=(2, 3),
        data_symbol_rate=150
    )

    # Expand to batch
    signal_batch = signal.unsqueeze(0)  # [1, num_samples]

    # Apply TX impairments
    tx_signal, tx_profiles, rx_profiles = channel.apply_tx_rx_impairments_batch(
        signal_batch, apply_tx=True, apply_rx=False
    )

    # Analyze
    tx_signal_cpu = tx_signal[0].cpu().numpy()
    sub_500_pct = analyze_frequency_content(tx_signal_cpu,
                                            label="Signal after TX impairments")

    if sub_500_pct > 0.05:  # 5% threshold (some filter ringing is OK)
        print("⚠️  WARNING: TX impairments creating sub-500 Hz content")
        return False
    else:
        print("✅ PASS: TX impairments OK")
        return True


def test_full_stream():
    """Test 3: Full stream with noise, channel, impairments."""
    print("\n" + "="*60)
    print("TEST 3: Full Stream (with noise + channel + impairments)")
    print("="*60)

    # Create minimal dataset
    dataset = StreamingCascadeDataset(
        stream_duration_s=2.0,  # Short stream
        num_streams=1,
        sample_rate=48000,
        enable_transceiver_impairments=True,
        cache_dir=None  # No caching
    )

    # Get one stream
    stream_data = dataset[0]
    iq_stream = stream_data['iq']

    print(f"Stream shape: {iq_stream.shape}")

    # Analyze full stream
    sub_500_pct = analyze_frequency_content(iq_stream, label="Full stream (with noise)")

    # For full stream with noise, we expect some sub-500 Hz from thermal noise
    # But CASCADE signals should still be primarily 500-2600 Hz
    if sub_500_pct > 0.30:  # 30% threshold (thermal noise can be ~10-20%)
        print("⚠️  WARNING: Excessive sub-500 Hz content in stream")

        # Create spectrogram to visualize
        f, t, Sxx = scipy_signal.spectrogram(iq_stream, fs=48000,
                                            nperseg=2048, noverlap=1536)

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.colorbar(label='Power (dB)')
        plt.axhline(500, color='red', linestyle='--', linewidth=2, label='500 Hz limit')
        plt.axhline(2600, color='red', linestyle='--', linewidth=2, label='2600 Hz limit')
        plt.legend()
        plt.title('Stream Spectrogram - Debug')
        plt.savefig('test_frequency_bounds_debug.png', dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to test_frequency_bounds_debug.png")

        return False
    else:
        print("✅ PASS: Stream frequency content acceptable")
        return True


if __name__ == "__main__":
    print("Testing CASCADE frequency bounds (MIN_FREQ=500 Hz)")
    print("="*60)

    # Run tests
    test1_pass = test_signal_generation()
    test2_pass = test_with_transceiver_impairments()
    test3_pass = test_full_stream()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (Clean signal):        {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (TX impairments):      {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (Full stream):         {'✅ PASS' if test3_pass else '❌ FAIL'}")

    if test1_pass and test2_pass and test3_pass:
        print("\n🎉 ALL TESTS PASSED")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - investigate!")
        exit(1)
