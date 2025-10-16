#!/usr/bin/env python3
"""
Test if transceiver impairments create sub-300 Hz artifacts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from gpu_channel_simulator import GPUChannelSimulator

def analyze_spectrum(signal, sample_rate, title):
    """Analyze and display frequency spectrum."""
    signal_np = signal.cpu().numpy()

    # FFT
    fft = np.fft.fft(signal_np)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    power = np.abs(fft) ** 2

    # Find sub-300 Hz power
    sub_300_mask = np.abs(freqs) < 300
    sub_300_power = np.sum(power[sub_300_mask])
    total_power = np.sum(power)
    sub_300_ratio = sub_300_power / total_power * 100

    # Find peak frequencies
    peak_threshold = np.max(power) * 0.01  # 1% of max
    peak_indices = np.where(power > peak_threshold)[0]
    peak_freqs = np.abs(freqs[peak_indices])

    print(f"\n{title}:")
    print(f"  Sub-300 Hz power: {sub_300_ratio:.3f}%")
    print(f"  Total power: {total_power:.2e}")
    print(f"  Peak frequencies (>1% max):")

    unique_peaks = np.unique(np.round(peak_freqs / 10) * 10)
    for freq in sorted(unique_peaks[:15]):
        if freq < 3000:  # Only show audio range
            print(f"    {freq:.0f} Hz")

    return sub_300_ratio

def test_impairments_create_artifacts():
    """Test if transceiver impairments create sub-300 Hz content."""

    print("=" * 70)
    print("TESTING TRANSCEIVER IMPAIRMENTS FOR SUB-300 HZ ARTIFACTS")
    print("=" * 70)

    # Create generator
    gen = GPUSignalGenerator(device='cuda', use_always_on=True)

    # Create channel simulator WITH impairments
    channel_sim = GPUChannelSimulator(sample_rate=48000, device='cuda',
                                     enable_transceiver_impairments=True)

    # Generate clean signal at frequency triple 0
    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([0], device='cuda'),  # Lowest triple
        modulations=['QPSK'],
        polar_rates=[(2, 3)],
        data_symbol_rates=torch.tensor([150], device='cuda')
    )

    messages = [b"TEST MESSAGE"]

    clean_signals, _ = gen.generate_batch(
        batch_params, messages, fixed_length=96000, num_centers=4
    )

    # Analyze clean signal
    ratio_clean = analyze_spectrum(clean_signals[0], gen.SAMPLE_RATE, "Clean signal (no impairments)")

    # Apply TX impairments
    tx_signals, tx_profiles, _ = channel_sim.apply_tx_rx_impairments_batch(
        clean_signals, apply_tx=True, apply_rx=False
    )

    ratio_tx = analyze_spectrum(tx_signals[0], gen.SAMPLE_RATE,
                                f"After TX impairments ({tx_profiles[0]})")

    # Apply RX impairments
    rx_signals = channel_sim.apply_rx_impairments_only(tx_signals, tx_profiles)

    ratio_rx = analyze_spectrum(rx_signals[0], gen.SAMPLE_RATE,
                                f"After RX impairments ({tx_profiles[0]})")

    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY:")
    print(f"  Clean signal:      {ratio_clean:.3f}% sub-300 Hz")
    print(f"  After TX impair:   {ratio_tx:.3f}% sub-300 Hz  (Δ = {ratio_tx - ratio_clean:+.3f}%)")
    print(f"  After RX impair:   {ratio_rx:.3f}% sub-300 Hz  (Δ = {ratio_rx - ratio_tx:+.3f}%)")

    if ratio_rx > 1.0:
        print(f"\n⚠️  PROBLEM: Transceiver impairments created {ratio_rx:.3f}% sub-300 Hz content!")
    else:
        print(f"\n✓ Transceiver impairments are clean (<1% sub-300 Hz)")

    print("="*70)

if __name__ == "__main__":
    test_impairments_create_artifacts()
