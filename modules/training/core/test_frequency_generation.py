#!/usr/bin/env python3
"""
Test to verify what frequencies are actually being generated.
"""

import torch
import numpy as np
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters

def test_frequency_generation():
    """Test what frequencies are actually generated."""

    print("=" * 70)
    print("TESTING ACTUAL FREQUENCY GENERATION")
    print("=" * 70)

    # Create generator
    gen = GPUSignalGenerator(device='cuda', use_always_on=True)

    print(f"\nGPUSignalGenerator constants:")
    print(f"  MIN_FREQ: {gen.MIN_FREQ} Hz")
    print(f"  MAX_FREQ: {gen.MAX_FREQ} Hz")
    print(f"  TONE_SPACING: {gen.TONE_SPACING} Hz")
    print(f"  NUM_FREQUENCY_TRIPLES: {gen.NUM_FREQUENCY_TRIPLES}")

    # Test with frequency triple 0 (lowest)
    print(f"\n" + "="*70)
    print("Testing frequency triple 0 (should be lowest allowed)")
    print("="*70)

    freq_triple = torch.tensor([0], device='cuda')
    tones_a, tones_b, tones_c = gen.get_tone_triple_frequencies_batch(freq_triple)

    print(f"\nBase tones (no offset):")
    print(f"  tone_a: {tones_a[0].item():.1f} Hz")
    print(f"  tone_b: {tones_b[0].item():.1f} Hz")
    print(f"  tone_c: {tones_c[0].item():.1f} Hz")

    # Now test with 4-center offsets
    print(f"\nWith 4-center asymmetric offsets [0, +30, +60, +90] Hz:")
    for center_idx in range(4):
        freq_offset = center_idx * 30
        tone_a_offset = tones_a[0].item() + freq_offset
        tone_b_offset = tones_b[0].item() + freq_offset
        tone_c_offset = tones_c[0].item() + freq_offset

        print(f"  Center {center_idx}: {tone_a_offset:.1f}, {tone_b_offset:.1f}, {tone_c_offset:.1f} Hz")

        if tone_a_offset < 300:
            print(f"    ⚠️  WARNING: tone_a below 300 Hz!")

    # Test frequency triple 49 (highest)
    print(f"\n" + "="*70)
    print(f"Testing frequency triple {gen.NUM_FREQUENCY_TRIPLES - 1} (should be highest allowed)")
    print("="*70)

    freq_triple = torch.tensor([gen.NUM_FREQUENCY_TRIPLES - 1], device='cuda')
    tones_a, tones_b, tones_c = gen.get_tone_triple_frequencies_batch(freq_triple)

    print(f"\nBase tones (no offset):")
    print(f"  tone_a: {tones_a[0].item():.1f} Hz")
    print(f"  tone_b: {tones_b[0].item():.1f} Hz")
    print(f"  tone_c: {tones_c[0].item():.1f} Hz")

    # Now test with 4-center offsets
    print(f"\nWith 4-center asymmetric offsets [0, +30, +60, +90] Hz:")
    for center_idx in range(4):
        freq_offset = center_idx * 30
        tone_a_offset = tones_a[0].item() + freq_offset
        tone_b_offset = tones_b[0].item() + freq_offset
        tone_c_offset = tones_c[0].item() + freq_offset

        print(f"  Center {center_idx}: {tone_a_offset:.1f}, {tone_b_offset:.1f}, {tone_c_offset:.1f} Hz")

        if tone_c_offset > 2800:
            print(f"    ⚠️  WARNING: tone_c above 2800 Hz!")

    # Actually generate a signal and check its spectrum
    print(f"\n" + "="*70)
    print("Generating actual signal and analyzing spectrum")
    print("="*70)

    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([0], device='cuda'),
        modulations=['QPSK'],
        polar_rates=[(2, 3)],
        data_symbol_rates=torch.tensor([150], device='cuda')
    )

    messages = [b"TEST"]

    signals, metadata = gen.generate_batch(
        batch_params, messages, fixed_length=96000, num_centers=4
    )

    # FFT to see actual frequency content
    signal_fft = torch.fft.fft(signals[0])
    freqs = torch.fft.fftfreq(len(signal_fft), 1/gen.SAMPLE_RATE).cpu().numpy()
    power_spectrum = torch.abs(signal_fft).cpu().numpy() ** 2

    # Find peak frequencies
    peak_threshold = np.max(power_spectrum) * 0.1  # 10% of max
    peak_indices = np.where(power_spectrum > peak_threshold)[0]
    peak_freqs = np.abs(freqs[peak_indices])

    print(f"\nPeak frequencies in generated signal (>10% of max power):")
    unique_peaks = np.unique(np.round(peak_freqs / 10) * 10)  # Round to 10 Hz
    for freq in sorted(unique_peaks[:20]):  # Show first 20
        print(f"  {freq:.0f} Hz")

    # Check for sub-300 Hz content
    sub_300_mask = np.abs(freqs) < 300
    sub_300_power = np.sum(power_spectrum[sub_300_mask])
    total_power = np.sum(power_spectrum)
    sub_300_ratio = sub_300_power / total_power * 100

    print(f"\nSub-300 Hz energy:")
    print(f"  Power ratio: {sub_300_ratio:.2f}% of total")

    if sub_300_ratio > 1.0:
        print(f"  ⚠️  WARNING: Significant energy below 300 Hz!")
    else:
        print(f"  ✓ Sub-300 Hz energy is negligible")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_frequency_generation()
