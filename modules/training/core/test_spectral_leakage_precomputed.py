#!/usr/bin/env python3
"""
Test spectral leakage in pre-computed GMSK patterns vs on-the-fly generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from pathlib import Path


def compute_spectrum(signal, sample_rate=48000):
    """Compute power spectral density."""
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    psd = np.abs(fft)**2 / len(signal)

    # Sort by frequency
    sort_idx = np.argsort(freqs)
    return freqs[sort_idx], psd[sort_idx]


def test_spectral_leakage():
    """Compare spectral leakage between pre-computed and on-the-fly GMSK."""

    print("Testing spectral leakage in always-on generation...")
    print("=" * 60)

    # Generate with pre-computed patterns (current implementation)
    patterns_dir = Path('/home/ubuntu/cascade-rf/cascade/modules/training/patterns/patterns')
    gen_precomputed = GPUSignalGenerator(
        device='cuda',
        patterns_dir=patterns_dir,
        use_gpu_polar=True,
        use_always_on=True
    )

    # Generate a signal
    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([20], device='cuda'),  # Middle of band
        modulations=['QPSK'],
        polar_rates=[(2, 3)],
        data_symbol_rates=torch.tensor([150], device='cuda')
    )

    message = [b'Test message for spectral analysis']

    signal_precomputed, _ = gen_precomputed.generate_batch(
        batch_params, message, fixed_length=None, num_centers=4
    )

    # Convert to numpy
    signal_np = signal_precomputed[0].cpu().numpy()

    # Compute spectrum
    freqs, psd = compute_spectrum(signal_np)

    # Find band edges (based on 4 centers + 2 alternating = 6 frequencies)
    # Center triple at 20: channels 60, 61, 62 (60*14 + 300 = 1140, 1154, 1168 Hz)
    # With 4 centers and 14 Hz spacing: 6 tones span ~84 Hz
    center_freq = 300 + 61 * 14  # Middle tone of triple 20
    bandwidth = 84  # 6 tones × 14 Hz

    band_low = center_freq - bandwidth/2
    band_high = center_freq + bandwidth/2

    # Calculate in-band and out-of-band power
    in_band_mask = (freqs >= band_low) & (freqs <= band_high)
    out_of_band_mask = ~in_band_mask

    in_band_power = np.sum(psd[in_band_mask])
    out_of_band_power = np.sum(psd[out_of_band_mask])
    total_power = np.sum(psd)

    in_band_pct = in_band_power / total_power * 100
    out_of_band_pct = out_of_band_power / total_power * 100

    # Out-of-band suppression
    if out_of_band_power > 0:
        suppression_db = 10 * np.log10(in_band_power / out_of_band_power)
    else:
        suppression_db = float('inf')

    print(f"\nSpectral Analysis (Always-On with Pre-computed GMSK):")
    print(f"  Center frequency: {center_freq:.0f} Hz")
    print(f"  Bandwidth: {bandwidth:.0f} Hz ({band_low:.0f} - {band_high:.0f} Hz)")
    print(f"  In-band power: {in_band_pct:.1f}%")
    print(f"  Out-of-band power: {out_of_band_pct:.1f}%")
    print(f"  Out-of-band suppression: {suppression_db:.1f} dB")

    if suppression_db < 35:
        print(f"\n⚠️  WARNING: Spectral leakage detected!")
        print(f"     Expected: >40 dB suppression (from GMSK BT=0.3)")
        print(f"     Actual: {suppression_db:.1f} dB")
    else:
        print(f"\n✓ Good spectral containment ({suppression_db:.1f} dB)")

    # Plot spectrum
    plt.figure(figsize=(12, 6))
    plt.semilogy(freqs, psd)
    plt.axvline(band_low, color='r', linestyle='--', label='Band edges', alpha=0.7)
    plt.axvline(band_high, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Always-On 4-Center Spectrum (Pre-computed GMSK)\nOOB Suppression: {suppression_db:.1f} dB')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-3000, 3000)
    plt.savefig('spectral_leakage_precomputed.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved spectrum to spectral_leakage_precomputed.png")

    # Also check the pre-computed patterns themselves
    print(f"\nPre-computed pattern lengths:")
    for pid in range(4):
        lower_len = len(gen_precomputed.precomputed_gmsk[pid]['lower'])
        upper_len = len(gen_precomputed.precomputed_gmsk[pid]['upper'])
        print(f"  Pattern {pid}: {lower_len} samples ({lower_len/48000:.2f}s)")


if __name__ == "__main__":
    test_spectral_leakage()
