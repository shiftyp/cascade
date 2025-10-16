#!/usr/bin/env python3
"""
Visualize always-on signal to diagnose gaps and width issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from pathlib import Path


def test_signal_visualization():
    """Visualize the always-on signal structure."""

    print("Generating always-on signal for visualization...")

    patterns_dir = Path('/home/ubuntu/cascade-rf/cascade/modules/training/patterns/patterns')
    gen = GPUSignalGenerator(device='cuda', patterns_dir=patterns_dir, use_gpu_polar=True, use_always_on=True)

    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([20], device='cuda'),
        modulations=['BPSK'],
        polar_rates=[(1, 2)],
        data_symbol_rates=torch.tensor([75], device='cuda')
    )

    # Generate pattern-only signal (no data)
    signal, _ = gen.generate_batch(batch_params, [b''], fixed_length=None, num_centers=4)
    signal_np = signal[0].cpu().numpy()

    # Plot time-domain envelope
    envelope = np.abs(signal_np)
    time_ms = np.arange(len(signal_np)) / 48000 * 1000

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time domain
    axes[0].plot(time_ms, envelope)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Signal Envelope (Should be constant with no gaps for always-on centers)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, min(200, time_ms[-1]))  # First 200ms

    # Check for gaps
    mean_envelope = np.mean(envelope[1000:-1000])  # Skip edges
    gap_threshold = mean_envelope * 0.1  # 10% of mean
    gaps = envelope < gap_threshold
    gap_pct = np.sum(gaps) / len(gaps) * 100

    axes[0].axhline(gap_threshold, color='r', linestyle='--', label=f'Gap threshold ({gap_pct:.1f}% below)')
    axes[0].legend()

    # Spectrogram
    from scipy import signal as sp_signal
    f, t, Sxx = sp_signal.spectrogram(signal_np, fs=48000, nperseg=1024, noverlap=512)
    axes[1].pcolormesh(t * 1000, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Spectrogram (Check for gaps and width issues)')
    axes[1].set_ylim(800, 1400)
    axes[1].set_xlim(0, min(200, t[-1] * 1000))

    plt.tight_layout()
    plt.savefig('signal_visualization_gaps.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to signal_visualization_gaps.png")

    print(f"\nDiagnostics:")
    print(f"  Signal length: {len(signal_np)} samples ({len(signal_np)/48000:.2f}s)")
    print(f"  Mean envelope: {mean_envelope:.3f}")
    print(f"  Samples below threshold: {np.sum(gaps)} ({gap_pct:.1f}%)")

    if gap_pct > 5:
        print(f"  ⚠️  WARNING: Signal has {gap_pct:.1f}% gaps!")
        print(f"     Center carriers should be always-on (no gaps)")
    else:
        print(f"  ✓ Minimal gaps")

    # Check spectral width
    fft = np.fft.fft(signal_np)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal_np), 1/48000))
    psd = np.fft.fftshift(np.abs(fft)**2 / len(signal_np))

    # Find -3dB bandwidth
    max_psd = np.max(psd)
    half_power = max_psd / 2
    above_half = psd > half_power
    bw_indices = np.where(above_half)[0]
    if len(bw_indices) > 0:
        bw_hz = freqs[bw_indices[-1]] - freqs[bw_indices[0]]
        print(f"\n  -3dB Bandwidth: {bw_hz:.0f} Hz")
        print(f"  Expected: ~200-250 Hz (4 centers spanning 208 Hz + GMSK sidebands)")

        if bw_hz > 400:
            print(f"  ⚠️  Signal is {bw_hz/250:.1f}× wider than expected!")


if __name__ == "__main__":
    test_signal_visualization()
