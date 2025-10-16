#!/usr/bin/env python3
"""
Generate spectrogram to diagnose gaps and width issues in always-on signals.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from pathlib import Path


def test_spectrogram():
    """Generate spectrogram to visualize always-on signal behavior."""

    print("Generating spectrogram for always-on signal...")

    patterns_dir = Path('/home/ubuntu/cascade-rf/cascade/modules/training/patterns/patterns')
    gen = GPUSignalGenerator(device='cuda', patterns_dir=patterns_dir, use_gpu_polar=True, use_always_on=True)

    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([20], device='cuda'),
        modulations=['QPSK'],
        polar_rates=[(2, 3)],
        data_symbol_rates=torch.tensor([150], device='cuda')
    )

    # Generate signal with data
    signal, metadata = gen.generate_batch(batch_params, [b'Test message for spectrogram'],
                                          fixed_length=None, num_centers=4)
    signal_np = signal[0].cpu().numpy()

    print(f"Signal length: {len(signal_np)} samples ({len(signal_np)/48000:.2f}s)")
    print(f"Metadata: {metadata[0]}")

    # Create high-resolution spectrogram
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Spectrogram - full time range
    f, t, Sxx = sp_signal.spectrogram(signal_np, fs=48000, nperseg=512, noverlap=480)

    axes[0].pcolormesh(t * 1000, f, 10 * np.log10(Sxx + 1e-12),
                      shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_title('Always-On Spectrogram - Full View')
    axes[0].set_ylim(900, 1400)
    axes[0].grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Mark expected frequencies
    # 4 centers: 1034, 1094, 1154, 1214 Hz
    for freq in [1034, 1094, 1154, 1214]:
        axes[0].axhline(freq, color='r', linestyle='--', alpha=0.5, linewidth=0.8)

    # Alternating tones: 1020, 1080, 1140, 1200 Hz (lower, even)
    #                    1048, 1108, 1168, 1228 Hz (upper, odd)
    for freq in [1020, 1080, 1140, 1200]:
        axes[0].axhline(freq, color='cyan', linestyle=':', alpha=0.5, linewidth=0.8, label='Lower (even)' if freq == 1020 else '')
    for freq in [1048, 1108, 1168, 1228]:
        axes[0].axhline(freq, color='yellow', linestyle=':', alpha=0.5, linewidth=0.8, label='Upper (odd)' if freq == 1048 else '')

    axes[0].legend(loc='upper right')

    # Zoom into first 100ms
    axes[1].pcolormesh(t * 1000, f, 10 * np.log10(Sxx + 1e-12),
                      shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Always-On Spectrogram - Zoom (First 100ms)')
    axes[1].set_ylim(900, 1400)
    axes[1].set_xlim(0, 100)
    axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Mark frequencies
    for freq in [1034, 1094, 1154, 1214]:
        axes[1].axhline(freq, color='r', linestyle='--', alpha=0.7, linewidth=1.0)

    plt.tight_layout()
    plt.savefig('spectrogram_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved spectrogram to spectrogram_diagnosis.png")

    print(f"\nWhat to look for:")
    print(f"  - Center tones (red dashed): Should be continuous horizontal lines (no gaps)")
    print(f"  - Alternating tones: Should appear on even/odd symbols (alternating pattern)")
    print(f"  - If centers have gaps: Pattern layer is being gated incorrectly")
    print(f"  - If signals appear twice as wide: Alternating tones spanning wrong time")


if __name__ == "__main__":
    test_spectrogram()
