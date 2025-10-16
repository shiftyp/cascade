#!/usr/bin/env python3
"""
Simple test: Check if signals are appearing below 500 Hz.

Creates one stream and analyzes frequency content.
"""

import torch
import numpy as np
from streaming_cascade_dataset import StreamingCascadeDataset
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal


def main():
    print("Creating test stream with transceiver impairments...")

    # Create minimal dataset
    dataset = StreamingCascadeDataset(
        num_streams=1,
        stream_duration_sec=2.0,  # Short 2s stream
        window_duration_sec=2.0,  # Full stream as one window
        sample_rate=48000,
        enable_transceiver_impairments=True,
        tx_impairments_enabled=True,
        rx_impairments_enabled=True,
        cache_dir=None  # No caching for test
    )

    # Get one stream
    print("Generating stream...")
    stream_data = dataset[0]
    iq_stream = stream_data['iq']

    print(f"Stream shape: {iq_stream.shape}")
    print(f"Stream dtype: {iq_stream.dtype}")

    # FFT analysis
    print("\nAnalyzing frequency content...")
    fft_result = np.fft.fft(iq_stream)
    freqs = np.fft.fftfreq(len(iq_stream), 1/48000)

    # Only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    fft_pos = np.abs(fft_result[pos_mask]) ** 2

    # Energy distribution
    below_500 = np.sum(fft_pos[freqs_pos < 500])
    band_500_2600 = np.sum(fft_pos[(freqs_pos >= 500) & (freqs_pos <= 2600)])
    above_2600 = np.sum(fft_pos[freqs_pos > 2600])
    total = below_500 + band_500_2600 + above_2600

    print(f"\nEnergy distribution:")
    print(f"  Below 500 Hz:     {below_500/total*100:6.2f}%")
    print(f"  CASCADE (500-2600): {band_500_2600/total*100:6.2f}%")
    print(f"  Above 2600 Hz:    {above_2600/total*100:6.2f}%")

    # Find peak frequencies below 500 Hz
    sub_500_mask = (freqs_pos < 500) & (freqs_pos > 10)  # Exclude DC
    if np.any(sub_500_mask):
        fft_sub500 = fft_pos[sub_500_mask]
        freqs_sub500 = freqs_pos[sub_500_mask]

        # Top 5 peaks
        peak_indices = np.argsort(fft_sub500)[-5:][::-1]

        print(f"\nTop peaks below 500 Hz:")
        for i, idx in enumerate(peak_indices):
            freq = freqs_sub500[idx]
            power = fft_sub500[idx]
            pct_of_total = power / total * 100
            print(f"  {i+1}. {freq:7.1f} Hz: {pct_of_total:6.3f}% of total energy")

    # Create spectrogram
    print("\nGenerating spectrogram...")
    f, t, Sxx = scipy_signal.spectrogram(iq_stream, fs=48000,
                                         nperseg=2048, noverlap=1536)

    plt.figure(figsize=(14, 8))

    # Main spectrogram
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Power (dB)')
    plt.axhline(500, color='red', linestyle='--', linewidth=2, alpha=0.7, label='500 Hz (MIN_FREQ)')
    plt.axhline(2600, color='red', linestyle='--', linewidth=2, alpha=0.7, label='2600 Hz (MAX_FREQ)')
    plt.legend(loc='upper right')
    plt.title('Full Spectrum')
    plt.ylim([0, 3000])

    # Zoomed view of 0-800 Hz
    plt.subplot(2, 1, 2)
    zoom_mask = f <= 800
    plt.pcolormesh(t, f[zoom_mask], 10*np.log10(Sxx[zoom_mask, :] + 1e-10),
                   shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Power (dB)')
    plt.axhline(500, color='red', linestyle='--', linewidth=2, alpha=0.7, label='500 Hz (MIN_FREQ)')
    plt.legend(loc='upper right')
    plt.title('Zoomed: 0-800 Hz (should be mostly noise below 500 Hz)')

    plt.tight_layout()
    plt.savefig('test_sub500_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved spectrogram to test_sub500_analysis.png")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT:")
    print("="*60)

    if below_500/total < 0.05:
        print(f"✅ EXCELLENT: Only {below_500/total*100:.2f}% below 500 Hz (likely just noise)")
    elif below_500/total < 0.15:
        print(f"✅ GOOD: {below_500/total*100:.2f}% below 500 Hz (acceptable for thermal noise)")
    elif below_500/total < 0.30:
        print(f"⚠️  WARNING: {below_500/total*100:.2f}% below 500 Hz (higher than expected)")
    else:
        print(f"❌ FAIL: {below_500/total*100:.2f}% below 500 Hz (SIGNALS leaking below 500 Hz!)")

    print("\nNote: Some sub-500 Hz content from thermal/atmospheric noise is expected.")
    print("The problem occurs when CASCADE SIGNALS appear below 500 Hz.")


if __name__ == "__main__":
    main()
