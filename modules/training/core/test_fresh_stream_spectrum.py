#!/usr/bin/env python3
"""
Generate a FRESH stream (not from cache) and check sub-300 Hz content.
"""

import torch
import numpy as np
from streaming_cascade_dataset import StreamingCascadeDataset

def test_fresh_stream():
    """Generate fresh stream and analyze."""

    print("="*80)
    print("GENERATING FRESH STREAM (NO CACHE)")
    print("="*80)

    # Create dataset with cache DISABLED
    dataset = StreamingCascadeDataset(
        num_streams=1,
        stream_duration_sec=10.0,
        use_cache=False,  # CRITICAL: No cache!
        regenerate_cache=True,
        batch_size=1,
        device='cuda',
        enable_transceiver_impairments=True,
        tx_impairments_enabled=True,
        rx_impairments_enabled=True
    )

    # Get first stream
    print("\nLoading stream 0...")
    stream_data = dataset[0]

    stream_iq = stream_data['stream_iq']  # [2, num_samples]

    # Reconstruct complex signal
    iq_complex = stream_iq[0] + 1j * stream_iq[1]

    # Analyze spectrum
    fft = np.fft.fft(iq_complex)
    freqs = np.fft.fftfreq(len(fft), 1/48000)
    power = np.abs(fft) ** 2

    # Sub-300 Hz analysis
    sub_300_mask = np.abs(freqs) < 300
    sub_300_power = np.sum(power[sub_300_mask])
    total_power = np.sum(power)
    sub_300_ratio = sub_300_power / total_power * 100

    # Time domain analysis
    amplitude = np.abs(iq_complex)
    mean_amp = np.mean(amplitude)
    max_amp = np.max(amplitude)

    # Find quiet periods (should have low amplitude)
    quiet_threshold = mean_amp * 0.5
    quiet_samples = np.sum(amplitude < quiet_threshold)
    quiet_ratio = quiet_samples / len(amplitude) * 100

    print("\n" + "="*80)
    print("ANALYSIS RESULTS:")
    print("="*80)
    print(f"\nSpectrum:")
    print(f"  Sub-300 Hz power: {sub_300_ratio:.3f}%")
    print(f"  Total power: {total_power:.2e}")

    print(f"\nTime domain:")
    print(f"  Mean amplitude: {mean_amp:.4f}")
    print(f"  Max amplitude: {max_amp:.4f}")
    print(f"  Quiet periods (<50% mean): {quiet_ratio:.1f}% of stream")

    print(f"\nMetadata:")
    print(f"  Num messages: {stream_data['num_messages']}")
    print(f"  Propagation: {stream_data['propagation_mode']}")
    print(f"  QRN type: {stream_data['qrn_type']}")

    # Check if problem exists
    print("\n" + "="*80)
    if sub_300_ratio > 2.0:
        print("⚠️  PROBLEM: Excessive sub-300 Hz energy ({:.3f}%)".format(sub_300_ratio))
    else:
        print("✓ Sub-300 Hz energy is acceptable ({:.3f}%)".format(sub_300_ratio))

    if quiet_ratio < 30:
        print(f"⚠️  PROBLEM: Not enough quiet periods ({quiet_ratio:.1f}%, expected >30%)")
        print("    This means noise floor is too high or signals too long!")
    else:
        print(f"✓ Quiet periods look good ({quiet_ratio:.1f}%)")

    print("="*80)

if __name__ == "__main__":
    test_fresh_stream()
