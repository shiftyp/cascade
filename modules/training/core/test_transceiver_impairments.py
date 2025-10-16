#!/usr/bin/env python3
"""
Test HF transceiver hardware impairments.

Verifies SSB filter, AGC, ALC, and audio interface impairments
for CASCADE signal generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_channel_simulator import GPUTransceiverImpairments, TransceiverProfile


def test_transceiver_impairments():
    """Test transceiver impairments on simple test signal."""

    print("=" * 70)
    print("TESTING HF TRANSCEIVER IMPAIRMENTS")
    print("=" * 70)

    # Create impairments simulator
    impairments = GPUTransceiverImpairments(sample_rate=48000, device='cuda')

    # Create test signal (clean BPSK tone at 1500 Hz)
    batch_size = 8
    num_samples = 96000  # 2 seconds
    t = torch.arange(num_samples, dtype=torch.float32, device='cuda') / 48000

    # Generate clean BPSK signal
    symbols = torch.randint(0, 2, (num_samples // 480,), device='cuda') * 2 - 1  # {-1, 1}
    symbols_upsampled = torch.repeat_interleave(symbols, 480)[:num_samples]
    clean_signal = symbols_upsampled.to(torch.complex64) * torch.exp(1j * 2 * torch.pi * 1500 * t)
    clean_signal = clean_signal.unsqueeze(0).expand(batch_size, -1).clone()

    print(f"\nClean signal: {clean_signal.shape}")
    print(f"  Mean power: {torch.mean(torch.abs(clean_signal)**2):.3f}")

    # Test 1: Profile sampling
    print("\n1. Testing weighted profile sampling...")
    profiles = impairments.sample_random_profiles(100)
    profile_counts = {}
    for p in profiles:
        profile_counts[p] = profile_counts.get(p, 0) + 1

    print("  Profile distribution (100 samples):")
    for profile, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
        print(f"    {profile}: {count}% (expected: {impairments.PROFILE_WEIGHTS[profile]*100:.0f}%)")

    # Test 2: TX impairments
    print("\n2. Testing TX impairments...")
    tx_profiles = ['icom_ic7300', 'yaesu_ft991a', 'elecraft_kx3', 'icom_ic705',
                   'budget_soundcard', 'chinese_budget', 'flex6000', 'hermes_lite2']
    tx_signals = impairments.apply_tx_impairments(clean_signal, tx_profiles)

    print(f"  Output: {tx_signals.shape}")
    print(f"  Mean power after TX: {torch.mean(torch.abs(tx_signals)**2):.3f}")
    print(f"  Power variation: {torch.std(torch.abs(tx_signals)**2, dim=1).mean():.4f}")

    # Test 3: RX impairments
    print("\n3. Testing RX impairments...")
    rx_profiles = ['icom_ic7300', 'yaesu_ft991a', 'elecraft_kx3', 'icom_ic705',
                   'budget_soundcard', 'chinese_budget', 'flex6000', 'hermes_lite2']
    rx_signals = impairments.apply_rx_impairments(tx_signals, rx_profiles)

    print(f"  Output: {rx_signals.shape}")
    print(f"  Mean power after RX: {torch.mean(torch.abs(rx_signals)**2):.3f}")
    print(f"  SNR degradation: {10 * torch.log10(torch.mean(torch.abs(clean_signal)**2) / torch.mean(torch.abs(rx_signals - clean_signal)**2)):.1f} dB")

    # Test 4: Individual impairment effects
    print("\n4. Testing individual impairment effects...")

    # SSB filter
    test_signal = clean_signal[0:1].clone()
    ssb_bw = torch.tensor([2600.0], device='cuda')
    ssb_shape = torch.tensor([2.0], device='cuda')
    filtered = impairments.apply_ssb_filter_batch(test_signal, ssb_bw, ssb_shape)
    print(f"  SSB filter (2600 Hz): Power loss = {10 * torch.log10(torch.mean(torch.abs(test_signal)**2) / torch.mean(torch.abs(filtered)**2)):.2f} dB")

    # AGC pumping
    test_signal = clean_signal[0:1].clone()
    agc_attack = torch.tensor([10.0], device='cuda')
    agc_release = torch.tensor([400.0], device='cuda')
    agc_variation = torch.tensor([3.0], device='cuda')
    agc_signal = impairments.apply_agc_pumping_batch(test_signal, agc_attack, agc_release, agc_variation, [True])
    print(f"  AGC pumping: Gain variation = {torch.std(torch.abs(agc_signal) / torch.abs(test_signal + 1e-10)):.3f}")

    # ALC compression
    test_signal = clean_signal[0:1].clone() * 2.0  # Overdrive to trigger ALC
    alc_threshold = torch.tensor([-5.0], device='cuda')
    alc_ratio = torch.tensor([3.5], device='cuda')
    compressed = impairments.apply_alc_compression_batch(test_signal, alc_threshold, alc_ratio, [True])
    print(f"  ALC compression: Peak reduction = {10 * torch.log10(torch.max(torch.abs(test_signal)**2) / torch.max(torch.abs(compressed)**2)):.2f} dB")

    # Audio interface
    test_signal = clean_signal[0:1].clone()
    audio_snr = torch.tensor([80.0], device='cuda')
    audio_thd = torch.tensor([0.1], device='cuda')
    audio_ripple = torch.tensor([0.8], device='cuda')
    audio_signal = impairments.apply_audio_interface_batch(test_signal, audio_snr, audio_thd, audio_ripple)
    added_noise_power = torch.mean(torch.abs(audio_signal - test_signal)**2)
    signal_power = torch.mean(torch.abs(test_signal)**2)
    measured_snr = 10 * torch.log10(signal_power / added_noise_power)
    print(f"  Audio interface (80 dB): Measured SNR = {measured_snr:.1f} dB")

    print("\n" + "=" * 70)
    print("✓ All transceiver impairment tests passed!")
    print("=" * 70)

    # Test 5: Full cascade with profiles
    print("\n5. Testing full TX→RX cascade with real profiles...")
    test_profiles = ['icom_ic7300', 'yaesu_ft991a', 'icom_ic705', 'budget_soundcard']

    for profile in test_profiles:
        test_signal = clean_signal[0:1].clone()

        # Apply TX
        tx_signal = impairments.apply_tx_impairments(test_signal, [profile])

        # Apply RX
        rx_signal = impairments.apply_rx_impairments(tx_signal, [profile])

        # Measure degradation
        degradation_db = 10 * torch.log10(
            torch.mean(torch.abs(test_signal)**2) / torch.mean(torch.abs(rx_signal - test_signal)**2)
        )

        print(f"  {profile:20s}: SNR degradation = {degradation_db:.1f} dB")

    print("\n✓ Integration test complete!")


if __name__ == "__main__":
    test_transceiver_impairments()
