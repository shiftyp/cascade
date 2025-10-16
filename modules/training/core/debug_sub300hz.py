#!/usr/bin/env python3
"""
Comprehensive debugging of sub-300 Hz issue.
Traces through the entire signal generation pipeline step-by-step.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from gpu_channel_simulator import GPUChannelSimulator
from gpu_qrn_generator import GPUQRNGenerator

def measure_sub300(signal, sample_rate, label):
    """Measure sub-300 Hz power and print results."""
    if signal is None or len(signal) == 0:
        print(f"  {label}: EMPTY/NONE")
        return 0.0

    signal_np = signal.cpu().numpy() if torch.is_tensor(signal) else signal

    # FFT
    fft = np.fft.fft(signal_np)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    power = np.abs(fft) ** 2

    # Sub-300 Hz power
    sub_300_mask = np.abs(freqs) < 300
    sub_300_power = np.sum(power[sub_300_mask])
    total_power = np.sum(power)
    ratio = (sub_300_power / total_power * 100) if total_power > 0 else 0.0

    # Find where the energy actually is
    peak_threshold = np.max(power) * 0.01
    peak_indices = np.where(power > peak_threshold)[0]
    peak_freqs = sorted(set(np.round(np.abs(freqs[peak_indices]) / 50) * 50))[:10]

    status = "✓" if ratio < 1.0 else "⚠️"
    print(f"  {status} {label:40s}: {ratio:6.3f}% sub-300Hz | Peaks: {[int(f) for f in peak_freqs if f < 1500]}")

    return ratio

def debug_streaming_dataset_single():
    """Debug a single stream generation step-by-step."""

    print("="*80)
    print("DEBUGGING SINGLE STREAM GENERATION (STEP-BY-STEP)")
    print("="*80)

    # Initialize generators
    sig_gen = GPUSignalGenerator(device='cuda', use_always_on=True)
    qrn_gen = GPUQRNGenerator(sample_rate=48000, device='cuda')
    channel_sim = GPUChannelSimulator(sample_rate=48000, device='cuda',
                                      enable_transceiver_impairments=True)

    num_samples = 96000  # 2 seconds

    # Step 1: Generate clean CASCADE signal
    print("\nSTEP 1: Generate clean CASCADE signal at frequency triple 0")
    print("-" * 80)

    batch_params = BatchKernelParameters(
        pattern_ids=torch.tensor([0], device='cuda'),
        frequency_triples=torch.tensor([0], device='cuda'),
        modulations=['QPSK'],
        polar_rates=[(2, 3)],
        data_symbol_rates=torch.tensor([150], device='cuda')
    )

    clean_signal, _ = sig_gen.generate_batch(
        batch_params, [b"TEST"], fixed_length=num_samples, num_centers=4
    )

    ratio_clean = measure_sub300(clean_signal[0], 48000, "Clean CASCADE signal")

    # Step 2: Generate QRN components separately
    print("\nSTEP 2: Generate QRN components")
    print("-" * 80)

    thermal = (torch.randn(1, num_samples, device='cuda') +
               1j * torch.randn(1, num_samples, device='cuda')) / np.sqrt(2)
    measure_sub300(thermal[0], 48000, "Thermal noise (white)")

    galactic = qrn_gen.generate_galactic_noise_batch(1, num_samples, noise_level=1.0)
    measure_sub300(galactic[0], 48000, "Galactic noise (after HPF)")

    atmospheric = qrn_gen.generate_atmospheric_qrn_batch(
        1, num_samples, burst_rate=2.0,
        k_index_batch=torch.tensor([5.0], device='cuda')
    )
    measure_sub300(atmospheric[0], 48000, "Atmospheric QRN (after HPF)")

    impulsive = qrn_gen.generate_impulsive_qrn_batch(1, num_samples, strength=0.3)
    measure_sub300(impulsive[0], 48000, "Impulsive QRN (after HPF)")

    # Step 3: Combine noise components using quadrature
    print("\nSTEP 3: Combine noise components (quadrature)")
    print("-" * 80)

    thermal_power = torch.abs(thermal) ** 2
    galactic_power = torch.abs(galactic) ** 2 * 2.0  # 3 dB above thermal
    atmospheric_power = torch.abs(atmospheric) ** 2
    impulsive_power = torch.abs(impulsive) ** 2

    total_power = thermal_power + galactic_power + atmospheric_power + impulsive_power
    phase = torch.angle(thermal)
    combined_noise = torch.sqrt(total_power + 1e-10) * torch.exp(1j * phase)

    measure_sub300(combined_noise[0], 48000, "Combined noise (quadrature)")

    # Step 4: Add noise to signal
    print("\nSTEP 4: Add noise to signal (15 dB SNR)")
    print("-" * 80)

    signal_power = torch.mean(torch.abs(clean_signal) ** 2)
    target_snr_db = 15.0
    noise_power = signal_power / (10 ** (target_snr_db / 10))

    scaled_noise = combined_noise * torch.sqrt(noise_power)
    signal_with_noise = clean_signal + scaled_noise

    measure_sub300(signal_with_noise[0], 48000, "Signal + noise")

    # Step 5: Apply TX impairments
    print("\nSTEP 5: Apply TX impairments")
    print("-" * 80)

    tx_signal, tx_prof, _ = channel_sim.apply_tx_rx_impairments_batch(
        signal_with_noise, apply_tx=True, apply_rx=False
    )
    measure_sub300(tx_signal[0], 48000, f"After TX impairments ({tx_prof[0]})")

    # Step 6: Apply RX impairments
    print("\nSTEP 6: Apply RX impairments")
    print("-" * 80)

    rx_signal = channel_sim.apply_rx_impairments_only(tx_signal, tx_prof)
    measure_sub300(rx_signal[0], 48000, f"After RX impairments ({tx_prof[0]})")

    # Step 7: Check thermal noise alone
    print("\nSTEP 7: Test thermal noise (should be uniform across frequencies)")
    print("-" * 80)

    # Generate new thermal noise
    thermal_test = (torch.randn(1, num_samples, device='cuda') +
                    1j * torch.randn(1, num_samples, device='cuda')) / np.sqrt(2)
    measure_sub300(thermal_test[0], 48000, "Fresh thermal noise")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("Check which step introduced significant sub-300 Hz content (>1%)")
    print("="*80)

if __name__ == "__main__":
    debug_streaming_dataset_single()
