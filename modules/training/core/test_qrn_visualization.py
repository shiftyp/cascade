#!/usr/bin/env python3
"""
Test QRN generation and create visualization to verify realism.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from gpu_qrn_generator import GPUQRNGenerator

# Generate a CASCADE signal
print("Generating CASCADE signal...")
gen = GPUSignalGenerator(device='cuda')

batch_params = BatchKernelParameters(
    pattern_ids=torch.tensor([0], dtype=torch.long, device='cuda'),
    frequency_triples=torch.tensor([21], dtype=torch.long, device='cuda'),
    modulations=['QPSK'],
    polar_rates=[(2, 3)],
    data_symbol_rates=torch.tensor([150], dtype=torch.long, device='cuda')
)

# Generate reasonable-sized message (200 bytes = ~1.5 second transmission)
medium_message = b"This is a test message for CASCADE protocol QRN visualization validation. " * 3
messages = [medium_message[:200]]  # Limit to 200 bytes
signals, metadata = gen.generate_batch(batch_params, messages, fixed_length=480000, num_centers=0)

# Signal should have power = 1.0
signal_power = torch.mean(torch.abs(signals)**2).item()
print(f"Signal power: {signal_power:.4f}")

# Generate realistic QRN
print("Generating realistic QRN...")
qrn_gen = GPUQRNGenerator(sample_rate=48000, device='cuda')

# Thermal noise (baseline)
thermal = (torch.randn(1, 480000, device='cuda') + 1j * torch.randn(1, 480000, device='cuda')) / np.sqrt(2)

# Galactic noise (10 dB above thermal)
galactic = qrn_gen.generate_galactic_noise_batch(1, 480000, noise_level=1.0)
galactic_power_ratio = 10.0  # 10 dB

# Atmospheric QRN (lightning bursts)
k_index = torch.tensor([5.0], device='cuda')
atmospheric = qrn_gen.generate_atmospheric_qrn_batch(1, 480000, burst_rate=0.5, k_index_batch=k_index)

# Impulsive QRN (powerline)
impulsive = qrn_gen.generate_impulsive_qrn_batch(1, 480000, powerline_freq=60.0, strength=0.3)

# Combine noise components
total_noise = thermal + galactic * np.sqrt(galactic_power_ratio) + atmospheric + impulsive

# Scale noise to achieve 10 dB SNR
target_snr_db = 10.0
noise_power_target = signal_power / (10 ** (target_snr_db / 10))
noise_power_actual = torch.mean(torch.abs(total_noise)**2).item()
noise_scale = np.sqrt(noise_power_target / noise_power_actual)
scaled_noise = total_noise * noise_scale

print(f"Noise power: {torch.mean(torch.abs(scaled_noise)**2).item():.6f}")
print(f"Target SNR: {target_snr_db:.1f} dB")

# Add signal + noise
received_signal = signals + scaled_noise
received_cpu = received_signal[0].cpu().numpy()

# Create spectrogram
print("Creating visualization...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Spectrogram of received signal
from scipy import signal as scipy_signal
f, t, Sxx = scipy_signal.spectrogram(
    received_cpu, fs=48000, nperseg=256, noverlap=128, mode='magnitude'
)

im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_xlabel('Time (s)')
axes[0].set_title('Received Signal (Signal + Realistic QRN)')
axes[0].set_ylim([0, 3000])
plt.colorbar(im1, ax=axes[0], label='Power (dB)')

# Spectrogram of noise only
noise_cpu = scaled_noise[0].cpu().numpy()
f, t, Sxx_noise = scipy_signal.spectrogram(
    noise_cpu, fs=48000, nperseg=256, noverlap=128, mode='magnitude'
)

im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_noise + 1e-10), shading='gouraud', cmap='viridis')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('QRN Only (Thermal + Galactic + Atmospheric + Impulsive)')
axes[1].set_ylim([0, 3000])
plt.colorbar(im2, ax=axes[1], label='Power (dB)')

# Spectrogram of clean signal
signal_cpu = signals[0].cpu().numpy()
f, t, Sxx_signal = scipy_signal.spectrogram(
    signal_cpu, fs=48000, nperseg=256, noverlap=128, mode='magnitude'
)

im3 = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx_signal + 1e-10), shading='gouraud', cmap='viridis')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Clean CASCADE Signal (No Noise)')
axes[2].set_ylim([0, 3000])
plt.colorbar(im3, ax=axes[2], label='Power (dB)')

plt.tight_layout()

output_path = Path('/tmp/qrn_test_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to {output_path}")
print("  Check the spectrogram - you should see:")
print("  - Top: CASCADE signal visible at ~1560-1600 Hz with realistic noise")
print("  - Middle: Frequency-shaped galactic noise + atmospheric bursts + impulsive spikes")
print("  - Bottom: Clean CASCADE 3-FSK signal clearly visible")
