#!/usr/bin/env python3
"""
Test complete stream generation with realistic signals and QRN.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters
from gpu_qrn_generator import GPUQRNGenerator

print("=" * 70)
print("TESTING COMPLETE STREAM GENERATION")
print("=" * 70)

# Initialize generators
gen = GPUSignalGenerator(device='cuda')
qrn_gen = GPUQRNGenerator(sample_rate=48000, device='cuda')

# Generate 3 CASCADE signals at different frequencies with always-on design
print("\n1. Generating 3 CASCADE signals...")
batch_params = BatchKernelParameters(
    pattern_ids=torch.tensor([0, 1, 2], dtype=torch.long, device='cuda'),
    frequency_triples=torch.tensor([15, 21, 28], dtype=torch.long, device='cuda'),  # Different freqs
    modulations=['QPSK', 'BPSK', 'QPSK'],
    polar_rates=[(2, 3), (1, 2), (2, 3)],
    data_symbol_rates=torch.tensor([150, 100, 175], dtype=torch.long, device='cuda')
)

messages = [
    b"CQ CQ CQ DE W1ABC EM42",
    b"W1ABC DE K2XYZ FN31 +10",
    b"K2XYZ DE W1ABC R-05"
]

signals, metadata = gen.generate_batch(batch_params, messages, fixed_length=None, num_centers=4)

print(f"Generated {signals.shape[0]} signals:")
for i in range(3):
    power = torch.mean(torch.abs(signals[i])**2).item()
    print(f"  Signal {i}: power={power:.4f}, len={signals.shape[1]} samples, {metadata[i]['modulation']}")

# Create 10-second stream
stream_samples = 480000
stream = torch.zeros(stream_samples, dtype=torch.complex64, device='cuda')

# Place signals at different times
print("\n2. Placing signals in 10-second stream...")
start_times = [48000, 192000, 336000]  # 1s, 4s, 7s
for i in range(3):
    sig = signals[i]
    start = start_times[i]
    sig_len = min(len(sig), stream_samples - start)
    if sig_len > 0:
        # Signals already have unit power, add directly
        stream[start:start + sig_len] += sig[:sig_len]
        print(f"  Signal {i} at t={start/48000:.1f}s, duration={sig_len/48000:.2f}s")

stream_power_signals_only = torch.mean(torch.abs(stream)**2).item()
print(f"Stream power (signals only): {stream_power_signals_only:.4f}")

# Add realistic QRN
print("\n3. Adding realistic QRN...")
k_index = torch.tensor([5.0], device='cuda')

# Thermal
thermal = (torch.randn(stream_samples, device='cuda') +
           1j * torch.randn(stream_samples, device='cuda')) / np.sqrt(2)

# Galactic (3 dB above thermal)
galactic = qrn_gen.generate_galactic_noise_batch(1, stream_samples, noise_level=1.0)[0]

# Atmospheric (lightning bursts)
atmospheric = qrn_gen.generate_atmospheric_qrn_batch(1, stream_samples, burst_rate=0.5, k_index_batch=k_index)[0]

# Impulsive (powerline)
impulsive = qrn_gen.generate_impulsive_qrn_batch(1, stream_samples, powerline_freq=60.0, strength=0.3)[0]

# Combine noise
total_noise = thermal + galactic * np.sqrt(2.0) + atmospheric + impulsive

# Scale noise for target SNR = 10 dB
target_snr_db = 10.0
noise_power_target = stream_power_signals_only / (10 ** (target_snr_db / 10))
noise_power_actual = torch.mean(torch.abs(total_noise)**2).item()
noise_scale = np.sqrt(noise_power_target / noise_power_actual)
scaled_noise = total_noise * noise_scale

print(f"Noise power: {torch.mean(torch.abs(scaled_noise)**2).item():.6f}")
print(f"Actual SNR: {10*np.log10(stream_power_signals_only / torch.mean(torch.abs(scaled_noise)**2).item()):.1f} dB")

# Final stream
final_stream = stream + scaled_noise
final_stream_cpu = final_stream.cpu().numpy()

# Create visualization
print("\n4. Creating visualization...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Top: Received signal (signal + noise)
f, t, Sxx = scipy_signal.spectrogram(final_stream_cpu, fs=48000, nperseg=512, noverlap=256, mode='magnitude')
im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis', vmin=-60, vmax=-20)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_xlabel('Time (s)')
axes[0].set_title('Received Signal (3 CASCADE messages + Realistic QRN) - SNR=10 dB')
axes[0].set_ylim([300, 2800])
axes[0].axhline(y=1560, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
axes[0].axhline(y=1580, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
axes[0].axhline(y=1600, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
plt.colorbar(im1, ax=axes[0], label='Power (dB)')

# Middle: Clean signals only
stream_clean_cpu = stream.cpu().numpy()
f, t, Sxx2 = scipy_signal.spectrogram(stream_clean_cpu, fs=48000, nperseg=512, noverlap=256, mode='magnitude')
im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx2 + 1e-10), shading='gouraud', cmap='viridis', vmin=-80, vmax=-30)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Clean CASCADE Signals (No Noise) - 3 messages at different times/frequencies')
axes[1].set_ylim([300, 2800])
plt.colorbar(im2, ax=axes[1], label='Power (dB)')

# Bottom: QRN only
noise_cpu = scaled_noise.cpu().numpy()
f, t, Sxx3 = scipy_signal.spectrogram(noise_cpu, fs=48000, nperseg=512, noverlap=256, mode='magnitude')
im3 = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx3 + 1e-10), shading='gouraud', cmap='viridis', vmin=-60, vmax=-20)
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('QRN Only (Thermal + Galactic + Atmospheric + Impulsive)')
axes[2].set_ylim([300, 2800])
plt.colorbar(im3, ax=axes[2], label='Power (dB)')

plt.tight_layout()
plt.savefig('/tmp/complete_stream_test.png', dpi=150)
print(f"\n✓ Visualization saved to /tmp/complete_stream_test.png")
print("\nExpected to see:")
print("  - Top: 3 CASCADE signals visible in noise at different times")
print("  - Middle: Clear 3 CASCADE signals (always-on centers + alternating outers)")
print("  - Bottom: Realistic QRN with atmospheric bursts + impulsive spikes")
