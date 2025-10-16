#!/usr/bin/env python3
"""Test and visualize SNR levels in CASCADE signal generation."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signal_generator.generator import SignalGenerator
from core.gpu_channel_simulator import GPUChannelSimulator, PropagationMode, MultipathProfile
from dataclasses import dataclass

@dataclass
class TestScenario:
    """Test scenario for channel simulation."""
    propagation_mode: str
    k_index: float = 3
    sfi: float = 100
    qrn_type: str = 'QUIET'

def generate_test_signal():
    """Generate a clean test signal."""
    gen = SignalGenerator()

    # Generate clean signal
    signal, metadata = gen.generate(
        pattern_id=0,
        frequency_triple=21,  # Middle of band
        modulation_scheme='QPSK',
        polar_rate=(2, 3),
        data_symbol_rate=150,
        message=b"Test CASCADE signal",
        seed=42
    )

    return signal.iq_samples, metadata

def test_snr_levels():
    """Test different SNR levels and visualize."""

    # Generate clean signal
    print("Generating clean signal...")
    clean_signal, metadata = generate_test_signal()

    # Convert to GPU tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_tensor = torch.from_numpy(clean_signal).unsqueeze(0).to(device)

    # Create channel simulator
    sim = GPUChannelSimulator(sample_rate=48000, device=device)

    # Test different propagation modes
    modes = ['AWGN', 'RAYLEIGH', 'MULTIPATH_DENSE']

    fig, axes = plt.subplots(len(modes), 3, figsize=(15, 12))
    fig.suptitle('CASCADE Signal Quality at Different SNR Levels', fontsize=16)

    for mode_idx, mode in enumerate(modes):
        print(f"\nTesting {mode} propagation...")

        # Create scenario
        scenario = TestScenario(propagation_mode=mode)

        # Apply channel (this will add noise based on mode)
        noisy_signal = sim.apply_batch(
            clean_tensor,
            [scenario],
            add_collisions=False,
            add_qrm=False
        )

        # Move back to CPU for plotting
        noisy_np = noisy_signal[0].cpu().numpy()
        clean_np = clean_tensor[0].cpu().numpy()

        # Calculate actual SNR
        signal_power = np.mean(np.abs(clean_np)**2)
        noise_power = np.mean(np.abs(noisy_np - clean_np)**2)
        actual_snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Time domain plot
        ax_time = axes[mode_idx, 0]
        time_ms = np.arange(1000) / 48  # First 1000 samples in ms
        ax_time.plot(time_ms, np.real(clean_np[:1000]), 'b-', alpha=0.5, label='Clean')
        ax_time.plot(time_ms, np.real(noisy_np[:1000]), 'r-', alpha=0.5, label='Noisy')
        ax_time.set_title(f'{mode} - Time Domain (SNR: {actual_snr_db:.1f} dB)')
        ax_time.set_xlabel('Time (ms)')
        ax_time.set_ylabel('Amplitude')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)

        # Spectrum plot
        ax_spec = axes[mode_idx, 1]
        fft_clean = np.fft.fft(clean_np[:4096])
        fft_noisy = np.fft.fft(noisy_np[:4096])
        freqs = np.fft.fftfreq(4096, 1/48000)[:2048]

        ax_spec.plot(freqs, 20*np.log10(np.abs(fft_clean[:2048])+1e-10), 'b-', alpha=0.5, label='Clean')
        ax_spec.plot(freqs, 20*np.log10(np.abs(fft_noisy[:2048])+1e-10), 'r-', alpha=0.5, label='Noisy')
        ax_spec.set_title(f'{mode} - Spectrum')
        ax_spec.set_xlabel('Frequency (Hz)')
        ax_spec.set_ylabel('Magnitude (dB)')
        ax_spec.set_xlim(0, 3000)
        ax_spec.legend()
        ax_spec.grid(True, alpha=0.3)

        # Constellation plot
        ax_const = axes[mode_idx, 2]
        # Sample constellation points from middle of signal
        mid_point = len(noisy_np) // 2
        constellation_points = noisy_np[mid_point:mid_point+500:10]  # Every 10th sample

        ax_const.scatter(np.real(constellation_points), np.imag(constellation_points),
                        alpha=0.5, s=10, c='red', label='Received')
        ax_const.set_title(f'{mode} - IQ Constellation')
        ax_const.set_xlabel('In-phase')
        ax_const.set_ylabel('Quadrature')
        ax_const.set_xlim(-3, 3)
        ax_const.set_ylim(-3, 3)
        ax_const.grid(True, alpha=0.3)
        ax_const.set_aspect('equal')

        # Print statistics
        print(f"  Mode: {mode}")
        print(f"  Target SNR range: {get_snr_range(mode)}")
        print(f"  Actual SNR: {actual_snr_db:.1f} dB")
        print(f"  Signal power: {signal_power:.6f}")
        print(f"  Noise power: {noise_power:.6f}")

    plt.tight_layout()
    plt.savefig('/home/ubuntu/cascade-rf/cascade/modules/training/core/snr_analysis.png', dpi=150)
    print(f"\nVisualization saved to snr_analysis.png")

def get_snr_range(mode):
    """Get the SNR range for a propagation mode."""
    if mode == 'AWGN':
        return "20-30 dB"
    elif mode == 'RAYLEIGH':
        return "5-15 dB"
    elif mode == 'MULTIPATH_DENSE':
        return "0-10 dB (floor: -10 dB)"
    elif mode == 'MULTIPATH_SPARSE':
        return "8-18 dB"
    elif mode == 'RICIAN':
        return "10-20 dB"
    else:
        return "10-20 dB"

if __name__ == "__main__":
    test_snr_levels()