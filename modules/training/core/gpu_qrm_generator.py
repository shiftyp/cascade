"""
GPU-accelerated QRM (interference) generator for CASCADE.

Generates realistic interfering signals from "other stations" with:
- Random frequencies across the band
- Random patterns and modulations
- Realistic power levels (weaker than target signal)
- Poisson arrival times

All operations GPU-accelerated for maximum performance.
"""

import torch
import numpy as np
from typing import Optional, List
from gpu_signal_generator import GPUSignalGenerator, BatchKernelParameters


class GPUQRMGenerator:
    """
    Generate realistic QRM (interference from other stations) on GPU.

    QRM characteristics:
    - Multiple interfering CASCADE signals
    - Random frequencies (avoid target signal)
    - Lower power than target (-10 to -30 dB relative)
    - Realistic modulation and patterns
    """

    def __init__(self, signal_generator: GPUSignalGenerator, sample_rate: int = 48000):
        """
        Initialize QRM generator.

        Args:
            signal_generator: Existing GPU signal generator instance
            sample_rate: Audio sample rate
        """
        self.signal_generator = signal_generator
        self.sample_rate = sample_rate
        self.device = signal_generator.device

    def generate_qrm_batch(self,
                          num_streams: int,
                          stream_duration_sec: float,
                          target_frequencies: Optional[List[int]] = None,
                          qrm_power_db: float = -20.0,
                          num_interferers: int = 3) -> torch.Tensor:
        """
        Generate batch of QRM (interfering signals) for multiple streams.

        Args:
            num_streams: Number of streams to generate QRM for
            stream_duration_sec: Duration of each stream in seconds
            target_frequencies: List of frequency triples to avoid (target signals)
            qrm_power_db: QRM power relative to target signal (negative = weaker)
            num_interferers: Number of interfering stations per stream (1-5)

        Returns:
            torch.Tensor: [num_streams, num_samples] QRM interference signals
        """
        num_samples = int(stream_duration_sec * self.sample_rate)

        # Initialize QRM signals
        qrm_signals = torch.zeros(num_streams, num_samples, dtype=torch.complex64, device=self.device)

        # Generate interferers for all streams in parallel
        # Total interferers = num_streams × num_interferers
        total_interferers = num_streams * num_interferers

        # Random parameters for all interferers
        interferer_patterns = torch.randint(0, 4, (total_interferers,), device=self.device)

        # Random frequencies (avoid target if specified)
        if target_frequencies is not None and len(target_frequencies) > 0:
            # Avoid target frequency ±5 triples
            all_triples = list(range(14))  # Updated: 50 Hz spacing, 2-center design (500-2600 Hz)
            for target in target_frequencies:
                for offset in range(-5, 6):
                    triple = target + offset
                    if 0 <= triple < 14 and triple in all_triples:
                        all_triples.remove(triple)

            # Sample from remaining triples
            if len(all_triples) > 0:
                interferer_freqs = torch.tensor(
                    np.random.choice(all_triples, size=total_interferers, replace=True),
                    device=self.device
                )
            else:
                # Fallback if all avoided
                interferer_freqs = torch.randint(0, 14, (total_interferers,), device=self.device)
        else:
            interferer_freqs = torch.randint(0, 50, (total_interferers,), device=self.device)

        # Random modulations (weighted toward lower orders for weaker signals)
        mod_choices = ['BPSK', 'BPSK', 'QPSK', 'QPSK', '8-PSK']  # Weighted
        interferer_mods = np.random.choice(mod_choices, size=total_interferers).tolist()

        # Random data rates (typically lower for QRM)
        rate_choices = [75, 100, 125, 150]  # Lower rates for interferers
        interferer_rates = torch.tensor(
            np.random.choice(rate_choices, size=total_interferers),
            device=self.device
        )

        # Random message sizes (shorter for QRM)
        interferer_messages = [np.random.bytes(np.random.randint(5, 30)) for _ in range(total_interferers)]

        # Batch parameters for all interferers
        batch_params = BatchKernelParameters(
            pattern_ids=interferer_patterns,
            frequency_triples=interferer_freqs,
            modulations=interferer_mods,
            polar_rates=[(2, 3)] * total_interferers,
            data_symbol_rates=interferer_rates
        )

        # GENERATE ALL INTERFERERS AT ONCE ON GPU!
        interferer_signals, _ = self.signal_generator.generate_batch(
            batch_params,
            interferer_messages,
            fixed_length=None,
            num_centers=0  # QRM uses standard 3-FSK (not always-on)
        )

        # Distribute interferers to streams with random start times
        for stream_idx in range(num_streams):
            for interferer_idx in range(num_interferers):
                global_idx = stream_idx * num_interferers + interferer_idx
                signal = interferer_signals[global_idx]

                # Random start time (Poisson-like)
                max_start = max(0, num_samples - len(signal))
                start_sample = np.random.randint(0, max_start + 1) if max_start > 0 else 0

                # Add to stream with QRM power scaling
                qrm_scale = 10 ** (qrm_power_db / 20.0)  # dB to linear amplitude
                signal_len = min(len(signal), num_samples - start_sample)

                if signal_len > 0:
                    qrm_signals[stream_idx, start_sample:start_sample + signal_len] += qrm_scale * signal[:signal_len]

        return qrm_signals


def test_qrm_generator():
    """Test QRM generator."""
    print("Testing GPU QRM Generator...")

    # Create signal generator (auto-detects pattern locations)
    sig_gen = GPUSignalGenerator(device='cuda', use_gpu_polar=True, use_always_on=True)

    # Create QRM generator
    qrm_gen = GPUQRMGenerator(sig_gen)

    # Generate QRM for 10 streams
    qrm_signals = qrm_gen.generate_qrm_batch(
        num_streams=10,
        stream_duration_sec=10.0,
        target_frequencies=[20, 21, 22],  # Avoid these triples
        qrm_power_db=-20.0,  # 20 dB weaker than target
        num_interferers=3
    )

    print(f"✓ Generated QRM for {len(qrm_signals)} streams")
    print(f"  Shape: {qrm_signals.shape}")
    print(f"  Mean power: {torch.mean(torch.abs(qrm_signals)**2).item():.6f}")
    print(f"  Max amplitude: {torch.max(torch.abs(qrm_signals)).item():.3f}")


if __name__ == "__main__":
    test_qrm_generator()
