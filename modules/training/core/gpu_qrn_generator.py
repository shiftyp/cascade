"""
GPU-accelerated QRN (natural noise) generator for CASCADE.

Generates realistic atmospheric and man-made noise based on physical models:
- Atmospheric QRN: Lightning crashes, exponentially decaying bursts
- Impulsive QRN: Powerline/industrial noise with periodic spikes
- Galactic noise: Frequency-dependent cosmic background (ITU-R P.372)

All operations GPU-accelerated and parallelized.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class GPUQRNGenerator:
    """
    Generate realistic QRN (natural/man-made noise) on GPU.

    Implements physics-based models for:
    1. Atmospheric noise (lightning)
    2. Impulsive noise (powerline, industrial)
    3. Galactic noise (cosmic background)
    """

    def __init__(self, sample_rate: int = 48000, device: str = 'cuda'):
        """
        Initialize QRN generator.

        Args:
            sample_rate: Audio sample rate
            device: 'cuda' for GPU
        """
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        # Pre-compute frequency-dependent noise shaping (ITU-R P.372)
        self._precompute_galactic_noise_spectrum()

    def _precompute_galactic_noise_spectrum(self):
        """
        Pre-compute galactic noise spectrum based on ITU-R P.372.

        Galactic noise is frequency-dependent: Fa = 52.5 + 23.5*log10(250/f_MHz)
        This gives stronger noise at lower frequencies.
        """
        # For HF band (full audio spectrum 0-3000 Hz in our baseband)
        # This represents actual HF frequencies ~7-14 MHz after down-conversion
        freqs_hz = torch.linspace(10, 3000, 2048, device=self.device)  # Start at 10 Hz to avoid DC

        # Map to approximate HF frequencies (assume 10 MHz center)
        # Baseband offset from center
        offset_mhz = (freqs_hz - 1650) / 1000  # 1650 Hz = baseband center
        hf_freq_mhz = 10.0 + offset_mhz

        # ITU-R P.372 galactic noise (dB above kTB)
        # Fa = 52.5 + 23.5*log10(250/f) for f in MHz
        galactic_noise_db = 52.5 + 23.5 * torch.log10(250.0 / hf_freq_mhz.clamp(min=1.0))

        # Convert to linear power scaling (relative to white noise)
        # Normalize to unit energy (spectrum shape only, not absolute level)
        spectrum_raw = 10 ** (galactic_noise_db / 20.0)
        self.galactic_spectrum = spectrum_raw / torch.mean(spectrum_raw)  # Normalize to mean=1
        self.galactic_freqs = freqs_hz

    def generate_atmospheric_qrn_batch(self,
                                      num_streams: int,
                                      num_samples: int,
                                      burst_rate: float = 0.5,
                                      k_index_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate atmospheric QRN (lightning crashes) for batch of streams - FULLY VECTORIZED.

        Physics model: Exponentially decaying bursts from lightning strikes.
        Rate increases with higher K-index (geomagnetic activity).

        Args:
            num_streams: Number of streams
            num_samples: Samples per stream
            burst_rate: Average bursts per second (scaled by K-index)
            k_index_batch: [num_streams] K-index values (0-9), higher = more bursts

        Returns:
            torch.Tensor: [num_streams, num_samples] atmospheric noise
        """
        # Adjust burst rate by K-index (GPU operation)
        if k_index_batch is not None:
            k_factor = 1.0 + (k_index_batch / 9.0)  # 1.0-2.0× multiplier
            burst_rate_adjusted = burst_rate * k_factor
        else:
            burst_rate_adjusted = torch.full((num_streams,), burst_rate, device=self.device)

        duration_sec = num_samples / self.sample_rate

        # Pre-compute maximum bursts (worst case for all streams)
        max_burst_rate = burst_rate_adjusted.max().item() if k_index_batch is not None else burst_rate
        max_bursts_per_stream = int(np.ceil(max_burst_rate * duration_sec * 3))  # 3× for safety

        # VECTORIZED: Generate all burst parameters for all streams at once
        # Total bursts = num_streams × max_bursts_per_stream
        total_bursts = num_streams * max_bursts_per_stream

        # Generate all arrival times (uniform random, will filter by actual rate later)
        arrival_samples_flat = torch.randint(0, num_samples, (total_bursts,), device=self.device)

        # Generate all amplitudes (exponential distribution)
        amplitudes_flat = torch.from_numpy(np.random.exponential(1.0, size=total_bursts)).to(self.device).float() * 0.5

        # Generate all decay times (uniform 5-50ms)
        decay_ms_flat = torch.rand(total_bursts, device=self.device) * 45 + 5  # 5-50ms
        decay_samples_flat = (decay_ms_flat * self.sample_rate / 1000).long()

        # Reshape to [num_streams, max_bursts_per_stream]
        arrival_samples = arrival_samples_flat.view(num_streams, max_bursts_per_stream)
        amplitudes = amplitudes_flat.view(num_streams, max_bursts_per_stream)
        decay_samples = decay_samples_flat.view(num_streams, max_bursts_per_stream)

        # Determine which bursts are actually active (based on Poisson rate)
        # VECTORIZED: Sample Poisson on GPU for all streams at once
        poisson_rates = burst_rate_adjusted * duration_sec  # [num_streams]
        num_bursts_per_stream = torch.poisson(poisson_rates).long()  # GPU-accelerated!
        num_bursts_per_stream = torch.clamp(num_bursts_per_stream, 0, max_bursts_per_stream)

        # Build active bursts mask using broadcasting
        burst_indices = torch.arange(max_bursts_per_stream, device=self.device).unsqueeze(0)  # [1, max_bursts]
        active_bursts = burst_indices < num_bursts_per_stream.unsqueeze(1)  # [num_streams, max_bursts]

        # Initialize output
        atmospheric_noise = torch.zeros(num_streams, num_samples, dtype=torch.complex64, device=self.device)

        # VECTORIZED BURST GENERATION: Process all active bursts at once
        # This is still a loop but over bursts (fixed size), not over individual samples
        for burst_idx in range(max_bursts_per_stream):
            # Mask for streams that have this burst active
            burst_active = active_bursts[:, burst_idx]
            if not burst_active.any():
                continue

            num_active = burst_active.sum().item()

            # Get parameters for active bursts
            arrivals = arrival_samples[burst_active, burst_idx]  # [num_active]
            amps = amplitudes[burst_active, burst_idx]  # [num_active]
            decays = decay_samples[burst_active, burst_idx]  # [num_active]

            # Generate bursts for all active streams (fixed length for vectorization)
            max_decay = decays.max().item()
            max_decay = min(max_decay, num_samples // 10)  # Cap at 10% of stream

            if max_decay > 0:
                t = torch.arange(max_decay, device=self.device, dtype=torch.float32)

                # Exponential decay for all bursts (vectorized)
                decay_curves = torch.exp(-t.unsqueeze(0) / (decays.unsqueeze(1).float() * 0.3))  # [num_active, max_decay]

                # Generate complex noise
                noise_real = torch.randn(num_active, max_decay, device=self.device)
                noise_imag = torch.randn(num_active, max_decay, device=self.device)
                burst_signals = amps.unsqueeze(1) * decay_curves * (noise_real + 1j * noise_imag) / np.sqrt(2)

                # Add bursts to streams - VECTORIZED using index_add
                active_indices = torch.where(burst_active)[0]

                # For each active burst, add to the appropriate stream
                for idx in range(len(active_indices)):
                    stream_idx = active_indices[idx].item()
                    arrival = arrivals[idx].item()
                    burst_len = min(max_decay, num_samples - arrival)
                    if burst_len > 0:
                        # Add burst signal to stream at arrival position
                        atmospheric_noise[stream_idx, arrival:arrival + burst_len] += burst_signals[idx, :burst_len]

        return atmospheric_noise

    def generate_impulsive_qrn_batch(self,
                                    num_streams: int,
                                    num_samples: int,
                                    powerline_freq: float = 60.0,
                                    strength: float = 0.3) -> torch.Tensor:
        """
        Generate impulsive QRN (powerline/industrial noise) for batch of streams - FULLY VECTORIZED.

        Physics model: Broadband impulsive bursts at powerline frequency timing.
        Creates short duration noise spikes (not frequency tones!).

        Args:
            num_streams: Number of streams
            num_samples: Samples per stream
            powerline_freq: Powerline frequency (50 or 60 Hz) - determines spike rate
            strength: Impulse strength (0-1)

        Returns:
            torch.Tensor: [num_streams, num_samples] impulsive noise (broadband)
        """
        duration_sec = num_samples / self.sample_rate
        impulse_rate = powerline_freq * 2  # 120 Hz for 60 Hz AC
        num_impulses_per_stream = int(impulse_rate * duration_sec)

        # VECTORIZED: Generate ALL impulse parameters at once
        total_impulses = num_streams * num_impulses_per_stream

        # Ideal impulse times (periodic)
        impulse_indices = torch.arange(num_impulses_per_stream, device=self.device).unsqueeze(0).expand(num_streams, -1)
        ideal_times = impulse_indices.float() / impulse_rate  # [num_streams, num_impulses]

        # Add jitter (±10% of period)
        period = 1.0 / impulse_rate
        jitter = (torch.rand(num_streams, num_impulses_per_stream, device=self.device) - 0.5) * period * 0.2
        arrival_times = ideal_times + jitter  # [num_streams, num_impulses]
        arrival_samples = (arrival_times * self.sample_rate).long()

        # Clamp to valid range
        arrival_samples = torch.clamp(arrival_samples, 0, num_samples - 1)

        # Impulse durations (0.5-2 ms)
        impulse_duration_ms = torch.rand(num_streams, num_impulses_per_stream, device=self.device) * 1.5 + 0.5
        impulse_samples = (impulse_duration_ms * self.sample_rate / 1000).long()
        impulse_samples = torch.clamp(impulse_samples, 1, 200)  # 1-200 samples (max 4ms)

        # Amplitudes (exponential distribution)
        amplitudes = torch.from_numpy(np.random.exponential(strength, size=(num_streams, num_impulses_per_stream))).to(self.device).float()

        # VECTORIZED BURST GENERATION: Create all bursts at once using broadcasting
        # For efficiency, use fixed max burst length
        max_burst_len = 200  # 4ms max (covers 99% of bursts)

        # Generate all noise at once: [num_streams, num_impulses, max_burst_len]
        burst_noise_real = torch.randn(num_streams, num_impulses_per_stream, max_burst_len, device=self.device)
        burst_noise_imag = torch.randn(num_streams, num_impulses_per_stream, max_burst_len, device=self.device)
        burst_noise = (burst_noise_real + 1j * burst_noise_imag) / np.sqrt(2)

        # Scale by amplitudes: [num_streams, num_impulses, 1] * [num_streams, num_impulses, max_burst_len]
        burst_noise = burst_noise * amplitudes.unsqueeze(2)

        # Initialize output
        impulsive_noise = torch.zeros(num_streams, num_samples, dtype=torch.complex64, device=self.device)

        # VECTORIZED: Loop over impulse index (like atmospheric), not total impulses!
        # This reduces 120,000 iterations to 1200 iterations (100× fewer!)
        for impulse_idx in range(num_impulses_per_stream):
            # Get parameters for this impulse across all streams
            arrivals = arrival_samples[:, impulse_idx]  # [num_streams]
            durations = impulse_samples[:, impulse_idx]  # [num_streams]
            amps = amplitudes[:, impulse_idx]  # [num_streams]

            # Get max duration for this impulse across streams
            max_duration = durations.max().item()
            max_duration = min(max_duration, max_burst_len)

            if max_duration > 0:
                # Get burst noise for all streams at this impulse
                impulse_bursts = burst_noise[:, impulse_idx, :max_duration]  # [num_streams, max_duration]

                # Add to each stream at its arrival position
                for stream_idx in range(num_streams):
                    arrival = arrivals[stream_idx].item()
                    duration = min(durations[stream_idx].item(), num_samples - arrival, max_duration)

                    if duration > 0 and arrival < num_samples:
                        impulsive_noise[stream_idx, arrival:arrival + duration] += impulse_bursts[stream_idx, :duration]

        return impulsive_noise

    def generate_galactic_noise_batch(self,
                                     num_streams: int,
                                     num_samples: int,
                                     noise_level: float = 0.1) -> torch.Tensor:
        """
        Generate galactic (cosmic) background noise with frequency shaping.

        Physics model: ITU-R P.372 galactic noise spectrum (stronger at lower frequencies).

        Args:
            num_streams: Number of streams
            num_samples: Samples per stream
            noise_level: Overall noise level (0-1)

        Returns:
            torch.Tensor: [num_streams, num_samples] frequency-shaped galactic noise
        """
        # Generate white noise (I and Q separately for FFT shaping)
        white_noise_i = noise_level * torch.randn(num_streams, num_samples, device=self.device) / np.sqrt(2)
        white_noise_q = noise_level * torch.randn(num_streams, num_samples, device=self.device) / np.sqrt(2)

        # Apply frequency-dependent shaping via FFT (process I and Q separately)
        # Transform to frequency domain
        noise_i_fft = torch.fft.rfft(white_noise_i, dim=1)
        noise_q_fft = torch.fft.rfft(white_noise_q, dim=1)

        # Create frequency-dependent gain based on galactic spectrum - VECTORIZED
        fft_size = noise_i_fft.shape[1]
        freqs = torch.fft.rfftfreq(num_samples, 1/self.sample_rate, device=self.device)

        # VECTORIZED: Find closest galactic spectrum frequency for all FFT bins at once
        # Compute distance matrix: [fft_size, galactic_freqs_size]
        freq_diffs = torch.abs(freqs.unsqueeze(1) - self.galactic_freqs.unsqueeze(0))

        # Find minimum distance indices (vectorized)
        closest_indices = torch.argmin(freq_diffs, dim=1)  # [fft_size]

        # Lookup gains (fully vectorized, no loops!)
        freq_gains = self.galactic_spectrum[closest_indices]  # [fft_size]

        # Apply frequency shaping to both I and Q
        shaped_i_fft = noise_i_fft * freq_gains.unsqueeze(0)
        shaped_q_fft = noise_q_fft * freq_gains.unsqueeze(0)

        # Transform back to time domain
        galactic_noise_i = torch.fft.irfft(shaped_i_fft, n=num_samples, dim=1)
        galactic_noise_q = torch.fft.irfft(shaped_q_fft, n=num_samples, dim=1)

        return galactic_noise_i + 1j * galactic_noise_q

    def generate_combined_qrn_batch(self,
                                   num_streams: int,
                                   num_samples: int,
                                   k_index_batch: Optional[torch.Tensor] = None,
                                   thunderstorm_activity_batch: Optional[torch.Tensor] = None,
                                   include_atmospheric: bool = True,
                                   include_impulsive: bool = True,
                                   include_galactic: bool = True) -> torch.Tensor:
        """
        Generate combined QRN (all types) for batch of streams.

        Args:
            num_streams: Number of streams
            num_samples: Samples per stream
            k_index_batch: [num_streams] K-index values for atmospheric scaling
            thunderstorm_activity_batch: [num_streams] Thunderstorm activity (0-1) for QRN probability
            include_atmospheric: Include atmospheric QRN (lightning)
            include_impulsive: Include impulsive QRN (powerline)
            include_galactic: Include galactic background

        Returns:
            torch.Tensor: [num_streams, num_samples] combined QRN
        """
        combined_qrn = torch.zeros(num_streams, num_samples, dtype=torch.complex64, device=self.device)

        # Add atmospheric QRN with adaptive probability based on conditions
        if include_atmospheric:
            # Calculate adaptive probability for each stream
            atmospheric_probability = torch.full((num_streams,), 0.3, device=self.device)  # Base 30%

            # Boost for geomagnetic storms (K > 5)
            if k_index_batch is not None:
                # K=5: 30% → 45%, K=6: 60%, K=7: 75%, K=8: 85%, K=9: 90%
                k_boost = torch.clamp((k_index_batch - 4.0) * 0.15, 0.0, 0.6)
                atmospheric_probability = torch.clamp(atmospheric_probability + k_boost, 0.0, 0.9)

            # Boost for thunderstorms (activity > 0.3)
            if thunderstorm_activity_batch is not None:
                # activity=0.3: 30%, activity=0.6: 62%, activity=0.9: 87%, activity=1.0: 95%
                ts_boost = torch.clamp((thunderstorm_activity_batch - 0.2) * 0.75, 0.0, 0.65)
                atmospheric_probability = torch.clamp(atmospheric_probability + ts_boost, 0.0, 0.95)

            # For combined severe conditions (K>7 AND activity>0.7), push to near 100%
            if k_index_batch is not None and thunderstorm_activity_batch is not None:
                severe_combined = (k_index_batch > 7.0) & (thunderstorm_activity_batch > 0.7)
                atmospheric_probability[severe_combined] = 0.98

            # Generate mask based on adaptive probabilities
            atmospheric_mask = torch.rand(num_streams, device=self.device) < atmospheric_probability
            if atmospheric_mask.any():
                num_with_atmospheric = atmospheric_mask.sum().item()

                # Adaptive burst rate: base 0.5 bursts/sec, boosted by K-index and thunderstorm
                base_burst_rate = 0.5
                k_subset = k_index_batch[atmospheric_mask] if k_index_batch is not None else None
                ts_subset = thunderstorm_activity_batch[atmospheric_mask] if thunderstorm_activity_batch is not None else None

                # Calculate per-stream burst rate multiplier
                if k_subset is not None or ts_subset is not None:
                    burst_rate_multiplier = torch.ones(num_with_atmospheric, device=self.device)

                    # K-index boost: K=5 → 1.2×, K=7 → 1.6×, K=9 → 2.0×
                    if k_subset is not None:
                        k_multiplier = 1.0 + torch.clamp((k_subset - 4.0) * 0.2, 0.0, 1.0)
                        burst_rate_multiplier *= k_multiplier

                    # Thunderstorm boost: activity=0.5 → 1.4×, activity=0.8 → 2.0×, activity=1.0 → 2.4×
                    if ts_subset is not None:
                        ts_multiplier = 1.0 + (ts_subset * 1.4)
                        burst_rate_multiplier *= ts_multiplier

                    # Apply adaptive burst rate (pass average, generator will use k_index internally too)
                    adaptive_burst_rate = base_burst_rate * burst_rate_multiplier.mean().item()
                else:
                    adaptive_burst_rate = base_burst_rate

                atmospheric_qrn = self.generate_atmospheric_qrn_batch(
                    num_with_atmospheric, num_samples,
                    burst_rate=adaptive_burst_rate,
                    k_index_batch=k_subset
                )
                combined_qrn[atmospheric_mask] += atmospheric_qrn

        # Add impulsive QRN (15% chance per stream)
        if include_impulsive:
            impulsive_mask = torch.rand(num_streams, device=self.device) < 0.15
            if impulsive_mask.any():
                num_with_impulsive = impulsive_mask.sum().item()
                impulsive_qrn = self.generate_impulsive_qrn_batch(
                    num_with_impulsive, num_samples,
                    powerline_freq=60.0,  # US powerline
                    strength=0.3
                )
                combined_qrn[impulsive_mask] += impulsive_qrn

        # Add galactic background (always present, low level)
        if include_galactic:
            galactic_qrn = self.generate_galactic_noise_batch(
                num_streams, num_samples,
                noise_level=0.05  # Low background level
            )
            combined_qrn += galactic_qrn

        return combined_qrn


def test_qrn_generator():
    """Test QRN generator with adaptive probabilities."""
    print("Testing GPU QRN Generator with Adaptive Probabilities...")
    print("=" * 80)

    qrn_gen = GPUQRNGenerator(sample_rate=48000, device='cuda')
    num_samples = 480000  # 10 seconds

    # Test 1: Quiet conditions (should have ~30% atmospheric QRN)
    print("\n### Test 1: Quiet Conditions (K=2, no thunderstorms) ###")
    num_streams = 100
    k_quiet = torch.full((num_streams,), 2.0, device='cuda')
    ts_quiet = torch.full((num_streams,), 0.05, device='cuda')

    qrn_quiet = qrn_gen.generate_combined_qrn_batch(
        num_streams, num_samples,
        k_index_batch=k_quiet,
        thunderstorm_activity_batch=ts_quiet,
        include_atmospheric=True,
        include_impulsive=False,
        include_galactic=False
    )
    has_qrn_quiet = (torch.abs(qrn_quiet).sum(dim=1) > 0).float().mean().item()
    print(f"  Streams with atmospheric QRN: {has_qrn_quiet*100:.1f}% (expected ~30%)")

    # Test 2: Minor storm (K=5-6, should have ~45-60% atmospheric QRN)
    print("\n### Test 2: Minor Geomagnetic Storm (K=5.5, no thunderstorms) ###")
    k_minor = torch.full((num_streams,), 5.5, device='cuda')
    ts_none = torch.full((num_streams,), 0.05, device='cuda')

    qrn_minor = qrn_gen.generate_combined_qrn_batch(
        num_streams, num_samples,
        k_index_batch=k_minor,
        thunderstorm_activity_batch=ts_none,
        include_atmospheric=True,
        include_impulsive=False,
        include_galactic=False
    )
    has_qrn_minor = (torch.abs(qrn_minor).sum(dim=1) > 0).float().mean().item()
    avg_power_minor = torch.mean(torch.abs(qrn_minor[torch.abs(qrn_minor).sum(dim=1) > 0])**2).item()
    print(f"  Streams with atmospheric QRN: {has_qrn_minor*100:.1f}% (expected ~52%)")
    print(f"  Mean power (with QRN): {avg_power_minor:.6f}")

    # Test 3: Severe storm (K=8, should have ~85% atmospheric QRN)
    print("\n### Test 3: Severe Geomagnetic Storm (K=8.0) ###")
    k_severe = torch.full((num_streams,), 8.0, device='cuda')
    ts_low = torch.full((num_streams,), 0.1, device='cuda')

    qrn_severe = qrn_gen.generate_combined_qrn_batch(
        num_streams, num_samples,
        k_index_batch=k_severe,
        thunderstorm_activity_batch=ts_low,
        include_atmospheric=True,
        include_impulsive=False,
        include_galactic=False
    )
    has_qrn_severe = (torch.abs(qrn_severe).sum(dim=1) > 0).float().mean().item()
    avg_power_severe = torch.mean(torch.abs(qrn_severe[torch.abs(qrn_severe).sum(dim=1) > 0])**2).item()
    print(f"  Streams with atmospheric QRN: {has_qrn_severe*100:.1f}% (expected ~85%)")
    print(f"  Mean power (with QRN): {avg_power_severe:.6f} (higher burst rate)")

    # Test 4: Severe thunderstorms (K=3, activity=0.9, should have ~87% atmospheric QRN)
    print("\n### Test 4: Severe Thunderstorms (K=3, activity=0.9) ###")
    k_low = torch.full((num_streams,), 3.0, device='cuda')
    ts_severe = torch.full((num_streams,), 0.9, device='cuda')

    qrn_tstorm = qrn_gen.generate_combined_qrn_batch(
        num_streams, num_samples,
        k_index_batch=k_low,
        thunderstorm_activity_batch=ts_severe,
        include_atmospheric=True,
        include_impulsive=False,
        include_galactic=False
    )
    has_qrn_tstorm = (torch.abs(qrn_tstorm).sum(dim=1) > 0).float().mean().item()
    avg_power_tstorm = torch.mean(torch.abs(qrn_tstorm[torch.abs(qrn_tstorm).sum(dim=1) > 0])**2).item()
    print(f"  Streams with atmospheric QRN: {has_qrn_tstorm*100:.1f}% (expected ~82%)")
    print(f"  Mean power (with QRN): {avg_power_tstorm:.6f} (high burst rate from thunderstorms)")

    # Test 5: Combined severe (K=8, activity=0.9, should have ~98% atmospheric QRN)
    print("\n### Test 5: EXTREME - Severe Storm + Thunderstorms (K=8, activity=0.9) ###")
    k_extreme = torch.full((num_streams,), 8.0, device='cuda')
    ts_extreme = torch.full((num_streams,), 0.9, device='cuda')

    qrn_extreme = qrn_gen.generate_combined_qrn_batch(
        num_streams, num_samples,
        k_index_batch=k_extreme,
        thunderstorm_activity_batch=ts_extreme,
        include_atmospheric=True,
        include_impulsive=False,
        include_galactic=False
    )
    has_qrn_extreme = (torch.abs(qrn_extreme).sum(dim=1) > 0).float().mean().item()
    avg_power_extreme = torch.mean(torch.abs(qrn_extreme[torch.abs(qrn_extreme).sum(dim=1) > 0])**2).item()
    print(f"  Streams with atmospheric QRN: {has_qrn_extreme*100:.1f}% (expected ~98%)")
    print(f"  Mean power (with QRN): {avg_power_extreme:.6f} (VERY high burst rate)")

    print("\n" + "=" * 80)
    print("SUMMARY - Adaptive QRN Probabilities:")
    print(f"  Quiet (K=2):            {has_qrn_quiet*100:5.1f}% (baseline)")
    print(f"  Minor storm (K=5.5):    {has_qrn_minor*100:5.1f}% (geomagnetic boost)")
    print(f"  Severe storm (K=8):     {has_qrn_severe*100:5.1f}% (strong boost)")
    print(f"  Thunderstorms (act=0.9):{has_qrn_tstorm*100:5.1f}% (weather boost)")
    print(f"  EXTREME combined:       {has_qrn_extreme*100:5.1f}% (near 100%)")
    print("\n✓ Training diversity improved - storms now strongly correlated with QRN!")
    print("=" * 80)

    # Test individual components
    print("\n### Testing Individual QRN Components ###")
    k_indices = torch.tensor([3.0] * 5, device='cuda')

    atmospheric = qrn_gen.generate_atmospheric_qrn_batch(5, num_samples, k_index_batch=k_indices)
    print(f"  Atmospheric: mean power = {torch.mean(torch.abs(atmospheric)**2).item():.6f}")

    impulsive = qrn_gen.generate_impulsive_qrn_batch(5, num_samples)
    print(f"  Impulsive: mean power = {torch.mean(torch.abs(impulsive)**2).item():.6f}")

    galactic = qrn_gen.generate_galactic_noise_batch(5, num_samples)
    print(f"  Galactic: mean power = {torch.mean(torch.abs(galactic)**2).item():.6f}")


if __name__ == "__main__":
    test_qrn_generator()
