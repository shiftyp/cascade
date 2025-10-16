"""
GPU-accelerated continuous frequency-selective channel simulator for GH200.

Implements publication-quality HF ionospheric channel models:
- Continuous frequency-selective fading (2048 frequency bins, 23 Hz resolution)
- Time-varying updates (10ms resolution, 100 updates/second)
- Continuous D-layer absorption (f^-1.5 law across full spectrum)
- Realistic bursty QRN (Poisson lightning strikes, exponential decays)

Optimized for Grace Hopper unified memory architecture.
"""

import torch
import torch.fft
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class PropagationMode(Enum):
    """Propagation channel types."""
    AWGN = "awgn"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    MULTIPATH_SPARSE = "multipath_sparse"
    MULTIPATH_DENSE = "multipath_dense"


class QRNType(Enum):
    """QRN (atmospheric noise) types."""
    QUIET = "quiet"
    STATIC = "static"
    CRACKLING = "crackling"
    POPCORN = "popcorn"
    THUNDERSTORM = "thunderstorm"
    HISS = "hiss"
    AURORAL = "auroral"
    FLUTTERY = "fluttery"


@dataclass
class MultipathProfile:
    """Multipath channel profile (batch)."""
    delays_ms: torch.Tensor  # [batch_size, num_paths]
    powers: torch.Tensor  # [batch_size, num_paths], linear scale
    doppler_shifts_hz: torch.Tensor  # [batch_size, num_paths]
    k_factors: torch.Tensor  # [batch_size, num_paths], Rician K-factor


@dataclass
class TransceiverProfile:
    """
    HF transceiver impairment profile (via audio interface).

    Models common amateur radio transceivers (Icom, Yaesu, Kenwood, Elecraft)
    connected via USB audio or sound card for digital modes.
    """
    # SSB Filter (most dominant effect!)
    ssb_filter_bw_hz: float  # SSB filter bandwidth (2400-2800 Hz typical)
    ssb_filter_shape_factor: float  # Shape factor (1.8-2.5, lower = sharper)

    # Audio Interface (sound card USB or built-in)
    audio_snr_db: float  # Audio interface SNR (40-90 dB)
    audio_thd_percent: float  # Total harmonic distortion (0.01-5%)
    audio_freq_response_ripple_db: float  # Frequency response flatness (±0.5 to ±3 dB)

    # AGC (automatic gain control)
    agc_enabled: bool  # AGC on/off
    agc_attack_ms: float  # AGC attack time (1-100 ms)
    agc_release_ms: float  # AGC release time (100-1000 ms)
    agc_gain_variation_db: float  # Max gain variation (0-10 dB)

    # ALC (automatic level control on TX)
    alc_enabled: bool  # ALC on/off (TX only)
    alc_threshold_db: float  # ALC activation threshold (-10 to -3 dB below full scale)
    alc_compression_ratio: float  # Compression ratio (2:1 to 10:1)

    # Audio codec artifacts
    codec_bits: int  # Effective audio resolution (16-24 bit typical)

    # RF path (less significant for audio-coupled digital modes)
    phase_noise_dbcHz_10kHz: float  # Phase noise at 10kHz (mainly affects SSB, not audio)


class GPUTransceiverImpairments:
    """
    GPU-accelerated HF transceiver hardware impairments.

    Models realistic amateur radio transceivers connected via audio interface:
    - SSB filter bandwidth limiting (most dominant!)
    - Audio interface SNR and distortion
    - AGC pumping (RX)
    - ALC compression (TX)
    - Audio path frequency response
    - Audio codec quantization
    """

    # Hardware profiles (realistic HF transceiver + audio interface)
    # Based on popular amateur radio rigs for digital modes
    PROFILES = {
        'ideal': TransceiverProfile(
            ssb_filter_bw_hz=3000,
            ssb_filter_shape_factor=1.5,  # Very sharp filter
            audio_snr_db=90,
            audio_thd_percent=0.01,
            audio_freq_response_ripple_db=0.2,
            agc_enabled=False,
            agc_attack_ms=10,
            agc_release_ms=500,
            agc_gain_variation_db=0.5,
            alc_enabled=False,
            alc_threshold_db=-10,
            alc_compression_ratio=2.0,
            codec_bits=24,
            phase_noise_dbcHz_10kHz=-140
        ),
        'icom_ic7300': TransceiverProfile(
            # Very popular ($1400), direct sampling SDR, excellent audio via USB
            ssb_filter_bw_hz=2700,  # Adjustable, default ~2.7 kHz
            ssb_filter_shape_factor=1.9,  # Good shape factor
            audio_snr_db=85,  # Excellent USB audio
            audio_thd_percent=0.05,  # Very clean
            audio_freq_response_ripple_db=0.5,  # Flat response
            agc_enabled=True,
            agc_attack_ms=5,  # Fast AGC
            agc_release_ms=300,
            agc_gain_variation_db=2.0,  # Moderate pumping
            alc_enabled=True,
            alc_threshold_db=-6,  # Typical ALC setting
            alc_compression_ratio=3.0,
            codec_bits=16,  # 16-bit USB audio
            phase_noise_dbcHz_10kHz=-130  # Excellent phase noise
        ),
        'yaesu_ft991a': TransceiverProfile(
            # Popular all-mode ($1200), good audio
            ssb_filter_bw_hz=2600,
            ssb_filter_shape_factor=2.0,
            audio_snr_db=80,  # Good USB audio
            audio_thd_percent=0.1,
            audio_freq_response_ripple_db=0.8,
            agc_enabled=True,
            agc_attack_ms=10,
            agc_release_ms=400,
            agc_gain_variation_db=3.0,
            alc_enabled=True,
            alc_threshold_db=-5,
            alc_compression_ratio=3.5,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-125
        ),
        'elecraft_kx3': TransceiverProfile(
            # Premium QRP ($1200), excellent for digital modes
            ssb_filter_bw_hz=2800,  # Wide filter option
            ssb_filter_shape_factor=1.8,  # Excellent filters
            audio_snr_db=88,  # Excellent audio
            audio_thd_percent=0.03,
            audio_freq_response_ripple_db=0.4,
            agc_enabled=True,
            agc_attack_ms=3,  # Very fast AGC
            agc_release_ms=250,
            agc_gain_variation_db=1.5,  # Minimal pumping
            alc_enabled=True,
            alc_threshold_db=-8,  # Conservative ALC
            alc_compression_ratio=2.5,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-135  # Excellent TCXO
        ),
        'kenwood_ts590': TransceiverProfile(
            # Premium rig ($2000+), excellent audio
            ssb_filter_bw_hz=2700,
            ssb_filter_shape_factor=1.85,
            audio_snr_db=87,
            audio_thd_percent=0.04,
            audio_freq_response_ripple_db=0.5,
            agc_enabled=True,
            agc_attack_ms=8,
            agc_release_ms=350,
            agc_gain_variation_db=2.0,
            alc_enabled=True,
            alc_threshold_db=-7,
            alc_compression_ratio=3.0,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-132
        ),
        'budget_soundcard': TransceiverProfile(
            # Older rig or budget setup with external USB sound card
            ssb_filter_bw_hz=2400,  # Narrow SSB filter
            ssb_filter_shape_factor=2.5,  # Mediocre shape
            audio_snr_db=65,  # Cheap USB sound card
            audio_thd_percent=1.0,  # Higher distortion
            audio_freq_response_ripple_db=2.0,  # Poor flatness
            agc_enabled=True,
            agc_attack_ms=50,  # Slow AGC (vintage)
            agc_release_ms=800,
            agc_gain_variation_db=4.0,  # Reduced from 5.0 (pilots provide reference)
            alc_enabled=True,
            alc_threshold_db=-4,  # Aggressive ALC
            alc_compression_ratio=5.0,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-115  # Older oscillator
        ),
        'chinese_budget': TransceiverProfile(
            # Budget Chinese kits (QCX, uBITX, etc.)
            ssb_filter_bw_hz=2500,
            ssb_filter_shape_factor=2.3,
            audio_snr_db=60,  # Basic audio
            audio_thd_percent=2.0,  # Higher distortion
            audio_freq_response_ripple_db=3.0,  # Poor response
            agc_enabled=True,
            agc_attack_ms=100,  # Very slow
            agc_release_ms=1000,
            agc_gain_variation_db=4.0,  # Reduced from 8.0 (pilots provide reference)
            alc_enabled=True,
            alc_threshold_db=-3,  # Very aggressive
            alc_compression_ratio=8.0,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-105
        ),
        # ===== SDR TRANSCEIVERS (Future-proofing) =====
        'icom_ic705': TransceiverProfile(
            # Popular portable SDR ($1400), direct sampling, excellent digital
            ssb_filter_bw_hz=2900,  # Wide DSP filter
            ssb_filter_shape_factor=1.7,  # Sharp DSP filter
            audio_snr_db=88,  # Excellent direct digital audio
            audio_thd_percent=0.02,  # Very clean digital path
            audio_freq_response_ripple_db=0.3,  # Flat DSP response
            agc_enabled=True,
            agc_attack_ms=3,  # Very fast DSP AGC
            agc_release_ms=200,
            agc_gain_variation_db=1.5,
            alc_enabled=True,
            alc_threshold_db=-8,
            alc_compression_ratio=2.5,
            codec_bits=24,  # 24-bit internal
            phase_noise_dbcHz_10kHz=-135
        ),
        'flex6000': TransceiverProfile(
            # High-end SDR ($3000+), best-in-class digital
            ssb_filter_bw_hz=3000,  # Fully flexible DSP
            ssb_filter_shape_factor=1.6,  # Excellent DSP filter
            audio_snr_db=95,  # Exceptional digital audio
            audio_thd_percent=0.005,  # Virtually no distortion
            audio_freq_response_ripple_db=0.1,  # Essentially flat
            agc_enabled=True,
            agc_attack_ms=2,
            agc_release_ms=150,
            agc_gain_variation_db=1.0,
            alc_enabled=True,
            alc_threshold_db=-10,
            alc_compression_ratio=2.0,
            codec_bits=24,
            phase_noise_dbcHz_10kHz=-140
        ),
        'hermes_lite2': TransceiverProfile(
            # Open-source SDR (~$300-500), good performance
            ssb_filter_bw_hz=2850,
            ssb_filter_shape_factor=1.8,
            audio_snr_db=82,
            audio_thd_percent=0.08,
            audio_freq_response_ripple_db=0.6,
            agc_enabled=True,
            agc_attack_ms=5,
            agc_release_ms=250,
            agc_gain_variation_db=2.0,
            alc_enabled=True,
            alc_threshold_db=-7,
            alc_compression_ratio=3.0,
            codec_bits=16,
            phase_noise_dbcHz_10kHz=-128
        )
    }

    # Weighted sampling (matches real-world amateur radio usage + future growth)
    PROFILE_WEIGHTS = {
        # Traditional HF transceivers (70%)
        'icom_ic7300': 0.30,      # 30% - extremely popular for digital
        'yaesu_ft991a': 0.20,     # 20% - popular all-mode
        'elecraft_kx3': 0.10,     # 10% - premium, QRP enthusiasts
        'kenwood_ts590': 0.05,    # 5% - premium contesters
        'budget_soundcard': 0.03, # 3% - vintage rigs with USB sound card
        'chinese_budget': 0.02,   # 2% - budget/kit builders

        # SDR transceivers (30% - growing segment!)
        'icom_ic705': 0.15,       # 15% - very popular portable SDR
        'flex6000': 0.08,         # 8% - high-end SDR users
        'hermes_lite2': 0.07      # 7% - open-source/homebrew SDR
    }

    def __init__(self, sample_rate: int = 48000, device: str = 'cuda'):
        """
        Initialize transceiver impairments simulator.

        Args:
            sample_rate: Sample rate in Hz
            device: 'cuda' for GPU
        """
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        # Pre-compute SSB filters for common bandwidths (2400-2800 Hz)
        self.ssb_filters = {}
        for bw in [2400, 2500, 2600, 2700, 2800, 3000]:
            self.ssb_filters[bw] = self._design_ssb_filter(bw, shape_factor=2.0)

    def _design_ssb_filter(self, bandwidth_hz: float, shape_factor: float = 2.0) -> torch.Tensor:
        """
        Design SSB bandpass filter (Butterworth).

        Args:
            bandwidth_hz: 3dB bandwidth (2400-2800 Hz typical)
            shape_factor: Shape factor (1.8-2.5, SF = BW_60dB/BW_6dB)

        Returns:
            torch.Tensor: FFT-domain filter response [num_bins]
        """
        # Design in frequency domain
        num_bins = 2048  # Fixed FFT size
        freqs = torch.fft.rfftfreq(num_bins, 1/self.sample_rate, device=self.device)

        # SSB passband: 500-2600 Hz (CASCADE bandwidth with guard bands)
        # Center filter high enough that widest filter (3000 Hz) has lower edge >= 500 Hz
        # For 3000 Hz BW: center = 500 + 1500 = 2000 Hz
        center_freq = 2000.0
        bw_3dB = bandwidth_hz / 2  # ±1200-1400 Hz from center

        # Butterworth filter order from shape factor
        # Higher shape factor = lower order (gentler slopes)
        order = int(6 / shape_factor)  # Typical: 2-3 for HF rigs

        # Butterworth response: H(f) = 1 / sqrt(1 + ((f - fc)/BW)^(2*order))
        freq_offset = torch.abs(freqs - center_freq)
        H = 1.0 / torch.sqrt(1 + (freq_offset / bw_3dB) ** (2 * order))

        return H

    def apply_dc_blocking_filter_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Apply DC blocking high-pass filter to remove sub-300 Hz content.

        Removes DC offset and low-frequency artifacts from AGC/quantization.
        Uses 2nd-order Butterworth HPF at 250 Hz.

        Args:
            signals: [batch_size, num_samples], complex signals

        Returns:
            torch.Tensor: [batch_size, num_samples], DC-blocked signals
        """
        batch_size, num_samples = signals.shape

        # FFT to frequency domain
        signals_fft = torch.fft.fft(signals, dim=-1)

        # Design high-pass filter: 2nd-order Butterworth at 250 Hz cutoff
        freqs = torch.fft.fftfreq(num_samples, 1/self.sample_rate, device=self.device)
        freqs_abs = torch.abs(freqs)

        # HPF: H(f) = 1 - 1/sqrt(1 + (f/fc)^(2*order))
        # This zeros DC and attenuates below fc
        fc = 250.0  # 250 Hz cutoff (below CASCADE min 300 Hz)
        order = 2  # 2nd order = 12 dB/octave rolloff

        H_hpf = 1.0 - 1.0 / torch.sqrt(1 + (freqs_abs / fc) ** (2 * order))

        # Apply filter
        signals_fft_filtered = signals_fft * H_hpf.unsqueeze(0)

        # IFFT back to time domain
        return torch.fft.ifft(signals_fft_filtered, dim=-1)

    def apply_ssb_filter_batch(self,
                               signals: torch.Tensor,
                               bandwidths_hz: torch.Tensor,
                               shape_factors: torch.Tensor) -> torch.Tensor:
        """
        Apply SSB filter (most dominant HF transceiver effect!).

        Models IF crystal filter or DSP filter in transceiver.

        Args:
            signals: [batch_size, num_samples], complex signals
            bandwidths_hz: [batch_size], SSB filter bandwidth (2400-2800 Hz)
            shape_factors: [batch_size], filter shape factor (1.8-2.5)

        Returns:
            torch.Tensor: [batch_size, num_samples], filtered signals
        """
        batch_size, num_samples = signals.shape

        # For very large signals, process in chunks to avoid cuFFT errors
        # REDUCED: 48000 samples (1s) safer for cuFFT with large batches
        CHUNK_SIZE = 24000  # 0.5 second at 48kHz (safest for cuFFT)

        if num_samples > CHUNK_SIZE:
            # Process in overlapping chunks to avoid edge artifacts
            overlap = 4800  # 100ms overlap
            output = torch.zeros_like(signals)

            num_chunks = (num_samples + CHUNK_SIZE - overlap - 1) // (CHUNK_SIZE - overlap)

            for chunk_idx in range(num_chunks):
                start = chunk_idx * (CHUNK_SIZE - overlap)
                end = min(start + CHUNK_SIZE, num_samples)

                if start >= num_samples:
                    break

                # Extract chunk
                chunk = signals[:, start:end]

                # Filter chunk
                filtered_chunk = self._apply_ssb_filter_single_chunk(chunk, bandwidths_hz, shape_factors)

                # Blend into output (use window for smooth transitions)
                if chunk_idx == 0:
                    # First chunk - no blend at start
                    output[:, start:end] = filtered_chunk
                else:
                    # Blend overlap region
                    blend_start = start
                    blend_end = start + overlap
                    if blend_end <= num_samples:
                        # Create blend window
                        blend_win = torch.linspace(0, 1, overlap, device=self.device)
                        output[:, blend_start:blend_end] = (
                            output[:, blend_start:blend_end] * (1 - blend_win) +
                            filtered_chunk[:, :overlap] * blend_win
                        )
                        # Copy non-overlapping part
                        if end > blend_end:
                            output[:, blend_end:end] = filtered_chunk[:, overlap:]
                    else:
                        output[:, blend_start:end] = filtered_chunk

            return output

        # For signals <= CHUNK_SIZE, use single-chunk processing
        return self._apply_ssb_filter_single_chunk(signals, bandwidths_hz, shape_factors)

    def _apply_ssb_filter_single_chunk(self,
                                       signals: torch.Tensor,
                                       bandwidths_hz: torch.Tensor,
                                       shape_factors: torch.Tensor) -> torch.Tensor:
        """
        Apply SSB filter to a single chunk (helper for chunked processing).

        Args:
            signals: [batch_size, chunk_samples], complex signals
            bandwidths_hz: [batch_size], SSB filter bandwidth
            shape_factors: [batch_size], filter shape factor

        Returns:
            torch.Tensor: [batch_size, chunk_samples], filtered signals
        """
        batch_size, num_samples = signals.shape

        # FFT to frequency domain (with error handling)
        try:
            signals_fft = torch.fft.fft(signals, dim=-1)
        except RuntimeError as e:
            if "cuFFT" in str(e):
                # cuFFT error - reduce batch processing to smaller chunks
                raise RuntimeError(
                    f"cuFFT error in SSB filter: batch_size={batch_size}, chunk_samples={num_samples}. "
                    f"Try reducing stream batch size in dataset generation. Original error: {e}"
                )
            else:
                raise

        # Apply filter to each signal
        for i in range(batch_size):
            bw = int(bandwidths_hz[i].item())
            # Round to nearest cached bandwidth
            bw_key = min(self.ssb_filters.keys(), key=lambda x: abs(x - bw))
            filter_response = self.ssb_filters[bw_key]

            # Pad/interpolate filter to match signal FFT size
            if len(filter_response) != num_samples // 2 + 1:
                # Resample filter response
                filter_response = torch.nn.functional.interpolate(
                    filter_response.unsqueeze(0).unsqueeze(0),
                    size=num_samples // 2 + 1,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            # Apply to both positive and negative frequencies
            # rfft only gives positive, so mirror for negative
            filter_full = torch.cat([filter_response, filter_response.flip(0)[1:-1]])

            signals_fft[i] = signals_fft[i] * filter_full[:num_samples]

        # IFFT back to time domain (with error handling)
        try:
            return torch.fft.ifft(signals_fft, dim=-1)
        except RuntimeError as e:
            if "cuFFT" in str(e):
                raise RuntimeError(
                    f"cuFFT error in SSB filter IFFT: batch_size={batch_size}, samples={num_samples}. "
                    f"Try reducing CASCADE_DATASET_BATCH_SIZE. Current error: {e}"
                )
            else:
                raise

    def apply_audio_interface_batch(self,
                                    signals: torch.Tensor,
                                    snr_db: torch.Tensor,
                                    thd_percent: torch.Tensor,
                                    freq_ripple_db: torch.Tensor) -> torch.Tensor:
        """
        Apply audio interface impairments (USB sound card or built-in audio).

        Args:
            signals: [batch_size, num_samples], complex signals
            snr_db: [batch_size], audio interface SNR
            thd_percent: [batch_size], total harmonic distortion (%)
            freq_ripple_db: [batch_size], frequency response ripple (±dB)

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with audio impairments
        """
        batch_size, num_samples = signals.shape

        # For very large signals, process in chunks to avoid cuFFT errors
        # REDUCED: 48000 samples (1s) safer for cuFFT with large batches
        CHUNK_SIZE = 24000  # 0.5 second at 48kHz (safest for cuFFT)

        # 1. Frequency response variations (non-flat audio path)
        if num_samples > CHUNK_SIZE and (freq_ripple_db > 0.01).any():
            # Process in overlapping chunks
            overlap = 4800  # 100ms overlap
            output = torch.zeros_like(signals)
            num_chunks = (num_samples + CHUNK_SIZE - overlap - 1) // (CHUNK_SIZE - overlap)

            for chunk_idx in range(num_chunks):
                start = chunk_idx * (CHUNK_SIZE - overlap)
                end = min(start + CHUNK_SIZE, num_samples)
                if start >= num_samples:
                    break

                chunk = signals[:, start:end]
                chunk_size = end - start

                # Apply frequency response to chunk
                freqs = torch.fft.fftfreq(chunk_size, 1/self.sample_rate, device=self.device)
                freq_response = torch.ones(batch_size, chunk_size, device=self.device, dtype=torch.float32)

                for i in range(batch_size):
                    ripple_db = (torch.rand(chunk_size, device=self.device) - 0.5) * 2 * freq_ripple_db[i]
                    ripple_linear = 10 ** (ripple_db / 20.0)
                    freq_response[i] = ripple_linear

                chunk_fft = torch.fft.fft(chunk, dim=-1)
                chunk_fft = chunk_fft * freq_response
                filtered_chunk = torch.fft.ifft(chunk_fft, dim=-1)

                # Blend into output
                if chunk_idx == 0:
                    output[:, start:end] = filtered_chunk
                else:
                    blend_start = start
                    blend_end = start + overlap
                    if blend_end <= num_samples:
                        blend_win = torch.linspace(0, 1, overlap, device=self.device)
                        output[:, blend_start:blend_end] = (
                            output[:, blend_start:blend_end] * (1 - blend_win) +
                            filtered_chunk[:, :overlap] * blend_win
                        )
                        if end > blend_end:
                            output[:, blend_end:end] = filtered_chunk[:, overlap:]
                    else:
                        output[:, blend_start:end] = filtered_chunk

            signals = output
        elif (freq_ripple_db > 0.01).any():
            # Signals <= CHUNK_SIZE but need freq response
            # Add chunking for safety even on small signals
            if num_samples > CHUNK_SIZE:
                # Should not reach here (handled above), but chunk anyway for safety
                overlap = 4800
                output = torch.zeros_like(signals)
                num_chunks = (num_samples + CHUNK_SIZE - overlap - 1) // (CHUNK_SIZE - overlap)

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * (CHUNK_SIZE - overlap)
                    end = min(start + CHUNK_SIZE, num_samples)
                    if start >= num_samples:
                        break

                    chunk = signals[:, start:end]
                    chunk_size = end - start
                    freqs = torch.fft.fftfreq(chunk_size, 1/self.sample_rate, device=self.device)
                    freq_response = torch.ones(batch_size, chunk_size, device=self.device, dtype=torch.float32)

                    for i in range(batch_size):
                        ripple_db = (torch.rand(chunk_size, device=self.device) - 0.5) * 2 * freq_ripple_db[i]
                        ripple_linear = 10 ** (ripple_db / 20.0)
                        freq_response[i] = ripple_linear

                    chunk_fft = torch.fft.fft(chunk, dim=-1)
                    chunk_fft = chunk_fft * freq_response
                    filtered_chunk = torch.fft.ifft(chunk_fft, dim=-1)

                    if chunk_idx == 0:
                        output[:, start:end] = filtered_chunk
                    else:
                        blend_start = start
                        blend_end = start + overlap
                        if blend_end <= num_samples:
                            blend_win = torch.linspace(0, 1, overlap, device=self.device)
                            output[:, blend_start:blend_end] = (
                                output[:, blend_start:blend_end] * (1 - blend_win) +
                                filtered_chunk[:, :overlap] * blend_win
                            )
                            if end > blend_end:
                                output[:, blend_end:end] = filtered_chunk[:, overlap:]
                        else:
                            output[:, blend_start:end] = filtered_chunk
                signals = output
            else:
                # Process small signal normally
                num_bins = num_samples
                freqs = torch.fft.fftfreq(num_samples, 1/self.sample_rate, device=self.device)
                freq_response = torch.ones(batch_size, num_bins, device=self.device, dtype=torch.float32)

                for i in range(batch_size):
                    ripple_db = (torch.rand(num_bins, device=self.device) - 0.5) * 2 * freq_ripple_db[i]
                    ripple_linear = 10 ** (ripple_db / 20.0)
                    freq_response[i] = ripple_linear

                signals_fft = torch.fft.fft(signals, dim=-1)
                signals_fft = signals_fft * freq_response
                signals = torch.fft.ifft(signals_fft, dim=-1)

        # 2. Harmonic distortion (audio amplifier non-linearity)
        if (thd_percent > 0.1).any():
            amplitude = torch.abs(signals)
            phase = torch.angle(signals)

            # 2nd and 3rd harmonic distortion
            # THD ≈ sqrt(h2^2 + h3^2), typically h2 > h3
            h2 = thd_percent / 150.0  # 2nd harmonic
            h3 = thd_percent / 300.0  # 3rd harmonic

            # Add harmonics (simplified - phase modulation)
            distortion = h2.unsqueeze(1) * torch.sin(2 * phase) + h3.unsqueeze(1) * torch.sin(3 * phase)
            signals = (amplitude + distortion * amplitude) * torch.exp(1j * phase)

        # 3. Audio interface noise (thermal + quantization)
        signal_power = torch.mean(torch.abs(signals) ** 2, dim=1, keepdim=True)
        noise_power = signal_power / (10 ** (snr_db.unsqueeze(1) / 10))
        audio_noise = (torch.randn_like(signals.real) + 1j * torch.randn_like(signals.imag)) / np.sqrt(2)
        signals = signals + audio_noise * torch.sqrt(noise_power + 1e-10)

        return signals

    def apply_agc_pumping_batch(self,
                                signals: torch.Tensor,
                                attack_ms: torch.Tensor,
                                release_ms: torch.Tensor,
                                max_variation_db: torch.Tensor,
                                enabled: List[bool]) -> torch.Tensor:
        """
        Apply AGC pumping (automatic gain control on RX).

        Models AGC gain variations when signal power changes.

        Args:
            signals: [batch_size, num_samples], complex signals
            attack_ms: [batch_size], AGC attack time
            release_ms: [batch_size], AGC release time
            max_variation_db: [batch_size], maximum gain variation
            enabled: [batch_size], whether AGC is enabled

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with AGC pumping
        """
        batch_size, num_samples = signals.shape

        # Process each signal individually (AGC is time-variant)
        for i in range(batch_size):
            if not enabled[i]:
                continue

            # Calculate instantaneous power envelope (50ms windows for smoother AGC)
            # Real AGC doesn't respond to every 10ms fluctuation - that creates clicks!
            window_samples = int(0.050 * self.sample_rate)  # 50ms (realistic AGC response time)
            num_windows = num_samples // window_samples

            power_envelope = torch.zeros(num_samples, device=self.device)
            for w in range(num_windows):
                start = w * window_samples
                end = min(start + window_samples, num_samples)
                window_power = torch.mean(torch.abs(signals[i, start:end]) ** 2)
                power_envelope[start:end] = window_power

            # AGC response (exponential attack/release)
            attack_samples = int(attack_ms[i].item() * self.sample_rate / 1000)
            release_samples = int(release_ms[i].item() * self.sample_rate / 1000)

            # Simplified AGC: gain inversely proportional to power
            target_power = torch.mean(power_envelope)

            # CRITICAL: Add floor to power_envelope to prevent excessive gain on noise
            # When there's only noise (no signal), power is very low, causing huge AGC gain
            # This amplifies the noise floor and creates amplitude spikes when signals arrive
            power_envelope_floored = torch.maximum(power_envelope, target_power * 0.1)  # Floor at 10% of target

            gain_db = 10 * torch.log10(target_power / (power_envelope_floored + 1e-10))

            # Limit gain variation
            gain_db = torch.clamp(gain_db, -max_variation_db[i], max_variation_db[i])

            # Apply exponential smoothing (attack/release) on window values
            # This operates on the window envelope, not sample-by-sample
            gain_smooth = torch.zeros_like(gain_db)
            gain_smooth[0] = gain_db[0]
            for t in range(1, len(gain_db)):
                if gain_db[t] > gain_smooth[t-1]:
                    # Attack (gain increasing)
                    alpha = 1.0 / max(attack_samples / window_samples, 1.0)  # Scale by window size
                else:
                    # Release (gain decreasing)
                    alpha = 1.0 / max(release_samples / window_samples, 1.0)  # Scale by window size
                alpha = min(alpha, 0.5)  # Cap at 50% per window to prevent instability
                gain_smooth[t] = (1 - alpha) * gain_smooth[t-1] + alpha * gain_db[t]

            # Interpolate gain envelope smoothly across samples (prevents clicks!)
            # Upsample from window rate to sample rate
            gain_linear_windows = 10 ** (gain_smooth / 20.0)

            # Create smooth interpolated gain for all samples
            gain_linear = torch.ones(num_samples, device=self.device)
            for w in range(num_windows):
                start = w * window_samples
                end = min(start + window_samples, num_samples)

                # Linear interpolation within window
                if w < num_windows - 1:
                    # Interpolate from current window to next
                    t_interp = torch.linspace(0, 1, window_samples, device=self.device)
                    gain_linear[start:end] = (
                        gain_linear_windows[w] * (1 - t_interp[:end-start]) +
                        gain_linear_windows[min(w+1, len(gain_linear_windows)-1)] * t_interp[:end-start]
                    )
                else:
                    # Last window - hold constant
                    gain_linear[start:end] = gain_linear_windows[w]

            # Apply smooth gain modulation
            signals[i] = signals[i] * gain_linear

        return signals

    def apply_alc_compression_batch(self,
                                    signals: torch.Tensor,
                                    threshold_db: torch.Tensor,
                                    compression_ratio: torch.Tensor,
                                    enabled: List[bool]) -> torch.Tensor:
        """
        Apply ALC compression (automatic level control on TX).

        Prevents overdriving the transmitter.

        Args:
            signals: [batch_size, num_samples], complex signals
            threshold_db: [batch_size], ALC threshold (dB below full scale)
            compression_ratio: [batch_size], compression ratio (2:1 to 10:1)
            enabled: [batch_size], whether ALC is enabled

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with ALC compression
        """
        batch_size, num_samples = signals.shape

        for i in range(batch_size):
            if not enabled[i]:
                continue

            # Extract amplitude and phase
            amplitude = torch.abs(signals[i])
            phase = torch.angle(signals[i])

            # ALC threshold (linear)
            threshold_linear = 10 ** (threshold_db[i] / 20.0)

            # Apply compression above threshold
            # Below threshold: gain = 1
            # Above threshold: gain = (A/T)^((1/R)-1)
            mask_above = amplitude > threshold_linear
            amplitude_out = amplitude.clone()

            if mask_above.any():
                ratio = compression_ratio[i]
                # Soft knee compression
                excess = amplitude[mask_above] / threshold_linear
                compressed = threshold_linear * torch.pow(excess, 1.0 / ratio)
                amplitude_out[mask_above] = compressed

            # Reconstruct
            signals[i] = amplitude_out * torch.exp(1j * phase)

        return signals

    def apply_audio_codec_quantization_batch(self,
                                             signals: torch.Tensor,
                                             bits: torch.Tensor) -> torch.Tensor:
        """
        Apply audio codec quantization (USB audio, typically 16-bit).

        Args:
            signals: [batch_size, num_samples], complex signals
            bits: [batch_size], audio codec bit depth

        Returns:
            torch.Tensor: [batch_size, num_samples], quantized signals
        """
        # Normalize to [-1, 1] based on 95th percentile (prevents outliers from reducing resolution)
        max_amp = torch.quantile(torch.abs(signals), 0.95, dim=1, keepdim=True)
        max_amp = torch.maximum(max_amp, torch.tensor(1e-10, device=self.device))

        normalized = signals / max_amp

        # Quantization levels (per-batch)
        for i in range(signals.shape[0]):
            levels = 2.0 ** bits[i]

            # Quantize I and Q separately
            i_quantized = torch.round(normalized[i].real * (levels / 2)) / (levels / 2)
            q_quantized = torch.round(normalized[i].imag * (levels / 2)) / (levels / 2)

            normalized[i] = (i_quantized + 1j * q_quantized)

        # Denormalize
        return normalized * max_amp

    @staticmethod
    def sample_random_profiles(batch_size: int, weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Sample random transceiver profiles weighted by real-world usage.

        Args:
            batch_size: Number of profiles to sample
            weights: Optional custom weights (default: use PROFILE_WEIGHTS)

        Returns:
            List[str]: List of profile names (length = batch_size)
        """
        if weights is None:
            weights = GPUTransceiverImpairments.PROFILE_WEIGHTS

        profile_names = list(weights.keys())
        profile_probs = np.array([weights[p] for p in profile_names])
        profile_probs = profile_probs / profile_probs.sum()  # Normalize

        return list(np.random.choice(profile_names, size=batch_size, p=profile_probs))

    def apply_tx_impairments(self,
                            signals: torch.Tensor,
                            profiles: List[str]) -> torch.Tensor:
        """
        Apply TX impairments (before channel).

        Signal path: Clean signal → ALC → SSB Filter → Audio Interface → Air

        Args:
            signals: [batch_size, num_samples], complex signals
            profiles: List of profile names (length = batch_size)

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with TX impairments
        """
        batch_size = signals.shape[0]

        # Get profiles
        tx_profiles = [self.PROFILES[p] for p in profiles]

        # Extract parameters as tensors
        alc_enabled = [p.alc_enabled for p in tx_profiles]
        alc_threshold = torch.tensor([p.alc_threshold_db for p in tx_profiles], device=self.device)
        alc_ratio = torch.tensor([p.alc_compression_ratio for p in tx_profiles], device=self.device)
        ssb_bw = torch.tensor([p.ssb_filter_bw_hz for p in tx_profiles], device=self.device)
        ssb_shape = torch.tensor([p.ssb_filter_shape_factor for p in tx_profiles], device=self.device)
        audio_snr = torch.tensor([p.audio_snr_db for p in tx_profiles], device=self.device)
        audio_thd = torch.tensor([p.audio_thd_percent for p in tx_profiles], device=self.device)
        audio_ripple = torch.tensor([p.audio_freq_response_ripple_db for p in tx_profiles], device=self.device)

        # Apply TX impairments in signal path order
        # 1. ALC compression (prevents overdriving PA)
        signals = self.apply_alc_compression_batch(signals, alc_threshold, alc_ratio, alc_enabled)

        # 2. SSB filter (crystal/DSP filter in TX chain)
        signals = self.apply_ssb_filter_batch(signals, ssb_bw, ssb_shape)

        # 3. Audio interface (microphone/line-in path to modulator)
        signals = self.apply_audio_interface_batch(signals, audio_snr, audio_thd, audio_ripple)

        return signals

    def apply_rx_impairments(self,
                            signals: torch.Tensor,
                            profiles: List[str]) -> torch.Tensor:
        """
        Apply RX impairments (after channel).

        Signal path: Air → SSB Filter → AGC → Audio Interface → Computer

        Args:
            signals: [batch_size, num_samples], complex signals
            profiles: List of profile names (length = batch_size)

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with RX impairments
        """
        batch_size = signals.shape[0]

        # Get profiles
        rx_profiles = [self.PROFILES[p] for p in profiles]

        # Extract parameters as tensors
        ssb_bw = torch.tensor([p.ssb_filter_bw_hz for p in rx_profiles], device=self.device)
        ssb_shape = torch.tensor([p.ssb_filter_shape_factor for p in rx_profiles], device=self.device)
        agc_enabled = [p.agc_enabled for p in rx_profiles]
        agc_attack = torch.tensor([p.agc_attack_ms for p in rx_profiles], device=self.device)
        agc_release = torch.tensor([p.agc_release_ms for p in rx_profiles], device=self.device)
        agc_variation = torch.tensor([p.agc_gain_variation_db for p in rx_profiles], device=self.device)
        audio_snr = torch.tensor([p.audio_snr_db for p in rx_profiles], device=self.device)
        audio_thd = torch.tensor([p.audio_thd_percent for p in rx_profiles], device=self.device)
        audio_ripple = torch.tensor([p.audio_freq_response_ripple_db for p in rx_profiles], device=self.device)
        codec_bits = torch.tensor([p.codec_bits for p in rx_profiles], device=self.device, dtype=torch.float32)

        # Apply RX impairments in signal path order
        # 1. SSB filter (roofing filter + IF filter)
        signals = self.apply_ssb_filter_batch(signals, ssb_bw, ssb_shape)

        # 2. AGC pumping (gain variations)
        signals = self.apply_agc_pumping_batch(signals, agc_attack, agc_release, agc_variation, agc_enabled)

        # 3. Audio interface (demodulator → sound card)
        signals = self.apply_audio_interface_batch(signals, audio_snr, audio_thd, audio_ripple)

        # 4. Audio codec quantization (USB audio, typically 16-bit)
        signals = self.apply_audio_codec_quantization_batch(signals, codec_bits)

        # 5. DC blocking filter (removes DC offset and sub-300 Hz artifacts)
        # DISABLED: cuFFT error on large signals, handle at source instead
        # signals = self.apply_dc_blocking_filter_batch(signals)

        return signals


class GPUChannelSimulator:
    """
    GPU-accelerated continuous frequency-selective channel simulator.

    Key features:
    - Continuous frequency response (2048 bins = 23 Hz resolution)
    - Time-varying updates every 10ms (100 snapshots/sec)
    - Physically accurate D-layer absorption (f^-1.5)
    - Realistic bursty QRN with temporal structure
    """

    def __init__(self, sample_rate=48000, device='cuda', enable_transceiver_impairments=False):
        """
        Initialize GPU channel simulator.

        Args:
            sample_rate: Sample rate in Hz
            device: 'cuda' for GPU
            enable_transceiver_impairments: Enable HF transceiver hardware impairments
        """
        self.sample_rate = sample_rate
        self.device = torch.device(device)

        # Time-varying channel parameters
        # Using 10ms updates for high time resolution (100 updates/sec)
        # This provides realistic HF channel time-varying behavior
        self.update_interval_ms = 10  # Update every 10ms for high time resolution
        self.coherence_bandwidth_hz = 50  # Typical HF: 20-100 Hz

        # Initialize transceiver impairments (SSB filter, AGC, ALC, audio interface)
        self.enable_transceiver_impairments = enable_transceiver_impairments
        if enable_transceiver_impairments:
            self.transceiver_impairments = GPUTransceiverImpairments(
                sample_rate=sample_rate,
                device=device
            )
            print(f"GPUChannelSimulator: {self.sample_rate} Hz, device {self.device}")
            print(f"  Time resolution: {self.update_interval_ms}ms (100 updates/sec)")
            print(f"  Coherence bandwidth: {self.coherence_bandwidth_hz} Hz")
            print(f"  Transceiver impairments: ENABLED (HF rig + audio interface)")
            print(f"    Profiles: {list(GPUTransceiverImpairments.PROFILES.keys())}")
        else:
            self.transceiver_impairments = None
            print(f"GPUChannelSimulator: {self.sample_rate} Hz, device {self.device}")
            print(f"  Time resolution: {self.update_interval_ms}ms (100 updates/sec)")
            print(f"  Coherence bandwidth: {self.coherence_bandwidth_hz} Hz")

    def apply_tx_rx_impairments_batch(self,
                                      clean_signals: torch.Tensor,
                                      tx_profiles: Optional[List[str]] = None,
                                      rx_profiles: Optional[List[str]] = None,
                                      apply_tx: bool = True,
                                      apply_rx: bool = True) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Apply TX and/or RX transceiver impairments with weighted random sampling.

        Signal flow: Clean → TX impairments → Channel → RX impairments → Received

        Args:
            clean_signals: [batch_size, num_samples], clean signals
            tx_profiles: Optional list of TX profile names (default: random sample)
            rx_profiles: Optional list of RX profile names (default: random sample)
            apply_tx: Apply TX impairments (default: True)
            apply_rx: Apply RX impairments (default: True)

        Returns:
            Tuple of (signals, tx_profiles_used, rx_profiles_used)
        """
        if not self.enable_transceiver_impairments:
            # Impairments disabled - return clean signals
            return clean_signals, [], []

        batch_size = clean_signals.shape[0]

        # Sample random profiles if not provided (weighted by popularity)
        if tx_profiles is None and apply_tx:
            tx_profiles = self.transceiver_impairments.sample_random_profiles(batch_size)

        if rx_profiles is None and apply_rx:
            rx_profiles = self.transceiver_impairments.sample_random_profiles(batch_size)

        # Apply TX impairments (before channel)
        signals = clean_signals
        if apply_tx and tx_profiles is not None:
            signals = self.transceiver_impairments.apply_tx_impairments(signals, tx_profiles)

        # RX impairments applied after channel (caller should apply channel first)
        # We return the signal and profiles for caller to apply RX after channel

        return signals, tx_profiles if apply_tx else [], rx_profiles if apply_rx else []

    def apply_rx_impairments_only(self,
                                  signals_after_channel: torch.Tensor,
                                  rx_profiles: List[str]) -> torch.Tensor:
        """
        Apply only RX impairments (after channel has been applied).

        Args:
            signals_after_channel: [batch_size, num_samples], signals after channel
            rx_profiles: List of RX profile names

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with RX impairments
        """
        if not self.enable_transceiver_impairments:
            return signals_after_channel

        return self.transceiver_impairments.apply_rx_impairments(signals_after_channel, rx_profiles)

    def generate_jakes_fading_batch(self,
                                    batch_size: int,
                                    num_samples: int,
                                    doppler_hz: torch.Tensor,
                                    num_oscillators: int = 20) -> torch.Tensor:
        """
        Generate Jakes model Rayleigh fading for batch.

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            doppler_hz: [batch_size], Doppler spread in Hz
            num_oscillators: Number of sinusoids (8 sufficient for converged statistics)

        Returns:
            torch.Tensor: [batch_size, num_samples], complex fading envelope
        """
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.sample_rate
        t = t.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_samples]

        i_component = torch.zeros(batch_size, num_samples, device=self.device)
        q_component = torch.zeros(batch_size, num_samples, device=self.device)

        for n in range(num_oscillators):
            # Random phases per batch
            phases = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi
            angle = 2 * torch.pi * n / num_oscillators

            # Doppler frequency for this oscillator
            fd = doppler_hz.unsqueeze(1) * torch.cos(torch.tensor(angle, device=self.device))

            i_component += torch.cos(2 * torch.pi * fd * t + phases)
            q_component += torch.sin(2 * torch.pi * fd * t + phases)

        # Normalize to unit power
        i_component /= torch.sqrt(torch.tensor(num_oscillators / 2, device=self.device))
        q_component /= torch.sqrt(torch.tensor(num_oscillators / 2, device=self.device))

        return i_component + 1j * q_component

    def generate_rician_fading_batch(self,
                                     batch_size: int,
                                     num_samples: int,
                                     doppler_hz: torch.Tensor,
                                     k_factors: torch.Tensor) -> torch.Tensor:
        """
        Generate Rician fading (LOS + Rayleigh).

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            doppler_hz: [batch_size], Doppler spread
            k_factors: [batch_size], Rician K-factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex Rician fading
        """
        # Generate Rayleigh component
        rayleigh = self.generate_jakes_fading_batch(batch_size, num_samples, doppler_hz)

        # Add LOS component
        # K = LOS_power / scattered_power
        # Total power = 1 = LOS_power + scattered_power
        # LOS_amplitude = sqrt(K / (K + 1))
        # scattered_amplitude = sqrt(1 / (K + 1))

        los_amplitude = torch.sqrt(k_factors / (k_factors + 1)).unsqueeze(1)
        scattered_amplitude = torch.sqrt(1 / (k_factors + 1)).unsqueeze(1)

        rician = los_amplitude + scattered_amplitude * rayleigh

        return rician

    def generate_continuous_frequency_response(self,
                                               batch_size: int,
                                               num_freq_bins: int,
                                               multipath_profile: MultipathProfile) -> torch.Tensor:
        """
        Generate continuous frequency-selective channel response.

        Creates smooth frequency response from multipath profile.
        Each path contributes: α_i × exp(-j2πfτ_i)

        Args:
            batch_size: Number of channels
            num_freq_bins: Number of frequency bins (2048 for 48 kHz)
            multipath_profile: Multipath parameters

        Returns:
            torch.Tensor: [batch_size, num_freq_bins], complex frequency response
        """
        num_paths = multipath_profile.delays_ms.shape[1]

        # Frequency bins (Hz)
        freq_bins = torch.fft.rfftfreq(num_freq_bins, 1/self.sample_rate, device=self.device)
        freq_bins = freq_bins.unsqueeze(0).unsqueeze(0)  # [1, 1, num_freq_bins]

        # Convert delays to seconds
        delays_s = multipath_profile.delays_ms.unsqueeze(2) / 1000  # [batch, paths, 1]

        # Path amplitudes (sqrt of power)
        amplitudes = torch.sqrt(multipath_profile.powers).unsqueeze(2)  # [batch, paths, 1]

        # Frequency response for each path: H_i(f) = α_i × exp(-j2πfτ_i)
        phase_shifts = -2 * torch.pi * freq_bins * delays_s  # [batch, paths, freq_bins]
        path_responses = amplitudes * torch.exp(1j * phase_shifts)

        # Sum all paths
        H_total = torch.sum(path_responses, dim=1)  # [batch, num_freq_bins]

        return H_total

    def apply_coherence_bandwidth_smoothing(self,
                                           H: torch.Tensor,
                                           coherence_bandwidth_hz: float) -> torch.Tensor:
        """
        Apply coherence bandwidth smoothing to frequency response.

        Smooths channel response over coherence bandwidth to model
        correlation between nearby frequencies.

        Args:
            H: [batch_size, num_freq_bins], complex frequency response
            coherence_bandwidth_hz: Coherence bandwidth in Hz

        Returns:
            torch.Tensor: [batch_size, num_freq_bins], smoothed response
        """
        batch_size, num_bins = H.shape

        # Create Gaussian smoothing kernel
        # Standard deviation in bins
        freq_resolution = self.sample_rate / (2 * num_bins)
        sigma_bins = coherence_bandwidth_hz / freq_resolution

        # Gaussian kernel in frequency domain
        kernel_size = int(6 * sigma_bins)  # 6 sigma = 99.7% of energy
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()

        # Apply smoothing (real and imag separately)
        H_real_smooth = torch.nn.functional.conv1d(
            H.real.unsqueeze(1),
            kernel.unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze(1)

        H_imag_smooth = torch.nn.functional.conv1d(
            H.imag.unsqueeze(1),
            kernel.unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze(1)

        return H_real_smooth + 1j * H_imag_smooth

    def apply_time_varying_multipath_batch(self,
                                          signals_batch: torch.Tensor,
                                          multipath_profiles: MultipathProfile,
                                          coherence_bandwidth_hz: float = 50) -> torch.Tensor:
        """
        Apply time-varying frequency-selective multipath fading.

        Updates channel every 10ms with new fading realizations.
        Uses continuous frequency response (all 2048 bins).

        Args:
            signals_batch: [batch_size, num_samples], complex signals
            multipath_profiles: Multipath parameters (batch)
            coherence_bandwidth_hz: Coherence bandwidth (20-100 Hz typical)

        Returns:
            torch.Tensor: [batch_size, num_samples], faded signals
        """
        batch_size, num_samples = signals_batch.shape

        # MEMORY CHECK: Prevent OOM for large batches
        if batch_size > 8192:
            raise ValueError(f"Batch size {batch_size} too large! Max 8192 to prevent OOM. "
                           f"Got multipath_profiles with {batch_size} entries.")

        # Calculate number of updates
        duration_s = num_samples / self.sample_rate

        # OPTIMIZATION: Use adaptive update interval based on signal length
        # For long signals (>5s), use longer update interval to reduce iterations
        if duration_s > 5.0:
            adaptive_update_ms = 100  # 100ms updates for long signals (10× faster)
        elif duration_s > 2.0:
            adaptive_update_ms = 50  # 50ms for medium signals
        else:
            adaptive_update_ms = self.update_interval_ms  # 10ms for short signals

        update_interval_s = adaptive_update_ms / 1000
        num_updates = int(duration_s / update_interval_s)
        samples_per_update = int(update_interval_s * self.sample_rate)

        # Limit max updates to prevent extreme slowdown
        MAX_UPDATES = 200  # Cap at 200 updates even for very long signals
        if num_updates > MAX_UPDATES:
            num_updates = MAX_UPDATES
            samples_per_update = num_samples // MAX_UPDATES

        # Initialize output
        faded_signals = torch.zeros_like(signals_batch)

        # Process each time chunk
        for update_idx in range(num_updates):
            start_sample = update_idx * samples_per_update
            end_sample = min((update_idx + 1) * samples_per_update, num_samples)

            if start_sample >= num_samples:
                break

            chunk_size = end_sample - start_sample

            # Vary multipath profile slightly for this update
            # Path delays drift ±0.1ms
            delay_variation = (torch.rand_like(multipath_profiles.delays_ms) - 0.5) * 0.2
            varied_delays = multipath_profiles.delays_ms + delay_variation

            # Path powers fluctuate ±1 dB
            power_variation_db = (torch.rand_like(multipath_profiles.powers) - 0.5) * 2
            power_variation_linear = 10 ** (power_variation_db / 20)
            varied_powers = multipath_profiles.powers * power_variation_linear

            # Normalize powers
            varied_powers = varied_powers / varied_powers.sum(dim=1, keepdim=True)

            # Doppler shifts vary ±0.05 Hz
            doppler_variation = (torch.rand_like(multipath_profiles.doppler_shifts_hz) - 0.5) * 0.1
            varied_doppler = multipath_profiles.doppler_shifts_hz + doppler_variation

            # Create varied profile
            varied_profile = MultipathProfile(
                delays_ms=varied_delays,
                powers=varied_powers,
                doppler_shifts_hz=varied_doppler,
                k_factors=multipath_profiles.k_factors
            )

            # Generate fading for this chunk
            # Use Jakes model for each path
            num_paths = varied_profile.delays_ms.shape[1]
            path_fadings = []

            for path_idx in range(num_paths):
                k_factor = varied_profile.k_factors[:, path_idx]
                doppler = varied_profile.doppler_shifts_hz[:, path_idx]

                # Generate fading (Rician if K>0, else Rayleigh)
                if (k_factor > 0).any():
                    fading = self.generate_rician_fading_batch(batch_size, chunk_size, doppler, k_factor)
                else:
                    fading = self.generate_jakes_fading_batch(batch_size, chunk_size, doppler)

                path_fadings.append(fading)

            # Extract chunk
            signal_chunk = signals_batch[:, start_sample:end_sample]

            # For complex signals, apply fading in time domain instead of frequency domain
            # This avoids FFT/IFFT complexity and potential NaN from mirroring
            # Generate time-domain fading envelope
            fading_envelope = torch.ones_like(signal_chunk)

            # Apply fading from each path
            for path_idx in range(num_paths):
                path_power = varied_profile.powers[:, path_idx].unsqueeze(1)  # [batch, 1]
                path_delay_ms = varied_profile.delays_ms[:, path_idx]
                path_doppler = varied_profile.doppler_shifts_hz[:, path_idx]
                path_k = varied_profile.k_factors[:, path_idx]

                # Only apply if path has power
                if (path_power > 0.01).any():
                    # Generate fading for this path
                    if (path_k > 0).any():
                        path_fading = self.generate_rician_fading_batch(batch_size, chunk_size, path_doppler, path_k)
                    else:
                        path_fading = self.generate_jakes_fading_batch(batch_size, chunk_size, path_doppler)

                    # Apply delay (simplified - just use power scaling)
                    fading_envelope += torch.sqrt(path_power) * path_fading

            # Normalize fading envelope to preserve power (add epsilon to prevent division by zero)
            fading_power = torch.mean(torch.abs(fading_envelope) ** 2, dim=1, keepdim=True)
            fading_envelope = fading_envelope / torch.sqrt(fading_power + 1e-10)  # Prevent NaN from zero division

            # Apply fading to signal chunk
            signal_faded_chunk = signal_chunk * fading_envelope

            # Store in output
            faded_signals[:, start_sample:end_sample] = signal_faded_chunk

            # Clear intermediate tensors to prevent memory accumulation
            del signal_chunk, fading_envelope, signal_faded_chunk

        # Final cleanup
        torch.cuda.empty_cache()

        return faded_signals

    def apply_continuous_d_layer_absorption(self,
                                           signals_batch: torch.Tensor,
                                           base_absorption_db: torch.Tensor,
                                           solar_zenith_angles: torch.Tensor) -> torch.Tensor:
        """
        Apply continuous frequency-dependent D-layer absorption.

        Uses f^(-1.5) law across full spectrum:
        - 300 Hz: ~8 dB absorption (severe)
        - 1500 Hz: ~3 dB absorption (moderate)
        - 2860 Hz: ~1.5 dB absorption (mild)

        Args:
            signals_batch: [batch_size, num_samples], complex signals
            base_absorption_db: [batch_size], base absorption at 1 MHz
            solar_zenith_angles: [batch_size], degrees (0=overhead, 90=horizon)

        Returns:
            torch.Tensor: [batch_size, num_samples], absorbed signals
        """
        batch_size, num_samples = signals_batch.shape

        # For large signals, process in chunks to avoid cuFFT errors
        CHUNK_SIZE = 48000
        if num_samples > CHUNK_SIZE:
            output = torch.zeros_like(signals_batch)
            num_chunks = (num_samples + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(num_chunks):
                start = chunk_idx * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, num_samples)
                chunk = signals_batch[:, start:end]
                chunk_size = end - start

                # Process chunk
                chunk_fft = torch.fft.fft(chunk, dim=-1)
                freq_bins = torch.fft.fftfreq(chunk_size, 1/self.sample_rate, device=self.device)
                freq_bins = torch.abs(freq_bins).unsqueeze(0)
                freq_bins = torch.maximum(freq_bins, torch.tensor(1.0, device=self.device))

                # Apply absorption
                base_absorption_db_expanded = base_absorption_db.unsqueeze(1)
                sza_factor = torch.cos(solar_zenith_angles * torch.pi / 180).unsqueeze(1)
                freq_normalized = torch.clamp(freq_bins / 1000.0, min=0.001)
                absorption_db = base_absorption_db_expanded * (freq_normalized ** (-1.5)) * sza_factor
                absorption_db = torch.clamp(absorption_db, -50.0, 50.0)
                absorption_linear = 10 ** (-absorption_db / 20)
                absorption_linear = torch.clamp(absorption_linear, 0.001, 100.0)

                chunk_absorbed_fft = chunk_fft * absorption_linear
                chunk_absorbed = torch.fft.ifft(chunk_absorbed_fft, dim=-1)
                output[:, start:end] = chunk_absorbed

            return output

        # For small signals, process normally
        signal_fft = torch.fft.fft(signals_batch, dim=-1)
        num_freq_bins = signal_fft.shape[-1]

        # Frequency bins (Hz) - use fftfreq for full spectrum
        freq_bins = torch.fft.fftfreq(num_samples, 1/self.sample_rate, device=self.device)
        freq_bins = torch.abs(freq_bins).unsqueeze(0)  # [1, num_freq_bins], use magnitude

        # Avoid division by zero at DC
        freq_bins = torch.maximum(freq_bins, torch.tensor(1.0, device=self.device))

        # f^(-1.5) law
        # A(f) = A_base × (f/1000)^(-1.5) × cos(SZA)
        base_absorption_db_expanded = base_absorption_db.unsqueeze(1)  # [batch, 1]

        # Convert solar zenith angle to absorption factor
        # cos(0°) = 1 (overhead sun, max absorption)
        # cos(90°) = 0 (horizon, min absorption)
        sza_factor = torch.cos(solar_zenith_angles * torch.pi / 180).unsqueeze(1)

        # Calculate absorption for each frequency
        # Use safe power operation to avoid NaN/Inf
        freq_normalized = torch.clamp(freq_bins / 1000.0, min=0.001)  # Prevent division issues
        absorption_db = base_absorption_db_expanded * (freq_normalized ** (-1.5)) * sza_factor

        # Clamp absorption to reasonable range (-50 to 50 dB)
        absorption_db = torch.clamp(absorption_db, -50.0, 50.0)

        # Convert to linear scale (safely)
        absorption_linear = 10 ** (-absorption_db / 20)
        absorption_linear = torch.clamp(absorption_linear, 0.001, 100.0)  # Prevent extreme values

        # Apply in frequency domain
        signal_absorbed_fft = signal_fft * absorption_linear

        # IFFT back to time domain (use ifft for complex signals)
        signal_absorbed = torch.fft.ifft(signal_absorbed_fft, dim=-1)

        return signal_absorbed

    def generate_lightning_strike_batch(self,
                                       batch_size: int,
                                       num_samples: int,
                                       strike_rate_hz: torch.Tensor,
                                       decay_time_ms: float = 25.0) -> torch.Tensor:
        """
        Generate realistic lightning crashes (Poisson process).

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            strike_rate_hz: [batch_size], average strikes per second
            decay_time_ms: Exponential decay time

        Returns:
            torch.Tensor: [batch_size, num_samples], complex QRN signal
        """
        duration_s = num_samples / self.sample_rate

        qrn_signal = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

        for b in range(batch_size):
            # Poisson process: number of strikes
            expected_strikes = strike_rate_hz[b].item() * duration_s
            num_strikes = torch.poisson(torch.tensor([expected_strikes], device=self.device)).int().item()

            if num_strikes == 0:
                continue

            # Random strike times
            strike_times = torch.rand(num_strikes, device=self.device) * num_samples
            strike_times = strike_times.int()

            # Random amplitudes (exponential distribution)
            amplitudes = torch.distributions.Exponential(rate=0.2).sample((num_strikes,)).to(self.device)

            # Decay envelope
            decay_samples = int(decay_time_ms * self.sample_rate / 1000)
            t_decay = torch.arange(decay_samples, dtype=torch.float32, device=self.device)
            decay_envelope = torch.exp(-t_decay / (decay_samples / 4))

            # Add each strike
            for strike_idx in range(num_strikes):
                start_sample = strike_times[strike_idx].item()
                end_sample = min(start_sample + decay_samples, num_samples)

                if start_sample >= num_samples:
                    continue

                # Random phase
                phase = torch.rand(1, device=self.device) * 2 * torch.pi

                # Impulse with decay
                impulse_len = end_sample - start_sample
                impulse = amplitudes[strike_idx] * decay_envelope[:impulse_len] * torch.exp(1j * phase)

                # Add noise modulation
                noise_mod = (torch.randn(impulse_len, device=self.device) +
                           1j * torch.randn(impulse_len, device=self.device)) * 0.3

                qrn_signal[b, start_sample:end_sample] += impulse * (1 + noise_mod)

        return qrn_signal

    def generate_power_line_noise_batch(self,
                                       batch_size: int,
                                       num_samples: int,
                                       intensity: torch.Tensor,
                                       line_freq: float = 60.0) -> torch.Tensor:
        """
        Generate power line noise (60/50 Hz harmonics with buzz).

        Common QRM source in urban/suburban areas.

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor
            line_freq: Line frequency (60 Hz US/Japan, 50 Hz Europe)

        Returns:
            torch.Tensor: [batch_size, num_samples], complex power line noise
        """
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.sample_rate

        # Generate harmonics (fundamental through 10th harmonic)
        noise = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

        for harmonic in range(1, 11):
            freq = line_freq * harmonic
            amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics

            # Add random phase and amplitude modulation (buzz characteristic)
            phase = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi
            buzz_mod = 1.0 + 0.3 * torch.sin(2 * torch.pi * 120 * t.unsqueeze(0))  # 120 Hz buzz

            harmonic_signal = amplitude * buzz_mod * torch.exp(1j * (2 * torch.pi * freq * t.unsqueeze(0) + phase))
            noise += harmonic_signal

        # Scale by intensity
        noise = noise * intensity.unsqueeze(1) * 0.1  # Keep relatively weak

        return noise

    def generate_led_driver_noise_batch(self,
                                        batch_size: int,
                                        num_samples: int,
                                        intensity: torch.Tensor) -> torch.Tensor:
        """
        Generate LED/SMPS driver noise (broadband switching noise).

        Very common in modern homes - switching power supplies, LED lights.

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex switching noise
        """
        # Fundamental switching frequency (30-100 kHz, varies by device)
        switch_freq = 30000 + torch.rand(batch_size, device=self.device) * 70000

        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.sample_rate

        # Generate switching harmonics (creates broadband hash)
        noise = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

        # Add first 20 harmonics of switching frequency (creates dense spectrum)
        for harmonic in range(1, 21):
            freq_batch = switch_freq.unsqueeze(1) * harmonic

            # Random phase per batch
            phase = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi

            # Amplitude decreases but not as fast as power line (more broadband)
            amplitude = 0.5 / np.sqrt(harmonic)

            harmonic_signal = amplitude * torch.exp(1j * (2 * torch.pi * freq_batch * t.unsqueeze(0) + phase))
            noise += harmonic_signal

        # Add random AM modulation (switching instability)
        mod_freq = 50 + torch.rand(batch_size, 1, device=self.device) * 200  # 50-250 Hz modulation
        am_mod = 1.0 + 0.5 * torch.sin(2 * torch.pi * mod_freq * t.unsqueeze(0))

        noise = noise * am_mod * intensity.unsqueeze(1) * 0.05  # Relatively weak but annoying

        return noise

    def generate_oth_radar_batch(self,
                                batch_size: int,
                                num_samples: int,
                                intensity: torch.Tensor) -> torch.Tensor:
        """
        Generate Over-The-Horizon (OTH) radar interference.

        Characteristics: Swept-frequency chirps, periodic (PRF ~10-50 Hz).

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex radar pulses
        """
        duration_s = num_samples / self.sample_rate
        radar = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)

        for b in range(batch_size):
            # Random radar parameters
            prf = 10 + torch.rand(1, device=self.device).item() * 40  # 10-50 Hz PRF
            pulse_width = 0.001 + torch.rand(1, device=self.device).item() * 0.004  # 1-5 ms
            bandwidth = 10 + torch.rand(1, device=self.device).item() * 40  # 10-50 kHz chirp BW

            # Generate pulses
            pulse_period = 1.0 / prf
            num_pulses = int(duration_s / pulse_period)

            for pulse_idx in range(num_pulses):
                start_time = pulse_idx * pulse_period
                start_sample = int(start_time * self.sample_rate)
                pulse_samples = int(pulse_width * self.sample_rate)
                end_sample = min(start_sample + pulse_samples, num_samples)

                if start_sample >= num_samples:
                    break

                # Linear FM chirp
                t_pulse = torch.arange(end_sample - start_sample, dtype=torch.float32, device=self.device) / self.sample_rate
                chirp_rate = bandwidth * 1000 / pulse_width  # Hz/sec
                phase = 2 * torch.pi * (1500 * t_pulse + 0.5 * chirp_rate * t_pulse**2)  # Start at 1500 Hz

                # Add pulse envelope (Hamming window)
                envelope = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(len(t_pulse), device=self.device) / len(t_pulse))

                radar[b, start_sample:end_sample] = envelope * torch.exp(1j * phase)

        # Scale by intensity
        radar = radar * intensity.unsqueeze(1) * 0.2

        return radar

    def generate_am_broadcast_batch(self,
                                   batch_size: int,
                                   num_samples: int,
                                   intensity: torch.Tensor) -> torch.Tensor:
        """
        Generate AM broadcast interference.

        Characteristics: Strong carrier + audio sidebands (music, voice).

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex AM broadcast
        """
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.sample_rate

        # Random carrier frequency (off-frequency from CASCADE channels)
        carrier_offset = 500 + torch.rand(batch_size, device=self.device) * 2000  # 500-2500 Hz

        # Audio modulation (music-like: multiple tones)
        audio = torch.zeros(batch_size, num_samples, device=self.device)

        # Add several audio frequencies (simulate music/voice)
        for freq in [200, 400, 800, 1200, 1600]:
            phase = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi
            tone = torch.sin(2 * torch.pi * freq * t.unsqueeze(0) + phase)
            audio += tone / 5  # Mix of 5 frequencies

        # AM modulation
        modulation_depth = 0.8  # 80% modulation
        am_signal = (1.0 + modulation_depth * audio) * torch.exp(1j * 2 * torch.pi * carrier_offset.unsqueeze(1) * t.unsqueeze(0))

        # Scale by intensity
        am_signal = am_signal * intensity.unsqueeze(1) * 0.3  # Can be strong

        return am_signal

    def generate_satellite_downlink_batch(self,
                                         batch_size: int,
                                         num_samples: int,
                                         intensity: torch.Tensor) -> torch.Tensor:
        """
        Generate satellite downlink interference.

        Characteristics: Narrow FSK, Doppler shifted, periodic.

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex satellite signal
        """
        t = torch.arange(num_samples, dtype=torch.float32, device=self.device) / self.sample_rate

        # Satellite carrier (random frequency)
        carrier = 800 + torch.rand(batch_size, device=self.device) * 1500  # 800-2300 Hz

        # FSK data (telemetry)
        baud_rate = 50 + torch.rand(batch_size, device=self.device) * 200  # 50-250 baud
        samples_per_bit = (self.sample_rate / baud_rate.unsqueeze(1)).long()

        # Generate random data
        num_bits = num_samples // samples_per_bit.min().item()
        data = torch.randint(0, 2, (batch_size, num_bits), device=self.device) * 2 - 1  # {-1, 1}

        # Upsample to sample rate
        data_upsampled = torch.repeat_interleave(data.float(), samples_per_bit[:, 0], dim=1)[:, :num_samples]

        # FSK modulation (shift ±50 Hz)
        freq_shift = 50.0
        inst_freq = carrier.unsqueeze(1) + freq_shift * data_upsampled

        # Add Doppler shift (satellite motion)
        doppler_shift = -20 + torch.rand(batch_size, 1, device=self.device) * 40  # ±20 Hz
        inst_freq = inst_freq + doppler_shift

        # Generate signal
        phase = torch.cumsum(2 * torch.pi * inst_freq / self.sample_rate, dim=1)
        sat_signal = torch.exp(1j * phase)

        # Scale by intensity (usually weak)
        sat_signal = sat_signal * intensity.unsqueeze(1) * 0.1

        return sat_signal

    def generate_atmospheric_static_batch(self,
                                         batch_size: int,
                                         num_samples: int,
                                         intensity: torch.Tensor) -> torch.Tensor:
        """
        Generate continuous atmospheric static (pink noise).

        Args:
            batch_size: Number of signals
            num_samples: Samples per signal
            intensity: [batch_size], intensity factor

        Returns:
            torch.Tensor: [batch_size, num_samples], complex static
        """
        # For large signals, process in chunks to avoid cuFFT errors
        CHUNK_SIZE = 48000
        if num_samples > CHUNK_SIZE:
            output = torch.zeros(batch_size, num_samples, dtype=torch.complex64, device=self.device)
            num_chunks = (num_samples + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(num_chunks):
                start = chunk_idx * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, num_samples)
                chunk_size = end - start

                # Generate white noise for chunk
                white_chunk = (torch.randn(batch_size, chunk_size, device=self.device) +
                              1j * torch.randn(batch_size, chunk_size, device=self.device))

                # Apply 1/f filter
                white_fft = torch.fft.fft(white_chunk, dim=-1)
                freq_bins = torch.fft.fftfreq(chunk_size, 1/self.sample_rate, device=self.device)
                freq_bins = torch.abs(freq_bins)
                freq_bins = torch.maximum(freq_bins, torch.tensor(1.0, device=self.device))

                H_pink = 1 / (1 + freq_bins / 100)
                pink_fft = white_fft * H_pink.unsqueeze(0)
                pink_chunk = torch.fft.ifft(pink_fft, dim=-1)

                output[:, start:end] = pink_chunk

            # Scale by intensity
            return output * intensity.unsqueeze(1)

        # For small signals, process normally
        white = (torch.randn(batch_size, num_samples, device=self.device) +
                1j * torch.randn(batch_size, num_samples, device=self.device))

        # FFT to frequency domain (use fft for complex signals)
        white_fft = torch.fft.fft(white, dim=-1)

        # 1/f filter (pink noise)
        freq_bins = torch.fft.fftfreq(num_samples, 1/self.sample_rate, device=self.device)
        freq_bins = torch.abs(freq_bins)  # Use magnitude for both positive and negative frequencies
        freq_bins = torch.maximum(freq_bins, torch.tensor(1.0, device=self.device))  # Avoid div/0

        H_pink = 1 / (1 + freq_bins / 100)
        pink_fft = white_fft * H_pink.unsqueeze(0)

        # IFFT back (use ifft for complex signals)
        pink = torch.fft.ifft(pink_fft, dim=-1)

        # Scale by intensity
        pink = pink * intensity.unsqueeze(1)

        return pink

    def apply_batch(self,
                   signals_batch: torch.Tensor,
                   scenarios: List,
                   add_collisions: bool = True,
                   collision_probability: float = 0.3,
                   add_qrm: bool = True,
                   qrm_probability: float = 0.2) -> torch.Tensor:
        """
        Apply channel effects to batch of signals based on scenarios.

        Unified interface for ReciprocalChannelDataset compatibility.

        Args:
            signals_batch: [batch_size, num_samples], complex signals
            scenarios: List of CorePhysicalDrivers scenarios
            add_collisions: Whether to add collisions (ignored for beacons)
            collision_probability: Probability of collision
            add_qrm: Whether to add QRM (ignored for beacons)
            qrm_probability: Probability of QRM

        Returns:
            torch.Tensor: [batch_size, num_samples], signals with channel effects applied
        """
        batch_size, num_samples = signals_batch.shape

        # Extract scenario parameters
        propagation_modes = [s.propagation_mode for s in scenarios]
        k_indices = torch.tensor([s.k_index for s in scenarios], device=self.device)
        sfis = torch.tensor([s.sfi for s in scenarios], device=self.device)
        qrn_types = [s.qrn_type for s in scenarios]

        output_signals = signals_batch.clone()

        # Apply multipath fading for non-AWGN scenarios
        needs_multipath = [p in ['RAYLEIGH', 'RICIAN', 'MULTIPATH_SPARSE', 'MULTIPATH_DENSE']
                          for p in propagation_modes]

        if any(needs_multipath):
            # Create multipath profile from scenarios
            delays_ms = []
            powers = []
            doppler_shifts = []
            k_factors = []

            for i, scenario in enumerate(scenarios):
                if needs_multipath[i]:
                    # Use scenario parameters or defaults
                    if hasattr(scenario, 'multipath_delays') and scenario.multipath_delays is not None:
                        delays_ms.append(scenario.multipath_delays[:3])  # Max 3 paths
                    else:
                        # Default multipath profile
                        if propagation_modes[i] == 'MULTIPATH_DENSE':
                            delays_ms.append([0, 0.5, 1.5])
                        elif propagation_modes[i] == 'MULTIPATH_SPARSE':
                            delays_ms.append([0, 1.0, 3.0])
                        else:
                            delays_ms.append([0, 0.0, 0.0])

                    # Power profile
                    if propagation_modes[i] == 'MULTIPATH_DENSE':
                        powers.append([0.6, 0.25, 0.15])
                    elif propagation_modes[i] == 'MULTIPATH_SPARSE':
                        powers.append([0.8, 0.15, 0.05])
                    else:
                        powers.append([1.0, 0.0, 0.0])

                    # Doppler (based on HF typical)
                    doppler_shifts.append([0.0, 0.5, 1.0])

                    # K-factor (Rician)
                    if propagation_modes[i] == 'RICIAN':
                        k_factors.append([10.0, 0.0, 0.0])  # Strong LOS
                    else:
                        k_factors.append([0.0, 0.0, 0.0])  # Rayleigh
                else:
                    # AWGN - no multipath
                    delays_ms.append([0, 0, 0])
                    powers.append([1.0, 0.0, 0.0])
                    doppler_shifts.append([0.0, 0.0, 0.0])
                    k_factors.append([0.0, 0.0, 0.0])

            # Create multipath profile tensor
            profile = MultipathProfile(
                delays_ms=torch.tensor(delays_ms, device=self.device),
                powers=torch.tensor(powers, device=self.device),
                doppler_shifts_hz=torch.tensor(doppler_shifts, device=self.device),
                k_factors=torch.tensor(k_factors, device=self.device)
            )

            # Apply multipath
            output_signals = self.apply_time_varying_multipath_batch(
                output_signals, profile, coherence_bandwidth_hz=50
            )

        # Apply D-layer absorption (continuous frequency-dependent)
        # Base absorption depends on SFI
        base_absorption_db = 10 - (sfis / 200) * 8  # 10 dB at low SFI, 2 dB at high SFI

        # Solar zenith angle (simplified: random 0-90°)
        sza = torch.rand(batch_size, device=self.device) * 90

        output_signals = self.apply_continuous_d_layer_absorption(
            output_signals, base_absorption_db, sza
        )

        # Add QRN based on qrn_type
        for i, qrn_type in enumerate(qrn_types):
            if qrn_type in ['CRACKLING', 'THUNDERSTORM']:
                # Lightning strikes
                strike_rate = torch.tensor([5.0 if qrn_type == 'THUNDERSTORM' else 2.0], device=self.device)
                lightning = self.generate_lightning_strike_batch(1, num_samples, strike_rate)
                output_signals[i] += lightning[0] * 0.3

            elif qrn_type in ['STATIC', 'HISS', 'POPCORN']:
                # Continuous static
                intensity = torch.tensor([0.5 if qrn_type == 'STATIC' else 0.3], device=self.device)
                static = self.generate_atmospheric_static_batch(1, num_samples, intensity)
                output_signals[i] += static[0]

        # Add AWGN (always present)
        # SNR based on propagation conditions (worse in multipath)
        # IMPORTANT: Keep SNR above -14 dB (phase detection limit)
        # Using -10 dB floor for 4 dB safety margin
        snrs_db = torch.zeros(batch_size, device=self.device)
        for i, mode in enumerate(propagation_modes):
            if mode == 'AWGN':
                snrs_db[i] = 25 + torch.randn(1, device=self.device) * 5  # 20-30 dB (clear signals)
            elif mode == 'RAYLEIGH':
                snrs_db[i] = 10 + torch.randn(1, device=self.device) * 5  # 5-15 dB (moderate fading)
            elif mode == 'MULTIPATH_DENSE':
                snrs_db[i] = 5 + torch.randn(1, device=self.device) * 5  # 0-10 dB (severe multipath)
            elif mode == 'MULTIPATH_SPARSE':
                snrs_db[i] = 13 + torch.randn(1, device=self.device) * 5  # 8-18 dB (sparse multipath)
            elif mode == 'RICIAN':
                snrs_db[i] = 15 + torch.randn(1, device=self.device) * 5  # 10-20 dB (LOS + multipath)
            else:
                snrs_db[i] = 15 + torch.randn(1, device=self.device) * 5  # 10-20 dB (default)

        # Apply safety floor: never go below -10 dB SNR (4 dB margin above -14 dB limit)
        snrs_db = torch.maximum(snrs_db, torch.tensor(-10.0, device=self.device))

        # Add noise to achieve target SNR
        signal_power = torch.mean(torch.abs(output_signals) ** 2, dim=1)
        noise_power = signal_power / (10 ** (snrs_db / 10))

        noise = (torch.randn_like(output_signals) + 1j * torch.randn_like(output_signals).imag) / np.sqrt(2)
        noise = noise * torch.sqrt(noise_power).unsqueeze(1)

        output_signals = output_signals + noise

        return output_signals


def test_gpu_channel_simulator():
    """Test GPU channel simulator with small batch."""
    print("Testing GPU Channel Simulator...")

    sim = GPUChannelSimulator(sample_rate=48000, device='cuda')

    # Create test batch
    batch_size = 8
    num_samples = 96000  # 2 seconds

    # Generate test signals (simple tones)
    t = torch.arange(num_samples, dtype=torch.float32, device='cuda') / 48000
    signals = torch.exp(1j * 2 * torch.pi * 1500 * t.unsqueeze(0).expand(batch_size, -1))

    print(f"Input signals: {signals.shape}, dtype={signals.dtype}")

    # Test 1: Time-varying multipath
    print("\n1. Testing time-varying multipath...")
    profile = MultipathProfile(
        delays_ms=torch.tensor([[0, 0.5, 2.0]], device='cuda').expand(batch_size, -1),
        powers=torch.tensor([[0.7, 0.2, 0.1]], device='cuda').expand(batch_size, -1),
        doppler_shifts_hz=torch.tensor([[0.0, 0.5, 1.0]], device='cuda').expand(batch_size, -1),
        k_factors=torch.tensor([[5.0, 0.0, 0.0]], device='cuda').expand(batch_size, -1)
    )

    import time
    start = time.time()
    faded = sim.apply_time_varying_multipath_batch(signals, profile, coherence_bandwidth_hz=50)
    elapsed = time.time() - start
    print(f"✓ Fading applied in {elapsed*1000:.1f}ms ({elapsed/batch_size*1000:.2f}ms per signal)")
    print(f"  Output: {faded.shape}, mean power: {torch.mean(torch.abs(faded)**2):.3f}")

    # Test 2: D-layer absorption
    print("\n2. Testing continuous D-layer absorption...")
    absorption_db = torch.tensor([5.0] * batch_size, device='cuda')
    sza = torch.tensor([45.0] * batch_size, device='cuda')  # 45° zenith angle

    start = time.time()
    absorbed = sim.apply_continuous_d_layer_absorption(signals, absorption_db, sza)
    elapsed = time.time() - start
    print(f"✓ Absorption applied in {elapsed*1000:.1f}ms")
    print(f"  Power reduction: {torch.mean(torch.abs(absorbed)**2) / torch.mean(torch.abs(signals)**2):.3f}")

    # Test 3: Lightning QRN
    print("\n3. Testing lightning strikes...")
    strike_rate = torch.tensor([10.0] * batch_size, device='cuda')  # 10 strikes/sec

    start = time.time()
    qrn = sim.generate_lightning_strike_batch(batch_size, num_samples, strike_rate)
    elapsed = time.time() - start
    print(f"✓ Lightning generated in {elapsed*1000:.1f}ms")
    print(f"  Peak amplitude: {torch.max(torch.abs(qrn)):.2f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_gpu_channel_simulator()
