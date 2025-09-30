"""Multi-channel QRN extraction for enhanced noise characterization.

Implements T032a: Multi-channel QRN extractor (FR-029).
Extracts 9 overlapping 2.5kHz channels from 12kHz IQ with 50% overlap
for frequency-dependent noise analysis.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""
    id: int
    center: float  # Hz offset from IQ center
    start: float   # Hz
    end: float     # Hz
    bandwidth: float  # Hz


class MultiChannelQRNExtractor:
    """Extracts multiple overlapping channels from IQ data (FR-029)."""

    def __init__(
        self,
        sample_rate: int = 12000,
        channel_width: int = 2500,
        overlap: float = 0.5,
    ):
        """Initialize multi-channel extractor.

        Args:
            sample_rate: IQ sample rate in Hz
            channel_width: Channel bandwidth in Hz
            overlap: Overlap fraction (0-1)
        """
        self.sample_rate = sample_rate
        self.channel_width = channel_width
        self.overlap = overlap

        # Calculate channel spacing
        self.channel_spacing = int(channel_width * (1 - overlap))

        # Generate channel configuration
        self.channels = self._generate_channel_config()

        logger.info(
            f"Multi-channel QRN extractor initialized: {len(self.channels)} channels, "
            f"{channel_width}Hz width, {overlap*100}% overlap"
        )

    def _generate_channel_config(self) -> List[ChannelConfig]:
        """Generate configuration for all channels.

        Returns:
            List of channel configurations
        """
        channels = []

        # IQ bandwidth is sample_rate
        total_bandwidth = self.sample_rate
        half_bandwidth = total_bandwidth / 2

        # Start from negative frequency
        current_center = -half_bandwidth + self.channel_width / 2

        channel_id = 0

        while current_center + self.channel_width / 2 <= half_bandwidth:
            start_freq = current_center - self.channel_width / 2
            end_freq = current_center + self.channel_width / 2

            channels.append(
                ChannelConfig(
                    id=channel_id,
                    center=current_center,
                    start=start_freq,
                    end=end_freq,
                    bandwidth=self.channel_width,
                )
            )

            current_center += self.channel_spacing
            channel_id += 1

        return channels

    async def get_channel_config(self) -> List[Dict[str, Any]]:
        """Get channel configuration.

        Returns:
            List of channel configs as dicts
        """
        return [
            {
                "id": ch.id,
                "center": ch.center,
                "start": ch.start,
                "end": ch.end,
                "bandwidth": ch.bandwidth,
            }
            for ch in self.channels
        ]

    async def extract_channels(
        self, iq_data: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Extract all channels from IQ data (FR-029).

        Args:
            iq_data: Complex IQ samples

        Returns:
            List of channel data dicts
        """
        logger.info(f"Extracting {len(self.channels)} channels from {len(iq_data)} samples")

        channels_data = []

        for channel_config in self.channels:
            # Extract this channel
            channel_iq = await self._extract_single_channel(
                iq_data, channel_config
            )

            channels_data.append({
                "id": channel_config.id,
                "center": channel_config.center,
                "start": channel_config.start,
                "end": channel_config.end,
                "bandwidth": channel_config.bandwidth,
                "data": channel_iq,
            })

        return channels_data

    async def _extract_single_channel(
        self,
        iq_data: np.ndarray,
        channel: ChannelConfig,
    ) -> np.ndarray:
        """Extract a single channel from IQ data.

        Args:
            iq_data: Complex IQ samples
            channel: Channel configuration

        Returns:
            Filtered and decimated IQ for this channel
        """
        # Frequency shift to center this channel at DC
        if channel.center != 0:
            t = np.arange(len(iq_data)) / self.sample_rate
            shift_signal = np.exp(-1j * 2 * np.pi * channel.center * t)
            shifted_iq = iq_data * shift_signal
        else:
            shifted_iq = iq_data

        # Design low-pass filter for channel bandwidth
        nyquist = self.sample_rate / 2
        cutoff = channel.bandwidth / 2
        normalized_cutoff = cutoff / nyquist

        # Butterworth filter
        sos = signal.butter(
            6,  # Order
            normalized_cutoff,
            btype="low",
            output="sos",
        )

        # Apply filter
        filtered_iq = signal.sosfilt(sos, shifted_iq)

        # Decimate to channel sample rate
        # Target: 2.5kHz bandwidth -> 5kHz sample rate (Nyquist)
        target_rate = int(channel.bandwidth * 2)
        decimation_factor = self.sample_rate // target_rate

        if decimation_factor > 1:
            # Decimate
            decimated_iq = signal.decimate(
                filtered_iq,
                decimation_factor,
                ftype="fir",
                zero_phase=True,
            )
        else:
            decimated_iq = filtered_iq

        return decimated_iq

    async def calculate_channel_statistics(
        self, channel_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate power statistics for a channel.

        Args:
            channel_data: Complex IQ for channel

        Returns:
            Power statistics
        """
        # Calculate power
        power = np.abs(channel_data) ** 2
        power_dbm = 10 * np.log10(power + 1e-12) + 30  # Convert to dBm

        # Statistics
        mean_power = float(np.mean(power_dbm))
        peak_power = float(np.max(power_dbm))

        # Estimate noise floor (lower percentile)
        noise_floor = float(np.percentile(power_dbm, 10))

        # Dynamic range
        dynamic_range = peak_power - noise_floor

        return {
            "mean_power_dbm": mean_power,
            "peak_power_dbm": peak_power,
            "noise_floor_dbm": noise_floor,
            "dynamic_range_db": dynamic_range,
        }

    async def extract_spectral_features(
        self, channel_data: np.ndarray
    ) -> Dict[str, float]:
        """Extract spectral features from channel.

        Args:
            channel_data: Complex IQ for channel

        Returns:
            Spectral features
        """
        # Compute power spectral density
        freqs, psd = signal.welch(
            channel_data,
            fs=self.channel_width * 2,  # Channel sample rate
            nperseg=min(256, len(channel_data) // 4),
        )

        # Convert to linear scale
        psd_linear = psd / np.sum(psd)

        # Spectral centroid
        spectral_centroid = float(np.sum(freqs * psd_linear))

        # Spectral bandwidth (standard deviation)
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_linear))
        )

        # Spectral rolloff (95% energy)
        cumsum = np.cumsum(psd_linear)
        rolloff_idx = np.where(cumsum >= 0.95)[0][0] if len(np.where(cumsum >= 0.95)[0]) > 0 else len(freqs) - 1
        spectral_rolloff = float(freqs[rolloff_idx])

        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-12)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-12))

        # Zero crossing rate
        magnitude = np.abs(channel_data)
        zero_crossings = np.sum(np.diff(np.sign(magnitude)) != 0)
        zero_crossing_rate = float(zero_crossings / len(magnitude))

        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flatness": min(1.0, spectral_flatness),
            "zero_crossing_rate": zero_crossing_rate,
        }

    async def extract_temporal_features(
        self, channel_data: np.ndarray
    ) -> Dict[str, float]:
        """Extract temporal features from channel.

        Args:
            channel_data: Complex IQ for channel

        Returns:
            Temporal features
        """
        # Envelope (magnitude)
        envelope = np.abs(channel_data)

        # Envelope statistics
        envelope_mean = float(np.mean(envelope))
        envelope_std = float(np.std(envelope))

        # Crest factor (peak / RMS)
        rms = float(np.sqrt(np.mean(envelope ** 2)))
        peak = float(np.max(envelope))
        crest_factor = peak / (rms + 1e-12)

        # Peak to average ratio
        avg = float(np.mean(envelope))
        peak_to_average = peak / (avg + 1e-12)

        return {
            "envelope_mean": envelope_mean,
            "envelope_std": envelope_std,
            "crest_factor": crest_factor,
            "peak_to_average_ratio": peak_to_average,
        }

    async def calculate_correlation_matrix(
        self, channels: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate inter-channel correlation matrix.

        Args:
            channels: List of channel data dicts

        Returns:
            Correlation matrix (NxN)
        """
        n_channels = len(channels)
        correlation_matrix = np.zeros((n_channels, n_channels))

        # Extract magnitude for each channel
        magnitudes = []
        for channel in channels:
            magnitude = np.abs(channel["data"])
            magnitudes.append(magnitude)

        # Calculate correlations
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Resample to same length if needed
                    mag_i = magnitudes[i]
                    mag_j = magnitudes[j]

                    min_len = min(len(mag_i), len(mag_j))
                    mag_i = mag_i[:min_len]
                    mag_j = mag_j[:min_len]

                    # Calculate correlation
                    correlation = np.corrcoef(mag_i, mag_j)[0, 1]
                    correlation_matrix[i, j] = correlation

        return correlation_matrix

    async def measure_phase_coherence(
        self, channels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Measure phase coherence between channels.

        Args:
            channels: List of channel data dicts

        Returns:
            Phase coherence metrics
        """
        n_channels = len(channels)
        pairwise_coherence = []

        # Calculate phase difference between adjacent channels
        for i in range(n_channels - 1):
            ch1_data = channels[i]["data"]
            ch2_data = channels[i + 1]["data"]

            # Resample to same length
            min_len = min(len(ch1_data), len(ch2_data))
            ch1_data = ch1_data[:min_len]
            ch2_data = ch2_data[:min_len]

            # Calculate phase difference
            phase_diff = np.angle(ch1_data) - np.angle(ch2_data)

            # Wrap to [-pi, pi]
            phase_diff = np.angle(np.exp(1j * phase_diff))

            # Calculate coherence (inverse of phase variance)
            phase_var = np.var(phase_diff)
            coherence = 1.0 / (1.0 + phase_var)

            pairwise_coherence.append(float(coherence))

        mean_coherence = float(np.mean(pairwise_coherence))

        return {
            "mean_coherence": mean_coherence,
            "pairwise_coherence": pairwise_coherence,
        }

    async def estimate_snr(self, channel_data: np.ndarray) -> float:
        """Estimate SNR for channel.

        Args:
            channel_data: Complex IQ for channel

        Returns:
            SNR in dB
        """
        # Calculate power
        power = np.abs(channel_data) ** 2

        # Estimate signal power (upper percentile)
        signal_power = np.percentile(power, 90)

        # Estimate noise power (lower percentile)
        noise_power = np.percentile(power, 10)

        # Calculate SNR
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = float(10 * np.log10(snr_linear))

        return snr_db