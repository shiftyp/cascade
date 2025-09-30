"""QRN (atmospheric noise) processor for CASCADE."""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


@dataclass
class QRNCharacteristics:
    """Characteristics of QRN in a recording."""
    timestamp: datetime
    frequency: float
    bandwidth: float
    noise_floor_dbm: float
    peak_power_dbm: float
    average_power_dbm: float
    impulse_rate: float  # Impulses per second
    spectral_occupancy: float  # Percentage of spectrum occupied
    kurtosis: float
    skewness: float
    noise_type: str  # "static", "impulse", "burst", "mixed"


class QRNProcessor:
    """Processes and analyzes QRN (atmospheric noise) from recordings."""

    def __init__(self):
        """Initialize QRN processor."""
        self.samples_processed = 0
        self.qrn_segments_extracted = 0
        self.fft_size = 2048
        self.overlap = 0.5

    async def process_recording(self, audio_data: np.ndarray,
                               sample_rate: int,
                               frequency: float,
                               bandwidth: float = 12000) -> QRNCharacteristics:
        """Process recording for QRN characteristics.

        Args:
            audio_data: Audio samples (IQ or real)
            sample_rate: Sample rate in Hz
            frequency: Center frequency in Hz
            bandwidth: Recording bandwidth in Hz

        Returns:
            QRN characteristics
        """
        self.samples_processed += len(audio_data)

        # Convert to complex IQ if needed
        iq_data = self._ensure_complex(audio_data)

        # Calculate basic statistics
        noise_floor = self._calculate_noise_floor(iq_data)
        peak_power = self._calculate_peak_power(iq_data)
        avg_power = self._calculate_average_power(iq_data)

        # Detect impulses
        impulse_rate = self._detect_impulses(iq_data, sample_rate)

        # Calculate spectral occupancy
        spectral_occupancy = self._calculate_spectral_occupancy(iq_data, sample_rate)

        # Calculate statistical moments
        kurt = kurtosis(np.abs(iq_data))
        skewness = skew(np.abs(iq_data))

        # Classify noise type
        noise_type = self._classify_noise_type(kurt, skewness, impulse_rate)

        characteristics = QRNCharacteristics(
            timestamp=datetime.now(timezone.utc),
            frequency=frequency,
            bandwidth=bandwidth,
            noise_floor_dbm=noise_floor,
            peak_power_dbm=peak_power,
            average_power_dbm=avg_power,
            impulse_rate=impulse_rate,
            spectral_occupancy=spectral_occupancy,
            kurtosis=kurt,
            skewness=skewness,
            noise_type=noise_type
        )

        logger.info(f"QRN analysis: {noise_type} noise, "
                   f"floor={noise_floor:.1f}dBm, impulses={impulse_rate:.1f}/s")

        return characteristics

    def _ensure_complex(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is complex IQ format.

        Args:
            data: Input data

        Returns:
            Complex IQ data
        """
        if np.iscomplexobj(data):
            return data
        elif len(data.shape) == 2 and data.shape[1] == 2:
            # Two-channel real (I, Q)
            return data[:, 0] + 1j * data[:, 1]
        elif len(data.shape) == 1:
            # Assume alternating I/Q samples
            i_data = data[0::2]
            q_data = data[1::2]
            return i_data + 1j * q_data
        else:
            # Treat as real signal
            return scipy_signal.hilbert(data)

    def _calculate_noise_floor(self, iq_data: np.ndarray) -> float:
        """Calculate noise floor in dBm.

        Args:
            iq_data: Complex IQ data

        Returns:
            Noise floor in dBm
        """
        # Use percentile to estimate noise floor (avoid outliers)
        power = np.abs(iq_data) ** 2
        noise_floor_linear = np.percentile(power, 10)

        # Convert to dBm (assuming 50 ohm impedance)
        if noise_floor_linear > 0:
            noise_floor_dbm = 10 * np.log10(noise_floor_linear * 1000)
        else:
            noise_floor_dbm = -120  # Floor value

        return noise_floor_dbm

    def _calculate_peak_power(self, iq_data: np.ndarray) -> float:
        """Calculate peak power in dBm.

        Args:
            iq_data: Complex IQ data

        Returns:
            Peak power in dBm
        """
        power = np.abs(iq_data) ** 2
        peak_power_linear = np.max(power)

        if peak_power_linear > 0:
            peak_power_dbm = 10 * np.log10(peak_power_linear * 1000)
        else:
            peak_power_dbm = -120

        return peak_power_dbm

    def _calculate_average_power(self, iq_data: np.ndarray) -> float:
        """Calculate average power in dBm.

        Args:
            iq_data: Complex IQ data

        Returns:
            Average power in dBm
        """
        power = np.abs(iq_data) ** 2
        avg_power_linear = np.mean(power)

        if avg_power_linear > 0:
            avg_power_dbm = 10 * np.log10(avg_power_linear * 1000)
        else:
            avg_power_dbm = -120

        return avg_power_dbm

    def _detect_impulses(self, iq_data: np.ndarray, sample_rate: int) -> float:
        """Detect impulse rate in signal.

        Args:
            iq_data: Complex IQ data
            sample_rate: Sample rate in Hz

        Returns:
            Impulses per second
        """
        # Calculate envelope
        envelope = np.abs(iq_data)

        # Dynamic threshold (3 sigma above mean)
        threshold = np.mean(envelope) + 3 * np.std(envelope)

        # Find peaks
        peaks, properties = scipy_signal.find_peaks(
            envelope,
            height=threshold,
            distance=sample_rate // 100  # Min 10ms between impulses
        )

        # Calculate rate
        duration_seconds = len(iq_data) / sample_rate
        impulse_rate = len(peaks) / duration_seconds

        return impulse_rate

    def _calculate_spectral_occupancy(self, iq_data: np.ndarray,
                                     sample_rate: int) -> float:
        """Calculate spectral occupancy percentage.

        Args:
            iq_data: Complex IQ data
            sample_rate: Sample rate in Hz

        Returns:
            Percentage of spectrum occupied by signals
        """
        # Calculate spectrogram
        f, t, Sxx = scipy_signal.spectrogram(
            iq_data,
            fs=sample_rate,
            nperseg=self.fft_size,
            noverlap=int(self.fft_size * self.overlap)
        )

        # Calculate power in dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Threshold for signal detection (10 dB above median)
        threshold = np.median(Sxx_db) + 10

        # Calculate occupancy
        occupied_bins = np.sum(Sxx_db > threshold)
        total_bins = Sxx_db.size
        occupancy = (occupied_bins / total_bins) * 100

        return occupancy

    def _classify_noise_type(self, kurt: float, skewness: float,
                            impulse_rate: float) -> str:
        """Classify type of noise based on statistics.

        Args:
            kurt: Kurtosis value
            skewness: Skewness value
            impulse_rate: Impulses per second

        Returns:
            Noise type classification
        """
        if impulse_rate > 10:
            return "impulse"
        elif kurt > 10:
            return "burst"
        elif abs(skewness) < 0.5 and abs(kurt - 3) < 1:
            return "static"  # Gaussian-like
        else:
            return "mixed"

    def extract_qrn_segments(self, iq_data: np.ndarray,
                            segment_duration: float = 1.0,
                            sample_rate: int = 12000) -> List[np.ndarray]:
        """Extract QRN-rich segments from recording.

        Args:
            iq_data: Complex IQ data
            segment_duration: Duration of each segment in seconds
            sample_rate: Sample rate in Hz

        Returns:
            List of QRN segments
        """
        segment_samples = int(segment_duration * sample_rate)
        segments = []

        # Split into segments
        for i in range(0, len(iq_data) - segment_samples, segment_samples):
            segment = iq_data[i:i + segment_samples]

            # Check if segment has interesting QRN
            envelope = np.abs(segment)
            variation = np.std(envelope) / (np.mean(envelope) + 1e-10)

            # Keep segments with high variation (interesting QRN)
            if variation > 0.5:
                segments.append(segment)
                self.qrn_segments_extracted += 1

        logger.info(f"Extracted {len(segments)} QRN-rich segments")
        return segments

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "samples_processed": self.samples_processed,
            "qrn_segments_extracted": self.qrn_segments_extracted,
            "avg_segments_per_mb": self.qrn_segments_extracted / max(1,
                self.samples_processed / 1024 / 1024)
        }