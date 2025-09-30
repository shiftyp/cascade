"""Equipment signature extraction from signal characteristics.

T076: Extract equipment-specific characteristics (phase noise, drift, linearity)
from received signals to fingerprint transmitter hardware.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import signal
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


@dataclass
class EquipmentSignature:
    """Hardware-specific characteristics of a transmitter."""

    station_hash: str
    timestamp: str

    # Oscillator characteristics
    phase_noise_dbc: float  # Phase noise at 1kHz offset (dBc/Hz)
    frequency_drift_hz_per_min: float
    frequency_stability_ppm: float
    allan_deviation: float  # Short-term stability

    # Power amplifier characteristics
    pa_linearity_imd3: float  # 3rd order intermodulation (dB)
    pa_linearity_imd5: float  # 5th order intermodulation (dB)
    spectral_regrowth_db: float
    compression_point_db: float  # 1dB compression estimate

    # Modulation characteristics
    evm_percent: float  # Error vector magnitude
    symbol_rate_error_ppm: float
    rise_time_us: float
    fall_time_us: float

    # Statistical confidence
    confidence_score: float  # 0-1 confidence in measurements
    sample_count: int
    snr_db: float


class EquipmentSignatureExtractor:
    """Extracts equipment-specific signatures from IQ samples."""

    def __init__(self, sample_rate: int = 12000):
        """Initialize signature extractor.

        Args:
            sample_rate: IQ sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.signatures: Dict[str, List[EquipmentSignature]] = {}

    def extract_signature(self, iq_samples: np.ndarray,
                         station_hash: str,
                         timestamp: str,
                         snr_db: float = 0) -> EquipmentSignature:
        """Extract equipment signature from IQ samples.

        Args:
            iq_samples: Complex IQ samples
            station_hash: Anonymized station identifier
            timestamp: UTC timestamp
            snr_db: Signal-to-noise ratio

        Returns:
            EquipmentSignature with extracted characteristics
        """
        # Extract oscillator characteristics
        phase_noise = self._measure_phase_noise(iq_samples)
        drift = self._measure_frequency_drift(iq_samples)
        stability = self._measure_frequency_stability(iq_samples)
        allan_dev = self._calculate_allan_deviation(iq_samples)

        # Extract PA characteristics
        imd3, imd5 = self._measure_intermodulation(iq_samples)
        regrowth = self._measure_spectral_regrowth(iq_samples)
        compression = self._estimate_compression_point(iq_samples)

        # Extract modulation characteristics
        evm = self._calculate_evm(iq_samples)
        symbol_error = self._measure_symbol_rate_error(iq_samples)
        rise, fall = self._measure_keying_times(iq_samples)

        # Calculate confidence score
        confidence = self._calculate_confidence(snr_db, len(iq_samples))

        signature = EquipmentSignature(
            station_hash=station_hash,
            timestamp=timestamp,
            phase_noise_dbc=phase_noise,
            frequency_drift_hz_per_min=drift,
            frequency_stability_ppm=stability,
            allan_deviation=allan_dev,
            pa_linearity_imd3=imd3,
            pa_linearity_imd5=imd5,
            spectral_regrowth_db=regrowth,
            compression_point_db=compression,
            evm_percent=evm,
            symbol_rate_error_ppm=symbol_error,
            rise_time_us=rise,
            fall_time_us=fall,
            confidence_score=confidence,
            sample_count=len(iq_samples),
            snr_db=snr_db
        )

        # Store signature
        if station_hash not in self.signatures:
            self.signatures[station_hash] = []
        self.signatures[station_hash].append(signature)

        return signature

    def _measure_phase_noise(self, iq_samples: np.ndarray) -> float:
        """Measure phase noise at 1kHz offset.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Phase noise in dBc/Hz at 1kHz offset
        """
        # Extract instantaneous phase
        phase = np.unwrap(np.angle(iq_samples))

        # Remove linear trend (carrier frequency)
        time = np.arange(len(phase)) / self.sample_rate
        coeffs = np.polyfit(time, phase, 1)
        phase_detrended = phase - np.polyval(coeffs, time)

        # Calculate phase noise PSD
        f, psd = signal.welch(phase_detrended, fs=self.sample_rate,
                             nperseg=min(1024, len(phase)//4))

        # Find noise at 1kHz offset
        target_freq = 1000  # Hz
        idx = np.argmin(np.abs(f - target_freq))

        # Convert to dBc/Hz (relative to carrier)
        phase_noise_dbc = 10 * np.log10(psd[idx] + 1e-20)

        return phase_noise_dbc

    def _measure_frequency_drift(self, iq_samples: np.ndarray) -> float:
        """Measure frequency drift rate.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Drift rate in Hz/minute
        """
        # Estimate instantaneous frequency
        phase = np.unwrap(np.angle(iq_samples))
        inst_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)

        # Smooth to remove noise
        window = min(100, len(inst_freq)//10)
        if window > 1:
            inst_freq_smooth = np.convolve(inst_freq, np.ones(window)/window, 'valid')
        else:
            inst_freq_smooth = inst_freq

        # Calculate drift rate
        time_seconds = np.arange(len(inst_freq_smooth)) / self.sample_rate
        if len(time_seconds) > 1:
            coeffs = np.polyfit(time_seconds, inst_freq_smooth, 1)
            drift_hz_per_sec = coeffs[0]
            drift_hz_per_min = drift_hz_per_sec * 60
        else:
            drift_hz_per_min = 0.0

        return drift_hz_per_min

    def _measure_frequency_stability(self, iq_samples: np.ndarray) -> float:
        """Measure frequency stability in PPM.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Frequency stability in parts per million
        """
        # Estimate instantaneous frequency
        phase = np.unwrap(np.angle(iq_samples))
        inst_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)

        # Calculate stability
        if len(inst_freq) > 0:
            mean_freq = np.mean(inst_freq)
            std_freq = np.std(inst_freq)

            # Assume nominal frequency around 0 Hz (baseband)
            # In practice, this would be relative to known carrier
            nominal_freq = 14_000_000  # 14 MHz as example
            stability_ppm = (std_freq / nominal_freq) * 1e6
        else:
            stability_ppm = 0.0

        return stability_ppm

    def _calculate_allan_deviation(self, iq_samples: np.ndarray) -> float:
        """Calculate Allan deviation for short-term stability.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Allan deviation
        """
        # Extract frequency samples
        phase = np.unwrap(np.angle(iq_samples))
        inst_freq = np.diff(phase) * self.sample_rate / (2 * np.pi)

        # Calculate Allan deviation with tau = 1 second
        tau_samples = self.sample_rate

        if len(inst_freq) >= 2 * tau_samples:
            # Average frequency over tau intervals
            n_intervals = len(inst_freq) // tau_samples
            freq_avg = np.array([np.mean(inst_freq[i*tau_samples:(i+1)*tau_samples])
                                for i in range(n_intervals)])

            # Allan deviation formula
            if len(freq_avg) >= 2:
                diff = np.diff(freq_avg)
                allan_dev = np.sqrt(0.5 * np.mean(diff**2))
            else:
                allan_dev = 0.0
        else:
            allan_dev = np.std(inst_freq) if len(inst_freq) > 0 else 0.0

        return allan_dev

    def _measure_intermodulation(self, iq_samples: np.ndarray) -> Tuple[float, float]:
        """Measure intermodulation distortion products.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Tuple of (IMD3 in dB, IMD5 in dB)
        """
        # Calculate spectrum
        spectrum = np.fft.fft(iq_samples)
        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-20)

        # Find fundamental frequency
        fund_idx = np.argmax(spectrum_db[:len(spectrum)//2])
        fund_power = spectrum_db[fund_idx]

        # Estimate IMD3 and IMD5 locations
        # In real implementation, these would be at specific frequency offsets
        if fund_idx > 0 and 3*fund_idx < len(spectrum)//2:
            imd3_idx = 3 * fund_idx
            imd3_power = spectrum_db[imd3_idx]
            imd3 = fund_power - imd3_power
        else:
            imd3 = 60.0  # Default good value

        if fund_idx > 0 and 5*fund_idx < len(spectrum)//2:
            imd5_idx = 5 * fund_idx
            imd5_power = spectrum_db[imd5_idx]
            imd5 = fund_power - imd5_power
        else:
            imd5 = 70.0  # Default good value

        return imd3, imd5

    def _measure_spectral_regrowth(self, iq_samples: np.ndarray) -> float:
        """Measure spectral regrowth from PA nonlinearity.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Spectral regrowth in dB
        """
        # Calculate power spectrum
        f, psd = signal.welch(iq_samples, fs=self.sample_rate,
                             nperseg=min(512, len(iq_samples)//4))
        psd_db = 10 * np.log10(psd + 1e-20)

        # Find main signal bandwidth (3dB points)
        peak_idx = np.argmax(psd_db)
        peak_power = psd_db[peak_idx]
        threshold_3db = peak_power - 3

        # Find bandwidth
        above_threshold = psd_db > threshold_3db
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            bw_indices = indices[-1] - indices[0]

            # Measure power outside main bandwidth
            in_band = np.zeros_like(psd_db, dtype=bool)
            in_band[indices[0]:indices[-1]+1] = True

            in_band_power = np.sum(psd[in_band])
            out_band_power = np.sum(psd[~in_band])

            if in_band_power > 0:
                regrowth_db = 10 * np.log10(out_band_power / in_band_power + 1e-20)
            else:
                regrowth_db = -60.0
        else:
            regrowth_db = -60.0

        return abs(regrowth_db)

    def _estimate_compression_point(self, iq_samples: np.ndarray) -> float:
        """Estimate 1dB compression point from signal statistics.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Estimated 1dB compression point
        """
        # Calculate amplitude statistics
        amplitude = np.abs(iq_samples)

        # Look for compression indicators
        # High kurtosis suggests compression
        kurt = kurtosis(amplitude)

        # Estimate based on statistics
        # This is a simplified model
        if kurt < 0:  # Platykurtic - likely compressed
            compression = 20.0  # Low compression point
        elif kurt > 3:  # Leptokurtic - likely linear
            compression = 30.0  # High compression point
        else:
            compression = 25.0 + kurt  # Scale with kurtosis

        return compression

    def _calculate_evm(self, iq_samples: np.ndarray) -> float:
        """Calculate Error Vector Magnitude.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            EVM as percentage
        """
        # For FT8, assume 8-GFSK modulation
        # Simplified EVM calculation

        # Normalize samples
        samples_norm = iq_samples / (np.max(np.abs(iq_samples)) + 1e-20)

        # Estimate ideal constellation points (simplified)
        # For 8-GFSK, we'd have 8 frequency states
        phases = np.angle(samples_norm)

        # Quantize to nearest ideal point
        n_states = 8
        ideal_phases = np.linspace(-np.pi, np.pi, n_states, endpoint=False)

        errors = []
        for phase in phases:
            nearest_ideal = ideal_phases[np.argmin(np.abs(ideal_phases - phase))]
            error = np.abs(phase - nearest_ideal)
            errors.append(error)

        # Calculate RMS error
        evm = np.sqrt(np.mean(np.array(errors)**2))
        evm_percent = evm * 100 / np.pi  # Normalize to percentage

        return evm_percent

    def _measure_symbol_rate_error(self, iq_samples: np.ndarray) -> float:
        """Measure symbol rate error in PPM.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Symbol rate error in PPM
        """
        # For FT8: 6.25 Hz symbol rate
        expected_symbol_rate = 6.25  # Hz

        # Detect symbol transitions
        amplitude = np.abs(iq_samples)

        # Find edges using derivative
        if len(amplitude) > 2:
            edges = np.abs(np.diff(amplitude))
            threshold = np.mean(edges) + np.std(edges)

            # Find transition points
            transitions = np.where(edges > threshold)[0]

            if len(transitions) > 1:
                # Calculate average symbol period
                periods = np.diff(transitions) / self.sample_rate
                avg_period = np.mean(periods)
                measured_rate = 1 / avg_period if avg_period > 0 else expected_symbol_rate

                # Calculate error in PPM
                error_ppm = ((measured_rate - expected_symbol_rate) /
                           expected_symbol_rate) * 1e6
            else:
                error_ppm = 0.0
        else:
            error_ppm = 0.0

        return abs(error_ppm)

    def _measure_keying_times(self, iq_samples: np.ndarray) -> Tuple[float, float]:
        """Measure rise and fall times of keying.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Tuple of (rise_time_us, fall_time_us)
        """
        amplitude = np.abs(iq_samples)

        # Find 10% and 90% levels
        max_amp = np.max(amplitude)
        level_10 = 0.1 * max_amp
        level_90 = 0.9 * max_amp

        rise_times = []
        fall_times = []

        # Scan for transitions
        for i in range(1, len(amplitude)-1):
            # Rising edge
            if amplitude[i-1] < level_10 and amplitude[i+1] > level_90:
                # Count samples in transition
                rise_start = i
                while i < len(amplitude) and amplitude[i] < level_90:
                    i += 1
                rise_samples = i - rise_start
                rise_times.append(rise_samples / self.sample_rate * 1e6)  # Convert to microseconds

            # Falling edge
            elif amplitude[i-1] > level_90 and amplitude[i+1] < level_10:
                fall_start = i
                while i < len(amplitude) and amplitude[i] > level_10:
                    i += 1
                fall_samples = i - fall_start
                fall_times.append(fall_samples / self.sample_rate * 1e6)

        # Average times
        rise_time = np.mean(rise_times) if rise_times else 100.0  # Default 100us
        fall_time = np.mean(fall_times) if fall_times else 100.0

        return rise_time, fall_time

    def _calculate_confidence(self, snr_db: float, sample_count: int) -> float:
        """Calculate confidence score for measurements.

        Args:
            snr_db: Signal-to-noise ratio
            sample_count: Number of samples

        Returns:
            Confidence score 0-1
        """
        # SNR component (sigmoid)
        snr_confidence = 1 / (1 + np.exp(-(snr_db - 10) / 5))

        # Sample count component (logarithmic)
        sample_confidence = min(1.0, np.log10(sample_count) / 5)  # Max at 100k samples

        # Combined confidence
        confidence = 0.7 * snr_confidence + 0.3 * sample_confidence

        return confidence

    def aggregate_signatures(self, station_hash: str) -> Optional[EquipmentSignature]:
        """Aggregate multiple signatures for a station.

        Args:
            station_hash: Station identifier

        Returns:
            Aggregated signature or None
        """
        if station_hash not in self.signatures:
            return None

        sigs = self.signatures[station_hash]
        if not sigs:
            return None

        # Weight by confidence
        weights = np.array([s.confidence_score for s in sigs])
        weights = weights / np.sum(weights)

        # Weighted average of measurements
        aggregated = EquipmentSignature(
            station_hash=station_hash,
            timestamp=sigs[-1].timestamp,  # Most recent
            phase_noise_dbc=np.average([s.phase_noise_dbc for s in sigs], weights=weights),
            frequency_drift_hz_per_min=np.average([s.frequency_drift_hz_per_min for s in sigs], weights=weights),
            frequency_stability_ppm=np.average([s.frequency_stability_ppm for s in sigs], weights=weights),
            allan_deviation=np.average([s.allan_deviation for s in sigs], weights=weights),
            pa_linearity_imd3=np.average([s.pa_linearity_imd3 for s in sigs], weights=weights),
            pa_linearity_imd5=np.average([s.pa_linearity_imd5 for s in sigs], weights=weights),
            spectral_regrowth_db=np.average([s.spectral_regrowth_db for s in sigs], weights=weights),
            compression_point_db=np.average([s.compression_point_db for s in sigs], weights=weights),
            evm_percent=np.average([s.evm_percent for s in sigs], weights=weights),
            symbol_rate_error_ppm=np.average([s.symbol_rate_error_ppm for s in sigs], weights=weights),
            rise_time_us=np.average([s.rise_time_us for s in sigs], weights=weights),
            fall_time_us=np.average([s.fall_time_us for s in sigs], weights=weights),
            confidence_score=np.mean([s.confidence_score for s in sigs]),
            sample_count=sum(s.sample_count for s in sigs),
            snr_db=np.average([s.snr_db for s in sigs], weights=weights)
        )

        return aggregated