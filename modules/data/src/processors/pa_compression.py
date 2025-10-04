"""Power amplifier compression detection from spectral analysis.

T096: Detect PA overdrive and compression from harmonic content, spectral
regrowth, and intermodulation products to estimate actual vs reported power.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from scipy import signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


@dataclass
class CompressionIndicators:
    """Indicators of PA compression from signal analysis."""

    station_hash: str
    timestamp: datetime

    # Compression indicators
    harmonic_ratio_db: float  # Fundamental to harmonic ratio
    imd3_level_db: float  # 3rd order intermodulation
    imd5_level_db: float  # 5th order intermodulation
    spectral_regrowth_db: float  # Out-of-band energy
    peak_to_average_ratio_db: float  # PAPR

    # Statistical indicators
    amplitude_kurtosis: float  # Deviation from Gaussian
    phase_distortion: float  # Phase nonlinearity
    evm_percent: float  # Error vector magnitude

    # Compression estimate
    compression_level: str  # 'none', 'mild', 'moderate', 'severe'
    estimated_backoff_db: float  # Estimated dB below saturation
    actual_vs_reported_db: float  # Difference from linear operation

    confidence_score: float


class PACompressionAnalyzer:
    """Analyzes PA compression from signal characteristics."""

    def __init__(self, sample_rate: int = 12000):
        """Initialize PA compression analyzer.

        Args:
            sample_rate: IQ sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.compression_history: Dict[str, List[CompressionIndicators]] = {}

        # Compression thresholds
        self.thresholds = {
            'harmonic_ratio': {'none': -40, 'mild': -30, 'moderate': -20, 'severe': -15},
            'imd3': {'none': -35, 'mild': -30, 'moderate': -25, 'severe': -20},
            'spectral_regrowth': {'none': -40, 'mild': -35, 'moderate': -30, 'severe': -25},
            'papr_reduction': {'none': 0.5, 'mild': 1.0, 'moderate': 2.0, 'severe': 3.0}
        }

    def analyze_compression(self, iq_samples: np.ndarray,
                          station_hash: str,
                          timestamp: datetime,
                          expected_power_dbm: Optional[float] = None) -> CompressionIndicators:
        """Analyze PA compression from IQ samples.

        Args:
            iq_samples: Complex IQ samples
            station_hash: Station identifier
            timestamp: Sample timestamp
            expected_power_dbm: Expected linear power (optional)

        Returns:
            CompressionIndicators with analysis results
        """
        # Calculate spectral indicators
        harmonic_ratio = self._measure_harmonic_content(iq_samples)
        imd3, imd5 = self._measure_intermodulation(iq_samples)
        spectral_regrowth = self._measure_spectral_regrowth(iq_samples)

        # Calculate time-domain indicators
        papr = self._calculate_papr(iq_samples)
        amp_kurtosis = self._calculate_amplitude_kurtosis(iq_samples)
        phase_dist = self._measure_phase_distortion(iq_samples)
        evm = self._calculate_evm(iq_samples)

        # Determine compression level
        compression_level = self._classify_compression(
            harmonic_ratio, imd3, spectral_regrowth, papr
        )

        # Estimate backoff from saturation
        estimated_backoff = self._estimate_backoff(compression_level, imd3, papr)

        # Estimate actual vs reported power
        if expected_power_dbm is not None:
            actual_vs_reported = self._estimate_power_difference(
                compression_level, estimated_backoff
            )
        else:
            actual_vs_reported = 0.0

        # Calculate confidence
        confidence = self._calculate_confidence(
            iq_samples, harmonic_ratio, imd3, spectral_regrowth
        )

        indicators = CompressionIndicators(
            station_hash=station_hash,
            timestamp=timestamp,
            harmonic_ratio_db=harmonic_ratio,
            imd3_level_db=imd3,
            imd5_level_db=imd5,
            spectral_regrowth_db=spectral_regrowth,
            peak_to_average_ratio_db=papr,
            amplitude_kurtosis=amp_kurtosis,
            phase_distortion=phase_dist,
            evm_percent=evm,
            compression_level=compression_level,
            estimated_backoff_db=estimated_backoff,
            actual_vs_reported_db=actual_vs_reported,
            confidence_score=confidence
        )

        # Store in history
        if station_hash not in self.compression_history:
            self.compression_history[station_hash] = []
        self.compression_history[station_hash].append(indicators)

        return indicators

    def _measure_harmonic_content(self, iq_samples: np.ndarray) -> float:
        """Measure harmonic distortion products.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Harmonic to fundamental ratio in dB
        """
        # Compute spectrum
        fft_size = min(2048, len(iq_samples))
        spectrum = np.fft.fft(iq_samples, fft_size)
        spectrum_mag = np.abs(spectrum)
        spectrum_db = 20 * np.log10(spectrum_mag + 1e-20)

        # Find fundamental frequency (highest peak)
        fund_idx = np.argmax(spectrum_db[:fft_size//2])
        fund_power = spectrum_db[fund_idx]

        # Look for harmonics
        harmonic_powers = []

        # 2nd harmonic
        if 2 * fund_idx < fft_size//2:
            h2_idx = 2 * fund_idx
            h2_power = spectrum_db[h2_idx]
            harmonic_powers.append(h2_power)

        # 3rd harmonic
        if 3 * fund_idx < fft_size//2:
            h3_idx = 3 * fund_idx
            h3_power = spectrum_db[h3_idx]
            harmonic_powers.append(h3_power)

        if harmonic_powers:
            # Total harmonic power
            harmonic_power_linear = np.sum(10**(np.array(harmonic_powers)/10))
            harmonic_power_db = 10 * np.log10(harmonic_power_linear + 1e-20)
            harmonic_ratio = harmonic_power_db - fund_power
        else:
            harmonic_ratio = -60.0  # No harmonics detected

        return harmonic_ratio

    def _measure_intermodulation(self, iq_samples: np.ndarray) -> Tuple[float, float]:
        """Measure intermodulation distortion products.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Tuple of (IMD3 level, IMD5 level) in dBc
        """
        # For FT8, look for IMD in adjacent channels
        # Compute spectrum
        fft_size = min(2048, len(iq_samples))
        spectrum = np.fft.fft(iq_samples, fft_size)
        spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-20)

        # Find signal bandwidth
        peak_idx = np.argmax(spectrum_db[:fft_size//2])
        peak_power = spectrum_db[peak_idx]

        # Estimate signal bandwidth (3dB points)
        threshold = peak_power - 3
        signal_indices = np.where(spectrum_db[:fft_size//2] > threshold)[0]

        if len(signal_indices) > 0:
            signal_bw = signal_indices[-1] - signal_indices[0]

            # Look for IMD3 at ±2*BW
            imd3_lower_idx = max(0, peak_idx - 2*signal_bw)
            imd3_upper_idx = min(fft_size//2-1, peak_idx + 2*signal_bw)

            imd3_lower = spectrum_db[imd3_lower_idx]
            imd3_upper = spectrum_db[imd3_upper_idx]
            imd3_level = max(imd3_lower, imd3_upper) - peak_power

            # Look for IMD5 at ±3*BW
            imd5_lower_idx = max(0, peak_idx - 3*signal_bw)
            imd5_upper_idx = min(fft_size//2-1, peak_idx + 3*signal_bw)

            imd5_lower = spectrum_db[imd5_lower_idx]
            imd5_upper = spectrum_db[imd5_upper_idx]
            imd5_level = max(imd5_lower, imd5_upper) - peak_power
        else:
            imd3_level = -50.0
            imd5_level = -60.0

        return imd3_level, imd5_level

    def _measure_spectral_regrowth(self, iq_samples: np.ndarray) -> float:
        """Measure spectral regrowth from PA nonlinearity.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Spectral regrowth in dB
        """
        # Calculate power spectral density
        f, psd = signal.welch(iq_samples, fs=self.sample_rate,
                             nperseg=min(512, len(iq_samples)//4))
        psd_db = 10 * np.log10(psd + 1e-20)

        # Find main signal
        peak_idx = np.argmax(psd_db)
        peak_power = psd_db[peak_idx]

        # Define in-band (6dB bandwidth)
        threshold_6db = peak_power - 6
        in_band_indices = np.where(psd_db > threshold_6db)[0]

        if len(in_band_indices) > 0:
            # Calculate in-band and out-of-band power
            in_band_mask = np.zeros_like(psd_db, dtype=bool)
            in_band_mask[in_band_indices[0]:in_band_indices[-1]+1] = True

            in_band_power = np.sum(psd[in_band_mask])
            out_band_power = np.sum(psd[~in_band_mask])

            if in_band_power > 0:
                regrowth_db = 10 * np.log10(out_band_power / in_band_power + 1e-20)
            else:
                regrowth_db = -60.0
        else:
            regrowth_db = -60.0

        return regrowth_db

    def _calculate_papr(self, iq_samples: np.ndarray) -> float:
        """Calculate Peak-to-Average Power Ratio.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            PAPR in dB
        """
        amplitude = np.abs(iq_samples)

        # Remove zeros to avoid log issues
        amplitude = amplitude[amplitude > 0]

        if len(amplitude) == 0:
            return 0.0

        peak_power = np.max(amplitude)**2
        avg_power = np.mean(amplitude**2)

        if avg_power > 0:
            papr_db = 10 * np.log10(peak_power / avg_power)
        else:
            papr_db = 0.0

        return papr_db

    def _calculate_amplitude_kurtosis(self, iq_samples: np.ndarray) -> float:
        """Calculate kurtosis of amplitude distribution.

        Compression tends to reduce kurtosis (platykurtic).
        """
        amplitude = np.abs(iq_samples)
        return kurtosis(amplitude)

    def _measure_phase_distortion(self, iq_samples: np.ndarray) -> float:
        """Measure phase distortion from AM-PM conversion.

        Args:
            iq_samples: Complex IQ samples

        Returns:
            Phase distortion metric (0-1)
        """
        amplitude = np.abs(iq_samples)
        phase = np.unwrap(np.angle(iq_samples))

        # Remove linear phase trend
        if len(phase) > 2:
            time = np.arange(len(phase))
            coeffs = np.polyfit(time, phase, 1)
            phase_detrended = phase - np.polyval(coeffs, time)
        else:
            phase_detrended = phase

        # Look for correlation between amplitude and phase
        if len(amplitude) > 10:
            # Normalize
            amp_norm = (amplitude - np.mean(amplitude)) / (np.std(amplitude) + 1e-8)
            phase_norm = (phase_detrended - np.mean(phase_detrended)) / (np.std(phase_detrended) + 1e-8)

            # Calculate correlation
            correlation = np.abs(np.corrcoef(amp_norm, phase_norm)[0, 1])
        else:
            correlation = 0.0

        return correlation

    def _calculate_evm(self, iq_samples: np.ndarray) -> float:
        """Calculate Error Vector Magnitude.

        Higher EVM indicates more distortion.
        """
        # Normalize samples
        samples_norm = iq_samples / (np.max(np.abs(iq_samples)) + 1e-20)

        # For FT8 8-GFSK, estimate ideal symbols
        # This is simplified - actual would need proper demodulation
        phases = np.angle(samples_norm)

        # Quantize to nearest ideal phase
        n_states = 8
        ideal_phases = np.linspace(-np.pi, np.pi, n_states, endpoint=False)

        errors = []
        for i, phase in enumerate(phases):
            nearest_idx = np.argmin(np.abs(ideal_phases - phase))
            ideal_phase = ideal_phases[nearest_idx]

            # Create ideal symbol
            ideal_symbol = np.exp(1j * ideal_phase)
            actual_symbol = samples_norm[i] if i < len(samples_norm) else 0

            # Calculate error
            error = np.abs(actual_symbol - ideal_symbol)
            errors.append(error)

        # RMS error as percentage
        evm = 100 * np.sqrt(np.mean(np.array(errors)**2))

        return evm

    def _classify_compression(self, harmonic_ratio: float, imd3: float,
                            spectral_regrowth: float, papr: float) -> str:
        """Classify compression severity.

        Args:
            harmonic_ratio: Harmonic distortion ratio
            imd3: IMD3 level
            spectral_regrowth: Spectral regrowth level
            papr: Peak-to-average ratio

        Returns:
            Compression level classification
        """
        scores = {'none': 0, 'mild': 0, 'moderate': 0, 'severe': 0}

        # Score based on harmonics
        for level, threshold in self.thresholds['harmonic_ratio'].items():
            if harmonic_ratio >= threshold:
                scores[level] += 1

        # Score based on IMD3
        for level, threshold in self.thresholds['imd3'].items():
            if imd3 >= threshold:
                scores[level] += 1

        # Score based on spectral regrowth
        for level, threshold in self.thresholds['spectral_regrowth'].items():
            if spectral_regrowth >= threshold:
                scores[level] += 1

        # Score based on PAPR reduction
        papr_reduction = 10 - papr  # Ideal PAPR ~10dB for FT8
        for level, threshold in self.thresholds['papr_reduction'].items():
            if papr_reduction >= threshold:
                scores[level] += 1

        # Return highest scoring level
        return max(scores.keys(), key=lambda k: scores[k])

    def _estimate_backoff(self, compression_level: str, imd3: float, papr: float) -> float:
        """Estimate input backoff from saturation.

        Args:
            compression_level: Compression classification
            imd3: IMD3 level in dBc
            papr: Measured PAPR

        Returns:
            Estimated backoff in dB
        """
        # Base estimate from compression level
        base_backoff = {
            'none': 10.0,    # Well backed off
            'mild': 6.0,     # Near linear region edge
            'moderate': 3.0,  # In compression
            'severe': 0.0    # At saturation
        }

        backoff = base_backoff[compression_level]

        # Adjust based on IMD3 (empirical relationship)
        # IMD3 increases ~2dB for each 1dB closer to saturation
        if imd3 > -40:
            imd_adjustment = (imd3 + 40) / 2
            backoff = max(0, backoff - imd_adjustment)

        return backoff

    def _estimate_power_difference(self, compression_level: str, backoff: float) -> float:
        """Estimate difference between actual and reported power.

        Args:
            compression_level: Compression classification
            backoff: Estimated backoff from saturation

        Returns:
            Power difference in dB (positive = actual > reported)
        """
        # Compression causes actual output to be less than expected
        # Based on typical PA compression curves
        compression_loss = {
            'none': 0.0,
            'mild': 0.5,      # ~0.5 dB compression
            'moderate': 1.5,   # ~1.5 dB compression
            'severe': 3.0     # ~3 dB compression
        }

        return -compression_loss[compression_level]

    def _calculate_confidence(self, iq_samples: np.ndarray, harmonic_ratio: float,
                             imd3: float, spectral_regrowth: float) -> float:
        """Calculate confidence in compression analysis.

        Args:
            iq_samples: IQ samples
            harmonic_ratio: Measured harmonic ratio
            imd3: Measured IMD3
            spectral_regrowth: Measured spectral regrowth

        Returns:
            Confidence score 0-1
        """
        # Sample count factor
        sample_factor = min(1.0, len(iq_samples) / 10000)

        # SNR factor (estimate from sample statistics)
        signal_power = np.mean(np.abs(iq_samples)**2)
        noise_estimate = np.std(np.abs(iq_samples))
        if noise_estimate > 0:
            snr_estimate = 10 * np.log10(signal_power / (noise_estimate**2))
            snr_factor = min(1.0, max(0.0, (snr_estimate - 10) / 20))
        else:
            snr_factor = 0.5

        # Measurement consistency (if multiple indicators agree)
        indicators = [harmonic_ratio > -40, imd3 > -35, spectral_regrowth > -40]
        consistency_factor = sum(indicators) / 3

        # Combined confidence
        confidence = sample_factor * 0.3 + snr_factor * 0.4 + consistency_factor * 0.3

        return confidence

    def get_station_compression_profile(self, station_hash: str) -> Dict[str, Any]:
        """Get compression profile for a station.

        Args:
            station_hash: Station identifier

        Returns:
            Compression profile summary
        """
        if station_hash not in self.compression_history:
            return {'status': 'no_data'}

        history = self.compression_history[station_hash]

        # Aggregate statistics
        compression_levels = [h.compression_level for h in history]
        level_counts = {level: compression_levels.count(level)
                       for level in ['none', 'mild', 'moderate', 'severe']}

        avg_backoff = np.mean([h.estimated_backoff_db for h in history])
        avg_imd3 = np.mean([h.imd3_level_db for h in history])
        avg_confidence = np.mean([h.confidence_score for h in history])

        # Dominant compression level
        dominant_level = max(level_counts.keys(), key=lambda k: level_counts[k])

        return {
            'dominant_compression': dominant_level,
            'compression_distribution': level_counts,
            'avg_backoff_db': avg_backoff,
            'avg_imd3_db': avg_imd3,
            'confidence': avg_confidence,
            'num_analyses': len(history),
            'likely_overdriving': dominant_level in ['moderate', 'severe']
        }