"""QRN (Atmospheric Noise) analyzer for quiet period detection.

Implements T031: QRN analyzer (FR-029, FR-030).
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import json

from ..config import config
from ..config.frequencies import BAND_CONFIGS
from ..models import SessionLocal, QRNSample, AtmosphericEvent

logger = logging.getLogger(__name__)


@dataclass
class NoiseMetrics:
    """Noise analysis metrics."""

    timestamp: datetime
    rms_level: float
    peak_level: float
    spectral_density: Dict[str, float]  # Frequency bands
    impulse_count: int
    impulse_rate: float  # impulses per second
    quiet_ratio: float  # 0-1, higher = quieter
    band_utilization: Dict[str, float]  # Band occupancy
    atmospheric_index: float  # 0-100, atmospheric noise level


@dataclass
class QuietPeriod:
    """Identified quiet period."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float
    avg_noise_level: float
    min_noise_level: float
    quality_score: float  # 0-1, higher = better for neural training
    band: str
    frequency_khz: float


@dataclass
class AtmosphericImpulse:
    """Detected atmospheric impulse."""

    timestamp: datetime
    peak_amplitude: float
    duration_ms: float
    frequency_content: Dict[str, float]
    rise_time_us: float
    decay_time_ms: float
    impulse_type: str  # lightning, sferics, etc.


class QRNAnalyzer:
    """Analyzes atmospheric noise for quiet periods and characteristics."""

    def __init__(self):
        """Initialize QRN analyzer."""
        self.db = SessionLocal()

        # Analysis parameters
        self.analysis_window_seconds = 10.0  # 10-second analysis windows
        self.quiet_threshold_db = -40  # dBFS threshold for quiet periods
        self.impulse_threshold_factor = 3.0  # Factor above RMS for impulse detection
        self.min_quiet_duration = 30.0  # Minimum 30-second quiet periods

        # Frequency bands for analysis
        self.analysis_bands = {
            "vhf": (30e6, 300e6),
            "hf_high": (14e6, 30e6),
            "hf_mid": (7e6, 14e6),
            "hf_low": (3e6, 7e6),
            "mf": (300e3, 3e6),
            "lf": (30e3, 300e3),
        }

    async def analyze_iq_data(
        self,
        iq_data: np.ndarray,
        sample_rate: int,
        center_frequency_khz: float,
        band: str,
        start_time: datetime,
    ) -> Tuple[List[NoiseMetrics], List[QuietPeriod], List[AtmosphericImpulse]]:
        """Analyze IQ data for QRN characteristics.

        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate in Hz
            center_frequency_khz: Center frequency
            band: Band name
            start_time: Recording start time

        Returns:
            Tuple of (noise_metrics, quiet_periods, impulses)
        """
        logger.info(f"Analyzing {len(iq_data)} samples for QRN on {band}")

        # Convert to power samples
        power_data = np.abs(iq_data) ** 2

        # Analysis window size
        window_samples = int(self.analysis_window_seconds * sample_rate)

        noise_metrics = []
        impulses = []

        # Process in overlapping windows
        hop_samples = window_samples // 2

        for i in range(0, len(power_data) - window_samples, hop_samples):
            window_data = iq_data[i : i + window_samples]
            window_power = power_data[i : i + window_samples]
            window_start = start_time + timedelta(seconds=i / sample_rate)

            # Analyze this window
            metrics = await self._analyze_noise_window(
                window_data, window_power, sample_rate, center_frequency_khz, window_start
            )

            noise_metrics.append(metrics)

            # Detect impulses in this window
            window_impulses = await self._detect_impulses(
                window_data, sample_rate, window_start
            )
            impulses.extend(window_impulses)

        # Identify quiet periods from metrics
        quiet_periods = await self._identify_quiet_periods(noise_metrics, band, center_frequency_khz)

        logger.info(
            f"QRN analysis complete: {len(noise_metrics)} windows, "
            f"{len(quiet_periods)} quiet periods, {len(impulses)} impulses"
        )

        return noise_metrics, quiet_periods, impulses

    async def _analyze_noise_window(
        self,
        iq_window: np.ndarray,
        power_window: np.ndarray,
        sample_rate: int,
        center_frequency_khz: float,
        window_start: datetime,
    ) -> NoiseMetrics:
        """Analyze noise characteristics in a window.

        Args:
            iq_window: Complex IQ samples
            power_window: Power samples
            sample_rate: Sample rate
            center_frequency_khz: Center frequency
            window_start: Window start time

        Returns:
            Noise metrics
        """
        # Basic power statistics
        rms_level = np.sqrt(np.mean(power_window))
        peak_level = np.max(power_window)
        rms_db = 20 * np.log10(rms_level + 1e-12)

        # Spectral analysis
        freqs, psd = signal.welch(
            iq_window, sample_rate, nperseg=min(1024, len(iq_window) // 4)
        )

        # Categorize spectral density by bands
        spectral_density = await self._categorize_spectral_density(
            freqs, psd, center_frequency_khz * 1000
        )

        # Impulse detection
        impulse_threshold = rms_level * self.impulse_threshold_factor
        impulse_mask = power_window > impulse_threshold
        impulse_count = np.sum(np.diff(impulse_mask.astype(int)) > 0)
        impulse_rate = impulse_count / len(power_window) * sample_rate

        # Quiet ratio (inverse of variability)
        power_std = np.std(power_window)
        quiet_ratio = 1.0 / (1.0 + power_std / (rms_level + 1e-12))

        # Band utilization (frequency domain occupancy)
        band_utilization = await self._calculate_band_utilization(freqs, psd)

        # Atmospheric index (combines multiple factors)
        atmospheric_index = self._calculate_atmospheric_index(
            rms_level, impulse_rate, quiet_ratio, spectral_density
        )

        return NoiseMetrics(
            timestamp=window_start,
            rms_level=rms_level,
            peak_level=peak_level,
            spectral_density=spectral_density,
            impulse_count=impulse_count,
            impulse_rate=impulse_rate,
            quiet_ratio=quiet_ratio,
            band_utilization=band_utilization,
            atmospheric_index=atmospheric_index,
        )

    async def _categorize_spectral_density(
        self, freqs: np.ndarray, psd: np.ndarray, center_freq_hz: float
    ) -> Dict[str, float]:
        """Categorize spectral power density by frequency bands.

        Args:
            freqs: Frequency array (relative to center)
            psd: Power spectral density
            center_freq_hz: Center frequency

        Returns:
            Spectral density by band
        """
        # Convert relative frequencies to absolute
        abs_freqs = freqs + center_freq_hz

        spectral_density = {}

        for band_name, (f_min, f_max) in self.analysis_bands.items():
            # Find frequencies in this band
            band_mask = (abs_freqs >= f_min) & (abs_freqs <= f_max)

            if np.any(band_mask):
                # Average power in this band
                band_power = np.mean(psd[band_mask])
                spectral_density[band_name] = float(10 * np.log10(band_power + 1e-12))
            else:
                spectral_density[band_name] = -100.0  # Very low

        return spectral_density

    async def _calculate_band_utilization(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> Dict[str, float]:
        """Calculate frequency band utilization.

        Args:
            freqs: Frequency array
            psd: Power spectral density

        Returns:
            Band utilization ratios (0-1)
        """
        # Define relative frequency bands for utilization
        bands = {
            "low": (0, 0.2),  # 0-20% of bandwidth
            "mid": (0.2, 0.8),  # 20-80% of bandwidth
            "high": (0.8, 1.0),  # 80-100% of bandwidth
        }

        total_power = np.sum(psd)
        utilization = {}

        for band_name, (f_start, f_end) in bands.items():
            # Frequency indices for this band
            start_idx = int(f_start * len(freqs))
            end_idx = int(f_end * len(freqs))

            band_power = np.sum(psd[start_idx:end_idx])
            utilization[band_name] = float(band_power / (total_power + 1e-12))

        return utilization

    def _calculate_atmospheric_index(
        self,
        rms_level: float,
        impulse_rate: float,
        quiet_ratio: float,
        spectral_density: Dict[str, float],
    ) -> float:
        """Calculate atmospheric noise index.

        Args:
            rms_level: RMS power level
            impulse_rate: Impulse rate
            quiet_ratio: Quiet ratio
            spectral_density: Spectral density by band

        Returns:
            Atmospheric index (0-100)
        """
        # Combine factors into single index
        # Higher impulse rate = more atmospheric
        impulse_factor = min(100, impulse_rate * 10)

        # Lower quiet ratio = more atmospheric
        quiet_factor = (1.0 - quiet_ratio) * 50

        # VHF/HF ratio (atmospheric noise stronger at lower frequencies)
        hf_power = spectral_density.get("hf_mid", -60)
        vhf_power = spectral_density.get("vhf", -60)
        freq_factor = max(0, (hf_power - vhf_power)) * 2

        # Combine factors
        atmospheric_index = (impulse_factor + quiet_factor + freq_factor) / 3
        return min(100, max(0, atmospheric_index))

    async def _detect_impulses(
        self,
        iq_window: np.ndarray,
        sample_rate: int,
        window_start: datetime,
    ) -> List[AtmosphericImpulse]:
        """Detect atmospheric impulses in window.

        Args:
            iq_window: IQ samples
            sample_rate: Sample rate
            window_start: Window start time

        Returns:
            Detected impulses
        """
        impulses = []

        # Convert to magnitude
        magnitude = np.abs(iq_window)

        # Detect impulse peaks
        rms_level = np.sqrt(np.mean(magnitude**2))
        threshold = rms_level * self.impulse_threshold_factor

        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            magnitude,
            height=threshold,
            distance=int(sample_rate * 0.001),  # Min 1ms separation
            width=1,
        )

        for peak_idx in peaks:
            try:
                # Calculate impulse characteristics
                peak_amplitude = magnitude[peak_idx]
                peak_time = window_start + timedelta(seconds=peak_idx / sample_rate)

                # Estimate impulse duration
                half_height = peak_amplitude / 2
                duration_samples = await self._estimate_impulse_duration(
                    magnitude, peak_idx, half_height
                )
                duration_ms = duration_samples / sample_rate * 1000

                # Analyze frequency content around impulse
                impulse_start = max(0, peak_idx - duration_samples // 2)
                impulse_end = min(len(iq_window), peak_idx + duration_samples // 2)
                impulse_data = iq_window[impulse_start:impulse_end]

                freq_content = await self._analyze_impulse_spectrum(
                    impulse_data, sample_rate
                )

                # Estimate rise/decay times
                rise_time_us, decay_time_ms = await self._estimate_impulse_times(
                    magnitude, peak_idx, sample_rate
                )

                # Classify impulse type
                impulse_type = self._classify_impulse(
                    peak_amplitude, duration_ms, freq_content
                )

                impulse = AtmosphericImpulse(
                    timestamp=peak_time,
                    peak_amplitude=float(peak_amplitude),
                    duration_ms=duration_ms,
                    frequency_content=freq_content,
                    rise_time_us=rise_time_us,
                    decay_time_ms=decay_time_ms,
                    impulse_type=impulse_type,
                )

                impulses.append(impulse)

            except Exception as e:
                logger.debug(f"Error analyzing impulse at {peak_idx}: {e}")

        return impulses

    async def _estimate_impulse_duration(
        self, magnitude: np.ndarray, peak_idx: int, half_height: float
    ) -> int:
        """Estimate impulse duration at half-height.

        Args:
            magnitude: Magnitude array
            peak_idx: Peak index
            half_height: Half-height threshold

        Returns:
            Duration in samples
        """
        # Find left and right half-height points
        left_idx = peak_idx
        right_idx = peak_idx

        # Search left
        for i in range(peak_idx, max(0, peak_idx - 100), -1):
            if magnitude[i] < half_height:
                left_idx = i
                break

        # Search right
        for i in range(peak_idx, min(len(magnitude), peak_idx + 100)):
            if magnitude[i] < half_height:
                right_idx = i
                break

        return right_idx - left_idx

    async def _analyze_impulse_spectrum(
        self, impulse_data: np.ndarray, sample_rate: int
    ) -> Dict[str, float]:
        """Analyze frequency content of impulse.

        Args:
            impulse_data: Impulse IQ samples
            sample_rate: Sample rate

        Returns:
            Frequency content analysis
        """
        if len(impulse_data) < 8:
            return {"total_power": 0.0}

        # FFT analysis
        fft_data = fft(impulse_data)
        freqs = fftfreq(len(impulse_data), 1 / sample_rate)
        power_spectrum = np.abs(fft_data) ** 2

        # Analyze frequency bands
        freq_bands = {
            "low": (0, sample_rate / 8),
            "mid": (sample_rate / 8, sample_rate / 4),
            "high": (sample_rate / 4, sample_rate / 2),
        }

        freq_content = {}
        total_power = np.sum(power_spectrum)

        for band_name, (f_min, f_max) in freq_bands.items():
            band_mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
            band_power = np.sum(power_spectrum[band_mask])
            freq_content[band_name] = float(band_power / (total_power + 1e-12))

        freq_content["total_power"] = float(total_power)
        return freq_content

    async def _estimate_impulse_times(
        self, magnitude: np.ndarray, peak_idx: int, sample_rate: int
    ) -> Tuple[float, float]:
        """Estimate impulse rise and decay times.

        Args:
            magnitude: Magnitude array
            peak_idx: Peak index
            sample_rate: Sample rate

        Returns:
            (rise_time_us, decay_time_ms)
        """
        peak_amplitude = magnitude[peak_idx]

        # Rise time (10% to 90%)
        rise_90_threshold = peak_amplitude * 0.9
        rise_10_threshold = peak_amplitude * 0.1

        rise_start_idx = peak_idx
        rise_end_idx = peak_idx

        # Find rise start (10%)
        for i in range(peak_idx, max(0, peak_idx - 50), -1):
            if magnitude[i] < rise_10_threshold:
                rise_start_idx = i
                break

        # Find rise end (90%)
        for i in range(rise_start_idx, peak_idx):
            if magnitude[i] > rise_90_threshold:
                rise_end_idx = i
                break

        rise_time_us = (rise_end_idx - rise_start_idx) / sample_rate * 1e6

        # Decay time (90% to 10%)
        decay_start_idx = peak_idx
        decay_end_idx = peak_idx

        # Find decay end (10%)
        for i in range(peak_idx, min(len(magnitude), peak_idx + 200)):
            if magnitude[i] < rise_10_threshold:
                decay_end_idx = i
                break

        decay_time_ms = (decay_end_idx - decay_start_idx) / sample_rate * 1000

        return max(0, rise_time_us), max(0, decay_time_ms)

    def _classify_impulse(
        self,
        peak_amplitude: float,
        duration_ms: float,
        freq_content: Dict[str, float],
    ) -> str:
        """Classify impulse type.

        Args:
            peak_amplitude: Peak amplitude
            duration_ms: Duration in ms
            freq_content: Frequency content

        Returns:
            Impulse type
        """
        # Simple classification based on characteristics
        if duration_ms < 1.0:
            return "sferics"  # Very short - atmospheric
        elif duration_ms < 10.0:
            if freq_content.get("high", 0) > 0.3:
                return "lightning"  # Short with high frequency content
            else:
                return "atmospheric"
        elif duration_ms < 100.0:
            return "distant_lightning"
        else:
            return "interference"  # Likely not atmospheric

    async def _identify_quiet_periods(
        self, noise_metrics: List[NoiseMetrics], band: str, center_frequency_khz: float
    ) -> List[QuietPeriod]:
        """Identify quiet periods from noise metrics.

        Args:
            noise_metrics: Noise analysis results
            band: Band name
            center_frequency_khz: Center frequency

        Returns:
            Identified quiet periods
        """
        if not noise_metrics:
            return []

        quiet_periods = []

        # Convert metrics to arrays for analysis
        timestamps = [m.timestamp for m in noise_metrics]
        rms_levels = np.array([m.rms_level for m in noise_metrics])
        quiet_ratios = np.array([m.quiet_ratio for m in noise_metrics])
        atmospheric_indices = np.array([m.atmospheric_index for m in noise_metrics])

        # Convert RMS to dB
        rms_db = 20 * np.log10(rms_levels + 1e-12)

        # Define quiet criteria
        quiet_mask = (
            (rms_db < self.quiet_threshold_db)
            & (quiet_ratios > 0.6)  # High quiet ratio
            & (atmospheric_indices < 30)  # Low atmospheric activity
        )

        # Find continuous quiet segments
        quiet_segments = self._find_quiet_segments(quiet_mask, timestamps)

        for start_time, end_time in quiet_segments:
            duration = (end_time - start_time).total_seconds()

            if duration >= self.min_quiet_duration:
                # Calculate quality metrics for this period
                start_idx = next(i for i, t in enumerate(timestamps) if t >= start_time)
                end_idx = next(i for i, t in enumerate(timestamps) if t > end_time)

                period_rms = rms_levels[start_idx:end_idx]
                period_quiet_ratios = quiet_ratios[start_idx:end_idx]

                avg_noise_level = np.mean(period_rms)
                min_noise_level = np.min(period_rms)

                # Quality score (0-1, higher = better)
                quality_score = np.mean(period_quiet_ratios) * (1.0 - min_noise_level)

                quiet_period = QuietPeriod(
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    avg_noise_level=float(avg_noise_level),
                    min_noise_level=float(min_noise_level),
                    quality_score=float(quality_score),
                    band=band,
                    frequency_khz=center_frequency_khz,
                )

                quiet_periods.append(quiet_period)

        return quiet_periods

    def _find_quiet_segments(
        self, quiet_mask: np.ndarray, timestamps: List[datetime]
    ) -> List[Tuple[datetime, datetime]]:
        """Find continuous quiet segments.

        Args:
            quiet_mask: Boolean mask of quiet periods
            timestamps: Timestamp array

        Returns:
            List of (start_time, end_time) tuples
        """
        segments = []
        in_quiet = False
        start_time = None

        for i, is_quiet in enumerate(quiet_mask):
            if is_quiet and not in_quiet:
                # Start of quiet period
                start_time = timestamps[i]
                in_quiet = True
            elif not is_quiet and in_quiet:
                # End of quiet period
                end_time = timestamps[i - 1] if i > 0 else timestamps[i]
                segments.append((start_time, end_time))
                in_quiet = False

        # Handle quiet period that goes to end
        if in_quiet and start_time:
            segments.append((start_time, timestamps[-1]))

        return segments

    async def store_qrn_analysis(
        self,
        noise_metrics: List[NoiseMetrics],
        quiet_periods: List[QuietPeriod],
        impulses: List[AtmosphericImpulse],
        session_id: str,
    ):
        """Store QRN analysis results in database.

        Args:
            noise_metrics: Noise metrics
            quiet_periods: Quiet periods
            impulses: Atmospheric impulses
            session_id: Recording session ID
        """
        try:
            # Store noise metrics as QRN samples
            for metrics in noise_metrics:
                qrn_sample = QRNSample(
                    session_id=session_id,
                    timestamp=metrics.timestamp,
                    rms_level=metrics.rms_level,
                    peak_level=metrics.peak_level,
                    impulse_count=metrics.impulse_count,
                    impulse_rate=metrics.impulse_rate,
                    quiet_ratio=metrics.quiet_ratio,
                    atmospheric_index=metrics.atmospheric_index,
                    spectral_data=json.dumps(metrics.spectral_density),
                    band_utilization=json.dumps(metrics.band_utilization),
                )
                self.db.add(qrn_sample)

            # Store atmospheric events (impulses and quiet periods)
            for impulse in impulses:
                event = AtmosphericEvent(
                    session_id=session_id,
                    timestamp=impulse.timestamp,
                    event_type="impulse",
                    peak_amplitude=impulse.peak_amplitude,
                    duration_ms=impulse.duration_ms,
                    rise_time_us=impulse.rise_time_us,
                    decay_time_ms=impulse.decay_time_ms,
                    frequency_content=json.dumps(impulse.frequency_content),
                    classification=impulse.impulse_type,
                )
                self.db.add(event)

            for quiet_period in quiet_periods:
                event = AtmosphericEvent(
                    session_id=session_id,
                    timestamp=quiet_period.start_time,
                    event_type="quiet_period",
                    duration_ms=quiet_period.duration_seconds * 1000,
                    avg_noise_level=quiet_period.avg_noise_level,
                    min_noise_level=quiet_period.min_noise_level,
                    quality_score=quiet_period.quality_score,
                    classification=f"quiet_{quiet_period.band}",
                )
                self.db.add(event)

            self.db.commit()
            logger.info(
                f"Stored QRN analysis: {len(noise_metrics)} metrics, "
                f"{len(quiet_periods)} quiet periods, {len(impulses)} impulses"
            )

        except Exception as e:
            logger.error(f"Error storing QRN analysis: {e}")
            self.db.rollback()

    async def generate_qrn_summary(
        self, session_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Generate QRN summary for a session.

        Args:
            session_id: Session ID
            start_time: Analysis start time
            end_time: Analysis end time

        Returns:
            QRN summary
        """
        try:
            # Query QRN samples for this session
            qrn_samples = (
                self.db.query(QRNSample)
                .filter(
                    QRNSample.session_id == session_id,
                    QRNSample.timestamp >= start_time,
                    QRNSample.timestamp <= end_time,
                )
                .order_by(QRNSample.timestamp)
                .all()
            )

            if not qrn_samples:
                return {"error": "No QRN data found"}

            # Calculate summary statistics
            rms_levels = [s.rms_level for s in qrn_samples]
            quiet_ratios = [s.quiet_ratio for s in qrn_samples]
            atmospheric_indices = [s.atmospheric_index for s in qrn_samples]

            summary = {
                "session_id": session_id,
                "analysis_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600,
                },
                "noise_statistics": {
                    "avg_rms_level": float(np.mean(rms_levels)),
                    "min_rms_level": float(np.min(rms_levels)),
                    "max_rms_level": float(np.max(rms_levels)),
                    "rms_std": float(np.std(rms_levels)),
                },
                "quiet_analysis": {
                    "avg_quiet_ratio": float(np.mean(quiet_ratios)),
                    "quiet_periods_count": len([q for q in quiet_ratios if q > 0.6]),
                    "quiet_time_percentage": float(
                        len([q for q in quiet_ratios if q > 0.6]) / len(quiet_ratios) * 100
                    ),
                },
                "atmospheric_activity": {
                    "avg_atmospheric_index": float(np.mean(atmospheric_indices)),
                    "max_atmospheric_index": float(np.max(atmospheric_indices)),
                    "high_activity_periods": len([a for a in atmospheric_indices if a > 70]),
                },
                "data_quality": {
                    "total_samples": len(qrn_samples),
                    "coverage_percentage": 100.0,  # Would calculate actual coverage
                    "recommended_for_training": np.mean(quiet_ratios) > 0.4,
                },
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating QRN summary: {e}")
            return {"error": str(e)}

    def close(self):
        """Close analyzer resources."""
        if self.db:
            self.db.close()


async def analyze_qrn_file(file_path: str, band: str) -> Dict[str, Any]:
    """Analyze QRN from IQ file (for testing).

    Args:
        file_path: IQ file path
        band: Band name

    Returns:
        Analysis results
    """
    analyzer = QRNAnalyzer()

    try:
        # Load IQ data (placeholder - would use proper IQ file format)
        iq_data = np.fromfile(file_path, dtype=np.complex64)
        sample_rate = 12000  # Default

        # Analyze
        noise_metrics, quiet_periods, impulses = await analyzer.analyze_iq_data(
            iq_data, sample_rate, 14080, band, datetime.utcnow()
        )

        return {
            "file": file_path,
            "band": band,
            "sample_rate": sample_rate,
            "duration_seconds": len(iq_data) / sample_rate,
            "noise_windows": len(noise_metrics),
            "quiet_periods_found": len(quiet_periods),
            "impulses_detected": len(impulses),
            "avg_atmospheric_index": float(
                np.mean([m.atmospheric_index for m in noise_metrics])
            ),
            "best_quiet_period": (
                asdict(max(quiet_periods, key=lambda q: q.quality_score))
                if quiet_periods
                else None
            ),
        }

    finally:
        analyzer.close()