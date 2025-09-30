"""Data quality validation for IQ recordings.

Implements T037: Quality validation (FR-013, FR-019).
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from enum import Enum

from scipy import signal
import soundfile as sf

logger = logging.getLogger(__name__)


class QualityStatus(Enum):
    """Quality check status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class QualityMetrics:
    """IQ data quality metrics."""

    # Basic parameters
    sample_rate: int
    bit_depth: int
    duration_seconds: float
    total_samples: int

    # IQ balance
    iq_balance_db: float
    iq_phase_error_deg: float
    dc_offset_i: float
    dc_offset_q: float

    # Dynamic range
    snr_db: float
    dynamic_range_db: float
    noise_floor_dbfs: float
    peak_level_dbfs: float

    # Signal quality
    clipping_percent: float
    saturation_percent: float
    dropout_count: int
    dropout_duration_ms: float

    # GPS validation (FR-013)
    gps_lock_present: bool
    gps_timestamp_valid: bool
    gps_accuracy_meters: Optional[float]

    # Overall score
    quality_score: float
    status: str
    issues: List[str]


class QualityCheck:
    """IQ recording quality validator."""

    # Quality thresholds
    MIN_SNR_DB = 20.0
    MAX_DC_OFFSET = 0.05  # 5% of full scale
    MAX_IQ_IMBALANCE_DB = 3.0
    MAX_PHASE_ERROR_DEG = 10.0
    MAX_CLIPPING_PERCENT = 1.0
    MAX_DROPOUT_MS = 100.0
    MIN_DYNAMIC_RANGE_DB = 40.0

    def __init__(self):
        """Initialize quality checker."""
        pass

    def check_file(self, file_path: Path, metadata: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """Perform comprehensive quality check on FLAC file.

        Args:
            file_path: Path to FLAC file
            metadata: Optional metadata with GPS info

        Returns:
            QualityMetrics with assessment
        """
        try:
            # Load IQ data
            iq_data, sample_rate = sf.read(file_path, dtype="float32")

            # Validate basic parameters
            basic_metrics = self._check_basic_parameters(iq_data, sample_rate, file_path)

            # Check IQ balance
            iq_metrics = self._check_iq_balance(iq_data)

            # Check dynamic range
            dynamic_metrics = self._check_dynamic_range(iq_data)

            # Check for clipping and saturation
            clipping_metrics = self._check_clipping(iq_data)

            # Check for dropouts
            dropout_metrics = self._check_dropouts(iq_data, sample_rate)

            # Check GPS lock (FR-013)
            gps_metrics = self._check_gps_lock(metadata)

            # Calculate overall quality score
            quality_score, status, issues = self._calculate_quality_score(
                iq_metrics, dynamic_metrics, clipping_metrics, dropout_metrics, gps_metrics
            )

            return QualityMetrics(
                **basic_metrics,
                **iq_metrics,
                **dynamic_metrics,
                **clipping_metrics,
                **dropout_metrics,
                **gps_metrics,
                quality_score=quality_score,
                status=status.value,
                issues=issues,
            )

        except Exception as e:
            logger.error(f"Quality check failed for {file_path}: {e}")
            raise

    def check_live_stream(self, iq_data: np.ndarray, sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> QualityMetrics:
        """Perform quality check on live IQ stream.

        Args:
            iq_data: IQ samples (Nx2 array)
            sample_rate: Sample rate in Hz
            metadata: Optional metadata with GPS info

        Returns:
            QualityMetrics with assessment
        """
        # Basic parameters
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        basic_metrics = {
            "sample_rate": sample_rate,
            "bit_depth": 16,  # Assume 16-bit
            "duration_seconds": len(iq_data) / sample_rate,
            "total_samples": len(iq_data),
        }

        # IQ balance
        iq_metrics = self._check_iq_balance(iq_data)

        # Dynamic range
        dynamic_metrics = self._check_dynamic_range(iq_data)

        # Clipping
        clipping_metrics = self._check_clipping(iq_data)

        # Dropouts
        dropout_metrics = self._check_dropouts(iq_data, sample_rate)

        # Check GPS lock (FR-013)
        gps_metrics = self._check_gps_lock(metadata)

        # Overall score
        quality_score, status, issues = self._calculate_quality_score(
            iq_metrics, dynamic_metrics, clipping_metrics, dropout_metrics, gps_metrics
        )

        return QualityMetrics(
            **basic_metrics,
            **iq_metrics,
            **dynamic_metrics,
            **clipping_metrics,
            **dropout_metrics,
            **gps_metrics,
            quality_score=quality_score,
            status=status.value,
            issues=issues,
        )

    def _check_basic_parameters(self, iq_data: np.ndarray, sample_rate: int, file_path: Path) -> Dict[str, Any]:
        """Check basic recording parameters.

        Args:
            iq_data: IQ samples
            sample_rate: Sample rate
            file_path: File path

        Returns:
            Basic metrics dict
        """
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        total_samples = len(iq_data)
        duration = total_samples / sample_rate

        # Check file info
        info = sf.info(file_path)

        return {
            "sample_rate": sample_rate,
            "bit_depth": 16 if info.subtype == "PCM_16" else 24,
            "duration_seconds": duration,
            "total_samples": total_samples,
        }

    def _check_iq_balance(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Check IQ balance and DC offset.

        Args:
            iq_data: IQ samples (Nx2 array)

        Returns:
            IQ balance metrics
        """
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        i_channel = iq_data[:, 0]
        q_channel = iq_data[:, 1]

        # DC offset
        dc_offset_i = float(np.mean(i_channel))
        dc_offset_q = float(np.mean(q_channel))

        # Remove DC for power calculations
        i_centered = i_channel - dc_offset_i
        q_centered = q_channel - dc_offset_q

        # Calculate power in each channel
        i_power = np.mean(i_centered**2)
        q_power = np.mean(q_centered**2)

        # IQ balance in dB
        if q_power > 0:
            iq_balance_db = float(10 * np.log10(i_power / q_power))
        else:
            iq_balance_db = 0.0

        # Phase error (should be 90 degrees)
        # Calculate cross-correlation
        correlation = np.corrcoef(i_centered, q_centered)[0, 1]
        phase_error_deg = float(np.abs(np.arcsin(correlation) * 180 / np.pi))

        return {
            "iq_balance_db": iq_balance_db,
            "iq_phase_error_deg": phase_error_deg,
            "dc_offset_i": dc_offset_i,
            "dc_offset_q": dc_offset_q,
        }

    def _check_dynamic_range(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Check dynamic range and SNR.

        Args:
            iq_data: IQ samples

        Returns:
            Dynamic range metrics
        """
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        # Convert to complex
        iq_complex = iq_data[:, 0] + 1j * iq_data[:, 1]

        # Calculate power spectrum
        freqs, psd = signal.welch(iq_complex, nperseg=1024, return_onesided=False)

        # Convert to dBFS
        psd_db = 10 * np.log10(psd + 1e-10)

        # Noise floor (median of lower 10% of spectrum)
        noise_floor_dbfs = float(np.median(np.sort(psd_db)[: len(psd_db) // 10]))

        # Peak level
        peak_level_dbfs = float(np.max(psd_db))

        # Dynamic range
        dynamic_range_db = peak_level_dbfs - noise_floor_dbfs

        # Estimate SNR (difference between signal peaks and noise floor)
        # Signal is top 10% of spectrum
        signal_level = float(np.median(np.sort(psd_db)[-(len(psd_db) // 10) :]))
        snr_db = signal_level - noise_floor_dbfs

        return {
            "snr_db": snr_db,
            "dynamic_range_db": dynamic_range_db,
            "noise_floor_dbfs": noise_floor_dbfs,
            "peak_level_dbfs": peak_level_dbfs,
        }

    def _check_clipping(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Check for clipping and saturation.

        Args:
            iq_data: IQ samples

        Returns:
            Clipping metrics
        """
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        # Clipping threshold (99% of full scale)
        clip_threshold = 0.99

        # Count samples exceeding threshold
        clipped = np.abs(iq_data) > clip_threshold
        clipping_count = np.sum(clipped)
        clipping_percent = float(clipping_count / (iq_data.size) * 100)

        # Saturation (consecutive samples at max)
        saturated = np.abs(iq_data) > 0.999
        saturation_runs = self._find_runs(saturated.flatten())
        saturation_percent = float(len(saturation_runs) / len(iq_data) * 100)

        return {"clipping_percent": clipping_percent, "saturation_percent": saturation_percent}

    def _check_dropouts(self, iq_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Check for data dropouts and gaps.

        Args:
            iq_data: IQ samples
            sample_rate: Sample rate

        Returns:
            Dropout metrics
        """
        if len(iq_data.shape) == 1:
            iq_data = iq_data.reshape(-1, 2)

        # Detect dropouts (consecutive zeros or very low power)
        power = np.sum(iq_data**2, axis=1)
        threshold = np.median(power) * 0.01  # 1% of median power

        dropout_mask = power < threshold
        dropout_runs = self._find_runs(dropout_mask)

        # Calculate dropout metrics
        dropout_count = len(dropout_runs)

        if dropout_count > 0:
            max_dropout_samples = max(end - start for start, end in dropout_runs)
            dropout_duration_ms = float(max_dropout_samples / sample_rate * 1000)
        else:
            dropout_duration_ms = 0.0

        return {"dropout_count": dropout_count, "dropout_duration_ms": dropout_duration_ms}

    def _find_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find consecutive runs of True values.

        Args:
            mask: Boolean array

        Returns:
            List of (start, end) index tuples
        """
        # Pad with False to detect runs at edges
        padded = np.concatenate([[False], mask, [False]])

        # Find transitions
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        return list(zip(starts, ends))

    def _check_gps_lock(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check GPS lock status and timestamp validity.

        Args:
            metadata: Optional metadata containing GPS info

        Returns:
            GPS lock metrics
        """
        # Default values if no metadata
        if metadata is None:
            return {
                "gps_lock_present": False,
                "gps_timestamp_valid": False,
                "gps_accuracy_meters": None,
            }

        # Check GPS lock status
        gps_lock_present = metadata.get("gps_lock", False)

        # Check timestamp validity
        gps_timestamp_valid = False
        if "gps_timestamp" in metadata and metadata["gps_timestamp"]:
            try:
                # Validate timestamp is reasonable (within last 24 hours)
                from datetime import datetime, timezone, timedelta
                if isinstance(metadata["gps_timestamp"], str):
                    gps_time = datetime.fromisoformat(metadata["gps_timestamp"].replace('Z', '+00:00'))
                else:
                    gps_time = metadata["gps_timestamp"]

                now = datetime.now(timezone.utc)
                time_diff = abs((now - gps_time).total_seconds())
                # GPS time should be within 24 hours of current time
                gps_timestamp_valid = time_diff < 86400
            except Exception as e:
                logger.debug(f"GPS timestamp validation failed: {e}")
                gps_timestamp_valid = False

        # Get GPS accuracy if available
        gps_accuracy_meters = metadata.get("gps_accuracy_meters", None)
        if gps_accuracy_meters is not None:
            try:
                gps_accuracy_meters = float(gps_accuracy_meters)
            except (ValueError, TypeError):
                gps_accuracy_meters = None

        return {
            "gps_lock_present": gps_lock_present,
            "gps_timestamp_valid": gps_timestamp_valid,
            "gps_accuracy_meters": gps_accuracy_meters,
        }

    def _calculate_quality_score(
        self,
        iq_metrics: Dict[str, float],
        dynamic_metrics: Dict[str, float],
        clipping_metrics: Dict[str, float],
        dropout_metrics: Dict[str, Any],
        gps_metrics: Dict[str, Any] = None,
    ) -> Tuple[float, QualityStatus, List[str]]:
        """Calculate overall quality score.

        Args:
            iq_metrics: IQ balance metrics
            dynamic_metrics: Dynamic range metrics
            clipping_metrics: Clipping metrics
            dropout_metrics: Dropout metrics
            gps_metrics: GPS lock metrics (optional for backward compatibility)

        Returns:
            Tuple of (score, status, issues)
        """
        score = 100.0
        issues = []

        # Check IQ balance
        if abs(iq_metrics["iq_balance_db"]) > self.MAX_IQ_IMBALANCE_DB:
            score -= 10
            issues.append(f"IQ imbalance: {iq_metrics['iq_balance_db']:.1f} dB")

        # Check phase error
        if iq_metrics["iq_phase_error_deg"] > self.MAX_PHASE_ERROR_DEG:
            score -= 10
            issues.append(f"Phase error: {iq_metrics['iq_phase_error_deg']:.1f} deg")

        # Check DC offset
        if abs(iq_metrics["dc_offset_i"]) > self.MAX_DC_OFFSET or abs(iq_metrics["dc_offset_q"]) > self.MAX_DC_OFFSET:
            score -= 5
            issues.append(f"DC offset: I={iq_metrics['dc_offset_i']:.3f}, Q={iq_metrics['dc_offset_q']:.3f}")

        # Check SNR
        if dynamic_metrics["snr_db"] < self.MIN_SNR_DB:
            score -= 15
            issues.append(f"Low SNR: {dynamic_metrics['snr_db']:.1f} dB")

        # Check dynamic range
        if dynamic_metrics["dynamic_range_db"] < self.MIN_DYNAMIC_RANGE_DB:
            score -= 10
            issues.append(f"Low dynamic range: {dynamic_metrics['dynamic_range_db']:.1f} dB")

        # Check clipping
        if clipping_metrics["clipping_percent"] > self.MAX_CLIPPING_PERCENT:
            score -= 20
            issues.append(f"Clipping: {clipping_metrics['clipping_percent']:.2f}%")

        # Check saturation
        if clipping_metrics["saturation_percent"] > 0.1:
            score -= 15
            issues.append(f"Saturation: {clipping_metrics['saturation_percent']:.2f}%")

        # Check dropouts
        if dropout_metrics["dropout_duration_ms"] > self.MAX_DROPOUT_MS:
            score -= 20
            issues.append(f"Dropouts: {dropout_metrics['dropout_count']} ({dropout_metrics['dropout_duration_ms']:.1f} ms max)")

        # Check GPS lock (FR-013) - CRITICAL for timestamp accuracy
        if gps_metrics:
            if not gps_metrics["gps_lock_present"]:
                score -= 25  # Significant penalty for missing GPS lock
                issues.append("GPS lock missing - timestamps may be inaccurate")

            if not gps_metrics["gps_timestamp_valid"]:
                score -= 15
                issues.append("GPS timestamp invalid or stale")

            # Check GPS accuracy if available
            if gps_metrics["gps_accuracy_meters"] is not None:
                if gps_metrics["gps_accuracy_meters"] > 100:
                    score -= 5
                    issues.append(f"Poor GPS accuracy: {gps_metrics['gps_accuracy_meters']:.1f} meters")

        # Clamp score
        score = max(0.0, min(100.0, score))

        # Determine status
        if score >= 80 and len(issues) == 0:
            status = QualityStatus.PASSED
        elif score >= 60:
            status = QualityStatus.WARNING
        else:
            status = QualityStatus.FAILED

        return score, status, issues

    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate recording metadata completeness.

        Args:
            metadata: Metadata dict

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        required_fields = [
            "session_id",
            "kiwisdr_url",
            "center_frequency_hz",
            "sample_rate",
            "start_time",
            "end_time",
            "band",
        ]

        missing = []
        for field in required_fields:
            if field not in metadata or metadata[field] is None:
                missing.append(field)

        return len(missing) == 0, missing


# Convenience functions
def check_recording_quality(file_path: Path) -> QualityMetrics:
    """Check recording quality.

    Args:
        file_path: Path to FLAC file

    Returns:
        QualityMetrics
    """
    checker = QualityCheck()
    return checker.check_file(file_path)


def quick_quality_check(iq_data: np.ndarray, sample_rate: int) -> float:
    """Quick quality score for live monitoring.

    Args:
        iq_data: IQ samples
        sample_rate: Sample rate

    Returns:
        Quality score (0-100)
    """
    checker = QualityCheck()
    metrics = checker.check_live_stream(iq_data, sample_rate)
    return metrics.quality_score