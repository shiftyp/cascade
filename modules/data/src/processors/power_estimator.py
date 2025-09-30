"""SNR-distance based power estimation for FT8/WSPR signals.

T093: Triangulate transmitter power using SNR reports from multiple receivers
with known locations and distances.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
from scipy import optimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class PowerEstimate:
    """Power estimate from SNR-distance triangulation."""

    station_hash: str
    timestamp: datetime
    estimated_power_dbm: float
    confidence_score: float  # 0-1 confidence in estimate

    # Components of the estimate
    num_receivers: int
    avg_snr_db: float
    avg_distance_km: float
    std_dev_dbm: float  # Standard deviation of estimates

    # Method used
    estimation_method: str  # 'triangulation', 'single_rx', 'statistical'

    # Raw measurements
    receiver_reports: List[Dict[str, Any]]  # SNR, distance, grid for each RX


class SNRDistancePowerEstimator:
    """Estimates transmitter power from multiple receiver SNR reports."""

    def __init__(self):
        """Initialize power estimator."""
        self.estimates: Dict[str, List[PowerEstimate]] = {}

        # Path loss model parameters
        self.freq_mhz_default = 14.0  # Default to 20m band
        self.ground_reflection_gain = 6.0  # dB for low angles
        self.atmospheric_loss_db_per_1000km = 0.5
        self.ionospheric_absorption_db = 2.0  # Average

        # Receiver assumptions
        self.rx_noise_figure_db = 10.0  # Typical amateur receiver
        self.rx_antenna_gain_dbi = 3.0  # Typical dipole
        self.thermal_noise_dbm = -174.0  # dBm/Hz at 290K
        self.bandwidth_hz = 50.0  # FT8 effective bandwidth

    def estimate_power(self,
                      station_hash: str,
                      receiver_reports: List[Dict[str, Any]],
                      timestamp: datetime,
                      frequency_hz: Optional[float] = None) -> PowerEstimate:
        """Estimate transmitter power from receiver reports.

        Args:
            station_hash: Anonymized station identifier
            receiver_reports: List of receiver reports with:
                - rx_hash: Receiver station hash
                - snr_db: Reported SNR
                - distance_km: Distance to transmitter
                - rx_grid: Receiver grid square
                - timestamp: Reception time
            timestamp: Transmission timestamp
            frequency_hz: Operating frequency (optional)

        Returns:
            PowerEstimate with triangulated power
        """
        if not receiver_reports:
            return self._create_null_estimate(station_hash, timestamp)

        # Filter valid reports
        valid_reports = [r for r in receiver_reports
                        if r.get('snr_db') is not None
                        and r.get('distance_km', 0) > 0]

        if not valid_reports:
            return self._create_null_estimate(station_hash, timestamp)

        # Use provided frequency or default
        freq_mhz = (frequency_hz / 1e6) if frequency_hz else self.freq_mhz_default

        # Estimate power from each receiver
        power_estimates = []
        for report in valid_reports:
            power_dbm = self._estimate_from_single_rx(
                report['snr_db'],
                report['distance_km'],
                freq_mhz
            )
            power_estimates.append(power_dbm)

        # Combine estimates
        if len(power_estimates) >= 3:
            # Triangulation with outlier rejection
            estimated_power, confidence = self._triangulate_power(power_estimates)
            method = 'triangulation'
        elif len(power_estimates) == 2:
            # Simple average
            estimated_power = np.mean(power_estimates)
            confidence = 0.6
            method = 'dual_rx'
        else:
            # Single receiver
            estimated_power = power_estimates[0]
            confidence = 0.4
            method = 'single_rx'

        # Calculate statistics
        avg_snr = np.mean([r['snr_db'] for r in valid_reports])
        avg_distance = np.mean([r['distance_km'] for r in valid_reports])
        std_dev = np.std(power_estimates) if len(power_estimates) > 1 else 0.0

        # Adjust confidence based on consistency
        if std_dev > 10:  # High variance
            confidence *= 0.7
        elif std_dev < 3:  # Very consistent
            confidence = min(1.0, confidence * 1.2)

        estimate = PowerEstimate(
            station_hash=station_hash,
            timestamp=timestamp,
            estimated_power_dbm=estimated_power,
            confidence_score=confidence,
            num_receivers=len(valid_reports),
            avg_snr_db=avg_snr,
            avg_distance_km=avg_distance,
            std_dev_dbm=std_dev,
            estimation_method=method,
            receiver_reports=valid_reports
        )

        # Store estimate
        if station_hash not in self.estimates:
            self.estimates[station_hash] = []
        self.estimates[station_hash].append(estimate)

        return estimate

    def _estimate_from_single_rx(self, snr_db: float, distance_km: float,
                                freq_mhz: float) -> float:
        """Estimate power from single receiver report.

        Args:
            snr_db: Reported SNR in dB
            distance_km: Distance to transmitter
            freq_mhz: Frequency in MHz

        Returns:
            Estimated power in dBm
        """
        # Calculate free space path loss (Friis equation)
        fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(freq_mhz) + 32.45

        # Add atmospheric losses
        atmospheric_loss = (distance_km / 1000) * self.atmospheric_loss_db_per_1000km

        # Add ionospheric losses (simplified model)
        if distance_km > 1000:
            # Skip distance with ionospheric reflection
            ionospheric_loss = self.ionospheric_absorption_db
        else:
            # Ground wave
            ionospheric_loss = 0

        # Total path loss
        path_loss_db = fspl_db + atmospheric_loss + ionospheric_loss

        # Account for ground reflection gain at low angles
        if distance_km > 2000:
            path_loss_db -= self.ground_reflection_gain

        # Calculate noise floor
        noise_floor_dbm = (self.thermal_noise_dbm +
                          10 * np.log10(self.bandwidth_hz) +
                          self.rx_noise_figure_db)

        # Signal power at receiver
        rx_signal_dbm = snr_db + noise_floor_dbm

        # Add receiver antenna gain
        rx_signal_dbm -= self.rx_antenna_gain_dbi

        # Estimate transmitter power (assuming unity TX antenna gain)
        tx_power_dbm = rx_signal_dbm + path_loss_db

        # Sanity check
        if tx_power_dbm < -10:  # Less than 0.1mW
            tx_power_dbm = -10
        elif tx_power_dbm > 63:  # More than 2kW
            tx_power_dbm = 63

        return tx_power_dbm

    def _triangulate_power(self, estimates: List[float]) -> Tuple[float, float]:
        """Triangulate power from multiple estimates with outlier rejection.

        Args:
            estimates: List of power estimates in dBm

        Returns:
            Tuple of (estimated_power_dbm, confidence_score)
        """
        estimates = np.array(estimates)

        # Remove outliers using IQR method
        q1, q3 = np.percentile(estimates, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = estimates[(estimates >= lower_bound) & (estimates <= upper_bound)]

        if len(filtered) < len(estimates) * 0.5:
            # Too many outliers, use median instead
            estimated_power = np.median(estimates)
            confidence = 0.5
        else:
            # Use mean of filtered values
            estimated_power = np.mean(filtered)

            # Confidence based on consistency and number of measurements
            std_dev = np.std(filtered)
            num_factor = min(1.0, len(filtered) / 10)  # Max confidence at 10+ RX
            consistency_factor = max(0.3, 1.0 - std_dev / 20)  # Lower confidence with high variance

            confidence = num_factor * consistency_factor

        return estimated_power, confidence

    def _create_null_estimate(self, station_hash: str, timestamp: datetime) -> PowerEstimate:
        """Create null estimate when insufficient data."""
        return PowerEstimate(
            station_hash=station_hash,
            timestamp=timestamp,
            estimated_power_dbm=30.0,  # Default 1W
            confidence_score=0.0,
            num_receivers=0,
            avg_snr_db=0.0,
            avg_distance_km=0.0,
            std_dev_dbm=0.0,
            estimation_method='none',
            receiver_reports=[]
        )

    def get_station_power_history(self, station_hash: str) -> List[PowerEstimate]:
        """Get power estimation history for a station.

        Args:
            station_hash: Station identifier

        Returns:
            List of power estimates
        """
        return self.estimates.get(station_hash, [])

    def get_average_power(self, station_hash: str,
                         min_confidence: float = 0.5) -> Optional[float]:
        """Get average estimated power for a station.

        Args:
            station_hash: Station identifier
            min_confidence: Minimum confidence threshold

        Returns:
            Average power in dBm or None
        """
        estimates = self.get_station_power_history(station_hash)

        # Filter by confidence
        valid = [e for e in estimates if e.confidence_score >= min_confidence]

        if not valid:
            return None

        # Weight by confidence
        weights = np.array([e.confidence_score for e in valid])
        powers = np.array([e.estimated_power_dbm for e in valid])

        return np.average(powers, weights=weights)

    def calibrate_with_known_power(self, station_hash: str,
                                  actual_power_dbm: float):
        """Calibrate estimator with known transmitter power.

        Args:
            station_hash: Station with known power
            actual_power_dbm: Actual transmitter power
        """
        estimates = self.get_station_power_history(station_hash)

        if not estimates:
            return

        # Calculate average error
        errors = [actual_power_dbm - e.estimated_power_dbm for e in estimates]
        avg_error = np.mean(errors)

        logger.info(f"Calibration for {station_hash[:8]}: avg error = {avg_error:.1f} dB")

        # Could store calibration factor for future use
        # self.calibration_offsets[station_hash] = avg_error