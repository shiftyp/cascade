"""Multi-band power correlation for consistency checking.

T097: Track stations across multiple bands and correlate power estimates
accounting for band-specific propagation differences.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BandObservation:
    """Power observation on a specific band."""

    station_hash: str
    timestamp: datetime
    band: str
    frequency_mhz: float

    # Power estimates
    estimated_power_dbm: float
    snr_db: float
    distance_km: float

    # Propagation context
    solar_flux: Optional[float]
    k_index: Optional[float]
    time_of_day: str  # 'day', 'night', 'gray_line'


@dataclass
class MultiBandPowerProfile:
    """Correlated power profile across multiple bands."""

    station_hash: str
    analysis_time: datetime

    # Per-band estimates
    band_powers: Dict[str, float]  # band -> power_dbm
    band_observations: Dict[str, int]  # band -> count

    # Correlation analysis
    power_consistency_score: float  # 0-1, higher = more consistent
    likely_power_dbm: float  # Most likely actual power
    confidence_score: float

    # Anomalies
    anomalous_bands: List[str]  # Bands with inconsistent power
    power_variance_db: float  # Variance across bands

    # Pattern detection
    uses_auto_tuner: bool  # Consistent power across bands
    manual_adjustments: bool  # Different power on different bands


class MultiBandPowerCorrelator:
    """Correlates power estimates across multiple amateur bands."""

    def __init__(self):
        """Initialize multi-band correlator."""
        self.observations: Dict[str, List[BandObservation]] = defaultdict(list)
        self.profiles: Dict[str, MultiBandPowerProfile] = {}

        # Band characteristics for propagation adjustment
        self.band_properties = {
            '160m': {'freq_mhz': 1.9, 'prop_factor': 1.5, 'noise_floor': -105},
            '80m': {'freq_mhz': 3.7, 'prop_factor': 1.3, 'noise_floor': -110},
            '40m': {'freq_mhz': 7.1, 'prop_factor': 1.2, 'noise_floor': -115},
            '30m': {'freq_mhz': 10.1, 'prop_factor': 1.1, 'noise_floor': -118},
            '20m': {'freq_mhz': 14.1, 'prop_factor': 1.0, 'noise_floor': -120},
            '17m': {'freq_mhz': 18.1, 'prop_factor': 0.95, 'noise_floor': -121},
            '15m': {'freq_mhz': 21.1, 'prop_factor': 0.9, 'noise_floor': -122},
            '12m': {'freq_mhz': 24.9, 'prop_factor': 0.85, 'noise_floor': -122},
            '10m': {'freq_mhz': 28.5, 'prop_factor': 0.8, 'noise_floor': -123},
            '6m': {'freq_mhz': 50.1, 'prop_factor': 0.7, 'noise_floor': -124}
        }

        # Correlation parameters
        self.min_bands_for_correlation = 3
        self.time_window_hours = 24
        self.anomaly_threshold_db = 10  # Power difference to flag as anomalous

    def add_observation(self, observation: BandObservation):
        """Add a power observation for correlation.

        Args:
            observation: Band-specific power observation
        """
        self.observations[observation.station_hash].append(observation)

    def correlate_station_power(self, station_hash: str,
                              time_window: Optional[timedelta] = None) -> Optional[MultiBandPowerProfile]:
        """Correlate power estimates across bands for a station.

        Args:
            station_hash: Station to analyze
            time_window: Time window for observations (default 24h)

        Returns:
            MultiBandPowerProfile or None if insufficient data
        """
        if station_hash not in self.observations:
            return None

        # Filter observations within time window
        if time_window is None:
            time_window = timedelta(hours=self.time_window_hours)

        now = datetime.now()
        recent_obs = [
            obs for obs in self.observations[station_hash]
            if now - obs.timestamp <= time_window
        ]

        # Group by band
        band_groups = defaultdict(list)
        for obs in recent_obs:
            band_groups[obs.band].append(obs)

        if len(band_groups) < self.min_bands_for_correlation:
            logger.debug(f"Insufficient bands for {station_hash}: {len(band_groups)}")
            return None

        # Calculate average power per band with propagation correction
        band_powers = {}
        band_counts = {}
        corrected_powers = []

        for band, obs_list in band_groups.items():
            # Average power for this band
            powers = [obs.estimated_power_dbm for obs in obs_list]
            avg_power = np.mean(powers)
            band_powers[band] = avg_power
            band_counts[band] = len(obs_list)

            # Apply propagation correction for correlation
            if band in self.band_properties:
                prop_factor = self.band_properties[band]['prop_factor']
                corrected_power = avg_power / prop_factor
                corrected_powers.append(corrected_power)

        # Analyze consistency
        corrected_powers = np.array(corrected_powers)
        power_variance = np.var(corrected_powers)
        power_std = np.sqrt(power_variance)

        # Consistency score (lower variance = higher consistency)
        consistency_score = max(0, 1 - power_std / 10)  # 10 dB std = 0 consistency

        # Identify anomalous bands
        median_corrected = np.median(corrected_powers)
        anomalous_bands = []

        for band, power in band_powers.items():
            if band in self.band_properties:
                corrected = power / self.band_properties[band]['prop_factor']
                if abs(corrected - median_corrected) > self.anomaly_threshold_db:
                    anomalous_bands.append(band)

        # Estimate most likely power (weighted by observations and consistency)
        weights = []
        powers_for_average = []

        for band, power in band_powers.items():
            if band not in anomalous_bands:
                weight = band_counts[band] * (1 if band == '20m' else 0.8)  # Prefer 20m
                weights.append(weight)
                powers_for_average.append(power)

        if weights:
            likely_power = np.average(powers_for_average, weights=weights)
        else:
            likely_power = np.median(list(band_powers.values()))

        # Detect patterns
        uses_auto_tuner = power_std < 3  # Very consistent = auto-tuner
        manual_adjustments = len(anomalous_bands) > 0 or power_std > 5

        # Calculate confidence
        confidence = self._calculate_confidence(
            len(band_groups), consistency_score, len(anomalous_bands)
        )

        profile = MultiBandPowerProfile(
            station_hash=station_hash,
            analysis_time=now,
            band_powers=band_powers,
            band_observations=band_counts,
            power_consistency_score=consistency_score,
            likely_power_dbm=likely_power,
            confidence_score=confidence,
            anomalous_bands=anomalous_bands,
            power_variance_db=power_variance,
            uses_auto_tuner=uses_auto_tuner,
            manual_adjustments=manual_adjustments
        )

        self.profiles[station_hash] = profile
        return profile

    def _calculate_confidence(self, num_bands: int, consistency: float,
                            num_anomalies: int) -> float:
        """Calculate confidence in multi-band correlation.

        Args:
            num_bands: Number of bands observed
            consistency: Consistency score
            num_anomalies: Number of anomalous bands

        Returns:
            Confidence score 0-1
        """
        # More bands = higher confidence
        band_factor = min(1.0, num_bands / 6)

        # Higher consistency = higher confidence
        consistency_factor = consistency

        # Fewer anomalies = higher confidence
        anomaly_factor = max(0.5, 1.0 - num_anomalies * 0.2)

        confidence = band_factor * 0.3 + consistency_factor * 0.5 + anomaly_factor * 0.2

        return confidence

    def find_inconsistent_stations(self, threshold: float = 0.5) -> List[str]:
        """Find stations with inconsistent power across bands.

        Args:
            threshold: Maximum consistency score to be flagged

        Returns:
            List of station hashes with inconsistent power
        """
        inconsistent = []

        for station_hash in self.observations.keys():
            profile = self.correlate_station_power(station_hash)

            if profile and profile.power_consistency_score < threshold:
                inconsistent.append(station_hash)

        return inconsistent

    def detect_band_specific_patterns(self, station_hash: str) -> Dict[str, Any]:
        """Detect band-specific operating patterns.

        Args:
            station_hash: Station to analyze

        Returns:
            Dictionary of detected patterns
        """
        if station_hash not in self.observations:
            return {}

        observations = self.observations[station_hash]

        # Group by band and time
        band_time_patterns = defaultdict(lambda: {'day': [], 'night': [], 'gray_line': []})

        for obs in observations:
            if obs.time_of_day:
                band_time_patterns[obs.band][obs.time_of_day].append(obs.estimated_power_dbm)

        patterns = {}

        # Analyze each band
        for band, time_data in band_time_patterns.items():
            band_pattern = {}

            # Check for time-based power changes
            day_powers = time_data['day']
            night_powers = time_data['night']

            if day_powers and night_powers:
                day_avg = np.mean(day_powers)
                night_avg = np.mean(night_powers)
                power_difference = night_avg - day_avg

                if abs(power_difference) > 3:
                    band_pattern['time_based_adjustment'] = power_difference
                    band_pattern['higher_at_night'] = power_difference > 0

            # Check for gray-line enhancement
            gray_line_powers = time_data['gray_line']
            if gray_line_powers and (day_powers or night_powers):
                gray_avg = np.mean(gray_line_powers)
                other_avg = np.mean(day_powers + night_powers)

                if gray_avg - other_avg > 3:
                    band_pattern['gray_line_boost'] = gray_avg - other_avg

            if band_pattern:
                patterns[band] = band_pattern

        return patterns

    def correlate_with_propagation(self, station_hash: str,
                                  solar_data: Optional[Dict] = None) -> Dict[str, float]:
        """Correlate power adjustments with propagation conditions.

        Args:
            station_hash: Station to analyze
            solar_data: Optional solar/propagation data

        Returns:
            Correlation coefficients
        """
        if station_hash not in self.observations:
            return {}

        observations = self.observations[station_hash]

        # Extract power and propagation metrics
        powers = []
        solar_flux_values = []
        k_indices = []

        for obs in observations:
            powers.append(obs.estimated_power_dbm)

            if obs.solar_flux is not None:
                solar_flux_values.append(obs.solar_flux)

            if obs.k_index is not None:
                k_indices.append(obs.k_index)

        correlations = {}

        # Correlate with solar flux (if available)
        if len(solar_flux_values) >= 10 and len(powers) == len(solar_flux_values):
            corr = np.corrcoef(powers[:len(solar_flux_values)], solar_flux_values)[0, 1]
            correlations['solar_flux_correlation'] = corr

        # Correlate with K-index
        if len(k_indices) >= 10 and len(powers) == len(k_indices):
            corr = np.corrcoef(powers[:len(k_indices)], k_indices)[0, 1]
            correlations['k_index_correlation'] = corr

        return correlations

    def generate_report(self, station_hash: str) -> str:
        """Generate human-readable multi-band power report.

        Args:
            station_hash: Station to report on

        Returns:
            Text report
        """
        profile = self.profiles.get(station_hash)
        if not profile:
            profile = self.correlate_station_power(station_hash)

        if not profile:
            return f"No multi-band data for station {station_hash[:8]}..."

        report = f"Multi-Band Power Analysis for {station_hash[:8]}...\n"
        report += "=" * 50 + "\n\n"

        report += f"Likely Power: {profile.likely_power_dbm:.1f} dBm "
        report += f"({10**((profile.likely_power_dbm-30)/10):.0f}W)\n"
        report += f"Consistency Score: {profile.power_consistency_score:.2f}\n"
        report += f"Confidence: {profile.confidence_score:.0%}\n\n"

        report += "Band-Specific Powers:\n"
        for band in sorted(profile.band_powers.keys(),
                         key=lambda b: self.band_properties.get(b, {}).get('freq_mhz', 0)):
            power = profile.band_powers[band]
            count = profile.band_observations[band]
            anomaly = " [ANOMALY]" if band in profile.anomalous_bands else ""
            report += f"  {band:5s}: {power:5.1f} dBm ({count:3d} obs){anomaly}\n"

        report += f"\nPower Variance: {profile.power_variance_db:.1f} dB\n"

        if profile.uses_auto_tuner:
            report += "Pattern: Likely using auto-tuner (consistent power)\n"
        if profile.manual_adjustments:
            report += "Pattern: Manual power adjustments detected\n"

        # Add band-specific patterns
        patterns = self.detect_band_specific_patterns(station_hash)
        if patterns:
            report += "\nBand-Specific Patterns:\n"
            for band, pattern in patterns.items():
                report += f"  {band}: "
                if 'higher_at_night' in pattern:
                    report += f"{'Higher' if pattern['higher_at_night'] else 'Lower'} power at night "
                if 'gray_line_boost' in pattern:
                    report += f"Gray-line boost: +{pattern['gray_line_boost']:.1f} dB"
                report += "\n"

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for multi-band correlation.

        Returns:
            Statistics dictionary
        """
        total_stations = len(self.observations)
        stations_with_profiles = len(self.profiles)

        consistency_scores = [p.power_consistency_score for p in self.profiles.values()]
        auto_tuner_count = sum(1 for p in self.profiles.values() if p.uses_auto_tuner)
        manual_adjust_count = sum(1 for p in self.profiles.values() if p.manual_adjustments)

        band_coverage = defaultdict(int)
        for obs_list in self.observations.values():
            bands = set(obs.band for obs in obs_list)
            for band in bands:
                band_coverage[band] += 1

        return {
            'total_stations': total_stations,
            'stations_analyzed': stations_with_profiles,
            'avg_consistency_score': np.mean(consistency_scores) if consistency_scores else 0,
            'auto_tuner_stations': auto_tuner_count,
            'manual_adjustment_stations': manual_adjust_count,
            'band_coverage': dict(band_coverage),
            'inconsistent_stations': len(self.find_inconsistent_stations())
        }