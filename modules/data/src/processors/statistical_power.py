"""Statistical power analysis using amateur radio population distributions.

T094: Estimate transmitter power using Bayesian inference based on known
amateur radio power level distributions and band-specific patterns.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


@dataclass
class StatisticalPowerEstimate:
    """Power estimate from statistical analysis."""

    station_hash: str
    timestamp: datetime

    # Power level probabilities
    qrp_probability: float  # ≤5W
    low_probability: float  # 10-25W
    typical_probability: float  # 50-100W
    high_probability: float  # 200-500W
    qro_probability: float  # 1000-1500W

    # Most likely power
    most_likely_power_dbm: float
    expected_power_dbm: float  # Weighted average
    confidence_score: float

    # Input features
    snr_db: float
    distance_km: float
    band: str
    time_of_day_utc: int
    is_contest: bool


class StatisticalPowerAnalyzer:
    """Estimates power using amateur radio statistical distributions."""

    def __init__(self):
        """Initialize statistical power analyzer."""
        self.estimates: Dict[str, List[StatisticalPowerEstimate]] = {}

        # Power level definitions (in watts and dBm)
        self.power_levels = {
            'qrp': {'watts': 5, 'dbm': 37.0, 'prior': 0.10},
            'low': {'watts': 25, 'dbm': 44.0, 'prior': 0.15},
            'typical': {'watts': 100, 'dbm': 50.0, 'prior': 0.50},
            'high': {'watts': 400, 'dbm': 56.0, 'prior': 0.20},
            'qro': {'watts': 1500, 'dbm': 61.8, 'prior': 0.05}
        }

        # Band-specific adjustments to priors
        self.band_factors = {
            '160m': {'qrp': 0.5, 'qro': 2.0},  # More high power on 160m
            '80m': {'qrp': 0.7, 'qro': 1.5},
            '40m': {'qrp': 0.9, 'qro': 1.1},
            '30m': {'qrp': 1.5, 'typical': 0.8},  # 30m limited to 200W in many regions
            '20m': {'qrp': 1.0, 'qro': 1.0},  # Baseline
            '17m': {'qrp': 1.2, 'typical': 1.1},
            '15m': {'qrp': 1.1, 'typical': 1.0},
            '12m': {'qrp': 1.2, 'typical': 1.0},
            '10m': {'qrp': 0.8, 'qro': 1.3},  # More power needed on 10m
            '6m': {'qrp': 0.6, 'qro': 1.5}
        }

        # Time-of-day patterns
        self.time_factors = {
            'night': {'qrp': 1.3, 'qro': 0.8},  # More QRP at night
            'gray_line': {'typical': 1.2, 'high': 1.3},  # Higher power at gray-line
            'day': {'typical': 1.0, 'qro': 1.1},
            'contest': {'qro': 2.0, 'high': 1.5, 'qrp': 0.5}  # Contest stations run high power
        }

    def analyze_power(self,
                      station_hash: str,
                      snr_db: float,
                      distance_km: float,
                      band: str,
                      timestamp: datetime,
                      is_contest: bool = False,
                      additional_features: Optional[Dict] = None) -> StatisticalPowerEstimate:
        """Analyze likely transmitter power using Bayesian inference.

        Args:
            station_hash: Anonymized station identifier
            snr_db: Received SNR
            distance_km: Distance to receiver
            band: Operating band (e.g., '20m')
            timestamp: Observation time
            is_contest: Whether during contest period
            additional_features: Optional features like equipment type

        Returns:
            StatisticalPowerEstimate with probability distribution
        """
        # Determine time category
        hour_utc = timestamp.hour
        time_category = self._get_time_category(hour_utc)

        # Get adjusted priors
        priors = self._get_adjusted_priors(band, time_category, is_contest)

        # Calculate likelihoods for each power level
        likelihoods = {}
        for level, params in self.power_levels.items():
            likelihood = self._calculate_likelihood(
                snr_db, distance_km, params['dbm'], band
            )
            likelihoods[level] = likelihood

        # Bayesian inference
        posteriors = self._calculate_posteriors(priors, likelihoods)

        # Find most likely power
        most_likely_level = max(posteriors.keys(), key=lambda k: posteriors[k])
        most_likely_power_dbm = self.power_levels[most_likely_level]['dbm']

        # Calculate expected (weighted average) power
        expected_power_dbm = sum(
            self.power_levels[level]['dbm'] * prob
            for level, prob in posteriors.items()
        )

        # Calculate confidence
        entropy = -sum(p * np.log(p + 1e-10) for p in posteriors.values())
        max_entropy = -np.log(1/5)  # Maximum entropy for 5 categories
        confidence = 1.0 - (entropy / max_entropy)

        # Adjust confidence based on SNR and distance
        if snr_db < -20 or snr_db > 20:
            confidence *= 0.8  # Extreme SNRs less reliable
        if distance_km < 100 or distance_km > 15000:
            confidence *= 0.9  # Extreme distances less reliable

        estimate = StatisticalPowerEstimate(
            station_hash=station_hash,
            timestamp=timestamp,
            qrp_probability=posteriors['qrp'],
            low_probability=posteriors['low'],
            typical_probability=posteriors['typical'],
            high_probability=posteriors['high'],
            qro_probability=posteriors['qro'],
            most_likely_power_dbm=most_likely_power_dbm,
            expected_power_dbm=expected_power_dbm,
            confidence_score=confidence,
            snr_db=snr_db,
            distance_km=distance_km,
            band=band,
            time_of_day_utc=hour_utc,
            is_contest=is_contest
        )

        # Store estimate
        if station_hash not in self.estimates:
            self.estimates[station_hash] = []
        self.estimates[station_hash].append(estimate)

        return estimate

    def _get_time_category(self, hour_utc: int) -> str:
        """Determine time category from UTC hour."""
        if 22 <= hour_utc or hour_utc < 6:
            return 'night'
        elif 6 <= hour_utc < 9 or 18 <= hour_utc < 22:
            return 'gray_line'
        else:
            return 'day'

    def _get_adjusted_priors(self, band: str, time_category: str,
                            is_contest: bool) -> Dict[str, float]:
        """Get priors adjusted for band, time, and contest."""
        priors = {}

        # Start with base priors
        for level, params in self.power_levels.items():
            priors[level] = params['prior']

        # Apply band-specific factors
        if band in self.band_factors:
            for level, factor in self.band_factors[band].items():
                if level in priors:
                    priors[level] *= factor

        # Apply time factors
        category = 'contest' if is_contest else time_category
        if category in self.time_factors:
            for level, factor in self.time_factors[category].items():
                if level in priors:
                    priors[level] *= factor

        # Normalize to sum to 1
        total = sum(priors.values())
        return {k: v/total for k, v in priors.items()}

    def _calculate_likelihood(self, snr_db: float, distance_km: float,
                             power_dbm: float, band: str) -> float:
        """Calculate likelihood of observing SNR given power and distance.

        Args:
            snr_db: Observed SNR
            distance_km: Distance to receiver
            power_dbm: Hypothetical transmitter power
            band: Operating band

        Returns:
            Likelihood value
        """
        # Expected SNR for this power and distance
        expected_snr = self._predict_snr(power_dbm, distance_km, band)

        # Model uncertainty (standard deviation)
        # Increases with distance due to propagation variability
        sigma = 5.0 + (distance_km / 1000)  # 5 dB base + 1 dB per 1000 km

        # Gaussian likelihood
        likelihood = stats.norm.pdf(snr_db, expected_snr, sigma)

        return likelihood

    def _predict_snr(self, power_dbm: float, distance_km: float, band: str) -> float:
        """Predict expected SNR for given power and distance.

        Simplified propagation model for likelihood calculation.
        """
        # Get frequency for band
        freq_mhz = self._get_band_frequency(band)

        # Free space path loss
        fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(freq_mhz) + 32.45

        # Band-specific adjustments
        if distance_km > 3000:
            # Long distance - ionospheric enhancement
            if band in ['20m', '17m', '15m']:
                fspl_db -= 10  # F-layer enhancement
            elif band in ['40m', '30m']:
                fspl_db -= 5   # Some enhancement
        elif distance_km < 500:
            # Short distance - ground wave
            if band in ['160m', '80m', '40m']:
                fspl_db -= 3  # Ground wave enhancement

        # Noise floor estimate
        noise_floor_dbm = -174 + 10*np.log10(50) + 10  # 50 Hz BW, 10 dB NF

        # Expected signal strength
        rx_signal_dbm = power_dbm - fspl_db

        # Expected SNR
        expected_snr = rx_signal_dbm - noise_floor_dbm

        return expected_snr

    def _get_band_frequency(self, band: str) -> float:
        """Get center frequency for band in MHz."""
        frequencies = {
            '160m': 1.9, '80m': 3.7, '40m': 7.1, '30m': 10.1,
            '20m': 14.1, '17m': 18.1, '15m': 21.1, '12m': 24.9,
            '10m': 28.5, '6m': 50.1
        }
        return frequencies.get(band, 14.1)

    def _calculate_posteriors(self, priors: Dict[str, float],
                            likelihoods: Dict[str, float]) -> Dict[str, float]:
        """Calculate posterior probabilities using Bayes' theorem."""
        # Calculate unnormalized posteriors
        posteriors = {}
        for level in priors:
            posteriors[level] = priors[level] * likelihoods[level]

        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}
        else:
            # Fall back to priors if likelihoods are all zero
            posteriors = priors.copy()

        return posteriors

    def aggregate_estimates(self, station_hash: str) -> Optional[Dict[str, float]]:
        """Aggregate multiple estimates for a station.

        Args:
            station_hash: Station identifier

        Returns:
            Aggregated power distribution or None
        """
        if station_hash not in self.estimates:
            return None

        estimates = self.estimates[station_hash]
        if not estimates:
            return None

        # Weight by confidence
        weights = np.array([e.confidence_score for e in estimates])
        weights = weights / np.sum(weights)

        # Weighted average of probabilities
        aggregated = {
            'qrp': np.average([e.qrp_probability for e in estimates], weights=weights),
            'low': np.average([e.low_probability for e in estimates], weights=weights),
            'typical': np.average([e.typical_probability for e in estimates], weights=weights),
            'high': np.average([e.high_probability for e in estimates], weights=weights),
            'qro': np.average([e.qro_probability for e in estimates], weights=weights),
            'expected_power_dbm': np.average([e.expected_power_dbm for e in estimates], weights=weights),
            'confidence': np.mean([e.confidence_score for e in estimates])
        }

        return aggregated

    def get_power_profile(self, station_hash: str) -> str:
        """Get human-readable power profile for a station.

        Args:
            station_hash: Station identifier

        Returns:
            Power profile description
        """
        aggregated = self.aggregate_estimates(station_hash)

        if not aggregated:
            return "No power estimates available"

        # Find dominant category
        categories = ['qrp', 'low', 'typical', 'high', 'qro']
        probs = [aggregated[cat] for cat in categories]
        dominant = categories[np.argmax(probs)]

        profile = f"Power Profile: {dominant.upper()} "
        profile += f"({aggregated[dominant]:.0%} probability)\n"
        profile += f"Expected: {aggregated['expected_power_dbm']:.1f} dBm "
        profile += f"({10**((aggregated['expected_power_dbm']-30)/10):.0f}W)\n"
        profile += f"Confidence: {aggregated['confidence']:.0%}"

        return profile