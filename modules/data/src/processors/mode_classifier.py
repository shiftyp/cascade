"""Propagation mode classifier for FT8/WSPR signals.

Implements T032b: Mode classifier (FR-025).
Classifies propagation modes (Es, F2, TEP, EME, MS, Aurora) with
confidence scoring.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PropagationMode(Enum):
    """Propagation mode types."""
    F2 = "F2"  # F2 layer refraction
    ES = "Es"  # Sporadic-E
    TEP = "TEP"  # Trans-equatorial propagation
    EME = "EME"  # Earth-Moon-Earth
    MS = "MS"  # Meteor scatter
    AURORA = "Aurora"  # Auroral propagation
    NVIS = "NVIS"  # Near vertical incidence skywave
    GROUNDWAVE = "Groundwave"
    UNKNOWN = "Unknown"


@dataclass
class ModeClassification:
    """Propagation mode classification result."""
    mode: PropagationMode
    confidence: float  # 0-1
    features: Dict[str, float]
    indicators: Dict[str, Any]
    alternative_modes: Dict[PropagationMode, float]  # Alternative modes with scores


class ModeClassifier:
    """Classifies propagation modes from signal characteristics (FR-025)."""

    def __init__(self):
        """Initialize mode classifier."""
        # Classification thresholds and parameters
        self.es_distance_threshold = 2200  # km
        self.f2_distance_threshold = 3000  # km
        self.tep_latitude_range = (-30, 30)  # Equatorial region
        self.aurora_k_threshold = 4
        self.ms_drift_threshold = 2.0  # Hz

        logger.info("Propagation mode classifier initialized")

    async def classify(
        self,
        distance_km: float,
        frequency_mhz: float,
        snr_db: float,
        tx_grid: str,
        rx_grid: str,
        drift_hz: Optional[float] = None,
        k_index: Optional[int] = None,
        solar_flux: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> ModeClassification:
        """Classify propagation mode (FR-025).

        Args:
            distance_km: Path distance
            frequency_mhz: Operating frequency
            snr_db: Signal-to-noise ratio
            tx_grid: Transmitter grid square
            rx_grid: Receiver grid square
            drift_hz: Frequency drift
            k_index: Geomagnetic K-index
            solar_flux: Solar flux index
            timestamp: Signal timestamp

        Returns:
            Mode classification with confidence
        """
        # Extract features
        features = await self._extract_features(
            distance_km=distance_km,
            frequency_mhz=frequency_mhz,
            snr_db=snr_db,
            tx_grid=tx_grid,
            rx_grid=rx_grid,
            drift_hz=drift_hz,
            k_index=k_index,
            solar_flux=solar_flux,
            timestamp=timestamp,
        )

        # Calculate mode scores
        mode_scores = await self._calculate_mode_scores(features)

        # Select primary mode (highest score)
        primary_mode = max(mode_scores.items(), key=lambda x: x[1])
        mode = primary_mode[0]
        confidence = primary_mode[1]

        # Get alternative modes (sorted by score)
        alternative_modes = {
            m: s for m, s in mode_scores.items() if m != mode and s > 0.1
        }

        # Extract mode-specific indicators
        indicators = await self._extract_mode_indicators(mode, features)

        classification = ModeClassification(
            mode=mode,
            confidence=confidence,
            features=features,
            indicators=indicators,
            alternative_modes=alternative_modes,
        )

        logger.debug(
            f"Classified propagation: {mode.value} "
            f"(confidence: {confidence:.2f}, distance: {distance_km:.0f}km)"
        )

        return classification

    async def _extract_features(
        self,
        distance_km: float,
        frequency_mhz: float,
        snr_db: float,
        tx_grid: str,
        rx_grid: str,
        drift_hz: Optional[float],
        k_index: Optional[int],
        solar_flux: Optional[float],
        timestamp: Optional[datetime],
    ) -> Dict[str, float]:
        """Extract classification features.

        Args:
            Various propagation parameters

        Returns:
            Feature dictionary
        """
        features = {
            "distance_km": distance_km,
            "frequency_mhz": frequency_mhz,
            "snr_db": snr_db,
            "drift_hz": abs(drift_hz) if drift_hz is not None else 0.0,
            "k_index": float(k_index) if k_index is not None else 3.0,
            "solar_flux": solar_flux if solar_flux is not None else 100.0,
        }

        # Calculate path geometry features
        if tx_grid and rx_grid:
            path_features = self._calculate_path_features(tx_grid, rx_grid)
            features.update(path_features)

        # Time-based features
        if timestamp:
            time_features = self._calculate_time_features(timestamp)
            features.update(time_features)

        return features

    def _calculate_path_features(
        self, tx_grid: str, rx_grid: str
    ) -> Dict[str, float]:
        """Calculate path geometry features.

        Args:
            tx_grid: Transmitter grid square
            rx_grid: Receiver grid square

        Returns:
            Path features
        """
        # Extract latitude from grid squares (simplified)
        # Grid format: AANN (e.g., FN31)
        def grid_to_lat(grid: str) -> float:
            if len(grid) < 2:
                return 0.0
            # First letter: field (A=0, R=18, each = 10 degrees)
            field = ord(grid[0].upper()) - ord('A')
            lat = (field - 9) * 10
            # Second digit: square (0-9, each = 1 degree)
            if len(grid) >= 4:
                square = int(grid[3])
                lat += square
            return lat

        tx_lat = grid_to_lat(tx_grid)
        rx_lat = grid_to_lat(rx_grid)

        # Midpoint latitude
        midpoint_lat = (tx_lat + rx_lat) / 2

        # Path crosses equator?
        crosses_equator = (tx_lat * rx_lat) < 0

        return {
            "tx_latitude": tx_lat,
            "rx_latitude": rx_lat,
            "midpoint_latitude": midpoint_lat,
            "crosses_equator": float(crosses_equator),
            "latitude_span": abs(tx_lat - rx_lat),
        }

    def _calculate_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """Calculate time-based features.

        Args:
            timestamp: Signal timestamp

        Returns:
            Time features
        """
        # UTC hour
        hour = timestamp.hour

        # Day/night indicator (simplified)
        is_daytime = 6 <= hour <= 18

        # Season (northern hemisphere)
        month = timestamp.month
        if month in [12, 1, 2]:
            season = 0  # Winter
        elif month in [3, 4, 5]:
            season = 1  # Spring
        elif month in [6, 7, 8]:
            season = 2  # Summer
        else:
            season = 3  # Autumn

        return {
            "utc_hour": float(hour),
            "is_daytime": float(is_daytime),
            "season": float(season),
        }

    async def _calculate_mode_scores(
        self, features: Dict[str, float]
    ) -> Dict[PropagationMode, float]:
        """Calculate scores for each propagation mode.

        Args:
            features: Extracted features

        Returns:
            Mode scores (0-1)
        """
        scores = {}

        # F2 layer propagation
        scores[PropagationMode.F2] = self._score_f2(features)

        # Sporadic-E
        scores[PropagationMode.ES] = self._score_es(features)

        # Trans-equatorial propagation
        scores[PropagationMode.TEP] = self._score_tep(features)

        # Meteor scatter
        scores[PropagationMode.MS] = self._score_ms(features)

        # Aurora
        scores[PropagationMode.AURORA] = self._score_aurora(features)

        # NVIS
        scores[PropagationMode.NVIS] = self._score_nvis(features)

        # Groundwave
        scores[PropagationMode.GROUNDWAVE] = self._score_groundwave(features)

        # EME (very rare)
        scores[PropagationMode.EME] = self._score_eme(features)

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {mode: score / total for mode, score in scores.items()}

        return scores

    def _score_f2(self, features: Dict[str, float]) -> float:
        """Score F2 layer propagation likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        distance = features["distance_km"]
        freq = features["frequency_mhz"]
        sfi = features.get("solar_flux", 100)

        # Distance: 400-4000 km typical for F2
        if 400 <= distance <= 4000:
            score += 0.4
        elif distance > 4000:
            score += 0.2

        # Frequency: HF bands (7-30 MHz)
        if 7 <= freq <= 30:
            score += 0.3

        # Solar flux: higher flux supports higher frequencies
        if sfi > 100:
            score += 0.2
        else:
            score += 0.1

        # Time: F2 works day and night
        score += 0.1

        return min(1.0, score)

    def _score_es(self, features: Dict[str, float]) -> float:
        """Score Sporadic-E likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        distance = features["distance_km"]
        freq = features["frequency_mhz"]
        season = features.get("season", 0)

        # Distance: typically < 2200 km
        if distance < 2200:
            score += 0.5
        else:
            score -= 0.2

        # Frequency: VHF and high HF (28-50 MHz)
        if 28 <= freq <= 50:
            score += 0.4
        elif 20 <= freq <= 28:
            score += 0.2

        # Season: more common in summer
        if season == 2:  # Summer
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_tep(self, features: Dict[str, float]) -> float:
        """Score Trans-Equatorial Propagation likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        crosses_eq = features.get("crosses_equator", 0)
        midpoint_lat = features.get("midpoint_latitude", 90)
        distance = features["distance_km"]

        # Must cross equator
        if crosses_eq:
            score += 0.5

        # Midpoint near equator
        if abs(midpoint_lat) < 15:
            score += 0.3

        # Long distance
        if distance > 8000:
            score += 0.2

        return min(1.0, score)

    def _score_ms(self, features: Dict[str, float]) -> float:
        """Score Meteor Scatter likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        drift = features.get("drift_hz", 0)
        distance = features["distance_km"]
        freq = features["frequency_mhz"]

        # High drift is characteristic of MS
        if drift > 2.0:
            score += 0.6
        elif drift > 1.0:
            score += 0.3

        # Distance: 500-2000 km typical
        if 500 <= distance <= 2000:
            score += 0.2

        # Frequency: VHF preferred
        if freq > 50:
            score += 0.2

        return min(1.0, score)

    def _score_aurora(self, features: Dict[str, float]) -> float:
        """Score Auroral propagation likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        k_index = features.get("k_index", 3)
        tx_lat = features.get("tx_latitude", 0)
        rx_lat = features.get("rx_latitude", 0)

        # High K-index
        if k_index >= 5:
            score += 0.5
        elif k_index >= 4:
            score += 0.3

        # High latitude paths
        if abs(tx_lat) > 50 or abs(rx_lat) > 50:
            score += 0.3

        # Moderate distance
        distance = features["distance_km"]
        if 500 <= distance <= 2000:
            score += 0.2

        return min(1.0, score)

    def _score_nvis(self, features: Dict[str, float]) -> float:
        """Score NVIS likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        distance = features["distance_km"]
        freq = features["frequency_mhz"]

        # Short distance
        if distance < 400:
            score += 0.6

        # Low HF bands (3-7 MHz)
        if 3 <= freq <= 7:
            score += 0.4

        return min(1.0, score)

    def _score_groundwave(self, features: Dict[str, float]) -> float:
        """Score groundwave likelihood.

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        score = 0.0

        distance = features["distance_km"]
        freq = features["frequency_mhz"]

        # Very short distance
        if distance < 100:
            score += 0.6

        # Low frequency (MF/LF)
        if freq < 3:
            score += 0.4

        return min(1.0, score)

    def _score_eme(self, features: Dict[str, float]) -> float:
        """Score EME likelihood (very rare in FT8).

        Args:
            features: Feature dict

        Returns:
            Score (0-1)
        """
        # EME is extremely rare and requires special conditions
        # Would need additional indicators (moon position, etc.)
        return 0.01

    async def _extract_mode_indicators(
        self, mode: PropagationMode, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Extract mode-specific indicators.

        Args:
            mode: Classified mode
            features: Feature dict

        Returns:
            Mode indicators
        """
        indicators = {
            "classification_time": datetime.utcnow().isoformat(),
        }

        if mode == PropagationMode.F2:
            indicators["muf_estimate"] = features["frequency_mhz"] * 1.2
            indicators["hop_count"] = int(features["distance_km"] / 2000)

        elif mode == PropagationMode.ES:
            indicators["es_cloud_distance"] = features["distance_km"] / 2
            indicators["critical_frequency"] = features["frequency_mhz"]

        elif mode == PropagationMode.TEP:
            indicators["equatorial_crossing"] = True
            indicators["path_type"] = "trans_equatorial"

        elif mode == PropagationMode.MS:
            indicators["doppler_shift"] = features.get("drift_hz", 0)
            indicators["burst_mode"] = features.get("drift_hz", 0) > 2

        elif mode == PropagationMode.AURORA:
            indicators["auroral_flutter"] = features.get("k_index", 0) >= 5
            indicators["distortion_expected"] = True

        return indicators