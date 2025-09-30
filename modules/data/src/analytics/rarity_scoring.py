"""Rarity scoring algorithm with 18-month collection window multipliers.

Implements FR-055, FR-058: Cycle-aware rarity scoring with 5x-10x multipliers.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.models import SessionLocal, SpaceWeatherData, SolarCyclePhase, Season

logger = logging.getLogger(__name__)


@dataclass
class RarityScore:
    """Rarity score breakdown."""

    total_score: float
    base_rarity: float
    cycle_multiplier: float
    urgency_multiplier: float
    seasonal_factor: float
    event_type: str
    justification: str


class RarityScorer:
    """Calculates event rarity scores with 18-month window awareness."""

    def __init__(self):
        """Initialize rarity scorer."""
        self.db = SessionLocal()

        # Base rarity values for different event types
        self.base_rarity = {
            # Solar events
            "X_flare": 50.0,     # X-class flare
            "M_flare": 25.0,     # M-class flare
            "C_flare": 10.0,     # C-class flare
            "B_flare": 2.0,      # B-class flare
            "A_flare": 1.0,      # A-class flare

            # Geomagnetic events
            "K9_storm": 100.0,   # Extreme storm (K=9)
            "K8_storm": 75.0,    # Severe storm (K=8)
            "K7_storm": 50.0,    # Strong storm (K=7)
            "K6_storm": 25.0,    # Moderate storm (K=6)
            "K5_storm": 15.0,    # Minor storm (K=5)
            "K4_storm": 8.0,     # Unsettled (K=4)
            "K3_storm": 5.0,     # Quiet to unsettled (K=3)

            # Solar flux events
            "high_flux": 10.0,   # SFI > 200
            "low_flux": 5.0,     # SFI < 80 (solar minimum)

            # Combined events
            "super_storm": 200.0, # X+K9 combination
            "major_event": 150.0, # M+K7+ combination
        }

    def calculate_rarity_score(
        self,
        space_weather: SpaceWeatherData,
        event_context: Optional[Dict[str, Any]] = None
    ) -> RarityScore:
        """Calculate comprehensive rarity score for space weather event.

        Args:
            space_weather: Space weather conditions
            event_context: Additional context (recording session, etc.)

        Returns:
            RarityScore with breakdown
        """
        # Determine event type and base rarity
        event_type, base_rarity = self._classify_event(space_weather)

        # Calculate cycle-aware multiplier (FR-058)
        cycle_multiplier = self._calculate_cycle_multiplier(space_weather)

        # Calculate 18-month urgency multiplier (FR-055)
        urgency_multiplier = self._calculate_urgency_multiplier(space_weather)

        # Calculate seasonal factor
        seasonal_factor = self._calculate_seasonal_factor(space_weather)

        # Calculate total score
        total_score = (
            base_rarity *
            cycle_multiplier *
            urgency_multiplier *
            seasonal_factor
        )

        # Generate justification
        justification = self._generate_justification(
            event_type, base_rarity, cycle_multiplier,
            urgency_multiplier, seasonal_factor, space_weather
        )

        return RarityScore(
            total_score=min(total_score, 1000.0),  # Cap at 1000x
            base_rarity=base_rarity,
            cycle_multiplier=cycle_multiplier,
            urgency_multiplier=urgency_multiplier,
            seasonal_factor=seasonal_factor,
            event_type=event_type,
            justification=justification
        )

    def _classify_event(self, space_weather: SpaceWeatherData) -> tuple[str, float]:
        """Classify event type and get base rarity.

        Args:
            space_weather: Space weather data

        Returns:
            Tuple of (event_type, base_rarity_score)
        """
        # Check for combined super events first
        if (space_weather.xray_class and space_weather.xray_class.startswith('X') and
            space_weather.k_index and space_weather.k_index >= 9):
            return "super_storm", self.base_rarity["super_storm"]

        if (space_weather.xray_class and space_weather.xray_class.startswith('M') and
            space_weather.k_index and space_weather.k_index >= 7):
            return "major_event", self.base_rarity["major_event"]

        # Solar flare classification
        if space_weather.xray_class:
            flare_class = space_weather.xray_class[0]
            if flare_class == 'X':
                return "X_flare", self.base_rarity["X_flare"]
            elif flare_class == 'M':
                return "M_flare", self.base_rarity["M_flare"]
            elif flare_class == 'C':
                return "C_flare", self.base_rarity["C_flare"]
            elif flare_class == 'B':
                return "B_flare", self.base_rarity["B_flare"]
            elif flare_class == 'A':
                return "A_flare", self.base_rarity["A_flare"]

        # Geomagnetic storm classification
        if space_weather.k_index:
            if space_weather.k_index >= 9:
                return "K9_storm", self.base_rarity["K9_storm"]
            elif space_weather.k_index >= 8:
                return "K8_storm", self.base_rarity["K8_storm"]
            elif space_weather.k_index >= 7:
                return "K7_storm", self.base_rarity["K7_storm"]
            elif space_weather.k_index >= 6:
                return "K6_storm", self.base_rarity["K6_storm"]
            elif space_weather.k_index >= 5:
                return "K5_storm", self.base_rarity["K5_storm"]
            elif space_weather.k_index >= 4:
                return "K4_storm", self.base_rarity["K4_storm"]
            elif space_weather.k_index >= 3:
                return "K3_storm", self.base_rarity["K3_storm"]

        # Solar flux events
        if space_weather.solar_flux:
            if space_weather.solar_flux > 200:
                return "high_flux", self.base_rarity["high_flux"]
            elif space_weather.solar_flux < 80:
                return "low_flux", self.base_rarity["low_flux"]

        # Default for quiet conditions
        return "quiet", 1.0

    def _calculate_cycle_multiplier(self, space_weather: SpaceWeatherData) -> float:
        """Calculate cycle-aware multiplier (FR-058).

        Args:
            space_weather: Space weather data

        Returns:
            Cycle multiplier
        """
        if not space_weather.solar_cycle_phase:
            return 1.0

        # Different multipliers based on solar cycle phase
        if space_weather.solar_cycle_phase == SolarCyclePhase.MINIMUM:
            # Events during solar minimum are much rarer
            if space_weather.k_index and space_weather.k_index >= 5:
                return 8.0  # Very rare during minimum
            elif space_weather.k_index and space_weather.k_index >= 3:
                return 6.0  # Rare during minimum
            elif space_weather.xray_class and space_weather.xray_class[0] in ['M', 'X']:
                return 10.0  # Extremely rare during minimum
            elif space_weather.xray_class and space_weather.xray_class.startswith('C'):
                return 4.0  # Moderately rare during minimum
            else:
                return 2.0  # Even quiet periods have some rarity value

        elif space_weather.solar_cycle_phase == SolarCyclePhase.RISING:
            # Moderate rarity boost during rising phase
            return 2.0

        elif space_weather.solar_cycle_phase == SolarCyclePhase.MAXIMUM:
            # Events common during maximum, lower multiplier
            return 0.8

        else:  # DECLINING
            # Normal rarity during declining phase
            return 1.0

    def _calculate_urgency_multiplier(self, space_weather: SpaceWeatherData) -> float:
        """Calculate 18-month window urgency multiplier (FR-055).

        Args:
            space_weather: Space weather data

        Returns:
            Urgency multiplier (up to 10x)
        """
        if not space_weather.opportunity_limited_mode:
            return 1.0

        # Get base multiplier from space weather data
        base_multiplier = space_weather.get_rarity_multiplier()

        # Apply collection window time pressure
        window_factor = space_weather.collection_window_factor or 1.0

        # Combine multipliers (cap at 10x as per FR-055)
        total_multiplier = min(base_multiplier * window_factor, 10.0)

        return total_multiplier

    def _calculate_seasonal_factor(self, space_weather: SpaceWeatherData) -> float:
        """Calculate seasonal rarity factor.

        Args:
            space_weather: Space weather data

        Returns:
            Seasonal factor
        """
        if not space_weather.season:
            return 1.0

        # Base seasonal factors
        seasonal_factors = {
            Season.WINTER: 1.2,  # Winter events slightly rarer
            Season.SPRING: 1.0,  # Normal
            Season.SUMMER: 0.9,  # Summer events slightly more common
            Season.AUTUMN: 1.0,  # Normal
        }

        base_factor = seasonal_factors.get(space_weather.season, 1.0)

        # Equinoctial enhancement reduces rarity (events more common)
        if space_weather.equinoctial_enhancement:
            base_factor *= 0.8

        # Apply seasonal balance factor from space weather
        if space_weather.seasonal_balance_factor:
            base_factor *= space_weather.seasonal_balance_factor

        return base_factor

    def _generate_justification(
        self,
        event_type: str,
        base_rarity: float,
        cycle_multiplier: float,
        urgency_multiplier: float,
        seasonal_factor: float,
        space_weather: SpaceWeatherData
    ) -> str:
        """Generate human-readable justification for rarity score.

        Args:
            event_type: Type of event
            base_rarity: Base rarity score
            cycle_multiplier: Solar cycle multiplier
            urgency_multiplier: 18-month urgency multiplier
            seasonal_factor: Seasonal factor
            space_weather: Space weather data

        Returns:
            Justification string
        """
        parts = [f"Event: {event_type} (base rarity: {base_rarity:.1f})"]

        # Solar cycle justification
        if cycle_multiplier > 2.0:
            parts.append(
                f"Solar cycle: {space_weather.solar_cycle_phase.value} "
                f"({cycle_multiplier:.1f}x - very rare for this phase)"
            )
        elif cycle_multiplier > 1.5:
            parts.append(
                f"Solar cycle: {space_weather.solar_cycle_phase.value} "
                f"({cycle_multiplier:.1f}x - rare for this phase)"
            )

        # Urgency justification
        if urgency_multiplier >= 5.0:
            parts.append(
                f"18-month window: CRITICAL urgency ({urgency_multiplier:.1f}x - "
                f"maximum collection priority during limited opportunity window)"
            )
        elif urgency_multiplier >= 3.0:
            parts.append(
                f"18-month window: HIGH urgency ({urgency_multiplier:.1f}x - "
                f"elevated priority for rare event capture)"
            )
        elif urgency_multiplier > 1.5:
            parts.append(
                f"18-month window: Moderate urgency ({urgency_multiplier:.1f}x)"
            )

        # Seasonal justification
        if seasonal_factor > 1.1:
            parts.append(
                f"Season: {space_weather.season.value} "
                f"({seasonal_factor:.1f}x - enhanced rarity)"
            )

        # Special conditions
        if space_weather.is_100_percent_capture_required():
            parts.append("100% CAPTURE REQUIRED - solar minimum activity event")

        return "; ".join(parts)

    async def get_collection_priority(
        self,
        space_weather: SpaceWeatherData
    ) -> Dict[str, Any]:
        """Get collection priority recommendation.

        Args:
            space_weather: Space weather data

        Returns:
            Priority recommendation
        """
        rarity_score = self.calculate_rarity_score(space_weather)

        # Determine priority level
        if rarity_score.total_score >= 500:
            priority = "CRITICAL"
            recommended_sdrs = 50
        elif rarity_score.total_score >= 200:
            priority = "HIGH"
            recommended_sdrs = 40
        elif rarity_score.total_score >= 100:
            priority = "ELEVATED"
            recommended_sdrs = 30
        elif rarity_score.total_score >= 50:
            priority = "MODERATE"
            recommended_sdrs = 20
        elif rarity_score.total_score >= 20:
            priority = "LOW"
            recommended_sdrs = 12
        else:
            priority = "BASELINE"
            recommended_sdrs = 6

        return {
            "priority": priority,
            "recommended_sdrs": recommended_sdrs,
            "rarity_score": rarity_score.total_score,
            "justification": rarity_score.justification,
            "capture_required": space_weather.is_100_percent_capture_required(),
            "urgency_multiplier": rarity_score.urgency_multiplier,
            "event_type": rarity_score.event_type,
        }

    def close(self):
        """Close scorer resources."""
        if self.db:
            self.db.close()


async def score_current_conditions() -> Dict[str, Any]:
    """Score current space weather conditions (utility function).

    Returns:
        Current rarity score and priority
    """
    scorer = RarityScorer()

    try:
        # Get latest space weather
        latest = (
            scorer.db.query(SpaceWeatherData)
            .order_by(SpaceWeatherData.observation_time.desc())
            .first()
        )

        if not latest:
            return {"error": "No space weather data available"}

        # Calculate priority
        priority = await scorer.get_collection_priority(latest)

        return {
            "timestamp": latest.observation_time.isoformat(),
            "space_weather": {
                "k_index": latest.k_index,
                "xray_class": latest.xray_class,
                "solar_flux": latest.solar_flux,
                "solar_cycle_phase": latest.solar_cycle_phase.value if latest.solar_cycle_phase else None,
            },
            "priority": priority,
        }

    finally:
        scorer.close()