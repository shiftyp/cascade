"""Event-based SDR scaler for space weather events."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of space weather events."""
    SOLAR_FLARE = "solar_flare"
    GEOMAGNETIC_STORM = "geomagnetic_storm"
    SOLAR_MINIMUM = "solar_minimum"
    IONOSPHERIC_DISTURBANCE = "ionospheric_disturbance"
    AURORAL_ACTIVITY = "auroral_activity"


@dataclass
class SpaceWeatherEvent:
    """Space weather event data."""
    event_type: EventType
    severity: float  # 0-1 scale
    start_time: datetime
    duration_hours: float
    affected_bands: List[str]
    metadata: Dict[str, Any]


class EventScaler:
    """Scales SDR collection based on space weather events."""

    def __init__(self, base_stations: int = 6, max_stations: int = 20):
        """Initialize event scaler.

        Args:
            base_stations: Base number of stations
            max_stations: Maximum number of stations during events
        """
        self.base_stations = base_stations
        self.max_stations = max_stations
        self.current_stations = base_stations
        self.active_events: List[SpaceWeatherEvent] = []
        self.scaling_history: List[Dict] = []
        self._lock = asyncio.Lock()

    async def process_event(self, event: SpaceWeatherEvent) -> int:
        """Process a space weather event and determine scaling.

        Args:
            event: Space weather event

        Returns:
            Number of stations to deploy
        """
        async with self._lock:
            # Add to active events
            self.active_events.append(event)

            # Calculate required stations
            required_stations = self._calculate_required_stations(event)

            # Update current stations
            self.current_stations = min(required_stations, self.max_stations)

            # Record in history
            self.scaling_history.append({
                "timestamp": datetime.now(timezone.utc),
                "event": event.event_type.value,
                "severity": event.severity,
                "stations": self.current_stations
            })

            logger.info(f"Scaled to {self.current_stations} stations for "
                       f"{event.event_type.value} (severity: {event.severity})")

            return self.current_stations

    def _calculate_required_stations(self, event: SpaceWeatherEvent) -> int:
        """Calculate required stations based on event.

        Args:
            event: Space weather event

        Returns:
            Number of required stations
        """
        # Base calculation on event type and severity
        multiplier = 1.0

        if event.event_type == EventType.SOLAR_FLARE:
            multiplier = 1.5 + event.severity * 2.0
        elif event.event_type == EventType.GEOMAGNETIC_STORM:
            multiplier = 2.0 + event.severity * 1.5
        elif event.event_type == EventType.SOLAR_MINIMUM:
            # Aggressive collection during rare solar minimum
            multiplier = 3.0 + event.severity
        elif event.event_type == EventType.IONOSPHERIC_DISTURBANCE:
            multiplier = 1.3 + event.severity * 1.2
        elif event.event_type == EventType.AURORAL_ACTIVITY:
            multiplier = 1.8 + event.severity * 1.3

        # Calculate stations
        required = int(self.base_stations * multiplier)

        # Add extra stations for affected bands
        extra_per_band = max(1, int(event.severity * 2))
        required += len(event.affected_bands) * extra_per_band

        return required

    async def check_event_expiry(self) -> int:
        """Check for expired events and adjust scaling.

        Returns:
            Current number of stations
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Remove expired events
            self.active_events = [
                event for event in self.active_events
                if (event.start_time.timestamp() + event.duration_hours * 3600) > now.timestamp()
            ]

            # Recalculate stations
            if not self.active_events:
                self.current_stations = self.base_stations
            else:
                # Use highest requirement from active events
                max_required = self.base_stations
                for event in self.active_events:
                    required = self._calculate_required_stations(event)
                    max_required = max(max_required, required)
                self.current_stations = min(max_required, self.max_stations)

            return self.current_stations

    def get_scaling_factor(self) -> float:
        """Get current scaling factor.

        Returns:
            Scaling factor (1.0 = base level)
        """
        return self.current_stations / self.base_stations

    def get_status(self) -> Dict[str, Any]:
        """Get scaler status.

        Returns:
            Status dictionary
        """
        return {
            "current_stations": self.current_stations,
            "base_stations": self.base_stations,
            "max_stations": self.max_stations,
            "scaling_factor": self.get_scaling_factor(),
            "active_events": len(self.active_events),
            "events": [
                {
                    "type": event.event_type.value,
                    "severity": event.severity,
                    "affected_bands": event.affected_bands
                }
                for event in self.active_events
            ]
        }

    async def simulate_solar_minimum(self) -> int:
        """Simulate aggressive solar minimum collection.

        Returns:
            Number of stations deployed
        """
        # Create solar minimum event
        event = SpaceWeatherEvent(
            event_type=EventType.SOLAR_MINIMUM,
            severity=0.9,  # High severity for rare event
            start_time=datetime.now(timezone.utc),
            duration_hours=72,  # 3 days of aggressive collection
            affected_bands=["10m", "15m", "20m", "40m", "80m", "160m"],
            metadata={"reason": "Rare solar minimum conditions detected"}
        )

        return await self.process_event(event)

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get collection recommendations based on events.

        Returns:
            List of recommendations
        """
        recommendations = []

        for event in self.active_events:
            if event.event_type == EventType.SOLAR_MINIMUM:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Maximize QRN collection on all bands",
                    "reason": "Solar minimum - rare atmospheric conditions",
                    "bands": event.affected_bands,
                    "duration_hours": event.duration_hours
                })
            elif event.severity > 0.7:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": f"Increase collection on affected bands",
                    "reason": f"{event.event_type.value} with high severity",
                    "bands": event.affected_bands,
                    "duration_hours": event.duration_hours
                })

        return recommendations