"""Southern hemisphere priority collector (T089).

Prioritizes collection from southern hemisphere stations to address
geographic bias in global coverage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum

from ..models import SessionLocal, KiwiSDRSource
from ..collectors.geographic_quotas import GridSquareClassifier, Hemisphere
from ..processors.reciprocal_inference import ReciprocalPathInference

logger = logging.getLogger(__name__)


class PriorityLevel(Enum):
    """Priority levels for SDR selection."""
    CRITICAL = 4.0   # Antarctica and extreme south
    HIGH = 3.0       # Southern hemisphere
    MEDIUM = 2.0     # Equatorial
    NORMAL = 1.0     # Northern hemisphere


@dataclass
class SouthernSDR:
    """Southern hemisphere SDR with priority weighting."""

    sdr_id: str
    url: str
    grid_square: str
    hemisphere: Hemisphere
    latitude: float
    priority_level: PriorityLevel
    priority_weight: float  # T089b: 3x weight for southern
    availability_score: float
    last_used: Optional[datetime]
    daily_usage_minutes: int
    daily_limit_minutes: int


class SouthernHemispherePriorityCollector:
    """Manages prioritized collection from southern hemisphere (T089)."""

    def __init__(self):
        """Initialize southern hemisphere priority collector."""
        self.db = SessionLocal()
        self.classifier = GridSquareClassifier()
        self.reciprocal_inference = ReciprocalPathInference()

        # T089b: 3x collection weight for southern stations
        self.SOUTHERN_WEIGHT_MULTIPLIER = 3.0
        self.ANTARCTIC_WEIGHT_MULTIPLIER = 5.0  # Even higher for Antarctica

        # Cache for southern SDRs
        self.southern_sdr_cache = []
        self.cache_updated = None

    def maintain_southern_sdr_list(self) -> List[SouthernSDR]:
        """Maintain prioritized list of southern hemisphere SDRs (T089a).

        Returns:
            List of southern SDRs sorted by priority
        """
        # Check cache freshness
        if (self.cache_updated and
            datetime.utcnow() - self.cache_updated < timedelta(hours=1)):
            return self.southern_sdr_cache

        try:
            # Query all active SDRs
            all_sdrs = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.active == True,
                KiwiSDRSource.failure_count < 5
            ).all()

            southern_sdrs = []

            for sdr in all_sdrs:
                if not sdr.grid_square:
                    continue

                # Determine hemisphere
                hemisphere = self.classifier.get_hemisphere(sdr.grid_square)
                latitude = self.classifier.get_latitude_from_grid(sdr.grid_square)

                # Filter for southern, equatorial, or Antarctic
                if hemisphere in [Hemisphere.SOUTH, Hemisphere.EQUATORIAL] or latitude < -60:
                    # Determine priority level
                    if latitude < -66.5:
                        priority_level = PriorityLevel.CRITICAL  # Antarctic
                        weight = self.ANTARCTIC_WEIGHT_MULTIPLIER
                    elif hemisphere == Hemisphere.SOUTH:
                        priority_level = PriorityLevel.HIGH
                        weight = self.SOUTHERN_WEIGHT_MULTIPLIER
                    else:  # Equatorial
                        priority_level = PriorityLevel.MEDIUM
                        weight = 1.5

                    # Calculate availability
                    if sdr.should_reset_usage():
                        sdr.daily_usage_minutes = 0
                        sdr.last_usage_reset = datetime.utcnow()
                        self.db.commit()

                    remaining_minutes = sdr.remaining_daily_minutes
                    availability = remaining_minutes / sdr.daily_limit_minutes if sdr.daily_limit_minutes > 0 else 0

                    # Create Southern SDR entry
                    southern_sdr = SouthernSDR(
                        sdr_id=str(sdr.kiwisdr_id),
                        url=sdr.url,
                        grid_square=sdr.grid_square,
                        hemisphere=hemisphere,
                        latitude=latitude,
                        priority_level=priority_level,
                        priority_weight=weight,
                        availability_score=availability,
                        last_used=sdr.last_connected,
                        daily_usage_minutes=sdr.daily_usage_minutes,
                        daily_limit_minutes=sdr.daily_limit_minutes
                    )

                    southern_sdrs.append(southern_sdr)

            # Sort by priority (highest first)
            southern_sdrs.sort(
                key=lambda s: (s.priority_weight * s.availability_score),
                reverse=True
            )

            # Update cache
            self.southern_sdr_cache = southern_sdrs
            self.cache_updated = datetime.utcnow()

            logger.info(
                f"Updated southern SDR list: {len(southern_sdrs)} SDRs "
                f"({sum(1 for s in southern_sdrs if s.priority_level == PriorityLevel.CRITICAL)} Antarctic, "
                f"{sum(1 for s in southern_sdrs if s.priority_level == PriorityLevel.HIGH)} Southern, "
                f"{sum(1 for s in southern_sdrs if s.priority_level == PriorityLevel.MEDIUM)} Equatorial)"
            )

            return southern_sdrs

        except Exception as e:
            logger.error(f"Error maintaining southern SDR list: {e}")
            return self.southern_sdr_cache  # Return cached list on error

    def apply_collection_weight(
        self,
        sdr: SouthernSDR,
        base_collection_hours: float
    ) -> float:
        """Apply 3x collection weight for southern stations (T089b).

        Args:
            sdr: Southern SDR
            base_collection_hours: Base collection time

        Returns:
            Weighted collection hours
        """
        # Apply weight multiplier
        weighted_hours = base_collection_hours * sdr.priority_weight

        # Cap at daily limit
        max_hours = (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0
        weighted_hours = min(weighted_hours, max_hours)

        logger.debug(
            f"Applied {sdr.priority_weight}x weight to {sdr.grid_square}: "
            f"{base_collection_hours:.1f}h -> {weighted_hours:.1f}h"
        )

        return weighted_hours

    def failover_to_reciprocal(
        self,
        target_bands: List[str],
        required_hours: float
    ) -> Dict[str, Any]:
        """Failover to reciprocal inference when southern SDRs unavailable (T089c).

        Args:
            target_bands: Target frequency bands
            required_hours: Required collection hours

        Returns:
            Reciprocal inference results
        """
        logger.warning(
            f"Southern SDRs unavailable for {required_hours}h on bands {target_bands}. "
            f"Failing over to reciprocal inference."
        )

        # Get available southern SDRs
        southern_sdrs = self.maintain_southern_sdr_list()

        # Calculate total available hours
        total_available = sum(
            (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0
            for sdr in southern_sdrs
        )

        # If insufficient, use reciprocal inference
        if total_available < required_hours:
            deficit = required_hours - total_available

            logger.info(
                f"Deficit of {deficit:.1f}h. Applying reciprocal inference "
                f"for southern hemisphere coverage."
            )

            # Generate inferred data
            inferred_records = []

            try:
                # Get recent northern->southern paths
                northern_records = self._get_northern_to_southern_paths()

                # Generate reciprocal observations
                reciprocal_paths = self.reciprocal_inference.generate_southern_observations(
                    northern_records
                )

                # Weight the inferred data
                weighted_records = self.reciprocal_inference.weight_inferred_data(
                    reciprocal_paths
                )

                inferred_records = weighted_records

                logger.info(
                    f"Generated {len(inferred_records)} inferred records "
                    f"to cover {deficit:.1f}h deficit"
                )

            except Exception as e:
                logger.error(f"Reciprocal inference failed: {e}")

            return {
                "method": "reciprocal_inference",
                "southern_sdrs_available": len(southern_sdrs),
                "available_hours": total_available,
                "required_hours": required_hours,
                "deficit_hours": deficit,
                "inferred_records": len(inferred_records),
                "success": len(inferred_records) > 0
            }

        else:
            return {
                "method": "direct_collection",
                "southern_sdrs_available": len(southern_sdrs),
                "available_hours": total_available,
                "required_hours": required_hours,
                "deficit_hours": 0,
                "inferred_records": 0,
                "success": True
            }

    def _get_northern_to_southern_paths(self) -> List[Any]:
        """Get recent northern->southern propagation paths.

        Returns:
            List of propagation records
        """
        from ..models import PropagationRecord

        cutoff = datetime.utcnow() - timedelta(hours=24)

        records = self.db.query(PropagationRecord).filter(
            PropagationRecord.timestamp >= cutoff
        ).limit(1000).all()

        # Filter for north->south paths
        filtered = []
        for record in records:
            if not record.tx_grid_square or not record.rx_grid_square:
                continue

            tx_hem = self.classifier.get_hemisphere(record.tx_grid_square)
            rx_hem = self.classifier.get_hemisphere(record.rx_grid_square)

            if tx_hem == Hemisphere.NORTH and rx_hem == Hemisphere.SOUTH:
                filtered.append(record)

        return filtered

    def get_southern_collection_status(self) -> Dict[str, Any]:
        """Get status of southern hemisphere collection.

        Returns:
            Status dictionary
        """
        southern_sdrs = self.maintain_southern_sdr_list()

        # Group by priority level
        by_priority = {
            "critical": [],
            "high": [],
            "medium": []
        }

        for sdr in southern_sdrs:
            if sdr.priority_level == PriorityLevel.CRITICAL:
                by_priority["critical"].append(sdr)
            elif sdr.priority_level == PriorityLevel.HIGH:
                by_priority["high"].append(sdr)
            else:
                by_priority["medium"].append(sdr)

        # Calculate statistics
        total_available_hours = sum(
            (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0
            for sdr in southern_sdrs
        )

        weighted_hours = sum(
            (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0 * sdr.priority_weight
            for sdr in southern_sdrs
        )

        return {
            "total_southern_sdrs": len(southern_sdrs),
            "by_priority": {
                "critical_antarctic": len(by_priority["critical"]),
                "high_southern": len(by_priority["high"]),
                "medium_equatorial": len(by_priority["medium"])
            },
            "available_hours": {
                "raw": total_available_hours,
                "weighted": weighted_hours
            },
            "top_5_available": [
                {
                    "grid": sdr.grid_square,
                    "latitude": sdr.latitude,
                    "available_hours": (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0,
                    "priority": sdr.priority_level.name
                }
                for sdr in southern_sdrs[:5]
            ],
            "reciprocal_inference_ready": True,  # Always ready as fallback
            "weight_multipliers": {
                "southern": self.SOUTHERN_WEIGHT_MULTIPLIER,
                "antarctic": self.ANTARCTIC_WEIGHT_MULTIPLIER
            }
        }

    def select_optimal_southern_sdr(
        self,
        frequency_khz: float,
        duration_minutes: int,
        prefer_antarctic: bool = False
    ) -> Optional[SouthernSDR]:
        """Select optimal southern hemisphere SDR.

        Args:
            frequency_khz: Target frequency
            duration_minutes: Required duration
            prefer_antarctic: Prefer Antarctic stations if available

        Returns:
            Best available southern SDR or None
        """
        southern_sdrs = self.maintain_southern_sdr_list()

        # Filter by availability
        available_sdrs = [
            sdr for sdr in southern_sdrs
            if (sdr.daily_limit_minutes - sdr.daily_usage_minutes) >= duration_minutes
        ]

        if not available_sdrs:
            logger.warning("No southern SDRs available for required duration")
            return None

        # Prefer Antarctic if requested
        if prefer_antarctic:
            antarctic_sdrs = [
                sdr for sdr in available_sdrs
                if sdr.priority_level == PriorityLevel.CRITICAL
            ]
            if antarctic_sdrs:
                return antarctic_sdrs[0]

        # Return highest priority available
        return available_sdrs[0]

    def close(self):
        """Close database connections."""
        if self.db:
            self.db.close()
        if self.reciprocal_inference:
            self.reciprocal_inference.close()


# Utility function for scheduling
def schedule_southern_priority_collection(
    target_hours_per_day: float,
    prefer_antarctic: bool = True
) -> Dict[str, Any]:
    """Schedule prioritized southern hemisphere collection.

    Args:
        target_hours_per_day: Target collection hours per day
        prefer_antarctic: Prefer Antarctic stations when available

    Returns:
        Collection schedule
    """
    collector = SouthernHemispherePriorityCollector()

    try:
        # Get southern SDR status
        status = collector.get_southern_collection_status()

        # Check if we have enough weighted hours
        if status["available_hours"]["weighted"] >= target_hours_per_day:
            # Direct collection possible
            schedule_method = "direct"

            # Select SDRs for schedule
            selected_sdrs = []
            remaining_hours = target_hours_per_day

            for sdr in collector.maintain_southern_sdr_list():
                if remaining_hours <= 0:
                    break

                available = (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0
                if available > 0:
                    scheduled_hours = min(available * sdr.priority_weight, remaining_hours)
                    selected_sdrs.append({
                        "grid": sdr.grid_square,
                        "hours": scheduled_hours / sdr.priority_weight,  # Actual hours
                        "weighted_hours": scheduled_hours,
                        "priority": sdr.priority_level.name
                    })
                    remaining_hours -= scheduled_hours

        else:
            # Need reciprocal inference
            schedule_method = "hybrid_with_inference"
            selected_sdrs = []

            # Use all available southern SDRs
            for sdr in collector.maintain_southern_sdr_list():
                available = (sdr.daily_limit_minutes - sdr.daily_usage_minutes) / 60.0
                if available > 0:
                    selected_sdrs.append({
                        "grid": sdr.grid_square,
                        "hours": available,
                        "weighted_hours": available * sdr.priority_weight,
                        "priority": sdr.priority_level.name
                    })

        return {
            "schedule_method": schedule_method,
            "target_hours": target_hours_per_day,
            "scheduled_sdrs": selected_sdrs,
            "total_scheduled_hours": sum(s["weighted_hours"] for s in selected_sdrs),
            "requires_inference": schedule_method == "hybrid_with_inference",
            "status": status
        }

    finally:
        collector.close()