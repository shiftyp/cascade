"""Propagation-aware SDR rotation algorithm.

Implements T028: SDR rotation algorithm (FR-008, FR-014, FR-022).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

from sqlalchemy.orm import Session

from ..models import SessionLocal, KiwiSDRSource
from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class SDRScore:
    """SDR scoring for selection."""

    sdr: KiwiSDRSource
    score: float
    factors: Dict[str, float]


class SDRManager:
    """Manages SDR rotation and selection based on propagation conditions."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize SDR manager."""
        self.db = db_session or SessionLocal()
        self.active_sdrs: Dict[str, KiwiSDRSource] = {}
        self.failed_sdrs: List[KiwiSDRSource] = []
        self.usage_cache: Dict[str, float] = {}

    async def get_next_available(
        self,
        band: Optional[str] = None,
        prefer_location: Optional[str] = None,
    ) -> Optional[KiwiSDRSource]:
        """Get next available SDR based on usage limits (FR-008).

        Args:
            band: Preferred band
            prefer_location: Preferred geographic region

        Returns:
            Available SDR or None
        """
        # Query available SDRs
        query = self.db.query(KiwiSDRSource).filter(
            KiwiSDRSource.active == True,
            KiwiSDRSource.failure_count < 5,
        )

        sdrs = query.all()

        # Check and reset daily usage if needed
        for sdr in sdrs:
            if sdr.should_reset_usage():
                logger.info(f"Resetting daily usage for {sdr.url}")
                sdr.daily_usage_minutes = 0
                sdr.last_usage_reset = datetime.utcnow()
                self.db.commit()

        # Filter by availability (FR-014)
        available_sdrs = [
            sdr for sdr in sdrs
            if sdr.remaining_daily_minutes > 5  # At least 5 minutes remaining
        ]

        if not available_sdrs:
            logger.warning("No SDRs available within usage limits")
            return None

        # Score SDRs based on criteria
        if prefer_location:
            # Sort by location preference
            available_sdrs.sort(
                key=lambda s: (
                    0 if s.grid_square and s.grid_square[:2] == prefer_location[:2] else 1,
                    s.daily_usage_minutes,  # Then by least usage
                )
            )
        else:
            # Sort by least usage
            available_sdrs.sort(key=lambda s: s.daily_usage_minutes)

        selected = available_sdrs[0]
        self.active_sdrs[selected.url] = selected
        logger.info(f"Selected SDR: {selected.url} ({selected.remaining_daily_minutes:.1f} min remaining)")

        return selected

    async def get_best_for_propagation(
        self,
        target_time: datetime,
        propagation_type: str,
        frequency_mhz: float,
    ) -> Optional[KiwiSDRSource]:
        """Get best SDR for specific propagation conditions (FR-022).

        Args:
            target_time: Target observation time
            propagation_type: Type of propagation (gray_line, F2, Es, etc.)
            frequency_mhz: Operating frequency

        Returns:
            Best SDR for conditions
        """
        available = await self.get_all_available()
        if not available:
            return None

        scored_sdrs = []

        for sdr in available:
            score = self._calculate_propagation_score(
                sdr, target_time, propagation_type, frequency_mhz
            )
            scored_sdrs.append(SDRScore(sdr, score.score, score.factors))

        # Sort by score (highest first)
        scored_sdrs.sort(key=lambda x: x.score, reverse=True)

        if scored_sdrs:
            best = scored_sdrs[0]
            logger.info(
                f"Best SDR for {propagation_type}: {best.sdr.url} "
                f"(score: {best.score:.2f})"
            )
            return best.sdr

        return None

    def _calculate_propagation_score(
        self,
        sdr: KiwiSDRSource,
        target_time: datetime,
        propagation_type: str,
        frequency_mhz: float,
    ) -> SDRScore:
        """Calculate propagation suitability score for an SDR.

        Args:
            sdr: SDR to score
            target_time: Target time
            propagation_type: Propagation mode
            frequency_mhz: Frequency

        Returns:
            SDRScore with breakdown
        """
        factors = {}

        # Base availability score
        factors['availability'] = min(1.0, sdr.remaining_daily_minutes / 30)

        # Location score based on propagation type
        if propagation_type == 'gray_line' and sdr.latitude and sdr.longitude:
            # Calculate proximity to terminator
            solar_angle = self._calculate_solar_angle(
                sdr.latitude, sdr.longitude, target_time
            )
            # Best score near sunrise/sunset (solar angle near 90°)
            factors['gray_line'] = 1.0 - abs(90 - solar_angle) / 90

        elif propagation_type == 'F2':
            # F2 propagation favors mid-latitudes
            if sdr.latitude:
                factors['latitude'] = 1.0 - abs(sdr.latitude - 40) / 50

        elif propagation_type == 'Es':
            # Sporadic-E favors mid-latitudes in summer
            if sdr.latitude:
                factors['latitude'] = 1.0 - abs(sdr.latitude - 45) / 45

        elif propagation_type == 'Aurora':
            # Aurora favors high latitudes
            if sdr.latitude:
                factors['latitude'] = min(1.0, abs(sdr.latitude) / 60)

        # Reliability score
        factors['reliability'] = sdr.reliability_score or 0.5

        # Calculate weighted total
        weights = {
            'availability': 0.3,
            'gray_line': 0.4,
            'latitude': 0.2,
            'reliability': 0.1,
        }

        total_score = sum(
            factors.get(k, 0) * weights.get(k, 0)
            for k in weights
        )

        return SDRScore(sdr, total_score, factors)

    def _calculate_solar_angle(
        self,
        latitude: float,
        longitude: float,
        time: datetime,
    ) -> float:
        """Calculate solar elevation angle.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            time: UTC time

        Returns:
            Solar elevation angle in degrees
        """
        # Simplified calculation - would use pyephem in practice
        hour_angle = (time.hour + time.minute / 60 - 12) * 15 + longitude

        # Approximate declination
        day_of_year = time.timetuple().tm_yday
        declination = 23.45 * math.sin(math.radians((360 * (284 + day_of_year)) / 365))

        # Solar elevation
        elevation = math.degrees(math.asin(
            math.sin(math.radians(declination)) * math.sin(math.radians(latitude)) +
            math.cos(math.radians(declination)) * math.cos(math.radians(latitude)) *
            math.cos(math.radians(hour_angle))
        ))

        return max(0, min(90, elevation + 90))

    async def get_concurrent_sdrs(
        self,
        count: int,
        band: Optional[str] = None,
    ) -> List[KiwiSDRSource]:
        """Get multiple SDRs for concurrent operation.

        Args:
            count: Number of SDRs requested
            band: Preferred band

        Returns:
            List of available SDRs
        """
        sdrs = []
        used_urls = set()

        for _ in range(count):
            sdr = await self.get_next_available(band=band)
            if sdr and sdr.url not in used_urls:
                sdrs.append(sdr)
                used_urls.add(sdr.url)
            else:
                break

        logger.info(f"Allocated {len(sdrs)} concurrent SDRs (requested {count})")
        return sdrs

    async def update_usage(
        self,
        sdr: KiwiSDRSource,
        minutes: float,
    ):
        """Update SDR usage tracking (FR-008).

        Args:
            sdr: SDR to update
            minutes: Usage in minutes
        """
        sdr.daily_usage_minutes += minutes
        sdr.total_usage_minutes += minutes
        sdr.last_connected = datetime.utcnow()

        self.db.commit()

        logger.info(
            f"Updated {sdr.url} usage: {sdr.daily_usage_minutes:.1f}/{config.KIWI_DAILY_LIMIT_MINUTES} min today"
        )

        # Remove from active if limit reached
        if sdr.remaining_daily_minutes <= 0:
            logger.warning(f"SDR {sdr.url} reached daily limit")
            if sdr.url in self.active_sdrs:
                del self.active_sdrs[sdr.url]

    async def handle_sdr_failure(
        self,
        sdr: KiwiSDRSource,
        error: Exception,
    ):
        """Handle SDR connection failure.

        Args:
            sdr: Failed SDR
            error: Exception that occurred
        """
        sdr.failure_count += 1
        sdr.reliability_score = max(0, (sdr.reliability_score or 1.0) - 0.1)

        self.db.commit()

        logger.error(f"SDR {sdr.url} failed: {error} (failure #{sdr.failure_count})")

        # Mark as inactive if too many failures
        if sdr.failure_count >= 5:
            sdr.active = False
            self.db.commit()
            logger.warning(f"SDR {sdr.url} marked inactive after {sdr.failure_count} failures")

        self.failed_sdrs.append(sdr)
        if sdr.url in self.active_sdrs:
            del self.active_sdrs[sdr.url]

    async def get_all_available(self) -> List[KiwiSDRSource]:
        """Get all currently available SDRs.

        Returns:
            List of available SDRs
        """
        sdrs = self.db.query(KiwiSDRSource).filter(
            KiwiSDRSource.active == True,
            KiwiSDRSource.failure_count < 5,
        ).all()

        available = []
        for sdr in sdrs:
            if sdr.should_reset_usage():
                sdr.daily_usage_minutes = 0
                sdr.last_usage_reset = datetime.utcnow()
                self.db.commit()

            if sdr.remaining_daily_minutes > 0:
                available.append(sdr)

        return available

    async def ensure_minimum_collection(
        self,
        min_sdrs: int = 1,
    ) -> Dict[str, Any]:
        """Ensure minimum SDRs available for collection (FR-035).

        Args:
            min_sdrs: Minimum SDRs required

        Returns:
            Status dict
        """
        available = await self.get_all_available()

        result = {
            "available_count": len(available),
            "minimum_met": len(available) >= min_sdrs,
            "active_sdrs": available[:min_sdrs] if available else [],
        }

        if not result["minimum_met"]:
            logger.critical(
                f"Minimum SDR requirement not met: {len(available)}/{min_sdrs}"
            )

        return result

    async def check_and_reset_usage(self):
        """Check all SDRs and reset daily usage if needed."""
        sdrs = self.db.query(KiwiSDRSource).all()

        reset_count = 0
        for sdr in sdrs:
            if sdr.should_reset_usage():
                sdr.daily_usage_minutes = 0
                sdr.last_usage_reset = datetime.utcnow()
                reset_count += 1

        if reset_count > 0:
            self.db.commit()
            logger.info(f"Reset daily usage for {reset_count} SDRs")

    def close(self):
        """Close database session."""
        self.db.close()