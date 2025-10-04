"""Minimum viable collection scheduler for 1-SDR operation.

Implements T028d: Minimum scheduler (FR-035).
Maintains data collection even with only 1 SDR through intelligent
round-robin band switching.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models import SessionLocal, KiwiSDRSource, CollectionSchedule
from ..config import config
from ..config.frequencies import BANDS, BAND_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class MinimumSchedule:
    """Minimum collection schedule configuration."""
    active_sdrs: int
    collection_strategy: str
    bands: List[str]
    rotation_interval_minutes: int
    priority_bands: List[str]
    sessions_per_hour: int


class MinimumScheduler:
    """Scheduler for minimum viable collection with 1 SDR (FR-035)."""

    def __init__(self, sdr_manager, db_session: Optional[Session] = None):
        """Initialize minimum scheduler.

        Args:
            sdr_manager: SDR manager instance
            db_session: Database session (optional)
        """
        self.sdr_manager = sdr_manager
        self.db = db_session or SessionLocal()

        # Minimum operation parameters
        self.rotation_interval_minutes = 10  # Switch bands every 10 minutes
        self.minimum_sdrs = 1
        self.emergency_mode = False

        # Band priorities (higher = more important)
        self.band_priorities = {
            "20m": 5,  # Highest - most propagation
            "40m": 4,
            "15m": 3,
            "80m": 2,
            "10m": 2,
            "6m": 1,   # Lowest - sporadic coverage
        }

        # Current state
        self.current_band_index = 0
        self.last_rotation: Optional[datetime] = None

    async def create_minimum_schedule(self) -> Dict[str, Any]:
        """Create minimum viable collection schedule (FR-035).

        Returns:
            Minimum schedule configuration
        """
        # Get available SDRs
        available = await self.sdr_manager.get_all_available()

        if len(available) == 0:
            logger.critical("Cannot create minimum schedule - no SDRs available")
            return {
                "active_sdrs": 0,
                "collection_strategy": "offline",
                "bands": [],
                "rotation_interval_minutes": 0,
                "priority_bands": [],
                "error": "No SDRs available",
            }

        # Create schedule based on available SDRs
        if len(available) == 1:
            strategy = "minimum_viable"
            bands = self._get_priority_bands()
            rotation_interval = self.rotation_interval_minutes
        else:
            strategy = "degraded"
            bands = BANDS
            rotation_interval = max(5, self.rotation_interval_minutes // len(available))

        # Calculate sessions per hour
        sessions_per_hour = 60 // rotation_interval

        schedule = MinimumSchedule(
            active_sdrs=len(available),
            collection_strategy=strategy,
            bands=bands,
            rotation_interval_minutes=rotation_interval,
            priority_bands=self._get_priority_bands(),
            sessions_per_hour=sessions_per_hour,
        )

        logger.info(
            f"Minimum schedule created: {len(available)} SDR(s), "
            f"rotating {len(bands)} bands every {rotation_interval} minutes"
        )

        return {
            "active_sdrs": schedule.active_sdrs,
            "collection_strategy": schedule.collection_strategy,
            "bands": schedule.bands,
            "rotation_interval_minutes": schedule.rotation_interval_minutes,
            "priority_bands": schedule.priority_bands,
            "sessions_per_hour": schedule.sessions_per_hour,
        }

    def _get_priority_bands(self) -> List[str]:
        """Get bands sorted by priority.

        Returns:
            List of band names, highest priority first
        """
        sorted_bands = sorted(
            self.band_priorities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [band for band, _ in sorted_bands]

    async def get_next_band(self) -> str:
        """Get next band in rotation sequence.

        Returns:
            Band name
        """
        # Check if rotation interval has passed
        now = datetime.utcnow()

        if self.last_rotation is None:
            self.last_rotation = now

        time_since_rotation = (now - self.last_rotation).total_seconds() / 60

        # Rotate if interval has passed
        if time_since_rotation >= self.rotation_interval_minutes:
            priority_bands = self._get_priority_bands()

            # Move to next band
            self.current_band_index = (self.current_band_index + 1) % len(priority_bands)
            self.last_rotation = now

            logger.info(
                f"Rotating to band: {priority_bands[self.current_band_index]} "
                f"(index {self.current_band_index})"
            )

        priority_bands = self._get_priority_bands()
        return priority_bands[self.current_band_index]

    async def start_minimum_collection(self):
        """Start minimum viable collection loop."""
        logger.info("Starting minimum viable collection mode")

        while True:
            try:
                # Get available SDRs
                available = await self.sdr_manager.get_all_available()

                if len(available) == 0:
                    logger.warning("No SDRs available - entering standby mode")
                    await asyncio.sleep(60)
                    continue

                # Get next band to collect
                band = await self.get_next_band()
                band_config = BAND_CONFIGS.get(band)

                if not band_config:
                    logger.error(f"Unknown band: {band}")
                    continue

                # Start recording on first available SDR
                sdr = available[0]

                logger.info(
                    f"Minimum collection: {sdr.url} on {band} "
                    f"({band_config.center_khz} kHz)"
                )

                # Start recording
                # In practice, would use actual recorder
                from ..collectors.recorder import Recorder

                recorder = Recorder()

                try:
                    session_id = await recorder.start_recording(
                        kiwisdr_url=sdr.url,
                        frequency_khz=band_config.center_khz,
                        duration_seconds=self.rotation_interval_minutes * 60,
                        band=band,
                    )

                    logger.info(f"Started minimum collection session: {session_id}")

                    # Wait for rotation interval
                    await asyncio.sleep(self.rotation_interval_minutes * 60)

                except Exception as e:
                    logger.error(f"Error in minimum collection: {e}")
                    await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Minimum collection cancelled")
                break
            except Exception as e:
                logger.error(f"Error in minimum collection loop: {e}")
                await asyncio.sleep(60)

    async def calculate_coverage_estimate(
        self, hours: int = 24
    ) -> Dict[str, Any]:
        """Calculate coverage estimate for minimum operation.

        Args:
            hours: Hours to estimate

        Returns:
            Coverage estimate
        """
        # Get current schedule
        schedule = await self.create_minimum_schedule()

        if schedule["active_sdrs"] == 0:
            return {
                "total_hours": 0,
                "hours_per_band": {},
                "coverage_percentage": 0,
                "quality": "none",
            }

        # Calculate hours per band
        sessions_per_hour = schedule["sessions_per_hour"]
        bands = schedule["bands"]

        hours_per_band = {}
        for band in bands:
            # Each band gets equal time in rotation
            band_hours = (hours * sessions_per_hour) / len(bands) * (
                schedule["rotation_interval_minutes"] / 60
            )
            hours_per_band[band] = band_hours

        total_hours = sum(hours_per_band.values())

        # Quality assessment
        if schedule["active_sdrs"] >= 6:
            quality = "excellent"
            coverage_percentage = 100
        elif schedule["active_sdrs"] >= 3:
            quality = "good"
            coverage_percentage = 75
        elif schedule["active_sdrs"] >= 1:
            quality = "minimal"
            coverage_percentage = 25
        else:
            quality = "none"
            coverage_percentage = 0

        return {
            "total_hours": total_hours,
            "hours_per_band": hours_per_band,
            "coverage_percentage": coverage_percentage,
            "quality": quality,
            "sessions_per_hour": sessions_per_hour,
            "rotation_interval_minutes": schedule["rotation_interval_minutes"],
        }

    async def optimize_rotation_for_propagation(
        self, current_conditions: Dict[str, Any]
    ) -> List[str]:
        """Optimize band rotation based on propagation conditions.

        Args:
            current_conditions: Current space weather and propagation

        Returns:
            Optimized band rotation order
        """
        # Get base priorities
        priorities = dict(self.band_priorities)

        # Adjust based on conditions
        if "k_index" in current_conditions:
            k_index = current_conditions["k_index"]

            # High K-index favors lower bands
            if k_index >= 5:
                priorities["80m"] += 2
                priorities["40m"] += 2
                priorities["20m"] -= 1

        if "solar_flux" in current_conditions:
            sfi = current_conditions["solar_flux"]

            # High SFI favors higher bands
            if sfi > 150:
                priorities["15m"] += 2
                priorities["10m"] += 2
                priorities["6m"] += 1

        if "time_of_day" in current_conditions:
            hour = current_conditions["time_of_day"]

            # Daytime favors higher bands
            if 10 <= hour <= 18:
                priorities["20m"] += 1
                priorities["15m"] += 1
                priorities["10m"] += 1
            else:
                # Nighttime favors lower bands
                priorities["80m"] += 1
                priorities["40m"] += 1

        # Sort by adjusted priorities
        sorted_bands = sorted(
            priorities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        optimized_order = [band for band, _ in sorted_bands]

        logger.info(f"Optimized band rotation: {optimized_order}")

        return optimized_order

    def close(self):
        """Close minimum scheduler."""
        self.db.close()