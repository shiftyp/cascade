"""Graceful degradation handler for progressive SDR loss.

Implements T028b: Degradation handler (FR-032).
Handles graceful degradation when SDRs become unavailable, maintaining
minimum collection with as few as 1 SDR.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from ..models import SessionLocal, CollectionAlert, KiwiSDRSource
from ..config import config

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Degradation severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DegradationStatus:
    """Current degradation status."""
    available_sdrs: int
    total_sdrs: int
    degradation_level: DegradationLevel
    operational: bool
    minimum_met: bool
    affected_bands: List[str]
    timestamp: datetime


class DegradationHandler:
    """Handles graceful degradation during SDR failures (FR-032)."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize degradation handler.

        Args:
            db_session: Database session (optional)
        """
        self.db = db_session or SessionLocal()
        self.alert_callbacks: List[Callable] = []

        # Degradation thresholds
        self.warning_threshold = 0.5  # 50% of SDRs
        self.critical_threshold = 0.3  # 30% of SDRs
        self.minimum_sdrs = 1  # Absolute minimum for operation

        # State tracking
        self.last_status: Optional[DegradationStatus] = None
        self.degradation_start: Optional[datetime] = None

    async def connect(self):
        """Connect to resources."""
        logger.info("Degradation handler initialized")

    async def disconnect(self):
        """Disconnect from resources."""
        self.db.close()
        logger.info("Degradation handler disconnected")

    def register_alert_callback(self, callback: Callable):
        """Register callback for degradation alerts.

        Args:
            callback: Async function to call on alerts
        """
        self.alert_callbacks.append(callback)

    async def check_degradation_status(self) -> Dict[str, Any]:
        """Check current degradation status (FR-032).

        Returns:
            Status dictionary with degradation level and metrics
        """
        # Query all SDRs
        total_sdrs = self.db.query(KiwiSDRSource).count()
        available_sdrs = (
            self.db.query(KiwiSDRSource)
            .filter(
                KiwiSDRSource.active == True,
                KiwiSDRSource.failure_count < 5,
            )
            .count()
        )

        # Calculate degradation level
        if total_sdrs == 0:
            degradation_level = DegradationLevel.EMERGENCY
            operational = False
            minimum_met = False
        else:
            availability_ratio = available_sdrs / total_sdrs

            if availability_ratio >= self.warning_threshold:
                degradation_level = DegradationLevel.NORMAL
            elif availability_ratio >= self.critical_threshold:
                degradation_level = DegradationLevel.WARNING
            elif available_sdrs >= self.minimum_sdrs:
                degradation_level = DegradationLevel.CRITICAL
            else:
                degradation_level = DegradationLevel.EMERGENCY

            operational = available_sdrs >= self.minimum_sdrs
            minimum_met = available_sdrs >= self.minimum_sdrs

        # Check band coverage
        affected_bands = await self._check_affected_bands()

        status = DegradationStatus(
            available_sdrs=available_sdrs,
            total_sdrs=total_sdrs,
            degradation_level=degradation_level,
            operational=operational,
            minimum_met=minimum_met,
            affected_bands=affected_bands,
            timestamp=datetime.utcnow(),
        )

        # Detect state changes
        await self._handle_status_change(status)

        self.last_status = status

        return {
            "available_sdrs": available_sdrs,
            "total_sdrs": total_sdrs,
            "degradation_level": degradation_level.value,
            "operational": operational,
            "minimum_met": minimum_met,
            "affected_bands": affected_bands,
            "availability_ratio": available_sdrs / max(total_sdrs, 1),
            "timestamp": status.timestamp.isoformat(),
        }

    async def _check_affected_bands(self) -> List[str]:
        """Check which bands are affected by degradation.

        Returns:
            List of band names with insufficient coverage
        """
        bands = ["80m", "40m", "20m", "15m", "10m", "6m"]
        affected = []

        for band in bands:
            # Check if any SDRs available for this band
            available = (
                self.db.query(KiwiSDRSource)
                .filter(
                    KiwiSDRSource.active == True,
                    KiwiSDRSource.failure_count < 5,
                )
                .count()
            )

            # If less than 1 SDR available, band is affected
            if available < 1:
                affected.append(band)

        return affected

    async def _handle_status_change(self, new_status: DegradationStatus):
        """Handle degradation status changes.

        Args:
            new_status: New status
        """
        # Check if status changed
        if self.last_status is None:
            return

        old_level = self.last_status.degradation_level
        new_level = new_status.degradation_level

        if old_level != new_level:
            logger.warning(
                f"Degradation level changed: {old_level.value} -> {new_level.value}"
            )

            # Track degradation start
            if new_level != DegradationLevel.NORMAL and self.degradation_start is None:
                self.degradation_start = new_status.timestamp
            elif new_level == DegradationLevel.NORMAL:
                self.degradation_start = None

            # Generate alerts
            await self._generate_alert(new_status, old_level)

    async def _generate_alert(
        self, status: DegradationStatus, previous_level: DegradationLevel
    ):
        """Generate degradation alert.

        Args:
            status: Current status
            previous_level: Previous degradation level
        """
        # Map level to alert severity
        severity_map = {
            DegradationLevel.NORMAL: "info",
            DegradationLevel.WARNING: "warning",
            DegradationLevel.CRITICAL: "error",
            DegradationLevel.EMERGENCY: "critical",
        }

        severity = severity_map[status.degradation_level]

        # Create alert
        alert = CollectionAlert(
            alert_type="degradation",
            severity=severity,
            message=(
                f"SDR availability degraded to {status.degradation_level.value}: "
                f"{status.available_sdrs}/{status.total_sdrs} SDRs available"
            ),
            details=(
                f"Previous level: {previous_level.value}, "
                f"Affected bands: {', '.join(status.affected_bands) if status.affected_bands else 'none'}"
            ),
        )

        self.db.add(alert)
        self.db.commit()

        logger.warning(f"Degradation alert: {alert.message}")

        # Notify callbacks
        alert_dict = {
            "level": severity,
            "available_sdrs": status.available_sdrs,
            "total_sdrs": status.total_sdrs,
            "degradation_level": status.degradation_level.value,
            "timestamp": status.timestamp.isoformat(),
            "affected_bands": status.affected_bands,
        }

        for callback in self.alert_callbacks:
            try:
                await callback(alert_dict)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def allocate_sdrs_by_priority(
        self,
        available_sdrs: int,
        priorities: Dict[str, int],
    ) -> Dict[str, str]:
        """Allocate SDRs by band priority during degradation (FR-032).

        Args:
            available_sdrs: Number of available SDRs
            priorities: Band priorities (higher = more important)

        Returns:
            Dict mapping SDR URL to assigned band
        """
        # Get available SDRs
        sdrs = (
            self.db.query(KiwiSDRSource)
            .filter(
                KiwiSDRSource.active == True,
                KiwiSDRSource.failure_count < 5,
            )
            .limit(available_sdrs)
            .all()
        )

        # Sort bands by priority
        sorted_bands = sorted(priorities.items(), key=lambda x: x[1], reverse=True)

        # Allocate SDRs to highest priority bands
        allocation = {}
        sdr_idx = 0

        for band, priority in sorted_bands:
            if sdr_idx >= len(sdrs):
                break

            allocation[sdrs[sdr_idx].url] = band
            sdr_idx += 1

        logger.info(f"Allocated {len(allocation)} SDRs by priority: {allocation}")

        return allocation

    async def check_band_coverage(self) -> Dict[str, Any]:
        """Check band coverage status.

        Returns:
            Band coverage status
        """
        bands = ["80m", "40m", "20m", "15m", "10m", "6m"]
        covered_bands = []
        uncovered_bands = []

        available_sdrs = (
            self.db.query(KiwiSDRSource)
            .filter(
                KiwiSDRSource.active == True,
                KiwiSDRSource.failure_count < 5,
            )
            .all()
        )

        # Check each band
        for band in bands:
            # For now, assume any available SDR can cover any band
            # In practice, would check SDR capabilities and band-specific availability
            if available_sdrs:
                covered_bands.append(band)
            else:
                uncovered_bands.append(band)

        return {
            "covered_bands": covered_bands,
            "uncovered_bands": uncovered_bands,
            "coverage_ratio": len(covered_bands) / len(bands),
            "all_bands_covered": len(uncovered_bands) == 0,
        }

    async def get_degradation_history(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get degradation event history.

        Args:
            hours: Hours of history to retrieve

        Returns:
            List of degradation events
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        alerts = (
            self.db.query(CollectionAlert)
            .filter(
                CollectionAlert.alert_type == "degradation",
                CollectionAlert.created_at >= cutoff,
            )
            .order_by(CollectionAlert.created_at.desc())
            .all()
        )

        history = []
        for alert in alerts:
            history.append({
                "timestamp": alert.created_at.isoformat(),
                "severity": alert.severity,
                "message": alert.message,
                "details": alert.details,
                "acknowledged": alert.acknowledged,
            })

        return history

    def close(self):
        """Close degradation handler."""
        self.db.close()