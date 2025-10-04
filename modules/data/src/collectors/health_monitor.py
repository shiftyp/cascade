"""SDR health monitoring and auto-reconnection system.

Implements T028c: Health monitor (FR-033).
Auto-reconnects failed SDRs with exponential backoff and tracks
SDR availability and performance metrics.
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
from ..notifications.gmail_notifier import GmailNotifier

logger = logging.getLogger(__name__)


class SDRHealthStatus(Enum):
    """SDR health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HealthMetrics:
    """SDR health metrics."""
    sdr_url: str
    status: SDRHealthStatus
    uptime_percentage: float
    failure_count: int
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    reconnection_attempts: int
    avg_connection_time: float
    timestamp: datetime


class HealthMonitor:
    """Monitors SDR health and handles auto-reconnection (FR-033)."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize health monitor.

        Args:
            db_session: Database session (optional)
        """
        self.db = db_session or SessionLocal()
        self.callbacks: Dict[str, List[Callable]] = {}
        self.alert_callbacks: List[Callable] = []
        self.notification_callbacks: List[Callable] = []

        # Health monitoring parameters
        self.check_interval_seconds = 300  # Check every 5 minutes
        self.reconnection_interval_seconds = 300  # Reconnect every 5 minutes
        self.failure_threshold = 3  # Consecutive failures before marking failed

        # Backoff parameters
        self.initial_backoff_seconds = 60  # Start with 1 minute
        self.max_backoff_seconds = 3600  # Max 1 hour
        self.backoff_multiplier = 2.0

        # State tracking
        self.reconnection_state: Dict[str, Dict[str, Any]] = {}
        self.running = False

        # Initialize Gmail notifier for health alerts (FR-034)
        try:
            self.notifier = GmailNotifier()
            logger.info("Gmail notifier initialized for health monitoring")
        except ValueError as e:
            logger.warning(f"Gmail notifier not configured: {e}")
            self.notifier = None

    async def connect(self):
        """Connect to resources."""
        logger.info("Health monitor initialized")

    async def disconnect(self):
        """Disconnect from resources."""
        self.running = False
        self.db.close()
        logger.info("Health monitor disconnected")

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for health events.

        Args:
            event_type: Event type (e.g., "reconnection_attempt", "total_failure")
            callback: Async function to call
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts.

        Args:
            callback: Async function to call
        """
        self.alert_callbacks.append(callback)

    def register_notification_callback(self, callback: Callable):
        """Register callback for notifications.

        Args:
            callback: Async function to call
        """
        self.notification_callbacks.append(callback)

    async def monitor_and_recover(
        self,
        initial_backoff_seconds: Optional[float] = None,
        max_backoff_seconds: Optional[float] = None,
    ):
        """Main monitoring and recovery loop (FR-033).

        Args:
            initial_backoff_seconds: Override initial backoff
            max_backoff_seconds: Override max backoff
        """
        self.running = True

        # Override backoff parameters if provided
        if initial_backoff_seconds is not None:
            self.initial_backoff_seconds = initial_backoff_seconds
        if max_backoff_seconds is not None:
            self.max_backoff_seconds = max_backoff_seconds

        logger.info("Starting SDR health monitoring and recovery")

        while self.running:
            try:
                # Check all SDRs
                await self._check_all_sdrs()

                # Attempt reconnections
                await self._attempt_reconnections()

                # Wait before next check
                await asyncio.sleep(self.check_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(30)

    async def _check_all_sdrs(self):
        """Check health of all SDRs."""
        sdrs = self.db.query(KiwiSDRSource).all()

        healthy_count = 0
        failed_count = 0

        for sdr in sdrs:
            status = await self._check_sdr_health(sdr)

            if status == SDRHealthStatus.HEALTHY:
                healthy_count += 1
            elif status == SDRHealthStatus.FAILED:
                failed_count += 1

        logger.info(
            f"Health check complete: {healthy_count} healthy, "
            f"{failed_count} failed of {len(sdrs)} total"
        )

        # Check for total failure
        if healthy_count == 0 and len(sdrs) > 0:
            await self._notify_total_failure()

    async def _check_sdr_health(self, sdr: KiwiSDRSource) -> SDRHealthStatus:
        """Check health of a single SDR.

        Args:
            sdr: SDR to check

        Returns:
            Health status
        """
        # Check if SDR is marked as active
        if not sdr.active:
            return SDRHealthStatus.FAILED

        # Check failure count
        if sdr.failure_count >= self.failure_threshold:
            return SDRHealthStatus.FAILED

        # Check if recently used successfully
        if sdr.last_connected:
            age = datetime.utcnow() - sdr.last_connected.replace(tzinfo=None)
            if age < timedelta(hours=1):
                return SDRHealthStatus.HEALTHY

        # Check reliability score
        if sdr.reliability_score and sdr.reliability_score < 0.5:
            return SDRHealthStatus.DEGRADED

        # If no recent activity, consider degraded
        if sdr.last_connected:
            age = datetime.utcnow() - sdr.last_connected.replace(tzinfo=None)
            if age > timedelta(hours=24):
                return SDRHealthStatus.DEGRADED

        return SDRHealthStatus.HEALTHY

    async def _attempt_reconnections(self):
        """Attempt to reconnect failed SDRs with exponential backoff."""
        # Get failed SDRs
        failed_sdrs = (
            self.db.query(KiwiSDRSource)
            .filter(
                (KiwiSDRSource.active == False) | (KiwiSDRSource.failure_count >= self.failure_threshold)
            )
            .all()
        )

        if not failed_sdrs:
            return

        reconnection_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "sdrs_checked": len(failed_sdrs),
        }

        await self._fire_callbacks("reconnection_attempt", reconnection_event)

        for sdr in failed_sdrs:
            # Check if enough time has passed since last attempt
            if sdr.url not in self.reconnection_state:
                self.reconnection_state[sdr.url] = {
                    "last_attempt": None,
                    "attempts": 0,
                    "backoff_seconds": self.initial_backoff_seconds,
                }

            state = self.reconnection_state[sdr.url]

            # Check if we should attempt reconnection
            if state["last_attempt"]:
                time_since_last = datetime.utcnow() - state["last_attempt"]
                if time_since_last.total_seconds() < state["backoff_seconds"]:
                    continue  # Still in backoff period

            # Attempt reconnection
            logger.info(f"Attempting reconnection to {sdr.url}")

            success = await self._attempt_single_reconnection(sdr)

            state["last_attempt"] = datetime.utcnow()
            state["attempts"] += 1

            if success:
                logger.info(f"Successfully reconnected to {sdr.url}")

                # Reset failure count
                sdr.failure_count = 0
                sdr.active = True
                sdr.reliability_score = min(1.0, (sdr.reliability_score or 0.5) + 0.1)
                self.db.commit()

                # Reset backoff
                del self.reconnection_state[sdr.url]

            else:
                logger.warning(f"Failed to reconnect to {sdr.url}")

                # Increase backoff exponentially
                state["backoff_seconds"] = min(
                    state["backoff_seconds"] * self.backoff_multiplier,
                    self.max_backoff_seconds,
                )

    async def _attempt_single_reconnection(self, sdr: KiwiSDRSource) -> bool:
        """Attempt to reconnect to a single SDR.

        Args:
            sdr: SDR to reconnect

        Returns:
            True if successful
        """
        try:
            # In practice, would attempt actual connection
            # For now, simulate with a simple check
            from ..collectors.kiwi_client import KiwiClient

            client = KiwiClient()

            # Test connection with short timeout
            connected = await client.test_connection(
                sdr.url,
                timeout_seconds=10,
            )

            return connected

        except Exception as e:
            logger.debug(f"Reconnection attempt failed for {sdr.url}: {e}")
            return False

    async def _notify_total_failure(self):
        """Notify about total SDR failure (FR-034)."""
        logger.critical("TOTAL SDR FAILURE DETECTED")

        # Send operator alert via Gmail (FR-034)
        if self.notifier:
            try:
                alert_body = """CRITICAL ALERT: Total SDR Failure

All KiwiSDRs are currently unavailable. Data collection has been halted.

Automatic reconnection attempts are in progress with exponential backoff.

Actions being taken:
1. Attempting to reconnect to all known SDRs
2. Exponential backoff applied (1 min -> 1 hour max)
3. Monitoring for SDR recovery
4. Will resume collection when SDRs become available

This is a critical situation requiring immediate attention.
Review SDR availability at: https://kiwisdr.com/public/

Check system logs for more details.
"""
                success = self.notifier.send_notification(
                    subject="[CASCADE CRITICAL] Total SDR Failure - Collection Halted",
                    body=alert_body,
                    priority="high"
                )
                if success:
                    logger.info("Critical alert sent to operators via email")
                else:
                    logger.error("Failed to send critical email alert")
            except Exception as e:
                logger.error(f"Error sending Gmail alert: {e}")

        # Create critical alert
        alert = CollectionAlert(
            alert_type="total_sdr_failure",
            severity="critical",
            message="All SDRs are unavailable - collection halted",
            details="No healthy SDRs found. Automatic reconnection in progress.",
        )

        self.db.add(alert)
        self.db.commit()

        # Fire callbacks
        failure_event = {
            "event_type": "total_sdr_failure",
            "available_sdrs": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._fire_callbacks("total_failure", failure_event)

        # Fire alert callbacks
        for callback in self.alert_callbacks:
            try:
                alert_dict = {
                    "level": "critical",
                    "event_type": "total_sdr_failure",
                    "available_sdrs": 0,
                    "notification_required": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await callback(alert_dict)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def check_recovery_status(self) -> Dict[str, Any]:
        """Check recovery status after failures.

        Returns:
            Recovery status
        """
        total_sdrs = self.db.query(KiwiSDRSource).count()
        available_sdrs = (
            self.db.query(KiwiSDRSource)
            .filter(
                KiwiSDRSource.active == True,
                KiwiSDRSource.failure_count < self.failure_threshold,
            )
            .count()
        )

        if total_sdrs == 0:
            recovery_type = "none"
            operational_mode = "offline"
        elif available_sdrs == 0:
            recovery_type = "none"
            operational_mode = "failed"
        elif available_sdrs == total_sdrs:
            recovery_type = "full"
            operational_mode = "normal"
        elif available_sdrs < total_sdrs * 0.5:
            recovery_type = "partial"
            operational_mode = "degraded"
        else:
            recovery_type = "partial"
            operational_mode = "normal"

        recovery_percentage = (available_sdrs / total_sdrs * 100) if total_sdrs > 0 else 0

        return {
            "recovery_type": recovery_type,
            "available_sdrs": available_sdrs,
            "total_sdrs": total_sdrs,
            "recovery_percentage": recovery_percentage,
            "operational_mode": operational_mode,
        }

    async def get_sdr_metrics(self, sdr_url: str) -> HealthMetrics:
        """Get health metrics for a specific SDR.

        Args:
            sdr_url: SDR URL

        Returns:
            Health metrics
        """
        sdr = (
            self.db.query(KiwiSDRSource)
            .filter(KiwiSDRSource.url == sdr_url)
            .first()
        )

        if not sdr:
            raise ValueError(f"SDR not found: {sdr_url}")

        status = await self._check_sdr_health(sdr)

        # Calculate uptime percentage
        # In practice, would track actual uptime
        uptime_percentage = (
            100.0 - (sdr.failure_count * 10.0) if sdr.failure_count < 10 else 0.0
        )

        # Get reconnection attempts
        reconnection_attempts = 0
        if sdr_url in self.reconnection_state:
            reconnection_attempts = self.reconnection_state[sdr_url]["attempts"]

        return HealthMetrics(
            sdr_url=sdr_url,
            status=status,
            uptime_percentage=uptime_percentage,
            failure_count=sdr.failure_count or 0,
            last_success=sdr.last_connected,
            last_failure=None,  # Would track separately
            reconnection_attempts=reconnection_attempts,
            avg_connection_time=0.0,  # Would track separately
            timestamp=datetime.utcnow(),
        )

    async def _fire_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Fire callbacks for an event type.

        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type not in self.callbacks:
            return

        for callback in self.callbacks[event_type]:
            try:
                await callback(event_data)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}")

    def close(self):
        """Close health monitor."""
        self.running = False
        self.db.close()