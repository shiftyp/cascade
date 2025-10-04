"""SDR Manager for coordinating SDR connections and rotations."""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class SDRStatus(Enum):
    """SDR connection status."""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECORDING = "recording"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class SDRConnection:
    """SDR connection details."""
    sdr_id: str
    url: str
    band: str
    status: SDRStatus = SDRStatus.IDLE
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    usage_minutes: float = 0
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SDRManager:
    """Manages SDR connections and rotation."""

    def __init__(self, max_connections: int = 6):
        """Initialize SDR manager.

        Args:
            max_connections: Maximum concurrent SDR connections
        """
        self.max_connections = max_connections
        self.connections: Dict[str, SDRConnection] = {}
        self.active_sdrs: Set[str] = set()
        self.rotation_interval = timedelta(minutes=30)
        self.usage_limit = 90  # minutes per day per SDR
        self._lock = asyncio.Lock()
        self.rotation_task: Optional[asyncio.Task] = None

    async def connect_sdr(self, sdr_id: str, url: str, band: str) -> bool:
        """Connect to an SDR.

        Args:
            sdr_id: SDR identifier
            url: SDR URL
            band: Frequency band

        Returns:
            True if connected successfully
        """
        async with self._lock:
            # Check if already connected
            if sdr_id in self.active_sdrs:
                logger.warning(f"SDR {sdr_id} already connected")
                return False

            # Check connection limit
            if len(self.active_sdrs) >= self.max_connections:
                logger.warning(f"Max connections reached ({self.max_connections})")
                return False

            # Create connection
            connection = SDRConnection(
                sdr_id=sdr_id,
                url=url,
                band=band,
                status=SDRStatus.CONNECTING,
                connected_at=datetime.now(timezone.utc)
            )

            self.connections[sdr_id] = connection
            self.active_sdrs.add(sdr_id)

            # Simulate connection (in real implementation, would connect to actual SDR)
            await asyncio.sleep(0.5)
            connection.status = SDRStatus.CONNECTED

            logger.info(f"Connected to SDR {sdr_id} for band {band}")
            return True

    async def disconnect_sdr(self, sdr_id: str) -> bool:
        """Disconnect from an SDR.

        Args:
            sdr_id: SDR identifier

        Returns:
            True if disconnected successfully
        """
        async with self._lock:
            if sdr_id not in self.active_sdrs:
                logger.warning(f"SDR {sdr_id} not connected")
                return False

            connection = self.connections[sdr_id]
            connection.status = SDRStatus.DISCONNECTED
            connection.disconnected_at = datetime.now(timezone.utc)

            # Update usage time
            if connection.connected_at:
                duration = (connection.disconnected_at - connection.connected_at).total_seconds() / 60
                connection.usage_minutes += duration

            self.active_sdrs.remove(sdr_id)

            logger.info(f"Disconnected from SDR {sdr_id}")
            return True

    async def rotate_sdrs(self, available_sdrs: List[Dict[str, str]]) -> int:
        """Rotate SDR connections.

        Args:
            available_sdrs: List of available SDRs with 'id', 'url', 'band'

        Returns:
            Number of SDRs rotated
        """
        rotated = 0

        async with self._lock:
            # Find SDRs that have been connected too long
            now = datetime.now(timezone.utc)
            sdrs_to_rotate = []

            for sdr_id in list(self.active_sdrs):
                connection = self.connections[sdr_id]
                if connection.connected_at:
                    duration = now - connection.connected_at
                    if duration > self.rotation_interval:
                        sdrs_to_rotate.append(sdr_id)

            # Disconnect old SDRs
            for sdr_id in sdrs_to_rotate:
                await self.disconnect_sdr(sdr_id)
                rotated += 1

            # Connect new SDRs
            for sdr_info in available_sdrs:
                if len(self.active_sdrs) >= self.max_connections:
                    break

                sdr_id = sdr_info['id']
                if sdr_id not in self.active_sdrs:
                    # Check usage limit
                    if sdr_id in self.connections:
                        if self.connections[sdr_id].usage_minutes >= self.usage_limit:
                            continue

                    success = await self.connect_sdr(
                        sdr_id=sdr_id,
                        url=sdr_info['url'],
                        band=sdr_info['band']
                    )
                    if success:
                        rotated += 1

        logger.info(f"Rotated {rotated} SDR connections")
        return rotated

    async def start_auto_rotation(self, available_sdrs: List[Dict[str, str]]):
        """Start automatic SDR rotation.

        Args:
            available_sdrs: List of available SDRs
        """
        if self.rotation_task and not self.rotation_task.done():
            logger.warning("Auto-rotation already running")
            return

        async def rotation_loop():
            while True:
                try:
                    await asyncio.sleep(self.rotation_interval.total_seconds())
                    await self.rotate_sdrs(available_sdrs)
                except Exception as e:
                    logger.error(f"Rotation error: {e}")

        self.rotation_task = asyncio.create_task(rotation_loop())
        logger.info("Started automatic SDR rotation")

    async def stop_auto_rotation(self):
        """Stop automatic SDR rotation."""
        if self.rotation_task and not self.rotation_task.done():
            self.rotation_task.cancel()
            try:
                await self.rotation_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped automatic SDR rotation")

    def get_status(self) -> Dict[str, Any]:
        """Get manager status.

        Returns:
            Status dictionary
        """
        return {
            "active_sdrs": len(self.active_sdrs),
            "max_connections": self.max_connections,
            "total_connections": len(self.connections),
            "connections": [
                {
                    "sdr_id": conn.sdr_id,
                    "band": conn.band,
                    "status": conn.status.value,
                    "usage_minutes": round(conn.usage_minutes, 2),
                    "error_count": conn.error_count
                }
                for conn in self.connections.values()
            ]
        }

    def get_connection(self, sdr_id: str) -> Optional[SDRConnection]:
        """Get connection details for an SDR.

        Args:
            sdr_id: SDR identifier

        Returns:
            SDRConnection or None if not found
        """
        return self.connections.get(sdr_id)

    def get_active_bands(self) -> Set[str]:
        """Get set of bands with active connections.

        Returns:
            Set of band identifiers
        """
        bands = set()
        for sdr_id in self.active_sdrs:
            if sdr_id in self.connections:
                bands.add(self.connections[sdr_id].band)
        return bands

    async def handle_error(self, sdr_id: str, error: str):
        """Handle SDR error.

        Args:
            sdr_id: SDR identifier
            error: Error message
        """
        async with self._lock:
            if sdr_id in self.connections:
                connection = self.connections[sdr_id]
                connection.error_count += 1
                connection.last_error = error
                connection.status = SDRStatus.ERROR

                # Disconnect if too many errors
                if connection.error_count >= 3:
                    logger.error(f"Too many errors for SDR {sdr_id}, disconnecting")
                    await self.disconnect_sdr(sdr_id)

    async def cleanup_stale_connections(self, hours: int = 24) -> int:
        """Clean up stale connections.

        Args:
            hours: Hours before considering connection stale

        Returns:
            Number of connections cleaned
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=hours)
            cleaned = 0

            for sdr_id in list(self.connections.keys()):
                conn = self.connections[sdr_id]
                if conn.disconnected_at and conn.disconnected_at < cutoff:
                    del self.connections[sdr_id]
                    cleaned += 1

            if cleaned:
                logger.info(f"Cleaned {cleaned} stale connections")

            return cleaned

    async def reset_daily_usage(self):
        """Reset daily usage counters."""
        async with self._lock:
            for connection in self.connections.values():
                connection.usage_minutes = 0
            logger.info("Reset daily usage counters")

    async def get_best_sdrs_for_bands(self, bands: List[str], available_sdrs: List[Dict]) -> List[Dict]:
        """Get best available SDRs for specified bands.

        Args:
            bands: List of frequency bands
            available_sdrs: List of available SDRs

        Returns:
            List of selected SDRs
        """
        selected = []

        for band in bands:
            # Filter SDRs for this band
            band_sdrs = [sdr for sdr in available_sdrs if sdr.get('band') == band]

            if not band_sdrs:
                logger.warning(f"No SDRs available for band {band}")
                continue

            # Sort by usage (prefer less used SDRs)
            band_sdrs.sort(key=lambda x: self.connections.get(x['id'], SDRConnection(
                sdr_id=x['id'], url='', band=''
            )).usage_minutes)

            # Select SDR with lowest usage
            best_sdr = band_sdrs[0]

            # Check usage limit
            if best_sdr['id'] in self.connections:
                if self.connections[best_sdr['id']].usage_minutes < self.usage_limit:
                    selected.append(best_sdr)
            else:
                selected.append(best_sdr)

        return selected