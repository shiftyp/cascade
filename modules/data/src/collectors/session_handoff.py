"""Graceful session handoff manager for SDR time limit management.

Implements FR-064: Graceful session management with automatic disconnection
before time limits and seamless handoff to alternative SDRs.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..models import SessionLocal, KiwiSDRSource, WebSDRSource, RecordingSession
from .kiwi_client import KiwiClient
from .websdr_client import WebSDRClient
from .hybrid_sdr_selector import HybridSDRSelector

logger = logging.getLogger(__name__)


class HandoffState(Enum):
    """Session handoff states."""
    NORMAL = "normal"
    WARNING = "warning"  # 85% of time limit
    PREPARING = "preparing"  # 90% - allocating replacement
    HANDOFF = "handoff"  # 95% - active transition
    COMPLETE = "complete"


@dataclass
class SessionInfo:
    """Active session information."""
    session_id: str
    sdr_url: str
    sdr_type: str  # "kiwi" or "websdr"
    start_time: datetime
    time_limit_minutes: int
    frequency_khz: float
    band: str
    client: Any  # KiwiClient or WebSDRClient
    buffer: bytearray  # Overlap buffer for seamless transition


class SessionHandoffManager:
    """Manages graceful handoff between SDRs before time limits."""

    def __init__(self):
        """Initialize handoff manager."""
        self.db = SessionLocal()
        self.hybrid_selector = HybridSDRSelector()
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.pending_handoffs: Dict[str, str] = {}  # session_id -> new_sdr_url
        self.handoff_buffers: Dict[str, bytearray] = {}  # Overlap audio

        # Handoff thresholds
        self.warning_threshold = 0.85  # Warn at 85% of time limit
        self.prepare_threshold = 0.90  # Start preparing at 90%
        self.handoff_threshold = 0.95  # Execute handoff at 95%
        self.overlap_seconds = 5  # Record overlap for seamless merge

    async def monitor_sessions(self):
        """Monitor all active sessions for approaching time limits."""
        while True:
            try:
                for session_id, session in list(self.active_sessions.items()):
                    state = self._get_session_state(session)

                    if state == HandoffState.WARNING:
                        await self._handle_warning(session)
                    elif state == HandoffState.PREPARING:
                        await self._prepare_handoff(session)
                    elif state == HandoffState.HANDOFF:
                        await self._execute_handoff(session)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in session monitor: {e}")
                await asyncio.sleep(30)

    def _get_session_state(self, session: SessionInfo) -> HandoffState:
        """Determine current handoff state for a session."""
        elapsed = (datetime.utcnow() - session.start_time).total_seconds() / 60
        usage_percent = elapsed / session.time_limit_minutes

        if usage_percent >= self.handoff_threshold:
            return HandoffState.HANDOFF
        elif usage_percent >= self.prepare_threshold:
            return HandoffState.PREPARING
        elif usage_percent >= self.warning_threshold:
            return HandoffState.WARNING
        else:
            return HandoffState.NORMAL

    async def _handle_warning(self, session: SessionInfo):
        """Handle warning state - log and notify."""
        remaining = session.time_limit_minutes - \
                   (datetime.utcnow() - session.start_time).total_seconds() / 60

        logger.warning(
            f"Session {session.session_id} approaching time limit: "
            f"{remaining:.1f} minutes remaining on {session.sdr_url}"
        )

    async def _prepare_handoff(self, session: SessionInfo):
        """Prepare for handoff by allocating replacement SDR."""
        if session.session_id in self.pending_handoffs:
            return  # Already preparing

        logger.info(f"Preparing handoff for session {session.session_id}")

        # Find replacement SDR
        remaining_time = session.time_limit_minutes - \
                        (datetime.utcnow() - session.start_time).total_seconds() / 60

        # Request SDR for remaining recording time + buffer
        replacement = await self._find_replacement_sdr(
            session.frequency_khz,
            session.band,
            remaining_time + 10,  # Add buffer time
            exclude_url=session.sdr_url
        )

        if replacement:
            self.pending_handoffs[session.session_id] = replacement.url
            logger.info(
                f"Allocated replacement SDR {replacement.url} "
                f"for session {session.session_id}"
            )

            # Pre-connect to reduce handoff time
            await self._preconnect_sdr(replacement.url, session.frequency_khz)
        else:
            logger.error(f"No replacement SDR available for {session.session_id}")

    async def _execute_handoff(self, session: SessionInfo):
        """Execute seamless handoff to new SDR."""
        if session.session_id not in self.pending_handoffs:
            logger.error(f"No replacement SDR for emergency handoff of {session.session_id}")
            await self._emergency_disconnect(session)
            return

        new_sdr_url = self.pending_handoffs[session.session_id]
        logger.info(f"Executing handoff: {session.sdr_url} -> {new_sdr_url}")

        try:
            # Start recording on new SDR with overlap
            new_client = await self._start_new_recording(
                new_sdr_url,
                session.frequency_khz,
                session.band
            )

            # Record overlap period on both SDRs
            overlap_task1 = asyncio.create_task(
                self._record_overlap(session.client, session.session_id + "_old")
            )
            overlap_task2 = asyncio.create_task(
                self._record_overlap(new_client, session.session_id + "_new")
            )

            # Wait for overlap recording
            old_data, new_data = await asyncio.gather(overlap_task1, overlap_task2)

            # Store overlap for seamless merge
            self.handoff_buffers[session.session_id] = {
                'old': old_data,
                'new': new_data,
                'crossfade_samples': int(self.overlap_seconds * 12000 / 2)  # Half overlap for crossfade
            }

            # Disconnect from old SDR
            await self._disconnect_sdr(session.client, session.sdr_url)

            # Update session info
            session.sdr_url = new_sdr_url
            session.client = new_client
            session.start_time = datetime.utcnow()  # Reset timer
            session.sdr_type = self._get_sdr_type(new_sdr_url)

            # Update database
            await self._update_session_database(session)

            logger.info(f"Handoff complete for session {session.session_id}")

            # Cleanup
            del self.pending_handoffs[session.session_id]

        except Exception as e:
            logger.error(f"Handoff failed for {session.session_id}: {e}")
            await self._emergency_disconnect(session)

    async def _record_overlap(self, client: Any, buffer_id: str) -> np.ndarray:
        """Record overlap period for seamless transition."""
        samples = []
        duration = self.overlap_seconds

        for _ in range(duration):
            data = await client.get_iq_data()
            if data:
                samples.extend(data)
            await asyncio.sleep(1)

        return np.array(samples)

    async def _find_replacement_sdr(
        self,
        frequency_khz: float,
        band: str,
        duration_minutes: float,
        exclude_url: str
    ) -> Optional[Any]:
        """Find replacement SDR for handoff."""
        # Use hybrid selector to find best replacement
        candidates = self.hybrid_selector.get_available_sdrs(
            frequency_khz=frequency_khz,
            duration_minutes=duration_minutes,
            exclude_urls=[exclude_url]
        )

        if candidates:
            # Prefer WebSDR for longer remaining time
            if duration_minutes > 90:
                websdr_candidates = [c for c in candidates if c.sdr_type == "websdr"]
                if websdr_candidates:
                    return websdr_candidates[0]

            return candidates[0]

        return None

    async def _preconnect_sdr(self, url: str, frequency_khz: float):
        """Pre-connect to SDR to reduce handoff time."""
        try:
            if self._get_sdr_type(url) == "kiwi":
                client = KiwiClient(url)
            else:
                client = WebSDRClient()

            # Just test connection, don't start recording yet
            await client.connect(frequency_khz)
            await client.disconnect()

        except Exception as e:
            logger.warning(f"Pre-connection failed for {url}: {e}")

    async def _start_new_recording(
        self,
        url: str,
        frequency_khz: float,
        band: str
    ) -> Any:
        """Start recording on new SDR."""
        if self._get_sdr_type(url) == "kiwi":
            client = KiwiClient(url)
        else:
            client = WebSDRClient()

        await client.connect(frequency_khz, mode="iq", bandwidth_khz=12.0)
        return client

    async def _disconnect_sdr(self, client: Any, url: str):
        """Gracefully disconnect from SDR."""
        try:
            await client.disconnect()
            logger.info(f"Disconnected from {url}")
        except Exception as e:
            logger.error(f"Error disconnecting from {url}: {e}")

    async def _emergency_disconnect(self, session: SessionInfo):
        """Emergency disconnect when no handoff possible."""
        logger.error(f"Emergency disconnect for session {session.session_id}")

        try:
            await session.client.disconnect()
        except:
            pass

        # Mark session as ended with data loss warning
        await self._mark_session_incomplete(session.session_id)

        # Remove from active sessions
        if session.session_id in self.active_sessions:
            del self.active_sessions[session.session_id]

    async def _update_session_database(self, session: SessionInfo):
        """Update session record in database after handoff."""
        db_session = (
            self.db.query(RecordingSession)
            .filter(RecordingSession.session_id == session.session_id)
            .first()
        )

        if db_session:
            db_session.notes = f"Handoff executed at {datetime.utcnow().isoformat()}"
            self.db.commit()

    async def _mark_session_incomplete(self, session_id: str):
        """Mark session as incomplete due to handoff failure."""
        db_session = (
            self.db.query(RecordingSession)
            .filter(RecordingSession.session_id == session_id)
            .first()
        )

        if db_session:
            db_session.quality_check = {"status": "incomplete", "reason": "handoff_failed"}
            db_session.notes = "Session ended early due to handoff failure"
            self.db.commit()

    def _get_sdr_type(self, url: str) -> str:
        """Determine if URL is KiwiSDR or WebSDR."""
        # Simple heuristic - can be improved
        if "websdr" in url.lower():
            return "websdr"
        return "kiwi"

    def register_session(
        self,
        session_id: str,
        sdr_url: str,
        start_time: datetime,
        time_limit_minutes: int,
        frequency_khz: float,
        band: str,
        client: Any
    ):
        """Register a new session for monitoring."""
        self.active_sessions[session_id] = SessionInfo(
            session_id=session_id,
            sdr_url=sdr_url,
            sdr_type=self._get_sdr_type(sdr_url),
            start_time=start_time,
            time_limit_minutes=time_limit_minutes,
            frequency_khz=frequency_khz,
            band=band,
            client=client,
            buffer=bytearray()
        )

        logger.info(
            f"Registered session {session_id} for handoff monitoring "
            f"(limit: {time_limit_minutes} minutes)"
        )

    def unregister_session(self, session_id: str):
        """Remove session from monitoring."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Unregistered session {session_id}")

    def get_handoff_buffer(self, session_id: str) -> Optional[Dict]:
        """Get overlap buffer for seamless audio merge."""
        return self.handoff_buffers.get(session_id)