"""IQ recording orchestrator.

Implements T026: Recorder orchestrator (FR-002, FR-007, FR-015, FR-020).
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4

import numpy as np
import soundfile as sf

from ..collectors.kiwi_client import KiwiClient
from ..collectors.sdr_manager import SDRManager
from ..config import config
from ..models import SessionLocal, RecordingSession
from ..storage.compression import compress_to_flac

logger = logging.getLogger(__name__)


class Recorder:
    """Orchestrates IQ recording from KiwiSDR sources."""

    def __init__(self, sdr_manager: Optional[SDRManager] = None):
        """Initialize recorder.

        Args:
            sdr_manager: SDR manager for usage tracking (FR-008)
        """
        self.active_sessions: Dict[str, RecordingSession] = {}
        self.kiwi_clients: Dict[str, KiwiClient] = {}
        self.sdr_manager = sdr_manager or SDRManager()

    async def start_recording(
        self,
        kiwisdr_url: str,
        frequency_khz: float,
        duration_seconds: int,
        band: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Start a recording session (FR-002, FR-020).

        Args:
            kiwisdr_url: KiwiSDR URL
            frequency_khz: Center frequency in kHz
            duration_seconds: Recording duration
            band: Band designation (e.g., "20m")
            session_id: Optional session ID (generated if not provided)

        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid4())

        logger.info(
            f"Starting recording: {session_id} at {frequency_khz} kHz for {duration_seconds}s"
        )

        # Create database session record
        db = SessionLocal()
        try:
            # Get or create KiwiSDR source
            from ..models.kiwisdr_source import KiwiSDRSource

            kiwi_source = (
                db.query(KiwiSDRSource)
                .filter_by(url=kiwisdr_url)
                .first()
            )

            if not kiwi_source:
                kiwi_source = KiwiSDRSource(
                    url=kiwisdr_url,
                    name=kiwisdr_url.split(":")[0],
                )
                db.add(kiwi_source)
                db.commit()

            # Create recording session
            session = RecordingSession(
                session_id=session_id,
                kiwisdr_id=kiwi_source.kiwisdr_id,
                frequency_khz=frequency_khz,
                bandwidth_khz=12.0,  # FR-020: 12 kHz windows
                duration_seconds=duration_seconds,
                band=band,
                status="pending",
            )
            db.add(session)
            db.commit()

            # Store in active sessions
            self.active_sessions[session_id] = session

            # Create KiwiSDR client
            client = KiwiClient(kiwisdr_url)
            self.kiwi_clients[session_id] = client

            # Connect to KiwiSDR
            connected = await client.connect(
                frequency_khz=frequency_khz,
                mode="iq",
                bandwidth_khz=12.0,
            )

            if not connected:
                session.status = "failed"
                session.error_message = "Failed to connect to KiwiSDR"
                db.commit()
                raise ConnectionError(f"Failed to connect to {kiwisdr_url}")

            # Update session status
            session.status = "recording"
            session.start_time = datetime.utcnow()
            db.commit()

            # Start recording task
            asyncio.create_task(
                self._record_and_save(session_id, duration_seconds, db)
            )

            return session_id

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "failed"
                self.active_sessions[session_id].error_message = str(e)
            db.rollback()
            raise
        finally:
            db.close()

    async def _record_and_save(
        self,
        session_id: str,
        duration_seconds: int,
        db,
    ):
        """Record and save IQ data (FR-007, FR-015).

        Args:
            session_id: Session ID
            duration_seconds: Recording duration
            db: Database session
        """
        try:
            session = self.active_sessions[session_id]
            client = self.kiwi_clients[session_id]

            # Record IQ data
            iq_data, metadata = await client.start_recording(
                duration_seconds=duration_seconds
            )

            # Save to file (FR-015: organized by date/frequency/location)
            file_path = self._generate_file_path(session, metadata)
            await self._save_recording(iq_data, metadata, file_path)

            # Update session record
            session.end_time = datetime.utcnow()
            session.status = "completed"
            session.file_path = str(file_path)
            session.file_size_bytes = file_path.stat().st_size
            session.avg_snr_db = metadata.get("avg_snr_db")
            session.samples_received = len(iq_data)

            # Calculate actual usage duration
            if session.start_time:
                actual_duration_minutes = (session.end_time - session.start_time).total_seconds() / 60.0
            else:
                actual_duration_minutes = duration_seconds / 60.0

            # Update SDR usage tracking (FR-008, FR-014) - CRITICAL!
            kiwi_source = session.kiwisdr_source
            if kiwi_source:
                await self.sdr_manager.update_usage(kiwi_source, actual_duration_minutes)
                logger.info(
                    f"SDR {kiwi_source.url}: {actual_duration_minutes:.1f} min used, "
                    f"{kiwi_source.remaining_daily_minutes:.1f} min remaining today"
                )

            db.commit()
            logger.info(f"Recording {session_id} completed: {file_path}")

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            session.status = "failed"
            session.error_message = str(e)

            # Still track usage even on failure (FR-008)
            if session.start_time:
                actual_duration_minutes = (datetime.utcnow() - session.start_time).total_seconds() / 60.0
                kiwi_source = session.kiwisdr_source
                if kiwi_source:
                    await self.sdr_manager.update_usage(kiwi_source, actual_duration_minutes)
                    logger.info(f"Tracked {actual_duration_minutes:.1f} min usage despite failure")

            db.commit()
        finally:
            # Cleanup
            if session_id in self.kiwi_clients:
                self.kiwi_clients[session_id].disconnect()
                del self.kiwi_clients[session_id]

    def _generate_file_path(
        self,
        session: RecordingSession,
        metadata: Dict[str, Any],
    ) -> Path:
        """Generate organized file path (FR-015).

        Format: /data/recordings/YYYY/MM/DD/band/freq_time_location.flac

        Args:
            session: Recording session
            metadata: Recording metadata

        Returns:
            Path object for file
        """
        timestamp = metadata["start_time"]
        date_path = timestamp.strftime("%Y/%m/%d")

        # Band or frequency-based organization
        if session.band:
            band_path = session.band
        else:
            band_path = f"{int(session.frequency_khz)}khz"

        # Location component (grid square if available)
        location = ""
        if session.kiwisdr_source and session.kiwisdr_source.grid_square:
            location = f"_{session.kiwisdr_source.grid_square}"

        # Generate filename
        filename = (
            f"{int(session.frequency_khz)}khz"
            f"_{timestamp.strftime('%H%M%S')}"
            f"{location}"
            f"_{session.session_id[:8]}"
            ".flac"
        )

        # Construct full path
        file_path = (
            config.RECORDINGS_DIR / date_path / band_path / filename
        )

        # Create directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path

    async def _save_recording(
        self,
        iq_data: np.ndarray,
        metadata: Dict[str, Any],
        file_path: Path,
    ):
        """Save IQ data to FLAC file (FR-007).

        Args:
            iq_data: IQ samples array
            metadata: Recording metadata
            file_path: Destination path
        """
        # Normalize IQ data to [-1, 1] for audio format
        if iq_data.dtype == np.int16:
            iq_normalized = iq_data.astype(np.float32) / 32768.0
        else:
            iq_normalized = iq_data

        # Ensure 2-channel format (I and Q)
        if len(iq_normalized.shape) == 1:
            # Assume alternating I/Q samples
            iq_normalized = iq_normalized.reshape(-1, 2)

        # Save as FLAC
        sf.write(
            file_path,
            iq_normalized,
            samplerate=metadata.get("sample_rate", 12000),
            subtype="PCM_16",
            format="FLAC",
        )

        logger.info(f"Saved recording to {file_path}")

    async def stop_recording(self, session_id: str) -> Dict[str, Any]:
        """Stop an active recording.

        Args:
            session_id: Session ID to stop

        Returns:
            Status information
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        # Disconnect client if active
        if session_id in self.kiwi_clients:
            self.kiwi_clients[session_id].disconnect()
            del self.kiwi_clients[session_id]

        # Update session status
        db = SessionLocal()
        try:
            session.status = "stopped"
            session.end_time = datetime.utcnow()
            db.commit()

            return {
                "session_id": session_id,
                "status": "stopped",
                "stop_time": session.end_time,
                "file_path": session.file_path,
            }
        finally:
            db.close()
            del self.active_sessions[session_id]

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active recording sessions.

        Returns:
            List of session information dicts
        """
        return [
            {
                "session_id": str(session.session_id),
                "frequency_khz": session.frequency_khz,
                "band": session.band,
                "status": session.status,
                "start_time": session.start_time,
                "kiwisdr_url": session.kiwisdr_source.url
                if session.kiwisdr_source
                else None,
            }
            for session in self.active_sessions.values()
        ]