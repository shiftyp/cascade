"""IQ recording orchestrator.

Implements T026: Recorder orchestrator (FR-002, FR-007, FR-015, FR-020).
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4

import numpy as np
import soundfile as sf
import boto3
from botocore.exceptions import ClientError

from ..collectors.kiwi_client import KiwiClient
from ..collectors.websdr_client import WebSDRClient
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
        self.websdr_clients: Dict[str, WebSDRClient] = {}
        self.sdr_manager = sdr_manager or SDRManager()

        # Initialize Tigris S3 client for cloud storage
        self.s3_client = None
        self.tigris_bucket = os.getenv('TIGRIS_BUCKET', 'cascade-iq-data')

        # Initialize Tigris if credentials available
        tigris_access_key = os.getenv('TIGRIS_ACCESS_KEY') or os.getenv('AWS_ACCESS_KEY_ID')
        tigris_secret_key = os.getenv('TIGRIS_SECRET_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY')

        if tigris_access_key and tigris_secret_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3', 'https://fly.storage.tigris.dev'),
                    aws_access_key_id=tigris_access_key,
                    aws_secret_access_key=tigris_secret_key,
                    region_name=os.getenv('AWS_REGION', 'auto'),
                )
                logger.info(f"Tigris S3 client initialized for bucket: {self.tigris_bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize Tigris client: {e}")
                logger.warning("Recordings will be stored locally only")
        else:
            logger.warning("Tigris credentials not found - recordings will be stored locally only")

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
            # Determine SDR type
            is_websdr = "websdr" in kiwisdr_url.lower() or kiwisdr_url.startswith("http")

            if is_websdr:
                # Get or create WebSDR source
                from ..models.websdr_source import WebSDRSource

                websdr_source = (
                    db.query(WebSDRSource)
                    .filter_by(url=kiwisdr_url)
                    .first()
                )

                if not websdr_source:
                    websdr_source = WebSDRSource(
                        url=kiwisdr_url,
                        name=kiwisdr_url.split("/")[2] if "/" in kiwisdr_url else kiwisdr_url,
                    )
                    db.add(websdr_source)
                    db.commit()

                # Create recording session with WebSDR
                session = RecordingSession(
                    session_id=session_id,
                    websdr_id=websdr_source.websdr_id,
                    frequency_khz=frequency_khz,
                    bandwidth_khz=12.0,  # FR-020: 12 kHz windows
                    duration_seconds=duration_seconds,
                    band=band,
                    status="pending",
                )
            else:
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

                # Create recording session with KiwiSDR
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

            # Determine SDR type and create appropriate client
            # Only treat as WebSDR if explicitly marked in URL
            is_websdr = "websdr" in kiwisdr_url.lower()

            if is_websdr:
                # Create WebSDR client
                client = WebSDRClient(kiwisdr_url, session_id)
                self.websdr_clients[session_id] = client

                # Connect to WebSDR
                connected = await client.connect(
                    frequency_khz=frequency_khz,
                    mode="iq",
                    bandwidth_khz=12.0,
                )
            else:
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
                session.error_message = f"Failed to connect to {'WebSDR' if is_websdr else 'KiwiSDR'}"
                db.commit()
                raise ConnectionError(f"Failed to connect to {kiwisdr_url}")

            # Update session status
            session.status = "recording"
            session.start_time = datetime.now(timezone.utc)
            db.commit()

            # Commit session to database before starting async task
            db.commit()

            # Start recording task (will create its own db session)
            asyncio.create_task(
                self._record_and_save(session_id, duration_seconds)
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
    ):
        """Record and save IQ data (FR-007, FR-015).

        Args:
            session_id: Session ID
            duration_seconds: Recording duration
        """
        # Create new database session for this async task
        db = SessionLocal()

        try:
            # Get session from active sessions (in-memory)
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found in active sessions")
                return

            # Reload from database to attach to this thread's session
            session = db.query(RecordingSession).filter(
                RecordingSession.session_id == session_id
            ).first()

            if not session:
                logger.error(f"Session {session_id} not found in database")
                return

            client = self.kiwi_clients.get(session_id) or self.websdr_clients.get(session_id)
            if not client:
                logger.error(f"No client found for session {session_id}")
                return

            # Record IQ data
            iq_data, metadata = await client.start_recording(
                duration_seconds=duration_seconds
            )

            # Save to file (FR-015: organized by date/frequency/location)
            file_path = self._generate_file_path(session, metadata)
            await self._save_recording(iq_data, metadata, file_path)

            # Upload to Tigris if configured
            tigris_path = None
            if self.s3_client:
                tigris_path = await self._upload_to_tigris(file_path, session, metadata)

            # Update session record
            session.end_time = datetime.now(timezone.utc)
            session.status = "completed"
            session.file_path = str(file_path)
            session.tigris_path = tigris_path
            session.file_size_bytes = file_path.stat().st_size
            session.avg_snr_db = metadata.get("avg_snr_db")
            session.samples_received = len(iq_data)

            # Calculate actual usage duration
            if session.start_time:
                actual_duration_minutes = (session.end_time - session.start_time).total_seconds() / 60.0
            else:
                actual_duration_minutes = duration_seconds / 60.0

            # Update SDR usage tracking (FR-008, FR-014, FR-066) - CRITICAL!
            # Handle both KiwiSDR and WebSDR sources
            sdr_source = None
            if hasattr(session, 'kiwisdr_source') and session.kiwisdr_source:
                sdr_source = session.kiwisdr_source
            elif hasattr(session, 'websdr_source') and session.websdr_source:
                sdr_source = session.websdr_source

            if sdr_source:
                await self.sdr_manager.update_usage(sdr_source, actual_duration_minutes)

                # Log remaining time based on type
                from ..models import WebSDRSource
                if isinstance(sdr_source, WebSDRSource):
                    if sdr_source.daily_limit_minutes:
                        remaining = sdr_source.daily_limit_minutes - sdr_source.daily_usage_minutes
                        logger.info(
                            f"WebSDR {sdr_source.url}: {actual_duration_minutes:.1f} min used, "
                            f"{remaining:.1f} min remaining today"
                        )
                    else:
                        logger.info(
                            f"WebSDR {sdr_source.url}: {actual_duration_minutes:.1f} min used (unlimited)"
                        )
                else:
                    logger.info(
                        f"KiwiSDR {sdr_source.url}: {actual_duration_minutes:.1f} min used, "
                        f"{sdr_source.remaining_daily_minutes:.1f} min remaining today"
                    )

            db.commit()
            logger.info(f"Recording {session_id} completed: {file_path}")

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            session.status = "failed"
            session.error_message = str(e)

            # Still track usage even on failure (FR-008, FR-066)
            if session.start_time:
                # Use timezone-aware datetime to match session.start_time
                actual_duration_minutes = (datetime.now(timezone.utc) - session.start_time).total_seconds() / 60.0

                # Handle both KiwiSDR and WebSDR sources
                sdr_source = None
                if hasattr(session, 'kiwisdr_source') and session.kiwisdr_source:
                    sdr_source = session.kiwisdr_source
                elif hasattr(session, 'websdr_source') and session.websdr_source:
                    sdr_source = session.websdr_source

                if sdr_source:
                    await self.sdr_manager.update_usage(sdr_source, actual_duration_minutes)
                    logger.info(f"Tracked {actual_duration_minutes:.1f} min usage despite failure")

            db.commit()
        finally:
            # Cleanup clients
            if session_id in self.kiwi_clients:
                self.kiwi_clients[session_id].disconnect()
                del self.kiwi_clients[session_id]

            if session_id in self.websdr_clients:
                await self.websdr_clients[session_id].disconnect()
                del self.websdr_clients[session_id]

            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Close database session
            db.close()

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
        # Convert UUID to string before slicing
        session_id_str = str(session.session_id)
        filename = (
            f"{int(session.frequency_khz)}khz"
            f"_{timestamp.strftime('%H%M%S')}"
            f"{location}"
            f"_{session_id_str[:8]}"
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

    async def _upload_to_tigris(
        self,
        file_path: Path,
        session: RecordingSession,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Upload recording to Tigris S3.

        Args:
            file_path: Local file path
            session: Recording session
            metadata: Recording metadata

        Returns:
            Tigris object key if successful, None otherwise
        """
        try:
            # Generate S3 key matching local path structure
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

            # S3 key structure: recordings/YYYY/MM/DD/band/filename
            s3_key = f"recordings/{date_path}/{band_path}/{file_path.name}"

            # Upload with metadata
            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.tigris_bucket,
                    Key=s3_key,
                    Body=f,
                    Metadata={
                        'session_id': str(session.session_id),
                        'frequency_khz': str(session.frequency_khz),
                        'bandwidth_khz': str(session.bandwidth_khz),
                        'sample_rate': str(session.sample_rate),
                        'grid_square': session.kiwisdr_source.grid_square if session.kiwisdr_source else '',
                        'gps_locked': str(session.gps_locked),
                    }
                )

            logger.info(f"Uploaded recording to Tigris: {s3_key}")

            # Optionally delete local file to save space (keep for now during testing)
            # file_path.unlink()

            return s3_key

        except ClientError as e:
            logger.error(f"Failed to upload to Tigris: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to Tigris: {e}")
            return None

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
            session.end_time = datetime.now(timezone.utc)
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