"""Recording pipeline for KiwiSDR data collection."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import numpy as np
import wave
import json
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class RecordingConfig:
    """Configuration for recording session."""
    frequency: float
    sample_rate: int = 12000
    duration_seconds: int = 300  # 5 minutes default
    output_dir: str = "/tmp/recordings"
    format: str = "wav"
    compress: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    buffer_size: int = 4096
    channels: int = 2  # IQ data


@dataclass
class RecordingStatus:
    """Status of a recording session."""
    session_id: str
    start_time: datetime
    frequency: float
    sample_rate: int
    samples_recorded: int = 0
    bytes_written: int = 0
    state: str = "idle"  # idle, recording, paused, completed, error
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    duration_recorded: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Main recorder for handling KiwiSDR data streams."""

    def __init__(self, config: Optional[RecordingConfig] = None):
        """Initialize recorder.

        Args:
            config: Recording configuration
        """
        self.config = config or RecordingConfig()
        self._sessions: Dict[str, RecordingStatus] = {}
        self._active_session: Optional[str] = None
        self._recording_task: Optional[asyncio.Task] = None
        self._buffer: List[np.ndarray] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_stop': [],
            'on_error': [],
            'on_data': []
        }
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    async def start_recording(
        self,
        kiwi_client,
        frequency: Optional[float] = None,
        duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new recording session.

        Args:
            kiwi_client: Connected KiwiSDR client
            frequency: Center frequency (Hz)
            duration: Recording duration (seconds)
            metadata: Additional metadata

        Returns:
            Session ID

        Raises:
            RuntimeError: If already recording
        """
        if self._active_session:
            raise RuntimeError(f"Already recording session {self._active_session}")

        # Create new session
        session_id = str(uuid4())[:8]
        frequency = frequency or self.config.frequency
        duration = duration or self.config.duration_seconds

        # Generate output path
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_{frequency}Hz_{session_id}.wav"
        output_path = Path(self.config.output_dir) / filename

        # Create session status
        status = RecordingStatus(
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
            frequency=frequency,
            sample_rate=self.config.sample_rate,
            state="recording",
            output_path=str(output_path),
            metadata=metadata or {}
        )

        self._sessions[session_id] = status
        self._active_session = session_id

        # Start recording task
        self._recording_task = asyncio.create_task(
            self._record_stream(kiwi_client, status, duration)
        )

        # Trigger callbacks
        await self._trigger_callbacks('on_start', status)

        logger.info(f"Started recording session {session_id} at {frequency} Hz for {duration} seconds")

        return session_id

    async def _record_stream(
        self,
        kiwi_client,
        status: RecordingStatus,
        duration: int
    ) -> None:
        """Record data stream from KiwiSDR.

        Args:
            kiwi_client: Connected KiwiSDR client
            status: Recording status object
            duration: Recording duration in seconds
        """
        try:
            # Open output file
            with wave.open(status.output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit samples
                wav_file.setframerate(self.config.sample_rate)

                start_time = time.time()
                samples_needed = duration * self.config.sample_rate

                while status.samples_recorded < samples_needed:
                    if status.state != "recording":
                        break

                    # Get data from client (simulated for now)
                    data = await self._get_data_from_client(kiwi_client)
                    if data is None:
                        break

                    # Write to file
                    wav_file.writeframes(data.tobytes())

                    # Update status
                    status.samples_recorded += len(data) // self.config.channels
                    status.bytes_written += len(data.tobytes())
                    status.duration_recorded = time.time() - start_time

                    # Trigger data callback
                    await self._trigger_callbacks('on_data', status, data)

                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.01)

            # Mark as completed
            status.state = "completed"
            logger.info(f"Recording session {status.session_id} completed: "
                       f"{status.samples_recorded} samples, "
                       f"{status.duration_recorded:.1f} seconds")

        except Exception as e:
            status.state = "error"
            status.error_message = str(e)
            logger.error(f"Recording error in session {status.session_id}: {e}")
            await self._trigger_callbacks('on_error', status, e)

        finally:
            self._active_session = None
            await self._trigger_callbacks('on_stop', status)

    async def _get_data_from_client(self, kiwi_client) -> Optional[np.ndarray]:
        """Get data from KiwiSDR client.

        Args:
            kiwi_client: Connected KiwiSDR client

        Returns:
            NumPy array of IQ samples or None if no data
        """
        # Simulate getting data - in reality would interface with kiwiclient
        # Generate test IQ data
        samples = self.config.buffer_size // (2 * 2)  # 2 channels, 2 bytes per sample
        i_data = np.random.randn(samples) * 0.1
        q_data = np.random.randn(samples) * 0.1

        # Interleave IQ data
        iq_data = np.empty(samples * 2, dtype=np.float32)
        iq_data[0::2] = i_data
        iq_data[1::2] = q_data

        # Convert to 16-bit integer
        iq_int = (iq_data * 32767).astype(np.int16)

        return iq_int

    async def stop_recording(self, session_id: Optional[str] = None) -> RecordingStatus:
        """Stop recording session.

        Args:
            session_id: Session ID to stop (or current if None)

        Returns:
            Final recording status

        Raises:
            ValueError: If session not found
        """
        session_id = session_id or self._active_session
        if not session_id or session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        status = self._sessions[session_id]

        if status.state == "recording":
            status.state = "stopping"
            logger.info(f"Stopping recording session {session_id}")

            # Cancel recording task
            if self._recording_task and not self._recording_task.done():
                self._recording_task.cancel()
                try:
                    await self._recording_task
                except asyncio.CancelledError:
                    pass

        return status

    def get_status(self, session_id: Optional[str] = None) -> Optional[RecordingStatus]:
        """Get status of recording session.

        Args:
            session_id: Session ID (or current if None)

        Returns:
            Recording status or None if not found
        """
        session_id = session_id or self._active_session
        return self._sessions.get(session_id) if session_id else None

    def get_all_sessions(self) -> Dict[str, RecordingStatus]:
        """Get all recording sessions.

        Returns:
            Dictionary of session_id -> RecordingStatus
        """
        return self._sessions.copy()

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback.

        Args:
            event: Event name (on_start, on_stop, on_error, on_data)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _trigger_callbacks(self, event: str, *args) -> None:
        """Trigger event callbacks.

        Args:
            event: Event name
            *args: Arguments to pass to callbacks
        """
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")

    def save_metadata(self, session_id: str, metadata_path: Optional[str] = None) -> str:
        """Save session metadata to JSON file.

        Args:
            session_id: Session ID
            metadata_path: Output path (auto-generated if None)

        Returns:
            Path to metadata file

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        status = self._sessions[session_id]

        if not metadata_path:
            base_path = Path(status.output_path).with_suffix('.json')
            metadata_path = str(base_path)

        metadata = {
            'session_id': status.session_id,
            'start_time': status.start_time.isoformat(),
            'frequency': status.frequency,
            'sample_rate': status.sample_rate,
            'samples_recorded': status.samples_recorded,
            'bytes_written': status.bytes_written,
            'duration_recorded': status.duration_recorded,
            'state': status.state,
            'output_path': status.output_path,
            'metadata': status.metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata for session {session_id} to {metadata_path}")

        return metadata_path

    async def cleanup(self) -> None:
        """Clean up recorder resources."""
        # Stop any active recording
        if self._active_session:
            await self.stop_recording()

        # Clear sessions
        self._sessions.clear()
        self._buffer.clear()

        logger.info("Recorder cleanup completed")