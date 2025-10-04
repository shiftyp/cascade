"""KiwiSDR connection manager.

Implements T025: KiwiClient connection manager (FR-001, FR-008, FR-009).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import threading
import queue

import numpy as np

# Use the real kiwiclient library
try:
    from kiwiclient import KiwiSDRStream, KiwiWorker
    KIWICLIENT_AVAILABLE = True
except ImportError:
    logger.warning("kiwiclient library not available, using fallback WebSocket implementation")
    from websocket import WebSocket, WebSocketTimeoutException
    KIWICLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class KiwiClient:
    """Manages connection to a KiwiSDR receiver."""

    def __init__(self, url: str, timeout: int = 30):
        """Initialize KiwiSDR client.

        Args:
            url: KiwiSDR URL (format: "host:port" or "ws://host:port")
            timeout: Connection timeout in seconds
        """
        self.url = self._normalize_url(url)
        self.timeout = timeout
        self.connected = False
        self.session_id: Optional[str] = None

        # Connection parameters
        self.frequency_khz: Optional[float] = None
        self.mode: str = "iq"
        self.bandwidth_khz: float = 12.0
        self.sample_rate: int = 12000

        # Usage tracking (FR-008)
        self.connection_start: Optional[datetime] = None
        self.total_usage_seconds: float = 0

        # Audio buffer
        self.audio_buffer = bytearray()
        self.samples_received = 0

        # Use real kiwiclient if available
        if KIWICLIENT_AVAILABLE:
            self.kiwi_stream: Optional[KiwiSDRStream] = None
            self.audio_queue = queue.Queue()
            self.worker_thread: Optional[threading.Thread] = None
        else:
            self.ws: Optional[WebSocket] = None

    def _normalize_url(self, url: str) -> str:
        """Normalize KiwiSDR URL to WebSocket format."""
        if not url.startswith(("ws://", "wss://")):
            # Parse as host:port
            if "://" not in url:
                url = f"ws://{url}"
            else:
                url = url.replace("http://", "ws://").replace("https://", "wss://")

        parsed = urlparse(url)
        if not parsed.port:
            # Default KiwiSDR port
            url = f"{url}:8073"

        return f"{url}/kiwi"

    async def connect(
        self,
        frequency_khz: float,
        mode: str = "iq",
        bandwidth_khz: float = 12.0,
        auth: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Connect to KiwiSDR (FR-001).

        Args:
            frequency_khz: Center frequency in kHz
            mode: Recording mode (iq, am, usb, lsb)
            bandwidth_khz: Bandwidth in kHz
            auth: Optional authentication credentials

        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to KiwiSDR: {self.url}")

            if KIWICLIENT_AVAILABLE:
                # Use real kiwiclient library
                # Parse host and port from URL
                parsed = urlparse(self.url)
                host = parsed.hostname or self.url.split(':')[0]
                port = parsed.port or 8073

                # Create KiwiSDRStream
                self.kiwi_stream = KiwiSDRStream(
                    host=host,
                    port=port,
                    tlimit=self.timeout,
                )

                # Configure for IQ mode
                self.kiwi_stream.set_mod('iq')
                self.kiwi_stream.set_freq(frequency_khz)
                self.kiwi_stream.set_srate(self.sample_rate)

                # Start worker thread for audio collection
                def audio_worker():
                    self.kiwi_stream.connect()
                    for samples in self.kiwi_stream:
                        self.audio_queue.put(samples)

                self.worker_thread = threading.Thread(target=audio_worker, daemon=True)
                self.worker_thread.start()

            else:
                # Fallback WebSocket implementation
                self.ws = WebSocket()
                self.ws.settimeout(self.timeout)
                self.ws.connect(self.url)

                # Send connection parameters
                params = {
                    "type": "connect",
                    "frequency": int(frequency_khz * 1000),  # Convert to Hz
                    "mode": mode,
                    "bandwidth": int(bandwidth_khz * 1000),
                    "compression": False,
                }

                if auth:
                    params["user"] = auth.get("username", "")
                params["password"] = auth.get("password", "")

            self.ws.send_json(params)

            # Wait for connection acknowledgment
            response = self.ws.recv_json(timeout=self.timeout)

            if response.get("status") == "connected":
                self.connected = True
                self.frequency_khz = frequency_khz
                self.mode = mode
                self.bandwidth_khz = bandwidth_khz
                self.connection_start = datetime.utcnow()
                self.session_id = response.get("session_id")

                logger.info(f"Connected to {self.url} at {frequency_khz} kHz")
                return True
            else:
                logger.error(f"Connection failed: {response.get('error')}")
                return False

        except WebSocketTimeoutException:
            logger.error(f"Connection timeout to {self.url}")
            raise TimeoutError(f"Connection timeout to {self.url}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.disconnect()
            raise

    def disconnect(self):
        """Disconnect from KiwiSDR (FR-009)."""
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self.connected = False
        self.ws = None

        # Update usage tracking
        if self.connection_start:
            duration = (datetime.utcnow() - self.connection_start).total_seconds()
            self.total_usage_seconds += duration
            self.connection_start = None

        logger.info(f"Disconnected from {self.url}")

    async def start_recording(
        self,
        duration_seconds: int,
        callback=None,
        max_session_minutes: int = 30
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start recording IQ data with session limit enforcement (FR-014).

        Args:
            duration_seconds: Recording duration
            callback: Optional callback for streaming data
            max_session_minutes: Maximum session length (default 30 min per FR-014)

        Returns:
            Tuple of (IQ samples array, metadata dict)

        Raises:
            ValueError: If duration exceeds session limit
        """
        if not self.connected:
            raise RuntimeError("Not connected to KiwiSDR")

        # Enforce session limit (FR-014: 30-minute max per session)
        duration_minutes = duration_seconds / 60.0
        if duration_minutes > max_session_minutes:
            raise ValueError(
                f"Recording duration {duration_minutes:.1f} min exceeds "
                f"session limit of {max_session_minutes} min (FR-014). "
                f"Split into multiple shorter sessions."
            )

        logger.info(f"Starting {duration_seconds}s recording at {self.frequency_khz} kHz")

        # Send start command
        self.ws.send_json({
            "type": "start",
            "duration": duration_seconds,
        })

        # Collect samples
        samples = []
        metadata = {
            "start_time": datetime.utcnow(),
            "frequency_khz": self.frequency_khz,
            "bandwidth_khz": self.bandwidth_khz,
            "sample_rate": self.sample_rate,
            "mode": self.mode,
        }

        start_time = asyncio.get_event_loop().time()
        timeout = duration_seconds + 10  # Extra time for buffering

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Receive IQ data chunk
                data = self.ws.recv()

                if isinstance(data, bytes):
                    # Parse IQ samples (assuming 16-bit signed integers)
                    iq_chunk = np.frombuffer(data, dtype=np.int16)
                    # Reshape to I/Q pairs
                    iq_chunk = iq_chunk.reshape(-1, 2)
                    samples.append(iq_chunk)

                    if callback:
                        callback(iq_chunk)

                    # Check if recording complete
                    total_samples = sum(len(s) for s in samples)
                    if total_samples >= duration_seconds * self.sample_rate:
                        break

            except WebSocketTimeoutException:
                logger.warning("Timeout receiving data")
                continue
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break

        # Combine all samples
        if samples:
            iq_data = np.vstack(samples)
        else:
            iq_data = np.array([])

        metadata["end_time"] = datetime.utcnow()
        metadata["samples_collected"] = len(iq_data)
        metadata["actual_duration"] = (
            metadata["end_time"] - metadata["start_time"]
        ).total_seconds()

        logger.info(f"Recording complete: {len(iq_data)} samples collected")
        return iq_data, metadata

    def get_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "connected": self.connected,
            "url": self.url,
            "frequency_khz": self.frequency_khz,
            "mode": self.mode,
            "bandwidth_khz": self.bandwidth_khz,
            "session_id": self.session_id,
            "usage_seconds": self.total_usage_seconds,
        }

    async def handle_reconnect(self, max_retries: int = 3) -> bool:
        """Handle reconnection after failure (FR-009).

        Args:
            max_retries: Maximum reconnection attempts

        Returns:
            True if reconnected successfully
        """
        for attempt in range(max_retries):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}")

            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

            try:
                if self.frequency_khz:
                    success = await self.connect(
                        self.frequency_khz,
                        self.mode,
                        self.bandwidth_khz
                    )
                    if success:
                        logger.info("Reconnected successfully")
                        return True
            except Exception as e:
                logger.warning(f"Reconnection failed: {e}")

        logger.error(f"Failed to reconnect after {max_retries} attempts")
        return False

    def validate_usage_limit(self, duration_minutes: float) -> bool:
        """Check if usage is within daily limits (FR-008).

        Args:
            duration_minutes: Planned recording duration

        Returns:
            True if within limits
        """
        from ..config import config

        current_usage_minutes = self.total_usage_seconds / 60
        total_planned = current_usage_minutes + duration_minutes

        return total_planned <= config.KIWI_DAILY_LIMIT_MINUTES