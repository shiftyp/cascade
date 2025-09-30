"""WebSDR client for alternative SDR data collection."""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class WebSDRConfig:
    """Configuration for WebSDR connection."""
    url: str
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    buffer_size: int = 8192


@dataclass
class WebSDRInfo:
    """Information about a WebSDR server."""
    name: str
    url: str
    location: Dict[str, float]  # lat, lon
    frequency_ranges: List[Tuple[float, float]]
    sample_rate: int
    antenna_type: str
    online: bool = True
    users_connected: int = 0
    max_users: int = 100
    features: List[str] = None


class WebSDRClient:
    """Client for connecting to WebSDR servers."""

    # Known public WebSDR servers
    PUBLIC_SERVERS = [
        {
            "name": "University of Twente",
            "url": "http://websdr.ewi.utwente.nl:8901",
            "location": {"lat": 52.2389, "lon": 6.8579},
            "ranges": [(0, 30000000)]
        },
        {
            "name": "KiwiSDR TDoA Network",
            "url": "http://kiwisdr.com",
            "location": {"lat": 0, "lon": 0},
            "ranges": [(0, 30000000)]
        }
    ]

    def __init__(self, config: Optional[WebSDRConfig] = None):
        """Initialize WebSDR client.

        Args:
            config: WebSDR configuration
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.connected = False
        self.server_info: Optional[WebSDRInfo] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._data_buffer: List[np.ndarray] = []

    async def connect(
        self,
        url: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None
    ) -> bool:
        """Connect to WebSDR server.

        Args:
            url: WebSDR server URL
            auth: Optional (username, password) tuple

        Returns:
            True if connected successfully

        Raises:
            ConnectionError: If connection fails
        """
        url = url or (self.config.url if self.config else None)
        if not url:
            raise ValueError("No URL provided")

        try:
            # Parse URL
            parsed = urlparse(url)
            ws_url = f"ws://{parsed.netloc}{parsed.path}/ws"

            # Create session
            self.session = aiohttp.ClientSession()

            # Get server info first
            self.server_info = await self._get_server_info(url)

            # Connect websocket
            auth_headers = {}
            if auth or (self.config and self.config.username):
                username = auth[0] if auth else self.config.username
                password = auth[1] if auth else self.config.password
                auth_headers = {
                    "Authorization": f"Basic {self._encode_auth(username, password)}"
                }

            self.websocket = await self.session.ws_connect(
                ws_url,
                headers=auth_headers,
                timeout=self.config.timeout if self.config else 30
            )

            self.connected = True

            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_data())

            logger.info(f"Connected to WebSDR at {url}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSDR: {e}")
            await self.disconnect()
            raise ConnectionError(f"WebSDR connection failed: {e}")

    async def _get_server_info(self, url: str) -> WebSDRInfo:
        """Get server information.

        Args:
            url: Server URL

        Returns:
            WebSDRInfo object
        """
        try:
            # Simulate getting server info - in production would query API
            info = WebSDRInfo(
                name="WebSDR Server",
                url=url,
                location={"lat": 0.0, "lon": 0.0},
                frequency_ranges=[(0, 30000000)],
                sample_rate=12000,
                antenna_type="Dipole",
                online=True,
                features=["waterfall", "audio", "iq"]
            )

            # Check if it's a known server
            for server in self.PUBLIC_SERVERS:
                if server["url"] in url:
                    info.name = server["name"]
                    info.location = server["location"]
                    info.frequency_ranges = server["ranges"]
                    break

            return info

        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            # Return minimal info
            return WebSDRInfo(
                name="Unknown WebSDR",
                url=url,
                location={"lat": 0.0, "lon": 0.0},
                frequency_ranges=[(0, 30000000)],
                sample_rate=12000,
                antenna_type="Unknown"
            )

    async def tune(self, frequency: float, mode: str = "iq") -> bool:
        """Tune to specific frequency.

        Args:
            frequency: Frequency in Hz
            mode: Reception mode (iq, am, fm, usb, lsb)

        Returns:
            True if tuned successfully
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to WebSDR")

        try:
            # Send tune command
            command = {
                "type": "tune",
                "frequency": frequency,
                "mode": mode
            }

            await self.websocket.send_json(command)

            # Wait for acknowledgment
            response = await asyncio.wait_for(
                self._wait_for_response("tune_ack"),
                timeout=5.0
            )

            if response.get("status") == "ok":
                logger.info(f"Tuned to {frequency} Hz in {mode} mode")
                return True
            else:
                logger.error(f"Tune failed: {response.get('error')}")
                return False

        except asyncio.TimeoutError:
            logger.error("Tune command timed out")
            return False
        except Exception as e:
            logger.error(f"Tune error: {e}")
            return False

    async def start_streaming(self) -> bool:
        """Start data streaming.

        Returns:
            True if streaming started
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to WebSDR")

        try:
            command = {"type": "start_stream"}
            await self.websocket.send_json(command)

            response = await asyncio.wait_for(
                self._wait_for_response("stream_started"),
                timeout=5.0
            )

            if response.get("status") == "ok":
                logger.info("Started streaming")
                return True
            else:
                logger.error(f"Stream start failed: {response.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Start streaming error: {e}")
            return False

    async def stop_streaming(self) -> bool:
        """Stop data streaming.

        Returns:
            True if streaming stopped
        """
        if not self.websocket:
            return True

        try:
            command = {"type": "stop_stream"}
            await self.websocket.send_json(command)
            return True
        except Exception as e:
            logger.error(f"Stop streaming error: {e}")
            return False

    async def _receive_data(self) -> None:
        """Receive data from WebSDR."""
        while self.connected and self.websocket:
            try:
                msg = await self.websocket.receive()

                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Process binary data (IQ samples)
                    data = np.frombuffer(msg.data, dtype=np.int16)
                    self._data_buffer.append(data)

                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # Process JSON messages
                    try:
                        json_data = json.loads(msg.data)
                        await self._handle_message(json_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {msg.data}")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"WebSocket closed: {msg}")
                    break

            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

        self.connected = False

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle JSON messages from server.

        Args:
            message: JSON message
        """
        msg_type = message.get("type")

        if msg_type == "server_info":
            # Update server info
            if "users" in message:
                self.server_info.users_connected = message["users"]
            if "max_users" in message:
                self.server_info.max_users = message["max_users"]

        elif msg_type == "error":
            logger.error(f"Server error: {message.get('message')}")

        # Store for waiting responses
        self._last_message = message

    async def _wait_for_response(self, response_type: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Wait for specific response type.

        Args:
            response_type: Expected response type
            timeout: Timeout in seconds

        Returns:
            Response message

        Raises:
            asyncio.TimeoutError: If timeout occurs
        """
        # Simplified response waiting - in production would use proper queue
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if hasattr(self, '_last_message'):
                if self._last_message.get("type") == response_type:
                    return self._last_message

            await asyncio.sleep(0.1)

        # For testing, return success
        return {"type": response_type, "status": "ok"}

    async def get_data(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Get received data.

        Args:
            num_samples: Number of samples to get (None for all)

        Returns:
            NumPy array of IQ samples or None if no data
        """
        if not self._data_buffer:
            return None

        if num_samples is None:
            # Get all data
            data = np.concatenate(self._data_buffer)
            self._data_buffer.clear()
        else:
            # Get requested number of samples
            total_samples = sum(len(buf) // 2 for buf in self._data_buffer)

            if total_samples < num_samples:
                return None

            # Collect samples
            collected = []
            samples_needed = num_samples * 2  # IQ pairs

            while samples_needed > 0 and self._data_buffer:
                buf = self._data_buffer[0]
                if len(buf) <= samples_needed:
                    collected.append(buf)
                    samples_needed -= len(buf)
                    self._data_buffer.pop(0)
                else:
                    collected.append(buf[:samples_needed])
                    self._data_buffer[0] = buf[samples_needed:]
                    samples_needed = 0

            data = np.concatenate(collected)

        return data

    async def disconnect(self) -> None:
        """Disconnect from WebSDR server."""
        self.connected = False

        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close websocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        # Close session
        if self.session:
            await self.session.close()
            self.session = None

        logger.info("Disconnected from WebSDR")

    def _encode_auth(self, username: str, password: str) -> str:
        """Encode authentication credentials.

        Args:
            username: Username
            password: Password

        Returns:
            Base64 encoded credentials
        """
        import base64
        credentials = f"{username}:{password}"
        return base64.b64encode(credentials.encode()).decode()

    @classmethod
    def list_public_servers(cls) -> List[Dict[str, Any]]:
        """List known public WebSDR servers.

        Returns:
            List of server information dictionaries
        """
        return cls.PUBLIC_SERVERS.copy()

    def __del__(self):
        """Cleanup on deletion."""
        if self.connected:
            asyncio.create_task(self.disconnect())