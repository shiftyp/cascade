"""WebSDR client connection manager for hybrid collection strategy.

Implements T025b: WebSDR client connection manager (FR-001, FR-065, FR-066).
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4
from urllib.parse import urljoin

import httpx
import websockets

from ..config import config

logger = logging.getLogger(__name__)


class WebSDRClient:
    """Client for WebSDR receivers with institutional backing."""

    def __init__(self, websdr_url: str, session_id: Optional[str] = None):
        """Initialize WebSDR client.

        Args:
            websdr_url: Base URL of WebSDR receiver
            session_id: Optional session ID for tracking
        """
        self.websdr_url = websdr_url.rstrip('/')
        self.session_id = session_id or str(uuid4())
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # WebSDR state
        self.connected = False
        self.recording = False
        self.current_frequency = None
        self.current_mode = None
        self.bandwidth = None
        self.sample_rate = None

        # Session tracking
        self.start_time: Optional[datetime] = None
        self.bytes_received = 0
        self.connection_quality = 1.0

    async def discover_capabilities(self) -> Dict[str, Any]:
        """Discover WebSDR capabilities and configuration.

        Returns:
            WebSDR capabilities information
        """
        try:
            # Get WebSDR info page
            info_url = urljoin(self.websdr_url, "/~~info")
            response = await self.http_client.get(info_url)
            response.raise_for_status()

            # Parse WebSDR configuration
            config_data = await self._parse_websdr_config(response.text)

            # Get receiver status
            status_url = urljoin(self.websdr_url, "/~~status")
            status_response = await self.http_client.get(status_url)

            if status_response.status_code == 200:
                status_data = await self._parse_websdr_status(status_response.text)
                config_data.update(status_data)

            logger.info(f"Discovered WebSDR capabilities: {self.websdr_url}")
            return config_data

        except Exception as e:
            logger.error(f"Failed to discover WebSDR capabilities: {e}")
            return {}

    async def connect(
        self,
        frequency_khz: float,
        mode: str = "iq",
        bandwidth_khz: float = 12.0
    ) -> bool:
        """Connect to WebSDR and configure for IQ streaming.

        Args:
            frequency_khz: Frequency in kHz
            mode: Reception mode ("iq", "usb", "lsb", "am")
            bandwidth_khz: Bandwidth in kHz

        Returns:
            True if connection successful
        """
        try:
            if self.connected:
                await self.disconnect()

            # Validate frequency range
            capabilities = await self.discover_capabilities()
            if not self._validate_frequency(frequency_khz, capabilities):
                raise ValueError(f"Frequency {frequency_khz} kHz not supported")

            # Connect to WebSDR WebSocket
            ws_url = self._build_websocket_url(frequency_khz, mode, bandwidth_khz)

            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={"User-Agent": "CASCADE-Collector/1.0"},
                ping_interval=30,
                ping_timeout=10
            )

            # Configure receiver
            await self._configure_receiver(frequency_khz, mode, bandwidth_khz)

            # Wait for configuration confirmation
            await asyncio.sleep(1.0)

            self.connected = True
            self.current_frequency = frequency_khz
            self.current_mode = mode
            self.bandwidth = bandwidth_khz
            self.start_time = datetime.utcnow()

            logger.info(
                f"Connected to WebSDR: {self.websdr_url}, "
                f"freq: {frequency_khz} kHz, mode: {mode}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSDR {self.websdr_url}: {e}")
            return False

    async def start_iq_stream(self) -> bool:
        """Start IQ data streaming from WebSDR.

        Returns:
            True if streaming started successfully
        """
        if not self.connected or not self.websocket:
            logger.error("Not connected to WebSDR")
            return False

        try:
            # Send IQ streaming command
            command = {
                "type": "start_iq",
                "session_id": self.session_id,
                "frequency": self.current_frequency,
                "bandwidth": self.bandwidth,
                "format": "complex_float32"
            }

            await self.websocket.send(json.dumps(command))

            # Wait for acknowledgment
            response = await asyncio.wait_for(
                self.websocket.recv(), timeout=5.0
            )

            response_data = json.loads(response)
            if response_data.get("status") == "started":
                self.recording = True
                self.sample_rate = response_data.get("sample_rate", 12000)
                logger.info(f"Started IQ streaming: {self.sample_rate} Hz")
                return True
            else:
                logger.error(f"Failed to start IQ stream: {response_data}")
                return False

        except Exception as e:
            logger.error(f"Error starting IQ stream: {e}")
            return False

    async def read_iq_data(self, timeout: float = 5.0) -> Optional[bytes]:
        """Read IQ data from WebSDR stream.

        Args:
            timeout: Read timeout in seconds

        Returns:
            IQ data bytes or None
        """
        if not self.recording or not self.websocket:
            return None

        try:
            # Read from WebSocket with timeout
            data = await asyncio.wait_for(
                self.websocket.recv(), timeout=timeout
            )

            if isinstance(data, bytes):
                # Binary IQ data
                self.bytes_received += len(data)
                return data
            else:
                # JSON message (status, error, etc.)
                message = json.loads(data)
                if message.get("type") == "error":
                    logger.error(f"WebSDR error: {message.get('message')}")
                elif message.get("type") == "status":
                    # Update connection quality
                    self.connection_quality = message.get("quality", 1.0)

                return None

        except asyncio.TimeoutError:
            # Normal timeout - no data available
            return None
        except Exception as e:
            logger.error(f"Error reading IQ data: {e}")
            return None

    async def stop_recording(self) -> bool:
        """Stop IQ data recording.

        Returns:
            True if stopped successfully
        """
        if not self.recording:
            return True

        try:
            if self.websocket:
                # Send stop command
                command = {
                    "type": "stop_iq",
                    "session_id": self.session_id
                }
                await self.websocket.send(json.dumps(command))

                # Wait for confirmation
                response = await asyncio.wait_for(
                    self.websocket.recv(), timeout=5.0
                )

                response_data = json.loads(response)
                if response_data.get("status") == "stopped":
                    self.recording = False
                    logger.info("Stopped IQ recording")
                    return True

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

        self.recording = False
        return True

    async def disconnect(self):
        """Disconnect from WebSDR and track usage."""
        try:
            # Calculate session duration for usage tracking
            session_duration_minutes = 0
            if self.start_time:
                session_duration_minutes = (datetime.utcnow() - self.start_time).total_seconds() / 60.0

            if self.recording:
                await self.stop_recording()

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            if self.http_client:
                await self.http_client.aclose()

            self.connected = False

            duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
            logger.info(
                f"Disconnected from WebSDR: {self.websdr_url}, "
                f"duration: {duration:.1f}s, data: {self.bytes_received:,} bytes, "
                f"session_minutes: {session_duration_minutes:.1f}"
            )

            # Return session duration for caller to track usage
            return session_duration_minutes

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            return 0  # Return 0 minutes on error

    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information.

        Returns:
            Connection information dictionary
        """
        duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0

        return {
            "session_id": self.session_id,
            "websdr_url": self.websdr_url,
            "connected": self.connected,
            "recording": self.recording,
            "frequency_khz": self.current_frequency,
            "mode": self.current_mode,
            "bandwidth_khz": self.bandwidth,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration,
            "bytes_received": self.bytes_received,
            "connection_quality": self.connection_quality,
        }

    async def _parse_websdr_config(self, html_content: str) -> Dict[str, Any]:
        """Parse WebSDR configuration from HTML.

        Args:
            html_content: HTML content from info page

        Returns:
            Parsed configuration
        """
        # Basic WebSDR configuration parsing
        # In practice, would use proper HTML parsing
        config_data = {
            "type": "WebSDR",
            "url": self.websdr_url,
            "frequency_min_khz": 0,
            "frequency_max_khz": 30000,
            "max_users": 100,  # WebSDRs typically support more users
            "iq_capable": True,
            "institution_type": "UNIVERSITY",  # Default assumption
            "usage_policy": "RESEARCH_AGREEMENT",
            "daily_limit_minutes": 0,  # Many WebSDRs have no strict daily limits
            "discovered_at": datetime.utcnow().isoformat()
        }

        # Try to extract actual configuration
        if "frequency range" in html_content.lower():
            # Would parse frequency ranges from HTML
            pass

        if "university" in html_content.lower():
            config_data["institution_type"] = "UNIVERSITY"
        elif "research" in html_content.lower():
            config_data["institution_type"] = "RESEARCH_INSTITUTE"

        return config_data

    async def _parse_websdr_status(self, html_content: str) -> Dict[str, Any]:
        """Parse WebSDR status information.

        Args:
            html_content: HTML content from status page

        Returns:
            Parsed status
        """
        # Parse current users, system load, etc.
        return {
            "current_users": 0,  # Would parse from HTML
            "system_load": 0.5,
            "last_checked": datetime.utcnow().isoformat()
        }

    def _validate_frequency(self, frequency_khz: float, capabilities: Dict[str, Any]) -> bool:
        """Validate frequency is within WebSDR range.

        Args:
            frequency_khz: Frequency to validate
            capabilities: WebSDR capabilities

        Returns:
            True if frequency is valid
        """
        min_freq = capabilities.get("frequency_min_khz", 0)
        max_freq = capabilities.get("frequency_max_khz", 30000)

        return min_freq <= frequency_khz <= max_freq

    def _build_websocket_url(self, frequency_khz: float, mode: str, bandwidth_khz: float) -> str:
        """Build WebSocket URL for WebSDR connection.

        Args:
            frequency_khz: Frequency
            mode: Mode
            bandwidth_khz: Bandwidth

        Returns:
            WebSocket URL
        """
        # WebSDR WebSocket URL format (simplified)
        # Real implementation would follow specific WebSDR protocol
        base_ws_url = self.websdr_url.replace("http://", "ws://").replace("https://", "wss://")

        return f"{base_ws_url}/ws?freq={frequency_khz}&mode={mode}&bw={bandwidth_khz}&session={self.session_id}"

    async def _configure_receiver(self, frequency_khz: float, mode: str, bandwidth_khz: float):
        """Configure WebSDR receiver settings.

        Args:
            frequency_khz: Frequency
            mode: Mode
            bandwidth_khz: Bandwidth
        """
        if not self.websocket:
            return

        # Send configuration command
        config_command = {
            "type": "configure",
            "frequency": frequency_khz,
            "mode": mode,
            "bandwidth": bandwidth_khz,
            "gain": "auto",
            "format": "iq"
        }

        await self.websocket.send(json.dumps(config_command))

    async def get_receiver_status(self) -> Dict[str, Any]:
        """Get current receiver status from WebSDR.

        Returns:
            Status information
        """
        try:
            status_url = urljoin(self.websdr_url, "/~~status")
            response = await self.http_client.get(status_url)

            if response.status_code == 200:
                return await self._parse_websdr_status(response.text)
            else:
                return {"error": f"Status request failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Failed to get receiver status: {e}")
            return {"error": str(e)}

    def estimate_session_limit(self) -> int:
        """Estimate maximum session duration for this WebSDR.

        Returns:
            Maximum session duration in minutes
        """
        # WebSDRs often have more generous limits than KiwiSDRs
        # University/research WebSDRs may have no strict time limits
        # but we should be considerate

        # Conservative estimate for research usage
        return 180  # 3 hours max session

    def get_usage_policy(self) -> Dict[str, Any]:
        """Get usage policy information for this WebSDR.

        Returns:
            Usage policy details
        """
        return {
            "type": "WebSDR",
            "usage_policy": "RESEARCH_AGREEMENT",
            "daily_limit_minutes": 0,  # No strict daily limit
            "session_limit_minutes": self.estimate_session_limit(),
            "peak_hours_restriction": False,
            "concurrent_users_limit": 100,
            "research_friendly": True,
            "contact_required": False,  # Usually no prior contact needed
            "notes": "WebSDR typically allows longer sessions for research"
        }


class WebSDRManager:
    """Manages multiple WebSDR connections for hybrid collection."""

    def __init__(self):
        """Initialize WebSDR manager."""
        self.active_clients: Dict[str, WebSDRClient] = {}
        self.websdr_registry: List[Dict[str, Any]] = []

    async def discover_websdrs(self) -> List[Dict[str, Any]]:
        """Discover available WebSDR receivers.

        Returns:
            List of WebSDR information
        """
        # Known WebSDR URLs (would be expanded with actual discovery)
        known_websdrs = [
            "http://websdr.ewi.utwente.nl:8901",  # University of Twente
            "http://kiwisdr.dayton.net:8073",     # Hybrid KiwiSDR/WebSDR
            "http://websdr.pi4tht.nl",            # Netherlands
            "http://websdr.marcus.org.uk",        # UK
            "http://websdr.suws.org.uk",          # UK Amateur Radio
        ]

        discovered = []
        for url in known_websdrs:
            try:
                client = WebSDRClient(url)
                capabilities = await client.discover_capabilities()

                if capabilities:
                    websdr_info = {
                        "url": url,
                        "type": "WebSDR",
                        "capabilities": capabilities,
                        "usage_policy": client.get_usage_policy(),
                        "discovered_at": datetime.utcnow().isoformat()
                    }
                    discovered.append(websdr_info)

                await client.disconnect()

            except Exception as e:
                logger.debug(f"WebSDR {url} not available: {e}")

        self.websdr_registry = discovered
        logger.info(f"Discovered {len(discovered)} available WebSDRs")
        return discovered

    async def connect_to_websdr(
        self,
        websdr_url: str,
        frequency_khz: float,
        mode: str = "iq",
        bandwidth_khz: float = 12.0
    ) -> Optional[str]:
        """Connect to a specific WebSDR.

        Args:
            websdr_url: WebSDR URL
            frequency_khz: Frequency
            mode: Mode
            bandwidth_khz: Bandwidth

        Returns:
            Session ID if successful, None if failed
        """
        try:
            session_id = str(uuid4())
            client = WebSDRClient(websdr_url, session_id)

            success = await client.connect(frequency_khz, mode, bandwidth_khz)

            if success:
                self.active_clients[session_id] = client
                logger.info(f"WebSDR connection established: {session_id}")
                return session_id
            else:
                await client.disconnect()
                return None

        except Exception as e:
            logger.error(f"Failed to connect to WebSDR {websdr_url}: {e}")
            return None

    async def start_recording(self, session_id: str) -> bool:
        """Start recording from WebSDR session.

        Args:
            session_id: WebSDR session ID

        Returns:
            True if recording started
        """
        client = self.active_clients.get(session_id)
        if not client:
            logger.error(f"WebSDR session not found: {session_id}")
            return False

        return await client.start_iq_stream()

    async def read_iq_data(self, session_id: str, timeout: float = 5.0) -> Optional[bytes]:
        """Read IQ data from WebSDR session.

        Args:
            session_id: Session ID
            timeout: Read timeout

        Returns:
            IQ data or None
        """
        client = self.active_clients.get(session_id)
        if not client:
            return None

        return await client.read_iq_data(timeout)

    async def stop_recording(self, session_id: str) -> bool:
        """Stop recording from WebSDR session.

        Args:
            session_id: Session ID

        Returns:
            True if stopped successfully
        """
        client = self.active_clients.get(session_id)
        if not client:
            return False

        return await client.stop_recording()

    async def disconnect_session(self, session_id: str):
        """Disconnect WebSDR session.

        Args:
            session_id: Session ID to disconnect
        """
        client = self.active_clients.get(session_id)
        if client:
            await client.disconnect()
            del self.active_clients[session_id]

    async def disconnect_all(self):
        """Disconnect all active WebSDR sessions."""
        for session_id in list(self.active_clients.keys()):
            await self.disconnect_session(session_id)

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about active WebSDR sessions.

        Returns:
            List of session information
        """
        sessions = []
        for session_id, client in self.active_clients.items():
            sessions.append(client.get_connection_info())

        return sessions

    def select_best_websdr(
        self,
        frequency_khz: float,
        duration_minutes: int,
        prefer_institution: str = "UNIVERSITY"
    ) -> Optional[str]:
        """Select best WebSDR for given requirements.

        Args:
            frequency_khz: Required frequency
            duration_minutes: Expected session duration
            prefer_institution: Preferred institution type

        Returns:
            Best WebSDR URL or None
        """
        suitable_websdrs = []

        for websdr in self.websdr_registry:
            capabilities = websdr.get("capabilities", {})
            policy = websdr.get("usage_policy", {})

            # Check frequency support
            min_freq = capabilities.get("frequency_min_khz", 0)
            max_freq = capabilities.get("frequency_max_khz", 30000)

            if not (min_freq <= frequency_khz <= max_freq):
                continue

            # Check session duration compatibility
            session_limit = policy.get("session_limit_minutes", 180)
            if duration_minutes > session_limit:
                continue

            # Score based on preferences
            score = 100

            # Prefer specified institution type
            if capabilities.get("institution_type") == prefer_institution:
                score += 20

            # Prefer research-friendly receivers
            if policy.get("research_friendly"):
                score += 15

            # Prefer less loaded receivers
            current_users = capabilities.get("current_users", 0)
            max_users = capabilities.get("max_users", 100)
            load_factor = current_users / max_users if max_users > 0 else 0
            score -= load_factor * 30

            suitable_websdrs.append((websdr["url"], score))

        if suitable_websdrs:
            # Sort by score (highest first)
            suitable_websdrs.sort(key=lambda x: x[1], reverse=True)
            return suitable_websdrs[0][0]

        return None


# Utility functions for testing
async def test_websdr_connection(websdr_url: str) -> Dict[str, Any]:
    """Test WebSDR connection (utility function).

    Args:
        websdr_url: WebSDR URL to test

    Returns:
        Test results
    """
    client = WebSDRClient(websdr_url)

    try:
        # Test discovery
        capabilities = await client.discover_capabilities()

        if not capabilities:
            return {"success": False, "error": "Failed to discover capabilities"}

        # Test connection
        success = await client.connect(14080, "iq", 12.0)

        if success:
            # Test brief recording
            recording_started = await client.start_iq_stream()

            if recording_started:
                # Read some data
                data = await client.read_iq_data(timeout=2.0)
                await client.stop_recording()

                return {
                    "success": True,
                    "capabilities": capabilities,
                    "data_received": len(data) if data else 0,
                    "connection_info": client.get_connection_info()
                }
            else:
                return {"success": False, "error": "Failed to start recording"}
        else:
            return {"success": False, "error": "Failed to connect"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        await client.disconnect()