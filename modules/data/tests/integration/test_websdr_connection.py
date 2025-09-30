"""
Integration tests for WebSDR connection and session management
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid
import json

# These imports will fail initially (TDD approach)
from cascade_collector.collectors.websdr_client import WebSDRClient
from cascade_collector.models.sdr_source import SDRSource


class TestWebSDRConnection:
    """Test WebSDR connection, authentication, and data streaming"""

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        return ws

    @pytest.fixture
    def websdr_config(self):
        """WebSDR configuration"""
        return {
            'url': 'websdr.ewi.utwente.nl',
            'port': 8901,
            'max_bandwidth_khz': 192,  # WebSDR supports wider bandwidth
            'session_timeout': 3600,  # 1 hour sessions
            'reconnect_attempts': 3,
        }

    @pytest.mark.asyncio
    async def test_websdr_initial_connection(self, mock_websocket, websdr_config):
        """Test initial WebSDR connection and handshake"""
        client = WebSDRClient(**websdr_config)

        with patch('websockets.connect', return_value=mock_websocket):
            # Mock initial server response
            mock_websocket.recv.side_effect = [
                json.dumps({
                    'type': 'welcome',
                    'server': 'WebSDR',
                    'version': '11.2',
                    'bandwidth': 192000,
                    'users': 45
                }),
                json.dumps({
                    'type': 'ready',
                    'session_id': 'ws_123456'
                })
            ]

            connected = await client.connect()

            assert connected
            assert client.session_id == 'ws_123456'
            assert client.bandwidth == 192000
            mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_websdr_frequency_tuning(self, mock_websocket, websdr_config):
        """Test WebSDR frequency tuning and mode selection"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True

        # Tune to frequency
        await client.tune(
            frequency_khz=14074,
            mode='usb',
            bandwidth_hz=12000
        )

        # Verify tuning command sent
        mock_websocket.send.assert_called()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data['cmd'] == 'tune'
        assert sent_data['frequency'] == 14074000
        assert sent_data['mode'] == 'usb'

    @pytest.mark.asyncio
    async def test_websdr_audio_stream_start(self, mock_websocket, websdr_config):
        """Test starting audio/IQ stream from WebSDR"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True

        # Start streaming
        await client.start_stream(
            stream_type='iq',
            sample_rate=12000
        )

        # Verify stream start command
        mock_websocket.send.assert_called()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data['cmd'] == 'startstream'
        assert sent_data['type'] == 'iq'
        assert sent_data['rate'] == 12000

    @pytest.mark.asyncio
    async def test_websdr_data_reception(self, mock_websocket, websdr_config):
        """Test receiving and processing WebSDR data stream"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True
        client.streaming = True

        # Mock incoming data packets
        mock_websocket.recv.side_effect = [
            b'AUDIO:' + bytes(range(256)),  # Audio data packet
            b'AUDIO:' + bytes(range(256)),
            json.dumps({'type': 'keepalive'}).encode(),  # Keepalive
            b'AUDIO:' + bytes(range(256)),
        ]

        # Collect data
        data_chunks = []
        async def collect_data():
            async for chunk in client.stream_data():
                data_chunks.append(chunk)
                if len(data_chunks) >= 3:
                    break

        await collect_data()

        assert len(data_chunks) == 3
        assert all(len(chunk) == 256 for chunk in data_chunks)

    @pytest.mark.asyncio
    async def test_websdr_session_keepalive(self, mock_websocket, websdr_config):
        """Test WebSDR session keepalive mechanism"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True
        client.last_activity = datetime.utcnow() - timedelta(seconds=50)

        # Send keepalive
        await client.send_keepalive()

        mock_websocket.send.assert_called()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data['cmd'] == 'keepalive'
        assert client.last_activity > datetime.utcnow() - timedelta(seconds=5)

    @pytest.mark.asyncio
    async def test_websdr_reconnection_on_failure(self, mock_websocket, websdr_config):
        """Test automatic reconnection on connection failure"""
        client = WebSDRClient(**websdr_config)

        connect_attempts = 0

        async def mock_connect(*args, **kwargs):
            nonlocal connect_attempts
            connect_attempts += 1
            if connect_attempts < 2:
                raise ConnectionError("Connection failed")
            return mock_websocket

        with patch('websockets.connect', side_effect=mock_connect):
            mock_websocket.recv.return_value = json.dumps({
                'type': 'welcome',
                'session_id': 'ws_new'
            })

            connected = await client.connect_with_retry(max_attempts=3)

            assert connected
            assert connect_attempts == 2

    @pytest.mark.asyncio
    async def test_websdr_bandwidth_limitation(self, mock_websocket, websdr_config):
        """Test WebSDR bandwidth limitations and adaptation"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True

        # Request excessive bandwidth
        result = await client.set_bandwidth(500000)  # 500 kHz

        # Should be limited to max bandwidth
        assert not result  # Failed due to bandwidth limit

        # Request valid bandwidth
        result = await client.set_bandwidth(48000)  # 48 kHz
        assert result

    @pytest.mark.asyncio
    async def test_websdr_multi_channel_support(self, mock_websocket, websdr_config):
        """Test WebSDR multi-channel streaming capability"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True

        # Start multiple channels
        channels = await client.start_multi_channel([
            {'freq': 3573000, 'bw': 12000},
            {'freq': 7074000, 'bw': 12000},
            {'freq': 14074000, 'bw': 12000},
        ])

        assert len(channels) == 3
        # Verify commands sent for each channel
        assert mock_websocket.send.call_count >= 3

    @pytest.mark.asyncio
    async def test_websdr_graceful_disconnect(self, mock_websocket, websdr_config):
        """Test graceful WebSDR disconnection"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True
        client.streaming = True

        await client.disconnect()

        # Verify stop stream and close commands
        mock_websocket.send.assert_called()
        mock_websocket.close.assert_called()
        assert not client.connected
        assert not client.streaming

    @pytest.mark.asyncio
    async def test_websdr_error_handling(self, mock_websocket, websdr_config):
        """Test WebSDR error message handling"""
        client = WebSDRClient(**websdr_config)
        client._ws = mock_websocket
        client.connected = True

        # Mock error response
        mock_websocket.recv.return_value = json.dumps({
            'type': 'error',
            'code': 'FREQ_OUT_OF_RANGE',
            'message': 'Frequency 50000000 out of range'
        })

        with pytest.raises(ValueError) as exc_info:
            await client.tune(frequency_khz=50000)

        assert 'out of range' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_websdr_concurrent_connections(self, mock_websocket, websdr_config):
        """Test handling multiple concurrent WebSDR connections"""
        clients = []
        for i in range(3):
            config = websdr_config.copy()
            client = WebSDRClient(**config)
            clients.append(client)

        with patch('websockets.connect', return_value=mock_websocket):
            mock_websocket.recv.return_value = json.dumps({
                'type': 'welcome',
                'session_id': 'ws_test'
            })

            # Connect all clients concurrently
            results = await asyncio.gather(*[
                c.connect() for c in clients
            ])

            assert all(results)
            assert len(set(c.session_id for c in clients if c.session_id)) == len(clients)