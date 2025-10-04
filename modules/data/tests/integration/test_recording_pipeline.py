"""
Integration tests for end-to-end recording pipeline
"""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid
import tempfile
import os

# These imports will fail initially (TDD approach)
from cascade_collector.collectors.recorder import Recorder
from cascade_collector.collectors.kiwi_client import KiwiClient
from cascade_collector.storage.compression import FLACCompressor
from cascade_collector.storage.file_manager import FileManager
from cascade_collector.models.recording_session import RecordingSession


class TestRecordingPipeline:
    """Test complete recording pipeline from SDR to storage"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def mock_sdr_client(self):
        """Mock SDR client"""
        client = AsyncMock(spec=KiwiClient)
        client.connected = True
        client.sample_rate = 12000
        client.center_frequency = 14074000
        return client

    @pytest.fixture
    def sample_iq_data(self):
        """Generate sample IQ data"""
        # 10 seconds of IQ data at 12 kHz
        samples = 12000 * 10
        i_data = np.random.randn(samples) * 0.1
        q_data = np.random.randn(samples) * 0.1
        return np.vstack([i_data, q_data]).T.astype(np.float32)

    @pytest.mark.asyncio
    async def test_complete_recording_flow(self, mock_db_session, mock_sdr_client, sample_iq_data):
        """Test complete flow: connect -> record -> compress -> store"""
        recorder = Recorder(mock_db_session)

        # Mock SDR streaming
        async def stream_generator():
            chunk_size = 1024
            for i in range(0, len(sample_iq_data), chunk_size):
                yield sample_iq_data[i:i+chunk_size]

        mock_sdr_client.stream_iq.return_value = stream_generator()

        # Start recording
        session = await recorder.start_recording(
            sdr_client=mock_sdr_client,
            frequency_hz=14074000,
            duration_seconds=10,
            band="20m"
        )

        assert session is not None
        assert session.status == 'recording'

        # Process streaming data
        await recorder.process_stream(session)

        # Verify recording completed
        assert session.status == 'completed'
        assert session.file_path is not None
        assert session.file_size > 0

    @pytest.mark.asyncio
    async def test_realtime_compression(self, mock_db_session, mock_sdr_client, sample_iq_data):
        """Test real-time FLAC compression during recording"""
        recorder = Recorder(mock_db_session)
        compressor = FLACCompressor()

        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp_file:
            try:
                # Stream and compress
                compressed_size = await recorder.record_and_compress(
                    sdr_client=mock_sdr_client,
                    output_path=tmp_file.name,
                    duration_seconds=5
                )

                # Verify compression
                assert compressed_size > 0
                assert os.path.exists(tmp_file.name)

                # Check compression ratio
                uncompressed_size = len(sample_iq_data) * 4  # float32
                compression_ratio = compressed_size / uncompressed_size
                assert compression_ratio < 0.6  # Expect 40%+ compression

            finally:
                os.unlink(tmp_file.name)

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, mock_db_session, mock_sdr_client):
        """Test metadata extraction during recording"""
        recorder = Recorder(mock_db_session)

        # Mock GPS and SDR metadata
        mock_sdr_client.get_gps.return_value = {'lat': 42.3601, 'lon': -71.0589}
        mock_sdr_client.get_metadata.return_value = {
            'version': '1.364',
            'users': 3,
            'gps_good': True
        }

        session = await recorder.start_recording(
            sdr_client=mock_sdr_client,
            frequency_hz=7074000,
            duration_seconds=30
        )

        # Verify metadata captured
        assert session.gps_latitude == 42.3601
        assert session.gps_longitude == -71.0589
        assert session.sdr_metadata['version'] == '1.364'
        assert session.sdr_metadata['gps_good'] is True

    @pytest.mark.asyncio
    async def test_signal_quality_monitoring(self, mock_db_session, mock_sdr_client, sample_iq_data):
        """Test signal quality monitoring during recording"""
        recorder = Recorder(mock_db_session)

        # Add signals to IQ data
        sample_iq_data[1000:1100] += 0.5  # Strong signal
        sample_iq_data[5000:5050] += 0.2  # Weak signal

        async def stream_with_signals():
            yield sample_iq_data

        mock_sdr_client.stream_iq.return_value = stream_with_signals()

        session = await recorder.record_with_monitoring(
            sdr_client=mock_sdr_client,
            monitor_interval=1.0
        )

        # Verify quality metrics
        assert session.signal_count > 0
        assert session.avg_noise_floor_dbm < -90
        assert session.quality_score > 0.5

    @pytest.mark.asyncio
    async def test_error_recovery_during_recording(self, mock_db_session, mock_sdr_client):
        """Test error recovery during recording"""
        recorder = Recorder(mock_db_session)

        # Simulate connection drop
        async def stream_with_error():
            yield np.random.randn(1024, 2)
            yield np.random.randn(1024, 2)
            raise ConnectionError("SDR disconnected")

        mock_sdr_client.stream_iq.return_value = stream_with_error()

        session = await recorder.record_with_retry(
            sdr_client=mock_sdr_client,
            max_retries=3
        )

        # Should attempt recovery
        assert session.status in ['partial', 'recovered']
        assert session.error_count > 0

    @pytest.mark.asyncio
    async def test_concurrent_multi_band_recording(self, mock_db_session):
        """Test concurrent recording on multiple bands"""
        recorder = Recorder(mock_db_session)

        # Create mock clients for different bands
        clients = []
        for freq in [3573000, 7074000, 14074000]:
            client = AsyncMock(spec=KiwiClient)
            client.center_frequency = freq
            client.stream_iq.return_value = self._generate_stream()
            clients.append(client)

        # Start concurrent recordings
        sessions = await asyncio.gather(*[
            recorder.start_recording(client, client.center_frequency, 60)
            for client in clients
        ])

        # Verify all recordings started
        assert len(sessions) == 3
        assert all(s.status == 'recording' for s in sessions)
        assert len(set(s.center_frequency_hz for s in sessions)) == 3

    @pytest.mark.asyncio
    async def test_storage_path_organization(self, mock_db_session, mock_sdr_client):
        """Test proper file organization in storage"""
        recorder = Recorder(mock_db_session)
        file_manager = FileManager()

        session = await recorder.start_recording(
            sdr_client=mock_sdr_client,
            frequency_hz=14074000,
            band="20m"
        )

        # Verify path structure
        expected_pattern = r'recordings/20m/\d{4}-\d{2}-\d{2}/.*\.flac'
        assert file_manager.matches_pattern(session.file_path, expected_pattern)

    @pytest.mark.asyncio
    async def test_correlation_id_preservation(self, mock_db_session, mock_sdr_client):
        """Test correlation ID for paired recordings"""
        recorder = Recorder(mock_db_session)
        correlation_id = str(uuid.uuid4())

        # Start correlated recordings
        sessions = []
        for i in range(2):
            session = await recorder.start_recording(
                sdr_client=mock_sdr_client,
                frequency_hz=14074000,
                correlation_id=correlation_id
            )
            sessions.append(session)

        # Verify correlation preserved
        assert all(s.correlation_id == correlation_id for s in sessions)

    @pytest.mark.asyncio
    async def test_recording_size_limits(self, mock_db_session, mock_sdr_client):
        """Test recording size limits and auto-splitting"""
        recorder = Recorder(mock_db_session)
        recorder.max_file_size_mb = 100  # 100 MB limit

        # Long recording that exceeds size
        session = await recorder.start_recording(
            sdr_client=mock_sdr_client,
            duration_seconds=3600  # 1 hour
        )

        # Should split into multiple files
        assert session.split_count > 1
        assert all(f.size < 100 * 1024 * 1024 for f in session.files)

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_db_session, mock_sdr_client):
        """Test graceful shutdown of recording pipeline"""
        recorder = Recorder(mock_db_session)

        # Start recording
        session = await recorder.start_recording(
            sdr_client=mock_sdr_client,
            duration_seconds=300
        )

        # Simulate shutdown signal
        await recorder.shutdown_gracefully()

        # Verify clean shutdown
        assert session.status == 'stopped'
        assert session.file_path is not None  # Data saved
        mock_db_session.commit.assert_called()

    async def _generate_stream(self):
        """Helper to generate IQ stream"""
        for _ in range(10):
            yield np.random.randn(1024, 2).astype(np.float32)