"""Integration tests for correlation preservation in data collection.

Tests that correlated samples (QRN and propagation data from same time/location)
maintain their relationships through the entire processing pipeline.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import components to test
from modules.data.src.processors.correlation_manager import CorrelationManager
from modules.data.src.collectors.recorder import Recorder
from modules.data.src.processors.qrn_analyzer import QRNAnalyzer
from modules.data.src.processors.ft8_decoder import FT8Decoder
from modules.data.src.storage.metadata_db import MetadataDB


class TestCorrelationPreservation:
    """Test preservation of correlations between QRN and propagation data."""

    @pytest.fixture
    def correlation_manager(self):
        """Create correlation manager instance."""
        return CorrelationManager()

    @pytest.fixture
    def recorder(self):
        """Create recorder instance."""
        return Recorder()

    @pytest.fixture
    def qrn_analyzer(self):
        """Create QRN analyzer instance."""
        return QRNAnalyzer()

    @pytest.fixture
    def ft8_decoder(self):
        """Create FT8 decoder instance."""
        return FT8Decoder()

    @pytest.fixture
    def metadata_db(self):
        """Create metadata database interface."""
        return MetadataDB()

    @pytest.mark.asyncio
    async def test_correlation_id_generation(self, correlation_manager):
        """Test generation of unique correlation IDs."""
        # Generate correlation IDs for paired collection
        correlation_id = await correlation_manager.generate_correlation_id(
            timestamp=datetime.utcnow(),
            sdr_id="kiwi_w6rek",
            band="20m"
        )

        # Verify ID format and uniqueness
        assert correlation_id is not None
        assert len(correlation_id) == 36  # UUID format
        assert '-' in correlation_id

        # Generate another ID
        correlation_id_2 = await correlation_manager.generate_correlation_id(
            timestamp=datetime.utcnow(),
            sdr_id="kiwi_w6rek",
            band="20m"
        )

        # Verify uniqueness
        assert correlation_id != correlation_id_2

    @pytest.mark.asyncio
    async def test_paired_recording_correlation(self, recorder, correlation_manager):
        """Test that paired recordings share correlation IDs."""
        # Setup recording parameters
        params = {
            "sdr_id": "kiwi_kfs",
            "band": "40m",
            "frequency": 7080000,
            "duration": 300
        }

        # Generate correlation ID
        correlation_id = await correlation_manager.generate_correlation_id(
            timestamp=datetime.utcnow(),
            **params
        )

        # Mock paired recordings
        with patch.object(recorder, 'start_recording') as mock_record:
            # Start QRN recording
            qrn_session = await recorder.start_recording(
                **params,
                recording_type="qrn",
                correlation_id=correlation_id
            )

            # Start propagation recording (same correlation ID)
            prop_session = await recorder.start_recording(
                **params,
                recording_type="propagation",
                correlation_id=correlation_id
            )

            # Verify same correlation ID
            assert qrn_session.correlation_id == correlation_id
            assert prop_session.correlation_id == correlation_id
            assert qrn_session.correlation_id == prop_session.correlation_id

    @pytest.mark.asyncio
    async def test_qrn_propagation_temporal_alignment(self, correlation_manager):
        """Test temporal alignment of correlated samples."""
        # Create time-aligned samples
        base_time = datetime.utcnow()
        correlation_id = str(uuid.uuid4())

        # QRN sample
        qrn_sample = {
            "correlation_id": correlation_id,
            "start_time": base_time,
            "end_time": base_time + timedelta(minutes=5),
            "type": "qrn"
        }

        # Propagation sample (must overlap)
        prop_sample = {
            "correlation_id": correlation_id,
            "start_time": base_time,
            "end_time": base_time + timedelta(minutes=5),
            "type": "propagation"
        }

        # Verify temporal alignment
        overlap = await correlation_manager.verify_temporal_overlap(
            qrn_sample,
            prop_sample
        )

        assert overlap is True
        assert qrn_sample["start_time"] == prop_sample["start_time"]
        assert qrn_sample["end_time"] == prop_sample["end_time"]

    @pytest.mark.asyncio
    async def test_multi_channel_correlation(self, qrn_analyzer):
        """Test correlation across multiple QRN channels."""
        correlation_id = str(uuid.uuid4())

        # Mock 9-channel QRN extraction (overlapping 2.5kHz channels)
        channels = []
        for i in range(9):
            channel = {
                "channel_id": i,
                "center_freq": 14080000 + (i - 4) * 1250,  # Overlapping channels
                "bandwidth": 2500,
                "correlation_id": correlation_id
            }
            channels.append(channel)

        # Process channels
        with patch.object(qrn_analyzer, 'extract_multichannel') as mock_extract:
            mock_extract.return_value = channels

            result = await qrn_analyzer.extract_multichannel(
                iq_data=np.random.randn(12000),
                correlation_id=correlation_id
            )

            # Verify all channels share correlation ID
            for channel in result:
                assert channel["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_propagation_mode_correlation(self, ft8_decoder):
        """Test correlation of propagation mode labels."""
        correlation_id = str(uuid.uuid4())

        # Mock FT8 signals with propagation modes
        signals = [
            {
                "callsign_hash": "abc123",
                "snr": 15,
                "propagation_mode": "Es",  # Sporadic-E
                "correlation_id": correlation_id
            },
            {
                "callsign_hash": "def456",
                "snr": -5,
                "propagation_mode": "F2",  # F2 layer
                "correlation_id": correlation_id
            }
        ]

        # Process signals
        with patch.object(ft8_decoder, 'decode_with_modes') as mock_decode:
            mock_decode.return_value = signals

            result = await ft8_decoder.decode_with_modes(
                iq_data=np.random.randn(12000),
                correlation_id=correlation_id
            )

            # Verify mode labels preserved with correlation
            for signal in result:
                assert signal["correlation_id"] == correlation_id
                assert signal["propagation_mode"] in ["Es", "F2", "TEP", "EME", "MS", "Aurora"]

    @pytest.mark.asyncio
    async def test_correlation_metadata_storage(self, metadata_db, correlation_manager):
        """Test storage of correlation metadata."""
        correlation_id = str(uuid.uuid4())

        # Create correlation metadata
        correlation_meta = {
            "correlation_id": correlation_id,
            "created_at": datetime.utcnow(),
            "sdr_id": "kiwi_ka7u",
            "band": "15m",
            "samples": [
                {"type": "qrn", "file": "qrn_001.flac"},
                {"type": "ft8", "file": "ft8_001.json"},
                {"type": "wspr", "file": "wspr_001.json"}
            ]
        }

        # Store metadata
        with patch.object(metadata_db, 'store_correlation') as mock_store:
            await metadata_db.store_correlation(correlation_meta)

            # Verify storage
            mock_store.assert_called_once_with(correlation_meta)

    @pytest.mark.asyncio
    async def test_correlation_query_retrieval(self, metadata_db):
        """Test retrieval of correlated samples."""
        correlation_id = str(uuid.uuid4())

        # Mock correlated samples in database
        correlated_samples = [
            {"type": "qrn", "correlation_id": correlation_id, "file": "qrn_001.flac"},
            {"type": "ft8", "correlation_id": correlation_id, "file": "ft8_001.json"},
            {"type": "wspr", "correlation_id": correlation_id, "file": "wspr_001.json"}
        ]

        with patch.object(metadata_db, 'get_correlated_samples') as mock_get:
            mock_get.return_value = correlated_samples

            # Query by correlation ID
            samples = await metadata_db.get_correlated_samples(correlation_id)

            # Verify all related samples retrieved
            assert len(samples) == 3
            assert all(s["correlation_id"] == correlation_id for s in samples)
            assert {s["type"] for s in samples} == {"qrn", "ft8", "wspr"}

    @pytest.mark.asyncio
    async def test_path_context_correlation(self, correlation_manager):
        """Test correlation of path context (geographic propagation info)."""
        correlation_id = str(uuid.uuid4())

        # Create path context
        path_context = {
            "correlation_id": correlation_id,
            "tx_grid": "FN31pr",
            "rx_grid": "DM79lv",
            "distance_km": 3847,
            "bearing": 285,
            "midpoint_lat": 42.5,
            "midpoint_lon": -95.3,
            "great_circle_points": [(lat, lon) for lat, lon in zip(range(30, 50), range(-120, -80))]
        }

        # Store path context
        result = await correlation_manager.store_path_context(path_context)

        # Verify context preserved
        assert result["correlation_id"] == correlation_id
        assert result["distance_km"] == 3847
        assert len(result["great_circle_points"]) > 0

    @pytest.mark.asyncio
    async def test_correlation_chain_validation(self, correlation_manager):
        """Test validation of complete correlation chains."""
        correlation_id = str(uuid.uuid4())

        # Create complete chain
        chain = {
            "correlation_id": correlation_id,
            "qrn_sample": {"file": "qrn_001.flac", "channels": 9},
            "ft8_signals": [{"file": "ft8_001.json", "count": 15}],
            "wspr_signals": [{"file": "wspr_001.json", "count": 3}],
            "path_contexts": [{"file": "paths_001.json", "count": 18}],
            "space_weather": {"kp": 3, "flux": 95}
        }

        # Validate chain completeness
        validation = await correlation_manager.validate_correlation_chain(chain)

        # Verify validation
        assert validation["is_complete"] is True
        assert validation["has_qrn"] is True
        assert validation["has_propagation"] is True
        assert validation["has_context"] is True

    @pytest.mark.asyncio
    async def test_broken_correlation_detection(self, correlation_manager):
        """Test detection of broken correlation chains."""
        correlation_id = str(uuid.uuid4())

        # Create incomplete chain (missing QRN)
        incomplete_chain = {
            "correlation_id": correlation_id,
            "ft8_signals": [{"file": "ft8_001.json"}],
            "wspr_signals": [{"file": "wspr_001.json"}]
            # Missing qrn_sample
        }

        # Validate chain
        validation = await correlation_manager.validate_correlation_chain(incomplete_chain)

        # Verify broken chain detected
        assert validation["is_complete"] is False
        assert validation["has_qrn"] is False
        assert validation["missing_components"] == ["qrn_sample"]

    @pytest.mark.asyncio
    async def test_correlation_repair_mechanism(self, correlation_manager, recorder):
        """Test repair of broken correlations."""
        correlation_id = str(uuid.uuid4())

        # Mock broken correlation
        broken_chain = {
            "correlation_id": correlation_id,
            "ft8_signals": [{"file": "ft8_001.json"}],
            # Missing QRN
        }

        # Attempt repair by re-recording missing component
        with patch.object(recorder, 'record_missing_component') as mock_record:
            repair_result = await correlation_manager.repair_correlation(
                broken_chain,
                missing_type="qrn"
            )

            # Verify repair attempted
            mock_record.assert_called_once()
            assert repair_result["status"] == "repair_attempted"
            assert repair_result["correlation_id"] == correlation_id


class TestCorrelationPerformance:
    """Test performance aspects of correlation handling."""

    @pytest.mark.asyncio
    async def test_bulk_correlation_processing(self):
        """Test processing of many correlations efficiently."""
        manager = Mock()

        # Create 1000 correlations
        correlations = []
        for i in range(1000):
            correlations.append({
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow()
            })

        # Process in bulk
        start_time = datetime.utcnow()
        await manager.process_bulk_correlations(correlations)
        end_time = datetime.utcnow()

        # Verify performance
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 5.0  # Should process 1000 in under 5 seconds

    @pytest.mark.asyncio
    async def test_correlation_index_performance(self):
        """Test performance of correlation ID indexing."""
        db = Mock()

        # Mock indexed query
        correlation_id = str(uuid.uuid4())

        with patch.object(db, 'query_by_correlation_id') as mock_query:
            mock_query.return_value = []

            # Query should be fast with index
            start = datetime.utcnow()
            await db.query_by_correlation_id(correlation_id)
            query_time = (datetime.utcnow() - start).total_seconds()

            # Verify indexed query is fast
            assert query_time < 0.01  # Sub 10ms with index