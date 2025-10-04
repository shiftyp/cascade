"""
Unit tests for QA Waterfall Viewer components (T051)

Tests waterfall generation, IQ reading, and metadata aggregation.
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import asyncio
import asyncpg

# Import components to test
from modules.data.src.dashboard.waterfall_generator import WaterfallGenerator
from modules.data.src.dashboard.iq_reader import IQStreamReader as IQReader
from modules.data.src.processors.qa_metadata_aggregator import (
    QAMetadataAggregator,
    QAMetadata,
    AggregatedStats
)

class TestWaterfallGenerator:
    """Test waterfall generation from IQ samples (T051a)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = WaterfallGenerator()

        # Generate test IQ data - complex sinusoid at 1 kHz
        self.sample_rate = 12000
        self.duration = 1.0
        self.frequency = 1000  # 1 kHz tone

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_iq = np.exp(2j * np.pi * self.frequency * t)

        # Add some noise
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1
        self.test_iq += noise

    def test_waterfall_generation(self):
        """Test basic waterfall generation."""
        result = self.generator.generate(
            iq_data=self.test_iq,
            sample_rate=self.sample_rate,
            fft_size=1024,
            overlap=0.5,
            colormap='viridis'
        )

        # Check keys based on actual implementation
        assert 'data' in result
        assert 'data_db' in result
        assert 'frequencies' in result
        assert 'timestamps' in result
        assert 'colormap' in result
        assert 'config' in result

        # Check dimensions
        assert result['data'].ndim == 2
        assert len(result['frequencies']) == result['data'].shape[0]
        assert len(result['timestamps']) == result['data'].shape[1]

        # Check frequency range (should be -6 kHz to +6 kHz for 12 kHz sample rate)
        assert result['frequencies'][0] == pytest.approx(-6000, rel=0.1)
        assert result['frequencies'][-1] == pytest.approx(6000, rel=0.1)

        # Check that data is within expected dB range
        assert np.all(result['data_db'] >= -120.0)  # Above noise floor
        assert np.all(result['data_db'] <= 0.0)  # Below max (0 dB)

    def test_different_fft_sizes(self):
        """Test waterfall generation with different FFT sizes."""
        for fft_size in [256, 512, 1024, 2048]:
            result = self.generator.generate(
                iq_data=self.test_iq,
                sample_rate=self.sample_rate,
                fft_size=fft_size,
                overlap=0.5
            )

            # Frequency resolution should match FFT size
            assert len(result['frequencies']) == fft_size

    def test_overlap_parameter(self):
        """Test different overlap values."""
        for overlap in [0, 0.25, 0.5, 0.75]:
            result = self.generator.generate(
                iq_data=self.test_iq,
                sample_rate=self.sample_rate,
                fft_size=512,
                overlap=overlap
            )

            # More overlap = more time bins
            if overlap > 0:
                assert len(result['timestamps']) > 1

    def test_colormap_generation(self):
        """Test different colormap outputs."""
        for colormap in ['viridis', 'jet', 'hot', 'cool']:
            result = self.generator.generate(
                iq_data=self.test_iq,
                sample_rate=self.sample_rate,
                fft_size=512,
                overlap=0.5,
                colormap=colormap
            )

            # Should have data and colormap specified
            assert 'data' in result
            assert result['colormap'] == colormap

    def test_empty_iq_data(self):
        """Test handling of empty IQ data."""
        # The implementation handles empty data gracefully
        result = self.generator.generate(
            iq_data=np.array([]),
            sample_rate=self.sample_rate,
            fft_size=512
        )
        # Should return some result structure even for empty data
        assert 'data' in result

    def test_invalid_fft_size(self):
        """Test handling of invalid FFT size."""
        # The implementation will use a default if invalid
        result = self.generator.generate(
            iq_data=self.test_iq,
            sample_rate=self.sample_rate,
            fft_size=0  # Invalid - will be replaced with default
        )
        # Should still generate a result
        assert 'data' in result


class TestIQReader:
    """Test IQ file streaming reader (T051b)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reader = IQReader()

        # Create mock FLAC file data
        self.mock_file_path = "/tmp/test_iq.flac"
        self.sample_rate = 12000
        self.duration = 10.0

        # Generate test IQ data
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        i_data = np.sin(2 * np.pi * 1000 * t)
        q_data = np.cos(2 * np.pi * 1000 * t)
        self.test_iq_array = np.column_stack([i_data, q_data])

    @pytest.mark.asyncio
    async def test_read_segment(self):
        """Test reading a segment of IQ data."""
        with patch('soundfile.info') as mock_info, \
             patch('soundfile.read') as mock_read:

            # Mock soundfile.info
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = self.sample_rate
            mock_info_obj.frames = len(self.test_iq_array)
            mock_info_obj.channels = 2
            mock_info.return_value = mock_info_obj

            # Mock soundfile.read to return test data
            start_time = 2.0
            duration = 3.0
            start_sample = int(start_time * self.sample_rate)
            num_samples = int(duration * self.sample_rate)

            mock_read.return_value = (
                self.test_iq_array[start_sample:start_sample + num_samples],
                self.sample_rate
            )

            # Test reading segment
            iq_data = await self.reader.read_segment(
                self.mock_file_path,
                start_time=start_time,
                duration=duration,
                return_complex=True
            )

            assert iq_data is not None
            assert len(iq_data) == num_samples
            assert np.iscomplexobj(iq_data)

    @pytest.mark.asyncio
    async def test_read_from_bytes(self):
        """Test reading IQ data from bytes."""
        # Create mock bytes data
        test_bytes = b"mock_flac_data"

        with patch('soundfile.read') as mock_read:
            mock_read.return_value = (self.test_iq_array[:1000], self.sample_rate)

            iq_data = await self.reader.read_from_bytes(
                test_bytes,
                start_time=0,
                duration=1.0,
                return_complex=True
            )

            assert iq_data is not None
            assert np.iscomplexobj(iq_data)

    @pytest.mark.asyncio
    async def test_stream_chunks(self):
        """Test streaming IQ data in chunks."""
        with patch('soundfile.info') as mock_info, \
             patch('soundfile.read') as mock_read:

            # Mock soundfile.info
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = self.sample_rate
            mock_info_obj.frames = len(self.test_iq_array)
            mock_info_obj.channels = 2
            mock_info.return_value = mock_info_obj

            # Mock soundfile.read to return chunks
            mock_read.side_effect = [
                (self.test_iq_array[i:i+1024], self.sample_rate)
                for i in range(0, 5120, 1024)
            ]

            chunks = []
            async for chunk in self.reader.stream_chunks(
                self.mock_file_path,
                chunk_size=1024,
                overlap=0,
                return_complex=True
            ):
                chunks.append(chunk)
                if len(chunks) >= 5:
                    break

            assert len(chunks) == 5
            assert all(len(chunk) == 1024 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_get_file_info(self):
        """Test getting file information."""
        with patch('soundfile.info') as mock_info:
            mock_info_obj = MagicMock()
            mock_info_obj.samplerate = self.sample_rate
            mock_info_obj.frames = 120000
            mock_info_obj.channels = 2
            mock_info_obj.duration = 10.0
            mock_info.return_value = mock_info_obj

            info = await self.reader.get_file_info(self.mock_file_path)

            assert info['sample_rate'] == self.sample_rate
            assert info['duration'] == 10.0
            assert info['channels'] == 2
            assert info['frames'] == 120000


class TestQAMetadataAggregator:
    """Test QA metadata aggregation (T051e)."""

    @pytest.fixture
    async def aggregator(self):
        """Create test aggregator with mock database."""
        with patch('asyncpg.create_pool') as mock_pool:
            # Create mock pool
            mock_pool_obj = AsyncMock()
            mock_pool.return_value = mock_pool_obj

            # Create aggregator
            agg = QAMetadataAggregator("postgresql://test")
            await agg.initialize()

            yield agg

            await agg.close()

    @pytest.mark.asyncio
    async def test_ingest_qa_sample(self, aggregator):
        """Test ingesting a QA sample."""
        # Create test metadata
        metadata = QAMetadata(
            sample_id="test_001",
            session_id="session_001",
            timestamp=datetime.now(timezone.utc),
            frequency_khz=14074.0,
            band="20m",
            duration=10.0,
            grid_square="FN42",
            snr=12.5,
            signal_type="FT8",
            propagation_mode="F2",
            file_size_bytes=240000
        )

        # Mock database execute
        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Ingest sample
            sample_id = await aggregator.ingest_qa_sample(metadata)

            assert sample_id == "test_001"
            mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_aggregate_stats(self, aggregator):
        """Test aggregating statistics."""
        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Mock query results
            mock_conn.fetchrow.return_value = {
                'total': 1000,
                'total_duration': 36000,  # 10 hours
                'total_size': 10737418240,  # 10 GB
                'avg_snr': 8.5,
                'avg_quality': 0.75,
                'clipping_count': 50
            }

            mock_conn.fetch.side_effect = [
                # Band distribution
                [{'band': '20m', 'count': 500, 'duration': 18000},
                 {'band': '40m', 'count': 500, 'duration': 18000}],
                # Grid distribution
                [{'grid_square': 'FN42', 'count': 100},
                 {'grid_square': 'EM48', 'count': 50}],
                # Signal types
                [{'signal_type': 'FT8', 'count': 800},
                 {'signal_type': 'WSPR', 'count': 200}],
                # Propagation modes
                [{'propagation_mode': 'F2', 'count': 600},
                 {'propagation_mode': 'Es', 'count': 400}],
                # Hourly distribution
                [{'hour': 0, 'count': 40}, {'hour': 12, 'count': 60}],
                # Daily trend
                [{'day': datetime.now(timezone.utc).date(), 'duration': 3600}],
                # Space weather
                [{'space_weather_condition': 'quiet', 'count': 700}]
            ]

            mock_conn.fetchval.return_value = 25  # High K-index samples

            # Get aggregated stats
            stats = await aggregator.aggregate_stats()

            assert stats.total_samples == 1000
            assert stats.total_duration_hours == pytest.approx(10.0)
            assert stats.total_size_gb == pytest.approx(10.0)
            assert stats.average_snr == 8.5
            assert stats.average_quality_score == 0.75
            assert stats.clipping_percentage == 5.0
            assert stats.band_counts['20m'] == 500
            assert stats.band_counts['40m'] == 500
            assert stats.signal_types['FT8'] == 800
            assert stats.propagation_modes['F2'] == 600
            assert stats.high_k_index_samples == 25

    @pytest.mark.asyncio
    async def test_find_similar_samples(self, aggregator):
        """Test finding similar samples."""
        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Mock reference sample
            mock_conn.fetchrow.return_value = {
                'id': 'ref_001',
                'frequency_khz': 14074.0,
                'band': '20m',
                'snr': 10.0,
                'propagation_mode': 'F2'
            }

            # Mock similar samples
            mock_conn.fetch.return_value = [
                {'id': 'sim_001', 'frequency_khz': 14075.0, 'snr': 9.5},
                {'id': 'sim_002', 'frequency_khz': 14073.0, 'snr': 10.5}
            ]

            # Find similar samples
            similar = await aggregator.find_similar_samples('ref_001', limit=10)

            assert len(similar) == 2
            assert similar[0]['id'] == 'sim_001'

    @pytest.mark.asyncio
    async def test_export_metadata(self, aggregator):
        """Test exporting metadata."""
        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Mock query results
            mock_conn.fetch.return_value = [
                {
                    'id': 'test_001',
                    'session_id': 'session_001',
                    'timestamp': datetime.now(timezone.utc),
                    'frequency_khz': 14074.0,
                    'band': '20m'
                }
            ]

            # Export as JSON
            json_data = await aggregator.export_metadata(output_format='json')
            assert json_data is not None
            assert isinstance(json_data, bytes)

            # Verify JSON is valid
            parsed = json.loads(json_data)
            assert len(parsed) == 1
            assert parsed[0]['id'] == 'test_001'

    @pytest.mark.asyncio
    async def test_cache_ttl(self, aggregator):
        """Test cache TTL behavior."""
        cache_key = "stats_None_None_None"

        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Setup mock responses
            mock_conn.fetchrow.return_value = {'total': 100}
            mock_conn.fetch.return_value = []
            mock_conn.fetchval.return_value = 0

            # First call should query database
            stats1 = await aggregator.aggregate_stats()
            assert mock_conn.fetchrow.call_count == 1

            # Second call should use cache
            stats2 = await aggregator.aggregate_stats()
            assert mock_conn.fetchrow.call_count == 1  # No additional calls

            # Simulate cache expiry
            aggregator._cache[f"{cache_key}_time"] = datetime.now(timezone.utc) - timedelta(seconds=301)

            # Third call should query database again
            stats3 = await aggregator.aggregate_stats()
            assert mock_conn.fetchrow.call_count == 2  # New call made

    @pytest.mark.asyncio
    async def test_quality_distribution(self, aggregator):
        """Test getting quality distribution."""
        with patch.object(aggregator._pool, 'acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_acquire.return_value.__aenter__.return_value = mock_conn

            # Mock query results
            mock_conn.fetch.return_value = [
                {'quality_category': 'excellent', 'count': 100, 'avg_snr': 15.0},
                {'quality_category': 'good', 'count': 300, 'avg_snr': 10.0},
                {'quality_category': 'fair', 'count': 400, 'avg_snr': 5.0},
                {'quality_category': 'poor', 'count': 150, 'avg_snr': 0.0},
                {'quality_category': 'very_poor', 'count': 50, 'avg_snr': -5.0}
            ]

            # Get quality distribution
            dist = await aggregator.get_quality_distribution()

            assert 'categories' in dist
            assert len(dist['categories']) == 5
            assert dist['categories'][0]['category'] == 'excellent'
            assert dist['categories'][0]['count'] == 100
            assert dist['categories'][0]['average_snr'] == 15.0


# Integration test for the complete QA viewer system
@pytest.mark.integration
class TestQAViewerIntegration:
    """Integration tests for the complete QA viewer system."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from IQ data to waterfall display."""
        # Generate test IQ data
        sample_rate = 12000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create multi-tone signal
        iq_data = np.zeros(len(t), dtype=complex)
        for freq in [500, 1500, 2500]:  # Multiple tones
            iq_data += np.exp(2j * np.pi * freq * t)

        # Add noise
        noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1
        iq_data += noise

        # Generate waterfall
        generator = WaterfallGenerator()
        waterfall = generator.generate(
            iq_data=iq_data,
            sample_rate=sample_rate,
            fft_size=1024,
            overlap=0.5,
            colormap='viridis'
        )

        # Verify waterfall contains the tones
        assert waterfall is not None
        assert 'data' in waterfall
        assert 'data_db' in waterfall
        assert 'frequencies' in waterfall

        # Check that we can detect all three tones
        for target_freq in [500, 1500, 2500]:
            freq_idx = np.argmin(np.abs(waterfall['frequencies'] - target_freq))
            power_at_tone = np.mean(waterfall['data_db'][freq_idx, :])
            mean_power = np.mean(waterfall['data_db'])

            # Each tone should be visible (less negative in dB)
            assert power_at_tone > mean_power, f"Tone at {target_freq} Hz not detected"

    @pytest.mark.asyncio
    async def test_api_search_integration(self):
        """Test API search functionality."""
        from modules.data.src.dashboard.qa_sample_api import SearchRequest

        # Create search request
        search = SearchRequest(
            band="20m",
            callsign_hash="abc123",
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc),
            min_snr=5.0,
            limit=50
        )

        # Validate request
        assert search.band == "20m"
        assert search.callsign_hash == "abc123"
        assert search.min_snr == 5.0
        assert search.limit == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])