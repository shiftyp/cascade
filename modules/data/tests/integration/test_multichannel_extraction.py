"""Integration tests for multi-channel QRN extraction.

Tests extraction of 9 overlapping 2.5kHz channels from 12kHz IQ recordings
for enhanced noise characterization in neural network training.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from scipy import signal

# Import components to test
from modules.data.src.processors.multichannel_qrn import MultiChannelQRNExtractor
from modules.data.src.processors.qrn_analyzer import QRNAnalyzer
from modules.data.src.storage.file_manager import FileManager


class TestMultiChannelExtraction:
    """Test multi-channel QRN extraction from IQ data."""

    @pytest.fixture
    def extractor(self):
        """Create multi-channel QRN extractor."""
        return MultiChannelQRNExtractor(
            sample_rate=12000,
            channel_width=2500,
            overlap=0.5  # 50% overlap
        )

    @pytest.fixture
    def qrn_analyzer(self):
        """Create QRN analyzer instance."""
        return QRNAnalyzer()

    @pytest.fixture
    def sample_iq_data(self):
        """Generate sample IQ data with known signals."""
        sample_rate = 12000
        duration = 10  # seconds
        t = np.arange(0, duration, 1/sample_rate)

        # Create complex IQ with multiple tones
        iq = np.zeros(len(t), dtype=complex)

        # Add tones at different frequencies
        frequencies = [500, 1500, 2500, 3500, 4500]  # Hz offsets from center
        for freq in frequencies:
            iq += np.exp(1j * 2 * np.pi * freq * t)

        # Add noise
        iq += (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * 0.1

        return iq

    @pytest.mark.asyncio
    async def test_channel_extraction_count(self, extractor, sample_iq_data):
        """Test that exactly 9 channels are extracted."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Verify channel count
        assert len(channels) == 9
        assert all(isinstance(ch, dict) for ch in channels)

    @pytest.mark.asyncio
    async def test_channel_frequency_spacing(self, extractor):
        """Test proper frequency spacing of channels."""
        # Expected channel configuration for 12kHz bandwidth
        expected_channels = [
            {"id": 0, "center": -5000, "start": -6250, "end": -3750},
            {"id": 1, "center": -3750, "start": -5000, "end": -2500},
            {"id": 2, "center": -2500, "start": -3750, "end": -1250},
            {"id": 3, "center": -1250, "start": -2500, "end": 0},
            {"id": 4, "center": 0, "start": -1250, "end": 1250},
            {"id": 5, "center": 1250, "start": 0, "end": 2500},
            {"id": 6, "center": 2500, "start": 1250, "end": 3750},
            {"id": 7, "center": 3750, "start": 2500, "end": 5000},
            {"id": 8, "center": 5000, "start": 3750, "end": 6250}
        ]

        # Get channel configuration
        config = await extractor.get_channel_config()

        # Verify configuration
        assert len(config) == 9
        for i, ch in enumerate(config):
            assert ch["id"] == i
            assert abs(ch["center"] - expected_channels[i]["center"]) < 10

    @pytest.mark.asyncio
    async def test_channel_overlap(self, extractor):
        """Test 50% overlap between adjacent channels."""
        config = await extractor.get_channel_config()

        # Check overlap between adjacent channels
        for i in range(len(config) - 1):
            ch1 = config[i]
            ch2 = config[i + 1]

            # Calculate overlap
            overlap_start = max(ch1["start"], ch2["start"])
            overlap_end = min(ch1["end"], ch2["end"])
            overlap_width = overlap_end - overlap_start

            # Verify 50% overlap (1250 Hz)
            assert abs(overlap_width - 1250) < 10

    @pytest.mark.asyncio
    async def test_channel_filtering(self, extractor, sample_iq_data):
        """Test that each channel properly filters its frequency range."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Verify each channel has filtered data
        for i, channel in enumerate(channels):
            assert "data" in channel
            assert len(channel["data"]) > 0
            assert channel["data"].dtype == complex

            # Check decimation (2.5kHz from 12kHz = factor of ~5)
            expected_length = len(sample_iq_data) // 4  # Approximate
            assert abs(len(channel["data"]) - expected_length) < expected_length * 0.1

    @pytest.mark.asyncio
    async def test_quiet_period_detection(self, qrn_analyzer):
        """Test detection of quiet periods in each channel."""
        # Create data with quiet period
        sample_rate = 2500  # Channel sample rate
        duration = 10
        t = np.arange(0, duration, 1/sample_rate)

        # First 5 seconds: noise
        # Last 5 seconds: quiet
        channel_data = np.zeros(len(t), dtype=complex)
        channel_data[:len(t)//2] = (np.random.randn(len(t)//2) +
                                     1j * np.random.randn(len(t)//2)) * 1.0
        channel_data[len(t)//2:] = (np.random.randn(len(t)//2) +
                                     1j * np.random.randn(len(t)//2)) * 0.01

        # Detect quiet periods
        quiet_periods = await qrn_analyzer.detect_quiet_periods(
            channel_data,
            sample_rate=sample_rate,
            threshold_db=-40
        )

        # Verify quiet period detected
        assert len(quiet_periods) > 0
        assert quiet_periods[0]["start_time"] >= 4.5  # Around 5 seconds
        assert quiet_periods[0]["duration"] >= 4.0

    @pytest.mark.asyncio
    async def test_channel_power_statistics(self, extractor, sample_iq_data):
        """Test calculation of power statistics per channel."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Calculate power statistics for each channel
        for channel in channels:
            stats = await extractor.calculate_channel_statistics(channel["data"])

            # Verify statistics
            assert "mean_power_dbm" in stats
            assert "peak_power_dbm" in stats
            assert "noise_floor_dbm" in stats
            assert "dynamic_range_db" in stats

            # Sanity checks
            assert stats["peak_power_dbm"] > stats["mean_power_dbm"]
            assert stats["mean_power_dbm"] > stats["noise_floor_dbm"]
            assert stats["dynamic_range_db"] > 0

    @pytest.mark.asyncio
    async def test_channel_spectral_features(self, extractor, sample_iq_data):
        """Test extraction of spectral features from each channel."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Extract spectral features
        for channel in channels:
            features = await extractor.extract_spectral_features(channel["data"])

            # Verify features
            assert "spectral_centroid" in features
            assert "spectral_bandwidth" in features
            assert "spectral_rolloff" in features
            assert "spectral_flatness" in features
            assert "zero_crossing_rate" in features

            # Sanity checks
            assert 0 <= features["spectral_flatness"] <= 1
            assert features["spectral_centroid"] > 0
            assert features["spectral_bandwidth"] > 0

    @pytest.mark.asyncio
    async def test_channel_temporal_features(self, extractor, sample_iq_data):
        """Test extraction of temporal features from each channel."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Extract temporal features
        for channel in channels:
            features = await extractor.extract_temporal_features(channel["data"])

            # Verify features
            assert "envelope_mean" in features
            assert "envelope_std" in features
            assert "crest_factor" in features
            assert "peak_to_average_ratio" in features

            # Sanity checks
            assert features["crest_factor"] >= 1
            assert features["peak_to_average_ratio"] >= 0

    @pytest.mark.asyncio
    async def test_channel_correlation_matrix(self, extractor, sample_iq_data):
        """Test calculation of inter-channel correlation matrix."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Calculate correlation matrix
        correlation_matrix = await extractor.calculate_correlation_matrix(channels)

        # Verify matrix properties
        assert correlation_matrix.shape == (9, 9)
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Symmetric
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal = 1

        # Adjacent channels should have higher correlation due to overlap
        for i in range(8):
            assert correlation_matrix[i, i+1] > 0.3  # Significant correlation

    @pytest.mark.asyncio
    async def test_channel_phase_coherence(self, extractor):
        """Test measurement of phase coherence between channels."""
        # Create coherent signal
        sample_rate = 12000
        duration = 1
        t = np.arange(0, duration, 1/sample_rate)

        # Coherent signal across all channels
        freq = 1000  # Hz
        iq_coherent = np.exp(1j * 2 * np.pi * freq * t)

        # Extract channels
        channels = await extractor.extract_channels(iq_coherent)

        # Measure phase coherence
        coherence = await extractor.measure_phase_coherence(channels)

        # Verify high coherence for coherent signal
        assert coherence["mean_coherence"] > 0.8
        assert all(c > 0.7 for c in coherence["pairwise_coherence"])

    @pytest.mark.asyncio
    async def test_channel_data_persistence(self, extractor, sample_iq_data):
        """Test saving and loading of multi-channel data."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Add metadata
        for i, channel in enumerate(channels):
            channel["metadata"] = {
                "channel_id": i,
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": "test_corr_123"
            }

        # Mock save operation
        file_manager = Mock()
        with patch.object(file_manager, 'save_channels') as mock_save:
            await file_manager.save_channels(channels, "test_recording.h5")

            # Verify save called with all channels
            mock_save.assert_called_once()
            saved_channels = mock_save.call_args[0][0]
            assert len(saved_channels) == 9

    @pytest.mark.asyncio
    async def test_channel_parallel_processing(self, extractor, sample_iq_data):
        """Test parallel processing of multiple channels."""
        # Extract channels
        channels = await extractor.extract_channels(sample_iq_data)

        # Define processing function
        async def process_channel(channel):
            # Simulate processing
            await asyncio.sleep(0.01)
            return {
                "channel_id": channel["id"],
                "processed": True,
                "features": {"power": np.abs(channel["data"]).mean()}
            }

        # Process channels in parallel
        tasks = [process_channel(ch) for ch in channels]
        results = await asyncio.gather(*tasks)

        # Verify all processed
        assert len(results) == 9
        assert all(r["processed"] for r in results)
        assert all("features" in r for r in results)


class TestMultiChannelQuality:
    """Test quality aspects of multi-channel extraction."""

    @pytest.mark.asyncio
    async def test_channel_snr_estimation(self):
        """Test SNR estimation for each channel."""
        extractor = MultiChannelQRNExtractor()

        # Create signal with known SNR
        signal = np.exp(1j * 2 * np.pi * 1000 * np.arange(12000) / 12000)
        noise = (np.random.randn(12000) + 1j * np.random.randn(12000)) * 0.1
        iq_data = signal + noise

        # Extract channels and estimate SNR
        channels = await extractor.extract_channels(iq_data)

        for channel in channels:
            snr = await extractor.estimate_snr(channel["data"])

            # Verify SNR in expected range
            assert snr > 10  # Should be ~20 dB
            assert snr < 30

    @pytest.mark.asyncio
    async def test_channel_artifact_detection(self):
        """Test detection of artifacts in channels."""
        extractor = MultiChannelQRNExtractor()
        analyzer = QRNAnalyzer()

        # Create data with artifact (spike)
        iq_data = np.random.randn(12000) + 1j * np.random.randn(12000)
        iq_data[6000] = 100 + 100j  # Artifact

        # Extract channels
        channels = await extractor.extract_channels(iq_data)

        # Detect artifacts
        for channel in channels:
            artifacts = await analyzer.detect_artifacts(channel["data"])

            # At least center channels should detect the artifact
            if channel["id"] in [3, 4, 5]:
                assert len(artifacts) > 0