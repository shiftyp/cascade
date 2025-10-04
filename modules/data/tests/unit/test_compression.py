"""Unit tests for FLAC compression utility."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
import tempfile
import os

from src.storage.compression import FLACCompressor


class TestFLACCompressor:
    """Test FLAC compression for IQ data storage."""

    @pytest.fixture
    def compressor(self):
        """Create FLACCompressor instance."""
        return FLACCompressor(
            compression_level=5,
            block_size=4096
        )

    @pytest.fixture
    def sample_iq_data(self):
        """Generate sample IQ data."""
        # Generate complex IQ samples
        samples = 12000
        t = np.linspace(0, 1, samples)
        i_data = np.sin(2 * np.pi * 1000 * t) * 0.5
        q_data = np.cos(2 * np.pi * 1000 * t) * 0.5
        return i_data + 1j * q_data

    def test_compressor_initialization(self, compressor):
        """Test compressor initialization."""
        assert compressor.compression_level == 5
        assert compressor.block_size == 4096
        assert compressor.sample_rate == 12000

    def test_compress_iq_data(self, compressor, sample_iq_data):
        """Test IQ data compression."""
        compressed = compressor.compress(sample_iq_data)

        assert compressed is not None
        assert len(compressed) > 0

        # Check compression ratio
        original_size = sample_iq_data.nbytes
        compressed_size = len(compressed)
        compression_ratio = compressed_size / original_size

        assert compression_ratio < 0.7  # Expect at least 30% compression

    def test_decompress_iq_data(self, compressor, sample_iq_data):
        """Test IQ data decompression."""
        compressed = compressor.compress(sample_iq_data)
        decompressed = compressor.decompress(compressed)

        assert decompressed is not None
        assert len(decompressed) == len(sample_iq_data)

        # Check data integrity (allowing for small numerical errors)
        np.testing.assert_allclose(
            decompressed.real, sample_iq_data.real, rtol=1e-5
        )
        np.testing.assert_allclose(
            decompressed.imag, sample_iq_data.imag, rtol=1e-5
        )

    def test_compress_with_metadata(self, compressor, sample_iq_data):
        """Test compression with metadata."""
        metadata = {
            'frequency': 14100000,
            'timestamp': '2025-01-01T00:00:00Z',
            'sdr_name': 'test_sdr'
        }

        compressed = compressor.compress_with_metadata(
            sample_iq_data, metadata
        )

        assert compressed is not None
        assert 'data' in compressed
        assert 'metadata' in compressed
        assert compressed['metadata'] == metadata

    @patch('builtins.open', new_callable=mock_open)
    def test_compress_to_file(self, mock_file, compressor, sample_iq_data):
        """Test compression directly to file."""
        output_path = '/tmp/test.flac'

        result = compressor.compress_to_file(sample_iq_data, output_path)

        assert result is True
        mock_file.assert_called_with(output_path, 'wb')
        mock_file().write.assert_called()

    def test_compress_empty_data(self, compressor):
        """Test compression of empty data."""
        empty_data = np.array([], dtype=np.complex64)

        compressed = compressor.compress(empty_data)

        assert compressed is not None
        assert len(compressed) > 0  # Header still present

    def test_compress_large_data(self, compressor):
        """Test compression of large dataset."""
        # 1 minute of IQ data at 12 kHz
        large_data = np.random.randn(12000 * 60) + 1j * np.random.randn(12000 * 60)

        compressed = compressor.compress(large_data)

        assert compressed is not None
        compression_ratio = len(compressed) / large_data.nbytes
        assert compression_ratio < 0.8  # Some compression achieved

    def test_compression_levels(self, sample_iq_data):
        """Test different compression levels."""
        sizes = {}

        for level in [0, 5, 8]:  # Fast, medium, best
            compressor = FLACCompressor(compression_level=level)
            compressed = compressor.compress(sample_iq_data)
            sizes[level] = len(compressed)

        # Higher compression levels should produce smaller files
        assert sizes[8] <= sizes[5]
        assert sizes[5] <= sizes[0]

    def test_block_size_impact(self, sample_iq_data):
        """Test impact of different block sizes."""
        sizes = {}

        for block_size in [1152, 4096, 8192]:
            compressor = FLACCompressor(block_size=block_size)
            compressed = compressor.compress(sample_iq_data)
            sizes[block_size] = len(compressed)

        # All should compress successfully
        assert all(size > 0 for size in sizes.values())

    def test_bit_depth_handling(self, compressor):
        """Test handling of different bit depths."""
        # 16-bit data
        data_16 = (np.random.randn(1000) * 32767).astype(np.int16)
        compressed_16 = compressor.compress_int16(data_16)
        assert compressed_16 is not None

        # 24-bit data (stored as 32-bit)
        data_24 = (np.random.randn(1000) * 8388607).astype(np.int32)
        compressed_24 = compressor.compress_int24(data_24)
        assert compressed_24 is not None

    def test_streaming_compression(self, compressor, sample_iq_data):
        """Test streaming compression for real-time data."""
        chunk_size = 1024
        chunks = [sample_iq_data[i:i+chunk_size]
                 for i in range(0, len(sample_iq_data), chunk_size)]

        compressor.start_stream()

        for chunk in chunks:
            compressed_chunk = compressor.compress_chunk(chunk)
            assert compressed_chunk is not None

        final = compressor.finish_stream()
        assert final is not None

    def test_compression_statistics(self, compressor, sample_iq_data):
        """Test compression statistics calculation."""
        compressed = compressor.compress(sample_iq_data)
        stats = compressor.get_statistics()

        assert 'original_size' in stats
        assert 'compressed_size' in stats
        assert 'compression_ratio' in stats
        assert 'bits_per_sample' in stats

        assert stats['compression_ratio'] > 1.0  # Compression achieved

    def test_error_handling(self, compressor):
        """Test error handling for invalid data."""
        # Invalid data type
        with pytest.raises(TypeError):
            compressor.compress("not_an_array")

        # Invalid shape
        invalid_data = np.array([[1, 2], [3, 4]], dtype=np.complex64)
        with pytest.raises(ValueError):
            compressor.compress(invalid_data)

    def test_parallel_compression(self, compressor, sample_iq_data):
        """Test parallel compression of multiple files."""
        files = [sample_iq_data.copy() for _ in range(5)]

        results = compressor.compress_batch(files, n_jobs=2)

        assert len(results) == 5
        assert all(r is not None for r in results)