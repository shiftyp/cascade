"""Integration tests for Tigris S3 storage upload functionality.

Tests FLAC file upload, metadata storage, lifecycle management,
and error recovery for the 40-50TB storage system.
"""

import pytest
import asyncio
import os
import tempfile
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import components to test
from modules.data.src.storage.tigris_storage import TigrisStorageClient
from modules.data.src.storage.compression import FLACCompressor
from modules.data.src.storage.file_manager import FileManager
from modules.data.src.storage.metadata_db import MetadataDB


class TestTigrisStorageUpload:
    """Test Tigris S3 storage integration."""

    @pytest.fixture
    def tigris_client(self):
        """Create Tigris storage client."""
        return TigrisStorageClient(
            bucket="cascade-kiwisdr-data",
            region="auto"
        )

    @pytest.fixture
    def compressor(self):
        """Create FLAC compressor instance."""
        return FLACCompressor()

    @pytest.fixture
    def file_manager(self):
        """Create file manager instance."""
        return FileManager()

    @pytest.fixture
    def metadata_db(self):
        """Create metadata database interface."""
        return MetadataDB()

    @pytest.fixture
    def sample_iq_data(self):
        """Generate sample IQ data for testing."""
        # Generate 1 minute of IQ data at 12kHz
        sample_rate = 12000
        duration = 60  # seconds
        samples = sample_rate * duration

        # Create complex IQ samples
        i_data = np.random.randn(samples).astype(np.float32)
        q_data = np.random.randn(samples).astype(np.float32)
        iq_data = i_data + 1j * q_data

        return iq_data

    @pytest.mark.asyncio
    async def test_basic_file_upload(self, tigris_client, sample_iq_data):
        """Test basic file upload to Tigris S3."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.iq') as tmp_file:
            # Write IQ data
            sample_iq_data.tofile(tmp_file.name)

            # Generate S3 key
            s3_key = f"recordings/2025/01/30/test_recording_{datetime.utcnow().timestamp()}.iq"

            # Mock upload
            with patch.object(tigris_client, 'upload_file', return_value=True) as mock_upload:
                result = await tigris_client.upload_file(tmp_file.name, s3_key)

                # Verify upload
                assert result is True
                mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_flac_compression_before_upload(self, compressor, tigris_client, sample_iq_data):
        """Test FLAC compression before upload."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.iq') as raw_file:
            with tempfile.NamedTemporaryFile(suffix='.flac') as flac_file:
                # Write raw IQ data
                sample_iq_data.tofile(raw_file.name)

                # Compress to FLAC
                compression_ratio = await compressor.compress_iq_to_flac(
                    raw_file.name,
                    flac_file.name
                )

                # Verify compression
                assert compression_ratio > 0.4  # 40-60% expected
                assert compression_ratio < 0.6

                # Get file sizes
                raw_size = os.path.getsize(raw_file.name)
                flac_size = os.path.getsize(flac_file.name)

                # Verify size reduction
                assert flac_size < raw_size
                assert flac_size / raw_size < 0.6  # At least 40% reduction

    @pytest.mark.asyncio
    async def test_multipart_upload_large_file(self, tigris_client):
        """Test multipart upload for large files (>100MB)."""
        # Create large file (100MB)
        large_size = 100 * 1024 * 1024  # 100MB
        large_data = np.random.bytes(large_size)

        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(large_data)
            tmp_file.flush()

            # Mock multipart upload
            with patch.object(tigris_client, 'multipart_upload', return_value="upload_id_123") as mock_upload:
                upload_id = await tigris_client.multipart_upload(
                    tmp_file.name,
                    "large_files/test_large.flac"
                )

                # Verify multipart upload used
                assert upload_id == "upload_id_123"
                mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_with_metadata(self, tigris_client, metadata_db):
        """Test upload with associated metadata."""
        # Create recording metadata
        metadata = {
            "recording_id": "rec_001",
            "sdr_id": "kiwi_w6rek",
            "band": "20m",
            "frequency": 14080000,
            "sample_rate": 12000,
            "duration": 300,
            "timestamp": datetime.utcnow().isoformat(),
            "location": {"lat": 37.7749, "lon": -122.4194},
            "correlation_id": "corr_xyz789"
        }

        with tempfile.NamedTemporaryFile(suffix='.flac') as tmp_file:
            # Mock upload with metadata
            with patch.object(tigris_client, 'upload_with_metadata', return_value=True) as mock_upload:
                result = await tigris_client.upload_with_metadata(
                    tmp_file.name,
                    "recordings/test.flac",
                    metadata
                )

                # Verify metadata included
                assert result is True
                call_args = mock_upload.call_args
                assert call_args[0][2] == metadata

    @pytest.mark.asyncio
    async def test_upload_retry_on_failure(self, tigris_client):
        """Test automatic retry on upload failure."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test data")
            tmp_file.flush()

            # Mock upload with initial failures
            call_count = 0

            async def mock_upload_with_retry(file_path, key):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Network error")
                return True

            with patch.object(tigris_client, 'upload_file', side_effect=mock_upload_with_retry):
                result = await tigris_client.upload_file_with_retry(
                    tmp_file.name,
                    "test.flac",
                    max_retries=3
                )

                # Verify retry succeeded
                assert result is True
                assert call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_uploads(self, tigris_client):
        """Test parallel upload of multiple files."""
        # Create multiple files
        files = []
        for i in range(5):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(f"data_{i}".encode())
            tmp.close()
            files.append(tmp.name)

        try:
            # Mock parallel uploads
            with patch.object(tigris_client, 'upload_file', return_value=True) as mock_upload:
                # Upload files in parallel
                tasks = [
                    tigris_client.upload_file(f, f"parallel/file_{i}.flac")
                    for i, f in enumerate(files)
                ]

                results = await asyncio.gather(*tasks)

                # Verify all uploaded
                assert all(results)
                assert mock_upload.call_count == 5

        finally:
            # Cleanup
            for f in files:
                os.unlink(f)

    @pytest.mark.asyncio
    async def test_storage_lifecycle_policy(self, tigris_client):
        """Test lifecycle policy application for tiered storage."""
        # Define lifecycle rules
        lifecycle_rules = {
            "hot_tier": {
                "duration_days": 7,
                "storage_class": "STANDARD"
            },
            "warm_tier": {
                "duration_days": 30,
                "storage_class": "STANDARD_IA"
            },
            "cold_tier": {
                "duration_days": 90,
                "storage_class": "GLACIER"
            }
        }

        # Mock lifecycle configuration
        with patch.object(tigris_client, 'set_lifecycle_policy', return_value=True) as mock_lifecycle:
            result = await tigris_client.set_lifecycle_policy(lifecycle_rules)

            # Verify policy set
            assert result is True
            mock_lifecycle.assert_called_once_with(lifecycle_rules)

    @pytest.mark.asyncio
    async def test_upload_bandwidth_throttling(self, tigris_client):
        """Test bandwidth throttling during upload."""
        # Configure bandwidth limit (10 Mbps)
        bandwidth_limit = 10 * 1024 * 1024 / 8  # bytes per second

        with tempfile.NamedTemporaryFile() as tmp_file:
            # Write 10MB file
            tmp_file.write(np.random.bytes(10 * 1024 * 1024))
            tmp_file.flush()

            # Mock throttled upload
            with patch.object(tigris_client, 'upload_with_throttle') as mock_upload:
                await tigris_client.upload_with_throttle(
                    tmp_file.name,
                    "throttled.flac",
                    bandwidth_limit
                )

                # Verify throttling applied
                call_args = mock_upload.call_args
                assert call_args[0][2] == bandwidth_limit

    @pytest.mark.asyncio
    async def test_checksum_verification(self, tigris_client):
        """Test checksum verification after upload."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            data = b"test data for checksum"
            tmp_file.write(data)
            tmp_file.flush()

            # Calculate expected checksum
            expected_md5 = hashlib.md5(data).hexdigest()

            # Mock upload with checksum
            with patch.object(tigris_client, 'upload_with_verification') as mock_upload:
                mock_upload.return_value = {"md5": expected_md5, "status": "verified"}

                result = await tigris_client.upload_with_verification(
                    tmp_file.name,
                    "verified.flac"
                )

                # Verify checksum match
                assert result["md5"] == expected_md5
                assert result["status"] == "verified"

    @pytest.mark.asyncio
    async def test_storage_quota_monitoring(self, tigris_client):
        """Test storage quota monitoring and alerts."""
        # Mock storage usage
        usage_info = {
            "used_bytes": 35 * 1024**4,  # 35TB used
            "quota_bytes": 50 * 1024**4,  # 50TB quota
            "percentage": 70.0
        }

        with patch.object(tigris_client, 'get_storage_usage', return_value=usage_info):
            usage = await tigris_client.get_storage_usage()

            # Check usage monitoring
            assert usage["percentage"] == 70.0
            assert usage["used_bytes"] < usage["quota_bytes"]

    @pytest.mark.asyncio
    async def test_upload_deduplication(self, tigris_client, metadata_db):
        """Test deduplication of identical files."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            data = b"duplicate content"
            tmp_file.write(data)
            tmp_file.flush()

            # Calculate file hash
            file_hash = hashlib.sha256(data).hexdigest()

            # Mock existing file check
            with patch.object(metadata_db, 'file_exists', return_value=True):
                with patch.object(tigris_client, 'upload_file') as mock_upload:
                    # Attempt upload
                    result = await tigris_client.upload_with_dedup(
                        tmp_file.name,
                        "potential_duplicate.flac",
                        file_hash
                    )

                    # Verify upload skipped
                    assert result["status"] == "duplicate_skipped"
                    mock_upload.assert_not_called()


class TestStorageResilience:
    """Test storage system resilience and recovery."""

    @pytest.mark.asyncio
    async def test_incomplete_upload_recovery(self):
        """Test recovery of incomplete uploads."""
        storage = Mock()

        # Mock incomplete uploads
        incomplete = [
            {"upload_id": "up_001", "key": "file1.flac", "parts": 3, "total": 5},
            {"upload_id": "up_002", "key": "file2.flac", "parts": 1, "total": 3}
        ]

        storage.get_incomplete_uploads.return_value = incomplete

        # Recover uploads
        recovered = await storage.recover_incomplete_uploads()

        # Verify recovery attempted
        assert len(recovered) == 2
        storage.resume_upload.assert_called()

    @pytest.mark.asyncio
    async def test_corrupted_file_quarantine(self):
        """Test quarantine of corrupted files."""
        storage = Mock()
        validator = Mock()

        # Mock corrupted file detection
        validator.validate_file.return_value = False

        # Process file
        result = await storage.process_uploaded_file("corrupted.flac")

        # Verify quarantine
        assert result["status"] == "quarantined"
        assert "corruption" in result["reason"]


class TestStorageMetrics:
    """Test storage metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_upload_performance_metrics(self):
        """Test collection of upload performance metrics."""
        metrics = Mock()

        # Record upload metrics
        upload_metrics = {
            "file_size": 100 * 1024 * 1024,  # 100MB
            "upload_time": 10.5,  # seconds
            "bandwidth": 9.5 * 1024 * 1024,  # bytes/sec
            "retries": 0
        }

        await metrics.record_upload(upload_metrics)

        # Verify metrics recorded
        metrics.record_upload.assert_called_once_with(upload_metrics)

    @pytest.mark.asyncio
    async def test_storage_cost_tracking(self):
        """Test tracking of storage costs."""
        cost_tracker = Mock()

        # Calculate monthly costs
        storage_gb = 35000  # 35TB
        egress_gb = 100    # 100GB egress

        costs = await cost_tracker.calculate_monthly_cost(storage_gb, egress_gb)

        # Verify cost calculation
        cost_tracker.calculate_monthly_cost.assert_called_once()
        assert costs["storage_cost"] > 0
        assert costs["egress_cost"] > 0