"""File manager for FLAC compressed IQ data storage.

Implements T034: File manager for FLAC storage.
"""

import os
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from ..storage import compression

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a stored file."""

    file_path: str
    file_size: int
    checksum: str
    created_at: datetime
    compression_ratio: float
    original_size: int


@dataclass
class StorageStats:
    """Storage utilization statistics."""

    total_files: int
    total_size_bytes: int
    total_compressed_size: bytes
    avg_compression_ratio: float
    oldest_file: datetime
    newest_file: datetime
    free_space_bytes: int


class FileManager:
    """Manages FLAC file storage with compression and organization."""

    def __init__(self, base_storage_path: str = "/data/cascade"):
        """Initialize file manager.

        Args:
            base_storage_path: Base directory for file storage
        """
        self.base_path = Path(base_storage_path)
        # Use compression module functions

        # Create directory structure
        self.iq_path = self.base_path / "iq_data"
        self.qa_path = self.base_path / "qa_samples"
        self.temp_path = self.base_path / "temp"
        self.archive_path = self.base_path / "archive"

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.iq_path, self.qa_path, self.temp_path, self.archive_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")

    def store_iq_data(
        self,
        iq_samples: Union[bytes, str],
        session_id: str,
        metadata: Dict[str, Any]
    ) -> FileInfo:
        """Store IQ data with FLAC compression.

        Args:
            iq_samples: IQ data (bytes or file path)
            session_id: Recording session ID
            metadata: Session metadata

        Returns:
            FileInfo for stored file
        """
        # Generate file path based on session metadata
        file_path = self._generate_file_path(session_id, metadata)

        # Handle input data
        if isinstance(iq_samples, str):
            # Input is file path
            input_path = iq_samples
            original_size = os.path.getsize(input_path)
        else:
            # Input is raw bytes
            temp_input = self.temp_path / f"{session_id}_input.tmp"
            with open(temp_input, 'wb') as f:
                f.write(iq_samples)
            input_path = str(temp_input)
            original_size = len(iq_samples)

        try:
            # Load IQ data for compression
            if isinstance(iq_samples, bytes):
                # Convert bytes to complex array
                iq_array = np.frombuffer(iq_samples, dtype=np.complex64)
            else:
                # Load from file
                iq_array = np.fromfile(input_path, dtype=np.complex64)

            # Compress to FLAC
            compressed_path, compression_ratio_calc = compression.compress_to_flac(
                iq_array,
                metadata.get('sample_rate', 12000),
                file_path
            )

            # Calculate actual metrics
            compressed_size = os.path.getsize(compressed_path)
            actual_compression_ratio = original_size / compressed_size if compressed_size > 0 else compression_ratio_calc
            checksum = self._calculate_checksum(compressed_path)

            # Create file info
            file_info = FileInfo(
                file_path=str(compressed_path),
                file_size=compressed_size,
                checksum=checksum,
                created_at=datetime.utcnow(),
                compression_ratio=actual_compression_ratio,
                original_size=original_size
            )

            # Store metadata alongside
            self._store_metadata(file_path, metadata, file_info)

            logger.info(
                f"Stored IQ data: {session_id}, "
                f"size: {compressed_size:,} bytes "
                f"(ratio: {compression_ratio:.1f}x)"
            )

            return file_info

        finally:
            # Clean up temporary files
            if isinstance(iq_samples, bytes):
                if os.path.exists(input_path):
                    os.remove(input_path)

    def store_qa_sample(
        self,
        iq_samples: Union[bytes, str],
        session_id: str,
        metadata: Dict[str, Any]
    ) -> FileInfo:
        """Store QA sample in hot storage.

        Args:
            iq_samples: IQ data for QA sample
            session_id: Recording session ID
            metadata: Sample metadata including quality metrics

        Returns:
            FileInfo for stored QA sample
        """
        # QA samples go to separate hot storage
        qa_file_path = self.qa_path / self._generate_qa_filename(session_id, metadata)

        # Handle input data
        if isinstance(iq_samples, str):
            input_path = iq_samples
            original_size = os.path.getsize(input_path)
        else:
            temp_input = self.temp_path / f"{session_id}_qa_input.tmp"
            with open(temp_input, 'wb') as f:
                f.write(iq_samples)
            input_path = str(temp_input)
            original_size = len(iq_samples)

        try:
            # Load IQ data for compression
            if isinstance(iq_samples, bytes):
                iq_array = np.frombuffer(iq_samples, dtype=np.complex64)
            else:
                iq_array = np.fromfile(input_path, dtype=np.complex64)

            # Compress QA sample to FLAC
            compressed_path, compression_ratio_calc = compression.compress_to_flac(
                iq_array,
                metadata.get('sample_rate', 12000),
                qa_file_path
            )

            # Calculate metrics
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            checksum = self._calculate_checksum(compressed_path)

            file_info = FileInfo(
                file_path=str(compressed_path),
                file_size=compressed_size,
                checksum=checksum,
                created_at=datetime.utcnow(),
                compression_ratio=compression_ratio,
                original_size=original_size
            )

            # Store QA metadata
            self._store_qa_metadata(qa_file_path, metadata, file_info)

            logger.info(f"Stored QA sample: {session_id}, quality: {metadata.get('quality_score', 'unknown')}")

            return file_info

        finally:
            # Clean up
            if isinstance(iq_samples, bytes) and os.path.exists(input_path):
                os.remove(input_path)

    def retrieve_iq_data(self, file_path: str) -> bytes:
        """Retrieve and decompress IQ data.

        Args:
            file_path: Path to compressed file

        Returns:
            Decompressed IQ data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IQ file not found: {file_path}")

        # Decompress FLAC data using compression module
        try:
            iq_data = compression.decompress_flac(file_path)

            # Convert back to bytes if needed
            if isinstance(iq_data, np.ndarray):
                iq_bytes = iq_data.astype(np.complex64).tobytes()
            else:
                iq_bytes = iq_data

            logger.debug(f"Retrieved IQ data: {len(iq_bytes):,} bytes from {file_path}")
            return iq_bytes

        except Exception as e:
            logger.error(f"Failed to decompress IQ data from {file_path}: {e}")
            raise

    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """Get information about a stored file.

        Args:
            file_path: Path to file

        Returns:
            FileInfo or None if not found
        """
        if not os.path.exists(file_path):
            return None

        stat = os.stat(file_path)
        checksum = self._calculate_checksum(file_path)

        # Try to load metadata
        metadata_path = file_path + ".meta"
        compression_ratio = 1.0
        original_size = stat.st_size

        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    compression_ratio = meta.get('compression_ratio', 1.0)
                    original_size = meta.get('original_size', stat.st_size)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {file_path}: {e}")

        return FileInfo(
            file_path=file_path,
            file_size=stat.st_size,
            checksum=checksum,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            compression_ratio=compression_ratio,
            original_size=original_size
        )

    def list_files(self, pattern: str = "*.flac", directory: str = "iq_data") -> List[FileInfo]:
        """List files in storage.

        Args:
            pattern: File pattern to match
            directory: Directory to search ("iq_data", "qa_samples", "archive")

        Returns:
            List of FileInfo objects
        """
        if directory == "iq_data":
            search_path = self.iq_path
        elif directory == "qa_samples":
            search_path = self.qa_path
        elif directory == "archive":
            search_path = self.archive_path
        else:
            search_path = self.base_path / directory

        files = []
        for file_path in search_path.glob(pattern):
            if file_path.is_file():
                file_info = self.get_file_info(str(file_path))
                if file_info:
                    files.append(file_info)

        # Sort by creation time (newest first)
        files.sort(key=lambda f: f.created_at, reverse=True)
        return files

    def delete_file(self, file_path: str, keep_metadata: bool = False) -> bool:
        """Delete a file from storage.

        Args:
            file_path: Path to file to delete
            keep_metadata: Whether to keep metadata file

        Returns:
            True if deletion successful
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")

            # Remove metadata unless requested to keep
            if not keep_metadata:
                metadata_path = file_path + ".meta"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)

            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def archive_old_files(self, days_old: int = 30) -> int:
        """Archive files older than specified days.

        Args:
            days_old: Archive files older than this many days

        Returns:
            Number of files archived
        """
        cutoff_time = datetime.utcnow().timestamp() - (days_old * 24 * 3600)
        archived_count = 0

        for file_path in self.iq_path.glob("*.flac"):
            if file_path.stat().st_ctime < cutoff_time:
                # Move to archive
                archive_dest = self.archive_path / file_path.name

                try:
                    shutil.move(str(file_path), str(archive_dest))

                    # Move metadata too
                    metadata_src = str(file_path) + ".meta"
                    metadata_dest = str(archive_dest) + ".meta"
                    if os.path.exists(metadata_src):
                        shutil.move(metadata_src, metadata_dest)

                    archived_count += 1
                    logger.debug(f"Archived file: {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to archive {file_path}: {e}")

        logger.info(f"Archived {archived_count} files older than {days_old} days")
        return archived_count

    def get_storage_stats(self) -> StorageStats:
        """Get storage utilization statistics.

        Returns:
            Storage statistics
        """
        all_files = self.list_files("*", "iq_data") + self.list_files("*", "qa_samples")

        if not all_files:
            return StorageStats(
                total_files=0,
                total_size_bytes=0,
                total_compressed_size=0,
                avg_compression_ratio=0,
                oldest_file=datetime.utcnow(),
                newest_file=datetime.utcnow(),
                free_space_bytes=self._get_free_space()
            )

        total_files = len(all_files)
        total_size = sum(f.file_size for f in all_files)
        total_original = sum(f.original_size for f in all_files)
        avg_ratio = total_original / total_size if total_size > 0 else 0

        oldest = min(f.created_at for f in all_files)
        newest = max(f.created_at for f in all_files)

        return StorageStats(
            total_files=total_files,
            total_size_bytes=total_size,
            total_compressed_size=total_size,
            avg_compression_ratio=avg_ratio,
            oldest_file=oldest,
            newest_file=newest,
            free_space_bytes=self._get_free_space()
        )

    def cleanup_temp_files(self) -> int:
        """Clean up temporary files.

        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        for temp_file in self.temp_path.glob("*"):
            try:
                if temp_file.is_file():
                    os.remove(temp_file)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean temp file {temp_file}: {e}")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} temporary files")

        return cleaned

    def _generate_file_path(self, session_id: str, metadata: Dict[str, Any]) -> Path:
        """Generate organized file path for IQ data.

        Args:
            session_id: Session ID
            metadata: Session metadata

        Returns:
            Path for file storage
        """
        # Extract date from metadata
        timestamp = metadata.get('start_time', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        # Create date-based directory structure
        date_dir = timestamp.strftime("%Y/%m/%d")
        hour_dir = timestamp.strftime("%H")

        # Include frequency and band in filename
        frequency = metadata.get('center_frequency_hz', 0)
        band = metadata.get('band', 'unknown')

        filename = f"{session_id}_{band}_{frequency}_{timestamp.strftime('%H%M%S')}.flac"

        return self.iq_path / date_dir / hour_dir / filename

    def _generate_qa_filename(self, session_id: str, metadata: Dict[str, Any]) -> str:
        """Generate filename for QA sample.

        Args:
            session_id: Session ID
            metadata: QA metadata

        Returns:
            QA sample filename
        """
        timestamp = datetime.utcnow()
        quality = metadata.get('quality_score', 0)
        band = metadata.get('band', 'unknown')

        return f"qa_{session_id}_{band}_q{quality:.2f}_{timestamp.strftime('%Y%m%d_%H%M%S')}.flac"

    def _store_metadata(self, file_path: Path, metadata: Dict[str, Any], file_info: FileInfo):
        """Store metadata alongside file.

        Args:
            file_path: Path to main file
            metadata: Session metadata
            file_info: File information
        """
        metadata_path = str(file_path) + ".meta"

        combined_metadata = {
            "session_metadata": metadata,
            "file_info": {
                "size": file_info.file_size,
                "checksum": file_info.checksum,
                "created_at": file_info.created_at.isoformat(),
                "compression_ratio": file_info.compression_ratio,
                "original_size": file_info.original_size
            }
        }

        try:
            import json
            with open(metadata_path, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to store metadata: {e}")

    def _store_qa_metadata(self, file_path: Path, metadata: Dict[str, Any], file_info: FileInfo):
        """Store QA metadata.

        Args:
            file_path: Path to QA file
            metadata: QA metadata
            file_info: File information
        """
        self._store_metadata(file_path, metadata, file_info)

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 checksum
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_free_space(self) -> int:
        """Get free space on storage device.

        Returns:
            Free space in bytes
        """
        try:
            stat = shutil.disk_usage(self.base_path)
            return stat.free
        except Exception:
            return 0