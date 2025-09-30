"""File manager for CASCADE storage operations."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json
import shutil

logger = logging.getLogger(__name__)


class FileManager:
    """Manages local file storage and organization."""

    def __init__(self, base_dir: str = "/tmp/cascade"):
        """Initialize file manager.

        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self.recordings_dir = self.base_dir / "recordings"
        self.processed_dir = self.base_dir / "processed"
        self.quarantine_dir = self.base_dir / "quarantine"
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for dir_path in [self.recordings_dir, self.processed_dir, self.quarantine_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_recording_path(self, sdr_id: str, band: str, timestamp: datetime) -> Path:
        """Get path for a new recording.

        Args:
            sdr_id: SDR identifier
            band: Frequency band
            timestamp: Recording timestamp

        Returns:
            Path for recording file
        """
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")
        filename = f"{sdr_id}_{band}_{date_str}_{time_str}.wav"

        # Organize by date and band
        dir_path = self.recordings_dir / date_str / band
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path / filename

    def get_processed_path(self, original_path: Path, format: str = "flac") -> Path:
        """Get path for processed file.

        Args:
            original_path: Original recording path
            format: Output format

        Returns:
            Path for processed file
        """
        relative = original_path.relative_to(self.recordings_dir)
        new_path = self.processed_dir / relative
        new_path = new_path.with_suffix(f".{format}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        return new_path

    def quarantine_file(self, file_path: Path, reason: str) -> Path:
        """Move file to quarantine.

        Args:
            file_path: File to quarantine
            reason: Reason for quarantine

        Returns:
            Quarantine path
        """
        quarantine_path = self.quarantine_dir / file_path.name

        # Add metadata
        metadata_path = quarantine_path.with_suffix('.json')
        metadata = {
            "original_path": str(file_path),
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        }

        # Move file
        shutil.move(str(file_path), str(quarantine_path))

        # Write metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.warning(f"Quarantined {file_path.name}: {reason}")
        return quarantine_path

    def cleanup_old_files(self, days: int = 7) -> int:
        """Clean up old processed files.

        Args:
            days: Age threshold in days

        Returns:
            Number of files deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted = 0

        for file_path in self.processed_dir.rglob("*"):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    file_path.unlink()
                    deleted += 1

        if deleted:
            logger.info(f"Cleaned up {deleted} old files")

        return deleted

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Storage statistics
        """
        def get_dir_size(path: Path) -> int:
            """Get total size of directory."""
            total = 0
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size
            return total

        recordings_size = get_dir_size(self.recordings_dir)
        processed_size = get_dir_size(self.processed_dir)
        quarantine_size = get_dir_size(self.quarantine_dir)

        return {
            "recordings_mb": recordings_size / 1024 / 1024,
            "processed_mb": processed_size / 1024 / 1024,
            "quarantine_mb": quarantine_size / 1024 / 1024,
            "total_mb": (recordings_size + processed_size + quarantine_size) / 1024 / 1024,
            "recordings_count": len(list(self.recordings_dir.rglob("*.wav"))),
            "processed_count": len(list(self.processed_dir.rglob("*.flac"))),
            "quarantine_count": len(list(self.quarantine_dir.rglob("*")))
        }

    def list_recordings(self, band: Optional[str] = None,
                       date: Optional[datetime] = None) -> List[Path]:
        """List recording files.

        Args:
            band: Filter by band
            date: Filter by date

        Returns:
            List of recording paths
        """
        pattern = "*.wav"

        if date:
            date_str = date.strftime("%Y%m%d")
            if band:
                search_path = self.recordings_dir / date_str / band
            else:
                search_path = self.recordings_dir / date_str
        elif band:
            search_path = self.recordings_dir
            pattern = f"*/{band}/*.wav"
        else:
            search_path = self.recordings_dir

        if search_path.exists():
            return list(search_path.rglob(pattern))
        return []