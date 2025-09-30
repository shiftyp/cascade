"""Metadata store for CASCADE recordings."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class RecordingMetadata:
    """Metadata for a recording."""
    recording_id: str
    sdr_id: str
    band: str
    frequency: float
    sample_rate: int
    duration_seconds: float
    timestamp: datetime
    file_path: str
    file_size: int
    compressed_path: Optional[str] = None
    compressed_size: Optional[int] = None
    ft8_count: int = 0
    wspr_count: int = 0
    qrn_type: Optional[str] = None
    noise_floor_dbm: Optional[float] = None
    correlation_id: Optional[str] = None


class MetadataStore:
    """Stores and manages recording metadata."""

    def __init__(self):
        """Initialize metadata store."""
        self.metadata: Dict[str, RecordingMetadata] = {}
        self.by_band: Dict[str, List[str]] = {}
        self.by_sdr: Dict[str, List[str]] = {}

    def add_recording(self, metadata: RecordingMetadata) -> None:
        """Add recording metadata.

        Args:
            metadata: Recording metadata
        """
        self.metadata[metadata.recording_id] = metadata

        # Update indexes
        if metadata.band not in self.by_band:
            self.by_band[metadata.band] = []
        self.by_band[metadata.band].append(metadata.recording_id)

        if metadata.sdr_id not in self.by_sdr:
            self.by_sdr[metadata.sdr_id] = []
        self.by_sdr[metadata.sdr_id].append(metadata.recording_id)

        logger.debug(f"Added metadata for recording {metadata.recording_id}")

    def get_recording(self, recording_id: str) -> Optional[RecordingMetadata]:
        """Get recording metadata.

        Args:
            recording_id: Recording identifier

        Returns:
            Recording metadata or None
        """
        return self.metadata.get(recording_id)

    def update_recording(self, recording_id: str, updates: Dict[str, Any]) -> bool:
        """Update recording metadata.

        Args:
            recording_id: Recording identifier
            updates: Fields to update

        Returns:
            True if updated successfully
        """
        if recording_id not in self.metadata:
            return False

        metadata = self.metadata[recording_id]
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        return True

    def find_recordings(self, band: Optional[str] = None,
                       sdr_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[RecordingMetadata]:
        """Find recordings matching criteria.

        Args:
            band: Filter by band
            sdr_id: Filter by SDR
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of matching recordings
        """
        results = []

        # Start with band or SDR filter for efficiency
        if band and band in self.by_band:
            recording_ids = self.by_band[band]
        elif sdr_id and sdr_id in self.by_sdr:
            recording_ids = self.by_sdr[sdr_id]
        else:
            recording_ids = self.metadata.keys()

        for recording_id in recording_ids:
            metadata = self.metadata[recording_id]

            # Apply filters
            if band and metadata.band != band:
                continue
            if sdr_id and metadata.sdr_id != sdr_id:
                continue
            if start_time and metadata.timestamp < start_time:
                continue
            if end_time and metadata.timestamp > end_time:
                continue

            results.append(metadata)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata statistics.

        Returns:
            Statistics dictionary
        """
        if not self.metadata:
            return {
                "total_recordings": 0,
                "total_duration_hours": 0,
                "total_size_gb": 0,
                "bands": {},
                "sdrs": {}
            }

        total_duration = sum(m.duration_seconds for m in self.metadata.values())
        total_size = sum(m.file_size for m in self.metadata.values())
        total_ft8 = sum(m.ft8_count for m in self.metadata.values())
        total_wspr = sum(m.wspr_count for m in self.metadata.values())

        band_stats = {}
        for band, ids in self.by_band.items():
            band_stats[band] = {
                "count": len(ids),
                "duration_hours": sum(self.metadata[id].duration_seconds for id in ids) / 3600
            }

        sdr_stats = {}
        for sdr, ids in self.by_sdr.items():
            sdr_stats[sdr] = {
                "count": len(ids),
                "duration_hours": sum(self.metadata[id].duration_seconds for id in ids) / 3600
            }

        return {
            "total_recordings": len(self.metadata),
            "total_duration_hours": total_duration / 3600,
            "total_size_gb": total_size / 1024 / 1024 / 1024,
            "total_ft8": total_ft8,
            "total_wspr": total_wspr,
            "bands": band_stats,
            "sdrs": sdr_stats
        }

    def export_json(self, file_path: str) -> None:
        """Export metadata to JSON.

        Args:
            file_path: Output file path
        """
        data = {
            recording_id: asdict(metadata)
            for recording_id, metadata in self.metadata.items()
        }

        # Convert datetime objects
        for record in data.values():
            if isinstance(record["timestamp"], datetime):
                record["timestamp"] = record["timestamp"].isoformat()

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(data)} records to {file_path}")

    def import_json(self, file_path: str) -> int:
        """Import metadata from JSON.

        Args:
            file_path: Input file path

        Returns:
            Number of records imported
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        count = 0
        for recording_id, record in data.items():
            # Convert timestamp
            if "timestamp" in record:
                record["timestamp"] = datetime.fromisoformat(record["timestamp"])

            metadata = RecordingMetadata(**record)
            self.add_recording(metadata)
            count += 1

        logger.info(f"Imported {count} records from {file_path}")
        return count
