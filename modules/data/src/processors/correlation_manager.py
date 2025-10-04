"""Correlation manager for preserving sample relationships across processing stages."""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import numpy as np
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetadata:
    """Metadata for correlation tracking."""
    recording_id: str
    sample_id: str
    timestamp: datetime
    frequency: float
    band: str
    location: Dict[str, float]
    processing_chain: List[str]
    parent_samples: List[str]
    child_samples: List[str]
    correlation_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class CorrelationManager:
    """Manages correlation preservation across QRN processing stages."""

    def __init__(self, redis_client=None):
        """Initialize correlation manager.

        Args:
            redis_client: Optional Redis client for distributed correlation tracking
        """
        self.redis_client = redis_client
        self._local_cache: Dict[str, CorrelationMetadata] = {}
        self._correlation_index: Dict[str, List[str]] = {}

    def create_correlation_id(
        self,
        recording_id: str,
        timestamp: datetime,
        frequency: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create unique correlation ID for a sample.

        Args:
            recording_id: Recording session ID
            timestamp: Sample timestamp
            frequency: Center frequency
            additional_data: Optional additional data for ID generation

        Returns:
            Unique correlation ID
        """
        # Create deterministic ID based on sample properties
        id_data = {
            'recording_id': recording_id,
            'timestamp': timestamp.isoformat(),
            'frequency': frequency
        }

        if additional_data:
            id_data.update(additional_data)

        id_string = json.dumps(id_data, sort_keys=True)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def register_sample(
        self,
        sample_id: str,
        recording_id: str,
        timestamp: datetime,
        frequency: float,
        band: str,
        location: Dict[str, float],
        processing_stage: str,
        parent_samples: Optional[List[str]] = None
    ) -> CorrelationMetadata:
        """Register a new sample in the correlation system.

        Args:
            sample_id: Unique sample identifier
            recording_id: Recording session ID
            timestamp: Sample timestamp
            frequency: Center frequency
            band: Frequency band
            location: Geographic location
            processing_stage: Current processing stage
            parent_samples: Parent sample IDs if derived

        Returns:
            CorrelationMetadata object
        """
        metadata = CorrelationMetadata(
            recording_id=recording_id,
            sample_id=sample_id,
            timestamp=timestamp,
            frequency=frequency,
            band=band,
            location=location,
            processing_chain=[processing_stage],
            parent_samples=parent_samples or [],
            child_samples=[],
            correlation_scores={}
        )

        # Store in local cache
        self._local_cache[sample_id] = metadata

        # Update parent-child relationships
        if parent_samples:
            for parent_id in parent_samples:
                if parent_id in self._local_cache:
                    self._local_cache[parent_id].child_samples.append(sample_id)

        # Store in Redis if available
        if self.redis_client:
            self._store_in_redis(sample_id, metadata)

        # Update correlation index
        if recording_id not in self._correlation_index:
            self._correlation_index[recording_id] = []
        self._correlation_index[recording_id].append(sample_id)

        logger.debug(f"Registered sample {sample_id} with {len(parent_samples or [])} parents")

        return metadata

    def update_processing_chain(
        self,
        sample_id: str,
        processing_stage: str,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update processing chain for a sample.

        Args:
            sample_id: Sample identifier
            processing_stage: New processing stage
            metadata_update: Optional metadata updates

        Returns:
            True if successful
        """
        if sample_id not in self._local_cache:
            if self.redis_client:
                metadata = self._load_from_redis(sample_id)
                if not metadata:
                    logger.warning(f"Sample {sample_id} not found")
                    return False
            else:
                logger.warning(f"Sample {sample_id} not found")
                return False
        else:
            metadata = self._local_cache[sample_id]

        # Add processing stage
        metadata.processing_chain.append(processing_stage)

        # Apply metadata updates
        if metadata_update:
            for key, value in metadata_update.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        # Update storage
        if self.redis_client:
            self._store_in_redis(sample_id, metadata)

        return True

    def calculate_correlation(
        self,
        sample_id1: str,
        sample_id2: str,
        correlation_type: str = "temporal"
    ) -> float:
        """Calculate correlation between two samples.

        Args:
            sample_id1: First sample ID
            sample_id2: Second sample ID
            correlation_type: Type of correlation (temporal, spectral, etc.)

        Returns:
            Correlation score between 0 and 1
        """
        metadata1 = self._get_metadata(sample_id1)
        metadata2 = self._get_metadata(sample_id2)

        if not metadata1 or not metadata2:
            return 0.0

        score = 0.0

        if correlation_type == "temporal":
            # Calculate temporal proximity
            time_diff = abs((metadata1.timestamp - metadata2.timestamp).total_seconds())
            # Exponential decay with 60-second half-life
            score = np.exp(-time_diff / 60.0)

        elif correlation_type == "spectral":
            # Calculate frequency proximity
            freq_diff = abs(metadata1.frequency - metadata2.frequency)
            # Exponential decay with 1kHz half-life
            score = np.exp(-freq_diff / 1000.0)

        elif correlation_type == "spatial":
            # Calculate geographic proximity
            if metadata1.location and metadata2.location:
                lat_diff = abs(metadata1.location.get('lat', 0) - metadata2.location.get('lat', 0))
                lon_diff = abs(metadata1.location.get('lon', 0) - metadata2.location.get('lon', 0))
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                # Exponential decay with 10-degree half-life
                score = np.exp(-distance / 10.0)

        # Store correlation score
        if sample_id1 in self._local_cache:
            self._local_cache[sample_id1].correlation_scores[sample_id2] = score
        if sample_id2 in self._local_cache:
            self._local_cache[sample_id2].correlation_scores[sample_id1] = score

        return score

    def get_correlated_samples(
        self,
        sample_id: str,
        min_correlation: float = 0.5,
        correlation_types: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get samples correlated with the given sample.

        Args:
            sample_id: Sample identifier
            min_correlation: Minimum correlation threshold
            correlation_types: Types of correlation to consider

        Returns:
            List of (sample_id, correlation_score) tuples
        """
        metadata = self._get_metadata(sample_id)
        if not metadata:
            return []

        correlation_types = correlation_types or ["temporal", "spectral"]
        correlated = []

        # Check samples in the same recording
        recording_samples = self._correlation_index.get(metadata.recording_id, [])

        for other_id in recording_samples:
            if other_id == sample_id:
                continue

            max_score = 0.0
            for corr_type in correlation_types:
                score = self.calculate_correlation(sample_id, other_id, corr_type)
                max_score = max(max_score, score)

            if max_score >= min_correlation:
                correlated.append((other_id, max_score))

        # Sort by correlation score
        correlated.sort(key=lambda x: x[1], reverse=True)

        return correlated

    def get_lineage(self, sample_id: str) -> Dict[str, Any]:
        """Get complete lineage for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary with parent and child relationships
        """
        metadata = self._get_metadata(sample_id)
        if not metadata:
            return {}

        lineage = {
            'sample_id': sample_id,
            'parents': metadata.parent_samples,
            'children': metadata.child_samples,
            'processing_chain': metadata.processing_chain,
            'generation': self._calculate_generation(sample_id)
        }

        # Recursively get parent lineage
        if metadata.parent_samples:
            lineage['parent_lineage'] = [
                self.get_lineage(parent_id)
                for parent_id in metadata.parent_samples
            ]

        return lineage

    def _get_metadata(self, sample_id: str) -> Optional[CorrelationMetadata]:
        """Get metadata for a sample."""
        if sample_id in self._local_cache:
            return self._local_cache[sample_id]

        if self.redis_client:
            return self._load_from_redis(sample_id)

        return None

    def _calculate_generation(self, sample_id: str, visited: Optional[set] = None) -> int:
        """Calculate generation depth for a sample."""
        if visited is None:
            visited = set()

        if sample_id in visited:
            return 0

        visited.add(sample_id)

        metadata = self._get_metadata(sample_id)
        if not metadata or not metadata.parent_samples:
            return 0

        max_parent_gen = 0
        for parent_id in metadata.parent_samples:
            parent_gen = self._calculate_generation(parent_id, visited)
            max_parent_gen = max(max_parent_gen, parent_gen)

        return max_parent_gen + 1

    def _store_in_redis(self, sample_id: str, metadata: CorrelationMetadata) -> None:
        """Store metadata in Redis."""
        if not self.redis_client:
            return

        try:
            key = f"correlation:{sample_id}"
            self.redis_client.set(
                key,
                json.dumps(metadata.to_dict()),
                ex=86400  # 24-hour TTL
            )
        except Exception as e:
            logger.error(f"Failed to store correlation in Redis: {e}")

    def _load_from_redis(self, sample_id: str) -> Optional[CorrelationMetadata]:
        """Load metadata from Redis."""
        if not self.redis_client:
            return None

        try:
            key = f"correlation:{sample_id}"
            data = self.redis_client.get(key)
            if data:
                metadata_dict = json.loads(data)
                metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                return CorrelationMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Failed to load correlation from Redis: {e}")

        return None

    def flush_to_storage(self) -> int:
        """Flush local cache to persistent storage.

        Returns:
            Number of items flushed
        """
        count = 0
        if self.redis_client:
            for sample_id, metadata in self._local_cache.items():
                self._store_in_redis(sample_id, metadata)
                count += 1

        logger.info(f"Flushed {count} correlation entries to storage")
        return count