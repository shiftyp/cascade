"""Hybrid SDR selector for optimal source selection."""

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SDRType(Enum):
    """SDR source types."""
    KIWISDR = "kiwisdr"
    WEBSDR = "websdr"
    RTLSDR = "rtlsdr"
    HACKRF = "hackrf"


@dataclass
class SDRMetrics:
    """Metrics for SDR performance."""
    latency_ms: float
    packet_loss: float
    snr_db: float
    bandwidth_khz: float
    users_connected: int
    cpu_usage: float
    uptime_hours: float
    error_rate: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SDRSource:
    """SDR source information."""
    id: str
    name: str
    type: SDRType
    url: str
    location: Dict[str, float]  # lat, lon, altitude
    frequency_range: Tuple[float, float]
    sample_rate: int
    antenna_type: str
    online: bool = True
    priority: int = 50  # 0-100, higher is better
    metrics: SDRMetrics = field(default_factory=lambda: SDRMetrics(
        latency_ms=0, packet_loss=0, snr_db=0, bandwidth_khz=0,
        users_connected=0, cpu_usage=0, uptime_hours=0, error_rate=0
    ))
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_limit_minutes: int = 90
    usage_today_minutes: float = 0
    last_connected: Optional[datetime] = None


class SelectionStrategy(Enum):
    """SDR selection strategies."""
    BEST_QUALITY = "best_quality"
    LOWEST_LATENCY = "lowest_latency"
    GEOGRAPHIC_DIVERSITY = "geographic_diversity"
    LOAD_BALANCED = "load_balanced"
    FREQUENCY_OPTIMIZED = "frequency_optimized"
    ROUND_ROBIN = "round_robin"


class HybridSDRSelector:
    """Intelligent SDR source selector."""

    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.BEST_QUALITY):
        """Initialize SDR selector.

        Args:
            strategy: Selection strategy to use
        """
        self.strategy = strategy
        self.sources: Dict[str, SDRSource] = {}
        self.active_connections: Dict[str, str] = {}  # session_id -> source_id
        self.rotation_index = 0
        self.performance_history: Dict[str, List[SDRMetrics]] = {}
        self._lock = asyncio.Lock()

    async def register_source(self, source: SDRSource) -> None:
        """Register an SDR source.

        Args:
            source: SDR source to register
        """
        async with self._lock:
            self.sources[source.id] = source
            self.performance_history[source.id] = []
            logger.info(f"Registered SDR source: {source.name} ({source.type.value})")

    async def update_metrics(self, source_id: str, metrics: SDRMetrics) -> None:
        """Update metrics for an SDR source.

        Args:
            source_id: Source identifier
            metrics: Updated metrics
        """
        async with self._lock:
            if source_id in self.sources:
                self.sources[source_id].metrics = metrics

                # Keep history
                history = self.performance_history[source_id]
                history.append(metrics)

                # Keep only last 100 entries
                if len(history) > 100:
                    history.pop(0)

                logger.debug(f"Updated metrics for {source_id}: "
                           f"latency={metrics.latency_ms}ms, SNR={metrics.snr_db}dB")

    async def select_source(
        self,
        frequency: float,
        bandwidth: float = 12000,
        location_preference: Optional[Dict[str, float]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> Optional[SDRSource]:
        """Select optimal SDR source.

        Args:
            frequency: Target frequency in Hz
            bandwidth: Required bandwidth in Hz
            location_preference: Preferred location (lat, lon)
            exclude_ids: Source IDs to exclude

        Returns:
            Selected SDR source or None if none available
        """
        async with self._lock:
            # Filter available sources
            candidates = await self._filter_candidates(
                frequency, bandwidth, exclude_ids
            )

            if not candidates:
                logger.warning(f"No suitable SDR sources for {frequency} Hz")
                return None

            # Apply selection strategy
            selected = await self._apply_strategy(
                candidates, frequency, location_preference
            )

            if selected:
                # Update usage tracking
                selected.last_connected = datetime.now(timezone.utc)
                logger.info(f"Selected {selected.name} for {frequency} Hz "
                          f"using {self.strategy.value} strategy")

            return selected

    async def _filter_candidates(
        self,
        frequency: float,
        bandwidth: float,
        exclude_ids: Optional[List[str]]
    ) -> List[SDRSource]:
        """Filter candidate SDR sources.

        Args:
            frequency: Target frequency
            bandwidth: Required bandwidth
            exclude_ids: IDs to exclude

        Returns:
            List of candidate sources
        """
        candidates = []
        exclude_ids = exclude_ids or []

        for source in self.sources.values():
            # Skip if excluded
            if source.id in exclude_ids:
                continue

            # Check if online
            if not source.online:
                continue

            # Check frequency range
            if not (source.frequency_range[0] <= frequency <= source.frequency_range[1]):
                continue

            # Check bandwidth capability
            if source.metrics.bandwidth_khz * 1000 < bandwidth:
                if source.sample_rate < bandwidth:
                    continue

            # Check usage limits (for KiwiSDR)
            if source.type == SDRType.KIWISDR:
                if source.usage_today_minutes >= source.usage_limit_minutes:
                    logger.debug(f"Skipping {source.name}: daily limit reached")
                    continue

            # Check if not overloaded
            if source.metrics.cpu_usage > 90:
                logger.debug(f"Skipping {source.name}: high CPU usage")
                continue

            candidates.append(source)

        return candidates

    async def _apply_strategy(
        self,
        candidates: List[SDRSource],
        frequency: float,
        location_preference: Optional[Dict[str, float]]
    ) -> Optional[SDRSource]:
        """Apply selection strategy.

        Args:
            candidates: Candidate sources
            frequency: Target frequency
            location_preference: Preferred location

        Returns:
            Selected source
        """
        if not candidates:
            return None

        if self.strategy == SelectionStrategy.BEST_QUALITY:
            return self._select_best_quality(candidates)

        elif self.strategy == SelectionStrategy.LOWEST_LATENCY:
            return self._select_lowest_latency(candidates)

        elif self.strategy == SelectionStrategy.GEOGRAPHIC_DIVERSITY:
            return self._select_geographic_diversity(candidates, location_preference)

        elif self.strategy == SelectionStrategy.LOAD_BALANCED:
            return self._select_load_balanced(candidates)

        elif self.strategy == SelectionStrategy.FREQUENCY_OPTIMIZED:
            return self._select_frequency_optimized(candidates, frequency)

        elif self.strategy == SelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates)

        # Default to first available
        return candidates[0]

    def _select_best_quality(self, candidates: List[SDRSource]) -> SDRSource:
        """Select based on signal quality."""
        def quality_score(source: SDRSource) -> float:
            metrics = source.metrics
            # Weighted score: SNR is most important
            score = (
                metrics.snr_db * 3.0 +
                (100 - metrics.packet_loss) * 2.0 +
                (100 - metrics.error_rate) * 1.0 +
                max(0, 100 - metrics.latency_ms) * 0.5
            )
            return score

        return max(candidates, key=quality_score)

    def _select_lowest_latency(self, candidates: List[SDRSource]) -> SDRSource:
        """Select based on lowest latency."""
        return min(candidates, key=lambda s: s.metrics.latency_ms)

    def _select_geographic_diversity(
        self,
        candidates: List[SDRSource],
        location_preference: Optional[Dict[str, float]]
    ) -> SDRSource:
        """Select for geographic diversity."""
        if location_preference:
            # Select closest to preference
            def distance(source: SDRSource) -> float:
                lat_diff = source.location["lat"] - location_preference["lat"]
                lon_diff = source.location["lon"] - location_preference["lon"]
                return np.sqrt(lat_diff**2 + lon_diff**2)

            return min(candidates, key=distance)
        else:
            # Random selection for diversity
            return random.choice(candidates)

    def _select_load_balanced(self, candidates: List[SDRSource]) -> SDRSource:
        """Select based on load balancing."""
        def load_score(source: SDRSource) -> float:
            metrics = source.metrics
            # Lower is better
            return (
                metrics.users_connected * 2.0 +
                metrics.cpu_usage +
                (100 - source.priority)
            )

        return min(candidates, key=load_score)

    def _select_frequency_optimized(
        self,
        candidates: List[SDRSource],
        frequency: float
    ) -> SDRSource:
        """Select optimized for specific frequency."""
        def freq_score(source: SDRSource) -> float:
            # Check if frequency is in optimal range
            range_size = source.frequency_range[1] - source.frequency_range[0]
            center = (source.frequency_range[0] + source.frequency_range[1]) / 2

            # Distance from center normalized
            distance = abs(frequency - center) / (range_size / 2)

            # Combine with quality metrics
            return (
                (1 - distance) * 100 +
                source.metrics.snr_db +
                source.priority
            )

        return max(candidates, key=freq_score)

    def _select_round_robin(self, candidates: List[SDRSource]) -> SDRSource:
        """Select using round-robin."""
        selected = candidates[self.rotation_index % len(candidates)]
        self.rotation_index += 1
        return selected

    async def mark_source_offline(self, source_id: str, reason: str = "") -> None:
        """Mark source as offline.

        Args:
            source_id: Source identifier
            reason: Reason for marking offline
        """
        async with self._lock:
            if source_id in self.sources:
                self.sources[source_id].online = False
                logger.warning(f"Marked {source_id} as offline: {reason}")

    async def mark_source_online(self, source_id: str) -> None:
        """Mark source as online.

        Args:
            source_id: Source identifier
        """
        async with self._lock:
            if source_id in self.sources:
                self.sources[source_id].online = True
                logger.info(f"Marked {source_id} as online")

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self.sources)
        online = sum(1 for s in self.sources.values() if s.online)

        type_counts = {}
        for source in self.sources.values():
            type_name = source.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        avg_latency = np.mean([
            s.metrics.latency_ms for s in self.sources.values()
            if s.metrics.latency_ms > 0
        ]) if self.sources else 0

        avg_snr = np.mean([
            s.metrics.snr_db for s in self.sources.values()
            if s.metrics.snr_db > 0
        ]) if self.sources else 0

        return {
            "total_sources": total,
            "online_sources": online,
            "active_connections": len(self.active_connections),
            "types": type_counts,
            "average_latency_ms": round(avg_latency, 2),
            "average_snr_db": round(avg_snr, 2),
            "strategy": self.strategy.value
        }

    async def cleanup_stale_connections(self, timeout_hours: int = 24) -> int:
        """Clean up stale connections.

        Args:
            timeout_hours: Hours before considering connection stale

        Returns:
            Number of connections cleaned
        """
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
            cleaned = 0

            for source in self.sources.values():
                if source.last_connected and source.last_connected < cutoff:
                    source.usage_today_minutes = 0
                    cleaned += 1

            if cleaned:
                logger.info(f"Cleaned {cleaned} stale connections")

            return cleaned