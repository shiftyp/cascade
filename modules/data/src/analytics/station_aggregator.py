"""Privacy-safe aggregation for station statistics.

T078: Aggregate station data while preserving privacy through k-anonymity,
differential privacy, and statistical obfuscation.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AggregatedStatistics:
    """Privacy-safe aggregated statistics."""

    timestamp: datetime
    aggregation_level: str  # 'grid', 'band', 'global'

    # Station counts (with k-anonymity)
    total_stations: int
    active_stations: int
    new_stations: int

    # Activity statistics (with noise)
    avg_observations_per_station: float
    median_observations: float
    avg_duty_cycle: float

    # Signal statistics (aggregated)
    avg_snr_db: float
    snr_distribution: Dict[str, int]  # Binned distribution
    band_distribution: Dict[str, int]

    # Geographic distribution (grid square level only)
    grid_distribution: Dict[str, int]  # 4-char grids only
    distance_distribution: Dict[str, int]  # Binned distances

    # Privacy parameters
    k_anonymity: int
    epsilon: float  # Differential privacy parameter
    suppressed_count: int  # Number of suppressed entries


class PrivacySafeAggregator:
    """Aggregates station data with privacy protection."""

    def __init__(self, k_anonymity: int = 5, epsilon: float = 1.0):
        """Initialize aggregator with privacy parameters.

        Args:
            k_anonymity: Minimum group size for reporting
            epsilon: Differential privacy parameter (lower = more privacy)
        """
        self.k_anonymity = k_anonymity
        self.epsilon = epsilon
        self.aggregated_stats: List[AggregatedStatistics] = []

    def aggregate_stations(self, station_data: List[Dict]) -> AggregatedStatistics:
        """Aggregate station data with privacy protection.

        Args:
            station_data: List of station fingerprint dictionaries

        Returns:
            Privacy-safe aggregated statistics
        """
        if len(station_data) < self.k_anonymity:
            logger.warning(f"Insufficient data for aggregation: {len(station_data)} < {self.k_anonymity}")
            return self._create_empty_stats()

        # Apply k-anonymity filtering
        filtered_data = self._apply_k_anonymity(station_data)

        # Calculate basic statistics
        total_stations = len(filtered_data)
        active_stations = self._count_active_stations(filtered_data)
        new_stations = self._count_new_stations(filtered_data)

        # Add differential privacy noise
        total_stations = self._add_laplace_noise(total_stations, sensitivity=1)
        active_stations = self._add_laplace_noise(active_stations, sensitivity=1)
        new_stations = self._add_laplace_noise(new_stations, sensitivity=1)

        # Calculate activity statistics
        observations = [s.get('total_observations', 0) for s in filtered_data]
        avg_observations = self._noisy_mean(observations)
        median_observations = self._noisy_median(observations)

        duty_cycles = [s.get('duty_cycle', 0) for s in filtered_data]
        avg_duty_cycle = self._noisy_mean(duty_cycles)

        # Calculate signal statistics
        snr_values = [s.get('avg_snr_db', 0) for s in filtered_data]
        avg_snr = self._noisy_mean(snr_values)
        snr_distribution = self._create_binned_distribution(snr_values, 'snr')

        # Band distribution
        band_distribution = self._aggregate_band_usage(filtered_data)

        # Geographic distribution (4-char grids only)
        grid_distribution = self._aggregate_grid_distribution(filtered_data)

        # Distance distribution (binned)
        distances = [s.get('max_distance_km', 0) for s in filtered_data]
        distance_distribution = self._create_binned_distribution(distances, 'distance')

        # Count suppressed entries
        suppressed_count = len(station_data) - len(filtered_data)

        stats = AggregatedStatistics(
            timestamp=datetime.now(),
            aggregation_level='global',
            total_stations=max(0, int(total_stations)),
            active_stations=max(0, int(active_stations)),
            new_stations=max(0, int(new_stations)),
            avg_observations_per_station=avg_observations,
            median_observations=median_observations,
            avg_duty_cycle=avg_duty_cycle,
            avg_snr_db=avg_snr,
            snr_distribution=snr_distribution,
            band_distribution=band_distribution,
            grid_distribution=grid_distribution,
            distance_distribution=distance_distribution,
            k_anonymity=self.k_anonymity,
            epsilon=self.epsilon,
            suppressed_count=suppressed_count
        )

        self.aggregated_stats.append(stats)
        return stats

    def _create_empty_stats(self) -> AggregatedStatistics:
        """Create empty statistics when insufficient data."""
        return AggregatedStatistics(
            timestamp=datetime.now(),
            aggregation_level='global',
            total_stations=0,
            active_stations=0,
            new_stations=0,
            avg_observations_per_station=0.0,
            median_observations=0.0,
            avg_duty_cycle=0.0,
            avg_snr_db=0.0,
            snr_distribution={},
            band_distribution={},
            grid_distribution={},
            distance_distribution={},
            k_anonymity=self.k_anonymity,
            epsilon=self.epsilon,
            suppressed_count=0
        )

    def _apply_k_anonymity(self, station_data: List[Dict]) -> List[Dict]:
        """Apply k-anonymity filtering to station data.

        Args:
            station_data: Original station data

        Returns:
            Filtered data meeting k-anonymity requirements
        """
        # Group stations by quasi-identifiers
        groups = defaultdict(list)

        for station in station_data:
            # Create quasi-identifier from attributes
            # Use coarse-grained attributes to create groups
            qi_parts = []

            # 4-character grid (not 6)
            grid = station.get('primary_grid', '')[:4]
            qi_parts.append(grid)

            # Binned SNR
            snr = station.get('avg_snr_db', 0)
            snr_bin = self._get_snr_bin(snr)
            qi_parts.append(snr_bin)

            # Primary band
            bands = station.get('primary_bands', [])
            primary_band = bands[0] if bands else 'unknown'
            qi_parts.append(primary_band)

            # Activity level (binned)
            obs = station.get('total_observations', 0)
            activity_bin = self._get_activity_bin(obs)
            qi_parts.append(activity_bin)

            quasi_identifier = '|'.join(qi_parts)
            groups[quasi_identifier].append(station)

        # Filter groups that meet k-anonymity
        filtered = []
        for group_stations in groups.values():
            if len(group_stations) >= self.k_anonymity:
                filtered.extend(group_stations)

        logger.debug(f"K-anonymity filtering: {len(station_data)} -> {len(filtered)} stations")
        return filtered

    def _add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for differential privacy.

        Args:
            value: Original value
            sensitivity: Query sensitivity

        Returns:
            Value with noise added
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def _noisy_mean(self, values: List[float]) -> float:
        """Calculate mean with differential privacy noise."""
        if not values:
            return 0.0

        mean = np.mean(values)
        # Sensitivity is range/n for mean
        if len(values) > 1:
            sensitivity = (max(values) - min(values)) / len(values)
        else:
            sensitivity = 1.0

        return self._add_laplace_noise(mean, sensitivity)

    def _noisy_median(self, values: List[float]) -> float:
        """Calculate median with differential privacy noise."""
        if not values:
            return 0.0

        median = np.median(values)
        # Sensitivity for median is harder to bound, use conservative estimate
        sensitivity = (max(values) - min(values)) / len(values) if values else 1.0

        return self._add_laplace_noise(median, sensitivity)

    def _count_active_stations(self, station_data: List[Dict]) -> int:
        """Count recently active stations."""
        cutoff = datetime.now() - timedelta(days=7)
        count = 0

        for station in station_data:
            last_seen = station.get('last_seen')
            if last_seen:
                if isinstance(last_seen, str):
                    last_seen = datetime.fromisoformat(last_seen)
                if last_seen >= cutoff:
                    count += 1

        return count

    def _count_new_stations(self, station_data: List[Dict]) -> int:
        """Count new stations (first seen in last 30 days)."""
        cutoff = datetime.now() - timedelta(days=30)
        count = 0

        for station in station_data:
            first_seen = station.get('first_seen')
            if first_seen:
                if isinstance(first_seen, str):
                    first_seen = datetime.fromisoformat(first_seen)
                if first_seen >= cutoff:
                    count += 1

        return count

    def _create_binned_distribution(self, values: List[float],
                                   dist_type: str) -> Dict[str, int]:
        """Create binned distribution with k-anonymity.

        Args:
            values: Raw values
            dist_type: 'snr' or 'distance'

        Returns:
            Binned distribution with counts
        """
        if not values:
            return {}

        bins = []
        if dist_type == 'snr':
            bins = [(-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)]
        elif dist_type == 'distance':
            bins = [(0, 100), (100, 500), (500, 1000), (1000, 5000),
                   (5000, 10000), (10000, 20000)]

        distribution = {}
        for low, high in bins:
            count = sum(1 for v in values if low <= v < high)
            # Apply k-anonymity threshold
            if count >= self.k_anonymity:
                # Add noise
                count = int(self._add_laplace_noise(count, sensitivity=1))
                distribution[f"{low}-{high}"] = max(0, count)

        return distribution

    def _aggregate_band_usage(self, station_data: List[Dict]) -> Dict[str, int]:
        """Aggregate band usage with privacy protection."""
        band_counts = defaultdict(int)

        for station in station_data:
            bands = station.get('primary_bands', [])
            for band in bands:
                band_counts[band] += 1

        # Apply k-anonymity and noise
        protected_counts = {}
        for band, count in band_counts.items():
            if count >= self.k_anonymity:
                noisy_count = int(self._add_laplace_noise(count, sensitivity=1))
                protected_counts[band] = max(0, noisy_count)

        return protected_counts

    def _aggregate_grid_distribution(self, station_data: List[Dict]) -> Dict[str, int]:
        """Aggregate grid distribution (4-char only for privacy)."""
        grid_counts = defaultdict(int)

        for station in station_data:
            grid = station.get('primary_grid', '')
            if len(grid) >= 4:
                # Use only 4-character grid for privacy
                grid_4char = grid[:4]
                grid_counts[grid_4char] += 1

        # Apply k-anonymity and noise
        protected_counts = {}
        for grid, count in grid_counts.items():
            if count >= self.k_anonymity:
                noisy_count = int(self._add_laplace_noise(count, sensitivity=1))
                protected_counts[grid] = max(0, noisy_count)

        return protected_counts

    def _get_snr_bin(self, snr: float) -> str:
        """Get SNR bin for quasi-identifier."""
        if snr < -20:
            return 'very_low'
        elif snr < -10:
            return 'low'
        elif snr < 0:
            return 'medium'
        elif snr < 10:
            return 'high'
        else:
            return 'very_high'

    def _get_activity_bin(self, observations: int) -> str:
        """Get activity bin for quasi-identifier."""
        if observations < 10:
            return 'minimal'
        elif observations < 100:
            return 'low'
        elif observations < 1000:
            return 'medium'
        elif observations < 10000:
            return 'high'
        else:
            return 'very_high'

    def aggregate_by_grid(self, station_data: List[Dict]) -> Dict[str, AggregatedStatistics]:
        """Aggregate statistics by grid square (4-char only).

        Args:
            station_data: Station data to aggregate

        Returns:
            Dictionary of grid -> statistics
        """
        grid_groups = defaultdict(list)

        # Group by 4-character grid
        for station in station_data:
            grid = station.get('primary_grid', '')[:4]
            if grid:
                grid_groups[grid].append(station)

        grid_stats = {}

        for grid, stations in grid_groups.items():
            # Only report if meets k-anonymity
            if len(stations) >= self.k_anonymity:
                stats = self.aggregate_stations(stations)
                stats.aggregation_level = f'grid:{grid}'
                grid_stats[grid] = stats

        return grid_stats

    def aggregate_by_band(self, station_data: List[Dict]) -> Dict[str, AggregatedStatistics]:
        """Aggregate statistics by band.

        Args:
            station_data: Station data to aggregate

        Returns:
            Dictionary of band -> statistics
        """
        band_groups = defaultdict(list)

        # Group by primary band
        for station in station_data:
            bands = station.get('primary_bands', [])
            if bands:
                primary_band = bands[0]
                band_groups[primary_band].append(station)

        band_stats = {}

        for band, stations in band_groups.items():
            # Only report if meets k-anonymity
            if len(stations) >= self.k_anonymity:
                stats = self.aggregate_stations(stations)
                stats.aggregation_level = f'band:{band}'
                band_stats[band] = stats

        return band_stats

    def create_time_series(self, station_data: List[Dict],
                         window_hours: int = 24) -> List[AggregatedStatistics]:
        """Create time series of aggregated statistics.

        Args:
            station_data: Station data
            window_hours: Time window for aggregation

        Returns:
            List of statistics over time
        """
        # Sort stations by last_seen time
        sorted_data = sorted(station_data,
                           key=lambda s: s.get('last_seen', datetime.min))

        if not sorted_data:
            return []

        # Determine time range
        first_time = sorted_data[0].get('first_seen', datetime.now())
        last_time = sorted_data[-1].get('last_seen', datetime.now())

        if isinstance(first_time, str):
            first_time = datetime.fromisoformat(first_time)
        if isinstance(last_time, str):
            last_time = datetime.fromisoformat(last_time)

        # Create time windows
        time_series = []
        current_time = first_time

        while current_time <= last_time:
            window_end = current_time + timedelta(hours=window_hours)

            # Filter stations active in this window
            window_stations = []
            for station in sorted_data:
                station_first = station.get('first_seen')
                station_last = station.get('last_seen')

                if isinstance(station_first, str):
                    station_first = datetime.fromisoformat(station_first)
                if isinstance(station_last, str):
                    station_last = datetime.fromisoformat(station_last)

                # Check if station was active in this window
                if station_first <= window_end and station_last >= current_time:
                    window_stations.append(station)

            # Aggregate if sufficient data
            if len(window_stations) >= self.k_anonymity:
                stats = self.aggregate_stations(window_stations)
                stats.timestamp = current_time
                time_series.append(stats)

            current_time = window_end

        return time_series

    def export_safe_statistics(self, stats: AggregatedStatistics) -> Dict[str, Any]:
        """Export statistics in privacy-safe format.

        Args:
            stats: Aggregated statistics

        Returns:
            Dictionary safe for export
        """
        return {
            'timestamp': stats.timestamp.isoformat(),
            'aggregation_level': stats.aggregation_level,
            'total_stations': stats.total_stations,
            'active_stations': stats.active_stations,
            'avg_snr_db': round(stats.avg_snr_db, 1),
            'avg_duty_cycle': round(stats.avg_duty_cycle, 1),
            'band_distribution': stats.band_distribution,
            'grid_distribution': stats.grid_distribution,
            'privacy_params': {
                'k_anonymity': stats.k_anonymity,
                'epsilon': stats.epsilon,
                'suppressed_count': stats.suppressed_count
            }
        }