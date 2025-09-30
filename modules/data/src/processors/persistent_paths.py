"""Persistent path characterization using TX-RX hash pairs.

T074: Track and characterize propagation paths between anonymous stations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class PropagationPath:
    """Characterized propagation path between two stations."""

    tx_hash: str  # Transmitter station hash
    rx_hash: str  # Receiver station hash
    tx_grid: str  # TX grid square (preserved)
    rx_grid: str  # RX grid square (preserved)

    # Path geometry
    distance_km: float
    bearing_degrees: float
    reverse_bearing_degrees: float
    midpoint_grid: str

    # Signal statistics
    observation_count: int = 0
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None

    snr_history: List[float] = field(default_factory=list)
    avg_snr_db: float = 0
    max_snr_db: float = -100
    min_snr_db: float = 100
    snr_variance: float = 0

    # Propagation characteristics
    modes_observed: Dict[str, int] = field(default_factory=dict)  # Mode -> count
    best_hours_utc: List[int] = field(default_factory=list)
    worst_hours_utc: List[int] = field(default_factory=list)
    seasonal_performance: Dict[str, float] = field(default_factory=dict)  # Month -> avg SNR

    # Success metrics
    total_attempts: int = 0
    successful_decodes: int = 0
    decode_success_rate: float = 0
    bidirectional: bool = False

    # Time patterns
    median_duration_sec: float = 0
    typical_modes: List[str] = field(default_factory=list)  # FT8, WSPR, etc.

    # Environmental correlation
    solar_flux_correlation: Optional[float] = None
    k_index_correlation: Optional[float] = None


class PersistentPathTracker:
    """Tracks and characterizes persistent propagation paths."""

    def __init__(self):
        """Initialize path tracker."""
        self.paths: Dict[Tuple[str, str], PropagationPath] = {}
        self.reverse_paths: Dict[Tuple[str, str], bool] = {}  # Track bidirectional
        self.grid_calculator = GridCalculator()

    def record_observation(self, tx_hash: str, rx_hash: str,
                          tx_grid: str, rx_grid: str,
                          snr: float, timestamp: datetime,
                          mode: str = 'FT8') -> Optional[PropagationPath]:
        """Record an observation and update path characterization.

        Args:
            tx_hash: Transmitter station hash
            rx_hash: Receiver station hash
            tx_grid: TX grid square
            rx_grid: RX grid square
            snr: Signal-to-noise ratio
            timestamp: Observation timestamp
            mode: Operating mode

        Returns:
            Updated PropagationPath or None
        """
        qso_data = {
            'tx_hash': tx_hash,
            'rx_hash': rx_hash,
            'tx_grid': tx_grid,
            'rx_grid': rx_grid,
            'snr': snr,
            'mode': mode,
            'timestamp': timestamp,
            'successful': True
        }
        return self.record_qso(qso_data)

    def record_qso(self, qso_data: Dict[str, Any]) -> Optional[PropagationPath]:
        """Record a QSO and update path characterization.

        Args:
            qso_data: QSO data including:
                - tx_hash: Transmitter station hash
                - rx_hash: Receiver station hash
                - tx_grid: TX grid square
                - rx_grid: RX grid square
                - snr: Signal-to-noise ratio
                - mode: Operating mode (FT8, WSPR, etc.)
                - timestamp: QSO timestamp
                - successful: Whether decode was successful

        Returns:
            Updated PropagationPath or None
        """
        tx_hash = qso_data.get('tx_hash')
        rx_hash = qso_data.get('rx_hash')

        if not tx_hash or not rx_hash:
            return None

        # Create path key
        path_key = (tx_hash, rx_hash)

        # Get or create path
        if path_key not in self.paths:
            path = self._create_path(qso_data)
            if not path:
                return None
            self.paths[path_key] = path
        else:
            path = self.paths[path_key]

        # Update path with observation
        self._update_path(path, qso_data)

        # Check for bidirectional communication
        reverse_key = (rx_hash, tx_hash)
        if reverse_key in self.paths:
            path.bidirectional = True
            self.paths[reverse_key].bidirectional = True

        return path

    def _create_path(self, qso_data: Dict[str, Any]) -> Optional[PropagationPath]:
        """Create new propagation path.

        Args:
            qso_data: QSO data

        Returns:
            New PropagationPath or None
        """
        tx_grid = qso_data.get('tx_grid')
        rx_grid = qso_data.get('rx_grid')

        if not tx_grid or not rx_grid:
            return None

        # Calculate path geometry
        distance_km = self.grid_calculator.calculate_distance(tx_grid, rx_grid)
        bearing = self.grid_calculator.calculate_bearing(tx_grid, rx_grid)
        reverse_bearing = self.grid_calculator.calculate_bearing(rx_grid, tx_grid)
        midpoint_grid = self.grid_calculator.calculate_midpoint_grid(tx_grid, rx_grid)

        timestamp = qso_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return PropagationPath(
            tx_hash=qso_data['tx_hash'],
            rx_hash=qso_data['rx_hash'],
            tx_grid=tx_grid,
            rx_grid=rx_grid,
            distance_km=distance_km,
            bearing_degrees=bearing,
            reverse_bearing_degrees=reverse_bearing,
            midpoint_grid=midpoint_grid,
            first_observed=timestamp,
            last_observed=timestamp
        )

    def _update_path(self, path: PropagationPath, qso_data: Dict[str, Any]):
        """Update path with new observation.

        Args:
            path: PropagationPath to update
            qso_data: New QSO data
        """
        timestamp = qso_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Update observation count and timestamps
        path.observation_count += 1
        path.last_observed = timestamp

        # Update SNR statistics
        snr = qso_data.get('snr', 0)
        path.snr_history.append(snr)
        if len(path.snr_history) > 100:  # Keep last 100 observations
            path.snr_history.pop(0)

        # Recalculate statistics
        path.avg_snr_db = np.mean(path.snr_history)
        path.max_snr_db = max(path.max_snr_db, snr)
        path.min_snr_db = min(path.min_snr_db, snr)
        path.snr_variance = np.var(path.snr_history) if len(path.snr_history) > 1 else 0

        # Update mode statistics
        mode = qso_data.get('mode', 'UNKNOWN')
        path.modes_observed[mode] = path.modes_observed.get(mode, 0) + 1

        # Update time patterns
        hour = timestamp.hour
        if snr > path.avg_snr_db:  # Better than average
            if hour not in path.best_hours_utc:
                path.best_hours_utc.append(hour)
        else:
            if hour not in path.worst_hours_utc:
                path.worst_hours_utc.append(hour)

        # Update seasonal performance
        month_key = timestamp.strftime('%B')
        if month_key not in path.seasonal_performance:
            path.seasonal_performance[month_key] = snr
        else:
            # Running average
            alpha = 0.1
            path.seasonal_performance[month_key] = (
                (1 - alpha) * path.seasonal_performance[month_key] + alpha * snr
            )

        # Update success metrics
        path.total_attempts += 1
        if qso_data.get('successful', True):
            path.successful_decodes += 1
        path.decode_success_rate = path.successful_decodes / path.total_attempts

    def find_best_paths(self, min_distance_km: float = 100,
                        min_observations: int = 10) -> List[PropagationPath]:
        """Find best performing propagation paths.

        Args:
            min_distance_km: Minimum distance filter
            min_observations: Minimum observations filter

        Returns:
            List of best paths sorted by success rate and SNR
        """
        candidates = [
            path for path in self.paths.values()
            if path.distance_km >= min_distance_km
            and path.observation_count >= min_observations
        ]

        # Score paths
        scored_paths = []
        for path in candidates:
            score = (
                path.decode_success_rate * 100 +
                path.avg_snr_db +
                (10 if path.bidirectional else 0) +
                min(10, path.observation_count / 10)  # Bonus for many observations
            )
            scored_paths.append((score, path))

        # Sort by score
        scored_paths.sort(key=lambda x: x[0], reverse=True)

        return [path for _, path in scored_paths]

    def find_paths_for_station(self, station_hash: str) -> Dict[str, List[PropagationPath]]:
        """Find all paths involving a station.

        Args:
            station_hash: Station identifier

        Returns:
            Dict with 'tx' and 'rx' path lists
        """
        result = {'tx': [], 'rx': []}

        for path in self.paths.values():
            if path.tx_hash == station_hash:
                result['tx'].append(path)
            if path.rx_hash == station_hash:
                result['rx'].append(path)

        return result

    def predict_propagation(self, tx_hash: str, rx_hash: str,
                          timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Predict propagation between stations.

        Args:
            tx_hash: Transmitter station hash
            rx_hash: Receiver station hash
            timestamp: Time for prediction

        Returns:
            Prediction dictionary or None
        """
        path = self.paths.get((tx_hash, rx_hash))
        if not path:
            return None

        hour = timestamp.hour

        # Calculate probability based on historical data
        if hour in path.best_hours_utc:
            probability = 0.8
        elif hour in path.worst_hours_utc:
            probability = 0.2
        else:
            probability = 0.5

        # Best frequency based on time
        if 6 <= hour < 10:
            best_freq = 14074000  # 20m morning
        elif 10 <= hour < 16:
            best_freq = 21074000  # 15m daytime
        elif 16 <= hour < 20:
            best_freq = 14074000  # 20m afternoon
        else:
            best_freq = 7074000   # 40m night

        return {
            'probability': probability,
            'best_frequency': best_freq,
            'expected_snr': path.avg_snr_db,
            'confidence': min(1.0, path.observation_count / 100)
        }

    def predict_propagation_by_grid(self, tx_grid: str, rx_grid: str,
                          hour_utc: int, month: str) -> Dict[str, Any]:
        """Predict propagation conditions based on historical paths by grid.

        Args:
            tx_grid: Transmitter grid
            rx_grid: Receiver grid
            hour_utc: Hour in UTC
            month: Month name

        Returns:
            Prediction including probability and expected SNR
        """
        # Find similar paths
        similar_paths = self._find_similar_paths(tx_grid, rx_grid)

        if not similar_paths:
            return {
                'probability': 0.0,
                'expected_snr': None,
                'confidence': 'low',
                'based_on_paths': 0
            }

        # Aggregate predictions
        probabilities = []
        expected_snrs = []

        for path in similar_paths:
            # Check if hour is in best/worst hours
            if hour_utc in path.best_hours_utc:
                hour_factor = 1.2
            elif hour_utc in path.worst_hours_utc:
                hour_factor = 0.8
            else:
                hour_factor = 1.0

            # Get seasonal factor
            seasonal_snr = path.seasonal_performance.get(month, path.avg_snr_db)
            seasonal_factor = seasonal_snr / path.avg_snr_db if path.avg_snr_db != 0 else 1

            # Calculate probability and expected SNR
            base_probability = path.decode_success_rate
            adjusted_probability = min(1.0, base_probability * hour_factor * seasonal_factor)
            probabilities.append(adjusted_probability)

            expected_snr = path.avg_snr_db * hour_factor * seasonal_factor
            expected_snrs.append(expected_snr)

        # Average predictions
        avg_probability = np.mean(probabilities)
        avg_expected_snr = np.mean(expected_snrs)

        # Determine confidence
        if len(similar_paths) >= 5:
            confidence = 'high'
        elif len(similar_paths) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'probability': avg_probability,
            'expected_snr': avg_expected_snr,
            'confidence': confidence,
            'based_on_paths': len(similar_paths),
            'recommended_mode': self._recommend_mode(avg_expected_snr)
        }

    def _find_similar_paths(self, tx_grid: str, rx_grid: str,
                           max_distance_diff_km: float = 100) -> List[PropagationPath]:
        """Find paths similar to a given TX-RX pair.

        Args:
            tx_grid: Transmitter grid
            rx_grid: Receiver grid
            max_distance_diff_km: Maximum distance difference

        Returns:
            List of similar paths
        """
        target_distance = self.grid_calculator.calculate_distance(tx_grid, rx_grid)
        similar = []

        for path in self.paths.values():
            # Check if grids are close
            tx_dist = self.grid_calculator.calculate_distance(tx_grid, path.tx_grid)
            rx_dist = self.grid_calculator.calculate_distance(rx_grid, path.rx_grid)

            # Both ends should be reasonably close
            if tx_dist < max_distance_diff_km and rx_dist < max_distance_diff_km:
                similar.append(path)
            # Or if distance is similar
            elif abs(path.distance_km - target_distance) < max_distance_diff_km:
                similar.append(path)

        return similar

    def _recommend_mode(self, expected_snr: float) -> str:
        """Recommend operating mode based on expected SNR.

        Args:
            expected_snr: Expected SNR in dB

        Returns:
            Recommended mode
        """
        if expected_snr > -10:
            return "FT8"  # Good conditions
        elif expected_snr > -25:
            return "WSPR"  # Weak signal work
        else:
            return "WSPR_QRP"  # Very weak, use low power WSPR

    def get_all_paths(self) -> List[PropagationPath]:
        """Get all tracked paths.

        Returns:
            List of all PropagationPath objects
        """
        return list(self.paths.values())

    def get_path(self, tx_hash: str, rx_hash: str) -> Optional[PropagationPath]:
        """Get a specific path.

        Args:
            tx_hash: Transmitter station hash
            rx_hash: Receiver station hash

        Returns:
            PropagationPath or None
        """
        return self.paths.get((tx_hash, rx_hash))

    def find_bidirectional_paths(self) -> List[Tuple[PropagationPath, PropagationPath]]:
        """Find paths that work bidirectionally.

        Returns:
            List of bidirectional path pairs
        """
        bidirectional = []
        checked = set()

        for (tx, rx), path in self.paths.items():
            if (tx, rx) in checked or (rx, tx) in checked:
                continue

            reverse_path = self.paths.get((rx, tx))
            if reverse_path:
                bidirectional.append((path, reverse_path))
                checked.add((tx, rx))
                checked.add((rx, tx))

        return bidirectional

    def update_band_data(self, tx_hash: str, rx_hash: str,
                        band: str, frequency: float):
        """Update band-specific data for a path.

        Args:
            tx_hash: Transmitter station hash
            rx_hash: Receiver station hash
            band: Band name (e.g., '20m')
            frequency: Frequency in Hz
        """
        path = self.paths.get((tx_hash, rx_hash))
        if path:
            if not hasattr(path, 'band_data'):
                path.band_data = {}
            path.band_data[band] = frequency

    def get_best_propagation_times(self, tx_hash: str, rx_hash: str) -> List[int]:
        """Get best propagation times for a path.

        Args:
            tx_hash: Transmitter station hash
            rx_hash: Receiver station hash

        Returns:
            List of best hours (UTC)
        """
        path = self.paths.get((tx_hash, rx_hash))
        if path:
            return path.best_hours_utc
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get path tracking statistics.

        Returns:
            Statistics dictionary
        """
        if not self.paths:
            return {'total_paths': 0}

        distances = [p.distance_km for p in self.paths.values()]
        observations = [p.observation_count for p in self.paths.values()]
        success_rates = [p.decode_success_rate for p in self.paths.values()]

        bidirectional_count = sum(1 for p in self.paths.values() if p.bidirectional)

        return {
            'total_paths': len(self.paths),
            'bidirectional_paths': bidirectional_count,
            'avg_distance_km': np.mean(distances),
            'max_distance_km': np.max(distances),
            'min_distance_km': np.min(distances),
            'avg_observations': np.mean(observations),
            'total_observations': np.sum(observations),
            'avg_success_rate': np.mean(success_rates),
            'paths_over_1000km': sum(1 for d in distances if d > 1000),
            'paths_over_5000km': sum(1 for d in distances if d > 5000)
        }


class GridCalculator:
    """Helper class for grid square calculations."""

    def __init__(self):
        """Initialize calculator."""
        self.earth_radius_km = 6371.0

    def calculate_distance(self, grid1: str, grid2: str) -> float:
        """Calculate distance between grid squares.

        Args:
            grid1, grid2: Maidenhead grid squares

        Returns:
            Distance in kilometers
        """
        lat1, lon1 = self.grid_to_latlon(grid1)
        lat2, lon2 = self.grid_to_latlon(grid2)

        # Haversine formula
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        return self.earth_radius_km * c

    def calculate_bearing(self, grid1: str, grid2: str) -> float:
        """Calculate bearing from grid1 to grid2.

        Args:
            grid1, grid2: Maidenhead grid squares

        Returns:
            Bearing in degrees
        """
        lat1, lon1 = self.grid_to_latlon(grid1)
        lat2, lon2 = self.grid_to_latlon(grid2)

        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)

        y = math.sin(dlon) * math.cos(lat2_r)
        x = (math.cos(lat1_r) * math.sin(lat2_r) -
             math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))

        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360

    def calculate_midpoint_grid(self, grid1: str, grid2: str) -> str:
        """Calculate midpoint grid square.

        Args:
            grid1, grid2: Maidenhead grid squares

        Returns:
            Midpoint grid square
        """
        lat1, lon1 = self.grid_to_latlon(grid1)
        lat2, lon2 = self.grid_to_latlon(grid2)

        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2

        return self.latlon_to_grid(mid_lat, mid_lon)

    def grid_to_latlon(self, grid: str) -> Tuple[float, float]:
        """Convert Maidenhead grid to lat/lon.

        Args:
            grid: Maidenhead grid square

        Returns:
            (latitude, longitude) in degrees
        """
        grid = grid.upper()

        lon = (ord(grid[0]) - ord('A')) * 20 - 180
        lat = (ord(grid[1]) - ord('A')) * 10 - 90

        if len(grid) >= 4:
            lon += int(grid[2]) * 2
            lat += int(grid[3]) * 1

        if len(grid) >= 6:
            lon += (ord(grid[4]) - ord('A')) * 5/60
            lat += (ord(grid[5]) - ord('A')) * 2.5/60

        # Return center of grid square
        lon += 1
        lat += 0.5

        return lat, lon

    def latlon_to_grid(self, lat: float, lon: float, precision: int = 6) -> str:
        """Convert lat/lon to Maidenhead grid.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            precision: Grid precision (4 or 6)

        Returns:
            Maidenhead grid square
        """
        lon_adj = lon + 180
        lat_adj = lat + 90

        field_lon = int(lon_adj / 20)
        field_lat = int(lat_adj / 10)

        square_lon = int((lon_adj % 20) / 2)
        square_lat = int((lat_adj % 10) / 1)

        grid = (chr(ord('A') + field_lon) +
                chr(ord('A') + field_lat) +
                str(square_lon) +
                str(square_lat))

        if precision >= 6:
            subsquare_lon = int(((lon_adj % 20) % 2) * 12)
            subsquare_lat = int(((lat_adj % 10) % 1) * 24)
            grid += (chr(ord('A') + subsquare_lon) +
                    chr(ord('A') + subsquare_lat))

        return grid