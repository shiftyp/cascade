"""Reciprocal path power detection from bidirectional QSOs.

T095: Identify power asymmetries in bidirectional communications to estimate
relative power differences between stations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QSOPair:
    """Bidirectional QSO observation."""

    timestamp: datetime
    station_a_hash: str
    station_b_hash: str

    # A transmitting to B
    a_to_b_snr_db: float
    a_to_b_grid: str  # B's grid

    # B transmitting to A
    b_to_a_snr_db: float
    b_to_a_grid: str  # A's grid

    distance_km: float
    band: str


@dataclass
class PowerAsymmetry:
    """Power asymmetry detection from reciprocal paths."""

    station_a_hash: str
    station_b_hash: str
    timestamp: datetime

    # Power difference estimate
    power_difference_db: float  # Positive means A > B
    confidence_score: float

    # Statistics
    num_observations: int
    avg_snr_asymmetry_db: float
    std_dev_db: float

    # Relative power estimates (normalized)
    station_a_relative_power: float  # 0-1 scale
    station_b_relative_power: float  # 0-1 scale


class ReciprocalPathPowerDetector:
    """Detects power differences from reciprocal propagation paths."""

    def __init__(self):
        """Initialize reciprocal path detector."""
        self.qso_pairs: List[QSOPair] = []
        self.asymmetries: Dict[Tuple[str, str], List[PowerAsymmetry]] = defaultdict(list)

        # Detection parameters
        self.time_window_minutes = 30  # Max time between reciprocal transmissions
        self.min_observations = 3  # Minimum QSOs for reliable estimate
        self.propagation_symmetry_window = 5  # Minutes for symmetric propagation

    def process_ft8_log(self, ft8_entries: List[Dict[str, Any]]) -> List[QSOPair]:
        """Process FT8 log entries to identify bidirectional QSOs.

        Args:
            ft8_entries: List of FT8 decode entries with:
                - timestamp: UTC timestamp
                - tx_hash: Transmitter station hash
                - rx_hash: Receiver station hash
                - snr_db: Reported SNR
                - tx_grid: Transmitter grid
                - rx_grid: Receiver grid
                - band: Operating band
                - message: FT8 message content

        Returns:
            List of identified QSO pairs
        """
        # Sort by timestamp
        sorted_entries = sorted(ft8_entries, key=lambda x: x['timestamp'])

        # Group by station pairs
        station_pairs = defaultdict(list)
        for entry in sorted_entries:
            pair_key = tuple(sorted([entry['tx_hash'], entry['rx_hash']]))
            station_pairs[pair_key].append(entry)

        # Identify reciprocal QSOs
        qso_pairs = []
        for (station_a, station_b), entries in station_pairs.items():
            pairs = self._find_reciprocal_pairs(entries, station_a, station_b)
            qso_pairs.extend(pairs)

        self.qso_pairs.extend(qso_pairs)
        return qso_pairs

    def _find_reciprocal_pairs(self, entries: List[Dict],
                               station_a: str, station_b: str) -> List[QSOPair]:
        """Find reciprocal transmission pairs between two stations.

        Args:
            entries: Chronologically sorted entries between two stations
            station_a, station_b: Station hashes

        Returns:
            List of QSO pairs
        """
        pairs = []

        # Separate by direction
        a_to_b = [e for e in entries if e['tx_hash'] == station_a]
        b_to_a = [e for e in entries if e['tx_hash'] == station_b]

        if not a_to_b or not b_to_a:
            return pairs

        # Match reciprocal transmissions within time window
        for a_tx in a_to_b:
            # Find closest B transmission
            best_match = None
            min_time_diff = timedelta(minutes=self.time_window_minutes)

            for b_tx in b_to_a:
                time_diff = abs(b_tx['timestamp'] - a_tx['timestamp'])

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = b_tx

            if best_match and min_time_diff < timedelta(minutes=self.propagation_symmetry_window):
                # Create QSO pair
                pair = QSOPair(
                    timestamp=a_tx['timestamp'],
                    station_a_hash=station_a,
                    station_b_hash=station_b,
                    a_to_b_snr_db=a_tx['snr_db'],
                    a_to_b_grid=a_tx['rx_grid'],
                    b_to_a_snr_db=best_match['snr_db'],
                    b_to_a_grid=best_match['rx_grid'],
                    distance_km=self._calculate_distance(
                        a_tx.get('tx_grid'), a_tx.get('rx_grid')
                    ),
                    band=a_tx['band']
                )
                pairs.append(pair)

        return pairs

    def detect_power_asymmetry(self, station_a: str, station_b: str) -> Optional[PowerAsymmetry]:
        """Detect power asymmetry between two stations.

        Args:
            station_a, station_b: Station hashes to compare

        Returns:
            PowerAsymmetry or None if insufficient data
        """
        # Find all QSOs between these stations
        relevant_qsos = [
            qso for qso in self.qso_pairs
            if {qso.station_a_hash, qso.station_b_hash} == {station_a, station_b}
        ]

        if len(relevant_qsos) < self.min_observations:
            return None

        # Calculate SNR asymmetries
        asymmetries = []
        for qso in relevant_qsos:
            if qso.station_a_hash == station_a:
                # A to B vs B to A
                asymmetry = qso.a_to_b_snr_db - qso.b_to_a_snr_db
            else:
                # B to A vs A to B (flip sign)
                asymmetry = qso.b_to_a_snr_db - qso.a_to_b_snr_db

            asymmetries.append(asymmetry)

        asymmetries = np.array(asymmetries)

        # Remove outliers
        q1, q3 = np.percentile(asymmetries, [25, 75])
        iqr = q3 - q1
        mask = (asymmetries >= q1 - 1.5*iqr) & (asymmetries <= q3 + 1.5*iqr)
        filtered_asymmetries = asymmetries[mask]

        if len(filtered_asymmetries) < self.min_observations:
            filtered_asymmetries = asymmetries  # Use all if too many filtered

        # Calculate statistics
        avg_asymmetry = np.mean(filtered_asymmetries)
        std_dev = np.std(filtered_asymmetries)

        # Power difference is approximately equal to SNR asymmetry
        # (assuming reciprocal propagation)
        power_difference_db = avg_asymmetry

        # Confidence based on consistency and number of observations
        consistency_factor = max(0.3, 1.0 - std_dev/10)  # Lower confidence with high variance
        observation_factor = min(1.0, len(filtered_asymmetries)/10)  # More observations = higher confidence
        confidence = consistency_factor * observation_factor

        # Calculate relative power (normalized 0-1)
        if power_difference_db > 0:
            # Station A has more power
            station_a_relative = 1.0
            station_b_relative = 10**(-abs(power_difference_db)/20)  # Convert dB to linear ratio
        else:
            # Station B has more power
            station_a_relative = 10**(-abs(power_difference_db)/20)
            station_b_relative = 1.0

        asymmetry_result = PowerAsymmetry(
            station_a_hash=station_a,
            station_b_hash=station_b,
            timestamp=relevant_qsos[-1].timestamp,  # Most recent
            power_difference_db=power_difference_db,
            confidence_score=confidence,
            num_observations=len(filtered_asymmetries),
            avg_snr_asymmetry_db=avg_asymmetry,
            std_dev_db=std_dev,
            station_a_relative_power=station_a_relative,
            station_b_relative_power=station_b_relative
        )

        # Store result
        pair_key = tuple(sorted([station_a, station_b]))
        self.asymmetries[pair_key].append(asymmetry_result)

        return asymmetry_result

    def aggregate_station_power(self, station_hash: str) -> Dict[str, float]:
        """Aggregate power estimates for a station from all reciprocal paths.

        Args:
            station_hash: Station to analyze

        Returns:
            Dictionary with aggregated power metrics
        """
        # Find all asymmetries involving this station
        station_asymmetries = []
        for (s1, s2), asymmetry_list in self.asymmetries.items():
            if station_hash in (s1, s2):
                station_asymmetries.extend(asymmetry_list)

        if not station_asymmetries:
            return {'relative_power': 0.5, 'confidence': 0.0, 'num_comparisons': 0}

        # Collect relative powers for this station
        powers = []
        weights = []

        for asym in station_asymmetries:
            if asym.station_a_hash == station_hash:
                powers.append(asym.station_a_relative_power)
            else:
                powers.append(asym.station_b_relative_power)

            weights.append(asym.confidence_score)

        # Weighted average
        powers = np.array(powers)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        avg_relative_power = np.average(powers, weights=weights)
        confidence = np.mean([a.confidence_score for a in station_asymmetries])

        return {
            'relative_power': avg_relative_power,
            'confidence': confidence,
            'num_comparisons': len(station_asymmetries),
            'std_dev': np.std(powers)
        }

    def _calculate_distance(self, grid1: Optional[str], grid2: Optional[str]) -> float:
        """Calculate distance between grid squares in km.

        Simplified calculation - in production would use proper grid square math.
        """
        if not grid1 or not grid2 or len(grid1) < 4 or len(grid2) < 4:
            return 1000.0  # Default 1000 km

        # Extract field and square
        lat1 = (ord(grid1[1]) - ord('A')) * 10 + int(grid1[3])
        lon1 = (ord(grid1[0]) - ord('A')) * 20 + int(grid1[2]) * 2

        lat2 = (ord(grid2[1]) - ord('A')) * 10 + int(grid2[3])
        lon2 = (ord(grid2[0]) - ord('A')) * 20 + int(grid2[2]) * 2

        # Convert to approximate coordinates
        lat1 = lat1 - 90
        lon1 = lon1 - 180
        lat2 = lat2 - 90
        lon2 = lon2 - 180

        # Haversine formula (simplified)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c  # Earth radius in km

        return distance

    def identify_power_clusters(self, min_cluster_size: int = 5) -> Dict[str, List[str]]:
        """Identify clusters of stations with similar power levels.

        Args:
            min_cluster_size: Minimum stations per cluster

        Returns:
            Dictionary mapping power level to station hashes
        """
        # Aggregate power for all stations
        station_powers = {}
        for (s1, s2) in self.asymmetries.keys():
            for station in [s1, s2]:
                if station not in station_powers:
                    station_powers[station] = self.aggregate_station_power(station)

        if len(station_powers) < min_cluster_size:
            return {}

        # Simple clustering by relative power
        clusters = {
            'qrp': [],      # relative_power < 0.2
            'low': [],      # 0.2 <= relative_power < 0.4
            'typical': [],  # 0.4 <= relative_power < 0.7
            'high': [],     # 0.7 <= relative_power < 0.9
            'qro': []       # relative_power >= 0.9
        }

        for station, metrics in station_powers.items():
            power = metrics['relative_power']

            if power < 0.2:
                clusters['qrp'].append(station)
            elif power < 0.4:
                clusters['low'].append(station)
            elif power < 0.7:
                clusters['typical'].append(station)
            elif power < 0.9:
                clusters['high'].append(station)
            else:
                clusters['qro'].append(station)

        # Filter out small clusters
        return {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about reciprocal path analysis.

        Returns:
            Statistics dictionary
        """
        total_pairs = len(self.qso_pairs)
        unique_stations = set()

        for qso in self.qso_pairs:
            unique_stations.add(qso.station_a_hash)
            unique_stations.add(qso.station_b_hash)

        asymmetry_values = []
        for asym_list in self.asymmetries.values():
            asymmetry_values.extend([a.power_difference_db for a in asym_list])

        return {
            'total_qso_pairs': total_pairs,
            'unique_stations': len(unique_stations),
            'station_pairs_analyzed': len(self.asymmetries),
            'avg_power_asymmetry_db': np.mean(asymmetry_values) if asymmetry_values else 0,
            'max_power_asymmetry_db': np.max(np.abs(asymmetry_values)) if asymmetry_values else 0,
            'power_clusters': {k: len(v) for k, v in self.identify_power_clusters().items()}
        }