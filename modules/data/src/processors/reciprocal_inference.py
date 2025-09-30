"""Reciprocal path inference for sparse regions (T086).

Generates synthetic southern observations from northern TX->southern RX paths
to help balance geographic coverage in underrepresented regions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import numpy as np

from ..models import SessionLocal, PropagationRecord
from ..collectors.geographic_quotas import GridSquareClassifier, Hemisphere

logger = logging.getLogger(__name__)


@dataclass
class ReciprocalPath:
    """Represents a reciprocal propagation path."""

    original_tx_grid: str
    original_rx_grid: str
    original_snr: float
    original_frequency_mhz: float
    original_timestamp: datetime

    inferred_tx_grid: str  # Swapped
    inferred_rx_grid: str  # Swapped
    inferred_snr: float    # Adjusted for reciprocity
    confidence: float       # 0.5x weight for inferred data

    tx_hemisphere: Hemisphere
    rx_hemisphere: Hemisphere
    path_distance_km: float


class ReciprocalPathInference:
    """Infers reciprocal paths for sparse region coverage (T086)."""

    def __init__(self):
        """Initialize reciprocal path inference."""
        self.db = SessionLocal()
        self.classifier = GridSquareClassifier()

        # Cache for bidirectional path detection
        self.bidirectional_cache = {}
        self.cache_expires = None

        # Confidence weight for inferred data (T086c)
        self.INFERRED_DATA_WEIGHT = 0.5

    def identify_bidirectional_paths(
        self,
        time_window_hours: int = 24
    ) -> List[Tuple[str, str]]:
        """Identify bidirectional paths from existing data (T086a).

        Args:
            time_window_hours: Time window to look for reciprocal paths

        Returns:
            List of bidirectional path pairs (tx_grid, rx_grid)
        """
        # Check cache
        if self.cache_expires and datetime.utcnow() < self.cache_expires:
            return list(self.bidirectional_cache.keys())

        try:
            # Query recent propagation records
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

            records = self.db.query(PropagationRecord).filter(
                PropagationRecord.timestamp >= cutoff_time
            ).all()

            # Build path map
            path_map = {}
            for record in records:
                if record.tx_grid_square and record.rx_grid_square:
                    # Create path key (sorted to handle both directions)
                    path_key = tuple(sorted([
                        record.tx_grid_square[:4],  # Use 4-char precision
                        record.rx_grid_square[:4]
                    ]))

                    # Track both directions
                    if path_key not in path_map:
                        path_map[path_key] = {
                            "forward": [],
                            "reverse": []
                        }

                    # Determine direction
                    if record.tx_grid_square[:4] == path_key[0]:
                        path_map[path_key]["forward"].append(record)
                    else:
                        path_map[path_key]["reverse"].append(record)

            # Find bidirectional paths
            bidirectional = []
            for path_key, directions in path_map.items():
                if directions["forward"] and directions["reverse"]:
                    bidirectional.append(path_key)

                    # Store average SNRs for inference
                    avg_forward_snr = np.mean([
                        r.snr_db for r in directions["forward"]
                        if r.snr_db is not None
                    ])
                    avg_reverse_snr = np.mean([
                        r.snr_db for r in directions["reverse"]
                        if r.snr_db is not None
                    ])

                    self.bidirectional_cache[path_key] = {
                        "forward_snr": avg_forward_snr,
                        "reverse_snr": avg_reverse_snr,
                        "count": len(directions["forward"]) + len(directions["reverse"])
                    }

            # Update cache
            self.cache_expires = datetime.utcnow() + timedelta(hours=1)

            logger.info(
                f"Found {len(bidirectional)} bidirectional paths "
                f"from {len(records)} records"
            )

            return bidirectional

        except Exception as e:
            logger.error(f"Error identifying bidirectional paths: {e}")
            return []

    def generate_southern_observations(
        self,
        northern_tx_records: List[PropagationRecord]
    ) -> List[ReciprocalPath]:
        """Generate synthetic southern observations from northern TX data (T086b).

        Args:
            northern_tx_records: Northern transmitter records with southern receivers

        Returns:
            List of inferred reciprocal paths
        """
        inferred_paths = []

        for record in northern_tx_records:
            # Check if this is a north->south path
            if not record.tx_grid_square or not record.rx_grid_square:
                continue

            tx_hemisphere = self.classifier.get_hemisphere(record.tx_grid_square)
            rx_hemisphere = self.classifier.get_hemisphere(record.rx_grid_square)

            # We want northern TX -> southern RX paths
            if tx_hemisphere != Hemisphere.NORTH or rx_hemisphere != Hemisphere.SOUTH:
                continue

            # Check if we have bidirectional data for SNR adjustment
            path_key = tuple(sorted([
                record.tx_grid_square[:4],
                record.rx_grid_square[:4]
            ]))

            # Calculate inferred SNR
            if path_key in self.bidirectional_cache:
                # Use bidirectional data for better inference
                cache_data = self.bidirectional_cache[path_key]
                snr_adjustment = cache_data["reverse_snr"] - cache_data["forward_snr"]
                inferred_snr = record.snr_db + snr_adjustment
                confidence = min(0.8, 0.5 + cache_data["count"] / 100)
            else:
                # Simple reciprocity assumption
                inferred_snr = record.snr_db - 3.0  # Assume 3dB loss
                confidence = self.INFERRED_DATA_WEIGHT

            # Create reciprocal path
            reciprocal = ReciprocalPath(
                original_tx_grid=record.tx_grid_square,
                original_rx_grid=record.rx_grid_square,
                original_snr=record.snr_db,
                original_frequency_mhz=record.frequency_mhz,
                original_timestamp=record.timestamp,

                # Swap TX and RX for reciprocal
                inferred_tx_grid=record.rx_grid_square,
                inferred_rx_grid=record.tx_grid_square,
                inferred_snr=inferred_snr,
                confidence=confidence,

                tx_hemisphere=tx_hemisphere,
                rx_hemisphere=rx_hemisphere,
                path_distance_km=self._calculate_distance(
                    record.tx_grid_square,
                    record.rx_grid_square
                )
            )

            inferred_paths.append(reciprocal)

        logger.info(
            f"Generated {len(inferred_paths)} southern observations "
            f"from {len(northern_tx_records)} northern TX records"
        )

        return inferred_paths

    def weight_inferred_data(
        self,
        inferred_paths: List[ReciprocalPath]
    ) -> List[Dict[str, Any]]:
        """Apply 0.5x weight to inferred data (T086c).

        Args:
            inferred_paths: List of inferred reciprocal paths

        Returns:
            List of weighted data records
        """
        weighted_records = []

        for path in inferred_paths:
            # Create weighted record
            record = {
                "tx_grid_square": path.inferred_tx_grid,
                "rx_grid_square": path.inferred_rx_grid,
                "snr_db": path.inferred_snr,
                "frequency_mhz": path.original_frequency_mhz,
                "timestamp": path.original_timestamp,
                "weight": path.confidence,  # 0.5x for simple inference
                "is_inferred": True,
                "original_path": f"{path.original_tx_grid}->{path.original_rx_grid}",

                # Metadata for tracking
                "inference_method": "reciprocal",
                "confidence": path.confidence,
                "hemisphere_pair": f"{path.tx_hemisphere.value}->{path.rx_hemisphere.value}",
                "distance_km": path.path_distance_km
            }

            # Add hash for deduplication
            path_str = f"{path.inferred_tx_grid}{path.inferred_rx_grid}{path.original_frequency_mhz}"
            record["path_hash"] = hashlib.md5(path_str.encode()).hexdigest()[:8]

            weighted_records.append(record)

        return weighted_records

    def _calculate_distance(self, grid1: str, grid2: str) -> float:
        """Calculate approximate distance between grid squares.

        Args:
            grid1: First grid square
            grid2: Second grid square

        Returns:
            Distance in kilometers
        """
        # Simplified distance calculation
        # In production, use proper great circle distance

        lat1 = self.classifier.get_latitude_from_grid(grid1)
        lat2 = self.classifier.get_latitude_from_grid(grid2)

        # Very rough approximation
        lat_diff = abs(lat1 - lat2)

        # Assume average longitude difference of 30 degrees
        lon_diff = 30

        # Rough distance (111 km per degree latitude)
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111

        return distance

    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get statistics about reciprocal path inference.

        Returns:
            Dictionary of inference statistics
        """
        bidirectional_paths = self.identify_bidirectional_paths()

        # Count hemispheric pairs
        hemisphere_pairs = {
            "north_north": 0,
            "north_south": 0,
            "south_north": 0,
            "south_south": 0,
            "equatorial": 0
        }

        for tx_grid, rx_grid in bidirectional_paths:
            tx_hem = self.classifier.get_hemisphere(tx_grid)
            rx_hem = self.classifier.get_hemisphere(rx_grid)

            if tx_hem == Hemisphere.EQUATORIAL or rx_hem == Hemisphere.EQUATORIAL:
                hemisphere_pairs["equatorial"] += 1
            elif tx_hem == Hemisphere.NORTH and rx_hem == Hemisphere.NORTH:
                hemisphere_pairs["north_north"] += 1
            elif tx_hem == Hemisphere.NORTH and rx_hem == Hemisphere.SOUTH:
                hemisphere_pairs["north_south"] += 1
            elif tx_hem == Hemisphere.SOUTH and rx_hem == Hemisphere.NORTH:
                hemisphere_pairs["south_north"] += 1
            else:
                hemisphere_pairs["south_south"] += 1

        # Calculate inference potential
        total_paths = sum(hemisphere_pairs.values())
        inferable = hemisphere_pairs["north_south"] + hemisphere_pairs["south_north"]
        inference_potential = (inferable / total_paths * 100) if total_paths > 0 else 0

        return {
            "bidirectional_paths_found": len(bidirectional_paths),
            "hemisphere_distribution": hemisphere_pairs,
            "inference_potential_percentage": inference_potential,
            "cache_size": len(self.bidirectional_cache),
            "inferred_data_weight": self.INFERRED_DATA_WEIGHT,
            "recommendations": {
                "increase_southern_tx": hemisphere_pairs["north_south"] > hemisphere_pairs["south_north"],
                "focus_regions": self._identify_sparse_regions()
            }
        }

    def _identify_sparse_regions(self) -> List[str]:
        """Identify sparse regions that would benefit from inference.

        Returns:
            List of sparse region descriptions
        """
        # This would query the database for underrepresented areas
        # Simplified for now
        return [
            "Southern Pacific (PG, PH grids)",
            "Southern Atlantic (GG, GH grids)",
            "Antarctica (all grids)",
            "Southern Indian Ocean (HG, HH grids)"
        ]

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()


# Utility functions
def apply_reciprocal_inference_for_region(
    target_hemisphere: Hemisphere,
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """Apply reciprocal inference for a specific hemisphere.

    Args:
        target_hemisphere: Target hemisphere to generate data for
        min_confidence: Minimum confidence threshold

    Returns:
        List of inferred records
    """
    inference = ReciprocalPathInference()

    try:
        # Get recent northern TX records
        db = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(hours=24)

        northern_records = db.query(PropagationRecord).filter(
            PropagationRecord.timestamp >= cutoff
        ).all()

        # Filter for northern TX with southern RX
        relevant_records = []
        classifier = GridSquareClassifier()

        for record in northern_records:
            if not record.tx_grid_square or not record.rx_grid_square:
                continue

            tx_hem = classifier.get_hemisphere(record.tx_grid_square)
            rx_hem = classifier.get_hemisphere(record.rx_grid_square)

            if tx_hem == Hemisphere.NORTH and rx_hem == target_hemisphere:
                relevant_records.append(record)

        # Generate inferred observations
        inferred_paths = inference.generate_southern_observations(relevant_records)

        # Filter by confidence
        filtered_paths = [
            path for path in inferred_paths
            if path.confidence >= min_confidence
        ]

        # Apply weights and return
        weighted_records = inference.weight_inferred_data(filtered_paths)

        logger.info(
            f"Generated {len(weighted_records)} inferred records "
            f"for {target_hemisphere.value} hemisphere"
        )

        return weighted_records

    finally:
        inference.close()
        if db:
            db.close()