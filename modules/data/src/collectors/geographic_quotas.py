"""Geographic quota system for ensuring global data diversity (T083).

Implements latitude band quotas and hemispheric balance requirements
to mitigate geographic bias in data collection.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class LatitudeBand(Enum):
    """Latitude band classifications (T083a)."""
    ARCTIC = "arctic"          # >66.5°N
    TEMPERATE = "temperate"    # 23.5-66.5° (both N and S)
    TROPICAL = "tropical"      # ±23.5°
    ANTARCTIC = "antarctic"    # <-66.5°S


class Hemisphere(Enum):
    """Hemisphere classifications (T083c)."""
    NORTH = "north"           # >10°N
    SOUTH = "south"           # <-10°S
    EQUATORIAL = "equatorial" # ±10°


@dataclass
class QuotaConfiguration:
    """Quota configuration for geographic diversity (T083b, T083c)."""

    # Minimum percentage for each latitude band (T083b)
    latitude_band_minimum_percent: float = 20.0

    # Hemisphere balance targets (T083c)
    # Default: 40% North, 40% South, 20% Equatorial
    hemisphere_targets: Dict[Hemisphere, float] = field(default_factory=lambda: {
        Hemisphere.NORTH: 0.4,
        Hemisphere.SOUTH: 0.4,
        Hemisphere.EQUATORIAL: 0.2
    })

    # Ocean path minimum (T084c)
    ocean_path_minimum_percent: float = 30.0

    # Underrepresented region boost multiplier
    scarcity_boost_multiplier: float = 2.5

    # Rebalancing thresholds
    critical_threshold_percent: float = 50.0  # Trigger warnings below this
    rebalance_threshold_percent: float = 70.0  # Suggest rebalancing below this

    def get_band_quota(self, band: LatitudeBand) -> float:
        """Get quota for a specific latitude band."""
        return self.latitude_band_minimum_percent / 100.0

    def get_hemisphere_target(self, hemisphere: Hemisphere) -> float:
        """Get target for a specific hemisphere."""
        return self.hemisphere_targets.get(hemisphere, 0.33)


@dataclass
class CollectionProgress:
    """Tracks collection progress across geographic regions."""

    total_hours: float = 0.0
    latitude_band_hours: Dict[LatitudeBand, float] = field(default_factory=dict)
    hemisphere_hours: Dict[Hemisphere, float] = field(default_factory=dict)
    ocean_path_hours: float = 0.0
    land_path_hours: float = 0.0

    # Percentages
    latitude_band_percentages: Dict[LatitudeBand, float] = field(default_factory=dict)
    hemisphere_percentages: Dict[Hemisphere, float] = field(default_factory=dict)
    ocean_path_percentage: float = 0.0

    def calculate_percentages(self):
        """Calculate percentage distributions."""
        if self.total_hours == 0:
            return

        # Latitude band percentages
        for band in LatitudeBand:
            hours = self.latitude_band_hours.get(band, 0)
            self.latitude_band_percentages[band] = (hours / self.total_hours) * 100

        # Hemisphere percentages
        for hemisphere in Hemisphere:
            hours = self.hemisphere_hours.get(hemisphere, 0)
            self.hemisphere_percentages[hemisphere] = (hours / self.total_hours) * 100

        # Ocean path percentage
        total_paths = self.ocean_path_hours + self.land_path_hours
        if total_paths > 0:
            self.ocean_path_percentage = (self.ocean_path_hours / total_paths) * 100


class GridSquareClassifier:
    """Classifies grid squares into geographic regions."""

    def __init__(self):
        """Initialize the classifier."""
        # Grid square to latitude mapping
        self.grid_to_latitude = self._build_grid_latitude_map()

        # Ocean grid squares (simplified - would use detailed database in production)
        self.ocean_grids = self._load_ocean_grids()

    def _build_grid_latitude_map(self) -> Dict[str, float]:
        """Build mapping from grid square prefix to latitude."""
        # This is a simplified version - full implementation would use
        # Maidenhead grid square calculations
        grid_map = {}

        # Each letter represents 10 degrees of latitude
        # A=90°S to 80°S, B=80°S to 70°S, ..., R=80°N to 90°N
        for i, letter in enumerate("ABCDEFGHIJKLMNOPQR"):
            base_lat = -90 + (i * 10)
            for j in range(10):
                grid_map[f"{letter}{j}"] = base_lat + j

        return grid_map

    def _load_ocean_grids(self) -> set:
        """Load set of ocean grid squares."""
        # Simplified ocean grids - production would use comprehensive database
        ocean_grids = {
            "DM", "CM", "BM",  # Pacific
            "FK", "EK", "DK",  # Pacific
            "JM", "IM", "HM",  # Atlantic
            "GH", "FH", "EH",  # Indian Ocean
        }
        return ocean_grids

    def get_latitude_from_grid(self, grid_square: str) -> float:
        """Convert grid square to approximate latitude.

        Args:
            grid_square: 4-6 character Maidenhead grid square

        Returns:
            Latitude in degrees (-90 to +90)
        """
        if not grid_square or len(grid_square) < 2:
            return 0.0

        # Extract field (first two characters)
        field = grid_square[:2].upper()

        # Maidenhead grid system:
        # First char: longitude (A-R, each 20 degrees, A=-180)
        # Second char: latitude (A-R, each 10 degrees, A=-90)

        # Latitude component (second character)
        lat_char = field[1]
        lat_index = ord(lat_char) - ord('A')
        latitude = -90 + (lat_index * 10) + 5  # Center of 10-degree band

        # Refine with square if available (third and fourth characters)
        if len(grid_square) >= 4:
            # Fourth character is latitude square (0-9, each 1 degree)
            try:
                lat_square = int(grid_square[3])
                latitude += lat_square - 5  # Adjust to use square center
            except (ValueError, IndexError):
                pass

        return latitude

    def get_latitude_band(self, grid_square: str) -> LatitudeBand:
        """Classify grid square into latitude band (T083a).

        Args:
            grid_square: Maidenhead grid square

        Returns:
            Latitude band classification
        """
        latitude = self.get_latitude_from_grid(grid_square)

        if latitude > 66.5:
            return LatitudeBand.ARCTIC
        elif latitude < -66.5:
            return LatitudeBand.ANTARCTIC
        elif -23.5 <= latitude <= 23.5:
            return LatitudeBand.TROPICAL
        else:
            return LatitudeBand.TEMPERATE

    def get_hemisphere(self, grid_square: str) -> Hemisphere:
        """Classify grid square into hemisphere (T083c).

        Args:
            grid_square: Maidenhead grid square

        Returns:
            Hemisphere classification
        """
        latitude = self.get_latitude_from_grid(grid_square)

        if latitude > 10:
            return Hemisphere.NORTH
        elif latitude < -10:
            return Hemisphere.SOUTH
        else:
            return Hemisphere.EQUATORIAL

    def is_ocean_grid(self, grid_square: str) -> bool:
        """Check if grid square is over ocean.

        Args:
            grid_square: Maidenhead grid square

        Returns:
            True if grid is primarily over ocean
        """
        if not grid_square or len(grid_square) < 2:
            return False

        prefix = grid_square[:2].upper()
        return prefix in self.ocean_grids


class GeographicQuotaManager:
    """Manages geographic quotas for data collection (T083)."""

    def __init__(self, config: Optional[QuotaConfiguration] = None):
        """Initialize the quota manager.

        Args:
            config: Quota configuration (uses defaults if None)
        """
        self.config = config or QuotaConfiguration()
        self.classifier = GridSquareClassifier()

        # Collection history
        self.collection_history: List[Dict[str, Any]] = []
        self.last_rebalance = datetime.now(timezone.utc)

    def add_collection_record(self, grid_square: str, hours: float,
                             is_ocean_path: bool = False):
        """Add a collection record.

        Args:
            grid_square: Grid square of collection
            hours: Hours collected
            is_ocean_path: Whether this is an ocean path
        """
        record = {
            "grid_square": grid_square,
            "hours": hours,
            "timestamp": datetime.now(timezone.utc),
            "latitude_band": self.classifier.get_latitude_band(grid_square),
            "hemisphere": self.classifier.get_hemisphere(grid_square),
            "is_ocean_path": is_ocean_path
        }
        self.collection_history.append(record)

    def get_collection_progress(self) -> CollectionProgress:
        """Get current collection progress.

        Returns:
            Collection progress summary
        """
        progress = CollectionProgress()

        # Initialize counters
        for band in LatitudeBand:
            progress.latitude_band_hours[band] = 0.0
        for hemisphere in Hemisphere:
            progress.hemisphere_hours[hemisphere] = 0.0

        # Aggregate from history
        for record in self.collection_history:
            hours = record["hours"]
            progress.total_hours += hours

            # Latitude band
            band = record["latitude_band"]
            progress.latitude_band_hours[band] += hours

            # Hemisphere
            hemisphere = record["hemisphere"]
            progress.hemisphere_hours[hemisphere] += hours

            # Ocean/land path
            if record.get("is_ocean_path", False):
                progress.ocean_path_hours += hours
            else:
                progress.land_path_hours += hours

        # Calculate percentages
        progress.calculate_percentages()

        return progress

    def get_underrepresented_bands(self) -> List[LatitudeBand]:
        """Get list of underrepresented latitude bands.

        Returns:
            List of bands below quota
        """
        progress = self.get_collection_progress()
        underrepresented = []

        if progress.total_hours == 0:
            return list(LatitudeBand)  # All bands underrepresented initially

        quota = self.config.latitude_band_minimum_percent

        for band in LatitudeBand:
            percentage = progress.latitude_band_percentages.get(band, 0)
            if percentage < quota:
                underrepresented.append(band)

        return underrepresented

    def get_hemisphere_balance_score(self) -> float:
        """Calculate hemispheric balance score.

        Returns:
            Balance score (0.0 = very imbalanced, 1.0 = perfectly balanced)
        """
        progress = self.get_collection_progress()

        if progress.total_hours == 0:
            return 1.0  # Perfect balance when no data

        score = 0.0
        count = 0

        for hemisphere, target in self.config.hemisphere_targets.items():
            actual = progress.hemisphere_percentages.get(hemisphere, 0) / 100.0
            if target > 0:
                # Calculate deviation from target
                deviation = abs(actual - target) / target
                hemisphere_score = max(0, 1.0 - deviation)
                score += hemisphere_score
                count += 1

        return score / count if count > 0 else 0.0

    def should_prioritize(self, grid_square: str) -> float:
        """Calculate priority score for a grid square.

        Args:
            grid_square: Grid square to evaluate

        Returns:
            Priority score (higher = more important to collect)
        """
        band = self.classifier.get_latitude_band(grid_square)
        hemisphere = self.classifier.get_hemisphere(grid_square)
        is_ocean = self.classifier.is_ocean_grid(grid_square)

        progress = self.get_collection_progress()

        # Base priority
        priority = 1.0

        # Boost for underrepresented latitude band
        if progress.total_hours > 0:
            band_percentage = progress.latitude_band_percentages.get(band, 0)
            if band_percentage < self.config.latitude_band_minimum_percent:
                deficit = self.config.latitude_band_minimum_percent - band_percentage
                priority *= (1.0 + deficit / 10.0)  # Up to 2x for 10% deficit

        # Boost for underrepresented hemisphere
        if progress.total_hours > 0:
            hemisphere_target = self.config.hemisphere_targets[hemisphere] * 100
            hemisphere_actual = progress.hemisphere_percentages.get(hemisphere, 0)
            if hemisphere_actual < hemisphere_target:
                deficit = hemisphere_target - hemisphere_actual
                priority *= (1.0 + deficit / 20.0)  # Up to 1.5x for 10% deficit

        # Boost for ocean paths if below target
        if is_ocean and progress.ocean_path_percentage < self.config.ocean_path_minimum_percent:
            priority *= 1.3

        # Special boost for Antarctic (hardest to get)
        if band == LatitudeBand.ANTARCTIC:
            priority *= 1.5

        return priority

    def is_ocean_path(self, tx_grid: str, rx_grid: str) -> bool:
        """Determine if a path crosses ocean (T084c).

        Args:
            tx_grid: Transmitter grid square
            rx_grid: Receiver grid square

        Returns:
            True if path likely crosses ocean
        """
        # Simple heuristic: if either endpoint is ocean, or they're far apart
        if self.classifier.is_ocean_grid(tx_grid) or self.classifier.is_ocean_grid(rx_grid):
            return True

        # Check distance (simplified - would use great circle calculation)
        tx_lat = self.classifier.get_latitude_from_grid(tx_grid)
        rx_lat = self.classifier.get_latitude_from_grid(rx_grid)

        # If > 30 degrees apart, likely ocean path
        if abs(tx_lat - rx_lat) > 30:
            return True

        return False

    def get_quota_warnings(self) -> List[str]:
        """Get warnings for quotas not being met.

        Returns:
            List of warning messages
        """
        warnings = []
        progress = self.get_collection_progress()

        if progress.total_hours == 0:
            return []

        # Check latitude bands
        for band in LatitudeBand:
            percentage = progress.latitude_band_percentages.get(band, 0)
            if percentage < self.config.critical_threshold_percent:
                warnings.append(
                    f"{band.value.capitalize()} band critically underrepresented: "
                    f"{percentage:.1f}% (target: {self.config.latitude_band_minimum_percent}%)"
                )

        # Check hemisphere balance
        balance_score = self.get_hemisphere_balance_score()
        if balance_score < 0.5:
            warnings.append(
                f"Hemispheric imbalance detected: score {balance_score:.2f} "
                f"(North: {progress.hemisphere_percentages.get(Hemisphere.NORTH, 0):.1f}%, "
                f"South: {progress.hemisphere_percentages.get(Hemisphere.SOUTH, 0):.1f}%)"
            )

        # Check ocean paths
        if progress.ocean_path_percentage < self.config.ocean_path_minimum_percent * 0.5:
            warnings.append(
                f"Ocean path collection too low: {progress.ocean_path_percentage:.1f}% "
                f"(target: {self.config.ocean_path_minimum_percent}%)"
            )

        return warnings

    def get_rebalancing_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for rebalancing collection.

        Returns:
            List of rebalancing recommendations
        """
        recommendations = []
        progress = self.get_collection_progress()

        if progress.total_hours == 0:
            return []

        # Recommend underrepresented bands
        for band in self.get_underrepresented_bands():
            percentage = progress.latitude_band_percentages.get(band, 0)
            deficit = self.config.latitude_band_minimum_percent - percentage

            recommendations.append({
                "region": f"{band.value.capitalize()} band",
                "current_percentage": percentage,
                "target_percentage": self.config.latitude_band_minimum_percent,
                "deficit": deficit,
                "priority_multiplier": 2.0 + (deficit / 10.0),
                "action": f"Increase {band.value} band collection by {deficit:.1f}%"
            })

        # Recommend hemisphere rebalancing
        for hemisphere, target in self.config.hemisphere_targets.items():
            actual = progress.hemisphere_percentages.get(hemisphere, 0) / 100.0
            if actual < target * 0.7:  # More than 30% below target
                deficit = (target - actual) * 100
                recommendations.append({
                    "region": f"{hemisphere.value.capitalize()} hemisphere",
                    "current_percentage": actual * 100,
                    "target_percentage": target * 100,
                    "deficit": deficit,
                    "priority_multiplier": 2.0,
                    "action": f"Prioritize {hemisphere.value} hemisphere stations"
                })

        # Sort by deficit (highest priority first)
        recommendations.sort(key=lambda x: x["deficit"], reverse=True)

        return recommendations

    def get_diversity_score(self) -> float:
        """Calculate overall geographic diversity score.

        Returns:
            Diversity score (0.0 = no diversity, 1.0 = perfect diversity)
        """
        progress = self.get_collection_progress()

        if progress.total_hours == 0:
            return 0.0

        # Simpson's Diversity Index for latitude bands
        simpson_sum = 0.0
        for band in LatitudeBand:
            proportion = progress.latitude_band_percentages.get(band, 0) / 100.0
            if proportion > 0:
                simpson_sum += proportion * proportion

        simpson_diversity = 1.0 - simpson_sum if simpson_sum > 0 else 0.0

        # Hemisphere balance score
        hemisphere_score = self.get_hemisphere_balance_score()

        # Ocean path score
        ocean_target = self.config.ocean_path_minimum_percent
        ocean_actual = progress.ocean_path_percentage
        ocean_score = min(1.0, ocean_actual / ocean_target) if ocean_target > 0 else 1.0

        # Weighted average
        diversity_score = (
            simpson_diversity * 0.4 +
            hemisphere_score * 0.4 +
            ocean_score * 0.2
        )

        return diversity_score

    def get_priority_multipliers(self, total_progress_percent: float) -> Dict[str, float]:
        """Get priority multipliers based on collection progress.

        Args:
            total_progress_percent: Overall collection progress (0-100)

        Returns:
            Dictionary of priority multipliers
        """
        # Early stage: strict enforcement
        if total_progress_percent < 30:
            return {
                "underrepresented": 3.0,
                "balanced": 1.0,
                "overrepresented": 0.5
            }
        # Middle stage: moderate enforcement
        elif total_progress_percent < 70:
            return {
                "underrepresented": 2.0,
                "balanced": 1.0,
                "overrepresented": 0.7
            }
        # Late stage: relaxed enforcement
        else:
            return {
                "underrepresented": 1.5,
                "balanced": 1.0,
                "overrepresented": 0.9
            }

    def export_state(self) -> Dict[str, Any]:
        """Export quota manager state for persistence.

        Returns:
            Serializable state dictionary
        """
        return {
            "collection_history": [
                {
                    "grid_square": r["grid_square"],
                    "hours": r["hours"],
                    "timestamp": r["timestamp"].isoformat(),
                    "latitude_band": r["latitude_band"].value,
                    "hemisphere": r["hemisphere"].value,
                    "is_ocean_path": r.get("is_ocean_path", False)
                }
                for r in self.collection_history
            ],
            "last_rebalance": self.last_rebalance.isoformat()
        }

    def import_state(self, state: Dict[str, Any]):
        """Import quota manager state.

        Args:
            state: Previously exported state
        """
        self.collection_history = []

        for record in state.get("collection_history", []):
            self.collection_history.append({
                "grid_square": record["grid_square"],
                "hours": record["hours"],
                "timestamp": datetime.fromisoformat(record["timestamp"]),
                "latitude_band": LatitudeBand(record["latitude_band"]),
                "hemisphere": Hemisphere(record["hemisphere"]),
                "is_ocean_path": record.get("is_ocean_path", False)
            })

        if "last_rebalance" in state:
            self.last_rebalance = datetime.fromisoformat(state["last_rebalance"])