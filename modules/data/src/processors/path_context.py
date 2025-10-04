"""Path context calculator for propagation analysis.

Implements T032c: Path context (FR-030).
Calculates great circle paths, midpoint features, and geographic
context for propagation characterization.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class PathContext:
    """Complete path context information."""
    tx_grid: str
    rx_grid: str
    distance_km: float
    bearing_degrees: float
    reverse_bearing_degrees: float
    midpoint_lat: float
    midpoint_lon: float
    midpoint_grid: str
    crosses_terminator: bool
    terminator_crossings: int
    path_type: str
    elevation_profile: Optional[List[float]] = None
    geographic_features: Optional[Dict[str, Any]] = None


class PathContextCalculator:
    """Calculates path context for propagation analysis (FR-030)."""

    def __init__(self):
        """Initialize path context calculator."""
        # Earth radius in km
        self.earth_radius = 6371.0

        # Grid square parameters
        self.grid_field_size = 10.0  # degrees
        self.grid_square_size = 1.0  # degrees
        self.grid_subsquare_size = 1.0 / 12  # degrees

        logger.info("Path context calculator initialized")

    async def calculate_path_context(
        self,
        tx_grid: str,
        rx_grid: str,
        timestamp: Optional[datetime] = None,
    ) -> PathContext:
        """Calculate complete path context (FR-030).

        Args:
            tx_grid: Transmitter grid square (4-6 characters)
            rx_grid: Receiver grid square (4-6 characters)
            timestamp: Time for terminator calculation

        Returns:
            Complete path context
        """
        # Validate grid squares
        if not self.validate_grid_square(tx_grid):
            raise ValueError(f"Invalid TX grid square: {tx_grid}")
        if not self.validate_grid_square(rx_grid):
            raise ValueError(f"Invalid RX grid square: {rx_grid}")

        # Convert grid squares to lat/lon
        tx_lat, tx_lon = self.grid_to_latlon(tx_grid)
        rx_lat, rx_lon = self.grid_to_latlon(rx_grid)

        # Calculate great circle distance and bearing
        distance_km = self.calculate_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        bearing = self.calculate_bearing(tx_lat, tx_lon, rx_lat, rx_lon)
        reverse_bearing = self.calculate_bearing(rx_lat, rx_lon, tx_lat, tx_lon)

        # Calculate midpoint
        midpoint_lat, midpoint_lon = self.calculate_midpoint(
            tx_lat, tx_lon, rx_lat, rx_lon
        )
        midpoint_grid = self.latlon_to_grid(midpoint_lat, midpoint_lon)

        # Check terminator crossings
        crosses_terminator = False
        terminator_crossings = 0

        if timestamp:
            crosses_terminator, terminator_crossings = self.check_terminator_crossing(
                tx_lat, tx_lon, rx_lat, rx_lon, timestamp
            )

        # Classify path type
        path_type = self.classify_path_type(
            distance_km, tx_lat, rx_lat, crosses_terminator
        )

        # Calculate elevation profile (simplified)
        elevation_profile = await self._calculate_elevation_profile(
            tx_lat, tx_lon, rx_lat, rx_lon
        )

        # Extract geographic features
        geographic_features = await self._extract_geographic_features(
            midpoint_lat, midpoint_lon, tx_lat, tx_lat, rx_lat, rx_lon
        )

        context = PathContext(
            tx_grid=tx_grid,
            rx_grid=rx_grid,
            distance_km=distance_km,
            bearing_degrees=bearing,
            reverse_bearing_degrees=reverse_bearing,
            midpoint_lat=midpoint_lat,
            midpoint_lon=midpoint_lon,
            midpoint_grid=midpoint_grid,
            crosses_terminator=crosses_terminator,
            terminator_crossings=terminator_crossings,
            path_type=path_type,
            elevation_profile=elevation_profile,
            geographic_features=geographic_features,
        )

        logger.debug(
            f"Path context: {tx_grid}->{rx_grid}, "
            f"{distance_km:.0f}km, {bearing:.0f}°, {path_type}"
        )

        return context

    def grid_to_latlon(self, grid: str) -> Tuple[float, float]:
        """Convert Maidenhead grid square to lat/lon.

        Args:
            grid: Grid square (4-6 characters, e.g., FN31pr)

        Returns:
            (latitude, longitude) in degrees
        """
        if len(grid) < 4:
            raise ValueError(f"Grid square too short: {grid}")

        grid = grid.upper()

        # Field (first 2 characters)
        lon = (ord(grid[0]) - ord('A')) * 20 - 180
        lat = (ord(grid[1]) - ord('A')) * 10 - 90

        # Square (next 2 characters)
        lon += int(grid[2]) * 2
        lat += int(grid[3]) * 1

        # Subsquare (optional, next 2 characters)
        if len(grid) >= 6:
            lon += (ord(grid[4]) - ord('A')) * (2.0 / 24.0)
            lat += (ord(grid[5]) - ord('A')) * (1.0 / 24.0)

        # Return center of grid square
        lon += 1.0  # Center of 2-degree square
        lat += 0.5  # Center of 1-degree square

        return lat, lon

    def latlon_to_grid(self, lat: float, lon: float, precision: int = 4) -> str:
        """Convert lat/lon to Maidenhead grid square.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            precision: Grid precision (4 or 6 characters)

        Returns:
            Grid square string
        """
        # Adjust to grid coordinates
        lon_adj = lon + 180
        lat_adj = lat + 90

        # Field
        field_lon = int(lon_adj / 20)
        field_lat = int(lat_adj / 10)

        # Square
        square_lon = int((lon_adj % 20) / 2)
        square_lat = int((lat_adj % 10) / 1)

        grid = (
            chr(ord('A') + field_lon) +
            chr(ord('A') + field_lat) +
            str(square_lon) +
            str(square_lat)
        )

        if precision >= 6:
            # Subsquare
            subsquare_lon = int(((lon_adj % 20) % 2) / (2.0 / 24.0))
            subsquare_lat = int(((lat_adj % 10) % 1) / (1.0 / 24.0))

            grid += (
                chr(ord('a') + subsquare_lon) +
                chr(ord('a') + subsquare_lat)
            )

        return grid

    def calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate great circle distance between two points.

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.asin(math.sqrt(a))

        distance = self.earth_radius * c

        return distance

    def calculate_bearing(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate initial bearing from point 1 to point 2.

        Args:
            lat1, lon1: Start point (degrees)
            lat2, lon2: End point (degrees)

        Returns:
            Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad

        y = math.sin(dlon) * math.cos(lat2_rad)
        x = (
            math.cos(lat1_rad) * math.sin(lat2_rad) -
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        )

        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360

        return bearing_deg

    def calculate_midpoint(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> Tuple[float, float]:
        """Calculate midpoint of great circle path.

        Args:
            lat1, lon1: Start point (degrees)
            lat2, lon2: End point (degrees)

        Returns:
            (midpoint_lat, midpoint_lon) in degrees
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad

        # Calculate midpoint
        bx = math.cos(lat2_rad) * math.cos(dlon)
        by = math.cos(lat2_rad) * math.sin(dlon)

        mid_lat_rad = math.atan2(
            math.sin(lat1_rad) + math.sin(lat2_rad),
            math.sqrt((math.cos(lat1_rad) + bx) ** 2 + by ** 2)
        )

        mid_lon_rad = lon1_rad + math.atan2(by, math.cos(lat1_rad) + bx)

        mid_lat = math.degrees(mid_lat_rad)
        mid_lon = math.degrees(mid_lon_rad)

        return mid_lat, mid_lon

    def check_terminator_crossing(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        timestamp: datetime,
    ) -> Tuple[bool, int]:
        """Check if path crosses day/night terminator.

        Args:
            lat1, lon1: Start point (degrees)
            lat2, lon2: End point (degrees)
            timestamp: UTC timestamp

        Returns:
            (crosses_terminator, number_of_crossings)
        """
        # Calculate solar position (simplified)
        solar_lon = self._calculate_solar_longitude(timestamp)

        # Calculate if endpoints are on different sides of terminator
        # Terminator is approximately at solar_lon ± 90 degrees

        # Normalize longitudes relative to solar longitude
        lon1_rel = ((lon1 - solar_lon + 180) % 360) - 180
        lon2_rel = ((lon2 - solar_lon + 180) % 360) - 180

        # Check if path crosses terminator (180° apart from sun)
        # Simplified: check if longitudes are on opposite sides of terminator
        crosses = (lon1_rel * lon2_rel < 0) and (abs(lon1_rel - lon2_rel) > 90)

        # Count crossings (simplified - would need path interpolation for accuracy)
        crossings = 1 if crosses else 0

        return crosses, crossings

    def _calculate_solar_longitude(self, timestamp: datetime) -> float:
        """Calculate solar longitude at given time.

        Args:
            timestamp: UTC timestamp

        Returns:
            Solar longitude in degrees
        """
        # Simplified calculation
        # Solar longitude changes by 15 degrees per hour
        hour = timestamp.hour + timestamp.minute / 60.0
        solar_lon = (hour - 12) * 15.0  # Noon at 0° longitude

        return solar_lon

    def classify_path_type(
        self,
        distance_km: float,
        lat1: float,
        lat2: float,
        crosses_terminator: bool,
    ) -> str:
        """Classify path type based on characteristics.

        Args:
            distance_km: Path distance
            lat1, lat2: Endpoint latitudes
            crosses_terminator: Whether path crosses terminator

        Returns:
            Path type string
        """
        # Check if trans-equatorial
        if (lat1 * lat2) < 0:  # Opposite signs
            return "trans_equatorial"

        # Check if polar
        if abs(lat1) > 60 or abs(lat2) > 60:
            return "polar"

        # Check if gray-line
        if crosses_terminator:
            return "gray_line"

        # Distance-based classification
        if distance_km < 300:
            return "short"
        elif distance_km < 1000:
            return "medium"
        elif distance_km < 3000:
            return "long"
        else:
            return "dx"

    async def _calculate_elevation_profile(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        num_points: int = 10,
    ) -> List[float]:
        """Calculate elevation profile along path.

        Args:
            lat1, lon1: Start point
            lat2, lon2: End point
            num_points: Number of sample points

        Returns:
            List of elevation angles (degrees)
        """
        # Simplified elevation profile
        # In practice, would use actual terrain data

        # Calculate points along great circle
        elevations = []

        for i in range(num_points):
            fraction = i / (num_points - 1)

            # Interpolate along great circle (simplified)
            lat = lat1 + (lat2 - lat1) * fraction
            lon = lon1 + (lon2 - lon1) * fraction

            # Calculate elevation angle (simplified)
            # Assumes spherical earth and radio horizon
            distance_from_start = fraction * self.calculate_distance(
                lat1, lon1, lat2, lon2
            )

            # Radio horizon calculation
            elevation = self._calculate_radio_horizon_angle(distance_from_start)

            elevations.append(elevation)

        return elevations

    def _calculate_radio_horizon_angle(self, distance_km: float) -> float:
        """Calculate radio horizon elevation angle.

        Args:
            distance_km: Distance from transmitter

        Returns:
            Elevation angle in degrees
        """
        # Simplified calculation
        # Uses 4/3 earth radius for radio refraction
        effective_radius = self.earth_radius * 4.0 / 3.0

        # Elevation angle
        if distance_km == 0:
            return 0.0

        angle_rad = math.asin(
            min(1.0, distance_km / (2 * effective_radius))
        )

        elevation_deg = math.degrees(angle_rad)

        return elevation_deg

    async def _extract_geographic_features(
        self,
        midpoint_lat: float,
        midpoint_lon: float,
        tx_lat: float,
        tx_lon: float,
        rx_lat: float,
        rx_lon: float,
    ) -> Dict[str, Any]:
        """Extract geographic features for path.

        Args:
            midpoint_lat, midpoint_lon: Path midpoint
            tx_lat, tx_lon: Transmitter location
            rx_lat, rx_lon: Receiver location

        Returns:
            Geographic features
        """
        features = {}

        # Midpoint characteristics
        features["midpoint_latitude_zone"] = self._classify_latitude_zone(midpoint_lat)

        # Ocean vs land (simplified - would use actual geography data)
        features["over_ocean"] = self._is_over_ocean(midpoint_lat, midpoint_lon)

        # Magnetic latitude (simplified)
        features["magnetic_latitude"] = self._estimate_magnetic_latitude(midpoint_lat)

        # Terrain roughness indicator (simplified)
        features["terrain_roughness"] = 0.5  # Would calculate from elevation data

        return features

    def _classify_latitude_zone(self, lat: float) -> str:
        """Classify latitude zone.

        Args:
            lat: Latitude in degrees

        Returns:
            Zone name
        """
        abs_lat = abs(lat)

        if abs_lat < 23.5:
            return "tropical"
        elif abs_lat < 35:
            return "subtropical"
        elif abs_lat < 60:
            return "temperate"
        else:
            return "polar"

    def _is_over_ocean(self, lat: float, lon: float) -> bool:
        """Check if point is over ocean (simplified).

        Args:
            lat, lon: Coordinates

        Returns:
            True if likely over ocean
        """
        # Very simplified - would use actual land/ocean data
        # This is just a placeholder
        return False

    def _estimate_magnetic_latitude(self, geographic_lat: float) -> float:
        """Estimate magnetic latitude from geographic latitude.

        Args:
            geographic_lat: Geographic latitude

        Returns:
            Estimated magnetic latitude
        """
        # Simplified approximation
        # Actual calculation would use IGRF model
        # Magnetic pole offset roughly 11 degrees
        magnetic_lat = geographic_lat + (11 if geographic_lat > 0 else -11)

        return magnetic_lat

    def validate_grid_square(self, grid: str) -> bool:
        """Validate Maidenhead grid square format.

        Args:
            grid: Grid square string

        Returns:
            True if valid grid square
        """
        if not grid:
            return False

        grid = grid.upper()

        # Check length (4 or 6 characters)
        if len(grid) not in [4, 6]:
            return False

        # Validate field (first 2 chars: A-R)
        if len(grid) >= 2:
            if not (ord('A') <= ord(grid[0]) <= ord('R') and
                    ord('A') <= ord(grid[1]) <= ord('R')):
                return False

        # Validate square (next 2 chars: 0-9)
        if len(grid) >= 4:
            if not (grid[2].isdigit() and grid[3].isdigit()):
                return False

        # Validate subsquare if present (last 2 chars: A-X)
        if len(grid) == 6:
            if not (ord('A') <= ord(grid[4]) <= ord('X') and
                    ord('A') <= ord(grid[5]) <= ord('X')):
                return False

        return True

    def get_grid_precision(self, grid: str) -> int:
        """Get precision level of grid square.

        Args:
            grid: Grid square string

        Returns:
            Precision in km (approximate)
        """
        if not self.validate_grid_square(grid):
            return 0

        if len(grid) == 4:
            return 111  # ~111km resolution (1 degree)
        elif len(grid) == 6:
            return 5    # ~5km resolution (2.5 minutes)
        else:
            return 0

    def calculate_grid_distance_km(self, grid1: str, grid2: str) -> float:
        """Calculate distance between two grid squares.

        Args:
            grid1: First grid square
            grid2: Second grid square

        Returns:
            Distance in kilometers
        """
        if not self.validate_grid_square(grid1) or not self.validate_grid_square(grid2):
            raise ValueError("Invalid grid square(s)")

        lat1, lon1 = self.grid_to_latlon(grid1)
        lat2, lon2 = self.grid_to_latlon(grid2)

        return self.calculate_distance(lat1, lon1, lat2, lon2)
