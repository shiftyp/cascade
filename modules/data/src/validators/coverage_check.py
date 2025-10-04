"""Coverage validation for geographic, temporal, and band diversity.

Implements T038: Coverage validation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class BandCoverage:
    """Coverage metrics for a single band."""

    band: str
    frequency_khz: int
    hours_collected: float
    target_hours: float
    percent_complete: float
    unique_sdrs: int
    unique_grids: int
    utc_hours_covered: int
    days_covered: int
    quality_avg: float


@dataclass
class GeographicCoverage:
    """Geographic coverage metrics."""

    total_grids: int
    grid_prefixes: List[str]
    coverage_percent: float
    min_spacing_km: float
    region_distribution: Dict[str, int]


@dataclass
class TemporalCoverage:
    """Temporal coverage metrics."""

    total_days: int
    days_with_data: int
    utc_hours_covered: int
    missing_hours: List[int]
    seasonal_balance: Dict[str, float]
    monthly_distribution: Dict[str, float]


@dataclass
class CoverageReport:
    """Comprehensive coverage report."""

    timestamp: datetime
    total_hours: float
    target_hours: float
    overall_percent: float
    band_coverage: List[BandCoverage]
    geographic_coverage: GeographicCoverage
    temporal_coverage: TemporalCoverage
    issues: List[str]
    recommendations: List[str]


class CoverageCheck:
    """Coverage validator for collection diversity."""

    # Target values per spec
    TARGET_TOTAL_HOURS = 200_000  # Updated to 200k-250k range
    MIN_HOURS_PER_BAND = 10_000
    MIN_GEOGRAPHIC_DIVERSITY = 2000  # km between stations
    MIN_GRID_PREFIXES = 20  # Minimum grid square prefixes
    MIN_UTC_HOURS = 24  # Full 24-hour coverage

    def __init__(self, db_session=None):
        """Initialize coverage checker.

        Args:
            db_session: Optional database session
        """
        from ..models import SessionLocal

        self.db = db_session or SessionLocal()
        self.owns_session = db_session is None

    def check_band_coverage(self, band: Optional[str] = None) -> List[BandCoverage]:
        """Check coverage for HF bands.

        Args:
            band: Optional specific band to check

        Returns:
            List of BandCoverage metrics
        """
        from ..models import CollectionSchedule, RecordingSession
        from sqlalchemy import func, extract

        try:
            # Get all collection schedules
            query = self.db.query(CollectionSchedule)
            if band:
                query = query.filter(CollectionSchedule.band_name == band)

            schedules = query.all()

            band_metrics = []

            for schedule in schedules:
                # Calculate hours collected
                hours_query = (
                    self.db.query(
                        func.sum(
                            extract("epoch", RecordingSession.end_time - RecordingSession.start_time) / 3600
                        )
                    )
                    .filter(
                        RecordingSession.center_frequency_hz == schedule.frequency_hz,
                        RecordingSession.end_time.isnot(None),
                    )
                    .scalar()
                )

                hours_collected = float(hours_query or 0)

                # Unique SDRs
                unique_sdrs = (
                    self.db.query(func.count(func.distinct(RecordingSession.kiwisdr_id)))
                    .filter(RecordingSession.center_frequency_hz == schedule.frequency_hz)
                    .scalar()
                    or 0
                )

                # Geographic diversity
                unique_grids = self._count_unique_grids(schedule.frequency_hz)

                # Temporal coverage
                utc_hours = self._count_utc_hours(schedule.frequency_hz)
                days_covered = (
                    self.db.query(func.count(func.distinct(func.date(RecordingSession.start_time))))
                    .filter(RecordingSession.center_frequency_hz == schedule.frequency_hz)
                    .scalar()
                    or 0
                )

                # Average quality
                quality_avg = (
                    self.db.query(func.avg(RecordingSession.quality_score))
                    .filter(
                        RecordingSession.center_frequency_hz == schedule.frequency_hz,
                        RecordingSession.quality_score.isnot(None),
                    )
                    .scalar()
                    or 0.0
                )

                band_metrics.append(
                    BandCoverage(
                        band=schedule.band_name,
                        frequency_khz=schedule.frequency_hz // 1000,
                        hours_collected=hours_collected,
                        target_hours=schedule.target_hours or self.MIN_HOURS_PER_BAND,
                        percent_complete=(hours_collected / (schedule.target_hours or self.MIN_HOURS_PER_BAND)) * 100,
                        unique_sdrs=unique_sdrs,
                        unique_grids=unique_grids,
                        utc_hours_covered=utc_hours,
                        days_covered=days_covered,
                        quality_avg=float(quality_avg),
                    )
                )

            return band_metrics

        except Exception as e:
            logger.error(f"Failed to check band coverage: {e}")
            return []

    def check_geographic_coverage(self) -> GeographicCoverage:
        """Check geographic diversity of SDR coverage.

        Returns:
            GeographicCoverage metrics
        """
        from ..models import KiwiSDRSource, RecordingSession
        from sqlalchemy import func

        try:
            # Get all SDRs used in recordings
            active_sdrs = (
                self.db.query(KiwiSDRSource)
                .join(RecordingSession, KiwiSDRSource.kiwisdr_id == RecordingSession.kiwisdr_id)
                .distinct()
                .all()
            )

            if not active_sdrs:
                return GeographicCoverage(
                    total_grids=0, grid_prefixes=[], coverage_percent=0.0, min_spacing_km=0.0, region_distribution={}
                )

            # Extract grid squares
            grid_squares = [sdr.grid_square for sdr in active_sdrs if sdr.grid_square]

            # Count grid prefixes (first 2 characters)
            grid_prefixes = list(set(grid[:2] for grid in grid_squares if len(grid) >= 2))

            # Calculate minimum spacing
            min_spacing = self._calculate_min_spacing(active_sdrs)

            # Region distribution (by grid prefix)
            region_dist = defaultdict(int)
            for grid in grid_squares:
                if len(grid) >= 2:
                    region_dist[grid[:2]] += 1

            # Coverage percent (assuming ~500 possible grid prefixes worldwide)
            coverage_percent = (len(grid_prefixes) / 500) * 100

            return GeographicCoverage(
                total_grids=len(grid_squares),
                grid_prefixes=sorted(grid_prefixes),
                coverage_percent=coverage_percent,
                min_spacing_km=min_spacing,
                region_distribution=dict(region_dist),
            )

        except Exception as e:
            logger.error(f"Failed to check geographic coverage: {e}")
            return GeographicCoverage(
                total_grids=0, grid_prefixes=[], coverage_percent=0.0, min_spacing_km=0.0, region_distribution={}
            )

    def check_temporal_coverage(self) -> TemporalCoverage:
        """Check temporal coverage across UTC hours and seasons.

        Returns:
            TemporalCoverage metrics
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            # Total days with data
            days_with_data = (
                self.db.query(func.count(func.distinct(func.date(RecordingSession.start_time)))).scalar() or 0
            )

            # Total recording period
            first_session = (
                self.db.query(func.min(RecordingSession.start_time)).filter(RecordingSession.end_time.isnot(None)).scalar()
            )
            last_session = (
                self.db.query(func.max(RecordingSession.start_time)).filter(RecordingSession.end_time.isnot(None)).scalar()
            )

            if first_session and last_session:
                total_days = (last_session - first_session).days + 1
            else:
                total_days = 0

            # UTC hours covered
            utc_hours_query = (
                self.db.query(func.distinct(extract("hour", RecordingSession.start_time)))
                .filter(RecordingSession.end_time.isnot(None))
                .all()
            )
            covered_hours = set(int(h[0]) for h in utc_hours_query)
            missing_hours = [h for h in range(24) if h not in covered_hours]

            # Seasonal balance
            seasonal_dist = self._calculate_seasonal_distribution()

            # Monthly distribution
            monthly_dist = self._calculate_monthly_distribution()

            return TemporalCoverage(
                total_days=total_days,
                days_with_data=days_with_data,
                utc_hours_covered=len(covered_hours),
                missing_hours=missing_hours,
                seasonal_balance=seasonal_dist,
                monthly_distribution=monthly_dist,
            )

        except Exception as e:
            logger.error(f"Failed to check temporal coverage: {e}")
            return TemporalCoverage(
                total_days=0, days_with_data=0, utc_hours_covered=0, missing_hours=[], seasonal_balance={}, monthly_distribution={}
            )

    def generate_coverage_report(self) -> CoverageReport:
        """Generate comprehensive coverage report.

        Returns:
            CoverageReport with all metrics
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            # Calculate total hours
            total_hours_query = (
                self.db.query(func.sum(extract("epoch", RecordingSession.end_time - RecordingSession.start_time) / 3600))
                .filter(RecordingSession.end_time.isnot(None))
                .scalar()
            )

            total_hours = float(total_hours_query or 0)
            overall_percent = (total_hours / self.TARGET_TOTAL_HOURS) * 100

            # Get all coverage metrics
            band_coverage = self.check_band_coverage()
            geographic_coverage = self.check_geographic_coverage()
            temporal_coverage = self.check_temporal_coverage()

            # Identify issues and recommendations
            issues, recommendations = self._analyze_coverage(band_coverage, geographic_coverage, temporal_coverage)

            return CoverageReport(
                timestamp=datetime.utcnow(),
                total_hours=total_hours,
                target_hours=self.TARGET_TOTAL_HOURS,
                overall_percent=overall_percent,
                band_coverage=band_coverage,
                geographic_coverage=geographic_coverage,
                temporal_coverage=temporal_coverage,
                issues=issues,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to generate coverage report: {e}")
            raise

    def _count_unique_grids(self, frequency_hz: int) -> int:
        """Count unique grid squares for a frequency.

        Args:
            frequency_hz: Center frequency

        Returns:
            Count of unique grids
        """
        from ..models import RecordingSession, KiwiSDRSource
        from sqlalchemy import func

        try:
            count = (
                self.db.query(func.count(func.distinct(KiwiSDRSource.grid_square)))
                .join(RecordingSession, KiwiSDRSource.kiwisdr_id == RecordingSession.kiwisdr_id)
                .filter(RecordingSession.center_frequency_hz == frequency_hz)
                .scalar()
                or 0
            )

            return int(count)

        except Exception as e:
            logger.error(f"Failed to count unique grids: {e}")
            return 0

    def _count_utc_hours(self, frequency_hz: int) -> int:
        """Count UTC hours covered for a frequency.

        Args:
            frequency_hz: Center frequency

        Returns:
            Number of UTC hours covered
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            hours = (
                self.db.query(func.count(func.distinct(extract("hour", RecordingSession.start_time))))
                .filter(
                    RecordingSession.center_frequency_hz == frequency_hz, RecordingSession.end_time.isnot(None)
                )
                .scalar()
                or 0
            )

            return int(hours)

        except Exception as e:
            logger.error(f"Failed to count UTC hours: {e}")
            return 0

    def _calculate_min_spacing(self, sdrs: List) -> float:
        """Calculate minimum spacing between SDRs.

        Args:
            sdrs: List of KiwiSDRSource objects

        Returns:
            Minimum spacing in km
        """
        if len(sdrs) < 2:
            return 0.0

        min_distance = float("inf")

        for i, sdr1 in enumerate(sdrs):
            for sdr2 in sdrs[i + 1 :]:
                if sdr1.latitude and sdr1.longitude and sdr2.latitude and sdr2.longitude:
                    distance = self._haversine_distance(
                        sdr1.latitude, sdr1.longitude, sdr2.latitude, sdr2.longitude
                    )
                    min_distance = min(min_distance, distance)

        return min_distance if min_distance != float("inf") else 0.0

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points.

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in km
        """
        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_seasonal_distribution(self) -> Dict[str, float]:
        """Calculate seasonal balance of recordings.

        Returns:
            Dict of season to percentage
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            # Get month distribution
            monthly = (
                self.db.query(
                    extract("month", RecordingSession.start_time).label("month"),
                    func.count(RecordingSession.session_id).label("count"),
                )
                .filter(RecordingSession.end_time.isnot(None))
                .group_by("month")
                .all()
            )

            # Map months to seasons (Northern Hemisphere)
            seasons = {"Winter": 0, "Spring": 0, "Summer": 0, "Autumn": 0}

            for month, count in monthly:
                if month in [12, 1, 2]:
                    seasons["Winter"] += count
                elif month in [3, 4, 5]:
                    seasons["Spring"] += count
                elif month in [6, 7, 8]:
                    seasons["Summer"] += count
                else:
                    seasons["Autumn"] += count

            # Convert to percentages
            total = sum(seasons.values())
            if total > 0:
                return {season: (count / total) * 100 for season, count in seasons.items()}
            else:
                return seasons

        except Exception as e:
            logger.error(f"Failed to calculate seasonal distribution: {e}")
            return {"Winter": 0, "Spring": 0, "Summer": 0, "Autumn": 0}

    def _calculate_monthly_distribution(self) -> Dict[str, float]:
        """Calculate monthly distribution of recordings.

        Returns:
            Dict of month to percentage
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            monthly = (
                self.db.query(
                    extract("month", RecordingSession.start_time).label("month"),
                    func.count(RecordingSession.session_id).label("count"),
                )
                .filter(RecordingSession.end_time.isnot(None))
                .group_by("month")
                .all()
            )

            total = sum(count for _, count in monthly)

            if total > 0:
                month_names = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                return {month_names[int(month) - 1]: (count / total) * 100 for month, count in monthly}
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to calculate monthly distribution: {e}")
            return {}

    def _analyze_coverage(
        self, band_coverage: List[BandCoverage], geo_coverage: GeographicCoverage, temp_coverage: TemporalCoverage
    ) -> Tuple[List[str], List[str]]:
        """Analyze coverage and generate issues/recommendations.

        Args:
            band_coverage: Band coverage metrics
            geo_coverage: Geographic coverage metrics
            temp_coverage: Temporal coverage metrics

        Returns:
            Tuple of (issues, recommendations)
        """
        issues = []
        recommendations = []

        # Check band coverage
        for band in band_coverage:
            if band.hours_collected < self.MIN_HOURS_PER_BAND:
                issues.append(
                    f"Band {band.band} below minimum: {band.hours_collected:.0f}/{self.MIN_HOURS_PER_BAND} hours"
                )
                recommendations.append(f"Increase collection priority for {band.band}")

            if band.utc_hours_covered < 20:
                issues.append(f"Band {band.band} missing UTC hour coverage: {band.utc_hours_covered}/24 hours")
                recommendations.append(f"Schedule {band.band} recordings across all UTC hours")

        # Check geographic diversity
        if geo_coverage.min_spacing_km < self.MIN_GEOGRAPHIC_DIVERSITY:
            issues.append(f"Insufficient geographic spacing: {geo_coverage.min_spacing_km:.0f} km minimum")
            recommendations.append("Select more geographically diverse SDRs")

        if len(geo_coverage.grid_prefixes) < self.MIN_GRID_PREFIXES:
            issues.append(f"Limited grid coverage: {len(geo_coverage.grid_prefixes)} grid prefixes")
            recommendations.append("Expand SDR selection to cover more grid squares")

        # Check temporal coverage
        if temp_coverage.utc_hours_covered < self.MIN_UTC_HOURS:
            issues.append(f"Incomplete UTC coverage: {temp_coverage.utc_hours_covered}/24 hours")
            recommendations.append(f"Schedule recordings for missing hours: {temp_coverage.missing_hours}")

        # Check seasonal balance
        if temp_coverage.seasonal_balance:
            max_season = max(temp_coverage.seasonal_balance.values())
            min_season = min(temp_coverage.seasonal_balance.values())
            if max_season > 0 and (max_season - min_season) / max_season > 0.3:  # >30% imbalance
                issues.append("Seasonal imbalance detected")
                recommendations.append("Balance collection across all seasons")

        return issues, recommendations

    def close(self):
        """Close database connection if owned."""
        if self.owns_session and self.db:
            self.db.close()


# Convenience functions
def check_overall_coverage() -> CoverageReport:
    """Get overall coverage report.

    Returns:
        CoverageReport
    """
    checker = CoverageCheck()
    try:
        return checker.generate_coverage_report()
    finally:
        checker.close()


def check_band_progress(band: str) -> Optional[BandCoverage]:
    """Check progress for specific band.

    Args:
        band: Band name

    Returns:
        BandCoverage or None
    """
    checker = CoverageCheck()
    try:
        coverage = checker.check_band_coverage(band)
        return coverage[0] if coverage else None
    finally:
        checker.close()