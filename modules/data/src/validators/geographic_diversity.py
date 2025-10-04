"""Geographic diversity metrics for data collection (T085).

Tracks and validates geographic distribution to ensure global coverage.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats
import numpy as np

from ..collectors.geographic_quotas import (
    GeographicQuotaManager,
    GridSquareClassifier,
    LatitudeBand,
    Hemisphere,
    CollectionProgress
)

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Container for geographic diversity metrics."""

    simpson_diversity_index: float
    hemisphere_balance_score: float
    continental_coverage: Dict[str, bool]
    latitude_distribution_chi_square: float
    latitude_distribution_p_value: float
    ocean_path_percentage: float
    overall_diversity_score: float
    underrepresented_regions: List[str]
    collection_gaps: List[str]


class GeographicDiversityValidator:
    """Validates and tracks geographic diversity metrics (T085)."""

    # Continental grid square prefixes (simplified)
    CONTINENTAL_GRIDS = {
        "north_america": ["CM", "DM", "EM", "FM", "CN", "DN", "EN", "FN"],
        "south_america": ["FG", "FH", "FI", "FJ", "GF", "GG", "GH", "GI"],
        "europe": ["IO", "JO", "JN", "KN", "KO", "KP"],
        "africa": ["JI", "JJ", "JK", "JL", "KH", "KI", "KJ", "KK"],
        "asia": ["MM", "NM", "OM", "PM", "MN", "NN", "ON", "PN"],
        "oceania": ["OF", "OG", "OH", "PF", "PG", "PH", "QF", "QG"],
        "antarctica": ["AA", "AB", "BA", "BB", "CA", "CB", "DA", "DB"]
    }

    def __init__(self, quota_manager: Optional[GeographicQuotaManager] = None):
        """Initialize the validator.

        Args:
            quota_manager: Geographic quota manager (creates new if None)
        """
        self.quota_manager = quota_manager or GeographicQuotaManager()
        self.classifier = GridSquareClassifier()

    def calculate_simpsons_diversity_index(
        self, collection_records: List[Dict]
    ) -> float:
        """Calculate Simpson's diversity index (T085a).

        Simpson's index measures the probability that two randomly selected
        samples will be from different geographic regions.

        Args:
            collection_records: List of collection records with grid squares

        Returns:
            Simpson's diversity index (0 = no diversity, 1 = perfect diversity)
        """
        if not collection_records:
            return 0.0

        # Count samples per grid square prefix
        grid_counts = {}
        total_samples = 0

        for record in collection_records:
            grid_square = record.get("grid_square", "")
            if len(grid_square) >= 2:
                prefix = grid_square[:2].upper()
                grid_counts[prefix] = grid_counts.get(prefix, 0) + 1
                total_samples += 1

        if total_samples == 0:
            return 0.0

        # Calculate Simpson's index
        simpson_sum = 0.0
        for count in grid_counts.values():
            proportion = count / total_samples
            simpson_sum += proportion * proportion

        # Return 1 - sum for diversity (higher = more diverse)
        return 1.0 - simpson_sum

    def calculate_hemisphere_balance(
        self, progress: CollectionProgress
    ) -> float:
        """Calculate hemispheric balance score (T085b).

        Target ratio: 0.8-1.2 between hemispheres

        Args:
            progress: Collection progress data

        Returns:
            Balance score (0 = very imbalanced, 1 = perfect balance)
        """
        north = progress.hemisphere_percentages.get(Hemisphere.NORTH, 0)
        south = progress.hemisphere_percentages.get(Hemisphere.SOUTH, 0)

        if north == 0 and south == 0:
            return 0.0
        elif north == 0 or south == 0:
            return 0.0  # Complete imbalance

        # Calculate ratio (ensure smaller/larger)
        ratio = min(north, south) / max(north, south)

        # Target range is 0.8-1.2 ratio
        # Map ratio to score
        if ratio >= 0.8:
            return 1.0  # Within target range
        else:
            # Linear decrease from 0.8 to 0
            return ratio / 0.8

    def check_continental_coverage(
        self, collection_records: List[Dict]
    ) -> Dict[str, bool]:
        """Check continental coverage (T085c).

        Args:
            collection_records: List of collection records

        Returns:
            Dictionary of continent -> covered (True/False)
        """
        covered_grids = set()

        for record in collection_records:
            grid_square = record.get("grid_square", "")
            if len(grid_square) >= 2:
                prefix = grid_square[:2].upper()
                covered_grids.add(prefix)

        # Check each continent
        coverage = {}
        for continent, grids in self.CONTINENTAL_GRIDS.items():
            # Continent is covered if any of its grids are present
            coverage[continent] = any(grid in covered_grids for grid in grids)

        return coverage

    def calculate_latitude_distribution(
        self, collection_records: List[Dict]
    ) -> Tuple[float, float]:
        """Calculate latitude distribution uniformity (T085d).

        Uses Chi-square test to check if latitude distribution is uniform.

        Args:
            collection_records: List of collection records

        Returns:
            Tuple of (chi_square_statistic, p_value)
        """
        if not collection_records:
            return 0.0, 1.0

        # Create latitude bins (every 10 degrees)
        latitude_bins = np.arange(-90, 91, 10)
        observed_counts = np.zeros(len(latitude_bins) - 1)

        # Count samples in each latitude bin
        for record in collection_records:
            grid_square = record.get("grid_square", "")
            if grid_square:
                latitude = self.classifier.get_latitude_from_grid(grid_square)
                # Find which bin this latitude falls into
                bin_index = np.digitize(latitude, latitude_bins) - 1
                if 0 <= bin_index < len(observed_counts):
                    observed_counts[bin_index] += record.get("hours", 1)

        # Expected counts (uniform distribution)
        total_hours = np.sum(observed_counts)
        if total_hours == 0:
            return 0.0, 1.0

        expected_counts = np.full_like(observed_counts, total_hours / len(observed_counts))

        # Remove bins with zero expected counts to avoid division by zero
        mask = expected_counts > 0
        observed = observed_counts[mask]
        expected = expected_counts[mask]

        if len(observed) < 2:
            return 0.0, 1.0

        # Perform Chi-square test
        chi_square, p_value = stats.chisquare(observed, expected)

        return float(chi_square), float(p_value)

    def identify_collection_gaps(
        self, collection_records: List[Dict]
    ) -> List[str]:
        """Identify geographic collection gaps.

        Args:
            collection_records: List of collection records

        Returns:
            List of gap descriptions
        """
        gaps = []

        # Check latitude bands
        progress = self.quota_manager.get_collection_progress()

        for band in LatitudeBand:
            percentage = progress.latitude_band_percentages.get(band, 0)
            if percentage < 5.0:  # Less than 5% is a gap
                gaps.append(f"{band.value.capitalize()} latitude band: {percentage:.1f}%")

        # Check continental coverage
        continental_coverage = self.check_continental_coverage(collection_records)
        for continent, covered in continental_coverage.items():
            if not covered:
                gaps.append(f"{continent.replace('_', ' ').title()} not covered")

        # Check hemisphere balance
        north = progress.hemisphere_percentages.get(Hemisphere.NORTH, 0)
        south = progress.hemisphere_percentages.get(Hemisphere.SOUTH, 0)

        if abs(north - south) > 30:  # More than 30% difference
            underrepresented = "Southern" if north > south else "Northern"
            gaps.append(f"{underrepresented} hemisphere underrepresented")

        # Check ocean paths
        if progress.ocean_path_percentage < 15:
            gaps.append(f"Ocean paths: only {progress.ocean_path_percentage:.1f}%")

        return gaps

    def get_diversity_metrics(self) -> DiversityMetrics:
        """Get comprehensive diversity metrics.

        Returns:
            DiversityMetrics object with all calculated metrics
        """
        # Get collection records from quota manager
        collection_records = self.quota_manager.collection_history
        progress = self.quota_manager.get_collection_progress()

        # Calculate all metrics
        simpson = self.calculate_simpsons_diversity_index(collection_records)
        hemisphere_balance = self.calculate_hemisphere_balance(progress)
        continental = self.check_continental_coverage(collection_records)
        chi_square, p_value = self.calculate_latitude_distribution(collection_records)
        ocean_percentage = progress.ocean_path_percentage

        # Overall diversity score (weighted average)
        continents_covered = sum(1 for covered in continental.values() if covered)
        continental_score = continents_covered / 7.0

        overall_score = (
            simpson * 0.25 +
            hemisphere_balance * 0.25 +
            continental_score * 0.25 +
            min(1.0, ocean_percentage / 30.0) * 0.25
        )

        # Identify underrepresented regions
        underrepresented = []
        for band in self.quota_manager.get_underrepresented_bands():
            underrepresented.append(band.value.capitalize())

        # Get collection gaps
        gaps = self.identify_collection_gaps(collection_records)

        return DiversityMetrics(
            simpson_diversity_index=simpson,
            hemisphere_balance_score=hemisphere_balance,
            continental_coverage=continental,
            latitude_distribution_chi_square=chi_square,
            latitude_distribution_p_value=p_value,
            ocean_path_percentage=ocean_percentage,
            overall_diversity_score=overall_score,
            underrepresented_regions=underrepresented,
            collection_gaps=gaps
        )

    def validate_diversity_requirements(
        self,
        min_simpson: float = 0.6,
        min_hemisphere_balance: float = 0.5,
        min_continents: int = 5,
        min_ocean_percentage: float = 20.0
    ) -> Tuple[bool, List[str]]:
        """Validate diversity requirements are met.

        Args:
            min_simpson: Minimum Simpson's diversity index
            min_hemisphere_balance: Minimum hemisphere balance score
            min_continents: Minimum number of continents covered
            min_ocean_percentage: Minimum ocean path percentage

        Returns:
            Tuple of (all_requirements_met, list_of_failures)
        """
        metrics = self.get_diversity_metrics()
        failures = []

        if metrics.simpson_diversity_index < min_simpson:
            failures.append(
                f"Simpson's diversity index {metrics.simpson_diversity_index:.2f} "
                f"< {min_simpson} minimum"
            )

        if metrics.hemisphere_balance_score < min_hemisphere_balance:
            failures.append(
                f"Hemisphere balance {metrics.hemisphere_balance_score:.2f} "
                f"< {min_hemisphere_balance} minimum"
            )

        continents_covered = sum(1 for c in metrics.continental_coverage.values() if c)
        if continents_covered < min_continents:
            failures.append(
                f"Only {continents_covered} continents covered "
                f"< {min_continents} minimum"
            )

        if metrics.ocean_path_percentage < min_ocean_percentage:
            failures.append(
                f"Ocean paths {metrics.ocean_path_percentage:.1f}% "
                f"< {min_ocean_percentage}% minimum"
            )

        return len(failures) == 0, failures

    def generate_diversity_report(self) -> str:
        """Generate a human-readable diversity report.

        Returns:
            Formatted diversity report string
        """
        metrics = self.get_diversity_metrics()

        report = []
        report.append("=" * 60)
        report.append("Geographic Diversity Report")
        report.append("=" * 60)
        report.append("")

        # Overall score
        report.append(f"Overall Diversity Score: {metrics.overall_diversity_score:.2f}/1.00")
        report.append("")

        # Key metrics
        report.append("Key Metrics:")
        report.append(f"  • Simpson's Diversity Index: {metrics.simpson_diversity_index:.3f}")
        report.append(f"  • Hemisphere Balance: {metrics.hemisphere_balance_score:.2f}")
        report.append(f"  • Ocean Path Coverage: {metrics.ocean_path_percentage:.1f}%")
        report.append("")

        # Continental coverage
        report.append("Continental Coverage:")
        for continent, covered in sorted(metrics.continental_coverage.items()):
            status = "✓" if covered else "✗"
            report.append(f"  {status} {continent.replace('_', ' ').title()}")
        report.append("")

        # Latitude distribution
        report.append("Latitude Distribution:")
        report.append(f"  • Chi-square statistic: {metrics.latitude_distribution_chi_square:.2f}")
        report.append(f"  • P-value: {metrics.latitude_distribution_p_value:.4f}")

        if metrics.latitude_distribution_p_value < 0.05:
            report.append("  ⚠ Distribution significantly non-uniform")
        else:
            report.append("  ✓ Distribution reasonably uniform")
        report.append("")

        # Underrepresented regions
        if metrics.underrepresented_regions:
            report.append("Underrepresented Regions:")
            for region in metrics.underrepresented_regions:
                report.append(f"  • {region}")
            report.append("")

        # Collection gaps
        if metrics.collection_gaps:
            report.append("Collection Gaps:")
            for gap in metrics.collection_gaps:
                report.append(f"  • {gap}")
        else:
            report.append("No significant collection gaps identified.")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)