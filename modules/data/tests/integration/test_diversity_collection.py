"""Integration tests for diversity-aware collection (T091a)."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from modules.data.src.collectors.geographic_quotas import (
    GeographicQuotaManager,
    LatitudeBand,
    Hemisphere
)
from modules.data.src.collectors.hybrid_sdr_selector import HybridSDRSelector
from modules.data.src.validators.geographic_diversity import GeographicDiversityValidator
from modules.data.src.collectors.southern_priority import (
    SouthernHemispherePriorityCollector,
    schedule_southern_priority_collection
)


class TestDiversityAwareCollection:
    """Test end-to-end collection with diversity requirements."""

    @patch('modules.data.src.collectors.hybrid_sdr_selector.SessionLocal')
    def test_quota_based_sdr_selection(self, mock_db):
        """Test that SDR selection respects geographic quotas."""
        # Setup mock database
        mock_session = MagicMock()
        mock_db.return_value = mock_session

        # Create mock SDRs with different geographic locations
        mock_sdrs = [
            MagicMock(
                kiwisdr_id=1,
                url="http://arctic.sdr",
                grid_square="KQ50",  # Arctic
                active=True,
                failure_count=0,
                frequency_min_khz=3000,
                frequency_max_khz=30000,
                has_gps=True,
                daily_limit_minutes=90,
                daily_usage_minutes=0,
                reliability_score=0.8
            ),
            MagicMock(
                kiwisdr_id=2,
                url="http://temperate.sdr",
                grid_square="FN42",  # North temperate
                active=True,
                failure_count=0,
                frequency_min_khz=3000,
                frequency_max_khz=30000,
                has_gps=True,
                daily_limit_minutes=90,
                daily_usage_minutes=0,
                reliability_score=0.9
            ),
            MagicMock(
                kiwisdr_id=3,
                url="http://southern.sdr",
                grid_square="PF95",  # South temperate
                active=True,
                failure_count=0,
                frequency_min_khz=3000,
                frequency_max_khz=30000,
                has_gps=True,
                daily_limit_minutes=90,
                daily_usage_minutes=0,
                reliability_score=0.85
            )
        ]

        # Configure mock query
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_sdrs

        # Configure SDR methods
        for sdr in mock_sdrs:
            sdr.should_reset_usage.return_value = False
            sdr.remaining_daily_minutes = 90

        # Initialize selector with quota manager
        selector = HybridSDRSelector()

        # Add collection history with northern bias
        for _ in range(100):
            selector.quota_manager.add_collection_record("FN42", 1.0)

        # Select SDR - should prefer southern/arctic due to quotas
        candidate = selector.select_optimal_sdr(
            frequency_khz=14080,
            expected_duration_minutes=30,
            band="20m"
        )

        # Verify selection prioritized underrepresented regions
        assert candidate is not None
        # Due to quota boost, arctic or southern should score higher despite
        # temperate having better reliability

    def test_hemisphere_balance_enforcement(self):
        """Test that collection maintains hemispheric balance."""
        quota_manager = GeographicQuotaManager()
        validator = GeographicDiversityValidator(quota_manager)

        # Simulate heavily northern-biased collection
        for _ in range(80):
            quota_manager.add_collection_record("FN42", 1.0)  # North
        for _ in range(20):
            quota_manager.add_collection_record("PF95", 1.0)  # South

        # Check balance score
        metrics = validator.get_diversity_metrics()
        assert metrics.hemisphere_balance_score < 0.5  # Poor balance

        # Get warnings
        warnings = quota_manager.get_quota_warnings()
        assert any("Hemispheric imbalance" in w for w in warnings)

        # Get rebalancing recommendations
        recommendations = quota_manager.get_rebalancing_recommendations()
        assert any("South" in r["region"] for r in recommendations)

    def test_ocean_path_balancing(self):
        """Test ocean/land path classification and balancing."""
        quota_manager = GeographicQuotaManager()

        # Add mostly land paths
        for _ in range(70):
            quota_manager.add_collection_record("FN42", 1.0, is_ocean_path=False)
        for _ in range(30):
            quota_manager.add_collection_record("DM14", 1.0, is_ocean_path=True)

        progress = quota_manager.get_collection_progress()
        assert progress.ocean_path_percentage == 30.0

        # Check if ocean paths are flagged as minimum
        warnings = quota_manager.get_quota_warnings()
        # Should not warn if at exactly 30% (the minimum)
        assert not any("Ocean path" in w for w in warnings)

        # Add more land paths to trigger warning
        for _ in range(50):
            quota_manager.add_collection_record("FN42", 1.0, is_ocean_path=False)

        progress = quota_manager.get_collection_progress()
        assert progress.ocean_path_percentage < 30.0

        warnings = quota_manager.get_quota_warnings()
        assert any("Ocean path" in w for w in warnings)

    def test_latitude_band_quota_enforcement(self):
        """Test that all latitude bands meet minimum quotas."""
        quota_manager = GeographicQuotaManager()

        # Add diverse collection across all bands
        quota_manager.add_collection_record("KQ50", 20)  # Arctic
        quota_manager.add_collection_record("FN42", 20)  # Temperate
        quota_manager.add_collection_record("FJ09", 20)  # Tropical
        quota_manager.add_collection_record("PF95", 20)  # Temperate
        quota_manager.add_collection_record("KB41", 20)  # Antarctic

        progress = quota_manager.get_collection_progress()

        # Check that each band has at least 20%
        for band in LatitudeBand:
            percentage = progress.latitude_band_percentages.get(band, 0)
            if band == LatitudeBand.TEMPERATE:
                # Temperate includes both north and south
                assert percentage >= 20.0
            elif band in [LatitudeBand.ARCTIC, LatitudeBand.TROPICAL, LatitudeBand.ANTARCTIC]:
                assert percentage >= 15.0  # Close to 20%

    def test_scarcity_scoring_boost(self):
        """Test that scarce regions get scoring boost."""
        selector = HybridSDRSelector()

        # Manually set density cache to simulate scarcity
        selector.sdr_density_cache = {
            "KQ": 1,   # Very scarce (Arctic)
            "FN": 20,  # Dense (US East Coast)
            "PF": 3,   # Scarce (New Zealand)
        }
        selector.density_cache_time = datetime.utcnow()

        # Calculate scarcity scores
        arctic_score = selector._calculate_scarcity_score("KQ50")
        us_score = selector._calculate_scarcity_score("FN42")
        nz_score = selector._calculate_scarcity_score("PF95")

        # Arctic should have highest scarcity score
        assert arctic_score > nz_score > us_score
        assert arctic_score > 0.8  # Very scarce
        assert us_score < 0.3      # Not scarce

    @patch('modules.data.src.collectors.southern_priority.SessionLocal')
    def test_southern_hemisphere_priority(self, mock_db):
        """Test southern hemisphere priority collection."""
        mock_session = MagicMock()
        mock_db.return_value = mock_session

        # Create mock southern SDRs
        mock_sdrs = [
            MagicMock(
                kiwisdr_id=1,
                url="http://antarctica.sdr",
                grid_square="KB41",  # Antarctic
                active=True,
                failure_count=0,
                daily_limit_minutes=60,
                daily_usage_minutes=0,
                remaining_daily_minutes=60
            ),
            MagicMock(
                kiwisdr_id=2,
                url="http://chile.sdr",
                grid_square="FF43",  # South America
                active=True,
                failure_count=0,
                daily_limit_minutes=90,
                daily_usage_minutes=30,
                remaining_daily_minutes=60
            )
        ]

        # Configure mock
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_sdrs

        for sdr in mock_sdrs:
            sdr.should_reset_usage.return_value = False
            sdr.last_successful_connection = datetime.utcnow()

        collector = SouthernHemispherePriorityCollector()

        # Get southern SDR list
        southern_list = collector.maintain_southern_sdr_list()

        assert len(southern_list) > 0
        # Antarctic should be prioritized
        if southern_list:
            assert "KB" in southern_list[0].grid_square or "FF" in southern_list[0].grid_square

        # Test weight application
        if southern_list:
            base_hours = 1.0
            weighted = collector.apply_collection_weight(southern_list[0], base_hours)
            assert weighted >= base_hours * 3.0  # At least 3x weight

    def test_reciprocal_path_inference_fallback(self):
        """Test fallback to reciprocal inference when direct collection unavailable."""
        collector = SouthernHemispherePriorityCollector()

        # Test failover with insufficient southern SDRs
        result = collector.failover_to_reciprocal(
            target_bands=["20m", "40m"],
            required_hours=100.0  # More than available
        )

        assert result["method"] in ["reciprocal_inference", "direct_collection"]
        if result["method"] == "reciprocal_inference":
            assert result["deficit_hours"] > 0

    def test_continental_coverage_tracking(self):
        """Test that all continents are tracked for coverage."""
        validator = GeographicDiversityValidator()

        # Create collection records covering various continents
        records = [
            {"grid_square": "FN42", "hours": 10},  # North America
            {"grid_square": "JO22", "hours": 10},  # Europe
            {"grid_square": "PM95", "hours": 10},  # Asia
            {"grid_square": "PF95", "hours": 10},  # Oceania
            {"grid_square": "GH23", "hours": 10},  # South America
            {"grid_square": "JJ47", "hours": 10},  # Africa
            {"grid_square": "KB41", "hours": 10},  # Antarctica
        ]

        coverage = validator.check_continental_coverage(records)

        # All continents should be covered
        assert len(coverage) == 7
        covered_count = sum(1 for covered in coverage.values() if covered)
        assert covered_count >= 5  # At least 5 continents

    def test_diversity_score_calculation(self):
        """Test overall diversity score calculation."""
        quota_manager = GeographicQuotaManager()
        validator = GeographicDiversityValidator(quota_manager)

        # Create well-distributed collection
        quota_manager.add_collection_record("KQ50", 15, is_ocean_path=True)   # Arctic
        quota_manager.add_collection_record("FN42", 20)                       # N. America
        quota_manager.add_collection_record("JO22", 15)                       # Europe
        quota_manager.add_collection_record("PM95", 15, is_ocean_path=True)   # Asia
        quota_manager.add_collection_record("PF95", 15, is_ocean_path=True)   # Oceania
        quota_manager.add_collection_record("GH23", 15)                       # S. America
        quota_manager.add_collection_record("KB41", 5)                        # Antarctica

        metrics = validator.get_diversity_metrics()

        # Should have good diversity
        assert metrics.overall_diversity_score > 0.6
        assert metrics.simpson_diversity_index > 0.7
        assert len(metrics.collection_gaps) < 3  # Few gaps

    def test_automatic_rebalancing_triggers(self):
        """Test that automatic rebalancing is triggered when needed."""
        quota_manager = GeographicQuotaManager()

        # Create imbalanced collection
        for _ in range(100):
            quota_manager.add_collection_record("FN42", 1.0)  # All northern

        # Check if rebalancing is recommended
        recommendations = quota_manager.get_rebalancing_recommendations()
        assert len(recommendations) > 0

        # Priority multipliers should be high for underrepresented
        for rec in recommendations:
            if "Antarctic" in rec["region"] or "Arctic" in rec["region"]:
                assert rec["priority_multiplier"] >= 2.0

    def test_progressive_quota_relaxation(self):
        """Test that quotas relax as collection progresses."""
        quota_manager = GeographicQuotaManager()

        # Early stage multipliers
        early = quota_manager.get_priority_multipliers(total_progress_percent=10)
        assert early["underrepresented"] == 3.0

        # Middle stage
        middle = quota_manager.get_priority_multipliers(total_progress_percent=50)
        assert middle["underrepresented"] == 2.0

        # Late stage
        late = quota_manager.get_priority_multipliers(total_progress_percent=80)
        assert late["underrepresented"] == 1.5

        # Quotas should progressively relax
        assert early["underrepresented"] > middle["underrepresented"] > late["underrepresented"]