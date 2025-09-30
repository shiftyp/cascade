"""Unit tests for geographic quota system (T090a)."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from modules.data.src.collectors.geographic_quotas import (
    LatitudeBand,
    Hemisphere,
    GeographicQuotaManager,
    QuotaConfiguration,
    GridSquareClassifier,
    CollectionProgress
)


class TestLatitudeBandClassification:
    """Test latitude band classification (T083a)."""

    def test_arctic_band_classification(self):
        """Test Arctic band classification (>66.5°N)."""
        classifier = GridSquareClassifier()

        # Arctic locations (need Q and R fields for >66.5°N)
        assert classifier.get_latitude_band("KQ50") == LatitudeBand.ARCTIC  # Alaska Arctic
        assert classifier.get_latitude_band("JQ90") == LatitudeBand.ARCTIC  # Svalbard
        assert classifier.get_latitude_band("GQ87") == LatitudeBand.ARCTIC  # North Greenland

    def test_temperate_band_classification(self):
        """Test Temperate band classification (23.5-66.5°)."""
        classifier = GridSquareClassifier()

        # Northern temperate
        assert classifier.get_latitude_band("FN42") == LatitudeBand.TEMPERATE  # New York
        assert classifier.get_latitude_band("JO22") == LatitudeBand.TEMPERATE  # Netherlands
        assert classifier.get_latitude_band("PM95") == LatitudeBand.TEMPERATE  # Japan

        # Southern temperate
        assert classifier.get_latitude_band("PF95") == LatitudeBand.TEMPERATE  # New Zealand
        assert classifier.get_latitude_band("GF05") == LatitudeBand.TEMPERATE  # Argentina

    def test_tropical_band_classification(self):
        """Test Tropical band classification (±23.5°)."""
        classifier = GridSquareClassifier()

        # Tropical locations
        assert classifier.get_latitude_band("FJ09") == LatitudeBand.TROPICAL  # Colombia
        assert classifier.get_latitude_band("OJ11") == LatitudeBand.TROPICAL  # Kenya
        assert classifier.get_latitude_band("PH17") == LatitudeBand.TROPICAL  # Thailand
        assert classifier.get_latitude_band("GH23") == LatitudeBand.TROPICAL  # Brazil

    def test_antarctic_band_classification(self):
        """Test Antarctic band classification (<-66.5°S)."""
        classifier = GridSquareClassifier()

        # Antarctic locations (need A, B, or C fields for <-66.5°S)
        assert classifier.get_latitude_band("KC41") == LatitudeBand.ANTARCTIC  # Antarctica
        assert classifier.get_latitude_band("LB59") == LatitudeBand.ANTARCTIC  # Antarctica
        assert classifier.get_latitude_band("GA00") == LatitudeBand.ANTARCTIC  # South Pole area

    def test_hemisphere_classification(self):
        """Test hemisphere classification (T083c)."""
        classifier = GridSquareClassifier()

        # Northern hemisphere
        assert classifier.get_hemisphere("FN42") == Hemisphere.NORTH
        assert classifier.get_hemisphere("JO22") == Hemisphere.NORTH

        # Southern hemisphere
        assert classifier.get_hemisphere("PF95") == Hemisphere.SOUTH
        assert classifier.get_hemisphere("GF05") == Hemisphere.SOUTH

        # Equatorial (within ±10°)
        assert classifier.get_hemisphere("FJ09") == Hemisphere.EQUATORIAL
        assert classifier.get_hemisphere("OI99") == Hemisphere.EQUATORIAL


class TestQuotaConfiguration:
    """Test quota configuration (T083b)."""

    def test_default_quota_configuration(self):
        """Test default quota values."""
        config = QuotaConfiguration()

        # Latitude band minimums (20% each)
        assert config.latitude_band_minimum_percent == 20
        assert config.get_band_quota(LatitudeBand.ARCTIC) == 0.2
        assert config.get_band_quota(LatitudeBand.TEMPERATE) == 0.2
        assert config.get_band_quota(LatitudeBand.TROPICAL) == 0.2
        assert config.get_band_quota(LatitudeBand.ANTARCTIC) == 0.2

    def test_hemisphere_balance_requirements(self):
        """Test hemispheric balance requirements (T083c)."""
        config = QuotaConfiguration()

        # 40% North, 40% South, 20% Equatorial
        assert config.hemisphere_targets[Hemisphere.NORTH] == 0.4
        assert config.hemisphere_targets[Hemisphere.SOUTH] == 0.4
        assert config.hemisphere_targets[Hemisphere.EQUATORIAL] == 0.2

    def test_ocean_path_requirement(self):
        """Test ocean path minimum requirement (30%)."""
        config = QuotaConfiguration()
        assert config.ocean_path_minimum_percent == 30

    def test_custom_quota_configuration(self):
        """Test custom quota configuration."""
        config = QuotaConfiguration(
            latitude_band_minimum_percent=25,
            hemisphere_targets={
                Hemisphere.NORTH: 0.35,
                Hemisphere.SOUTH: 0.45,
                Hemisphere.EQUATORIAL: 0.20
            },
            ocean_path_minimum_percent=35
        )

        assert config.latitude_band_minimum_percent == 25
        assert config.hemisphere_targets[Hemisphere.SOUTH] == 0.45
        assert config.ocean_path_minimum_percent == 35


class TestGeographicQuotaManager:
    """Test geographic quota management."""

    def test_quota_initialization(self):
        """Test quota manager initialization."""
        manager = GeographicQuotaManager()

        assert manager.config is not None
        assert manager.classifier is not None
        assert len(manager.collection_history) == 0

    def test_should_prioritize_location(self):
        """Test location prioritization based on quotas."""
        manager = GeographicQuotaManager()

        # Add collection history with Northern bias
        manager.add_collection_record("FN42", 100)  # North temperate
        manager.add_collection_record("JO22", 80)   # North temperate
        manager.add_collection_record("PM95", 60)   # North temperate
        manager.add_collection_record("GF05", 10)   # South temperate

        # Should prioritize Southern locations
        assert manager.should_prioritize("PF95") > manager.should_prioritize("FN42")

        # Should prioritize Antarctic over well-represented temperate
        assert manager.should_prioritize("KB41") > manager.should_prioritize("FN42")

    def test_get_underrepresented_bands(self):
        """Test identification of underrepresented bands."""
        manager = GeographicQuotaManager()

        # Add biased collection history
        for _ in range(100):
            manager.add_collection_record("FN42", 1.0)  # North temperate

        for _ in range(10):
            manager.add_collection_record("GF05", 1.0)  # South temperate

        underrepresented = manager.get_underrepresented_bands()

        # Arctic, Tropical, and Antarctic should be underrepresented
        assert LatitudeBand.ARCTIC in underrepresented
        assert LatitudeBand.TROPICAL in underrepresented
        assert LatitudeBand.ANTARCTIC in underrepresented

    def test_get_hemisphere_balance_score(self):
        """Test hemispheric balance scoring."""
        manager = GeographicQuotaManager()

        # Balanced collection
        for _ in range(40):
            manager.add_collection_record("FN42", 1.0)  # North
        for _ in range(40):
            manager.add_collection_record("PF95", 1.0)  # South
        for _ in range(20):
            manager.add_collection_record("FJ09", 1.0)  # Equatorial

        score = manager.get_hemisphere_balance_score()
        assert 0.9 <= score <= 1.0  # Good balance

        # Imbalanced collection
        manager.collection_history.clear()
        for _ in range(90):
            manager.add_collection_record("FN42", 1.0)  # North only
        for _ in range(10):
            manager.add_collection_record("PF95", 1.0)  # South

        score = manager.get_hemisphere_balance_score()
        assert score < 0.5  # Poor balance

    def test_ocean_land_path_classification(self):
        """Test ocean vs land path classification (T084c)."""
        manager = GeographicQuotaManager()

        # Ocean path (both endpoints near ocean)
        is_ocean = manager.is_ocean_path("DM14", "FK52")  # California to Hawaii
        assert is_ocean == True

        # Land path (both endpoints inland)
        is_ocean = manager.is_ocean_path("EN82", "EM48")  # Inland US
        assert is_ocean == False

    def test_get_collection_progress(self):
        """Test collection progress reporting."""
        manager = GeographicQuotaManager()

        # Add diverse collection data
        manager.add_collection_record("KQ50", 10)   # Arctic
        manager.add_collection_record("FN42", 30)   # North temperate
        manager.add_collection_record("FJ09", 20)   # Tropical
        manager.add_collection_record("PF95", 25)   # South temperate
        manager.add_collection_record("KB41", 5)    # Antarctic

        progress = manager.get_collection_progress()

        assert progress.total_hours == 90
        assert len(progress.latitude_band_hours) == 4  # 4 bands, not 5 (temperate is combined N+S)
        assert len(progress.hemisphere_hours) == 3
        assert progress.latitude_band_percentages[LatitudeBand.ARCTIC] > 0
        assert progress.latitude_band_percentages[LatitudeBand.TEMPERATE] > 0
        assert progress.hemisphere_percentages[Hemisphere.NORTH] > 0

    def test_quota_enforcement_warnings(self):
        """Test quota enforcement and warning generation."""
        manager = GeographicQuotaManager()

        # Create heavily biased collection
        for _ in range(100):
            manager.add_collection_record("FN42", 1.0)  # All North temperate

        warnings = manager.get_quota_warnings()

        # Should have warnings for underrepresented bands
        assert any("Arctic" in w for w in warnings)
        assert any("Antarctic" in w for w in warnings)
        assert any("South" in w for w in warnings)  # Hemispheric imbalance warning

    def test_rebalancing_recommendations(self):
        """Test automatic rebalancing recommendations."""
        manager = GeographicQuotaManager()

        # Create biased collection
        for _ in range(100):
            manager.add_collection_record("FN42", 1.0)  # North temperate

        recommendations = manager.get_rebalancing_recommendations()

        # Should recommend prioritizing underrepresented regions
        assert any("Antarctic" in r["region"] for r in recommendations)
        assert any("South" in r["region"] for r in recommendations)
        assert all(r["priority_multiplier"] >= 2.0 for r in recommendations)

    def test_diversity_score_calculation(self):
        """Test overall geographic diversity score."""
        manager = GeographicQuotaManager()

        # Diverse collection with some ocean paths
        manager.add_collection_record("KQ50", 20, is_ocean_path=True)   # Arctic
        manager.add_collection_record("FN42", 20)   # North temperate
        manager.add_collection_record("FJ09", 20, is_ocean_path=True)   # Tropical
        manager.add_collection_record("PF95", 20, is_ocean_path=True)   # South temperate
        manager.add_collection_record("KB41", 20)   # Antarctic

        diversity_score = manager.get_diversity_score()
        assert diversity_score > 0.75  # High diversity (adjusted for realistic score)

        # Biased collection
        manager.collection_history.clear()
        for _ in range(100):
            manager.add_collection_record("FN42", 1.0)  # All same region

        diversity_score = manager.get_diversity_score()
        assert diversity_score < 0.3  # Low diversity


class TestQuotaPersistence:
    """Test quota state persistence and recovery."""

    def test_save_and_load_progress(self):
        """Test saving and loading collection progress."""
        manager = GeographicQuotaManager()

        # Add collection data
        manager.add_collection_record("FN42", 50)
        manager.add_collection_record("PF95", 30)
        manager.add_collection_record("KB41", 20)

        # Save progress
        saved_state = manager.export_state()

        # Create new manager and import state
        new_manager = GeographicQuotaManager()
        new_manager.import_state(saved_state)

        # Verify state was preserved
        assert new_manager.get_collection_progress().total_hours == 100
        assert len(new_manager.collection_history) == 3

    def test_quota_adjustment_over_time(self):
        """Test dynamic quota adjustment as collection progresses."""
        manager = GeographicQuotaManager()

        # Early stage: strict quotas
        early_multipliers = manager.get_priority_multipliers(total_progress_percent=10)

        # Late stage: relaxed quotas
        late_multipliers = manager.get_priority_multipliers(total_progress_percent=90)

        # Quotas should relax over time
        assert early_multipliers["underrepresented"] > late_multipliers["underrepresented"]