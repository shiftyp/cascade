"""Integration tests for automatic rebalancing triggers (T091b)."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import json

from modules.data.src.collectors.geographic_quotas import (
    GeographicQuotaManager,
    LatitudeBand,
    Hemisphere
)
from modules.data.src.collectors.scheduler import Scheduler


class TestAutomaticRebalancing:
    """Test automatic rebalancing triggers and execution."""

    def test_rebalancing_trigger_on_hemisphere_imbalance(self):
        """Test that rebalancing triggers on hemispheric imbalance."""
        quota_manager = GeographicQuotaManager()

        # Create significant hemispheric imbalance
        for _ in range(90):
            quota_manager.add_collection_record("FN42", 1.0)  # North
        for _ in range(10):
            quota_manager.add_collection_record("PF95", 1.0)  # South

        # Check hemisphere balance
        balance_score = quota_manager.get_hemisphere_balance_score()
        assert balance_score < 0.5  # Poor balance should trigger rebalancing

        # Verify warnings are generated
        warnings = quota_manager.get_quota_warnings()
        assert any("Hemispheric imbalance" in w for w in warnings)

        # Verify rebalancing recommendations
        recommendations = quota_manager.get_rebalancing_recommendations()
        assert len(recommendations) > 0

        # South should be top priority
        south_rec = next((r for r in recommendations if "South" in r["region"]), None)
        assert south_rec is not None
        assert south_rec["priority_multiplier"] >= 2.0

    def test_rebalancing_trigger_on_latitude_band_deficit(self):
        """Test rebalancing when latitude bands fall below quota."""
        quota_manager = GeographicQuotaManager()

        # Create collection missing Arctic and Antarctic
        for _ in range(50):
            quota_manager.add_collection_record("FN42", 1.0)  # Temperate
        for _ in range(50):
            quota_manager.add_collection_record("FJ09", 1.0)  # Tropical

        # Check underrepresented bands
        underrepresented = quota_manager.get_underrepresented_bands()
        assert LatitudeBand.ARCTIC in underrepresented
        assert LatitudeBand.ANTARCTIC in underrepresented

        # Verify critical warnings
        warnings = quota_manager.get_quota_warnings()
        assert any("Arctic" in w and "critically" in w for w in warnings)
        assert any("Antarctic" in w and "critically" in w for w in warnings)

    def test_rebalancing_priority_adjustment(self):
        """Test that priorities adjust based on deficit size."""
        quota_manager = GeographicQuotaManager()

        # Create varying levels of imbalance
        quota_manager.add_collection_record("FN42", 70)   # North temperate
        quota_manager.add_collection_record("FJ09", 20)   # Tropical
        quota_manager.add_collection_record("PF95", 8)    # South temperate
        quota_manager.add_collection_record("KB41", 2)    # Antarctic

        recommendations = quota_manager.get_rebalancing_recommendations()

        # Sort by deficit
        sorted_recs = sorted(recommendations, key=lambda x: x["deficit"], reverse=True)

        # Highest deficit should have highest priority
        if len(sorted_recs) >= 2:
            assert sorted_recs[0]["deficit"] > sorted_recs[1]["deficit"]
            assert sorted_recs[0]["priority_multiplier"] >= sorted_recs[1]["priority_multiplier"]

    def test_ocean_path_rebalancing(self):
        """Test rebalancing for ocean path coverage."""
        quota_manager = GeographicQuotaManager()

        # Add mostly land paths
        for _ in range(90):
            quota_manager.add_collection_record("FN42", 1.0, is_ocean_path=False)
        for _ in range(10):
            quota_manager.add_collection_record("DM14", 1.0, is_ocean_path=True)

        progress = quota_manager.get_collection_progress()
        assert progress.ocean_path_percentage < 30.0  # Below minimum

        # Should generate ocean path warning
        warnings = quota_manager.get_quota_warnings()
        assert any("Ocean path" in w for w in warnings)

    def test_progressive_rebalancing_thresholds(self):
        """Test that rebalancing thresholds change over time."""
        quota_manager = GeographicQuotaManager()

        # Early collection phase (strict thresholds)
        early_multipliers = quota_manager.get_priority_multipliers(10)
        assert early_multipliers["underrepresented"] == 3.0
        assert early_multipliers["overrepresented"] == 0.5

        # Late collection phase (relaxed thresholds)
        late_multipliers = quota_manager.get_priority_multipliers(80)
        assert late_multipliers["underrepresented"] == 1.5
        assert late_multipliers["overrepresented"] == 0.9

        # Verify progressive relaxation
        assert early_multipliers["underrepresented"] > late_multipliers["underrepresented"]

    def test_rebalancing_state_persistence(self):
        """Test that rebalancing state persists across sessions."""
        quota_manager = GeographicQuotaManager()

        # Add imbalanced data
        for _ in range(80):
            quota_manager.add_collection_record("FN42", 1.0)
        for _ in range(20):
            quota_manager.add_collection_record("PF95", 1.0)

        # Export state
        state = quota_manager.export_state()
        assert "collection_history" in state
        assert len(state["collection_history"]) == 100

        # Create new manager and import state
        new_manager = GeographicQuotaManager()
        new_manager.import_state(state)

        # Verify state was restored
        assert len(new_manager.collection_history) == 100

        # Should still detect imbalance
        balance_score = new_manager.get_hemisphere_balance_score()
        assert balance_score < 0.5

    def test_rebalancing_with_scarcity_weighting(self):
        """Test that scarce regions get extra weight during rebalancing."""
        quota_manager = GeographicQuotaManager()

        # Antarctic is always scarce
        priority_antarctic = quota_manager.should_prioritize("KB41")

        # North America is usually well-represented
        priority_na = quota_manager.should_prioritize("FN42")

        # Antarctic should have higher priority when no data collected
        assert priority_antarctic > priority_na

        # Add some North American data
        for _ in range(50):
            quota_manager.add_collection_record("FN42", 1.0)

        # Priority difference should increase
        new_priority_antarctic = quota_manager.should_prioritize("KB41")
        new_priority_na = quota_manager.should_prioritize("FN42")

        assert new_priority_antarctic > priority_antarctic  # Increased
        assert new_priority_na <= priority_na  # Same or decreased

    def test_multi_factor_rebalancing_decision(self):
        """Test rebalancing with multiple factors considered."""
        quota_manager = GeographicQuotaManager()

        # Create complex imbalance scenario
        # Heavy northern bias
        for _ in range(60):
            quota_manager.add_collection_record("FN42", 1.0, is_ocean_path=False)

        # Some tropical ocean
        for _ in range(20):
            quota_manager.add_collection_record("FJ09", 1.0, is_ocean_path=True)

        # Minimal southern
        for _ in range(10):
            quota_manager.add_collection_record("PF95", 1.0, is_ocean_path=False)

        # No arctic or antarctic
        # Total: 90% land paths, northern bias, missing polar regions

        recommendations = quota_manager.get_rebalancing_recommendations()
        warnings = quota_manager.get_quota_warnings()

        # Should identify multiple issues
        assert len(warnings) >= 3  # Hemisphere, Arctic, Antarctic at minimum
        assert len(recommendations) >= 3

        # Verify comprehensive detection
        warning_topics = " ".join(warnings)
        assert "Arctic" in warning_topics
        assert "Antarctic" in warning_topics
        assert ("South" in warning_topics or "imbalance" in warning_topics)

    def test_rebalancing_impact_on_selection(self):
        """Test that rebalancing actually affects SDR selection."""
        from modules.data.src.collectors.hybrid_sdr_selector import HybridSDRSelector

        selector = HybridSDRSelector()

        # Create heavy northern bias in quota manager
        for _ in range(100):
            selector.quota_manager.add_collection_record("FN42", 1.0)

        # Calculate priority scores for different locations
        north_priority = selector.quota_manager.should_prioritize("FN42")
        south_priority = selector.quota_manager.should_prioritize("PF95")
        antarctic_priority = selector.quota_manager.should_prioritize("KB41")

        # Southern and Antarctic should have much higher priority
        assert south_priority > north_priority * 1.5
        assert antarctic_priority > north_priority * 2.0

        # This should influence geographic scoring in selection
        north_score = selector._calculate_geographic_score("FN42", None)
        south_score = selector._calculate_geographic_score("PF95", None)

        # Even without preference, south should score higher due to quota boost
        assert south_score > north_score

    def test_rebalancing_recommendation_actionability(self):
        """Test that rebalancing recommendations are actionable."""
        quota_manager = GeographicQuotaManager()

        # Create specific imbalance
        for _ in range(100):
            quota_manager.add_collection_record("FN42", 1.0)

        recommendations = quota_manager.get_rebalancing_recommendations()

        # Each recommendation should have required fields
        for rec in recommendations:
            assert "region" in rec
            assert "current_percentage" in rec
            assert "target_percentage" in rec
            assert "deficit" in rec
            assert "priority_multiplier" in rec
            assert "action" in rec

            # Action should be specific
            assert len(rec["action"]) > 10  # Not just a placeholder
            assert "%" in rec["action"] or "increase" in rec["action"].lower()

    def test_rebalancing_convergence(self):
        """Test that repeated rebalancing converges toward targets."""
        quota_manager = GeographicQuotaManager()

        # Start with imbalance
        for _ in range(80):
            quota_manager.add_collection_record("FN42", 1.0)  # North
        for _ in range(20):
            quota_manager.add_collection_record("PF95", 1.0)  # South

        initial_balance = quota_manager.get_hemisphere_balance_score()

        # Simulate rebalancing by adding southern data
        recommendations = quota_manager.get_rebalancing_recommendations()
        for rec in recommendations:
            if "South" in rec["region"]:
                # Add data according to recommendation
                for _ in range(int(rec["deficit"])):
                    quota_manager.add_collection_record("PF95", 1.0)

        # Check improved balance
        new_balance = quota_manager.get_hemisphere_balance_score()
        assert new_balance > initial_balance

        # Fewer recommendations should remain
        new_recommendations = quota_manager.get_rebalancing_recommendations()
        assert len(new_recommendations) <= len(recommendations)