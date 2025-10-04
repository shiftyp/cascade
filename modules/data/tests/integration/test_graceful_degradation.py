"""Integration tests for graceful degradation when SDRs become unavailable.

Implements T017c: Test graceful degradation scenarios (FR-032).
These tests MUST fail initially (TDD approach).
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

# These imports will fail until implementation exists (TDD)
from modules.data.src.collectors.sdr_manager import SDRManager
from modules.data.src.collectors.degradation_handler import DegradationHandler
from modules.data.src.collectors.minimum_scheduler import MinimumScheduler
from modules.data.src.collectors.scheduler import CollectionScheduler
from modules.data.src.models.collection_alerts import CollectionAlert


class TestGracefulDegradation:
    """Test graceful degradation when SDRs fail."""

    @pytest.fixture
    async def sdr_manager(self):
        """Create SDR manager with mock SDRs."""
        manager = SDRManager()
        await manager.connect()
        yield manager
        await manager.disconnect()

    @pytest.fixture
    async def degradation_handler(self):
        """Create degradation handler."""
        handler = DegradationHandler()
        await handler.connect()
        yield handler
        await handler.disconnect()

    @pytest.fixture
    def mock_sdr_pool(self) -> List[Mock]:
        """Create pool of mock SDRs."""
        sdrs = []
        for i in range(10):
            sdr = Mock()
            sdr.url = f"sdr{i}.example.com"
            sdr.location = f"Location-{i}"
            sdr.band = ["20m", "40m", "80m"][i % 3]
            sdr.available = True
            sdr.daily_usage_minutes = 0
            sdr.max_daily_minutes = 90
            sdrs.append(sdr)
        return sdrs

    @pytest.mark.asyncio
    async def test_graceful_degradation_from_10_to_1_sdr(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test graceful degradation from 10 SDRs down to minimum 1 SDR."""
        # Arrange - Add all SDRs to pool
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Act - Simulate progressive failures
        available_counts = []
        for i in range(9):
            # Mark SDR as unavailable
            mock_sdr_pool[i].available = False
            await sdr_manager.update_sdr_status(mock_sdr_pool[i].url, available=False)

            # Check degradation response
            status = await degradation_handler.check_degradation_status()
            available_counts.append(status["available_sdrs"])

            # Verify degradation level
            if status["available_sdrs"] <= 3:
                assert status["degradation_level"] == "critical"
            elif status["available_sdrs"] <= 5:
                assert status["degradation_level"] == "warning"
            else:
                assert status["degradation_level"] == "normal"

        # Assert - System still operational with 1 SDR
        assert available_counts[-1] == 1
        assert status["operational"] is True
        assert status["minimum_met"] is True

    @pytest.mark.asyncio
    async def test_fallback_to_minimum_viable_collection(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test fallback to minimum viable collection (1 SDR)."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        scheduler = MinimumScheduler(sdr_manager)

        # Act - Simulate failure of all but one SDR
        for sdr in mock_sdr_pool[:-1]:
            await sdr_manager.mark_failed(sdr.url, "Connection timeout")

        # Start minimum collection
        schedule = await scheduler.create_minimum_schedule()

        # Assert - Minimum schedule created
        assert schedule is not None
        assert schedule["active_sdrs"] == 1
        assert schedule["collection_strategy"] == "minimum_viable"
        assert len(schedule["bands"]) >= 1  # At least one band covered

        # Verify rotation schedule for maximum coverage
        assert schedule["rotation_interval_minutes"] <= 60
        assert "priority_bands" in schedule

    @pytest.mark.asyncio
    async def test_priority_based_sdr_allocation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test priority-based SDR allocation during degradation."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Define priorities
        priorities = {
            "20m": 3,  # Highest priority
            "40m": 2,
            "80m": 1,  # Lowest priority
        }

        # Act - Reduce to 3 SDRs
        for sdr in mock_sdr_pool[3:]:
            await sdr_manager.mark_failed(sdr.url, "Unavailable")

        # Request allocation based on priorities
        allocation = await degradation_handler.allocate_sdrs_by_priority(
            available_sdrs=3,
            priorities=priorities,
        )

        # Assert - High priority bands get SDRs first
        assert len(allocation) == 3

        # Count band assignments
        band_counts = {}
        for sdr_url, band in allocation.items():
            band_counts[band] = band_counts.get(band, 0) + 1

        # 20m should have at least one SDR
        assert band_counts.get("20m", 0) >= 1

        # Higher priority bands should have more allocations
        if len(band_counts) > 1:
            assert band_counts.get("20m", 0) >= band_counts.get("80m", 0)

    @pytest.mark.asyncio
    async def test_automatic_recovery_when_sdrs_return(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test automatic recovery when SDRs become available again."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Simulate failures
        for sdr in mock_sdr_pool[:7]:
            await sdr_manager.mark_failed(sdr.url, "Connection lost")

        initial_status = await degradation_handler.check_degradation_status()
        assert initial_status["degradation_level"] == "critical"

        # Act - SDRs come back online
        recovery_events = []
        for i, sdr in enumerate(mock_sdr_pool[:7]):
            # Simulate SDR recovery
            await sdr_manager.mark_recovered(sdr.url)

            # Check status after each recovery
            status = await degradation_handler.check_degradation_status()
            recovery_events.append({
                "recovered_count": i + 1,
                "available_sdrs": status["available_sdrs"],
                "degradation_level": status["degradation_level"],
            })

            # Small delay to simulate real-world timing
            await asyncio.sleep(0.1)

        # Assert - System recovered to normal operation
        final_status = await degradation_handler.check_degradation_status()
        assert final_status["available_sdrs"] == 10
        assert final_status["degradation_level"] == "normal"
        assert final_status["operational"] is True

        # Verify progressive recovery
        assert recovery_events[0]["degradation_level"] == "critical"
        assert recovery_events[-1]["degradation_level"] in ["normal", "warning"]

    @pytest.mark.asyncio
    async def test_degradation_alerts_generation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test that alerts are generated during degradation events."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        alerts = []

        async def alert_callback(alert: Dict[str, Any]):
            alerts.append(alert)

        degradation_handler.register_alert_callback(alert_callback)

        # Act - Trigger degradation events
        # Fail 6 SDRs - should trigger WARNING
        for sdr in mock_sdr_pool[:6]:
            await sdr_manager.mark_failed(sdr.url, "Connection timeout")
            await asyncio.sleep(0.1)

        # Fail 3 more - should trigger CRITICAL
        for sdr in mock_sdr_pool[6:9]:
            await sdr_manager.mark_failed(sdr.url, "Connection timeout")
            await asyncio.sleep(0.1)

        # Assert - Alerts generated
        assert len(alerts) >= 2

        # Check for warning alert
        warning_alerts = [a for a in alerts if a["level"] == "warning"]
        assert len(warning_alerts) >= 1

        # Check for critical alert
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        assert len(critical_alerts) >= 1

        # Verify alert content
        for alert in alerts:
            assert "available_sdrs" in alert
            assert "timestamp" in alert
            assert "degradation_level" in alert

    @pytest.mark.asyncio
    async def test_band_coverage_during_degradation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test that band coverage is maintained during degradation."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Define required bands
        required_bands = ["20m", "40m", "80m"]

        # Act - Reduce SDRs gradually
        for i in range(7):
            await sdr_manager.mark_failed(mock_sdr_pool[i].url, "Unavailable")

            # Check band coverage
            coverage = await degradation_handler.check_band_coverage()

            # Assert - All required bands still covered
            for band in required_bands:
                assert band in coverage["covered_bands"], \
                    f"Band {band} not covered with {3-i} SDRs remaining"

    @pytest.mark.asyncio
    async def test_collection_quality_metrics_during_degradation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test collection quality metrics during degradation."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        scheduler = CollectionScheduler(sdr_manager)
        await scheduler.start_collection()

        # Collect baseline metrics
        baseline_metrics = await scheduler.get_collection_metrics()

        # Act - Trigger degradation
        for sdr in mock_sdr_pool[:7]:
            await sdr_manager.mark_failed(sdr.url, "Connection lost")

        # Wait for scheduler to adapt
        await asyncio.sleep(1)

        # Collect degraded metrics
        degraded_metrics = await scheduler.get_collection_metrics()

        # Assert - Quality metrics tracked
        assert "hours_per_day" in degraded_metrics
        assert "geographic_diversity" in degraded_metrics
        assert "band_coverage" in degraded_metrics

        # Collection rate should decrease but remain operational
        assert degraded_metrics["hours_per_day"] < baseline_metrics["hours_per_day"]
        assert degraded_metrics["hours_per_day"] > 0

        # Geographic diversity should decrease
        assert degraded_metrics["geographic_diversity"] <= baseline_metrics["geographic_diversity"]

    @pytest.mark.asyncio
    async def test_scheduler_adaptation_to_degradation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test that scheduler adapts to degradation conditions."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        scheduler = CollectionScheduler(sdr_manager)

        # Start with full pool
        initial_schedule = await scheduler.generate_schedule()
        assert len(initial_schedule["sessions"]) >= 6

        # Act - Trigger degradation
        for sdr in mock_sdr_pool[:7]:
            await sdr_manager.mark_failed(sdr.url, "Unavailable")

        # Scheduler should adapt
        adapted_schedule = await scheduler.generate_schedule()

        # Assert - Schedule adapted
        assert len(adapted_schedule["sessions"]) < len(initial_schedule["sessions"])
        assert adapted_schedule["strategy"] == "degraded"

        # All sessions should use available SDRs only
        available_urls = [sdr.url for sdr in mock_sdr_pool[7:]]
        for session in adapted_schedule["sessions"]:
            assert session["sdr_url"] in available_urls

    @pytest.mark.asyncio
    async def test_no_degradation_with_sufficient_sdrs(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test that no degradation is triggered with sufficient SDRs."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Act - Check status with all SDRs available
        status = await degradation_handler.check_degradation_status()

        # Assert - Normal operation
        assert status["degradation_level"] == "normal"
        assert status["operational"] is True
        assert status["available_sdrs"] == 10
        assert status["minimum_met"] is True

    @pytest.mark.asyncio
    async def test_database_persistence_during_degradation(
        self,
        sdr_manager,
        degradation_handler,
        mock_sdr_pool,
    ):
        """Test that degradation events are persisted to database."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Act - Trigger degradation events
        for sdr in mock_sdr_pool[:8]:
            await sdr_manager.mark_failed(sdr.url, "Connection timeout")

        # Get degradation events from database
        from modules.data.src.storage.metadata_db import MetadataDB
        db = MetadataDB()

        events = await db.get_degradation_events(
            since=datetime.utcnow() - timedelta(minutes=5)
        )

        # Assert - Events persisted
        assert len(events) >= 1

        for event in events:
            assert event.event_type in ["degradation_warning", "degradation_critical"]
            assert event.available_sdrs <= 10
            assert event.timestamp is not None