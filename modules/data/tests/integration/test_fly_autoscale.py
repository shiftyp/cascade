"""Integration tests for Fly.io auto-scaling behavior.

Implements T017g: Test Fly.io auto-scaling (FR-041, FR-042, FR-023).
These tests MUST fail initially (TDD approach).
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List, Dict, Any

# These imports will fail until implementation exists (TDD)
from modules.data.src.collectors.event_scaler import EventScaler
from modules.data.src.collectors.scheduler import CollectionScheduler
from modules.data.src.collectors.sdr_manager import SDRManager
from modules.data.src.models.space_weather_data import SpaceWeatherData


class TestFlyAutoScale:
    """Test Fly.io auto-scaling behavior."""

    @pytest.fixture
    async def event_scaler(self):
        """Create event scaler."""
        scaler = EventScaler()
        await scaler.connect()
        yield scaler
        await scaler.disconnect()

    @pytest.fixture
    async def scheduler(self):
        """Create scheduler."""
        sdr_manager = SDRManager()
        await sdr_manager.connect()

        scheduler = CollectionScheduler(sdr_manager)
        yield scheduler

        await sdr_manager.disconnect()

    @pytest.fixture
    def mock_fly_api(self):
        """Mock Fly.io API client."""
        mock_api = Mock()
        mock_api.get_machine_count = AsyncMock(return_value=2)
        mock_api.scale_machines = AsyncMock(return_value={"success": True})
        mock_api.get_app_status = AsyncMock(return_value={
            "machines": [
                {"id": "machine1", "state": "started"},
                {"id": "machine2", "state": "started"},
            ]
        })
        return mock_api

    @pytest.mark.asyncio
    async def test_scale_up_on_solar_event(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test scale-up during solar event."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Simulate X-class flare
        space_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="X5.0",
            xray_flux=5.0e-4,
            sunspot_number=150,
            solar_flux_10_7=200,
        )

        # Act - Trigger scaling
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=2,
        )

        # Assert - Scale up decision
        assert scale_decision["action"] == "scale_up"
        assert scale_decision["target_machines"] > 2
        assert scale_decision["reason"] == "solar_event"
        assert scale_decision["event_severity"] == "high"

        # Verify scale-up would be executed
        if scale_decision["action"] == "scale_up":
            await event_scaler.execute_scaling(scale_decision)
            mock_fly_api.scale_machines.assert_called_once()

    @pytest.mark.asyncio
    async def test_scale_down_during_quiet_period(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test scale-down during quiet periods."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Simulate quiet conditions
        space_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="A1.0",
            xray_flux=1.0e-8,
            sunspot_number=0,
            solar_flux_10_7=70,
        )

        # High machine count during quiet period
        current_machines = 10

        # Act - Evaluate scaling
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=current_machines,
        )

        # Assert - Scale down decision
        assert scale_decision["action"] == "scale_down"
        assert scale_decision["target_machines"] < current_machines
        assert scale_decision["reason"] == "quiet_period"

    @pytest.mark.asyncio
    async def test_scale_up_trigger_thresholds(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test auto-scale trigger thresholds."""
        event_scaler.fly_api = mock_fly_api

        test_cases = [
            # (xray_class, expected_action, expected_machines)
            ("X5.0", "scale_up", 20),    # Major flare
            ("M5.0", "scale_up", 10),    # Moderate flare
            ("C5.0", "maintain", 2),     # Minor flare
            ("B5.0", "maintain", 2),     # Background
            ("A1.0", "scale_down", 2),   # Very quiet
        ]

        for xray_class, expected_action, expected_range in test_cases:
            # Arrange
            xray_flux = self._xray_class_to_flux(xray_class)
            space_weather = SpaceWeatherData(
                timestamp=datetime.utcnow(),
                xray_class=xray_class,
                xray_flux=xray_flux,
                sunspot_number=100,
                solar_flux_10_7=150,
            )

            # Act
            scale_decision = await event_scaler.evaluate_scaling(
                space_weather=space_weather,
                current_machines=2,
            )

            # Assert
            assert scale_decision["action"] == expected_action, \
                f"Wrong action for {xray_class}: expected {expected_action}, got {scale_decision['action']}"

    def _xray_class_to_flux(self, xray_class: str) -> float:
        """Convert X-ray class to flux value."""
        class_letter = xray_class[0]
        class_number = float(xray_class[1:])

        flux_map = {
            "X": 1e-4,
            "M": 1e-5,
            "C": 1e-6,
            "B": 1e-7,
            "A": 1e-8,
        }

        return flux_map[class_letter] * class_number

    @pytest.mark.asyncio
    async def test_gradual_scale_up_progression(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test gradual scale-up progression during events."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Start with M-class event
        m_class_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="M2.0",
            xray_flux=2.0e-5,
            sunspot_number=100,
            solar_flux_10_7=150,
        )

        # Act - First scale decision
        decision1 = await event_scaler.evaluate_scaling(
            space_weather=m_class_weather,
            current_machines=2,
        )

        # Escalate to X-class
        x_class_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="X3.0",
            xray_flux=3.0e-4,
            sunspot_number=150,
            solar_flux_10_7=200,
        )

        decision2 = await event_scaler.evaluate_scaling(
            space_weather=x_class_weather,
            current_machines=decision1["target_machines"],
        )

        # Assert - Progressive scaling
        assert decision1["target_machines"] > 2
        assert decision2["target_machines"] > decision1["target_machines"]

    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test that resource limits are enforced."""
        # Arrange
        event_scaler.fly_api = mock_fly_api
        event_scaler.max_machines = 50  # Set maximum

        # Extreme solar event
        extreme_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="X10.0",
            xray_flux=1.0e-3,
            sunspot_number=200,
            solar_flux_10_7=250,
        )

        # Act - Request scaling beyond limit
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=extreme_weather,
            current_machines=45,
        )

        # Assert - Limited to maximum
        assert scale_decision["target_machines"] <= event_scaler.max_machines
        assert scale_decision["limited_by_max"] is True

    @pytest.mark.asyncio
    async def test_minimum_machines_enforcement(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test that minimum machine count is maintained."""
        # Arrange
        event_scaler.fly_api = mock_fly_api
        event_scaler.min_machines = 2  # Minimum 2 machines

        # Very quiet conditions
        quiet_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="A1.0",
            xray_flux=1.0e-8,
            sunspot_number=0,
            solar_flux_10_7=65,
        )

        # Act - Try to scale down from minimum
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=quiet_weather,
            current_machines=2,
        )

        # Assert - Maintains minimum
        assert scale_decision["target_machines"] >= event_scaler.min_machines
        assert scale_decision["action"] in ["maintain", "scale_down"]

        if scale_decision["action"] == "scale_down":
            assert scale_decision["target_machines"] >= 2

    @pytest.mark.asyncio
    async def test_scale_based_on_queue_depth(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test scaling based on queue depth."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Normal space weather but high queue
        normal_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="C1.0",
            xray_flux=1.0e-6,
            sunspot_number=50,
            solar_flux_10_7=120,
        )

        # Act - Evaluate with high queue depth
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=normal_weather,
            current_machines=2,
            queue_depth=500,  # High queue
        )

        # Assert - Scale up due to queue
        assert scale_decision["action"] == "scale_up"
        assert "queue_backlog" in scale_decision["reason"]

    @pytest.mark.asyncio
    async def test_scale_cooldown_period(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test cooldown period between scaling operations."""
        # Arrange
        event_scaler.fly_api = mock_fly_api
        event_scaler.scale_cooldown_seconds = 300  # 5 minutes

        space_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="M5.0",
            xray_flux=5.0e-5,
            sunspot_number=100,
            solar_flux_10_7=150,
        )

        # Act - First scale operation
        decision1 = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=2,
        )

        await event_scaler.execute_scaling(decision1)

        # Immediately try to scale again
        decision2 = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=decision1["target_machines"],
        )

        # Assert - Second scale blocked by cooldown
        assert decision2["action"] == "maintain"
        assert "cooldown" in decision2.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_event_duration_based_scaling(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test scaling considers event duration."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Short M-class event
        space_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="M5.0",
            xray_flux=5.0e-5,
            sunspot_number=100,
            solar_flux_10_7=150,
        )

        # Act - Initial scale decision
        decision1 = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=2,
            event_duration_minutes=5,  # Just started
        )

        # Long-duration event
        decision2 = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=2,
            event_duration_minutes=120,  # 2 hours
        )

        # Assert - Longer events may get more resources
        # Or might scale down if event is ending
        assert decision1["target_machines"] >= 2
        assert decision2["target_machines"] >= 2

    @pytest.mark.asyncio
    async def test_geographic_load_balancing(
        self,
        event_scaler,
        scheduler,
        mock_fly_api,
    ):
        """Test geographic load balancing across regions."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Add SDRs from different regions
        regions = ["US", "EU", "Asia", "Oceania"]
        for region in regions:
            sdr = Mock()
            sdr.url = f"sdr-{region}.example.com"
            sdr.location = region
            sdr.available = True
            await scheduler.sdr_manager.add_sdr(sdr)

        # Act - Request regional scaling
        scale_decision = await event_scaler.evaluate_regional_scaling(
            target_regions=["US", "EU"],
            event_type="gray_line",
        )

        # Assert - Regional scaling planned
        assert "regions" in scale_decision
        assert "US" in scale_decision["regions"]
        assert "EU" in scale_decision["regions"]

    @pytest.mark.asyncio
    async def test_cost_optimization_during_scale_down(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test cost optimization during scale-down."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        quiet_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="B1.0",
            xray_flux=1.0e-7,
            sunspot_number=10,
            solar_flux_10_7=80,
        )

        # Act - Scale down from 10 machines
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=quiet_weather,
            current_machines=10,
            queue_depth=5,  # Low queue
        )

        # Assert - Aggressive scale-down for cost savings
        assert scale_decision["action"] == "scale_down"
        assert scale_decision["target_machines"] < 10

        # Should suggest specific machines to stop
        assert "machines_to_stop" in scale_decision or scale_decision["target_machines"] >= 2

    @pytest.mark.asyncio
    async def test_metrics_based_autoscale_decision(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test auto-scale decision based on system metrics."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        metrics = {
            "cpu_usage_percent": 85,
            "memory_usage_percent": 70,
            "queue_depth": 200,
            "active_recordings": 8,
            "available_sdrs": 50,
        }

        # Act
        scale_decision = await event_scaler.evaluate_scaling_from_metrics(
            current_machines=2,
            metrics=metrics,
        )

        # Assert - Scale up due to high utilization
        assert scale_decision["action"] == "scale_up"
        assert "cpu_usage" in scale_decision["reason"] or "queue_depth" in scale_decision["reason"]

    @pytest.mark.asyncio
    async def test_predictive_scaling_for_known_events(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test predictive scaling for known events (contests, gray-line)."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Upcoming contest event
        upcoming_event = {
            "type": "contest",
            "name": "CQ WW DX Contest",
            "start_time": datetime.utcnow() + timedelta(minutes=30),
            "expected_duration_hours": 48,
            "expected_activity_level": "very_high",
        }

        # Act - Pre-scale for event
        scale_decision = await event_scaler.evaluate_predictive_scaling(
            upcoming_event=upcoming_event,
            current_machines=2,
        )

        # Assert - Pre-emptive scale-up
        assert scale_decision["action"] == "scale_up"
        assert scale_decision["reason"] == "predictive"
        assert "contest" in scale_decision.get("event_type", "").lower()

    @pytest.mark.asyncio
    async def test_scale_down_delay_after_event(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test delayed scale-down after event ends."""
        # Arrange
        event_scaler.fly_api = mock_fly_api
        event_scaler.scale_down_delay_minutes = 30

        # Event just ended
        space_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="C1.0",  # Back to normal
            xray_flux=1.0e-6,
            sunspot_number=50,
            solar_flux_10_7=120,
        )

        # Act - Check scaling immediately after event
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=space_weather,
            current_machines=15,  # Still scaled up
            last_event_end=datetime.utcnow() - timedelta(minutes=5),  # 5 min ago
        )

        # Assert - Maintains high machine count during delay period
        assert scale_decision["action"] in ["maintain", "scale_down"]

        if scale_decision["action"] == "scale_down":
            # Should be gradual
            assert scale_decision["target_machines"] > 10

    @pytest.mark.asyncio
    async def test_emergency_scale_up_for_rare_events(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test emergency scale-up for rare high-value events."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Extremely rare X10+ flare
        rare_event = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="X15.0",
            xray_flux=1.5e-3,
            sunspot_number=250,
            solar_flux_10_7=300,
        )

        # Act
        scale_decision = await event_scaler.evaluate_scaling(
            space_weather=rare_event,
            current_machines=5,
        )

        # Assert - Maximum scale-up
        assert scale_decision["action"] == "scale_up"
        assert scale_decision["priority"] == "emergency"
        assert scale_decision["target_machines"] >= 20

    @pytest.mark.asyncio
    async def test_autoscale_respects_time_of_day(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test that auto-scale considers time of day patterns."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        normal_weather = SpaceWeatherData(
            timestamp=datetime.utcnow(),
            xray_class="C5.0",
            xray_flux=5.0e-6,
            sunspot_number=75,
            solar_flux_10_7=130,
        )

        # Act - Peak hours (gray-line period)
        peak_decision = await event_scaler.evaluate_scaling(
            space_weather=normal_weather,
            current_machines=2,
            time_of_day="gray_line",  # Peak propagation
        )

        # Off-peak hours
        offpeak_decision = await event_scaler.evaluate_scaling(
            space_weather=normal_weather,
            current_machines=2,
            time_of_day="midday",  # Poor propagation
        )

        # Assert - Different scaling for different times
        # Peak time should maintain or scale up more
        assert peak_decision["target_machines"] >= offpeak_decision["target_machines"]

    @pytest.mark.asyncio
    async def test_health_check_triggers_replacement(
        self,
        event_scaler,
        mock_fly_api,
    ):
        """Test that unhealthy machines trigger replacement."""
        # Arrange
        event_scaler.fly_api = mock_fly_api

        # Mock unhealthy machine
        mock_fly_api.get_app_status = AsyncMock(return_value={
            "machines": [
                {"id": "machine1", "state": "started", "health": "healthy"},
                {"id": "machine2", "state": "started", "health": "unhealthy"},
                {"id": "machine3", "state": "started", "health": "healthy"},
            ]
        })

        # Act
        health_decision = await event_scaler.evaluate_health_scaling(
            target_machines=3,
        )

        # Assert - Replace unhealthy machine
        assert health_decision["action"] == "replace"
        assert "machine2" in health_decision["machines_to_replace"]
        assert health_decision["target_machines"] == 3  # Maintain count