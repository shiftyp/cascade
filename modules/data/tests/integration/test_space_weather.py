"""Integration tests for space weather event triggering and SDR scaling.

Tests the integration between NOAA space weather API, event detection,
and automatic SDR scaling for rare event capture.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json

# Import components to test
from modules.data.src.external.noaa_client import NOAASpaceWeatherClient
from modules.data.src.collectors.event_scaler import EventBasedScaler
from modules.data.src.collectors.scheduler import CollectionScheduler


class TestSpaceWeatherTrigger:
    """Test space weather event detection and response."""

    @pytest.fixture
    def noaa_client(self):
        """Create NOAA client instance."""
        return NOAASpaceWeatherClient()

    @pytest.fixture
    def event_scaler(self):
        """Create event scaler instance."""
        return EventBasedScaler()

    @pytest.fixture
    def scheduler(self):
        """Create collection scheduler instance."""
        return CollectionScheduler()

    @pytest.mark.asyncio
    async def test_xray_flare_detection(self, noaa_client):
        """Test detection of X-ray flares from NOAA data."""
        # Mock NOAA API response
        mock_response = {
            "x_ray_flux": 1.5e-4,  # M-class flare level
            "x_ray_class": "M1.5",
            "timestamp": datetime.utcnow().isoformat()
        }

        with patch.object(noaa_client, 'fetch_current_conditions', return_value=mock_response):
            conditions = await noaa_client.fetch_current_conditions()

            # Verify flare detection
            assert conditions['x_ray_class'] == 'M1.5'
            assert conditions['x_ray_flux'] > 1e-5  # Above C-class threshold

    @pytest.mark.asyncio
    async def test_solar_wind_spike_detection(self, noaa_client):
        """Test detection of solar wind velocity spikes."""
        # Mock enhanced solar wind conditions
        mock_response = {
            "solar_wind_speed": 650,  # km/s - enhanced conditions
            "particle_density": 15,    # p/cm³
            "magnetic_field": 12,      # nT
            "timestamp": datetime.utcnow().isoformat()
        }

        with patch.object(noaa_client, 'fetch_solar_wind', return_value=mock_response):
            wind_data = await noaa_client.fetch_solar_wind()

            # Verify enhanced conditions detected
            assert wind_data['solar_wind_speed'] > 500  # Enhanced threshold
            assert wind_data['particle_density'] > 10

    @pytest.mark.asyncio
    async def test_kp_index_storm_detection(self, noaa_client):
        """Test geomagnetic storm detection via Kp index."""
        # Mock storm conditions
        mock_response = {
            "kp_index": 7,  # G3 storm level
            "ap_index": 120,
            "storm_level": "G3",
            "timestamp": datetime.utcnow().isoformat()
        }

        with patch.object(noaa_client, 'fetch_geomagnetic_indices', return_value=mock_response):
            indices = await noaa_client.fetch_geomagnetic_indices()

            # Verify storm detection
            assert indices['kp_index'] >= 7
            assert indices['storm_level'] == 'G3'

    @pytest.mark.asyncio
    async def test_event_triggers_scaling(self, event_scaler, noaa_client):
        """Test that space weather events trigger SDR scaling."""
        # Mock X-class flare event
        mock_event = {
            "type": "x_ray_flare",
            "x_ray_class": "X2.1",
            "x_ray_flux": 2.1e-4,
            "timestamp": datetime.utcnow().isoformat()
        }

        with patch.object(noaa_client, 'fetch_current_conditions', return_value=mock_event):
            # Process event
            scaling_response = await event_scaler.process_space_weather_event(mock_event)

            # Verify scaling triggered
            assert scaling_response['action'] == 'scale_up'
            assert scaling_response['target_sdrs'] >= 20  # Solar minimum boost
            assert scaling_response['priority'] == 'high'

    @pytest.mark.asyncio
    async def test_multiple_event_correlation(self, event_scaler):
        """Test correlation of multiple simultaneous events."""
        # Mock multiple events
        events = [
            {"type": "x_ray_flare", "x_ray_class": "M5.0"},
            {"type": "solar_wind", "speed": 700},
            {"type": "geomagnetic", "kp_index": 6}
        ]

        # Process correlated events
        response = await event_scaler.process_correlated_events(events)

        # Verify enhanced response for multiple events
        assert response['correlation_detected'] is True
        assert response['target_sdrs'] >= 30  # Higher for correlated events
        assert response['collection_duration'] >= 6  # Extended duration

    @pytest.mark.asyncio
    async def test_event_priority_queue(self, scheduler):
        """Test prioritization of event-driven collections."""
        # Create normal and event collections
        normal_task = {
            "type": "routine",
            "band": "20m",
            "priority": "normal"
        }

        event_task = {
            "type": "space_weather",
            "band": "all",
            "priority": "critical",
            "event": "X1.5_flare"
        }

        # Add tasks to scheduler
        await scheduler.add_task(normal_task)
        await scheduler.add_task(event_task)

        # Verify event task gets priority
        next_task = await scheduler.get_next_task()
        assert next_task['type'] == 'space_weather'
        assert next_task['priority'] == 'critical'

    @pytest.mark.asyncio
    async def test_scaling_resource_limits(self, event_scaler):
        """Test that scaling respects resource limits."""
        # Mock extreme event requiring many SDRs
        extreme_event = {
            "type": "x_ray_flare",
            "x_ray_class": "X10.0",
            "requested_sdrs": 200  # More than available
        }

        # Process with resource constraints
        response = await event_scaler.process_with_limits(extreme_event)

        # Verify capped at maximum available
        assert response['allocated_sdrs'] <= 100  # Maximum pool size
        assert response['status'] == 'partial_allocation'

    @pytest.mark.asyncio
    async def test_event_duration_calculation(self, event_scaler):
        """Test calculation of collection duration based on event type."""
        # Test different event types
        events = [
            {"type": "x_ray_flare", "class": "M1.0"},  # 2-4 hours
            {"type": "x_ray_flare", "class": "X1.0"},  # 4-6 hours
            {"type": "cme_arrival", "speed": 800},     # 6-12 hours
            {"type": "geomagnetic_storm", "kp": 8}     # 12-24 hours
        ]

        for event in events:
            duration = await event_scaler.calculate_collection_duration(event)

            # Verify appropriate durations
            if event['type'] == 'x_ray_flare':
                if 'X' in event.get('class', ''):
                    assert 4 <= duration <= 6
                else:
                    assert 2 <= duration <= 4
            elif event['type'] == 'cme_arrival':
                assert 6 <= duration <= 12
            elif event['type'] == 'geomagnetic_storm':
                assert 12 <= duration <= 24

    @pytest.mark.asyncio
    async def test_event_cooldown_period(self, event_scaler):
        """Test cooldown period after event to prevent thrashing."""
        # Trigger an event
        event = {"type": "x_ray_flare", "class": "M5.0"}
        await event_scaler.process_space_weather_event(event)

        # Try to trigger similar event immediately
        similar_event = {"type": "x_ray_flare", "class": "M3.0"}
        response = await event_scaler.process_space_weather_event(similar_event)

        # Verify cooldown prevents immediate rescaling
        assert response['status'] == 'cooldown_active'
        assert response['remaining_cooldown'] > 0

    @pytest.mark.asyncio
    async def test_event_metadata_recording(self, scheduler, event_scaler):
        """Test that event metadata is properly recorded."""
        # Process an event
        event = {
            "type": "x_ray_flare",
            "x_ray_class": "X1.5",
            "peak_flux": 1.5e-4,
            "start_time": datetime.utcnow().isoformat()
        }

        response = await event_scaler.process_space_weather_event(event)

        # Verify metadata captured
        assert 'correlation_id' in response
        assert response['event_metadata']['x_ray_class'] == 'X1.5'
        assert response['event_metadata']['peak_flux'] == 1.5e-4
        assert 'collection_start' in response
        assert 'collection_end' in response


class TestEventNotifications:
    """Test event notification system."""

    @pytest.mark.asyncio
    async def test_critical_event_notification(self):
        """Test notifications for critical space weather events."""
        notifier = Mock()

        # Mock critical event
        event = {
            "type": "x_ray_flare",
            "x_ray_class": "X5.0",
            "severity": "critical"
        }

        # Send notification
        await notifier.send_event_alert(event)

        # Verify notification sent
        notifier.send_event_alert.assert_called_once()
        call_args = notifier.send_event_alert.call_args[0][0]
        assert call_args['severity'] == 'critical'

    @pytest.mark.asyncio
    async def test_scaling_status_notification(self):
        """Test notifications for scaling status changes."""
        notifier = Mock()

        # Mock scaling event
        scaling_event = {
            "action": "scale_up",
            "from_sdrs": 6,
            "to_sdrs": 30,
            "reason": "X2.1 flare detected"
        }

        # Send notification
        await notifier.send_scaling_alert(scaling_event)

        # Verify notification sent
        notifier.send_scaling_alert.assert_called_once()


class TestEventPersistence:
    """Test event data persistence and recovery."""

    @pytest.mark.asyncio
    async def test_event_history_storage(self):
        """Test storage of event history for analysis."""
        storage = Mock()

        # Create event record
        event_record = {
            "event_id": "evt_001",
            "type": "x_ray_flare",
            "x_ray_class": "M8.7",
            "timestamp": datetime.utcnow().isoformat(),
            "sdrs_allocated": 25,
            "collection_hours": 4.5
        }

        # Store event
        await storage.store_event_record(event_record)

        # Verify storage
        storage.store_event_record.assert_called_once_with(event_record)

    @pytest.mark.asyncio
    async def test_event_recovery_after_crash(self):
        """Test recovery of active events after system crash."""
        storage = Mock()
        scheduler = Mock()

        # Mock active events from storage
        active_events = [
            {"event_id": "evt_001", "status": "collecting", "remaining_hours": 2},
            {"event_id": "evt_002", "status": "queued", "remaining_hours": 4}
        ]

        storage.get_active_events.return_value = active_events

        # Recover events
        recovered = await scheduler.recover_active_events()

        # Verify recovery
        assert len(recovered) == 2
        assert recovered[0]['status'] == 'collecting'