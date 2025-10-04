"""Integration tests for recovery from complete SDR pool failure.

Implements T017d: Test total failure recovery scenarios (FR-033, FR-034).
These tests MUST fail initially (TDD approach).
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

# These imports will fail until implementation exists (TDD)
from modules.data.src.collectors.sdr_manager import SDRManager
from modules.data.src.collectors.health_monitor import HealthMonitor
from modules.data.src.collectors.queue_manager import QueueManager
from modules.data.src.collectors.scheduler import CollectionScheduler
from modules.data.src.models.collection_alerts import CollectionAlert


class TestTotalFailureRecovery:
    """Test recovery from complete SDR pool failure."""

    @pytest.fixture
    async def sdr_manager(self):
        """Create SDR manager."""
        manager = SDRManager()
        await manager.connect()
        yield manager
        await manager.disconnect()

    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor."""
        monitor = HealthMonitor()
        await monitor.connect()
        yield monitor
        await monitor.disconnect()

    @pytest.fixture
    async def queue_manager(self):
        """Create queue manager."""
        queue = QueueManager()
        await queue.connect()
        yield queue
        await queue.disconnect()

    @pytest.fixture
    def mock_sdr_pool(self) -> List[Mock]:
        """Create pool of mock SDRs."""
        sdrs = []
        for i in range(6):
            sdr = Mock()
            sdr.url = f"sdr{i}.example.com"
            sdr.location = f"Location-{i}"
            sdr.band = ["20m", "40m", "80m"][i % 3]
            sdr.available = True
            sdrs.append(sdr)
        return sdrs

    @pytest.mark.asyncio
    async def test_complete_sdr_pool_failure(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test detection of complete SDR pool failure."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Register failure callback
        failures_detected = []

        async def failure_callback(event):
            failures_detected.append(event)

        health_monitor.register_callback("total_failure", failure_callback)

        # Act - Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Network outage")
            await asyncio.sleep(0.05)

        # Wait for health monitor to detect failure
        await asyncio.sleep(0.5)

        # Assert - Total failure detected
        assert len(failures_detected) >= 1

        failure_event = failures_detected[0]
        assert failure_event["event_type"] == "total_sdr_failure"
        assert failure_event["available_sdrs"] == 0
        assert "timestamp" in failure_event

    @pytest.mark.asyncio
    async def test_queue_preservation_during_outage(
        self,
        sdr_manager,
        queue_manager,
        mock_sdr_pool,
    ):
        """Test that job queue is preserved during SDR outage."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Queue up collection jobs
        job_ids = []
        for i in range(10):
            job_id = await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {
                    "sdr_url": mock_sdr_pool[i % 6].url,
                    "frequency_khz": 14074 + i,
                    "duration_seconds": 300,
                },
            )
            job_ids.append(job_id)

        # Get initial queue state
        initial_stats = await queue_manager.get_queue_stats()
        initial_length = initial_stats[queue_manager.COLLECTION_QUEUE]

        # Act - Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Connection lost")

        # Wait during outage
        await asyncio.sleep(1)

        # Get queue state during outage
        outage_stats = await queue_manager.get_queue_stats()
        outage_length = outage_stats[queue_manager.COLLECTION_QUEUE]

        # Assert - Queue preserved
        assert outage_length == initial_length
        assert outage_length == 10

        # Verify job data intact
        for job_id in job_ids:
            job_status = await queue_manager.client.hget(f"job:{job_id}", "status")
            assert job_status in ["queued", "processing"]

    @pytest.mark.asyncio
    async def test_automatic_reconnection_attempts(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test automatic reconnection attempts during outage."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Network timeout")

        # Track reconnection attempts
        reconnection_attempts = []

        async def attempt_callback(event):
            reconnection_attempts.append(event)

        health_monitor.register_callback("reconnection_attempt", attempt_callback)

        # Act - Start health monitoring (should trigger reconnection attempts)
        monitor_task = asyncio.create_task(health_monitor.monitor_and_recover())

        # Wait for multiple reconnection cycles
        await asyncio.sleep(3)

        # Stop monitoring
        monitor_task.cancel()

        # Assert - Multiple reconnection attempts made
        assert len(reconnection_attempts) >= 2

        for attempt in reconnection_attempts:
            assert "timestamp" in attempt
            assert "sdrs_checked" in attempt
            assert attempt["sdrs_checked"] == len(mock_sdr_pool)

    @pytest.mark.asyncio
    async def test_exponential_backoff_for_reconnection(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test exponential backoff for reconnection attempts."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Connection refused")

        # Track attempt timing
        attempt_times = []

        async def timing_callback(event):
            attempt_times.append(datetime.utcnow())

        health_monitor.register_callback("reconnection_attempt", timing_callback)

        # Act - Monitor with backoff
        monitor_task = asyncio.create_task(
            health_monitor.monitor_and_recover(
                initial_backoff_seconds=1,
                max_backoff_seconds=8,
            )
        )

        # Wait for several attempts
        await asyncio.sleep(15)
        monitor_task.cancel()

        # Assert - Backoff increases
        assert len(attempt_times) >= 3

        # Calculate intervals between attempts
        intervals = []
        for i in range(1, len(attempt_times)):
            interval = (attempt_times[i] - attempt_times[i-1]).total_seconds()
            intervals.append(interval)

        # Verify exponential increase (with some tolerance)
        for i in range(1, len(intervals)):
            # Each interval should be roughly 2x previous (±30% tolerance)
            expected_min = intervals[i-1] * 1.7
            assert intervals[i] >= expected_min or intervals[i] >= 8, \
                f"Backoff not exponential: {intervals}"

    @pytest.mark.asyncio
    async def test_data_backfill_after_recovery(
        self,
        sdr_manager,
        queue_manager,
        mock_sdr_pool,
    ):
        """Test data backfill after SDR pool recovery."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        scheduler = CollectionScheduler(sdr_manager)

        # Schedule continuous collection
        await scheduler.start_collection()

        # Queue initial jobs
        await scheduler.schedule_next_batch()
        initial_jobs = await queue_manager.get_queue_stats()

        # Act - Simulate outage
        outage_start = datetime.utcnow()
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Network outage")

        # Wait during outage (2 minutes simulated)
        await asyncio.sleep(2)

        # Recover SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_recovered(sdr.url)

        outage_end = datetime.utcnow()
        outage_duration_minutes = (outage_end - outage_start).total_seconds() / 60

        # Trigger backfill
        backfill_jobs = await scheduler.schedule_backfill(
            start_time=outage_start,
            end_time=outage_end,
        )

        # Assert - Backfill jobs created
        assert len(backfill_jobs) > 0

        # Verify backfill coverage
        # Should have jobs to cover missed collection periods
        expected_min_jobs = int(outage_duration_minutes / 5)  # Assuming 5-min collection periods
        assert len(backfill_jobs) >= expected_min_jobs

        # Verify backfill jobs have correct time range
        for job in backfill_jobs:
            job_time = datetime.fromisoformat(job["scheduled_time"])
            assert outage_start <= job_time <= outage_end

    @pytest.mark.asyncio
    async def test_alert_notification_on_total_failure(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test that critical alerts are sent on total failure."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        alerts_sent = []

        async def alert_callback(alert):
            alerts_sent.append(alert)

        health_monitor.register_alert_callback(alert_callback)

        # Act - Trigger total failure
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Total network failure")

        # Wait for alerts
        await asyncio.sleep(0.5)

        # Assert - Critical alert sent
        assert len(alerts_sent) >= 1

        critical_alert = alerts_sent[0]
        assert critical_alert["level"] == "critical"
        assert critical_alert["event_type"] == "total_sdr_failure"
        assert critical_alert["available_sdrs"] == 0
        assert "notification_required" in critical_alert
        assert critical_alert["notification_required"] is True

    @pytest.mark.asyncio
    async def test_recovery_notification_after_restoration(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test that recovery notification is sent after restoration."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        notifications = []

        async def notification_callback(notification):
            notifications.append(notification)

        health_monitor.register_notification_callback(notification_callback)

        # Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Outage")

        await asyncio.sleep(0.5)

        # Act - Recover SDRs
        for sdr in mock_sdr_pool[:4]:  # Recover 4 out of 6
            await sdr_manager.mark_recovered(sdr.url)

        await asyncio.sleep(0.5)

        # Assert - Recovery notification sent
        recovery_notifications = [
            n for n in notifications
            if n["type"] == "recovery"
        ]

        assert len(recovery_notifications) >= 1

        recovery_notif = recovery_notifications[0]
        assert recovery_notif["available_sdrs"] >= 4
        assert "recovered_from_failure" in recovery_notif
        assert recovery_notif["recovered_from_failure"] is True

    @pytest.mark.asyncio
    async def test_partial_recovery_handling(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test handling of partial recovery (not all SDRs return)."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Network issue")

        # Act - Recover only 2 SDRs
        await sdr_manager.mark_recovered(mock_sdr_pool[0].url)
        await sdr_manager.mark_recovered(mock_sdr_pool[1].url)

        # Check recovery status
        status = await health_monitor.check_recovery_status()

        # Assert - Partial recovery detected
        assert status["recovery_type"] == "partial"
        assert status["available_sdrs"] == 2
        assert status["total_sdrs"] == 6
        assert status["recovery_percentage"] == pytest.approx(33.3, abs=1)

        # Should still be in degraded mode
        assert status["operational_mode"] == "degraded"

    @pytest.mark.asyncio
    async def test_queue_processing_resumes_after_recovery(
        self,
        sdr_manager,
        queue_manager,
        mock_sdr_pool,
    ):
        """Test that queue processing resumes after recovery."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Queue jobs
        for i in range(5):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {
                    "sdr_url": mock_sdr_pool[i].url,
                    "frequency_khz": 14074,
                    "duration_seconds": 60,
                },
            )

        # Fail all SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Failure")

        # Act - Recover SDRs
        await asyncio.sleep(0.5)
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_recovered(sdr.url)

        # Start worker to process queue
        from modules.data.src.collectors.worker import CollectionWorker
        worker = CollectionWorker()

        # Process jobs
        process_task = asyncio.create_task(worker.start())
        await asyncio.sleep(2)
        await worker.stop()
        process_task.cancel()

        # Assert - Jobs processed
        assert worker.jobs_completed > 0

        # Queue should be drained
        final_stats = await queue_manager.get_queue_stats()
        assert final_stats[queue_manager.COLLECTION_QUEUE] < 5

    @pytest.mark.asyncio
    async def test_failure_metrics_persistence(
        self,
        sdr_manager,
        health_monitor,
        mock_sdr_pool,
    ):
        """Test that failure metrics are persisted to database."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Act - Trigger total failure
        failure_start = datetime.utcnow()
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Total failure")

        await asyncio.sleep(1)

        # Recover after 1 second
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_recovered(sdr.url)

        failure_end = datetime.utcnow()

        # Assert - Metrics stored
        from modules.data.src.storage.metadata_db import MetadataDB
        db = MetadataDB()

        failure_events = await db.get_failure_events(
            since=failure_start - timedelta(seconds=5)
        )

        assert len(failure_events) >= 1

        # Check for total failure event
        total_failures = [
            e for e in failure_events
            if e.event_type == "total_sdr_failure"
        ]
        assert len(total_failures) >= 1

        # Verify failure duration
        failure_event = total_failures[0]
        assert failure_event.timestamp >= failure_start
        assert failure_event.timestamp <= failure_end

    @pytest.mark.asyncio
    async def test_no_data_loss_during_recovery(
        self,
        sdr_manager,
        queue_manager,
        mock_sdr_pool,
    ):
        """Test that no data is lost during recovery process."""
        # Arrange
        for sdr in mock_sdr_pool:
            await sdr_manager.add_sdr(sdr)

        # Create tracking of all jobs
        all_job_ids = set()

        # Queue 20 jobs
        for i in range(20):
            job_id = await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {
                    "sdr_url": mock_sdr_pool[i % 6].url,
                    "frequency_khz": 14074,
                    "duration_seconds": 60,
                },
            )
            all_job_ids.add(job_id)

        # Act - Fail and recover SDRs
        for sdr in mock_sdr_pool:
            await sdr_manager.mark_failed(sdr.url, "Outage")

        await asyncio.sleep(0.5)

        for sdr in mock_sdr_pool:
            await sdr_manager.mark_recovered(sdr.url)

        # Assert - All jobs still tracked
        for job_id in all_job_ids:
            job_data = await queue_manager.client.hgetall(f"job:{job_id}")
            assert job_data is not None
            assert "status" in job_data
            assert job_data["status"] in ["queued", "processing", "completed", "failed"]