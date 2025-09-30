"""Integration tests for Redis/KeyDB queue operations.

Implements T017e: Test Redis queue operations (FR-039).
These tests MUST fail initially (TDD approach).
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

# These imports will fail until implementation exists (TDD)
from modules.data.src.collectors.queue_manager import QueueManager


class TestRedisQueue:
    """Test Redis/KeyDB queue operations."""

    @pytest.fixture
    async def queue_manager(self):
        """Create queue manager with test Redis instance."""
        manager = QueueManager()
        await manager.connect()

        # Clean up test queues
        await manager.client.flushdb()

        yield manager
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_basic_queue_push_pop(self, queue_manager):
        """Test basic queue push and pop operations."""
        # Arrange
        job_data = {
            "sdr_url": "test.sdr.com",
            "frequency_khz": 14074,
            "duration_seconds": 300,
        }

        # Act - Push job
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            job_data,
        )

        # Pop job
        retrieved_job = await queue_manager.pop_job(
            queue_manager.COLLECTION_QUEUE,
            timeout=1,
        )

        # Assert
        assert job_id is not None
        assert retrieved_job is not None
        assert retrieved_job["sdr_url"] == job_data["sdr_url"]
        assert retrieved_job["frequency_khz"] == job_data["frequency_khz"]
        assert retrieved_job["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_fifo_queue_ordering(self, queue_manager):
        """Test FIFO ordering for regular queue."""
        # Arrange - Push multiple jobs
        job_ids = []
        for i in range(5):
            job_id = await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {"task_id": i, "priority": 0},
            )
            job_ids.append(job_id)

        # Act - Pop all jobs
        retrieved_ids = []
        for _ in range(5):
            job = await queue_manager.pop_job(
                queue_manager.COLLECTION_QUEUE,
                timeout=1,
            )
            retrieved_ids.append(job["job_id"])

        # Assert - FIFO order preserved
        assert retrieved_ids == job_ids

    @pytest.mark.asyncio
    async def test_priority_queue_handling(self, queue_manager):
        """Test priority queue with high-priority jobs."""
        # Arrange - Push jobs with different priorities
        jobs = [
            {"name": "low1", "priority": 0},
            {"name": "high1", "priority": 10},
            {"name": "low2", "priority": 0},
            {"name": "high2", "priority": 20},
            {"name": "medium", "priority": 5},
        ]

        for job in jobs:
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                job,
                priority=job["priority"],
            )

        # Act - Pop all jobs
        retrieved_jobs = []
        for _ in range(5):
            job = await queue_manager.pop_job(
                queue_manager.COLLECTION_QUEUE,
                timeout=1,
            )
            retrieved_jobs.append(job)

        # Assert - High priority jobs come first
        assert retrieved_jobs[0]["name"] == "high2"  # Priority 20
        assert retrieved_jobs[1]["name"] == "high1"  # Priority 10
        assert retrieved_jobs[2]["name"] == "medium"  # Priority 5

        # Low priority jobs come last (in FIFO order)
        low_priority_names = [j["name"] for j in retrieved_jobs[3:]]
        assert "low1" in low_priority_names
        assert "low2" in low_priority_names

    @pytest.mark.asyncio
    async def test_blocking_pop_with_timeout(self, queue_manager):
        """Test blocking pop with timeout."""
        # Arrange - Empty queue
        await queue_manager.client.delete(queue_manager.COLLECTION_QUEUE)

        # Act - Pop with timeout
        start_time = datetime.utcnow()
        job = await queue_manager.pop_job(
            queue_manager.COLLECTION_QUEUE,
            timeout=2,
        )
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Assert - Returns None after timeout
        assert job is None
        assert 1.8 <= elapsed <= 2.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_non_blocking_pop_on_empty_queue(self, queue_manager):
        """Test non-blocking pop returns immediately on empty queue."""
        # Arrange - Empty queue
        await queue_manager.client.delete(queue_manager.COLLECTION_QUEUE)

        # Act - Pop without timeout
        start_time = datetime.utcnow()
        job = await queue_manager.pop_job(
            queue_manager.COLLECTION_QUEUE,
            timeout=0,
        )
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Assert - Returns immediately
        assert job is None
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_job_status_tracking(self, queue_manager):
        """Test job status tracking throughout lifecycle."""
        # Arrange
        job_data = {"task": "test_recording"}

        # Act - Push job
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            job_data,
        )

        # Check initial status
        initial_status = await queue_manager.client.hget(f"job:{job_id}", "status")
        assert initial_status == "queued"

        # Pop job (status should change to processing)
        job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
        processing_status = await queue_manager.client.hget(f"job:{job_id}", "status")
        assert processing_status == "processing"

        # Complete job
        await queue_manager.complete_job(job_id, {"result": "success"})
        final_status = await queue_manager.client.hget(f"job:{job_id}", "status")
        assert final_status == "completed"

    @pytest.mark.asyncio
    async def test_job_completion_with_result(self, queue_manager):
        """Test job completion with result data."""
        # Arrange
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {"task": "recording"},
        )

        await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)

        # Act - Complete with result
        result = {
            "session_id": "test-session-123",
            "duration": 300,
            "file_size": 1024000,
        }

        await queue_manager.complete_job(job_id, result)

        # Assert - Result stored
        stored_result = await queue_manager.client.hget(f"job:{job_id}", "result")
        assert stored_result is not None

        result_data = json.loads(stored_result)
        assert result_data["session_id"] == "test-session-123"
        assert result_data["duration"] == 300

        # Check results queue
        results_job = await queue_manager.client.rpop(queue_manager.RESULTS_QUEUE)
        assert results_job is not None

        results_data = json.loads(results_job)
        assert results_data["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_job_failure_with_retry(self, queue_manager):
        """Test job failure handling with automatic retry."""
        # Arrange
        job_data = {"task": "test", "retryable": True}
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            job_data,
        )

        await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)

        # Act - Fail job (should retry)
        await queue_manager.fail_job(job_id, "Connection timeout", retry=True)

        # Assert - Job requeued
        retry_status = await queue_manager.client.hget(f"job:{job_id}", "status")
        assert retry_status == "retry"

        retry_count = await queue_manager.client.hget(f"job:{job_id}", "retry_count")
        assert int(retry_count) == 1

        # Job should be back in queue
        stats = await queue_manager.get_queue_stats()
        assert stats[queue_manager.COLLECTION_QUEUE] == 1

    @pytest.mark.asyncio
    async def test_job_failure_moves_to_dead_letter(self, queue_manager):
        """Test that failed jobs move to dead letter queue after max retries."""
        # Arrange
        job_data = {"task": "failing_task"}
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            job_data,
        )

        # Act - Fail job 3 times
        for attempt in range(4):
            job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE, timeout=1)
            if job:
                await queue_manager.fail_job(job_id, f"Failure attempt {attempt}", retry=True)

        # Assert - Job in dead letter queue
        final_status = await queue_manager.client.hget(f"job:{job_id}", "status")
        assert final_status == "failed"

        # Check dead letter queue
        stats = await queue_manager.get_queue_stats()
        assert stats[queue_manager.DEAD_LETTER_QUEUE] >= 1

    @pytest.mark.asyncio
    async def test_distributed_work_assignment(self, queue_manager):
        """Test distributed work assignment across multiple workers."""
        # Arrange - Create multiple queue managers (simulating workers)
        worker_managers = [queue_manager]

        for _ in range(2):
            manager = QueueManager()
            await manager.connect()
            worker_managers.append(manager)

        # Push 10 jobs
        for i in range(10):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {"task_id": i},
            )

        # Act - Each worker pops jobs concurrently
        async def worker_pop(manager, worker_id):
            jobs = []
            for _ in range(4):  # Try to get 4 jobs each
                job = await manager.pop_job(
                    manager.COLLECTION_QUEUE,
                    timeout=1,
                )
                if job:
                    jobs.append(job)
            return worker_id, jobs

        results = await asyncio.gather(*[
            worker_pop(manager, i)
            for i, manager in enumerate(worker_managers)
        ])

        # Assert - Work distributed
        total_jobs = sum(len(jobs) for _, jobs in results)
        assert total_jobs == 10

        # Each worker should get some jobs
        for worker_id, jobs in results:
            assert len(jobs) > 0, f"Worker {worker_id} got no jobs"

        # Clean up
        for manager in worker_managers[1:]:
            await manager.disconnect()

    @pytest.mark.asyncio
    async def test_queue_persistence_across_reconnections(self, queue_manager):
        """Test that queue persists across reconnections."""
        # Arrange - Push jobs
        job_ids = []
        for i in range(5):
            job_id = await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {"task_id": i},
            )
            job_ids.append(job_id)

        # Act - Disconnect and reconnect
        await queue_manager.disconnect()
        await queue_manager.connect()

        # Assert - Jobs still in queue
        stats = await queue_manager.get_queue_stats()
        assert stats[queue_manager.COLLECTION_QUEUE] == 5

        # Can still pop jobs
        job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
        assert job is not None
        assert job["job_id"] in job_ids

    @pytest.mark.asyncio
    async def test_job_expiration(self, queue_manager):
        """Test that job tracking expires after configured time."""
        # Arrange
        job_id = await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {"task": "test"},
        )

        # Act - Check TTL
        ttl = await queue_manager.client.ttl(f"job:{job_id}")

        # Assert - TTL set (should be 7 days = 604800 seconds)
        assert ttl > 0
        assert ttl <= 604800

    @pytest.mark.asyncio
    async def test_queue_statistics(self, queue_manager):
        """Test queue statistics reporting."""
        # Arrange - Push jobs to different queues
        await queue_manager.push_job(queue_manager.COLLECTION_QUEUE, {"task": 1})
        await queue_manager.push_job(queue_manager.COLLECTION_QUEUE, {"task": 2})
        await queue_manager.push_job(queue_manager.PROCESSING_QUEUE, {"task": 3})

        # Complete one job and push to results
        job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
        await queue_manager.complete_job(job["job_id"], {"result": "ok"})

        # Fail one job to dead letter
        job2 = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
        for _ in range(4):
            await queue_manager.fail_job(job2["job_id"], "Error", retry=True)
            job2 = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE, timeout=0)
            if not job2:
                break

        # Act - Get stats
        stats = await queue_manager.get_queue_stats()

        # Assert
        assert queue_manager.COLLECTION_QUEUE in stats
        assert queue_manager.PROCESSING_QUEUE in stats
        assert queue_manager.RESULTS_QUEUE in stats
        assert queue_manager.DEAD_LETTER_QUEUE in stats

        assert stats[queue_manager.PROCESSING_QUEUE] == 1
        assert stats[queue_manager.RESULTS_QUEUE] >= 1

    @pytest.mark.asyncio
    async def test_pubsub_event_publishing(self, queue_manager):
        """Test pub/sub event publishing."""
        # Arrange
        received_events = []

        async def event_callback(channel, event):
            received_events.append((channel, event))

        # Subscribe to channel
        subscribe_task = asyncio.create_task(
            queue_manager.subscribe_events(
                ["collection:events"],
                event_callback,
            )
        )

        # Wait for subscription to be ready
        await asyncio.sleep(0.2)

        # Act - Publish events
        await queue_manager.publish_event(
            "collection:events",
            "recording_started",
            {"session_id": "test-123"},
        )

        await queue_manager.publish_event(
            "collection:events",
            "recording_completed",
            {"session_id": "test-123"},
        )

        # Wait for events to be received
        await asyncio.sleep(0.2)

        # Clean up
        subscribe_task.cancel()

        # Assert - Events received
        assert len(received_events) >= 2

        event_types = [event["type"] for _, event in received_events]
        assert "recording_started" in event_types
        assert "recording_completed" in event_types

    @pytest.mark.asyncio
    async def test_concurrent_queue_operations(self, queue_manager):
        """Test concurrent queue operations don't cause corruption."""
        # Arrange
        async def producer(n_jobs):
            for i in range(n_jobs):
                await queue_manager.push_job(
                    queue_manager.COLLECTION_QUEUE,
                    {"producer_job": i},
                )
                await asyncio.sleep(0.01)

        async def consumer():
            jobs = []
            for _ in range(15):
                job = await queue_manager.pop_job(
                    queue_manager.COLLECTION_QUEUE,
                    timeout=1,
                )
                if job:
                    jobs.append(job)
                await asyncio.sleep(0.01)
            return jobs

        # Act - Run concurrent producers and consumers
        results = await asyncio.gather(
            producer(10),
            producer(10),
            consumer(),
            consumer(),
        )

        consumer_jobs = [r for r in results if isinstance(r, list)]
        all_jobs = [job for jobs in consumer_jobs for job in jobs]

        # Assert - All jobs accounted for
        assert len(all_jobs) == 20

        # No duplicate job IDs
        job_ids = [job["job_id"] for job in all_jobs]
        assert len(job_ids) == len(set(job_ids))

    @pytest.mark.asyncio
    async def test_priority_queue_with_multiple_levels(self, queue_manager):
        """Test priority queue with multiple priority levels."""
        # Arrange - Push jobs with various priorities
        priorities = [1, 5, 10, 15, 20, 3, 7, 12, 18, 25]

        for i, priority in enumerate(priorities):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {"job_num": i},
                priority=priority,
            )

        # Act - Pop all jobs
        retrieved_priorities = []
        for _ in range(len(priorities)):
            job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
            retrieved_priorities.append(job["priority"])

        # Assert - Jobs retrieved in priority order (highest first)
        assert retrieved_priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_queue_recovery_after_redis_restart(self, queue_manager):
        """Test queue behavior after Redis connection loss and recovery."""
        # Arrange - Push jobs
        for i in range(3):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {"task_id": i},
            )

        # Simulate connection loss and recovery
        await queue_manager.disconnect()

        # Wait a moment
        await asyncio.sleep(0.5)

        # Reconnect
        await queue_manager.connect()

        # Assert - Can still access queue
        stats = await queue_manager.get_queue_stats()
        assert stats[queue_manager.COLLECTION_QUEUE] == 3

        # Can still pop jobs
        job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)
        assert job is not None