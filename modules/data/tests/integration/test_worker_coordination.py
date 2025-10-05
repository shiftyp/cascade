"""Integration tests for distributed worker coordination.

Implements T017f: Test distributed worker coordination (FR-040, FR-043).
These tests MUST fail initially (TDD approach).
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

# These imports will fail until implementation exists (TDD)
from src.collectors.worker import CollectionWorker
from src.collectors.queue_manager import QueueManager
from src.collectors.sdr_manager import SDRManager


class TestWorkerCoordination:
    """Test distributed worker coordination."""

    @pytest.fixture
    async def lock_manager(self):
        """Create lock manager."""
        manager = LockManager()
        await manager.connect()
        yield manager
        await manager.disconnect()

    @pytest.fixture
    async def queue_manager(self):
        """Create queue manager."""
        manager = QueueManager()
        await manager.connect()
        # Clean test queues
        await manager.client.flushdb()
        yield manager
        await manager.disconnect()

    @pytest.fixture
    async def sdr_manager(self):
        """Create SDR manager."""
        manager = SDRManager()
        await manager.connect()
        yield manager
        await manager.disconnect()

    @pytest.fixture
    def mock_sdrs(self):
        """Create mock SDR pool."""
        sdrs = []
        for i in range(5):
            sdr = Mock()
            sdr.url = f"sdr{i}.example.com"
            sdr.available = True
            sdrs.append(sdr)
        return sdrs

    @pytest.mark.asyncio
    async def test_multiple_workers_process_different_jobs(
        self,
        queue_manager,
        sdr_manager,
        mock_sdrs,
    ):
        """Test that multiple workers process different jobs."""
        # Arrange
        for sdr in mock_sdrs:
            await sdr_manager.add_sdr(sdr)

        # Create workers
        workers = [CollectionWorker(worker_id=f"worker-{i}") for i in range(3)]

        # Queue 15 jobs
        for i in range(15):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {
                    "sdr_url": mock_sdrs[i % 5].url,
                    "frequency_khz": 14074 + i,
                    "duration_seconds": 60,
                },
            )

        # Act - Start workers
        worker_tasks = [
            asyncio.create_task(worker.start())
            for worker in workers
        ]

        # Let workers process jobs
        await asyncio.sleep(3)

        # Stop workers
        for worker in workers:
            await worker.stop()

        for task in worker_tasks:
            task.cancel()

        # Assert - Jobs distributed across workers
        total_completed = sum(w.jobs_completed for w in workers)
        assert total_completed > 0

        # All workers should have done some work
        completed_counts = [w.jobs_completed for w in workers]
        assert all(count > 0 for count in completed_counts)

    @pytest.mark.asyncio
    async def test_sdr_claim_locking_prevents_conflicts(
        self,
        lock_manager,
        queue_manager,
        mock_sdrs,
    ):
        """Test SDR claim locking prevents concurrent access."""
        # Arrange
        sdr_url = mock_sdrs[0].url

        # Act - Two workers try to claim same SDR
        lock1 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=60)
        lock2 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=60)

        # Assert - Only one succeeds
        assert lock1 is not None
        assert lock2 is None

        # First worker can release lock
        released = await lock_manager.release_lock(f"sdr:{sdr_url}", lock1)
        assert released is True

        # Now second worker can acquire
        lock3 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=60)
        assert lock3 is not None

        # Clean up
        await lock_manager.release_lock(f"sdr:{sdr_url}", lock3)

    @pytest.mark.asyncio
    async def test_lock_expiration_releases_stale_claims(
        self,
        lock_manager,
        mock_sdrs,
    ):
        """Test that lock expiration releases stale claims."""
        # Arrange
        sdr_url = mock_sdrs[0].url

        # Acquire lock with short duration
        lock1 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=2)
        assert lock1 is not None

        # Try to acquire again immediately - should fail
        lock2 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=10)
        assert lock2 is None

        # Wait for lock to expire
        await asyncio.sleep(2.5)

        # Act - Try to acquire again after expiration
        lock3 = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=10)

        # Assert - Lock acquired after expiration
        assert lock3 is not None

        # Clean up
        await lock_manager.release_lock(f"sdr:{sdr_url}", lock3)

    @pytest.mark.asyncio
    async def test_worker_health_monitoring(
        self,
        queue_manager,
    ):
        """Test worker health reporting and monitoring."""
        # Arrange
        worker = CollectionWorker(worker_id="test-worker")

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for health reports
        await asyncio.sleep(2)

        # Act - Check worker health in Redis
        health_key = f"worker:{worker.worker_id}:health"
        health_data_str = await queue_manager.client.get(health_key)

        # Stop worker
        await worker.stop()
        worker_task.cancel()

        # Assert - Health data reported
        assert health_data_str is not None

        import json
        health_data = json.loads(health_data_str)

        assert health_data["worker_id"] == "test-worker"
        assert health_data["status"] == "healthy"
        assert "jobs_completed" in health_data
        assert "jobs_failed" in health_data
        assert "uptime_seconds" in health_data
        assert "timestamp" in health_data

    @pytest.mark.asyncio
    async def test_worker_failure_detection(
        self,
        queue_manager,
    ):
        """Test detection of worker failures."""
        # Arrange
        worker = CollectionWorker(worker_id="failing-worker")

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for initial health report
        await asyncio.sleep(1)

        # Act - Simulate worker crash (cancel task)
        worker_task.cancel()

        # Wait for health to expire (60 seconds TTL)
        # For testing, we'll check immediately and after delay
        health_key = f"worker:{worker.worker_id}:health"

        # Check TTL
        ttl = await queue_manager.client.ttl(health_key)
        assert ttl > 0

        # Wait for expiration (we'll wait a bit less for test speed)
        # In production, monitoring would check for stale health reports

        # Assert - Worker can be detected as unhealthy based on stale data
        health_data_str = await queue_manager.client.get(health_key)
        if health_data_str:
            import json
            health_data = json.loads(health_data_str)
            timestamp = datetime.fromisoformat(health_data["timestamp"])

            # If timestamp is old, worker is unhealthy
            age = (datetime.utcnow() - timestamp).total_seconds()
            # Should be recent since we just stopped
            assert age < 5

    @pytest.mark.asyncio
    async def test_load_balancing_across_workers(
        self,
        queue_manager,
        sdr_manager,
        mock_sdrs,
    ):
        """Test load balancing across multiple workers."""
        # Arrange
        for sdr in mock_sdrs:
            await sdr_manager.add_sdr(sdr)

        workers = [CollectionWorker(worker_id=f"worker-{i}") for i in range(3)]

        # Queue 30 jobs
        for i in range(30):
            await queue_manager.push_job(
                queue_manager.COLLECTION_QUEUE,
                {
                    "sdr_url": mock_sdrs[i % 5].url,
                    "frequency_khz": 14074,
                    "duration_seconds": 30,
                },
            )

        # Act - Start all workers
        worker_tasks = [
            asyncio.create_task(worker.start())
            for worker in workers
        ]

        # Let them process
        await asyncio.sleep(4)

        # Stop workers
        for worker in workers:
            await worker.stop()

        for task in worker_tasks:
            task.cancel()

        # Assert - Load reasonably balanced
        completed_counts = [w.jobs_completed for w in workers]
        total = sum(completed_counts)

        # Each worker should have processed some jobs
        assert all(count > 0 for count in completed_counts)

        # No worker should have processed all jobs (reasonable balance)
        assert all(count < total for count in completed_counts)

        # Standard deviation should be reasonable (within ~30% of mean)
        mean = total / len(workers)
        variance = sum((count - mean) ** 2 for count in completed_counts) / len(workers)
        std_dev = variance ** 0.5
        assert std_dev < mean * 0.5  # Allow up to 50% deviation

    @pytest.mark.asyncio
    async def test_worker_respects_sdr_locks(
        self,
        queue_manager,
        lock_manager,
        sdr_manager,
        mock_sdrs,
    ):
        """Test that workers respect SDR locks."""
        # Arrange
        for sdr in mock_sdrs:
            await sdr_manager.add_sdr(sdr)

        sdr_url = mock_sdrs[0].url

        # Lock SDR externally
        external_lock = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=60)
        assert external_lock is not None

        # Queue job for locked SDR
        await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {
                "sdr_url": sdr_url,
                "frequency_khz": 14074,
                "duration_seconds": 60,
            },
        )

        # Act - Start worker
        worker = CollectionWorker(worker_id="test-worker")
        worker_task = asyncio.create_task(worker.start())

        # Let worker try to process
        await asyncio.sleep(2)

        # Stop worker
        await worker.stop()
        worker_task.cancel()

        # Assert - Job not processed (SDR was locked)
        assert worker.jobs_completed == 0

        # Job should be requeued
        stats = await queue_manager.get_queue_stats()
        assert stats[queue_manager.COLLECTION_QUEUE] >= 1

        # Clean up
        await lock_manager.release_lock(f"sdr:{sdr_url}", external_lock)

    @pytest.mark.asyncio
    async def test_worker_coordination_with_priority_jobs(
        self,
        queue_manager,
        sdr_manager,
        mock_sdrs,
    ):
        """Test worker coordination with priority jobs."""
        # Arrange
        for sdr in mock_sdrs:
            await sdr_manager.add_sdr(sdr)

        # Queue mix of priority and regular jobs
        await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {"job_type": "regular1"},
            priority=0,
        )

        await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {"job_type": "high_priority"},
            priority=10,
        )

        await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {"job_type": "regular2"},
            priority=0,
        )

        # Act - Start worker
        worker = CollectionWorker(worker_id="priority-worker")

        # Process first job
        first_job = await queue_manager.pop_job(queue_manager.COLLECTION_QUEUE)

        # Assert - High priority job processed first
        assert first_job["job_type"] == "high_priority"

    @pytest.mark.asyncio
    async def test_worker_graceful_shutdown_with_active_job(
        self,
        queue_manager,
        sdr_manager,
        mock_sdrs,
    ):
        """Test worker graceful shutdown with active job."""
        # Arrange
        for sdr in mock_sdrs:
            await sdr_manager.add_sdr(sdr)

        # Queue long-running job
        await queue_manager.push_job(
            queue_manager.COLLECTION_QUEUE,
            {
                "sdr_url": mock_sdrs[0].url,
                "frequency_khz": 14074,
                "duration_seconds": 300,  # 5 minutes
            },
        )

        # Start worker
        worker = CollectionWorker(worker_id="shutdown-worker")
        worker_task = asyncio.create_task(worker.start())

        # Wait for job to start
        await asyncio.sleep(1)

        # Act - Graceful shutdown
        await worker.stop()
        worker_task.cancel()

        # Assert - Worker tracked current job
        assert worker.current_job is None  # Cleared after stop

    @pytest.mark.asyncio
    async def test_distributed_lock_ownership_tracking(
        self,
        lock_manager,
        mock_sdrs,
    ):
        """Test distributed lock ownership tracking."""
        # Arrange
        sdr_urls = [sdr.url for sdr in mock_sdrs[:3]]

        # Acquire locks
        locks = []
        for url in sdr_urls:
            lock = await lock_manager.acquire_sdr_lock(url, duration_seconds=60)
            locks.append((url, lock))

        # Act - List owned locks
        owned_locks = await lock_manager.list_locks(owner_only=True)

        # Assert - All locks tracked
        assert len(owned_locks) == 3

        owned_resources = [lock.resource for lock in owned_locks]
        for url in sdr_urls:
            assert f"sdr:{url}" in owned_resources

        # Clean up
        for url, lock in locks:
            await lock_manager.release_lock(f"sdr:{url}", lock)

    @pytest.mark.asyncio
    async def test_lock_refresh_mechanism(
        self,
        lock_manager,
        mock_sdrs,
    ):
        """Test automatic lock refresh mechanism."""
        # Arrange
        sdr_url = mock_sdrs[0].url

        # Acquire lock with short duration
        lock = await lock_manager.acquire_sdr_lock(sdr_url, duration_seconds=10)
        assert lock is not None

        # Get initial lock info
        initial_info = await lock_manager.get_lock_info(f"sdr:{sdr_url}")
        assert initial_info is not None

        initial_expires_at = initial_info.expires_at

        # Wait a bit
        await asyncio.sleep(2)

        # Act - Extend lock
        extended = await lock_manager.extend_lock(f"sdr:{sdr_url}", lock, 30)
        assert extended is True

        # Get updated lock info
        updated_info = await lock_manager.get_lock_info(f"sdr:{sdr_url}")

        # Assert - Expiration extended
        assert updated_info.expires_at > initial_expires_at

        # Clean up
        await lock_manager.release_lock(f"sdr:{sdr_url}", lock)

    @pytest.mark.asyncio
    async def test_worker_statistics_aggregation(
        self,
        queue_manager,
    ):
        """Test worker statistics aggregation."""
        # Arrange
        workers = [CollectionWorker(worker_id=f"stats-worker-{i}") for i in range(3)]

        # Simulate some work
        workers[0].jobs_completed = 10
        workers[0].jobs_failed = 1

        workers[1].jobs_completed = 15
        workers[1].jobs_failed = 2

        workers[2].jobs_completed = 8
        workers[2].jobs_failed = 0

        # Report health for each worker
        import json
        for worker in workers:
            health_data = {
                "worker_id": worker.worker_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "jobs_completed": worker.jobs_completed,
                "jobs_failed": worker.jobs_failed,
            }

            await queue_manager.client.set(
                f"worker:{worker.worker_id}:health",
                json.dumps(health_data),
                ex=60,
            )

        # Act - Aggregate statistics
        worker_keys = await queue_manager.client.keys("worker:*:health")
        total_completed = 0
        total_failed = 0

        for key in worker_keys:
            data_str = await queue_manager.client.get(key)
            if data_str:
                data = json.loads(data_str)
                total_completed += data.get("jobs_completed", 0)
                total_failed += data.get("jobs_failed", 0)

        # Assert - Correct aggregation
        assert total_completed == 33
        assert total_failed == 3

    @pytest.mark.asyncio
    async def test_lock_contention_handling(
        self,
        lock_manager,
        mock_sdrs,
    ):
        """Test handling of lock contention."""
        # Arrange
        sdr_url = mock_sdrs[0].url

        # Create multiple lock managers (simulating workers)
        managers = [lock_manager]
        for _ in range(2):
            manager = LockManager()
            await manager.connect()
            managers.append(manager)

        # Act - All try to acquire same lock simultaneously
        results = await asyncio.gather(*[
            manager.acquire_sdr_lock(sdr_url, duration_seconds=60)
            for manager in managers
        ])

        # Assert - Only one succeeds
        successful_locks = [r for r in results if r is not None]
        assert len(successful_locks) == 1

        # Find which manager got the lock
        for i, result in enumerate(results):
            if result:
                await managers[i].release_lock(f"sdr:{sdr_url}", result)

        # Clean up
        for manager in managers[1:]:
            await manager.disconnect()