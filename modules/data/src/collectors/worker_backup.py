"""Distributed worker process for collection.

Implements T027a: Distributed worker process.
"""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from uuid import uuid4

from ..collectors.kiwi_client import KiwiClient
from ..collectors.recorder import Recorder
from ..collectors.queue_manager import QueueManager
from ..config import config
from ..models import SessionLocal

logger = logging.getLogger(__name__)


class CollectionWorker:
    """Worker process for distributed collection with concurrent SDR support.

    Modified to handle 5-10 concurrent SDRs per worker to reduce costs.
    """

    def __init__(self, worker_id: Optional[str] = None, max_concurrent_sdrs: Optional[int] = None):
        """Initialize worker with dynamic concurrent SDR support.

        Args:
            worker_id: Unique worker identifier
            max_concurrent_sdrs: Maximum SDRs to handle (None = unlimited until resources exhausted)
        """
        self.worker_id = worker_id or str(uuid4())[:8]
        self.queue_manager = QueueManager()
        self.recorder = Recorder()
        self.db = SessionLocal()
        self.running = False

        # Dynamic scaling - consume until resource limits hit
        # If not specified, keep consuming until CPU/memory pressure
        self.max_concurrent_sdrs = max_concurrent_sdrs or 100  # Effectively unlimited

        # Track concurrent jobs
        self.current_jobs: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.health_task: Optional[asyncio.Task] = None

        # Lock management for restart recovery
        self.held_locks: Dict[str, str] = {}  # lock_key -> token
        self.lock_cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.jobs_retried = 0
        self.locks_recovered = 0
        self.start_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()

        logger.info(f"🚀 Initialized worker {self.worker_id} with {max_concurrent_sdrs} concurrent SDR slots")
        logger.info(f"📊 Worker stats tracking: completed={self.jobs_completed}, failed={self.jobs_failed}")

    async def start(self):
        """Start the worker process."""
        logger.info(f"🔄 Starting worker {self.worker_id}")
        logger.info(f"🏗️  Worker process PID: {os.getpid() if 'os' in globals() else 'unknown'}")

        try:
            # Initialize Tigris buckets
            logger.info(f"🗂️  Initializing Tigris storage buckets for worker {self.worker_id}")
            from ..storage.tigris_init import initialize_tigris_buckets
            if not initialize_tigris_buckets():
                logger.warning("⚠️  Tigris bucket initialization failed - uploads may not work")
            else:
                logger.info(f"✅ Tigris buckets initialized successfully")

            # Connect to queue
            logger.info(f"🔗 Connecting to Redis queue manager")
            await self.queue_manager.connect()
            logger.info(f"✅ Connected to queue manager successfully")

            # Check for orphaned locks from previous worker instances
            await self._recover_orphaned_locks()

            # Register signal handlers
            logger.info(f"📡 Registering signal handlers for graceful shutdown")
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            self.running = True
            logger.info(f"🟢 Worker {self.worker_id} is now RUNNING")

            # Start background tasks
            logger.info(f"🔄 Starting background health reporting task")
            self.health_task = asyncio.create_task(self._health_reporter())
            
            logger.info(f"🔄 Starting lock cleanup monitoring task")
            self.lock_cleanup_task = asyncio.create_task(self._lock_cleanup_monitor())

            # Main work loop
            logger.info(f"🎯 Entering main work loop")
            await self._work_loop()
            
        except Exception as e:
            logger.error(f"❌ Fatal error during worker startup: {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop the worker gracefully."""
        logger.info(f"🛑 Stopping worker {self.worker_id}")
        self.running = False

        # Cancel background tasks
        logger.info(f"🔄 Cancelling background tasks")
        if self.health_task:
            self.health_task.cancel()
            logger.debug(f"✅ Health reporting task cancelled")
            
        if self.lock_cleanup_task:
            self.lock_cleanup_task.cancel()
            logger.debug(f"✅ Lock cleanup task cancelled")

        # Release all held locks before shutting down
        await self._release_all_locks()

        # Wait for all current jobs to complete
        if self.active_tasks:
            logger.info(f"⏳ Waiting for {len(self.active_tasks)} jobs to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                    timeout=30
                )
                logger.info(f"✅ All jobs completed gracefully")
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout waiting for jobs, cancelling {len(self.active_tasks)} tasks")
                for job_id, task in self.active_tasks.items():
                    logger.warning(f"🚫 Cancelling job {job_id}")
                    task.cancel()

        # Final statistics
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        logger.info(f"📊 Final worker statistics:")
        logger.info(f"   ✅ Jobs completed: {self.jobs_completed}")
        logger.info(f"   ❌ Jobs failed: {self.jobs_failed}")
        logger.info(f"   🔄 Jobs retried: {self.jobs_retried}")
        logger.info(f"   🔧 Locks recovered: {self.locks_recovered}")
        logger.info(f"   ⏱️  Uptime: {uptime:.1f} seconds")

        # Disconnect
        logger.info(f"🔌 Disconnecting from queue manager")
        await self.queue_manager.disconnect()
        
        logger.info(f"🗄️  Closing database connection")
        try:
            self.db.close()
        except Exception as e:
            logger.error(f"❌ Error closing database: {e}")

    async def _recover_orphaned_locks(self):
        """Recover locks that may have been left by previous worker instances."""
        try:
            logger.info(f"🔍 Checking for orphaned locks from previous worker instances")
            
            # Check for locks with this worker ID that might be stale
            lock_pattern = f"🏁 sdr:*"
            
            # Get all SDR locks
            try:
                keys = await self.queue_manager.client.keys(f"lock:{lock_pattern}")
                logger.info(f"📋 Found {len(keys)} existing SDR locks to examine")
                
                recovered_count = 0
                for key in keys:
                    try:
                        # Check lock metadata
                        lock_info = await self.queue_manager.client.get(key)
                        if lock_info:
                            # If lock contains our worker ID but we just started, it's orphaned
                            if self.worker_id in lock_info:
                                logger.warning(f"🚨 Found orphaned lock: {key} with our worker ID")
                                await self.queue_manager.client.delete(key)
                                recovered_count += 1
                                logger.info(f"🧹 Cleaned up orphaned lock: {key}")
                    except Exception as e:
                        logger.error(f"❌ Error examining lock {key}: {e}")
                        
                if recovered_count > 0:
                    logger.warning(f"🔧 Recovered {recovered_count} orphaned locks")
                    self.locks_recovered = recovered_count
                else:
                    logger.info(f"✅ No orphaned locks found")
                    
            except Exception as e:
                logger.error(f"❌ Error retrieving lock keys: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error during lock recovery: {e}")

    async def _lock_cleanup_monitor(self):
        """Monitor and cleanup stale locks periodically."""
        while self.running:
            try:
                logger.debug(f"🔍 Monitoring {len(self.held_locks)} held locks for staleness")
                
                # Check each held lock is still valid
                stale_locks = []
                for lock_key, token in list(self.held_locks.items()):
                    try:
                        # Verify lock still exists and matches our token
                        current_token = await self.queue_manager.client.get(f"lock:{lock_key}")
                        if current_token != token:
                            logger.warning(f"🚨 Lock {lock_key} token mismatch - marking as stale")
                            stale_locks.append(lock_key)
                    except Exception as e:
                        logger.error(f"❌ Error checking lock {lock_key}: {e} stopped gracefully")

    async def _release_all_locks(self):
        """Release all locks held by this worker."""
        if self.held_locks:
            logger.info(f"🔓 Releasing {len(self.held_locks)} held locks")
            
            for lock_key, token in list(self.held_locks.items()):
                try:
                   ")
                        stale_locks.append(lock_key)
                
                # Clean up stale locks
                for lock_key in stale_locks:
                    logger.warning(f"🧹 Removing await self.queue_manager.release_lock(lock_key, tale lock tracking: {lock_key}")
                    self.held_loken)
                    logger.debug(f"✅ Released lock: {lock_key}")
                excecks.pt Exceoption as e:
                    logger.(lock_kerror(f"❌ Error releasing lock {lock_key}: {e}")
                finally:
                    self.held_locks.pop(lock_key, None)
                    
            logger.info(f"🔓 All locks releasey, None)
                
                # Upd")
        else:
            logger.info(f"✅ No locks to releaseate heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in lock cleanup monitor: {e}")
                await asyncio.sleep(60)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.running = False

    async def _work_loop(self):
        """Main work loop - process jobs until resources exhausted."""
        logger.info(f"Worker {self.worker_id} entering work loop (resource-based scaling)")

        while self.running:
            try:
                # Clean up completed tasks
                await self._cleanup_completed_tasks()

                # Check resource usage
                cpu_percent, mem_percent = await self._get_resource_usage()

                # Dynamic scaling decision based on resources
                # Keep consuming until we hit 70% CPU or 80% memory
                can_accept_more = (
                    len(self.active_tasks) < self.max_concurrent_sdrs and
                    cpu_percent < 70 and
                    mem_percent < 80
                )

                if can_accept_more:
                    # Get job from queue (non-blocking)
                    job = await self.queue_manager.pop_job(
                        self.queue_manager.COLLECTION_QUEUE,
                        timeout=1,
                    )

                    if job:
                        # Process job concurrently
                        job_id = job.get("job_id", str(uuid4())[:8])
                        task = asyncio.create_task(self._process_job(job))
                        self.active_tasks[job_id] = task
                        logger.info(
                            f"Started job {job_id} ({len(self.active_tasks)} active, "
                            f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%)"
                        )
                    else:
                        # No job available, wait a bit
                        await asyncio.sleep(1)
                else:
                    # At resource capacity, wait for resources to free up
                    await asyncio.sleep(2)
                    logger.debug(
                        f"Worker at resource limits: {len(self.active_tasks)} tasks, "
                        f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%"
                    )

            except Exception as e:
                logger.error(f"Error in work loop: {e}")
                await asyncio.sleep(5)

        # Wait for all active tasks to complete before exiting
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        logger.info(f"Worker {self.worker_id} exiting work loop")

    async def _cleanup_completed_tasks(self):
        """Remove completed tasks from active tasks dict."""
        completed_jobs = []
        for job_id, task in list(self.active_tasks.items()):
            if task.done():
                completed_jobs.append(job_id)
                try:
                    # Get result to check for exceptions
                    await task
                except Exception as e:
                    logger.error(f"Job {job_id} completed with error: {e}")

        for job_id in completed_jobs:
            del self.active_tasks[job_id]
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]

    async def _get_resource_usage(self) -> Tuple[float, float]:
        """Get current CPU and memory usage percentages.

        Returns:
            Tuple of (cpu_percent, memory_percent)
        """
        try:
            import psutil

            # Get current process
            process = psutil.Process()

            # CPU usage (percentage of total system CPU)
            cpu_percent = process.cpu_percent(interval=0.1)

            # Memory usage (percentage of available memory)
            mem_info = process.memory_info()
            mem_percent = (mem_info.rss / psutil.virtual_memory().total) * 100

            return cpu_percent, mem_percent
        except ImportError:
            # If psutil not available, return conservative estimates
            # This will limit to ~10 SDRs per worker
            estimated_cpu = len(self.active_tasks) * 7
            estimated_mem = len(self.active_tasks) * 8
            return estimated_cpu, estimated_mem
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            # Return high values to prevent overload
            return 70, 80

    async def _process_job(self, job: Dict[str, Any]):
        """Process a collection job.

        Args:
            job: Job data from queue
        """
        job_id = job.get("job_id", str(uuid4())[:8])
        logger.info(f"Processing job {job_id} on worker {self.worker_id}")

        self.current_jobs[job_id] = job

        try:
            # Extract job parameters
            sdr_url = job["sdr_url"]
            frequency_khz = job["frequency_khz"]
            duration_seconds = job.get("duration_seconds", 360)
            band = job.get("band")

            # Attempt to acquire lock on SDR
            lock_token = await self.queue_manager.acquire_lock(
                f"sdr:{sdr_url}",
                ttl=duration_seconds + 60,
            )

            if not lock_token:
                logger.warning(f"SDR {sdr_url} is locked, requeuing job")
                await self.queue_manager.push_job(
                    self.queue_manager.COLLECTION_QUEUE,
                    job,
                )
                return

            try:
                # Start recording with retry logic
                max_retries = 3
                retry_count = job.get("retry_count", 0)

                try:
                    session_id = await self.recorder.start_recording(
                        kiwisdr_url=sdr_url,
                        frequency_khz=frequency_khz,
                        duration_seconds=duration_seconds,
                        band=band,
                    )
                except (ConnectionError, asyncio.TimeoutError) as e:
                    # Connection failed - retry with exponential backoff
                    if retry_count < max_retries:
                        retry_delay = (2 ** retry_count) * 5  # 5, 10, 20 seconds
                        logger.warning(
                            f"Connection failed to {sdr_url}: {e}. "
                            f"Retrying in {retry_delay}s (attempt {retry_count + 1}/{max_retries})"
                        )

                        # Requeue with increased retry count
                        job["retry_count"] = retry_count + 1
                        await asyncio.sleep(retry_delay)
                        await self.queue_manager.push_job(
                            self.queue_manager.COLLECTION_QUEUE,
                            job,
                        )
                        return
                    else:
                        logger.error(f"Max retries exceeded for {sdr_url}")
                        raise

                # Wait for recording to complete
                # In practice, this would monitor the recording status
                await asyncio.sleep(min(duration_seconds, 10))

                # Mark job as complete
                await self.queue_manager.complete_job(
                    job_id,
                    result={
                        "session_id": session_id,
                        "worker_id": self.worker_id,
                        "completed_at": datetime.utcnow().isoformat(),
                    },
                )

                self.jobs_completed += 1
                logger.info(f"Job {job_id} completed successfully")

                # Publish completion event
                await self.queue_manager.publish_event(
                    "collection:events",
                    "recording_completed",
                    {
                        "job_id": job_id,
                        "session_id": session_id,
                        "worker_id": self.worker_id,
                    },
                )

            finally:
                # Release SDR lock
                await self.queue_manager.release_lock(f"sdr:{sdr_url}", lock_token)

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            await self.queue_manager.fail_job(job_id, str(e))
            self.jobs_failed += 1

        finally:
            # Cleanup job tracking
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]

    async def _health_reporter(self):
        """Report worker health to Redis (FR-043)."""
        health_key = f"worker:{self.worker_id}:health"

        while self.running:
            try:
                # Get current job IDs
                current_job_ids = list(self.current_jobs.keys())

                # Prepare health data
                health_data = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "healthy",
                    "jobs_completed": self.jobs_completed,
                    "jobs_failed": self.jobs_failed,
                    "concurrent_jobs": len(self.current_jobs),
                    "max_concurrent": self.max_concurrent_sdrs,
                    "current_job_ids": current_job_ids,
                    "capacity_percent": (len(self.current_jobs) / self.max_concurrent_sdrs) * 100,
                    "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                }

                # Store in Redis
                await self.queue_manager.client.set(
                    health_key,
                    json.dumps(health_data),
                    ex=60,  # Expire after 60 seconds
                )

                # Wait 30 seconds before next report
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Failed to report health: {e}")
                await asyncio.sleep(10)

    async def process_single_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single job directly (for testing).

        Args:
            job_data: Job data

        Returns:
            Result dictionary
        """
        job_data["job_id"] = job_data.get("job_id", str(uuid4()))
        await self._process_job(job_data)

        return {
            "worker_id": self.worker_id,
            "job_id": job_data["job_id"],
            "status": "completed" if self.jobs_completed > 0 else "failed",
        }


async def main():
    """Main entry point for worker."""
    import argparse

    parser = argparse.ArgumentParser(description="CASCADE Collection Worker")
    parser.add_argument(
        "--worker-id",
        help="Worker ID",
        default=None,
    )
    parser.add_argument(
        "--test-job",
        help="Test job JSON",
        default=None,
    )

    args = parser.parse_args()

    worker = CollectionWorker(worker_id=args.worker_id)

    if args.test_job:
        # Test mode - process single job
        job_data = json.loads(args.test_job)
        result = await worker.process_single_job(job_data)
        print(json.dumps(result, indent=2))
    else:
        # Normal mode - run worker
        try:
            await worker.start()
        except KeyboardInterrupt:
            logger.info("Received interrupt")
        finally:
            await worker.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())