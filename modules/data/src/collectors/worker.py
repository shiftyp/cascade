"""Distributed worker process for collection with enhanced logging and lock management.

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
    """Worker process for distributed collection with concurrent SDR support and enhanced logging."""

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
        self.max_concurrent_sdrs = max_concurrent_sdrs or 100

        # Track concurrent jobs
        self.current_jobs: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.health_task: Optional[asyncio.Task] = None

        # Lock management for restart recovery
        self.held_locks: Dict[str, str] = {}  # lock_key -> token
        self.lock_cleanup_task: Optional[asyncio.Task] = None

        # Enhanced statistics
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.jobs_retried = 0
        self.locks_recovered = 0
        self.connection_errors = 0
        self.start_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()

        logger.info(f"🚀 Initialized worker {self.worker_id} with {max_concurrent_sdrs} concurrent SDR slots")
        logger.info(f"📊 Worker stats tracking: completed={self.jobs_completed}, failed={self.jobs_failed}")
        logger.info(f"🏗️  Worker process PID: {os.getpid()}")

    async def start(self):
        """Start the worker process with comprehensive logging."""
        logger.info(f"🔄 Starting worker {self.worker_id}")
        
        try:
            # Initialize storage
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

            # Check for orphaned locks from previous instances
            await self._recover_orphaned_locks()

            # Register signal handlers
            logger.info(f"📡 Registering signal handlers for graceful shutdown")
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            self.running = True
            logger.info(f"🟢 Worker {self.worker_id} is now RUNNING")

            # Start background tasks
            logger.info(f"🔄 Starting background tasks")
            self.health_task = asyncio.create_task(self._health_reporter())
            self.lock_cleanup_task = asyncio.create_task(self._lock_cleanup_monitor())
            logger.info(f"✅ Background tasks started")

            # Main work loop
            logger.info(f"🎯 Entering main work loop")
            await self._work_loop()
            
        except Exception as e:
            logger.error(f"❌ Fatal error during worker startup: {e}", exc_info=True)
            await self._emergency_cleanup()
            raise

    async def stop(self):
        """Stop the worker gracefully with detailed logging."""
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
        logger.info(f"   🔌 Connection errors: {self.connection_errors}")
        logger.info(f"   ⏱️  Uptime: {uptime:.1f} seconds")

        # Disconnect
        logger.info(f"🔌 Disconnecting from queue manager")
        await self.queue_manager.disconnect()
        
        logger.info(f"🗄️  Closing database connection")
        try:
            self.db.rollback()  # Ensure clean state
            self.db.close()
        except Exception as e:
            logger.error(f"❌ Error closing database: {e}")

        logger.info(f"🏁 Worker {self.worker_id} stopped gracefully")

    async def _recover_orphaned_locks(self):
        """Recover locks that may have been left by previous worker instances."""
        try:
            logger.info(f"🔍 Checking for orphaned locks from previous worker instances")
            
            # Check for locks with this worker ID that might be stale
            lock_pattern = f"sdr:*"
            
            try:
                keys = await self.queue_manager.client.keys(f"lock:{lock_pattern}")
                logger.info(f"📋 Found {len(keys)} existing SDR locks to examine")
                
                recovered_count = 0
                for key in keys:
                    try:
                        # Check lock metadata
                        lock_info = await self.queue_manager.client.get(key)
                        if lock_info and self.worker_id in str(lock_info):
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
        logger.info(f"🧹 Starting lock cleanup monitor")
        
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
                        logger.error(f"❌ Error checking lock {lock_key}: {e}")
                        stale_locks.append(lock_key)
                
                # Clean up stale locks
                for lock_key in stale_locks:
                    logger.warning(f"🧹 Removing stale lock tracking: {lock_key}")
                    self.held_locks.pop(lock_key, None)
                
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in lock cleanup monitor: {e}")
                await asyncio.sleep(60)

    async def _release_all_locks(self):
        """Release all locks held by this worker."""
        if self.held_locks:
            logger.info(f"🔓 Releasing {len(self.held_locks)} held locks")
            
            for lock_key, token in list(self.held_locks.items()):
                try:
                    await self.queue_manager.release_lock(lock_key, token)
                    logger.debug(f"✅ Released lock: {lock_key}")
                except Exception as e:
                    logger.error(f"❌ Error releasing lock {lock_key}: {e}")
                finally:
                    self.held_locks.pop(lock_key, None)
                    
            logger.info(f"🔓 All locks released")
        else:
            logger.info(f"✅ No locks to release")

    async def _emergency_cleanup(self):
        """Emergency cleanup on startup failure."""
        logger.error(f"🚨 Performing emergency cleanup for worker {self.worker_id}")
        
        try:
            await self._release_all_locks()
        except Exception as e:
            logger.error(f"❌ Error during emergency lock cleanup: {e}")
        
        try:
            if hasattr(self, 'queue_manager'):
                await self.queue_manager.disconnect()
        except Exception as e:
            logger.error(f"❌ Error during emergency queue disconnect: {e}")
        
        try:
            if hasattr(self, 'db'):
                self.db.close()
        except Exception as e:
            logger.error(f"❌ Error during emergency db close: {e}")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals with logging."""
        logger.info(f"📡 Received signal {signum} - initiating graceful shutdown")
        self.running = False

    async def _work_loop(self):
        """Main work loop with enhanced logging."""
        logger.info(f"🎯 Worker {self.worker_id} entering work loop (resource-based scaling)")

        while self.running:
            try:
                # Clean up completed tasks
                await self._cleanup_completed_tasks()

                # Check resource usage
                cpu_percent, mem_percent = await self._get_resource_usage()

                # Dynamic scaling decision
                can_accept_more = (
                    len(self.active_tasks) < self.max_concurrent_sdrs and
                    cpu_percent < 70 and
                    mem_percent < 80
                )

                if can_accept_more:
                    # Get job from queue
                    job = await self.queue_manager.pop_job(
                        self.queue_manager.COLLECTION_QUEUE,
                        timeout=1,
                    )

                    if job:
                        # Process job concurrently
                        job_id = job.get("job_id", str(uuid4())[:8])
                        logger.info(f"📥 Received job {job_id}: {job.get('sdr_url', 'unknown')} @ {job.get('frequency_khz', 'unknown')}kHz")
                        
                        task = asyncio.create_task(self._process_job(job))
                        self.active_tasks[job_id] = task
                        
                        logger.info(
                            f"🚀 Started job {job_id} ({len(self.active_tasks)} active, "
                            f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%)"
                        )
                    else:
                        # No job available
                        logger.debug(f"⌛ No jobs available, waiting...")
                        await asyncio.sleep(1)
                else:
                    # At resource capacity
                    logger.debug(
                        f"🔴 Worker at resource limits: {len(self.active_tasks)} tasks, "
                        f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%"
                    )
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ Error in work loop: {e}", exc_info=True)
                try:
                    self.db.rollback()
                except Exception:
                    pass
                await asyncio.sleep(5)

        # Wait for all active tasks to complete before exiting
        if self.active_tasks:
            logger.info(f"⏳ Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        logger.info(f"🏁 Worker {self.worker_id} exiting work loop")

    async def _process_job(self, job: Dict[str, Any]):
        """Process a collection job with enhanced logging and error handling."""
        job_id = job.get("job_id", str(uuid4())[:8])
        sdr_url = job.get("sdr_url", "unknown")
        frequency_khz = job.get("frequency_khz", "unknown")
        
        logger.info(f"⚙️  Processing job {job_id} on worker {self.worker_id}")
        logger.info(f"📡 SDR: {sdr_url}")
        logger.info(f"📻 Frequency: {frequency_khz} kHz")
        logger.info(f"⏱️  Duration: {job.get('duration_seconds', 360)} seconds")

        self.current_jobs[job_id] = job

        try:
            # Extract job parameters
            duration_seconds = job.get("duration_seconds", 360)
            band = job.get("band")

            # Attempt to acquire lock on SDR
            lock_key = f"sdr:{sdr_url}"
            logger.info(f"🔒 Attempting to acquire lock for {sdr_url}")
            
            lock_token = await self.queue_manager.acquire_lock(
                lock_key,
                ttl=duration_seconds + 60,
            )

            if not lock_token:
                # SDR is locked - use exponential backoff for requeue
                retry_count = job.get("lock_retry_count", 0)

                # Calculate delay: 5s, 15s, 30s, 60s (max)
                delay_seconds = min(5 * (3 ** retry_count), 60)

                logger.warning(
                    f"⚠️  SDR {sdr_url} is locked by another worker, "
                    f"requeuing job {job_id} with {delay_seconds}s delay (attempt {retry_count + 1})"
                )

                # Track lock retry count
                job["lock_retry_count"] = retry_count + 1

                # Use delayed requeue to prevent hot loop
                await self.queue_manager.push_job_delayed(
                    self.queue_manager.COLLECTION_QUEUE,
                    job,
                    delay_seconds=delay_seconds,
                )
                return

            # Track the lock
            self.held_locks[lock_key] = lock_token
            logger.info(f"🔓 Acquired lock for {sdr_url} with token {lock_token[:8]}...")

            try:
                # Start recording with retry logic
                max_retries = 3
                retry_count = job.get("retry_count", 0)

                logger.info(f"🎬 Starting recording (attempt {retry_count + 1}/{max_retries + 1})")

                try:
                    session_id = await self.recorder.start_recording(
                        kiwisdr_url=sdr_url,
                        frequency_khz=frequency_khz,
                        duration_seconds=duration_seconds,
                        band=band,
                    )
                    logger.info(f"✅ Recording session {session_id} started successfully")

                except (ConnectionError, asyncio.TimeoutError) as e:
                    self.connection_errors += 1
                    logger.error(f"🔌 Connection failed to {sdr_url}: {e}")
                    
                    # Connection failed - retry with exponential backoff
                    if retry_count < max_retries:
                        retry_delay = (2 ** retry_count) * 5  # 5, 10, 20 seconds
                        logger.warning(
                            f"🔄 Retrying in {retry_delay}s (attempt {retry_count + 1}/{max_retries})"
                        )

                        # Requeue with increased retry count
                        job["retry_count"] = retry_count + 1
                        self.jobs_retried += 1
                        
                        await asyncio.sleep(retry_delay)
                        await self.queue_manager.push_job(
                            self.queue_manager.COLLECTION_QUEUE,
                            job,
                        )
                        return
                    else:
                        logger.error(f"❌ Max retries exceeded for {sdr_url}")
                        raise

                # Wait for recording to complete
                logger.info(f"⏳ Monitoring recording progress for {duration_seconds}s...")
                await asyncio.sleep(duration_seconds + 5)  # Wait full duration + 5s buffer

                # Mark job as complete
                logger.info(f"✅ Job {job_id} completed, marking as successful")
                await self.queue_manager.complete_job(
                    job_id,
                    result={
                        "session_id": session_id,
                        "worker_id": self.worker_id,
                        "completed_at": datetime.utcnow().isoformat(),
                    },
                )

                self.jobs_completed += 1
                logger.info(f"🎉 Job {job_id} completed successfully (total: {self.jobs_completed})")

                # Publish completion event
                await self.queue_manager.publish_event(
                    "collection:events",
                    "recording_completed",
                    {
                        "job_id": job_id,
                        "session_id": session_id,
                        "worker_id": self.worker_id,
                        "sdr_url": sdr_url,
                        "frequency_khz": frequency_khz,
                    },
                )

            finally:
                # Release SDR lock
                try:
                    await self.queue_manager.release_lock(lock_key, lock_token)
                    self.held_locks.pop(lock_key, None)
                    logger.info(f"🔓 Released lock for {sdr_url}")
                except Exception as e:
                    logger.error(f"❌ Error releasing lock for {sdr_url}: {e}")

        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
            await self.queue_manager.fail_job(job_id, str(e))
            self.jobs_failed += 1
            
            # Rollback any database changes
            try:
                self.db.rollback()
            except Exception:
                pass

        finally:
            # Cleanup job tracking
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
                logger.debug(f"🧹 Cleaned up job tracking for {job_id}")

    async def _cleanup_completed_tasks(self):
        """Remove completed tasks from active tasks dict with logging."""
        completed_jobs = []
        for job_id, task in list(self.active_tasks.items()):
            if task.done():
                completed_jobs.append(job_id)
                try:
                    # Get result to check for exceptions
                    await task
                    logger.debug(f"✅ Task {job_id} completed successfully")
                except Exception as e:
                    logger.error(f"❌ Task {job_id} completed with error: {e}")

        for job_id in completed_jobs:
            del self.active_tasks[job_id]
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
                
        if completed_jobs:
            logger.debug(f"🧹 Cleaned up {len(completed_jobs)} completed tasks")

    async def _get_resource_usage(self) -> Tuple[float, float]:
        """Get current CPU and memory usage percentages."""
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
            estimated_cpu = len(self.active_tasks) * 7
            estimated_mem = len(self.active_tasks) * 8
            return estimated_cpu, estimated_mem
        except Exception as e:
            logger.error(f"❌ Error getting resource usage: {e}")
            # Return high values to prevent overload
            return 70, 80

    async def _health_reporter(self):
        """Report worker health to Redis with enhanced logging."""
        health_key = f"worker:{self.worker_id}:health"
        logger.info(f"💓 Starting health reporter for worker {self.worker_id}")

        while self.running:
            try:
                # Get current job IDs
                current_job_ids = list(self.current_jobs.keys())
                cpu_percent, mem_percent = await self._get_resource_usage()

                # Prepare health data
                health_data = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "healthy",
                    "jobs_completed": self.jobs_completed,
                    "jobs_failed": self.jobs_failed,
                    "jobs_retried": self.jobs_retried,
                    "connection_errors": self.connection_errors,
                    "locks_recovered": self.locks_recovered,
                    "concurrent_jobs": len(self.current_jobs),
                    "max_concurrent": self.max_concurrent_sdrs,
                    "current_job_ids": current_job_ids,
                    "capacity_percent": (len(self.current_jobs) / self.max_concurrent_sdrs) * 100,
                    "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": mem_percent,
                    "held_locks": len(self.held_locks),
                }

                # Store in Redis
                await self.queue_manager.client.set(
                    health_key,
                    json.dumps(health_data),
                    ex=60,  # Expire after 60 seconds
                )

                logger.debug(
                    f"💓 Health reported: {len(self.current_jobs)} jobs, "
                    f"CPU: {cpu_percent:.1f}%, MEM: {mem_percent:.1f}%"
                )

                # Wait 30 seconds before next report
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"❌ Failed to report health: {e}")
                await asyncio.sleep(10)

    async def process_single_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single job directly (for testing)."""
        job_data["job_id"] = job_data.get("job_id", str(uuid4()))
        logger.info(f"🧪 Processing test job: {job_data['job_id']}")
        
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
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum concurrent SDRs",
        default=None,
    )

    args = parser.parse_args()

    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )

    worker = CollectionWorker(
        worker_id=args.worker_id,
        max_concurrent_sdrs=args.max_concurrent
    )

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
            logger.info("📡 Received interrupt")
        except Exception as e:
            logger.error(f"❌ Worker failed: {e}", exc_info=True)
        finally:
            try:
                await worker.stop()
            except Exception as e:
                logger.error(f"❌ Error during worker shutdown: {e}")


if __name__ == "__main__":
    asyncio.run(main())