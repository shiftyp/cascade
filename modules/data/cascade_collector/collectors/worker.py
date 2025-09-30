"""Worker for distributed collection tasks."""

import asyncio
import logging
import signal
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import uuid

from .queue_manager import QueueManager, CollectionTask
from .recorder import Recorder, RecordingConfig
from .hybrid_sdr_selector import HybridSDRSelector

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""
    worker_id: str
    redis_url: str = "redis://localhost:6379"
    bands: List[str] = None
    max_concurrent_tasks: int = 2
    heartbeat_interval: int = 30
    output_dir: str = "/tmp/recordings"


class Worker:
    """Worker for processing collection tasks."""

    def __init__(self, config: WorkerConfig):
        """Initialize worker.

        Args:
            config: Worker configuration
        """
        self.config = config
        self.worker_id = config.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.queue_manager = QueueManager(config.redis_url)
        self.sdr_selector = HybridSDRSelector()
        self.running = False
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start worker."""
        logger.info(f"Starting worker {self.worker_id}")

        # Connect to Redis
        await self.queue_manager.connect()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self.running = True

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start processing loop
        processing_task = asyncio.create_task(self._processing_loop())

        try:
            # Wait for stop signal
            await self._stop_event.wait()
        finally:
            # Cleanup
            self.running = False
            heartbeat_task.cancel()
            processing_task.cancel()

            # Wait for current tasks to complete
            await self._graceful_shutdown()

            # Disconnect
            await self.queue_manager.disconnect()

            logger.info(f"Worker {self.worker_id} stopped. "
                       f"Completed: {self.tasks_completed}, Failed: {self.tasks_failed}")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._stop_event.set()

    async def _heartbeat_loop(self):
        """Send heartbeats to Redis."""
        while self.running:
            try:
                await self.queue_manager.worker_heartbeat(self.worker_id)
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Check if we can take more tasks
                if len(self.current_tasks) < self.config.max_concurrent_tasks:
                    # Get next task
                    task = await self.queue_manager.get_task(
                        self.worker_id,
                        self.config.bands
                    )

                    if task:
                        # Process task
                        task_future = asyncio.create_task(self._process_task(task))
                        self.current_tasks[task.task_id] = task_future

                        # Cleanup completed tasks
                        completed = []
                        for task_id, future in self.current_tasks.items():
                            if future.done():
                                completed.append(task_id)

                        for task_id in completed:
                            del self.current_tasks[task_id]
                    else:
                        # No tasks available, wait
                        await asyncio.sleep(5)
                else:
                    # At capacity, wait
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(5)

    async def _process_task(self, task: CollectionTask):
        """Process a collection task.

        Args:
            task: Collection task
        """
        logger.info(f"Processing task {task.task_id}: "
                   f"SDR={task.sdr_id}, band={task.band}, "
                   f"duration={task.duration_seconds}s")

        try:
            # Create recorder
            config = RecordingConfig(
                sample_rate=12000,
                duration_seconds=task.duration_seconds,
                frequency=task.frequency,
                output_dir=self.config.output_dir
            )
            recorder = Recorder(config)

            # Start recording
            output_path = await recorder.start_recording(
                sdr_id=task.sdr_id,
                frequency=task.frequency
            )

            # Wait for recording to complete
            while recorder.is_recording:
                await asyncio.sleep(1)

            # Get final status
            metadata = await recorder.stop_recording()

            # Validate recording
            validation = await recorder.validate_recording(output_path)

            if validation["valid"]:
                # Mark task complete
                await self.queue_manager.complete_task(task.task_id, self.worker_id)
                self.tasks_completed += 1
                logger.info(f"Task {task.task_id} completed successfully")
            else:
                # Mark task failed
                error = f"Recording validation failed: {validation.get('errors', [])}"
                await self.queue_manager.fail_task(task.task_id, self.worker_id, error)
                self.tasks_failed += 1
                logger.error(f"Task {task.task_id} failed: {error}")

            # Cleanup
            await recorder.cleanup()

        except Exception as e:
            logger.error(f"Task {task.task_id} error: {e}")
            await self.queue_manager.fail_task(task.task_id, self.worker_id, str(e))
            self.tasks_failed += 1

    async def _graceful_shutdown(self):
        """Gracefully shutdown worker."""
        logger.info("Starting graceful shutdown...")

        # Cancel new task fetching
        self.running = False

        # Wait for current tasks with timeout
        if self.current_tasks:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")

            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.current_tasks.values(), return_exceptions=True),
                    timeout=30
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling...")
                for task in self.current_tasks.values():
                    task.cancel()

    def get_status(self) -> Dict[str, Any]:
        """Get worker status.

        Returns:
            Status dictionary
        """
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "current_tasks": len(self.current_tasks),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "bands": self.config.bands or "all"
        }


async def run_worker(worker_id: Optional[str] = None,
                    redis_url: str = "redis://localhost:6379",
                    bands: Optional[List[str]] = None):
    """Run a worker.

    Args:
        worker_id: Worker ID (auto-generated if None)
        redis_url: Redis URL
        bands: List of bands to process
    """
    config = WorkerConfig(
        worker_id=worker_id,
        redis_url=redis_url,
        bands=bands
    )

    worker = Worker(config)
    await worker.start()


if __name__ == "__main__":
    # Run worker from command line
    import argparse

    parser = argparse.ArgumentParser(description="CASCADE Collection Worker")
    parser.add_argument("--id", help="Worker ID")
    parser.add_argument("--redis", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--bands", nargs="+", help="Bands to process")

    args = parser.parse_args()

    asyncio.run(run_worker(
        worker_id=args.id,
        redis_url=args.redis,
        bands=args.bands
    ))