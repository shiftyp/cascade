"""Auto-scaling manager for collection workers.

Dynamically scales workers based on queue depth and collection targets.
"""

import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis.asyncio as redis

from ..config import config

logger = logging.getLogger(__name__)


class WorkerAutoscaler:
    """Manages automatic scaling of collection workers based on demand."""

    def __init__(self):
        """Initialize autoscaler."""
        self.redis_client: Optional[redis.Redis] = None
        self.running = False

        # Scaling parameters
        self.min_workers = 2  # Minimum workers to maintain
        self.max_workers = 20  # Maximum workers (for 50-100 SDRs with 5 SDRs/worker)
        self.sdrs_per_worker = 5  # SDRs handled concurrently per worker

        # Auto-scaling thresholds
        self.scale_up_queue_depth = 10  # Scale up if > 10 jobs queued
        self.scale_down_idle_time = 300  # Scale down after 5 min idle
        self.check_interval = 30  # Check every 30 seconds

        # Track active workers
        self.active_workers: Dict[str, Dict] = {}
        self.worker_processes: Dict[str, subprocess.Popen] = {}

    async def initialize(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis_client.ping()
            logger.info("Autoscaler connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def start(self):
        """Start the autoscaler."""
        await self.initialize()
        self.running = True

        logger.info(f"Starting worker autoscaler (min={self.min_workers}, max={self.max_workers})")

        # Start minimum workers
        await self._ensure_minimum_workers()

        # Start monitoring loop
        await self._monitor_loop()

    async def stop(self):
        """Stop the autoscaler and all workers."""
        self.running = False

        # Stop all workers
        for worker_id in list(self.worker_processes.keys()):
            await self._stop_worker(worker_id)

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Autoscaler stopped")

    async def _monitor_loop(self):
        """Main monitoring loop for auto-scaling."""
        while self.running:
            try:
                # Get current metrics
                metrics = await self._get_scaling_metrics()

                # Determine scaling action
                action = await self._determine_scaling_action(metrics)

                # Execute scaling
                if action == "scale_up":
                    await self._scale_up()
                elif action == "scale_down":
                    await self._scale_down()

                # Log status
                logger.info(
                    f"Autoscaler: {len(self.active_workers)} workers, "
                    f"queue_depth={metrics['queue_depth']}, "
                    f"capacity={metrics['total_capacity_percent']:.1f}%"
                )

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in autoscaler monitor loop: {e}")
                await asyncio.sleep(60)

    async def _get_scaling_metrics(self) -> Dict:
        """Get metrics for scaling decisions."""
        metrics = {
            "queue_depth": 0,
            "active_workers": 0,
            "idle_workers": 0,
            "total_capacity_percent": 0,
            "sdr_target": 6,  # Default baseline
        }

        try:
            # Get queue depth
            queue_length = await self.redis_client.llen("queue:collection")
            metrics["queue_depth"] = queue_length

            # Get worker health
            worker_keys = await self.redis_client.keys("worker:*:health")

            total_capacity = 0
            active_count = 0
            idle_count = 0

            for key in worker_keys:
                health_data = await self.redis_client.get(key)
                if health_data:
                    import json
                    health = json.loads(health_data)

                    # Check if worker is recent
                    last_update = datetime.fromisoformat(health["timestamp"])
                    age = (datetime.utcnow() - last_update).total_seconds()

                    if age < 60:  # Worker is alive
                        active_count += 1
                        capacity = health.get("capacity_percent", 0)
                        total_capacity += capacity

                        if capacity < 20:  # Less than 20% utilized
                            idle_count += 1

            metrics["active_workers"] = active_count
            metrics["idle_workers"] = idle_count

            if active_count > 0:
                metrics["total_capacity_percent"] = total_capacity / active_count

            # Get SDR target from scheduler
            scheduler_metrics = await self.redis_client.get("scheduler:metrics")
            if scheduler_metrics:
                import json
                sched_data = json.loads(scheduler_metrics)
                metrics["sdr_target"] = sched_data.get("sdr_target", 6)

        except Exception as e:
            logger.error(f"Error getting scaling metrics: {e}")

        return metrics

    async def _determine_scaling_action(self, metrics: Dict) -> Optional[str]:
        """Determine if scaling is needed."""
        current_workers = metrics["active_workers"]

        # Calculate required workers based on SDR target
        required_workers = max(
            self.min_workers,
            min(
                self.max_workers,
                (metrics["sdr_target"] + self.sdrs_per_worker - 1) // self.sdrs_per_worker
            )
        )

        # Scale up conditions
        if current_workers < required_workers:
            return "scale_up"

        if metrics["queue_depth"] > self.scale_up_queue_depth and current_workers < self.max_workers:
            return "scale_up"

        if metrics["total_capacity_percent"] > 80 and current_workers < self.max_workers:
            return "scale_up"

        # Scale down conditions
        if current_workers > self.min_workers:
            if metrics["idle_workers"] > current_workers * 0.5:  # More than 50% idle
                return "scale_down"

            if metrics["queue_depth"] == 0 and metrics["total_capacity_percent"] < 20:
                return "scale_down"

        return None

    async def _scale_up(self):
        """Add a new worker."""
        try:
            worker_id = await self._start_worker()
            logger.info(f"Scaled up: started worker {worker_id}")
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")

    async def _scale_down(self):
        """Remove an idle worker."""
        try:
            # Find the most idle worker
            idle_worker = await self._find_idle_worker()
            if idle_worker:
                await self._stop_worker(idle_worker)
                logger.info(f"Scaled down: stopped worker {idle_worker}")
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")

    async def _ensure_minimum_workers(self):
        """Ensure minimum number of workers are running."""
        current_count = len(self.active_workers)

        while current_count < self.min_workers:
            await self._start_worker()
            current_count += 1

    async def _start_worker(self) -> str:
        """Start a new worker process."""
        from uuid import uuid4
        worker_id = str(uuid4())[:8]

        # Start worker subprocess
        cmd = [
            "python", "-m",
            "modules.data.src.collectors.worker",
            "--worker-id", worker_id,
            "--max-concurrent", str(self.sdrs_per_worker)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": "/app"}
        )

        self.worker_processes[worker_id] = process
        self.active_workers[worker_id] = {
            "started": datetime.utcnow(),
            "process": process
        }

        logger.info(f"Started worker {worker_id} (PID: {process.pid})")
        return worker_id

    async def _stop_worker(self, worker_id: str):
        """Stop a worker process."""
        if worker_id in self.worker_processes:
            process = self.worker_processes[worker_id]

            # Send termination signal
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not responsive
                process.kill()

            del self.worker_processes[worker_id]
            del self.active_workers[worker_id]

            logger.info(f"Stopped worker {worker_id}")

    async def _find_idle_worker(self) -> Optional[str]:
        """Find the most idle worker to stop."""
        min_capacity = 100
        idle_worker = None

        for worker_id in self.active_workers:
            health_key = f"worker:{worker_id}:health"
            health_data = await self.redis_client.get(health_key)

            if health_data:
                import json
                health = json.loads(health_data)
                capacity = health.get("capacity_percent", 0)

                if capacity < min_capacity:
                    min_capacity = capacity
                    idle_worker = worker_id

        # Don't stop workers that still have some load
        if min_capacity > 20:
            return None

        return idle_worker


async def main():
    """Run the autoscaler."""
    import os
    autoscaler = WorkerAutoscaler()

    try:
        await autoscaler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down autoscaler...")
        await autoscaler.stop()


if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())