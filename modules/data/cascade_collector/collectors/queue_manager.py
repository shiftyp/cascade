"""Redis queue manager for distributed task coordination."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import redis.asyncio as aioredis
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CollectionTask:
    """Collection task for queue."""
    task_id: str
    sdr_id: str
    band: str
    frequency: float
    duration_seconds: int
    priority: TaskPriority
    created_at: datetime
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    metadata: Dict[str, Any] = None


class QueueManager:
    """Manages task queues in Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize queue manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.queue_prefix = "cascade:queue"
        self.task_prefix = "cascade:task"
        self.worker_prefix = "cascade:worker"
        self._connected = False

    async def connect(self):
        """Connect to Redis."""
        if self._connected:
            return

        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            self._connected = True
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False
            logger.info("Disconnected from Redis")

    async def add_task(self, task: CollectionTask) -> bool:
        """Add task to queue.

        Args:
            task: Collection task

        Returns:
            True if added successfully
        """
        if not self._connected:
            await self.connect()

        try:
            # Store task data
            task_key = f"{self.task_prefix}:{task.task_id}"
            task_data = self._serialize_task(task)
            await self.redis_client.hset(task_key, mapping=task_data)

            # Add to priority queue
            queue_key = f"{self.queue_prefix}:{task.band}"
            score = task.priority.value
            await self.redis_client.zadd(queue_key, {task.task_id: score})

            # Set expiry (24 hours)
            await self.redis_client.expire(task_key, 86400)

            logger.debug(f"Added task {task.task_id} to queue")
            return True

        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return False

    async def get_task(self, worker_id: str, bands: List[str] = None) -> Optional[CollectionTask]:
        """Get next task from queue.

        Args:
            worker_id: Worker identifier
            bands: List of bands to check (None = all)

        Returns:
            Next task or None
        """
        if not self._connected:
            await self.connect()

        bands = bands or ["20m", "40m", "80m", "160m", "10m", "15m"]

        for band in bands:
            queue_key = f"{self.queue_prefix}:{band}"

            try:
                # Pop highest priority task
                result = await self.redis_client.zpopmax(queue_key)

                if result:
                    task_id = result[0][0]

                    # Get task data
                    task_key = f"{self.task_prefix}:{task_id}"
                    task_data = await self.redis_client.hgetall(task_key)

                    if task_data:
                        task = self._deserialize_task(task_data)

                        # Update assignment
                        task.assigned_at = datetime.now(timezone.utc)
                        task.worker_id = worker_id
                        task.status = TaskStatus.ASSIGNED

                        # Update in Redis
                        await self.redis_client.hset(
                            task_key,
                            mapping=self._serialize_task(task)
                        )

                        # Register worker assignment
                        worker_key = f"{self.worker_prefix}:{worker_id}"
                        await self.redis_client.hset(
                            worker_key,
                            mapping={
                                "current_task": task_id,
                                "assigned_at": task.assigned_at.isoformat()
                            }
                        )

                        logger.info(f"Assigned task {task_id} to worker {worker_id}")
                        return task

            except Exception as e:
                logger.error(f"Failed to get task for band {band}: {e}")

        return None

    async def complete_task(self, task_id: str, worker_id: str) -> bool:
        """Mark task as completed.

        Args:
            task_id: Task identifier
            worker_id: Worker identifier

        Returns:
            True if marked successfully
        """
        if not self._connected:
            await self.connect()

        try:
            task_key = f"{self.task_prefix}:{task_id}"

            # Update task status
            await self.redis_client.hset(
                task_key,
                mapping={
                    "status": TaskStatus.COMPLETED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                }
            )

            # Clear worker assignment
            worker_key = f"{self.worker_prefix}:{worker_id}"
            await self.redis_client.hdel(worker_key, "current_task")

            logger.info(f"Task {task_id} completed by worker {worker_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to complete task: {e}")
            return False

    async def fail_task(self, task_id: str, worker_id: str, error: str) -> bool:
        """Mark task as failed.

        Args:
            task_id: Task identifier
            worker_id: Worker identifier
            error: Error message

        Returns:
            True if marked successfully
        """
        if not self._connected:
            await self.connect()

        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                return False

            task = self._deserialize_task(task_data)
            task.status = TaskStatus.FAILED
            task.retry_count += 1

            # Requeue if retries available
            if task.retry_count < 3:
                task.status = TaskStatus.PENDING
                task.assigned_at = None
                task.worker_id = None

                # Re-add to queue with lower priority
                queue_key = f"{self.queue_prefix}:{task.band}"
                score = max(1, task.priority.value - task.retry_count * 2)
                await self.redis_client.zadd(queue_key, {task_id: score})

                logger.info(f"Requeued task {task_id} (retry {task.retry_count})")

            # Update task
            await self.redis_client.hset(
                task_key,
                mapping=self._serialize_task(task)
            )

            # Add error to history
            error_key = f"{task_key}:errors"
            await self.redis_client.rpush(error_key, json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "worker_id": worker_id,
                "error": error
            }))

            # Clear worker assignment
            worker_key = f"{self.worker_prefix}:{worker_id}"
            await self.redis_client.hdel(worker_key, "current_task")

            return True

        except Exception as e:
            logger.error(f"Failed to fail task: {e}")
            return False

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Statistics dictionary
        """
        if not self._connected:
            await self.connect()

        stats = {
            "bands": {},
            "total_pending": 0,
            "total_assigned": 0,
            "total_completed": 0,
            "total_failed": 0
        }

        bands = ["20m", "40m", "80m", "160m", "10m", "15m"]

        for band in bands:
            queue_key = f"{self.queue_prefix}:{band}"
            count = await self.redis_client.zcard(queue_key)
            stats["bands"][band] = count
            stats["total_pending"] += count

        # Count task statuses
        task_keys = await self.redis_client.keys(f"{self.task_prefix}:*")

        for key in task_keys:
            status = await self.redis_client.hget(key, "status")
            if status == TaskStatus.ASSIGNED.value:
                stats["total_assigned"] += 1
            elif status == TaskStatus.COMPLETED.value:
                stats["total_completed"] += 1
            elif status == TaskStatus.FAILED.value:
                stats["total_failed"] += 1

        return stats

    async def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker status.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker status or None
        """
        if not self._connected:
            await self.connect()

        worker_key = f"{self.worker_prefix}:{worker_id}"
        data = await self.redis_client.hgetall(worker_key)

        if data:
            return {
                "worker_id": worker_id,
                "current_task": data.get("current_task"),
                "assigned_at": data.get("assigned_at"),
                "last_heartbeat": data.get("last_heartbeat")
            }

        return None

    async def worker_heartbeat(self, worker_id: str):
        """Update worker heartbeat.

        Args:
            worker_id: Worker identifier
        """
        if not self._connected:
            await self.connect()

        worker_key = f"{self.worker_prefix}:{worker_id}"
        await self.redis_client.hset(
            worker_key,
            "last_heartbeat",
            datetime.now(timezone.utc).isoformat()
        )

    def _serialize_task(self, task: CollectionTask) -> Dict[str, str]:
        """Serialize task to dict.

        Args:
            task: Collection task

        Returns:
            Serialized task data
        """
        data = {
            "task_id": task.task_id,
            "sdr_id": task.sdr_id,
            "band": task.band,
            "frequency": str(task.frequency),
            "duration_seconds": str(task.duration_seconds),
            "priority": str(task.priority.value),
            "created_at": task.created_at.isoformat(),
            "status": task.status.value,
            "retry_count": str(task.retry_count)
        }

        if task.assigned_at:
            data["assigned_at"] = task.assigned_at.isoformat()
        if task.completed_at:
            data["completed_at"] = task.completed_at.isoformat()
        if task.worker_id:
            data["worker_id"] = task.worker_id
        if task.metadata:
            data["metadata"] = json.dumps(task.metadata)

        return data

    def _deserialize_task(self, data: Dict[str, str]) -> CollectionTask:
        """Deserialize task from dict.

        Args:
            data: Task data

        Returns:
            Collection task
        """
        task = CollectionTask(
            task_id=data["task_id"],
            sdr_id=data["sdr_id"],
            band=data["band"],
            frequency=float(data["frequency"]),
            duration_seconds=int(data["duration_seconds"]),
            priority=TaskPriority(int(data["priority"])),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=TaskStatus(data["status"]),
            retry_count=int(data.get("retry_count", 0))
        )

        if data.get("assigned_at"):
            task.assigned_at = datetime.fromisoformat(data["assigned_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("worker_id"):
            task.worker_id = data["worker_id"]
        if data.get("metadata"):
            task.metadata = json.loads(data["metadata"])

        return task