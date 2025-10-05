"""Redis queue manager for distributed coordination.

Implements T027b: Redis queue manager (FR-039).
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4

import redis.asyncio as redis
from redis.exceptions import RedisError

from ..config import config

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages distributed work queue using Redis/KeyDB (FR-039)."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize queue manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or config.REDIS_URL
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.connected = False

        # Queue names
        self.COLLECTION_QUEUE = "cascade:collection:queue"
        self.PROCESSING_QUEUE = "cascade:processing:queue"
        self.RESULTS_QUEUE = "cascade:results:queue"
        self.DEAD_LETTER_QUEUE = "cascade:dead_letter:queue"

    async def connect(self):
        """Connect to Redis."""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
            )
            await self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.pubsub:
            await self.pubsub.close()
        if self.client:
            await self.client.close()
        self.connected = False
        logger.info("Disconnected from Redis")

    async def push_job(
        self,
        queue: str,
        job_data: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """Push a job to the queue.

        Args:
            queue: Queue name
            job_data: Job data dictionary
            priority: Job priority (higher = more important)

        Returns:
            Job ID
        """
        if not self.connected:
            await self.connect()

        # Generate job ID
        job_id = str(uuid4())
        job_data["job_id"] = job_id
        job_data["queued_at"] = datetime.utcnow().isoformat()
        job_data["priority"] = priority

        # Serialize job
        job_json = json.dumps(job_data)

        # Push to queue (left push for FIFO, or use sorted set for priority)
        if priority > 0:
            # Use sorted set for priority queue
            await self.client.zadd(
                f"{queue}:priority",
                {job_json: -priority},  # Negative for highest first
            )
        else:
            # Regular FIFO queue
            await self.client.lpush(queue, job_json)

        # Track job status
        await self.client.hset(
            f"job:{job_id}",
            mapping={
                "status": "queued",
                "queue": queue,
                "data": job_json,
            },
        )

        # Set expiry for job tracking (7 days)
        await self.client.expire(f"job:{job_id}", 604800)

        logger.debug(f"Pushed job {job_id} to {queue}")
        return job_id

    async def push_job_delayed(
        self,
        queue: str,
        job_data: Dict[str, Any],
        delay_seconds: int = 5,
    ) -> str:
        """Push a job to the queue with a delay (using sorted set with future timestamp).

        Args:
            queue: Queue name
            job_data: Job data dictionary
            delay_seconds: Delay before job becomes available (seconds)

        Returns:
            Job ID
        """
        if not self.connected:
            await self.connect()

        # Generate job ID if not present
        job_id = job_data.get("job_id", str(uuid4()))
        job_data["job_id"] = job_id
        job_data["queued_at"] = datetime.utcnow().isoformat()
        job_data["delayed_until"] = (datetime.utcnow().timestamp() + delay_seconds)

        # Serialize job
        job_json = json.dumps(job_data)

        # Add to delayed queue (sorted set with timestamp as score)
        delayed_queue = f"{queue}:delayed"
        await self.client.zadd(
            delayed_queue,
            {job_json: datetime.utcnow().timestamp() + delay_seconds}
        )

        # Track job status
        await self.client.hset(
            f"job:{job_id}",
            mapping={
                "status": "delayed",
                "queue": queue,
                "data": job_json,
            },
        )

        # Set expiry for job tracking (7 days)
        await self.client.expire(f"job:{job_id}", 604800)

        logger.debug(f"Pushed job {job_id} to {queue} with {delay_seconds}s delay")
        return job_id

    async def pop_job(
        self,
        queue: str,
        timeout: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Pop a job from the queue.

        Args:
            queue: Queue name
            timeout: Blocking timeout in seconds (0 = non-blocking)

        Returns:
            Job data or None
        """
        if not self.connected:
            await self.connect()

        # Check delayed queue first - move ready jobs to priority queue
        delayed_queue = f"{queue}:delayed"
        current_time = datetime.utcnow().timestamp()

        # Get all jobs that are ready (score <= current_time)
        ready_jobs = await self.client.zrangebyscore(
            delayed_queue,
            min=0,
            max=current_time,
            start=0,
            num=10,  # Process up to 10 at a time
        )

        # Move ready jobs from delayed to priority queue
        if ready_jobs:
            for job_json in ready_jobs:
                # Remove from delayed queue
                await self.client.zrem(delayed_queue, job_json)
                # Add to priority queue with high priority
                await self.client.zadd(
                    f"{queue}:priority",
                    {job_json: -100}  # High priority for delayed jobs
                )
                logger.debug(f"Moved delayed job to priority queue")

        # Check priority queue first
        priority_queue = f"{queue}:priority"
        job_json = await self.client.zpopmax(priority_queue)

        if job_json:
            # Got priority job
            job_data = json.loads(job_json[0][0])
        else:
            # Try regular queue
            if timeout > 0:
                result = await self.client.brpop(queue, timeout=timeout)
                if result:
                    job_json = result[1]
                else:
                    return None
            else:
                job_json = await self.client.rpop(queue)
                if not job_json:
                    return None

            job_data = json.loads(job_json)

        # Update job status
        job_id = job_data.get("job_id")
        if job_id:
            await self.client.hset(
                f"job:{job_id}",
                mapping={
                    "status": "processing",
                    "started_at": datetime.utcnow().isoformat(),
                },
            )

        return job_data

    async def complete_job(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Mark a job as completed.

        Args:
            job_id: Job ID
            result: Job result data
        """
        await self.client.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": json.dumps(result) if result else "",
            },
        )

        # Push result to results queue if provided
        if result:
            result["job_id"] = job_id
            await self.client.lpush(self.RESULTS_QUEUE, json.dumps(result))

        logger.debug(f"Completed job {job_id}")

    async def fail_job(
        self,
        job_id: str,
        error: str,
        retry: bool = True,
    ):
        """Mark a job as failed.

        Args:
            job_id: Job ID
            error: Error message
            retry: Whether to retry the job
        """
        job_info = await self.client.hgetall(f"job:{job_id}")

        if job_info:
            retry_count = int(job_info.get("retry_count", 0))

            if retry and retry_count < 3:
                # Retry the job
                await self.client.hset(
                    f"job:{job_id}",
                    mapping={
                        "status": "retry",
                        "retry_count": retry_count + 1,
                        "last_error": error,
                    },
                )

                # Re-queue the job
                job_data = json.loads(job_info["data"])
                queue = job_info["queue"]
                await self.push_job(queue, job_data)

                logger.info(f"Retrying job {job_id} (attempt {retry_count + 1})")
            else:
                # Move to dead letter queue
                await self.client.hset(
                    f"job:{job_id}",
                    mapping={
                        "status": "failed",
                        "failed_at": datetime.utcnow().isoformat(),
                        "error": error,
                    },
                )

                # Push to dead letter queue
                await self.client.lpush(
                    self.DEAD_LETTER_QUEUE,
                    job_info.get("data", "{}"),
                )

                logger.error(f"Job {job_id} failed: {error}")

    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics.

        Returns:
            Dictionary of queue lengths
        """
        stats = {}

        queues = [
            self.COLLECTION_QUEUE,
            self.PROCESSING_QUEUE,
            self.RESULTS_QUEUE,
            self.DEAD_LETTER_QUEUE,
        ]

        for queue in queues:
            # Regular queue length
            length = await self.client.llen(queue)

            # Priority queue length
            priority_length = await self.client.zcard(f"{queue}:priority")

            stats[queue] = length + priority_length

        return stats

    async def publish_event(
        self,
        channel: str,
        event_type: str,
        data: Dict[str, Any],
    ):
        """Publish an event to a channel.

        Args:
            channel: Channel name
            event_type: Event type
            data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        await self.client.publish(channel, json.dumps(event))
        logger.debug(f"Published {event_type} to {channel}")

    async def subscribe_events(
        self,
        channels: List[str],
        callback,
    ):
        """Subscribe to event channels.

        Args:
            channels: List of channel names
            callback: Async callback function(channel, event)
        """
        if not self.pubsub:
            self.pubsub = self.client.pubsub()

        await self.pubsub.subscribe(*channels)

        logger.info(f"Subscribed to channels: {channels}")

        # Start listening
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    event = json.loads(message["data"])
                    await callback(message["channel"], event)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")

    async def acquire_lock(
        self,
        resource: str,
        ttl: int = 60,
    ) -> Optional[str]:
        """Acquire a distributed lock.

        Args:
            resource: Resource identifier
            ttl: Lock TTL in seconds

        Returns:
            Lock token or None
        """
        lock_key = f"lock:{resource}"
        lock_token = str(uuid4())

        # Try to acquire lock
        acquired = await self.client.set(
            lock_key,
            lock_token,
            nx=True,  # Only set if not exists
            ex=ttl,
        )

        if acquired:
            logger.debug(f"Acquired lock for {resource}")
            return lock_token
        else:
            return None

    async def release_lock(
        self,
        resource: str,
        token: str,
    ) -> bool:
        """Release a distributed lock.

        Args:
            resource: Resource identifier
            token: Lock token

        Returns:
            True if released successfully
        """
        lock_key = f"lock:{resource}"

        # Use Lua script for atomic check and delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        result = await self.client.eval(lua_script, 1, lock_key, token)

        if result:
            logger.debug(f"Released lock for {resource}")
            return True
        else:
            logger.warning(f"Failed to release lock for {resource}")
            return False