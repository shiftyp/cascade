"""Redis configuration for CASCADE Data Collector."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = os.getenv("REDIS_HOST", "cascade-keydb.internal")
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    url: Optional[str] = None

    @property
    def connection_url(self) -> str:
        """Get connection URL."""
        if self.url:
            return self.url
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class RedisKeys:
    """Redis key namespaces."""
    # Queue keys
    QUEUE_PREFIX = "cascade:queue"
    TASK_PREFIX = "cascade:task"
    WORKER_PREFIX = "cascade:worker"

    # Lock keys
    LOCK_PREFIX = "cascade:lock"
    SDR_LOCK_PREFIX = "cascade:lock:sdr"

    # Status keys
    STATUS_PREFIX = "cascade:status"
    METRICS_PREFIX = "cascade:metrics"

    # Event keys
    EVENT_PREFIX = "cascade:event"
    ALERT_PREFIX = "cascade:alert"

    @classmethod
    def queue_key(cls, band: str) -> str:
        """Get queue key for band."""
        return f"{cls.QUEUE_PREFIX}:{band}"

    @classmethod
    def task_key(cls, task_id: str) -> str:
        """Get task key."""
        return f"{cls.TASK_PREFIX}:{task_id}"

    @classmethod
    def worker_key(cls, worker_id: str) -> str:
        """Get worker key."""
        return f"{cls.WORKER_PREFIX}:{worker_id}"

    @classmethod
    def sdr_lock_key(cls, sdr_id: str) -> str:
        """Get SDR lock key."""
        return f"{cls.SDR_LOCK_PREFIX}:{sdr_id}"