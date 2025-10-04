"""
Redis/KeyDB configuration for distributed coordination
"""
import os
from typing import Optional

class RedisConfig:
    """Redis configuration for distributed worker coordination"""

    def __init__(self):
        # Connection settings
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.password = os.getenv('REDIS_PASSWORD', None)
        self.db = int(os.getenv('REDIS_DB', '0'))
        self.ssl = os.getenv('REDIS_SSL', 'false').lower() == 'true'

        # Connection pool settings
        self.max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '50'))
        self.socket_timeout = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
        self.socket_connect_timeout = int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
        self.socket_keepalive = True
        self.socket_keepalive_options = {}

        # Retry settings
        self.retry_on_timeout = True
        self.retry_on_error = [ConnectionError, TimeoutError]
        self.max_retries = int(os.getenv('REDIS_MAX_RETRIES', '3'))

        # Queue settings
        self.sdr_queue_key = 'cascade:sdr:queue'
        self.sdr_locks_prefix = 'cascade:sdr:lock:'
        self.worker_health_prefix = 'cascade:worker:health:'
        self.event_channel = 'cascade:events'

        # TTL settings (in seconds)
        self.sdr_lock_ttl = 1800  # 30 minutes
        self.worker_health_ttl = 60  # 1 minute
        self.queue_item_ttl = 3600  # 1 hour

    @property
    def connection_url(self) -> str:
        """Build Redis connection URL"""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

    def get_connection_kwargs(self) -> dict:
        """Get connection kwargs for redis-py"""
        kwargs = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'password': self.password,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.socket_connect_timeout,
            'socket_keepalive': self.socket_keepalive,
            'socket_keepalive_options': self.socket_keepalive_options,
            'retry_on_timeout': self.retry_on_timeout,
            'max_connections': self.max_connections,
        }

        if self.ssl:
            kwargs['ssl'] = True
            kwargs['ssl_cert_reqs'] = 'required'

        return kwargs


class RedisKeys:
    """Centralized Redis key management"""

    # Queue keys
    SDR_ASSIGNMENT_QUEUE = 'cascade:queue:sdr_assignments'
    RECORDING_QUEUE = 'cascade:queue:recordings'
    PROCESSING_QUEUE = 'cascade:queue:processing'
    FAILED_QUEUE = 'cascade:queue:failed'

    # Lock keys
    SDR_LOCK_PREFIX = 'cascade:lock:sdr:'
    WORKER_LOCK_PREFIX = 'cascade:lock:worker:'

    # Status keys
    SDR_STATUS_PREFIX = 'cascade:status:sdr:'
    WORKER_STATUS_PREFIX = 'cascade:status:worker:'
    COLLECTION_STATS = 'cascade:stats:collection'

    # Event channels
    EVENT_SPACE_WEATHER = 'cascade:event:space_weather'
    EVENT_SDR_FAILURE = 'cascade:event:sdr_failure'
    EVENT_SCALING = 'cascade:event:scaling'

    # Cache keys
    CACHE_SDR_LIST = 'cascade:cache:sdr_list'
    CACHE_SPACE_WEATHER = 'cascade:cache:space_weather'

    @staticmethod
    def sdr_lock(sdr_id: str) -> str:
        """Generate SDR lock key"""
        return f"{RedisKeys.SDR_LOCK_PREFIX}{sdr_id}"

    @staticmethod
    def worker_lock(worker_id: str) -> str:
        """Generate worker lock key"""
        return f"{RedisKeys.WORKER_LOCK_PREFIX}{worker_id}"

    @staticmethod
    def sdr_status(sdr_id: str) -> str:
        """Generate SDR status key"""
        return f"{RedisKeys.SDR_STATUS_PREFIX}{sdr_id}"

    @staticmethod
    def worker_status(worker_id: str) -> str:
        """Generate worker status key"""
        return f"{RedisKeys.WORKER_STATUS_PREFIX}{worker_id}"


# Queue message schemas
SDR_ASSIGNMENT_SCHEMA = {
    'sdr_id': str,
    'sdr_url': str,
    'frequency_hz': int,
    'duration_seconds': int,
    'priority': int,
    'correlation_id': Optional[str],
    'retry_count': int,
    'assigned_at': str,  # ISO timestamp
    'expires_at': str,  # ISO timestamp
}

RECORDING_TASK_SCHEMA = {
    'session_id': str,
    'sdr_id': str,
    'frequency_hz': int,
    'start_time': str,
    'duration_seconds': int,
    'file_path': str,
    'status': str,  # 'pending', 'recording', 'completed', 'failed'
    'worker_id': str,
    'error_message': Optional[str],
}

PROCESSING_TASK_SCHEMA = {
    'session_id': str,
    'file_path': str,
    'task_type': str,  # 'ft8', 'wspr', 'qrn'
    'priority': int,
    'status': str,  # 'pending', 'processing', 'completed', 'failed'
    'worker_id': Optional[str],
    'result': Optional[dict],
    'error_message': Optional[str],
}