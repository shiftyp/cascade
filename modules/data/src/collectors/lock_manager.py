"""Distributed lock manager for SDR resource claims.

Implements T027c: Distributed lock manager (FR-040).
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from uuid import uuid4
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from redis.exceptions import RedisError

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class LockInfo:
    """Information about an acquired lock."""

    resource: str
    token: str
    owner_id: str
    acquired_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any]


class LockManager:
    """Manages distributed locks for SDR resources and collection coordination."""

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize lock manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or config.REDIS_URL
        self.client: Optional[redis.Redis] = None
        self.owned_locks: Dict[str, LockInfo] = {}
        self.lock_refresh_task: Optional[asyncio.Task] = None
        self.owner_id = str(uuid4())[:12]

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
            logger.info(f"Lock manager connected to Redis (owner: {self.owner_id})")

            # Start lock refresh task
            self.lock_refresh_task = asyncio.create_task(self._refresh_locks())

        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect and release all locks."""
        # Cancel refresh task
        if self.lock_refresh_task:
            self.lock_refresh_task.cancel()

        # Release all owned locks
        await self.release_all_locks()

        if self.client:
            await self.client.close()
        logger.info("Lock manager disconnected")

    async def acquire_sdr_lock(
        self,
        sdr_url: str,
        duration_seconds: int = 600,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Acquire exclusive lock on an SDR.

        Args:
            sdr_url: SDR URL to lock
            duration_seconds: Lock duration
            metadata: Additional metadata

        Returns:
            Lock token or None if failed
        """
        resource = f"sdr:{sdr_url}"
        return await self.acquire_lock(resource, duration_seconds, metadata)

    async def acquire_frequency_lock(
        self,
        frequency_khz: float,
        band: str,
        duration_seconds: int = 300,
    ) -> Optional[str]:
        """Acquire lock on frequency within band.

        Args:
            frequency_khz: Frequency to lock
            band: Band name
            duration_seconds: Lock duration

        Returns:
            Lock token or None
        """
        resource = f"freq:{band}:{frequency_khz}"
        metadata = {"band": band, "frequency_khz": frequency_khz}
        return await self.acquire_lock(resource, duration_seconds, metadata)

    async def acquire_collection_slot(
        self,
        session_id: str,
        duration_seconds: int = 900,
    ) -> Optional[str]:
        """Acquire collection session slot.

        Args:
            session_id: Collection session ID
            duration_seconds: Session duration

        Returns:
            Lock token or None
        """
        resource = f"session:{session_id}"
        metadata = {"session_id": session_id}
        return await self.acquire_lock(resource, duration_seconds, metadata)

    async def acquire_lock(
        self,
        resource: str,
        duration_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Acquire a distributed lock.

        Args:
            resource: Resource identifier
            duration_seconds: Lock duration
            metadata: Additional metadata

        Returns:
            Lock token or None if failed
        """
        if not self.client:
            await self.connect()

        lock_key = f"lock:{resource}"
        token = str(uuid4())
        acquired_at = datetime.utcnow()
        expires_at = acquired_at + timedelta(seconds=duration_seconds)

        # Lock data
        lock_data = {
            "token": token,
            "owner_id": self.owner_id,
            "acquired_at": acquired_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "metadata": metadata or {},
        }

        # Try to acquire lock atomically
        acquired = await self.client.set(
            lock_key,
            json.dumps(lock_data),
            nx=True,  # Only set if not exists
            ex=duration_seconds,
        )

        if acquired:
            # Track owned lock
            lock_info = LockInfo(
                resource=resource,
                token=token,
                owner_id=self.owner_id,
                acquired_at=acquired_at,
                expires_at=expires_at,
                metadata=metadata or {},
            )
            self.owned_locks[resource] = lock_info

            logger.info(f"Acquired lock for {resource} (expires: {expires_at})")
            return token
        else:
            # Check who owns the lock
            existing_data = await self.client.get(lock_key)
            if existing_data:
                existing_lock = json.loads(existing_data)
                logger.debug(
                    f"Lock for {resource} held by {existing_lock.get('owner_id', 'unknown')}"
                )
            return None

    async def release_lock(self, resource: str, token: str) -> bool:
        """Release a specific lock.

        Args:
            resource: Resource identifier
            token: Lock token

        Returns:
            True if released successfully
        """
        lock_key = f"lock:{resource}"

        # Use Lua script for atomic check and delete
        lua_script = """
        local lock_data = redis.call("get", KEYS[1])
        if lock_data then
            local lock = cjson.decode(lock_data)
            if lock.token == ARGV[1] and lock.owner_id == ARGV[2] then
                return redis.call("del", KEYS[1])
            end
        end
        return 0
        """

        result = await self.client.eval(
            lua_script, 1, lock_key, token, self.owner_id
        )

        if result:
            # Remove from owned locks
            if resource in self.owned_locks:
                del self.owned_locks[resource]
            logger.info(f"Released lock for {resource}")
            return True
        else:
            logger.warning(f"Failed to release lock for {resource}")
            return False

    async def release_all_locks(self):
        """Release all locks owned by this manager."""
        resources = list(self.owned_locks.keys())

        for resource in resources:
            lock_info = self.owned_locks[resource]
            await self.release_lock(resource, lock_info.token)

        logger.info(f"Released {len(resources)} locks")

    async def extend_lock(
        self,
        resource: str,
        token: str,
        additional_seconds: int,
    ) -> bool:
        """Extend an existing lock.

        Args:
            resource: Resource identifier
            token: Lock token
            additional_seconds: Seconds to add

        Returns:
            True if extended successfully
        """
        lock_key = f"lock:{resource}"

        # Lua script for atomic extend
        lua_script = """
        local lock_data = redis.call("get", KEYS[1])
        if lock_data then
            local lock = cjson.decode(lock_data)
            if lock.token == ARGV[1] and lock.owner_id == ARGV[2] then
                local new_expiry = tonumber(ARGV[3])
                redis.call("expire", KEYS[1], new_expiry)

                -- Update expires_at in lock data
                local current_time = redis.call("time")
                local current_timestamp = current_time[1]
                lock.expires_at = os.date("!%Y-%m-%dT%H:%M:%S", current_timestamp + new_expiry)
                redis.call("set", KEYS[1], cjson.encode(lock), "EX", new_expiry)

                return 1
            end
        end
        return 0
        """

        result = await self.client.eval(
            lua_script, 1, lock_key, token, self.owner_id, additional_seconds
        )

        if result:
            # Update local tracking
            if resource in self.owned_locks:
                lock_info = self.owned_locks[resource]
                lock_info.expires_at += timedelta(seconds=additional_seconds)

            logger.debug(f"Extended lock for {resource} by {additional_seconds}s")
            return True
        else:
            return False

    async def get_lock_info(self, resource: str) -> Optional[LockInfo]:
        """Get information about a lock.

        Args:
            resource: Resource identifier

        Returns:
            LockInfo or None
        """
        lock_key = f"lock:{resource}"
        lock_data = await self.client.get(lock_key)

        if not lock_data:
            return None

        data = json.loads(lock_data)
        return LockInfo(
            resource=resource,
            token=data["token"],
            owner_id=data["owner_id"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            metadata=data.get("metadata", {}),
        )

    async def list_locks(
        self,
        pattern: str = "*",
        owner_only: bool = False,
    ) -> List[LockInfo]:
        """List current locks.

        Args:
            pattern: Resource pattern (Redis glob)
            owner_only: Only return locks owned by this manager

        Returns:
            List of LockInfo objects
        """
        lock_keys = await self.client.keys(f"lock:{pattern}")
        locks = []

        for lock_key in lock_keys:
            resource = lock_key[5:]  # Remove 'lock:' prefix
            lock_info = await self.get_lock_info(resource)

            if lock_info:
                if owner_only and lock_info.owner_id != self.owner_id:
                    continue
                locks.append(lock_info)

        return locks

    async def cleanup_expired_locks(self):
        """Clean up expired locks (maintenance task)."""
        all_locks = await self.list_locks()
        current_time = datetime.utcnow()
        cleaned = 0

        for lock_info in all_locks:
            if current_time > lock_info.expires_at:
                # Force delete expired lock
                lock_key = f"lock:{lock_info.resource}"
                deleted = await self.client.delete(lock_key)
                if deleted:
                    cleaned += 1
                    logger.debug(f"Cleaned expired lock: {lock_info.resource}")

        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} expired locks")

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get lock statistics.

        Returns:
            Statistics dictionary
        """
        all_locks = await self.list_locks()
        owned_locks = [l for l in all_locks if l.owner_id == self.owner_id]

        # Categorize by resource type
        sdr_locks = [l for l in all_locks if l.resource.startswith("sdr:")]
        freq_locks = [l for l in all_locks if l.resource.startswith("freq:")]
        session_locks = [l for l in all_locks if l.resource.startswith("session:")]

        return {
            "total_locks": len(all_locks),
            "owned_locks": len(owned_locks),
            "sdr_locks": len(sdr_locks),
            "frequency_locks": len(freq_locks),
            "session_locks": len(session_locks),
            "owner_id": self.owner_id,
        }

    async def _refresh_locks(self):
        """Background task to refresh owned locks."""
        while True:
            try:
                current_time = datetime.utcnow()
                refresh_threshold = timedelta(minutes=2)  # Refresh 2min before expiry

                for resource, lock_info in list(self.owned_locks.items()):
                    time_to_expiry = lock_info.expires_at - current_time

                    if time_to_expiry < refresh_threshold:
                        # Extend lock by 5 minutes
                        success = await self.extend_lock(
                            resource, lock_info.token, 300
                        )

                        if not success:
                            logger.warning(f"Failed to refresh lock: {resource}")
                            # Remove from owned locks
                            del self.owned_locks[resource]

                # Sleep for 30 seconds
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in lock refresh task: {e}")
                await asyncio.sleep(10)