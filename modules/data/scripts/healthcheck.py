#!/usr/bin/env python3
"""
Universal health check script for CASCADE processes.

Usage:
    python scripts/healthcheck.py <process_type>

Process types:
    - scheduler: Check scheduler health via file
    - worker: Check worker health via Redis
    - api: Check API HTTP endpoint
    - qa_api: Check QA API HTTP endpoint
    - dashboard: Check dashboard HTTP endpoint
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import httpx


def check_scheduler_health() -> bool:
    """Check scheduler health via file."""
    try:
        health_file = Path("/tmp/scheduler_health.json")

        if not health_file.exists():
            print("ERROR: Scheduler health file not found", file=sys.stderr)
            return False

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(health_file.stat().st_mtime)
        if file_age > timedelta(minutes=2):
            print(f"ERROR: Scheduler health file is stale ({file_age.total_seconds():.0f}s old)", file=sys.stderr)
            return False

        # Check content
        health_data = json.loads(health_file.read_text())

        if health_data.get("status") != "healthy":
            print(f"ERROR: Scheduler status is {health_data.get('status')}", file=sys.stderr)
            return False

        print(f"OK: Scheduler healthy (uptime: {health_data.get('uptime_seconds', 0):.0f}s)")
        return True

    except Exception as e:
        print(f"ERROR: Failed to check scheduler health: {e}", file=sys.stderr)
        return False


async def check_worker_health() -> bool:
    """Check worker health via Redis."""
    try:
        import redis.asyncio as redis

        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        client = redis.from_url(redis_url, decode_responses=True)

        # Check if any workers have reported health in last 2 minutes
        worker_keys = await client.keys("worker:*:health")

        if not worker_keys:
            print("WARNING: No worker health records found", file=sys.stderr)
            return False

        healthy_workers = 0
        for key in worker_keys:
            try:
                health_data = await client.get(key)
                if health_data:
                    health = json.loads(health_data)
                    timestamp = datetime.fromisoformat(health['timestamp'])
                    age = (datetime.utcnow() - timestamp).total_seconds()

                    if age < 120:  # Healthy if updated within 2 minutes
                        healthy_workers += 1
            except Exception:
                pass

        await client.close()

        if healthy_workers == 0:
            print(f"ERROR: No healthy workers found (checked {len(worker_keys)} workers)", file=sys.stderr)
            return False

        print(f"OK: {healthy_workers}/{len(worker_keys)} workers healthy")
        return True

    except Exception as e:
        print(f"ERROR: Failed to check worker health: {e}", file=sys.stderr)
        return False


async def check_http_health(url: str, path: str = "/health") -> bool:
    """Check HTTP health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}{path}")

            if response.status_code != 200:
                print(f"ERROR: HTTP {response.status_code} from {url}{path}", file=sys.stderr)
                return False

            data = response.json()
            status = data.get("status", "unknown")

            if status in ["healthy", "ok"]:
                print(f"OK: {url}{path} returned {status}")
                return True
            else:
                print(f"WARNING: {url}{path} returned status: {status}", file=sys.stderr)
                return False

    except httpx.TimeoutException:
        print(f"ERROR: Timeout connecting to {url}{path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Failed to check {url}{path}: {e}", file=sys.stderr)
        return False


async def main():
    if len(sys.argv) < 2:
        print("Usage: healthcheck.py <process_type>", file=sys.stderr)
        print("Process types: scheduler, worker, api, qa_api, dashboard", file=sys.stderr)
        sys.exit(2)

    process_type = sys.argv[1].lower()

    if process_type == "scheduler":
        success = check_scheduler_health()
    elif process_type == "worker":
        success = await check_worker_health()
    elif process_type == "api":
        port = os.getenv("DASHBOARD_PORT", "8000")
        success = await check_http_health(f"http://localhost:{port}")
    elif process_type == "qa_api":
        port = os.getenv("QA_API_PORT", "8001")
        success = await check_http_health(f"http://localhost:{port}")
    elif process_type == "dashboard":
        success = await check_http_health("http://localhost:3000", path="/")
    else:
        print(f"ERROR: Unknown process type: {process_type}", file=sys.stderr)
        print("Valid types: scheduler, worker, api, qa_api, dashboard", file=sys.stderr)
        sys.exit(2)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
