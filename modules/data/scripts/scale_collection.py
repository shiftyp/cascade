#!/usr/bin/env python3
"""
Live scaling control for CASCADE Data Collector.

Allows adjusting collection parameters without restart or data loss.
"""

import argparse
import json
import os
import redis
import sys
from typing import Optional


def update_config(
    baseline: Optional[int] = None,
    max_sdrs: Optional[int] = None,
    min_hours: Optional[int] = None,
    redis_url: str = os.getenv("REDIS_URL", "redis://cascade-keydb.internal:6379"),
):
    """Update collection scaling parameters live.

    Args:
        baseline: New baseline SDR count (e.g., 50)
        max_sdrs: New max SDR count for events (e.g., 200)
        min_hours: New minimum collection hours/day alert threshold
        redis_url: Redis connection URL
    """
    try:
        # Connect to Redis
        r = redis.from_url(redis_url, decode_responses=True)

        # Get existing config
        config_key = "scheduler:dynamic_config"
        existing = r.get(config_key)
        config = json.loads(existing) if existing else {}

        # Update values
        if baseline is not None:
            config['baseline_sdr_count'] = baseline
            print(f"✓ Setting baseline SDR count to {baseline}")

        if max_sdrs is not None:
            config['max_sdr_count'] = max_sdrs
            print(f"✓ Setting max SDR count to {max_sdrs}")

        if min_hours is not None:
            if 'alert_thresholds' not in config:
                config['alert_thresholds'] = {}
            config['alert_thresholds']['collection_rate_hours_per_day'] = min_hours
            print(f"✓ Setting minimum hours/day alert to {min_hours}")

        # Save to Redis
        r.set(config_key, json.dumps(config))
        print(f"\n✅ Configuration updated successfully!")
        print(f"   Changes will take effect within 30 seconds.")
        print(f"   No restart required. No data loss.")

        # Show current config
        print(f"\nCurrent configuration:")
        print(json.dumps(config, indent=2))

    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_url}")
        print("   Make sure Redis is running and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Scale CASCADE Data Collection without restart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start small for testing
  %(prog)s --baseline 6 --max 50

  # Scale up after successful testing
  %(prog)s --baseline 50 --max 200

  # Target production scale (FR-016, FR-018)
  %(prog)s --baseline 100 --max 300 --min-hours 200

  # Connect to Fly.io Redis
  %(prog)s --redis redis://cascade-keydb.internal:6379 --baseline 50
        """
    )

    parser.add_argument(
        '--baseline',
        type=int,
        help='Baseline SDR count (default: keep current)'
    )
    parser.add_argument(
        '--max',
        type=int,
        dest='max_sdrs',
        help='Maximum SDRs during events (default: keep current)'
    )
    parser.add_argument(
        '--min-hours',
        type=int,
        help='Minimum collection hours/day before alert (default: keep current)'
    )
    parser.add_argument(
        '--redis',
        default=os.getenv('REDIS_URL', 'redis://cascade-keydb.internal:6379'),
        help='Redis URL (default: from REDIS_URL env var or redis://cascade-keydb.internal:6379)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration only'
    )

    args = parser.parse_args()

    if args.show:
        try:
            r = redis.from_url(args.redis, decode_responses=True)
            config = r.get("scheduler:dynamic_config")
            if config:
                print("Current configuration:")
                print(json.dumps(json.loads(config), indent=2))
            else:
                print("No dynamic configuration set (using defaults from environment)")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not any([args.baseline, args.max_sdrs, args.min_hours]):
            parser.print_help()
            sys.exit(1)

        update_config(
            baseline=args.baseline,
            max_sdrs=args.max_sdrs,
            min_hours=args.min_hours,
            redis_url=args.redis
        )


if __name__ == "__main__":
    main()