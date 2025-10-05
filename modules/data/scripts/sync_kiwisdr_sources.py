#!/usr/bin/env python3
"""
Sync KiwiSDR sources from public directory to local database.

Fetches the public KiwiSDR list from kiwisdr.com/public/ and updates
the local kiwisdr_sources table.

Usage:
    python scripts/sync_kiwisdr_sources.py [--dry-run]
"""

import os
import sys
import logging
import re
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

import httpx
import psycopg2
from psycopg2.extras import execute_values

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KiwiSDR JSON data source (updated automatically from kiwisdr.com/public/)
KIWI_JSON_URL = "http://rx.linkfanel.net/kiwisdr_com.js"

# Fallback sources
ALTERNATIVE_SOURCES = [
    "http://kiwisdr.com/public/",  # HTML page (requires parsing)
]


def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment.

    Reads from DATABASE_URL (postgresql://user:pass@host:port/dbname) if available,
    otherwise falls back to individual env vars.
    """
    database_url = os.getenv('DATABASE_URL')

    if database_url:
        # Parse DATABASE_URL (format: postgresql://user:pass@host:port/dbname)
        from urllib.parse import urlparse
        parsed = urlparse(database_url)

        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/') if parsed.path else 'cascade_data',
            'user': parsed.username or 'postgres',
            'password': parsed.password or 'postgres'
        }

    # Fallback to individual env vars
    return {
        'host': os.getenv('CASCADE_DB_HOST', os.getenv('DB_HOST', 'localhost')),
        'port': int(os.getenv('CASCADE_DB_PORT', '5432')),
        'database': os.getenv('CASCADE_DB_NAME', 'cascade_data'),
        'user': os.getenv('CASCADE_DB_USER', 'postgres'),
        'password': os.getenv('CASCADE_DB_PASSWORD', 'postgres')
    }


def parse_grid_square(location: str) -> Optional[str]:
    """Extract Maidenhead grid square from location string."""
    # Look for 4 or 6 character grid square pattern
    match = re.search(r'\b([A-R]{2}[0-9]{2}[a-x]{0,2})\b', location, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_coordinates(location: str) -> tuple[Optional[float], Optional[float]]:
    """Parse latitude/longitude from location string."""
    # Look for coordinate patterns like "40.7N, 74.0W" or "(40.7, -74.0)"
    # This is a simplified parser - adjust based on actual format
    lat_match = re.search(r'(-?\d+\.?\d*)[°\s]*([NS])?', location, re.IGNORECASE)
    lon_match = re.search(r'(-?\d+\.?\d*)[°\s]*([EW])?', location, re.IGNORECASE)

    lat = None
    lon = None

    if lat_match:
        lat = float(lat_match.group(1))
        if lat_match.group(2) and lat_match.group(2).upper() == 'S':
            lat = -abs(lat)

    if lon_match:
        lon = float(lon_match.group(1))
        if lon_match.group(2) and lon_match.group(2).upper() == 'W':
            lon = -abs(lon)

    # Round for privacy (to ~11km precision)
    if lat:
        lat = round(lat, 1)
    if lon:
        lon = round(lon, 1)

    return lat, lon


def fetch_kiwisdr_list() -> List[Dict[str, any]]:
    """
    Fetch and parse KiwiSDR list from rx.linkfanel.net JSON source.

    This source is automatically updated from kiwisdr.com/public/ and provides
    structured JSON data wrapped in a JavaScript variable declaration.

    Returns:
        List of KiwiSDR receiver dictionaries
    """
    logger.info(f"Fetching KiwiSDR list from {KIWI_JSON_URL}")

    try:
        # Fetch JavaScript file with timeout
        client = httpx.Client(timeout=30.0, follow_redirects=True)
        response = client.get(KIWI_JSON_URL)
        response.raise_for_status()

        js_content = response.text
        logger.info(f"Fetched {len(js_content)} bytes of JavaScript")

        # Parse JavaScript variable declaration
        # Format: var kiwisdr_com = [ ... ];
        # Extract the JSON array between [ and ];

        # Find start of array
        start_idx = js_content.find('[')
        if start_idx == -1:
            raise ValueError("Could not find JSON array start in JavaScript file")

        # Find end of array (last closing bracket before semicolon)
        end_idx = js_content.rfind(']')
        if end_idx == -1:
            raise ValueError("Could not find JSON array end in JavaScript file")

        # Extract JSON string
        json_str = js_content[start_idx:end_idx + 1]

        # Clean up trailing commas (JSON doesn't allow them, but JavaScript does)
        # Replace ",]" with "]" to fix trailing comma before closing bracket
        json_str = re.sub(r',\s*]', ']', json_str)
        # Also handle ",}" for trailing commas in objects
        json_str = re.sub(r',\s*}', '}', json_str)

        # Parse JSON
        kiwisdr_data = json.loads(json_str)

        if not isinstance(kiwisdr_data, list):
            raise ValueError(f"Expected list, got {type(kiwisdr_data)}")

        logger.info(f"Parsed {len(kiwisdr_data)} receivers from JSON")

        # Convert to our internal format
        receivers = []
        for item in kiwisdr_data:
            receiver = parse_kiwisdr_json(item)
            if receiver:
                receivers.append(receiver)

        logger.info(f"Processed {len(receivers)} valid receivers")
        return receivers

    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching KiwiSDR list: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing KiwiSDR list: {e}")
        raise


def parse_kiwisdr_json(data: Dict[str, any]) -> Optional[Dict[str, any]]:
    """Parse receiver data from kiwisdr_com.js JSON format.

    Args:
        data: JSON object from kiwisdr_com.js

    Returns:
        Receiver dictionary in our internal format, or None if invalid
    """
    try:
        # Skip offline or inactive receivers
        status = data.get('status', 'offline')
        offline = data.get('offline', 'yes')

        if status != 'active' or offline == 'yes':
            return None

        # Extract URL
        url = data.get('url')
        if not url:
            return None

        # Extract name and location
        name = data.get('name', '')
        location = data.get('loc', '')

        # Extract GPS coordinates
        # Format: "(lat, lon)" as string
        gps_str = data.get('gps', '')
        lat, lon = None, None

        if gps_str:
            # Parse GPS string like "(-45.91, 170.34)"
            gps_match = re.search(r'\(([^,]+),\s*([^)]+)\)', gps_str)
            if gps_match:
                try:
                    lat = float(gps_match.group(1))
                    lon = float(gps_match.group(2))
                    # Round for privacy (to ~11km precision)
                    lat = round(lat, 1)
                    lon = round(lon, 1)
                except ValueError:
                    pass

        # Extract Maidenhead grid square
        grid_square = data.get('grid')

        # Extract antenna info
        antenna = data.get('antenna')

        # Extract max users
        users_max = data.get('users_max')
        try:
            max_users = int(users_max) if users_max else 4
        except (ValueError, TypeError):
            max_users = 4

        # Check if GPS is available
        has_gps = bool(data.get('gps_good'))

        # Extract additional metadata
        sw_version = data.get('sw_version', '')
        bands = data.get('bands', '0-30000000')

        return {
            'url': url,
            'name': name,
            'grid_square': grid_square,
            'latitude': lat,
            'longitude': lon,
            'antenna_type': antenna,
            'has_gps': has_gps,
            'max_users': max_users,
            'location': location,
            'sw_version': sw_version,
            'bands': bands,
        }

    except Exception as e:
        logger.warning(f"Error parsing receiver JSON: {e}")
        return None


# Legacy HTML parsing functions (kept for fallback, but not used by default)
def parse_table_row(cols: List) -> Optional[Dict[str, any]]:
    """Parse receiver data from HTML table row (fallback parser)."""
    # Not used with JSON source, kept for future fallback implementation
    return None


def parse_receiver_div(div) -> Optional[Dict[str, any]]:
    """Parse receiver data from HTML div element (fallback parser)."""
    # Not used with JSON source, kept for future fallback implementation
    return None


def sync_to_database(receivers: List[Dict[str, any]], dry_run: bool = False):
    """
    Sync receiver list to database.

    Args:
        receivers: List of receiver dictionaries
        dry_run: If True, don't actually update database
    """
    if dry_run:
        logger.info("DRY RUN MODE - No database changes will be made")

    config = get_db_config()

    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        now = datetime.utcnow()

        logger.info(f"Syncing {len(receivers)} receivers to database")

        new_count = 0
        updated_count = 0
        marked_inactive = 0

        # Get existing URLs
        cur.execute("SELECT url FROM kiwisdr_sources")
        existing_urls = {row[0] for row in cur.fetchall()}

        seen_urls = set()

        for receiver in receivers:
            url = receiver['url']
            seen_urls.add(url)

            # Check if exists
            if url in existing_urls:
                # Update existing
                if not dry_run:
                    cur.execute("""
                        UPDATE kiwisdr_sources
                        SET
                            name = COALESCE(%s, name),
                            grid_square = COALESCE(%s, grid_square),
                            latitude = COALESCE(%s, latitude),
                            longitude = COALESCE(%s, longitude),
                            antenna_type = COALESCE(%s, antenna_type),
                            last_seen = %s,
                            active = true,
                            updated_at = %s
                        WHERE url = %s
                    """, (
                        receiver.get('name'),
                        receiver.get('grid_square'),
                        receiver.get('latitude'),
                        receiver.get('longitude'),
                        receiver.get('antenna_type'),
                        now,
                        now,
                        url
                    ))
                updated_count += 1
            else:
                # Insert new
                if not dry_run:
                    cur.execute("""
                        INSERT INTO kiwisdr_sources (
                            url, name, grid_square, latitude, longitude,
                            antenna_type, has_gps, max_users, active,
                            last_seen, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (url) DO NOTHING
                    """, (
                        url,
                        receiver.get('name'),
                        receiver.get('grid_square'),
                        receiver.get('latitude'),
                        receiver.get('longitude'),
                        receiver.get('antenna_type'),
                        receiver.get('has_gps', True),
                        receiver.get('max_users', 4),
                        True,
                        now,
                        now,
                        now
                    ))
                new_count += 1

        # Mark receivers not in current list as inactive
        missing_urls = existing_urls - seen_urls
        if missing_urls and not dry_run:
            cur.execute("""
                UPDATE kiwisdr_sources
                SET active = false, updated_at = %s
                WHERE url = ANY(%s) AND active = true
            """, (now, list(missing_urls)))
            marked_inactive = cur.rowcount

        if not dry_run:
            conn.commit()

        logger.info(f"Sync complete:")
        logger.info(f"  New receivers: {new_count}")
        logger.info(f"  Updated receivers: {updated_count}")
        logger.info(f"  Marked inactive: {marked_inactive}")

        # Show totals
        if not dry_run:
            cur.execute("SELECT COUNT(*) FROM kiwisdr_sources WHERE active = true")
            active_count = cur.fetchone()[0]
            logger.info(f"  Total active receivers: {active_count}")

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Database error: {e}")
        raise


def main():
    """Main sync function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync KiwiSDR sources from public directory"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    logger.info("CASCADE KiwiSDR Source Sync")
    logger.info("=" * 60)

    try:
        # Fetch list
        receivers = fetch_kiwisdr_list()

        if not receivers:
            logger.error("No receivers found. Check HTML parsing logic.")
            sys.exit(1)

        # Sync to database
        sync_to_database(receivers, dry_run=args.dry_run)

        logger.info("=" * 60)
        logger.info("Sync complete!")

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
