#!/usr/bin/env python3
"""Terminal-based dashboard for CASCADE data collection monitoring."""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import curses
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    filename='/tmp/cascade_dashboard.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DashboardData:
    """Fetches and caches dashboard data from PostgreSQL."""

    def __init__(self, db_config: Dict[str, str]):
        """Initialize with database configuration."""
        self.db_config = db_config
        self.conn = None
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 30  # seconds

    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from database."""
        if self.conn:
            self.conn.close()

    def _fetch_query(self, query: str, cache_key: str) -> List[Dict]:
        """Fetch query results with caching."""
        # Check cache
        if cache_key in self.cache:
            if time.time() - self.cache_time.get(cache_key, 0) < self.cache_ttl:
                return self.cache[cache_key]

        # Fetch fresh data
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                self.cache[cache_key] = results
                self.cache_time[cache_key] = time.time()
                return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_collection_status(self) -> Dict:
        """Get overall collection status."""
        query = "SELECT * FROM v_collection_status"
        results = self._fetch_query(query, "collection_status")
        return results[0] if results else {}

    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        """Get hourly collection statistics."""
        query = f"""
            SELECT * FROM v_hourly_collection_stats
            WHERE hour_bin >= NOW() - INTERVAL '{hours} hours'
            ORDER BY hour_bin DESC
        """
        return self._fetch_query(query, f"hourly_stats_{hours}")

    def get_sdr_performance(self) -> List[Dict]:
        """Get SDR performance metrics."""
        query = """
            SELECT * FROM v_sdr_performance
            WHERE last_used >= NOW() - INTERVAL '24 hours'
            ORDER BY is_active DESC, total_hours DESC
            LIMIT 20
        """
        return self._fetch_query(query, "sdr_performance")

    def get_band_coverage(self) -> List[Dict]:
        """Get band coverage statistics."""
        query = """
            SELECT frequency_band,
                   SUM(avg_hours) as total_hours,
                   AVG(avg_sdrs) as avg_sdrs
            FROM v_band_coverage
            GROUP BY frequency_band
            ORDER BY frequency_band
        """
        return self._fetch_query(query, "band_coverage")

    def get_space_weather(self) -> List[Dict]:
        """Get recent space weather events."""
        query = """
            SELECT * FROM v_space_weather_events
            WHERE timestamp >= NOW() - INTERVAL '48 hours'
            ORDER BY timestamp DESC
            LIMIT 10
        """
        return self._fetch_query(query, "space_weather")

    def get_storage_usage(self) -> List[Dict]:
        """Get storage usage breakdown."""
        query = "SELECT * FROM v_storage_usage ORDER BY total_gb DESC"
        return self._fetch_query(query, "storage_usage")


class TerminalDashboard:
    """Terminal-based dashboard interface."""

    def __init__(self, data_source: DashboardData):
        """Initialize dashboard with data source."""
        self.data = data_source
        self.stdscr = None
        self.height = 0
        self.width = 0
        self.current_page = 0
        self.pages = ["Overview", "SDRs", "Bands", "Weather", "Storage"]

    def run(self, stdscr):
        """Main dashboard loop."""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(1000)  # Refresh every second

        # Color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

        while True:
            self.height, self.width = stdscr.getmaxyx()
            stdscr.clear()

            # Draw header
            self._draw_header()

            # Draw current page
            if self.current_page == 0:
                self._draw_overview()
            elif self.current_page == 1:
                self._draw_sdr_status()
            elif self.current_page == 2:
                self._draw_band_coverage()
            elif self.current_page == 3:
                self._draw_space_weather()
            elif self.current_page == 4:
                self._draw_storage()

            # Draw footer
            self._draw_footer()

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == curses.KEY_RIGHT:
                self.current_page = (self.current_page + 1) % len(self.pages)
            elif key == curses.KEY_LEFT:
                self.current_page = (self.current_page - 1) % len(self.pages)
            elif key == ord('r'):
                self.data.cache.clear()  # Force refresh

    def _draw_header(self):
        """Draw dashboard header."""
        title = "CASCADE Data Collection Dashboard"
        page_name = self.pages[self.current_page]
        header = f"{title} - {page_name}"

        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(0, (self.width - len(header)) // 2, header)
        self.stdscr.attroff(curses.A_BOLD)

        # Current time
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.stdscr.addstr(1, self.width - len(time_str) - 2, time_str)

        # Separator
        self.stdscr.hline(2, 0, "─", self.width)

    def _draw_footer(self):
        """Draw dashboard footer."""
        self.stdscr.hline(self.height - 3, 0, "─", self.width)

        help_text = "←/→: Navigate | r: Refresh | q: Quit"
        self.stdscr.addstr(self.height - 2, 2, help_text)

        # Page indicators
        page_indicator = " ".join(
            f"[{p}]" if i == self.current_page else f" {p} "
            for i, p in enumerate(self.pages)
        )
        self.stdscr.addstr(
            self.height - 2,
            self.width - len(page_indicator) - 2,
            page_indicator
        )

    def _draw_overview(self):
        """Draw overview page."""
        status = self.data.get_collection_status()
        if not status:
            self.stdscr.addstr(4, 2, "No data available")
            return

        y = 4
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Collection Status")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        # Format statistics
        stats = [
            ("Total Sessions", f"{status.get('total_sessions', 0):,}"),
            ("Unique SDRs", f"{status.get('unique_sdrs', 0)}"),
            ("Total Hours", f"{status.get('total_hours_collected', 0):,.1f}"),
            ("Storage Used", f"{status.get('total_storage_gb', 0):,.1f} GB"),
            ("Bands Covered", f"{status.get('bands_covered', 0)}"),
            ("Collection Days", f"{status.get('collection_days', 0)}"),
        ]

        for label, value in stats:
            self.stdscr.addstr(y, 4, f"{label}:")
            self.stdscr.addstr(y, 25, value, curses.color_pair(1))
            y += 1

        # Hourly collection chart
        y += 2
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "24-Hour Collection Rate")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        hourly = self.data.get_hourly_stats(24)
        if hourly:
            self._draw_hourly_chart(y, hourly)

    def _draw_hourly_chart(self, start_y: int, data: List[Dict]):
        """Draw hourly collection chart."""
        # Group by hour
        hourly_totals = defaultdict(float)
        for row in data:
            hour = row['hour_bin'].hour if hasattr(row['hour_bin'], 'hour') else 0
            hourly_totals[hour] += row.get('hours_collected', 0)

        # Find max for scaling
        max_hours = max(hourly_totals.values()) if hourly_totals else 1
        chart_height = min(10, self.height - start_y - 5)
        chart_width = min(50, self.width - 10)

        # Draw chart
        for h in range(24):
            x = 4 + (h * 2)
            if x >= self.width - 2:
                break

            hours = hourly_totals.get(h, 0)
            bar_height = int((hours / max_hours) * chart_height)

            # Draw bar
            for i in range(bar_height):
                y = start_y + chart_height - i
                self.stdscr.addstr(y, x, "█", curses.color_pair(4))

            # Hour label
            self.stdscr.addstr(start_y + chart_height + 1, x, f"{h:02d}"[:2])

    def _draw_sdr_status(self):
        """Draw SDR status page."""
        sdrs = self.data.get_sdr_performance()

        y = 4
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "SDR Performance (Last 24 Hours)")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        # Headers
        headers = ["SDR Name", "Grid", "Sessions", "Hours", "Success", "Status"]
        x_positions = [2, 20, 30, 40, 50, 60]

        for header, x in zip(headers, x_positions):
            self.stdscr.addstr(y, x, header, curses.A_UNDERLINE)
        y += 1

        # SDR data
        for sdr in sdrs[:self.height - y - 4]:
            name = str(sdr.get('sdr_name', 'Unknown'))[:17]
            grid = str(sdr.get('grid_square', '--'))[:6]
            sessions = f"{sdr.get('total_sessions', 0)}"
            hours = f"{sdr.get('total_hours', 0):.1f}"
            success = f"{sdr.get('success_rate', 0):.0f}%"
            active = "Active" if sdr.get('is_active') else "Inactive"

            # Color based on status
            color = curses.color_pair(1) if sdr.get('is_active') else curses.color_pair(3)

            self.stdscr.addstr(y, x_positions[0], name)
            self.stdscr.addstr(y, x_positions[1], grid)
            self.stdscr.addstr(y, x_positions[2], sessions)
            self.stdscr.addstr(y, x_positions[3], hours)
            self.stdscr.addstr(y, x_positions[4], success)
            self.stdscr.addstr(y, x_positions[5], active, color)
            y += 1

    def _draw_band_coverage(self):
        """Draw band coverage page."""
        bands = self.data.get_band_coverage()

        y = 4
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Band Coverage Statistics")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        if not bands:
            self.stdscr.addstr(y, 2, "No band data available")
            return

        # Draw band bars
        for band_data in bands:
            band = band_data.get('frequency_band', 'Unknown')
            hours = band_data.get('total_hours', 0)
            sdrs = band_data.get('avg_sdrs', 0)

            # Band label
            self.stdscr.addstr(y, 4, f"{band:>5}:")

            # Hours bar
            bar_width = min(int(hours / 10), self.width - 20)
            self.stdscr.addstr(y, 12, "█" * bar_width, curses.color_pair(4))
            self.stdscr.addstr(y, 12 + bar_width + 2, f"{hours:.0f}h")
            y += 1

    def _draw_space_weather(self):
        """Draw space weather events."""
        events = self.data.get_space_weather()

        y = 4
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Recent Space Weather Events")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        if not events:
            self.stdscr.addstr(y, 2, "No recent space weather events")
            return

        for event in events[:self.height - y - 4]:
            timestamp = event.get('timestamp', '')
            xray_class = event.get('xray_class', '--')
            recordings = event.get('concurrent_recordings', 0)
            bands = event.get('bands_recorded', '--')

            # Format line
            if isinstance(timestamp, str):
                time_str = timestamp[:16]
            else:
                time_str = timestamp.strftime("%Y-%m-%d %H:%M")

            # Color based on class
            if xray_class.startswith('X'):
                color = curses.color_pair(3)
            elif xray_class.startswith('M'):
                color = curses.color_pair(2)
            else:
                color = curses.color_pair(1)

            line = f"{time_str} | {xray_class:>6} | {recordings:>3} recordings | {bands}"
            self.stdscr.addstr(y, 4, line[:self.width-6], color)
            y += 1

    def _draw_storage(self):
        """Draw storage usage page."""
        storage = self.data.get_storage_usage()

        y = 4
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(y, 2, "Storage Usage by Band")
        self.stdscr.attroff(curses.A_BOLD)
        y += 2

        if not storage:
            self.stdscr.addstr(y, 2, "No storage data available")
            return

        total_gb = sum(s.get('total_gb', 0) for s in storage)
        target_gb = 75000  # 75TB target

        # Overall usage
        percent = (total_gb / target_gb) * 100
        self.stdscr.addstr(y, 4, f"Total: {total_gb:,.1f} GB / {target_gb:,} GB ({percent:.2f}%)")
        y += 2

        # Per-band breakdown
        for item in storage[:self.height - y - 4]:
            band = item.get('frequency_band', 'Unknown')
            size_gb = item.get('total_gb', 0)
            files = item.get('file_count', 0)
            ratio = item.get('compression_ratio', 1.0)

            line = f"{band:>5}: {size_gb:>8.1f} GB | {files:>5} files | {ratio:.1%} compression"
            self.stdscr.addstr(y, 4, line[:self.width-6])
            y += 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CASCADE Terminal Dashboard")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="cascade", help="Database name")
    parser.add_argument("--user", default="cascade", help="Database user")
    parser.add_argument("--password", help="Database password")

    args = parser.parse_args()

    # Database configuration
    db_config = {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "user": args.user,
    }
    if args.password:
        db_config["password"] = args.password

    # Create data source
    data_source = DashboardData(db_config)

    # Connect to database
    if not data_source.connect():
        print("Failed to connect to database")
        sys.exit(1)

    try:
        # Run dashboard
        dashboard = TerminalDashboard(data_source)
        curses.wrapper(dashboard.run)
    finally:
        data_source.disconnect()


if __name__ == "__main__":
    main()