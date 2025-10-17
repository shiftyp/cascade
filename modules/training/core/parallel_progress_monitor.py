#!/usr/bin/env python3
"""
Parallel Progress Monitor for CASCADE Dataset Generation

Monitors multiple worker log files and displays combined progress in terminal.
Uses rich library for beautiful multi-progress bar display.
"""

import time
import re
from pathlib import Path
from collections import defaultdict
import sys

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  rich library not available. Using simple progress display.")


class ParallelProgressMonitor:
    """Monitor progress of multiple parallel workers."""

    def __init__(self, cache_dir: Path, num_workers: int):
        self.cache_dir = Path(cache_dir)
        self.num_workers = num_workers
        self.worker_stats = {}

        for i in range(num_workers):
            self.worker_stats[i] = {
                'current': 0,
                'total': 0,
                'rate': 0.0,
                'status': 'Starting...'
            }

    def parse_log_file(self, worker_id: int):
        """Parse worker log file to extract current progress."""
        log_file = self.cache_dir / f'worker_{worker_id:02d}.log'

        if not log_file.exists():
            return

        try:
            # Read last 1000 lines for speed
            with open(log_file, 'r') as f:
                lines = f.readlines()[-1000:]

            # Look for progress bar updates and status
            for line in reversed(lines):
                # Match tqdm progress: "Chunk 1/1:  45%|████▌     | 234/520 [01:23<01:45, 2.7streams/s]"
                match = re.search(r'(\d+)/(\d+) \[.+?(\d+\.?\d*)(it|streams)/s\]', line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    rate = float(match.group(3))
                    self.worker_stats[worker_id]['current'] = current
                    self.worker_stats[worker_id]['total'] = total
                    self.worker_stats[worker_id]['rate'] = rate
                    self.worker_stats[worker_id]['status'] = 'Generating'
                    break

                # Check for completion
                if '✓ Worker' in line and 'completed' in line:
                    self.worker_stats[worker_id]['status'] = 'Complete'
                    break

                # Check for errors
                if 'FAILED' in line or 'Error' in line:
                    self.worker_stats[worker_id]['status'] = 'Failed'
                    break

        except Exception as e:
            pass

    def display_rich(self):
        """Display using rich library with multiple progress bars."""
        console = Console()

        with Live(console=console, refresh_per_second=2) as live:
            while True:
                # Update stats from log files
                for worker_id in range(self.num_workers):
                    self.parse_log_file(worker_id)

                # Check if all complete
                statuses = [self.worker_stats[i]['status'] for i in range(self.num_workers)]
                if all(s in ['Complete', 'Failed'] for s in statuses):
                    break

                # Create table
                table = Table(title="CASCADE Parallel Dataset Generation")
                table.add_column("Worker", style="cyan", width=8)
                table.add_column("GPU", style="magenta", width=5)
                table.add_column("Progress", width=30)
                table.add_column("Rate", style="green", width=15)
                table.add_column("Status", width=12)

                total_current = 0
                total_total = 0
                total_rate = 0

                for worker_id in range(self.num_workers):
                    stats = self.worker_stats[worker_id]
                    current = stats['current']
                    total = stats['total']
                    rate = stats['rate']
                    status = stats['status']

                    total_current += current
                    total_total += total
                    total_rate += rate

                    # Progress bar
                    if total > 0:
                        pct = current / total * 100
                        bar_len = 20
                        filled = int(bar_len * current / total)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        progress_str = f"{bar} {pct:5.1f}% ({current}/{total})"
                    else:
                        progress_str = "Waiting..."

                    # Status color
                    if status == 'Complete':
                        status_str = f"[green]{status}[/green]"
                    elif status == 'Failed':
                        status_str = f"[red]{status}[/red]"
                    else:
                        status_str = f"[yellow]{status}[/yellow]"

                    table.add_row(
                        f"{worker_id}",
                        f"{worker_id}",
                        progress_str,
                        f"{rate:.1f} streams/s" if rate > 0 else "-",
                        status_str
                    )

                # Add total row
                if total_total > 0:
                    total_pct = total_current / total_total * 100
                    table.add_row(
                        "TOTAL",
                        "All",
                        f"{total_pct:5.1f}% ({total_current:,}/{total_total:,})",
                        f"[bold green]{total_rate:.1f} streams/s[/bold green]",
                        f"{sum(1 for s in statuses if s=='Complete')}/{self.num_workers} done"
                    )

                live.update(table)
                time.sleep(2)

        console.print("\n✓ All workers completed!")

    def display_simple(self):
        """Simple text-based display without rich."""
        while True:
            # Update stats
            for worker_id in range(self.num_workers):
                self.parse_log_file(worker_id)

            # Clear screen and display
            print("\033[2J\033[H")  # Clear screen
            print("=" * 80)
            print("CASCADE Parallel Dataset Generation")
            print("=" * 80)

            total_current = 0
            total_total = 0
            total_rate = 0.0

            for worker_id in range(self.num_workers):
                stats = self.worker_stats[worker_id]
                current = stats['current']
                total = stats['total']
                rate = stats['rate']
                status = stats['status']

                total_current += current
                total_total += total
                total_rate += rate

                pct = current / total * 100 if total > 0 else 0
                print(f"Worker {worker_id} (GPU {worker_id}): {current:6d}/{total:6d} ({pct:5.1f}%) | "
                      f"{rate:6.1f} streams/s | {status}")

            print("-" * 80)
            if total_total > 0:
                total_pct = total_current / total_total * 100
                print(f"TOTAL: {total_current:8d}/{total_total:8d} ({total_pct:5.1f}%) | "
                      f"{total_rate:6.1f} streams/s")

            # Check if done
            statuses = [self.worker_stats[i]['status'] for i in range(self.num_workers)]
            if all(s in ['Complete', 'Failed'] for s in statuses):
                print("\n✓ All workers completed!")
                break

            time.sleep(2)

    def monitor(self):
        """Monitor worker progress with best available display."""
        if RICH_AVAILABLE:
            self.display_rich()
        else:
            self.display_simple()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    monitor = ParallelProgressMonitor(Path(args.cache_dir), args.num_workers)
    monitor.monitor()
