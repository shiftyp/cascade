"""Rich terminal UI dashboard for tournament pattern generation

Provides a real-time dashboard similar to 'top' showing trial status,
statistics, and activity log.
"""

import time
import threading
from datetime import datetime
from typing import List, Optional
from collections import deque

from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich.align import Align


class PatternGeneratorDashboard:
    """Rich terminal UI for pattern generation monitoring"""

    def __init__(self, tournament_optimizer=None):
        # Force terminal detection for proper rendering
        self.console = Console(force_terminal=True, force_interactive=True)
        self.optimizer = tournament_optimizer
        self.layout = Layout()
        self.running = False
        self.recent_logs = deque(maxlen=12)
        self.start_time = None

        # UI update thread
        self.ui_thread = None

        # Setup layout structure
        self._setup_layout()

    def _setup_layout(self):
        """Create the dashboard layout structure"""
        # Main layout split
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1)
        )

        # Body split into two columns
        self.layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", ratio=1)
        )

        # Main area split
        self.layout["main"].split(
            Layout(name="trials", size=15),
            Layout(name="log", size=12)
        )

        # Sidebar split
        self.layout["sidebar"].split(
            Layout(name="stats", size=15),
            Layout(name="info", size=12)
        )

    def generate_header(self) -> Panel:
        """Generate the header panel"""
        if self.optimizer:
            compute_pct = (self.optimizer.compute_used / self.optimizer.total_budget * 100) if self.optimizer.total_budget > 0 else 0
            phase_color = {
                'exploration': 'cyan',
                'evaluation': 'yellow',
                'exploitation': 'orange1',
                'refinement': 'red'
            }.get(self.optimizer.current_phase, 'white')

            header_text = Text()
            header_text.append("CASCADE Pattern Tournament", style="bold white")
            header_text.append(" | ", style="dim")
            header_text.append(f"Phase: {self.optimizer.current_phase.upper()}", style=f"bold {phase_color}")
            header_text.append(" | ", style="dim")
            header_text.append(f"Progress: {compute_pct:.1f}%", style="green")
            header_text.append(f" ({self.optimizer.compute_used:,}/{self.optimizer.total_budget:,})", style="dim")

            return Panel(
                Align.center(header_text),
                style="bold blue on black",
                padding=0
            )
        else:
            return Panel(
                Align.center(Text("CASCADE Pattern Tournament - Initializing...", style="bold white")),
                style="bold blue on black",
                padding=0
            )

    def generate_trials_table(self) -> Panel:
        """Generate the trials status table"""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            title="Trial Status",
            title_style="bold white",
            expand=True,
            show_lines=False
        )

        # Define columns
        table.add_column("ID", style="cyan", width=4, justify="center")
        table.add_column("Status", width=8, justify="center")
        table.add_column("Core", width=8, justify="center")
        table.add_column("Iteration", justify="right", width=10)
        table.add_column("Best Score", justify="right", width=12)
        table.add_column("Conv Rate", justify="right", width=10)
        table.add_column("Progress", width=20)
        table.add_column("ETA", width=8, justify="center")

        if self.optimizer and self.optimizer.trials:
            for trial in self.optimizer.trials:
                # Status icon and color
                status_display = self._get_status_display(trial.status, trial.is_best)

                # Progress bar
                progress_bar = self._create_progress_bar(trial.progress)

                # Row style
                row_style = self._get_row_style(trial)

                # Core display
                if trial.p_cores:
                    core_display = f"P{trial.p_cores[0]}" if len(trial.p_cores) == 1 else f"P{trial.p_cores[0]}-{trial.p_cores[-1]}"
                else:
                    core_display = "-"

                table.add_row(
                    str(trial.trial_id),
                    status_display,
                    core_display,
                    f"{trial.iterations:,}",
                    f"{trial.best_score:.2f} dB" if trial.best_score != float('inf') else "-",
                    f"{trial.convergence_rate:.4f}",
                    progress_bar,
                    trial.eta,
                    style=row_style
                )
        else:
            # Empty table
            table.add_row("-", "-", "-", "-", "-", "-", "-", "-", style="dim")

        return Panel(table, border_style="bright_blue", padding=(0, 1))

    def generate_stats_panel(self) -> Panel:
        """Generate the statistics panel"""
        if self.optimizer:
            # Calculate statistics
            active_count = len(self.optimizer.active_trials)
            eliminated_count = len([t for t in self.optimizer.trials if t.eliminated])
            compute_remaining = self.optimizer.total_budget - self.optimizer.compute_used

            # Runtime calculation
            if self.optimizer.start_time:
                runtime = datetime.now() - self.optimizer.start_time
                runtime_str = f"{int(runtime.total_seconds() // 3600):02d}:{int((runtime.total_seconds() % 3600) // 60):02d}"
            else:
                runtime_str = "00:00"

            # Create stats text
            # Format best score
            if self.optimizer.global_best_score == float('inf'):
                best_score_str = "-"
            else:
                best_score_str = f"{self.optimizer.global_best_score:.2f} dB"

            stats_lines = [
                f"[bold cyan]Active Trials:[/]    {active_count}",
                f"[bold red]Eliminated:[/]       {eliminated_count}",
                f"[bold yellow]Compute Used:[/]     {self.optimizer.compute_used:,}",
                f"[bold green]Compute Left:[/]     {compute_remaining:,}",
                "",
                f"[bold]Best Score:[/]       {best_score_str}",
                f"[bold]Best Trial:[/]       #{self.optimizer.global_best_trial_id or '-'}",
                "",
                f"[bold]Runtime:[/]          {runtime_str}",
                f"[bold]Phase:[/]            {self.optimizer.current_phase.title()}"
            ]

            stats_text = "\n".join(stats_lines)
        else:
            stats_text = "[dim]No data available[/dim]"

        return Panel(
            stats_text,
            title="[bold white]Statistics[/bold white]",
            border_style="green",
            padding=(0, 1)
        )

    def generate_info_panel(self) -> Panel:
        """Generate the system info panel"""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)

            # Memory usage
            memory = psutil.virtual_memory()
            mem_used_gb = memory.used / (1024**3)
            mem_total_gb = memory.total / (1024**3)

            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                freq_ghz = cpu_freq.current / 1000
            else:
                freq_ghz = 0

            info_lines = [
                f"[bold cyan]System Resources[/bold cyan]",
                f"────────────────",
                f"CPU Usage:    {cpu_percent:5.1f}%",
                f"CPU Freq:     {freq_ghz:4.2f} GHz",
                f"Memory:       {mem_used_gb:4.1f}/{mem_total_gb:4.1f} GB",
                "",
                f"[bold cyan]Configuration[/bold cyan]",
                f"────────────────",
                f"Trials:       {self.optimizer.num_initial_trials if self.optimizer else '-'}",
                f"Eval Interval: {self.optimizer.eval_interval:,} " if self.optimizer else "-",
                f"Min Iterations: {self.optimizer.min_iterations:,}" if self.optimizer else "-"
            ]

            info_text = "\n".join(info_lines)
        except ImportError:
            info_text = "[dim]System info unavailable\n(install psutil)[/dim]"

        return Panel(
            info_text,
            title="[bold white]System Info[/bold white]",
            border_style="magenta",
            padding=(0, 1)
        )

    def generate_log_panel(self) -> Panel:
        """Generate the activity log panel"""
        if self.recent_logs:
            # Format log entries with timestamps
            log_lines = []
            for timestamp, level, message in self.recent_logs:
                time_str = timestamp.strftime("%H:%M:%S")

                # Color based on level
                level_colors = {
                    'INFO': 'white',
                    'SUCCESS': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'ELIMINATE': 'red',
                    'PHASE': 'cyan'
                }
                color = level_colors.get(level, 'white')

                log_lines.append(f"[dim]{time_str}[/dim] [{color}]{message}[/{color}]")

            log_text = "\n".join(log_lines)
        else:
            log_text = "[dim]Waiting for activity...[/dim]"

        return Panel(
            log_text,
            title="[bold white]Activity Log[/bold white]",
            border_style="yellow",
            padding=(0, 1)
        )

    def generate_footer(self) -> Panel:
        """Generate the footer panel"""
        footer_text = "[dim]Press Ctrl+C to stop | F5 to refresh | H for help[/dim]"
        return Panel(
            Align.center(Text(footer_text)),
            style="dim white on black",
            padding=0,
            height=1
        )

    def _get_status_display(self, status: str, is_best: bool) -> str:
        """Get status display with icon"""
        if is_best:
            return "[bold yellow]👑 BEST[/bold yellow]"

        status_map = {
            'running': '[green]● RUN[/green]',
            'paused': '[yellow]⏸ PAUSE[/yellow]',
            'eliminated': '[red]✗ ELIM[/red]',
            'completed': '[bold green]✓ DONE[/bold green]',
            'pending': '[dim]○ WAIT[/dim]',
            'error': '[red]⚠ ERROR[/red]'
        }
        return status_map.get(status, '[dim]?[/dim]')

    def _create_progress_bar(self, progress: float) -> str:
        """Create a text-based progress bar"""
        if progress <= 0:
            return "[dim]" + "─" * 20 + "[/dim]"

        filled = int(progress * 20)
        bar = "█" * filled + "─" * (20 - filled)

        # Color based on progress
        if progress < 0.33:
            color = "red"
        elif progress < 0.66:
            color = "yellow"
        else:
            color = "green"

        return f"[{color}]{bar}[/{color}]"

    def _get_row_style(self, trial) -> str:
        """Get row style based on trial status"""
        if trial.is_best:
            return "bold yellow"
        elif trial.eliminated:
            return "dim red"
        elif trial.status == 'completed':
            return "green"
        elif trial.status == 'running':
            return "white"
        else:
            return "dim white"

    def add_log(self, message: str, level: str = "INFO"):
        """Add a message to the activity log"""
        self.recent_logs.append((datetime.now(), level, message))

    def run(self, refresh_rate: float = 1.0):
        """Run the dashboard with live updates"""
        self.running = True
        self.start_time = datetime.now()

        # Clear screen and hide cursor for clean display
        self.console.clear()
        self.console.show_cursor(False)

        # Detect Windows for different rendering strategy
        import platform
        is_windows = platform.system() == "Windows"

        try:
            with Live(
                self.layout,
                refresh_per_second=1.0 / max(refresh_rate, 2.0) if is_windows else 1.0 / refresh_rate,
                screen=not is_windows,  # Disable screen mode on Windows to reduce blinking
                console=self.console,
                transient=False,
                auto_refresh=True
            ) as live:
                while self.running:
                    # Update all panels
                    self.layout["header"].update(self.generate_header())
                    self.layout["trials"].update(self.generate_trials_table())
                    self.layout["stats"].update(self.generate_stats_panel())
                    self.layout["info"].update(self.generate_info_panel())
                    self.layout["log"].update(self.generate_log_panel())
                    self.layout["footer"].update(self.generate_footer())

                    # Force Rich to refresh the display
                    live.refresh()

                    time.sleep(refresh_rate)
        finally:
            # Restore cursor on exit
            self.console.show_cursor(True)

    def stop(self):
        """Stop the dashboard"""
        self.running = False