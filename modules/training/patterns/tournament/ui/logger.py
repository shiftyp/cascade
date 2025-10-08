"""Dual logging system for tournament pattern generation

Logs to both file (complete record) and UI (filtered display).
Also maintains statistics logging for post-run analysis.
"""

import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
from collections import deque
from enum import Enum


class LogLevel(Enum):
    """Log levels for UI display"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    ELIMINATE = "ELIMINATE"
    PHASE = "PHASE"


class DualLogger:
    """Logger that outputs to both file and UI"""

    def __init__(
        self,
        log_dir: str = "./logs",
        ui_dashboard=None,
        file_level: int = logging.DEBUG,
        ui_level: str = "INFO",
        use_console: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.ui_dashboard = ui_dashboard
        self.ui_level = ui_level
        self.use_console = use_console

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup file logging
        self._setup_file_logging(file_level)

        # Setup statistics logging
        self._setup_stats_logging()

        # Thread-safe UI log queue
        self.ui_log_queue = deque(maxlen=100)
        self.lock = threading.Lock()

        # Trial-specific loggers
        self.trial_loggers = {}

    def _setup_file_logging(self, level: int):
        """Setup comprehensive file logging"""
        # Main log file
        log_file = self.log_dir / f"tournament_{self.timestamp}.log"

        # Create main logger
        self.file_logger = logging.getLogger("tournament")
        self.file_logger.setLevel(level)

        # Remove existing handlers
        self.file_logger.handlers.clear()

        # File handler with detailed formatting (UTF-8 encoding for Unicode)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)

        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.file_logger.addHandler(file_handler)

        # Console handler for debugging - ONLY if not using Rich UI
        # Rich UI handles its own display, console output corrupts it
        if self.use_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(
                logging.Formatter('%(levelname)-8s | %(message)s')
            )
            self.file_logger.addHandler(console_handler)

        # Log initialization
        self.file_logger.info("=" * 80)
        self.file_logger.info("CASCADE Pattern Tournament Logger Initialized")
        self.file_logger.info(f"Log file: {log_file}")
        self.file_logger.info("=" * 80)

    def _setup_stats_logging(self):
        """Setup statistics logging for analysis"""
        self.stats_file = self.log_dir / f"stats_{self.timestamp}.jsonl"
        self.stats_handle = open(self.stats_file, 'w', encoding='utf-8')

        # Write header
        self.log_stats({
            'event': 'initialization',
            'timestamp': self.timestamp,
            'version': '1.0'
        })

    def get_trial_logger(self, trial_id: int) -> logging.Logger:
        """Get or create a trial-specific logger"""
        if trial_id not in self.trial_loggers:
            # Create trial log directory
            trial_log_dir = self.log_dir / f"trial_{trial_id}"
            trial_log_dir.mkdir(exist_ok=True, parents=True)

            # Create trial logger
            logger_name = f"trial_{trial_id}"
            trial_logger = logging.getLogger(logger_name)
            trial_logger.setLevel(logging.DEBUG)

            # Trial log file with UTF-8 encoding
            trial_log_file = trial_log_dir / f"trial_{trial_id}_{self.timestamp}.log"
            handler = logging.FileHandler(trial_log_file, encoding='utf-8')
            handler.setFormatter(
                logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            )
            trial_logger.addHandler(handler)

            self.trial_loggers[trial_id] = trial_logger

            # Log creation
            trial_logger.info(f"Trial {trial_id} logger initialized")

        return self.trial_loggers[trial_id]

    def log(
        self,
        message: str,
        level: str = "INFO",
        trial_id: Optional[int] = None,
        to_ui: bool = True,
        **kwargs
    ):
        """Main logging method

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, etc.)
            trial_id: Optional trial ID for trial-specific logging
            to_ui: Whether to send to UI dashboard
            **kwargs: Additional data for statistics logging
        """
        # File logging
        log_level = getattr(logging, level.upper(), logging.INFO)
        if trial_id is not None:
            # Log to trial-specific logger
            trial_logger = self.get_trial_logger(trial_id)
            trial_logger.log(log_level, message)

            # Also log to main with trial prefix
            self.file_logger.log(log_level, f"[Trial {trial_id}] {message}")
        else:
            self.file_logger.log(log_level, message)

        # UI logging
        if to_ui and self.ui_dashboard and self._should_log_to_ui(level):
            with self.lock:
                self.ui_dashboard.add_log(message, level)

        # Statistics logging if data provided
        if kwargs:
            stats_data = {
                'timestamp': time.time(),
                'level': level,
                'message': message,
                **kwargs
            }
            if trial_id is not None:
                stats_data['trial_id'] = trial_id
            self.log_stats(stats_data)

    def _should_log_to_ui(self, level: str) -> bool:
        """Check if message should be sent to UI based on level"""
        ui_levels = ['ERROR', 'WARNING', 'SUCCESS', 'INFO', 'ELIMINATE', 'PHASE']
        min_level_index = ui_levels.index(self.ui_level) if self.ui_level in ui_levels else 3

        try:
            level_index = ui_levels.index(level.upper())
            return level_index <= min_level_index
        except ValueError:
            return True  # Unknown levels go to UI

    def log_trial_event(
        self,
        trial_id: int,
        event_type: str,
        data: Dict[str, Any],
        message: Optional[str] = None
    ):
        """Log a trial-specific event

        Args:
            trial_id: Trial identifier
            event_type: Type of event (start, checkpoint, eliminate, etc.)
            data: Event data
            message: Optional human-readable message
        """
        # Generate message if not provided
        if message is None:
            message = self._format_trial_event(trial_id, event_type, data)

        # Determine log level based on event type
        level_map = {
            'start': 'INFO',
            'checkpoint': 'DEBUG',
            'improvement': 'SUCCESS',
            'eliminate': 'ELIMINATE',
            'complete': 'SUCCESS',
            'error': 'ERROR',
            'stagnant': 'WARNING'
        }
        level = level_map.get(event_type, 'INFO')

        # Log with full data
        self.log(
            message,
            level=level,
            trial_id=trial_id,
            event_type=event_type,
            **data
        )

    def _format_trial_event(self, trial_id: int, event_type: str, data: Dict) -> str:
        """Format trial event into human-readable message"""
        if event_type == 'improvement':
            return f"Trial {trial_id} improved: {data.get('old_score', 0):.2f} -> {data.get('new_score', 0):.2f} dB"
        elif event_type == 'eliminate':
            return f"Trial {trial_id} eliminated: {data.get('reason', 'unknown reason')}"
        elif event_type == 'checkpoint':
            return f"Trial {trial_id} checkpoint at iteration {data.get('iteration', 0)}"
        elif event_type == 'start':
            return f"Trial {trial_id} started with seed {data.get('seed', 0)}"
        elif event_type == 'complete':
            return f"Trial {trial_id} completed with score {data.get('score', 0):.2f} dB"
        else:
            return f"Trial {trial_id} event: {event_type}"

    def log_phase_change(self, old_phase: str, new_phase: str):
        """Log optimization phase change"""
        message = f"Phase transition: {old_phase.upper()} -> {new_phase.upper()}"
        self.log(message, level='PHASE', to_ui=True)

    def log_elimination(self, trial_id: int, reason: str, score: float):
        """Log trial elimination"""
        message = f"Eliminating Trial {trial_id} (score: {score:.2f} dB) - {reason}"
        self.log(message, level='ELIMINATE', trial_id=trial_id, to_ui=True)

    def log_stats(self, data: Dict[str, Any]):
        """Log statistics data for analysis"""
        if hasattr(self, 'stats_handle'):
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = time.time()

            # Write as JSON line
            json_line = json.dumps(data, default=str) + "\n"
            self.stats_handle.write(json_line)
            self.stats_handle.flush()

    def log_summary(self, summary_data: Dict[str, Any]):
        """Log run summary"""
        self.file_logger.info("=" * 80)
        self.file_logger.info("TOURNAMENT SUMMARY")
        self.file_logger.info("=" * 80)

        for key, value in summary_data.items():
            self.file_logger.info(f"{key}: {value}")

        # Also save as JSON
        summary_file = self.log_dir / f"summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

    def close(self):
        """Close all log files"""
        # Close stats file
        if hasattr(self, 'stats_handle'):
            self.stats_handle.close()

        # Close trial loggers
        for logger in self.trial_loggers.values():
            for handler in logger.handlers:
                handler.close()

        # Log closure
        self.file_logger.info("Logger closed")

        # Close main logger handlers
        for handler in self.file_logger.handlers:
            handler.close()


import time  # Add missing import at module level


class PerformanceLogger:
    """Specialized logger for performance metrics"""

    def __init__(self, logger: DualLogger):
        self.logger = logger
        self.timers = {}

    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        return 0.0

    def log_timing(self, operation: str, elapsed: float):
        """Log timing information"""
        self.logger.log_stats({
            'event': 'timing',
            'operation': operation,
            'elapsed_seconds': elapsed
        })

    def log_resource_usage(self):
        """Log current resource usage"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            self.logger.log_stats({
                'event': 'resources',
                'cpu_percent': cpu_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent
            })
        except ImportError:
            pass  # psutil not available