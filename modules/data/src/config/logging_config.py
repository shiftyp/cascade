"""Logging configuration for CASCADE data collector.

Provides consistent logging format across all modules.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    format: Optional[str] = None,
    log_file: Optional[str] = None,
):
    """Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format string
        log_file: Optional file path for logging
    """
    # Get level from env or parameter
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Default format for production
    if format is None:
        if os.getenv("FLY_APP_NAME"):
            # Simplified format for Fly.io (they add timestamps)
            format = "%(name)s [%(levelname)s] %(message)s"
        else:
            # Full format for local development
            format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format))
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set specific module levels if needed
    if os.getenv("DEBUG_REDIS") == "true":
        logging.getLogger("redis").setLevel(logging.DEBUG)
    else:
        # Redis is quite verbose at INFO level
        logging.getLogger("redis").setLevel(logging.WARNING)

    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {level} level")
    if log_file:
        logger.info(f"Logging to file: {log_file}")