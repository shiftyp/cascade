"""Configuration module for CASCADE Data Collector."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Main configuration class for the data collector."""

    # Database
    POSTGRES_URL: str = os.getenv(
        "POSTGRES_URL", "postgresql://cascade:cascade@localhost/cascade_data"
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Storage
    TIGRIS_ACCESS_KEY: str = os.getenv("TIGRIS_ACCESS_KEY", "")
    TIGRIS_SECRET_KEY: str = os.getenv("TIGRIS_SECRET_KEY", "")
    TIGRIS_BUCKET: str = os.getenv("TIGRIS_BUCKET", "cascade-iq-data")
    TIGRIS_ENDPOINT: str = os.getenv("TIGRIS_ENDPOINT", "https://fly.storage.tigris.dev")

    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/data"))
    RECORDINGS_DIR: Path = DATA_DIR / "recordings"
    CACHE_DIR: Path = DATA_DIR / "cache"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # Collection parameters
    DEFAULT_SAMPLE_RATE: int = 12000  # 12 kHz
    DEFAULT_BIT_DEPTH: int = 16
    DEFAULT_RECORDING_DURATION: int = 300  # 5 minutes
    MAX_CONCURRENT_SDRS: int = int(os.getenv("MAX_CONCURRENT_SDRS", "50"))
    MIN_CONCURRENT_SDRS: int = int(os.getenv("MIN_CONCURRENT_SDRS", "6"))

    # KiwiSDR limits
    KIWI_DAILY_LIMIT_MINUTES: int = 90
    KIWI_CONNECTION_TIMEOUT: int = 30
    KIWI_RETRY_ATTEMPTS: int = 3
    KIWI_RETRY_DELAY: int = 5

    # QA sampling (FR-036) - Progressive intelligent sampling
    QA_SAMPLE_PERCENTAGE: float = 0.01  # Default 1% (overridden by intelligent sampler)
    QA_HOT_STORAGE_DAYS: int = 7  # Keep in hot storage for 7 days

    # Progressive QA collection phases (18-month strategy)
    QA_PHASE_BOOTSTRAP_MONTHS = [1, 2]  # Months 1-2: 3% random sampling
    QA_PHASE_HYBRID_MONTHS = [3, 4]     # Months 3-4: 8% mixed sampling
    QA_PHASE_PRODUCTION_MONTHS = range(5, 19)  # Months 5-18: 12% intelligent

    QA_BOOTSTRAP_RATE: float = 0.03  # 3% random for initial training data
    QA_HYBRID_RATE: float = 0.08     # 5% random + 3% intelligent
    QA_PRODUCTION_RATE: float = 0.12 # 10% intelligent + 2% random baseline

    # Worker configuration
    WORKER_HEALTH_CHECK_INTERVAL: int = 30  # seconds (FR-043)
    WORKER_SHUTDOWN_GRACE_PERIOD: int = 60  # seconds

    # External APIs
    NOAA_API_URL: str = "https://services.swpc.noaa.gov/json"
    NOAA_API_KEY: Optional[str] = os.getenv("NOAA_API_KEY")

    # Privacy and anonymization
    CALLSIGN_SALT: str = os.getenv("CALLSIGN_SALT", "cascade_default_salt_2025")
    STORE_RAW_MESSAGES: bool = os.getenv("STORE_RAW_MESSAGES", "false").lower() == "true"

    # Dashboard
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [
            cls.RECORDINGS_DIR,
            cls.CACHE_DIR,
            cls.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()