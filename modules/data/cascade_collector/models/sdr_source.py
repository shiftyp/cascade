"""SDR Source model."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime


@dataclass
class SDRSource:
    """SDR source configuration and metadata."""

    id: str
    name: str
    url: str
    type: str  # "kiwisdr", "websdr", etc.
    location: Dict[str, float]  # lat, lon, altitude
    frequency_range: Tuple[float, float]  # Min, max frequency in Hz
    sample_rate: int
    bandwidth: int
    online: bool = True
    priority: int = 50  # 0-100
    usage_minutes_today: float = 0
    usage_limit_minutes: int = 90
    last_connected: Optional[datetime] = None
    metadata: Optional[Dict] = None