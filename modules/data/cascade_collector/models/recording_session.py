"""Recording session model for CASCADE."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class RecordingSession:
    """Recording session information."""

    session_id: str
    sdr_id: str
    band: str
    frequency: float
    sample_rate: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    status: str = "pending"  # pending, recording, completed, failed
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None