"""CollectionSchedule model.

Implements T023: CollectionSchedule model.
"""

from datetime import datetime, time
from uuid import uuid4

from sqlalchemy import Column, DateTime, Integer, String, Time, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class CollectionSchedule(Base):
    """Automated recording schedule configuration."""

    __tablename__ = "collection_schedules"

    schedule_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    name = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)

    # Schedule timing
    start_time = Column(Time, nullable=True)  # Daily start time
    end_time = Column(Time, nullable=True)  # Daily end time
    days_of_week = Column(JSON, nullable=True)  # [0-6] for days

    # Recording parameters
    frequency_khz = Column(Integer, nullable=False)
    band = Column(String(10), nullable=True)
    duration_seconds = Column(Integer, default=300)
    interval_minutes = Column(Integer, default=60)

    # Target SDRs
    preferred_sdrs = Column(JSON, nullable=True)  # List of SDR URLs
    min_sdrs = Column(Integer, default=1)
    max_sdrs = Column(Integer, default=10)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)