"""AtmosphericEvent model for QRN impulses and quiet periods."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class AtmosphericEvent(Base):
    """Atmospheric events detected during QRN analysis."""

    __tablename__ = "atmospheric_events"

    # Primary key
    event_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique event identifier",
    )

    # Session reference
    session_id = Column(
        PostgresUUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Reference to recording session",
    )

    # Event timing
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Event timestamp (UTC)",
    )

    # Event type
    event_type = Column(
        String(20),
        nullable=False,
        comment="Event type (impulse, quiet_period, etc.)",
    )

    # Event characteristics
    peak_amplitude = Column(
        Float,
        nullable=True,
        comment="Peak amplitude for impulses",
    )

    duration_ms = Column(
        Float,
        nullable=True,
        comment="Event duration in milliseconds",
    )

    rise_time_us = Column(
        Float,
        nullable=True,
        comment="Rise time in microseconds",
    )

    decay_time_ms = Column(
        Float,
        nullable=True,
        comment="Decay time in milliseconds",
    )

    # Quality metrics for quiet periods
    avg_noise_level = Column(
        Float,
        nullable=True,
        comment="Average noise level during quiet period",
    )

    min_noise_level = Column(
        Float,
        nullable=True,
        comment="Minimum noise level during quiet period",
    )

    quality_score = Column(
        Float,
        nullable=True,
        comment="Quality score for quiet periods (0-1)",
    )

    # Spectral data
    frequency_content = Column(
        JSON,
        nullable=True,
        comment="Frequency content analysis",
    )

    # Classification
    classification = Column(
        String(50),
        nullable=True,
        comment="Event classification (lightning, sferics, etc.)",
    )

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Record creation time",
    )

    def __repr__(self):
        return f"<AtmosphericEvent(type={self.event_type}, time={self.timestamp}, duration={self.duration_ms}ms)>"