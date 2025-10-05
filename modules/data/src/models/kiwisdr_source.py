"""KiwiSDRSource model for receiver registry.

Implements T019: KiwiSDRSource model.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from .base import Base


class KiwiSDRSource(Base):
    """Represents a public KiwiSDR receiver."""

    __tablename__ = "kiwisdr_sources"

    # Primary key
    kiwisdr_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier for the KiwiSDR",
    )

    # Connection info
    url = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="KiwiSDR URL (host:port)",
    )

    name = Column(
        String(255),
        nullable=True,
        comment="Friendly name for the SDR",
    )

    # Location (anonymized)
    grid_square = Column(
        String(10),
        nullable=True,
        comment="Maidenhead grid square (anonymized, up to 10 chars for extended locators)",
    )

    latitude = Column(
        Float,
        nullable=True,
        comment="Latitude (rounded for privacy)",
    )

    longitude = Column(
        Float,
        nullable=True,
        comment="Longitude (rounded for privacy)",
    )

    timezone = Column(
        String(50),
        nullable=True,
        default="UTC",
        comment="Local timezone",
    )

    # Capabilities
    min_freq_khz = Column(
        Float,
        nullable=False,
        default=10,
        comment="Minimum frequency in kHz",
    )

    max_freq_khz = Column(
        Float,
        nullable=False,
        default=30000,
        comment="Maximum frequency in kHz",
    )

    max_users = Column(
        Integer,
        nullable=False,
        default=4,
        comment="Maximum concurrent users",
    )

    has_gps = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="GPS timing available",
    )

    antenna_type = Column(
        String(255),
        nullable=True,
        comment="Antenna type description",
    )

    # Usage policies (FR-061, FR-063)
    peak_hours_utc = Column(
        JSON,
        nullable=True,
        comment="Peak hours to avoid (UTC) as list of [start, end] pairs",
    )

    owner_contact = Column(
        Text,
        nullable=True,
        comment="Encrypted owner contact information for coordination",
    )

    has_research_agreement = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether research access agreement exists (FR-063)",
    )

    usage_policy_notes = Column(
        Text,
        nullable=True,
        comment="Special usage restrictions or policies",
    )

    # Usage tracking (FR-008, FR-014)
    daily_limit_minutes = Column(
        Float,
        nullable=False,
        default=90,
        comment="Daily usage limit in minutes (typically 90 for KiwiSDR)",
    )

    daily_usage_minutes = Column(
        Float,
        nullable=False,
        default=0,
        comment="Current daily usage in minutes",
    )

    last_usage_reset = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Last usage counter reset",
    )

    total_usage_minutes = Column(
        Float,
        nullable=False,
        default=0,
        comment="Total historical usage",
    )

    last_connected = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful connection time",
    )

    last_seen = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time SDR was seen in public directory",
    )

    # Status
    active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether SDR is active/available",
    )

    reliability_score = Column(
        Float,
        nullable=True,
        comment="Reliability score (0-1)",
    )

    failure_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of connection failures",
    )

    consecutive_failures = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Consecutive failures without success (for blacklist detection)",
    )

    last_failure_type = Column(
        String(50),
        nullable=True,
        comment="Type of last failure: timeout, refused, auth_failed, blacklist",
    )

    last_failure_time = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Time of last connection failure",
    )

    potentially_blacklisted = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Flag indicating possible IP blacklist (10+ consecutive connection refused)",
    )

    # Authentication (optional)
    requires_auth = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether authentication is required",
    )

    auth_config = Column(
        JSON,
        nullable=True,
        comment="Encrypted auth configuration",
    )

    # Metadata
    notes = Column(
        Text,
        nullable=True,
        comment="Additional notes",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Record creation time",
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Last update time",
    )

    # Relationships
    sessions = relationship("RecordingSession", back_populates="kiwisdr_source")

    def __repr__(self):
        return f"<KiwiSDRSource(url={self.url}, grid={self.grid_square})>"

    @property
    def remaining_daily_minutes(self) -> float:
        """Calculate remaining daily usage allowance."""
        from src.config import config

        return max(0, config.KIWI_DAILY_LIMIT_MINUTES - self.daily_usage_minutes)

    @property
    def is_available(self) -> bool:
        """Check if SDR is available for use."""
        return (
            self.active
            and self.remaining_daily_minutes > 0
            and self.failure_count < 5
        )

    def should_reset_usage(self) -> bool:
        """Check if daily usage should be reset."""
        now = datetime.utcnow()
        last_reset = self.last_usage_reset.replace(tzinfo=None) if self.last_usage_reset else now
        return (now - last_reset).days >= 1