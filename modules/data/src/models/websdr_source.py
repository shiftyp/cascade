"""WebSDRSource model for WebSDR receiver registry.

Implements FR-065, FR-066, FR-068: WebSDR integration and tracking.
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
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship
import enum

from .base import Base


class InstitutionType(enum.Enum):
    """Institution types for WebSDR operators."""
    UNIVERSITY = "university"
    RESEARCH_INSTITUTE = "research_institute"
    AMATEUR_CLUB = "amateur_club"
    GOVERNMENT = "government"
    INDIVIDUAL = "individual"
    COMMERCIAL = "commercial"


class WebSDRSource(Base):
    """Represents a WebSDR receiver with institutional backing."""

    __tablename__ = "websdr_sources"

    # Primary key
    websdr_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier for the WebSDR",
    )

    # Connection info
    url = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="WebSDR URL",
    )

    name = Column(
        String(255),
        nullable=False,
        comment="WebSDR name/title",
    )

    # Institution information (FR-068)
    institution_type = Column(
        SQLEnum(InstitutionType),
        nullable=False,
        default=InstitutionType.INDIVIDUAL,
        comment="Type of institution operating the WebSDR",
    )

    institution_name = Column(
        String(255),
        nullable=True,
        comment="Name of institution",
    )

    # Contact information for research coordination (FR-068)
    owner_contact = Column(
        Text,
        nullable=True,
        comment="Encrypted contact information for owner/operator",
    )

    contact_email = Column(
        String(255),
        nullable=True,
        comment="Contact email for coordination",
    )

    # Research agreements (FR-063, FR-068)
    has_research_agreement = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether research access agreement exists",
    )

    agreement_details = Column(
        JSON,
        nullable=True,
        comment="Research agreement terms and conditions",
    )

    agreement_expiry = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Agreement expiration date",
    )

    extended_usage_allowed = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether extended usage periods are permitted",
    )

    # Location
    grid_square = Column(
        String(6),
        nullable=True,
        comment="Maidenhead grid square",
    )

    latitude = Column(
        Float,
        nullable=True,
        comment="Latitude",
    )

    longitude = Column(
        Float,
        nullable=True,
        comment="Longitude",
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
        default=0,
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
        default=100,  # WebSDRs typically support many more users
        comment="Maximum concurrent users",
    )

    bandwidth_khz = Column(
        Float,
        nullable=False,
        default=192,  # Many WebSDRs have wider bandwidth
        comment="Total bandwidth in kHz",
    )

    # Usage policies (FR-066)
    daily_limit_minutes = Column(
        Integer,
        nullable=True,  # Many WebSDRs have no limit
        comment="Daily usage limit in minutes (null = unlimited)",
    )

    session_limit_minutes = Column(
        Integer,
        nullable=True,
        default=180,  # Typical WebSDR allows longer sessions
        comment="Per-session limit in minutes",
    )

    peak_hours_utc = Column(
        JSON,
        nullable=True,
        comment="Peak hours to avoid (UTC) as list of [start, end] pairs",
    )

    usage_policy_notes = Column(
        Text,
        nullable=True,
        comment="Special usage policies or restrictions",
    )

    # Usage tracking (FR-066)
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

    # Status
    active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether WebSDR is active/available",
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

    # Priority for longer sessions (FR-067)
    preferred_for_long_sessions = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Prioritize for sessions > 90 minutes",
    )

    # Authentication
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
    antenna_description = Column(
        Text,
        nullable=True,
        comment="Antenna system description",
    )

    receiver_description = Column(
        Text,
        nullable=True,
        comment="Receiver hardware description",
    )

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
    sessions = relationship(
        "RecordingSession",
        back_populates="websdr",
        foreign_keys="RecordingSession.websdr_id",
    )

    def __repr__(self):
        return f"<WebSDRSource(name={self.name}, url={self.url}, institution={self.institution_name})>"

    def can_use_for_duration(self, duration_minutes: int) -> bool:
        """Check if WebSDR can be used for the requested duration.

        Args:
            duration_minutes: Requested session duration

        Returns:
            True if duration is within limits
        """
        # Check session limit
        if self.session_limit_minutes and duration_minutes > self.session_limit_minutes:
            return False

        # Check daily limit
        if self.daily_limit_minutes:
            remaining = self.daily_limit_minutes - self.daily_usage_minutes
            if duration_minutes > remaining:
                return False

        return True

    def is_in_peak_hours(self, utc_hour: int) -> bool:
        """Check if current UTC hour is during peak hours.

        Args:
            utc_hour: Current UTC hour (0-23)

        Returns:
            True if in peak hours
        """
        if not self.peak_hours_utc:
            return False

        for start, end in self.peak_hours_utc:
            if start <= utc_hour < end:
                return True

        return False

    def get_priority_score(self, session_duration_minutes: int) -> float:
        """Calculate priority score for SDR selection.

        Higher scores are preferred.

        Args:
            session_duration_minutes: Expected session duration

        Returns:
            Priority score (0-100)
        """
        score = 50.0  # Base score

        # Boost for long sessions (FR-067)
        if session_duration_minutes > 90 and self.preferred_for_long_sessions:
            score += 30

        # Boost for research agreements
        if self.has_research_agreement:
            score += 20

        # Boost for unlimited usage
        if not self.daily_limit_minutes:
            score += 10

        # Reduce for low reliability
        if self.reliability_score:
            score *= self.reliability_score

        # Reduce during peak hours
        current_hour = datetime.utcnow().hour
        if self.is_in_peak_hours(current_hour):
            score *= 0.5

        return min(100, max(0, score))