"""RecordingSession model with correlation support.

Implements T018: RecordingSession model with correlation_id.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from .base import Base


class RecordingSession(Base):
    """Represents a single data collection session."""

    __tablename__ = "recording_sessions"

    # Primary key
    session_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier for the session",
    )

    # Foreign keys
    kiwisdr_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("kiwisdr_sources.kiwisdr_id"),
        nullable=False,
        comment="Reference to KiwiSDR source",
    )

    # Correlation support (FR-028)
    correlation_id = Column(
        PostgresUUID(as_uuid=True),
        default=uuid4,
        nullable=False,
        index=True,
        comment="Links related QRN and propagation samples",
    )

    # Session metadata
    start_time = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Session start time (UTC)",
    )

    end_time = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Session end time (UTC)",
    )

    duration_seconds = Column(
        Integer,
        nullable=True,
        comment="Planned or actual duration in seconds",
    )

    # Recording parameters
    frequency_khz = Column(
        Float,
        nullable=False,
        comment="Center frequency in kHz",
    )

    bandwidth_khz = Column(
        Float,
        nullable=False,
        default=12.0,
        comment="Bandwidth in kHz",
    )

    sample_rate = Column(
        Integer,
        nullable=False,
        default=12000,
        comment="Sample rate in Hz",
    )

    mode = Column(
        String(10),
        nullable=False,
        default="iq",
        comment="Recording mode (iq, am, usb, lsb)",
    )

    # Propagation mode (FR-025)
    propagation_mode = Column(
        String(20),
        nullable=True,
        comment="Propagation mode (F2, Aurora, Sporadic-E, TEP, meteor scatter)",
    )

    # Quality metrics
    gps_locked = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="GPS lock status during recording",
    )

    avg_snr_db = Column(
        Float,
        nullable=True,
        comment="Average SNR in dB",
    )

    sample_rate_accuracy = Column(
        Float,
        nullable=True,
        comment="Sample rate accuracy (±%)",
    )

    gaps_detected = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of gaps > 1 second detected",
    )

    # Storage
    file_path = Column(
        Text,
        nullable=True,
        comment="Local file path",
    )

    tigris_path = Column(
        Text,
        nullable=True,
        comment="Tigris S3 storage path",
    )

    file_size_bytes = Column(
        Integer,
        nullable=True,
        comment="File size in bytes",
    )

    compressed = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether FLAC compression was applied",
    )

    # Status
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        comment="Status: pending, recording, completed, failed, quarantined",
    )

    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if failed",
    )

    # Metadata
    band = Column(
        String(10),
        nullable=True,
        comment="Band designation (80m, 40m, 20m, etc.)",
    )

    notes = Column(
        Text,
        nullable=True,
        comment="Additional notes or metadata",
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
    kiwisdr_source = relationship("KiwiSDRSource", back_populates="sessions")
    qrn_samples = relationship("QRNSample", back_populates="session")
    propagation_records = relationship("PropagationRecord", back_populates="session")

    def __repr__(self):
        return f"<RecordingSession(id={self.session_id}, freq={self.frequency_khz}kHz, status={self.status})>"

    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status == "recording"

    @property
    def duration_actual(self) -> Optional[float]:
        """Calculate actual duration if ended."""
        if self.end_time and self.start_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds()
        return None

    def validate_quality(self) -> bool:
        """Validate quality metrics per FR-013."""
        if not self.gps_locked:
            return False
        if self.sample_rate_accuracy and abs(self.sample_rate_accuracy) > 1.0:
            return False
        if self.gaps_detected > 0:
            return False
        return True