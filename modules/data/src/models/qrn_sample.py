"""QRNSample model with quiet period detection.

Implements T020: QRNSample model with quiet_periods and correlation_id.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from .base import Base


class QRNSample(Base):
    """Atmospheric noise recording with characterization."""

    __tablename__ = "qrn_samples"

    # Primary key
    sample_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier for the sample",
    )

    # Foreign key
    session_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("recording_sessions.session_id"),
        nullable=False,
        comment="Reference to RecordingSession",
    )

    # Correlation support (FR-028)
    correlation_id = Column(
        PostgresUUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Links to propagation data from same recording",
    )

    # Sample metadata
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Sample timestamp (UTC)",
    )

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

    duration_seconds = Column(
        Float,
        nullable=False,
        comment="Sample duration in seconds",
    )

    # Signal characteristics
    avg_power_dbm = Column(
        Float,
        nullable=True,
        comment="Average power in dBm",
    )

    peak_power_dbm = Column(
        Float,
        nullable=True,
        comment="Peak power in dBm",
    )

    noise_floor_dbm = Column(
        Float,
        nullable=True,
        comment="Noise floor in dBm",
    )

    snr_db = Column(
        Float,
        nullable=True,
        comment="Signal-to-noise ratio in dB",
    )

    # Quiet period detection (FR-026)
    quiet_periods = Column(
        JSON,
        nullable=True,
        comment="Array of quiet period timestamps [{start, end, duration_ms}]",
    )

    quiet_percentage = Column(
        Float,
        nullable=True,
        comment="Percentage of sample that is quiet",
    )

    has_quiet_zones = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether quiet periods were detected",
    )

    # Geographic context
    grid_square = Column(
        String(6),
        nullable=True,
        comment="Maidenhead grid square",
    )

    geographic_region = Column(
        String(50),
        nullable=True,
        comment="Geographic region identifier",
    )

    # QRN characterization
    qrn_type = Column(
        String(50),
        nullable=True,
        comment="Type of QRN detected (storm, precipitation, etc.)",
    )

    impulsiveness = Column(
        Float,
        nullable=True,
        comment="Impulsiveness metric (0-1)",
    )

    spectral_occupancy = Column(
        Float,
        nullable=True,
        comment="Spectral occupancy percentage",
    )

    # Multi-channel extraction (FR-029)
    channel_data = Column(
        JSON,
        nullable=True,
        comment="9x 2.5kHz overlapping channel measurements",
    )

    # Storage
    file_path = Column(
        Text,
        nullable=True,
        comment="Local file path for sample",
    )

    file_size_bytes = Column(
        Integer,
        nullable=True,
        comment="File size in bytes",
    )

    # Quality
    quality_score = Column(
        Float,
        nullable=True,
        comment="Quality score (0-1)",
    )

    validated = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether sample has been validated",
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

    # Relationships
    session = relationship("RecordingSession", back_populates="qrn_samples")

    def __repr__(self):
        return f"<QRNSample(id={self.sample_id}, freq={self.frequency_khz}kHz, quiet={self.quiet_percentage}%)>"

    def extract_quiet_periods(self, threshold_dbm: float = -120) -> list:
        """Extract quiet periods from sample.

        Args:
            threshold_dbm: Power threshold for quiet detection

        Returns:
            List of quiet period dictionaries
        """
        # This would analyze the actual IQ data
        # Placeholder for implementation
        return []

    @property
    def is_quiet_sample(self) -> bool:
        """Check if this is primarily a quiet sample."""
        return self.quiet_percentage and self.quiet_percentage > 50