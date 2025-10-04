"""FT8Signal model for decoded FT8 signals."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class FT8Signal(Base):
    """Decoded FT8 signal data."""

    __tablename__ = "ft8_signals"

    # Primary key
    signal_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique signal identifier",
    )

    # Session reference
    session_id = Column(
        PostgresUUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Reference to recording session",
    )

    # Signal timing
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Signal timestamp (UTC)",
    )

    # Signal characteristics
    frequency_hz = Column(
        Float,
        nullable=False,
        comment="Signal frequency in Hz",
    )

    snr_db = Column(
        Float,
        nullable=False,
        comment="Signal-to-noise ratio in dB",
    )

    dt_seconds = Column(
        Float,
        nullable=True,
        comment="Time offset from nominal in seconds",
    )

    # Anonymized content
    message_hash = Column(
        String(32),
        nullable=True,
        comment="Anonymized callsign hash",
    )

    grid_square = Column(
        String(6),
        nullable=True,
        comment="4 or 6 character grid square",
    )

    # Band and mode
    band = Column(
        String(10),
        nullable=True,
        comment="Amateur band (20m, 40m, etc.)",
    )

    mode = Column(
        String(20),
        nullable=False,
        default="FT8",
        comment="Signal mode",
    )

    # Raw message (optional - controlled by config)
    raw_message = Column(
        String(50),
        nullable=True,
        comment="Raw message content (if enabled)",
    )

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Record creation time",
    )

    def __repr__(self):
        return f"<FT8Signal(time={self.timestamp}, freq={self.frequency_hz}, SNR={self.snr_db})>"