"""WSPRSignal model for decoded WSPR signals."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class WSPRSignal(Base):
    """Decoded WSPR signal data."""

    __tablename__ = "wspr_signals"

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

    drift_hz = Column(
        Float,
        nullable=True,
        comment="Frequency drift in Hz",
    )

    power_dbm = Column(
        Integer,
        nullable=True,
        comment="Transmitter power in dBm",
    )

    # Anonymized identity
    callsign_hash = Column(
        String(32),
        nullable=True,
        comment="Anonymized callsign hash",
    )

    grid_square = Column(
        String(6),
        nullable=True,
        comment="4 or 6 character grid square",
    )

    # Propagation data
    distance_km = Column(
        Float,
        nullable=True,
        comment="Great circle distance in km",
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
        default="WSPR",
        comment="Signal mode",
    )

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="Record creation time",
    )

    def __repr__(self):
        return f"<WSPRSignal(time={self.timestamp}, freq={self.frequency_hz}, SNR={self.snr_db}, power={self.power_dbm})>"