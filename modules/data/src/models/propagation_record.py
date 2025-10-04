"""PropagationRecord model for FT8/WSPR data.

Implements T021: PropagationRecord model with propagation_mode and correlation_id.
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
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from .base import Base


class PropagationRecord(Base):
    """Extracted FT8/WSPR propagation data."""

    __tablename__ = "propagation_records"

    # Primary key
    record_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        comment="Unique identifier for the record",
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
        comment="Links to QRN data from same recording",
    )

    # Signal metadata
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="Detection timestamp (UTC)",
    )

    frequency_hz = Column(
        Integer,
        nullable=False,
        comment="Exact frequency in Hz",
    )

    mode = Column(
        String(10),
        nullable=False,
        comment="Mode: FT8, WSPR, etc.",
    )

    # Signal characteristics
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

    signal_strength_dbm = Column(
        Float,
        nullable=True,
        comment="Signal strength in dBm",
    )

    # Path information (FR-030)
    tx_grid = Column(
        String(6),
        nullable=True,
        comment="Transmitter grid square (anonymized)",
    )

    rx_grid = Column(
        String(6),
        nullable=True,
        comment="Receiver grid square",
    )

    distance_km = Column(
        Float,
        nullable=True,
        comment="Path distance in km",
    )

    azimuth_degrees = Column(
        Float,
        nullable=True,
        comment="Path azimuth in degrees",
    )

    # Propagation mode detection (FR-025)
    propagation_mode = Column(
        String(50),
        nullable=True,
        comment="Detected mode: F2, Es, Aurora, TEP, MS, etc.",
    )

    propagation_confidence = Column(
        Float,
        nullable=True,
        comment="Mode detection confidence (0-1)",
    )

    mode_indicators = Column(
        JSON,
        nullable=True,
        comment="Mode-specific indicators and features",
    )

    # Environmental context
    solar_flux = Column(
        Float,
        nullable=True,
        comment="Solar flux index at time of detection",
    )

    k_index = Column(
        Integer,
        nullable=True,
        comment="K-index at time of detection",
    )

    # Message content (anonymized)
    callsign_hash = Column(
        String(64),
        nullable=True,
        comment="One-way hash of callsign (FR-005)",
    )

    message_type = Column(
        String(20),
        nullable=True,
        comment="Message type (CQ, reply, telemetry, etc.)",
    )

    # Quality metrics
    decode_confidence = Column(
        Float,
        nullable=True,
        comment="Decode confidence (0-1)",
    )

    false_decode_probability = Column(
        Float,
        nullable=True,
        comment="Probability of false decode",
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
    session = relationship("RecordingSession", back_populates="propagation_records")

    def __repr__(self):
        return f"<PropagationRecord(mode={self.mode}, snr={self.snr_db}dB, prop={self.propagation_mode})>"

    def calculate_path_geometry(self) -> dict:
        """Calculate detailed path geometry."""
        if not (self.tx_grid and self.rx_grid):
            return {}

        # Would calculate:
        # - Great circle distance
        # - Azimuth and elevation angles
        # - Midpoint location
        # - Possible reflection points
        return {
            "distance_km": self.distance_km,
            "azimuth": self.azimuth_degrees,
            "midpoint_grid": None,  # Calculate from grids
        }

    def classify_propagation_mode(self) -> str:
        """Classify propagation mode based on characteristics."""
        # Simplified logic - would use ML model in practice
        if self.distance_km and self.frequency_hz:
            freq_mhz = self.frequency_hz / 1e6

            # Basic classification rules
            if self.distance_km > 3000 and freq_mhz < 10:
                return "F2"
            elif self.distance_km < 2200 and freq_mhz > 28:
                return "Es"  # Sporadic-E
            elif self.k_index and self.k_index >= 5:
                return "Aurora"
            elif self.drift_hz and abs(self.drift_hz) > 2:
                return "MS"  # Meteor scatter

        return "Unknown"