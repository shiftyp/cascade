"""CollectionAlert model.

Implements T024a: CollectionAlerts model.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class CollectionAlert(Base):
    """SDR availability and error alerts."""

    __tablename__ = "collection_alerts"

    alert_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    alert_type = Column(String(50), nullable=False)  # sdr_unavailable, storage_low, etc.
    severity = Column(String(20), nullable=False)  # info, warning, error, critical
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)

    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(255), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)