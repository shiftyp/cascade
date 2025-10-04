"""NotificationConfig model.

Implements T024: NotificationConfig model.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, Boolean, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class NotificationConfig(Base):
    """Alert configuration for SDR availability and errors."""

    __tablename__ = "notification_configs"

    config_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    name = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)

    # Notification channels
    email_enabled = Column(Boolean, default=True)
    email_addresses = Column(JSON)  # List of emails

    # Alert conditions
    min_sdr_threshold = Column(Integer, default=3)
    max_failure_count = Column(Integer, default=5)
    storage_threshold_gb = Column(Integer, default=1000)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)