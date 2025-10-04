"""NotificationTemplate model.

Implements T024b: NotificationTemplates model.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from .base import Base


class NotificationTemplate(Base):
    """Message templates for alerts."""

    __tablename__ = "notification_templates"

    template_id = Column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    name = Column(String(255), nullable=False, unique=True)
    alert_type = Column(String(50), nullable=False)
    subject = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)