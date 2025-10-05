"""Database models for CASCADE Data Collector."""

from .base import Base, engine, SessionLocal, get_db

# Import all models to register them - using relative imports
from .recording_session import RecordingSession
from .kiwisdr_source import KiwiSDRSource
from .websdr_source import WebSDRSource
from .qrn_sample import QRNSample
from .propagation_record import PropagationRecord
from .space_weather_data import SpaceWeatherData, SolarCyclePhase, QBOPhase, Season
from .ft8_signal import FT8Signal
from .wspr_signal import WSPRSignal
from .atmospheric_event import AtmosphericEvent
from .collection_schedule import CollectionSchedule
from .notification_config import NotificationConfig
from .collection_alerts import CollectionAlert
from .notification_templates import NotificationTemplate

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "RecordingSession",
    "KiwiSDRSource",
    "WebSDRSource",
    "QRNSample",
    "PropagationRecord",
    "SpaceWeatherData",
    "SolarCyclePhase",
    "QBOPhase",
    "Season",
    "FT8Signal",
    "WSPRSignal",
    "AtmosphericEvent",
    "CollectionSchedule",
    "NotificationConfig",
    "CollectionAlert",
    "NotificationTemplate",
]